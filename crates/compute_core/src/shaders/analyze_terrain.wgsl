// import utils.wgsl;
// BEGIN utils.wgsl
struct Particle {
    position: vec3f,
    mass: f32,
    velocity: vec3f,
    snow_thickness: f32,
    C: mat2x2f,
    stopped: u32,
};

struct SimInfo {
  timestep: u32,
  number_particles: u32,
  elevation_threshold: f32,
  max_velocity: f32,
};

struct SimSettings {
    num_steps: u32,
    model_type: u32,
    friction_model: u32,
    released_particles_per_cell: u32,
    grid_shape: vec2u,

    world_size: vec2f,
    snow_density: f32,
    slab_thickness: f32,
    friction_coefficient: f32,
    drag_coefficient: f32,
    cfl: f32,
    cell_size: f32,
    min_slope_angle: f32,
    max_slope_angle: f32,
    min_elevation: f32,
    velocity_threshold: f32,
    roughness_threshold: f32,
};

struct AtomicValue {
    value: atomic<u32>,
};

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;
const MAX_VELOCITY_FACTOR: f32 = 1e7;

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cell3_to_uv(cell: vec3u) -> vec2f {
    return (vec2f(cell.xy) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cellf_to_uv(cell: vec2f) -> vec2f {
    return (cell + 0.5) / vec2f(sim_settings.grid_shape);
}

fn position_to_uv(position: vec3f) -> vec2f {
    return position.xy / vec2f(sim_settings.world_size);
}

fn position_to_cell_index(position: vec3f) -> u32 {
    let uv = position_to_uv(position);
    return uv_to_cell_index(uv);
}


fn uv_to_cell(uv: vec2f) -> vec2u {
    return vec2u(clamp(uv * vec2f(sim_settings.grid_shape), vec2f(0.0), vec2f(sim_settings.grid_shape - 1u)));
}

fn uv_to_cell_index(uv: vec2f) -> u32 {
    let cell = uv_to_cell(uv);
    // return cell.x * sim_settings.grid_shape.y + cell.y;
    return (cell.y % sim_settings.grid_shape.y * sim_settings.grid_shape.x +
              (cell.x % sim_settings.grid_shape.x));
}

fn compute_centroid(points: ptr<function, array<vec2<f32>, 256>>, count: u32) -> vec2<f32> {
    var area: f32 = 0.0;
    var cx: f32 = 0.0;
    var cy: f32 = 0.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let j = (i + 1u) % count;
        let p0 = (*points)[i];
        let p1 = (*points)[j];
        let cross = p0.x * p1.y - p1.x * p0.y;

        area = area + cross;
        cx = cx + (p0.x + p1.x) * cross;
        cy = cy + (p0.y + p1.y) * cross;
    }

    area = area * 0.5;

    if (abs(area) < 1e-6) {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(cx, cy) / (6.0 * area);
}



// END utils.wgsl

@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var wind_shelter_texture: texture_2d<f32>;

@group(0) @binding(3) var normals_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var slope_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var curvature_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var<storage, read_write> debug: array<f32>;


@compute @workgroup_size(16, 16, 1)
fn analyze_terrain(@builtin(global_invocation_id) cell: vec3<u32>) {
    // exit if thread cell is out of the image dimensions (i.e. thread is not supposed to be doing any work)
    if (cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y) {
        return;
    }

    let resolution = sim_settings.cell_size;
    let resolution2 = resolution * resolution;
    let coord = vec2<i32>(cell.xy);

    // Sample center and neighbors
    // selects handle domain edges
    let center = textureLoad(dem_texture, coord + vec2<i32>(0, 0), 0).r;                    
    let left   = select(textureLoad(dem_texture, coord + vec2<i32>(-1, 0), 0).r, center, cell.x==0);
    let right  = select(textureLoad(dem_texture, coord + vec2<i32>( 1, 0), 0).r, center, cell.x==sim_settings.grid_shape.x - 1);
    let down   = select(textureLoad(dem_texture, coord + vec2<i32>(0, -1), 0).r, center, cell.y==0);
    let up     = select(textureLoad(dem_texture, coord + vec2<i32>(0,  1), 0).r, center, cell.y==sim_settings.grid_shape.y - 1);

    let up_right    = select(textureLoad(dem_texture, coord + vec2<i32>(1,  1), 0).r, center, cell.x==sim_settings.grid_shape.x - 1 || cell.y==sim_settings.grid_shape.y - 1);
    let down_right  = select(textureLoad(dem_texture, coord + vec2<i32>(-1,  1), 0).r, center, cell.x==sim_settings.grid_shape.x - 1  || cell.y==0);
    let up_left     = select(textureLoad(dem_texture, coord + vec2<i32>(1,  -1), 0).r, center, cell.x==0 || cell.y==0);
    let down_left   = select(textureLoad(dem_texture, coord + vec2<i32>(-1,  -1), 0).r, center, cell.x==0 || cell.y==sim_settings.grid_shape.y - 1);
    // selects handle domain edges
    let dx = select((right - left) / (2.0 * resolution), (right - left) / resolution, cell.x==0 || cell.x==sim_settings.grid_shape.x - 1);
    let dy = select((up - down) / (2.0 * resolution), (up - down) / resolution, cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    let normal = normalize(vec3f(-dx, -dy, 1.0));

    // TODO: 2nd derivative needs more care at the edges
    let dxx = select((left - 2*center + right) / resolution2, (left - 2*center + right) / (4 * resolution2), cell.x==0 || cell.x==sim_settings.grid_shape.x - 1);
    let dyy = select((up - 2*center + down) / resolution2, (up - 2*center + down) / (4 * resolution2), cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    let dxy = select((up_right - down_right - up_left + down_left) / (4 * resolution2), (up_right - down_right - up_left + down_left) / (16 * resolution2), cell.x==0 || cell.x==sim_settings.grid_shape.x - 1 || cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);

    let denom = pow(dx*dx + dy*dy + 1e-12, 1.5);
    let p = dx;
    let q = dy;
    let profile_curvature = select((dxx*dx*dx + 2.0*dxy*dx*dy + dyy*dy*dy) / denom, 0.0, cell.x==0 || cell.x==sim_settings.grid_shape.x - 1 || cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    // let planCurvature    = (dxx*dy*dy - 2.0*dxy*dx*dy + dyy*dx*dx) / denom;
    // let meanCurvature = ((1.0 + dy*dy)*dxx - 2.0*dx*dy*dxy + (1.0 + dx*dx)*dyy) / (2.0 * pow(1.0 + dx*dx + dy*dy, 1.5));
    // let normal_f16: vec4<f16> = vec4<f16>(
    //     f16(normal.x), f16(normal.y), f16(normal.z), f16(1.0)
    // );
    // textureStore(normalsTexture, coord, normal_f16);
    textureStore(normals_texture, coord, vec4f(normal, profile_curvature));
    textureStore(curvature_texture, coord, vec4f(dxx, dxy, dyy, 0.0));

    let slope_angle = acos(normal.z) * RAD_TO_DEG;
    let slope_aspect = (atan2(normal.x, normal.y) * RAD_TO_DEG + 360.0) % 360.0;
    // let slopeAngle = normal.z;
    // let slopeAspect = 2.3;
    let wind_shelter_index = textureLoad(wind_shelter_texture, coord, 0).r;
    textureStore(slope_texture, coord, vec4f(slope_angle, slope_aspect, wind_shelter_index, 0f));

    if(cell.x == 0 && cell.y == 0) {
        debug[0] = normal.x;
        debug[1] = normal.y;
        debug[2] = normal.z;
        debug[3] = resolution;
        debug[4] = dx;
        debug[5] = dy;
        debug[6] = dxx;
        debug[7] = dyy;
        debug[8] = dxy;
        debug[9] = profile_curvature;
        debug[10] = left;
        debug[11] = right;
        debug[12] = up;
        debug[13] = down;
        debug[14] = center;
        debug[15] = slope_angle;
        debug[16] = slope_aspect;
        debug[17] = up_right;
        debug[18] = down_right;
        debug[19] = up_left;
        debug[20] = down_left;
    }

}
