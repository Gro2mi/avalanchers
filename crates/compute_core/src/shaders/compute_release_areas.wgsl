// @group(0) @binding(0) var<uniform> sim_settings: sim_settings;
@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var slope_texture: texture_2d<f32>;
@group(0) @binding(3) var roughness_texture: texture_2d<f32>;
// @group(0) @binding(4) var landcoverTexture: texture_2d<u32>;

@group(0) @binding(4) var release_areas_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<storage, read_write> debug: array<f32>;
@group(0) @binding(6) var<storage, read_write> atomic_values: AtomicValues;

const confers_trees = vec4u(34, 139, 34, 255);
const broadleaf_deciduous_trees = vec4u(128, 255, 0, 255);
const broadleaf_evergreen_trees = vec4u(0, 255, 8, 255);

// fn is_forest(id: vec2u) -> bool {
//     // check if the pixel is within the bounds of the landcover texture
//     if (id.x >= textureDimensions(landcoverTexture).x || id.y >= textureDimensions(landcoverTexture).y) {
//         return false;
//     }
//     // load the landcover value at the given id
//     let landcover_value = textureLoad(landcoverTexture, id, 0).r; // assuming landcover is stored in the red channel
//     // check if the value corresponds to forest (e.g., 1.0 for forest)
//     return landcover_value > 0.5; // adjust threshold as needed
// }

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn compute_release_areas(@builtin(global_invocation_id) id: vec3<u32>) {
    // exit if thread id is outside image dimensions (i.e. thread is not supposed to be doing any work)
    let texture_size = textureDimensions(release_areas_texture);
    if id.x >= texture_size.x || id.y >= texture_size.y {
        return;
    }
    let tex_pos = id.xy;
    var rgba = textureLoad(slope_texture, tex_pos, 0);
    let slope_angle = rgba.r;
    let wind_shelter_index = rgba.b;

    let elevation = textureLoad(dem_texture, tex_pos, 0).x;
    rgba = textureLoad(roughness_texture, tex_pos, 0);
    let roughness = rgba.r;
    let forest = rgba.g;

    let gpx_mask = 0f;
    let predictor = 0f;

    if slope_angle < sim_settings.min_slope_angle || slope_angle > sim_settings.max_slope_angle 
        || elevation < sim_settings.min_elevation
        || roughness > sim_settings.roughness_threshold
        || forest > 0.1 {
        // no release cell
        textureStore(release_areas_texture, tex_pos, vec4f(0f, gpx_mask, predictor, 0f));
    } else {
        // release cell
        textureStore(release_areas_texture, tex_pos, vec4f(sim_settings.slab_thickness, gpx_mask, predictor, 0f));
        atomicAdd(&atomic_values.number_release_cells, 1u);
        debug[0] = f32(sim_settings.slab_thickness);
    }
    // needs to stay here, otherwise texture is not used
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR; 

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

struct Particle {
    position: vec3f,
    mass: f32,
    velocity: vec3f,
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

struct AtomicValues {
    peak_velocity: atomic<u32>,
    peak_flow_thickness: atomic<u32>,
    alpha: atomic<u32>,
    travel_length: atomic<u32>,
    release_volume: atomic<u32>,
    number_release_cells: atomic<u32>,
    number_release_particles: atomic<u32>,
};

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

fn xy_to_idx(x: u32, y: u32) -> u32 {
    return y * sim_settings.grid_shape.x + x;
}

fn quadratic_weight(d: f32) -> f32 {
    let abs_d = abs(d);
    if abs_d < 0.5 {
        return 0.75 - abs_d * abs_d;
    } else if abs_d < 1.5 {
        return 0.5 * pow(1.5 - abs_d, 2.0);
    }
    return 0.0;
}

fn calculate_weight(particle_position: vec2f, node_position: vec2i) -> f32 {
    let dist = particle_position - vec2f(node_position);
    return quadratic_weight(dist.x) * quadratic_weight(dist.y);
}

fn get_base_node(grid_pos: vec2f) -> vec2i {
    return vec2i(floor(grid_pos - vec2f(0.5)));
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

    if abs(area) < 1e-6 {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(cx, cy) / (6.0 * area);
}
// END utils.wgsl