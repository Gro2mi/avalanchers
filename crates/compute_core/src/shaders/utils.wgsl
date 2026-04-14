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


