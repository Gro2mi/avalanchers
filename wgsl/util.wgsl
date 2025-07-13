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
const maxVelocityFactor: f32 = 1e7;

@group(0) @binding(0) var<uniform> simSettings: SimSettings;

fn cellToUV(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(simSettings.grid_shape);
}
fn cell3ToUV(cell: vec3u) -> vec2f {
    return (vec2f(cell.xy) + 0.5) / vec2f(simSettings.grid_shape);
}
fn cellfToUV(cell: vec2f) -> vec2f {
    return (cell + 0.5) / vec2f(simSettings.grid_shape);
}

fn positionToUV(position: vec3f) -> vec2f {
    return position.xy / vec2f(simSettings.world_size);
}


fn uvToCell(uv: vec2f) -> vec2u {
    return vec2u(clamp(uv * vec2f(simSettings.grid_shape), vec2f(0.0), vec2f(simSettings.grid_shape - 1u)));
}

fn uvToCellIndex(uv: vec2f) -> u32 {
    let cell = uvToCell(uv);
    // return cell.x * simSettings.grid_shape.y + cell.y;
    return (cell.y % simSettings.grid_shape.y * simSettings.grid_shape.x +
              (cell.x % simSettings.grid_shape.x));
}
