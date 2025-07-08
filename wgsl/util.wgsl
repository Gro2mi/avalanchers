struct Particle {
    position: vec3f,
    mass: f32,
    velocity: vec3f,
    snow_thickness: f32,
    C: mat2x2f,
};

struct SimSettings {
    num_steps: u32,
    model_type: u32,
    friction_model: u32,
    released_particles_per_cell: u32,

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

struct AtomicData {
    counter: atomic<u32>,
};

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;


fn cellIndexToUV(cell: vec2u, resolution: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(resolution);
}
fn cellIndexfToUV(cell: vec2f, resolution: vec2u) -> vec2f {
    return (cell + 0.5) / vec2f(resolution);
}

fn uvToCellIndex(uv: vec2f, resolution: vec2u) -> vec2u {
    return vec2u(clamp(uv * vec2f(resolution), vec2f(0.0), vec2f(resolution - 1u)));
}
