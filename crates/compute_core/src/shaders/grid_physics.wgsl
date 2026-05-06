@group(0) @binding(1) var<storage, read_write> grid_h_atomic: array<atomic<i32>>;
@group(0) @binding(2) var normals_texture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> grid_forces: array<vec2f>;

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics(@builtin(global_invocation_id) id: vec3u) {
    let idx = xy_to_idx(id.x, id.y);

    // 1. Decode height and velocity[cite: 3]
    let h = f32(atomicLoad(&grid_h_atomic[idx])) * INV_H_FACTOR;
    // let u = f32(atomicLoad(&grid_mom_atomic[idx * 2])) / (h * SCALE_FACTOR + EPSILON);
    // let v = f32(atomicLoad(&grid_mom_atomic[idx * 2 + 1])) / (h * SCALE_FACTOR + EPSILON);

    // 2. Compute Divergence for Active/Passive state[cite: 3]
    // TODO calculate divergence and earth pressure coefficient
    // let div_u = (get_u(id.x + 1, id.y) - get_u(id.x - 1, id.y)) / (2.0 * dx);
    // let k = calculate_k(div_u); // Returns k_act, k_pass, or 1.0 based on div_u[cite: 3]
    let k = 1.0;
    // 3. Lateral Pressure Force[cite: 3]
    // Force = -0.5 * g * cos(theta) * k * gradient(h^2)
    // TODO do I need to apply a filter to the height field to prevent noise in the gradient?
    // e. g. h_ij = (1-alpha)h_ij + alpha/4 * (h_ij-1 + h_i+1j + h_ij+1 + h_i-1j) to smooth the height field and prevent noise in the gradient?
    // TODO do i need a cutoff for small h to prevent noise in the gradient? if h < h_min -> h = 0
    let grad_h2 = vec2f(
        // TODO account for slope in x and y direction. multiply by cos_theta_x
        (get_h2(id.x + 1, id.y) - get_h2(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size),
        (get_h2(id.x, id.y + 1) - get_h2(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size)
    );
    // TODO do i need a slope limiter like minmod?
    let n = textureLoad(normals_texture, id.xy, 0);
    let slope_corrected_grad_h2 = grad_h2 * sqrt(1.0 - n.x * n.x);
    grid_forces[idx] = -0.5 * g * n.z * k * slope_corrected_grad_h2;
}

fn get_h2(x: u32, y: u32) -> f32 {
    let idx = xy_to_idx(x, y);
    return pow(f32(atomicLoad(&grid_h_atomic[idx])) * INV_H_FACTOR, 2.0);
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const H_FACTOR: f32 = 1e6; // u32 limit is 4.3km thickness
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_H_FACTOR: f32 = 1 / H_FACTOR; // u32 limit is 4.3km thickness

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