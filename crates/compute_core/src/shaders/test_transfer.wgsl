
@group(0) @binding(1) var<storage, read_write> particles_position: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particles_velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_mass: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_mass_atomic: array<atomic<f32>>;
@group(0) @binding(5) var<storage, read_write> grid_momentum_atomic: array<atomic<f32>>;
@group(0) @binding(6) var<storage, read_write> grid_velocity: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> test_results: array<f32>;
@group(0) @binding(8) var<storage, read_write> particles_affine_matrix: array<mat2x2<f32>>;

@compute @workgroup_size(1)
fn test_transfer(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 1u) {
        return;
    }
    let idx = position_to_idx(particles_position[0]);
    let idx_y_minus = position_to_idx(particles_position[0] - vec2<f32>(0.0, sim_settings.cell_size));
    let idx_y_plus = position_to_idx(particles_position[0] + vec2<f32>(0.0, sim_settings.cell_size));
    transfer_p2g(0u);

    for (var i: u32 = 0; i < sim_settings.grid_shape.x * sim_settings.grid_shape.y; i++) {
        // let mass = f32(atomicLoad(&grid_mass_atomic[i])) * INV_MASS_FACTOR;
        // let u = f32(atomicLoad(&grid_momentum_atomic[i * 2])) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
        // let v = f32(atomicLoad(&grid_momentum_atomic[i * 2 + 1])) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
        let mass = atomicLoad(&grid_mass_atomic[i]);
        let u = atomicLoad(&grid_momentum_atomic[i * 2]) / (mass + 1e-6);
        let v = atomicLoad(&grid_momentum_atomic[i * 2 + 1]) / (mass + 1e-6);
        grid_velocity[i] = vec2f(u, v);
    }
    // let mass = f32(atomicLoad(&grid_mass_atomic[idx])) * INV_MASS_FACTOR;
    // var u = f32(atomicLoad(&grid_momentum_atomic[idx * 2])) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
    // var v = f32(atomicLoad(&grid_momentum_atomic[idx * 2 + 1])) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
    let mass = atomicLoad(&grid_mass_atomic[idx]);
    var u = atomicLoad(&grid_momentum_atomic[idx * 2]) / (mass + 1e-6);
    var v = atomicLoad(&grid_momentum_atomic[idx * 2 + 1]) / (mass + 1e-6);

    let interpolated_velocity = transfer_g2p(0u);
    // for affine transfer
    // let update = transfer_g2p(0u);
    // let interpolated_velocity = update.velocity;
    // let new_affine_matrix = update.affine_matrix;


    test_results[0] = particles_position[0].x;
    test_results[1] = particles_position[0].y;
    test_results[2] = particles_velocity[0].x;
    test_results[3] = particles_velocity[0].y;
    test_results[4] = particles_mass[0];
    test_results[5] = f32(atomicLoad(&grid_mass_atomic[32u]));
    test_results[6] = f32(atomicLoad(&grid_mass_atomic[33u]));
    test_results[7] = f32(atomicLoad(&grid_mass_atomic[34u]));

    test_results[8] = f32(atomicLoad(&grid_mass_atomic[42u]));
    test_results[9] = f32(atomicLoad(&grid_mass_atomic[43u]));
    test_results[10] = f32(atomicLoad(&grid_mass_atomic[44u]));

    test_results[11] = f32(atomicLoad(&grid_mass_atomic[52u]));
    test_results[12] = f32(atomicLoad(&grid_mass_atomic[53u]));
    test_results[13] = f32(atomicLoad(&grid_mass_atomic[54u]));

    test_results[14] = f32(atomicLoad(&grid_mass_atomic[41u]));
    test_results[15] = f32(atomicLoad(&grid_mass_atomic[51u]));

    let grid_pos = particles_position[0] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    test_results[16] = f32(base_node.x);
    test_results[17] = f32(base_node.y);
    test_results[18] = interpolated_velocity.x;
    test_results[19] = interpolated_velocity.y;

    
    test_results[20] = f32(atomicLoad(&grid_momentum_atomic[64u]));
    test_results[21] = f32(atomicLoad(&grid_momentum_atomic[66u]));
    test_results[22] = f32(atomicLoad(&grid_momentum_atomic[68u]));

    test_results[23] = f32(atomicLoad(&grid_momentum_atomic[84u]));
    test_results[24] = f32(atomicLoad(&grid_momentum_atomic[86u]));
    test_results[25] = f32(atomicLoad(&grid_momentum_atomic[88u]));

    test_results[26] = f32(atomicLoad(&grid_momentum_atomic[104u]));
    test_results[27] = f32(atomicLoad(&grid_momentum_atomic[106u]));
    test_results[28] = f32(atomicLoad(&grid_momentum_atomic[108u]));

    test_results[30] = f32(atomicLoad(&grid_momentum_atomic[65u]));
    test_results[31] = f32(atomicLoad(&grid_momentum_atomic[67u]));
    test_results[32] = f32(atomicLoad(&grid_momentum_atomic[69u]));

    test_results[33] = f32(atomicLoad(&grid_momentum_atomic[85u]));
    test_results[34] = f32(atomicLoad(&grid_momentum_atomic[87u]));
    test_results[35] = f32(atomicLoad(&grid_momentum_atomic[89u]));

    test_results[36] = f32(atomicLoad(&grid_momentum_atomic[105u]));
    test_results[37] = f32(atomicLoad(&grid_momentum_atomic[107u]));
    test_results[38] = f32(atomicLoad(&grid_momentum_atomic[109u]));

    
    test_results[40] = f32(idx);
}

// import transfer_p2g.wgsl;
// BEGIN transfer_p2g.wgsl
fn transfer_p2g(p_idx: u32) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    // let affine_matrix = particles_affine_matrix[p_idx];

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            // let affine_velocity = p_velocity + (affine_matrix * distance);

            
            atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * p_velocity.x * weight);
            atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * p_velocity.y * weight);
            // for affine transfer
            // atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);

            // no_atomic_float atomicAdd(&grid_mass_atomic[idx], u32(round(p_mass * weight * MASS_FACTOR)));
            // no_atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u], i32(round(p_mass * p_velocity.x * weight * MOMENTUM_FACTOR)));
            // no_atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], i32(round(p_mass * p_velocity.y * weight * MOMENTUM_FACTOR)));
        }
    }
}
// END transfer_p2g.wgsl
// import transfer_g2p.wgsl;
// BEGIN transfer_g2p.wgsl
fn transfer_g2p(p_idx: u32) -> vec2f {
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    var interpolated_velocity = vec2f(0.0);

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            
            interpolated_velocity += weight * grid_velocity[idx];
        }
    }
    return interpolated_velocity;
}

// END transfer_g2p.wgsl
// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
// TODO calculate momentum factor
const MOMENTUM_FACTOR: f32 = 1e-2; 
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR; 
const INV_MOMENTUM_FACTOR: f32 = 1 / MOMENTUM_FACTOR;

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

struct SimInfo {
    timestep: u32,
    dt: f32,
    elapsed_time: f32,
    number_particles: u32,
    elevation_threshold: f32,
    max_velocity: f32,
    max_flow_thickness: f32,
    flags: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;

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
    n0: f32,
    i0: f32,
    mu0: f32,
    mu2: f32,
    grain_diameter: f32,
    internal_friction_angle: f32,
    basal_friction_angle: f32,
    cfl: f32,
    cell_size: f32,
    min_slope_angle: f32,
    max_slope_angle: f32,
    min_elevation: f32,
    velocity_threshold: f32,
    roughness_threshold: f32,
    flags: u32,
};

struct AtomicValues {
    peak_velocity: atomic<u32>,
    peak_flow_thickness: atomic<u32>,
    expected_max_velocity: atomic<u32>,
    travel_length: atomic<u32>,
    release_volume: atomic<u32>,
    number_release_cells: atomic<u32>,
    number_release_particles: atomic<u32>,
    stopped_particles: atomic<u32>,
};

struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

fn is_nan(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    return (bits & 0x7F800000u) == 0x7F800000u
          && (bits & 0x007FFFFFu) != 0u;
}

fn is_inf(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    return (bits == 0x7F800000u || bits == 0xFF800000u);
}

fn is_finite(x: f32) -> bool {
    return !is_nan(x) && !is_inf(x);
}

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cell3_to_uv(cell: vec3u) -> vec2f {
    return (vec2f(cell.xy) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cellf_to_uv(cell: vec2f) -> vec2f {
    return (cell + 0.5) / vec2f(sim_settings.grid_shape);
}

fn position_to_uv(position: vec2f) -> vec2f {
    return (position.xy) / (vec2f(sim_settings.world_size)); // add some padding to ensure particles outside the world bounds are still captured in the simulation info
}

fn position_to_idx(position: vec2f) -> u32 {
    let uv = position_to_uv(position);
    return uv_to_idx(uv);
}

fn position_to_cell(position: vec2f) -> vec2u {
    return vec2u(floor(position / sim_settings.cell_size));
}

fn uv_to_cell(uv: vec2f) -> vec2u {
    let epsilon = 1e-5f; // A tiny offset to counteract negative rounding bias
    let scaled_uv = uv * vec2f(sim_settings.grid_shape) + epsilon;
    let max_bound = vec2f(sim_settings.grid_shape - 1u);

    return vec2u(clamp(scaled_uv, vec2f(0.0), max_bound));
}

fn uv_to_idx(uv: vec2f) -> u32 {
    let cell = uv_to_cell(uv);
    // return cell.x * sim_settings.grid_shape.y + cell.y;
    return (cell.y % sim_settings.grid_shape.y * sim_settings.grid_shape.x +
              (cell.x % sim_settings.grid_shape.x));
}

fn x_y_to_idx(x: u32, y: u32) -> u32 {
    return y * sim_settings.grid_shape.x + x;
}

fn xy_to_idx(xy: vec2<u32>) -> u32 {
    return xy.y * sim_settings.grid_shape.x + xy.x;
}

fn idx_to_xy(idx: u32) -> vec2<u32> {
    let x = idx % sim_settings.grid_shape.x;
    let y = idx / sim_settings.grid_shape.x;
    return vec2u(x, y);
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

fn calculate_weight(distance: vec2f) -> f32 {
    return quadratic_weight(distance.x) * quadratic_weight(distance.y);
}

fn calculate_distance_to_node(particle_position: vec2f, node_position: vec2u) -> vec2f {
    return particle_position - vec2f(node_position);
}

fn get_base_node(grid_pos: vec2f) -> vec2u {
    return vec2u(floor(grid_pos - vec2f(0.5)));
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