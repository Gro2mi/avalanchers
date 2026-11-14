struct RelaxParams {
    num_particles: u32,
    // fraction of the overlap corrected per iteration
    move_factor: f32,
    // per-iteration displacement limit as a fraction of the target spacing
    max_step_fraction: f32,
    _padding: u32,
};

@group(0) @binding(1) var<uniform> relax_params: RelaxParams;
// written by the relax kernel; the previous iteration's positions are read from
// the snapshot so every thread sees a consistent state
@group(0) @binding(2) var<storage, read_write> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_position_snapshot: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> particles_velocity: array<vec2<f32>>;
@group(0) @binding(5) var<storage> release_areas: array<f32>;
// fixed hash grid over the simulation cells, chained through particle_next:
// grid_head[cell] is the index of the first particle in the cell or
// LINKED_LIST_END; lets each particle find its neighbors in O(cell occupancy)
// instead of scanning all particles
@group(0) @binding(6) var<storage, read_write> grid_head: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> particle_next: array<u32>;
@group(0) @binding(8) var dem_texture: texture_2d<f32>;
@group(0) @binding(9) var tex_sampler: sampler;
// start elevation of each particle, set by initialize_particles at the seed position
@group(0) @binding(10) var<storage, read_write> start_elevation: array<f32>;

const LINKED_LIST_END: u32 = 0xFFFFFFFFu;
// must match RELEASE_THICKNESS_THRESHOLD in lib.rs and initialize_particles.wgsl
const RELEASE_THICKNESS_THRESHOLD: f32 = 0.01;

override WG_SIZE_1D: u32 = 1u;

fn clamp_to_grid(cellf: vec2f) -> vec2u {
    let max_cell = vec2f(sim_settings.grid_shape) - 1.0;
    return vec2u(clamp(cellf, vec2f(0.0), max_cell));
}

// true if p lies inside a cell with snow (a release cell)
fn is_release_position(p: vec2f) -> bool {
    let cellf = floor(p / sim_settings.cell_size);
    if (cellf.x < 0.0 || cellf.y < 0.0
        || cellf.x >= f32(sim_settings.grid_shape.x)
        || cellf.y >= f32(sim_settings.grid_shape.y)) {
        return false;
    }
    return release_areas[xy_to_idx(vec2u(cellf))] > RELEASE_THICKNESS_THRESHOLD;
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn relax_clear_grid(@builtin(global_invocation_id) cell: vec3<u32>) {
    if cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y {
        return;
    }
    atomicStore(&grid_head[xy_to_idx(cell.xy)], LINKED_LIST_END);
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }
    let cell = clamp_to_grid(floor(particles_position_snapshot[index] / sim_settings.cell_size));
    // lock-free push onto the cell's linked list
    particle_next[index] = atomicExchange(&grid_head[xy_to_idx(cell)], index);
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }

    // hexagonal packing: each particle owns A_particle = A_cell / N =
    // sqrt(3)/2 * h^2, so the equilibrium center spacing is h; h <= cell_size
    // for N >= 2, so the 3x3 cell neighborhood always contains every particle
    // within the target spacing
    let particles_per_cell = max(sim_settings.released_particles_per_cell, 1u);
    let target_spacing = sim_settings.cell_size
        * sqrt(2.0 / (sqrt(3.0) * f32(particles_per_cell)));

    let position = particles_position_snapshot[index];
    let base_cellf = floor(position / sim_settings.cell_size);
    var displacement = vec2f(0.0);

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cellf = base_cellf + vec2f(f32(dx), f32(dy));
            // skip cells outside the grid instead of clamping, so edge cells
            // are not visited (and their particles not repelled) multiple times
            if (cellf.x < 0.0 || cellf.y < 0.0
                || cellf.x >= f32(sim_settings.grid_shape.x)
                || cellf.y >= f32(sim_settings.grid_shape.y)) {
                continue;
            }
            var j = atomicLoad(&grid_head[xy_to_idx(vec2u(cellf))]);
            loop {
                if (j == LINKED_LIST_END) {
                    break;
                }
                if (j != index) {
                    let diff = position - particles_position_snapshot[j];
                    let dist = length(diff);
                    // repulsion if closer than the desired equilibrium spacing
                    if (dist < target_spacing && dist > 1e-4) {
                        let overlap = target_spacing - dist;
                        displacement += diff / dist * (relax_params.move_factor * overlap);
                    }
                }
                j = particle_next[j];
            }
        }
    }

    // clamp the per-iteration displacement for stability
    let max_step = relax_params.max_step_fraction * target_spacing;
    let displacement_len = length(displacement);
    if (displacement_len > max_step) {
        displacement = displacement / displacement_len * max_step;
    }

    // a step leaving the release area is reduced to one axis, larger component
    // first, so particles pushed against the edge still slide along it
    let step_x = vec2f(displacement.x, 0.0);
    let step_y = vec2f(0.0, displacement.y);
    let primary = select(step_y, step_x, abs(displacement.x) >= abs(displacement.y));
    let secondary = select(step_x, step_y, abs(displacement.x) >= abs(displacement.y));
    if (is_release_position(position + displacement)) {
        particles_position[index] = position + displacement;
    } else if (is_release_position(position + primary)) {
        particles_position[index] = position + primary;
    } else if (is_release_position(position + secondary)) {
        particles_position[index] = position + secondary;
    } else {
        // no admissible step: stay at the previous position
        particles_position[index] = position;
        particles_velocity[index] = vec2f(0.0);
    }
}

// run once after the last iteration: the particles moved away from their seed
// position, so the start elevation is re-sampled at the relaxed position
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_sample_elevation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }
    start_elevation[index] = textureSampleLevel(dem_texture, tex_sampler, position_to_uv(particles_position[index]), 0).x;
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const H_FACTOR: f32 = 1e6;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_H_FACTOR: f32 = 1 / H_FACTOR;
// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The factors fill the integer range with the largest node values a
// flow can reach: a MAX_FLOW_THICKNESS deep column over a cell
// (density * thickness * cell_size^2) moving at MAX_FLOW_VELOCITY, times
// QUANTIZATION_SAFETY (which also covers the slope factor of the node mass).
// They scale with the cell size like the particle masses, so a particle of a
// release h_r thick with N particles per cell holds about
// U32_RANGE * h_r / (N * QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS) mass units
// (2e6 for 1 m and 64 per cell) at every cell size. Must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 30.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
const QUANTIZATION_SAFETY: f32 = 2.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
fn max_node_mass() -> f32 {
    return QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / (max_node_mass() * MAX_FLOW_VELOCITY);
}
fn inv_momentum_factor() -> f32 {
    return max_node_mass() * MAX_FLOW_VELOCITY / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (15 g for 5 m cells and 200 kg/m3). Lighter nodes
// count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// depth-integrated internal force (h * sigma * grad_w * area, ~1e4..1e5 N per node contribution)
const FORCE_FACTOR: f32 = 1e-3; // i32 limit is 2.1e6 N per node
const INV_FORCE_FACTOR: f32 = 1 / FORCE_FACTOR;
// weak-form pressure sum m_p h_p / 2 grad_w (kg) of the curvilinear model,
// node sum = node_mass * |grad h|: 0.001 kg resolves thin front nodes
//   (rho*cell^2*h*J ~ 6e4 kg * 1 * 1e3 = 6e7 < 2.1e9)
const PRESSURE_FACTOR: f32 = 1e3;
const INV_PRESSURE_FACTOR: f32 = 1 / PRESSURE_FACTOR;

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
    grid_elapsed_time: f32,
    travel_length_2d: f32,
    travel_length_3d: f32,
    travel_vertical_drop: f32,
    travel_angle: f32,
    _first_center_of_mass_timestep: u32,
    _last_center_of_mass_timestep: u32,
    _pad: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

// reversible low-speed state; the low bits hold the rest start step while resting
// and the consecutive slow-step count while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
const PARTICLE_FLYING: u32 = 1u << 27u;
const PARTICLE_OUT_OF_BOUNDS: u32 = 1u << 28u;
const PARTICLE_IS_NAN: u32 = 1u << 29u;
const PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 30u;
const PARTICLE_STOPPED: u32 = 1u << 31u;

struct SimSettings {
    num_steps: u32,
    model_type: u32,
    friction_model: u32,
    released_particles_per_cell: u32,
    grid_shape: vec2u,
    world_size: vec2f,
    snow_density: f32,
    release_thickness: f32,
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
    release_max_elevation: f32,
    peak_flow_thickness_threshold: f32,
    // MPMDAC constitutive model (must mirror the Rust POD layout)
    constitutive_model: u32,
    shear_modulus: f32,
    hardening_modulus: f32,
    // MPMDAC compressibility; bulk_modulus 0 = incompressible
    bulk_modulus: f32,
    compaction_pressure: f32,
    // stop once total momentum < fraction * its peak; 0 disables (curvilinear)
    momentum_stop_fraction: f32,
    // front travel in cells covered by the no-new-cells stop window
    no_new_cells_travel: f32,
    // Gaussian width (m) of the curvilinear reference surface; 0 disables it
    reference_surface_length: f32,
    // snow water equivalent (kg/m2) of the release cells; NaN if unset
    release_swe: f32,
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
    // resting particles counted during the current step, reset by update_sim_info
    resting_particles: atomic<u32>,
    // total grid momentum (kg m/s) of the current step as a 64-bit sum, reset by update_sim_info
    momentum_lo: atomic<u32>,
    momentum_hi: atomic<u32>,
    // f32 bits of the largest total momentum so far
    peak_momentum: atomic<u32>,
};

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_RESTING - 1u;
    var resting = (previous_state & PARTICLE_RESTING) != 0u;
    var low_bits = previous_state & low_mask;
    if resting {
        if speed > 2.0 * sim_settings.velocity_threshold {
            resting = false;
            low_bits = 0u;
        }
    } else if speed < sim_settings.velocity_threshold {
        low_bits = min(low_bits + 1u, REST_STEPS);
        if low_bits >= REST_STEPS {
            resting = true;
            low_bits = timestep & low_mask;
        }
    } else {
        low_bits = 0u;
    }
    return (previous_state & ~(PARTICLE_RESTING | low_mask))
        | select(0u, PARTICLE_RESTING, resting)
        | low_bits;
}

struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// length in steps of the no-new-cells rolling window: the fastest material
// moves at most cfl cells per step, so round(travel / cfl) steps cover the same
// front travel at any cfl; at least 10 steps
fn no_new_cells_window() -> u32 {
    return max(10u, u32(round(sim_settings.no_new_cells_travel / sim_settings.cfl)));
}

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

fn position3_to_cell(position: vec3f) -> vec2u {
    return position_to_cell(position.xy);
}

fn position_to_cell(position: vec2f) -> vec2u {
    return vec2u(
        floor(position.xy / sim_settings.cell_size)
    );
}

fn cell_center_xy(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) * sim_settings.cell_size;
}

fn position_to_uv(position: vec2f) -> vec2f {
    return (position.xy) / (vec2f(sim_settings.world_size)); // add some padding to ensure particles outside the world bounds are still captured in the simulation info
}

fn position_to_idx(position: vec2f) -> u32 {
    let uv = position_to_uv(position);
    return uv_to_idx(uv);
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

// derivative of quadratic_weight with respect to its (cell-unit) argument
fn quadratic_weight_gradient(d: f32) -> f32 {
    let abs_d = abs(d);
    if abs_d < 0.5 {
        return -2.0 * d;
    } else if abs_d < 1.5 {
        return -sign(d) * (1.5 - abs_d);
    }
    return 0.0;
}

// physical gradient of the 2D B-spline weight, in 1/m
fn calculate_weight_gradient(distance: vec2f) -> vec2f {
    return vec2f(
        quadratic_weight_gradient(distance.x) * quadratic_weight(distance.y),
        quadratic_weight(distance.x) * quadratic_weight_gradient(distance.y)
    ) / sim_settings.cell_size;
}

fn determinant_2x2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

fn identity_2x2() -> mat2x2<f32> {
    return mat2x2<f32>(vec2f(1.0, 0.0), vec2f(0.0, 1.0));
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
