@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_velocity: array<vec2<f32>>;
@group(0) @binding(4) var<storage> particles_mass: array<f32>;
// atomic_float @group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<f32>>;
// atomic_float @group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<f32>>; 
@group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<u32>>;  // no_atomic_float 
@group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<i32>>;  // no_atomic_float 
@group(0) @binding(7) var<storage> particles_affine_matrix: array<mat2x2f>;
@group(0) @binding(8) var<storage> particles_velocity_z: array<f32>;
// atomic_float @group(0) @binding(9) var<storage, read_write> grid_momentum_z_atomic: array<atomic<f32>>;
@group(0) @binding(9) var<storage, read_write> grid_momentum_z_atomic: array<atomic<i32>>;  // no_atomic_float 
@group(0) @binding(10) var<storage> particles_affine_z: array<vec2f>;
// 3D model only: detached (flying) particles deposit nothing
@group(0) @binding(11) var<storage> particles_state: array<u32>;
// 3D model only: cells that hold a depositing particle in this step
@group(0) @binding(12) var<storage, read_write> grid_occupied: array<atomic<u32>>;

override WG_SIZE_1D: u32 = 1u;
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn p2g(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles {
        return;
    }
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }    
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    if sim_settings.model_type == 0 && !use_particle_interaction {
        return;
    }
    let cell = position_to_cell(particles_position[id.x].xy);
    if cell.x < 3 || cell.x >= (sim_settings.grid_shape.x - 3) || cell.y < 3 || cell.y >= (sim_settings.grid_shape.y - 3) {
        return;
    }
    
    transfer_p2g(id.x);
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn p2g_2d(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles || (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    let cell = position_to_cell(particles_position[id.x]);
    if cell.x < 3 || cell.x >= sim_settings.grid_shape.x - 3 || cell.y < 3 || cell.y >= sim_settings.grid_shape.y - 3 {
        return;
    }
    transfer_p2g_with_affine(id.x, particles_affine_matrix[id.x]);
}

// 3D Cartesian APIC transfer: xy through the shared transfer, the z
// component with the z row of the affine matrix on the same stencil
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn p2g_3d(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles || (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    let cell = position_to_cell(particles_position[id.x]);
    if cell.x < 3 || cell.x >= sim_settings.grid_shape.x - 3 || cell.y < 3 || cell.y >= sim_settings.grid_shape.y - 3 {
        return;
    }
    if (particles_state[id.x] & PARTICLE_FLYING) != 0u {
        return;
    }
    transfer_p2g_3d(id.x, particles_affine_matrix[id.x]);
    if (sim_settings.flags & SIM_FLAG_GRID_OCCUPIED) != 0u {
        atomicOr(&grid_occupied[xy_to_idx(cell)], 1u);
    }
}

// import transfer_p2g_z.wgsl;
// BEGIN transfer_p2g_z.wgsl
// 3D APIC transfer: mass, xy momentum and the z momentum (with the z row of
// the affine matrix) in one pass over the stencil, so the 3D model computes
// the weights once. Pure library module: the importing module provides the
// particle buffers, grid_mass_atomic, grid_momentum_atomic and
// grid_momentum_z_atomic, and imports transfer_p2g.wgsl (quantize_*).
fn transfer_p2g_3d(p_idx: u32, affine_matrix: mat2x2f) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    let p_velocity_z = particles_velocity_z[p_idx];
    let affine_z = particles_affine_z[p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let affine_velocity = p_velocity - affine_matrix * distance;
            let affine_velocity_z = p_velocity_z - dot(affine_z, distance);
            let node_mass = p_mass * weight;
            // atomic_float atomicAdd(&grid_mass_atomic[idx], node_mass);
            // resting particles carry no momentum: skip their zero adds
            // atomic_float let momentum = node_mass * vec3f(affine_velocity, affine_velocity_z);
            // atomic_float if momentum.x != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u], momentum.x); }
            // atomic_float if momentum.y != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], momentum.y); }
            // atomic_float if momentum.z != 0.0 { atomicAdd(&grid_momentum_z_atomic[idx], momentum.z); }
            let quantized_mass = quantize_mass(node_mass); // no_atomic_float
            if quantized_mass == 0u { continue; } // no momentum without mass, see quantize_mass // no_atomic_float
            atomicAdd(&grid_mass_atomic[idx], quantized_mass); // no_atomic_float
            let quantized_x = quantize_momentum_component(node_mass * affine_velocity.x * momentum_factor()); // no_atomic_float
            let quantized_y = quantize_momentum_component(node_mass * affine_velocity.y * momentum_factor()); // no_atomic_float
            let quantized_z = quantize_momentum_component(node_mass * affine_velocity_z * momentum_factor()); // no_atomic_float
            if quantized_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], quantized_x); } // no_atomic_float
            if quantized_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantized_y); } // no_atomic_float
            if quantized_z != 0 { atomicAdd(&grid_momentum_z_atomic[idx], quantized_z); } // no_atomic_float
        }
    }
}

// END transfer_p2g_z.wgsl
// import transfer_p2g.wgsl;
// BEGIN transfer_p2g.wgsl
fn quantize_momentum_component(value: f32) -> i32 {
    // Keep conversion inside i32 range before atomicAdd.
    let i32_max_f = 2147483647.0;
    let clamped = clamp(value, -i32_max_f, i32_max_f);
    return i32(round(clamped));
}

// Quantized mass contribution of the integer grid (mass_factor() in
// quantization.wgsl). A contribution whose mass rounds to zero must not deposit
// momentum either, or it would leave momentum without mass and an unbounded
// node velocity. With the range-based factors that needs a contribution faster
// than 2 * MAX_FLOW_VELOCITY, so the guard only protects against misuse.
fn quantize_mass(value: f32) -> u32 {
    return u32(round(value * mass_factor()));
}

fn transfer_p2g(p_idx: u32) {
    transfer_p2g_with_affine(p_idx, mat2x2f(vec2f(0.0), vec2f(0.0)));
}

fn transfer_p2g_with_affine(p_idx: u32, affine_matrix: mat2x2f) {
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
            let affine_velocity = p_velocity - affine_matrix * distance;

            
            
            // atomic_float atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            // resting particles carry no momentum: skip their zero adds
            // atomic_float let momentum = p_mass * weight * affine_velocity;
            // atomic_float if momentum.x != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u], momentum.x); }
            // atomic_float if momentum.y != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], momentum.y); }
            // for affine transfer
            // atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);

            let quantized_mass = quantize_mass(p_mass * weight); // no_atomic_float
            if quantized_mass == 0u { continue; } // no_atomic_float
            atomicAdd(&grid_mass_atomic[idx], quantized_mass); // no_atomic_float
            let momentum_x = p_mass * affine_velocity.x * weight * momentum_factor(); // no_atomic_float
            let momentum_y = p_mass * affine_velocity.y * weight * momentum_factor(); // no_atomic_float
            // resting particles carry no momentum; skipping zero adds saves most of their atomics
            let quantized_x = quantize_momentum_component(momentum_x); // no_atomic_float
            let quantized_y = quantize_momentum_component(momentum_y); // no_atomic_float
            if quantized_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], quantized_x); } // no_atomic_float
            if quantized_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantized_y); } // no_atomic_float
        }
    }
}
// END transfer_p2g.wgsl

// import sim_settings.wgsl;
// BEGIN sim_settings.wgsl
// Simulation settings uniform (group 0, binding 0) and the values derived from
// it. Must mirror the Rust POD layout of SimSettings.

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

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
    // 3D lift-off: terminal velocity (m/s) of the quadratic air drag in
    // flight; 0 disables the drag
    flight_terminal_velocity: f32,
    // forest model (forest.wgsl), read only with SIM_FLAG_FOREST
    forest_drag_coefficient: f32,
    forest_trunk_diameter: f32,
    forest_crown_diameter: f32,
    forest_tree_height: f32,
    forest_modulus_of_rupture: f32,
    forest_decay_coefficient: f32,
    forest_release_max_cover: f32,
    // entrainment (entrainment.wgsl), read only with SIM_FLAG_ENTRAINMENT by
    // the DAMPM global model: erosion energy e_b (J/kg), density (kg/m3) and
    // uniform depth (m) of the erodible snow cover
    erosion_energy: f32,
    entrainment_density: f32,
    entrainment_thickness: f32,
    // entrainment rate law (EntrainmentModel): 0 erosion energy, 1 excess
    // shear stress, 2 velocity-proportional, with their parameters tau_c (Pa),
    // k (s/m) and kappa (-)
    entrainment_model: u32,
    critical_shear_stress: f32,
    shear_erosion_coefficient: f32,
    entrainment_velocity_coefficient: f32,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// SimFlags::GRID_OCCUPIED (settings.rs): p2g marks the cells holding a
// depositing particle in grid_occupied and only those record peak maps
const SIM_FLAG_GRID_OCCUPIED: u32 = 1u << 9u;

// SimFlags::FOREST (settings.rs): a forest cover is loaded; the DAMPM global
// model decelerates the flow by the drag of the standing stems
const SIM_FLAG_FOREST: u32 = 1u << 10u;

fn forest_enabled() -> bool {
    return (sim_settings.flags & SIM_FLAG_FOREST) != 0u;
}

// SimFlags::ENTRAINMENT (settings.rs): the DAMPM global model entrains the
// erodible snow cover (entrainment.wgsl)
const SIM_FLAG_ENTRAINMENT: u32 = 1u << 3u;

fn entrainment_enabled() -> bool {
    return (sim_settings.flags & SIM_FLAG_ENTRAINMENT) != 0u;
}

// Gully walls (reference surface, wall friction) need a reference surface at
// least one cell wide; must match SimSettings::gully_walls_enabled.
fn gully_walls_enabled() -> bool {
    return sim_settings.reference_surface_length >= sim_settings.cell_size;
}

// length in steps of the no-new-cells rolling window: the fastest material
// moves at most cfl cells per step, so round(travel / cfl) steps cover the same
// front travel at any cfl; at least 10 steps
fn no_new_cells_window() -> u32 {
    return max(10u, u32(round(sim_settings.no_new_cells_travel / sim_settings.cfl)));
}

// END sim_settings.wgsl
// import sim_info.wgsl;
// BEGIN sim_info.wgsl
// Simulation state shared between the passes: the SimInfo buffer with its
// status flags and the atomic accumulators. Must mirror the Rust layouts.

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
    // ranges of the integer-atomic grid in units of the reference node (see
    // quantization.wgsl); previous_mass_scale quantized the last step's grid
    mass_scale: f32,
    momentum_scale: f32,
    previous_mass_scale: f32,
    _pad0: u32,
    _pad1: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
// a particle exceeded MAX_PARTICLE_SPEED and was stopped
const SIM_INFO_UNPHYSICAL_SPEED: u32 = 1u << 4u;
// a node filled more than QUANTIZATION_OVERFLOW_FILL of an integer-atomic
// grid range, which it may have overflowed (see quantization.wgsl)
const SIM_INFO_QUANTIZATION_OVERFLOW: u32 = 1u << 5u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

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
    // largest node mass (u32 units) and momentum component (|i32| units) of
    // the integer-atomic grid this step, read and reset by reset_grid
    max_mass_units: atomic<u32>,
    max_momentum_units: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

// END sim_info.wgsl
// import particle_state.wgsl;
// BEGIN particle_state.wgsl
// Particle state flags and the reversible resting state. Needs sim_settings.

// Fastest particle speed (m/s) taken as physical. Dense flow avalanches reach
// about 60-80 m/s and the integer atomic grid holds momentum up to
// MAX_FLOW_VELOCITY = 100 m/s for a full MAX_FLOW_THICKNESS node (faster for
// thinner nodes); a faster particle comes from a numerical blow-up (e.g. a cliff of no-data elevations) and is stopped.
// Mirrors compute_core::MAX_PARTICLE_SPEED.
const MAX_PARTICLE_SPEED: f32 = 150.0;
// stopped for exceeding MAX_PARTICLE_SPEED
const PARTICLE_UNPHYSICAL_SPEED: u32 = 1u << 25u;
// reversible low-speed state; the low bits (below PARTICLE_UNPHYSICAL_SPEED)
// hold the rest start step while resting and the consecutive slow-step count
// while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
const PARTICLE_FLYING: u32 = 1u << 27u;
const PARTICLE_OUT_OF_BOUNDS: u32 = 1u << 28u;
const PARTICLE_IS_NAN: u32 = 1u << 29u;
const PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 30u;
const PARTICLE_STOPPED: u32 = 1u << 31u;

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_UNPHYSICAL_SPEED - 1u;
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

// END particle_state.wgsl
// import grid_index.wgsl;
// BEGIN grid_index.wgsl
// Conversions between world positions, cells, uv coordinates and flat grid
// indices. Needs sim_settings.

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
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

// END grid_index.wgsl
// import bspline.wgsl;
// BEGIN bspline.wgsl
// Quadratic B-spline weights of the particle <-> grid transfers. Needs
// sim_settings (calculate_weight_gradient).

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

fn calculate_distance_to_node(particle_position: vec2f, node_position: vec2u) -> vec2f {
    return particle_position - vec2f(node_position);
}

fn get_base_node(grid_pos: vec2f) -> vec2u {
    return vec2u(floor(grid_pos - vec2f(0.5)));
}

// END bspline.wgsl

// import quantization.wgsl;
// BEGIN quantization.wgsl
// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The ranges adapt to the flow. reset_grid doubles a range once the
// fullest node of the last step filled half of it, and halves it once that
// node stayed below an eighth, so the integers resolve thin flows finely and
// still hold deep deposits. sim_info.mass_scale counts the mass range in
// reference nodes: a MAX_FLOW_THICKNESS deep column over a cell (density *
// thickness * cell_size^2); sim_info.momentum_scale counts the momentum range
// in that column moving at MAX_FLOW_VELOCITY. Both start at 1. Needs
// sim_settings and sim_info; must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 50.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
// fill of the fullest node at which a range doubles, and below which it halves
const QUANTIZATION_GROW_FILL: f32 = 0.5;
const QUANTIZATION_SHRINK_FILL: f32 = 0.125;
// fill at which the fullest node probably wrapped around
const QUANTIZATION_OVERFLOW_FILL: f32 = 0.9;
const MIN_MASS_SCALE: f32 = 1.0 / 256.0;
const MAX_QUANTIZATION_SCALE: f32 = 256.0;
// Smallest momentum range relative to the mass range: the full mass range at
// MAX_FLOW_VELOCITY / 8 = 12.5 m/s. A flow starting from rest (no momentum to
// measure) cannot outgrow it within one step.
const MIN_MOMENTUM_PER_MASS_SCALE: f32 = 0.125;

// mass (kg) of the reference node, a MAX_FLOW_THICKNESS deep column
fn reference_node_mass() -> f32 {
    return MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// largest node mass (kg) and momentum component (kg m/s) the integers hold
fn max_node_mass() -> f32 {
    return sim_info.mass_scale * reference_node_mass();
}
fn max_node_momentum() -> f32 {
    return sim_info.momentum_scale * reference_node_mass() * MAX_FLOW_VELOCITY;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// kg per unit of the last step's grid mass (MPMDAC's grid_mass_previous)
fn previous_inv_mass_factor() -> f32 {
    return sim_info.previous_mass_scale * reference_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / max_node_momentum();
}
fn inv_momentum_factor() -> f32 {
    return max_node_momentum() / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (12.5 g for 5 m cells and 200 kg/m3 at the start
// range). Lighter nodes count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// next range scale for a fullest node filling `fill` of the current range
fn adapted_quantization_scale(scale: f32, fill: f32) -> f32 {
    if fill >= QUANTIZATION_GROW_FILL {
        return scale * 2.0;
    }
    if fill < QUANTIZATION_SHRINK_FILL {
        return scale * 0.5;
    }
    return scale;
}

// END quantization.wgsl
