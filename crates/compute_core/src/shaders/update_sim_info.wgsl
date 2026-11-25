
@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(3) var<storage, read_write> new_cells_rolling_window: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn update_sim_info(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    sim_info.grid_elapsed_time = sim_info.elapsed_time;
    sim_info.elapsed_time = sim_info.elapsed_time + sim_info.dt;
    if atomicLoad(&atomic_values.stopped_particles) == sim_info.number_particles {
        sim_info.flags = sim_info.flags | SIM_INFO_ALL_PARTICLES_STOPPED;
        sim_info.flags = sim_info.flags | SIM_INFO_STOPPED;
        return;
    }
    // Grid flow models (terrain-following, curvilinear, 3D)
    let grid_flow_model = sim_settings.model_type <= 1u || sim_settings.model_type == 4u;
    // Stop once 99% of the particles rest or are stopped; the remaining few
    // moving particles do not change the outline materially.
    let resting = atomicLoad(&atomic_values.resting_particles);
    atomicStore(&atomic_values.resting_particles, 0u);
    let settled = atomicLoad(&atomic_values.stopped_particles) + resting;
    if grid_flow_model && f32(settled) >= 0.99 * f32(sim_info.number_particles) {
        sim_info.flags = sim_info.flags | SIM_INFO_AT_REST;
        sim_info.flags = sim_info.flags | SIM_INFO_STOPPED;
        return;
    }
    // Optional: stop once the total momentum of this step falls
    // below momentum_stop_fraction of the largest total momentum so far.
    let momentum = f32(atomicLoad(&atomic_values.momentum_hi)) * 4294967296.0
        + f32(atomicLoad(&atomic_values.momentum_lo));
    atomicStore(&atomic_values.momentum_lo, 0u);
    atomicStore(&atomic_values.momentum_hi, 0u);
    if grid_flow_model && sim_settings.momentum_stop_fraction > 0.0 {
        let peak_momentum = max(bitcast<f32>(atomicLoad(&atomic_values.peak_momentum)), momentum);
        atomicStore(&atomic_values.peak_momentum, bitcast<u32>(peak_momentum));
        if peak_momentum > 0.0 && momentum < sim_settings.momentum_stop_fraction * peak_momentum {
            sim_info.flags = sim_info.flags | SIM_INFO_LOW_MOMENTUM;
            sim_info.flags = sim_info.flags | SIM_INFO_STOPPED;
            return;
        }
    }
    var sum_new_cells: u32 = 0u;
    for (var i: u32 = 0u; i < no_new_cells_window(); i = i + 1u) {
        sum_new_cells = sum_new_cells + new_cells_rolling_window[i];
    }
    if sum_new_cells < 4u {
        sim_info.flags = sim_info.flags | SIM_INFO_NO_NEW_CELLS;
        sim_info.flags = sim_info.flags | SIM_INFO_STOPPED;
        return;
    } 

    let max_flow_thickness = bitcast<f32>(atomicLoad(&atomic_values.peak_flow_thickness));
    sim_info.timestep = sim_info.timestep + 1u;
    new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] = 0u; // reset the count for the new cells in the current timestep
    sim_info.max_velocity = bitcast<f32>(atomicLoad(&atomic_values.peak_velocity));
    // MPM case
    var expected_max_velocity = bitcast<f32>(atomicLoad(&atomic_values.expected_max_velocity));
    // Particle case: no grid shader tracks the CFL speed of the sliding block,
    // so its run maximum is kept here
    if sim_settings.model_type == 3u || expected_max_velocity < 1e-6 {
        expected_max_velocity = max(expected_max_velocity, sim_info.max_velocity + sqrt(g * max_flow_thickness));
        atomicStore(&atomic_values.expected_max_velocity, bitcast<u32>(expected_max_velocity));
    }
    let expected_dt = sim_settings.cfl * sim_settings.cell_size / expected_max_velocity;
    if !is_finite(expected_dt) {
        sim_info.dt = 0.05;
    } else {
        sim_info.dt = expected_dt;
    }
    // The CFL speed follows the current flow instead of the run's all-time
    // maximum, so dt grows again once the flow slows down.
    if grid_flow_model {
        atomicStore(&atomic_values.expected_max_velocity, 0u);
    }
    sim_info.max_flow_thickness = max_flow_thickness;
}

// import constants.wgsl;
// BEGIN constants.wgsl
// Shared constants without dependencies.
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// END constants.wgsl
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
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

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
// import float_checks.wgsl;
// BEGIN float_checks.wgsl
// NaN / infinity checks on the bit pattern (comparisons may be optimized away).

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

// END float_checks.wgsl