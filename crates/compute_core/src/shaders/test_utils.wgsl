@group(0) @binding(1) var<storage, read_write> test_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> atomic_values: AtomicValues;


@compute @workgroup_size(1)
fn test_utils(@builtin(global_invocation_id) id: vec3u) {
    let nan = bitcast<f32>(0x7FC00000u);
    let neg_nan = bitcast<f32>(0xFF800001u);
    let inf = bitcast<f32>(0x7F800000u);
    let neg_inf = bitcast<f32>(0xFF800000u);
    let low_val: f32 = 3.1415;
    let high_val: f32 = 42.0;
    atomicStore(&atomic_values.peak_velocity, bitcast<u32>(low_val));
    test_output[0] = bitcast<f32>(atomicLoad(&atomic_values.peak_velocity));
    atomicMax(&atomic_values.peak_velocity, bitcast<u32>(high_val));
    test_output[1] = bitcast<f32>(atomicLoad(&atomic_values.peak_velocity));
    atomicMax(&atomic_values.peak_velocity, bitcast<u32>(low_val));
    test_output[2] = bitcast<f32>(atomicLoad(&atomic_values.peak_velocity));

    // is_nan tests
    test_output[3] = f32(is_nan(nan)); // NaN -> true
    test_output[4] = f32(is_nan(neg_nan)); // -NaN -> true
    test_output[5] = f32(is_nan(inf)); // Inf -> false
    test_output[6] = f32(is_nan(neg_inf)); // -Inf -> false
    test_output[7] = f32(is_nan(1.2345)); // valid f32 -> false

    // is_inf tests
    test_output[8] = f32(is_inf(inf)); // Inf -> true
    test_output[9] = f32(is_inf(neg_inf)); // -Inf -> true
    test_output[10] = f32(is_inf(nan)); // NaN -> false
    test_output[11] = f32(is_inf(neg_nan)); // -NaN -> false
    test_output[12] = f32(is_inf(1.2345)); // valid f32 -> false

    // is_finite tests
    test_output[13] = f32(is_finite(nan)); // NaN -> false
    test_output[14] = f32(is_finite(neg_nan)); // -NaN -> false
    test_output[15] = f32(is_finite(inf)); // Inf -> false
    test_output[16] = f32(is_finite(neg_inf)); // -Inf -> false
    test_output[17] = f32(is_finite(1.2345)); // valid f32 -> true

    
    atomicStore(&atomic_values.peak_flow_thickness, bitcast<u32>(2.71828));
    atomicStore(&atomic_values.expected_max_velocity, bitcast<u32>(1.618));
    atomicStore(&atomic_values.peak_velocity, bitcast<u32>(1.4142));
    atomicStore(&atomic_values.travel_length, bitcast<u32>(1.732));

    atomicStore(&atomic_values.release_volume, 73u);
    atomicStore(&atomic_values.number_release_cells, 37u);
    atomicStore(&atomic_values.number_release_particles, 42u);
    atomicStore(&atomic_values.stopped_particles, 99u);

}

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