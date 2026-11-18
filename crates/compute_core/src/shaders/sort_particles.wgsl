// Counting sort of particle indices by 8x8-cell tiles. Only an index
// permutation is written, so particle buffers and identities stay unchanged.
// Passes: sort_clear_bins -> sort_count_particles -> sort_scan_bins -> sort_scatter_particles.
@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> sort_bins: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> particle_bin_rank: array<u32>;
@group(0) @binding(5) var<storage, read_write> sorted_particle_index: array<u32>;

const SORT_TILE: u32 = 8u;
const SCAN_WG_SIZE: u32 = 256u;

fn sort_bins_shape() -> vec2u {
    return (sim_settings.grid_shape + vec2u(SORT_TILE - 1u)) / SORT_TILE;
}

fn number_sort_bins() -> u32 {
    let shape = sort_bins_shape();
    return shape.x * shape.y;
}

fn particle_sort_bin(p_idx: u32) -> u32 {
    let position = particles_position[p_idx];
    // NaN or out-of-grid positions (terminally stopped particles) go to a valid bin
    var cell = vec2u(0u);
    if all(position == position) {
        let max_cell = vec2f(sim_settings.grid_shape - vec2u(1u));
        cell = vec2u(clamp(floor(position / sim_settings.cell_size), vec2f(0.0), max_cell));
    }
    let tile = cell / SORT_TILE;
    return tile.y * sort_bins_shape().x + tile.x;
}

override WG_SIZE_1D: u32 = 1u;

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn sort_clear_bins(@builtin(global_invocation_id) id: vec3u) {
    if id.x < number_sort_bins() {
        atomicStore(&sort_bins[id.x], 0u);
    }
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn sort_count_particles(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles {
        return;
    }
    particle_bin_rank[id.x] = atomicAdd(&sort_bins[particle_sort_bin(id.x)], 1u);
}

var<workgroup> scan_sums: array<u32, SCAN_WG_SIZE>;

// exclusive prefix sum of the bin counts in place, in a single workgroup
@compute @workgroup_size(SCAN_WG_SIZE, 1, 1)
fn sort_scan_bins(@builtin(local_invocation_id) lid: vec3u) {
    let bins = number_sort_bins();
    let chunk = (bins + SCAN_WG_SIZE - 1u) / SCAN_WG_SIZE;
    let start = min(lid.x * chunk, bins);
    let end = min(start + chunk, bins);
    var sum = 0u;
    for (var i = start; i < end; i++) {
        sum += atomicLoad(&sort_bins[i]);
    }
    scan_sums[lid.x] = sum;
    workgroupBarrier();
    // Hillis-Steele inclusive scan of the chunk sums
    for (var offset = 1u; offset < SCAN_WG_SIZE; offset *= 2u) {
        var value = scan_sums[lid.x];
        if lid.x >= offset {
            value += scan_sums[lid.x - offset];
        }
        workgroupBarrier();
        scan_sums[lid.x] = value;
        workgroupBarrier();
    }
    var running = scan_sums[lid.x] - sum;
    for (var i = start; i < end; i++) {
        let count = atomicLoad(&sort_bins[i]);
        atomicStore(&sort_bins[i], running);
        running += count;
    }
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn sort_scatter_particles(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles {
        return;
    }
    let offset = atomicLoad(&sort_bins[particle_sort_bin(id.x)]);
    sorted_particle_index[offset + particle_bin_rank[id.x]] = id.x;
}

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
