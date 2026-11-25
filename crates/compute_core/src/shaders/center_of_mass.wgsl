// Center of mass of the flattened grid mass buffer, in three kernels sharing
// this module:
// - center_of_mass_seed: seeds the blob label buffer for the biggest-blob
//   mode. Every mass-holding cell gets its own cell index as label, all
//   others are marked NO_BLOB.
// - center_of_mass_propagate: propagates the minimum blob label across each
//   blob after the seed. Each invocation performs a fixed number of rounds of
//   8-neighbor hooking plus pointer compression (adopting the label that its
//   own label points to), which shrinks the remaining label distance
//   exponentially per round - long, sinuous blobs converge in a handful of
//   rounds instead of one pass per path length.
// - center_of_mass_partials: whole-grid mode only, one workgroup per
//   WG_SIZE_2D x WG_SIZE_2D tile of the grid sums the mass and moment of its
//   tile into tile_partials, so the final stage reduces a few
//   hundred partial sums instead of every cell on a single workgroup.
// - compute_center_of_mass: the final reduction stage, dispatched as exactly
//   one workgroup after seed and propagate have filled the label buffer
//   (both early-out when the whole-grid mode is selected).
//
// Two modes, selected with sim_settings flag bit 4 (the
// center_of_mass_biggest_blob setting, off by default):
// - biggest blob (bit set): the mass can split into multiple
//   disconnected blobs during the simulation, so the biggest blob (by total
//   mass, 8-connected) is detected first and only its cells contribute.
// - whole grid (bit clear, default): every mass cell contributes.
//
// Blob mode in the reduction:
// 1. verifies the blob labels converged (unit-step hooking with pointer
//    compression until nothing changes - usually one pass)
// 2. accumulates the encoded mass per blob label
// 3. picks the label of the biggest blob and accumulates its center of mass
//
// Labels can only decrease, so racing with other invocations updating
// neighbor cells within the same dispatch is safe. The final reduction
// verifies convergence and guarantees exactness.
//
// The reduction runs as a single workgroup with a strided loop over all
// cells, so barriers provide full global synchronization and no atomics on f32
// are needed. workgroupBarrier only orders workgroup memory, so every hand-off
// of blob_labels or blob_mass between invocations needs a storageBarrier.
//
// Output: total_mass in the same unit as the decoded grid mass,
// com in world coordinates (same units as sim_settings.cell_size);
// divide by cell_size to get the center of mass in cell coordinates.

struct CenterOfMassResult {
    com: vec3f,
    total_mass: f32,
    time: f32,
}

const CENTER_OF_MASS_BIGGEST_BLOB: u32 = 1u << 4u;
const WG_SIZE: u32 = 256u;
const NO_BLOB: u32 = 0xFFFFFFFFu;
const PROPAGATION_ROUNDS: u32 = 16u;

@group(0) @binding(1) var<storage, read> mass_buffer: array<u32>; // no_atomic_float
// atomic_float @group(0) @binding(1) var<storage, read> mass_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> center_of_mass: array<CenterOfMassResult>;
@group(0) @binding(3) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(4) var dem_texture: texture_2d<f32>;
@group(0) @binding(5) var tex_sampler: sampler;
@group(0) @binding(6) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(7) var<storage, read_write> blob_labels: array<u32>;
@group(0) @binding(8) var<storage, read_write> blob_mass: array<atomic<u32>>; // no_atomic_float
// atomic_float @group(0) @binding(8) var<storage, read_write> blob_mass: array<atomic<f32>>;
// per-tile (mass, moment x, moment y, unused) of the whole-grid mode, one per
// workgroup of center_of_mass_partials in row-major tile order
@group(0) @binding(9) var<storage, read_write> tile_partials: array<vec4f>;

var<workgroup> wg_mass: array<f32, WG_SIZE>;
var<workgroup> wg_moment: array<vec2f, WG_SIZE>;
var<workgroup> wg_best: array<vec2u, WG_SIZE>;
var<workgroup> wg_changed: atomic<u32>;
// Barriers need uniform control flow, and reads of read_write storage or atomics
// count as non-uniform (Tint rejects them, naga does not check). Branches ahead
// of a barrier therefore stage their value here for workgroupUniformLoad.
var<workgroup> wg_uniform_flag: u32;

fn min_neighbor_label(cell: vec2i, label: u32, grid_shape: vec2i) -> u32 {
    var best = label;
    for (var dy = -1i; dy <= 1; dy = dy + 1) {
        for (var dx = -1i; dx <= 1; dx = dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let n = cell + vec2i(dx, dy);
            if n.x < 0 || n.y < 0 || n.x >= grid_shape.x || n.y >= grid_shape.y {
                continue;
            }
            let neighbor_label = blob_labels[u32(n.y) * sim_settings.grid_shape.x + u32(n.x)];
            best = min(best, neighbor_label);
        }
    }
    return best;
}

fn get_elevation(position_xy: vec2f) -> f32 {
    let uv = position_to_uv(position_xy);
    return textureSampleLevel(dem_texture, tex_sampler, uv, 0).x;
}

fn trajectory_distance(delta: vec3f) -> f32 {
    let absolute = abs(delta);
    let scale = max(max(absolute.x, absolute.y), absolute.z);
    if scale == 0.0 || !is_finite(scale) {
        return scale;
    }
    // Scale before squaring, like hypot, to avoid overflow and underflow.
    return scale * length(absolute / scale);
}

fn update_trajectory(timestep: u32) {
    let current = center_of_mass[timestep];
    let mass_bits = bitcast<u32>(current.total_mass);
    if (mass_bits & 0x80000000u) != 0u || mass_bits == 0u || mass_bits > 0x7f800000u
        || !is_finite(current.com.x) || !is_finite(current.com.y) || !is_finite(current.com.z) {
        return;
    }
    let previous_timestep = sim_info._last_center_of_mass_timestep;
    if previous_timestep == 0xffffffffu {
        sim_info._first_center_of_mass_timestep = timestep;
        sim_info.travel_length_2d = 0.0;
        sim_info.travel_length_3d = 0.0;
    } else {
        if timestep <= previous_timestep {
            return;
        }
        let delta = current.com - center_of_mass[previous_timestep].com;
        sim_info.travel_length_2d += trajectory_distance(vec3f(delta.xy, 0.0));
        sim_info.travel_length_3d += trajectory_distance(delta);
    }
    sim_info._last_center_of_mass_timestep = timestep;
    sim_info.travel_vertical_drop =
        center_of_mass[sim_info._first_center_of_mass_timestep].com.z - current.com.z;
    sim_info.travel_angle = 0.0;
    if sim_info.travel_length_2d > 0.0 {
        sim_info.travel_angle = degrees(atan2(sim_info.travel_vertical_drop, sim_info.travel_length_2d));
    }
}

// Whole-grid mode: mass and moment of one tile of the grid, reduced in
// workgroup memory (WG_SIZE_2D^2 == WG_SIZE threads). No early return before
// the barriers: cells outside the grid contribute nothing.
@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn center_of_mass_partials(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) tile: vec3u,
    @builtin(num_workgroups) tiles: vec3u,
) {
    var mass = 0.0;
    var moment = vec2f(0.0, 0.0);
    if id.x < sim_settings.grid_shape.x && id.y < sim_settings.grid_shape.y {
        let i = xy_to_idx(id.xy);
        let m = f32(mass_buffer[i]) * inv_mass_factor(); // no_atomic_float
        // atomic_float let m = mass_buffer[i];
        if m > 0.0 && is_finite(m) {
            mass = m;
            moment = cell_center_xy(id.xy) * m;
        }
    }
    wg_mass[li] = mass;
    wg_moment[li] = moment;
    workgroupBarrier();
    var stride_size = WG_SIZE / 2u;
    loop {
        if li < stride_size {
            wg_mass[li] = wg_mass[li] + wg_mass[li + stride_size];
            wg_moment[li] = wg_moment[li] + wg_moment[li + stride_size];
        }
        workgroupBarrier();
        if stride_size == 1u {
            break;
        }
        stride_size = stride_size >> 1u;
    }
    if li == 0u {
        tile_partials[tile.y * tiles.x + tile.x] =
            vec4f(wg_mass[0], wg_moment[0], 0.0);
    }
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn center_of_mass_seed(@builtin(global_invocation_id) id: vec3u) {
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    if (sim_settings.flags & CENTER_OF_MASS_BIGGEST_BLOB) == 0u {
        return;
    }
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let idx = xy_to_idx(id.xy);
    let has_mass = mass_buffer[idx] > 0u; // no_atomic_float
    // atomic_float let has_mass = mass_buffer[idx] > 0.0;
    blob_labels[idx] = select(NO_BLOB, idx, has_mass);
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn center_of_mass_propagate(@builtin(global_invocation_id) id: vec3u) {
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    if (sim_settings.flags & CENTER_OF_MASS_BIGGEST_BLOB) == 0u {
        return;
    }
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let idx = xy_to_idx(id.xy);
    var label = blob_labels[idx];
    if label == NO_BLOB {
        return;
    }
    let cell = vec2i(id.xy);
    let grid_shape_i = vec2i(sim_settings.grid_shape);
    for (var round = 0u; round < PROPAGATION_ROUNDS; round = round + 1u) {
        label = min(label, min_neighbor_label(cell, label, grid_shape_i));
        // pointer compression: adopt the label our label points to
        label = min(label, blob_labels[label]);
        blob_labels[idx] = label;
    }
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn compute_center_of_mass(@builtin(local_invocation_index) li: u32) {
    if li == 0u {
        wg_uniform_flag = sim_info.flags & SIM_INFO_STOPPED;
    }
    if workgroupUniformLoad(&wg_uniform_flag) != 0u {
        return;
    }
    let num_cells = sim_settings.grid_shape.x * sim_settings.grid_shape.y;
    let grid_shape_i = vec2i(sim_settings.grid_shape);
    let use_biggest_blob = (sim_settings.flags & CENTER_OF_MASS_BIGGEST_BLOB) != 0u;

    var mass = 0.0;
    var moment = vec2f(0.0, 0.0);

    if use_biggest_blob {
        // 1. hook the labels with their 8 neighbors and compress label chains
        //    until nothing changes. Values can only decrease, so racing with
        //    other invocations updating neighbors in the same pass is safe. The
        //    propagate shader has already done the heavy lifting, so this
        //    usually converges after a single verification pass.
        var iterations = 0u;
        loop {
            atomicStore(&wg_changed, 0u);
            workgroupBarrier();
            for (var i = li; i < num_cells; i = i + WG_SIZE) {
                let current = blob_labels[i];
                if current != NO_BLOB {
                    var label = min(current, min_neighbor_label(vec2i(idx_to_xy(i)), current, grid_shape_i));
                    // pointer compression: adopt the label our label points to
                    label = min(label, blob_labels[label]);
                    if label < current {
                        blob_labels[i] = label;
                        atomicStore(&wg_changed, 1u);
                    }
                }
            }
            workgroupBarrier();
            storageBarrier();
            if li == 0u {
                wg_uniform_flag = atomicLoad(&wg_changed);
            }
            // also keeps the next pass from resetting wg_changed before it was read
            if workgroupUniformLoad(&wg_uniform_flag) == 0u {
                break;
            }
            iterations = iterations + 1u;
            if iterations > sim_settings.grid_shape.x + sim_settings.grid_shape.y {
                break; // a blob cannot be longer than the grid perimeter
            }
        }

        // 2. accumulate the encoded mass per blob label
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            atomicStore(&blob_mass[i], 0u); // no_atomic_float
            // atomic_float atomicStore(&blob_mass[i], 0.0);
        }
        storageBarrier();
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            if blob_labels[i] != NO_BLOB {
                // whole kg on the integer grid: its mass units resolve a single
                // node and the sum over a blob would overflow u32 (4.3e9 kg
                // fit); rounded up so every node with mass counts
                atomicAdd(&blob_mass[blob_labels[i]], u32(ceil(f32(mass_buffer[i]) * inv_mass_factor()))); // no_atomic_float
                // atomic_float atomicAdd(&blob_mass[blob_labels[i]], mass_buffer[i]);
            }
        }
        storageBarrier();

        // 3. find the biggest blob (ties keep the lower label)
        var best_mass = 0u; // no_atomic_float
        // atomic_float var best_mass = 0.0;
        var best_label = NO_BLOB;
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            let label = blob_labels[i];
            if label != NO_BLOB {
                let m = atomicLoad(&blob_mass[label]);
                if m > best_mass {
                    best_mass = m;
                    best_label = label;
                }
            }
        }
        wg_best[li] = vec2u(best_mass, best_label); // no_atomic_float
        // atomic_float wg_best[li] = vec2u(u32(best_mass), best_label);
        workgroupBarrier();

        var best_stride = WG_SIZE / 2u;
        loop {
            if li < best_stride {
                let a = wg_best[li];
                let b = wg_best[li + best_stride];
                wg_best[li] = select(a, b, b.x > a.x);
            }
            workgroupBarrier();
            if best_stride == 1u {
                break;
            }
            best_stride = best_stride >> 1u;
        }
        let biggest_label = wg_best[0].y;

        // only the cells of the biggest blob contribute
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            if blob_labels[i] == biggest_label {
                let m = f32(mass_buffer[i]) * inv_mass_factor(); // no_atomic_float
                // atomic_float let m = mass_buffer[i];
                if m > 0.0 && is_finite(m) {
                    let position = cell_center_xy(idx_to_xy(i));
                    mass = mass + m;
                    moment = moment + position * m;
                }
            }
        }
    } else {
        // whole-grid mode: every mass cell contributes, summed per tile by
        // center_of_mass_partials
        let tiles = (sim_settings.grid_shape + vec2u(WG_SIZE_2D - 1u)) / WG_SIZE_2D;
        for (var i = li; i < tiles.x * tiles.y; i = i + WG_SIZE) {
            let partial = tile_partials[i];
            mass = mass + partial.x;
            moment = moment + partial.yz;
        }
    }

    wg_mass[li] = mass;
    wg_moment[li] = moment;
    workgroupBarrier();

    // tree reduction of the per-thread partial sums
    var stride_size = WG_SIZE / 2u;
    loop {
        if li < stride_size {
            wg_mass[li] = wg_mass[li] + wg_mass[li + stride_size];
            wg_moment[li] = wg_moment[li] + wg_moment[li + stride_size];
        }
        workgroupBarrier();
        if stride_size == 1u {
            break;
        }
        stride_size = stride_size >> 1u;
    }

    if li == 0u {
        let timestep = sim_info.timestep;
        let total_mass = wg_mass[0];
        let x = wg_moment[0].x / total_mass;
        let y = wg_moment[0].y / total_mass;
        // new_position = p_star;
        var elevation = get_elevation(vec2f(x, y));
        center_of_mass[timestep].total_mass = total_mass;
        center_of_mass[timestep].time = sim_info.elapsed_time;
        center_of_mass[timestep].com = select(vec3f(0.0, 0.0, 0.0), vec3f(x, y, elevation), total_mass > 0.0);
        update_trajectory(timestep);
    }
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
    // forest model (forest.wgsl), read only with SIM_FLAG_FOREST
    forest_drag_coefficient: f32,
    forest_trunk_diameter: f32,
    forest_crown_diameter: f32,
    forest_tree_height: f32,
    forest_modulus_of_rupture: f32,
    forest_decay_coefficient: f32,
    forest_release_max_cover: f32,
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
