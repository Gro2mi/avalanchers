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
// - compute_center_of_mass: the final reduction stage, dispatched as exactly
//   one workgroup after seed and propagate have filled the label buffer
//   (both early-out when the whole-grid mode is selected).
//
// Two modes, selected with sim_settings flag bit 4 (the
// center_of_mass_biggest_blob setting):
// - biggest blob (bit set, default): the mass can split into multiple
//   disconnected blobs during the simulation, so the biggest blob (by total
//   mass, 8-connected) is detected first and only its cells contribute.
// - whole grid (bit clear): every mass cell contributes.
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
// cells, so the workgroup barriers provide full global synchronization and no
// atomics on f32 are needed.
//
// Output: total_mass in the same unit as the decoded grid mass,
// com in world coordinates (same units as sim_settings.cell_size);
// divide by cell_size to get the center of mass in cell coordinates.

struct CenterOfMassResult {
    com: vec3f,
    total_mass: f32,
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

var<workgroup> wg_mass: array<f32, WG_SIZE>;
var<workgroup> wg_moment: array<vec2f, WG_SIZE>;
var<workgroup> wg_best: array<vec2u, WG_SIZE>;
var<workgroup> wg_changed: atomic<u32>;

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
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
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
            if atomicLoad(&wg_changed) == 0u {
                break;
            }
            workgroupBarrier();
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
        workgroupBarrier();
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            if blob_labels[i] != NO_BLOB {
                atomicAdd(&blob_mass[blob_labels[i]], mass_buffer[i]);
            }
        }
        workgroupBarrier();

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
                let m = f32(mass_buffer[i]) * INV_MASS_FACTOR; // no_atomic_float
                // atomic_float let m = mass_buffer[i];
                if m > 0.0 && is_finite(m) {
                    let position = cell_center_xy(idx_to_xy(i));
                    mass = mass + m;
                    moment = moment + position * m;
                }
            }
        }
    } else {
        // whole-grid mode: every mass cell contributes
        for (var i = li; i < num_cells; i = i + WG_SIZE) {
            let m = f32(mass_buffer[i]) * INV_MASS_FACTOR; // no_atomic_float
            // atomic_float let m = mass_buffer[i];
            if m > 0.0 && is_finite(m) {
                let position = cell_center_xy(idx_to_xy(i));
                mass = mass + m;
                moment = moment + position * m;
            }
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
        center_of_mass[timestep].com = select(vec3f(0.0, 0.0, 0.0), vec3f(x, y, elevation), total_mass > 0.0);
    }
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
// Momentum is quantized per particle->node contribution before atomicAdd, so
// MOMENTUM_FACTOR sets the velocity resolution of that contribution:
// the smallest non-zero contribution is v = 1 / (particle_mass * weight * MOMENTUM_FACTOR).
// It must stay far below the slow hydrostatic spreading velocities (~0.01-0.1 m/s)
// or p2g rounds them to zero every step and the flow never spreads laterally.
// i32 budget: node sum = node_mass * v_max * MOMENTUM_FACTOR
//   (rho*cell^2*h*J ~ 6e4 kg * 40 m/s * 1e2 = 2.4e8 < 2.1e9)
const MOMENTUM_FACTOR: f32 = 1e2;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR;
const INV_MOMENTUM_FACTOR: f32 = 1 / MOMENTUM_FACTOR;
// depth-integrated internal force (h * sigma * grad_w * area, ~1e4..1e5 N per node contribution)
const FORCE_FACTOR: f32 = 1e-3; // i32 limit is 2.1e6 N per node
const INV_FORCE_FACTOR: f32 = 1 / FORCE_FACTOR;

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
    release_max_elevation: f32,
    peak_flow_thickness_threshold: f32,
    // MPMDAC constitutive model (must mirror the Rust POD layout)
    constitutive_model: u32,
    shear_modulus: f32,
    hardening_modulus: f32,
    // MPMDAC compressibility; bulk_modulus 0 = incompressible
    bulk_modulus: f32,
    compaction_pressure: f32,
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
