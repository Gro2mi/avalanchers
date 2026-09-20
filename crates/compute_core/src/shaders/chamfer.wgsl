// Chamfer distance between the simulated cells
// (grid_peak_flow_thickness > sim_settings.peak_flow_thickness_threshold)
// and the region-of-interest bitmask, in three kernels sharing this module:
// - chamfer_prepare: seeds the nearest-neighbor fields. Per cell, both fields
//   store the cell's own coordinates if it is a seed (region_of_interest bit
//   set, respectively a simulated cell), NO_SEED otherwise.
// - chamfer_flood: one pass of the jump flooding algorithm (Rong & Tan, "Jump
//   Flooding in GPU with Applications to Voronoi Diagram and Distance
//   Transform") on both nearest-seed fields. The run method dispatches this
//   kernel once per step size, with chamfer_params.step going n/2, n/4, ..., 1
//   for n the next power of two >= the larger grid dimension. The fields are
//   read from the snapshot buffers written by the previous pass, so every
//   invocation sees a consistent state and the result is the exact
//   nearest-seed Voronoi.
// - chamfer_reduce: accumulates the raw sums (sum/count of simulated cells to
//   their nearest region-of-interest cell and vice versa) into the unified
//   evaluation result buffer. Runs as a single workgroup with a strided loop,
//   so no atomics are needed. Dispatch exactly one workgroup; the CPU combines
//   the sums into the diagonal-normalized chamfer distance.

struct ChamferParams {
    step: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

// The first 32 bytes of the unified evaluation result buffer hold the mass
// movement counts written by the evaluate shaders; the chamfer sums follow.
struct ChamferDistanceResult {
    _evaluation_counts: array<u32, 8>,
    sum_sim_to_roi: f32,
    count_sim: f32,
    sum_roi_to_sim: f32,
    count_roi: f32,
}

const NO_SEED: u32 = 0xFFFFFFFFu;
const MAX_DISTANCE_SQUARED: i32 = 2147483647;
const WG_SIZE: u32 = 256u;

@group(0) @binding(1) var<uniform> chamfer_params: ChamferParams;
@group(0) @binding(2) var<storage, read> grid_peak_flow_thickness: array<f32>;
@group(0) @binding(3) var<storage, read> region_of_interest: array<u32>;
@group(0) @binding(4) var<storage, read_write> chamfer_nearest_roi: array<vec2u>;
@group(0) @binding(5) var<storage, read> chamfer_nearest_roi_snapshot: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> chamfer_nearest_sim: array<vec2u>;
@group(0) @binding(7) var<storage, read> chamfer_nearest_sim_snapshot: array<vec2u>;
@group(0) @binding(8) var<storage, read_write> chamfer_result: ChamferDistanceResult;

fn bit_is_set(word: u32, index: u32) -> bool {
    return (word & (1u << (index % 32u))) != 0u;
}

fn distance_squared(cell: vec2i, seed: vec2i) -> i32 {
    let d = seed - cell;
    return d.x * d.x + d.y * d.y;
}

fn cell_distance(cell: vec2u, seed: vec2u) -> f32 {
    return length(vec2f(seed) - vec2f(cell)) * sim_settings.cell_size;
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn chamfer_prepare(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let idx = xy_to_idx(id.xy);
    let no_seed = vec2u(NO_SEED);

    chamfer_nearest_roi[idx] = select(no_seed, id.xy, bit_is_set(region_of_interest[idx / 32u], idx));
    chamfer_nearest_sim[idx] = select(
        no_seed,
        id.xy,
        grid_peak_flow_thickness[idx] > sim_settings.peak_flow_thickness_threshold,
    );
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn chamfer_flood(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let cell = vec2i(id.xy);
    let step = i32(chamfer_params.step);
    let idx = xy_to_idx(id.xy);

    var best_roi = chamfer_nearest_roi_snapshot[idx];
    var best_roi_d = select(distance_squared(cell, vec2i(best_roi)), MAX_DISTANCE_SQUARED, best_roi.x == NO_SEED);
    var best_sim = chamfer_nearest_sim_snapshot[idx];
    var best_sim_d = select(distance_squared(cell, vec2i(best_sim)), MAX_DISTANCE_SQUARED, best_sim.x == NO_SEED);

    for (var dy = -step; dy <= step; dy = dy + step) {
        for (var dx = -step; dx <= step; dx = dx + step) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let n = cell + vec2i(dx, dy);
            if n.x < 0 || n.y < 0 || n.x >= i32(sim_settings.grid_shape.x) || n.y >= i32(sim_settings.grid_shape.y) {
                continue;
            }
            let nidx = u32(n.y) * sim_settings.grid_shape.x + u32(n.x);

            let neighbor_roi = chamfer_nearest_roi_snapshot[nidx];
            if neighbor_roi.x != NO_SEED {
                let d = distance_squared(cell, vec2i(neighbor_roi));
                if d < best_roi_d {
                    best_roi_d = d;
                    best_roi = neighbor_roi;
                }
            }

            let neighbor_sim = chamfer_nearest_sim_snapshot[nidx];
            if neighbor_sim.x != NO_SEED {
                let d = distance_squared(cell, vec2i(neighbor_sim));
                if d < best_sim_d {
                    best_sim_d = d;
                    best_sim = neighbor_sim;
                }
            }
        }
    }

    chamfer_nearest_roi[idx] = best_roi;
    chamfer_nearest_sim[idx] = best_sim;
}

var<workgroup> wg_partial: array<vec4f, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn chamfer_reduce(@builtin(local_invocation_index) li: u32) {
    let num_cells = sim_settings.grid_shape.x * sim_settings.grid_shape.y;

    // x: sum_sim_to_roi, y: count_sim, z: sum_roi_to_sim, w: count_roi
    var partial = vec4f(0.0, 0.0, 0.0, 0.0);
    for (var i = li; i < num_cells; i = i + WG_SIZE) {
        let simulated = grid_peak_flow_thickness[i] > sim_settings.peak_flow_thickness_threshold;
        let reference = bit_is_set(region_of_interest[i / 32u], i);

        if simulated {
            let seed = chamfer_nearest_roi[i];
            if seed.x != NO_SEED {
                partial.x = partial.x + cell_distance(idx_to_xy(i), seed);
            }
            partial.y = partial.y + 1.0;
        }
        if reference {
            let seed = chamfer_nearest_sim[i];
            if seed.x != NO_SEED {
                partial.z = partial.z + cell_distance(idx_to_xy(i), seed);
            }
            partial.w = partial.w + 1.0;
        }
    }
    wg_partial[li] = partial;
    workgroupBarrier();

    // tree reduction of the per-thread partial sums
    var stride_size = WG_SIZE / 2u;
    loop {
        if li < stride_size {
            wg_partial[li] = wg_partial[li] + wg_partial[li + stride_size];
        }
        workgroupBarrier();
        if stride_size == 1u {
            break;
        }
        stride_size = stride_size >> 1u;
    }

    if li == 0u {
        chamfer_result.sum_sim_to_roi = wg_partial[0].x;
        chamfer_result.count_sim = wg_partial[0].y;
        chamfer_result.sum_roi_to_sim = wg_partial[0].z;
        chamfer_result.count_roi = wg_partial[0].w;
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
