// Computes the 3D beeline distance between the highest and the lowest point
// of the avalanche: among the cells with peak_flow_thickness above
// sim_settings.peak_flow_thickness_threshold, the one with the maximum and
// the one with the minimum terrain elevation (from the DEM) are the extreme
// points, and the distance is measured between their cell centers in world
// coordinates plus the elevation difference.
//
// Runs as a single workgroup with a strided loop over all cells, so the
// workgroup barriers provide full global synchronization and the elevation
// stays paired with its cell index during the reduction - no atomics needed.
// Dispatch exactly one workgroup.

// The unified evaluation result buffer starts with the mass movement counts
// (32 bytes) and the chamfer sums (16 bytes) written by the other evaluation
// shaders; the beeline section follows.
struct BeelineDistanceResult {
    _evaluation_counts: array<u32, 8>,
    _chamfer: array<f32, 4>,
    distance: f32,
    min_elevation: f32,
    max_elevation: f32,
    min_cell: u32,
    max_cell: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

const NO_CELL: u32 = 0xFFFFFFFFu;
const WG_SIZE: u32 = 256u;

@group(0) @binding(1) var<storage, read> grid_peak_flow_thickness: array<f32>;
@group(0) @binding(2) var dem_texture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> beeline_result: BeelineDistanceResult;

var<workgroup> wg_max_elevation: array<f32, WG_SIZE>;
var<workgroup> wg_max_cell: array<u32, WG_SIZE>;
var<workgroup> wg_min_elevation: array<f32, WG_SIZE>;
var<workgroup> wg_min_cell: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn compute_beeline_distance(@builtin(local_invocation_index) li: u32) {
    let num_cells = sim_settings.grid_shape.x * sim_settings.grid_shape.y;

    // per-thread extremes, elevation and cell index stay paired
    var max_elevation = -1e30;
    var max_cell = NO_CELL;
    var min_elevation = 1e30;
    var min_cell = NO_CELL;
    for (var i = li; i < num_cells; i = i + WG_SIZE) {
        if grid_peak_flow_thickness[i] > sim_settings.peak_flow_thickness_threshold {
            let elevation = textureLoad(dem_texture, vec2<i32>(idx_to_xy(i)), 0).r;
            if elevation > max_elevation {
                max_elevation = elevation;
                max_cell = i;
            }
            if elevation < min_elevation {
                min_elevation = elevation;
                min_cell = i;
            }
        }
    }
    wg_max_elevation[li] = max_elevation;
    wg_max_cell[li] = max_cell;
    wg_min_elevation[li] = min_elevation;
    wg_min_cell[li] = min_cell;
    workgroupBarrier();

    // tree reduction of both extremes with their paired cell indices
    var stride_size = WG_SIZE / 2u;
    loop {
        if li < stride_size {
            if wg_max_elevation[li + stride_size] > wg_max_elevation[li] {
                wg_max_elevation[li] = wg_max_elevation[li + stride_size];
                wg_max_cell[li] = wg_max_cell[li + stride_size];
            }
            if wg_min_elevation[li + stride_size] < wg_min_elevation[li] {
                wg_min_elevation[li] = wg_min_elevation[li + stride_size];
                wg_min_cell[li] = wg_min_cell[li + stride_size];
            }
        }
        workgroupBarrier();
        if stride_size == 1u {
            break;
        }
        stride_size = stride_size >> 1u;
    }

    if li == 0u {
        beeline_result.max_elevation = wg_max_elevation[0u];
        beeline_result.min_elevation = wg_min_elevation[0u];
        beeline_result.max_cell = wg_max_cell[0u];
        beeline_result.min_cell = wg_min_cell[0u];
        let has_avalanche_cells = wg_max_cell[0u] != NO_CELL && wg_min_cell[0u] != NO_CELL;
        let d = cell_center_xy(idx_to_xy(wg_max_cell[0u])) - cell_center_xy(idx_to_xy(wg_min_cell[0u]));
        let dz = wg_max_elevation[0u] - wg_min_elevation[0u];
        beeline_result.distance = select(0.0, length(vec3f(d.x, d.y, dz)), has_avalanche_cells);
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
// TODO calculate momentum factor 
// 2147483647.0 / (120.0 * 100.0 * 25 * 200)
// use override
const MOMENTUM_FACTOR: f32 =  1e-2; 
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
