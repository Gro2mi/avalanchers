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
    _chamfer: array<f32, 6>,
    distance: f32,
    min_elevation: f32,
    max_elevation: f32,
    min_cell: u32,
    max_cell: u32,
    _padding: u32,
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
