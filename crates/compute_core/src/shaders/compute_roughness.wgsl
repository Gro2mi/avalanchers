@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> roughness_buffer: array<f32>;

fn dem_at(x: i32, y: i32) -> f32 {
    return textureLoad(dem_texture, vec2<i32>(x, y), 0).r;
}

// Unit surface normal from central differences of the DEM, one-sided at the
// grid border. Computed from the DEM directly because the terrain geometry
// texture holds model-dependent quantities (unit normal for the
// terrain-following models, metric coefficients for the curvilinear model).
fn surface_normal(cell: vec2<i32>) -> vec3f {
    let last = vec2<i32>(sim_settings.grid_shape) - 1;
    let lo = max(cell - 1, vec2<i32>(0));
    let hi = min(cell + 1, last);
    let dz_dx = (dem_at(hi.x, cell.y) - dem_at(lo.x, cell.y))
        / (f32(hi.x - lo.x) * sim_settings.cell_size);
    let dz_dy = (dem_at(cell.x, hi.y) - dem_at(cell.x, lo.y))
        / (f32(hi.y - lo.y) * sim_settings.cell_size);
    return normalize(vec3f(-dz_dx, -dz_dy, 1.0));
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn compute_roughness(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.xy;
    if cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y {
        return;
    }
    var roughness = 1.0; // high roughness at borders
    // vector ruggedness measure according to doi:10.5194/nhess-16-2211-2016:
    // 1 - |sum of the 9 unit normals| / 9, 0 on a plane
    if cell.x > 0 && cell.y > 0 && cell.x < sim_settings.grid_shape.x - 1 && cell.y < sim_settings.grid_shape.y - 1 {
        // TODO implement kernel size depending on resolution and average snow height
        // 2 * (HS_mean^2 * C_v) + 1; C_v = 0.35 for regular smoothing, 0.2 for low smoothing (low winds)
        var normal_sum = vec3f(0.0);
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                normal_sum = normal_sum + surface_normal(vec2<i32>(cell) + vec2<i32>(x, y));
            }
        }
        roughness = 1.0 - length(normal_sum) / 9.0;
        // no-data cells (NaN elevations) cannot carry a slab
        if !is_finite(roughness) {
            roughness = 1.0;
        }
        // f32 rounding can push a perfect plane slightly below zero
        roughness = max(roughness, 0.0);
    }
    roughness_buffer[xy_to_idx(cell)] = roughness;
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

// SimFlags::GRID_OCCUPIED (settings.rs): p2g marks the cells holding a
// depositing particle in grid_occupied and only those record peak maps
const SIM_FLAG_GRID_OCCUPIED: u32 = 1u << 9u;

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