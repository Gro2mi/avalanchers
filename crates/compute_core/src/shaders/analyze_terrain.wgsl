@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var terrain_geometry_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var curvature_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> slope_angle_buffer: array<f32>;
@group(0) @binding(5) var<storage, read_write> slope_aspect_buffer: array<f32>;
@group(0) @binding(6) var<storage, read_write> debug: array<f32>;
// Gaussian-smoothed DEM, written only when gully_walls_enabled()
@group(0) @binding(7) var dem_reference_texture: texture_2d<f32>;
// x = cos(beta) between the bed and reference surface normals
@group(0) @binding(8) var basal_topography_texture: texture_storage_2d<rgba32float, write>;

// Gully walls steeper than 70 degrees against the reference surface load
// the flow like 70 degree walls (keeps 1 / cos(beta) bounded at cliffs)
const MIN_COS_BETA: f32 = 0.34202014;
// terrain_geometry_texture.w: 0 where the normal and curvature of the texel
// come from valid elevations only, 1 where any of its 3 x 3 DEM samples is
// missing (and at the grid border). Bilinear filtering of zeros is exactly
// zero, so a sampled w == 0 means the whole footprint holds valid terrain;
// see on_valid_terrain in g2p_terrain_following.wgsl.
const TERRAIN_INVALID: f32 = 1.0;


// The reference surface at coord and its neighbours: center, right, left, up,
// down, as in the DEM differences of analyze_terrain.
fn reference_neighbourhood(coord: vec2<i32>) -> array<f32, 5> {
    let offsets = array<vec2<i32>, 5>(
        vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
    );
    var values: array<f32, 5>;
    for (var k = 0; k < 5; k++) {
        values[k] = textureLoad(dem_reference_texture, coord + offsets[k], 0).r;
    }
    return values;
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn analyze_terrain(@builtin(global_invocation_id) cell: vec3<u32>) {
    // boundary guard
    if cell.x < 1 || cell.x >= sim_settings.grid_shape.x - 1 || cell.y < 1 || cell.y >= sim_settings.grid_shape.y - 1 {
        if cell.x < sim_settings.grid_shape.x && cell.y < sim_settings.grid_shape.y {
            textureStore(terrain_geometry_texture, vec2<i32>(cell.xy), vec4f(0.0, 0.0, 0.0, TERRAIN_INVALID));
        }
        return;
    }

    let resolution = sim_settings.cell_size;
    let resolution2 = resolution * resolution;
    let coord = vec2<i32>(cell.xy);

    // Sample center and neighbors
    // selects handle domain edges
    let center = textureLoad(dem_texture, coord + vec2<i32>(0, 0), 0).r;                    
    let left   = select(textureLoad(dem_texture, coord + vec2<i32>(-1, 0), 0).r, center, cell.x==0);
    let right  = select(textureLoad(dem_texture, coord + vec2<i32>( 1, 0), 0).r, center, cell.x==sim_settings.grid_shape.x - 1);
    let down   = select(textureLoad(dem_texture, coord + vec2<i32>(0, -1), 0).r, center, cell.y==0);
    let up     = select(textureLoad(dem_texture, coord + vec2<i32>(0,  1), 0).r, center, cell.y==sim_settings.grid_shape.y - 1);

    let up_right    = select(textureLoad(dem_texture, coord + vec2<i32>(1,  1), 0).r, center, cell.x==sim_settings.grid_shape.x - 1 || cell.y==sim_settings.grid_shape.y - 1);
    let down_right  = select(textureLoad(dem_texture, coord + vec2<i32>(-1,  1), 0).r, center, cell.x==sim_settings.grid_shape.x - 1  || cell.y==0);
    let up_left     = select(textureLoad(dem_texture, coord + vec2<i32>(1,  -1), 0).r, center, cell.x==0 || cell.y==0);
    let down_left   = select(textureLoad(dem_texture, coord + vec2<i32>(-1,  -1), 0).r, center, cell.x==0 || cell.y==sim_settings.grid_shape.y - 1);
    // selects handle domain edges
    let dx = select((right - left) / (2.0 * resolution), (right - left) / resolution, cell.x==0 || cell.x==sim_settings.grid_shape.x - 1);
    let dy = select((up - down) / (2.0 * resolution), (up - down) / resolution, cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    let normal = normalize(vec3f(-dx, -dy, 1.0));

    // TODO: 2nd derivative needs more care at the edges
    let dxx = select((left - 2*center + right) / resolution2, (left - 2*center + right) / (4 * resolution2), cell.x==0 || cell.x==sim_settings.grid_shape.x - 1);
    let dyy = select((up - 2*center + down) / resolution2, (up - 2*center + down) / (4 * resolution2), cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    let dxy = select((up_right - down_right - up_left + down_left) / (4 * resolution2), (up_right - down_right - up_left + down_left) / (16 * resolution2), cell.x==0 || cell.x==sim_settings.grid_shape.x - 1 || cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);

    let denom = pow(dx*dx + dy*dy + 1e-12, 1.5);
    let p = dx;
    let q = dy;
    let profile_curvature = select((dxx*dx*dx + 2.0*dxy*dx*dy + dyy*dy*dy) / denom, 0.0, cell.x==0 || cell.x==sim_settings.grid_shape.x - 1 || cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    let plan_curvature = select((dxx*dy*dy - 2.0*dxy*dx*dy + dyy*dx*dx) / denom, 0.0, cell.x==0 || cell.x==sim_settings.grid_shape.x - 1 || cell.y==0 || cell.y==sim_settings.grid_shape.y - 1);
    // let planCurvature    = (dxx*dy*dy - 2.0*dxy*dx*dy + dyy*dx*dx) / denom;
    // let meanCurvature = ((1.0 + dy*dy)*dxx - 2.0*dx*dy*dxy + (1.0 + dx*dx)*dyy) / (2.0 * pow(1.0 + dx*dx + dy*dy, 1.5));
    // let normal_f16: vec4<f16> = vec4<f16>(
    //     f16(normal.x), f16(normal.y), f16(normal.z), f16(1.0)
    // );
    // textureStore(normalsTexture, coord, normal_f16);
    // tested on the loaded values: the compiler may assume arithmetic results
    // are never NaN and drop a test on the derived normal or curvature
    let valid = is_finite(center) && is_finite(left) && is_finite(right) && is_finite(up)
        && is_finite(down) && is_finite(up_right) && is_finite(down_right)
        && is_finite(up_left) && is_finite(down_left);
    textureStore(terrain_geometry_texture, coord, vec4f(normal, select(TERRAIN_INVALID, 0.0, valid)));
    textureStore(curvature_texture, coord, vec4f(dxx, dxy, dyy, plan_curvature));

    // Inclination beta of the bed against the smoothed reference surface, the
    // gully wall angle (cos(beta) = 1 exactly when disabled)
    var basal_topography = vec4f(1.0, 0.0, 0.0, 0.0);
    if gully_walls_enabled() {
        let reference_at = reference_neighbourhood(coord);
        let ref_dx = (reference_at[1] - reference_at[2]) / (2.0 * resolution);
        let ref_dy = (reference_at[3] - reference_at[4]) / (2.0 * resolution);
        let cos_beta = dot(normal, normalize(vec3f(-ref_dx, -ref_dy, 1.0)));
        basal_topography.x = clamp(cos_beta, MIN_COS_BETA, 1.0);
    }
    textureStore(basal_topography_texture, coord, basal_topography);

    let slope_angle = degrees(acos(normal.z));

    var slope_aspect_deg = 0.0;
    if length(vec2f(dx, dy)) > 1e-5 {
        // GIS aspect: the bearing the slope faces (downhill), clockwise from north
        let aspect_rad = atan2(-dy, -dx);
        slope_aspect_deg = ((90.0 - degrees(aspect_rad)) % 360.0 + 360.0) % 360.0;
    } else {
        slope_aspect_deg = -1.0;
    }

    let index = xy_to_idx(cell.xy);
    slope_angle_buffer[index] = slope_angle;
    slope_aspect_buffer[index] = slope_aspect_deg;

    if(cell.x == 0 && cell.y == 0) {
        debug[0] = normal.x;
        debug[1] = normal.y;
        debug[2] = normal.z;
        debug[3] = resolution;
        debug[4] = dx;
        debug[5] = dy;
        debug[6] = dxx;
        debug[7] = dyy;
        debug[8] = dxy;
        debug[9] = profile_curvature;
        debug[10] = left;
        debug[11] = right;
        debug[12] = up;
        debug[13] = down;
        debug[14] = center;
        debug[15] = slope_angle;
        debug[16] = slope_aspect_deg;
        debug[17] = up_right;
        debug[18] = down_right;
        debug[19] = up_left;
        debug[20] = down_left;
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