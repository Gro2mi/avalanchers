@group(0) @binding(1) var dem_texture: texture_2d<f32>;
// metrics: vec4<f32>,   // x = l_x, y = l_y, z = J, w = g_y
@group(0) @binding(2) var terrain_geometry_texture: texture_storage_2d<rgba32float, write>;
// dynamics: vec4<f32>,  // x = K_xx, y = K_yy, z = K_xy, w = g_x
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

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn analyze_terrain_curvilinear(@builtin(global_invocation_id) id: vec3<u32>) {
    // boundary guard
    if id.x < 1 || id.x >= sim_settings.grid_shape.x - 1 || id.y < 1 || id.y >= sim_settings.grid_shape.y - 1 {
        return;
    }

    let dx = sim_settings.cell_size;
    let dy = sim_settings.cell_size;
    let coords = vec2<i32>(id.xy);

    // 2. Fetch 9-Point Stencil Neighborhood
    let z_cc = textureLoad(dem_texture, coords, 0).r;

    let z_l = textureLoad(dem_texture, coords + vec2<i32>(-1, 0), 0).r;
    let z_r = textureLoad(dem_texture, coords + vec2<i32>(1, 0), 0).r;
    let z_d = textureLoad(dem_texture, coords + vec2<i32>(0, -1), 0).r;
    let z_u = textureLoad(dem_texture, coords + vec2<i32>(0, 1), 0).r;

    let z_lu = textureLoad(dem_texture, coords + vec2<i32>(-1, 1), 0).r;
    let z_ru = textureLoad(dem_texture, coords + vec2<i32>(1, 1), 0).r;
    let z_ld = textureLoad(dem_texture, coords + vec2<i32>(-1, -1), 0).r;
    let z_rd = textureLoad(dem_texture, coords + vec2<i32>(1, -1), 0).r;

    // 3. First Derivatives (Slopes)
    let dB_dx = (z_r - z_l) / (2.0 * dx);
    let dB_dy = (z_u - z_d) / (2.0 * dy);

    // 4. Metric Coefficients & Surface Jacobian
    let l_x = sqrt(1.0 + dB_dx * dB_dx);
    let l_y = sqrt(1.0 + dB_dy * dB_dy);
    let J = sqrt(1.0 + dB_dx * dB_dx + dB_dy * dB_dy);

    // 5. Second Derivatives (Hessian Matrix)
    let dB_dx2 = (z_r - 2.0 * z_cc + z_l) / (dx * dx);
    let dB_dy2 = (z_u - 2.0 * z_cc + z_d) / (dy * dy);
    let dB_dxdy = (z_ru + z_ld - z_lu - z_rd) / (4.0 * dx * dy);

    // 6. Curvatures
    let K_xx = dB_dx2 / J;
    let K_yy = dB_dy2 / J;
    let K_xy = dB_dxdy / J;

    // 7. Slope-parallel Driving Gravity Forces (m/s^2)
    let g_x = -g * dB_dx / l_x;
    let g_y = -g * dB_dy / l_y;

    // 8. --- CALCULATE SLOPE ANGLE & ASPECT IN DEGREES ---

    // Magnitude of the terrain gradient vector
    let slope_magnitude = sqrt(dB_dx * dB_dx + dB_dy * dB_dy);

    // Angle computation: atan(gradient)
    let slope_angle_deg = degrees(atan(slope_magnitude));

    var slope_aspect_deg = 0.0;

    // Safety check: prevent division/undefined states on perfectly flat terrain
    if slope_magnitude > 1e-5 {
        // atan2(y, x) returns the math angle counter-clockwise from East (+x)
        let aspect_rad = atan2(-dB_dy, -dB_dx);

        // GIS aspect: the bearing the slope faces (downhill), clockwise from north
        slope_aspect_deg = ((90.0 - degrees(aspect_rad)) % 360.0 + 360.0) % 360.0;
    } else {
        // GIS convention representation for flat terrain (Aspect = -1)
        slope_aspect_deg = -1.0;
    }

    // 9. Inclination beta of the bed against the smoothed reference surface,
    //    the gully wall angle (cos(beta) = 1 exactly when disabled)
    var basal_topography = vec4f(1.0, 0.0, 0.0, 0.0);
    if gully_walls_enabled() {
        let ref_dx = (textureLoad(dem_reference_texture, coords + vec2<i32>(1, 0), 0).r
            - textureLoad(dem_reference_texture, coords + vec2<i32>(-1, 0), 0).r) / (2.0 * dx);
        let ref_dy = (textureLoad(dem_reference_texture, coords + vec2<i32>(0, 1), 0).r
            - textureLoad(dem_reference_texture, coords + vec2<i32>(0, -1), 0).r) / (2.0 * dy);
        let ref_J = sqrt(1.0 + ref_dx * ref_dx + ref_dy * ref_dy);
        let cos_beta = (dB_dx * ref_dx + dB_dy * ref_dy + 1.0) / (J * ref_J);
        basal_topography.x = clamp(cos_beta, MIN_COS_BETA, 1.0);
    }

    // 10. Write everything to WebGPU Storage Buffers
    let index = xy_to_idx(id.xy);

    // Curvilinear physics buffer
    textureStore(curvature_texture, coords, vec4f(K_xx, K_yy, K_xy, g_x));
    textureStore(terrain_geometry_texture, coords, vec4f(l_x, l_y, J, g_y));
    textureStore(basal_topography_texture, coords, basal_topography);

    // New analysis buffers
    slope_angle_buffer[index] = slope_angle_deg;
    slope_aspect_buffer[index] = slope_aspect_deg;

    // if(cell.x == 0 && cell.y == 0) {
    //     debug[0] = normal.x;
    //     debug[1] = normal.y;
    //     debug[2] = normal.z;
    //     debug[3] = resolution;
    //     debug[4] = dx;
    //     debug[5] = dy;
    //     debug[6] = dxx;
    //     debug[7] = dyy;
    //     debug[8] = dxy;
    //     debug[9] = profile_curvature;
    //     debug[10] = left;
    //     debug[11] = right;
    //     debug[12] = up;
    //     debug[13] = down;
    //     debug[14] = center;
    //     debug[15] = slope_angle;
    //     debug[16] = slope_aspect;
    //     debug[17] = up_right;
    //     debug[18] = down_right;
    //     debug[19] = up_left;
    //     debug[20] = down_left;
    // }
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