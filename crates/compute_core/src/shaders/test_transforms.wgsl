struct CellTestData {
    idx_from_uv: u32,
    idx_from_xy: u32,
    idx_from_x_y: u32,
    error_code: i32,     // -1 = Pass, 0 = didnt run, 1 = Index Flatten, 2 = Round-trip, 3 = Position

    cell_x: u32,
    cell_y: u32,
    rt_cell_x: u32,
    rt_cell_y: u32,

    mock_pos_x: f32,
    mock_pos_y: f32,
    computed_idx: u32,
    expected_idx: u32,
}

@group(0) @binding(1) var<storage, read_write> test_results: array<CellTestData>;

@compute @workgroup_size(16, 16, 1)
fn test_transforms(@builtin(global_invocation_id) global_id: vec3u) {
    if (global_id.x >= sim_settings.grid_shape.x || global_id.y >= sim_settings.grid_shape.y) {
        return;
    }

    let cell = global_id.xy;
    let linear_idx = xy_to_idx(cell);
    let uv = cell_to_uv(cell);

    // Prepare our diagnostic snapshot
    var data: CellTestData;
    data.cell_x = cell.x;
    data.cell_y = cell.y;
    data.error_code = -1i;

    // -------------------------------------------------------------------------
    // TEST 1: Index Flattening
    // -------------------------------------------------------------------------
    data.idx_from_uv  = uv_to_idx(uv);
    data.idx_from_xy  = xy_to_idx(cell);
    data.idx_from_x_y = x_y_to_idx(cell.x, cell.y);

    if (data.idx_from_uv != data.idx_from_xy || data.idx_from_xy != data.idx_from_x_y) {
        data.error_code = 1i;
        test_results[linear_idx] = data;
        return;
    }

    // -------------------------------------------------------------------------
    // TEST 2: Round-Tripping Cell
    // -------------------------------------------------------------------------
    let cell_roundtrip = uv_to_cell(uv);
    data.rt_cell_x = cell_roundtrip.x;
    data.rt_cell_y = cell_roundtrip.y;

    if (any(cell != cell_roundtrip)) {
        data.error_code = 2i;
        test_results[linear_idx] = data;
        return;
    }

    // -------------------------------------------------------------------------
    // TEST 3: Position Index Routing
    // -------------------------------------------------------------------------
    let mock_world_pos = (vec2f(cell) + 0.5) * sim_settings.cell_size;
    data.mock_pos_x    = mock_world_pos.x;
    data.mock_pos_y    = mock_world_pos.y;
    
    data.computed_idx  = position_to_idx(mock_world_pos);
    
    let shifted_uv     = position_to_uv(mock_world_pos);
    data.expected_idx  = uv_to_idx(shifted_uv);

    if (data.computed_idx != data.expected_idx) {
        data.error_code = 3i;
        test_results[linear_idx] = data;
        return;
    }

    // surface-plane velocities: inverse maps that keep the speed and stay tangent
    let bed_normal = normalize(vec3f(0.2 * f32(cell.x) - 0.3, 0.1 - 0.15 * f32(cell.y), 1.0));
    let surface_velocity = vec2f(1.5, -0.7);
    let tangent_velocity = surface_to_tangent(surface_velocity, bed_normal);
    if abs(dot(tangent_velocity, bed_normal)) > 1e-5
        || abs(length(tangent_velocity) - length(surface_velocity)) > 1e-5
        || length(tangent_to_surface(tangent_velocity, bed_normal) - surface_velocity) > 1e-5 {
        data.error_code = 4i;
        test_results[linear_idx] = data;
        return;
    }

    // Write out the pristine passing state
    test_results[linear_idx] = data;
}

// import cartesian_geometry.wgsl;
// BEGIN cartesian_geometry.wgsl
// Cartesian surface geometry for the terrain-following models.
// Pure library module: velocities are 3D Cartesian vectors (vx, vy, vz)
// tangent to the bed z = B(x, y) with unit normal n = (-grad(B), 1) / J.
// Hessian components are passed as (B_xx, B_xy, B_yy), the layout written by
// analyze_terrain.wgsl into the curvature texture.

// The terrain-following model stores velocities in the surface plane: the 3D
// tangent velocity rotated into the horizontal plane by the rotation that takes
// the bed normal to z. Unlike the horizontal projection this keeps the magnitude,
// so length(v2) is the true speed, and the map is linear on any plane, so grid
// averages and APIC stay exact there. The components are not dx/dt.
fn surface_to_tangent(v2: vec2f, n: vec3f) -> vec3f {
    let along_slope = dot(n.xy, v2);
    return vec3f(v2 - along_slope / (1.0 + n.z) * n.xy, -along_slope);
}

fn tangent_to_surface(v3: vec3f, n: vec3f) -> vec2f {
    return v3.xy - v3.z / (1.0 + n.z) * n.xy;
}

// Removes the bed-normal component.
fn tangent_project(v: vec3f, n: vec3f) -> vec3f {
    return v - dot(v, n) * n;
}

// Rotates v into the tangent plane while keeping its speed, so a particle
// moving onto a differently inclined bed does not lose kinetic energy.
fn tangent_project_preserve_speed(v: vec3f, n: vec3f) -> vec3f {
    let tangential = tangent_project(v, n);
    let tangential_speed = length(tangential);
    return select(tangential, tangential * (length(v) / tangential_speed), tangential_speed > 1e-6);
}

// Surface gradient of a scalar field given its horizontal gradient:
// P (grad_xy, 0) with P = I - n n^T. The xy part equals
// surface_inverse_metric(slope, grad_xy) of the curvilinear model.
fn surface_gradient(n: vec3f, grad_xy: vec2f) -> vec3f {
    return tangent_project(vec3f(grad_xy, 0.0), n);
}

// Bed curvature along a horizontal direction d: d^T H d.
fn hessian_quadratic_form(hessian: vec3f, d: vec2f) -> f32 {
    return hessian.x * d.x * d.x + 2.0 * hessian.y * d.x * d.y + hessian.z * d.y * d.y;
}

// Centripetal bed-normal acceleration kappa_n |v|^2 of a tangent velocity v.
// Equals the curvilinear K:qq with K = H / J and q = v.xy.
fn normal_curvature_acceleration(hessian: vec3f, v: vec3f, n: vec3f) -> f32 {
    return n.z * hessian_quadratic_form(hessian, v.xy);
}

// Velocity used for the trapezoidal position update.
fn arc_step_velocity(v_prev: vec3f, v_new: vec3f) -> vec3f {
    return 0.5 * (v_prev + v_new);
}

// Curvature-corrected trapezoidal step along the bed: arc length
// s = |v_step| dt along the tangent, bent by the normal curvature kappa.
// kappa > 0 on a concave bed (positive Hessian form), where the path bends
// towards the bed normal.
// The caller snaps the elevation back onto the DEM afterwards. Next to
// missing DEM data the curvature is undefined and the step stays straight,
// so the particle still reaches the data gap and is flagged there.
fn arc_step(position: vec3f, v_step: vec3f, n: vec3f, bed_kappa: f32, dt: f32) -> vec3f {
    let speed = length(v_step);
    let s = speed * dt;
    if s < 1e-9 {
        return position;
    }
    let kappa = select(0.0, bed_kappa, is_finite(bed_kappa));
    let tangent_scale = s - (kappa * kappa * s * s * s) / 6.0;
    let normal_scale = (kappa * s * s) / 2.0;
    return position + tangent_scale * (v_step / speed) + normal_scale * n;
}

// END cartesian_geometry.wgsl
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