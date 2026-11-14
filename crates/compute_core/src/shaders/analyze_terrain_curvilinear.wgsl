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

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const H_FACTOR: f32 = 1e6;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_H_FACTOR: f32 = 1 / H_FACTOR;
// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The factors fill the integer range with the largest node values a
// flow can reach: a MAX_FLOW_THICKNESS deep column over a cell
// (density * thickness * cell_size^2) moving at MAX_FLOW_VELOCITY, times
// QUANTIZATION_SAFETY (which also covers the slope factor of the node mass).
// They scale with the cell size like the particle masses, so a particle of a
// release h_r thick with N particles per cell holds about
// U32_RANGE * h_r / (N * QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS) mass units
// (2e6 for 1 m and 64 per cell) at every cell size. Must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 30.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
const QUANTIZATION_SAFETY: f32 = 2.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
fn max_node_mass() -> f32 {
    return QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / (max_node_mass() * MAX_FLOW_VELOCITY);
}
fn inv_momentum_factor() -> f32 {
    return max_node_mass() * MAX_FLOW_VELOCITY / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (15 g for 5 m cells and 200 kg/m3). Lighter nodes
// count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// Gully walls (reference surface, wall friction) need a reference surface at
// least one cell wide; must match SimSettings::gully_walls_enabled.
fn gully_walls_enabled() -> bool {
    return sim_settings.reference_surface_length >= sim_settings.cell_size;
}
// depth-integrated internal force (h * sigma * grad_w * area, ~1e4..1e5 N per node contribution)
const FORCE_FACTOR: f32 = 1e-3; // i32 limit is 2.1e6 N per node
const INV_FORCE_FACTOR: f32 = 1 / FORCE_FACTOR;
// weak-form pressure sum m_p h_p / 2 grad_w (kg) of the curvilinear model,
// node sum = node_mass * |grad h|: 0.001 kg resolves thin front nodes
//   (rho*cell^2*h*J ~ 6e4 kg * 1 * 1e3 = 6e7 < 2.1e9)
const PRESSURE_FACTOR: f32 = 1e3;
const INV_PRESSURE_FACTOR: f32 = 1 / PRESSURE_FACTOR;

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
    grid_elapsed_time: f32,
    travel_length_2d: f32,
    travel_length_3d: f32,
    travel_vertical_drop: f32,
    travel_angle: f32,
    _first_center_of_mass_timestep: u32,
    _last_center_of_mass_timestep: u32,
    _pad: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
// a particle exceeded MAX_PARTICLE_SPEED and was stopped
const SIM_INFO_UNPHYSICAL_SPEED: u32 = 1u << 4u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

// Fastest particle speed (m/s) taken as physical. Dense flow avalanches reach
// about 60-80 m/s and the integer atomic grid holds momentum up to
// QUANTIZATION_SAFETY * MAX_FLOW_VELOCITY = 200 m/s; a faster particle comes from
// a numerical blow-up (e.g. a cliff of no-data elevations) and is stopped.
// Mirrors compute_core::MAX_PARTICLE_SPEED.
const MAX_PARTICLE_SPEED: f32 = 150.0;
// stopped for exceeding MAX_PARTICLE_SPEED
const PARTICLE_UNPHYSICAL_SPEED: u32 = 1u << 25u;
// reversible low-speed state; the low bits (below PARTICLE_UNPHYSICAL_SPEED)
// hold the rest start step while resting and the consecutive slow-step count
// while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
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
    // resting particles counted during the current step, reset by update_sim_info
    resting_particles: atomic<u32>,
    // total grid momentum (kg m/s) of the current step as a 64-bit sum, reset by update_sim_info
    momentum_lo: atomic<u32>,
    momentum_hi: atomic<u32>,
    // f32 bits of the largest total momentum so far
    peak_momentum: atomic<u32>,
};

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_UNPHYSICAL_SPEED - 1u;
    var resting = (previous_state & PARTICLE_RESTING) != 0u;
    var low_bits = previous_state & low_mask;
    if resting {
        if speed > 2.0 * sim_settings.velocity_threshold {
            resting = false;
            low_bits = 0u;
        }
    } else if speed < sim_settings.velocity_threshold {
        low_bits = min(low_bits + 1u, REST_STEPS);
        if low_bits >= REST_STEPS {
            resting = true;
            low_bits = timestep & low_mask;
        }
    } else {
        low_bits = 0u;
    }
    return (previous_state & ~(PARTICLE_RESTING | low_mask))
        | select(0u, PARTICLE_RESTING, resting)
        | low_bits;
}

struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// length in steps of the no-new-cells rolling window: the fastest material
// moves at most cfl cells per step, so round(travel / cfl) steps cover the same
// front travel at any cfl; at least 10 steps
fn no_new_cells_window() -> u32 {
    return max(10u, u32(round(sim_settings.no_new_cells_travel / sim_settings.cfl)));
}

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