@group(0) @binding(1) var<storage> grid_mass_atomic: array<u32>; // no_atomic_float
// atomic_float @group(0) @binding(1) var<storage> grid_mass_atomic: array<f32>;
// xyz = unit bed normal, w = profile curvature (analyze_terrain.wgsl)
@group(0) @binding(2) var terrain_geometry_texture: texture_2d<f32>;
// xyz = Hessian (B_xx, B_xy, B_yy), w = plan curvature (analyze_terrain.wgsl)
@group(0) @binding(3) var curvature_texture: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> peak_flow_thickness: array<f32>;
@group(0) @binding(5) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(6) var<storage> grid_momentum_atomic: array<i32>; // no_atomic_float
// atomic_float @group(0) @binding(6) var<storage> grid_momentum_atomic: array<f32>; // Combined u, v
@group(0) @binding(7) var<storage, read_write> new_cells_rolling_window: array<u32>;
@group(0) @binding(8) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(9) var<storage, read_write> grid_velocity: array<vec2f>;
@group(0) @binding(10) var<storage, read_write> grid_peak_velocity: array<f32>;
// x = cos(beta) between the bed and the smoothed reference surface
@group(0) @binding(11) var basal_topography_texture: texture_2d<f32>;
// z components of the 3D model; the terrain-following layout ends at binding 11
@group(0) @binding(12) var<storage> grid_momentum_z_atomic: array<i32>; // no_atomic_float
// atomic_float @group(0) @binding(12) var<storage> grid_momentum_z_atomic: array<f32>;
@group(0) @binding(13) var<storage, read_write> grid_velocity_z: array<f32>;

// Cartesian counterpart of grid_physics_curvilinear: forces act on a 3D velocity
// tangent to the bed and are projected with the bed normal. The node keeps the
// velocity in the surface plane (see surface_to_tangent).
@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics_terrain_following(@builtin(global_invocation_id) id: vec3u) {
    if node_is_skipped(id.xy) {
        return;
    }
    let idx = xy_to_idx(id.xy);
    if neighbourhood_is_empty(id.xy) {
        grid_velocity[idx] = vec2f(0.0);
        return;
    }
    let n = textureLoad(terrain_geometry_texture, vec2<i32>(id.xy), 0).xyz;
    if !is_finite(n.z) {
        grid_velocity[idx] = vec2f(0.0);
        return;
    }
    let mass = node_mass(idx);
    let velocity = surface_to_tangent(node_velocity_xy(idx, mass), n);
    grid_velocity[idx] = tangent_to_surface(update_node(id.xy, n, velocity, mass), n);
}

// The same update on a node velocity carried as a full 3D vector.
@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics_3d(@builtin(global_invocation_id) id: vec3u) {
    if node_is_skipped(id.xy) {
        return;
    }
    let idx = xy_to_idx(id.xy);
    if neighbourhood_is_empty(id.xy) {
        grid_velocity[idx] = vec2f(0.0);
        grid_velocity_z[idx] = 0.0;
        return;
    }
    let n = textureLoad(terrain_geometry_texture, vec2<i32>(id.xy), 0).xyz;
    if !is_finite(n.z) {
        grid_velocity[idx] = vec2f(0.0);
        grid_velocity_z[idx] = 0.0;
        return;
    }
    let mass = node_mass(idx);
    let w = select(f32(grid_momentum_z_atomic[idx]) * INV_MOMENTUM_FACTOR / (mass + 1e-6), 0.0, mass < 1e-2); // no_atomic_float
    // atomic_float let w = select(grid_momentum_z_atomic[idx] / mass, 0.0, mass < 1e-2);
    // particles on differently inclined beds scatter into this node
    let velocity = tangent_project(vec3f(node_velocity_xy(idx, mass), w), n);
    let new_velocity = update_node(id.xy, n, velocity, mass);
    grid_velocity[idx] = new_velocity.xy;
    grid_velocity_z[idx] = new_velocity.z;
}

fn node_is_skipped(xy: vec2u) -> bool {
    return xy.x < 1 || xy.x >= (sim_settings.grid_shape.x - 1) || xy.y < 1 || xy.y >= (sim_settings.grid_shape.y - 1)
        || (sim_info.flags & SIM_INFO_STOPPED) != 0u;
}

// No particle gathers from a node whose 3x3 neighbourhood holds no mass: each
// particle deposits at least a quarter of its mass onto its stencil centre,
// which neighbours all nine nodes the particle gathers from.
fn neighbourhood_is_empty(xy: vec2u) -> bool {
    for (var y = xy.y - 1u; y <= xy.y + 1u; y++) {
        for (var x = xy.x - 1u; x <= xy.x + 1u; x++) {
            if f32(grid_mass_atomic[x_y_to_idx(x, y)]) > 0.0 {
                return false;
            }
        }
    }
    return true;
}

fn node_mass(idx: u32) -> f32 {
    return f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    // atomic_float return grid_mass_atomic[idx];
}

// Nodes holding less mass than the p2g quantization floor carry no meaningful
// velocity (see grid_physics_curvilinear).
fn node_velocity_xy(idx: u32, mass: f32) -> vec2f {
    let node_is_empty = mass < 1e-2;
    let u = select(f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR / (mass + 1e-6), 0.0, node_is_empty); // no_atomic_float
    let v = select(f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR / (mass + 1e-6), 0.0, node_is_empty); // no_atomic_float
    // atomic_float let u = select(grid_momentum_atomic[idx * 2] / mass, 0.0, node_is_empty);
    // atomic_float let v = select(grid_momentum_atomic[idx * 2 + 1] / mass, 0.0, node_is_empty);
    return vec2f(u, v);
}

// Applies gravity, pressure and friction to the tangent velocity of a node and
// records the node's diagnostics. Returns the new tangent velocity.
fn update_node(xy: vec2u, n: vec3f, velocity: vec3f, mass: f32) -> vec3f {
    let use_curvature: bool = (sim_settings.flags & (1u << 0u)) != 0u;
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    let use_earth_pressure_coefficient: bool = (sim_settings.flags & (1u << 2u)) != 0u;

    let idx = xy_to_idx(xy);
    let hessian = textureLoad(curvature_texture, vec2<i32>(xy), 0).xyz;

    let h = get_h(xy.x, xy.y);
    let current_peak_flow_thickness = peak_flow_thickness[idx];
    if current_peak_flow_thickness < h {
        // only if previously cell was empty, we count it as a new cell for diagnostics
        if current_peak_flow_thickness < 1e-5 {
            new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] = new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] + 1u; // update new cell count for diagnostics
        }
        peak_flow_thickness[idx] = h;
    }
    if bitcast<u32>(h) > atomicLoad(&atomic_values.peak_flow_thickness) {
        if !is_nan(h) {
            atomicMax(&atomic_values.peak_flow_thickness, bitcast<u32>(h));
        }
    }

    // 2. Bed-normal load
    var g_normal = g * n.z;
    if use_curvature {
        g_normal = max(g_normal + normal_curvature_acceleration(hessian, velocity, n), 0.0);
    }

    // 3. Driving accelerations: tangential gravity and hydrostatic pressure
    let gravity_acceleration = vec3f(0.0, 0.0, -g) + g * n.z * n;
    let grad_h = vec2f(
        (get_h(xy.x + 1, xy.y) - get_h(xy.x - 1, xy.y)) / (2.0 * sim_settings.cell_size),
        (get_h(xy.x, xy.y + 1) - get_h(xy.x, xy.y - 1)) / (2.0 * sim_settings.cell_size)
    );
    // Savage-Hutter earth pressure (Hutter et al. 1993): k_x along the fall
    // line, k_y across it, each active where the flow stretches in that
    // direction and passive where it is compressed; hydrostatic when disabled
    var k_grad_h = grad_h;
    var k_y = 1.0;
    if use_earth_pressure_coefficient {
        // horizontal bed gradient from the unit normal
        let slope = -n.xy / n.z;
        let earth = earth_pressure(xy, slope, velocity.xy, grad_h);
        k_grad_h = earth.k_grad_h;
        k_y = earth.k_y;
    }
    let pressure_acceleration = select(vec3f(0.0), -g_normal * surface_gradient(n, k_grad_h), use_particle_interaction);

    let proposed = tangent_project(velocity + sim_info.dt * (gravity_acceleration + pressure_acceleration), n);
    let proposed_speed = length(proposed);

    // 4. Basal friction, applied as a scaling factor that cannot reverse the
    // flow. It acts at every speed, so Coulomb resistance also acts as static
    // friction: a node at rest stays at rest while the driving acceleration is
    // below the yield value, independent of dt. Evaluating at no less than
    // velocity_threshold keeps the low-speed gate of the shared friction model
    // from switching it off.
    let friction_speed = max(proposed_speed, sim_settings.velocity_threshold);
    // gully walls, inclined by beta against the reference surface, carry the
    // cross-slope lateral stress k_y sigma_zz of the snow beside them
    let cos_beta = textureLoad(basal_topography_texture, vec2<i32>(xy), 0).x;
    let wall_load = g_normal * gully_wall_load_factor(cos_beta, k_y);
    let friction_acceleration = basal_friction_acceleration(wall_load, sim_settings.snow_density, friction_speed, h, sim_settings.friction_model);
    var new_velocity = vec3f(0.0);
    if (friction_acceleration * sim_info.dt) < proposed_speed {
        new_velocity = proposed * (1.0 - friction_acceleration * sim_info.dt / proposed_speed);
    }

    let v_mag = length(new_velocity);
    if !is_nan(v_mag) && h > 1e-3 && bitcast<u32>(v_mag) > atomicLoad(&atomic_values.peak_velocity) {
        atomicMax(&atomic_values.peak_velocity, bitcast<u32>(v_mag));
    }
    if !is_nan(v_mag) && h > 1e-3 {
        grid_peak_velocity[idx] = max(grid_peak_velocity[idx], v_mag);
    }
    if sim_settings.momentum_stop_fraction > 0.0 && !is_nan(v_mag) {
        // total momentum in kg m/s as a 64-bit sum of two u32 atomics
        let node_momentum = u32(round(mass * v_mag));
        if node_momentum > 0u {
            let previous = atomicAdd(&atomic_values.momentum_lo, node_momentum);
            if previous > 0xFFFFFFFFu - node_momentum {
                atomicAdd(&atomic_values.momentum_hi, 1u);
            }
        }
    }

    // gravity waves travel at sqrt(g cos(theta) h) on the inclined bed
    let expected_cell_velocity = v_mag + sqrt(g * h * n.z);
    if !is_nan(expected_cell_velocity) && h > 1e-3 && bitcast<u32>(expected_cell_velocity) > atomicLoad(&atomic_values.expected_max_velocity) {
        atomicMax(&atomic_values.expected_max_velocity, bitcast<u32>(expected_cell_velocity));
    }
    return new_velocity;
}

// bed-normal flow thickness of a node
fn get_h(x: u32, y: u32) -> f32 {
    let n_z = textureLoad(terrain_geometry_texture, vec2i(i32(x), i32(y)), 0).z;
    return node_mass(x_y_to_idx(x, y)) / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size) * n_z;
}

// import earth_pressure.wgsl;
// BEGIN earth_pressure.wgsl
// Savage-Hutter earth pressure coefficients (Hutter et al. 1993) for the grid
// physics of the depth-averaged models.
// Pure library module: it is textually imported and relies on the importing
// module providing grid_mass_atomic, grid_momentum_atomic and the utils.wgsl
// symbols (sim_settings, x_y_to_idx, INV_MASS_FACTOR, INV_MOMENTUM_FACTOR).

struct EarthPressure {
    // pressure gradient weighted with k_x along the fall line, k_y across it
    k_grad_h: vec2f,
    // cross-slope coefficient, also the lateral stress loading gully walls
    k_y: f32,
}

// Earth pressure at node xy from the horizontal bed gradient `slope`, the
// horizontal node velocity q and the horizontal gradient of the flow thickness.
// Each coefficient is active where the flow stretches in its direction and
// passive where it is compressed. The fall line is an eigenvector of the
// surface metric, so the weighting commutes with projecting onto the bed.
fn earth_pressure(xy: vec2u, slope: vec2f, q: vec2f, grad_h: vec2f) -> EarthPressure {
    let e_x = earth_pressure_axis(slope, q);
    let e_y = vec2f(-e_x.y, e_x.x);
    let grad_q = velocity_gradient(xy.x, xy.y);
    let k = earth_pressure_coefficients(
        radians(sim_settings.internal_friction_angle),
        radians(sim_settings.basal_friction_angle),
        dot(e_x, grad_q * e_x),
        dot(e_y, grad_q * e_y)
    );
    return EarthPressure(k.x * dot(e_x, grad_h) * e_x + k.y * dot(e_y, grad_h) * e_y, k.y);
}

// Horizontal velocity q of node (x, y) from the node's own mass and momentum;
// z = 1 when the node holds mass, 0 below the p2g mass floor where q is
// meaningless.
fn node_velocity(x: u32, y: u32) -> vec3f {
    let idx = x_y_to_idx(x, y);
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    if mass < 1e-2 {
        return vec3f(0.0);
    }
    let q = vec2f(
        f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR, // no_atomic_float
        f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR // no_atomic_float
        // atomic_float grid_momentum_atomic[idx * 2],
        // atomic_float grid_momentum_atomic[idx * 2 + 1]
    ) / mass;
    return vec3f(q, 1.0);
}

// Derivative of q along one grid axis: central where both neighbours hold
// mass, one-sided against the centre at the flow edge, zero for an isolated
// node. Empty nodes carry no velocity, so a spreading front is not mistaken
// for a compressed one.
fn edge_aware_derivative(low: vec3f, centre: vec2f, high: vec3f) -> vec2f {
    let dx = sim_settings.cell_size;
    if low.z > 0.0 && high.z > 0.0 {
        return (high.xy - low.xy) / (2.0 * dx);
    }
    if high.z > 0.0 {
        return (high.xy - centre) / dx;
    }
    if low.z > 0.0 {
        return (centre - low.xy) / dx;
    }
    return vec2f(0.0);
}

// Horizontal velocity gradient, columns dq/dx and dq/dy: grad_q * e is the
// derivative of q along e, dot(e, grad_q * e) the strain rate along e.
fn velocity_gradient(x: u32, y: u32) -> mat2x2f {
    let centre = node_velocity(x, y).xy;
    return mat2x2f(
        edge_aware_derivative(node_velocity(x - 1u, y), centre, node_velocity(x + 1u, y)),
        edge_aware_derivative(node_velocity(x, y - 1u), centre, node_velocity(x, y + 1u))
    );
}

// Savage-Hutter downslope axis: the fall line, the flow direction on flat
// ground.
fn earth_pressure_axis(slope: vec2f, q: vec2f) -> vec2f {
    if length(slope) > 1e-3 {
        return -normalize(slope);
    }
    if length(q) > 1e-6 {
        return normalize(q);
    }
    return vec2f(1.0, 0.0);
}

// Earth pressure coefficients (k_x, k_y) of Hutter et al. (1993) from the
// internal and bed friction angles and the strain rates along and across the
// downslope axis; >= 0 (stretching) selects the active state.
fn earth_pressure_coefficients(phi: f32, delta: f32, strain_x: f32, strain_y: f32) -> vec2f {
    let cos2_phi = cos(phi) * cos(phi);
    let cos2_delta = cos(delta) * cos(delta);
    let tan2_delta = tan(delta) * tan(delta);
    // delta > phi has no Mohr-Coulomb limit state; clamp to the phi = delta case
    let root = sqrt(max(1.0 - cos2_phi / cos2_delta, 0.0));
    let k_x = select(
        2.0 / cos2_phi * (1.0 + root) - 1.0,
        2.0 / cos2_phi * (1.0 - root) - 1.0,
        strain_x >= 0.0
    );
    let root_y = sqrt((1.0 - k_x) * (1.0 - k_x) + 4.0 * tan2_delta);
    let k_y = select(0.5 * (k_x + 1.0 + root_y), 0.5 * (k_x + 1.0 - root_y), strain_y >= 0.0);
    return vec2f(k_x, k_y);
}

// END earth_pressure.wgsl
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
// import friction.wgsl;
// BEGIN friction.wgsl
// Shared basal friction models.
// Pure library module: it is textually imported and relies on the importing
// module providing the utils.wgsl symbols (sim_settings, g).
// Returns a deceleration magnitude (m/s^2) acting against the flow direction,
// given the effective bed-normal acceleration g_eff (m/s^2), the flow density
// (kg/m^3), the proposed flow speed (m/s) and the flow depth h (m).
// model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAT,
// 4 voellmy with cohesion (stub), 5 mu(I)
fn basal_friction_acceleration(g_eff: f32, density: f32, proposed_speed: f32, h: f32, model: u32) -> f32 {
    if proposed_speed < sim_settings.velocity_threshold || model >= 6u {
        return 0.0;
    }
    // normal stress is a magnitude: some callers (curvilinear) pass the
    // bed-normal acceleration with a negative sign convention
    let g_eff_magnitude = abs(g_eff);

    let mass_per_area = density * max(h, 1e-3);
    let normal_stress = g_eff_magnitude * mass_per_area;
    let friction_coefficient = sim_settings.friction_coefficient;
    var shear_stress = 0.0;
    // Coulomb friction model
    if model == 0u || model == 1u || model == 2u {
        shear_stress = friction_coefficient * normal_stress;
    }
    // samosAT friction model: Coulomb-like shear term with a density/speed
    // dependent correction plus a runup-limited turbulent drag term
    else if model == 3u {
        let rs0 = 0.222;
        let rs = density * proposed_speed * proposed_speed / (normal_stress + 0.001);
        shear_stress = normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs));
        let kappa_inv = 2.32558; // 1/kappa, standard kappa = 0.43
        let r_inv = 20.0; // 1/r, standard r = 0.05
        let b = 4.13;
        var div = max(h * r_inv, 1.0);
        div = log(div) * kappa_inv + b;
        shear_stress = shear_stress + density * proposed_speed * proposed_speed / (div * div);
    }
    // check https://ramms.ch/ramms-avalanche/friction-parameters/
    else if model == 4u {
        // let n0 = sim_settings.n0;
        // shear_stress = friction_coefficient * normal_stress + (1 - friction_coefficient) * n0 - (1 - friction_coefficient) * n0 * exp(-normal_stress / n0);
    }
    // mu(I) friction model with the depth-averaged inertial number of a
    // Bagnold profile (shear rate 5u/2h): I = 5/2 d u / (h sqrt(g h))
    else if model == 5u {
        let grain_diameter = sim_settings.grain_diameter;
        let i0 = sim_settings.i0;
        let mu0 = sim_settings.mu0;
        let mu2 = sim_settings.mu2;
        let h_clamped = max(h, 1e-3);
        let inertial_number = 2.5 * grain_diameter * proposed_speed
            / (h_clamped * sqrt(max(g_eff_magnitude, 1e-6) * h_clamped));
        let mu_i = mu0 + (mu2 - mu0) / (i0 / max(inertial_number, 1e-9) + 1.0);
        shear_stress = mu_i * normal_stress;
    }

    // Voellmy-style turbulent drag contribution
    if model == 1u || model == 2u {
        shear_stress = shear_stress + density * proposed_speed * proposed_speed * g / sim_settings.drag_coefficient;
    }
    // Voellmy min shear: a constant basal shear independent of load
    if model == 2u {
        shear_stress = shear_stress + 70.0;
    }

    return shear_stress / max(mass_per_area, 1e-6);
}

// Gully confinement: snow filling a gully presses its walls (inclined by beta
// against the smoothed reference surface) with the cross-flow stress
// k_y sigma_zz of the material beside it. Relative to the bed-normal load of
// the column above, the wall normal stress grows by 1 + k_y tan^2(beta).
// The bed-normal h already equals the hydraulic radius, so the
// velocity-dependent drag needs no wetted-area correction.
// cos_beta = 1 (no reference surface) gives 1.
fn gully_wall_load_factor(cos_beta: f32, k_y: f32) -> f32 {
    let tan2_beta = 1.0 / (cos_beta * cos_beta) - 1.0;
    return 1.0 + k_y * tan2_beta;
}

// END friction.wgsl
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
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

// reversible low-speed state; the low bits hold the rest start step while resting
// and the consecutive slow-step count while moving
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
    let low_mask = PARTICLE_RESTING - 1u;
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
