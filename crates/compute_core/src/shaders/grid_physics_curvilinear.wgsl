@group(0) @binding(1) var<storage> grid_mass_atomic: array<u32>; // no_atomic_float
// atomic_float @group(0) @binding(1) var<storage> grid_mass_atomic: array<f32>;
@group(0) @binding(2) var terrain_geometry_texture: texture_2d<f32>;
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
// declared for curvilinear_geometry.wgsl, unused by this entry point
@group(0) @binding(12) var dem_texture: texture_2d<f32>;
@group(0) @binding(13) var tex_sampler: sampler;
@group(0) @binding(14) var<storage, read_write> particles_elevation: array<f32>;


@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics_curvilinear(@builtin(global_invocation_id) id: vec3u) {
    if id.x < 1 || id.x >= (sim_settings.grid_shape.x - 1) || id.y < 1 || id.y >= (sim_settings.grid_shape.y - 1) {
        return;
    }
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }

    let use_curvature: bool = (sim_settings.flags & (1u << 0u)) != 0u;
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    let use_earth_pressure_coefficient: bool = (sim_settings.flags & (1u << 2u)) != 0u;
    let use_entrainment: bool = (sim_settings.flags & (1u << 3u)) != 0u;

    let idx = xy_to_idx(id.xy);
    // No particle gathers from a node whose 3x3 neighbourhood holds no mass:
    // each particle deposits at least a quarter of its mass onto its stencil
    // centre, which neighbours all nine nodes the particle gathers from.
    if neighbourhood_is_empty(id.xy) {
        grid_velocity[idx] = vec2f(0.0);
        return;
    }
    let terrain_geometry_texture_data = textureLoad(terrain_geometry_texture, vec2<i32>(id.xy), 0);

    // 2. Unpack precalculated elements
    let l_x = terrain_geometry_texture_data.x;
    let l_y = terrain_geometry_texture_data.y;
    let J = terrain_geometry_texture_data.z;
    let g_y = terrain_geometry_texture_data.w;

    let curvature_texture_data = textureLoad(curvature_texture, vec2<i32>(id.xy), 0);
    let K_xx = curvature_texture_data.x;
    let K_yy = curvature_texture_data.y;
    let K_xy = curvature_texture_data.z;
    let g_x = curvature_texture_data.w;
    let slope = -vec2f(g_x * l_x, g_y * l_y) / g;
    // inclination of the bed against the smoothed reference surface (gully walls)
    let cos_beta = textureLoad(basal_topography_texture, vec2<i32>(id.xy), 0).x;

    // 1. Decode height and velocity
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    // Nodes holding less mass than the p2g quantization floor carry no
    // meaningful velocity: their mass can round to 0 while the momentum
    // quantization still survives, and momentum/mass then explodes (this
    // feeds straight into the PIC transfer). Treat such nodes as empty.
    let node_is_empty = mass < 1e-2;
    var u = select(f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR / (mass + 1e-6), 0.0, node_is_empty); // no_atomic_float
    var v = select(f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR / (mass + 1e-6), 0.0, node_is_empty); // no_atomic_float
    // atomic_float var u = grid_momentum_atomic[idx * 2] / mass;
    // atomic_float var v = grid_momentum_atomic[idx * 2 + 1] / mass;

    // Geometric transport is required even when the curvature correction
    // to basal normal load is disabled.
    let centripetal_acceleration = K_xx * u * u + 2.0 * K_xy * u * v + K_yy * v * v;
    let curvature_acceleration = -slope * centripetal_acceleration / J;
    var g_normal_magnitude = g / J;
    if use_curvature {
        g_normal_magnitude = max(g_normal_magnitude + centripetal_acceleration, 0.0);
    }
    let g_normal = -g_normal_magnitude;

    let h =  mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size * J);
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

    // let grad_h2 = vec2f(
    //     (get_h2(id.x + 1, id.y) - get_h2(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size / l_x),
    //     (get_h2(id.x, id.y + 1) - get_h2(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size / l_y)
    // );
    // let k = 1.0;
    // let grad_h = grad_h2 * 0.5 / (h + 1e-5);
    let grad_h = vec2f(
        (get_h(id.x + 1, id.y) - get_h(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size),
        (get_h(id.x, id.y + 1) - get_h(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size)
    );
    // Savage-Hutter earth pressure (Hutter et al. 1993): k_x along the fall
    // line, k_y across it, each active where the flow stretches in that
    // direction and passive where it is compressed; hydrostatic when disabled
    var k_grad_h = grad_h;
    var k_y = 1.0;
    if use_earth_pressure_coefficient {
        let e_x = earth_pressure_axis(slope, vec2f(u, v));
        let e_y = vec2f(-e_x.y, e_x.x);
        let grad_q = velocity_gradient(id.x, id.y);
        let k = earth_pressure_coefficients(
            radians(sim_settings.internal_friction_angle),
            radians(sim_settings.basal_friction_angle),
            dot(e_x, grad_q * e_x),
            dot(e_y, grad_q * e_y)
        );
        k_y = k.y;
        // e_x lies along the gradient of the bed, so this tensor commutes
        // with the inverse surface metric
        k_grad_h = k.x * dot(e_x, grad_h) * e_x + k.y * dot(e_y, grad_h) * e_y;
    }
    let pressure_gradient = select(vec2f(0.0), k_grad_h, use_particle_interaction);
    let hydrostatic_pressure_acceleration = -g_normal * surface_inverse_metric(slope, pressure_gradient);
    let gravity_acceleration = -g * slope / (J * J);

    // 1. Calculate velocity update from gravity alone first
    // var proposed_u = u + sim_info.dt * g  * sin(degrees(34f));//g_x;
    var proposed_u = u + sim_info.dt * (gravity_acceleration.x + curvature_acceleration.x - hydrostatic_pressure_acceleration.x);
    var proposed_v = v + sim_info.dt * (gravity_acceleration.y + curvature_acceleration.y - hydrostatic_pressure_acceleration.y);
    var proposed_V_mag = surface_speed(slope, vec2f(proposed_u, proposed_v));

    let g_eff = g_normal;

    // Friction is applied at every speed, so Coulomb resistance also acts as
    // static friction: a node at rest stays at rest while the driving
    // acceleration is below the yield value, independent of dt. Evaluating at
    // no less than velocity_threshold keeps the low-speed gate of the shared
    // friction model from switching it off.
    let friction_speed = max(proposed_V_mag, sim_settings.velocity_threshold);
    // Gully confinement: snow filling a gully presses its walls (inclined by
    // beta against the smoothed reference surface) with the cross-flow stress
    // k_y sigma_zz of the material beside it. Relative to the bed-normal load
    // of the column above, the wall normal stress grows by 1 + k_y tan^2(beta).
    // The bed-normal h already equals the hydraulic radius, so the
    // velocity-dependent drag needs no wetted-area correction.
    let tan2_beta = 1.0 / (cos_beta * cos_beta) - 1.0;
    let wall_load_factor = 1.0 + k_y * tan2_beta;
    let friction_acceleration = basal_friction_acceleration(g_eff * wall_load_factor, sim_settings.snow_density, friction_speed, h, sim_settings.friction_model);

    // If friction would reverse or completely zero out the momentum this step
    if (friction_acceleration * sim_info.dt) >= proposed_V_mag {
        u = 0.0;
        v = 0.0;
    } else {
        // Apply friction cleanly as a scaling factor
        let friction_scale = 1.0 - (friction_acceleration * sim_info.dt / proposed_V_mag);
        u = proposed_u * friction_scale;
        v = proposed_v * friction_scale;
    }

    // u = 

    grid_velocity[idx] = vec2f(u, v);

    let v_mag = surface_speed(slope, vec2f(u, v));
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

    // gravity waves travel at sqrt(g cos(theta) h) on the inclined bed;
    // cos(theta) = 1 / J
    let expected_cell_velocity = v_mag + sqrt(g * h / J);
    if !is_nan(expected_cell_velocity) && h > 1e-3 && bitcast<u32>(expected_cell_velocity) > atomicLoad(&atomic_values.expected_max_velocity) {
        atomicMax(&atomic_values.expected_max_velocity, bitcast<u32>(expected_cell_velocity));
    }
    // if !is_nan(V_mag) {
    //     grid_peak_velocity[idx] = max(grid_peak_velocity[idx], V_mag);
    // }
    //  grid_peak_velocity[idx] = V_mag;
    // V_mag = 2f;
    // if V_mag > 1f {
    //     grid_peak_velocity[idx] = max(grid_peak_velocity[idx], V_mag);
    // }
}

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

fn get_h2(x: u32, y: u32) -> f32 {
    let idx = x_y_to_idx(x, y);
    // return pow(f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0);
    return pow(f32(grid_mass_atomic[idx]) / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0);
}
fn get_h(x: u32, y: u32) -> f32 {
    let idx = x_y_to_idx(x, y);
    let J = textureLoad(terrain_geometry_texture, vec2i(i32(x), i32(y)), 0).z;
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    return mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size * J);
}

fn get_h2_atomic_u32(x: u32, y: u32) -> f32 {
    let idx = x_y_to_idx(x, y);
    return pow(f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0);
}

fn minmod(a: f32, b: f32) -> f32 {
    // Returns the smallest magnitude slope if signs match, otherwise 0.0
    return 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b));
}

// Velocity q of node (x, y) from the node's own mass and momentum; w = 1 when
// the node holds mass, 0 below the p2g mass floor where q is meaningless.
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

// Savage-Hutter downslope axis: the fall line of the reference surface, the
// flow direction on flat ground.
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

// import curvilinear_geometry.wgsl;
// BEGIN curvilinear_geometry.wgsl
// Velocities are horizontal contravariant components q = (dx/dt, dy/dt).
// For z = B(x,y), G = I + grad(B) grad(B)^T.
fn surface_inverse_metric(slope: vec2f, covector: vec2f) -> vec2f {
    return covector - slope * dot(slope, covector) / (1.0 + dot(slope, slope));
}

fn surface_velocity(slope: vec2f, velocity: vec2f) -> vec3f {
    return vec3f(velocity, dot(slope, velocity));
}

fn surface_speed(slope: vec2f, velocity: vec2f) -> f32 {
    return length(surface_velocity(slope, velocity));
}

fn terrain_slope_at_uv(uv: vec2f) -> vec2f {
    let offset = 1.0 / vec2f(sim_settings.grid_shape);
    return vec2f(
        textureSampleLevel(dem_texture, tex_sampler, uv + vec2f(offset.x, 0.0), 0).x
            - textureSampleLevel(dem_texture, tex_sampler, uv - vec2f(offset.x, 0.0), 0).x,
        textureSampleLevel(dem_texture, tex_sampler, uv + vec2f(0.0, offset.y), 0).x
            - textureSampleLevel(dem_texture, tex_sampler, uv - vec2f(0.0, offset.y), 0).x
    ) / (2.0 * sim_settings.cell_size);
}

// END curvilinear_geometry.wgsl
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
    // mu(I) friction model
    else if model == 5u {
        let grain_diameter = sim_settings.grain_diameter;
        let i0 = sim_settings.i0;
        let mu0 = sim_settings.mu0;
        let mu2 = sim_settings.mu2;
        let inertial_number = 2.5 * sqrt(proposed_speed) / h * grain_diameter / sqrt(max(g_eff_magnitude, 1e-6) * h);
        let mu_i = mu0 + (mu2 - mu0) / (i0 / inertial_number + 1.0);
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
    // stop once total momentum < fraction * its peak; 0 disables (curvilinear)
    momentum_stop_fraction: f32,
    // front travel in cells covered by the no-new-cells stop window
    no_new_cells_travel: f32,
    // Gaussian width (m) of the curvilinear reference surface; 0 disables it
    reference_surface_length: f32,
    _pad1: u32,
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