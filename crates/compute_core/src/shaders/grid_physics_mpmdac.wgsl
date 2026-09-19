// MPMDAC (MPM depth-averaged curvilinear) grid update: terrain-parallel
// gravity, internal stress divergence forces and selectable basal friction
// in terrain-following coordinates on the surface.
// atomic_float @group(0) @binding(1) var<storage> grid_mass_atomic: array<f32>;
@group(0) @binding(1) var<storage> grid_mass_atomic: array<u32>; // no_atomic_float
@group(0) @binding(2) var terrain_geometry_texture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> peak_flow_thickness: array<f32>;
@group(0) @binding(4) var<storage, read_write> atomic_values: AtomicValues;
// atomic_float @group(0) @binding(5) var<storage> grid_momentum_atomic: array<f32>; // Combined u, v
@group(0) @binding(5) var<storage> grid_momentum_atomic: array<i32>; // no_atomic_float
@group(0) @binding(6) var<storage, read_write> new_cells_rolling_window: array<u32>;
@group(0) @binding(7) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(8) var<storage, read_write> grid_velocity: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> grid_peak_velocity: array<f32>;
// atomic_float @group(0) @binding(10) var<storage> grid_forces_atomic: array<f32>;
@group(0) @binding(10) var<storage> grid_forces_atomic: array<i32>; // no_atomic_float

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics_mpmdac(@builtin(global_invocation_id) id: vec3u) {
    if id.x < 1 || id.x >= (sim_settings.grid_shape.x - 1) || id.y < 1 || id.y >= (sim_settings.grid_shape.y - 1) {
        return;
    }
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }

    let idx = xy_to_idx(id.xy);
    let terrain_data = textureLoad(terrain_geometry_texture, vec2<i32>(id.xy), 0);
    let normal = normalize(terrain_data.xyz);
    let n_z = max(normal.z, 0.2);
    // surface area per horizontal area
    let J = 1.0 / n_z;

    // 1. Decode height and velocity
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    let u = f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR / (mass + 1e-6); // no_atomic_float
    let v = f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR / (mass + 1e-6); // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    // atomic_float let u = grid_momentum_atomic[idx * 2] / mass;
    // atomic_float let v = grid_momentum_atomic[idx * 2 + 1] / mass;
    let h = mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size * J);

    let current_peak_flow_thickness = peak_flow_thickness[idx];
    if current_peak_flow_thickness < h {
        // only if previously cell was empty, we count it as a new cell for diagnostics
        if current_peak_flow_thickness < 1e-5 {
            new_cells_rolling_window[sim_info.timestep % 40u] = new_cells_rolling_window[sim_info.timestep % 40u] + 1u;
        }
        peak_flow_thickness[idx] = h;
    }
    if bitcast<u32>(h) > atomicLoad(&atomic_values.peak_flow_thickness) {
        if !is_nan(h) {
            atomicMax(&atomic_values.peak_flow_thickness, bitcast<u32>(h));
        }
    }

    // 2. Velocity update: internal stress divergence + terrain-parallel gravity.
    // g_t = g * n_z * normal.xy is the horizontal projection of gravity onto
    // the tangent plane (magnitude g * sin(theta) * cos(theta))
    var proposed_velocity = vec2f(0.0);
    if h > 1e-4 {
        let force_x = f32(grid_forces_atomic[idx * 2u]) * INV_FORCE_FACTOR; // no_atomic_float
        let force_y = f32(grid_forces_atomic[idx * 2u + 1u]) * INV_FORCE_FACTOR; // no_atomic_float
        // atomic_float var force_acceleration = vec2f(grid_forces_atomic[idx * 2u], grid_forces_atomic[idx * 2u + 1u]) / (mass + 1e-6);
        var force_acceleration = vec2f(force_x, force_y) / (mass + 1e-6); // no_atomic_float
        // safety cap: plastic pressure waves are O(g); anything far above
        // indicates numerical breakdown and must not launch the flow
        let accel_magnitude = length(force_acceleration);
        if accel_magnitude > 5.0 * g {
            force_acceleration = force_acceleration * (5.0 * g / accel_magnitude);
        }
        let gravity_acceleration = g * n_z * normal.xy;
        proposed_velocity = vec2f(u, v) + sim_info.dt * (gravity_acceleration + force_acceleration);
    }
    var proposed_speed = length(proposed_velocity);

    // 3. Basal friction against the proposed flow direction
    if proposed_speed > sim_settings.velocity_threshold {
        let g_eff = g * n_z;
        let friction_acceleration = basal_friction_acceleration(g_eff, sim_settings.snow_density, proposed_speed, h, sim_settings.friction_model);

        // If friction would reverse or completely zero out the momentum this step
        if (friction_acceleration * sim_info.dt) >= proposed_speed {
            proposed_velocity = vec2f(0.0);
        } else {
            proposed_velocity = proposed_velocity * (1.0 - friction_acceleration * sim_info.dt / proposed_speed);
        }
        proposed_speed = length(proposed_velocity);
    }

    grid_velocity[idx] = proposed_velocity;

    let v_mag = proposed_speed;
    if !is_nan(v_mag) && h > 1e-3 && bitcast<u32>(v_mag) > atomicLoad(&atomic_values.peak_velocity) {
        atomicMax(&atomic_values.peak_velocity, bitcast<u32>(v_mag));
    }
    if !is_nan(v_mag) && h > 1e-3 {
        grid_peak_velocity[idx] = max(grid_peak_velocity[idx], v_mag);
    }

    // CFL signal speed: shallow-water gravity wave plus, for a compressible
    // material, the volumetric wave speed sqrt(K / rho)
    let expected_cell_velocity = v_mag + sqrt(g * h)
        + sqrt(sim_settings.bulk_modulus / sim_settings.snow_density);
    if !is_nan(expected_cell_velocity) && h > 1e-3 && bitcast<u32>(expected_cell_velocity) > atomicLoad(&atomic_values.expected_max_velocity) {
        atomicMax(&atomic_values.expected_max_velocity, bitcast<u32>(expected_cell_velocity));
    }
}

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
