// MPMDAC (MPM depth-averaged curvilinear) particle-to-grid transfer: mass,
// APIC momentum and internal forces from the per-particle constitutive stress.
@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_velocity: array<vec2<f32>>;
@group(0) @binding(4) var<storage> particles_mass: array<f32>;
// atomic_float @group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<f32>>;
@group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<u32>>; // no_atomic_float
// atomic_float @group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<f32>>;
@group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<i32>>; // no_atomic_float
@group(0) @binding(7) var<storage> particles_affine_matrix: array<mat2x2<f32>>;
// atomic_float @group(0) @binding(8) var<storage, read_write> grid_forces_atomic: array<atomic<f32>>;
@group(0) @binding(8) var<storage, read_write> grid_forces_atomic: array<atomic<i32>>; // no_atomic_float
@group(0) @binding(9) var terrain_geometry_texture: texture_2d<f32>;
@group(0) @binding(10) var curvature_texture: texture_2d<f32>;
@group(0) @binding(11) var<storage> grid_mass_previous: array<u32>;
// per-particle stress state: column 0 = (tau_xx, tau_xy), column 1 =
// (tau_yy, cumulative plastic strain); the deviatoric stress is symmetric
@group(0) @binding(12) var<storage, read_write> particles_stress: array<mat2x2<f32>>;
// per-particle volumetric state: (elastic volumetric strain, plastic
// compaction strain); compaction strain > 0 means the material densified
@group(0) @binding(13) var<storage, read_write> particles_volumetric_strain: array<vec2<f32>>;

override WG_SIZE_1D: u32 = 1u;
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn p2g_mpmdac(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles {
        return;
    }
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }
    let cell = position_to_cell(particles_position[id.x]);
    if cell.x < 1 || cell.x >= (sim_settings.grid_shape.x - 1) || cell.y < 1 || cell.y >= (sim_settings.grid_shape.y - 1) {
        return;
    }

    transfer_p2g_mpmdac(id.x, cell);
}

fn quantize_i32(value: f32) -> i32 {
    // Keep conversion inside i32 range before atomicAdd.
    let i32_max_f = 2147483647.0;
    let clamped = clamp(value, -i32_max_f, i32_max_f);
    return i32(round(clamped));
}

fn transfer_p2g_mpmdac(p_idx: u32, cell: vec2u) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));

    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    let affine_matrix = particles_affine_matrix[p_idx];

    // terrain metrics at the particle cell (terrain analysis of the
    // terrain-following model: xyz = surface normal)
    let terrain_data = textureLoad(terrain_geometry_texture, vec2<i32>(cell), 0);
    let normal = normalize(terrain_data.xyz);
    let n_z = max(normal.z, 0.2);
    // surface area per horizontal area
    let J = 1.0 / n_z;

    // MPMDAC particle depth from the previous step's grid mass, gathered with the
    // same B-spline weights over the 3x3 node stencil. Unlike a per-particle
    // depth from det(F) this sees sub-cell clumping: particles that bunch up
    // sit on locally higher column mass and receive the spreading pressure
    // gradient, instead of accumulating invisible compression that releases
    // in bursts.
    var gathered_mass = 0.0;
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            gathered_mass += weight * f32(grid_mass_previous[xy_to_idx(node_coords)]) * INV_MASS_FACTOR;
        }
    }
    let h_p = gathered_mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size * J);

    // effective bed-normal acceleration, with optional centrifugal curvature
    // correction: convex along-flow curvature (kappa < 0) lifts the flow
    let use_curvature: bool = (sim_settings.flags & (1u << 0u)) != 0u;
    var g_perp = g * n_z;
    if use_curvature {
        let speed = length(p_velocity);
        if speed > 1e-4 {
            let flow_direction = p_velocity / speed;
            let curvature_data = textureLoad(curvature_texture, vec2<i32>(cell), 0);
            let kappa = flow_direction.x * flow_direction.x * curvature_data.x
                + 2.0 * flow_direction.x * flow_direction.y * curvature_data.y
                + flow_direction.y * flow_direction.y * curvature_data.z;
            g_perp = max(g_perp + speed * speed * kappa, 0.1);
        }
    }

    // Drucker-Prager / mu(I) elastic predictor-corrector. The stored affine
    // matrix is the velocity gradient in cell units -> convert to 1/s.
    let velocity_gradient = affine_matrix * (1.0 / sim_settings.cell_size);
    let eps_dot = strain_rate_tensor(velocity_gradient);
    let eps_dot_dev = mat2_deviatoric(eps_dot);

    // Volumetric (compressible) response: the elastic volumetric strain
    // evolves with the APIC divergence; its stress adds to the lithostatic
    // pressure so compression stiffens and dilation relieves the flow.
    // One-sided plastic compaction: above the compaction pressure the
    // material densifies irreversibly (the elastic part relaxes onto the
    // compaction plateau), dilation stays elastic and clamped.
    let volumetric_state = particles_volumetric_strain[p_idx];
    var eps_v_e = volumetric_state.x;
    var eps_v_p = volumetric_state.y;
    var p_vol = 0.0;
    if sim_settings.bulk_modulus > 0.0 {
        let divergence = velocity_gradient[0][0] + velocity_gradient[1][1];
        let d_eps_v = clamp(sim_info.dt * divergence, -0.5, 0.5);
        eps_v_e = eps_v_e + d_eps_v;
        p_vol = -sim_settings.bulk_modulus * eps_v_e;
        if p_vol > sim_settings.compaction_pressure {
            let delta = eps_v_e + sim_settings.compaction_pressure / sim_settings.bulk_modulus;
            eps_v_p = eps_v_p + delta;
            eps_v_e = -sim_settings.compaction_pressure / sim_settings.bulk_modulus;
            p_vol = sim_settings.compaction_pressure;
        }
        if eps_v_e > 0.5 {
            eps_v_e = 0.5;
            p_vol = -sim_settings.bulk_modulus * eps_v_e;
        }
        if is_nan(eps_v_e) {
            eps_v_e = 0.0;
            p_vol = 0.0;
        }
    }
    particles_volumetric_strain[p_idx] = vec2f(eps_v_e, eps_v_p);

    // plastic compaction densifies the material; the particle volume in the
    // force scatter shrinks accordingly
    let density_ratio = max(1.0 + eps_v_p, 0.5);
    let particle_volume = p_mass / (sim_settings.snow_density * density_ratio);

    // total mean pressure: lithostatic (from the grid depth) plus the elastic
    // volumetric response, floored at zero (no tension)
    let pressure = max(
        0.5 * sim_settings.snow_density * g_perp * h_p + p_vol,
        0.0
    );

    // 1. elastic predictor: trial stress = old stress + 2G * dt * eps_dot_dev
    let stress_state = particles_stress[p_idx];
    let tau_old = mat2x2<f32>(
        vec2f(stress_state[0][0], stress_state[0][1]),
        vec2f(stress_state[1][0], 0.0)
    );
    let plastic_strain = stress_state[1][1];
    let tau_trial = tau_old + (2.0 * sim_settings.shear_modulus * sim_info.dt) * eps_dot_dev;
    let tau_trial_magnitude = frobenius_norm(tau_trial);

    // 2. yield stress: Drucker-Prager (pressure-dependent) or mu(I)
    // (rate-dependent), both with linear isotropic hardening on the
    // accumulated plastic strain
    let strain_rate_magnitude = frobenius_norm(eps_dot_dev);
    var yield_stress = 0.0;
    if sim_settings.constitutive_model == 1u {
        let mu_i = mu_inertial(strain_rate_magnitude, pressure);
        yield_stress = sim_settings.n0 + mu_i * pressure + sim_settings.hardening_modulus * plastic_strain;
    } else {
        yield_stress = sim_settings.n0
            + pressure * tan(radians(sim_settings.internal_friction_angle))
            + sim_settings.hardening_modulus * plastic_strain;
    }

    // 3. plastic corrector: radial return to the yield surface
    var tau_new = tau_trial;
    var delta_plastic_strain = 0.0;
    if tau_trial_magnitude > yield_stress {
        tau_new = tau_trial * (yield_stress / tau_trial_magnitude);
        delta_plastic_strain = (tau_trial_magnitude - yield_stress)
            / (2.0 * sim_settings.shear_modulus + sim_settings.hardening_modulus);
    }
    if is_nan(tau_new[0][0]) || is_nan(tau_new[0][1]) || is_nan(tau_new[1][0]) {
        tau_new = mat2x2<f32>(vec2f(0.0), vec2f(0.0));
        delta_plastic_strain = 0.0;
    }
    particles_stress[p_idx] = mat2x2<f32>(
        vec2f(tau_new[0][0], tau_new[0][1]),
        vec2f(tau_new[1][0], plastic_strain + delta_plastic_strain)
    );

    // total stress = deviatoric + mean compression, depth-integrated
    let sigma = tau_new - pressure * identity_2x2();
    let stress_integral = h_p * sigma;

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);

            let affine_velocity = p_velocity + (affine_matrix * distance);
            // atomic_float atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            // atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);
            atomicAdd(&grid_mass_atomic[idx], u32(round(p_mass * weight * MASS_FACTOR))); // no_atomic_float
            let momentum_x = p_mass * affine_velocity.x * weight * MOMENTUM_FACTOR; // no_atomic_float
            let momentum_y = p_mass * affine_velocity.y * weight * MOMENTUM_FACTOR; // no_atomic_float
            atomicAdd(&grid_momentum_atomic[idx * 2u], quantize_i32(momentum_x)); // no_atomic_float
            atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantize_i32(momentum_y)); // no_atomic_float

            // internal elastic/plastic force f_i -= V_p * (h_p * sigma_p) * grad_w_ip
            let grad_w = calculate_weight_gradient(distance);
            let internal_force = (-particle_volume) * stress_integral * grad_w;
            // atomic_float atomicAdd(&grid_forces_atomic[idx * 2u], internal_force.x);
            // atomic_float atomicAdd(&grid_forces_atomic[idx * 2u + 1u], internal_force.y);
            atomicAdd(&grid_forces_atomic[idx * 2u], quantize_i32(internal_force.x * FORCE_FACTOR)); // no_atomic_float
            atomicAdd(&grid_forces_atomic[idx * 2u + 1u], quantize_i32(internal_force.y * FORCE_FACTOR)); // no_atomic_float
        }
    }
}

// import constitutive.wgsl;
// BEGIN constitutive.wgsl
// Drucker-Prager / Mohr-Coulomb elastic predictor-corrector with mu(I)
// option for the MPMDAC (MPM depth-averaged curvilinear) model. Stresses are
// evaluated per particle from the APIC velocity gradient; the yield stress
// scales with the lithostatic pressure so active/passive earth-pressure
// behaviour emerges naturally.

// symmetric strain-rate tensor from the velocity gradient
fn strain_rate_tensor(velocity_gradient: mat2x2<f32>) -> mat2x2<f32> {
    return 0.5 * (velocity_gradient + transpose(velocity_gradient));
}

fn mat2_deviatoric(m: mat2x2<f32>) -> mat2x2<f32> {
    let trace = m[0][0] + m[1][1];
    return m - (0.5 * trace) * identity_2x2();
}

fn frobenius_norm(m: mat2x2<f32>) -> f32 {
    return sqrt(max(
        m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1],
        0.0
    ));
}

// Rate-dependent mu(I) inertial rheology: the friction coefficient grows
// from mu0 at rest towards mu2 with the inertial number
//   I = |eps_dot_dev| * d / sqrt(P / rho)
// (the norm convention absorbs the order-unity shear-rate factor into i0).
fn mu_inertial(strain_rate_magnitude: f32, pressure: f32) -> f32 {
    let inertial_number = strain_rate_magnitude * sim_settings.grain_diameter
        / sqrt(max(pressure / sim_settings.snow_density, 1e-6));
    return sim_settings.mu0
        + (sim_settings.mu2 - sim_settings.mu0)
            / (sim_settings.i0 / max(inertial_number, 1e-9) + 1.0);
}

// END constitutive.wgsl
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
