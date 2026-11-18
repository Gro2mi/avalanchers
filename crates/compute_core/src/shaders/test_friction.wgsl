// Verifies every basal friction model on a fixed test case:
// g_eff = 9.81 m/s^2, density = 200 kg/m^3, proposed speed = 10 m/s,
// flow depth h = 1 m. The friction parameters (coefficient, drag,
// grain diameter, mu(I) parameters) come from sim_settings, which the CPU
// test initializes with known values before dispatching.
// Dispatch exactly one workgroup: invocation i evaluates friction model i
// and writes the resulting deceleration to test_results[i]; the CPU test
// compares each value against the same formula evaluated on the CPU.

@group(0) @binding(1) var<storage, read_write> test_results: array<f32>;

@compute @workgroup_size(6, 1, 1)
fn test_friction(@builtin(local_invocation_index) li: u32) {
    test_results[li] = basal_friction_acceleration(9.81, 200.0, 10.0, 1.0, li);
}

// import friction.wgsl;
// BEGIN friction.wgsl
// Shared basal friction models.
// Pure library module: it is textually imported and relies on the importing
// module importing constants.wgsl and sim_settings.wgsl (sim_settings, g).
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
