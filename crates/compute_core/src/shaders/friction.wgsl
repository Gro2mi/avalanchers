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
