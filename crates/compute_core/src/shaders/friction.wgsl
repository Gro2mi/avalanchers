// Shared basal friction models for the MPMDAC grid physics.
// All functions return a deceleration magnitude (m/s^2) acting against the
// flow direction, given the effective bed-normal acceleration g_perp (m/s^2),
// the proposed flow speed (m/s) and the flow depth h (m).
// friction_model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAT,
// 4 voellmy with cohesion (stub), 5 mu(I)
fn basal_friction_acceleration(g_perp: f32, proposed_speed: f32, h: f32) -> f32 {
    let model = sim_settings.friction_model;
    if proposed_speed < sim_settings.velocity_threshold || model >= 6u {
        return 0.0;
    }

    let mass_per_area = sim_settings.snow_density * max(h, 1e-3);
    let normal_stress = g_perp * mass_per_area;
    let friction_coefficient = sim_settings.friction_coefficient;
    var shear_stress = 0.0;
    // Coulomb friction model
    if model == 0u || model == 1u || model == 2u {
        shear_stress = friction_coefficient * normal_stress;
    }
    // samosAT friction model
    else if model == 3u {
        let rs0 = 0.222;
        let rs = sim_settings.snow_density * proposed_speed * proposed_speed / (normal_stress + 0.001);
        shear_stress = normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs));
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
        let inertial_number = 2.5 * sqrt(proposed_speed) / h * grain_diameter / sqrt(max(g_perp, 1e-6) * h);
        let mu_i = mu0 + (mu2 - mu0) / (i0 / inertial_number + 1.0);
        shear_stress = mu_i * normal_stress;
    }

    // Voellmy-style turbulent drag contribution
    if model == 1u || model == 2u {
        shear_stress = shear_stress + sim_settings.snow_density * proposed_speed * proposed_speed * g / sim_settings.drag_coefficient;
    }
    // Voellmy min shear: a constant basal shear independent of load
    if model == 2u {
        shear_stress = shear_stress + 70.0;
    }

    return shear_stress / max(mass_per_area, 1e-6);
}
