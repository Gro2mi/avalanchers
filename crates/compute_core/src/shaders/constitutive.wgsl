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
