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

// Equivalent shear measures of 2D deviatoric tensors (Jop et al. 2006),
// which reduce to tau_xy and the shear rate gamma_dot in simple shear, so
// literature friction coefficients and mu(I) parameters apply directly:
//   tau_eq = sqrt(1/2 tau:tau),  gamma_dot = sqrt(2 eps_dot_dev:eps_dot_dev)
fn equivalent_shear_stress(tau_dev: mat2x2<f32>) -> f32 {
    return frobenius_norm(tau_dev) / sqrt(2.0);
}

fn equivalent_shear_rate(eps_dot_dev: mat2x2<f32>) -> f32 {
    return sqrt(2.0) * frobenius_norm(eps_dot_dev);
}

// Rate-dependent mu(I) inertial rheology: the friction coefficient grows
// from mu0 at rest towards mu2 with the inertial number
//   I = gamma_dot * d / sqrt(P / rho)
fn mu_inertial(shear_rate: f32, pressure: f32) -> f32 {
    let inertial_number = shear_rate * sim_settings.grain_diameter
        / sqrt(max(pressure / sim_settings.snow_density, 1e-6));
    return sim_settings.mu0
        + (sim_settings.mu2 - sim_settings.mu0)
            / (sim_settings.i0 / max(inertial_number, 1e-9) + 1.0);
}
