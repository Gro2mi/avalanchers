// Forest model of the DAMPM global model, after the NAKSIN / MoT-Voellmy
// forest description (NGI): standing stems exert a quadratic drag on the flow
// and break once the bending moment of the flow exceeds their strength, after
// which the standing forest decays and its drag vanishes.
// Pure library module: it is textually imported and relies on the importing
// module importing sim_settings.wgsl (sim_settings).
//
// State per node: the standing canopy cover c (0..1), initialised with the
// input cover and lowered by the destruction. All tree parameters derive from
// the cover and the settings:
//   stem density   n = c / (pi/4 d_c^2)            (crowns tile the cover)
//   stem diameter  D = forest_trunk_diameter
// Drag per unit flow mass (the MoT form 1/2 rho C_D n D h u^2 / (rho h)),
// with the flow height loading a stem capped at the tree height H:
//   a_F = 1/2 C_D n D min(h, H) / h u^2
// Breaking: bending moment at the stem base of the drag force on one stem,
// acting at half the loaded height, against the elastic section modulus:
//   M   = 1/2 rho C_D D h_eff u^2 * h_eff / 2
//   M_c = sigma_MOR pi D^3 / 32
// Decay (assumption, the source formula is not available): the decay
// coefficient c_f (m/s) is read as the speed at which the fracture runs
// through a stem, so a loaded stand fails on the time scale D / c_f:
//   dc/dt = -c c_f / D   while M >= M_c
// integrated exactly over a step, c <- c exp(-dt c_f / D).
// All forest physics lives here so the closures can be swapped as a unit.

const FOREST_PI: f32 = 3.14159265358979;

// stems per m^2 of a canopy cover fraction
fn forest_stem_density(cover: f32) -> f32 {
    let d_c = sim_settings.forest_crown_diameter;
    return max(cover, 0.0) / (0.25 * FOREST_PI * d_c * d_c);
}

// flow height (m) loading a stem
fn forest_loaded_height(h: f32) -> f32 {
    return min(max(h, 0.0), sim_settings.forest_tree_height);
}

// deceleration (m/s^2) of a flow of bed-normal height h and speed u by the
// stems of the standing cover
fn forest_drag_acceleration(standing_cover: f32, h: f32, speed: f32) -> f32 {
    if standing_cover <= 0.0 {
        return 0.0;
    }
    let n = forest_stem_density(standing_cover);
    let h_safe = max(h, 1e-3);
    let loaded_fraction = min(forest_loaded_height(h_safe) / h_safe, 1.0);
    return 0.5 * sim_settings.forest_drag_coefficient * n * sim_settings.forest_trunk_diameter
        * loaded_fraction * speed * speed;
}

// whether the flow breaks the stems: bending moment at the base above the
// moment of rupture
fn forest_breaks(h: f32, speed: f32) -> bool {
    let d = sim_settings.forest_trunk_diameter;
    let h_eff = forest_loaded_height(h);
    let moment = 0.25 * sim_settings.snow_density * sim_settings.forest_drag_coefficient * d
        * h_eff * h_eff * speed * speed;
    let moment_of_rupture = sim_settings.forest_modulus_of_rupture * FOREST_PI * d * d * d / 32.0;
    return moment >= moment_of_rupture;
}

// standing cover after a step of dt of breaking load
fn forest_decay(standing_cover: f32, dt: f32) -> f32 {
    let rate = sim_settings.forest_decay_coefficient / sim_settings.forest_trunk_diameter;
    return standing_cover * exp(-dt * rate);
}
