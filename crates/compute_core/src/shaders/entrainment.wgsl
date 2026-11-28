// Snow entrainment of the DAMPM global model.
// Pure library module: it is textually imported and relies on the importing
// module importing sim_settings.wgsl (sim_settings).
//
// Rate laws (sim_settings.entrainment_model, EntrainmentModel in settings.rs)
// for the entrained mass per bed area and time q (kg m^-2 s^-1), with the
// basal shear stress tau_b = rho h a_fric of the flow:
//   0 erosion energy (com1DFA / SamosAT; Sampl & Zwinger 2004, Fischer et
//     al. 2015): the basal frictional power erodes
//       q = tau_b |u| / e_b
//   1 excess shear stress (Naaim et al. 2003, Bouchut et al. 2008): the bed
//     fails where the flow's shear stress exceeds its strength tau_c,
//       q = k max(tau_b - tau_c, 0)
//   2 velocity-proportional (RAMMS; Christen et al. 2010): the snow cover is
//     taken up at a rate proportional to the flow speed,
//       q = kappa rho_e |u|
// The node's bed area dx^2 / n_z takes up dm = q dx^2 / n_z dt in a step, at
// most the remaining erodible snow rho_e h_e dx^2 / n_z. The entrained snow
// is at rest, so the node keeps its momentum: u <- u m / (m + dm). The work to
// break and deform the snow cover (com1DFA's e_s, e_d, 0 by default) is not
// modelled.
//
// State per node (EntrainmentState): x = remaining erodible depth h_e (m,
// bed-normal), y = dm / m of the current step, which the entrainment mass
// pass spreads onto the particles with their p2g weights. Since the node mass
// is m_i = sum_p w_ip m_p, the particles gain exactly sum_i dm_i.
// All entrainment physics lives here so the closures can be swapped as a unit.

const ENTRAINMENT_EROSION_ENERGY: u32 = 0u;
const ENTRAINMENT_EXCESS_SHEAR_STRESS: u32 = 1u;
const ENTRAINMENT_VELOCITY_PROPORTIONAL: u32 = 2u;

// entrained mass per bed area and time (kg m^-2 s^-1) of a flow with basal
// shear stress tau_b (Pa) and speed `speed`
fn entrainment_rate(tau_b: f32, speed: f32) -> f32 {
    switch sim_settings.entrainment_model {
        case ENTRAINMENT_EXCESS_SHEAR_STRESS: {
            return sim_settings.shear_erosion_coefficient * max(tau_b - sim_settings.critical_shear_stress, 0.0);
        }
        case ENTRAINMENT_VELOCITY_PROPORTIONAL: {
            return sim_settings.entrainment_velocity_coefficient * sim_settings.entrainment_density * speed;
        }
        default: {
            return tau_b * speed / sim_settings.erosion_energy;
        }
    }
}

// mass (kg) a node of flow mass `mass`, bed-normal thickness h and speed
// `speed` erodes in a step of dt from the remaining erodible depth h_e;
// a_fric is the basal friction deceleration (m/s^2) of the flow and n_z the
// z component of the bed normal. A node at rest erodes nothing.
fn entrainment_mass(mass: f32, h: f32, a_fric: f32, speed: f32, n_z: f32, h_e: f32, dt: f32) -> f32 {
    if h_e <= 0.0 || mass <= 0.0 || speed <= 0.0 {
        return 0.0;
    }
    let bed_area = sim_settings.cell_size * sim_settings.cell_size / max(n_z, 1e-3);
    let tau_b = sim_settings.snow_density * h * max(a_fric, 0.0);
    let rate = entrainment_rate(tau_b, speed);
    let available = sim_settings.entrainment_density * h_e * bed_area;
    return min(max(rate, 0.0) * bed_area * dt, available);
}

// erodible depth (m) left after a node of bed normal z component n_z takes up
// dm kg from the depth h_e
fn entrainment_deplete(h_e: f32, dm: f32, n_z: f32) -> f32 {
    let bed_area = sim_settings.cell_size * sim_settings.cell_size / max(n_z, 1e-3);
    return max(h_e - dm / (sim_settings.entrainment_density * bed_area), 0.0);
}
