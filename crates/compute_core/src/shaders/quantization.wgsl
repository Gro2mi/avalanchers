// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The ranges adapt to the flow. reset_grid doubles a range once the
// fullest node of the last step filled half of it, and halves it once that
// node stayed below an eighth, so the integers resolve thin flows finely and
// still hold deep deposits. sim_info.mass_scale counts the mass range in
// reference nodes: a MAX_FLOW_THICKNESS deep column over a cell (density *
// thickness * cell_size^2); sim_info.momentum_scale counts the momentum range
// in that column moving at MAX_FLOW_VELOCITY. Both start at 1. Needs
// sim_settings and sim_info; must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 50.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
// fill of the fullest node at which a range doubles, and below which it halves
const QUANTIZATION_GROW_FILL: f32 = 0.5;
const QUANTIZATION_SHRINK_FILL: f32 = 0.125;
// fill at which the fullest node probably wrapped around
const QUANTIZATION_OVERFLOW_FILL: f32 = 0.9;
const MIN_MASS_SCALE: f32 = 1.0 / 256.0;
const MAX_QUANTIZATION_SCALE: f32 = 256.0;
// Smallest momentum range relative to the mass range: the full mass range at
// MAX_FLOW_VELOCITY / 8 = 12.5 m/s. A flow starting from rest (no momentum to
// measure) cannot outgrow it within one step.
const MIN_MOMENTUM_PER_MASS_SCALE: f32 = 0.125;

// mass (kg) of the reference node, a MAX_FLOW_THICKNESS deep column
fn reference_node_mass() -> f32 {
    return MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// largest node mass (kg) and momentum component (kg m/s) the integers hold
fn max_node_mass() -> f32 {
    return sim_info.mass_scale * reference_node_mass();
}
fn max_node_momentum() -> f32 {
    return sim_info.momentum_scale * reference_node_mass() * MAX_FLOW_VELOCITY;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// kg per unit of the last step's grid mass (MPMDAC's grid_mass_previous)
fn previous_inv_mass_factor() -> f32 {
    return sim_info.previous_mass_scale * reference_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / max_node_momentum();
}
fn inv_momentum_factor() -> f32 {
    return max_node_momentum() / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (12.5 g for 5 m cells and 200 kg/m3 at the start
// range). Lighter nodes count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// next range scale for a fullest node filling `fill` of the current range
fn adapted_quantization_scale(scale: f32, fill: f32) -> f32 {
    if fill >= QUANTIZATION_GROW_FILL {
        return scale * 2.0;
    }
    if fill < QUANTIZATION_SHRINK_FILL {
        return scale * 0.5;
    }
    return scale;
}
