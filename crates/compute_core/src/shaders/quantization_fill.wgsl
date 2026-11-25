// Records how full the integer-atomic grid ranges are (quantization.wgsl):
// the largest node mass (u32 units) and momentum component (|i32| units) of
// this step, which reset_grid reads at the start of the next one. Needs
// atomic_values; called from the grid physics on the integer grid only.
fn record_quantization_fill(mass_units: u32, momentum_units: vec3<i32>) {
    if mass_units > atomicLoad(&atomic_values.max_mass_units) {
        atomicMax(&atomic_values.max_mass_units, mass_units);
    }
    // i32 -> u32 keeps the bits, so abs(i32 min) still reads as 2^31
    let momentum = u32(max(max(abs(momentum_units.x), abs(momentum_units.y)), abs(momentum_units.z)));
    if momentum > atomicLoad(&atomic_values.max_momentum_units) {
        atomicMax(&atomic_values.max_momentum_units, momentum);
    }
}
