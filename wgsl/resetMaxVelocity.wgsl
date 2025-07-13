
@group(0) @binding(1) var<storage, read_write> simInfo: SimInfo;
@group(0) @binding(2) var<storage, read_write> maxVelocity: AtomicValue;

@compute @workgroup_size(1)
fn resetMaxVelocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    simInfo.max_velocity = (f32(atomicLoad(&maxVelocity.value)) / f32(maxVelocityFactor)); // Load the current max velocity
    atomicStore(&maxVelocity.value, u32(0)); // Reset max velocity to 0 for the new timestep
}