// import utils.wgsl;

@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage, read_write> max_velocity: AtomicValue;

@compute @workgroup_size(1)
fn reset_max_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    sim_info.max_velocity = (f32(atomicLoad(&max_velocity.value)) / f32(MAX_VELOCITY_FACTOR)); // Load the current max velocity
    atomicStore(&max_velocity.value, u32(0)); // Reset max velocity to 0 for the new timestep
}