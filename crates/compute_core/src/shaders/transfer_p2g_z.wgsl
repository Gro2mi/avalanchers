// z row of the 3D APIC transfer, on the stencil of transfer_p2g_with_affine.
// Pure library module: the importing module provides particles_velocity_z,
// particles_affine_z and grid_momentum_z_atomic.
fn transfer_p2g_z(p_idx: u32) {
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    let p_mass = particles_mass[p_idx];
    let p_velocity_z = particles_velocity_z[p_idx];
    let affine_z = particles_affine_z[p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let affine_velocity_z = p_velocity_z - dot(affine_z, distance);
            // atomic_float atomicAdd(&grid_momentum_z_atomic[idx], p_mass * affine_velocity_z * weight);
            let quantized_z = quantize_momentum_component(p_mass * affine_velocity_z * weight * MOMENTUM_FACTOR); // no_atomic_float
            if quantized_z != 0 { atomicAdd(&grid_momentum_z_atomic[idx], quantized_z); } // no_atomic_float
        }
    }
}
