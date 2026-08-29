fn transfer_p2g(p_idx: u32) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    // let affine_matrix = particles_affine_matrix[p_idx];

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            // let affine_velocity = p_velocity + (affine_matrix * distance);

            
            atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * p_velocity.x * weight);
            atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * p_velocity.y * weight);
            // for affine transfer
            // atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);

            // no_atomic_float atomicAdd(&grid_mass_atomic[idx], u32(round(p_mass * weight * MASS_FACTOR)));
            // no_atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u], i32(round(p_mass * p_velocity.x * weight * MOMENTUM_FACTOR)));
            // no_atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], i32(round(p_mass * p_velocity.y * weight * MOMENTUM_FACTOR)));
        }
    }
}