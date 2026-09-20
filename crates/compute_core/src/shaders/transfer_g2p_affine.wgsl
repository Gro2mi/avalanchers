fn transfer_g2p(p_idx: u32) -> G2PUpdate {
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    var interpolated_velocity = vec2f(0.0);
    var new_affine_matrix = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let grid_vel = grid_velocity[idx];
            interpolated_velocity += weight * grid_vel;
            new_affine_matrix += 4.0 * weight * mat2x2<f32>(
                grid_vel * distance.x,
                grid_vel * distance.y
            );
        }
    }
    return G2PUpdate(interpolated_velocity, new_affine_matrix);
}
