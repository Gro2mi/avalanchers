fn transfer_g2p(p_idx: u32) -> vec2f {
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    var interpolated_velocity = vec2f(0.0);

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let weight = calculate_weight(grid_pos, node_coords);
            let idx = xy_to_idx(node_coords);
            
            interpolated_velocity += weight * grid_velocity[idx];
        }
    }
    return interpolated_velocity;
}
