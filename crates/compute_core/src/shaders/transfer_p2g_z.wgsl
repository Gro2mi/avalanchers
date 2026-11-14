// 3D APIC transfer: mass, xy momentum and the z momentum (with the z row of
// the affine matrix) in one pass over the stencil, so the 3D model computes
// the weights once. Pure library module: the importing module provides the
// particle buffers, grid_mass_atomic, grid_momentum_atomic and
// grid_momentum_z_atomic, and imports transfer_p2g.wgsl (quantize_*).
fn transfer_p2g_3d(p_idx: u32, affine_matrix: mat2x2f) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    let p_velocity_z = particles_velocity_z[p_idx];
    let affine_z = particles_affine_z[p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let affine_velocity = p_velocity - affine_matrix * distance;
            let affine_velocity_z = p_velocity_z - dot(affine_z, distance);
            let node_mass = p_mass * weight;
            // atomic_float atomicAdd(&grid_mass_atomic[idx], node_mass);
            // resting particles carry no momentum: skip their zero adds
            // atomic_float let momentum = node_mass * vec3f(affine_velocity, affine_velocity_z);
            // atomic_float if momentum.x != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u], momentum.x); }
            // atomic_float if momentum.y != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], momentum.y); }
            // atomic_float if momentum.z != 0.0 { atomicAdd(&grid_momentum_z_atomic[idx], momentum.z); }
            let quantized_mass = quantize_mass(node_mass); // no_atomic_float
            if quantized_mass == 0u { continue; } // no momentum without mass, see quantize_mass // no_atomic_float
            atomicAdd(&grid_mass_atomic[idx], quantized_mass); // no_atomic_float
            let quantized_x = quantize_momentum_component(node_mass * affine_velocity.x * momentum_factor()); // no_atomic_float
            let quantized_y = quantize_momentum_component(node_mass * affine_velocity.y * momentum_factor()); // no_atomic_float
            let quantized_z = quantize_momentum_component(node_mass * affine_velocity_z * momentum_factor()); // no_atomic_float
            if quantized_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], quantized_x); } // no_atomic_float
            if quantized_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantized_y); } // no_atomic_float
            if quantized_z != 0 { atomicAdd(&grid_momentum_z_atomic[idx], quantized_z); } // no_atomic_float
        }
    }
}
