fn quantize_momentum_component(value: f32) -> i32 {
    // Keep conversion inside i32 range before atomicAdd.
    let i32_max_f = 2147483647.0;
    let clamped = clamp(value, -i32_max_f, i32_max_f);
    return i32(round(clamped));
}

// Quantized mass contribution of the integer grid (mass_factor() in
// utils.wgsl). A contribution whose mass rounds to zero must not deposit
// momentum either, or it would leave momentum without mass and an unbounded
// node velocity. With the range-based factors that needs a contribution faster
// than 2 * MAX_FLOW_VELOCITY, so the guard only protects against misuse.
fn quantize_mass(value: f32) -> u32 {
    return u32(round(value * mass_factor()));
}

fn transfer_p2g(p_idx: u32) {
    transfer_p2g_with_affine(p_idx, mat2x2f(vec2f(0.0), vec2f(0.0)));
}

fn transfer_p2g_with_affine(p_idx: u32, affine_matrix: mat2x2f) {
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
            let affine_velocity = p_velocity - affine_matrix * distance;

            
            
            // atomic_float atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            // resting particles carry no momentum: skip their zero adds
            // atomic_float let momentum = p_mass * weight * affine_velocity;
            // atomic_float if momentum.x != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u], momentum.x); }
            // atomic_float if momentum.y != 0.0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], momentum.y); }
            // for affine transfer
            // atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);

            let quantized_mass = quantize_mass(p_mass * weight); // no_atomic_float
            if quantized_mass == 0u { continue; } // no_atomic_float
            atomicAdd(&grid_mass_atomic[idx], quantized_mass); // no_atomic_float
            let momentum_x = p_mass * affine_velocity.x * weight * momentum_factor(); // no_atomic_float
            let momentum_y = p_mass * affine_velocity.y * weight * momentum_factor(); // no_atomic_float
            // resting particles carry no momentum; skipping zero adds saves most of their atomics
            let quantized_x = quantize_momentum_component(momentum_x); // no_atomic_float
            let quantized_y = quantize_momentum_component(momentum_y); // no_atomic_float
            if quantized_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], quantized_x); } // no_atomic_float
            if quantized_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantized_y); } // no_atomic_float
        }
    }
}