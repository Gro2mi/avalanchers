@group(0) @binding(1) var<storage, read_write> grid_h_atomic: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grid_forces: array<vec2f>;

@compute @workgroup_size(16, 16, 1)
fn compute_grid_physics(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.y * grid_width + id.x;
    
    // 1. Decode height and velocity[cite: 3]
    let h = f32(atomicLoad(&grid_h_atomic[idx])) / SCALE_FACTOR;
    let u = f32(atomicLoad(&grid_mom_atomic[idx * 2])) / (h * SCALE_FACTOR + EPSILON);
    let v = f32(atomicLoad(&grid_mom_atomic[idx * 2 + 1])) / (h * SCALE_FACTOR + EPSILON);

    // 2. Compute Divergence for Active/Passive state[cite: 3]
    let div_u = (get_u(id.x + 1, id.y) - get_u(id.x - 1, id.y)) / (2.0 * dx);
    let k = calculate_k(div_u); // Returns k_act, k_pass, or 1.0 based on div_u[cite: 3]

    // 3. Lateral Pressure Force[cite: 3]
    // Force = -0.5 * g * cos(theta) * k * gradient(h^2)
    let grad_h2 = vec2f(
        (pow(get_h(id.x + 1, id.y), 2.0) - pow(get_h(id.x - 1, id.y), 2.0)) / (2.0 * dx),
        (pow(get_h(id.x, id.y + 1), 2.0) - pow(get_h(id.x, id.y - 1), 2.0)) / (2.0 * dy)
    );
    
    grid_forces[idx] = -0.5 * gravity * cos_theta * k * grad_h2;
}