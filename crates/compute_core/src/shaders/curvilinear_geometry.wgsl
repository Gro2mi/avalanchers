// Velocities are horizontal contravariant components q = (dx/dt, dy/dt).
// For z = B(x,y), G = I + grad(B) grad(B)^T.
fn surface_inverse_metric(slope: vec2f, covector: vec2f) -> vec2f {
    return covector - slope * dot(slope, covector) / (1.0 + dot(slope, slope));
}

fn surface_velocity(slope: vec2f, velocity: vec2f) -> vec3f {
    return vec3f(velocity, dot(slope, velocity));
}

fn surface_speed(slope: vec2f, velocity: vec2f) -> f32 {
    return length(surface_velocity(slope, velocity));
}

fn terrain_slope_at_uv(uv: vec2f) -> vec2f {
    let offset = 1.0 / vec2f(sim_settings.grid_shape);
    return vec2f(
        textureSampleLevel(dem_texture, tex_sampler, uv + vec2f(offset.x, 0.0), 0).x
            - textureSampleLevel(dem_texture, tex_sampler, uv - vec2f(offset.x, 0.0), 0).x,
        textureSampleLevel(dem_texture, tex_sampler, uv + vec2f(0.0, offset.y), 0).x
            - textureSampleLevel(dem_texture, tex_sampler, uv - vec2f(0.0, offset.y), 0).x
    ) / (2.0 * sim_settings.cell_size);
}
