// Cartesian surface geometry for the terrain-following models.
// Pure library module: velocities are 3D Cartesian vectors (vx, vy, vz)
// tangent to the bed z = B(x, y) with unit normal n = (-grad(B), 1) / J.
// Hessian components are passed as (B_xx, B_xy, B_yy), the layout written by
// analyze_terrain.wgsl into the curvature texture.

// Removes the bed-normal component.
fn tangent_project(v: vec3f, n: vec3f) -> vec3f {
    return v - dot(v, n) * n;
}

// Rotates v into the tangent plane while keeping its speed, so a particle
// moving onto a differently inclined bed does not lose kinetic energy.
fn tangent_project_preserve_speed(v: vec3f, n: vec3f) -> vec3f {
    let tangential = tangent_project(v, n);
    let tangential_speed = length(tangential);
    return select(tangential, tangential * (length(v) / tangential_speed), tangential_speed > 1e-6);
}

// Surface gradient of a scalar field given its horizontal gradient:
// P (grad_xy, 0) with P = I - n n^T. The xy part equals
// surface_inverse_metric(slope, grad_xy) of the curvilinear model.
fn surface_gradient(n: vec3f, grad_xy: vec2f) -> vec3f {
    return tangent_project(vec3f(grad_xy, 0.0), n);
}

// Bed curvature along a horizontal direction d: d^T H d.
fn hessian_quadratic_form(hessian: vec3f, d: vec2f) -> f32 {
    return hessian.x * d.x * d.x + 2.0 * hessian.y * d.x * d.y + hessian.z * d.y * d.y;
}

// Centripetal bed-normal acceleration kappa_n |v|^2 of a tangent velocity v.
// Equals the curvilinear K:qq with K = H / J and q = v.xy.
fn normal_curvature_acceleration(hessian: vec3f, v: vec3f, n: vec3f) -> f32 {
    return n.z * hessian_quadratic_form(hessian, v.xy);
}

// Velocity used for the trapezoidal position update.
fn arc_step_velocity(v_prev: vec3f, v_new: vec3f) -> vec3f {
    return 0.5 * (v_prev + v_new);
}

// Curvature-corrected trapezoidal step along the bed: arc length
// s = |v_step| dt along the tangent, bent by the normal curvature kappa.
// kappa > 0 on a concave bed (positive Hessian form), where the path bends
// towards the bed normal.
// The caller snaps the elevation back onto the DEM afterwards. Next to
// missing DEM data the curvature is undefined and the step stays straight,
// so the particle still reaches the data gap and is flagged there.
fn arc_step(position: vec3f, v_step: vec3f, n: vec3f, bed_kappa: f32, dt: f32) -> vec3f {
    let speed = length(v_step);
    let s = speed * dt;
    if s < 1e-9 {
        return position;
    }
    let kappa = select(0.0, bed_kappa, is_finite(bed_kappa));
    let tangent_scale = s - (kappa * kappa * s * s * s) / 6.0;
    let normal_scale = (kappa * s * s) / 2.0;
    return position + tangent_scale * (v_step / speed) + normal_scale * n;
}
