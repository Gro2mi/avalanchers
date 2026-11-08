// Savage-Hutter earth pressure coefficients (Hutter et al. 1993) for the grid
// physics of the depth-averaged models.
// Pure library module: it is textually imported and relies on the importing
// module providing grid_mass_atomic, grid_momentum_atomic and the utils.wgsl
// symbols (sim_settings, x_y_to_idx, INV_MASS_FACTOR, INV_MOMENTUM_FACTOR).

struct EarthPressure {
    // pressure gradient weighted with k_x along the fall line, k_y across it
    k_grad_h: vec2f,
    // cross-slope coefficient, also the lateral stress loading gully walls
    k_y: f32,
}

// Earth pressure at node xy from the horizontal bed gradient `slope`, the
// horizontal node velocity q and the horizontal gradient of the flow thickness.
// Each coefficient is active where the flow stretches in its direction and
// passive where it is compressed. The fall line is an eigenvector of the
// surface metric, so the weighting commutes with projecting onto the bed.
fn earth_pressure(xy: vec2u, slope: vec2f, q: vec2f, grad_h: vec2f) -> EarthPressure {
    let e_x = earth_pressure_axis(slope, q);
    let e_y = vec2f(-e_x.y, e_x.x);
    let grad_q = velocity_gradient(xy.x, xy.y);
    let k = earth_pressure_coefficients(
        radians(sim_settings.internal_friction_angle),
        radians(sim_settings.basal_friction_angle),
        dot(e_x, grad_q * e_x),
        dot(e_y, grad_q * e_y)
    );
    return EarthPressure(k.x * dot(e_x, grad_h) * e_x + k.y * dot(e_y, grad_h) * e_y, k.y);
}

// Horizontal velocity q of node (x, y) from the node's own mass and momentum;
// z = 1 when the node holds mass, 0 below the p2g mass floor where q is
// meaningless.
fn node_velocity(x: u32, y: u32) -> vec3f {
    let idx = x_y_to_idx(x, y);
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR; // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    if mass < 1e-2 {
        return vec3f(0.0);
    }
    let q = vec2f(
        f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR, // no_atomic_float
        f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR // no_atomic_float
        // atomic_float grid_momentum_atomic[idx * 2],
        // atomic_float grid_momentum_atomic[idx * 2 + 1]
    ) / mass;
    return vec3f(q, 1.0);
}

// Derivative of q along one grid axis: central where both neighbours hold
// mass, one-sided against the centre at the flow edge, zero for an isolated
// node. Empty nodes carry no velocity, so a spreading front is not mistaken
// for a compressed one.
fn edge_aware_derivative(low: vec3f, centre: vec2f, high: vec3f) -> vec2f {
    let dx = sim_settings.cell_size;
    if low.z > 0.0 && high.z > 0.0 {
        return (high.xy - low.xy) / (2.0 * dx);
    }
    if high.z > 0.0 {
        return (high.xy - centre) / dx;
    }
    if low.z > 0.0 {
        return (centre - low.xy) / dx;
    }
    return vec2f(0.0);
}

// Horizontal velocity gradient, columns dq/dx and dq/dy: grad_q * e is the
// derivative of q along e, dot(e, grad_q * e) the strain rate along e.
fn velocity_gradient(x: u32, y: u32) -> mat2x2f {
    let centre = node_velocity(x, y).xy;
    return mat2x2f(
        edge_aware_derivative(node_velocity(x - 1u, y), centre, node_velocity(x + 1u, y)),
        edge_aware_derivative(node_velocity(x, y - 1u), centre, node_velocity(x, y + 1u))
    );
}

// Savage-Hutter downslope axis: the fall line, the flow direction on flat
// ground.
fn earth_pressure_axis(slope: vec2f, q: vec2f) -> vec2f {
    if length(slope) > 1e-3 {
        return -normalize(slope);
    }
    if length(q) > 1e-6 {
        return normalize(q);
    }
    return vec2f(1.0, 0.0);
}

// Earth pressure coefficients (k_x, k_y) of Hutter et al. (1993) from the
// internal and bed friction angles and the strain rates along and across the
// downslope axis; >= 0 (stretching) selects the active state.
fn earth_pressure_coefficients(phi: f32, delta: f32, strain_x: f32, strain_y: f32) -> vec2f {
    let cos2_phi = cos(phi) * cos(phi);
    let cos2_delta = cos(delta) * cos(delta);
    let tan2_delta = tan(delta) * tan(delta);
    // delta > phi has no Mohr-Coulomb limit state; clamp to the phi = delta case
    let root = sqrt(max(1.0 - cos2_phi / cos2_delta, 0.0));
    let k_x = select(
        2.0 / cos2_phi * (1.0 + root) - 1.0,
        2.0 / cos2_phi * (1.0 - root) - 1.0,
        strain_x >= 0.0
    );
    let root_y = sqrt((1.0 - k_x) * (1.0 - k_x) + 4.0 * tan2_delta);
    let k_y = select(0.5 * (k_x + 1.0 + root_y), 0.5 * (k_x + 1.0 - root_y), strain_y >= 0.0);
    return vec2f(k_x, k_y);
}
