struct TimestepDataArray {
    trajectories: array<TimestepData, 3>,
};

struct TimestepData {
    velocity: vec3f,
    dt: f32,
    acceleration_tangential: vec3f,
    acceleration_friction_magnitude: f32,
    position: vec3f,
    elevation: f32,
    normal: vec3f,
    g_eff: f32,
    acceleration_normal: vec3f,
    _pad1: f32,
    uv: vec2f,
    _pad2: vec2f,
};

@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
// xyz = unit bed normal, w = profile curvature (analyze_terrain.wgsl)
@group(0) @binding(2) var terrain_geometry_texture: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var<storage, read_write> particles_position: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> particles_velocity: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> particles_state: array<u32>;
@group(0) @binding(7) var<storage, read> grid_velocity: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(9) var<storage, read_write> out_timestep_data: array<TimestepDataArray>; // trajectory data, fixed size 3
@group(0) @binding(10) var<storage, read_write> out_debug: array<f32>;
@group(0) @binding(11) var<storage, read_write> particles_affine_matrix: array<mat2x2<f32>>;
@group(0) @binding(12) var dem_texture: texture_2d<f32>;
@group(0) @binding(13) var<storage, read_write> particles_elevation: array<f32>;
// xyz = Hessian (B_xx, B_xy, B_yy), w = plan curvature (analyze_terrain.wgsl)
@group(0) @binding(14) var curvature_texture: texture_2d<f32>;
// z components of the 3D model; the terrain-following layout ends at binding 14
@group(0) @binding(15) var<storage, read_write> particles_velocity_z: array<f32>;
@group(0) @binding(16) var<storage, read> grid_velocity_z: array<f32>;
@group(0) @binding(17) var<storage, read_write> particles_affine_z: array<vec2f>;

override WG_SIZE_1D: u32 = 1u;

// Velocities live in the surface plane (see surface_to_tangent). A particle that
// reaches a differently inclined bed loses the velocity component normal to it,
// as the grid projection of the 3D model removes it: kept at full speed, a flow
// would run up steep side walls instead of being deflected by them.
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn g2p_terrain_following(@builtin(global_invocation_id) pId: vec3<u32>) {
    let particleId = pId.x;
    if particle_is_inactive(particleId) {
        return;
    }
    let position_xy = particles_position[particleId];
    let uv = position_to_uv(position_xy);
    let terrain = sample_terrain(uv);
    if !on_valid_terrain(terrain) || is_nan(particles_elevation[particleId]) {
        stop_outside_dem(particleId);
        return;
    }
    let normal = normalize(terrain.xyz);

    let update = transfer_g2p(particleId);
    if stop_if_too_fast(particleId, length(surface_to_tangent(update.velocity, normal))) {
        return;
    }
    if update_rest_state(particleId, length(update.velocity)) {
        particles_velocity[particleId] = vec2f(0.0);
        particles_affine_matrix[particleId] = mat2x2<f32>(vec2f(0.0), vec2f(0.0));
        return;
    }

    let velocity = surface_to_tangent(update.velocity, normal);
    let v_prev = surface_to_tangent(particles_velocity[particleId], normal);
    let position = advance_on_bed(particleId, position_xy, uv, normal, v_prev, velocity);
    let uv_new = position_to_uv(position.xy);
    let terrain_new = sample_terrain(uv_new);
    let normal_new = normalize(terrain_new.xyz);
    // outside the DEM data the next step flags the particle
    let valid_new = on_valid_terrain(terrain_new);
    let new_velocity = select(velocity, tangent_project(velocity, normal_new), valid_new);
    let surface_velocity = select(update.velocity, tangent_to_surface(new_velocity, normal_new), valid_new);
    particles_velocity[particleId] = surface_velocity;
    // The gather uses particle-minus-node offsets; C uses node-minus-particle.
    particles_affine_matrix[particleId] = update.affine_matrix * -1.0;

    if particleId == sim_info.number_particles / 2u {
        record_output(position, new_velocity, normal_new, uv);
    }
    stop_if_invalid(particleId, position, surface_velocity.x);
}

// 3D APIC: the velocity is carried as a vector tangent to the bed and rotated
// into the tangent plane at the new position.
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn g2p_3d(@builtin(global_invocation_id) pId: vec3<u32>) {
    let particleId = pId.x;
    if particle_is_inactive(particleId) {
        return;
    }
    let position_xy = particles_position[particleId];
    let uv = position_to_uv(position_xy);
    let terrain = sample_terrain(uv);
    if !on_valid_terrain(terrain) || is_nan(particles_elevation[particleId]) {
        stop_outside_dem(particleId);
        clear_z(particleId);
        return;
    }
    let normal = normalize(terrain.xyz);

    // a detached particle moves under gravity alone, outside the grid
    if (particles_state[particleId] & PARTICLE_FLYING) != 0u {
        let start = vec3f(position_xy, particles_elevation[particleId]);
        let velocity = vec3f(particles_velocity[particleId], particles_velocity_z[particleId]);
        fly_step(particleId, flight_motion(start, velocity), uv);
        return;
    }

    // the shared xy gather is inlined here because the z component uses the
    // same stencil and weights
    let grid_pos = position_xy / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    var interpolated_velocity_xy = vec2f(0.0);
    var interpolated_velocity_z = 0.0;
    var affine_matrix = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    var affine_z = vec2f(0.0);
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let grid_vel = grid_velocity[idx];
            let grid_vel_z = grid_velocity_z[idx];
            interpolated_velocity_xy += weight * grid_vel;
            interpolated_velocity_z += weight * grid_vel_z;
            affine_matrix += 4.0 * weight * mat2x2<f32>(
                grid_vel * distance.x,
                grid_vel * distance.y
            );
            affine_z += 4.0 * weight * grid_vel_z * distance;
        }
    }
    let interpolated_velocity = vec3f(interpolated_velocity_xy, interpolated_velocity_z);
    if stop_if_too_fast(particleId, length(interpolated_velocity)) {
        clear_z(particleId);
        return;
    }
    if update_rest_state(particleId, length(interpolated_velocity)) {
        particles_velocity[particleId] = vec2f(0.0);
        particles_affine_matrix[particleId] = mat2x2<f32>(vec2f(0.0), vec2f(0.0));
        clear_z(particleId);
        return;
    }

    // The bed falls away faster than the particle could fall: one free-flight
    // step from the bed ends above the DEM, so it detaches. On a plane the
    // flight ends g dt^2 / 2 below the bed, and on concave terrain lower.
    let tangent_velocity = tangent_project(interpolated_velocity, normal);
    if detachment_enabled() {
        let start = vec3f(position_xy, particles_elevation[particleId]);
        let flight = flight_motion(start, tangent_velocity);
        // NaN off the DEM data: the particle stays on the bed
        if flight.position.z > get_elevation(position_to_uv(flight.position.xy)) {
            particles_state[particleId] |= PARTICLE_FLYING;
            fly_step(particleId, flight, uv);
            return;
        }
    }

    let v_prev = vec3f(particles_velocity[particleId], particles_velocity_z[particleId]);
    let position = advance_on_bed(particleId, position_xy, uv, normal, v_prev, interpolated_velocity);
    let uv_new = position_to_uv(position.xy);
    let terrain_new = sample_terrain(uv_new);
    let normal_new = normalize(terrain_new.xyz);
    // outside the DEM data the next step flags the particle
    let new_velocity = select(interpolated_velocity, tangent_project_preserve_speed(interpolated_velocity, normal_new), on_valid_terrain(terrain_new));
    particles_velocity[particleId] = new_velocity.xy;
    particles_velocity_z[particleId] = new_velocity.z;

    // The gather uses particle-minus-node offsets; C uses node-minus-particle.
    particles_affine_matrix[particleId] = affine_matrix * -1.0;
    particles_affine_z[particleId] = affine_z * -1.0;

    if particleId == sim_info.number_particles / 2u {
        record_output(position, new_velocity, normal_new, uv);
    }
    if stop_if_invalid(particleId, position, new_velocity.x) {
        clear_z(particleId);
    }
}

// SimFlags::DETACHMENT: particles may lift off the bed (enable_detachment)
fn detachment_enabled() -> bool {
    return (sim_settings.flags & (1u << 7u)) != 0u;
}

struct FlightMotion {
    position: vec3f,
    velocity: vec3f,
}

// One step of free flight under gravity and quadratic air drag
// a = -g |u| u / v_t^2 (terminal velocity v_t, none for v_t = 0). Gravity is
// explicit, drag implicit (stable at any dt); the trapezoidal position is the
// exact parabola without drag.
fn flight_motion(position: vec3f, velocity: vec3f) -> FlightMotion {
    let dt = sim_info.dt;
    var new_velocity = velocity + vec3f(0.0, 0.0, -g * dt);
    let terminal_velocity = sim_settings.flight_terminal_velocity;
    if terminal_velocity > 0.0 {
        new_velocity /= 1.0 + dt * g * length(new_velocity) / (terminal_velocity * terminal_velocity);
    }
    return FlightMotion(position + 0.5 * (velocity + new_velocity) * dt, new_velocity);
}

// Applies a flight step of a detached particle. It lands once it falls below
// the bed, keeping only the tangential part of its velocity (inelastic
// impact). It carries no APIC gradient and deposits nothing on the grid
// (p2g_3d), so its speed enters the CFL condition here.
fn fly_step(particleId: u32, flight: FlightMotion, uv: vec2f) {
    var position = flight.position;
    var new_velocity = flight.velocity;
    particles_affine_matrix[particleId] = mat2x2<f32>(vec2f(0.0), vec2f(0.0));
    particles_affine_z[particleId] = vec2f(0.0);

    let uv_new = position_to_uv(position.xy);
    let bed = get_elevation(uv_new);
    if stop_if_too_fast(particleId, length(new_velocity)) {
        particles_position[particleId] = position.xy;
        drop_to_bed(particleId, bed);
        clear_z(particleId);
        return;
    }
    let terrain_new = sample_terrain(uv_new);
    var normal_new = vec3f(0.0, 0.0, 1.0);
    // over missing DEM data it stays in flight, the next step flags it
    if position.z <= bed && on_valid_terrain(terrain_new) {
        normal_new = normalize(terrain_new.xyz);
        new_velocity = landing_velocity(new_velocity, normal_new);
        position.z = bed;
        particles_state[particleId] = particles_state[particleId] & ~PARTICLE_FLYING;
    }
    particles_position[particleId] = position.xy;
    particles_elevation[particleId] = position.z;
    particles_velocity[particleId] = new_velocity.xy;
    particles_velocity_z[particleId] = new_velocity.z;

    let speed = length(new_velocity);
    if is_finite(speed) && bitcast<u32>(speed) > atomicLoad(&atomic_values.expected_max_velocity) {
        atomicMax(&atomic_values.expected_max_velocity, bitcast<u32>(speed));
    }

    if particleId == sim_info.number_particles / 2u {
        record_output(position, new_velocity, normal_new, uv);
    }
    if stop_if_invalid(particleId, position, new_velocity.x) {
        drop_to_bed(particleId, bed);
        clear_z(particleId);
    }
}

// A flying particle that is stopped is put on the bed below it rather than
// left hanging in the air (not possible over missing DEM data).
fn drop_to_bed(particleId: u32, bed: f32) {
    if is_finite(bed) {
        particles_elevation[particleId] = bed;
    }
}

// Inelastic landing with Coulomb friction over the impact: the bed removes the
// normal velocity v_n with the impulse m |v_n|, and the friction it carries
// removes mu |v_n| of the tangential speed (without reversing it). This is
// the limit of a resolved concave bend, where the basal friction acts on the
// centrifugal load; velocity-dependent drag contributes nothing to an
// instantaneous impact.
fn landing_velocity(velocity: vec3f, normal: vec3f) -> vec3f {
    let impact_speed = max(-dot(velocity, normal), 0.0);
    let tangential = tangent_project(velocity, normal);
    let tangential_speed = length(tangential);
    if tangential_speed < 1e-6 {
        return vec3f(0.0);
    }
    let remaining = max(tangential_speed - landing_friction_coefficient() * impact_speed, 0.0);
    return tangential * (remaining / tangential_speed);
}

// Coulomb coefficient of the basal friction model: mu0 for mu(I), the
// friction coefficient for the others
fn landing_friction_coefficient() -> f32 {
    return select(sim_settings.friction_coefficient, sim_settings.mu0, sim_settings.friction_model == 5u);
}

fn clear_z(particleId: u32) {
    particles_velocity_z[particleId] = 0.0;
    particles_affine_z[particleId] = vec2f(0.0);
}

fn particle_is_inactive(particleId: u32) -> bool {
    return particleId >= sim_info.number_particles
        || (sim_info.flags & SIM_INFO_STOPPED) != 0u
        || (particles_state[particleId] & PARTICLE_STOPPED) != 0u;
}

// Stopped particles keep depositing mass, but no stale momentum.
fn stop_particle(particleId: u32, state_flags: u32) {
    particles_state[particleId] = sim_info.timestep | PARTICLE_STOPPED | state_flags;
    particles_velocity[particleId] = vec2f(0.0);
    particles_affine_matrix[particleId] = mat2x2<f32>(vec2f(0.0), vec2f(0.0));
    atomicAdd(&atomic_values.stopped_particles, 1u);
}

// Stops a particle above MAX_PARTICLE_SPEED and flags the run; returns
// whether it did.
fn stop_if_too_fast(particleId: u32, speed: f32) -> bool {
    if speed <= MAX_PARTICLE_SPEED {
        return false;
    }
    stop_particle(particleId, PARTICLE_UNPHYSICAL_SPEED);
    sim_info.flags |= SIM_INFO_UNPHYSICAL_SPEED;
    return true;
}

fn stop_outside_dem(particleId: u32) {
    stop_particle(particleId, PARTICLE_OUT_OF_DEM_DATA);
    sim_info.flags |= SIM_INFO_PARTICLE_OUT_OF_DEM_DATA;
}

// Returns whether the particle rests after this step.
fn update_rest_state(particleId: u32, speed: f32) -> bool {
    let state = next_rest_state(particles_state[particleId], speed, sim_info.timestep);
    particles_state[particleId] = state;
    let resting = (state & PARTICLE_RESTING) != 0u;
    if resting {
        atomicAdd(&atomic_values.resting_particles, 1u);
    }
    return resting;
}

// Curvature-corrected arc step, snapped onto the DEM.
fn advance_on_bed(particleId: u32, position_xy: vec2f, uv: vec2f, normal: vec3f, v_prev: vec3f, velocity: vec3f) -> vec3f {
    let dt = sim_info.dt;
    let v_step = arc_step_velocity(v_prev, velocity);
    let kappa = normal.z * get_bed_curvature(uv, v_step);
    var position = arc_step(vec3f(position_xy, particles_elevation[particleId]), v_step, normal, kappa, dt);
    let elevation = get_elevation(position_to_uv(position.xy));
    position.z = elevation;
    particles_position[particleId] = position.xy;
    particles_elevation[particleId] = elevation;
    return position;
}

fn record_output(position: vec3f, velocity: vec3f, normal: vec3f, uv: vec2f) {
    var current: TimestepData;
    current.position = position;
    current.velocity = velocity;
    current.uv = uv;
    current.dt = sim_info.dt;
    current.acceleration_tangential = vec3f(0.0);
    current.acceleration_friction_magnitude = 0.0;
    current.elevation = position.z;
    current.normal = normal;
    current.g_eff = 0.0;
    current.acceleration_normal = vec3f(0.0);
    current._pad1 = 0.0;
    current._pad2 = vec2f(0.0);
    update_output_data(0u, sim_info.timestep - 1, current);

    out_debug[0] = f32(position.x);
    out_debug[1] = f32(position.y);
    out_debug[2] = f32(uv.x);
    out_debug[3] = f32(uv.y);
    out_debug[5] = f32(sim_info.timestep);
    out_debug[6] = f32(sim_info.number_particles);
    out_debug[7] = f32(sim_settings.released_particles_per_cell);
    out_debug[8] = f32(sim_settings.grid_shape.x);
    out_debug[9] = f32(sim_settings.grid_shape.y);
    out_debug[10] = f32(sim_settings.world_size.x);
    out_debug[11] = f32(sim_settings.world_size.y);
    out_debug[12] = f32(sim_settings.friction_coefficient);
}

// Stops particles that became invalid or left the grid; returns whether it did.
fn stop_if_invalid(particleId: u32, position: vec3f, velocity_x: f32) -> bool {
    if is_nan(position.x) {
        stop_particle(particleId, PARTICLE_IS_NAN | PARTICLE_OUT_OF_DEM_DATA);
        sim_info.flags |= SIM_INFO_IS_NAN;
        sim_info.flags |= SIM_INFO_PARTICLE_OUT_OF_DEM_DATA;
        return true;
    }
    if is_nan(velocity_x) {
        stop_particle(particleId, PARTICLE_IS_NAN);
        sim_info.flags |= SIM_INFO_IS_NAN;
        return true;
    }

    // we leave a two cell boundary
    if position.x < 2.0 * sim_settings.cell_size
        || position.x > sim_settings.world_size.x - 2.0 * sim_settings.cell_size
        || position.y < 2.0 * sim_settings.cell_size
        || position.y > sim_settings.world_size.y - 2.0 * sim_settings.cell_size {
        stop_particle(particleId, PARTICLE_OUT_OF_BOUNDS);
        sim_info.flags |= SIM_INFO_OUT_OF_BOUNDS;
        return true;
    }
    return false;
}

// bed curvature d^T H d along the horizontal part of the unit velocity
fn get_bed_curvature(uv: vec2f, velocity: vec3f) -> f32 {
    let speed = length(velocity);
    if speed < 1e-6 {
        return 0.0;
    }
    let hessian = textureSampleLevel(curvature_texture, tex_sampler, uv, 0.0).xyz;
    // a no-data texel with zero bilinear weight still makes the sample NaN
    if is_nan(hessian.x) || is_nan(hessian.y) || is_nan(hessian.z) {
        return 0.0;
    }
    return hessian_quadratic_form(hessian, velocity.xy / speed);
}

fn get_elevation(uv: vec2f) -> f32 {
    return textureSampleLevel(dem_texture, tex_sampler, uv, 0).x;
}

// The filtered terrain geometry (normal, validity in w) at uv.
fn sample_terrain(uv: vec2f) -> vec4f {
    return textureSampleLevel(terrain_geometry_texture, tex_sampler, uv, 0);
}

// Whether a sample of sample_terrain read only valid terrain. Filtering a
// footprint that holds a NaN texel is undefined: the sample can be NaN or
// finite garbage, e.g. a normal with n.z <= 0, where surface_to_tangent
// divides by 1 + n.z ~ 0 and throws the particle across the grid. The normal
// and curvature are NaN one cell into the data from a no-data cell, so
// analyze_terrain marks those texels with w = 1 and the valid ones with
// w = 0; filtered zeros stay exactly zero, so any invalid texel in the
// footprint makes w > 0. A texel whose bilinear weight rounds to zero adds
// nothing to w but still turns the filtered normal into NaN (0 * NaN), so
// the sampled n.z is tested too; on the sampled value, because the compiler
// may assume arithmetic results are never NaN.
fn on_valid_terrain(terrain: vec4f) -> bool {
    return terrain.w == 0.0 && !is_nan(terrain.z) && terrain.z > 0.0;
}

fn update_output_data(trajectory: u32, timestep: u32, timestep_data: TimestepData) {
    out_timestep_data[timestep].trajectories[trajectory] = timestep_data;
}

// import transfer_g2p_affine.wgsl;
// BEGIN transfer_g2p_affine.wgsl
struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

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

// END transfer_g2p_affine.wgsl
// import cartesian_geometry.wgsl;
// BEGIN cartesian_geometry.wgsl
// Cartesian surface geometry for the terrain-following models.
// Pure library module: velocities are 3D Cartesian vectors (vx, vy, vz)
// tangent to the bed z = B(x, y) with unit normal n = (-grad(B), 1) / J.
// Hessian components are passed as (B_xx, B_xy, B_yy), the layout written by
// analyze_terrain.wgsl into the curvature texture.

// The terrain-following model stores velocities in the surface plane: the 3D
// tangent velocity rotated into the horizontal plane by the rotation that takes
// the bed normal to z. Unlike the horizontal projection this keeps the magnitude,
// so length(v2) is the true speed, and the map is linear on any plane, so grid
// averages and APIC stay exact there. The components are not dx/dt.
fn surface_to_tangent(v2: vec2f, n: vec3f) -> vec3f {
    let along_slope = dot(n.xy, v2);
    return vec3f(v2 - along_slope / (1.0 + n.z) * n.xy, -along_slope);
}

fn tangent_to_surface(v3: vec3f, n: vec3f) -> vec2f {
    return v3.xy - v3.z / (1.0 + n.z) * n.xy;
}

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

// END cartesian_geometry.wgsl
// import constants.wgsl;
// BEGIN constants.wgsl
// Shared constants without dependencies.
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// END constants.wgsl
// import sim_settings.wgsl;
// BEGIN sim_settings.wgsl
// Simulation settings uniform (group 0, binding 0) and the values derived from
// it. Must mirror the Rust POD layout of SimSettings.

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

struct SimSettings {
    num_steps: u32,
    model_type: u32,
    friction_model: u32,
    released_particles_per_cell: u32,
    grid_shape: vec2u,
    world_size: vec2f,
    snow_density: f32,
    release_thickness: f32,
    friction_coefficient: f32,
    drag_coefficient: f32,
    n0: f32,
    i0: f32,
    mu0: f32,
    mu2: f32,
    grain_diameter: f32,
    internal_friction_angle: f32,
    basal_friction_angle: f32,
    cfl: f32,
    cell_size: f32,
    min_slope_angle: f32,
    max_slope_angle: f32,
    min_elevation: f32,
    velocity_threshold: f32,
    roughness_threshold: f32,
    flags: u32,
    release_max_elevation: f32,
    peak_flow_thickness_threshold: f32,
    // MPMDAC constitutive model (must mirror the Rust POD layout)
    constitutive_model: u32,
    shear_modulus: f32,
    hardening_modulus: f32,
    // MPMDAC compressibility; bulk_modulus 0 = incompressible
    bulk_modulus: f32,
    compaction_pressure: f32,
    // stop once total momentum < fraction * its peak; 0 disables (curvilinear)
    momentum_stop_fraction: f32,
    // front travel in cells covered by the no-new-cells stop window
    no_new_cells_travel: f32,
    // Gaussian width (m) of the curvilinear reference surface; 0 disables it
    reference_surface_length: f32,
    // snow water equivalent (kg/m2) of the release cells; NaN if unset
    release_swe: f32,
    // 3D lift-off: terminal velocity (m/s) of the quadratic air drag in
    // flight; 0 disables the drag
    flight_terminal_velocity: f32,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// Gully walls (reference surface, wall friction) need a reference surface at
// least one cell wide; must match SimSettings::gully_walls_enabled.
fn gully_walls_enabled() -> bool {
    return sim_settings.reference_surface_length >= sim_settings.cell_size;
}

// length in steps of the no-new-cells rolling window: the fastest material
// moves at most cfl cells per step, so round(travel / cfl) steps cover the same
// front travel at any cfl; at least 10 steps
fn no_new_cells_window() -> u32 {
    return max(10u, u32(round(sim_settings.no_new_cells_travel / sim_settings.cfl)));
}

// END sim_settings.wgsl
// import sim_info.wgsl;
// BEGIN sim_info.wgsl
// Simulation state shared between the passes: the SimInfo buffer with its
// status flags and the atomic accumulators. Must mirror the Rust layouts.

struct SimInfo {
    timestep: u32,
    dt: f32,
    elapsed_time: f32,
    number_particles: u32,
    elevation_threshold: f32,
    max_velocity: f32,
    max_flow_thickness: f32,
    flags: u32,
    grid_elapsed_time: f32,
    travel_length_2d: f32,
    travel_length_3d: f32,
    travel_vertical_drop: f32,
    travel_angle: f32,
    _first_center_of_mass_timestep: u32,
    _last_center_of_mass_timestep: u32,
    // ranges of the integer-atomic grid in units of the reference node (see
    // quantization.wgsl); previous_mass_scale quantized the last step's grid
    mass_scale: f32,
    momentum_scale: f32,
    previous_mass_scale: f32,
    _pad0: u32,
    _pad1: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
// a particle exceeded MAX_PARTICLE_SPEED and was stopped
const SIM_INFO_UNPHYSICAL_SPEED: u32 = 1u << 4u;
// a node filled more than QUANTIZATION_OVERFLOW_FILL of an integer-atomic
// grid range, which it may have overflowed (see quantization.wgsl)
const SIM_INFO_QUANTIZATION_OVERFLOW: u32 = 1u << 5u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

struct AtomicValues {
    peak_velocity: atomic<u32>,
    peak_flow_thickness: atomic<u32>,
    expected_max_velocity: atomic<u32>,
    travel_length: atomic<u32>,
    release_volume: atomic<u32>,
    number_release_cells: atomic<u32>,
    number_release_particles: atomic<u32>,
    stopped_particles: atomic<u32>,
    // resting particles counted during the current step, reset by update_sim_info
    resting_particles: atomic<u32>,
    // total grid momentum (kg m/s) of the current step as a 64-bit sum, reset by update_sim_info
    momentum_lo: atomic<u32>,
    momentum_hi: atomic<u32>,
    // f32 bits of the largest total momentum so far
    peak_momentum: atomic<u32>,
    // largest node mass (u32 units) and momentum component (|i32| units) of
    // the integer-atomic grid this step, read and reset by reset_grid
    max_mass_units: atomic<u32>,
    max_momentum_units: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

// END sim_info.wgsl
// import particle_state.wgsl;
// BEGIN particle_state.wgsl
// Particle state flags and the reversible resting state. Needs sim_settings.

// Fastest particle speed (m/s) taken as physical. Dense flow avalanches reach
// about 60-80 m/s and the integer atomic grid holds momentum up to
// MAX_FLOW_VELOCITY = 100 m/s for a full MAX_FLOW_THICKNESS node (faster for
// thinner nodes); a faster particle comes from a numerical blow-up (e.g. a cliff of no-data elevations) and is stopped.
// Mirrors compute_core::MAX_PARTICLE_SPEED.
const MAX_PARTICLE_SPEED: f32 = 150.0;
// stopped for exceeding MAX_PARTICLE_SPEED
const PARTICLE_UNPHYSICAL_SPEED: u32 = 1u << 25u;
// reversible low-speed state; the low bits (below PARTICLE_UNPHYSICAL_SPEED)
// hold the rest start step while resting and the consecutive slow-step count
// while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
const PARTICLE_FLYING: u32 = 1u << 27u;
const PARTICLE_OUT_OF_BOUNDS: u32 = 1u << 28u;
const PARTICLE_IS_NAN: u32 = 1u << 29u;
const PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 30u;
const PARTICLE_STOPPED: u32 = 1u << 31u;

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_UNPHYSICAL_SPEED - 1u;
    var resting = (previous_state & PARTICLE_RESTING) != 0u;
    var low_bits = previous_state & low_mask;
    if resting {
        if speed > 2.0 * sim_settings.velocity_threshold {
            resting = false;
            low_bits = 0u;
        }
    } else if speed < sim_settings.velocity_threshold {
        low_bits = min(low_bits + 1u, REST_STEPS);
        if low_bits >= REST_STEPS {
            resting = true;
            low_bits = timestep & low_mask;
        }
    } else {
        low_bits = 0u;
    }
    return (previous_state & ~(PARTICLE_RESTING | low_mask))
        | select(0u, PARTICLE_RESTING, resting)
        | low_bits;
}

// END particle_state.wgsl
// import grid_index.wgsl;
// BEGIN grid_index.wgsl
// Conversions between world positions, cells, uv coordinates and flat grid
// indices. Needs sim_settings.

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
}

fn position_to_cell(position: vec2f) -> vec2u {
    return vec2u(
        floor(position.xy / sim_settings.cell_size)
    );
}

fn cell_center_xy(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) * sim_settings.cell_size;
}

fn position_to_uv(position: vec2f) -> vec2f {
    return (position.xy) / (vec2f(sim_settings.world_size)); // add some padding to ensure particles outside the world bounds are still captured in the simulation info
}

fn position_to_idx(position: vec2f) -> u32 {
    let uv = position_to_uv(position);
    return uv_to_idx(uv);
}

fn uv_to_cell(uv: vec2f) -> vec2u {
    let epsilon = 1e-5f; // A tiny offset to counteract negative rounding bias
    let scaled_uv = uv * vec2f(sim_settings.grid_shape) + epsilon;
    let max_bound = vec2f(sim_settings.grid_shape - 1u);

    return vec2u(clamp(scaled_uv, vec2f(0.0), max_bound));
}

fn uv_to_idx(uv: vec2f) -> u32 {
    let cell = uv_to_cell(uv);
    // return cell.x * sim_settings.grid_shape.y + cell.y;
    return (cell.y % sim_settings.grid_shape.y * sim_settings.grid_shape.x +
              (cell.x % sim_settings.grid_shape.x));
}

fn x_y_to_idx(x: u32, y: u32) -> u32 {
    return y * sim_settings.grid_shape.x + x;
}

fn xy_to_idx(xy: vec2<u32>) -> u32 {
    return xy.y * sim_settings.grid_shape.x + xy.x;
}

fn idx_to_xy(idx: u32) -> vec2<u32> {
    let x = idx % sim_settings.grid_shape.x;
    let y = idx / sim_settings.grid_shape.x;
    return vec2u(x, y);
}

// END grid_index.wgsl
// import bspline.wgsl;
// BEGIN bspline.wgsl
// Quadratic B-spline weights of the particle <-> grid transfers. Needs
// sim_settings (calculate_weight_gradient).

fn quadratic_weight(d: f32) -> f32 {
    let abs_d = abs(d);
    if abs_d < 0.5 {
        return 0.75 - abs_d * abs_d;
    } else if abs_d < 1.5 {
        return 0.5 * pow(1.5 - abs_d, 2.0);
    }
    return 0.0;
}

fn calculate_weight(distance: vec2f) -> f32 {
    return quadratic_weight(distance.x) * quadratic_weight(distance.y);
}

// derivative of quadratic_weight with respect to its (cell-unit) argument
fn quadratic_weight_gradient(d: f32) -> f32 {
    let abs_d = abs(d);
    if abs_d < 0.5 {
        return -2.0 * d;
    } else if abs_d < 1.5 {
        return -sign(d) * (1.5 - abs_d);
    }
    return 0.0;
}

// physical gradient of the 2D B-spline weight, in 1/m
fn calculate_weight_gradient(distance: vec2f) -> vec2f {
    return vec2f(
        quadratic_weight_gradient(distance.x) * quadratic_weight(distance.y),
        quadratic_weight(distance.x) * quadratic_weight_gradient(distance.y)
    ) / sim_settings.cell_size;
}

fn calculate_distance_to_node(particle_position: vec2f, node_position: vec2u) -> vec2f {
    return particle_position - vec2f(node_position);
}

fn get_base_node(grid_pos: vec2f) -> vec2u {
    return vec2u(floor(grid_pos - vec2f(0.5)));
}

// END bspline.wgsl
// import float_checks.wgsl;
// BEGIN float_checks.wgsl
// NaN / infinity checks on the bit pattern (comparisons may be optimized away).

fn is_nan(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    return (bits & 0x7F800000u) == 0x7F800000u
          && (bits & 0x007FFFFFu) != 0u;
}

fn is_inf(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    return (bits == 0x7F800000u || bits == 0xFF800000u);
}

fn is_finite(x: f32) -> bool {
    return !is_nan(x) && !is_inf(x);
}

// END float_checks.wgsl
