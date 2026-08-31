struct TimestepDataArray {
    trajectories: array<TimestepData, 3>,
};

struct TimestepData {
    velocity: vec3f,                        // 12 bytes     12
    dt: f32,                          //  4 bytes     16
    acceleration_tangential: vec3f,         // 12 bytes     28
    acceleration_friction_magnitude: f32,   //  4 bytes     32
    position: vec3f,                        // 12 bytes     44
    elevation: f32,                         //  4 bytes     48
    normal: vec3f,                          // 12 bytes     60
    g_eff: f32,                             //  4 bytes     64
    acceleration_normal: vec3f,             // 12 bytes     76
    // padding                                  4 bytes     80
    uv: vec2f,                              //  8 bytes     88
    // padding                                  12 bytes    96
};

// @group(0) @binding(0) var<uniform> sim_settings: sim_settings;
@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var dem_texture: texture_2d<f32>;
@group(0) @binding(3) var normals_texture: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;
@group(0) @binding(5) var<storage, read_write> particles_position: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> particles_elevation: array<f32>;
@group(0) @binding(7) var<storage, read_write> particles_velocity: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> particles_velocity_z: array<f32>;
@group(0) @binding(9) var<storage, read_write> particles_mass: array<f32>;
@group(0) @binding(10) var<storage, read_write> particles_stopped: array<u32>;

@group(0) @binding(11) var<storage, read_write> atomic_values: AtomicValues;

@group(0) @binding(12) var<storage, read_write> grid_peak_velocity_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(13) var<storage, read_write> out_timestep_data: array<TimestepDataArray>; // trajectory data, fixed size 3

@group(0) @binding(14) var curvature_texture: texture_2d<f32>;
@group(0) @binding(15) var<storage, read_write> out_debug: array<f32>;
@group(0) @binding(16) var<storage, read_write> grid_mass_atomic: array<u32>; // no_atomic_float
// atomic_float @group(0) @binding(16) var<storage, read_write> grid_mass_atomic: array<f32>; // atomic_float
@group(0) @binding(17) var<storage, read> grad_h: array<vec2f>;
// @group(0) @binding(11) var<storage, read_write> atomicBuffer: AtomicData;

override WG_SIZE_1D: u32 = 1u;
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn compute_particles(
    @builtin(global_invocation_id) pId: vec3<u32>,
    @builtin(local_invocation_id) lId: vec3<u32>
) {
    let particleId = pId.x;
    let localId = lId.x;

    if particleId >= sim_info.number_particles {
        return;
    }
    if sim_info.flags >= SIM_INFO_STOPPED {
        return;
    }
    var stopped = particles_stopped[particleId];
    if stopped != 0u {
        return;
    }
    let p = particles_position[particleId];
    var position = vec3f(p.x, p.y, particles_elevation[particleId]);
    let uv = position_to_uv(position.xy);

    let normal = get_normal(uv);

    if is_nan(normal.x) {
        particles_stopped[particleId] = 1000000000u + sim_info.timestep;
        atomicAdd(&atomic_values.stopped_particles, 1u);
        sim_info.flags |= SIM_INFO_PARTICLE_OUT_OF_DEM_DATA;
        return;
    }

    let use_curvature: bool = (sim_settings.flags & (1u << 0u)) != 0u;
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    let use_earth_pressure_coefficient: bool = (sim_settings.flags & (1u << 2u)) != 0u;
    let use_entrainment: bool = (sim_settings.flags & (1u << 3u)) != 0u;

    let mass = particles_mass[particleId];
    let velocity_xy = particles_velocity[particleId];
    let velocity_z = particles_velocity_z[particleId];
    var velocity = vec3f(velocity_xy.x, velocity_xy.y, velocity_z);
    // --- project velocity onto tangent plane ---
    velocity = velocity - dot(velocity, normal) * normal;
    let v_prev = velocity;
    let v = velocity;

    // --- compute driving accelerations ---
    const acceleration_gravity = vec3f(0.0, 0.0, -g);
    let acceleration_normal = g * normal.z * normal;
    let acceleration_tangential = acceleration_gravity + acceleration_normal;
    // uKu, K curvature matrix
    var centrifugal_acceleration = 0f;
    if use_curvature {
        centrifugal_acceleration = get_bed_curvature(position, velocity) * dot(velocity, velocity);
    }
    let effective_acceleration_normal = max(0f, centrifugal_acceleration + length(acceleration_normal));

    // pressure acceleration, G2P step, TODO account for slope angle in P2G and G2P
    var interpolated_grad_h = vec2f(0.0);
    var interpolated_h = 0.0;

    let cell_pos = position.xy / sim_settings.cell_size;
    let base_node = get_base_node(cell_pos);

    var accel_lateral = vec3f(0.0, 0.0, 0.0);
    if use_particle_interaction {
        var interpolated_mass = 0.0;
        let safe_normal_z = max(1e-3, normal.z);
        let grid_pos = position.xy / sim_settings.cell_size - vec2f(0.5);
        let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
        let inv_factor = 1 / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size) * safe_normal_z;
        for (var i: u32 = 0; i < 3; i++) {
            for (var j: u32 = 0; j < 3; j++) {
                let node_coords = base_node + vec2u(i, j);
                let distance = calculate_distance_to_node(grid_pos, node_coords);
                let weight = calculate_weight(distance);
                let node_idx = xy_to_idx(node_coords);
                interpolated_grad_h += weight * grad_h[node_idx];
                
                interpolated_mass += weight * f32(grid_mass_atomic[node_idx]) * INV_MASS_FACTOR; // no_atomic_float
                // atomic_float interpolated_h += weight * grid_mass_atomic[node_idx];
            }
        }
        interpolated_h = interpolated_mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size) * safe_normal_z;
        accel_lateral = vec3f(interpolated_grad_h.x, interpolated_grad_h.y, (normal.x * interpolated_grad_h.x + normal.y * interpolated_grad_h.y) / safe_normal_z) * min(g*normal.z, effective_acceleration_normal);
        // accel_lateral = vec3f(interpolated_grad_h.x, interpolated_grad_h.y, (normal.x * interpolated_grad_h.x + normal.y * interpolated_grad_h.y) / safe_normal_z);


        // accel_lateral = force_local_to_world(interpolated_grad_h, normal) * effective_acceleration_normal;
        // accel_lateral = vec3interpolated_grad_h * effective_acceleration_normal;
        // accel_lateral = (accel_lateral - dot(accel_lateral, normal) * normal);
    }
    // var dt = sim_settings.cfl * sim_settings.cell_size / (sim_info.max_velocity + sim_settings.velocity_threshold);
    var dt = sim_info.dt;

    // --- update velocity with driving accelerations ---
    velocity = velocity + (acceleration_tangential + accel_lateral) * dt;

    // --- compute resisting accelerations ---
    var acceleration_normal_friction_magnitude = acceleration_by_normal_friction(effective_acceleration_normal, mass, velocity, interpolated_h);
    let acceleration_drag_friction_magnitude = acceleration_by_drag_friction(effective_acceleration_normal, mass, velocity, interpolated_h);
    var acceleration_friction_magnitude = acceleration_drag_friction_magnitude + acceleration_normal_friction_magnitude;

    // --- update velocity with resisting accelerations ---
    let velocity_length = length(velocity);
    if velocity_length < acceleration_friction_magnitude * dt {
        dt = velocity_length / max(acceleration_friction_magnitude, 1e-6);
        stopped = sim_info.timestep;
    }
    if velocity_length > sim_settings.velocity_threshold {
        velocity -= acceleration_friction_magnitude * (velocity / velocity_length) * dt;
    }

    velocity = velocity * 0.99;
    let speed = length(velocity);
    let s = speed * dt;
    let kappa = normal.z * get_bed_curvature(position, velocity);
    let tangent_scale = s - (kappa * kappa * s * s * s) / 6.0;
    let normal_scale = (kappa * s * s) / 2.0;
    let p_star = position + (tangent_scale * normalize(velocity)) - (normal_scale * normal);

    // --- update position ---
    var relative_trajectory = (velocity + v_prev) * 0.5 * dt;
    var new_position = p_star + relative_trajectory;
    var new_uv = position_to_uv(new_position.xy);
    // new_position = p_star;
    var elevation = get_elevation(new_uv);
    position = new_position;
    // TODO more sophisticated projection methods
    position.z = elevation;

    if particleId == sim_info.number_particles / 2u {
        var current: TimestepData;
        current.position = position;
        current.velocity = velocity;
        current.dt = dt;
        current.acceleration_tangential = acceleration_tangential;
        current.acceleration_friction_magnitude = acceleration_friction_magnitude;
        current.normal = normal;
        current.acceleration_normal = acceleration_normal;
        current.elevation = elevation;
        current.uv = new_uv;
        current.g_eff = length(accel_lateral);
        update_output_data(0u, sim_info.timestep - 1, current);

        // out_debug[2] = f32(p.position.x);
    }

    // --- update output ---
    let v_mag = length(velocity);

    if is_finite(v_mag) && bitcast<u32>(v_mag) > atomicLoad(&atomic_values.peak_velocity) {
        atomicMax(&atomic_values.peak_velocity, bitcast<u32>(v_mag));
    }

    let cell_index = uv_to_idx(new_uv);
    // atomicAdd(&grid_cell_count_buffer[cell_index], 1u);
    // the host reads this buffer as f32, positive floats keep their ordering when bitcast to u32
    if is_finite(v_mag) && bitcast<u32>(v_mag) > atomicLoad(&grid_peak_velocity_buffer[cell_index]) {
            atomicMax(&grid_peak_velocity_buffer[cell_index], bitcast<u32>(v_mag)); 
        }
   
    if particleId == sim_info.number_particles / 2u {
        // atomicMax(&atomicBuffer.counter, step_count);
        out_debug[0] = f32(position.x);
        out_debug[1] = f32(position.y);
        out_debug[2] = f32(new_uv.x);
        out_debug[3] = f32(new_uv.y);
        out_debug[5] = f32(sim_info.timestep);
        out_debug[6] = f32(sim_info.number_particles);
        out_debug[7] = f32(sim_settings.released_particles_per_cell);
        out_debug[8] = f32(sim_settings.grid_shape.x);
        out_debug[9] = f32(sim_settings.grid_shape.y);
        out_debug[10] = f32(sim_settings.world_size.x);
        out_debug[11] = f32(sim_settings.world_size.y);
        out_debug[12] = f32(sim_settings.friction_coefficient);
        out_debug[13] = bitcast<f32>(atomicLoad(&atomic_values.peak_flow_thickness));
        out_debug[14] = f32(uv_to_cell(new_uv).x);
        out_debug[15] = f32(uv_to_cell(new_uv).y);
        out_debug[16] = f32(uv_to_idx(new_uv));
    }

    if is_nan(position.x) {
        particles_stopped[particleId] = 1100000000u + sim_info.timestep;
        atomicAdd(&atomic_values.stopped_particles, 1u);
        sim_info.flags |= SIM_INFO_IS_NAN;
        sim_info.flags |= SIM_INFO_PARTICLE_OUT_OF_DEM_DATA;
        return;
    }
    if is_nan(velocity.x) {
        particles_stopped[particleId] = 1200000000u + sim_info.timestep;
        atomicAdd(&atomic_values.stopped_particles, 1u);
        sim_info.flags |= SIM_INFO_IS_NAN;
        return;
    }

    // stop criterion friction
    if stopped != 0u || length(velocity) < sim_settings.velocity_threshold {
        stopped = sim_info.timestep;
        atomicAdd(&atomic_values.stopped_particles, 1u);
        update_particle(particleId, position, velocity, stopped);
        return;
    }
    // we leave two cells boundary
    if position.x < 2.1 * sim_settings.cell_size 
        || position.x > sim_settings.world_size.x - 2.1 * sim_settings.cell_size
        || position.y < 2.1 * sim_settings.cell_size 
        || position.y > sim_settings.world_size.y - 2.1 * sim_settings.cell_size {//|| elevation < sim_info.elevation_threshold {
        stopped = sim_info.timestep;
        atomicAdd(&atomic_values.stopped_particles, 1u);
        update_particle(particleId, position, velocity, stopped);
        sim_info.flags |= SIM_INFO_OUT_OF_BOUNDS;
        return;
    }
    update_particle(particleId, position, velocity, stopped);
}

fn get_bed_curvature(position: vec3f, velocity: vec3f) -> f32 {
    let uv = position_to_uv(position.xy);
    let K = textureSampleLevel(curvature_texture, tex_sampler, uv, 0.0).rgb;

    let Kxx = K.r;
    let Kyy = K.g;
    let Kxy = K.b;
    let t_hat = normalize(velocity);
    let bed_curvature = (Kxx * t_hat.x * t_hat.x + 2.0 * Kxy * t_hat.x * t_hat.y + Kyy * t_hat.y * t_hat.y);
    return bed_curvature;
}
fn force_local_to_world(
    force_local: vec2<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    // Choose a reference axis that is not parallel to the normal.
    var reference = vec3<f32>(1.0, 0.0, 0.0);

    if abs(normal.x) > 0.9 {
        reference = vec3<f32>(0.0, 0.0, 1.0);
    }

    // First tangent direction in the plane.
    let e1 = normalize(cross(reference, normal));

    // Second tangent direction in the plane.
    let e2 = cross(normal, e1);

    // Transform from local plane coordinates to world coordinates.
    return force_local.x * e1
         + force_local.y * e2
         + 0f * normal;
}

fn update_particle(particleId: u32, position: vec3f, velocity: vec3f, stopped: u32) {
    particles_position[particleId] = position.xy;
    particles_elevation[particleId] = position.z;
    particles_velocity[particleId] = velocity.xy;
    particles_velocity_z[particleId] = velocity.z;
    particles_stopped[particleId] = stopped;
    // mass is constant for now
    // particles_mass[particleId] = mass; 
}

fn update_output_data(trajectory: u32, timestep: u32, timestep_data: TimestepData) {
    out_timestep_data[timestep].trajectories[trajectory] = timestep_data;
}

fn acceleration_by_normal_friction(effective_acceleration_normal: f32, mass: f32, velocity: vec3f, h: f32) -> f32 {
    let mass_per_area = mass / (sim_settings.cell_size * sim_settings.cell_size) * f32(sim_settings.released_particles_per_cell);
    let velocity_magnitude = length(velocity);
    let model = sim_settings.friction_model;
    if velocity_magnitude < sim_settings.velocity_threshold || model >= 6u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = sim_settings.friction_coefficient;
    let normal_stress = effective_acceleration_normal * mass_per_area;
    const min_shear_stress = 70f;
    var shear_stress = 0.0f;
    //actually: friction model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAt, 4 voellmy with cohesion
    // Coulomb friction model
    if model == 0u || model == 1u || model == 2u {
        shear_stress = friction_coefficient * normal_stress;
    }
    // samosAT friction model
    else if model == 3 {
        let rs0 = 0.222;
        let rs = sim_settings.snow_density * velocity_magnitude * velocity_magnitude / (normal_stress + 0.001);
        shear_stress = normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs));
    }
    // check https://ramms.ch/ramms-avalanche/friction-parameters/
    else if model == 4u {
        // let n0 = sim_settings.n0;
        // shear_stress = friction_coefficient * normal_stress + (1 - friction_coefficient) * n0 - (1 - friction_coefficient) * n0 * exp(-normal_stress / n0);
    }
    // mu(I) friction model
    else if model == 5u {
        let grain_diameter = sim_settings.grain_diameter;
        let i0 = sim_settings.i0;
        let mu0 = sim_settings.mu0;
        let mu2 = sim_settings.mu2;
        let inertial_number = 2.5 * sqrt(velocity_magnitude) / h * grain_diameter / sqrt(max(effective_acceleration_normal, 1e-6) * h);
        let muI = mu0 + (mu2 - mu0) / (i0 / inertial_number + 1.0);
        shear_stress = muI * normal_stress;
    }
    let acceleration_magnitude = shear_stress / max(mass_per_area, 1e-6);
    return acceleration_magnitude;
}

fn acceleration_by_drag_friction(effective_acceleration_normal: f32, mass: f32, velocity: vec3f, h: f32) -> f32 {
    let model = sim_settings.friction_model;
    if model == 0u || model >= 4u {
        return 0.0f;
    }
    let velocity_magnitude2 = dot(velocity, velocity);
    if velocity_magnitude2 < 1e-8 {
        return 0.0f;
    }
    let mass_per_area = mass / (sim_settings.cell_size * sim_settings.cell_size) * f32(sim_settings.released_particles_per_cell);
    var shear_stress = 0.0f;
    let density_velocity_magnitude2 = sim_settings.snow_density * velocity_magnitude2;
    // friction model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAt
    // Voellmy friction model
    if model == 1u {
        shear_stress = density_velocity_magnitude2 * g / sim_settings.drag_coefficient;
    }
    // Voellmy min shear friction model
    else if model == 2u {
        const min_shear_stress = 70f;
        shear_stress = min_shear_stress + density_velocity_magnitude2 * g / sim_settings.drag_coefficient;
    }
    // samosAT friction model
    else if model == 3u {
        let min_shear_stress_samosat = 0f;
        let rs0 = 0.222;
        let kappa_inv = 2.32558; // 1/kappa, standard kappa = 0.43
        let r_inv = 20.0; // 1/r, standard r = 0.05
        let b = 4.13;
        let normal_stress = effective_acceleration_normal * mass_per_area;
        let rs = density_velocity_magnitude2 / (normal_stress + 0.001);
        var div = max(h * r_inv, 1.0);
        div = log(div) * kappa_inv + b;
        shear_stress = min_shear_stress_samosat + density_velocity_magnitude2 / (div * div);
    }
    let acceleration_magnitude = shear_stress / mass_per_area;
    return acceleration_magnitude;
}

const TEXTURE_GATHER_OFFSET = 1.0f / 512.0f;
// Samples height texture with bilinear filtering.
fn get_elevation(uv: vec2f) -> f32 {
    // TODO: fix interpolation at the edges of the texture
    return textureSampleLevel(dem_texture, tex_sampler, uv, 0).x;
}

fn get_normal(uv: vec2f) -> vec3f {
    return normalize(textureSampleLevel(normals_texture, tex_sampler, uv, 0).xyz);
}

fn get_curvature(uv: vec2f) -> vec3f {
    return textureSampleLevel(curvature_texture, tex_sampler, uv, 0).xyz;
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
// TODO calculate momentum factor 
// 2147483647.0 / (120.0 * 100.0 * 25 * 200)
// use override
const MOMENTUM_FACTOR: f32 =  1e-2; 
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR; 
const INV_MOMENTUM_FACTOR: f32 = 1 / MOMENTUM_FACTOR;

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

struct SimInfo {
    timestep: u32,
    dt: f32,
    elapsed_time: f32,
    number_particles: u32,
    elevation_threshold: f32,
    max_velocity: f32,
    max_flow_thickness: f32,
    flags: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;

struct SimSettings {
    num_steps: u32,
    model_type: u32,
    friction_model: u32,
    released_particles_per_cell: u32,
    grid_shape: vec2u,
    world_size: vec2f,
    snow_density: f32,
    slab_thickness: f32,
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
};

struct AtomicValues {
    peak_velocity: atomic<u32>,
    peak_flow_thickness: atomic<u32>,
    expected_max_velocity: atomic<u32>,
    travel_length: atomic<u32>,
    release_volume: atomic<u32>,
    number_release_cells: atomic<u32>,
    number_release_particles: atomic<u32>,
    stopped_particles: atomic<u32>,
};

struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

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

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cell3_to_uv(cell: vec3u) -> vec2f {
    return (vec2f(cell.xy) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cellf_to_uv(cell: vec2f) -> vec2f {
    return (cell + 0.5) / vec2f(sim_settings.grid_shape);
}

fn position3_to_cell(position: vec3f) -> vec2u {
    return position_to_cell(position.xy);
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

fn calculate_distance_to_node(particle_position: vec2f, node_position: vec2u) -> vec2f {
    return particle_position - vec2f(node_position);
}

fn get_base_node(grid_pos: vec2f) -> vec2u {
    return vec2u(floor(grid_pos - vec2f(0.5)));
}

fn compute_centroid(points: ptr<function, array<vec2<f32>, 256>>, count: u32) -> vec2<f32> {
    var area: f32 = 0.0;
    var cx: f32 = 0.0;
    var cy: f32 = 0.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let j = (i + 1u) % count;
        let p0 = (*points)[i];
        let p1 = (*points)[j];
        let cross = p0.x * p1.y - p1.x * p0.y;

        area = area + cross;
        cx = cx + (p0.x + p1.x) * cross;
        cy = cy + (p0.y + p1.y) * cross;
    }

    area = area * 0.5;

    if abs(area) < 1e-6 {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(cx, cy) / (6.0 * area);
}
// END utils.wgsl