// import utils.wgsl;
// BEGIN utils.wgsl
struct Particle {
    position: vec3f,
    mass: f32,
    velocity: vec3f,
    snow_thickness: f32,
    C: mat2x2f,
    stopped: u32,
};

struct SimInfo {
  timestep: u32,
  number_particles: u32,
  elevation_threshold: f32,
  max_velocity: f32,
};

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
    cfl: f32,
    cell_size: f32,
    min_slope_angle: f32,
    max_slope_angle: f32,
    min_elevation: f32,
    velocity_threshold: f32,
    roughness_threshold: f32,
};

struct AtomicValue {
    value: atomic<u32>,
};

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;
const MAX_VELOCITY_FACTOR: f32 = 1e7;

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

fn cell_to_uv(cell: vec2u) -> vec2f {
    return (vec2f(cell) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cell3_to_uv(cell: vec3u) -> vec2f {
    return (vec2f(cell.xy) + 0.5) / vec2f(sim_settings.grid_shape);
}
fn cellf_to_uv(cell: vec2f) -> vec2f {
    return (cell + 0.5) / vec2f(sim_settings.grid_shape);
}

fn position_to_uv(position: vec3f) -> vec2f {
    return position.xy / vec2f(sim_settings.world_size);
}

fn position_to_cell_index(position: vec3f) -> u32 {
    let uv = position_to_uv(position);
    return uv_to_cell_index(uv);
}


fn uv_to_cell(uv: vec2f) -> vec2u {
    return vec2u(clamp(uv * vec2f(sim_settings.grid_shape), vec2f(0.0), vec2f(sim_settings.grid_shape - 1u)));
}

fn uv_to_cell_index(uv: vec2f) -> u32 {
    let cell = uv_to_cell(uv);
    // return cell.x * sim_settings.grid_shape.y + cell.y;
    return (cell.y % sim_settings.grid_shape.y * sim_settings.grid_shape.x +
              (cell.x % sim_settings.grid_shape.x));
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

    if (abs(area) < 1e-6) {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(cx, cy) / (6.0 * area);
}



// END utils.wgsl

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
@group(0) @binding(5) var <storage, read_write> particles: array<Particle>;

@group(0) @binding(6) var<storage, read_write> maxVelocity: AtomicValue;

@group(0) @binding(7) var<storage, read_write> atomic_cell_count_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(8) var<storage, read_write> atomic_velocity_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(9) var<storage, read_write> out_timestep_data: array<TimestepDataArray>; // trajectory data, fixed size 3

@group(0) @binding(10) var curvature_texture: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> out_debug: array<f32>;
// @group(0) @binding(11) var<storage, read_write> atomicBuffer: AtomicData;


const density: f32 = 200.0;

@compute @workgroup_size(64, 1, 1)
fn compute_particles(
    @builtin(global_invocation_id) pId: vec3<u32>, 
    @builtin(local_invocation_id) lId: vec3<u32>) {
    let particleId = pId.x;
    let localId = lId.x;

    if (particleId >= sim_info.number_particles) {
        return;
    }
    // do at least one step, otherwise simulation might stop too early
    if (particles[particleId].stopped > 1u) {
        return;
    }
    let uv = position_to_uv(particles[particleId].position);
    
    let normal = get_normal(uv);
    let u = particles[particleId].velocity;
    particles[particleId].velocity = particles[particleId].velocity - dot(particles[particleId].velocity, normal) * normal; 
    
    // let curvature_acceleration = get_curvature(uv) * length(particles[particleId].velocity) * length(particles[particleId].velocity);
    const acceleration_gravity = vec3f(0.0, 0.0, -g);
    var acceleration_normal = g * normal.z * normal;
    let acceleration_tangential = acceleration_gravity + acceleration_normal;
    // uKu, K curvature matrix
    let curvature = get_curvature(uv);
    let centrifugal_acceleration = (u.x * u.x * curvature.x) + (2.0 * u.x * u.y * curvature.y) + (u.y * u.y * curvature.z);
    acceleration_normal = acceleration_normal + centrifugal_acceleration;
    var dt = sim_settings.cfl * sim_settings.cell_size / (sim_info.max_velocity + sim_settings.velocity_threshold);
    particles[particleId].velocity = particles[particleId].velocity + acceleration_tangential * dt; 
    var acceleration_normal_friction_magnitude = acceleration_by_normal_friction(acceleration_normal, particles[particleId]);
    let acceleration_drag_friction_magnitude = acceleration_by_drag_friction(acceleration_normal, particles[particleId]);
    var acceleration_friction_magnitude = acceleration_drag_friction_magnitude + acceleration_normal_friction_magnitude;
    if(length(particles[particleId].velocity) < acceleration_friction_magnitude * dt){
        dt = length(particles[particleId].velocity) / acceleration_friction_magnitude;
        particles[particleId].stopped = sim_info.timestep;
    }
    let velocity_length = length(particles[particleId].velocity);
    if (velocity_length > sim_settings.velocity_threshold) {
        particles[particleId].velocity -= acceleration_friction_magnitude * (particles[particleId].velocity / velocity_length) * dt;
    }
    var relative_trajectory = particles[particleId].velocity * dt;
    var new_position = particles[particleId].position + relative_trajectory;
    var new_uv = position_to_uv(new_position);
    var elevation = get_elevation(new_uv);
    var z_diff = new_position.z - elevation;
    let offset = dot(relative_trajectory, normal) * normal;
    let alignment = dot(offset, normal);
    // var z_diff = length(offset);
    // if (alignment > 0f){
    //     z_diff = -z_diff;
    // }
    var curvature_acceleration = z_diff * normal.z / dt / dt; // curvature acceleration
    var g_eff = (g + curvature_acceleration);
    if (g_eff < 0.0) {
        g_eff = 0.0;
    }
    acceleration_normal_friction_magnitude = acceleration_by_normal_friction(g_eff * normal.z * normal, particles[particleId]);
    acceleration_friction_magnitude = acceleration_drag_friction_magnitude + acceleration_normal_friction_magnitude;
    relative_trajectory = particles[particleId].velocity * dt;
    new_position = particles[particleId].position + relative_trajectory;
    new_uv = position_to_uv(new_position);
    elevation = get_elevation(new_uv);
    z_diff = new_position.z - elevation;

    particles[particleId].position = new_position;
    
        

    if (particleId == 0u) {
        var current: TimestepData;
        current.position = particles[particleId].position;
        current.velocity = particles[particleId].velocity;
        current.dt = dt;
        current.acceleration_tangential = acceleration_tangential;
        current.acceleration_friction_magnitude = acceleration_friction_magnitude;
        current.normal = normal;
        current.acceleration_normal = acceleration_normal;
        current.elevation = elevation;
        current.uv = new_uv;
        current.g_eff = g_eff;
        update_output_data(0u, sim_info.timestep, current);
        
    // out_debug[2] = f32(particles[particleId].position.x);
        sim_info.timestep = sim_info.timestep + 1u;
    }

    let converted_velocity = u32(MAX_VELOCITY_FACTOR * length(particles[particleId].velocity));
    atomicMax(&maxVelocity.value, converted_velocity);
    
    let cell_index = uv_to_cell_index(new_uv);
    atomicAdd(&atomic_cell_count_buffer[cell_index], 1u);
    atomicMax(&atomic_velocity_buffer[cell_index], converted_velocity); // ensure that the velocity is not zero, this is needed for the next step

    if(particleId == 0u){
    // atomicMax(&atomicBuffer.counter, step_count);
        out_debug[0] = f32(particles[particleId].position.x);
        out_debug[1] = f32(particles[particleId].position.y);
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
        out_debug[13] = (f32(atomicLoad(&maxVelocity.value)) / f32(MAX_VELOCITY_FACTOR));
        out_debug[14] = f32(uv_to_cell(new_uv).x);
        out_debug[15] = f32(uv_to_cell(new_uv).y);
        out_debug[16] = f32(uv_to_cell_index(new_uv));
    }
    // TODO more sophisticated projection methods
    particles[particleId].position.z = elevation; 

    // stop criterion friction
    if (length(particles[particleId].velocity) < sim_settings.velocity_threshold) {
        particles[particleId].stopped = sim_info.timestep;
        return;
    }
    // out of bounds or non rectangular terrain
    if(new_uv.x < 0.0 || new_uv.x > 1.0 || new_uv.y < 0.0 || new_uv.y > 1.0 
        || elevation < sim_info.elevation_threshold){
        particles[particleId].stopped = sim_info.timestep;
        return;
    }
    // atomicLoad(&maxVelocity.value)
}


fn update_output_data(trajectory: u32, timestep: u32, timestep_data: TimestepData) {
    out_timestep_data[timestep].trajectories[trajectory] = timestep_data;
}

fn acceleration_by_normal_friction(acceleration_normal: vec3f, particle: Particle) -> f32 {
    let mass_per_area = sim_settings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = sim_settings.friction_model;
    if velocity_magnitude < sim_settings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = sim_settings.friction_coefficient;
    let normal_stress = length(acceleration_normal) * mass_per_area;
    const min_shear_stress = 70f;
    var shear_stress = 0.0f;
    //actually: friction model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAt
    // Coulomb friction model
    if (model == 0u || model == 1u || model == 2u){
        shear_stress = friction_coefficient * normal_stress;
    }
    // samosAT friction model
    else if (model == 3){
        let rs0 = 0.222;
        let rs = density * velocity_magnitude * velocity_magnitude / (normal_stress + 0.001);
        shear_stress = normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs)) ;
    }
    let acceleration_magnitude = shear_stress / mass_per_area;
    return acceleration_magnitude;
}


fn acceleration_by_drag_friction(acceleration_normal: vec3f, particle: Particle) -> f32 {
    let mass_per_area = sim_settings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = sim_settings.friction_model;
    if velocity_magnitude < sim_settings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = sim_settings.friction_coefficient;
    let drag_coefficient = sim_settings.drag_coefficient; // only used for voellmy, standard 4000.
    const min_shear_stress = 70f;
    var shear_stress = 0.0f;
    //actually: friction model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAt
    // Coulomb friction model
    // Voellmy friction model
    if (model == 1){
        shear_stress = density * g * velocity_magnitude * velocity_magnitude / drag_coefficient;
    }
    // Voellmy min shear friction model
    else if (model == 2){
        shear_stress = min_shear_stress + density * g * velocity_magnitude * velocity_magnitude / drag_coefficient;
    }
    // samosAT friction model
    else if (model == 3){
        let min_shear_stress_samosat = 0f;
        let rs0 = 0.222;
        let kappa = 0.43;
        let r = 0.05;
        let b = 4.13;
        let normal_stress = length(acceleration_normal * mass_per_area);
        let rs = density * velocity_magnitude * velocity_magnitude / (normal_stress + 0.001);
        var div = particle.snow_thickness / r;
        if div < 1.0 {
            div = 1.0;
        }
        div = log(div) / kappa + b;
        shear_stress = min_shear_stress_samosat + density * velocity_magnitude * velocity_magnitude / (div * div);
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
    return textureSampleLevel(normals_texture, tex_sampler, uv, 0).xyz;
}

fn get_curvature(uv: vec2f) -> vec3f {
    return textureSampleLevel(curvature_texture, tex_sampler, uv, 0).xyz;
}

fn acceleration_by_friction(acceleration_normal: vec3f, particle: Particle) -> f32 {
    let mass_per_area = sim_settings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = sim_settings.friction_model;
    if velocity_magnitude < sim_settings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = sim_settings.friction_coefficient;
    let drag_coefficient = sim_settings.drag_coefficient; // only used for voellmy, standard 4000.
    let normal_stress = length(acceleration_normal * mass_per_area);
    const min_shear_stress = 70f;
    var shear_stress = 0.0f;
    //actually: friction model: 0 coulomb, 1 voellmy, 2 voellmy minshear, 3 samosAt
    // Coulomb friction model
    if (model == 0u){
        shear_stress = friction_coefficient * normal_stress;
    }
    // Voellmy friction model
    else if (model == 1){
        shear_stress = friction_coefficient * normal_stress + density * g * velocity_magnitude * velocity_magnitude / drag_coefficient;
    }
    // Voellmy min shear friction model
    else if (model == 2){
        shear_stress = min_shear_stress + friction_coefficient * normal_stress + density * g * velocity_magnitude * velocity_magnitude / drag_coefficient;
    }
    // samosAT friction model
    else if (model == 3){
        let min_shear_stress_samosat = 0f;
        let rs0 = 0.222;
        let kappa = 0.43;
        let r = 0.05;
        let b = 4.13;
        let rs = density * velocity_magnitude * velocity_magnitude / (normal_stress + 0.001);
        var div = 30 / r;
        if div < 1.0 {
            div = 1.0;
        }
        div = log(div) / kappa + b;
        shear_stress = min_shear_stress_samosat + normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs)) + density * velocity_magnitude * velocity_magnitude / (div * div);
    }
    let acceleration_magnitude = shear_stress / mass_per_area;
    return acceleration_magnitude;
}


