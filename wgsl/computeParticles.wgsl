

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
    acceleration_normal: vec3f,             // 12 bytes     76
    // padding                                  4 bytes     80
    uv: vec2f,                              //  8 bytes     88
    g_eff: f32,                               // padding  8 bytes     96
};

// @group(0) @binding(0) var<uniform> simSettings: SimSettings;
@group(0) @binding(1) var<storage, read_write> simInfo: SimInfo;
@group(0) @binding(2) var dem_texture: texture_2d<f32>;
@group(0) @binding(3) var normals_texture: texture_2d<f32>;
@group(0) @binding(4) var <storage, read_write> particles: array<Particle>;
@group(0) @binding(5) var tex_sampler: sampler;

@group(0) @binding(6) var<storage, read_write> maxVelocity: AtomicValue;
@group(0) @binding(7) var<storage, read_write> out_timestep_data: array<TimestepDataArray>; // trajectory data, fixed size 3
@group(0) @binding(8) var<storage, read_write> out_debug: array<f32>;

@group(0) @binding(9) var<storage, read_write> output_texture_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(10) var<storage, read_write> output_velocity_texture_buffer: array<atomic<u32>>; // trajectory texture

// @group(0) @binding(11) var<storage, read_write> atomicBuffer: AtomicData;

const density: f32 = 200.0;
var<workgroup> sharedMaxVelocity: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn computeParticles(
    @builtin(global_invocation_id) pId: vec3<u32>, 
    @builtin(local_invocation_id) lId: vec3<u32>) {
    let particleId = pId.x;
    let localId = lId.x;

    let particle = &particles[particleId];
    if (particleId >= simInfo.number_particles) {
        return;
    }
    if (particle.stopped > 0u) {
        return;
    }
    let uv = positionToUV(particle.position);
    
    if(particleId == 0u){
    // atomicMax(&atomicBuffer.counter, step_count);
        out_debug[0] = f32(particle.position.x);
        out_debug[1] = f32(particle.position.y);
        out_debug[2] = f32(uv.x);
        out_debug[3] = f32(uv.y);
        out_debug[6] = f32(simInfo.timestep);
        out_debug[7] = f32(simInfo.number_particles);
        out_debug[8] = f32(simSettings.grid_shape.x);
        out_debug[9] = f32(simSettings.grid_shape.y);
        out_debug[10] = f32(simSettings.world_size.x);
        out_debug[11] = f32(simSettings.world_size.y);
        out_debug[12] = f32(simSettings.friction_coefficient);
        out_debug[13] = (f32(atomicLoad(&maxVelocity.value)) / f32(maxVelocityFactor));
        out_debug[14] = f32(simSettings.world_size.x);
        out_debug[15] = f32(simSettings.world_size.y);
        out_debug[16] = f32(simSettings.world_size.x);
    }
    let normal = getNormal(uv); 
    particle.velocity = particle.velocity - dot(particle.velocity, normal) * normal; 
    const acceleration_gravity = vec3f(0.0, 0.0, -g);
    let acceleration_normal = g * normal.z * normal;

    let acceleration_tangential = acceleration_gravity + acceleration_normal;
    var dt = simSettings.cfl * simSettings.cell_size / (simInfo.max_velocity + simSettings.velocity_threshold);
    particle.velocity = particle.velocity + acceleration_tangential * dt; 
    var acceleration_normal_friction_magnitude = acceleration_by_normal_friction(acceleration_normal, particle);
    let acceleration_drag_friction_magnitude = acceleration_by_drag_friction(acceleration_normal, particle);
    var acceleration_friction_magnitude = acceleration_drag_friction_magnitude + acceleration_normal_friction_magnitude;
    if(length(particle.velocity) < acceleration_friction_magnitude * dt){
        dt = length(particle.velocity) / acceleration_friction_magnitude;
        particle.stopped = simInfo.timestep;
    }
    let velocity_length = length(particle.velocity);
    if (velocity_length > simSettings.velocity_threshold) {
        particle.velocity -= acceleration_friction_magnitude * (particle.velocity / velocity_length) * dt;
    }
    let relative_trajectory = particle.velocity * dt;
    particle.position = particle.position + relative_trajectory;
    let new_uv = positionToUV(particle.position);
    let elevation = get_elevation(new_uv);
        

    if (particleId == 0u) {
        var current: TimestepData;
        current.position = particle.position;
        current.velocity = particle.velocity;
        current.dt = dt;
        current.acceleration_tangential = acceleration_tangential;
        current.acceleration_friction_magnitude = acceleration_friction_magnitude;
        current.normal = normal;
        current.acceleration_normal = acceleration_normal;
        current.elevation = elevation;
        current.uv = new_uv;
        update_output_data(0u, simInfo.timestep, current);
        
    // out_debug[2] = f32(particle.position.x);
        simInfo.timestep = simInfo.timestep + 1u;
    }


    atomicMax(&maxVelocity.value, u32(maxVelocityFactor * length(particle.velocity)));
    
    let cell_index = uvToCellIndex(new_uv);
    atomicAdd(&output_texture_buffer[cell_index], 1u);
    atomicMax(&output_velocity_texture_buffer[cell_index], u32(length(particle.velocity))); // ensure that the velocity is not zero, this is needed for the next step

    if(particleId == 0u){
    // atomicMax(&atomicBuffer.counter, step_count);
    out_debug[0] = f32(particle.position.x);
    out_debug[1] = f32(particle.position.y);
    out_debug[2] = f32(new_uv.x);
    out_debug[3] = f32(new_uv.y);
    out_debug[6] = f32(simInfo.timestep);
    out_debug[7] = f32(simSettings.released_particles_per_cell);
    out_debug[8] = f32(simSettings.grid_shape.x);
    out_debug[9] = f32(simSettings.grid_shape.y);
    out_debug[10] = f32(simSettings.world_size.x);
    out_debug[11] = f32(simSettings.world_size.y);
    out_debug[12] = f32(simSettings.friction_coefficient);
    out_debug[13] = (f32(atomicLoad(&maxVelocity.value)) / f32(maxVelocityFactor));
    out_debug[14] = f32(uvToCell(new_uv).x);
    out_debug[15] = f32(uvToCell(new_uv).y);
    out_debug[16] = f32(uvToCellIndex(new_uv));
    }
    // TODO more sophisticated projection methods
    particle.position.z = elevation; 

    // stop criterion friction
    if (length(particle.velocity) < simSettings.velocity_threshold) {
        particle.stopped = simInfo.timestep;
        return;
    }
    // out of bounds or non rectangular terrain
    if(new_uv.x < 0.0 || new_uv.x > 1.0 || new_uv.y < 0.0 || new_uv.y > 1.0 
        || elevation < simInfo.elevation_threshold){
        particle.stopped = simInfo.timestep;
        return;
    }
    
    // atomicLoad(&maxVelocity.value)
}


fn update_output_data(trajectory: u32, timestep: u32, timestep_data: TimestepData) {
    out_timestep_data[timestep].trajectories[trajectory] = timestep_data;
}

fn acceleration_by_normal_friction(acceleration_normal: vec3f, particle: ptr<storage, Particle, read_write>) -> f32 {
    let mass_per_area = simSettings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = simSettings.friction_model;
    if velocity_magnitude < simSettings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = simSettings.friction_coefficient;
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


fn acceleration_by_drag_friction(acceleration_normal: vec3f, particle: ptr<storage, Particle, read_write>) -> f32 {
    let mass_per_area = simSettings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = simSettings.friction_model;
    if velocity_magnitude < simSettings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = simSettings.friction_coefficient;
    let drag_coefficient = simSettings.drag_coefficient; // only used for voellmy, standard 4000.
    let normal_stress = length(acceleration_normal * mass_per_area);
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

fn get_normal_and_curvature(uv: vec2f) -> vec4f {
    return textureSampleLevel(normals_texture, tex_sampler, uv, 0); // convert from [0, 1] to [-1, 1
}

fn getNormal(uv: vec2f) -> vec3f {
    return get_normal_and_curvature(uv).xyz; // convert from [0, 1] to [-1, 1
}

fn acceleration_by_friction(acceleration_normal: vec3f, particle: ptr<storage, Particle, read_write>) -> f32 {
    let mass_per_area = simSettings.snow_density * particle.snow_thickness;
    let velocity_magnitude = length(particle.velocity);
    let model = simSettings.friction_model;
    if velocity_magnitude < simSettings.velocity_threshold || model >= 4u {
        return 0.0f;
    }
    // standard 0.155, samos: standard 0.155, small 0.22, medium 0.17
    let friction_coefficient = simSettings.friction_coefficient;
    let drag_coefficient = simSettings.drag_coefficient; // only used for voellmy, standard 4000.
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
        var div = particle.snow_thickness / r;
        if div < 1.0 {
            div = 1.0;
        }
        div = log(div) / kappa + b;
        shear_stress = min_shear_stress_samosat + normal_stress * friction_coefficient * (1.0 + rs0 / (rs0 + rs)) + density * velocity_magnitude * velocity_magnitude / (div * div);
    }
    let acceleration_magnitude = shear_stress / mass_per_area;
    return acceleration_magnitude;
}


