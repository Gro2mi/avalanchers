@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var dem_texture: texture_2d<f32>;
@group(0) @binding(3) var release_areas: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;

@group(0) @binding(5) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(6) var<storage, read_write> number_release_particles: AtomicValue;
@group(0) @binding(7) var<storage, read_write> atomic_cell_count_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(8) var<storage, read_write> atomic_velocity_buffer: array<atomic<u32>>; // trajectory texture

@group(0) @binding(9) var<storage, read_write> debug: array<f32>;
// @group(0) @binding(9) var<uniform> tracked_particle_relative_positions: array<Particle>;
// @group(0) @binding(10) var<storage, read_write> tracked_particle_ids: array<u32>;

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn initialize_particles(@builtin(global_invocation_id) cell: vec3<u32>) {
    if cell.x == 0 && cell.y == 0 {
        // set max velocity for the first timestep, 
        // 3.5 m/s^2 for tangential acceleration at 50° slope
        sim_info.max_velocity = sqrt(2f * sim_settings.cfl * sim_settings.cell_size / 3.5);
        sim_info.elevation_threshold = minElevation() - 0.1;
    }

    if cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y {
        return;
    }
    let snow_thickness = textureLoad(release_areas, cell.xy, 0).r;
    if snow_thickness <= 0.01 {
        return;
    }

    // TODO: pass timestamp as seed
    let seed = 42u;
    var rng_seed = pcg_hash(cell.x + cell.y + seed);
    for (var n: u32 = 0; n < sim_settings.released_particles_per_cell; n++) {
        let particleIndex = atomicAdd(&number_release_particles.value, 1u);

        let r = rand2(&rng_seed);
        let cell_xy = (vec2f(cell.xy) + r);
        let elevation = textureSampleLevel(dem_texture, tex_sampler, cellf_to_uv(cell_xy), 0).x;

        particles[particleIndex].position = vec3f(cell_xy * sim_settings.cell_size, elevation);
        particles[particleIndex].mass = (sim_settings.snow_density * snow_thickness * sim_settings.cell_size * sim_settings.cell_size) / f32(sim_settings.released_particles_per_cell);
        // p.velocity = rand32u(cell.xy + seed) * 1e-5;
        particles[particleIndex].velocity = vec3f(0f);
        particles[particleIndex].snow_thickness = snow_thickness;
        particles[particleIndex].C = mat2x2f(0f, 0f, 0f, 0f);
        particles[particleIndex].stopped = 0u;

        let cell_index = position_to_cell_index(particles[particleIndex].position);
        atomicAdd(&atomic_cell_count_buffer[cell_index], 1u);
        atomicMax(&atomic_velocity_buffer[cell_index], u32(length(particles[particleIndex].velocity))); // ensure that the velocity is not zero, this is needed for the next step
    }
}

const MIN_VALID_ELEVATION: f32 = 0.9; 

fn minElevation() -> f32 {
    // find the minimum elevation in the height texture
    var min_val: f32 = 1e10;

    for (var y: u32 = 0; y < sim_settings.grid_shape.y; y++) {
        for (var x: u32 = 0; x < sim_settings.grid_shape.x; x++) {
            let value = textureLoad(dem_texture, vec2u(x, y), 0).x;
            // Only consider values above the minimum valid elevation threshold
            if value < min_val && value > MIN_VALID_ELEVATION {
                min_val = value;
            }
        }
    }
    return min_val;
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;
const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const H_FACTOR: f32 = 1e6; // u32 limit is 4.3km thickness
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_H_FACTOR: f32 = 1 / H_FACTOR; // u32 limit is 4.3km thickness

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

fn xy_to_idx(x: u32, y: u32) -> u32 {
    return y * sim_settings.grid_shape.x + x;
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

fn calculate_weight(particle_position: vec2f, node_position: vec2i) -> f32 {
    let dist = particle_position - vec2f(node_position);
    return quadratic_weight(dist.x) * quadratic_weight(dist.y);
}

fn get_base_node(grid_pos: vec2f) -> vec2i {
    return vec2i(floor(grid_pos - vec2f(0.5)));
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

// import random.wgsl;
// BEGIN random.wgsl
// A high-quality 32-bit hash (PCG)
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advances the seed and returns a float 0.0 -> 1.0
fn next_rand(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / f32(0xffffffffu);
}

fn rand1(seed: ptr<function, u32>) -> f32 {
    return next_rand(seed);
}

fn rand2(seed: ptr<function, u32>) -> vec2f {
    return vec2f(next_rand(seed), next_rand(seed));
}

fn rand3(seed: ptr<function, u32>) -> vec3f {
    return vec3f(next_rand(seed), next_rand(seed), next_rand(seed));
}

fn rand4(seed: ptr<function, u32>) -> vec4f {
    return vec4f(next_rand(seed), next_rand(seed), next_rand(seed), next_rand(seed));
}

// END random.wgsl