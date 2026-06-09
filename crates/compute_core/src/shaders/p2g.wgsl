@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> grid_mass_atomic: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_momentum_atomic: array<atomic<i32>>; // Combined u, v
@group(0) @binding(4) var<storage, read_write> sim_info: SimInfo;

override WG_SIZE_1D: u32 = 1u;
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn p2g(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_info.number_particles {
        return;
    }
    if sim_info.flags >= SIM_INFO_STOPPED {
        return;
    }
    
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    if !use_particle_interaction {
        return;
    }
    let p = particles[id.x];

    let grid_pos = p.position.xy / sim_settings.cell_size;

    let base_node = get_base_node(grid_pos);

    for (var i: i32 = 0; i < 3; i++) {
        for (var j: i32 = 0; j < 3; j++) {

            let node_coords = base_node + vec2i(i, j);

            if node_coords.x < 0 ||
                node_coords.y < 0 ||
                node_coords.x >= i32(sim_settings.grid_shape.x) ||
                node_coords.y >= i32(sim_settings.grid_shape.y) {
                continue;
            }

            let weight = calculate_weight(grid_pos, node_coords);

            let idx = u32(node_coords.y) * sim_settings.grid_shape.x +
                u32(node_coords.x);

            atomicAdd(
                &grid_mass_atomic[idx],
                u32(p.mass * weight * MASS_FACTOR)
            );
            atomicAdd(&grid_momentum_atomic[idx * 2], i32(p.mass * p.velocity.x * weight * MOMENTUM_FACTOR));
            atomicAdd(&grid_momentum_atomic[idx * 2 + 1], i32(p.mass * p.velocity.y * weight * MOMENTUM_FACTOR));
        }
    }
}

// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
const MOMENTUM_FACTOR: f32 = 1e2; 
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR; 
const INV_MOMENTUM_FACTOR: f32 = 1 / MOMENTUM_FACTOR;

// TODO precompute often used values on the cpu and pass them as uniforms to avoid redundant calculations on the gpu

struct Particle {
    position: vec3f,
    mass: f32,
    velocity: vec3f,
    stopped: u32,
    travel_length: f32,
};

struct ParticleAlpha {
    alpha: f32,
    start_elevation: f32,
};

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
};

struct AtomicValues {
    peak_velocity: atomic<u32>,
    peak_flow_thickness: atomic<u32>,
    alpha: atomic<u32>,
    travel_length: atomic<u32>,
    release_volume: atomic<u32>,
    number_release_cells: atomic<u32>,
    number_release_particles: atomic<u32>,
    stopped_particles: atomic<u32>,
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
    return (position.xy + 0.5 * sim_settings.cell_size) / (vec2f(sim_settings.world_size)); // add some padding to ensure particles outside the world bounds are still captured in the simulation info
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
