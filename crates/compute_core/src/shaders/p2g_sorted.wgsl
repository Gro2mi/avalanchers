// Curvilinear P2G over particles in tile-sorted order (see sort_particles.wgsl).
// A workgroup whose particles touch at most SHARED_TILE x SHARED_TILE nodes
// accumulates the quantized contributions in workgroup memory and flushes
// each non-zero node sum with one global atomic; otherwise it falls back to
// direct global atomics. On the integer-atomic grid both paths give identical
// grids. On the float-atomic grid the shared path keeps the integer
// quantization and converts each node sum once, so it deviates from the
// unquantized direct float deposit by at most the quantization step per node.
@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var<storage> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_velocity: array<vec2<f32>>;
@group(0) @binding(4) var<storage> particles_mass: array<f32>;
@group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<u32>>; // no_atomic_float
@group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<i32>>; // no_atomic_float
// atomic_float @group(0) @binding(5) var<storage, read_write> grid_mass_atomic: array<atomic<f32>>;
// atomic_float @group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<atomic<f32>>;
@group(0) @binding(7) var<storage> particles_affine_matrix: array<mat2x2f>;
@group(0) @binding(8) var<storage> sorted_particle_index: array<u32>;

const P2G_SORTED_WG_SIZE: u32 = 256u;
const SHARED_TILE: u32 = 32u;
const SHARED_TILE_NODES: u32 = SHARED_TILE * SHARED_TILE;

// zero-initialized at the start of every workgroup
var<workgroup> tile_mass: array<atomic<u32>, SHARED_TILE_NODES>;
var<workgroup> tile_momentum_x: array<atomic<i32>, SHARED_TILE_NODES>;
var<workgroup> tile_momentum_y: array<atomic<i32>, SHARED_TILE_NODES>;
var<workgroup> box_min: array<atomic<u32>, 2>;
var<workgroup> box_max: array<atomic<u32>, 2>;
var<workgroup> box: vec4<u32>;

@compute @workgroup_size(P2G_SORTED_WG_SIZE, 1, 1)
fn p2g_curvilinear_sorted(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    if lid.x == 0u {
        atomicStore(&box_min[0], 0xFFFFFFFFu);
        atomicStore(&box_min[1], 0xFFFFFFFFu);
    }
    workgroupBarrier();

    let slot = wid.x * P2G_SORTED_WG_SIZE + lid.x;
    var valid = slot < sim_info.number_particles && (sim_info.flags & SIM_INFO_STOPPED) == 0u;
    var p_idx = 0u;
    var grid_pos = vec2f(0.0);
    var base_node = vec2u(0u);
    if valid {
        p_idx = sorted_particle_index[slot];
        let cell = position_to_cell(particles_position[p_idx]);
        valid = cell.x >= 3u && cell.x < sim_settings.grid_shape.x - 3u
            && cell.y >= 3u && cell.y < sim_settings.grid_shape.y - 3u;
    }
    if valid {
        // same stencil as transfer_p2g_with_affine
        grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
        base_node = vec2u(floor(grid_pos - vec2f(0.5)));
        atomicMin(&box_min[0], base_node.x);
        atomicMin(&box_min[1], base_node.y);
        atomicMax(&box_max[0], base_node.x);
        atomicMax(&box_max[1], base_node.y);
    }
    workgroupBarrier();
    if lid.x == 0u {
        box = vec4u(atomicLoad(&box_min[0]), atomicLoad(&box_min[1]),
                    atomicLoad(&box_max[0]), atomicLoad(&box_max[1]));
    }
    let node_box = workgroupUniformLoad(&box);
    if node_box.x > node_box.z {
        return; // no valid particle in this workgroup
    }
    let fits = node_box.z + 3u - node_box.x <= SHARED_TILE && node_box.w + 3u - node_box.y <= SHARED_TILE;
    if !fits {
        if valid {
            transfer_p2g_with_affine(p_idx, particles_affine_matrix[p_idx]);
        }
        return;
    }

    if valid {
        let p_mass = particles_mass[p_idx];
        let p_velocity = particles_velocity[p_idx];
        let affine_matrix = particles_affine_matrix[p_idx];
        for (var i: u32 = 0; i < 3; i++) {
            for (var j: u32 = 0; j < 3; j++) {
                let node_coords = base_node + vec2u(i, j);
                let distance = calculate_distance_to_node(grid_pos, node_coords);
                let weight = calculate_weight(distance);
                let local = node_coords - node_box.xy;
                let tile_idx = local.y * SHARED_TILE + local.x;
                let affine_velocity = p_velocity - affine_matrix * distance;
                let mass = u32(round(p_mass * weight * MASS_FACTOR));
                let momentum_x = quantize_momentum_component(p_mass * affine_velocity.x * weight * MOMENTUM_FACTOR);
                let momentum_y = quantize_momentum_component(p_mass * affine_velocity.y * weight * MOMENTUM_FACTOR);
                if mass != 0u { atomicAdd(&tile_mass[tile_idx], mass); }
                if momentum_x != 0 { atomicAdd(&tile_momentum_x[tile_idx], momentum_x); }
                if momentum_y != 0 { atomicAdd(&tile_momentum_y[tile_idx], momentum_y); }
            }
        }
    }
    workgroupBarrier();

    for (var tile_idx = lid.x; tile_idx < SHARED_TILE_NODES; tile_idx += P2G_SORTED_WG_SIZE) {
        let node = node_box.xy + vec2u(tile_idx % SHARED_TILE, tile_idx / SHARED_TILE);
        if node.x >= sim_settings.grid_shape.x || node.y >= sim_settings.grid_shape.y {
            continue;
        }
        let idx = xy_to_idx(node);
        let mass = atomicLoad(&tile_mass[tile_idx]);
        let momentum_x = atomicLoad(&tile_momentum_x[tile_idx]);
        let momentum_y = atomicLoad(&tile_momentum_y[tile_idx]);
        if mass != 0u { atomicAdd(&grid_mass_atomic[idx], mass); } // no_atomic_float
        if momentum_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], momentum_x); } // no_atomic_float
        if momentum_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], momentum_y); } // no_atomic_float
        // atomic_float if mass != 0u { atomicAdd(&grid_mass_atomic[idx], f32(mass) * INV_MASS_FACTOR); }
        // atomic_float if momentum_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], f32(momentum_x) * INV_MOMENTUM_FACTOR); }
        // atomic_float if momentum_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], f32(momentum_y) * INV_MOMENTUM_FACTOR); }
    }
}

// import transfer_p2g.wgsl;
// BEGIN transfer_p2g.wgsl
fn quantize_momentum_component(value: f32) -> i32 {
    // Keep conversion inside i32 range before atomicAdd.
    let i32_max_f = 2147483647.0;
    let clamped = clamp(value, -i32_max_f, i32_max_f);
    return i32(round(clamped));
}

fn transfer_p2g(p_idx: u32) {
    transfer_p2g_with_affine(p_idx, mat2x2f(vec2f(0.0), vec2f(0.0)));
}

fn transfer_p2g_with_affine(p_idx: u32, affine_matrix: mat2x2f) {
    // the stencil is offset by half a cell as it is in the middle of the cell
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    
    let p_mass = particles_mass[p_idx];
    let p_velocity = particles_velocity[p_idx];
    // let affine_matrix = particles_affine_matrix[p_idx];

    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let affine_velocity = p_velocity - affine_matrix * distance;

            
            
            // atomic_float atomicAdd(&grid_mass_atomic[idx], p_mass * weight);
            // atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomic_float atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);
            // for affine transfer
            // atomicAdd(&grid_momentum_atomic[idx * 2u], p_mass * affine_velocity.x * weight);
            // atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], p_mass * affine_velocity.y * weight);

            atomicAdd(&grid_mass_atomic[idx], u32(round(p_mass * weight * MASS_FACTOR))); // no_atomic_float 
            let momentum_x = p_mass * affine_velocity.x * weight * MOMENTUM_FACTOR; // no_atomic_float
            let momentum_y = p_mass * affine_velocity.y * weight * MOMENTUM_FACTOR; // no_atomic_float
            // resting particles carry no momentum; skipping zero adds saves most of their atomics
            let quantized_x = quantize_momentum_component(momentum_x); // no_atomic_float
            let quantized_y = quantize_momentum_component(momentum_y); // no_atomic_float
            if quantized_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], quantized_x); } // no_atomic_float
            if quantized_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], quantized_y); } // no_atomic_float
        }
    }
}
// END transfer_p2g.wgsl
// import utils.wgsl;
// BEGIN utils.wgsl
const WG_SIZE_2D: u32 = 16u;

const g: f32 = 9.81;

// u32 limit is 4 294 967 296
const MAX_VELOCITY_FACTOR: f32 = 1e7; // u32 limit is 430 m/s
const MASS_FACTOR: f32 = 1e1; // u32 limit is 4.3t thickness
const H_FACTOR: f32 = 1e6;
// Momentum is quantized per particle->node contribution before atomicAdd, so
// MOMENTUM_FACTOR sets the velocity resolution of that contribution:
// the smallest non-zero contribution is v = 1 / (particle_mass * weight * MOMENTUM_FACTOR).
// It must stay far below the slow hydrostatic spreading velocities (~0.01-0.1 m/s)
// or p2g rounds them to zero every step and the flow never spreads laterally.
// i32 budget: node sum = node_mass * v_max * MOMENTUM_FACTOR
//   (rho*cell^2*h*J ~ 6e4 kg * 40 m/s * 1e2 = 2.4e8 < 2.1e9)
const MOMENTUM_FACTOR: f32 = 1e2;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_MASS_FACTOR: f32 = 1 / MASS_FACTOR; // u32 limit is 4.3km thickness
const INV_H_FACTOR: f32 = 1 / H_FACTOR;
const INV_MOMENTUM_FACTOR: f32 = 1 / MOMENTUM_FACTOR;
// depth-integrated internal force (h * sigma * grad_w * area, ~1e4..1e5 N per node contribution)
const FORCE_FACTOR: f32 = 1e-3; // i32 limit is 2.1e6 N per node
const INV_FORCE_FACTOR: f32 = 1 / FORCE_FACTOR;

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
    grid_elapsed_time: f32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;

// reversible low-speed state; the low bits hold the rest start step while resting
// and the consecutive slow-step count while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
const PARTICLE_FLYING: u32 = 1u << 27u;
const PARTICLE_OUT_OF_BOUNDS: u32 = 1u << 28u;
const PARTICLE_IS_NAN: u32 = 1u << 29u;
const PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 30u;
const PARTICLE_STOPPED: u32 = 1u << 31u;

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
    peak_flow_thickness_threshold: f32,
    // MPMDAC constitutive model (must mirror the Rust POD layout)
    constitutive_model: u32,
    shear_modulus: f32,
    hardening_modulus: f32,
    // MPMDAC compressibility; bulk_modulus 0 = incompressible
    bulk_modulus: f32,
    compaction_pressure: f32,
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
    // resting particles counted during the current step, reset by update_sim_info
    resting_particles: atomic<u32>,
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

fn determinant_2x2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

fn identity_2x2() -> mat2x2<f32> {
    return mat2x2<f32>(vec2f(1.0, 0.0), vec2f(0.0, 1.0));
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
