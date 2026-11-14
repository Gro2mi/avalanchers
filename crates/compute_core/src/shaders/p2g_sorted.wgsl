// P2G over particles in tile-sorted order (see sort_particles.wgsl), for the
// models with a 2D velocity (p2g_2d_sorted) and the 3D model (p2g_3d_sorted).
// A workgroup whose particles touch at most SHARED_TILE x SHARED_TILE nodes
// accumulates the quantized contributions in workgroup memory and flushes
// each non-zero node sum with one global atomic; otherwise it falls back to
// direct global atomics. On the integer-atomic grid both paths give identical
// grids. Float-atomic devices never run this shader (see prepare_particle_sort):
// the shared tile quantizes, naga allows atomic<f32> only in storage memory,
// and an f32 compare-exchange tile was ~3x slower than the direct float P2G.
// Its float variant still has to compile, because every pipeline is built.
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
// z components of the 3D model; the 2D layout ends at binding 8
@group(0) @binding(9) var<storage> particles_velocity_z: array<f32>;
@group(0) @binding(10) var<storage, read_write> grid_momentum_z_atomic: array<atomic<i32>>; // no_atomic_float
// atomic_float @group(0) @binding(10) var<storage, read_write> grid_momentum_z_atomic: array<atomic<f32>>;
@group(0) @binding(11) var<storage> particles_affine_z: array<vec2f>;

const P2G_SORTED_WG_SIZE: u32 = 256u;
// 31 keeps the four tiles of the 3D kernel inside the 16 KiB of workgroup
// memory WebGPU guarantees; 32 would exceed it by the box variables
const SHARED_TILE: u32 = 31u;
const SHARED_TILE_NODES: u32 = SHARED_TILE * SHARED_TILE;

// zero-initialized at the start of every workgroup
var<workgroup> tile_mass: array<atomic<u32>, SHARED_TILE_NODES>;
var<workgroup> tile_momentum_x: array<atomic<i32>, SHARED_TILE_NODES>;
var<workgroup> tile_momentum_y: array<atomic<i32>, SHARED_TILE_NODES>;
var<workgroup> tile_momentum_z: array<atomic<i32>, SHARED_TILE_NODES>;
var<workgroup> box_min: array<atomic<u32>, 2>;
var<workgroup> box_max: array<atomic<u32>, 2>;
var<workgroup> box: vec4<u32>;

struct SortedParticle {
    valid: bool,
    p_idx: u32,
    grid_pos: vec2f,
    base_node: vec2u,
}

@compute @workgroup_size(P2G_SORTED_WG_SIZE, 1, 1)
fn p2g_2d_sorted(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let particle = load_sorted_particle(lid.x, wid.x);
    let node_box = shared_node_box(lid.x);
    if node_box.x > node_box.z {
        return; // no valid particle in this workgroup
    }
    if !box_fits_tile(node_box) {
        if particle.valid {
            transfer_p2g_with_affine(particle.p_idx, particles_affine_matrix[particle.p_idx]);
        }
        return;
    }
    if particle.valid {
        deposit_to_tile(particle, node_box);
    }
    workgroupBarrier();
    flush_tile(lid.x, node_box);
}

@compute @workgroup_size(P2G_SORTED_WG_SIZE, 1, 1)
fn p2g_3d_sorted(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let particle = load_sorted_particle(lid.x, wid.x);
    let node_box = shared_node_box(lid.x);
    if node_box.x > node_box.z {
        return; // no valid particle in this workgroup
    }
    if !box_fits_tile(node_box) {
        if particle.valid {
            transfer_p2g_with_affine(particle.p_idx, particles_affine_matrix[particle.p_idx]);
            transfer_p2g_z(particle.p_idx);
        }
        return;
    }
    if particle.valid {
        deposit_to_tile(particle, node_box);
        deposit_z_to_tile(particle, node_box);
    }
    workgroupBarrier();
    flush_tile(lid.x, node_box);
    flush_tile_z(lid.x, node_box);
}

// Must be called in uniform control flow, it contains barriers.
fn load_sorted_particle(lane: u32, group: u32) -> SortedParticle {
    if lane == 0u {
        atomicStore(&box_min[0], 0xFFFFFFFFu);
        atomicStore(&box_min[1], 0xFFFFFFFFu);
    }
    workgroupBarrier();

    let slot = group * P2G_SORTED_WG_SIZE + lane;
    var particle = SortedParticle(false, 0u, vec2f(0.0), vec2u(0u));
    particle.valid = slot < sim_info.number_particles && (sim_info.flags & SIM_INFO_STOPPED) == 0u;
    if particle.valid {
        particle.p_idx = sorted_particle_index[slot];
        let cell = position_to_cell(particles_position[particle.p_idx]);
        particle.valid = cell.x >= 3u && cell.x < sim_settings.grid_shape.x - 3u
            && cell.y >= 3u && cell.y < sim_settings.grid_shape.y - 3u;
    }
    if particle.valid {
        // same stencil as transfer_p2g_with_affine
        particle.grid_pos = particles_position[particle.p_idx] / sim_settings.cell_size - vec2f(0.5);
        particle.base_node = vec2u(floor(particle.grid_pos - vec2f(0.5)));
        atomicMin(&box_min[0], particle.base_node.x);
        atomicMin(&box_min[1], particle.base_node.y);
        atomicMax(&box_max[0], particle.base_node.x);
        atomicMax(&box_max[1], particle.base_node.y);
    }
    workgroupBarrier();
    return particle;
}

// Node range of the workgroup, identical in all invocations.
fn shared_node_box(lane: u32) -> vec4u {
    if lane == 0u {
        box = vec4u(atomicLoad(&box_min[0]), atomicLoad(&box_min[1]),
                    atomicLoad(&box_max[0]), atomicLoad(&box_max[1]));
    }
    return workgroupUniformLoad(&box);
}

fn box_fits_tile(node_box: vec4u) -> bool {
    return node_box.z + 3u - node_box.x <= SHARED_TILE && node_box.w + 3u - node_box.y <= SHARED_TILE;
}

fn deposit_to_tile(particle: SortedParticle, node_box: vec4u) {
    let p_mass = particles_mass[particle.p_idx];
    let p_velocity = particles_velocity[particle.p_idx];
    let affine_matrix = particles_affine_matrix[particle.p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = particle.base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(particle.grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let local = node_coords - node_box.xy;
            let tile_idx = local.y * SHARED_TILE + local.x;
            let affine_velocity = p_velocity - affine_matrix * distance;
            let mass = quantize_mass(p_mass * weight);
            if mass == 0u { continue; } // no momentum without mass, see quantize_mass
            let momentum_x = quantize_momentum_component(p_mass * affine_velocity.x * weight * momentum_factor());
            let momentum_y = quantize_momentum_component(p_mass * affine_velocity.y * weight * momentum_factor());
            if mass != 0u { atomicAdd(&tile_mass[tile_idx], mass); }
            if momentum_x != 0 { atomicAdd(&tile_momentum_x[tile_idx], momentum_x); }
            if momentum_y != 0 { atomicAdd(&tile_momentum_y[tile_idx], momentum_y); }
        }
    }
}

fn deposit_z_to_tile(particle: SortedParticle, node_box: vec4u) {
    let p_mass = particles_mass[particle.p_idx];
    let p_velocity_z = particles_velocity_z[particle.p_idx];
    let affine_z = particles_affine_z[particle.p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = particle.base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(particle.grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let local = node_coords - node_box.xy;
            if quantize_mass(p_mass * weight) == 0u { continue; }
            let affine_velocity_z = p_velocity_z - dot(affine_z, distance);
            let momentum_z = quantize_momentum_component(p_mass * affine_velocity_z * weight * momentum_factor());
            if momentum_z != 0 { atomicAdd(&tile_momentum_z[local.y * SHARED_TILE + local.x], momentum_z); }
        }
    }
}

fn flush_tile(lane: u32, node_box: vec4u) {
    for (var tile_idx = lane; tile_idx < SHARED_TILE_NODES; tile_idx += P2G_SORTED_WG_SIZE) {
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
        // atomic_float if mass != 0u { atomicAdd(&grid_mass_atomic[idx], f32(mass) * inv_mass_factor()); }
        // atomic_float if momentum_x != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u], f32(momentum_x) * inv_momentum_factor()); }
        // atomic_float if momentum_y != 0 { atomicAdd(&grid_momentum_atomic[idx * 2u + 1u], f32(momentum_y) * inv_momentum_factor()); }
    }
}

fn flush_tile_z(lane: u32, node_box: vec4u) {
    for (var tile_idx = lane; tile_idx < SHARED_TILE_NODES; tile_idx += P2G_SORTED_WG_SIZE) {
        let node = node_box.xy + vec2u(tile_idx % SHARED_TILE, tile_idx / SHARED_TILE);
        if node.x >= sim_settings.grid_shape.x || node.y >= sim_settings.grid_shape.y {
            continue;
        }
        let momentum_z = atomicLoad(&tile_momentum_z[tile_idx]);
        if momentum_z != 0 { atomicAdd(&grid_momentum_z_atomic[xy_to_idx(node)], momentum_z); } // no_atomic_float
        // atomic_float if momentum_z != 0 { atomicAdd(&grid_momentum_z_atomic[xy_to_idx(node)], f32(momentum_z) * inv_momentum_factor()); }
    }
}

// import transfer_p2g_z.wgsl;
// BEGIN transfer_p2g_z.wgsl
// z row of the 3D APIC transfer, on the stencil of transfer_p2g_with_affine.
// Pure library module: the importing module provides particles_velocity_z,
// particles_affine_z and grid_momentum_z_atomic.
fn transfer_p2g_z(p_idx: u32) {
    let grid_pos = particles_position[p_idx] / sim_settings.cell_size - vec2f(0.5);
    let base_node = vec2u(floor(grid_pos - vec2f(0.5)));
    let p_mass = particles_mass[p_idx];
    let p_velocity_z = particles_velocity_z[p_idx];
    let affine_z = particles_affine_z[p_idx];
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            let node_coords = base_node + vec2u(i, j);
            let distance = calculate_distance_to_node(grid_pos, node_coords);
            let weight = calculate_weight(distance);
            let idx = xy_to_idx(node_coords);
            let affine_velocity_z = p_velocity_z - dot(affine_z, distance);
            if quantize_mass(p_mass * weight) == 0u { continue; } // no_atomic_float
            // atomic_float atomicAdd(&grid_momentum_z_atomic[idx], p_mass * affine_velocity_z * weight);
            let quantized_z = quantize_momentum_component(p_mass * affine_velocity_z * weight * momentum_factor()); // no_atomic_float
            if quantized_z != 0 { atomicAdd(&grid_momentum_z_atomic[idx], quantized_z); } // no_atomic_float
        }
    }
}

// END transfer_p2g_z.wgsl
// import transfer_p2g.wgsl;
// BEGIN transfer_p2g.wgsl
fn quantize_momentum_component(value: f32) -> i32 {
    // Keep conversion inside i32 range before atomicAdd.
    let i32_max_f = 2147483647.0;
    let clamped = clamp(value, -i32_max_f, i32_max_f);
    return i32(round(clamped));
}

// Quantized mass contribution of the integer grid (mass_factor() in
// utils.wgsl). A contribution whose mass rounds to zero must not deposit
// momentum either, or it would leave momentum without mass and an unbounded
// node velocity. With the range-based factors that needs a contribution faster
// than 2 * MAX_FLOW_VELOCITY, so the guard only protects against misuse.
fn quantize_mass(value: f32) -> u32 {
    return u32(round(value * mass_factor()));
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

            let quantized_mass = quantize_mass(p_mass * weight); // no_atomic_float
            if quantized_mass == 0u { continue; } // no_atomic_float
            atomicAdd(&grid_mass_atomic[idx], quantized_mass); // no_atomic_float
            let momentum_x = p_mass * affine_velocity.x * weight * momentum_factor(); // no_atomic_float
            let momentum_y = p_mass * affine_velocity.y * weight * momentum_factor(); // no_atomic_float
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
const H_FACTOR: f32 = 1e6;
const INV_MAX_VELOCITY_FACTOR: f32 = 1 / MAX_VELOCITY_FACTOR; // u32 limit is 430 m/s
const INV_H_FACTOR: f32 = 1 / H_FACTOR;
// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The factors fill the integer range with the largest node values a
// flow can reach: a MAX_FLOW_THICKNESS deep column over a cell
// (density * thickness * cell_size^2) moving at MAX_FLOW_VELOCITY, times
// QUANTIZATION_SAFETY (which also covers the slope factor of the node mass).
// They scale with the cell size like the particle masses, so a particle of a
// release h_r thick with N particles per cell holds about
// U32_RANGE * h_r / (N * QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS) mass units
// (2e6 for 1 m and 64 per cell) at every cell size. Must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 30.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
const QUANTIZATION_SAFETY: f32 = 2.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
fn max_node_mass() -> f32 {
    return QUANTIZATION_SAFETY * MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / (max_node_mass() * MAX_FLOW_VELOCITY);
}
fn inv_momentum_factor() -> f32 {
    return max_node_mass() * MAX_FLOW_VELOCITY / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (15 g for 5 m cells and 200 kg/m3). Lighter nodes
// count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// depth-integrated internal force (h * sigma * grad_w * area, ~1e4..1e5 N per node contribution)
const FORCE_FACTOR: f32 = 1e-3; // i32 limit is 2.1e6 N per node
const INV_FORCE_FACTOR: f32 = 1 / FORCE_FACTOR;
// weak-form pressure sum m_p h_p / 2 grad_w (kg) of the curvilinear model,
// node sum = node_mass * |grad h|: 0.001 kg resolves thin front nodes
//   (rho*cell^2*h*J ~ 6e4 kg * 1 * 1e3 = 6e7 < 2.1e9)
const PRESSURE_FACTOR: f32 = 1e3;
const INV_PRESSURE_FACTOR: f32 = 1 / PRESSURE_FACTOR;

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
    travel_length_2d: f32,
    travel_length_3d: f32,
    travel_vertical_drop: f32,
    travel_angle: f32,
    _first_center_of_mass_timestep: u32,
    _last_center_of_mass_timestep: u32,
    _pad: u32,
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
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

// reversible low-speed state; the low bits hold the rest start step while resting
// and the consecutive slow-step count while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
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
    // total grid momentum (kg m/s) of the current step as a 64-bit sum, reset by update_sim_info
    momentum_lo: atomic<u32>,
    momentum_hi: atomic<u32>,
    // f32 bits of the largest total momentum so far
    peak_momentum: atomic<u32>,
};

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_RESTING - 1u;
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

struct G2PUpdate {
    velocity: vec2f,
    affine_matrix: mat2x2<f32>,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// length in steps of the no-new-cells rolling window: the fastest material
// moves at most cfl cells per step, so round(travel / cfl) steps cover the same
// front travel at any cfl; at least 10 steps
fn no_new_cells_window() -> u32 {
    return max(10u, u32(round(sim_settings.no_new_cells_travel / sim_settings.cfl)));
}

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
