struct RelaxParams {
    num_particles: u32,
    // fraction of the overlap corrected per iteration
    move_factor: f32,
    // per-iteration displacement limit as a fraction of the target spacing
    max_step_fraction: f32,
    _padding: u32,
};

@group(0) @binding(1) var<uniform> relax_params: RelaxParams;
// written by the relax kernel; the previous iteration's positions are read from
// the snapshot so every thread sees a consistent state
@group(0) @binding(2) var<storage, read_write> particles_position: array<vec2<f32>>;
@group(0) @binding(3) var<storage> particles_position_snapshot: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> particles_velocity: array<vec2<f32>>;
@group(0) @binding(5) var<storage> release_areas: array<f32>;
// fixed hash grid over the simulation cells, chained through particle_next:
// grid_head[cell] is the index of the first particle in the cell or
// LINKED_LIST_END; lets each particle find its neighbors in O(cell occupancy)
// instead of scanning all particles
@group(0) @binding(6) var<storage, read_write> grid_head: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> particle_next: array<u32>;

const LINKED_LIST_END: u32 = 0xFFFFFFFFu;

override WG_SIZE_1D: u32 = 1u;

fn clamp_to_grid(cellf: vec2f) -> vec2u {
    let max_cell = vec2f(sim_settings.grid_shape) - 1.0;
    return vec2u(clamp(cellf, vec2f(0.0), max_cell));
}

// true if p lies inside a cell with snow (a release cell)
fn is_release_position(p: vec2f) -> bool {
    let cellf = floor(p / sim_settings.cell_size);
    if (cellf.x < 0.0 || cellf.y < 0.0
        || cellf.x >= f32(sim_settings.grid_shape.x)
        || cellf.y >= f32(sim_settings.grid_shape.y)) {
        return false;
    }
    return release_areas[xy_to_idx(vec2u(cellf))] > 0.01;
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn relax_clear_grid(@builtin(global_invocation_id) cell: vec3<u32>) {
    if cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y {
        return;
    }
    atomicStore(&grid_head[xy_to_idx(cell.xy)], LINKED_LIST_END);
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }
    let cell = clamp_to_grid(floor(particles_position_snapshot[index] / sim_settings.cell_size));
    // lock-free push onto the cell's linked list
    particle_next[index] = atomicExchange(&grid_head[xy_to_idx(cell)], index);
}

@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }

    // hexagonal packing: each particle owns A_particle = A_cell / N =
    // sqrt(3)/2 * h^2, so the equilibrium center spacing is h; h <= cell_size
    // for N >= 2, so the 3x3 cell neighborhood always contains every particle
    // within the target spacing
    let particles_per_cell = max(sim_settings.released_particles_per_cell, 1u);
    let target_spacing = sim_settings.cell_size
        * sqrt(2.0 / (sqrt(3.0) * f32(particles_per_cell)));

    let position = particles_position_snapshot[index];
    let base_cellf = floor(position / sim_settings.cell_size);
    var displacement = vec2f(0.0);

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cellf = base_cellf + vec2f(f32(dx), f32(dy));
            // skip cells outside the grid instead of clamping, so edge cells
            // are not visited (and their particles not repelled) multiple times
            if (cellf.x < 0.0 || cellf.y < 0.0
                || cellf.x >= f32(sim_settings.grid_shape.x)
                || cellf.y >= f32(sim_settings.grid_shape.y)) {
                continue;
            }
            var j = atomicLoad(&grid_head[xy_to_idx(vec2u(cellf))]);
            loop {
                if (j == LINKED_LIST_END) {
                    break;
                }
                if (j != index) {
                    let diff = position - particles_position_snapshot[j];
                    let dist = length(diff);
                    // repulsion if closer than the desired equilibrium spacing
                    if (dist < target_spacing && dist > 1e-4) {
                        let overlap = target_spacing - dist;
                        displacement += diff / dist * (relax_params.move_factor * overlap);
                    }
                }
                j = particle_next[j];
            }
        }
    }

    // clamp the per-iteration displacement for stability
    let max_step = relax_params.max_step_fraction * target_spacing;
    let displacement_len = length(displacement);
    if (displacement_len > max_step) {
        displacement = displacement / displacement_len * max_step;
    }

    let relaxed = position + displacement;
    if (is_release_position(relaxed)) {
        particles_position[index] = relaxed;
    } else {
        // hard clamp back to the last valid position
        particles_position[index] = position;
        particles_velocity[index] = vec2f(0.0);
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
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;

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
