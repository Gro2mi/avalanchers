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
@group(0) @binding(8) var dem_texture: texture_2d<f32>;
@group(0) @binding(9) var tex_sampler: sampler;
// start elevation of each particle, set by initialize_particles at the seed position
@group(0) @binding(10) var<storage, read_write> start_elevation: array<f32>;

const LINKED_LIST_END: u32 = 0xFFFFFFFFu;
// must match RELEASE_THICKNESS_THRESHOLD in lib.rs and initialize_particles.wgsl
const RELEASE_THICKNESS_THRESHOLD: f32 = 0.01;

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
    return release_areas[xy_to_idx(vec2u(cellf))] > RELEASE_THICKNESS_THRESHOLD;
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

    // a step leaving the release area is reduced to one axis, larger component
    // first, so particles pushed against the edge still slide along it
    let step_x = vec2f(displacement.x, 0.0);
    let step_y = vec2f(0.0, displacement.y);
    let primary = select(step_y, step_x, abs(displacement.x) >= abs(displacement.y));
    let secondary = select(step_x, step_y, abs(displacement.x) >= abs(displacement.y));
    if (is_release_position(position + displacement)) {
        particles_position[index] = position + displacement;
    } else if (is_release_position(position + primary)) {
        particles_position[index] = position + primary;
    } else if (is_release_position(position + secondary)) {
        particles_position[index] = position + secondary;
    } else {
        // no admissible step: stay at the previous position
        particles_position[index] = position;
        particles_velocity[index] = vec2f(0.0);
    }
}

// run once after the last iteration: the particles moved away from their seed
// position, so the start elevation is re-sampled at the relaxed position
@compute @workgroup_size(WG_SIZE_1D, 1, 1)
fn relax_sample_elevation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= relax_params.num_particles) {
        return;
    }
    start_elevation[index] = textureSampleLevel(dem_texture, tex_sampler, position_to_uv(particles_position[index]), 0).x;
}

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
    // forest model (forest.wgsl), read only with SIM_FLAG_FOREST
    forest_drag_coefficient: f32,
    forest_trunk_diameter: f32,
    forest_crown_diameter: f32,
    forest_tree_height: f32,
    forest_modulus_of_rupture: f32,
    forest_decay_coefficient: f32,
    forest_release_max_cover: f32,
    // entrainment (entrainment.wgsl), read only with SIM_FLAG_ENTRAINMENT by
    // the DAMPM global model: erosion energy e_b (J/kg), density (kg/m3) and
    // uniform depth (m) of the erodible snow cover
    erosion_energy: f32,
    entrainment_density: f32,
    entrainment_thickness: f32,
    // entrainment rate law (EntrainmentModel): 0 erosion energy, 1 excess
    // shear stress, 2 velocity-proportional, with their parameters tau_c (Pa),
    // k (s/m) and kappa (-)
    entrainment_model: u32,
    critical_shear_stress: f32,
    shear_erosion_coefficient: f32,
    entrainment_velocity_coefficient: f32,
};

@group(0) @binding(0) var<uniform> sim_settings: SimSettings;

// SimFlags::GRID_OCCUPIED (settings.rs): p2g marks the cells holding a
// depositing particle in grid_occupied and only those record peak maps
const SIM_FLAG_GRID_OCCUPIED: u32 = 1u << 9u;

// SimFlags::FOREST (settings.rs): a forest cover is loaded; the DAMPM global
// model decelerates the flow by the drag of the standing stems
const SIM_FLAG_FOREST: u32 = 1u << 10u;

fn forest_enabled() -> bool {
    return (sim_settings.flags & SIM_FLAG_FOREST) != 0u;
}

// SimFlags::ENTRAINMENT (settings.rs): the DAMPM global model entrains the
// erodible snow cover (entrainment.wgsl)
const SIM_FLAG_ENTRAINMENT: u32 = 1u << 3u;

fn entrainment_enabled() -> bool {
    return (sim_settings.flags & SIM_FLAG_ENTRAINMENT) != 0u;
}

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
