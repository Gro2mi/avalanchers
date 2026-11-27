// Evaluates the simulation against the region of interest: classifies every
// cell as intersection (region of interest AND simulated), undershoot (region
// of interest only) or overshoot (simulated only) and accumulates the three
// counts atomically into the unified evaluation result buffer. The CPU
// combines the counts into the final MassMovementEvaluation.
// Release cells are skipped: they lie in both masks by construction, so
// counting them would inflate the overlap with the release area fraction.
// A cell is simulated when it passes the thresholds of evaluation_params
// (see evaluation_mask.wgsl); the peak fields are only read.

struct EvaluationCounts {
    intersection: atomic<u32>,
    undershoot: atomic<u32>,
    overshoot: atomic<u32>,
    _padding: atomic<u32>,
}

@group(0) @binding(1) var<storage, read> region_of_interest: array<u32>;
@group(0) @binding(2) var<storage, read> grid_peak_flow_thickness: array<f32>;
@group(0) @binding(3) var<storage, read_write> counts: EvaluationCounts;
@group(0) @binding(4) var<storage, read> release_areas: array<f32>;
@group(0) @binding(5) var<uniform> evaluation_params: EvaluationParams;
@group(0) @binding(6) var<storage, read> grid_peak_velocity: array<f32>;

// must match RELEASE_THICKNESS_THRESHOLD in lib.rs and initialize_particles.wgsl
const RELEASE_THICKNESS_THRESHOLD: f32 = 0.01;

fn bit_is_set(word: u32, index: u32) -> bool {
    return (word & (1u << (index % 32u))) != 0u;
}

@compute @workgroup_size(16, 16, 1)
fn evaluate_mass_movement(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }

    let index = id.y * sim_settings.grid_shape.x + id.x;
    if release_areas[index] > RELEASE_THICKNESS_THRESHOLD {
        return;
    }
    let word_index = index / 32u;
    let reference_affected = bit_is_set(region_of_interest[word_index], index);
    let simulated_affected = is_simulated_cell(index);

    if reference_affected && simulated_affected {
        atomicAdd(&counts.intersection, 1u);
    } else if reference_affected {
        atomicAdd(&counts.undershoot, 1u);
    } else if simulated_affected {
        atomicAdd(&counts.overshoot, 1u);
    }
}

// import evaluation_mask.wgsl;
// BEGIN evaluation_mask.wgsl
// Simulated-avalanche mask shared by the evaluation shaders. A cell belongs to
// the simulated avalanche when its peak flow thickness reaches
// evaluation_params.min_peak_flow_thickness and its peak velocity reaches
// evaluation_params.min_peak_velocity. The peak fields themselves are never
// modified, so a stored run can be re-evaluated with other thresholds.
//
// The importing shader declares the evaluation_params uniform
// (EvaluationParams) and the grid_peak_flow_thickness / grid_peak_velocity
// storage buffers at its own binding slots.

struct EvaluationParams {
    min_peak_flow_thickness: f32,
    min_peak_velocity: f32,
    _padding: vec2<f32>,
}

fn is_simulated_cell(idx: u32) -> bool {
    return grid_peak_flow_thickness[idx] >= evaluation_params.min_peak_flow_thickness
        && grid_peak_velocity[idx] >= evaluation_params.min_peak_velocity;
}

// END evaluation_mask.wgsl
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
