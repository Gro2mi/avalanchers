// Reference surface of the curvilinear, terrain-following and 3D models: the
// DEM smoothed with a Gaussian of width reference_surface_length, separated
// into a row pass (DEM -> reference_rows) and a column pass (reference_rows ->
// dem_reference_texture). Same result as dem::smooth_gaussian on the CPU:
// non-finite cells stay as they are and are left out of their neighbours'
// sums (weights renormalized) and lines are extended by point reflection about
// their end samples. Only dispatched when gully_walls_enabled().
@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> reference_rows: array<f32>;
@group(0) @binding(3) var dem_reference_texture: texture_storage_2d<r32float, write>;

fn reference_sigma_cells() -> f32 {
    return sim_settings.reference_surface_length / sim_settings.cell_size;
}

fn reference_radius(sigma: f32, length: u32) -> i32 {
    return i32(min(u32(ceil(3.0 * sigma)), length - 1u));
}

fn dem_at(x: i32, y: i32) -> f32 {
    return textureLoad(dem_texture, vec2<i32>(x, y), 0).r;
}

fn rows_at(x: i32, y: i32) -> f32 {
    return reference_rows[u32(y) * sim_settings.grid_shape.x + u32(x)];
}

// sample i of a line of `length` samples, point-reflected beyond its ends
fn reflected_index(i: i32, length: i32) -> vec2<i32> {
    // x = the sample read, y = the end sample it is reflected about (or -1)
    if i < 0 {
        return vec2<i32>(-i, 0);
    }
    if i >= length {
        return vec2<i32>(2 * (length - 1) - i, length - 1);
    }
    return vec2<i32>(i, -1);
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn smooth_reference_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let center = dem_at(x, y);
    let sigma = reference_sigma_cells();
    let length = i32(sim_settings.grid_shape.x);
    // !(sigma > 0) also covers NaN
    if !(sigma > 0.0) || length < 2 || !is_finite(center) {
        reference_rows[id.y * sim_settings.grid_shape.x + id.x] = center;
        return;
    }
    let radius = reference_radius(sigma, sim_settings.grid_shape.x);
    // deviations from the center keep f32 precise at large elevations
    var sum = 0.0;
    var weight = 0.0;
    for (var k = -radius; k <= radius; k++) {
        let index = reflected_index(x + k, length);
        var value = dem_at(index.x, y);
        if index.y >= 0 {
            value = 2.0 * dem_at(index.y, y) - value;
        }
        if is_finite(value) {
            let r = f32(k) / sigma;
            let w = exp(-0.5 * r * r);
            sum += w * (value - center);
            weight += w;
        }
    }
    reference_rows[id.y * sim_settings.grid_shape.x + id.x] = center + sum / weight;
}

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn smooth_reference_columns(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let center = rows_at(x, y);
    let sigma = reference_sigma_cells();
    let length = i32(sim_settings.grid_shape.y);
    var smoothed = center;
    if sigma > 0.0 && length >= 2 && is_finite(center) {
        let radius = reference_radius(sigma, sim_settings.grid_shape.y);
        var sum = 0.0;
        var weight = 0.0;
        for (var k = -radius; k <= radius; k++) {
            let index = reflected_index(y + k, length);
            var value = rows_at(x, index.x);
            if index.y >= 0 {
                value = 2.0 * rows_at(x, index.y) - value;
            }
            if is_finite(value) {
                let r = f32(k) / sigma;
                let w = exp(-0.5 * r * r);
                sum += w * (value - center);
                weight += w;
            }
        }
        smoothed = center + sum / weight;
    }
    textureStore(dem_reference_texture, vec2<i32>(x, y), vec4<f32>(smoothed, 0.0, 0.0, 0.0));
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
// import float_checks.wgsl;
// BEGIN float_checks.wgsl
// NaN / infinity checks on the bit pattern (comparisons may be optimized away).

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

// END float_checks.wgsl
