@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var normals_texture: texture_2d<f32>;
@group(0) @binding(3) var curvature_texture: texture_2d<f32>;
@group(0) @binding(4) var<storage> grid_mass_atomic: array<u32>; // no_atomic_float
// atomic_float @group(0) @binding(4) var<storage> grid_mass_atomic: array<f32>;
@group(0) @binding(5) var<storage> grid_momentum_atomic: array<i32>; // no_atomic_float
// atomic_float @group(0) @binding(5) var<storage> grid_momentum_atomic: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_h_buffer: array<vec2f>;
@group(0) @binding(7) var<storage, read_write> peak_flow_thickness: array<f32>;
@group(0) @binding(8) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(9) var<storage, read_write> new_cells_rolling_window: array<u32>;
@group(0) @binding(10) var<storage, read_write> velocity_buffer: array<vec2f>;

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    if (sim_info.flags & SIM_INFO_STOPPED) != 0u {
        return;
    }

    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    if !use_particle_interaction {
        new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] = new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] + 1u; // update new cell count for diagnostics
        return;
    }
    let idx = xy_to_idx(id.xy);
    record_quantization_fill(grid_mass_atomic[idx], vec3i(grid_momentum_atomic[idx * 2u], grid_momentum_atomic[idx * 2u + 1u], 0)); // no_atomic_float
    let n = textureLoad(normals_texture, id.xy, 0);
    if !is_finite(n.z) {
        grad_h_buffer[idx] = vec2f(0.0, 0.0);
        return;
    }

    // 1. Decode height and velocity[cite: 3]
    let mass = f32(grid_mass_atomic[idx]) * inv_mass_factor(); // no_atomic_float 
    let u = f32(grid_momentum_atomic[idx * 2]) * inv_momentum_factor() / (mass + 1e-6); // no_atomic_float 
    let v = f32(grid_momentum_atomic[idx * 2 + 1]) * inv_momentum_factor() / (mass + 1e-6); // no_atomic_float 
    // atomic_float let mass = grid_mass_atomic[idx];
    // atomic_float let u = grid_momentum_atomic[idx * 2] / (mass + 1e-6);
    // atomic_float let v = grid_momentum_atomic[idx * 2 + 1] / (mass + 1e-6);
    velocity_buffer[idx] = vec2f(u, v);
    let h = mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size) * n.z;
    // new max value
    if h > sim_settings.peak_flow_thickness_threshold && h > peak_flow_thickness[idx] {
        // new cell reached
        new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] = new_cells_rolling_window[sim_info.timestep % no_new_cells_window()] + 1u; // update new cell count for diagnostics
        
        peak_flow_thickness[idx] = h;
        if is_finite(h) {
            atomicMax(&atomic_values.peak_flow_thickness, bitcast<u32>(h));
        }
    }
    if mass < 1e-6 {
        grad_h_buffer[idx] = vec2f(0.0, 0.0);
        return;
    }

    // 2. Compute Divergence for Active/Passive state[cite: 3]
    // TODO calculate divergence and earth pressure coefficient
    // let div_u = (get_u(id.x + 1, id.y) - get_u(id.x - 1, id.y)) / (2.0 * dx);
    var k = 1f;

    let use_earth_pressure_coefficient: bool = (sim_settings.flags & (1u << 2u)) != 0u;
    if use_earth_pressure_coefficient {
        let div_u = div_u(id.x, id.y);
        k = earth_pressure_coefficient(radians(sim_settings.internal_friction_angle), radians(sim_settings.basal_friction_angle), div_u);
    } else {
        k = 1.0;
    }
    // let k = 1.0;
    // 3. Lateral Pressure Force[cite: 3]
    // Force = -0.5 * g * cos(theta) * k * gradient(h^2)
    // TODO do I need to apply a filter to the height field to prevent noise in the gradient?
    // e. g. h_ij = (1-alpha)h_ij + alpha/4 * (h_ij-1 + h_i+1j + h_ij+1 + h_i-1j) to smooth the height field and prevent noise in the gradient?
    // TODO do i need a cutoff for small h to prevent noise in the gradient? if h < h_min -> h = 0
    // let grad_h2 = select(vec2f(
    //     // TODO account for slope in x and y direction. multiply by cos_theta_x
    //     (get_h2(id.x + 1, id.y) - get_h2(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size),
    //     (get_h2(id.x, id.y + 1) - get_h2(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size)
    // ), vec2f(0.0, 0.0), h < 1e-5);
    if h > 1e-3 {
        if use_earth_pressure_coefficient {
            let div_u = div_u(id.x, id.y);
            k = earth_pressure_coefficient(radians(sim_settings.internal_friction_angle), radians(sim_settings.basal_friction_angle), div_u);
        } else {
            k = 1.0;
        }
        let grad_h2 = vec2f(
        // TODO account for slope in x and y direction. multiply by cos_theta_x
            (get_h2(id.x + 1, id.y) - get_h2(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size),
            (get_h2(id.x, id.y + 1) - get_h2(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size)
        );
        let grad_h = grad_h2 * 0.5 / h;

        // TODO do i need a slope limiter like minmod?
        // 1. Fetch the center value once to save redundant lookups
        // let h2_c = get_h2(id.x, id.y);

        // // 2. Fetch the 4 neighbors
        // let h2_r = get_h2(id.x + 1, id.y);
        // let h2_l = get_h2(id.x - 1, id.y);
        // let h2_t = get_h2(id.x, id.y + 1);
        // let h2_b = get_h2(id.x, id.y - 1);

        // // 3. Calculate forward and backward differences in X
        // let dx_fwd = (h2_r - h2_c) / sim_settings.cell_size;
        // let dx_bwd = (h2_c - h2_l) / sim_settings.cell_size;

        // // 4. Calculate forward and backward differences in Y
        // let dy_fwd = (h2_t - h2_c) / sim_settings.cell_size;
        // let dy_bwd = (h2_c - h2_b) / sim_settings.cell_size;

        // // 5. Apply the Minmod limiter to get the stable gradient
        // let grad_h2 = vec2f(
        //     minmod(dx_fwd, dx_bwd),
        //     minmod(dy_fwd, dy_bwd)
        // );

        // 6. Recover grad_h (Chain rule: grad(h^2) = 2 * h * grad(h))
        // let grad_h = grad_h2 * 0.5 / (h + 1e-5);
        // correct for slope sqrt(1-nx²), and again sqrt(1-nx²) to rotate it into 3d coordinates
        // let slope_corrected_grad_h2 = grad_h2 * vec2f(sqrt(1.0 - n.x * n.x), sqrt(1.0 - n.y * n.y));
        let slope_corrected_grad_h = grad_h * vec2f(sqrt(1.0 - n.x * n.x), sqrt(1.0 - n.y * n.y));
        let lateral_factor = select(vec2f(0.0), slope_corrected_grad_h, length(slope_corrected_grad_h) > tan(radians(sim_settings.internal_friction_angle)));
        // k = 1;
        // grad_h_buffer[idx] = -g * n.z * k * slope_corrected_grad_h;
        grad_h_buffer[idx] = -k * slope_corrected_grad_h;
    } else {
        grad_h_buffer[idx] = vec2f(0.0, 0.0);
    }
}

fn minmod(a: f32, b: f32) -> f32 {
    // Returns the smallest magnitude slope if signs match, otherwise 0.0
    return 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b));
}

fn get_h2(x: u32, y: u32) -> f32 {
    let idx = x_y_to_idx(x, y);
    // atomic_float return pow(grid_mass_atomic[idx] / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0);
    return pow(f32(grid_mass_atomic[idx]) * inv_mass_factor() / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0); // no_atomic_float 
}

fn get_h(x: u32, y: u32) -> f32 {
    let idx = x_y_to_idx(x, y);
    // atomic_float return grid_mass_atomic[idx] / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size);
    return f32(grid_mass_atomic[idx]) * inv_mass_factor() / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size); // no_atomic_float 
}

// Velocity of node (x, y) from the node's own mass and momentum; z = 1 when
// the node holds mass, 0 below the p2g mass floor where it is meaningless.
fn node_velocity(x: u32, y: u32) -> vec3f {
    let idx = x_y_to_idx(x, y);
    let mass = f32(grid_mass_atomic[idx]) * inv_mass_factor(); // no_atomic_float
    // atomic_float let mass = grid_mass_atomic[idx];
    let node_is_empty = mass < min_node_mass(); // no_atomic_float
    // atomic_float let node_is_empty = mass < 1e-2;
    if node_is_empty {
        return vec3f(0.0);
    }
    let q = vec2f(
        f32(grid_momentum_atomic[idx * 2]) * inv_momentum_factor(), // no_atomic_float
        f32(grid_momentum_atomic[idx * 2 + 1]) * inv_momentum_factor() // no_atomic_float
        // atomic_float grid_momentum_atomic[idx * 2],
        // atomic_float grid_momentum_atomic[idx * 2 + 1]
    ) / mass;
    return vec3f(q, 1.0);
}

// Derivative of one velocity component: central where both neighbours hold
// mass, one-sided against the centre at the flow edge, zero for an isolated
// node. Empty nodes carry no velocity, so a spreading front is not mistaken
// for a compressed one.
fn edge_aware_difference(low: vec3f, centre: f32, high: vec3f, component: u32) -> f32 {
    let dx = sim_settings.cell_size;
    if low.z > 0.0 && high.z > 0.0 {
        return (high[component] - low[component]) / (2.0 * dx);
    }
    if high.z > 0.0 {
        return (high[component] - centre) / dx;
    }
    if low.z > 0.0 {
        return (centre - low[component]) / dx;
    }
    return 0.0;
}

fn div_u(x: u32, y: u32) -> f32 {
    let centre = node_velocity(x, y);
    let dudx = edge_aware_difference(node_velocity(x - 1u, y), centre.x, node_velocity(x + 1u, y), 0u);
    let dvdy = edge_aware_difference(node_velocity(x, y - 1u), centre.y, node_velocity(x, y + 1u), 1u);
    return dudx + dvdy;
}

fn earth_pressure_coefficient(
    phi: f32,
    delta: f32,
    div_u: f32
) -> f32 {

    let cos_phi = cos(phi);
    let cos_delta = cos(delta);

    let inside = 1.0 -
        (cos_phi * cos_phi) /
        (cos_delta * cos_delta);

    // numerical safety
    let root = sqrt(max(inside, 0.0));

    let sec_phi2 = 1.0 / (cos_phi * cos_phi);

    // active for expansion
    let Ka = 2.0 * (1.0 - root) * sec_phi2 - 1.0;

    // passive for compression
    let Kp = 2.0 * (1.0 + root) * sec_phi2 - 1.0;

    return select(Kp, Ka, div_u > 0.0);
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
// import sim_info.wgsl;
// BEGIN sim_info.wgsl
// Simulation state shared between the passes: the SimInfo buffer with its
// status flags and the atomic accumulators. Must mirror the Rust layouts.

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
    // ranges of the integer-atomic grid in units of the reference node (see
    // quantization.wgsl); previous_mass_scale quantized the last step's grid
    mass_scale: f32,
    momentum_scale: f32,
    previous_mass_scale: f32,
    _pad0: u32,
    _pad1: u32,
};

const SIM_INFO_OUT_OF_BOUNDS: u32 = 1u << 0u;
const SIM_INFO_CFL_EXCEEDED: u32 = 1u << 1u;
const SIM_INFO_IS_NAN: u32 = 1u << 2u;
const SIM_INFO_PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 3u;
// a particle exceeded MAX_PARTICLE_SPEED and was stopped
const SIM_INFO_UNPHYSICAL_SPEED: u32 = 1u << 4u;
// a node filled more than QUANTIZATION_OVERFLOW_FILL of an integer-atomic
// grid range, which it may have overflowed (see quantization.wgsl)
const SIM_INFO_QUANTIZATION_OVERFLOW: u32 = 1u << 5u;
const SIM_INFO_STOPPED: u32 = 1u << 31u;
const SIM_INFO_ALL_PARTICLES_STOPPED: u32 = 1u << 30u;
const SIM_INFO_NO_NEW_CELLS: u32 = 1u << 29u;
// almost all particles are resting or terminally stopped
const SIM_INFO_AT_REST: u32 = 1u << 28u;
// total momentum fell below momentum_stop_fraction of its peak
const SIM_INFO_LOW_MOMENTUM: u32 = 1u << 27u;

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
    // largest node mass (u32 units) and momentum component (|i32| units) of
    // the integer-atomic grid this step, read and reset by reset_grid
    max_mass_units: atomic<u32>,
    max_momentum_units: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

// END sim_info.wgsl
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

// import quantization.wgsl;
// BEGIN quantization.wgsl
// Integer-atomic grid (no_atomic_float): node masses and momenta are fixed
// point numbers in u32 / i32, quantized per particle->node contribution before
// atomicAdd. The ranges adapt to the flow. reset_grid doubles a range once the
// fullest node of the last step filled half of it, and halves it once that
// node stayed below an eighth, so the integers resolve thin flows finely and
// still hold deep deposits. sim_info.mass_scale counts the mass range in
// reference nodes: a MAX_FLOW_THICKNESS deep column over a cell (density *
// thickness * cell_size^2); sim_info.momentum_scale counts the momentum range
// in that column moving at MAX_FLOW_VELOCITY. Both start at 1. Needs
// sim_settings and sim_info; must match settings.rs.
const MAX_FLOW_THICKNESS: f32 = 50.0;
const MAX_FLOW_VELOCITY: f32 = 100.0;
// below 2^32 and 2^31, so a rounded maximum still converts without saturating
const QUANTIZATION_U32_RANGE: f32 = 4.0e9;
const QUANTIZATION_I32_RANGE: f32 = 2.0e9;
// fill of the fullest node at which a range doubles, and below which it halves
const QUANTIZATION_GROW_FILL: f32 = 0.5;
const QUANTIZATION_SHRINK_FILL: f32 = 0.125;
// fill at which the fullest node probably wrapped around
const QUANTIZATION_OVERFLOW_FILL: f32 = 0.9;
const MIN_MASS_SCALE: f32 = 1.0 / 256.0;
const MAX_QUANTIZATION_SCALE: f32 = 256.0;
// Smallest momentum range relative to the mass range: the full mass range at
// MAX_FLOW_VELOCITY / 8 = 12.5 m/s. A flow starting from rest (no momentum to
// measure) cannot outgrow it within one step.
const MIN_MOMENTUM_PER_MASS_SCALE: f32 = 0.125;

// mass (kg) of the reference node, a MAX_FLOW_THICKNESS deep column
fn reference_node_mass() -> f32 {
    return MAX_FLOW_THICKNESS * sim_settings.snow_density
        * sim_settings.cell_size * sim_settings.cell_size;
}
// largest node mass (kg) and momentum component (kg m/s) the integers hold
fn max_node_mass() -> f32 {
    return sim_info.mass_scale * reference_node_mass();
}
fn max_node_momentum() -> f32 {
    return sim_info.momentum_scale * reference_node_mass() * MAX_FLOW_VELOCITY;
}
// integer units per kg
fn mass_factor() -> f32 {
    return QUANTIZATION_U32_RANGE / max_node_mass();
}
fn inv_mass_factor() -> f32 {
    return max_node_mass() / QUANTIZATION_U32_RANGE;
}
// kg per unit of the last step's grid mass (MPMDAC's grid_mass_previous)
fn previous_inv_mass_factor() -> f32 {
    return sim_info.previous_mass_scale * reference_node_mass() / QUANTIZATION_U32_RANGE;
}
// integer units per kg m/s
fn momentum_factor() -> f32 {
    return QUANTIZATION_I32_RANGE / max_node_momentum();
}
fn inv_momentum_factor() -> f32 {
    return max_node_momentum() / QUANTIZATION_I32_RANGE;
}
// Lightest node whose velocity momentum / mass is resolved: one momentum unit
// changes it by 1 m/s (12.5 g for 5 m cells and 200 kg/m3 at the start
// range). Lighter nodes count as empty.
fn min_node_mass() -> f32 {
    return inv_momentum_factor(); // momentum unit / (1 m/s)
}
// next range scale for a fullest node filling `fill` of the current range
fn adapted_quantization_scale(scale: f32, fill: f32) -> f32 {
    if fill >= QUANTIZATION_GROW_FILL {
        return scale * 2.0;
    }
    if fill < QUANTIZATION_SHRINK_FILL {
        return scale * 0.5;
    }
    return scale;
}

// END quantization.wgsl

// import quantization_fill.wgsl;
// BEGIN quantization_fill.wgsl
// Records how full the integer-atomic grid ranges are (quantization.wgsl):
// the largest node mass (u32 units) and momentum component (|i32| units) of
// this step, which reset_grid reads at the start of the next one. Needs
// atomic_values; called from the grid physics on the integer grid only.
fn record_quantization_fill(mass_units: u32, momentum_units: vec3<i32>) {
    if mass_units > atomicLoad(&atomic_values.max_mass_units) {
        atomicMax(&atomic_values.max_mass_units, mass_units);
    }
    // i32 -> u32 keeps the bits, so abs(i32 min) still reads as 2^31
    let momentum = u32(max(max(abs(momentum_units.x), abs(momentum_units.y)), abs(momentum_units.z)));
    if momentum > atomicLoad(&atomic_values.max_momentum_units) {
        atomicMax(&atomic_values.max_momentum_units, momentum);
    }
}

// END quantization_fill.wgsl
