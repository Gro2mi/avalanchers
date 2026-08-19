// @atlas: Grid solve: recovers flow thickness, earth-pressure coefficient from velocity divergence, lateral pressure force ∝ ∇(h²). Tracks newly-conquered cells for the stop criterion.
@group(0) @binding(1) var<storage> grid_mass_atomic: array<u32>;
@group(0) @binding(2) var normals_texture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> grid_forces: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> peak_flow_thickness: array<f32>;
@group(0) @binding(5) var<storage, read_write> atomic_values: AtomicValues;
@group(0) @binding(6) var<storage, read_write> grid_momentum_atomic: array<i32>; // Combined u, v
@group(0) @binding(7) var curvature_texture: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> new_cells_rolling_window: array<u32>;
@group(0) @binding(9) var<storage, read_write> sim_info: SimInfo;


@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D, 1)
fn grid_physics(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= sim_settings.grid_shape.x || id.y >= sim_settings.grid_shape.y {
        return;
    }
    if sim_info.flags >= SIM_INFO_STOPPED {
        return;
    }
    
    let use_particle_interaction: bool = (sim_settings.flags & (1u << 1u)) != 0u;
    if !use_particle_interaction {
        new_cells_rolling_window[sim_info.timestep % 40u] = new_cells_rolling_window[sim_info.timestep % 40u] + 1u; // update new cell count for diagnostics
        return;
    }
    let idx = xy_to_idx(id.x, id.y);
    let n = textureLoad(normals_texture, id.xy, 0);

    // 1. Decode height and velocity[cite: 3]
    let mass = f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR;
    let h = mass / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size) * n.z;
    let u = f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
    let v = f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR / (mass + 1e-6);
    // peak_flow_thickness[idx] = max(peak_flow_thickness[idx], h);
    if peak_flow_thickness[idx] < h {
        if peak_flow_thickness[idx] < 1e-5 {
            new_cells_rolling_window[sim_info.timestep % 40u] = new_cells_rolling_window[sim_info.timestep % 40u] + 1u; // update new cell count for diagnostics
        }
        peak_flow_thickness[idx] = h;
    }
    // atomicAdd(&atomic_values.alpha, 1u);
    atomicMax(&atomic_values.peak_flow_thickness, u32(h * H_FACTOR)); // update peak flow thickness for cfl calculation, this is needed for the next step

    // 2. Compute Divergence for Active/Passive state[cite: 3]
    // TODO calculate divergence and earth pressure coefficient
    // let div_u = (get_u(id.x + 1, id.y) - get_u(id.x - 1, id.y)) / (2.0 * dx);
    var k = 0f;
    
    let use_earth_pressure_coefficient: bool = (sim_settings.flags & (1u << 2u)) != 0u;
    if use_earth_pressure_coefficient {
        let div_u = div_u(id.x, id.y, mass);
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
    let grad_h2 = vec2f(
        // TODO account for slope in x and y direction. multiply by cos_theta_x
        (get_h2(id.x + 1, id.y) - get_h2(id.x - 1, id.y)) / (2.0 * sim_settings.cell_size),
        (get_h2(id.x, id.y + 1) - get_h2(id.x, id.y - 1)) / (2.0 * sim_settings.cell_size)
    );
    let grad_h = grad_h2 * 0.5 / (h + 1e-5);

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
    grid_forces[idx] = -g * n.z * k * slope_corrected_grad_h;
}

fn minmod(a: f32, b: f32) -> f32 {
    // Returns the smallest magnitude slope if signs match, otherwise 0.0
    return 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b));
}

fn get_h2(x: u32, y: u32) -> f32 {
    let idx = xy_to_idx(x, y);
    return pow(f32(grid_mass_atomic[idx]) * INV_MASS_FACTOR / (sim_settings.snow_density * sim_settings.cell_size * sim_settings.cell_size), 2.0);
}

fn get_velocity(x: u32, y: u32, mass: f32) -> vec2f {
    let idx = xy_to_idx(x, y);
    return vec2f(
        f32(grid_momentum_atomic[idx * 2]) * INV_MOMENTUM_FACTOR / (mass + 1e-6),
        f32(grid_momentum_atomic[idx * 2 + 1]) * INV_MOMENTUM_FACTOR / (mass + 1e-6)
    );
}

fn div_u(x: u32, y: u32, mass: f32) -> f32 {

    let dx = sim_settings.cell_size;

    let uL = get_velocity(x - 1u, y, mass).x;
    let uR = get_velocity(x + 1u, y, mass).x;

    let vD = get_velocity(x, y - 1u, mass).y;
    let vU = get_velocity(x, y + 1u, mass).y;

    let dudx = (uR - uL) / (2.0 * dx);
    let dvdy = (vU - vD) / (2.0 * dx);

    return dudx + dvdy;
}

fn earth_pressure_coefficient(
    phi: f32,
    delta: f32,
    div_u: f32
) -> f32 {

    let cos_phi = cos(phi);
    let cos_delta = cos(delta);

    let inside =
        1.0 -
        (cos_phi * cos_phi) /
        (cos_delta * cos_delta);

    // numerical safety
    let root = sqrt(max(inside, 0.0));

    let sec_phi2 =
        1.0 / (cos_phi * cos_phi);

    // active for expansion
    let Ka =
        2.0 * (1.0 - root) * sec_phi2 - 1.0;

    // passive for compression
    let Kp =
        2.0 * (1.0 + root) * sec_phi2 - 1.0;

    return select(Kp, Ka, div_u > 0.0);
}

// import utils.wgsl;
// BEGIN utils.wgsl
// @atlas: Shared prelude: constants, `Particle`/`SimInfo`/`SimSettings`/`AtomicValues` structs, quantisation factors, cell↔uv↔index helpers, MPM quadratic weights.
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