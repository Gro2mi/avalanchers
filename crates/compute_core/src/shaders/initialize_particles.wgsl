// import utils.wgsl;
// import random.wgsl;

@group(0) @binding(1) var<storage, read_write> sim_info: SimInfo;
@group(0) @binding(2) var dem_texture: texture_2d<f32>;
@group(0) @binding(3) var release_areas: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;
// @group(0) @binding(9) var<uniform> tracked_particle_relative_positions: array<Particle>;

@group(0) @binding(5) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(6) var<storage, read_write> number_release_particles: AtomicValue;
@group(0) @binding(7) var<storage, read_write> atomic_cell_count_buffer: array<atomic<u32>>; // trajectory texture
@group(0) @binding(8) var<storage, read_write> atomic_velocity_buffer: array<atomic<u32>>; // trajectory texture
// @group(0) @binding(10) var<storage, read_write> tracked_particle_ids: array<u32>;


@compute @workgroup_size(16, 16, 1)
fn initialize_particles(@builtin(global_invocation_id) cell: vec3<u32>) {
    let seed = 42u;
    if(cell.x == 0 && cell.y == 0) {
        // set max velocity for the first timestep, 
        // 3.5 m/s^2 for tangential acceleration at 50° slope
        sim_info.max_velocity = sqrt(2f * sim_settings.cfl * sim_settings.cell_size / 3.5);
        sim_info.elevation_threshold = minElevation() - 0.1;
    }

    if (cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y) {
        return;
    }

    let snow_thickness = textureLoad(release_areas, cell.xy, 0).r;
    if (snow_thickness <= 0.01) {
        return;
    }

    for (var n: u32 = 0; n < sim_settings.released_particles_per_cell; n++) {
        let particleIndex = atomicAdd(&number_release_particles.value, 1u);

        let r = rand22u(cell.xy + seed) - 0.5;
        let cell_xy = (vec2f(cell.xy) + r);
        let elevation = textureSampleLevel(dem_texture, tex_sampler,  cellf_to_uv(cell_xy), 0).x;

        let p = &particles[particleIndex];
        p.position = vec3f(cell_xy * sim_settings.cell_size, elevation);
        p.mass = (sim_settings.snow_density * snow_thickness * sim_settings.cell_size * sim_settings.cell_size) / f32(sim_settings.released_particles_per_cell);
        // p.velocity = rand32u(cell.xy + seed) * 1e-5;
        p.velocity = vec3f(0f);
        p.snow_thickness = snow_thickness;
        p.C = mat2x2f(0f, 0f, 0f, 0f);
        p.stopped = 0u;
        
        let cell_index = position_to_cell_index(p.position);
        atomicAdd(&atomic_cell_count_buffer[cell_index], 1u);
        atomicMax(&atomic_velocity_buffer[cell_index], u32(length(p.velocity))); // ensure that the velocity is not zero, this is needed for the next step

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
            if (value < min_val && value > MIN_VALID_ELEVATION) {
                min_val = value;
            }
        }
    }
    return min_val;
}