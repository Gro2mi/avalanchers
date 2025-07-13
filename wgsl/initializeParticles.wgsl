// @group(0) @binding(0) var<uniform> simSettings: SimSettings;
@group(0) @binding(1) var<storage, read_write> simInfo: SimInfo;
@group(0) @binding(2) var demTex: texture_2d<f32>;
@group(0) @binding(3) var releaseTex: texture_2d<f32>;
@group(0) @binding(4) var texSampler: sampler;

@group(0) @binding(5) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(6) var<storage, read_write> numberReleaseParticles: AtomicValue;
@group(0) @binding(7) var<storage, read_write> maxVelocity: AtomicValue;


@compute @workgroup_size(16, 16, 1)
fn initializeParticles(@builtin(global_invocation_id) cell: vec3<u32>) {
    let seed = 42u;
    if(cell.x == 0 && cell.y == 0) {
        // set max velocity for the first timestep, 
        // 3.5 m/s^2 for tangential acceleration at 50° slope
        atomicStore(&maxVelocity.value, u32(sqrt(2f * simSettings.cfl * simSettings.cell_size / 3.5) * maxVelocityFactor)); // reset max velocity for new timestep
        simInfo.elevation_threshold = minElevation() - 0.1;
    }
    
    if (cell.x >= simSettings.grid_shape.x || cell.y >= simSettings.grid_shape.y) {
        return;
    }

    let snowThickness = textureLoad(releaseTex, cell.xy, 0).r;
    if (snowThickness <= 0.01) {
        return;
    }

    for (var n: u32 = 0; n < simSettings.released_particles_per_cell; n++) {
        let particleIndex = atomicAdd(&numberReleaseParticles.value, 1u);

        let r = rand22u(cell.xy + seed) - 0.5;
        let cellXY = (vec2f(cell.xy) + r);
        let elevation = textureSampleLevel(demTex, texSampler,  cellfToUV(cellXY), 0).x;

        let p = &particles[particleIndex];
        p.position = vec3f(cellXY * simSettings.cell_size, elevation);
        p.mass = (simSettings.snow_density * snowThickness * simSettings.cell_size * simSettings.cell_size) / f32(simSettings.released_particles_per_cell);
        // p.velocity = rand32u(cell.xy + seed) * 1e-5;
        p.velocity = vec3f(0f);
        p.snow_thickness = snowThickness;
        p.C = mat2x2f(0f, 0f, 0f, 0f);
        p.stopped = 0u;
    }
}

const MIN_VALID_ELEVATION: f32 = 0.9; 

fn minElevation() -> f32 {
    // find the minimum elevation in the height texture
    let tex_size = textureDimensions(demTex);
    var min_val: f32 = 1e10;

    for (var y: u32 = 0; y < tex_size.y; y++) {
        for (var x: u32 = 0; x < tex_size.x; x++) {
            let value = textureLoad(demTex, vec2u(x, y), 0).x;
            // Only consider values above the minimum valid elevation threshold
            if (value < min_val && value > MIN_VALID_ELEVATION) {
                min_val = value;
            }
        }
    }
    return min_val;
}