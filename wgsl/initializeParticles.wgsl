@group(0) @binding(0) var<uniform> settings: SimSettings;
@group(0) @binding(1) var demTex: texture_2d<f32>;
@group(0) @binding(2) var releaseTex: texture_2d<f32>;
@group(0) @binding(3) var texSampler: sampler;

@group(0) @binding(4) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(5) var<storage, read_write> atomicBuffer: AtomicData;


@compute @workgroup_size(16, 16, 1)
fn initializeParticles(@builtin(global_invocation_id) cell: vec3<u32>) {
    let seed = 42u;
    let gridSize = textureDimensions(demTex);
    if (cell.x >= gridSize.x || cell.y >= gridSize.y) {
        return;
    }

    let snowThickness = textureLoad(releaseTex, cell.xy, 0).r;
    if (snowThickness <= 0.01) {
        return;
    }

    for (var n: u32 = 0; n < settings.released_particles_per_cell; n++) {
        let particleIndex = atomicAdd(&atomicBuffer.counter, 1u);
        
        let r = rand22u(cell.xy + seed) - 0.5;
        let cellXY = (vec2f(cell.xy) + r);
        let elevation = textureSampleLevel(demTex, texSampler,  cellIndexfToUV(cellXY, gridSize), 0).x;

        let p = &particles[particleIndex];
        p.position = vec3f(cellXY * settings.cell_size, elevation);
        p.mass = (settings.snow_density * snowThickness * settings.cell_size * settings.cell_size) / f32(settings.released_particles_per_cell);
        p.velocity = vec3f(0f);
        p.snow_thickness = snowThickness;
        p.C = mat2x2f(0f, 0f, 0f, 0f);
    }
}
