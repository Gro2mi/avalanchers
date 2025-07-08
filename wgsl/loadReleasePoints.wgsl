@group(0) @binding(0) var releasePointsIn: texture_2d<u32>;

@group(0) @binding(1) var releasePointsOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> debug: array<f32>;
@group(0) @binding(3) var<storage, read_write> atomicBuffer: AtomicData;

@compute @workgroup_size(16, 16, 1)
fn loadReleasePoints(@builtin(global_invocation_id) id: vec3<u32>) {
    let gridSize = textureDimensions(releasePointsIn);
    if (id.x >= gridSize.x || id.y >= gridSize.y) {
        return;
    }
    let releaseThickness = f32(textureLoad(releasePointsIn, id.xy, 0).a) / 100;
    if (releaseThickness > 0.01){
        textureStore(releasePointsOut, id.xy, vec4f(releaseThickness, 0.0, 0.0, 0.0));
        atomicAdd(&atomicBuffer.counter, 1u);
    }
    if(id.x == 103 && id.y == 269) {
        debug[0] = releaseThickness;
        debug[1] = f32(atomicLoad(&atomicBuffer.counter));
    }
}