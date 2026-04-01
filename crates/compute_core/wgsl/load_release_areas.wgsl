
struct AtomicValue {
    value: atomic<u32>,
};

@group(0) @binding(0) var release_areas_in: texture_2d<u32>;

@group(0) @binding(1) var release_areas_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read_write> debug: array<f32>;
@group(0) @binding(3) var<storage, read_write> number_release_cells: AtomicValue;

@compute @workgroup_size(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)
fn load_release_areas(@builtin(global_invocation_id) id: vec3<u32>) {
    let gridSize = textureDimensions(release_areas_in);
    if (id.x >= gridSize.x || id.y >= gridSize.y) {
        return;
    }
    let release_thickness = f32(textureLoad(release_areas_in, id.xy, 0).a) / 100;
    if (release_thickness > 0.01){
        textureStore(release_areas_out, id.xy, vec4f(release_thickness, 0.0, 0.0, 0.0));
        atomicAdd(&number_release_cells.value, 1u);
    }
    if(id.x == 103 && id.y == 269) {
        debug[0] = release_thickness;
        debug[1] = f32(atomicLoad(&number_release_cells.value));
    }
}