@group(0) @binding(0) var<uniform> simSettings: SimSettings;
@group(0) @binding(1) var demTexture: texture_2d<f32>;
@group(0) @binding(2) var slopeTexture: texture_2d<f32>;
@group(0) @binding(3) var roughnessTexture: texture_2d<f32>;
// @group(0) @binding(2) var landcoverTexture: texture_2d<u32>;

@group(0) @binding(6) var releasePointsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<storage, read_write> out_debug: array<f32>;
@group(0) @binding(8) var<storage, read_write> atomicBuffer: AtomicData;


const confers_trees = vec4u(34, 139, 34, 255);
const broadleaf_deciduous_trees = vec4u(128, 255, 0, 255);
const broadleaf_evergreen_trees = vec4u(0, 255, 8, 255);


// fn is_forest(id: vec2u) -> bool {
//     // check if the pixel is within the bounds of the landcover texture
//     if (id.x >= textureDimensions(landcoverTexture).x || id.y >= textureDimensions(landcoverTexture).y) {
//         return false;
//     }
//     // load the landcover value at the given id
//     let landcover_value = textureLoad(landcoverTexture, id, 0).r; // assuming landcover is stored in the red channel
//     // check if the value corresponds to forest (e.g., 1.0 for forest)
//     return landcover_value > 0.5; // adjust threshold as needed
// }

@compute @workgroup_size(16, 16, 1)
fn computeReleasePoints(@builtin(global_invocation_id) id: vec3<u32>) {
    // exit if thread id is outside image dimensions (i.e. thread is not supposed to be doing any work)
    let texture_size = textureDimensions(releasePointsTexture);
    if (id.x >= texture_size.x || id.y >= texture_size.y) {
        return;
    }
    let tex_pos = id.xy;
    var rgba = textureLoad(slopeTexture, tex_pos, 0);
    let slopeAngle = rgba.r;
    let windShelterIndex = rgba.b;

    let elevation = textureLoad(demTexture, tex_pos, 0).x;
    rgba = textureLoad(roughnessTexture, tex_pos, 0);
    let roughness = rgba.r;
    let forest = rgba.g;
    
    let gpxMask = 0f;
    let predictor = 0f;

    if (slopeAngle < simSettings.min_slope_angle || slopeAngle > simSettings.max_slope_angle 
        || elevation < simSettings.min_elevation
        || roughness > simSettings.roughness_threshold
        || forest > 0.1
    ) {
        // no release cell
        textureStore(releasePointsTexture, tex_pos, vec4f(0f, gpxMask, predictor, 0f));
    } else {
        // release cell
        textureStore(releasePointsTexture, tex_pos, vec4f(1, gpxMask, predictor, 0f));
        atomicAdd(&atomicBuffer.counter, 1u);
    out_debug[0] = f32(simSettings.slab_thickness);
    }
    // needs to stay here, otherwise texture is not used
}
