import utils.wgsl;

// @group(0) @binding(0) var<uniform> sim_settings: sim_settings;
@group(0) @binding(1) var dem_texture: texture_2d<f32>;
@group(0) @binding(2) var slope_texture: texture_2d<f32>;
@group(0) @binding(3) var roughness_texture: texture_2d<f32>;
// @group(0) @binding(4) var landcoverTexture: texture_2d<u32>;

@group(0) @binding(4) var release_areas_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<storage, read_write> out_debug: array<f32>;
@group(0) @binding(6) var<storage, read_write> atomic_release_cell_counter: AtomicValue;


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

@compute @workgroup_size(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)
fn compute_release_areas(@builtin(global_invocation_id) id: vec3<u32>) {
    // exit if thread id is outside image dimensions (i.e. thread is not supposed to be doing any work)
    let texture_size = textureDimensions(release_areas_texture);
    if (id.x >= texture_size.x || id.y >= texture_size.y) {
        return;
    }
    let tex_pos = id.xy;
    var rgba = textureLoad(slope_texture, tex_pos, 0);
    let slope_angle = rgba.r;
    let wind_shelter_index = rgba.b;

    let elevation = textureLoad(dem_texture, tex_pos, 0).x;
    rgba = textureLoad(roughness_texture, tex_pos, 0);
    let roughness = rgba.r;
    let forest = rgba.g;
    
    let gpx_mask = 0f;
    let predictor = 0f;

    if (slope_angle < sim_settings.min_slope_angle || slope_angle > sim_settings.max_slope_angle 
        || elevation < sim_settings.min_elevation
        || roughness > sim_settings.roughness_threshold
        || forest > 0.1
    ) {
        // no release cell
        textureStore(release_areas_texture, tex_pos, vec4f(0f, gpx_mask, predictor, 0f));
    } else {
        // release cell
        textureStore(release_areas_texture, tex_pos, vec4f(1, gpx_mask, predictor, 0f));
        atomicAdd(&atomic_release_cell_counter.value, 1u);
    out_debug[0] = f32(sim_settings.slab_thickness);
    }
    // needs to stay here, otherwise texture is not used
}
