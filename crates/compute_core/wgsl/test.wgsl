@group(0) @binding(0) var outputTex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Write a color gradient to the texture
    let dims = textureDimensions(outputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let color = vec4<f32>(
        f32(gid.x) / f32(dims.x),  // R: normalized x
        f32(gid.y) / f32(dims.y),  // G: normalized y
        0.5,                       // B: constant
        1.0                        // A: opaque
    );

    textureStore(outputTex, vec2<i32>(gid.xy), color);
}
