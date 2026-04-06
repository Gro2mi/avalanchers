import utils.wgsl;

@group(0) @binding(1) var normals_texture: texture_2d<f32>;
@group(0) @binding(2) var forest_texture: texture_2d<u32>;
@group(0) @binding(3) var roughness_texture: texture_storage_2d<rgba32float, write>;


@compute @workgroup_size(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)
fn compute_roughness(@builtin(global_invocation_id) id: vec3<u32>) {
    let threshold = sim_settings.roughness_threshold;
    let cell = id.xy;
    if (cell.x >= sim_settings.grid_shape.x || cell.y >= sim_settings.grid_shape.y) {
        return;
    }
    var roughness = 0f;
    // according to doi:10.5194/nhess-16-2211-2016
    if (cell.x == 0 || cell.y == 0 || cell.x >= sim_settings.grid_shape.x - 1 || cell.y >= sim_settings.grid_shape.y - 1) {
        roughness = 1.0; // high roughness at borders
    }
    else {
        // TODO implement kernel size depending on resolution and average snow height
        // 2 * (HS_mean^2 * C_v) + 1; C_v = 0.35 for regular smoothing, 0.2 for low smoothing (low winds)
        var idx = 0u;
        var r: array<vec3f, 9>;
        var r_sum = vec3f(0.0, 0.0, 0.0);
        
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let normal = textureLoad(normals_texture, vec2<i32>(cell) + vec2(x, y), 0);
                let alpha = acos(normal.z);             // slope in rad
                let beta = atan2(normal.x, normal.y);   // aspect in rad
                r[idx].x = sin(alpha) * cos(beta);      // x component of roughness vector
                r[idx].y = sin(alpha) * sin(beta);      // y component of roughness vector
                r[idx].z = cos(alpha);                  // z component of roughness vector
                idx = idx + 1u;
            }
        }
        for (var i = 0u; i < 9u; i = i + 1u) {
            r_sum = r_sum + r[i];
        }
        let r_magnitude = length(r_sum);
        roughness = 1 - r_magnitude / 9.0;
    }
    let forest = textureLoad(forest_texture, cell, 0).r;
    textureStore(roughness_texture, cell, vec4f(roughness, f32(forest), 0.0, 0.0));
}