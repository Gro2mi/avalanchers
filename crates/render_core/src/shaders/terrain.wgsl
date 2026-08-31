// Renders a DEM as a shaded height field. The triangle grid is generated from the
// vertex index, so the only geometry input is the heightmap texture.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // width, height, cell_size, vertical exaggeration
    grid: vec4<f32>,
    // min elevation, max elevation (both already scaled by the exaggeration)
    elevation: vec4<f32>,
    // direction towards the light
    light_dir: vec4<f32>,
    // enabled, min, max, threshold
    overlay: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> overlay_grid: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) elevation: f32,
    // Whole quads are dropped, so this must not be interpolated.
    @location(2) @interpolate(flat) valid: f32,
    @location(3) overlay_value: f32,
};

// Returns the exaggerated height in `x` and the no-data flag in `y`.
fn sample_texel(x: i32, y: i32) -> vec2<f32> {
    let max_x = i32(u.grid.x) - 1;
    let max_y = i32(u.grid.y) - 1;
    let coord = vec2<i32>(clamp(x, 0, max_x), clamp(y, 0, max_y));
    let texel = textureLoad(heightmap, coord, 0);
    return vec2<f32>(texel.r * u.grid.w, texel.g);
}

// Falls back to the centre height so no-data neighbours do not distort the normal.
fn neighbour_height(x: i32, y: i32, fallback: f32) -> f32 {
    let texel = sample_texel(x, y);
    if (texel.y < 0.5) {
        return fallback;
    }
    return texel.x;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corner_offsets = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u),
        vec2<u32>(1u, 0u),
        vec2<u32>(0u, 1u),
        vec2<u32>(1u, 0u),
        vec2<u32>(1u, 1u),
        vec2<u32>(0u, 1u),
    );

    let quads_x = u32(u.grid.x) - 1u;
    let quad = vertex_index / 6u;
    let offset = corner_offsets[vertex_index % 6u];
    let base_x = i32(quad % quads_x);
    let base_y = i32(quad / quads_x);
    let gx = base_x + i32(offset.x);
    let gy = base_y + i32(offset.y);

    let cell = u.grid.z;
    let height = sample_texel(gx, gy).x;
    let world = vec3<f32>(f32(gx) * cell, height, f32(gy) * cell);

    // Central differences on the neighbouring samples give the surface normal.
    let left = neighbour_height(gx - 1, gy, height);
    let right = neighbour_height(gx + 1, gy, height);
    let down = neighbour_height(gx, gy - 1, height);
    let up = neighbour_height(gx, gy + 1, height);
    let normal = normalize(vec3<f32>(left - right, 2.0 * cell, down - up));

    // A quad is drawn only when all four of its samples hold real data.
    let valid = min(
        min(sample_texel(base_x, base_y).y, sample_texel(base_x + 1, base_y).y),
        min(sample_texel(base_x, base_y + 1).y, sample_texel(base_x + 1, base_y + 1).y),
    );

    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world, 1.0);
    out.normal = normal;
    out.elevation = height;
    out.valid = valid;
    out.overlay_value = sample_overlay(gx, gy);
    return out;
}

fn sample_overlay(x: i32, y: i32) -> f32 {
    if (u.overlay.x < 0.5) {
        return 0.0;
    }
    return overlay_grid[u32(y) * u32(u.grid.x) + u32(x)];
}

// Blue to red heat ramp for simulation values.
fn overlay_color(t: f32) -> vec3<f32> {
    let cold = vec3<f32>(0.13, 0.36, 0.86);
    let mid = vec3<f32>(0.15, 0.82, 0.75);
    let warm = vec3<f32>(0.98, 0.82, 0.20);
    let hot = vec3<f32>(0.86, 0.14, 0.10);

    if (t < 0.33) {
        return mix(cold, mid, t / 0.33);
    }
    if (t < 0.66) {
        return mix(mid, warm, (t - 0.33) / 0.33);
    }
    return mix(warm, hot, (t - 0.66) / 0.34);
}

fn terrain_color(t: f32) -> vec3<f32> {
    let forest = vec3<f32>(0.16, 0.32, 0.18);
    let meadow = vec3<f32>(0.38, 0.50, 0.24);
    let scree = vec3<f32>(0.64, 0.58, 0.40);
    let rock = vec3<f32>(0.55, 0.51, 0.48);
    let snow = vec3<f32>(0.97, 0.97, 1.0);

    if (t < 0.25) {
        return mix(forest, meadow, t / 0.25);
    }
    if (t < 0.5) {
        return mix(meadow, scree, (t - 0.25) / 0.25);
    }
    if (t < 0.75) {
        return mix(scree, rock, (t - 0.5) / 0.25);
    }
    return mix(rock, snow, (t - 0.75) / 0.25);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.valid < 0.5) {
        discard;
    }

    let normal = normalize(in.normal);
    let light = normalize(u.light_dir.xyz);
    let diffuse = max(dot(normal, light), 0.0);

    let range = max(u.elevation.y - u.elevation.x, 1.0);
    let t = clamp((in.elevation - u.elevation.x) / range, 0.0, 1.0);

    var albedo = terrain_color(t);

    if (u.overlay.x > 0.5 && in.overlay_value > u.overlay.w) {
        let span = max(u.overlay.z - u.overlay.y, 1e-6);
        let v = clamp((in.overlay_value - u.overlay.y) / span, 0.0, 1.0);
        // Fade in near the threshold so the flow edge does not look stamped on.
        let coverage = clamp(v * 4.0, 0.35, 1.0);
        albedo = mix(albedo, overlay_color(v), coverage);
    }

    let shaded = albedo * (0.35 + 0.75 * diffuse);
    return vec4<f32>(shaded, 1.0);
}
