// Renders a DEM as a shaded height field. The triangle grid is generated from the
// vertex index, so the only geometry input is the heightmap texture.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // width, height, cell_size, vertical exaggeration
    grid: vec4<f32>,
    // min elevation, max elevation (both already scaled by the exaggeration),
    // momentum overlay unit (kg m/s per i32 step), orthophoto enabled (> 0.5)
    elevation: vec4<f32>,
    // direction towards the light
    light_dir: vec4<f32>,
    // enabled, min, max, threshold
    overlay: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> overlay_grid: array<f32>;
// initial forest cover for the forest damage overlay (mode 4)
@group(0) @binding(3) var<storage, read> overlay_reference: array<f32>;
// orthophoto spanning the outer cell edges, row 0 at the south edge
@group(0) @binding(4) var ortho: texture_2d<f32>;
@group(0) @binding(5) var ortho_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) elevation: f32,
    // Whole quads are dropped, so this must not be interpolated.
    @location(2) @interpolate(flat) valid: f32,
    @location(3) overlay_value: f32,
    @location(4) ortho_uv: vec2<f32>,
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
    // Row 0 (the map's south edge) sits at the maximum z, keeping the world
    // right-handed: +x east, +z south.
    let world = vec3<f32>(f32(gx) * cell, height, (u.grid.y - 1.0 - f32(gy)) * cell);

    // Central differences on the neighbouring samples give the surface normal.
    // World z runs against the row index, so the z slope term is `up - down`.
    let left = neighbour_height(gx - 1, gy, height);
    let right = neighbour_height(gx + 1, gy, height);
    let down = neighbour_height(gx, gy - 1, height);
    let up = neighbour_height(gx, gy + 1, height);
    let normal = normalize(vec3<f32>(left - right, 2.0 * cell, up - down));

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
    // vertices sit on cell centres, the orthophoto spans the cell edges
    out.ortho_uv = (vec2<f32>(f32(gx), f32(gy)) + 0.5) / u.grid.xy;
    return out;
}

fn sample_overlay(x: i32, y: i32) -> f32 {
    if (u.overlay.x < 0.5) {
        return 0.0;
    }
    let cell = u32(y) * u32(u.grid.x) + u32(x);
    // Overlay mode 2: interleaved i32 momentum pairs (u, v per cell), as
    // quantized by the simulation's p2g pass, u.elevation.z kg m/s per step.
    // Show the magnitude in kg·m/s.
    // Overlay mode 4: destroyed fraction of the forest, 1 - standing / initial
    // cover; cells without forest fall below any threshold and stay bare.
    if (u.overlay.x > 3.5) {
        let cover = overlay_reference[cell];
        if (cover < 0.01) {
            return -1.0;
        }
        return clamp(1.0 - overlay_grid[cell] / cover, 0.0, 1.0);
    }
    if (abs(u.overlay.x - 2.0) < 0.5) {
        let mu = f32(bitcast<i32>(overlay_grid[cell * 2u]));
        let mv = f32(bitcast<i32>(overlay_grid[cell * 2u + 1u]));
        return length(vec2<f32>(mu, mv)) * u.elevation.z;
    }
    return overlay_grid[cell];
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

// Light to dark green ramp for canopy cover (overlay mode 3); keep in sync
// with ramp_color in colorbar.rs.
fn forest_color(t: f32) -> vec3<f32> {
    let sparse = vec3<f32>(0.86, 0.90, 0.56);
    let open = vec3<f32>(0.52, 0.76, 0.30);
    let closed = vec3<f32>(0.18, 0.52, 0.20);
    let dense = vec3<f32>(0.04, 0.28, 0.12);

    if (t < 0.33) {
        return mix(sparse, open, t / 0.33);
    }
    if (t < 0.66) {
        return mix(open, closed, (t - 0.33) / 0.33);
    }
    return mix(closed, dense, (t - 0.66) / 0.34);
}

// Intact (green) to destroyed (brown) forest for overlay mode 4; keep in sync
// with ramp_color in colorbar.rs.
fn forest_damage_color(t: f32) -> vec3<f32> {
    let intact = vec3<f32>(0.10, 0.42, 0.16);
    let bent = vec3<f32>(0.88, 0.80, 0.22);
    let broken = vec3<f32>(0.92, 0.45, 0.10);
    let destroyed = vec3<f32>(0.42, 0.20, 0.07);

    if (t < 0.33) {
        return mix(intact, bent, t / 0.33);
    }
    if (t < 0.66) {
        return mix(bent, broken, (t - 0.33) / 0.33);
    }
    return mix(broken, destroyed, (t - 0.66) / 0.34);
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
    // the photo already holds the real shading, so the lighting is softened
    var ambient = 0.35;
    var direct = 0.75;
    // sampled unconditionally: textureSample needs uniform control flow
    let photo = textureSample(ortho, ortho_sampler, in.ortho_uv);
    if (u.elevation.w > 0.5) {
        albedo = mix(albedo, photo.rgb, photo.a);
        ambient = mix(ambient, 0.65, photo.a);
        direct = mix(direct, 0.45, photo.a);
    }

    if (u.overlay.x > 0.5 && in.overlay_value > u.overlay.w) {
        let span = max(u.overlay.z - u.overlay.y, 1e-6);
        let v = clamp((in.overlay_value - u.overlay.y) / span, 0.0, 1.0);
        // Fade in near the threshold so the flow edge does not look stamped on.
        var coverage = clamp(v * 4.0, 0.35, 1.0);
        var color = overlay_color(v);
        if (u.overlay.x > 3.5) {
            // intact forest is a value of 0 and must stay clearly visible
            color = forest_damage_color(v);
            coverage = 0.9;
        } else if (u.overlay.x > 2.5) {
            color = forest_color(v);
            coverage = 0.9;
        }
        albedo = mix(albedo, color, coverage);
    }

    let shaded = albedo * (ambient + direct * diffuse);
    return vec4<f32>(shaded, 1.0);
}
