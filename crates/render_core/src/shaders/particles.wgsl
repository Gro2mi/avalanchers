// Draws simulation particles as camera-facing discs. Positions are read straight from the
// simulation's storage buffers, so no data round-trips through the CPU.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // camera right vector in xyz, particle radius in w
    right: vec4<f32>,
    // camera up vector in xyz, vertical exaggeration in w
    up: vec4<f32>,
    // particle count, velocity used as the top of the colour ramp
    params: vec4<f32>,
    // grid width, grid height, cell size, height offset above the surface
    grid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> stopped: array<u32>;
@group(0) @binding(4) var heightmap: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat) speed: f32,
    @location(2) @interpolate(flat) moving: f32,
};

fn load_height(x: i32, y: i32) -> f32 {
    let max_x = i32(u.grid.x) - 1;
    let max_y = i32(u.grid.y) - 1;
    let coord = vec2<i32>(clamp(x, 0, max_x), clamp(y, 0, max_y));
    return textureLoad(heightmap, coord, 0).r;
}

// The simulation never writes back particle elevation, so the surface height is
// interpolated from the DEM instead.
fn surface_elevation(plane: vec2<f32>) -> f32 {
    let cell = max(plane / u.grid.z, vec2<f32>(0.0, 0.0));
    let base = vec2<i32>(i32(floor(cell.x)), i32(floor(cell.y)));
    let f = cell - floor(cell);

    let h00 = load_height(base.x, base.y);
    let h10 = load_height(base.x + 1, base.y);
    let h01 = load_height(base.x, base.y + 1);
    let h11 = load_height(base.x + 1, base.y + 1);

    let height = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    return height * u.up.w;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corner_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    let index = vertex_index / 6u;
    let offset = corner_offsets[vertex_index % 6u];

    // The simulation stores x and y in the DEM plane; y maps to the renderer's z axis.
    let plane = positions[index];
    let centre = vec3<f32>(plane.x, surface_elevation(plane) + u.grid.w, plane.y);

    let radius = u.right.w;
    let world = centre + u.right.xyz * (offset.x * radius) + u.up.xyz * (offset.y * radius);

    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world, 1.0);
    out.offset = offset;
    out.speed = length(velocities[index]);
    out.moving = select(1.0, 0.0, stopped[index] != 0u);
    return out;
}

fn speed_color(t: f32) -> vec3<f32> {
    let slow = vec3<f32>(0.30, 0.65, 1.0);
    let mid = vec3<f32>(0.98, 0.90, 0.35);
    let fast = vec3<f32>(0.92, 0.20, 0.12);

    if (t < 0.5) {
        return mix(slow, mid, t / 0.5);
    }
    return mix(mid, fast, (t - 0.5) / 0.5);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radial = dot(in.offset, in.offset);
    if (radial > 1.0) {
        discard;
    }

    let t = clamp(in.speed / max(u.params.y, 1e-6), 0.0, 1.0);
    var color = speed_color(t);
    if (in.moving < 0.5) {
        color = mix(color, vec3<f32>(0.55, 0.55, 0.58), 0.75);
    }

    // Cheap spherical shading so overlapping particles stay readable.
    let lit = 0.55 + 0.45 * sqrt(max(1.0 - radial, 0.0));
    return vec4<f32>(color * lit, 1.0);
}
