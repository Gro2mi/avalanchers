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
    // jitter amplitude (0 = off), grid mass to flow thickness factor,
    // 1 if the grid mass is stored as f32 bits, unused
    jitter: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> stopped: array<u32>;
@group(0) @binding(4) var<storage, read> elevations: array<f32>;
@group(0) @binding(5) var<storage, read> velocities_z: array<f32>;
@group(0) @binding(6) var<storage, read> grid_mass: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat) speed: f32,
    @location(2) @interpolate(flat) moving: f32,
};

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

    // The simulation stores x and y in the DEM plane; y runs against the
    // renderer's z axis so the world stays right-handed (+z south).
    let plane = positions[index];
    let centre = vec3<f32>(
        plane.x,
        (elevations[index] + elevation_jitter(index, plane)) * u.up.w + u.grid.w,
        (u.grid.y - 1.0) * u.grid.z - plane.y,
    );

    let radius = u.right.w;
    let world = centre + u.right.xyz * (offset.x * radius) + u.up.xyz * (offset.y * radius);

    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world, 1.0);
    out.offset = offset;
    // Vertical velocity is zero for models without a z component, so the ramp still
    // shows horizontal speed there.
    out.speed = length(vec3<f32>(velocities[index], velocities_z[index]));
    out.moving = select(1.0, 0.0, stopped[index] != 0u);
    return out;
}

// Random share of the local flow thickness; visual only, see jitter uniform.
fn elevation_jitter(index: u32, plane: vec2<f32>) -> f32 {
    if (u.jitter.x <= 0.0) {
        return 0.0;
    }
    let size = vec2<u32>(u.grid.xy);
    let cell = min(vec2<u32>(max(plane / u.grid.z, vec2<f32>(0.0))), size - vec2<u32>(1u));
    let raw = grid_mass[cell.y * size.x + cell.x];
    let mass = select(f32(raw), bitcast<f32>(raw), u.jitter.z > 0.5);
    return u.jitter.x * mass * u.jitter.y * hash01(index);
}

// PCG hash of the particle index mapped to [0, 1)
fn hash01(index: u32) -> f32 {
    let state = index * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
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
