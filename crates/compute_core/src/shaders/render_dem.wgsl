struct Uniforms {
    view_proj: mat4x4<f32>,
    grid_size: vec2<f32>,
    height_scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var dem_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
    // Generate grid coordinates from index (assuming a triangle list)
    // For a real app, you'd use an index buffer. 
    // Here we assume v_idx maps to a grid [0..width, 0..height]
    let x = f32(v_idx % u32(uniforms.grid_size.x));
    let z = f32(v_idx / u32(uniforms.grid_size.x));
    
    let uv = vec2<f32>(x / (uniforms.grid_size.x - 1.0), z / (uniforms.grid_size.y - 1.0));
    
    // Displacement
    let height = textureSampleLevel(dem_tex, tex_sampler, uv, 0.0).r * uniforms.height_scale;
    
    var out: VertexOutput;
    out.uv = uv;
    out.clip_position = uniforms.view_proj * vec4<f32>(x, height, z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = textureSample(normal_tex, tex_sampler, in.uv).rgb;
    // Unpack normal from [0, 1] to [-1, 1] if necessary
    let n = normalize(normal * 2.0 - 1.0);
    
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(n, light_dir), 0.1); // 0.1 ambient base
    
    let snow_color = vec3<f32>(0.9, 0.95, 1.0);
    return vec4<f32>(snow_color * diffuse, 1.0);
}