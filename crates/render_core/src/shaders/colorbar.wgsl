// Colour bar legend for the grid overlay. All geometry arrives in normalised
// device coordinates with per-vertex UVs and straight-alpha colours; the font
// atlas is single-channel and doubles as a white fill for solid quads.

@group(0) @binding(0) var font: texture_2d<f32>;
@group(0) @binding(1) var font_sampler: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mask = textureSample(font, font_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * mask);
}
