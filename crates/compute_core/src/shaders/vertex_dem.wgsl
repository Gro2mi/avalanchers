struct Uniforms {
    modelViewProjectionMatrix: mat4x4<f32>,
    gridDimensions: vec2<f32>, // e.g., 1024.0, 1024.0
    terrainHeightScale: f32,   // Adjusts how "tall" the mountains are
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var demTexture: texture_2d<f32>;
@group(0) @binding(2) var demSampler: sampler;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) vUV: vec2<f32>,
    @location(1) height: f32,
};

@vertex
fn main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
    // 1. Determine grid coordinates (x, y) based on the vertex index
    let gridX = f32(VertexIndex % u32(uniforms.gridDimensions.x));
    let gridY = f32(VertexIndex / u32(uniforms.gridDimensions.x));
    
    // 2. Normalize to UV coordinates (0.0 to 1.0)
    let uv = vec2<f32>(
        gridX / (uniforms.gridDimensions.x - 1.0),
        gridY / (uniforms.gridDimensions.y - 1.0)
    );

    // 3. Sample the DEM texture for the height (assuming data is in Red channel)
    let demData = textureSampleLevel(demTexture, demSampler, uv, 0.0);
    let h = demData.r * uniforms.terrainHeightScale;

    // 4. Construct the 3D position (X and Y are scaled to the grid, Z is height)
    let pos = vec4<f32>(gridX, h, gridY, 1.0);

    var output: VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix * pos;
    output.vUV = uv;
    output.height = h;
    return output;
}