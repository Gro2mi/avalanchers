

@group(0) @binding(1) var demTexture: texture_2d<f32>;
@group(0) @binding(2) var windTexture: texture_2d<f32>;

@group(0) @binding(3) var normalsTexture: texture_storage_2d<rgba16float, write>; // ASSERT: same dimensions as heights_texture
@group(0) @binding(4) var slopeTexture: texture_storage_2d<rgba16float, write>; // ASSERT: same dimensions as heights_texture
@group(0) @binding(5) var<storage, read_write> debug: array<f32>;



@compute @workgroup_size(16, 16, 1)
fn computeNormals(@builtin(global_invocation_id) cell: vec3<u32>) {
    // exit if thread cell is outscelle image dimensions (i.e. thread is not supposed to be doing any work)
    let gridSize = textureDimensions(normalsTexture);
    if (cell.x >= gridSize.x || cell.y >= gridSize.y) {
        return;
    }

    let resolution = simSettings.cell_size;
    let resolution2 = resolution * resolution;
    let coord = vec2<i32>(cell.xy);

    // Sample center and neighbors
    let center = textureLoad(demTexture, coord + vec2<i32>(0, 0), 0).r;
    let left   = textureLoad(demTexture, coord + vec2<i32>(-1, 0), 0).r;
    let right  = textureLoad(demTexture, coord + vec2<i32>( 1, 0), 0).r;
    let down   = textureLoad(demTexture, coord + vec2<i32>(0, -1), 0).r;
    let up     = textureLoad(demTexture, coord + vec2<i32>(0,  1), 0).r;

    let upRight    = textureLoad(demTexture, coord + vec2<i32>(1,  1), 0).r;
    let downRight  = textureLoad(demTexture, coord + vec2<i32>(-1,  1), 0).r;
    let upLeft     = textureLoad(demTexture, coord + vec2<i32>(1,  -1), 0).r;
    let downLeft   = textureLoad(demTexture, coord + vec2<i32>(-1,  -1), 0).r;

    // TODO handle domain edges
    let dx = (right - left) / (2.0 * resolution);
    let dy = (up - down) / (2.0 * resolution);
    // Normal from gradient, assuming square cells and disregards the change in latitude correction 
    // within the texture which is in the magnitude of 1e-4 for a normal skitour
    let normal = normalize(vec3(-dx, -dy, 1.0));

    let dxx = (left - 2*center + right) / (resolution2);
    let dyy = (up - 2*center + down) / (resolution2);
    let dxy = (upRight - downRight - upLeft + downLeft) / (4 * resolution2);

    let denom = pow(dx*dx + dy*dy + 1e-12, 1.5);
    let p = dx;
    let q = dy;
    let profileCurvature = (dxx*dx*dx + 2.0*dxy*dx*dy + dyy*dy*dy) / denom;
    // let planCurvature    = (dxx*dy*dy - 2.0*dxy*dx*dy + dyy*dx*dx) / denom;
    // let meanCurvature = ((1.0 + dy*dy)*dxx - 2.0*dx*dy*dxy + (1.0 + dx*dx)*dyy) / (2.0 * pow(1.0 + dx*dx + dy*dy, 1.5));
    textureStore(normalsTexture, coord, vec4f(normal, profileCurvature));

    let slopeAngle = acos(normal.z) * RAD_TO_DEG;
    let slopeAspect = (atan2(normal.x, normal.y) * RAD_TO_DEG + 360.0) % 360.0;
    let windShelterIndex = textureLoad(windTexture, coord, 0).r;
    textureStore(slopeTexture, coord, vec4f(slopeAngle, slopeAspect, windShelterIndex, 0f));

    if(cell.x == 222 && cell.y == 222) {
        debug[0] = normal.x;
        debug[1] = normal.y;
        debug[2] = normal.z;
        debug[3] = simSettings.cell_size;
        debug[4] = dx;
        debug[5] = dy;
        debug[6] = dxx;
        debug[7] = dyy;
        debug[8] = dxy;
        debug[9] = profileCurvature;
        debug[10] = left;
        debug[11] = right;
        debug[12] = up;
        debug[13] = down;
        debug[14] = center;
    }
}
