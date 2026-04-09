const MAX_RAYS = 21;
const MAX_STEPS = 25;

@group(0) @binding(0) var dem_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var shelterOut: texture_storage_2d<rgba32float, write>;

struct Params {
  width: u32,
  height: u32,
  cellSize: f32,
  windDirDeg: f32,
};

fn deg_to_rad(deg: f32) -> f32 {
  return deg * PI / 180.0;
}

fn atan2(y: f32, x: f32) -> f32 {
  return atan2(y, x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;

  if (x >= params.width || y >= params.height) {
    return;
  }

  let center = vec2<i32>(i32(x), i32(y));
  let z0 = textureLoad(dem_texture, center, 0).r;

  var angles: array<f32, MAX_RAYS * MAX_STEPS>;
  var count: u32 = 0;

  let windDir = deg_to_rad(params.windDirDeg);
  let angleStart = windDir - deg_to_rad(10.0);
  let angleStep = deg_to_rad(1.0);

  for (var a = 0u; a < MAX_RAYS; a++) {
    let theta = angleStart + f32(a) * angleStep;
    let dx = cos(theta);
    let dy = sin(theta);

    for (var s = 1u; s <= MAX_STEPS; s++) {
      let dist = f32(s) * params.cellSize;
      let samplePos = vec2<i32>(
        i32(round(f32(center.x) + dx * f32(s))),
        i32(round(f32(center.y) + dy * f32(s)))
      );

      if (samplePos.x < 0 || samplePos.y < 0 ||
          samplePos.x >= i32(params.width) || samplePos.y >= i32(params.height)) {
        continue;
      }

      let zi = textureLoad(dem_texture, samplePos, 0).r;
      let angle = atan2(zi - z0, dist);

      angles[count] = angle;
      count += 1u;
    }
  }

  // Sort the angles to get the 75th percentile
  for (var i = 0u; i < count; i++) {
    for (var j = i + 1u; j < count; j++) {
      if (angles[j] < angles[i]) {
        let tmp = angles[i];
        angles[i] = angles[j];
        angles[j] = tmp;
      }
    }
  }

  let q75_index = u32(f32(count) * 0.75);
  let q75 = angles[min(q75_index, count - 1u)];

  textureStore(shelterOut, vec2<u32>(x, y), vec4<f32>(q75, 0.0, 0.0, 1.0));
}
