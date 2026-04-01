fn rand1(x: f32) -> f32 {
  return fract(sin(x) * 43758.5453123);
}
fn rand1u(x: u32) -> f32 {
  return rand1(f32(x));
}
fn rand2(x: f32) -> vec2<f32> {
  return vec2<f32>(
    fract(sin(x) * 43758.5453),
    fract(sin(x + 13.13) * 43758.5453)
  );
}

fn rand2u(x: u32) -> vec2f {
  return rand2(f32(x));
}

fn rand3(x: f32) -> vec3<f32> {
  return vec3<f32>(
    fract(sin(x) * 43758.5453),
    fract(sin(x + 21.21) * 43758.5453),
    fract(sin(x + 42.42) * 43758.5453)
  );
}
fn rand3u(x: u32) -> vec3f {
  return rand3(f32(x));
}

fn rand4(x: f32) -> vec4<f32> {
  return vec4<f32>(
    fract(sin(x) * 43758.5453),
    fract(sin(x + 21.21) * 43758.5453),
    fract(sin(x + 42.42) * 43758.5453),
    fract(sin(x + 84.84) * 43758.5453)
  );
}

fn rand21(v: vec2<f32>) -> f32 {
  return fract(sin(dot(v, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn rand31(v: vec3<f32>) -> f32 {
  return fract(sin(dot(v, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453123);
}

fn rand41(v: vec4<f32>) -> f32 {
  return fract(sin(dot(v, vec4<f32>(12.9898, 78.233, 37.719, 24.876))) * 43758.5453123);
}

fn rand22(v: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    rand21(v),
    rand21(v + 13.13)
  );
}
fn rand22u(v: vec2u) -> vec2f {
  return rand22(vec2f(v));
}

fn rand33(v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    rand31(v),
    rand31(v + vec3<f32>(21.21, 37.37, 59.59)),
    rand31(v + vec3<f32>(17.17, 31.31, 89.89))
  );
}

fn rand44(v: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    rand41(v),
    rand41(v + vec4<f32>(21.21, 37.37, 59.59, 83.83)),
    rand41(v + vec4<f32>(19.19, 29.29, 41.41, 61.61)),
    rand41(v + vec4<f32>(23.23, 31.31, 47.47, 67.67))
  );
}

fn rand32u(v: vec2u) -> vec3f {
  let vf = vec2f(v);
  return vec3<f32>(
    rand21(vf),
    rand21(vf + vec2f(21.21, 37.37)),
    rand21(vf + vec2f(17.17, 31.31))
  );
}