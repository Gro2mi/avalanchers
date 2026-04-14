// A high-quality 32-bit hash (PCG)
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advances the seed and returns a float 0.0 -> 1.0
fn next_rand(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / f32(0xffffffffu);
}

fn rand1(seed: ptr<function, u32>) -> f32 {
    return next_rand(seed);
}

fn rand2(seed: ptr<function, u32>) -> vec2f {
    return vec2f(next_rand(seed), next_rand(seed));
}

fn rand3(seed: ptr<function, u32>) -> vec3f {
    return vec3f(next_rand(seed), next_rand(seed), next_rand(seed));
}

fn rand4(seed: ptr<function, u32>) -> vec4f {
    return vec4f(next_rand(seed), next_rand(seed), next_rand(seed), next_rand(seed));
}
