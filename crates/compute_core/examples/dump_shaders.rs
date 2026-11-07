//! Writes every compute shader to a directory exactly as a browser receives
//! it: imports freshly expanded, in the variant without float atomics. CI feeds
//! the result to Tint, the WGSL compiler of Chrome, which enforces rules naga
//! does not. Skips import fragments, which do not compile on their own, and
//! test shaders, which only exist under `cfg(test)`.

use compute_core::shaders::{ShaderName, load_shader_source};
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let out_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: dump_shaders <out_dir>"),
    );
    std::fs::create_dir_all(&out_dir)?;
    for &shader in ShaderName::ALL {
        let source = load_shader_source(shader, false);
        let file = shader.source_file();
        if source.contains("@compute") && !file.starts_with("test_") {
            std::fs::write(out_dir.join(format!("{file}.wgsl")), source)?;
        }
    }
    Ok(())
}
