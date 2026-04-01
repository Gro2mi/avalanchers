use criterion::{Criterion, black_box, criterion_group, criterion_main};
use data_processor::*;

use std::env;
use std::path::PathBuf;

fn my_benchmark(c: &mut Criterion) {
    let tmp_dir = env::temp_dir();
    let file_path = tmp_dir.join("test_write_file");
    let png_path = PathBuf::from("../avaframe/avaArzlerUni.png");
    let (data, _width, _height) = read_png(&png_path).expect("Failed to load PNG");

    c.bench_function("write_zst", |b| {
        b.iter(|| {
            // write_zstd(black_box(&file_path), black_box(&data)).expect("write_zstd failed");
            println!("Benchmark finished")
        });
    });
    
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
