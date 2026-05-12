use compute_core::settings::Settings;
use criterion::{Criterion, criterion_group, criterion_main};
use pollster::block_on;
use simulation::Simulation;
use web_time::Duration;

fn setup_ava_mal() -> Simulation {
    let settings = Settings {
        dem_path: Some("../../frontend/data/avaframe/avaMal.png".to_string()),
        release_areas_path: Some(
            "../../frontend/data/avaframe/avaMalreleaseTexture.png".to_string(),
        ),
        ..Default::default()
    };
    let mut simulation: Simulation = block_on(Simulation::new()).unwrap();
    block_on(simulation.create(settings)).unwrap();
    simulation
}
fn setup_vals() -> Simulation {
    let settings = Settings {
        dem_path: Some("../../data/vals/PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif".to_string()),
        ..Default::default()
    };
    let mut simulation: Simulation = block_on(Simulation::new()).unwrap();
    block_on(simulation.create(settings)).unwrap();
    simulation
}

fn benchmark_avamal(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Benchmarks");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(20));
    let mut sim = setup_ava_mal();
    group.bench_function("gpu_sim_run", |b| {
        b.iter(|| {
            // block_on ensures the closure doesn't return
            // until the future (the simulation) is complete.
            pollster::block_on(async {
                let _ = sim.run().await;
                // CRITICAL: You still need to sync with the GPU here
                // e.g., sim.device.poll(wgpu::Maintain::Wait);
            })
        });
    });
}

fn benchmark_vals(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Benchmarks");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(150));
    let mut sim = setup_vals();
    group.bench_function("gpu_sim_run", |b| {
        b.iter(|| {
            // block_on ensures the closure doesn't return
            // until the future (the simulation) is complete.
            pollster::block_on(async {
                let _ = sim.run().await;
                // CRITICAL: You still need to sync with the GPU here
                // e.g., sim.device.poll(wgpu::Maintain::Wait);
            })
        });
    });
}

criterion_group!(benches, benchmark_vals, benchmark_avamal);
criterion_main!(benches);
