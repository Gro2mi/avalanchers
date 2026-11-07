//! Computation time of a full run of the grid flow models on the avaWog test case.
//!
//! cargo bench -p simulation --bench model_benchmark
//! cargo bench -p simulation --bench model_benchmark -- terrain-following
//!
//! Every iteration restores the initial particle state outside the timed region
//! and then times `run()` until the simulation stops by itself, so terrain
//! analysis and particle seeding are not part of the measurement.

use compute_core::settings::{Settings, SimModel};
use criterion::{Criterion, criterion_group, criterion_main};
use pollster::block_on;
use simulation::Simulation;
use web_time::{Duration, Instant};

const MODELS: [SimModel; 3] = [
    SimModel::TerrainFollowing,
    SimModel::ThreeD,
    SimModel::Curvilinear,
];

fn setup_ava_wog(model: SimModel) -> Simulation {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/avaframe");
    let settings = Settings {
        dem_path: Some(format!("{data}/avaWog.png")),
        release_areas_path: Some(format!("{data}/avaWogreleaseTexture.png")),
        sim_model: Some(model),
        ..Default::default()
    };
    let mut simulation = block_on(Simulation::new()).expect("Failed to create simulation");
    block_on(simulation.create(settings)).expect("Failed to load avaWog");
    block_on(simulation.store_initial_state()).expect("Failed to store the initial state");
    simulation
}

fn benchmark_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("avaWog");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    for model in MODELS {
        let mut sim = setup_ava_wog(model);
        // warm up the pipelines, then one run for the step count and time per step
        block_on(sim.run()).expect("Failed to run");
        block_on(sim.restore_initial_state()).expect("Failed to restore");
        let start = Instant::now();
        block_on(sim.run()).expect("Failed to run");
        let elapsed = start.elapsed();
        let info = block_on(sim.fetch_sim_info()).expect("Failed to read sim info");
        println!(
            "{model}: {} particles, {} steps, {:.3} ms/step, stopped by {:?}",
            info.number_particles,
            info.timestep,
            elapsed.as_secs_f64() * 1e3 / info.timestep.max(1) as f64,
            info.parsed_flags()
        );

        group.bench_function(model.to_string(), |b| {
            b.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    block_on(sim.restore_initial_state()).expect("Failed to restore");
                    let start = Instant::now();
                    block_on(sim.run()).expect("Failed to run");
                    total += start.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_models);
criterion_main!(benches);
