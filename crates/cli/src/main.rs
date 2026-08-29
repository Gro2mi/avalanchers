// compute_cli/src/main.rs
use anyhow::Result;
use clap::Parser;
use compute_core::settings::Settings;
#[allow(unused_imports)]
use compute_core::utils::{MaxValue, timer_checkpoint, timer_get_summary, timer_new};
use pollster::block_on;
use simulation::{Simulation, init_logging};
#[allow(unused_imports)]
use std::path::{Path, PathBuf};
use std::{env, time::Instant};
use tracing::{debug, error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "Avalanche Simulation")]
struct Args {
    /// Path to the input file
    #[arg()]
    file_path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    timer_new();
    init_logging();
    let start = Instant::now();
    match env::current_dir() {
        Ok(path) => debug!("Current working directory: {}", path.display()),
        Err(e) => error!("Failed to get current directory: {}", e),
    }
    let args = Args::parse();
    let file_path = match &args.file_path {
        Some(path) if path.exists() && path.is_file() => {
            info!("File path: {}", path.display());
            path.clone()
        }
        Some(path) => {
            error!(
                "Warning: File does not exist: {}. Using settings.json instead.",
                path.display()
            );
            PathBuf::from("settings.json")
        }
        None => {
            warn!(
                "No file path provided. Using {}/settings.json instead.",
                env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .display()
            );
            PathBuf::from("settings.json")
        }
    };
    timer_checkpoint("Startup");

    let settings = Settings::from_json(&file_path.to_string_lossy())
        .expect("Failed to load settings from JSON file");

    let mut simulation: Simulation = block_on(Simulation::new_with_settings(settings.clone()))?;

    block_on(simulation.run())?;
    timer_checkpoint("Fetch data from GPU");

    block_on(simulation.fetch_peak_velocity()).expect("Failed to get peak velocity");

    block_on(simulation.fetch_peak_flow_thickness()).expect("Failed to get peak flow thickness");

    let peak_velocity: Vec<f32> = block_on(simulation.fetch_peak_velocity())
        .expect("Failed to get peak velocity")
        .to_vec();
    info!(
        "Peak velocity during simulation: {:.2} m/s",
        peak_velocity.max_value().unwrap(),
    );

    // info!("{}", timer_get_summary());

    {
        let vel = block_on(simulation.fetch_particles_velocity())
            .unwrap()
            .clone();
        let pos = block_on(simulation.fetch_particles_position())
            .unwrap()
            .clone();
        let stopped = block_on(simulation.fetch_particles_stopped())
            .unwrap()
            .clone();
        let max_speed = vel
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f32, f32::max);
        debug!("DBG max particle speed: {}", max_speed);
        debug!("DBG first positions: {:?}", &pos[..5.min(pos.len())]);
        debug!("DBG first velocities: {:?}", &vel[..5.min(vel.len())]);
        let n_stopped = stopped.iter().filter(|&&s| s != 0).count();
        let max_stop_step = stopped.iter().copied().max().unwrap_or(0);
        debug!(
            "DBG stopped {} / {}, max stop marker {}",
            n_stopped,
            stopped.len(),
            max_stop_step
        );
        debug!(
            "DBG sim_info {:?}",
            block_on(simulation.fetch_sim_info()).unwrap()
        );
        debug!(
            "DBG atomics {:?}",
            block_on(simulation.fetch_atomic_values()).unwrap()
        );
        debug!(
            "DBG debug buffer {:?}",
            &block_on(simulation.get_compute_particles_debug()).unwrap()[..20]
        );
    }
    info!(
        "Total mass of particles: {:.2} kg",
        block_on(simulation.get_total_mass()).unwrap()
    );
    info!(
        "Total release volume: {:.2} m3",
        block_on(simulation.get_total_volume()).unwrap()
    );
    simulation.print_grid(&peak_velocity, 20, 20);
    let duration = start.elapsed();

    info!("Time elapsed is: {:?}", duration);
    Ok(())
}
