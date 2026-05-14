// compute_cli/src/main.rs
use anyhow::Result;
use clap::Parser;
use compute_core::settings::Settings;
use compute_core::utils::MaxValue;
use pollster::block_on;
use simulation::{Simulation, init_logging};
use std::path::PathBuf;
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

    let settings = Settings::from_json(&file_path.to_string_lossy())
        .expect("Failed to load settings from JSON file");

    let mut simulation: Simulation = block_on(Simulation::new())?;
    block_on(simulation.create(settings))?;

    block_on(simulation.run())?;
    let peak_velocity =
        block_on(simulation.fetch_peak_velocity()).expect("Failed to get peak velocity");
    info!(
        "Peak velocity during simulation: {:.2} m/s",
        peak_velocity.max_value().unwrap(),
    );

    let sim_info = block_on(simulation.fetch_sim_info()).expect("Failed to get sim info");
    info!("{:#?}", sim_info);
    let atomic_values =
        block_on(simulation.fetch_atomic_values()).expect("Failed to get atomic values");
    info!("{:#?}", atomic_values);
    let duration = start.elapsed();

    info!("Time elapsed is: {:?}", duration);
    Ok(())
}
