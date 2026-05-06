// compute_cli/src/main.rs
use anyhow::Result;
use clap::Parser;
use compute_core::settings::Settings;
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

    let mut simulation: Simulation = pollster::block_on(Simulation::new())?;
    pollster::block_on(simulation.create(settings))?;

    pollster::block_on(simulation.run())?;

    let duration = start.elapsed();

    info!("Time elapsed is: {:?}", duration);
    Ok(())
}
