// compute_cli/src/main.rs
use anyhow::Result;
use clap::Parser;
use compute_core::utils::*;
use compute_core::*;
use compute_core::{
    dem::Dem,
    settings::{Settings, SimSettings},
}; // Import from your new crate
use data_processor::*;
use std::path::PathBuf;
use std::{env, time::Instant};
use tracing::{debug, error, info, trace, warn};

#[derive(Parser, Debug)]
#[command(name = "Avalanche Simulation")]
struct Args {
    /// Path to the input file
    #[arg()]
    file_path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    compute_core::init_logging();
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
            error!("Warning: No file path provided. Using settings.json instead.");
            PathBuf::from("settings.json")
        }
    };

    let settings = Settings::from_json(&file_path.to_string_lossy())
        .expect("Failed to load settings from JSON file");

    info!("Loaded settings: {:?}", settings);
    let dem_path = std::path::PathBuf::from(&settings.dem_path);
    if !dem_path.exists() {
        error!("DEM file does not exist: {}", settings.dem_path);
        std::process::exit(1);
    }
    let dem = Dem::load_png_as_float32(dem_path);
    let sim_settings = SimSettings::from_settings(&settings, &dem);
    // sim_settings.set_dem(&dem);
    info!("Loaded simSettings: {:?}", sim_settings);

    // SimSettings::new()
    //     .to_json(file_path)
    //     .expect("Failed to write settings to JSON file");

    // dem = Dem::load_png_as_float32(&mut self, casename)

    // block_on(async {
    //     // Use the high-level orchestrator function
    //     let output_data = ComputeOrchestrator::initialize_and_run(
    //         data_len,
    //         Some(&input_data),
    //         args.do_addition,
    //     ).await?;

    //     let output_string = output_data.iter()
    //         .map(|f| f.to_string())
    //         .collect::<Vec<String>>()
    //         .join(",");

    //     if let Some(output_file) = args.output {
    //         std::fs::write(&output_file, output_string.as_bytes())?;
    //         println!("Results written to: {}", output_file);
    //     } else {
    //         println!("Input: {:?}", input_data);
    //         println!("Output: {}", output_string);
    //     }

    //     Ok(())
    // })
    let duration = start.elapsed();

    info!("Time elapsed is: {:?}", duration);
    Ok(())
}
