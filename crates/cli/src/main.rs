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

    let mut simulation: Simulation = block_on(Simulation::new())?;
    block_on(simulation.create(settings.clone()))?;

    block_on(simulation.run())?;
    timer_checkpoint("Fetch data from GPU");

    block_on(simulation.fetch_peak_velocity()).expect("Failed to get peak velocity");

    block_on(simulation.fetch_peak_flow_thickness()).expect("Failed to get peak flow thickness");

    let peak_velocity: Vec<f32> = block_on(simulation.fetch_peak_flow_thickness())
        .expect("Failed to get peak velocity")
        .to_vec();
    info!(
        "Peak velocity during simulation: {:.2} m/s",
        peak_velocity.max_value().unwrap(),
    );
    // timer_checkpoint("Write data to disk");
    // let bytes_v: &[u8] = unsafe {
    //     std::slice::from_raw_parts(
    //         peak_velocity.as_ptr() as *const u8,
    //         peak_velocity.len() * std::mem::size_of::<f32>(),
    //     )
    // };
    // data_processor::write_bin(Path::new("peak_velocity.bin"), bytes_v);
    // let peak_flow_thickness = block_on(simulation.fetch_peak_flow_thickness()).expect("Failed to get peak flow thickness");
    // let bytes_f: &[u8] = unsafe {
    //     std::slice::from_raw_parts(
    //         peak_flow_thickness.as_ptr() as *const u8,
    //         peak_flow_thickness.len() * std::mem::size_of::<f32>(),
    //     )
    // };
    // data_processor::write_bin(Path::new("peak_flow_thickness.bin"), bytes_f);

    // let cell_count = block_on(simulation.fetch_cell_count()).expect("Failed to get cell count");
    // let bytes_c: &[u8] = unsafe {
    //     std::slice::from_raw_parts(
    //         cell_count.as_ptr() as *const u8,
    //         cell_count.len() * std::mem::size_of::<f32>(),
    //     )
    // };
    // data_processor::write_bin(Path::new("cell_count.bin"), bytes_c);

    // info!("{}", timer_get_summary());

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
