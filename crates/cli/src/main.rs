// compute_cli/src/main.rs
use anyhow::Result;
use clap::Parser;
use compute_core::settings::{FrictionModel, Settings, SimModel};
#[allow(unused_imports)]
use compute_core::utils::{MaxValue, timer_checkpoint, timer_get_summary, timer_new};
use pollster::block_on;
use simulation::{Simulation, init_logging};
#[allow(unused_imports)]
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{env, time::Instant};
use tracing::{debug, error, info, warn};

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        other => anyhow::bail!("invalid boolean value '{other}', expected true/false"),
    }
}

fn parse_sim_model(value: &str) -> Result<SimModel> {
    SimModel::from_str(value).map_err(|err| anyhow::anyhow!("invalid sim model '{value}': {err}"))
}

fn parse_friction_model(value: &str) -> Result<FrictionModel> {
    FrictionModel::from_str(value)
        .map_err(|err| anyhow::anyhow!("invalid friction model '{value}': {err}"))
}

#[derive(Parser, Debug)]
#[command(
    name = "avalanchers",
    version,
    about = "Dense flow avalanche simulation on the GPU"
)]
struct Args {
    /// Path to the input file
    #[arg()]
    file_path: Option<std::path::PathBuf>,

    /// Show copyright and data attribution information
    #[arg(long)]
    about: bool,

    /// List available GPU devices
    #[arg(long)]
    list_devices: bool,

    #[arg(long)]
    outlines_path: Option<String>,
    #[arg(long)]
    outlines_padding: Option<f32>,
    #[arg(long)]
    dem_path: Option<String>,
    #[arg(long)]
    release_areas_path: Option<String>,
    #[arg(long)]
    output_path: Option<String>,
    #[arg(long)]
    max_steps: Option<u32>,
    #[arg(long, value_parser = parse_sim_model)]
    sim_model: Option<SimModel>,
    #[arg(long)]
    batch_compute_steps: Option<u32>,
    #[arg(long, value_parser = parse_friction_model)]
    friction_model: Option<FrictionModel>,
    #[arg(long)]
    released_particles_per_cell: Option<u32>,
    #[arg(long)]
    density: Option<f32>,
    #[arg(long)]
    slab_thickness_factor: Option<f32>,
    #[arg(long)]
    friction_coefficient: Option<f32>,
    #[arg(long)]
    drag_coefficient: Option<f32>,
    #[arg(long)]
    n0: Option<f32>,
    #[arg(long)]
    i0: Option<f32>,
    #[arg(long)]
    mu0: Option<f32>,
    #[arg(long)]
    mu2: Option<f32>,
    #[arg(long)]
    grain_diameter: Option<f32>,
    #[arg(long)]
    internal_friction_angle: Option<f32>,
    #[arg(long)]
    basal_friction_angle: Option<f32>,
    #[arg(long)]
    cfl: Option<f32>,
    #[arg(long)]
    min_slope_angle: Option<f32>,
    #[arg(long)]
    max_slope_angle: Option<f32>,
    #[arg(long)]
    release_min_elevation: Option<f32>,
    #[arg(long)]
    velocity_threshold: Option<f32>,
    #[arg(long)]
    roughness_threshold: Option<f32>,
    #[arg(long, value_parser = parse_bool)]
    enable_curvature: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_particle_interaction: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_earth_pressure_coefficient: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_entrainment: Option<bool>,
}

impl Args {
    fn apply_overrides(&self, settings: &mut Settings) -> Result<()> {
        if let Some(value) = self.outlines_path.clone() {
            settings.outlines_path = Some(value);
        }
        if let Some(value) = self.outlines_padding {
            settings.outlines_padding = Some(value);
        }
        if let Some(value) = self.dem_path.clone() {
            settings.dem_path = Some(value);
        }
        if let Some(value) = self.release_areas_path.clone() {
            settings.release_areas_path = Some(value);
        }
        if let Some(value) = self.output_path.clone() {
            settings.output_path = Some(value);
        }
        if let Some(value) = self.max_steps {
            settings.max_steps = Some(value);
        }
        if let Some(value) = self.sim_model {
            settings.sim_model = Some(value);
        }
        if let Some(value) = self.batch_compute_steps {
            settings.batch_compute_steps = Some(value);
        }
        if let Some(value) = self.friction_model {
            settings.friction_model = Some(value);
        }
        if let Some(value) = self.released_particles_per_cell {
            settings.released_particles_per_cell = Some(value);
        }
        if let Some(value) = self.density {
            settings.density = Some(value);
        }
        if let Some(value) = self.slab_thickness_factor {
            settings.slab_thickness_factor = Some(value);
        }
        if let Some(value) = self.friction_coefficient {
            settings.friction_coefficient = Some(value);
        }
        if let Some(value) = self.drag_coefficient {
            settings.drag_coefficient = Some(value);
        }
        if let Some(value) = self.n0 {
            settings.n0 = Some(value);
        }
        if let Some(value) = self.i0 {
            settings.i0 = Some(value);
        }
        if let Some(value) = self.mu0 {
            settings.mu0 = Some(value);
        }
        if let Some(value) = self.mu2 {
            settings.mu2 = Some(value);
        }
        if let Some(value) = self.grain_diameter {
            settings.grain_diameter = Some(value);
        }
        if let Some(value) = self.internal_friction_angle {
            settings.internal_friction_angle = Some(value);
        }
        if let Some(value) = self.basal_friction_angle {
            settings.basal_friction_angle = Some(value);
        }
        if let Some(value) = self.cfl {
            settings.cfl = Some(value);
        }
        if let Some(value) = self.min_slope_angle {
            settings.min_slope_angle = Some(value);
        }
        if let Some(value) = self.max_slope_angle {
            settings.max_slope_angle = Some(value);
        }
        if let Some(value) = self.release_min_elevation {
            settings.release_min_elevation = Some(value);
        }
        if let Some(value) = self.velocity_threshold {
            settings.velocity_threshold = Some(value);
        }
        if let Some(value) = self.roughness_threshold {
            settings.roughness_threshold = Some(value);
        }
        if let Some(value) = self.enable_curvature {
            settings.enable_curvature = Some(value);
        }
        if let Some(value) = self.enable_particle_interaction {
            settings.enable_particle_interaction = Some(value);
        }
        if let Some(value) = self.enable_earth_pressure_coefficient {
            settings.enable_earth_pressure_coefficient = Some(value);
        }
        if let Some(value) = self.enable_entrainment {
            settings.enable_entrainment = Some(value);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    timer_new();
    init_logging();
    let start = Instant::now();
    let args = Args::parse();
    if args.about {
        println!("avalanchers");
        println!("Copyright © 2026 Markus Rampp");
        println!();
        println!("Map data:");
        println!("  Austrian data source: basemap.at");
        println!("  https://www.basemap.at");
        println!("  Swiss data source: Federal Office of Topography swisstopo. © swisstopo");
        println!("  https://www.swisstopo.admin.ch");
        println!();
        println!("This program is licensed under the MIT License.");
        return Ok(());
    }
    if args.list_devices {
        let devices = block_on(compute_core::list_devices());
        for device in devices? {
            println!("{}", device);
        }
        return Ok(());
    }
    match env::current_dir() {
        Ok(path) => debug!("Current working directory: {}", path.display()),
        Err(e) => error!("Failed to get current directory: {}", e),
    }
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

    let mut settings = Settings::from_json(&file_path.to_string_lossy())
        .expect("Failed to load settings from JSON file");
    args.apply_overrides(&mut settings)?;

    let mut simulation: Simulation = block_on(Simulation::new())?;
    block_on(simulation.create(settings.clone()))?;

    block_on(simulation.run())?;
    timer_checkpoint("Fetch data from GPU");

    block_on(simulation.fetch_peak_velocity()).expect("Failed to get peak velocity");

    block_on(simulation.fetch_peak_flow_thickness()).expect("Failed to get peak flow thickness");

    block_on(simulation.fetch_cell_count()).expect("Failed to get cell count");

    let peak_velocity: Vec<f32> = block_on(simulation.fetch_peak_velocity())
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

    let particles = block_on(simulation.fetch_particles()).expect("Failed to get final positions");
    for particle in particles
        .iter()
        .filter(|p| p.velocity[0].is_nan() || p.velocity[1].is_nan() || p.velocity[2].is_nan())
    {
        info!(
            "Out of bounds particle: Position = ({:.2}, {:.2}, {:.2}), mass: {:.2}, velocity: ({:.2}, {:.2}, {:.2}), stopped: {}",
            // particle[0].stopped,
            particle.position[0],
            particle.position[1],
            particle.position[2],
            particle.mass,
            particle.velocity[0],
            particle.velocity[1],
            particle.velocity[2],
            particle.stopped
        );
    }
    let out_of_bounds_count = particles.iter().filter(|p| p.stopped > 100000).count();
    if out_of_bounds_count > 0 {
        warn!("{} particles stopped out of bounds.", out_of_bounds_count);
    }
    // for particle in particles.iter().filter(|p| p.stopped > 100000) {
    //     info!(
    //         "Out of bounds particle: Position = ({:.2}, {:.2}, {:.2}), mass: {:.2}, velocity: ({:.2}, {:.2}, {:.2}), stopped: {}",
    //         // particle[0].stopped,
    //         particle.position[0],
    //         particle.position[1],
    //         particle.position[2],
    //         particle.mass,
    //         particle.velocity[0],
    //         particle.velocity[1],
    //         particle.velocity[2],
    //         particle.stopped
    //     );
    // }
    info!(
        "Total mass of particles: {:.2} kg",
        block_on(simulation.get_release_mass()).unwrap()
    );
    info!(
        "Total release volume: {:.2} m3",
        block_on(simulation.get_release_volume()).unwrap()
    );
    simulation.print_grid(&peak_velocity, 20, 20);
    block_on(simulation.save()).expect("Failed to save simulation");
    let duration = start.elapsed();

    info!("Time elapsed is: {:?}", duration);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::settings::{FrictionModel, SimModel};

    #[test]
    fn cli_args_override_settings_file() {
        let mut settings = Settings::default();
        settings.max_steps = Some(10);
        settings.density = Some(123.4);
        settings.sim_model = Some(SimModel::ParticleInteraction);
        settings.friction_model = Some(FrictionModel::Voellmy);
        settings.enable_curvature = Some(false);

        let args = Args {
            file_path: None,
            about: false,
            list_devices: false,
            max_steps: Some(42),
            sim_model: Some(SimModel::Block),
            friction_model: Some(FrictionModel::Coulomb),
            density: Some(456.7),
            enable_curvature: Some(true),
            outlines_path: None,
            outlines_padding: None,
            dem_path: None,
            release_areas_path: None,
            output_path: None,
            batch_compute_steps: None,
            released_particles_per_cell: None,
            slab_thickness_factor: None,
            friction_coefficient: None,
            drag_coefficient: None,
            n0: None,
            i0: None,
            mu0: None,
            mu2: None,
            grain_diameter: None,
            internal_friction_angle: None,
            basal_friction_angle: None,
            cfl: None,
            min_slope_angle: None,
            max_slope_angle: None,
            release_min_elevation: None,
            velocity_threshold: None,
            roughness_threshold: None,
            enable_particle_interaction: None,
            enable_earth_pressure_coefficient: None,
            enable_entrainment: None,
        };

        args.apply_overrides(&mut settings)
            .expect("Failed to apply args");

        assert_eq!(settings.max_steps, Some(42));
        assert_eq!(settings.density, Some(456.7));
        assert_eq!(settings.sim_model, Some(SimModel::Block));
        assert_eq!(settings.friction_model, Some(FrictionModel::Coulomb));
        assert_eq!(settings.enable_curvature, Some(true));
    }
}
