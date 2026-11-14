// compute_cli/src/main.rs
use anyhow::{Context, Result, bail};
use clap::Parser;
use compute_core::settings::{
    ConstitutiveModel, CrownLineMethod, FrictionModel, ParticlePlacement, ReleaseFillMethod,
    Settings, SimModel,
};
use compute_core::utils::{MaxValue, timer_checkpoint, timer_new};
use pollster::block_on;
use simulation::{LogLevel, Simulation, init_logging, set_log_level};
#[allow(unused_imports)]
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{env, time::Instant};
use tracing::{Level, debug, info};

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

fn parse_crown_line_method(value: &str) -> Result<CrownLineMethod> {
    CrownLineMethod::from_str(value)
        .map_err(|err| anyhow::anyhow!("invalid crown line method '{value}': {err}"))
}

fn parse_release_fill_method(value: &str) -> Result<ReleaseFillMethod> {
    ReleaseFillMethod::from_str(value)
        .map_err(|err| anyhow::anyhow!("invalid release fill method '{value}': {err}"))
}

fn parse_particle_placement(value: &str) -> Result<ParticlePlacement> {
    ParticlePlacement::from_str(value)
        .map_err(|err| anyhow::anyhow!("invalid particle placement '{value}': {err}"))
}

fn parse_log_level(value: &str) -> Result<LogLevel> {
    LogLevel::from_str(value).map_err(|err| anyhow::anyhow!(err))
}

fn parse_constitutive_model(value: &str) -> Result<ConstitutiveModel> {
    ConstitutiveModel::from_str(value)
        .map_err(|err| anyhow::anyhow!("invalid constitutive model '{value}': {err}"))
}

#[derive(Parser, Debug)]
#[command(
    name = "avalanchers",
    // `<version>+<short commit>[.dirty]`, the build written to result stores.
    version = data_processor::store_schema::SOFTWARE_VERSION,
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

    /// Log level: off, error, warn, info, debug or trace (default: warn in
    /// release builds, info in debug builds, or the AVALANCHERS_LOG
    /// environment variable). Results are printed at every level.
    #[arg(long, value_parser = parse_log_level, value_name = "LEVEL")]
    log_level: Option<LogLevel>,

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
    #[arg(long)]
    particle_sort_interval: Option<u32>,
    #[arg(long)]
    momentum_stop_fraction: Option<f32>,
    #[arg(long)]
    no_new_cells_travel: Option<f32>,
    #[arg(long)]
    reference_surface_length: Option<f32>,
    #[arg(long, value_parser = parse_friction_model)]
    friction_model: Option<FrictionModel>,
    #[arg(long)]
    released_particles_per_cell: Option<u32>,
    #[arg(long)]
    density: Option<f32>,
    #[arg(long)]
    release_thickness: Option<f32>,
    #[arg(long)]
    release_swe: Option<f32>,
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
    #[arg(long, value_parser = parse_constitutive_model)]
    constitutive_model: Option<ConstitutiveModel>,
    #[arg(long)]
    shear_modulus: Option<f32>,
    #[arg(long)]
    hardening_modulus: Option<f32>,
    #[arg(long)]
    bulk_modulus: Option<f32>,
    #[arg(long)]
    compaction_pressure: Option<f32>,
    #[arg(long)]
    cfl: Option<f32>,
    #[arg(long)]
    min_slope_angle: Option<f32>,
    #[arg(long)]
    max_slope_angle: Option<f32>,
    #[arg(long)]
    release_min_elevation: Option<f32>,
    #[arg(long)]
    release_max_elevation: Option<f32>,
    #[arg(long)]
    release_area_fraction: Option<f32>,
    #[arg(long, value_parser = parse_crown_line_method)]
    crown_line_method: Option<CrownLineMethod>,
    #[arg(long)]
    crown_line_exit_distance: Option<f32>,
    #[arg(long)]
    crown_elevation_band: Option<f32>,
    #[arg(long)]
    min_particles_per_crown_cell: Option<u32>,
    #[arg(long, value_parser = parse_release_fill_method)]
    release_fill_method: Option<ReleaseFillMethod>,
    #[arg(long)]
    release_slope_weight: Option<f32>,
    #[arg(long)]
    release_slope_peak: Option<f32>,
    #[arg(long)]
    release_slope_width: Option<f32>,
    #[arg(long)]
    release_slope_smoothing_length: Option<f32>,
    #[arg(long)]
    velocity_threshold: Option<f32>,
    #[arg(long)]
    roughness_threshold: Option<f32>,
    #[arg(long)]
    peak_flow_thickness_threshold: Option<f32>,
    #[arg(long, value_parser = parse_bool)]
    enable_curvature: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_particle_interaction: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_earth_pressure_coefficient: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_entrainment: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_weak_pressure: Option<bool>,
    #[arg(long, value_parser = parse_bool)]
    enable_center_of_mass: Option<bool>,
    /// Integer (fixed point) grid atomics even on GPUs with float atomics;
    /// `--integer-atomics` alone means true
    #[arg(long, value_parser = parse_bool, num_args = 0..=1, default_missing_value = "true")]
    integer_atomics: Option<bool>,
    /// random, relaxed, uniform-random or uniform-relaxed
    #[arg(long, value_parser = parse_particle_placement)]
    particle_placement: Option<ParticlePlacement>,
    #[arg(long, value_parser = parse_bool)]
    center_of_mass_biggest_blob: Option<bool>,
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
        if let Some(value) = self.release_area_fraction {
            settings.release_area_fraction = Some(value);
        }
        if let Some(value) = self.crown_line_method {
            settings.crown_line_method = Some(value);
        }
        if let Some(value) = self.crown_line_exit_distance {
            settings.crown_line_exit_distance = Some(value);
        }
        if let Some(value) = self.crown_elevation_band {
            settings.crown_elevation_band = Some(value);
        }
        if let Some(value) = self.min_particles_per_crown_cell {
            settings.min_particles_per_crown_cell = Some(value);
        }
        if let Some(value) = self.release_fill_method {
            settings.release_fill_method = Some(value);
        }
        if let Some(value) = self.release_slope_weight {
            settings.release_slope_weight = Some(value);
        }
        if let Some(value) = self.release_slope_peak {
            settings.release_slope_peak = Some(value);
        }
        if let Some(value) = self.release_slope_width {
            settings.release_slope_width = Some(value);
        }
        if let Some(value) = self.release_slope_smoothing_length {
            settings.release_slope_smoothing_length = Some(value);
        }
        if let Some(value) = self.enable_center_of_mass {
            settings.enable_center_of_mass = Some(value);
        }
        if let Some(value) = self.integer_atomics {
            settings.integer_atomics = Some(value);
        }
        if let Some(value) = self.particle_placement {
            settings.particle_placement = Some(value);
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
        if let Some(value) = self.particle_sort_interval {
            settings.particle_sort_interval = Some(value);
        }
        if let Some(value) = self.momentum_stop_fraction {
            settings.momentum_stop_fraction = Some(value);
        }
        if let Some(value) = self.no_new_cells_travel {
            settings.no_new_cells_travel = Some(value);
        }
        if let Some(value) = self.reference_surface_length {
            settings.reference_surface_length = Some(value);
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
        if let Some(value) = self.release_thickness {
            settings.release_thickness = Some(value);
        }
        if let Some(value) = self.release_swe {
            settings.release_swe = Some(value);
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
        if let Some(value) = self.constitutive_model {
            settings.constitutive_model = Some(value);
        }
        if let Some(value) = self.shear_modulus {
            settings.shear_modulus = Some(value);
        }
        if let Some(value) = self.hardening_modulus {
            settings.hardening_modulus = Some(value);
        }
        if let Some(value) = self.bulk_modulus {
            settings.bulk_modulus = Some(value);
        }
        if let Some(value) = self.compaction_pressure {
            settings.compaction_pressure = Some(value);
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
        if let Some(value) = self.release_max_elevation {
            settings.release_max_elevation = Some(value);
        }
        if let Some(value) = self.velocity_threshold {
            settings.velocity_threshold = Some(value);
        }
        if let Some(value) = self.roughness_threshold {
            settings.roughness_threshold = Some(value);
        }
        if let Some(value) = self.peak_flow_thickness_threshold {
            settings.peak_flow_thickness_threshold = Some(value);
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
        if let Some(value) = self.enable_weak_pressure {
            settings.enable_weak_pressure = Some(value);
        }
        if let Some(value) = self.center_of_mass_biggest_blob {
            settings.center_of_mass_biggest_blob = Some(value);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    timer_new();
    let start = Instant::now();
    let args = Args::parse();
    match args.log_level {
        Some(level) => set_log_level(level),
        None => init_logging(),
    }
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
    let file_path = match &args.file_path {
        Some(path) if path.is_file() => path.clone(),
        Some(path) => bail!("Settings file {} does not exist", path.display()),
        None => {
            let path = PathBuf::from("settings.json");
            info!(
                "No settings file given; using {}",
                env::current_dir()
                    .map(|dir| dir.join(&path))
                    .unwrap_or_else(|_| path.clone())
                    .display()
            );
            path
        }
    };
    debug!("Reading settings from {}", file_path.display());
    timer_checkpoint("Startup");

    let mut settings = Settings::from_json(&file_path.to_string_lossy())
        .with_context(|| format!("Could not read the settings file {}", file_path.display()))?;
    args.apply_overrides(&mut settings)?;

    let mut simulation: Simulation = block_on(Simulation::new_with_settings(settings.clone()))?;

    block_on(simulation.run())?;
    timer_checkpoint("Fetch data from GPU");

    // caches the peak fields for saving
    block_on(simulation.fetch_peak_flow_thickness())
        .context("Could not read the peak flow thickness from the GPU")?;
    let peak_velocity: Vec<f32> = block_on(simulation.fetch_peak_velocity())
        .context("Could not read the peak velocity from the GPU")?
        .to_vec();

    // particle details for diagnosing a run; read back only when shown
    if tracing::enabled!(Level::DEBUG) {
        log_particle_details(&mut simulation)?;
    }

    let saved = block_on(simulation.save()).context("Could not save the results")?;
    print_summary(&mut simulation, &peak_velocity)?;
    println!();
    println!("Saved to {} (run {})", saved.store, saved.run_id);
    println!("Finished in {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}

/// One line of the summary: a label column and the value.
fn row(label: &str, value: impl std::fmt::Display) {
    println!("  {label:<21}{value}");
}

/// An area in m², or in km² from 0.1 km² on.
fn format_area(square_metres: f64) -> String {
    if square_metres >= 1.0e5 {
        format!("{:.2} km²", square_metres / 1.0e6)
    } else {
        format!("{square_metres:.0} m²")
    }
}

/// Compass direction of an aspect in degrees (0° = north, clockwise).
fn compass(aspect_deg: f32) -> &'static str {
    const POINTS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    POINTS[((aspect_deg.rem_euclid(360.0) + 22.5) / 45.0) as usize % 8]
}

/// Angle in degrees of a drop over a horizontal distance (0 if there is none).
fn angle_deg(drop: f64, distance: f64) -> f64 {
    if distance > 0.0 {
        drop.atan2(distance).to_degrees()
    } else {
        0.0
    }
}

/// Prints the setup and the results of a finished run. Post-processes the
/// simulation (the avalanche mask), so it runs after saving.
fn print_summary(simulation: &mut Simulation, peak_velocity: &[f32]) -> Result<()> {
    let settings = simulation.settings;
    let cell_area = f64::from(settings.cell_size).powi(2);
    let sim_info = block_on(simulation.fetch_sim_info())?;
    let peak_flow_thickness = block_on(simulation.fetch_peak_flow_thickness())?
        .max_value()
        .unwrap_or(0.0);
    let release_cells = block_on(simulation.fetch_release_areas())?
        .iter()
        .filter(|&&thickness| thickness > compute_core::RELEASE_THICKNESS_THRESHOLD)
        .count();
    let release_terrain = block_on(simulation.release_area_terrain())?;
    let mass = block_on(simulation.get_total_mass())?;
    let volume = block_on(simulation.get_total_volume())?;
    let run = block_on(simulation.run_summary())?;
    let outline_cells = simulation.roi.iter().filter(|&&inside| inside).count();
    let has_outline = outline_cells > 0 && outline_cells < simulation.roi.len();

    println!();
    println!("Setup");
    let dem = &simulation.dem;
    let dem_name = if simulation.dem_path.is_empty() {
        String::new()
    } else {
        format!("{}, ", simulation.dem_path)
    };
    row(
        "DEM",
        format!(
            "{dem_name}{} x {} cells of {} m",
            dem.width, dem.height, dem.cell_size
        ),
    );
    let model = SimModel::from_int(settings.sim_model)
        .map_or_else(|| settings.sim_model.to_string(), |model| model.to_string());
    row("Model", model);
    let friction = match FrictionModel::from_int(settings.friction_model) {
        Some(FrictionModel::Coulomb) => format!("coulomb (μ {})", settings.friction_coefficient),
        Some(model @ (FrictionModel::Voellmy | FrictionModel::VoellmyMinShear)) => format!(
            "{model} (μ {}, ξ {} m/s²)",
            settings.friction_coefficient, settings.drag_coefficient
        ),
        Some(model) => model.to_string(),
        None => settings.friction_model.to_string(),
    };
    row("Friction", friction);
    row(
        "Release",
        format!(
            "{release_cells} cells ({}), {volume:.0} m³, {:.0} t of snow",
            format_area(release_cells as f64 * cell_area),
            mass / 1000.0
        ),
    );
    row(
        "Release terrain",
        format!(
            "slope {:.1}° (steepest {:.1}°), aspect {:.0}° ({}), elevation {:.0} m",
            release_terrain.slope_angle,
            release_terrain.max_slope_angle,
            release_terrain.slope_aspect,
            if release_terrain.slope_aspect.is_finite() {
                compass(release_terrain.slope_aspect)
            } else {
                "-"
            },
            release_terrain.elevation
        ),
    );
    row(
        "Particles",
        format!(
            "{} ({} per release cell, {})",
            simulation.number_particles(),
            settings.released_particles_per_cell,
            simulation.particle_placement
        ),
    );
    row(
        "Run",
        format!(
            "{} steps, {:.1} s of flow; {}",
            sim_info.timestep,
            sim_info.elapsed_time,
            sim_info
                .stop_reason()
                .map_or("stopped at max_steps".to_string(), |reason| format!(
                    "stopped because {reason}"
                ))
        ),
    );
    let states = block_on(simulation.fetch_particles_state())?;
    let count = |test: fn(&compute_core::ParticleState) -> bool| {
        states.iter().filter(|state| test(state)).count()
    };
    let stopped_early = [
        (
            count(|s| s.unphysical_speed),
            format!("faster than {} m/s", compute_core::MAX_PARTICLE_SPEED),
        ),
        (
            count(|s| s.is_nan),
            "with an invalid (NaN) state".to_string(),
        ),
        (
            count(|s| s.out_of_dem_data && !s.is_nan),
            "outside the DEM data".to_string(),
        ),
        (
            count(|s| s.out_of_bounds),
            "at the edge of the DEM".to_string(),
        ),
    ]
    .into_iter()
    .filter(|(number, _)| *number > 0)
    .map(|(number, reason)| format!("{number} {reason}"))
    .collect::<Vec<_>>();
    if !stopped_early.is_empty() {
        row("Stopped early", stopped_early.join(", "));
    }

    println!();
    println!("Results");
    row(
        "Peak velocity",
        format!("{:.1} m/s", peak_velocity.max_value().unwrap_or(0.0)),
    );
    row("Peak flow thickness", format!("{peak_flow_thickness:.2} m"));

    // the avalanche: the largest connected area above the thickness threshold
    block_on(simulation.post_process())?;
    let affected_cells = simulation.ava_mask.iter().filter(|&&set| set).count();
    row(
        "Affected area",
        format!(
            "{} ({affected_cells} cells with peak flow thickness ≥ {} m)",
            format_area(affected_cells as f64 * cell_area),
            settings.peak_flow_thickness_threshold
        ),
    );
    let (runout_distance, runout_drop) = simulation
        .dem
        .get_elevation_extrema_distance_and_drop(&simulation.ava_mask)
        .unwrap_or((0.0, 0.0));
    row(
        "Runout",
        format!(
            "{runout_distance:.0} m horizontal, {runout_drop:.0} m drop, angle {:.1}° (top to \
             lowest affected cell)",
            angle_deg(f64::from(runout_drop), f64::from(runout_distance))
        ),
    );
    if simulation.enable_center_of_mass && run.travel_angle.is_finite() {
        row(
            "Center of mass",
            format!(
                "travel length {:.0} m (horizontal {:.0} m), drop {:.0} m, travel angle {:.1}°",
                run.travel_length_3d,
                run.travel_length_2d,
                run.travel_vertical_drop,
                run.travel_angle
            ),
        );
    } else {
        row(
            "Center of mass",
            "not tracked (set enable_center_of_mass to true for the travel length and angle)",
        );
    }

    if has_outline {
        let evaluation = block_on(simulation.evaluate())?;
        println!();
        println!("Compared with the outline ({outline_cells} cells)");
        row(
            "Overlap",
            format!(
                "{:.0} % shared (IoU), {:.0} % only simulated, {:.0} % only in the outline \
                 (of the combined area without the release)",
                evaluation.intersection * 100.0,
                evaluation.overshoot * 100.0,
                evaluation.undershoot * 100.0
            ),
        );
        row(
            "Runout",
            format!(
                "{:.0} m horizontal, {:.0} m drop, angle {:.1}° (outline)",
                evaluation.horizontal_distance_ref,
                evaluation.vertical_drop_ref,
                angle_deg(
                    evaluation.vertical_drop_ref,
                    evaluation.horizontal_distance_ref
                )
            ),
        );
    }

    println!();
    println!("Peak velocity map (denser characters are faster)");
    simulation.print_grid(peak_velocity, 40, 20)?;
    Ok(())
}

/// Debug output of the particle state at the end of a run.
fn log_particle_details(simulation: &mut Simulation) -> Result<()> {
    let velocities = block_on(simulation.fetch_particles_velocity())?.clone();
    let positions = block_on(simulation.fetch_particles_position())?.clone();
    let states = block_on(simulation.fetch_particles_state())?.clone();
    let max_speed = velocities
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .fold(0.0f32, f32::max);
    let number_stopped = states.iter().filter(|state| state.stopped).count();
    let last_stop_step = states.iter().map(|state| state.timestep).max().unwrap_or(0);
    debug!("Largest particle speed at the end: {max_speed:.2} m/s");
    debug!(
        "Stopped particles: {number_stopped} of {}, the last one at step {last_stop_step}",
        states.len()
    );
    debug!(
        "First particle positions: {:?}",
        &positions[..5.min(positions.len())]
    );
    debug!(
        "First particle velocities: {:?}",
        &velocities[..5.min(velocities.len())]
    );
    // simulation info and atomic values: logged by the run itself
    let debug_buffer = block_on(simulation.get_compute_particles_debug())?;
    debug!(
        "Particle debug buffer: {:?}",
        &debug_buffer[..20.min(debug_buffer.len())]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::settings::{FrictionModel, SimModel};

    #[test]
    fn cli_args_override_settings_file() {
        let mut settings = Settings {
            max_steps: Some(10),
            density: Some(123.4),
            sim_model: Some(SimModel::TerrainFollowing),
            friction_model: Some(FrictionModel::Voellmy),
            enable_curvature: Some(false),
            ..Default::default()
        };

        let args = Args {
            file_path: None,
            about: false,
            list_devices: false,
            log_level: None,
            max_steps: Some(42),
            sim_model: Some(SimModel::TerrainFollowing),
            friction_model: Some(FrictionModel::Coulomb),
            density: Some(456.7),
            enable_curvature: Some(true),
            outlines_path: None,
            outlines_padding: None,
            dem_path: None,
            release_areas_path: None,
            output_path: None,
            batch_compute_steps: None,
            particle_sort_interval: None,
            momentum_stop_fraction: None,
            no_new_cells_travel: None,
            reference_surface_length: None,
            released_particles_per_cell: None,
            release_thickness: None,
            release_swe: None,
            friction_coefficient: None,
            drag_coefficient: None,
            n0: None,
            i0: None,
            mu0: None,
            mu2: None,
            grain_diameter: None,
            internal_friction_angle: None,
            basal_friction_angle: None,
            constitutive_model: None,
            shear_modulus: None,
            hardening_modulus: None,
            bulk_modulus: None,
            compaction_pressure: None,
            cfl: None,
            min_slope_angle: None,
            max_slope_angle: None,
            release_min_elevation: None,
            release_max_elevation: Some(3000.0),
            velocity_threshold: None,
            roughness_threshold: None,
            peak_flow_thickness_threshold: None,
            enable_particle_interaction: None,
            enable_earth_pressure_coefficient: None,
            enable_entrainment: None,
            enable_weak_pressure: None,
            center_of_mass_biggest_blob: None,
            release_area_fraction: None,
            crown_line_method: None,
            crown_line_exit_distance: None,
            crown_elevation_band: None,
            min_particles_per_crown_cell: None,
            release_fill_method: None,
            release_slope_weight: None,
            release_slope_peak: None,
            release_slope_width: None,
            release_slope_smoothing_length: None,
            enable_center_of_mass: None,
            integer_atomics: None,
            particle_placement: Some(ParticlePlacement::UniformRelaxed),
        };

        args.apply_overrides(&mut settings)
            .expect("Failed to apply args");

        assert_eq!(settings.max_steps, Some(42));
        assert_eq!(settings.density, Some(456.7));
        assert_eq!(settings.sim_model, Some(SimModel::TerrainFollowing));
        assert_eq!(settings.friction_model, Some(FrictionModel::Coulomb));
        assert_eq!(settings.enable_curvature, Some(true));
        assert_eq!(settings.release_max_elevation, Some(3000.0));
        assert_eq!(
            settings.particle_placement,
            Some(ParticlePlacement::UniformRelaxed)
        );
    }
}
