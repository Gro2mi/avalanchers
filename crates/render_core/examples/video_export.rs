//! Headless video export of a live simulation.
//!
//! Loads the simulation from a settings.json (same format as the app and the
//! other native entry points), runs it without a window, renders every batch
//! of steps into an offscreen frame, and pipes the raw RGBA pixels into an
//! ffmpeg process for H.264 encoding. Requires `ffmpeg` on the PATH. Paths
//! inside settings.json resolve relative to the working directory.
//!
//! Usage:
//! `cargo run -p render_core --example video_export -- [settings.json] [out.mp4] [fps] [steps_per_frame] [exaggeration] [resolution] [overlay]`
//! example: cargo run -p render_core --example video_export -- settings.json my_sim.mp4 30 1 1.3 4k peak_flow_thickness
//! Defaults: `settings.json`, `simulation.mp4`, 30 fps, 1 step per frame, 1.3
//! vertical exaggeration, 4k, peak flow velocity overlay. `resolution` accepts
//! 480p/720p/1080p/1440p/2160p/4k or an explicit `WIDTHxHEIGHT`; `overlay`
//! accepts none, peak_velocity, peak_flow_thickness, grid_mass, release_areas,
//! slope_angle, slope_aspect or roughness. Particles are always drawn.

use std::io::Write;
use std::process::{Child, Command, Stdio};

use compute_core::settings::Settings;
use compute_core::{ComputeOrchestrator, SimInfo, buffers::BufferName};
use render_core::{OverlayRange, ParticleBuffers, Renderer, TerrainData};
use simulation::{Simulation, SimulationState};
use wgpu::util::DeviceExt;

const DEFAULT_SETTINGS: &str = "settings.json";
const DEFAULT_OUTPUT: &str = "simulation.mp4";
const DEFAULT_RESOLUTION: &str = "4k";
const MAX_VELOCITY: f32 = 30.0;

/// Resolution presets plus an explicit `WIDTHxHEIGHT` form.
fn parse_resolution(spec: &str) -> anyhow::Result<(u32, u32)> {
    let normalized = spec.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "480p" => Ok((854, 480)),
        "720p" => Ok((1280, 720)),
        "1080p" => Ok((1920, 1080)),
        "1440p" => Ok((2560, 1440)),
        "2160p" | "4k" => Ok((3840, 2160)),
        other => {
            let (width, height) = other
                .split_once(['x', 'X'])
                .ok_or_else(|| anyhow::anyhow!(
                    "unsupported resolution '{spec}': use 480p/720p/1080p/1440p/2160p/4k or WIDTHxHEIGHT"
                ))?;
            let (width, height) = (width.parse::<u32>()?, height.parse::<u32>()?);
            if width == 0 || height == 0 {
                anyhow::bail!("resolution dimensions must be positive, got '{spec}'");
            }
            Ok((width, height))
        }
    }
}

/// Grid buffers cloned from the simulation, mirroring the live viewer.
const GRID_BUFFERS: [BufferName; 7] = [
    BufferName::GridPeakVelocity,
    BufferName::GridPeakFlowThickness,
    BufferName::GridMass,
    BufferName::ReleaseAreas,
    BufferName::SlopeAngle,
    BufferName::SlopeAspect,
    BufferName::Roughness,
];

/// Scalar field tinting the terrain, with its colour ramp bounds. Names match
/// the wasm binding's `set_overlay`.
#[derive(Clone, Copy, PartialEq)]
enum Overlay {
    None,
    PeakFlowVelocity,
    PeakFlowThickness,
    GridMass,
    ReleaseAreas,
    SlopeAngle,
    SlopeAspect,
    Roughness,
}

impl Overlay {
    fn from_name(name: &str) -> anyhow::Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "peak_velocity" => Ok(Self::PeakFlowVelocity),
            "peak_flow_thickness" => Ok(Self::PeakFlowThickness),
            "grid_mass" => Ok(Self::GridMass),
            "release_areas" => Ok(Self::ReleaseAreas),
            "slope_angle" => Ok(Self::SlopeAngle),
            "slope_aspect" => Ok(Self::SlopeAspect),
            "roughness" => Ok(Self::Roughness),
            other => anyhow::bail!(
                "unknown overlay '{other}': use none, peak_velocity, peak_flow_thickness, \
                 grid_mass, release_areas, slope_angle, slope_aspect or roughness"
            ),
        }
    }

    fn buffer_name(self) -> Option<BufferName> {
        match self {
            Self::None => None,
            Self::PeakFlowVelocity => Some(BufferName::GridPeakVelocity),
            Self::PeakFlowThickness => Some(BufferName::GridPeakFlowThickness),
            Self::GridMass => Some(BufferName::GridMass),
            Self::ReleaseAreas => Some(BufferName::ReleaseAreas),
            Self::SlopeAngle => Some(BufferName::SlopeAngle),
            Self::SlopeAspect => Some(BufferName::SlopeAspect),
            Self::Roughness => Some(BufferName::Roughness),
        }
    }

    /// Colour ramp bounds in the units of each field, plus the legend label.
    fn range(self) -> OverlayRange {
        let range = match self {
            Self::None => OverlayRange::default(),
            Self::PeakFlowVelocity => OverlayRange::new(0.0, 40.0)
                .with_threshold(0.1)
                .with_unit("m/s"),
            Self::PeakFlowThickness => OverlayRange::new(0.0, 10.0)
                .with_threshold(0.01)
                .with_unit("m"),
            Self::GridMass => OverlayRange::new(0.0, 5_000.0)
                .with_threshold(1.0)
                .with_unit("kg"),
            // Slab thickness in metres; release textures typically hold 1.0 m.
            Self::ReleaseAreas => OverlayRange::new(0.0, 2.0)
                .with_threshold(0.01)
                .with_unit("m"),
            // Degrees; steeper than the default 60° release window saturates hot.
            Self::SlopeAngle => OverlayRange::new(0.0, 60.0).with_unit("deg"),
            // Degrees clockwise from north; flat cells hold -1 and stay bare.
            Self::SlopeAspect => OverlayRange::new(0.0, 360.0)
                .with_threshold(-0.5)
                .with_unit("deg"),
            // Dimensionless, 0 (smooth) to 1 (rough); border cells are forced to 1.
            Self::Roughness => OverlayRange::new(0.0, 1.0),
        };
        range.with_label(self.label())
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PeakFlowVelocity => "peak flow velocity",
            Self::PeakFlowThickness => "peak flow thickness",
            Self::GridMass => "grid mass",
            Self::ReleaseAreas => "release areas",
            Self::SlopeAngle => "slope angle",
            Self::SlopeAspect => "slope aspect",
            Self::Roughness => "roughness",
        }
    }
}

/// MPM keeps no vertical velocity buffer, so it gets a zero-filled stand-in and
/// colouring falls back to horizontal speed.
fn clone_particle_buffers(
    orchestrator: &ComputeOrchestrator,
    particle_count: u32,
) -> anyhow::Result<[wgpu::Buffer; 5]> {
    let clone_buffer = |name: BufferName| -> anyhow::Result<wgpu::Buffer> {
        orchestrator
            .resources
            .get_buffer(&name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("simulation buffer '{name}' is missing"))
    };

    let velocity_z = clone_buffer(BufferName::ParticlesVelocityZ).unwrap_or_else(|_| {
        orchestrator
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Zero Vertical Velocity"),
                contents: bytemuck::cast_slice(&vec![0.0f32; particle_count as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    });

    Ok([
        clone_buffer(BufferName::ParticlesPosition)?,
        clone_buffer(BufferName::ParticlesVelocity)?,
        velocity_z,
        clone_buffer(BufferName::ParticlesState)?,
        clone_buffer(BufferName::ParticlesElevation)?,
    ])
}

/// Spawns an ffmpeg process that encodes raw RGBA frames on stdin into an
/// H.264 mp4. Dropping `stdin` finalizes the file.
fn spawn_encoder(output: &str, width: u32, height: u32, fps: u32) -> anyhow::Result<Child> {
    Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| anyhow::anyhow!("failed to spawn ffmpeg (is it on the PATH?): {error}"))
}

struct HeadlessSimulation {
    simulation: Simulation,
    device: wgpu::Device,
    queue: wgpu::Queue,
    terrain: TerrainData,
    grids: Vec<(BufferName, wgpu::Buffer)>,
    particles: [wgpu::Buffer; 5],
    info: SimInfo,
}

fn start_simulation(settings: Settings, exaggeration: f32) -> anyhow::Result<HeadlessSimulation> {
    let mut simulation = pollster::block_on(Simulation::new())?;
    pollster::block_on(simulation.create(settings))?;
    // Lazily prepares the simulation and pipelines without advancing time.
    let info = pollster::block_on(simulation.run_n_steps(0))?;

    let terrain = TerrainData::new(
        simulation.dem.width as u32,
        simulation.dem.height as u32,
        simulation.dem.cell_size,
        simulation.dem.data1d.clone(),
    )?
    .with_vertical_exaggeration(exaggeration);

    let orchestrator = simulation.orchestrator();
    let grids = GRID_BUFFERS
        .iter()
        .map(|name| {
            let buffer = orchestrator
                .resources
                .get_buffer(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("simulation buffer '{name}' is missing"))?;
            Ok((name.clone(), buffer))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let particles = clone_particle_buffers(orchestrator, info.number_particles)?;

    Ok(HeadlessSimulation {
        device: orchestrator.device.clone(),
        queue: orchestrator.queue.clone(),
        terrain,
        grids,
        particles,
        info,
        simulation,
    })
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let settings_path = args.next().unwrap_or_else(|| DEFAULT_SETTINGS.to_string());
    let output = args.next().unwrap_or_else(|| DEFAULT_OUTPUT.to_string());
    let fps: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(30);
    let steps_per_frame: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    let exaggeration: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.3);
    let resolution = args
        .next()
        .unwrap_or_else(|| DEFAULT_RESOLUTION.to_string());
    let (width, height) = parse_resolution(&resolution)?;
    let overlay = Overlay::from_name(
        args.next()
            .unwrap_or_else(|| "peak_velocity".to_string())
            .as_str(),
    )?;

    let settings_json = std::fs::read_to_string(&settings_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to read settings file '{}': {error} (paths inside resolve relative to the \
             working directory; dem_path and release_areas_path are required)",
            settings_path
        )
    })?;
    let settings = Settings::loads(&settings_json)
        .map_err(|error| anyhow::anyhow!("invalid settings in '{settings_path}': {error}"))?;

    let mut sim = start_simulation(settings, exaggeration)?;
    println!(
        "Loaded DEM {}x{} at {} m; encoding {}x{} @ {fps} fps, {steps_per_frame} step(s) per frame, overlay '{}'",
        sim.simulation.dem.width,
        sim.simulation.dem.height,
        sim.simulation.dem.cell_size,
        width,
        height,
        overlay.label()
    );

    let mut encoder = spawn_encoder(&output, width, height, fps)?;
    let mut encoder_stdin = encoder.stdin.take().expect("ffmpeg stdin was piped");

    let overlay_buffer = overlay.buffer_name().map(|name| {
        &sim.grids
            .iter()
            .find(|(n, _)| *n == name)
            .expect("overlay grid was cloned at startup")
            .1
    });

    let mut renderer = Renderer::new(
        &sim.device,
        &sim.queue,
        wgpu::TextureFormat::Rgba8Unorm,
        width,
        height,
        &sim.terrain,
    );
    renderer.set_grid_overlay(&sim.device, overlay_buffer, overlay.range());
    let [position, velocity, velocity_z, stopped, elevation] = &sim.particles;
    renderer.set_particles(
        &sim.device,
        Some(ParticleBuffers {
            position,
            velocity,
            velocity_z,
            stopped,
            elevation,
        }),
    );
    renderer
        .particles_mut()
        .set_count(sim.info.number_particles);
    renderer.particles_mut().set_max_velocity(MAX_VELOCITY);
    renderer
        .particles_mut()
        .set_radius(sim.terrain.cell_size() * 0.9);

    let started = std::time::Instant::now();
    let mut frames = 0u64;
    loop {
        if sim.simulation.get_state() >= SimulationState::Finished {
            break;
        }
        let previous_timestep = sim.info.timestep;
        sim.info = pollster::block_on(sim.simulation.run_n_steps(steps_per_frame))?;
        if sim.info.timestep == previous_timestep {
            // No progress (e.g. steps clamped to zero); avoid an endless loop.
            break;
        }
        renderer
            .particles_mut()
            .set_count(sim.info.number_particles);

        let frame = render_core::capture::render_to_rgba8(
            &sim.device,
            &sim.queue,
            &mut renderer,
            width,
            height,
        )?;
        if let Err(error) = encoder_stdin.write_all(&frame) {
            let status = encoder.wait()?;
            anyhow::bail!("ffmpeg failed while receiving frames ({error}); exit status: {status}");
        }
        frames += 1;
        if frames % 100 == 0 {
            println!(
                "frame {frames}: step {} ({:.1} s simulated)",
                sim.info.timestep, sim.info.elapsed_time
            );
        }
    }

    // Closing stdin tells ffmpeg the stream is complete; it then finalizes
    // the mp4 container.
    drop(encoder_stdin);
    let status = encoder.wait()?;
    if !status.success() {
        anyhow::bail!("ffmpeg exited with {status}");
    }

    println!(
        "Wrote {output}: {frames} frames from {} steps in {:.1} s",
        sim.info.timestep,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
