//! Live avalanche simulation viewer.
//!
//! Each redraw advances the simulation by one step and then renders the simulation's own
//! storage buffers on the same wgpu queue. No particle or grid data is copied to the CPU.
//!
//! Usage: `cargo run -p render_core --example live_sim -- [settings.json] [exaggeration]`
//!
//! Controls: left drag orbits, right drag pans, scroll zooms, `R` restarts the
//! simulation, `V` resets the view, `0` hides the overlay, `1` peak flow velocity, `2`
//! peak flow thickness, `3` grid mass, `4` release areas, `5` slope angle, `6` slope
//! aspect, `7` roughness, `P` toggles particles. The egui panel switches the testcase
//! (the DEM pngs next to the configured one), the simulation model and the other
//! simulation settings; sliders fine-tune with the arrow keys once focused (click
//! the slider or its label, or `Tab` to it), and double-clicking a slider's label
//! resets the value to its default.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use compute_core::settings::{FrictionModel, Settings, SimModel};
use compute_core::{ComputeOrchestrator, SimInfo, buffers::BufferName};
use render_core::{OrbitCamera, OverlayRange, ParticleBuffers, Renderer, TerrainData};
use simulation::{Simulation, SimulationState};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Icon, Window, WindowId};

const ORBIT_SPEED: f32 = 0.005;
const ZOOM_SPEED: f32 = 0.1;
const STEPS_PER_FRAME: u32 = 1;
const DEFAULT_SIM_SPEED: f32 = 4.0;
/// Minimum time between rendered frames while the simulation runs. Stepping is
/// decoupled from presenting, so fast runs are not throttled to the display rate
/// and the GPU renders at most ~33 fps instead of once per step.
const MIN_RENDER_INTERVAL: Duration = Duration::from_millis(30);
/// Step batches drained per wake when catching up between render slots.
const MAX_STEP_BATCHES_PER_WAKE: usize = 16;
const DEFAULT_SETTINGS: &str = "settings.json";
/// Fallback directory the testcase dropdown scans when the configured DEM has no
/// usable parent directory.
const DEFAULT_TESTCASE_DIR: &str = "data/avaframe";
/// Release textures follow the frontend naming convention: `avaFoo` ->
/// `avaFooreleaseTexture.png`.
const RELEASE_TEXTURE_SUFFIX: &str = "releaseTexture";
const WINDOW_ICON: &[u8] = include_bytes!("../../../frontend/icons/android-chrome-512x512.png");

/// Grid buffers cloned from the simulation, both at startup and after a restart.
const GRID_BUFFERS: [BufferName; 7] = [
    BufferName::GridPeakVelocity,
    BufferName::GridPeakFlowThickness,
    BufferName::GridMass,
    BufferName::ReleaseAreas,
    BufferName::SlopeAngle,
    BufferName::SlopeAspect,
    BufferName::Roughness,
];

fn window_icon() -> anyhow::Result<Icon> {
    let image = image::load_from_memory(WINDOW_ICON)?.into_rgba8();
    let (width, height) = image.dimensions();
    Ok(Icon::from_rgba(image.into_raw(), width, height)?)
}

/// Engine defaults for the fields the settings panel edits (`SimSettings::new`).
/// They fill unset fields on load and are the double-click reset targets of the
/// sliders.
const DEFAULT_DENSITY: f32 = 200.0;
const DEFAULT_SLAB_THICKNESS: f32 = 1.0;
const DEFAULT_FRICTION_COEFFICIENT: f32 = 0.2;
const DEFAULT_DRAG_COEFFICIENT: f32 = 2000.0;
const DEFAULT_CFL: f32 = 0.5;
const DEFAULT_MIN_SLOPE_ANGLE: f32 = 28.0;
const DEFAULT_MAX_SLOPE_ANGLE: f32 = 60.0;
const DEFAULT_RELEASE_MIN_ELEVATION: f32 = 1500.0;
const DEFAULT_RELEASE_MAX_ELEVATION: f32 = 8848.0;
const DEFAULT_ROUGHNESS_THRESHOLD: f32 = 0.01;
const DEFAULT_INTERNAL_FRICTION_ANGLE: f32 = 40.0;
const DEFAULT_RELEASED_PARTICLES_PER_CELL: u32 = 8;
const DEFAULT_MAX_STEPS: u32 = 6000;

/// Fills in every field the settings panel edits with the engine defaults from
/// `SimSettings::new`. `from_settings` already falls back to those values for unset
/// fields, so this never changes simulation behaviour — it just keeps the panel
/// showing concrete values and makes the draft-vs-applied comparison exact.
fn apply_panel_defaults(settings: &mut Settings) {
    settings.density.get_or_insert(DEFAULT_DENSITY);
    settings
        .slab_thickness_factor
        .get_or_insert(DEFAULT_SLAB_THICKNESS);
    settings
        .friction_coefficient
        .get_or_insert(DEFAULT_FRICTION_COEFFICIENT);
    settings
        .drag_coefficient
        .get_or_insert(DEFAULT_DRAG_COEFFICIENT);
    settings.cfl.get_or_insert(DEFAULT_CFL);
    settings
        .min_slope_angle
        .get_or_insert(DEFAULT_MIN_SLOPE_ANGLE);
    settings
        .max_slope_angle
        .get_or_insert(DEFAULT_MAX_SLOPE_ANGLE);
    settings
        .release_min_elevation
        .get_or_insert(DEFAULT_RELEASE_MIN_ELEVATION);
    settings
        .release_max_elevation
        .get_or_insert(DEFAULT_RELEASE_MAX_ELEVATION);
    settings
        .roughness_threshold
        .get_or_insert(DEFAULT_ROUGHNESS_THRESHOLD);
    settings
        .internal_friction_angle
        .get_or_insert(DEFAULT_INTERNAL_FRICTION_ANGLE);
    settings
        .released_particles_per_cell
        .get_or_insert(DEFAULT_RELEASED_PARTICLES_PER_CELL);
    settings.max_steps.get_or_insert(DEFAULT_MAX_STEPS);
    settings.sim_model.get_or_insert(SimModel::Curvilinear);
    settings
        .friction_model
        .get_or_insert(FrictionModel::Voellmy);
    settings.enable_curvature.get_or_insert(true);
    settings.enable_particle_interaction.get_or_insert(true);
    settings.enable_particle_relaxation.get_or_insert(true);
    settings
        .enable_earth_pressure_coefficient
        .get_or_insert(true);
}

/// Directory the testcase dropdown scans: the folder holding the configured DEM
/// (the `data/avaframe` examples in the default settings), so the example also
/// works from other working directories.
fn testcase_dir(settings: &Settings) -> String {
    settings
        .dem_path
        .as_deref()
        .and_then(|path| Path::new(path).parent())
        .map(|dir| dir.to_string_lossy().into_owned())
        .filter(|dir| Path::new(dir).is_dir())
        .unwrap_or_else(|| DEFAULT_TESTCASE_DIR.to_string())
}

/// All testcase names found in `dir`, following the avaframe convention: `avaFoo.png`
/// is a DEM, `avaFooreleaseTexture.png` is its release areas.
fn discover_testcases(dir: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        tracing::warn!("testcase directory '{dir}' not found; testcase list stays empty");
        return Vec::new();
    };
    let mut cases = entries
        .flatten()
        .filter(|entry| entry.path().is_file())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| {
            let stem = name.strip_suffix(".png")?;
            (!stem.ends_with(RELEASE_TEXTURE_SUFFIX)).then(|| stem.to_string())
        })
        .collect::<Vec<_>>();
    cases.sort();
    cases.dedup();
    cases
}

/// DEM and release-area paths for a testcase. The release path is `None` when no
/// texture exists; the simulation then derives release areas from elevation and
/// slope thresholds (e.g. avaFlowPy, avaSimilaritySol).
fn testcase_paths(dir: &str, case: &str) -> (String, Option<String>) {
    let dem_path = Path::new(dir).join(format!("{case}.png"));
    let release_path = Path::new(dir).join(format!("{case}{RELEASE_TEXTURE_SUFFIX}.png"));
    (
        dem_path.to_string_lossy().into_owned(),
        release_path
            .is_file()
            .then(|| release_path.to_string_lossy().into_owned()),
    )
}

/// Testcase name shown in the dropdown for the configured DEM.
fn testcase_name(settings: &Settings) -> String {
    settings
        .dem_path
        .as_deref()
        .and_then(|path| Path::new(path).file_stem())
        .and_then(|stem| stem.to_str())
        .unwrap_or("custom")
        .to_string()
}

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
    fn buffer_name(self) -> Option<BufferName> {
        match self {
            Overlay::None => None,
            Overlay::PeakFlowVelocity => Some(BufferName::GridPeakVelocity),
            Overlay::PeakFlowThickness => Some(BufferName::GridPeakFlowThickness),
            Overlay::GridMass => Some(BufferName::GridMass),
            Overlay::ReleaseAreas => Some(BufferName::ReleaseAreas),
            Overlay::SlopeAngle => Some(BufferName::SlopeAngle),
            Overlay::SlopeAspect => Some(BufferName::SlopeAspect),
            Overlay::Roughness => Some(BufferName::Roughness),
        }
    }

    /// Colour ramp bounds in the units of each field, plus the legend label.
    fn range(self) -> OverlayRange {
        let range = match self {
            Overlay::None => OverlayRange::default(),
            Overlay::PeakFlowVelocity => OverlayRange::new(0.0, 40.0)
                .with_threshold(0.1)
                .with_unit("m/s"),
            Overlay::PeakFlowThickness => OverlayRange::new(0.0, 10.0)
                .with_threshold(0.01)
                .with_unit("m"),
            Overlay::GridMass => OverlayRange::new(0.0, 5_000.0)
                .with_threshold(1.0)
                .with_unit("kg"),
            // Slab thickness in metres; release textures typically hold 1.0 m.
            Overlay::ReleaseAreas => OverlayRange::new(0.0, 2.0)
                .with_threshold(0.01)
                .with_unit("m"),
            // Degrees; steeper than the default 60° release window saturates hot.
            Overlay::SlopeAngle => OverlayRange::new(0.0, 60.0).with_unit("deg"),
            // Degrees clockwise from north; flat cells hold -1 and stay bare.
            Overlay::SlopeAspect => OverlayRange::new(0.0, 360.0)
                .with_threshold(-0.5)
                .with_unit("deg"),
            // Dimensionless, 0 (smooth) to 1 (rough); border cells are forced to 1.
            Overlay::Roughness => OverlayRange::new(0.0, 1.0),
        };
        range.with_label(self.label())
    }

    fn label(self) -> &'static str {
        match self {
            Overlay::None => "none",
            Overlay::PeakFlowVelocity => "peak flow velocity",
            Overlay::PeakFlowThickness => "peak flow thickness",
            Overlay::GridMass => "grid mass",
            Overlay::ReleaseAreas => "release areas",
            Overlay::SlopeAngle => "slope angle",
            Overlay::SlopeAspect => "slope aspect",
            Overlay::Roughness => "roughness",
        }
    }
}

/// Clones the particle buffers the renderer binds. MPM keeps no vertical velocity
/// buffer, so it gets a zero-filled stand-in and colouring falls back to horizontal
/// speed; every other particle buffer exists for both models.
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
        tracing::debug!("no vertical velocity buffer; colouring by horizontal speed");
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

/// The simulation and cloned handles to its GPU state. wgpu resources are reference
/// counted, so the renderer observes the buffers `run_n_steps` keeps writing to.
struct SimulationView {
    simulation: Simulation,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    terrain: TerrainData,
    grids: Vec<(BufferName, wgpu::Buffer)>,
    particles: [wgpu::Buffer; 5],
    particle_count: u32,
    info: SimInfo,
    exaggeration: f32,
    failed: bool,
    final_info_reported: bool,
    step_timings: Vec<(u32, Duration)>,
    sim_speed: f32,
    next_step_at: Instant,
}

impl SimulationView {
    fn grid(&self, name: &BufferName) -> &wgpu::Buffer {
        &self
            .grids
            .iter()
            .find(|(n, _)| n == name)
            .expect("grid buffer was cloned at startup")
            .1
    }

    /// Advances one time-gated step batch. Returns `true` when simulation state
    /// changed and the frame needs redrawing.
    fn advance_frame(&mut self) -> bool {
        if self.failed || self.simulation.get_state() >= SimulationState::Finished {
            self.report_final_info();
            return false;
        }
        let now = Instant::now();
        if now < self.next_step_at {
            return false;
        }

        let scheduled_step_at = self.next_step_at;
        let previous_timestep = self.info.timestep;
        let started = Instant::now();
        match pollster::block_on(self.simulation.run_n_steps(STEPS_PER_FRAME)) {
            Ok(info) => {
                self.info = info;
                self.record_step_timing(previous_timestep, started.elapsed());
                if self.simulation.get_state() < SimulationState::Finished {
                    let interval = (self.info.dt / self.sim_speed).max(0.001);
                    self.next_step_at = scheduled_step_at + Duration::from_secs_f32(interval);
                }
                self.report_final_info();
            }
            Err(error) => {
                self.failed = true;
                tracing::error!("simulation step failed: {error}");
            }
        }
        true
    }

    fn adjust_speed(&mut self, factor: f32) {
        self.sim_speed = (self.sim_speed * factor).clamp(0.25, 256.0);
        self.next_step_at = Instant::now();
        tracing::info!("simulation speed: {:.2}x", self.sim_speed);
    }

    /// Re-clones the GPU buffers after the simulation was reset or rebuilt. Model
    /// switches change which buffers exist, so this must run after every rebuild.
    fn refresh_buffers(&mut self) -> anyhow::Result<()> {
        let orchestrator = self.simulation.orchestrator();
        let clone_grid = |name: BufferName| -> anyhow::Result<(BufferName, wgpu::Buffer)> {
            orchestrator
                .resources
                .get_buffer(&name)
                .cloned()
                .map(|buffer| (name.clone(), buffer))
                .ok_or_else(|| anyhow::anyhow!("simulation buffer '{name}' is missing"))
        };
        self.grids = GRID_BUFFERS
            .into_iter()
            .map(clone_grid)
            .collect::<anyhow::Result<Vec<_>>>()?;
        self.particles = clone_particle_buffers(orchestrator, self.info.number_particles)?;
        self.particle_count = self.info.number_particles;
        self.failed = false;
        self.final_info_reported = false;
        self.step_timings.clear();
        self.next_step_at = Instant::now();
        Ok(())
    }

    fn restart(&mut self) -> anyhow::Result<()> {
        self.simulation.reset();
        self.info = pollster::block_on(self.simulation.run_n_steps(0))?;
        self.refresh_buffers()?;
        tracing::info!("simulation restarted");
        Ok(())
    }

    /// Reloads the simulation in place from `settings` (`Simulation::create`):
    /// DEM, release areas and baked settings are replaced and the terrain and
    /// GPU buffers re-cloned, while the wgpu device/instance — and with them the
    /// window surface — are kept. Creating a second simulation here would panic
    /// once the old instance the surface belongs to is dropped. Returns `true`
    /// when the DEM dimensions changed.
    fn apply_settings(&mut self, settings: &Settings) -> anyhow::Result<bool> {
        let previous_dem = (
            self.simulation.dem.width,
            self.simulation.dem.height,
            self.simulation.dem.cell_size,
        );
        pollster::block_on(self.simulation.create(settings.clone()))?;
        self.terrain = TerrainData::new(
            self.simulation.dem.width as u32,
            self.simulation.dem.height as u32,
            self.simulation.dem.cell_size,
            self.simulation.dem.data1d.clone(),
        )?
        .with_vertical_exaggeration(self.exaggeration);
        self.info = pollster::block_on(self.simulation.run_n_steps(0))?;
        self.refresh_buffers()?;

        let terrain_switched = previous_dem
            != (
                self.simulation.dem.width,
                self.simulation.dem.height,
                self.simulation.dem.cell_size,
            );
        if terrain_switched {
            tracing::info!(
                "Loaded DEM {}x{} at {} m resolution",
                self.simulation.dem.width,
                self.simulation.dem.height,
                self.simulation.dem.cell_size
            );
        }
        Ok(terrain_switched)
    }

    fn run_to_completion(&mut self) -> anyhow::Result<()> {
        while self.simulation.get_state() < SimulationState::Finished {
            let previous_timestep = self.info.timestep;
            let started = Instant::now();
            self.info = pollster::block_on(self.simulation.run_n_steps(256))?;
            self.record_step_timing(previous_timestep, started.elapsed());
        }
        self.report_final_info();
        Ok(())
    }

    fn record_step_timing(&mut self, previous_timestep: u32, duration: Duration) {
        let steps = self.info.timestep.saturating_sub(previous_timestep);
        if steps == 0 {
            return;
        }
        // tracing::info!(
        //     "simulation step {} (+{}) took {:.3} ms",
        //     self.info.timestep,
        //     steps,
        //     duration.as_secs_f64() * 1000.0
        // );
        self.step_timings.push((steps, duration));
    }

    fn report_final_info(&mut self) {
        if !self.final_info_reported && self.simulation.get_state() >= SimulationState::Finished {
            tracing::info!("simulation finished: {:?}", self.info);
            self.report_timing_summary();
            self.final_info_reported = true;
        }
    }

    fn report_timing_summary(&self) {
        if self.step_timings.is_empty() {
            tracing::info!("simulation timing summary: no steps completed");
            return;
        }

        let total_steps: u32 = self.step_timings.iter().map(|(steps, _)| steps).sum();
        let total_duration: Duration = self
            .step_timings
            .iter()
            .map(|(_, duration)| *duration)
            .sum();
        let min_duration = self
            .step_timings
            .iter()
            .map(|(_, duration)| *duration)
            .min()
            .unwrap();
        let max_duration = self
            .step_timings
            .iter()
            .map(|(_, duration)| *duration)
            .max()
            .unwrap();
        let total_seconds = total_duration.as_secs_f64();
        let average_ms_per_step = total_seconds * 1000.0 / f64::from(total_steps);
        let steps_per_second = f64::from(total_steps) / total_seconds.max(f64::EPSILON);

        tracing::info!(
            "simulation timing summary: {} steps in {} batches, total {:.3} s, average {:.3} ms/step, min batch {:.3} ms, max batch {:.3} ms, {:.2} steps/s",
            total_steps,
            self.step_timings.len(),
            total_seconds,
            average_ms_per_step,
            min_duration.as_secs_f64() * 1000.0,
            max_duration.as_secs_f64() * 1000.0,
            steps_per_second,
        );
    }
}

fn start_simulation(settings: Settings, exaggeration: f32) -> anyhow::Result<SimulationView> {
    let mut sim: Simulation = pollster::block_on(Simulation::new_with_settings(settings))?;
    tracing::info!(
        "Loaded DEM {}x{} at {} m resolution",
        sim.dem.width,
        sim.dem.height,
        sim.dem.cell_size
    );

    let terrain = TerrainData::new(
        sim.dem.width as u32,
        sim.dem.height as u32,
        sim.dem.cell_size,
        sim.dem.data1d.clone(),
    )?
    .with_vertical_exaggeration(exaggeration);

    // Lazily prepares the simulation and incremental pipelines without advancing time.
    let info = pollster::block_on(sim.run_n_steps(0))?;
    let orchestrator = sim.orchestrator();
    let clone_buffer = |name: BufferName| -> anyhow::Result<wgpu::Buffer> {
        orchestrator
            .resources
            .get_buffer(&name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("simulation buffer '{name}' is missing"))
    };

    let grids = GRID_BUFFERS
        .into_iter()
        .map(|name| Ok((name.clone(), clone_buffer(name)?)))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let particles = clone_particle_buffers(orchestrator, info.number_particles)?;

    let device = orchestrator.device.clone();
    let queue = orchestrator.queue.clone();
    let adapter = orchestrator.adapter.clone();
    let instance = orchestrator.instance.clone();

    Ok(SimulationView {
        simulation: sim,
        device,
        queue,
        adapter,
        instance,
        terrain,
        grids,
        particles,
        particle_count: info.number_particles,
        info,
        exaggeration,
        failed: false,
        final_info_reported: false,
        step_timings: Vec::new(),
        sim_speed: DEFAULT_SIM_SPEED,
        next_step_at: Instant::now(),
    })
}

fn write_snapshot(sim: &SimulationView, overlay: Overlay, path: &str) -> anyhow::Result<()> {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 640;

    let mut renderer = Renderer::new(
        &sim.device,
        &sim.queue,
        wgpu::TextureFormat::Rgba8Unorm,
        WIDTH,
        HEIGHT,
        &sim.terrain,
    );
    renderer.set_grid_overlay(
        &sim.device,
        overlay.buffer_name().map(|name| sim.grid(&name)),
        overlay.range(),
    );
    renderer.set_particles(
        &sim.device,
        (std::env::var("LIVE_SIM_NO_PARTICLES").is_err()).then_some(ParticleBuffers {
            position: &sim.particles[0],
            velocity: &sim.particles[1],
            velocity_z: &sim.particles[2],
            stopped: &sim.particles[3],
            elevation: &sim.particles[4],
        }),
    );
    renderer.particles_mut().set_count(sim.particle_count);
    renderer.particles_mut().set_max_velocity(30.0);
    renderer
        .particles_mut()
        .set_radius(sim.terrain.cell_size() * 0.9);

    let pixels = render_core::capture::render_to_rgba8(
        &sim.device,
        &sim.queue,
        &mut renderer,
        WIDTH,
        HEIGHT,
    )?;
    image::save_buffer(path, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)?;
    tracing::info!("wrote {path} with overlay '{}'", overlay.label());
    Ok(())
}

/// Slider with an interactive label. Clicking the label focuses the slider so
/// the arrow keys fine-tune it; double-clicking the label resets the value to
/// `default`. Dragging the slider focuses it too.
fn f32_slider(
    ui: &mut egui::Ui,
    label: &str,
    default: f32,
    field: &mut Option<f32>,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
) -> (egui::Response, egui::Response) {
    let mut value = field.unwrap_or(default);
    let (mut slider, label) = ui
        .horizontal(|ui| {
            let slider = ui.add(egui::Slider::new(&mut value, range).step_by(step as f64));
            let label = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
            let slider = slider.labelled_by(label.id);
            (slider, label)
        })
        .inner;
    apply_slider_label_interactions(&mut slider, &label, &mut value, default);
    *field = Some(value);
    (slider, label)
}

fn u32_slider(
    ui: &mut egui::Ui,
    label: &str,
    default: u32,
    field: &mut Option<u32>,
    range: std::ops::RangeInclusive<u32>,
) -> (egui::Response, egui::Response) {
    let mut value = field.unwrap_or(default);
    let (mut slider, label) = ui
        .horizontal(|ui| {
            let slider = ui.add(egui::Slider::new(&mut value, range));
            let label = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
            let slider = slider.labelled_by(label.id);
            (slider, label)
        })
        .inner;
    apply_slider_label_interactions(&mut slider, &label, &mut value, default);
    *field = Some(value);
    (slider, label)
}

/// egui routes the arrow keys only to the focused widget, and a plain click does
/// not focus a slider — request focus when it is dragged or its label is clicked.
/// A double click on the label restores the default value.
fn apply_slider_label_interactions<T>(
    slider: &mut egui::Response,
    label: &egui::Response,
    value: &mut T,
    default: T,
) {
    if label.double_clicked() {
        *value = default;
        slider.request_focus();
        slider.mark_changed();
    } else if label.clicked() || slider.drag_started() {
        slider.request_focus();
    }
}

fn bool_checkbox(ui: &mut egui::Ui, label: &str, field: &mut Option<bool>) {
    let mut value = field.unwrap_or(false);
    ui.checkbox(&mut value, label);
    *field = Some(value);
}

struct ViewerWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct Viewer {
    sim: SimulationView,
    /// Settings the running simulation was created from.
    base: Settings,
    /// Settings edited in the panel; applied on the button click.
    draft: Settings,
    /// Testcase names for the dropdown, discovered next to the configured DEM.
    testcases: Vec<String>,
    window: Option<ViewerWindow>,
    cursor: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
    overlay: Overlay,
    show_particles: bool,
    /// Input or rebuild changed the view; the next redraw must render. Keeps an
    /// idle viewer at zero GPU work instead of redrawing identical frames.
    dirty: bool,
    /// Simulation state changed since the last rendered frame.
    pending_render: bool,
    last_render_at: Instant,
}

impl Viewer {
    fn new(sim: SimulationView, settings: Settings) -> Self {
        let testcases = discover_testcases(&testcase_dir(&settings));
        Self {
            draft: settings.clone(),
            base: settings,
            testcases,
            sim,
            window: None,
            cursor: None,
            orbiting: false,
            panning: false,
            overlay: Overlay::PeakFlowVelocity,
            show_particles: true,
            dirty: true,
            pending_render: false,
            last_render_at: Instant::now(),
        }
    }

    /// Flags the view as changed and wakes the event loop for a redraw.
    fn mark_dirty(&mut self) {
        self.dirty = true;
        if let Some(view) = self.window.as_ref() {
            view.window.request_redraw();
        }
    }

    fn apply_overlay(&mut self) {
        let Some(view) = self.window.as_mut() else {
            return;
        };
        let buffer = self.overlay.buffer_name().map(|name| self.sim.grid(&name));
        view.renderer
            .set_grid_overlay(&self.sim.device, buffer, self.overlay.range());
        self.dirty = true;
        tracing::info!("overlay: {}", self.overlay.label());
    }

    fn apply_particles(&mut self) {
        let Some(view) = self.window.as_mut() else {
            return;
        };
        let buffers = self.show_particles.then(|| ParticleBuffers {
            position: &self.sim.particles[0],
            velocity: &self.sim.particles[1],
            velocity_z: &self.sim.particles[2],
            stopped: &self.sim.particles[3],
            elevation: &self.sim.particles[4],
        });
        view.renderer.set_particles(&self.sim.device, buffers);
        self.dirty = true;
    }

    /// Applies `settings` to the running simulation. Every panel change needs
    /// this, since settings are baked in when the simulation prepares. The
    /// renderer is recreated because a testcase switch can change the terrain's
    /// size and cell size; the device/instance (and surface) are kept.
    fn rebuild(&mut self, settings: Settings) -> anyhow::Result<()> {
        let terrain_switched = self.sim.apply_settings(&settings)?;
        self.base = settings;
        self.draft = self.base.clone();

        if let Some(view) = self.window.as_mut() {
            let size = view.window.inner_size();
            let (width, height) = (size.width.max(1), size.height.max(1));
            view.config.width = width;
            view.config.height = height;
            view.surface.configure(&self.sim.device, &view.config);
            view.renderer = Renderer::new(
                &self.sim.device,
                &self.sim.queue,
                view.config.format,
                width,
                height,
                &self.sim.terrain,
            );
            // A different DEM needs a fresh framing; same-DEM restarts keep the camera.
            if terrain_switched {
                let aspect = width as f32 / height as f32;
                view.renderer.camera = OrbitCamera::framing(&self.sim.terrain, aspect);
            }
        }
        self.apply_overlay();
        self.apply_particles();
        if let Some(view) = self.window.as_mut() {
            let particles = view.renderer.particles_mut();
            particles.set_count(self.sim.particle_count);
            particles.set_max_velocity(30.0);
            particles.set_radius(self.sim.terrain.cell_size() * 0.9);
            view.window.request_redraw();
        }
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let device = &self.sim.device;
        let Some(view) = self.window.as_mut() else {
            return;
        };
        view.config.width = width;
        view.config.height = height;
        view.surface.configure(device, &view.config);
        view.renderer.resize(device, width, height);
        self.mark_dirty();
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let icon = match window_icon() {
            Ok(icon) => Some(icon),
            Err(error) => {
                tracing::warn!("failed to load window icon: {error}");
                None
            }
        };
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Avalanchers - Live Simulation")
                        .with_window_icon(icon)
                        .with_maximized(true),
                )
                .expect("failed to create window"),
        );

        let surface = self
            .sim
            .instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));
        let config = surface
            .get_default_config(&self.sim.adapter, width, height)
            .expect("the simulation adapter cannot present to this window");
        surface.configure(&self.sim.device, &config);

        let renderer = Renderer::new(
            &self.sim.device,
            &self.sim.queue,
            config.format,
            width,
            height,
            &self.sim.terrain,
        );

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &self.sim.device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );

        self.window = Some(ViewerWindow {
            window,
            surface,
            config,
            renderer,
            egui_state,
            egui_renderer,
        });
        if let Some(view) = self.window.as_ref() {
            view.window.request_redraw();
        }

        self.apply_overlay();
        self.apply_particles();
        if let Some(view) = self.window.as_mut() {
            let particles = view.renderer.particles_mut();
            particles.set_count(self.sim.particle_count);
            particles.set_max_velocity(30.0);
            particles.set_radius(self.sim.terrain.cell_size() * 0.9);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // RedrawRequested is not user input: egui_winit answers it with another
        // repaint request, which would keep an idle viewer rendering forever.
        if !matches!(event, WindowEvent::RedrawRequested) {
            // egui sees every event first. Pointer events consumed by egui (over the panel)
            // must not drive the camera, but keyboard focus alone must not swallow the
            // viewer's single-key shortcuts: egui grabs focus from any click on the panel
            // and keeps it, so keys are only consumed while a popup is open or a text
            // field is focused.
            if let Some(view) = self.window.as_mut() {
                let response = view.egui_state.on_window_event(&view.window, &event);
                let egui_wants_repaint = response.repaint;
                let egui_needs_keys = view.egui_state.egui_ctx().any_popup_open()
                    || view.egui_state.egui_ctx().text_edit_focused();
                let consumed = match event {
                    WindowEvent::KeyboardInput { .. } => response.consumed && egui_needs_keys,
                    _ => response.consumed,
                };
                if consumed {
                    if egui_wants_repaint {
                        self.mark_dirty();
                    }
                    return;
                }
                // egui wants repaints for almost every event (any cursor move),
                // so this must not return: camera input below still needs to run.
                if egui_wants_repaint {
                    self.mark_dirty();
                }
            }
        }

        let mut pending_settings: Option<Settings> = None;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size.width, size.height),
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.orbiting = pressed,
                    MouseButton::Right | MouseButton::Middle => self.panning = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let last = self.cursor.replace((position.x, position.y));
                let (Some((last_x, last_y)), Some(view)) = (last, self.window.as_mut()) else {
                    return;
                };
                let dx = (position.x - last_x) as f32;
                let dy = (position.y - last_y) as f32;

                if self.orbiting {
                    view.renderer
                        .camera
                        .orbit(dx * ORBIT_SPEED, dy * ORBIT_SPEED);
                } else if self.panning {
                    let width = view.config.width.max(1) as f32;
                    let height = view.config.height.max(1) as f32;
                    view.renderer.camera.pan(dx / width, dy / height);
                } else {
                    return;
                }
                self.mark_dirty();
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 60.0,
                };
                if let Some(view) = self.window.as_mut() {
                    view.renderer.camera.zoom(scroll * ZOOM_SPEED);
                    self.mark_dirty();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                    return;
                }

                match event.logical_key.as_ref() {
                    Key::Character("0") => {
                        self.overlay = Overlay::None;
                        self.apply_overlay();
                    }
                    Key::Character("1") => {
                        self.overlay = Overlay::PeakFlowVelocity;
                        self.apply_overlay();
                    }
                    Key::Character("2") => {
                        self.overlay = Overlay::PeakFlowThickness;
                        self.apply_overlay();
                    }
                    Key::Character("3") => {
                        self.overlay = Overlay::GridMass;
                        self.apply_overlay();
                    }
                    Key::Character("4") => {
                        self.overlay = Overlay::ReleaseAreas;
                        self.apply_overlay();
                    }
                    Key::Character("5") => {
                        self.overlay = Overlay::SlopeAngle;
                        self.apply_overlay();
                    }
                    Key::Character("6") => {
                        self.overlay = Overlay::SlopeAspect;
                        self.apply_overlay();
                    }
                    Key::Character("7") => {
                        self.overlay = Overlay::Roughness;
                        self.apply_overlay();
                    }
                    Key::Character("p" | "P") => {
                        self.show_particles = !self.show_particles;
                        self.apply_particles();
                    }
                    Key::Character("+") => self.sim.adjust_speed(2.0),
                    Key::Character("-") => self.sim.adjust_speed(0.5),
                    Key::Character("r" | "R") => {
                        if let Err(error) = self.sim.restart() {
                            self.sim.failed = true;
                            tracing::error!("simulation restart failed: {error}");
                        } else {
                            self.apply_overlay();
                            self.apply_particles();
                            if let Some(view) = self.window.as_ref() {
                                view.window.request_redraw();
                            }
                        }
                    }
                    Key::Character("v" | "V") => {
                        if let Some(view) = self.window.as_mut() {
                            let aspect =
                                view.config.width as f32 / view.config.height.max(1) as f32;
                            view.renderer.camera =
                                render_core::OrbitCamera::framing(&self.sim.terrain, aspect);
                        }
                        self.mark_dirty();
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                // Stepping happens in about_to_wait; a redraw only presents state.
                // Clean, unchanged frames keep the previous swapchain image.
                let Some(view) = self.window.as_mut() else {
                    return;
                };
                if !self.dirty && !self.pending_render {
                    return;
                }

                let device = self.sim.device.clone();
                let queue = self.sim.queue.clone();

                let frame = match view.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        view.surface.configure(&device, &view.config);
                        view.window.request_redraw();
                        return;
                    }
                    other => {
                        tracing::warn!("Skipping frame: {other:?}");
                        view.window.request_redraw();
                        return;
                    }
                };

                let overlay = self.overlay.label();
                let title = if self.sim.failed {
                    format!("Avalanchers - Simulation Failed - overlay: {overlay}")
                } else if self.sim.simulation.get_state() < SimulationState::Finished {
                    format!(
                        "Avalanchers - step {} - {:.2} s - {:.2}x - overlay: {overlay}",
                        self.sim.info.timestep, self.sim.info.elapsed_time, self.sim.sim_speed
                    )
                } else {
                    format!(
                        "Avalanchers - Simulation Finished - step {} - {:.2} s - {:.2}x - overlay: {overlay}",
                        self.sim.info.timestep, self.sim.info.elapsed_time, self.sim.sim_speed
                    )
                };
                view.window.set_title(&title);
                let target = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                view.renderer.render(&device, &queue, &target);

                // Settings panel, painted over the scene in the same frame.
                let (timestep, elapsed) = (self.sim.info.timestep, self.sim.info.elapsed_time);
                let base = &self.base;
                let cases = &self.testcases;
                let testcase_dir = testcase_dir(base);
                // The dropdown follows the draft, so a selection stays visible
                // until it is applied (the draft's paths are resolved below).
                let draft_case_before = testcase_name(&self.draft);
                let mut draft = self.draft.clone();
                let mut draft_case = draft_case_before.clone();
                let mut apply_clicked = false;
                let egui_ctx = view.egui_state.egui_ctx().clone();
                let input = view.egui_state.take_egui_input(&view.window);
                // `run_ui` (not bare `begin_pass`) so egui knows the root UI covers the
                // viewport; otherwise it treats the whole background as an egui surface
                // and consumes every mouse event meant for the camera.
                let mut full_output = egui_ctx.run_ui(input, |root| {
                    // Anchored top-right so the panel stays clear of the terrain drag area.
                    egui::Window::new("Simulation")
                        .pivot(egui::Align2::RIGHT_TOP)
                        .default_pos(
                            root.ctx().viewport_rect().right_top() + egui::vec2(-16.0, 16.0),
                        )
                        .default_width(280.0)
                        .show(root.ctx(), |ui| {
                            ui.label(format!("step {timestep} — {elapsed:.2} s"));
                            ui.add_space(4.0);

                            ui.horizontal(|ui| {
                                ui.label("test case");
                                egui::ComboBox::from_id_salt("testcase")
                                    .selected_text(draft_case.as_str())
                                    .show_ui(ui, |ui| {
                                        for case in cases {
                                            ui.selectable_value(
                                                &mut draft_case,
                                                case.clone(),
                                                case.as_str(),
                                            );
                                        }
                                    });
                            });
                            // A testcase switch swaps the DEM and its release texture;
                            // a missing texture falls back to derived release areas.
                            if draft_case != draft_case_before {
                                let (dem_path, release_areas_path) =
                                    testcase_paths(&testcase_dir, &draft_case);
                                draft.dem_path = Some(dem_path);
                                draft.release_areas_path = release_areas_path;
                            }

                            ui.horizontal(|ui| {
                                ui.label("model");
                                let mut model =
                                    draft.sim_model.unwrap_or(SimModel::TerrainFollowing);
                                egui::ComboBox::from_id_salt("sim_model")
                                    .selected_text(model.to_string())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut model,
                                            SimModel::TerrainFollowing,
                                            "Terrain-following",
                                        );
                                        ui.selectable_value(
                                            &mut model,
                                            SimModel::Curvilinear,
                                            "Curvilinear",
                                        );
                                        ui.selectable_value(&mut model, SimModel::MpmDaC, "MPMDAC");
                                    });
                                draft.sim_model = Some(model);
                            });

                            ui.horizontal(|ui| {
                                ui.label("friction");
                                let mut friction =
                                    draft.friction_model.unwrap_or(FrictionModel::Voellmy);
                                egui::ComboBox::from_id_salt("friction_model")
                                    .selected_text(friction.to_string())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut friction,
                                            FrictionModel::Coulomb,
                                            "Coulomb",
                                        );
                                        ui.selectable_value(
                                            &mut friction,
                                            FrictionModel::Voellmy,
                                            "Voellmy",
                                        );
                                        ui.selectable_value(
                                            &mut friction,
                                            FrictionModel::VoellmyMinShear,
                                            "Voellmy-min-shear",
                                        );
                                        ui.selectable_value(
                                            &mut friction,
                                            FrictionModel::SamosAT,
                                            "SamosAT",
                                        );
                                    });
                                if Some(friction) != draft.friction_model {
                                    // Same preset the web frontend applies for Coulomb.
                                    if friction == FrictionModel::Coulomb {
                                        draft.friction_coefficient = Some(0.4663);
                                    }
                                    draft.friction_model = Some(friction);
                                }
                            });

                            egui::CollapsingHeader::new("parameters")
                                .default_open(true)
                                .show(ui, |ui| {
                                    f32_slider(
                                        ui,
                                        "density",
                                        DEFAULT_DENSITY,
                                        &mut draft.density,
                                        50.0..=500.0,
                                        5.0,
                                    );
                                    f32_slider(
                                        ui,
                                        "slab thickness",
                                        DEFAULT_SLAB_THICKNESS,
                                        &mut draft.slab_thickness_factor,
                                        0.1..=5.0,
                                        0.05,
                                    );
                                    f32_slider(
                                        ui,
                                        "friction coefficient",
                                        DEFAULT_FRICTION_COEFFICIENT,
                                        &mut draft.friction_coefficient,
                                        0.05..=1.0,
                                        0.005,
                                    );
                                    // Coulomb and SamosAT have no drag term, like the
                                    // web frontend.
                                    let drag_active = !matches!(
                                        draft.friction_model,
                                        Some(FrictionModel::Coulomb | FrictionModel::SamosAT)
                                    );
                                    ui.add_enabled_ui(drag_active, |ui| {
                                        f32_slider(
                                            ui,
                                            "drag coefficient",
                                            DEFAULT_DRAG_COEFFICIENT,
                                            &mut draft.drag_coefficient,
                                            50.0..=10000.0,
                                            50.0,
                                        );
                                    });
                                    u32_slider(
                                        ui,
                                        "particles per cell",
                                        DEFAULT_RELEASED_PARTICLES_PER_CELL,
                                        &mut draft.released_particles_per_cell,
                                        1..=256,
                                    );
                                    f32_slider(
                                        ui,
                                        "CFL",
                                        DEFAULT_CFL,
                                        &mut draft.cfl,
                                        0.1..=1.0,
                                        0.05,
                                    );
                                    u32_slider(
                                        ui,
                                        "max steps",
                                        DEFAULT_MAX_STEPS,
                                        &mut draft.max_steps,
                                        1..=10000,
                                    );
                                    f32_slider(
                                        ui,
                                        "internal friction angle",
                                        DEFAULT_INTERNAL_FRICTION_ANGLE,
                                        &mut draft.internal_friction_angle,
                                        0.0..=60.0,
                                        0.1,
                                    );
                                });

                            egui::CollapsingHeader::new("terrain & release").show(ui, |ui| {
                                f32_slider(
                                    ui,
                                    "min slope angle",
                                    DEFAULT_MIN_SLOPE_ANGLE,
                                    &mut draft.min_slope_angle,
                                    0.0..=60.0,
                                    0.5,
                                );
                                f32_slider(
                                    ui,
                                    "max slope angle",
                                    DEFAULT_MAX_SLOPE_ANGLE,
                                    &mut draft.max_slope_angle,
                                    5.0..=90.0,
                                    0.5,
                                );
                                f32_slider(
                                    ui,
                                    "release min elevation",
                                    DEFAULT_RELEASE_MIN_ELEVATION,
                                    &mut draft.release_min_elevation,
                                    0.0..=5000.0,
                                    10.0,
                                );
                                f32_slider(
                                    ui,
                                    "release max elevation",
                                    DEFAULT_RELEASE_MAX_ELEVATION,
                                    &mut draft.release_max_elevation,
                                    0.0..=9000.0,
                                    10.0,
                                );
                                f32_slider(
                                    ui,
                                    "roughness threshold",
                                    DEFAULT_ROUGHNESS_THRESHOLD,
                                    &mut draft.roughness_threshold,
                                    0.001..=1.0,
                                    0.001,
                                );
                            });

                            egui::CollapsingHeader::new("features").show(ui, |ui| {
                                bool_checkbox(ui, "curvature", &mut draft.enable_curvature);
                                bool_checkbox(
                                    ui,
                                    "particle interaction",
                                    &mut draft.enable_particle_interaction,
                                );
                                bool_checkbox(
                                    ui,
                                    "particle relaxation",
                                    &mut draft.enable_particle_relaxation,
                                );
                                bool_checkbox(
                                    ui,
                                    "earth pressure coefficient",
                                    &mut draft.enable_earth_pressure_coefficient,
                                );
                            });

                            ui.add_space(4.0);
                            ui.add_enabled_ui(draft != *base, |ui| {
                                if ui.button("apply & restart").clicked() {
                                    apply_clicked = true;
                                }
                            });
                        });
                });
                view.egui_state
                    .handle_platform_output(&view.window, full_output.platform_output);
                self.draft = draft;
                if apply_clicked {
                    pending_settings = Some(self.draft.clone());
                }

                let pixels_per_point = egui_ctx.pixels_per_point();
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [view.config.width, view.config.height],
                    pixels_per_point,
                };
                let paint_jobs = egui_ctx.tessellate(full_output.shapes, pixels_per_point);
                for (id, deltas) in &full_output.textures_delta.set {
                    for delta in deltas {
                        view.egui_renderer
                            .update_texture(&device, &queue, *id, delta);
                    }
                }
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui Encoder"),
                });
                view.egui_renderer.update_buffers(
                    &device,
                    &queue,
                    &mut encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );
                {
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    view.egui_renderer.render(
                        &mut pass.forget_lifetime(),
                        &paint_jobs,
                        &screen_descriptor,
                    );
                }
                queue.submit(Some(encoder.finish()));
                for id in &full_output.textures_delta.free {
                    view.egui_renderer.free_texture(id);
                }
                // Mark the deltas as handled; epaint panics on drop otherwise.
                full_output.textures_delta.clear();

                queue.present(frame);
                self.dirty = false;
                self.pending_render = false;
                self.last_render_at = Instant::now();
            }
            _ => {}
        }

        // Panel changes rebuild the simulation; running after the frame means the
        // fresh buffers are only drawn on the next redraw.
        if let Some(settings) = pending_settings {
            if let Err(error) = self.rebuild(settings) {
                tracing::error!("simulation rebuild failed: {error}");
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(view) = self.window.as_ref() else {
            return;
        };

        // Drain every due step batch (bounded); rendering is capped separately so
        // fast runs are not throttled to the display rate.
        let mut stepped = false;
        for _ in 0..MAX_STEP_BATCHES_PER_WAKE {
            match self.sim.advance_frame() {
                true => stepped = true,
                false => break,
            }
        }
        if stepped {
            self.pending_render = true;
        }

        let active =
            !self.sim.failed && self.sim.simulation.get_state() < SimulationState::Finished;
        let render_slot = self.last_render_at + MIN_RENDER_INTERVAL;
        if self.dirty || (self.pending_render && Instant::now() >= render_slot) {
            view.window.request_redraw();
        } else if active || self.pending_render {
            // Sleep until the earlier of the next due step and — only when a
            // frame is waiting — the next render slot; input wakes us earlier.
            let mut wake = self.sim.next_step_at;
            if self.pending_render {
                wake = wake.min(render_slot);
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(wake.max(Instant::now())));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let settings_path = args.next().unwrap_or_else(|| DEFAULT_SETTINGS.to_string());
    let exaggeration: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let snapshot = std::env::var("LIVE_SIM_SNAPSHOT").ok();
    println!(
        "Settings path: {settings_path} exaggeration: {exaggeration} snapshot: {:?}",
        snapshot
    );

    let mut settings = Settings::from_json(&settings_path)
        .with_context(|| format!("failed to load settings from {settings_path}"))?;
    apply_panel_defaults(&mut settings);

    let mut sim = start_simulation(settings.clone(), exaggeration)?;

    if let Some(path) = snapshot {
        sim.run_to_completion()?;
        write_snapshot(&sim, Overlay::PeakFlowVelocity, &path)?;
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    // The viewer switches to WaitUntil dynamically; idle frames cost nothing.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut Viewer::new(sim, settings))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avaframe_dir() -> String {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/avaframe").to_string()
    }

    #[test]
    fn discovers_avaframe_cases_by_the_frontend_naming_convention() {
        let dir = avaframe_dir();
        let cases = discover_testcases(&dir);
        assert!(cases.contains(&"avaGar".to_string()));
        assert!(cases.contains(&"avaChannel".to_string()));
        assert!(cases.contains(&"avaFlowPy".to_string()));
        assert!(
            !cases
                .iter()
                .any(|case| case.ends_with(RELEASE_TEXTURE_SUFFIX))
        );
        let mut sorted = cases.clone();
        sorted.sort();
        assert_eq!(cases, sorted, "cases must be sorted for the dropdown");
    }

    #[test]
    fn release_path_is_none_without_a_release_texture() {
        let dir = avaframe_dir();
        let (dem, release) = testcase_paths(&dir, "avaGar");
        assert!(Path::new(&dem).is_file());
        assert!(
            release
                .as_deref()
                .is_some_and(|path| Path::new(path).is_file())
        );
        let (_, release) = testcase_paths(&dir, "avaFlowPy");
        assert!(release.is_none(), "avaFlowPy ships no release texture");
    }

    #[test]
    fn testcase_dir_falls_back_to_data_avaframe() {
        assert_eq!(testcase_dir(&Settings::default()), DEFAULT_TESTCASE_DIR);
        let custom = Settings {
            dem_path: Some("no/such/dir/avaX.png".to_string()),
            ..Settings::default()
        };
        assert_eq!(testcase_dir(&custom), DEFAULT_TESTCASE_DIR);
    }

    #[test]
    fn panel_defaults_match_the_engine_fallbacks() {
        let mut settings = Settings::default();
        apply_panel_defaults(&mut settings);
        let sim_settings = settings.get_sim_settings();
        assert_eq!(sim_settings.density, 200.0);
        assert_eq!(sim_settings.max_steps, 6000);
        assert_eq!(sim_settings.released_particles_per_cell, 8);
        assert_eq!(sim_settings.friction_model, FrictionModel::Voellmy.as_int());
        assert_eq!(sim_settings.min_slope_angle, 28.0);
        assert_eq!(sim_settings.max_slope_angle, 60.0);
    }

    /// The apply-button path: reload the same `Simulation` in place — once with
    /// a different testcase and once with a different model — and keep stepping.
    #[test]
    fn apply_settings_reloads_in_place_across_testcases_and_models() {
        let avaframe = format!("{}/../../data/avaframe", env!("CARGO_MANIFEST_DIR"));
        let mut view = start_simulation(
            Settings {
                dem_path: Some(format!("{avaframe}/avaGar.png")),
                release_areas_path: Some(format!("{avaframe}/avaGarreleaseTexture.png")),
                max_steps: Some(10),
                ..Settings::default()
            },
            1.0,
        )
        .unwrap();
        let first_width = view.simulation.dem.width;

        view.apply_settings(&Settings {
            dem_path: Some(format!("{avaframe}/avaParabola.png")),
            release_areas_path: Some(format!("{avaframe}/avaParabolareleaseTexture.png")),
            max_steps: Some(10),
            ..Settings::default()
        })
        .unwrap();
        assert!(
            view.simulation.dem.width != first_width,
            "different testcase should switch the terrain"
        );
        assert_eq!(view.grids.len(), GRID_BUFFERS.len());
        assert_eq!(view.particles.len(), 5);
        assert_eq!(view.particle_count, view.info.number_particles);
        assert!(view.particle_count > 0);

        view.apply_settings(&Settings {
            dem_path: Some(format!("{avaframe}/avaParabola.png")),
            release_areas_path: Some(format!("{avaframe}/avaParabolareleaseTexture.png")),
            sim_model: Some(SimModel::MpmDaC),
            max_steps: Some(10),
            ..Settings::default()
        })
        .unwrap();
        let info = pollster::block_on(view.simulation.run_n_steps(5)).unwrap();
        assert_eq!(info.timestep, 5);
    }

    /// Clicking a slider or its label must focus it, a focused slider must
    /// fine-tune by the configured step with the arrow keys, and double-clicking
    /// the label must reset the value to the default — driven through the real
    /// `f32_slider` helper with synthetic input, headless.
    #[test]
    fn slider_focuses_on_click_and_fine_tunes_with_arrow_keys() {
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 200.0));
        let (default, step) = (1.0_f32, 0.5_f32);
        let value = std::cell::Cell::new(Some(6.0_f32));
        let slider_id = std::cell::Cell::new(egui::Id::NULL);
        let slider_rect = std::cell::Cell::new(egui::Rect::ZERO);
        let label_rect = std::cell::Cell::new(egui::Rect::ZERO);

        let pass = |events: Vec<egui::Event>, time: f64| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    time: Some(time),
                    screen_rect: Some(screen),
                    events,
                    ..egui::RawInput::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        let mut draft = value.get();
                        let (slider, label) =
                            f32_slider(ui, "v", default, &mut draft, 0.0..=10.0, step);
                        value.set(draft);
                        slider_id.set(slider.id);
                        slider_rect.set(slider.rect);
                        label_rect.set(label.rect);
                    });
                },
            );
            // epaint panics on drop if deltas are left unhandled.
            output.textures_delta.clear();
        };

        // Layout pass, then a click on the slider: the drag must focus it.
        pass(vec![], 0.0);
        assert!(slider_rect.get().width() > 0.0, "slider must be laid out");
        assert!(label_rect.get().width() > 0.0, "label must be laid out");
        let click = slider_rect.get().center();
        pass(
            vec![
                egui::Event::PointerMoved(click),
                egui::Event::PointerButton {
                    pos: click,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            0.1,
        );
        pass(
            vec![egui::Event::PointerButton {
                pos: click,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
            0.2,
        );
        assert_eq!(
            ctx.memory(|memory| memory.focused()),
            Some(slider_id.get()),
            "clicking a slider must give it keyboard focus"
        );

        let before = value.get();
        pass(vec![key_press(egui::Key::ArrowRight)], 0.3);
        let after_right = value.get();
        assert_eq!(after_right, before.map(|v| v + step));
        pass(vec![key_press(egui::Key::ArrowLeft)], 0.4);
        assert_eq!(value.get(), before, "ArrowLeft must undo the step");

        // A double click on the label (two clicks inside the double-click
        // window) resets the value to the default.
        let label_center = label_rect.get().center();
        let button = |pos: egui::Pos2, pressed: bool| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        pass(vec![egui::Event::PointerMoved(label_center)], 0.5);
        pass(vec![button(label_center, true)], 0.6);
        pass(vec![button(label_center, false)], 0.7);
        pass(vec![button(label_center, true)], 0.8);
        pass(vec![button(label_center, false)], 0.9);
        assert_eq!(value.get(), Some(default), "double click must reset");
        assert_eq!(
            ctx.memory(|memory| memory.focused()),
            Some(slider_id.get()),
            "the reset slider stays focused"
        );
        pass(vec![key_press(egui::Key::ArrowRight)], 1.0);
        assert_eq!(value.get(), Some(default + step));
    }

    fn key_press(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }
    }
}
