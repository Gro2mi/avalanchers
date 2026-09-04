//! Live avalanche simulation viewer.
//!
//! Each redraw advances the simulation by one step and then renders the simulation's own
//! storage buffers on the same wgpu queue. No particle or grid data is copied to the CPU.
//!
//! Usage: `cargo run -p render_core --example live_sim -- [data/avaframe/avaAlr.png] [exaggeration]`
//!
//! Controls: left drag orbits, right drag pans, scroll zooms, `R` resets the view,
//! `0` hides the overlay, `1` peak flow velocity, `2` peak flow thickness, `3` grid mass,
//! `4` release areas, `5` slope angle, `6` slope aspect, `7` roughness, `P` toggles
//! particles. The egui panel switches the simulation model (particle/mpm).

use std::sync::Arc;
use std::time::{Duration, Instant};

use compute_core::settings::{Settings, SimModel};
use compute_core::{ComputeOrchestrator, SimInfo, buffers::BufferName};
use render_core::{OverlayRange, ParticleBuffers, Renderer, TerrainData};
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
        clone_buffer(BufferName::ParticlesStopped)?,
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

    fn current_model(&self) -> SimModel {
        SimModel::from_int(self.simulation.settings.sim_model).unwrap_or(SimModel::Particle)
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

    /// Rebuilds the simulation with a different model. Settings are baked in when the
    /// simulation prepares, so a model change means a full recreate; the DEM is re-read
    /// from disk as part of that.
    fn apply_model(&mut self, model: SimModel) -> anyhow::Result<()> {
        self.simulation.settings.sim_model = model.as_int();
        self.simulation.reset();
        self.info = pollster::block_on(self.simulation.run_n_steps(0))?;
        self.refresh_buffers()?;
        tracing::info!("simulation rebuilt with the {model} model");
        Ok(())
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
        tracing::info!(
            "simulation step {} (+{}) took {:.3} ms",
            self.info.timestep,
            steps,
            duration.as_secs_f64() * 1000.0
        );
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

fn start_simulation(settings_path: &str, exaggeration: f32) -> anyhow::Result<SimulationView> {
    // let mut sim = pollster::block_on(Simulation::new())?;
    // // Same setup as `Simulation::create_example`, kept explicit so the paths can be
    // // stored for model rebuilds.
    // let release_areas_path = settings_path.replace(".png", "releaseTexture.png");
    // let model = std::env::var("LIVE_SIM_MODEL")
    //     .ok()
    //     .and_then(|value| value.parse().ok())
    //     .unwrap_or(SimModel::Particle);
    // let settings = Settings {
    //     dem_path: Some(dem_path.to_string()),
    //     release_areas_path: Some(release_areas_path.clone()),
    //     sim_model: Some(model),
    //     ..Settings::default()
    // };
    // pollster::block_on(sim.create(settings))?;
    let settings =
        Settings::from_json(&settings_path).expect("Failed to load settings from JSON file");

    let mut sim: Simulation = pollster::block_on(Simulation::new_with_settings(settings.clone()))?;
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
    window: Option<ViewerWindow>,
    cursor: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
    overlay: Overlay,
    show_particles: bool,
    /// Model selected in the settings panel; applied on the button click.
    ui_model: SimModel,
    /// Input or rebuild changed the view; the next redraw must render. Keeps an
    /// idle viewer at zero GPU work instead of redrawing identical frames.
    dirty: bool,
    /// Simulation state changed since the last rendered frame.
    pending_render: bool,
    last_render_at: Instant,
}

impl Viewer {
    fn new(sim: SimulationView) -> Self {
        Self {
            ui_model: sim.current_model(),
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

    /// Rebuilds the simulation with the selected model and rebinds every buffer the
    /// renderer watches.
    fn rebuild_with_model(&mut self, model: SimModel) -> anyhow::Result<()> {
        self.ui_model = model;
        self.sim.apply_model(model)?;
        self.apply_overlay();
        self.apply_particles();
        if let Some(view) = self.window.as_mut() {
            view.renderer
                .particles_mut()
                .set_count(self.sim.particle_count);
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

        let mut pending_model: Option<SimModel> = None;
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
                let current = self.sim.current_model();
                let (timestep, elapsed) = (self.sim.info.timestep, self.sim.info.elapsed_time);
                let mut draft = self.ui_model;
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
                        .default_width(240.0)
                        .show(root.ctx(), |ui| {
                            ui.label(format!("step {timestep} — {elapsed:.2} s"));
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("model");
                                egui::ComboBox::from_id_salt("sim_model")
                                    .selected_text(draft.to_string())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut draft,
                                            SimModel::Particle,
                                            "particle",
                                        );
                                        ui.selectable_value(&mut draft, SimModel::MPM, "mpm");
                                    });
                            });
                            ui.add_enabled_ui(draft != current, |ui| {
                                if ui.button("apply & restart").clicked() {
                                    pending_model = Some(draft);
                                }
                            });
                        });
                });
                view.egui_state
                    .handle_platform_output(&view.window, full_output.platform_output);
                self.ui_model = draft;

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

        // Model switches rebuild the simulation; running after the frame means the
        // fresh buffers are only drawn on the next redraw.
        if let Some(model) = pending_model {
            if let Err(error) = self.rebuild_with_model(model) {
                self.sim.failed = true;
                tracing::error!("model rebuild failed: {error}");
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

    let mut sim = start_simulation(&settings_path, exaggeration)?;

    if let Some(path) = snapshot {
        sim.run_to_completion()?;
        write_snapshot(&sim, Overlay::PeakFlowVelocity, &path)?;
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    // The viewer switches to WaitUntil dynamically; idle frames cost nothing.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut Viewer::new(sim))?;
    Ok(())
}
