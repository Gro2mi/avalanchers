//! Live avalanche simulation viewer.
//!
//! Each redraw advances the simulation by one step and then renders the simulation's own
//! storage buffers on the same wgpu queue. No particle or grid data is copied to the CPU.
//!
//! Usage: `cargo run -p render_core --example live_sim -- [data/avaframe/avaAlr.png] [exaggeration]`
//!
//! Controls: left drag orbits, right drag pans, scroll zooms, `R` resets the view,
//! `0` hides the overlay, `1` peak flow velocity, `2` peak flow thickness, `3` grid mass,
//! `P` toggles particles.

use std::sync::Arc;
use std::time::{Duration, Instant};

use compute_core::{SimInfo, buffers::BufferName};
use render_core::{OverlayRange, ParticleBuffers, Renderer, TerrainData};
use simulation::{Simulation, SimulationState};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Icon, Window, WindowId};

const ORBIT_SPEED: f32 = 0.005;
const ZOOM_SPEED: f32 = 0.1;
const STEPS_PER_FRAME: u32 = 1;
const DEFAULT_SIM_SPEED: f32 = 4.0;
const DEFAULT_DEM: &str = "data/avaframe/avaAlr.png";
const WINDOW_ICON: &[u8] = include_bytes!("../../../frontend/icons/android-chrome-512x512.png");

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
}

impl Overlay {
    fn buffer_name(self) -> Option<BufferName> {
        match self {
            Overlay::None => None,
            Overlay::PeakFlowVelocity => Some(BufferName::GridPeakVelocity),
            Overlay::PeakFlowThickness => Some(BufferName::GridPeakFlowThickness),
            Overlay::GridMass => Some(BufferName::GridMass),
        }
    }

    /// Colour ramp bounds in the units of each field.
    fn range(self) -> OverlayRange {
        match self {
            Overlay::None => OverlayRange::default(),
            Overlay::PeakFlowVelocity => OverlayRange::new(0.0, 30.0).with_threshold(0.1),
            Overlay::PeakFlowThickness => OverlayRange::new(0.0, 3.0).with_threshold(0.01),
            Overlay::GridMass => OverlayRange::new(0.0, 5_000.0).with_threshold(1.0),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Overlay::None => "none",
            Overlay::PeakFlowVelocity => "peak flow velocity",
            Overlay::PeakFlowThickness => "peak flow thickness",
            Overlay::GridMass => "grid mass",
        }
    }
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
    particles: [wgpu::Buffer; 3],
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

    fn advance_frame(&mut self) {
        if self.failed || self.simulation.get_state() >= SimulationState::Finished {
            self.report_final_info();
            return;
        }
        let now = Instant::now();
        if now < self.next_step_at {
            return;
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
    }

    fn adjust_speed(&mut self, factor: f32) {
        self.sim_speed = (self.sim_speed * factor).clamp(0.25, 256.0);
        self.next_step_at = Instant::now();
        tracing::info!("simulation speed: {:.2}x", self.sim_speed);
    }

    fn restart(&mut self) -> anyhow::Result<()> {
        self.simulation.reset();
        self.info = pollster::block_on(self.simulation.run_n_steps(0))?;

        let orchestrator = self.simulation.orchestrator();
        let clone_buffer = |name: BufferName| -> anyhow::Result<wgpu::Buffer> {
            orchestrator
                .resources
                .get_buffer(&name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("simulation buffer '{name}' is missing"))
        };
        self.grids = [
            BufferName::GridPeakVelocity,
            BufferName::GridPeakFlowThickness,
            BufferName::GridMass,
        ]
        .into_iter()
        .map(|name| Ok((name.clone(), clone_buffer(name)?)))
        .collect::<anyhow::Result<Vec<_>>>()?;
        self.particles = [
            clone_buffer(BufferName::ParticlesPosition)?,
            clone_buffer(BufferName::ParticlesVelocity)?,
            clone_buffer(BufferName::ParticlesStopped)?,
        ];
        self.particle_count = self.info.number_particles;
        self.failed = false;
        self.final_info_reported = false;
        self.step_timings.clear();
        self.next_step_at = Instant::now();
        tracing::info!("simulation restarted");
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

fn start_simulation(dem_path: &str, exaggeration: f32) -> anyhow::Result<SimulationView> {
    let mut sim = pollster::block_on(Simulation::new())?;
    pollster::block_on(sim.create_example(dem_path))?;
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

    let grids = [
        BufferName::GridPeakVelocity,
        BufferName::GridPeakFlowThickness,
        BufferName::GridMass,
    ]
    .into_iter()
    .map(|name| Ok((name.clone(), clone_buffer(name)?)))
    .collect::<anyhow::Result<Vec<_>>>()?;

    let particles = [
        clone_buffer(BufferName::ParticlesPosition)?,
        clone_buffer(BufferName::ParticlesVelocity)?,
        clone_buffer(BufferName::ParticlesStopped)?,
    ];

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
            stopped: &sim.particles[2],
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
}

struct Viewer {
    sim: SimulationView,
    window: Option<ViewerWindow>,
    cursor: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
    overlay: Overlay,
    show_particles: bool,
}

impl Viewer {
    fn new(sim: SimulationView) -> Self {
        Self {
            sim,
            window: None,
            cursor: None,
            orbiting: false,
            panning: false,
            overlay: Overlay::PeakFlowVelocity,
            show_particles: true,
        }
    }

    fn apply_overlay(&mut self) {
        let Some(view) = self.window.as_mut() else {
            return;
        };
        let buffer = self.overlay.buffer_name().map(|name| self.sim.grid(&name));
        view.renderer
            .set_grid_overlay(&self.sim.device, buffer, self.overlay.range());
        tracing::info!("overlay: {}", self.overlay.label());
    }

    fn apply_particles(&mut self) {
        let Some(view) = self.window.as_mut() else {
            return;
        };
        let buffers = self.show_particles.then(|| ParticleBuffers {
            position: &self.sim.particles[0],
            velocity: &self.sim.particles[1],
            stopped: &self.sim.particles[2],
        });
        view.renderer.set_particles(&self.sim.device, buffers);
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

        self.window = Some(ViewerWindow {
            window,
            surface,
            config,
            renderer,
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
                }
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 60.0,
                };
                if let Some(view) = self.window.as_mut() {
                    view.renderer.camera.zoom(scroll * ZOOM_SPEED);
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
                    Key::Character("p" | "P") => {
                        self.show_particles = !self.show_particles;
                        self.apply_particles();
                    }
                    Key::Character("+") => self.sim.adjust_speed(2.0),
                    Key::Character("-") => self.sim.adjust_speed(0.5),
                    Key::Character("v" | "V") => {
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
                    Key::Character("r" | "R") => {
                        if let Some(view) = self.window.as_mut() {
                            let aspect =
                                view.config.width as f32 / view.config.height.max(1) as f32;
                            view.renderer.camera =
                                render_core::OrbitCamera::framing(&self.sim.terrain, aspect);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let device = self.sim.device.clone();
                let queue = self.sim.queue.clone();
                let Some(view) = self.window.as_mut() else {
                    return;
                };

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

                self.sim.advance_frame();
                let title = if self.sim.failed {
                    "Avalanchers - Simulation Failed".to_string()
                } else if self.sim.simulation.get_state() < SimulationState::Finished {
                    format!(
                        "Avalanchers - step {} - {:.2} s - {:.2}x",
                        self.sim.info.timestep, self.sim.info.elapsed_time, self.sim.sim_speed
                    )
                } else {
                    format!(
                        "Avalanchers - Simulation Finished - step {} - {:.2} s - {:.2}x",
                        self.sim.info.timestep, self.sim.info.elapsed_time, self.sim.sim_speed
                    )
                };
                view.window.set_title(&title);
                let target = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                view.renderer.render(&device, &queue, &target);
                queue.present(frame);
                view.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(view) = self.window.as_ref() {
            view.window.request_redraw();
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
    let dem_path = args.next().unwrap_or_else(|| DEFAULT_DEM.to_string());
    let exaggeration: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let snapshot = std::env::var("LIVE_SIM_SNAPSHOT").ok();
    println!(
        "DEM path: {dem_path} exaggeration: {exaggeration} snapshot: {:?}",
        snapshot
    );

    let mut sim = start_simulation(&dem_path, exaggeration)?;

    if let Some(path) = snapshot {
        sim.run_to_completion()?;
        write_snapshot(&sim, Overlay::PeakFlowVelocity, &path)?;
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut Viewer::new(sim))?;
    Ok(())
}
