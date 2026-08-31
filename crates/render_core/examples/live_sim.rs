//! Live avalanche simulation viewer.
//!
//! The simulation runs on a worker thread while the main thread renders. Both share one
//! wgpu device, and the renderer binds the simulation's own storage buffers, so particle
//! and grid state is displayed as it is produced - nothing is copied back to the CPU.
//!
//! Usage: `cargo run -p render_core --example live_sim -- [data/avaframe/avaAlr.png] [exaggeration]`
//!
//! Controls: left drag orbits, right drag pans, scroll zooms, `R` resets the view,
//! `0` hides the overlay, `1` peak flow velocity, `2` peak flow thickness, `3` grid mass,
//! `P` toggles particles.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use compute_core::buffers::BufferName;
use render_core::{OverlayRange, ParticleBuffers, Renderer, TerrainData};
use simulation::Simulation;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Icon, Window, WindowId};

const ORBIT_SPEED: f32 = 0.005;
const ZOOM_SPEED: f32 = 0.1;
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

/// Handles to the simulation's GPU state, cloned before the simulation moves to its
/// worker thread. wgpu resources are reference counted, so these keep pointing at the
/// buffers the simulation keeps writing to.
struct SimulationView {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    terrain: TerrainData,
    grids: Vec<(BufferName, wgpu::Buffer)>,
    particles: [wgpu::Buffer; 3],
    particle_count: u32,
    running: Arc<AtomicBool>,
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
}

fn start_simulation(
    dem_path: &str,
    exaggeration: f32,
    live: bool,
) -> anyhow::Result<SimulationView> {
    let mut sim = pollster::block_on(Simulation::new())?;
    pollster::block_on(sim.create_example(dem_path))?;
    tracing::info!(
        "Loaded DEM {}x{} at {} m resolution",
        sim.dem.width,
        sim.dem.height,
        sim.dem.cell_size
    );

    // Allocates and fills the particle buffers; they must exist before we clone the handles.
    pollster::block_on(sim.prepare())?;

    let terrain = TerrainData::new(
        sim.dem.width as u32,
        sim.dem.height as u32,
        sim.dem.cell_size,
        sim.dem.data1d.clone(),
    )?
    .with_vertical_exaggeration(exaggeration);

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

    let view = SimulationView {
        device: orchestrator.device.clone(),
        queue: orchestrator.queue.clone(),
        adapter: orchestrator.adapter.clone(),
        instance: orchestrator.instance.clone(),
        terrain,
        grids,
        particles,
        particle_count: sim.number_particles(),
        running: Arc::new(AtomicBool::new(true)),
    };

    let running = view.running.clone();
    if live {
        std::thread::spawn(move || {
            if let Err(e) = pollster::block_on(sim.compute_particles()) {
                tracing::error!("simulation failed: {e}");
            }
            running.store(false, Ordering::Relaxed);
            tracing::info!("simulation finished");
        });
    } else {
        pollster::block_on(sim.compute_particles())?;
        running.store(false, Ordering::Relaxed);

        let positions = pollster::block_on(
            sim.orchestrator()
                .read_buffer::<[f32; 2]>(BufferName::ParticlesPosition),
        )?;
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in positions.iter().take(view.particle_count as usize) {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        tracing::info!(
            "particle positions: x {min_x}..{max_x} y {min_y}..{max_y} (of {} read)",
            positions.len()
        );
    }

    Ok(view)
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
        .set_radius(sim.terrain.cell_size() * 6.0);

    let (min_e, max_e) = sim.terrain.elevation_range();
    let vp = renderer.camera.view_projection();
    let samples = sim.terrain.fit_samples();
    let max_ndc = samples
        .iter()
        .map(|p| {
            let n = render_core::math::transform_point(vp, *p);
            n.x.abs().max(n.y.abs())
        })
        .fold(0.0f32, f32::max);
    tracing::info!(
        "terrain extent {:?} elevation {min_e}..{max_e} target {:?} distance {} samples {} max_ndc {max_ndc}",
        sim.terrain.extent(),
        renderer.camera.target,
        renderer.camera.distance,
        samples.len(),
    );

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
                if let Some(view) = self.window.as_ref() {
                    let title = if self.sim.running.load(Ordering::Relaxed) {
                        "Avalanchers - Live Simulation"
                    } else {
                        "Avalanchers - Simulation Finished"
                    };
                    view.window.set_title(title);
                }
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

    let sim = start_simulation(&dem_path, exaggeration, snapshot.is_none())?;

    if let Some(path) = snapshot {
        write_snapshot(&sim, Overlay::PeakFlowVelocity, &path)?;
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut Viewer::new(sim))?;
    Ok(())
}
