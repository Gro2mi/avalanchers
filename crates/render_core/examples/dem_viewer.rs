//! Native DEM viewer.
//!
//! Usage: `cargo run -p render_core --example dem_viewer -- [path/to/dem.asc] [vertical_exaggeration]`
//! cargo run -q -p render_core --example dem_viewer -- "data/avaframe/avaGar_remeshedDEM5.00.asc"
//! Without a path a synthetic ridge is rendered.
//!
//! Controls: left drag orbits, right drag pans, scroll zooms, `R` resets the view.

use std::sync::Arc;

use render_core::{GpuContext, Renderer, TerrainData};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const ORBIT_SPEED: f32 = 0.005;
const ZOOM_SPEED: f32 = 0.1;

fn synthetic_terrain() -> TerrainData {
    let (width, height) = (256u32, 256u32);
    let heights = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                let ridge = (fx * std::f32::consts::PI * 2.0).sin() * 180.0;
                let valley = (fy * std::f32::consts::PI).cos() * 260.0;
                1500.0 + ridge + valley
            })
        })
        .collect();
    TerrainData::new(width, height, 20.0, heights).expect("synthetic terrain is valid")
}

fn load_terrain() -> TerrainData {
    let mut args = std::env::args().skip(1);
    let path = args.next();
    let exaggeration: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.3);

    let terrain = match path {
        Some(path) => {
            let dem = pollster::block_on(data_processor::load_dem(&path))
                .unwrap_or_else(|e| panic!("failed to load DEM '{path}': {e}"));
            tracing::info!(
                "Loaded DEM {}x{} at {} m resolution",
                dem.width,
                dem.height,
                dem.cell_size
            );
            TerrainData::new(
                dem.width as u32,
                dem.height as u32,
                dem.cell_size,
                dem.data1d,
            )
            .expect("DEM is not a valid terrain grid")
        }
        None => {
            tracing::info!("No DEM path given, rendering a synthetic terrain");
            synthetic_terrain()
        }
    };

    terrain.with_vertical_exaggeration(exaggeration)
}

struct ViewerWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

#[derive(Default)]
struct Viewer {
    terrain: Option<TerrainData>,
    window: Option<ViewerWindow>,
    cursor: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
}

impl Viewer {
    fn terrain(&mut self) -> &TerrainData {
        self.terrain.get_or_insert_with(load_terrain)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let device = &gpu().device;
        let Some(view) = self.window.as_mut() else {
            return;
        };
        view.config.width = width;
        view.config.height = height;
        view.surface.configure(device, &view.config);
        view.renderer.resize(device, width, height);
    }
}

// The context outlives every window and is shared by all of them.
static GPU: std::sync::OnceLock<GpuContext> = std::sync::OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get().expect("GPU context is initialised on resume")
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Avalanchers - DEM Viewer"))
                .expect("failed to create window"),
        );

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(window.clone())),
        );
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let gpu = GPU.get_or_init(|| {
            pollster::block_on(GpuContext::new(instance, Some(&surface)))
                .expect("failed to create GPU context")
        });

        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));
        let config = surface
            .get_default_config(&gpu.adapter, width, height)
            .expect("surface is not supported by this adapter");
        surface.configure(&gpu.device, &config);

        let terrain = self.terrain().clone();
        let renderer = Renderer::new(
            &gpu.device,
            &gpu.queue,
            config.format,
            width,
            height,
            &terrain,
        );

        self.window = Some(ViewerWindow {
            window,
            surface,
            config,
            renderer,
        });
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
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key.as_ref(), Key::Character("r" | "R"))
                    && let (Some(terrain), Some(view)) =
                        (self.terrain.as_ref(), self.window.as_mut())
                {
                    let aspect = view.config.width as f32 / view.config.height.max(1) as f32;
                    view.renderer.camera = render_core::OrbitCamera::framing(terrain, aspect);
                }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape)
                {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                let gpu = gpu();
                let Some(view) = self.window.as_mut() else {
                    return;
                };

                let frame = match view.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        view.surface.configure(&gpu.device, &view.config);
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
                view.renderer.render(&gpu.device, &gpu.queue, &target);
                gpu.queue.present(frame);
                view.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut Viewer::default())
        .expect("viewer crashed");
}
