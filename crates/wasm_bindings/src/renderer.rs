//! Live canvas rendering for a running simulation, the wasm counterpart of the
//! native `live_sim` example.
//!
//! The renderer lives on the simulation's own device and binds clones of the
//! orchestrator's storage buffers, so the frame loop never copies particle or
//! grid data back to the CPU. Input events stay in JS and drive the camera
//! through the `WasmSimulation` binding.

use anyhow::{Result, anyhow};
use compute_core::ComputeOrchestrator;
use compute_core::buffers::BufferName;
use compute_core::dem::Dem;
use render_core::{OrbitCamera, OverlayRange, ParticleBuffers, Renderer, TerrainData};

/// Camera input scaling shared with the native viewer, so both feel the same
/// for the same pixel deltas.
const ORBIT_SPEED: f32 = 0.005;
const ZOOM_SPEED: f32 = 0.1;

/// Grid buffers cloned from the simulation, both at attach time and after a rebuild.
const GRID_BUFFERS: [BufferName; 7] = [
    BufferName::GridPeakVelocity,
    BufferName::GridPeakFlowThickness,
    BufferName::GridMass,
    BufferName::ReleaseAreas,
    BufferName::SlopeAngle,
    BufferName::SlopeAspect,
    BufferName::Roughness,
];

#[derive(Clone, Copy, PartialEq)]
pub enum Overlay {
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
    /// Names accepted by `WasmSimulation::set_overlay`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "peak_velocity" => Some(Self::PeakFlowVelocity),
            "peak_flow_thickness" => Some(Self::PeakFlowThickness),
            "grid_mass" => Some(Self::GridMass),
            "release_areas" => Some(Self::ReleaseAreas),
            "slope_angle" => Some(Self::SlopeAngle),
            "slope_aspect" => Some(Self::SlopeAspect),
            "roughness" => Some(Self::Roughness),
            _ => None,
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

/// Clones whichever grid buffers exist. Before `prepare` none of them do, so the
/// view renders bare terrain until the simulation provides them.
fn clone_grid_buffers(orchestrator: &ComputeOrchestrator) -> Vec<(BufferName, wgpu::Buffer)> {
    GRID_BUFFERS
        .iter()
        .filter_map(|name| {
            orchestrator
                .resources
                .get_buffer(name)
                .cloned()
                .map(|buffer| (name.clone(), buffer))
        })
        .collect()
}

/// Clones the particle buffers the renderer binds, or `None` before the
/// simulation initializes particles. MPM keeps no vertical velocity buffer, so it
/// gets a zero-filled stand-in and colouring falls back to horizontal speed.
fn clone_particle_buffers(
    orchestrator: &ComputeOrchestrator,
    particle_count: u32,
) -> Option<[wgpu::Buffer; 5]> {
    use wgpu::util::DeviceExt;

    let get = |name: BufferName| orchestrator.resources.get_buffer(&name).cloned();
    let position = get(BufferName::ParticlesPosition)?;
    let velocity = get(BufferName::ParticlesVelocity)?;
    let stopped = get(BufferName::ParticlesStopped)?;
    let elevation = get(BufferName::ParticlesElevation)?;
    let velocity_z = get(BufferName::ParticlesVelocityZ).unwrap_or_else(|| {
        tracing::debug!("no vertical velocity buffer; colouring by horizontal speed");
        orchestrator
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Zero Vertical Velocity"),
                contents: bytemuck::cast_slice(&vec![0.0f32; particle_count as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    });

    Some([position, velocity, velocity_z, stopped, elevation])
}

/// The canvas surface plus the renderer watching the simulation's GPU buffers.
///
/// wgpu resources are reference counted, so the renderer observes the buffers
/// `run_n_steps` keeps writing to.
pub struct RenderView {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub renderer: Renderer,
    pub terrain: TerrainData,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    grids: Vec<(BufferName, wgpu::Buffer)>,
    particles: Option<[wgpu::Buffer; 5]>,
    pub show_particles: bool,
    pub overlay: Overlay,
}

impl RenderView {
    /// Builds the view on the simulation's device. Grid and particle buffers are
    /// bound only if they exist: before `prepare` the view shows bare terrain.
    /// The canvas `width`/`height` attributes must be set already.
    pub fn new(
        orchestrator: &ComputeOrchestrator,
        canvas: web_sys::HtmlCanvasElement,
        dem: &Dem,
        exaggeration: f32,
        particle_count: u32,
    ) -> Result<Self> {
        let device = orchestrator.device.clone();
        let queue = orchestrator.queue.clone();
        let canvas_size = (canvas.width(), canvas.height());
        let surface = orchestrator
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;
        let (width, height) = (canvas_size.0.max(1), canvas_size.1.max(1));
        let config = surface
            .get_default_config(&orchestrator.adapter, width, height)
            .ok_or_else(|| anyhow!("the simulation adapter cannot present to this canvas"))?;
        surface.configure(&device, &config);

        let terrain = TerrainData::new(
            dem.width as u32,
            dem.height as u32,
            dem.cell_size,
            dem.data1d.clone(),
        )?
        .with_vertical_exaggeration(exaggeration);

        let mut renderer = Renderer::new(&device, &queue, config.format, width, height, &terrain);
        renderer.particles_mut().set_count(particle_count);
        renderer.particles_mut().set_max_velocity(30.0);
        renderer
            .particles_mut()
            .set_radius(terrain.cell_size() * 0.9);

        let mut view = Self {
            surface,
            config,
            renderer,
            terrain,
            device,
            queue,
            grids: clone_grid_buffers(orchestrator),
            particles: clone_particle_buffers(orchestrator, particle_count),
            show_particles: true,
            overlay: Overlay::PeakFlowVelocity,
        };
        view.apply_overlay();
        view.apply_particles();
        Ok(view)
    }

    /// Re-clones the GPU buffers after the simulation was reset or rebuilt. Model
    /// switches change which buffers exist, so this must run after every rebuild.
    pub fn refresh_buffers(&mut self, orchestrator: &ComputeOrchestrator, particle_count: u32) {
        self.grids = clone_grid_buffers(orchestrator);
        self.particles = clone_particle_buffers(orchestrator, particle_count);
        let bound_count = if self.particles.is_some() {
            particle_count
        } else {
            0
        };
        self.renderer.particles_mut().set_count(bound_count);
        self.apply_overlay();
        self.apply_particles();
    }

    pub fn apply_overlay(&mut self) {
        // Overlays whose grid does not exist yet (before `prepare`) stay unbound
        // and the terrain renders bare until `refresh_buffers` re-applies them.
        let buffer = self
            .overlay
            .buffer_name()
            .and_then(|name| self.grids.iter().find(|(n, _)| *n == name))
            .map(|(_, buffer)| buffer.clone());
        self.renderer
            .set_grid_overlay(&self.device, buffer.as_ref(), self.overlay.range());
    }

    pub fn apply_particles(&mut self) {
        let buffers = self
            .particles
            .as_ref()
            .filter(|_| self.show_particles)
            .map(|particles| ParticleBuffers {
                position: &particles[0],
                velocity: &particles[1],
                velocity_z: &particles[2],
                stopped: &particles[3],
                elevation: &particles[4],
            });
        self.renderer.set_particles(&self.device, buffers);
    }

    /// Renders one frame onto the canvas; the caller advances the simulation first.
    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return; // the next frame draws
            }
            other => {
                tracing::warn!("Skipping frame: {other:?}");
                return;
            }
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(&self.device, &self.queue, &target);
        self.queue.present(frame);
    }

    /// Rotates the camera; inputs are pixel deltas from a drag.
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.renderer
            .camera
            .orbit(delta_x * ORBIT_SPEED, delta_y * ORBIT_SPEED);
    }

    /// Moves the target in the camera's screen plane; inputs are pixel deltas.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let (width, height) = self.renderer.size();
        self.renderer.camera.pan(
            delta_x / width.max(1) as f32,
            delta_y / height.max(1) as f32,
        );
    }

    /// Moves towards or away from the target; positive `delta` zooms in.
    pub fn zoom(&mut self, delta: f32) {
        self.renderer.camera.zoom(delta * ZOOM_SPEED);
    }

    /// Resets the camera to the default terrain framing.
    pub fn reset_view(&mut self) {
        let (width, height) = self.renderer.size();
        self.renderer.camera =
            OrbitCamera::framing(&self.terrain, width as f32 / height.max(1) as f32);
    }
}
