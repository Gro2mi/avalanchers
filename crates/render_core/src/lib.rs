//! GPU rendering for avalanche simulation data.
//!
//! The crate is windowing-agnostic: it renders into any [`wgpu::TextureView`], so the same
//! code path serves a native window surface, a browser canvas surface, and offscreen targets.

pub mod camera;
#[cfg(not(target_arch = "wasm32"))]
pub mod capture;
pub mod math;
pub mod particles;
pub mod terrain;

pub use camera::OrbitCamera;
pub use math::Vec3;
pub use particles::{ParticleBuffers, ParticleRenderer};
pub use terrain::{OverlayRange, TerrainData, TerrainRenderer};

use anyhow::Result;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Device, queue and adapter used for rendering.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Creates a rendering context. Pass the target surface so the adapter is guaranteed
    /// to be able to present to it.
    pub async fn new(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await?;

        let info = adapter.get_info();
        tracing::info!(
            "Render adapter: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Render Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Creates a context for offscreen rendering, without a presentation surface.
    pub async fn headless() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        Self::new(instance, None).await
    }
}

/// Owns the render passes and the camera for a single view of the simulation.
pub struct Renderer {
    pub camera: OrbitCamera,
    pub clear_color: wgpu::Color,
    terrain: TerrainRenderer,
    particles: ParticleRenderer,
    depth_view: wgpu::TextureView,
    color_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        terrain: &TerrainData,
    ) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let terrain_renderer =
            TerrainRenderer::new(device, queue, color_format, DEPTH_FORMAT, terrain);
        let particles = ParticleRenderer::new(
            device,
            color_format,
            DEPTH_FORMAT,
            terrain_renderer.heightmap_view().clone(),
            terrain,
        );

        Self {
            camera: OrbitCamera::framing(terrain, width as f32 / height as f32),
            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.07,
                b: 0.11,
                a: 1.0,
            },
            terrain: terrain_renderer,
            particles,
            depth_view: create_depth_view(device, width, height),
            color_format,
            width,
            height,
        }
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    /// Tints the terrain with a scalar simulation grid such as peak flow velocity,
    /// peak flow thickness or grid mass. The buffer must hold `width * height` `f32`
    /// values indexed as `y * width + x` and live on the same device.
    pub fn set_grid_overlay(
        &mut self,
        device: &wgpu::Device,
        buffer: Option<&wgpu::Buffer>,
        range: OverlayRange,
    ) {
        self.terrain.set_overlay(device, buffer, range);
    }

    /// Attaches the simulation's particle buffers. Passing `None` hides the particles.
    pub fn set_particles(&mut self, device: &wgpu::Device, buffers: Option<ParticleBuffers<'_>>) {
        self.particles.set_buffers(device, buffers);
    }

    /// Rescales the overlay colour ramp, for example as the simulation's peak values grow.
    pub fn set_overlay_range(&mut self, range: OverlayRange) {
        self.terrain.set_overlay_range(range);
    }

    pub fn particles_mut(&mut self) -> &mut ParticleRenderer {
        &mut self.particles
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.camera.set_aspect(width, height);
        self.depth_view = create_depth_view(device, width, height);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
    ) {
        self.terrain.update_camera(queue, &self.camera);
        self.particles.update_camera(queue, &self.camera);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.terrain.draw(&mut pass);
            self.particles.draw(&mut pass);
        }
        queue.submit(Some(encoder.finish()));
    }
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
