use crate::camera::OrbitCamera;
use crate::math::Vec3;
use anyhow::{Result, bail};
use wgpu::util::DeviceExt;

/// Values at or below this are treated as no-data markers found in common DEM sources.
const NODATA_THRESHOLD: f32 = -1.0;

/// A regular elevation grid, stored row-major starting at the south-west corner.
#[derive(Clone, Debug)]
pub struct TerrainData {
    width: u32,
    height: u32,
    cell_size: f32,
    heights: Vec<f32>,
    /// 1.0 for real samples, 0.0 for no-data cells.
    valid: Vec<f32>,
    min_elevation: f32,
    max_elevation: f32,
    vertical_exaggeration: f32,
}

impl TerrainData {
    pub fn new(width: u32, height: u32, cell_size: f32, heights: Vec<f32>) -> Result<Self> {
        if width < 2 || height < 2 {
            bail!("terrain needs at least 2x2 samples, got {width}x{height}");
        }
        let expected = width as usize * height as usize;
        if heights.len() != expected {
            bail!(
                "height count {} does not match {width}x{height} = {expected}",
                heights.len()
            );
        }
        if cell_size <= 0.0 || !cell_size.is_finite() {
            bail!("cell_size must be positive, got {cell_size}");
        }

        let mut min_elevation = f32::INFINITY;
        let mut max_elevation = f32::NEG_INFINITY;
        for &h in heights.iter().filter(|h| Self::is_valid(**h)) {
            min_elevation = min_elevation.min(h);
            max_elevation = max_elevation.max(h);
        }
        if !min_elevation.is_finite() {
            bail!("terrain contains no valid elevation samples");
        }

        // No-data cells keep a finite height so neighbouring geometry stays well defined,
        // but they are flagged so the shader can drop them.
        let valid: Vec<f32> = heights
            .iter()
            .map(|h| if Self::is_valid(*h) { 1.0 } else { 0.0 })
            .collect();
        let heights = heights
            .into_iter()
            .map(|h| if Self::is_valid(h) { h } else { min_elevation })
            .collect();

        Ok(Self {
            width,
            height,
            cell_size,
            heights,
            valid,
            min_elevation,
            max_elevation,
            vertical_exaggeration: 1.0,
        })
    }

    fn is_valid(h: f32) -> bool {
        h.is_finite() && h > NODATA_THRESHOLD
    }

    pub fn with_vertical_exaggeration(mut self, factor: f32) -> Self {
        self.vertical_exaggeration = factor.max(0.0);
        self
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn vertical_exaggeration(&self) -> f32 {
        self.vertical_exaggeration
    }

    pub fn elevation_range(&self) -> (f32, f32) {
        (self.min_elevation, self.max_elevation)
    }

    /// World-space size of the terrain in the x (columns) and z (rows) directions.
    pub fn extent(&self) -> (f32, f32) {
        (
            (self.width - 1) as f32 * self.cell_size,
            (self.height - 1) as f32 * self.cell_size,
        )
    }

    pub fn center(&self) -> Vec3 {
        let (size_x, size_z) = self.extent();
        Vec3::new(
            size_x * 0.5,
            (self.min_elevation + self.max_elevation) * 0.5 * self.vertical_exaggeration,
            size_z * 0.5,
        )
    }

    /// A coarse sample of surface points, used to frame the camera on the terrain itself
    /// rather than on its bounding box, which the surface never fully fills.
    pub fn fit_samples(&self) -> Vec<Vec3> {
        const SAMPLES_PER_AXIS: u32 = 48;
        let step_x = ((self.width - 1) / SAMPLES_PER_AXIS).max(1);
        let step_y = ((self.height - 1) / SAMPLES_PER_AXIS).max(1);

        let mut points = Vec::new();
        let mut y = 0;
        loop {
            let mut x = 0;
            loop {
                if self.valid[(y * self.width + x) as usize] > 0.5 {
                    points.push(self.world_position(x, y));
                }
                if x == self.width - 1 {
                    break;
                }
                x = (x + step_x).min(self.width - 1);
            }
            if y == self.height - 1 {
                break;
            }
            y = (y + step_y).min(self.height - 1);
        }

        if points.is_empty() {
            points.push(self.center());
        }
        points
    }

    /// Height and validity interleaved for upload as an `Rg32Float` texture.
    fn texels(&self) -> Vec<[f32; 2]> {
        self.heights
            .iter()
            .zip(&self.valid)
            .map(|(h, v)| [*h, *v])
            .collect()
    }

    fn world_position(&self, x: u32, y: u32) -> Vec3 {
        let elevation = self.heights[(y * self.width + x) as usize] * self.vertical_exaggeration;
        Vec3::new(
            x as f32 * self.cell_size,
            elevation,
            y as f32 * self.cell_size,
        )
    }

    /// Number of vertices needed to draw the grid as a non-indexed triangle list.
    fn vertex_count(&self) -> u32 {
        (self.width - 1) * (self.height - 1) * 6
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainUniforms {
    view_proj: [[f32; 4]; 4],
    /// width, height, cell_size, vertical exaggeration
    grid: [f32; 4],
    /// min elevation, max elevation, unused, unused
    elevation: [f32; 4],
    /// direction towards the light
    light_dir: [f32; 4],
    /// enabled, min, max, threshold
    overlay: [f32; 4],
}

/// How a scalar simulation grid is mapped onto the terrain surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayRange {
    /// Value mapped to the start of the colour ramp.
    pub min: f32,
    /// Value mapped to the end of the colour ramp.
    pub max: f32,
    /// Cells at or below this value stay bare terrain.
    pub threshold: f32,
    /// Variable name rendered along the colour bar legend, e.g. "peak flow velocity".
    pub label: &'static str,
    /// Unit suffix rendered beside the colour bar legend, e.g. "m/s".
    pub unit: &'static str,
}

impl Default for OverlayRange {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            threshold: 0.0,
            label: "",
            unit: "",
        }
    }
}

impl OverlayRange {
    pub fn new(min: f32, max: f32) -> Self {
        Self {
            min,
            max,
            threshold: min,
            label: "",
            unit: "",
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    pub fn with_unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }
}

/// Draws a DEM as a shaded height field, optionally tinted by a scalar simulation grid.
/// The grid is expanded on the GPU from the vertex index, so no vertex or index buffers
/// are uploaded.
pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    heightmap_view: wgpu::TextureView,
    /// Bound when no simulation grid is attached, so the layout stays valid.
    placeholder_overlay: wgpu::Buffer,
    uniforms: TerrainUniforms,
    vertex_count: u32,
}

impl TerrainRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        terrain: &TerrainData,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: terrain.width,
            height: terrain.height,
            depth_or_array_layers: 1,
        };
        let heightmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DEM Heightmap"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &heightmap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&terrain.texels()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(terrain.width * 8),
                rows_per_image: Some(terrain.height),
            },
            size,
        );
        let heightmap_view = heightmap.create_view(&wgpu::TextureViewDescriptor::default());

        let (min_elevation, max_elevation) = terrain.elevation_range();
        let exaggeration = terrain.vertical_exaggeration;
        let uniforms = TerrainUniforms {
            view_proj: crate::math::IDENTITY,
            grid: [
                terrain.width as f32,
                terrain.height as f32,
                terrain.cell_size,
                exaggeration,
            ],
            elevation: [
                min_elevation * exaggeration,
                max_elevation * exaggeration,
                0.0,
                0.0,
            ],
            light_dir: {
                let dir = Vec3::new(-0.4, 0.75, 0.5).normalize();
                [dir.x, dir.y, dir.z, 0.0]
            },
            overlay: [0.0, 0.0, 1.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let placeholder_overlay = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Overlay Placeholder"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &heightmap_view,
            &placeholder_overlay,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            heightmap_view,
            placeholder_overlay,
            uniforms,
            vertex_count: terrain.vertex_count(),
        }
    }

    /// Tints the terrain with a scalar simulation grid of `width * height` `f32` values,
    /// indexed as `y * width + x`. Passing `None` returns to plain terrain shading.
    ///
    /// The buffer may be owned by another part of the application, such as the simulation
    /// itself, as long as it lives on the same device and has `STORAGE` usage.
    pub fn set_overlay(
        &mut self,
        device: &wgpu::Device,
        buffer: Option<&wgpu::Buffer>,
        range: OverlayRange,
    ) {
        self.uniforms.overlay = match buffer {
            Some(_) => [1.0, range.min, range.max, range.threshold],
            None => [0.0, range.min, range.max, range.threshold],
        };
        self.bind_group = create_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.heightmap_view,
            buffer.unwrap_or(&self.placeholder_overlay),
        );
    }

    /// The DEM heightmap, shared with passes that need to place geometry on the surface.
    pub fn heightmap_view(&self) -> &wgpu::TextureView {
        &self.heightmap_view
    }

    /// Updates the colour ramp bounds without rebuilding the bind group.
    pub fn set_overlay_range(&mut self, range: OverlayRange) {
        self.uniforms.overlay[1] = range.min;
        self.uniforms.overlay[2] = range.max;
        self.uniforms.overlay[3] = range.threshold;
    }

    pub fn update_camera(&mut self, queue: &wgpu::Queue, camera: &OrbitCamera) {
        self.uniforms.view_proj = camera.view_projection();
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..self.vertex_count, 0..1);
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    heightmap: &wgpu::TextureView,
    overlay: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Terrain Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(heightmap),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: overlay.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_height_count() {
        let err = TerrainData::new(4, 4, 1.0, vec![0.0; 15]).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[test]
    fn rejects_degenerate_grid() {
        let err = TerrainData::new(1, 4, 1.0, vec![0.0; 4]).unwrap_err();
        assert!(err.to_string().contains("at least 2x2"), "{err}");
    }

    #[test]
    fn nodata_is_replaced_with_minimum_valid_elevation() {
        let terrain = TerrainData::new(2, 2, 5.0, vec![100.0, -9999.0, 300.0, f32::NAN]).unwrap();
        assert_eq!(terrain.elevation_range(), (100.0, 300.0));
        assert_eq!(terrain.heights(), &[100.0, 100.0, 300.0, 100.0]);
    }

    #[test]
    fn extent_and_vertex_count_follow_grid_size() {
        let terrain = TerrainData::new(4, 3, 10.0, vec![0.0; 12]).unwrap();
        assert_eq!(terrain.extent(), (30.0, 20.0));
        assert_eq!(terrain.vertex_count(), 3 * 2 * 6);
    }
}
