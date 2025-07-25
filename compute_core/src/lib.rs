use crate::buffers::{ComputeBuffers, create_buffers_and_texture_descriptions};
use crate::shaders::ComputeShaderConfig;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use wgpu::{
    Adapter, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    Queue, RequestAdapterOptions, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages,
};

pub mod buffers;
pub mod dem;
pub mod settings;
pub mod shaders;
pub mod utils;

pub struct ComputeOrchestrator {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub buffers: ComputeBuffers,
    shader_configs: HashMap<String, ComputeShaderConfig>,
}

impl ComputeOrchestrator {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        let max_workgroup_xy = utils::highest_power_of_two(
            (adapter.limits().max_compute_workgroup_size_x as f64).sqrt() as u32,
        );
        let max_invocations = adapter.limits().max_compute_invocations_per_workgroup;
        let max_workgroup_x = max_invocations;
        let max_storage = adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Compute Device"),
                required_features: Features::FLOAT32_FILTERABLE | Features::TIMESTAMP_QUERY,
                required_limits: Limits {
                    max_compute_workgroup_size_x: max_workgroup_x,
                    max_compute_workgroup_size_y: max_workgroup_xy,
                    max_compute_workgroup_size_z: 1,
                    max_compute_invocations_per_workgroup: max_invocations,
                    max_storage_buffer_binding_size: max_storage,
                    ..Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device and queue");

        let buffers = ComputeBuffers::new();
        let shader_configs = shaders::create_shader_configs(&device)?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            buffers,
            shader_configs,
        })
    }

    fn get_view(&self, name: &str) -> wgpu::BindingResource<'_> {
        let view = self
            .buffers
            .get_texture_view(name)
            .ok_or_else(|| anyhow!("Texture view '{}' not found", name))
            .expect("Texture view not found");
        wgpu::BindingResource::TextureView(view)
    }

    fn get_buffer_binding(&self, name: &str) -> wgpu::BindingResource {
        self.buffers
            .get_buffer(name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", name))
            .expect("Buffer not found")
            .as_entire_binding()
    }

    // `run_shader` now takes `resources` directly for flexibility
    pub async fn run_shader(
        &self,
        shader_name: &str,
        resources: &[wgpu::BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
        dispatch_workgroup_x: u32,
        dispatch_workgroup_y: u32,
        dispatch_workgroup_z: u32,
    ) -> Result<()> {
        let config = self
            .shader_configs
            .get(shader_name)
            .ok_or_else(|| anyhow!("Shader '{}' not found", shader_name))?;

        let bind_group = config.create_bind_group(&self.device, resources)?;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Pass", shader_name)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&config.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                dispatch_workgroup_x,
                dispatch_workgroup_y,
                dispatch_workgroup_z,
            );
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    pub async fn run_normals_shader(
        &self,
        shader_name: &str,
        resources: &[wgpu::BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
        dispatch_workgroup_x: u32,
        dispatch_workgroup_y: u32,
        dispatch_workgroup_z: u32,
    ) -> Result<()> {
        let config = self
            .shader_configs
            .get(shader_name)
            .ok_or_else(|| anyhow!("Shader '{}' not found", shader_name))?;

        let bind_group = config.create_bind_group(&self.device, resources)?;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Pass", shader_name)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&config.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                dispatch_workgroup_x,
                dispatch_workgroup_y,
                dispatch_workgroup_z,
            );
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }
    pub fn create_buffers(&mut self, sim_settings: &settings::SimSettings) -> Result<()> {
        let texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };
        // Create buffers based on simulation settings and DEM data
        self.buffers = create_buffers_and_texture_descriptions(&self.device, texture_size);
        Ok(())
    }

    // --- Example of a texture-based chain ---
    pub async fn run_texture_processing_chain(
        &mut self, // `&mut self` because we're adding textures
        &sim_settings: &settings::SimSettings,
        dem: &dem::Dem,
    ) -> Result<Vec<f32>> {
        let texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };

        let texture_format = TextureFormat::R32Float; // Single float per pixel

        let dem_texture_desc = buffers::texture_descriptor(
            "dem_texture",
            texture_size,
            TextureFormat::R32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );

        let output_texture_desc = TextureDescriptor {
            label: Some("Output Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: texture_format,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // Add input texture with data
        self.buffers.add_texture_with_data(
            &self.device,
            &self.queue,
            dem.data1d.as_slice(),
            &dem_texture_desc,
        )?;

        // Add empty output texture
        self.buffers.add_texture(&self.device, &output_texture_desc);

        let input_view = self
            .buffers
            .get_texture_view("input_image")
            .ok_or_else(|| anyhow!("Input texture view not found"))?;
        let output_view = self
            .buffers
            .get_texture_view("output_image")
            .ok_or_else(|| anyhow!("Output texture view not found"))?;

        let dispatch_x = (sim_settings.grid_shape_x + 7) / 8; // Example: 8x8 workgroup size in shader
        let dispatch_y = (sim_settings.grid_shape_y + 7) / 8;

        println!("Running texture processing shader...");
        self.run_shader(
            "texture_process",
            &[
                wgpu::BindingResource::TextureView(input_view),
                wgpu::BindingResource::TextureView(output_view),
            ],
            dispatch_x,
            dispatch_y,
            1,
        )
        .await?;

        // Read the final output texture
        let result: Vec<f32> = self
            .buffers
            .read_texture(&self.device, &self.queue, "output_image")
            .await?;
        Ok(result)
    }

    pub async fn run_normals(
        &mut self, // `&mut self` because we're adding textures
        &sim_settings: &settings::SimSettings,
        dem: &dem::Dem,
    ) -> Result<()> {
        let texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };

        self.buffers = create_buffers_and_texture_descriptions(&self.device, texture_size);

        self.buffers.add_buffer_with_data(
            &self.device,
            "sim_settings",
            &sim_settings.as_bytes(),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.buffers
            .add_texture_with_data(
                &self.device,
                &self.queue,
                dem.data1d.as_slice(),
                &buffers::texture_descriptor(
                    "dem_texture",
                    texture_size,
                    TextureFormat::R32Float,
                    texture_usage_input,
                ),
            )
            .expect("Failed to add texture with data");
        println!("Running texture processing shader...");
        let _ = self.buffers
            .write_buffer(&self.queue, "sim_settings", &sim_settings.as_bytes());
        self.run_shader(
            "compute_normals",
            &[
                self.get_buffer_binding("sim_settings"),
                self.get_view("dem_texture"),
                self.get_view("wind_texture"),
                self.get_view("normals_texture"),
                self.get_view("slope_texture"),
                self.get_buffer_binding("out_debug_normals_buffer"),
            ],
            16,
            16,
            1,
        )
        .await?;
        Ok(())
    }
    async fn get_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: &str,
    ) -> Result<Vec<T>> {
        let result: Vec<T> = self
            .buffers
            .read_texture(&self.device, &self.queue, name)
            .await?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::settings::Settings;
    use half::f16;
    use super::*;

    #[test]
    fn test_compute_orchestrator_creation() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let (sim_settings, dem) = Settings::create_from_path("../avaframe/avaInclinedPlane.png");
        pollster::block_on(orchestrator.run_normals(&sim_settings, &dem))
            .expect("Failed to run normals shader");
        let slope_texture: Vec<f16> = pollster::block_on(orchestrator.get_texture("slope_texture"))
            .expect("Failed to get slope_texture");
        println!("Read slope_texture: {} {:?}", slope_texture.len(), slope_texture[200..220].to_vec());
        let d: Vec<f32> = pollster::block_on(orchestrator.buffers.read_buffer(&orchestrator.device, "out_debug_normals_buffer")).expect("Failed to read out_debug_normals_buffer");
        println!("Read out_debug_normals_buffer: {:?}", d);
    }

    #[test]
    fn test_compute() {
        let orchestrator = pollster::block_on(ComputeOrchestrator::new());
        assert!(orchestrator.is_ok());
    }
}
