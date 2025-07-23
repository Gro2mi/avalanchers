use crate::buffers::ComputeBuffers;
use crate::shaders::ComputeShaderConfig;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use wgpu::{
    Adapter, BindingType, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages,
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
        let max_workgroup_x = adapter.limits().max_compute_workgroup_size_x;
        let max_invocations = adapter.limits().max_compute_invocations_per_workgroup;
        let max_workgroup_y =
            adapter.limits().max_compute_invocations_per_workgroup / max_workgroup_x;
        let max_storage = adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Compute Device"),
                required_features: Features::FLOAT32_FILTERABLE | Features::TIMESTAMP_QUERY,
                required_limits: Limits {
                    max_compute_workgroup_size_x: max_workgroup_x,
                    max_compute_workgroup_size_y: max_workgroup_y,
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

    // --- Example of a texture-based chain ---
    pub async fn run_texture_processing_chain(
        &mut self, // `&mut self` because we're adding textures
        width: u32,
        height: u32,
        dem_texture_data: &[f32],
    ) -> Result<Vec<f32>> {
        let texture_size = Extent3d {
            width,
            height,
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
            "input_image".to_string(),
            dem_texture_data,
            &dem_texture_desc,
        )?;

        // Add empty output texture
        self.buffers.add_texture(
            &self.device,
            "output_image".to_string(),
            &output_texture_desc,
        );

        let input_view = self
            .buffers
            .get_texture_view("input_image")
            .ok_or_else(|| anyhow!("Input texture view not found"))?;
        let output_view = self
            .buffers
            .get_texture_view("output_image")
            .ok_or_else(|| anyhow!("Output texture view not found"))?;

        let dispatch_x = (width + 7) / 8; // Example: 8x8 workgroup size in shader
        let dispatch_y = (height + 7) / 8;

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
        let result = self
            .buffers
            .read_texture(&self.device, &self.queue, "output_image")
            .await?;
        Ok(result)
    }
}
