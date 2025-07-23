use anyhow::Result;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

pub const SHADER_COMPUTE_NORMALS: &str = include_str!("../../wgsl/computeNormals.wgsl");
pub const SHADER_UTILS: &str = include_str!("../../wgsl/util.wgsl");
pub const SHADER_RESET_MAX_VELOCITY: &str = include_str!("../../wgsl/resetMaxVelocity.wgsl");
pub const SHADER_LOAD_RELEASE_POINTS: &str = include_str!("../../wgsl/loadReleasePoints.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");
// pub const SHADER_: &str = include_str!("../../wgsl/.wgsl");

pub struct ComputeShaderConfig {
    pub name: String,
    pub shader_src: &'static str,
    pub bind_group_layout: BindGroupLayout,
    pub pipeline: ComputePipeline,
}

impl ComputeShaderConfig {
    pub fn new(
        device: &Device,
        name: String,
        shader_src: &'static str,
        bindings: &[BindingType], // Now takes BindingType directly
    ) -> Result<Self> {
        let mut bgl_entries = Vec::new();
        for (i, binding_type) in bindings.iter().enumerate() {
            bgl_entries.push(BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: binding_type.clone(), // Clone BindingType
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", name)),
            entries: &bgl_entries,
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", name)),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&format!("{} Compute Pipeline", name)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            name,
            shader_src,
            bind_group_layout,
            pipeline,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &[wgpu::BindingResource],
    ) -> Result<BindGroup> {
        let mut bg_entries = Vec::new();
        for (i, resource) in resources.iter().enumerate() {
            bg_entries.push(BindGroupEntry {
                binding: i as u32,
                resource: resource.clone(),
            });
        }
        Ok(device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", self.name)),
            layout: &self.bind_group_layout,
            entries: &bg_entries,
        }))
    }
}

pub fn create_shader_configs(
    device: &Device,
) -> Result<std::collections::HashMap<String, ComputeShaderConfig>> {
    let mut shader_configs = std::collections::HashMap::new();

    shader_configs.insert(
        "compute_normals".to_string(),
        ComputeShaderConfig::new(
            &device,
            "compute_normals".to_string(),
            crate::shaders::SHADER_COMPUTE_NORMALS,
            &[
                // Binding 0: Uniform buffer (simSettingsBuffer)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Texture (demTexture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                // Binding 2: Texture (windTexture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                // Binding 3: Texture (normalsTexture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                // Binding 4: Texture (slopeTexture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                // Binding 5: Storage buffer (outDebugNormals)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );
    Ok(shader_configs)
}
