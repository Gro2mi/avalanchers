use std::fs;
use std::num::NonZero;

use anyhow::Result;
use regex::Regex;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureViewDimension,
};
pub const SHADER_UTILS: &str = include_str!("../wgsl/utils.wgsl");

fn read_shader_source(name: &str) -> String {
    let path = format!("wgsl/{}.wgsl", name);
    let shader = fs::read_to_string(&path).expect(&format!("Failed to read shader file {}", &path));
    shader
}
fn load_shader_source(name: &str, workgroup_size_1d: u32, workgroup_size_2d: u32) -> &'static str {
    let import_re = Regex::new(r#"import\s+([a-zA-Z0-9_./-]+)\.wgsl"#).unwrap();
    let shader_source = read_shader_source(name)
        .replace("WORKGROUP_SIZE_1D", &workgroup_size_1d.to_string())
        .replace("WORKGROUP_SIZE_2D", &workgroup_size_2d.to_string());
    let processed = import_re.replace_all(&shader_source, |caps: &regex::Captures| {
        let import_name = &caps[1];
        load_shader_source(import_name, workgroup_size_1d, workgroup_size_2d)
    });
    Box::leak(processed.into_owned().into_boxed_str())
}

// pub const SHADER_COMPUTE_NORMALS: &str = concat!(
//     include_str!("../../wgsl/utils.wgsl"),
//     "\n",
//     include_str!("../../wgsl/compute_normals.wgsl"),
// );
// pub const SHADER_RESET_MAX_VELOCITY: &str = include_str!("../../wgsl/reset_max_velocity.wgsl");
// pub const SHADER_LOAD_RELEASE_AREAS: &str = concat!(
//     include_str!("../../wgsl/utils.wgsl"),
//     "\n",
//     include_str!("../../wgsl/load_release_areas.wgsl")
// );
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&format!("{} Compute Pipeline", name)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&name),
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
    workgroup_size_1d: u32,
    workgroup_size_2d: u32,
) -> Result<std::collections::HashMap<String, ComputeShaderConfig>> {
    let mut shader_configs = std::collections::HashMap::new();
    shader_configs.insert(
        "compute_normals".to_string(),
        ComputeShaderConfig::new(
            &device,
            "compute_normals".to_string(),
            load_shader_source("compute_normals", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Uniform buffer (sim_settings_buffer)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Texture (dem_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 2: Texture (wind_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                // Binding 3: Texture (normals_texture)
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                // Binding 4: Texture (slope_texture)
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                // Binding 5: Storage buffer (out_debug_normals)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );

    shader_configs.insert(
        "load_release_areas".to_string(),
        ComputeShaderConfig::new(
            &device,
            "load_release_areas".to_string(),
            load_shader_source("load_release_areas", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Texture (release_areas_in)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Uint,
                },
                // Binding 1: Storage texture (release_areas_out)
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                // Binding 2: Storage buffer (out_debug_release)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 3: Storage buffer (number_release_cells)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );

    shader_configs.insert(
        "roughness".to_string(),
        ComputeShaderConfig::new(
            &device,
            "roughness".to_string(),
            load_shader_source("roughness", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Uniform buffer (sim_settings_buffer)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Texture (normals_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 2: Texture (landcover_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Uint,
                },
                // Binding 3: Storage texture (roughness_texture)
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
            ],
        )?,
    );

    shader_configs.insert(
        "compute_release_areas".to_string(),
        ComputeShaderConfig::new(
            &device,
            "compute_release_areas".to_string(),
            load_shader_source(
                "compute_release_areas",
                workgroup_size_1d,
                workgroup_size_2d,
            ),
            &[
                // Binding 0: Uniform buffer (sim_settings_buffer)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Texture (dem_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 2: Texture (slope_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 3: Texture (roughness_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 4: Storage texture (release_areas_out)
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                // Binding 5: Storage buffer (out_debug_release)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 6: Storage buffer (out_debug_release)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );

    shader_configs.insert(
        "initialize_particles".to_string(),
        ComputeShaderConfig::new(
            &device,
            "initialize_particles".to_string(),
            load_shader_source("initialize_particles", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Uniform buffer (sim_settings)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Uniform buffer (sim_info)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 2: Texture (dem_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 3: Storage texture (release_areas)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 4: Sampler
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                // Binding 5: Buffer particles
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 6: Buffer number of released particles
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZero::new(4).unwrap()),
                },
                // Binding 7: atomic_cell_count_buffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 8: atomic_velocity_buffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );
    shader_configs.insert(
        "compute_particles".to_string(),
        ComputeShaderConfig::new(
            &device,
            "compute_particles".to_string(),
            load_shader_source("compute_particles", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Uniform buffer (sim_settings)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Uniform buffer (sim_info)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 2: Texture (dem_texture)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 3: Storage texture (release_areas)
                BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                // Binding 4: Sampler
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                // Binding 5: Buffer particles
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 6: maxVelocityAtomicBuffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 7: atomic_cellcount_buffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 8: atomic_velocity_buffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 9: timestep_data
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 10:  debug_buffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );
    shader_configs.insert(
        "reset_max_velocity".to_string(),
        ComputeShaderConfig::new(
            &device,
            "reset_max_velocity".to_string(),
            load_shader_source("reset_max_velocity", workgroup_size_1d, workgroup_size_2d),
            &[
                // Binding 0: Uniform buffer (sim_settings)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 1: Uniform buffer (sim_info)
                BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                // Binding 2: maxVelocityAtomicBuffer
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ],
        )?,
    );
    Ok(shader_configs)
}
