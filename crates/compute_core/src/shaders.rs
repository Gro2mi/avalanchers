use crate::buffers::AtomicValues;
use crate::buffers::{BufferName, GpuResources, TextureName};
use anyhow::Result;
use core::panic;
use regex::Regex;
use std::num::NonZero;
use tracing::error;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureViewDimension,
};
pub const SHADER_UTILS: &str = include_str!("shaders/utils.wgsl");

macro_rules! define_shaders {
    ($($variant:ident => $filename:expr),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ShaderName {
            $($variant),*
        }

        impl std::str::FromStr for ShaderName {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($filename => Ok(ShaderName::$variant),)*
                    _ => Err(format!("'{}' is not a valid ShaderName", s)),
                }
            }
        }

        impl ShaderName {
            // Added: Helper to go from string (like "compute_normals") to Enum
            pub fn to_str(&self) -> &'static str {
                match self {
                    $(ShaderName::$variant => $filename,)*
                }
            }

            pub fn read_source(&self) -> String {
                #[cfg(any(target_arch = "wasm32", not(debug_assertions)))]
                {
                    match self {
                        $(ShaderName::$variant => include_str!(concat!("shaders/", $filename, ".wgsl")).to_string()),*
                    }
                }

                #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
                {
                    tracing::debug!("Loading shader source for {:?} from disk", self);
                    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("src").join("shaders").join(format!("{}.wgsl", self.to_str()));
                    std::fs::read_to_string(&path).expect("Shader file missing")
                }
            }
        }
    }
}

define_shaders! {
    AnalyzeTerrainCurvilinear => "analyze_terrain_curvilinear",
    AnalyzeTerrain => "analyze_terrain",
    ResetGrid => "reset_grid",
    ComputeRoughness => "compute_roughness",
    ComputeReleaseAreas => "compute_release_areas",
    InitializeParticles => "initialize_particles",
    ComputeParticles => "compute_particles",
    G2P => "g2p",
    P2GMPM => "p2g_mpm",
    P2G => "p2g",
    GridPhysicsMPM => "grid_physics_mpm",
    GridPhysics => "grid_physics",
    Utils => "utils",
    Random => "random",
    UpdateSimInfo => "update_sim_info",
    TestTransfer => "test_transfer",
    TransferG2P => "transfer_g2p",
    TransferG2PAffine => "transfer_g2p_affine",
    TransferP2G => "transfer_p2g",
    TestTransforms => "test_transforms",
    TestSampling => "test_sampling",
    TestUtils => "test_utils",
}

impl std::fmt::Display for ShaderName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

use std::sync::OnceLock;

// Using OnceLock for Regex is faster than Re-compiling every call
static STRIP_RE: OnceLock<Regex> = OnceLock::new();
static IMPORT_RE: OnceLock<Regex> = OnceLock::new();

fn load_shader_source_string(name_str: &str, atomic_float_support: bool) -> &'static str {
    // Get the enum variant from the string
    let shader_enum: ShaderName = name_str
        .parse()
        .unwrap_or_else(|_| panic!("Unknown shader name: {}", name_str));

    // Step 0: Get the raw source (from Disk or Binary)
    let shader_source = shader_enum.read_source();

    let strip_re = STRIP_RE.get_or_init(|| {
        Regex::new(r#"(?m)\s*// BEGIN [a-zA-Z0-9_./-]+\.wgsl[\s\S]*?// END [a-zA-Z0-9_./-]+\.wgsl"#)
            .unwrap()
    });

    let import_re = IMPORT_RE
        .get_or_init(|| Regex::new(r#"(?m)^//\s+import\s+([a-zA-Z0-9_./-]+)\.wgsl;?"#).unwrap());

    // Step 1: Strip
    let clean_source = strip_re.replace_all(&shader_source, "");

    // Step 2: Recursive Import
    let mut source_with_imports = import_re
        .replace_all(&clean_source, |caps: &regex::Captures| {
            let import_name = &caps[1];
            let import_line = &caps[0];

            let imported_content = load_shader_source_string(import_name, atomic_float_support);

            format!(
                "{}\n// BEGIN {}.wgsl\n{}\n// END {}.wgsl",
                import_line, import_name, imported_content, import_name
            )
        })
        .into_owned();

    #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
    {
        if shader_source != source_with_imports {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join("shaders")
                .join(format!("{}.wgsl", name_str));
            std::fs::write(path, &source_with_imports).ok();
        }
    }
    if atomic_float_support {
        // remove all lines containing '// no_atomic_float'
        source_with_imports = source_with_imports
            .lines()
            .filter(|line| !line.contains("// no_atomic_float"))
            .map(|line| line.replace("// atomic_float", ""))
            .collect::<Vec<_>>()
            .join("\n");
    }

    Box::leak(source_with_imports.into_boxed_str())
}

// The clean entry point
pub fn load_shader_source(name: ShaderName, has_float32_atomic: bool) -> &'static str {
    load_shader_source_string(name.to_str(), has_float32_atomic)
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
    pub name: ShaderName,
    pub shader_src: &'static str,
    pub bind_group_layout: BindGroupLayout,
    pub pipeline: ComputePipeline,
    pub binding_names: Vec<String>,
    pub binding_types: Vec<BindingType>,
}

impl ComputeShaderConfig {
    pub fn new(
        device: &Device,
        name: ShaderName,
        shader_src: &'static str,
        bindings: &[(String, BindingType)],
    ) -> Result<Self> {
        Self::new_with_constants(device, name, shader_src, bindings, &[])
    }

    pub fn new_with_constants(
        device: &Device,
        name: ShaderName,
        shader_src: &'static str,
        bindings: &[(String, BindingType)],
        constants: &[(&str, f64)],
    ) -> Result<Self> {
        let mut binding_names = Vec::new();
        let mut binding_types = Vec::new();
        let mut binding_group_layout_entries = Vec::new();
        for (i, (binding_name, binding_type)) in bindings.iter().enumerate() {
            binding_names.push(binding_name.clone());
            binding_types.push(*binding_type);

            binding_group_layout_entries.push(BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: *binding_type, // Clone BindingType
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", name)),
            entries: &binding_group_layout_entries,
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
            entry_point: Some(name.to_str()),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });

        Ok(Self {
            name,
            shader_src,
            bind_group_layout,
            pipeline,
            binding_names,
            binding_types,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &GpuResources,
    ) -> Result<BindGroup> {
        let binding_resources = self.create_resources(resources);
        let mut bg_entries = Vec::new();
        for (i, resource) in binding_resources.iter().enumerate() {
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

    pub fn create_resources<'a>(
        &'a self,
        gpu_resources: &'a GpuResources,
    ) -> Vec<BindingResource<'a>> {
        let mut resources: Vec<BindingResource<'a>> = Vec::new();
        for (binding_name, binding_type) in self.binding_names.iter().zip(self.binding_types.iter())
        {
            match binding_type {
                BindingType::Buffer { .. } => {
                    let buf_name: BufferName = binding_name
                        .parse()
                        .expect("Invalid buffer name in shader config");
                    let buf = gpu_resources.get_buffer(&buf_name).unwrap_or_else(|| {
                        panic!("Buffer not found in GpuResources: {}", buf_name)
                    });
                    resources.push(BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: None,
                    }));
                }
                BindingType::Texture { .. } | BindingType::StorageTexture { .. } => {
                    let texture_name: TextureName = binding_name
                        .parse()
                        .expect("Invalid texture name in shader config");
                    let view = gpu_resources
                        .get_texture_view(&texture_name)
                        .expect("Texture view not found in GpuResources");
                    resources.push(BindingResource::TextureView(view));
                }

                BindingType::Sampler(_) => {
                    // For simplicity, we use a single sampler for all shaders that need it.
                    // In a more complex implementation, you might want to allow different samplers.
                    let sampler = gpu_resources.get_sampler("sampler").expect("Sampler 'sampler' not found in GpuResources for shader that requires a sampler");
                    resources.push(BindingResource::Sampler(sampler));
                }
                _ => {
                    error!(
                        "Unsupported binding type for '{}': {:?}",
                        binding_name, binding_type
                    );
                }
            }
        }
        resources
    }
}

pub fn create_shader_configs(
    device: &Device,
    max_compute_invocations_per_workgroup: u32,
    has_float32_filterable: bool,
    has_float32_atomic: bool,
) -> Result<std::collections::HashMap<ShaderName, ComputeShaderConfig>> {
    let mut shader_configs = std::collections::HashMap::new();
    shader_configs.insert(
        ShaderName::AnalyzeTerrain,
        ComputeShaderConfig::new(
            device,
            ShaderName::AnalyzeTerrain,
            load_shader_source(ShaderName::AnalyzeTerrain, has_float32_atomic),
            &[
                // Binding 0: Uniform buffer (sim_settings_buffer)
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 2:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
                // Binding 3:
                (
                    TextureName::Curvature.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
                // Binding 4:
                (
                    BufferName::SlopeAngle.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::SlopeAspect.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::Debug.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::AnalyzeTerrainCurvilinear,
        ComputeShaderConfig::new(
            device,
            ShaderName::AnalyzeTerrainCurvilinear,
            load_shader_source(ShaderName::AnalyzeTerrainCurvilinear, has_float32_atomic),
            &[
                // Binding 0: Uniform buffer (sim_settings_buffer)
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 2:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
                // Binding 3:
                (
                    TextureName::Curvature.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
                // Binding 4:
                (
                    BufferName::SlopeAngle.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::SlopeAspect.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::Debug.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::ComputeRoughness,
        ComputeShaderConfig::new(
            device,
            ShaderName::ComputeRoughness,
            load_shader_source(ShaderName::ComputeRoughness, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    // Binding 0: Uniform buffer (sim_settings_buffer)
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 2:
                (
                    BufferName::Roughness.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::ComputeReleaseAreas,
        ComputeShaderConfig::new(
            device,
            ShaderName::ComputeReleaseAreas,
            load_shader_source(ShaderName::ComputeReleaseAreas, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 2:
                (
                    BufferName::SlopeAngle.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::Roughness.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::ReleaseAreas.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::Debug.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::InitializeParticles,
        ComputeShaderConfig::new(
            device,
            ShaderName::InitializeParticles,
            load_shader_source(ShaderName::InitializeParticles, has_float32_atomic),
            &[
                // Binding 0: Uniform buffer
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    BufferName::SlopeAngle.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::ReleaseAreas.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    "Sampler".to_string(),
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // Binding 6:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::ParticlesMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::ParticlesElevation.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZero::new(((size_of::<AtomicValues>() as u64 - 1) / 16 + 1) * 16)
                                .unwrap(),
                        ),
                    },
                ),
                // Binding 8:
                (
                    BufferName::GridCellCount.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::G2P,
        ComputeShaderConfig::new_with_constants(
            device,
            ShaderName::G2P,
            load_shader_source(ShaderName::G2P, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    "Sampler".to_string(),
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // Binding 4:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::ParticlesVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::ParticlesStopped.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::GridVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::TimestepData.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 10:
                (
                    BufferName::Debug.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 11:
                (
                    BufferName::ParticlesAffineMatrix.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::ComputeParticles,
        ComputeShaderConfig::new_with_constants(
            device,
            ShaderName::ComputeParticles,
            load_shader_source(ShaderName::ComputeParticles, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 4:
                (
                    "Sampler".to_string(),
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // Binding 5:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::ParticlesElevation.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::ParticlesVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::ParticlesVelocityZ.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::ParticlesMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 10:
                (
                    BufferName::ParticlesStopped.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 11:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 12:
                (
                    BufferName::GridPeakVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 13:
                (
                    BufferName::TimestepData.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 14:
                (
                    TextureName::Curvature.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 15:
                (
                    BufferName::Debug.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 16:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 17:
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::ResetGrid,
        ComputeShaderConfig::new(
            device,
            ShaderName::ResetGrid,
            load_shader_source(ShaderName::ResetGrid, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::UpdateSimInfo,
        ComputeShaderConfig::new(
            device,
            ShaderName::UpdateSimInfo,
            load_shader_source(ShaderName::UpdateSimInfo, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::NewCellsRollingWindow.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::P2G,
        ComputeShaderConfig::new_with_constants(
            device,
            ShaderName::P2G,
            load_shader_source(ShaderName::P2G, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::ParticlesVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::ParticlesMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::P2GMPM,
        ComputeShaderConfig::new_with_constants(
            device,
            ShaderName::P2GMPM,
            load_shader_source(ShaderName::P2GMPM, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::ParticlesVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::ParticlesMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::ParticlesAffineMatrix.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::GridPhysics,
        ComputeShaderConfig::new(
            device,
            ShaderName::GridPhysics,
            load_shader_source(ShaderName::GridPhysics, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    TextureName::Curvature.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 4:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::GridPeakFlowThickness.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::NewCellsRollingWindow.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::GridPhysicsMPM,
        ComputeShaderConfig::new(
            device,
            ShaderName::GridPhysicsMPM,
            load_shader_source(ShaderName::GridPhysicsMPM, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::TerrainGeometry.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    TextureName::Curvature.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 4:
                (
                    BufferName::GridPeakFlowThickness.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::NewCellsRollingWindow.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::GridVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 10:
                (
                    BufferName::GridPeakVelocity.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::TestTransforms,
        ComputeShaderConfig::new(
            device,
            ShaderName::TestTransforms,
            load_shader_source(ShaderName::TestTransforms, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::TestOutput.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::TestSampling,
        ComputeShaderConfig::new(
            device,
            ShaderName::TestSampling,
            load_shader_source(ShaderName::TestSampling, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 2:
                (
                    "Sampler".to_string(),
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // Binding 3:
                (
                    BufferName::TestOutput.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::TestUtils,
        ComputeShaderConfig::new(
            device,
            ShaderName::TestUtils,
            load_shader_source(ShaderName::TestUtils, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::TestOutput.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::AtomicValues.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::TestTransfer,
        ComputeShaderConfig::new(
            device,
            ShaderName::TestTransfer,
            load_shader_source(ShaderName::TestTransfer, has_float32_atomic),
            &[
                // Binding 0:
                (
                    BufferName::SimSettings.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 1:
                (
                    BufferName::ParticlesPosition.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    BufferName::ParticlesVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 3:
                (
                    BufferName::ParticlesMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::GridMass.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 5:
                (
                    BufferName::GridMomentum.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::GridVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::TestOutput.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::ParticlesAffineMatrix.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    Ok(shader_configs)
}

pub fn generate_shader_report(
    filename: Option<&str>,
    configs: &std::collections::HashMap<ShaderName, ComputeShaderConfig>,
    custom_order: Option<&[ShaderName]>,
) -> String {
    let mut html = String::from(
        r#"
    <style>
        body {
            background: #11111b;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
        }
        /* Flexbox wrapper to lay tables side-by-side */
        .report-container {
            display: flex;
            flex-wrap: nowrap;
            gap: 20px;
            justify-content: flex-start;
            align-items: flex-start;
            overflow-x: auto;
            white-space: nowrap;
        }
        .shader-table {
            font-family: 'Inter', system-ui, sans-serif;
            border-collapse: separate;
            border-spacing: 0;
            /* Flex properties: min base width 380px, max width 600px */
            flex: 1 1 380px;
            max-width: 600px;
            background: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #8b8b8b;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        }
        .shader-table th {
            background: #89b4fa;
            color: #11111b;
            padding: 12px;
            text-align: left;
            font-size: 1.1rem;
            white-space: nowrap;
        }
        .shader-table td {
            padding: 10px 15px;
            border-bottom: 1px solid #313244;
            white-space: nowrap;
        }
        tr:last-child td { border-bottom: none; }
        .binding-idx { color: #fab387; font-weight: bold; width: 30px; }
        .io-tag { font-size: 0.8rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
        .input { background: #a6e3a1; color: #11111b; }
        .output { background: #f38ba8; color: #11111b; }
        .type-info { color: #9399b2; font-style: italic; font-size: 0.9rem; }
        .res-name { font-family: 'JetBrains Mono', monospace; }
        
        /* Highlight state applied via JS */
        tr.highlighted { background: #0d6b66 !important; }
        tr:hover { background: #0d6b66; }
    </style>
    <div class="report-container">
    "#,
    );

    // Determine key iteration order
    let ordered_keys: Vec<&ShaderName> = match custom_order {
        Some(order) => order.iter().filter(|k| configs.contains_key(k)).collect(),
        None => {
            let mut keys: Vec<&ShaderName> = configs.keys().collect();
            keys.sort_by_key(|k| k.to_str());
            keys
        }
    };

    for name in ordered_keys {
        let config = &configs[name];

        html.push_str(&format!(
            r#"<table class="shader-table">
            <thead><tr><th colspan="3">Shader: {}</th></tr></thead>
            <tbody>"#,
            name.to_str()
        ));

        for (i, entry) in config.binding_types.iter().enumerate() {
            let (io_label, io_class, details) = match entry {
                wgpu::BindingType::Buffer { ty, .. } => {
                    let is_out =
                        matches!(ty, wgpu::BufferBindingType::Storage { read_only: false });
                    (
                        if is_out { "OUT" } else { "IN" },
                        if is_out { "output" } else { "input" },
                        format!("{:?}", ty),
                    )
                }
                wgpu::BindingType::Texture { sample_type, .. } => {
                    ("IN", "input", format!("Tex ({:?})", sample_type))
                }
                wgpu::BindingType::StorageTexture { format, .. } => {
                    ("OUT", "output", format!("StorageTex ({:?})", format))
                }
                wgpu::BindingType::Sampler(ty) => ("IN", "input", format!("Sampler ({:?})", ty)),
                &wgpu::BindingType::AccelerationStructure { .. }
                | &wgpu::BindingType::ExternalTexture => {
                    todo!()
                }
            };

            let resource_name = config
                .binding_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| "unnamed".to_string());

            html.push_str(&format!(
                r#"<tr data-resource="{}">
                    <td class="binding-idx">{}</td>
                    <td class="res-name">{} <span class="type-info">{}</span></td>
                    <td style="text-align: right;">
                        <span class="io-tag {}">{}</span>
                    </td>
                </tr>"#,
                resource_name,
                i,
                resource_name,
                details
                    .replace("{ read_only: true }", "(read-only)")
                    .replace("{ read_only: false }", "")
                    .replace("(Float { filterable: true })", "Float (filterable)"),
                io_class,
                io_label
            ));
        }
        html.push_str("</tbody></table>");
    }

    html.push_str("</div>"); // Close container div

    // Inject JS to synchronize hover states across all tables for the same resource name
    html.push_str(
        r#"
        <script>
            document.querySelectorAll('tr[data-resource]').forEach(row => {
                const resName = row.getAttribute('data-resource');
                row.addEventListener('mouseenter', () => {
                    document.querySelectorAll(`tr[data-resource="${CSS.escape(resName)}"]`).forEach(r => {
                        r.classList.add('highlighted');
                    });
                });
                row.addEventListener('mouseleave', () => {
                    document.querySelectorAll(`tr[data-resource="${CSS.escape(resName)}"]`).forEach(r => {
                        r.classList.remove('highlighted');
                    });
                });
            });
        </script>
        "#,
    );

    std::fs::write(filename.unwrap_or("shader_report.html"), &html)
        .expect("Unable to write shader report to file.");
    html
}
