use crate::buffers::AtomicValues;
use crate::buffers::{BufferName, GpuResources, TextureName};
use anyhow::Result;
use core::panic;
use std::num::NonZero;
use tracing::error;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureViewDimension,
};
pub const SHADER_UTILS: &str = include_str!("shaders/utils.wgsl");

/// Resolves to the entry point when one is declared, else to the source file
/// stem (the entry point defaults to the shader name).
macro_rules! shader_name_or_entry {
    ($_filename:literal, $entry:literal) => {
        $entry
    };
    ($_filename:literal,) => {
        $_filename
    };
}

macro_rules! define_shaders {
    ($($variant:ident => $filename:literal $(entry_point $entry:literal)?),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ShaderName {
            $($variant),*
        }

        impl std::str::FromStr for ShaderName {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                $(
                    if s == ShaderName::$variant.to_str() {
                        return Ok(ShaderName::$variant);
                    }
                )*
                Err(format!("'{}' is not a valid ShaderName", s))
            }
        }

        impl ShaderName {
            /// Every shader, including import fragments and test shaders.
            pub const ALL: &'static [ShaderName] = &[$(ShaderName::$variant),*];

            /// The shader's canonical name: the declared entry point if any,
            /// else the source file stem. Unique per variant; used for string
            /// parsing, labels, and as the dispatched WGSL entry point.
            pub fn to_str(&self) -> &'static str {
                match self {
                    $(ShaderName::$variant => {
                        shader_name_or_entry!($filename, $($entry)?)
                    })*
                }
            }

            /// The first shader whose source file is `file`.
            pub fn from_source_file(file: &str) -> Option<ShaderName> {
                Self::ALL.iter().copied().find(|shader| shader.source_file() == file)
            }

            /// The WGSL source file providing this shader's entry point.
            /// Several entry points can share one file (see `entry_point` in
            /// [`define_shaders!`]).
            pub fn source_file(&self) -> &'static str {
                match self {
                    $(ShaderName::$variant => $filename),*
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
                    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("src").join("shaders").join(format!("{}.wgsl", self.source_file()));
                    match std::fs::read_to_string(&path) {
                        Ok(source) => source,
                        Err(error) => {
                            tracing::warn!(
                                ?path,
                                ?error,
                                "Shader file was not available on disk; using embedded source"
                            );
                            match self {
                                $(ShaderName::$variant => include_str!(concat!("shaders/", $filename, ".wgsl")).to_string()),*
                            }
                        }
                    }
                }
            }
        }
    }
}

define_shaders! {
    AnalyzeTerrainCurvilinear => "analyze_terrain_curvilinear",
    AnalyzeTerrain => "analyze_terrain",
    // the reference surface passes share one source file
    SmoothReferenceRows => "smooth_reference" entry_point "smooth_reference_rows",
    SmoothReferenceColumns => "smooth_reference" entry_point "smooth_reference_columns",
    ResetGrid => "reset_grid",
    ComputeRoughness => "compute_roughness",
    ComputeReleaseAreas => "compute_release_areas",
    // the three center-of-mass kernels share one source file
    ComputeCenterOfMass => "center_of_mass" entry_point "compute_center_of_mass",
    CenterOfMassSeed => "center_of_mass" entry_point "center_of_mass_seed",
    CenterOfMassPropagate => "center_of_mass" entry_point "center_of_mass_propagate",
    CenterOfMassPartials => "center_of_mass" entry_point "center_of_mass_partials",
    // the three chamfer kernels share one source file
    ChamferPrepare => "chamfer" entry_point "chamfer_prepare",
    ChamferFlood => "chamfer" entry_point "chamfer_flood",
    ChamferReduce => "chamfer" entry_point "chamfer_reduce",
    ComputeBeelineDistance => "compute_beeline_distance",
    EvaluateMassMovement => "evaluate_mass_movement",
    InitializeParticles => "initialize_particles",
    // the four relaxation kernels share one source file; each entry point
    // dispatches one kernel from relax_particles.wgsl
    RelaxParticles => "relax_particles",
    RelaxClearGrid => "relax_particles" entry_point "relax_clear_grid",
    RelaxBuildGrid => "relax_particles" entry_point "relax_build_grid",
    RelaxSampleElevation => "relax_particles" entry_point "relax_sample_elevation",
    ComputeParticles => "compute_particles",
    G2P => "g2p",
    G2PMPMDAC => "g2p_mpmdac",
    P2G => "p2g",
    P2G2D => "p2g" entry_point "p2g_2d",
    P2G2DSorted => "p2g_sorted" entry_point "p2g_2d_sorted",
    P2GWeakPressure => "p2g_weak_pressure",
    // the four particle sort kernels share one source file
    SortClearBins => "sort_particles" entry_point "sort_clear_bins",
    SortCountParticles => "sort_particles" entry_point "sort_count_particles",
    SortScanBins => "sort_particles" entry_point "sort_scan_bins",
    SortScatterParticles => "sort_particles" entry_point "sort_scatter_particles",
    P2GMPMDAC => "p2g_mpmdac",
    GridPhysicsCurvilinear => "grid_physics_curvilinear",
    GridPhysicsMPMDAC => "grid_physics_mpmdac",
    GridPhysics => "grid_physics",
    GridPhysicsTerrainFollowing => "grid_physics_terrain_following",
    G2PTerrainFollowing => "g2p_terrain_following",
    // the 3D model shares its physics with the terrain-following sources
    P2G3D => "p2g" entry_point "p2g_3d",
    P2G3DSorted => "p2g_sorted" entry_point "p2g_3d_sorted",
    GridPhysics3D => "grid_physics_terrain_following" entry_point "grid_physics_3d",
    G2P3D => "g2p_terrain_following" entry_point "g2p_3d",
    Friction => "friction",
    EarthPressure => "earth_pressure",
    CurvilinearGeometry => "curvilinear_geometry",
    CartesianGeometry => "cartesian_geometry",
    Constitutive => "constitutive",
    Utils => "utils",
    Random => "random",
    UpdateSimInfo => "update_sim_info",
    TestTransfer => "test_transfer",
    TransferG2P => "transfer_g2p",
    TransferG2PAffine => "transfer_g2p_affine",
    TransferP2G => "transfer_p2g",
    TransferP2GZ => "transfer_p2g_z",
    TestTransforms => "test_transforms",
    TestSampling => "test_sampling",
    TestFriction => "test_friction",
    TestUtils => "test_utils",
}

impl std::fmt::Display for ShaderName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Expanded sources per (source file, float atomics), built once per
/// process: entry points that share a file share one `&'static str`, which
/// [`create_shader_configs`] uses to compile one module per source.
static SOURCES: OnceLock<Mutex<HashMap<(&'static str, bool), &'static str>>> = OnceLock::new();

/// `source` (LF) with the line endings of `original`, so that the debug
/// write-back of expanded imports only ever changes the inlined content.
#[cfg(any(test, all(not(target_arch = "wasm32"), debug_assertions)))]
fn restore_line_endings(source: &str, original: &str) -> String {
    if original.contains("\r\n") {
        source.replace('\n', "\r\n")
    } else {
        source.to_string()
    }
}

/// Removes every inlined import block (`// BEGIN x.wgsl` .. `// END x.wgsl`)
/// together with the whitespace before it, leaving the bare import lines.
fn strip_import_blocks(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(begin) = rest.find("// BEGIN ") {
        out.push_str(rest[..begin].trim_end());
        let block = &rest[begin..];
        let Some(end) = block.find("// END ") else {
            // unterminated block: keep it as it is
            out.push_str(block);
            return out;
        };
        let end_marker = &block[end..];
        let marker_len = end_marker
            .find(".wgsl")
            .map_or(end_marker.len(), |index| index + ".wgsl".len());
        rest = &end_marker[marker_len..];
    }
    out.push_str(rest);
    out
}

/// The import of `line` (`// import name.wgsl;`): the matched directive and
/// the imported file name.
fn parse_import(line: &str) -> Option<(&str, &str)> {
    let after_slashes = line.strip_prefix("//")?;
    let after_space = after_slashes.trim_start();
    if after_space.len() == after_slashes.len() {
        return None;
    }
    let after_keyword = after_space.strip_prefix("import")?;
    let name_start = after_keyword.trim_start();
    if name_start.len() == after_keyword.len() {
        return None;
    }
    let token_len = name_start
        .find(|c: char| !(c.is_ascii_alphanumeric() || "_./-".contains(c)))
        .unwrap_or(name_start.len());
    let name = name_start[..token_len].strip_suffix(".wgsl")?;
    if name.is_empty() {
        return None;
    }
    let mut matched_len = line.len() - name_start.len() + token_len;
    if line[matched_len..].starts_with(';') {
        matched_len += 1;
    }
    Some((&line[..matched_len], name))
}

/// Source of `file` with every `// import x.wgsl;` followed by the expanded
/// content of `x.wgsl` between `// BEGIN` / `// END` markers (both atomic
/// variants kept), LF line endings. Debug builds read the files from disk and
/// write a source back whose inlined imports changed, so the files always
/// carry the current imports.
fn expand_imports(file: &'static str) -> &'static str {
    cached_source(file, false, || {
        let shader = ShaderName::from_source_file(file)
            .unwrap_or_else(|| panic!("Unknown shader source: {file}"));
        let raw_source = shader.read_source();
        // CRLF working copies must not count as a change or leak into the imports
        let shader_source = raw_source.replace("\r\n", "\n");
        let clean_source = strip_import_blocks(&shader_source);
        let mut expanded = String::with_capacity(shader_source.len() + 1024);
        // jump between the few `import` occurrences (str::find stays fast in
        // unoptimized builds) instead of walking every line
        let mut copied = 0;
        let mut search = 0;
        while let Some(found) = clean_source[search..].find("import") {
            let at = search + found;
            let line_start = clean_source[..at].rfind('\n').map_or(0, |index| index + 1);
            let line_end = clean_source[at..]
                .find('\n')
                .map_or(clean_source.len(), |index| at + index);
            search = line_end;
            let line = &clean_source[line_start..line_end];
            let Some((directive, name)) = parse_import(line) else {
                continue;
            };
            let import_file = ShaderName::from_source_file(name)
                .unwrap_or_else(|| panic!("{file}.wgsl imports unknown {name}.wgsl"))
                .source_file();
            expanded.push_str(&clean_source[copied..line_start]);
            expanded.push_str(directive);
            expanded.push_str("\n// BEGIN ");
            expanded.push_str(name);
            expanded.push_str(".wgsl\n");
            expanded.push_str(expand_imports(import_file));
            expanded.push_str("\n// END ");
            expanded.push_str(name);
            expanded.push_str(".wgsl");
            copied = line_start + directive.len();
        }
        expanded.push_str(&clean_source[copied..]);

        #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
        if shader_source != expanded {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join("shaders")
                .join(format!("{file}.wgsl"));
            tracing::debug!("Writing the updated imports of {file}.wgsl");
            std::fs::write(path, restore_line_endings(&expanded, &raw_source)).ok();
        }
        expanded
    })
}

/// The cached source of (`file`, `float_atomics`), built by `build` on first use.
fn cached_source(
    file: &'static str,
    float_atomics: bool,
    build: impl FnOnce() -> String,
) -> &'static str {
    let sources = SOURCES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(source) = sources.lock().unwrap().get(&(file, float_atomics)) {
        return source;
    }
    // built outside the lock: expanding a file loads its imports recursively
    let source: &'static str = Box::leak(build().into_boxed_str());
    sources
        .lock()
        .unwrap()
        .entry((file, float_atomics))
        .or_insert(source)
}

/// Source of `name`'s file with its imports expanded and, for float-atomic
/// devices, the `// no_atomic_float` lines removed and the `// atomic_float`
/// lines enabled. Every entry point of a file returns the same string.
pub fn load_shader_source(name: ShaderName, has_float32_atomic: bool) -> &'static str {
    let file = name.source_file();
    let expanded = expand_imports(file);
    if !has_float32_atomic {
        return expanded;
    }
    cached_source(file, true, || {
        // only the marked lines change: jump between them with str::find
        const MARKER: &str = "atomic_float";
        let mut filtered = String::with_capacity(expanded.len());
        let mut copied = 0;
        let mut search = 0;
        while let Some(found) = expanded[search..].find(MARKER) {
            let at = search + found;
            let line_start = expanded[..at].rfind('\n').map_or(0, |index| index + 1);
            let line_end = expanded[at..]
                .find('\n')
                .map_or(expanded.len(), |index| at + index);
            search = line_end;
            let line = &expanded[line_start..line_end];
            if line.contains("// no_atomic_float") {
                // drop the line with its line break
                filtered.push_str(&expanded[copied..line_start]);
                copied = (line_end + 1).min(expanded.len());
                search = copied;
            } else if let Some(marker) = line.find("// atomic_float") {
                filtered.push_str(&expanded[copied..line_start + marker]);
                copied = line_start + marker + "// atomic_float".len();
            }
        }
        filtered.push_str(&expanded[copied..]);
        filtered
    })
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

/// A compute shader collected by [`create_shader_configs`] before its
/// pipeline is built: name, source and bindings.
pub struct PendingShader {
    pub name: ShaderName,
    pub shader_src: &'static str,
    pub binding_names: Vec<String>,
    pub binding_types: Vec<BindingType>,
    constants: Vec<(String, f64)>,
}

impl PendingShader {
    pub fn new(
        name: ShaderName,
        shader_src: &'static str,
        bindings: &[(String, BindingType)],
    ) -> Result<Self> {
        Self::new_with_constants(name, shader_src, bindings, &[])
    }

    pub fn new_with_constants(
        name: ShaderName,
        shader_src: &'static str,
        bindings: &[(String, BindingType)],
        constants: &[(&str, f64)],
    ) -> Result<Self> {
        Ok(Self {
            name,
            shader_src,
            binding_names: bindings.iter().map(|(name, _)| name.clone()).collect(),
            binding_types: bindings.iter().map(|(_, ty)| *ty).collect(),
            constants: constants
                .iter()
                .map(|(name, value)| (name.to_string(), *value))
                .collect(),
        })
    }

    /// Bind group layout and pipeline of this entry point in `module`.
    fn build(&self, device: &Device, module: &wgpu::ShaderModule) -> ComputeShaderConfig {
        let name = self.name;
        let binding_group_layout_entries: Vec<_> = self
            .binding_types
            .iter()
            .enumerate()
            .map(|(i, binding_type)| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: *binding_type,
                count: None,
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", name)),
            entries: &binding_group_layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", name)),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let constants: Vec<(&str, f64)> = self
            .constants
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&format!("{} Compute Pipeline", name)),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(name.to_str()),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        ComputeShaderConfig {
            name,
            shader_src: self.shader_src,
            bind_group_layout,
            pipeline,
            binding_names: self.binding_names.clone(),
            binding_types: self.binding_types.clone(),
        }
    }
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
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", name)),
            source: ShaderSource::Wgsl(shader_src.into()),
        });
        Ok(
            PendingShader::new_with_constants(name, shader_src, bindings, constants)?
                .build(device, &module),
        )
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
                    let error_msg = format!("Texture not found in GpuResources: {}", texture_name);
                    let view = gpu_resources
                        .get_texture_view(&texture_name)
                        .expect(&error_msg);
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

fn collect_shader_configs(
    max_compute_invocations_per_workgroup: u32,
    has_float32_filterable: bool,
    has_float32_atomic: bool,
) -> Result<HashMap<ShaderName, PendingShader>> {
    let mut shader_configs = HashMap::new();
    // reference surface: Gaussian-smoothed DEM, a row and a column pass
    let smooth_reference_bindings = [
        (
            BufferName::SimSettings.to_string(),
            BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
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
        (
            BufferName::ReferenceRows.to_string(),
            BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        (
            TextureName::DemReference.to_string(),
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::R32Float,
                view_dimension: TextureViewDimension::D2,
            },
        ),
    ];
    for shader in [
        ShaderName::SmoothReferenceRows,
        ShaderName::SmoothReferenceColumns,
    ] {
        shader_configs.insert(
            shader,
            PendingShader::new(
                shader,
                load_shader_source(shader, has_float32_atomic),
                &smooth_reference_bindings,
            )?,
        );
    }
    shader_configs.insert(
        ShaderName::AnalyzeTerrain,
        PendingShader::new(
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
                // Binding 7:
                (
                    TextureName::DemReference.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 8:
                (
                    TextureName::BasalTopography.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
            ],
        )?,
    );
    shader_configs.insert(
        ShaderName::AnalyzeTerrainCurvilinear,
        PendingShader::new(
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
                // Binding 7:
                (
                    TextureName::DemReference.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 8:
                (
                    TextureName::BasalTopography.to_string(),
                    BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::ComputeRoughness,
        PendingShader::new(
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
                // Binding 1: DEM; normals are derived here because the terrain
                // geometry texture layout depends on the simulation model
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

    // both evaluation kernels share one bind group layout and one module
    // (see evaluate_mass_movement.wgsl); evaluate_mass_movement only uses a
    // subset of the bindings
    let evaluation_bindings = vec![
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
            BufferName::RegionOfInterest.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 2:
        (
            BufferName::GridPeakFlowThickness.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 3:
        (
            BufferName::EvaluationResult.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 4: release cells are excluded from the counts
        (
            BufferName::ReleaseAreas.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
    ];
    shader_configs.insert(
        ShaderName::EvaluateMassMovement,
        PendingShader::new(
            ShaderName::EvaluateMassMovement,
            load_shader_source(ShaderName::EvaluateMassMovement, has_float32_atomic),
            &evaluation_bindings,
        )?,
    );
    // all three center-of-mass kernels share one bind group layout and one
    // module (see center_of_mass.wgsl); seed and propagate only use a
    // subset of the bindings
    let center_of_mass_bindings = vec![
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
            BufferName::CenterOfMass.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 3:
        (
            BufferName::SimInfo.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 4:
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
        // Binding 5:
        (
            "Sampler".to_string(),
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        // Binding 6:
        (
            BufferName::AtomicValues.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 7:
        (
            BufferName::CenterOfMassLabels.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 8:
        (
            BufferName::CenterOfMassBlobMass.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 9:
        (
            BufferName::CenterOfMassPartials.to_string(),
            BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
    ];
    shader_configs.insert(
        ShaderName::ComputeCenterOfMass,
        PendingShader::new(
            ShaderName::ComputeCenterOfMass,
            load_shader_source(ShaderName::ComputeCenterOfMass, has_float32_atomic),
            &center_of_mass_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::CenterOfMassPartials,
        PendingShader::new(
            ShaderName::CenterOfMassPartials,
            load_shader_source(ShaderName::CenterOfMassPartials, has_float32_atomic),
            &center_of_mass_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::CenterOfMassSeed,
        PendingShader::new(
            ShaderName::CenterOfMassSeed,
            load_shader_source(ShaderName::CenterOfMassSeed, has_float32_atomic),
            &center_of_mass_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::CenterOfMassPropagate,
        PendingShader::new(
            ShaderName::CenterOfMassPropagate,
            load_shader_source(ShaderName::CenterOfMassPropagate, has_float32_atomic),
            &center_of_mass_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::ComputeBeelineDistance,
        PendingShader::new(
            ShaderName::ComputeBeelineDistance,
            load_shader_source(ShaderName::ComputeBeelineDistance, has_float32_atomic),
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
                    BufferName::GridPeakFlowThickness.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 2:
                (
                    TextureName::Dem.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 3:
                (
                    BufferName::EvaluationResult.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    // all three chamfer kernels share one bind group layout and one module
    // (see chamfer.wgsl); each kernel only uses a subset of the bindings
    let chamfer_bindings = vec![
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
            BufferName::ChamferParams.to_string(),
            BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 2:
        (
            BufferName::GridPeakFlowThickness.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 3:
        (
            BufferName::RegionOfInterest.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 4:
        (
            BufferName::ChamferNearestRoi.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 5:
        (
            BufferName::ChamferNearestRoiSnapshot.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 6:
        (
            BufferName::ChamferNearestSim.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 7:
        (
            BufferName::ChamferNearestSimSnapshot.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 8:
        (
            BufferName::EvaluationResult.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 9: release cells are excluded from the reduced sums
        (
            BufferName::ReleaseAreas.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
    ];
    shader_configs.insert(
        ShaderName::ChamferPrepare,
        PendingShader::new(
            ShaderName::ChamferPrepare,
            load_shader_source(ShaderName::ChamferPrepare, has_float32_atomic),
            &chamfer_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::ChamferFlood,
        PendingShader::new(
            ShaderName::ChamferFlood,
            load_shader_source(ShaderName::ChamferFlood, has_float32_atomic),
            &chamfer_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::ChamferReduce,
        PendingShader::new(
            ShaderName::ChamferReduce,
            load_shader_source(ShaderName::ChamferReduce, has_float32_atomic),
            &chamfer_bindings,
        )?,
    );

    shader_configs.insert(
        ShaderName::ComputeReleaseAreas,
        PendingShader::new(
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
                // Binding 7:
                (
                    BufferName::RegionOfInterest.to_string(),
                    BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    shader_configs.insert(
        ShaderName::InitializeParticles,
        PendingShader::new(
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
    // all four relaxation kernels share one bind group layout; each kernel
    // only uses a subset of the bindings
    let relax_bindings = vec![
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
            BufferName::RelaxParams.to_string(),
            BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 2:
        (
            BufferName::ParticlesPosition.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 3:
        (
            BufferName::ParticlesPositionRelax.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 4:
        (
            BufferName::ParticlesVelocity.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 5:
        (
            BufferName::ReleaseAreas.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 6:
        (
            BufferName::RelaxGridHead.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 7:
        (
            BufferName::ParticleNext.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        // Binding 8:
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
        // Binding 9:
        (
            "Sampler".to_string(),
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        // Binding 10:
        (
            BufferName::ParticlesElevation.to_string(),
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
    ];
    shader_configs.insert(
        ShaderName::RelaxParticles,
        PendingShader::new_with_constants(
            ShaderName::RelaxParticles,
            load_shader_source(ShaderName::RelaxParticles, has_float32_atomic),
            &relax_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::RelaxClearGrid,
        PendingShader::new(
            ShaderName::RelaxClearGrid,
            load_shader_source(ShaderName::RelaxClearGrid, has_float32_atomic),
            &relax_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::RelaxBuildGrid,
        PendingShader::new_with_constants(
            ShaderName::RelaxBuildGrid,
            load_shader_source(ShaderName::RelaxBuildGrid, has_float32_atomic),
            &relax_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::RelaxSampleElevation,
        PendingShader::new_with_constants(
            ShaderName::RelaxSampleElevation,
            load_shader_source(ShaderName::RelaxSampleElevation, has_float32_atomic),
            &relax_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::G2P,
        PendingShader::new_with_constants(
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
                    BufferName::ParticlesState.to_string(),
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
                // Binding 12:
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
                // Binding 13:
                (
                    BufferName::ParticlesElevation.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                (
                    BufferName::ParticlesVelocityZ.to_string(),
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
        ShaderName::G2PMPMDAC,
        PendingShader::new_with_constants(
            ShaderName::G2PMPMDAC,
            load_shader_source(ShaderName::G2PMPMDAC, has_float32_atomic),
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
                    BufferName::ParticlesState.to_string(),
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
                // Binding 12:
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
                // Binding 13:
                (
                    BufferName::ParticlesElevation.to_string(),
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
        PendingShader::new_with_constants(
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
                    BufferName::ParticlesState.to_string(),
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
                // Binding 17:
                (
                    BufferName::GridVelocity.to_string(),
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
        PendingShader::new(
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
                // Binding 4:
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                (
                    BufferName::GridMomentumZ.to_string(),
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
        PendingShader::new(
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
        PendingShader::new_with_constants(
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
    let p2g = shader_configs
        .get(&ShaderName::P2G)
        .expect("P2G shader config was just inserted");
    let mut curvilinear_p2g_bindings: Vec<_> = p2g
        .binding_names
        .iter()
        .cloned()
        .zip(p2g.binding_types.iter().copied())
        .collect();
    curvilinear_p2g_bindings.push((
        BufferName::ParticlesAffineMatrix.to_string(),
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    ));
    shader_configs.insert(
        ShaderName::P2G2D,
        PendingShader::new_with_constants(
            ShaderName::P2G2D,
            load_shader_source(ShaderName::P2G2D, has_float32_atomic),
            &curvilinear_p2g_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    // the tile-sorted P2G sums quantized integer contributions in workgroup
    // memory and flushes them to the integer or float grid
    let storage_buffer = |read_only: bool| BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let mut sorted_p2g_bindings = curvilinear_p2g_bindings.clone();
    // Binding 8:
    sorted_p2g_bindings.push((
        BufferName::SortedParticleIndex.to_string(),
        storage_buffer(true),
    ));
    shader_configs.insert(
        ShaderName::P2G2DSorted,
        PendingShader::new(
            ShaderName::P2G2DSorted,
            load_shader_source(ShaderName::P2G2DSorted, has_float32_atomic),
            &sorted_p2g_bindings,
        )?,
    );
    // curvilinear weak-form pressure, a particle pass after either P2G
    shader_configs.insert(
        ShaderName::P2GWeakPressure,
        PendingShader::new_with_constants(
            ShaderName::P2GWeakPressure,
            load_shader_source(ShaderName::P2GWeakPressure, has_float32_atomic),
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
                (BufferName::SimInfo.to_string(), storage_buffer(false)),
                // Binding 2:
                (
                    BufferName::ParticlesPosition.to_string(),
                    storage_buffer(true),
                ),
                // Binding 3:
                (BufferName::ParticlesMass.to_string(), storage_buffer(true)),
                // Binding 4:
                (BufferName::GridMass.to_string(), storage_buffer(true)),
                // Binding 5:
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
                // Binding 6:
                (BufferName::GridForces.to_string(), storage_buffer(false)),
            ],
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    let sort_bindings = [
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
        (BufferName::SimInfo.to_string(), storage_buffer(false)),
        // Binding 2:
        (
            BufferName::ParticlesPosition.to_string(),
            storage_buffer(true),
        ),
        // Binding 3:
        (BufferName::SortBins.to_string(), storage_buffer(false)),
        // Binding 4:
        (
            BufferName::ParticleBinRank.to_string(),
            storage_buffer(false),
        ),
        // Binding 5:
        (
            BufferName::SortedParticleIndex.to_string(),
            storage_buffer(false),
        ),
    ];
    for shader in [
        ShaderName::SortClearBins,
        ShaderName::SortCountParticles,
        ShaderName::SortScatterParticles,
    ] {
        shader_configs.insert(
            shader,
            PendingShader::new_with_constants(
                shader,
                load_shader_source(shader, has_float32_atomic),
                &sort_bindings,
                &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
            )?,
        );
    }
    shader_configs.insert(
        ShaderName::SortScanBins,
        PendingShader::new(
            ShaderName::SortScanBins,
            load_shader_source(ShaderName::SortScanBins, has_float32_atomic),
            &sort_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::GridPhysics,
        PendingShader::new(
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
                // Binding 10:
                (
                    BufferName::GridVelocity.to_string(),
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
        ShaderName::GridPhysicsCurvilinear,
        PendingShader::new(
            ShaderName::GridPhysicsCurvilinear,
            load_shader_source(ShaderName::GridPhysicsCurvilinear, has_float32_atomic),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 11:
                (
                    TextureName::BasalTopography.to_string(),
                    BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: has_float32_filterable,
                        },
                    },
                ),
                // Binding 12: weak-form pressure sums of p2g_weak_pressure
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );
    // the terrain-following model reuses the curvilinear binding layouts; the 3D
    // model appends its z-component buffers, so its z-only bindings come last in
    // the sources the two models share
    let existing_bindings = |shader: ShaderName| -> Vec<(String, BindingType)> {
        let config = shader_configs
            .get(&shader)
            .expect("base shader config is inserted before the terrain-following configs");
        config
            .binding_names
            .iter()
            .cloned()
            .zip(config.binding_types.iter().copied())
            .collect()
    };
    let storage_buffer = |read_only: bool| BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let mut three_d_p2g_bindings = existing_bindings(ShaderName::P2G2D);
    three_d_p2g_bindings.extend([
        // Binding 8:
        (
            BufferName::ParticlesVelocityZ.to_string(),
            storage_buffer(true),
        ),
        // Binding 9:
        (BufferName::GridMomentumZ.to_string(), storage_buffer(false)),
        // Binding 10:
        (
            BufferName::ParticlesAffineZ.to_string(),
            storage_buffer(true),
        ),
    ]);
    let mut three_d_sorted_p2g_bindings = existing_bindings(ShaderName::P2G2DSorted);
    three_d_sorted_p2g_bindings.extend([
        // Binding 9:
        (
            BufferName::ParticlesVelocityZ.to_string(),
            storage_buffer(true),
        ),
        // Binding 10:
        (BufferName::GridMomentumZ.to_string(), storage_buffer(false)),
        // Binding 11:
        (
            BufferName::ParticlesAffineZ.to_string(),
            storage_buffer(true),
        ),
    ]);
    let mut terrain_following_grid_physics_bindings =
        existing_bindings(ShaderName::GridPhysicsCurvilinear);
    // binding 12 (weak-form pressure) is curvilinear only
    terrain_following_grid_physics_bindings
        .retain(|(name, _)| *name != BufferName::GridForces.to_string());
    let mut three_d_grid_physics_bindings = terrain_following_grid_physics_bindings.clone();
    three_d_grid_physics_bindings.extend([
        // Binding 12:
        (BufferName::GridMomentumZ.to_string(), storage_buffer(true)),
        // Binding 13:
        (BufferName::GridVelocityZ.to_string(), storage_buffer(false)),
    ]);
    let mut terrain_following_g2p_bindings = existing_bindings(ShaderName::G2P);
    // the curvilinear binding 14 is the z velocity, which this model does not have
    terrain_following_g2p_bindings
        .retain(|(name, _)| *name != BufferName::ParticlesVelocityZ.to_string());
    // Binding 14:
    terrain_following_g2p_bindings.push((
        TextureName::Curvature.to_string(),
        BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float {
                filterable: has_float32_filterable,
            },
        },
    ));
    let mut three_d_g2p_bindings = terrain_following_g2p_bindings.clone();
    three_d_g2p_bindings.extend([
        // Binding 15:
        (
            BufferName::ParticlesVelocityZ.to_string(),
            storage_buffer(false),
        ),
        // Binding 16:
        (BufferName::GridVelocityZ.to_string(), storage_buffer(true)),
        // Binding 17:
        (
            BufferName::ParticlesAffineZ.to_string(),
            storage_buffer(false),
        ),
    ]);
    shader_configs.insert(
        ShaderName::GridPhysicsTerrainFollowing,
        PendingShader::new(
            ShaderName::GridPhysicsTerrainFollowing,
            load_shader_source(ShaderName::GridPhysicsTerrainFollowing, has_float32_atomic),
            &terrain_following_grid_physics_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::G2PTerrainFollowing,
        PendingShader::new_with_constants(
            ShaderName::G2PTerrainFollowing,
            load_shader_source(ShaderName::G2PTerrainFollowing, has_float32_atomic),
            &terrain_following_g2p_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::P2G3D,
        PendingShader::new_with_constants(
            ShaderName::P2G3D,
            load_shader_source(ShaderName::P2G3D, has_float32_atomic),
            &three_d_p2g_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::P2G3DSorted,
        PendingShader::new(
            ShaderName::P2G3DSorted,
            load_shader_source(ShaderName::P2G3DSorted, has_float32_atomic),
            &three_d_sorted_p2g_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::GridPhysics3D,
        PendingShader::new(
            ShaderName::GridPhysics3D,
            load_shader_source(ShaderName::GridPhysics3D, has_float32_atomic),
            &three_d_grid_physics_bindings,
        )?,
    );
    shader_configs.insert(
        ShaderName::G2P3D,
        PendingShader::new_with_constants(
            ShaderName::G2P3D,
            load_shader_source(ShaderName::G2P3D, has_float32_atomic),
            &three_d_g2p_bindings,
            &[("WG_SIZE_1D", max_compute_invocations_per_workgroup as f64)],
        )?,
    );
    shader_configs.insert(
        ShaderName::P2GMPMDAC,
        PendingShader::new_with_constants(
            ShaderName::P2GMPMDAC,
            load_shader_source(ShaderName::P2GMPMDAC, has_float32_atomic),
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
                // Binding 8:
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
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
                // Binding 10:
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
                // Binding 11:
                (
                    BufferName::GridMassPrevious.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 12:
                (
                    BufferName::ParticlesStress.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 13:
                (
                    BufferName::ParticlesVolumetricStrain.to_string(),
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
        ShaderName::GridPhysicsMPMDAC,
        PendingShader::new(
            ShaderName::GridPhysicsMPMDAC,
            load_shader_source(ShaderName::GridPhysicsMPMDAC, has_float32_atomic),
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
                    BufferName::GridPeakFlowThickness.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 4:
                (
                    BufferName::AtomicValues.to_string(),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 6:
                (
                    BufferName::NewCellsRollingWindow.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 7:
                (
                    BufferName::SimInfo.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 8:
                (
                    BufferName::GridVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 9:
                (
                    BufferName::GridPeakVelocity.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // Binding 10:
                (
                    BufferName::GridForces.to_string(),
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        )?,
    );

    #[cfg(test)]
    {
        shader_configs.extend(create_test_shader_configs(
            has_float32_filterable,
            has_float32_atomic,
        )?);
    }

    Ok(shader_configs)
}

/// Collects every compute shader of the orchestrator. The WGSL sources are
/// expanded up front (in parallel; debug builds write changed imports back),
/// the pipelines are built on first use, see [`ShaderConfigs`].
pub fn create_shader_configs(
    device: &Device,
    max_compute_invocations_per_workgroup: u32,
    has_float32_filterable: bool,
    has_float32_atomic: bool,
) -> Result<ShaderConfigs> {
    // expand every source file once, in parallel
    let mut files: Vec<ShaderName> = Vec::new();
    for shader in ShaderName::ALL {
        if !files
            .iter()
            .any(|file| file.source_file() == shader.source_file())
        {
            files.push(*shader);
        }
    }
    build_in_parallel(files, |shader| {
        load_shader_source(shader, has_float32_atomic)
    });
    let pending = collect_shader_configs(
        max_compute_invocations_per_workgroup,
        has_float32_filterable,
        has_float32_atomic,
    )?;
    Ok(ShaderConfigs {
        device: device.clone(),
        shaders: pending
            .into_iter()
            .map(|(name, shader)| (name, (shader, OnceLock::new())))
            .collect(),
        modules: Mutex::new(HashMap::new()),
    })
}

/// The compute shaders of an orchestrator, each pipeline built on first use:
/// a run compiles only the shaders it dispatches, and the entry points of one
/// source file share one shader module (parsed and validated by naga once).
/// Pipelines are built on the calling thread (see [`Self::prepare`]); only
/// the WGSL sources are expanded in parallel.
pub struct ShaderConfigs {
    device: Device,
    shaders: HashMap<ShaderName, (PendingShader, OnceLock<ComputeShaderConfig>)>,
    /// Shader modules by the address of their (cached, shared) source.
    modules: Mutex<HashMap<usize, wgpu::ShaderModule>>,
}

impl ShaderConfigs {
    /// The pipeline of `name`, built now if it is not yet.
    pub fn get(&self, name: &ShaderName) -> Option<&ComputeShaderConfig> {
        let (shader, config) = self.shaders.get(name)?;
        Some(config.get_or_init(|| shader.build(&self.device, &self.module(shader))))
    }

    /// Builds the pipelines of `names` that do not exist yet, one after
    /// another: creating pipelines of one device on several threads crashed
    /// the process (access violation) in about 7 % of the test runs on an
    /// NVIDIA Vulkan driver, none in 36 runs when built sequentially.
    pub fn prepare(&self, names: &[ShaderName]) {
        for name in names {
            self.get(name);
        }
    }

    /// Whether a shader named `name` exists (built or not).
    pub fn contains(&self, name: &ShaderName) -> bool {
        self.shaders.contains_key(name)
    }

    /// Every shader name.
    pub fn names(&self) -> impl Iterator<Item = &ShaderName> {
        self.shaders.keys()
    }

    /// Name, source and bindings of `name`, without building its pipeline.
    pub fn pending(&self, name: &ShaderName) -> Option<&PendingShader> {
        self.shaders.get(name).map(|(shader, _)| shader)
    }

    /// The shader module of `shader`'s source, shared by its file's entry points.
    fn module(&self, shader: &PendingShader) -> wgpu::ShaderModule {
        let key = shader.shader_src.as_ptr() as usize;
        if let Some(module) = self.modules.lock().unwrap().get(&key) {
            return module.clone();
        }
        // parsed outside the lock, so other sources build meanwhile
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", shader.name.source_file())),
            source: ShaderSource::Wgsl(shader.shader_src.into()),
        });
        self.modules
            .lock()
            .unwrap()
            .entry(key)
            .or_insert(module)
            .clone()
    }
}

/// Maps `items` with `build` on all cores (popping from the end, results in
/// completion order).
#[cfg(not(target_arch = "wasm32"))]
fn build_in_parallel<T: Send, R: Send>(items: Vec<T>, build: impl Fn(T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len());
    if threads <= 1 {
        return items.into_iter().map(build).collect();
    }
    let queue = Mutex::new(items);
    let results = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let Some(item) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let result = build(item);
                    results.lock().unwrap().push(result);
                }
            });
        }
    });
    results.into_inner().unwrap()
}

/// Maps `items` with `build` in order: wasm has no threads here, and its
/// wgpu handles are not `Sync`.
#[cfg(target_arch = "wasm32")]
fn build_in_parallel<T, R>(items: Vec<T>, build: impl Fn(T) -> R) -> Vec<R> {
    items.into_iter().map(build).collect()
}

#[cfg(test)]
fn create_test_shader_configs(
    has_float32_filterable: bool,
    has_float32_atomic: bool,
) -> Result<HashMap<ShaderName, PendingShader>> {
    let mut shader_configs = HashMap::new();

    shader_configs.insert(
        ShaderName::TestTransforms,
        PendingShader::new(
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
        PendingShader::new(
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
        PendingShader::new(
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
        PendingShader::new(
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
    shader_configs.insert(
        ShaderName::TestFriction,
        PendingShader::new(
            ShaderName::TestFriction,
            load_shader_source(ShaderName::TestFriction, has_float32_atomic),
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
    Ok(shader_configs)
}

pub fn generate_shader_report(
    filename: Option<&str>,
    configs: &ShaderConfigs,
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
        Some(order) => order.iter().filter(|k| configs.contains(k)).collect(),
        None => {
            let mut keys: Vec<&ShaderName> = configs.names().collect();
            keys.sort_by_key(|k| k.to_str());
            keys
        }
    };

    for name in ordered_keys {
        let config = configs
            .pending(name)
            .expect("the report lists existing shaders");

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

#[cfg(test)]
mod tests {
    use super::*;

    // The float-atomic variants cannot run on GPUs without float atomics, so
    // at least validate the WGSL that would be compiled there.
    fn validate_wgsl(shader: ShaderName, has_float32_atomic: bool) -> &'static str {
        let source = load_shader_source(shader, has_float32_atomic);
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
            std::panic!(
                "{shader} (float atomics {has_float32_atomic}): {}",
                error.emit_to_string(source)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| {
            std::panic!("{shader} (float atomics {has_float32_atomic}): {error:?}")
        });
        source
    }

    /// Bytes of workgroup memory each entry point of a shader statically uses.
    fn workgroup_memory(shader: ShaderName) -> Vec<(String, u32)> {
        let source = load_shader_source(shader, false);
        let module = naga::front::wgsl::parse_str(source).expect("shader must parse");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("shader must validate");
        module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, entry_point)| {
                let bytes = module
                    .global_variables
                    .iter()
                    .filter(|(handle, variable)| {
                        variable.space == naga::AddressSpace::WorkGroup
                            && !info.get_entry_point(index)[*handle].is_empty()
                    })
                    .map(|(_, variable)| module.types[variable.ty].inner.size(module.to_ctx()))
                    .sum();
                (entry_point.name.clone(), bytes)
            })
            .collect()
    }

    // browsers only guarantee this much, and only pipeline creation checks it
    #[test]
    fn workgroup_memory_stays_within_the_webgpu_default() {
        const WEBGPU_DEFAULT: u32 = 16384;
        for &shader in ShaderName::ALL {
            let source = load_shader_source(shader, false);
            if !source.contains("@compute") || shader.source_file().starts_with("test_") {
                continue;
            }
            for (entry_point, bytes) in workgroup_memory(shader) {
                assert!(
                    bytes <= WEBGPU_DEFAULT,
                    "{entry_point} uses {bytes} bytes of workgroup memory"
                );
            }
        }
        // four 31 x 31 tiles of 4 bytes plus the box variables
        let sorted = workgroup_memory(ShaderName::P2G3DSorted);
        let bytes = sorted.iter().find(|(name, _)| name == "p2g_3d_sorted");
        assert_eq!(bytes.map(|(_, bytes)| *bytes), Some(4 * 31 * 31 * 4 + 32));
    }

    #[test]
    fn parse_import_reads_the_directive_and_the_file() {
        assert_eq!(
            parse_import("// import utils.wgsl;\n"),
            Some(("// import utils.wgsl;", "utils"))
        );
        assert_eq!(
            parse_import("//  import transfer_p2g.wgsl"),
            Some(("//  import transfer_p2g.wgsl", "transfer_p2g"))
        );
        assert_eq!(parse_import("// imports utils.wgsl;"), None);
        assert_eq!(parse_import("//import utils.wgsl;"), None);
        assert_eq!(parse_import("let x = 1; // import utils.wgsl;"), None);
        assert_eq!(parse_import("// import utils;"), None);
    }

    #[test]
    fn strip_import_blocks_leaves_the_import_lines() {
        let source = "a\n// import x.wgsl;\n// BEGIN x.wgsl\nfn x() {}\n// END x.wgsl\nb\n";
        assert_eq!(strip_import_blocks(source), "a\n// import x.wgsl;\nb\n");
    }

    /// The files on disk carry their expanded imports, so expanding them again
    /// must reproduce them (the debug write-back keeps them current).
    #[test]
    fn shader_files_are_expanded_and_share_one_source_per_file() {
        for shader in ShaderName::ALL {
            let source = load_shader_source(*shader, false);
            let on_disk = shader.read_source().replace("\r\n", "\n");
            assert_eq!(source, on_disk, "{shader}: stale inlined imports");
            let first = ShaderName::from_source_file(shader.source_file()).unwrap();
            assert!(std::ptr::eq(source, load_shader_source(first, false)));
            assert!(std::ptr::eq(
                load_shader_source(*shader, true),
                load_shader_source(first, true)
            ));
            assert!(!load_shader_source(*shader, true).contains("// no_atomic_float"));
        }
    }

    #[test]
    fn restore_line_endings_keeps_the_endings_of_the_original() {
        assert_eq!(restore_line_endings("a\nb\n", "x\ny\n"), "a\nb\n");
        assert_eq!(restore_line_endings("a\nb\n", "x\r\ny\r\n"), "a\r\nb\r\n");
    }

    #[test]
    fn center_of_mass_shaders_validate_with_integer_and_float_atomics() {
        for shader in [
            ShaderName::ComputeCenterOfMass,
            ShaderName::CenterOfMassSeed,
            ShaderName::CenterOfMassPropagate,
        ] {
            for has_float32_atomic in [false, true] {
                validate_wgsl(shader, has_float32_atomic);
            }
        }
    }

    #[test]
    fn p2g_shaders_validate_with_integer_and_float_atomics() {
        for shader in [
            ShaderName::P2G2D,
            ShaderName::P2G2DSorted,
            ShaderName::P2G3D,
            ShaderName::P2G3DSorted,
            ShaderName::P2GWeakPressure,
        ] {
            for has_float32_atomic in [false, true] {
                let source = validate_wgsl(shader, has_float32_atomic);
                let float_binding = source.lines().any(|line| {
                    line.trim_start().starts_with("@group") && line.contains("atomic<f32>")
                });
                assert_eq!(float_binding, has_float32_atomic);
            }
        }
    }

    #[test]
    fn grid_physics_shaders_validate_with_integer_and_float_atomics() {
        for shader in [
            ShaderName::AnalyzeTerrainCurvilinear,
            ShaderName::GridPhysicsCurvilinear,
            ShaderName::GridPhysicsTerrainFollowing,
            ShaderName::GridPhysics3D,
            ShaderName::GridPhysics,
        ] {
            for has_float32_atomic in [false, true] {
                validate_wgsl(shader, has_float32_atomic);
            }
        }
    }
}
