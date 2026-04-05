use crate::buffers::{
    BufferName, ComputeBuffers, TextureName, create_buffers_and_texture_descriptions,
};
use crate::shaders::ComputeShaderConfig;
use anyhow::{Ok, Result, anyhow};
use std::collections::HashMap;
use wgpu::{
    Adapter, BindingResource, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, Sampler, TextureFormat, TextureUsages,
};
// use log::{debug, info, warn, error};
pub mod buffers;
pub mod dem;
pub mod settings;
pub mod shaders;
pub mod utils;
use data_processor::*;
use tracing::{debug, info, trace};

use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

static INIT: Once = Once::new();

/// Initializes the global tracing subscriber.
/// Safe to call multiple times; subsequent calls do nothing.
pub fn init_logging() {
    let filter = EnvFilter::new("warn,compute_core=debug,data_processor=debug,cli=debug");
    INIT.call_once(|| {
        tracing_subscriber::registry()
            .with(fmt::layer().with_target(false)) // Clean console output
            .with(filter) // Default level
            .init();

        info!("Simulation logging initialized");
    });
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub mass: f32,
    pub velocity: [f32; 3],
    pub _pad0: f32,  // padding to align next field
    pub c: [f32; 4], // 2x2 matrix: [xx, xy, yx, yy]
    pub stopped: u32,
    pub _pad1: [u32; 3], // 3 * 4 bytes padding
}

impl Default for Particle {
    fn default() -> Self {
        Self::new()
    }
}

impl Particle {
    pub const BYTE_SIZE: usize = 16 * 4;

    pub fn new() -> Self {
        Self {
            position: [0.0; 3],
            mass: 0.0,
            velocity: [0.0; 3],
            _pad0: 0.0,
            c: [0.0; 4],
            stopped: 0,
            _pad1: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimInfo {
    pub timestep: u32,
    pub number_particles: u32,
    pub elevation_threshold: f32,
    pub max_velocity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimestepData {
    pub velocity: [f32; 3],                   // 12 bytes
    pub dt: f32,                              // 4 bytes
    pub acceleration_tangential: [f32; 3],    // 12 bytes
    pub acceleration_friction_magnitude: f32, // 4 bytes
    pub position: [f32; 3],                   // 12 bytes
    pub elevation: f32,                       // 4 bytes
    pub normal: [f32; 3],                     // 12 bytes
    pub acceleration_normal: [f32; 3],        // 12 bytes
    pub _pad0: f32,                           // 4 bytes (padding)
    pub uv: [f32; 2],                         // 8 bytes
    pub g_eff: f32,                           // 4 bytes
    pub _pad1: f32,                           // 4 bytes (padding to 96 bytes)
}
#[expect(dead_code)]
pub struct ComputeOrchestrator {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub workgroup_size_1d: u32,
    pub workgroup_size_2d: u32,
    pub buffers: ComputeBuffers,
    pub sampler: Sampler,
    texture_size: Extent3d,
    shader_configs: HashMap<String, ComputeShaderConfig>,
    dispatch_workgroup_size_1d: u32,
    dispatch_workgroup_size_2d: u32,
}

impl ComputeOrchestrator {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        trace!("Adapter: {:?}", adapter);

        trace!("Adapter limits: {:?}", adapter.limits());

        debug!("Adapter: {:?}", adapter);
        let workgroup_size_2d = utils::highest_power_of_two(
            (adapter.limits().max_compute_workgroup_size_x as f64).sqrt() as u32,
        );
        debug!("Workgroup size 2D: {}", workgroup_size_2d);
        let max_invocations = adapter.limits().max_compute_invocations_per_workgroup;
        let workgroup_size_1d = max_invocations;
        let max_storage = adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Compute Device"),
                required_features: Features::FLOAT32_FILTERABLE | Features::TIMESTAMP_QUERY,
                required_limits: Limits {
                    max_compute_workgroup_size_x: workgroup_size_1d,
                    max_compute_workgroup_size_y: workgroup_size_2d,
                    max_compute_workgroup_size_z: 1,
                    max_compute_invocations_per_workgroup: max_invocations,
                    max_storage_buffer_binding_size: max_storage,
                    ..Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device and queue");

        let buffers = ComputeBuffers::new();
        let shader_configs =
            shaders::create_shader_configs(&device, workgroup_size_1d, workgroup_size_2d)?;
        let texture_size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            workgroup_size_1d,
            workgroup_size_2d,
            buffers,
            shader_configs,
            texture_size,
            sampler,
            dispatch_workgroup_size_2d: workgroup_size_2d,
            dispatch_workgroup_size_1d: workgroup_size_1d,
        })
    }
    fn get_sampler(&self) -> BindingResource<'_> {
        BindingResource::Sampler(&self.sampler)
    }
    fn get_view(&self, name: TextureName) -> BindingResource<'_> {
        let view = self
            .buffers
            .get_texture_view(&name)
            .ok_or_else(|| anyhow!("Texture view '{}' not found", name))
            .expect("Texture view not found");
        BindingResource::TextureView(view)
    }

    fn get_buffer_binding(&self, name: BufferName) -> BindingResource<'_> {
        self.buffers
            .get_buffer(&name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", name))
            .expect("Buffer not found")
            .as_entire_binding()
    }

    // `run_shader` now takes `resources` directly for flexibility
    pub async fn run_shader(
        &self,
        shader_name: &str,
        resources: &[BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
        dispatch_workgroup_x: u32,
        dispatch_workgroup_y: u32,
        dispatch_workgroup_z: u32,
    ) -> Result<()> {
        assert_ne!(dispatch_workgroup_x, 0);
        assert_ne!(dispatch_workgroup_y, 0);
        assert_ne!(dispatch_workgroup_z, 0);
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
        resources: &[BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
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

    pub fn create_buffers_and_texture_descriptions(
        &mut self,
        sim_settings: &settings::SimSettings,
    ) -> Result<()> {
        self.texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };
        self.buffers = create_buffers_and_texture_descriptions(&self.device, self.texture_size);
        Ok(())
    }

    pub async fn run_normals(
        &mut self, // `&mut self` because we're adding textures
        sim_settings: &settings::SimSettings,
        dem: &dem::Dem,
    ) -> Result<()> {
        self.texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };

        self.dispatch_workgroup_size_2d =
            sim_settings.grid_shape_x.div_ceil(self.workgroup_size_2d);

        self.buffers = create_buffers_and_texture_descriptions(&self.device, self.texture_size);

        self.buffers.add_buffer_with_data(
            &self.device,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            true,
        );

        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.buffers
            .add_texture_with_data(
                &self.device,
                &self.queue,
                dem.data1d.as_slice(),
                TextureName::Dem,
                self.texture_size,
                TextureFormat::R32Float,
                texture_usage_input,
            )
            .expect("Failed to add texture with data");
        debug!("Running texture processing shader...");
        let _ = self.buffers.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        );

        self.run_shader(
            "compute_normals",
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::Wind),
                self.get_view(TextureName::Normals),
                self.get_view(TextureName::Slope),
                self.get_buffer_binding(BufferName::OutDebugNormals),
            ],
            self.dispatch_workgroup_size_2d,
            self.dispatch_workgroup_size_2d,
            1,
        )
        .await?;
        Ok(())
    }
    pub async fn run_load_release_areas(
        &mut self, // `&mut self` because we're adding textures
        data: &[u8],
    ) -> Result<u32> {
        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.buffers
            .add_texture_with_data(
                &self.device,
                &self.queue,
                data,
                TextureName::ReleaseAreasInput,
                self.texture_size,
                TextureFormat::Rgba8Uint,
                texture_usage_input,
            )
            .expect("Failed to add texture with data");
        debug!("Running texture processing shader...");
        self.run_shader(
            "load_release_areas",
            &[
                self.get_view(TextureName::ReleaseAreasInput),
                self.get_view(TextureName::ReleaseAreas),
                self.get_buffer_binding(BufferName::OutDebugRelease),
                self.get_buffer_binding(BufferName::NumberReleaseCells),
            ],
            self.dispatch_workgroup_size_2d,
            self.dispatch_workgroup_size_2d,
            1,
        )
        .await?;
    
        let number_release_cells: u32 = self.read_buffer::<u32>(BufferName::NumberReleaseCells).await.expect("Failed to read number_release_cells buffer")[0];
        Ok(number_release_cells)
    }
    pub async fn run_initialize_particles(&mut self) -> Result<()> {
        debug!("Running texture processing shader...");
        self.buffers.add_buffer_with_data(
            &self.device,
            BufferName::ParticleIndex,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            false,
        );
        self.run_shader(
            "initialize_particles",
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_buffer_binding(BufferName::SimInfo),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::ReleaseAreas),
                self.get_sampler(),
                self.get_buffer_binding(BufferName::Particles),
                self.get_buffer_binding(BufferName::ParticleIndex),
                self.get_buffer_binding(BufferName::CellCountGrid),
                self.get_buffer_binding(BufferName::MaxVelocityGrid),
            ],
            self.workgroup_size_2d,
            self.workgroup_size_2d,
            1,
        )
        .await?;
        Ok(())
    }
    pub async fn run_compute_particles(&mut self) -> Result<()> {
        debug!("Running texture processing shader...");
        self.buffers.add_buffer_with_data(
            &self.device,
            BufferName::ParticleIndex,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            false,
        );
        self.run_shader(
            "initialize_particles",
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_buffer_binding(BufferName::SimInfo),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::Normals),
                self.get_buffer_binding(BufferName::Particles),
                self.get_sampler(),
                self.get_buffer_binding(BufferName::ParticleIndex),
                self.get_buffer_binding(BufferName::CellCountGrid),
                self.get_buffer_binding(BufferName::MaxVelocityGrid),
            ],
            self.workgroup_size_2d,
            self.workgroup_size_2d,
            1,
        )
        .await?;
        Ok(())
    }
    #[allow(dead_code)]
    async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        self.buffers
            .read_texture(&self.device, &self.queue, name)
            .await
    }
    async fn read_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: BufferName,
    ) -> Result<Vec<T>> {
        self.buffers
            .read_buffer(&self.device, &self.queue, name)
            .await
    }
    #[expect(dead_code)]
    async fn add_buffer<T: bytemuck::Pod + Send + Sync>(&self, name: BufferName) -> Result<Vec<T>> {
        self.buffers
            .read_buffer(&self.device, &self.queue, name)
            .await
    }

    pub fn save_grid(&self, path: &str, data: Vec<f32>) -> Result<()> {
        let params = MetaGridParams {
            width: self.texture_size.width,
            height: self.texture_size.height,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        F32Data::new(&MetaGrid::new(&params), data)
            .save(path.as_ref())
            .unwrap_or_else(|_| panic!("Failed to save grid {}", path));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::settings::Settings;
    use data_processor::read_png;
    use pollster;
    use utils::{HistFloat, MaxValue, MinValue};
    const INCLINED_PLANE_PATH: &str = "../../data/avaframe/avaInclinedPlane.png";
    const RELEASE_TEXTURE_PATH: &str = "../../data/avaframe/avaInclinedPlanereleaseTexture.png";
    #[test_log::test]
    fn test_compute_orchestrator_creation() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        let (sim_settings, dem) = Settings::create_from_path(INCLINED_PLANE_PATH);
        pollster::block_on(orchestrator.run_normals(&sim_settings, &dem))
            .expect("Failed to run normals shader");

        let (slope_angle, slope_aspect, _, _) =
            pollster::block_on(orchestrator.read_texture::<f32>(TextureName::Slope)) // get_texture::<f32>("slope"))
                .expect("Failed to get slope texture");
        let (normal_x, normal_y, normal_z, profile_curvature) =
            pollster::block_on(orchestrator.read_texture::<f32>(TextureName::Normals))
                .expect("Failed to get normals texture");
        orchestrator
            .save_grid("slope_aspect.bin", slope_aspect.clone())
            .expect("Failed to save slope_aspect");
        orchestrator
            .save_grid("slope_angle.bin", slope_angle.clone())
            .expect("Failed to save slope_angle");
        orchestrator
            .save_grid("profile_curvature.bin", profile_curvature.clone())
            .expect("Failed to save profile_curvature");
        // println!("{}", slope_texture[5].to_f32());
        let debug_buffer: Vec<f32> = pollster::block_on(orchestrator.buffers.read_buffer(
            &orchestrator.device,
            &orchestrator.queue,
            BufferName::OutDebugNormals,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        info!("Read out_debug_normals_buffer: {:?}", debug_buffer);
        assert!(
            slope_angle.iter().all(|&x| (x - 34.00012).abs() < 4e-3),
            "Slope angle values are not as expected. Min: {:?}, Max: {:?}\nHist:\n{:?}",
            slope_angle.min_value(),
            slope_angle.max_value(),
            slope_angle.hist_float()
        );
        assert!(slope_aspect.iter().all(|&x| (x - 90.0).abs() < 1e-6));
        let epsilon = 1e-4;
        assert!(normal_x.iter().all(|&x| (x - 0.55919474).abs() < epsilon));
        assert!(normal_y.iter().all(|&x| (x - 0.0).abs() < epsilon));
        assert!(normal_z.iter().all(|&x| (x - 0.82903636).abs() < epsilon));
        assert!(profile_curvature.iter().all(|&x| (x - 0.0).abs() < epsilon));
    }
    #[test_log::test]
    fn test_load_release_areas() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        let (sim_settings, _dem) = Settings::create_from_path(INCLINED_PLANE_PATH);
        let (data, _, _) = read_png(&Path::new(RELEASE_TEXTURE_PATH)).expect("Failed to read PNG");
        info!("Max: {:?}", data.max_value());

        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .expect("Failed to create buffers and texture descriptions");
        let number_release_cells: u32 = pollster::block_on(orchestrator.run_load_release_areas(&data))
            .expect("Failed to run load_release_areas shader");
        let (release_thickness, _, _, _) =
            pollster::block_on(orchestrator.read_texture::<f32>(TextureName::ReleaseAreas))
                .expect("Failed to get release_areas");
        info!(
            "Read release_texture: len: {} max: {:?} {:?}",
            release_thickness.len(),
            release_thickness.max_value(),
            release_thickness[1020..1040].to_vec(),
        );
        assert_eq!(number_release_cells, 3245);
        info!("Read number_release_cells: {:?}", number_release_cells);
    }

    #[test_log::test]
    fn test_compute() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        let (sim_settings, dem) = Settings::create_from_path(INCLINED_PLANE_PATH);
        pollster::block_on(orchestrator.run_normals(&sim_settings, &dem))
            .expect("Failed to run normals shader");
        let (data, _, _) = read_png(&Path::new(RELEASE_TEXTURE_PATH)).expect("Failed to read PNG");
        let number_release_cells: u32 = pollster::block_on(orchestrator.run_load_release_areas(&data))
            .expect("Failed to run load_release_areas shader");
        assert_eq!(number_release_cells, 3245);
        let number_particles =
            (number_release_cells * sim_settings.released_particles_per_cell) as usize;
        orchestrator.buffers.add_buffer(
            &orchestrator.device,
            BufferName::Particles,
            number_particles * size_of::<Particle>(),
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        info!("Read number_release_cells: {:?}", number_release_cells);
        pollster::block_on(orchestrator.run_initialize_particles())
            .expect("Failed to run initialize_particles shader");
        
        // pollster::block_on(orchestrator.comp())
        //     .expect("Failed to run initialize_particles shader");
        // pollster::block_on(orchestrator.read_texture::<f32>(TextureName::CellCount))
        //         .expect("Failed to get cell_count texture");
        // orchestrator
        //     .save_grid("slope_aspect.bin", slope_aspect.clone())
        //     .expect("Failed to save slope_aspect");
    }
}
