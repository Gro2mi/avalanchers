use crate::buffers::{
    BufferName, ComputeBuffers, TextureName, create_buffers_and_texture_descriptions,
};
use crate::shaders::{ComputeShaderConfig, ShaderName, generate_shader_report};
use anyhow::{Ok, Result, anyhow};
use std::collections::HashMap;
use std::path::PathBuf;
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
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

static INIT: Once = Once::new();

/// Initializes the global tracing subscriber.
pub fn init_logging() {
    INIT.call_once(|| {
        #[cfg(debug_assertions)]
        let filter = EnvFilter::new("error,compute_core=trace,data_processor=debug,cli=debug");
        #[cfg(not(debug_assertions))]
        let filter = EnvFilter::new("error,compute_core=info,data_processor=info,cli=info");

        let _ = tracing_subscriber::registry()
            .with(fmt::layer().with_target(false))
            .with(filter)
            .try_init();

        info!("Simulation logging initialized");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SimulationState {
    Uninitialized,
    Ready,
    NormalsComputed,
    ReleaseAreasComputed,
    ParticlesInitialized,
    Running,
    Finished,
}

#[allow(dead_code)]
pub struct Simulation {
    orchestrator: ComputeOrchestrator,
    pub settings: settings::SimSettings,
    pub info: SimInfo,
    pub dem_path: String,
    pub dem: dem::Dem,
    pub normals: Vec<f32>,
    pub slope: Vec<f32>,
    pub cell_count: Vec<u32>,
    pub max_velocity: Vec<f32>,
    number_particles: u32,
    particles: Vec<Particle>,
    state: SimulationState,
    gpu_cache: GpuCache,
}

impl Simulation {
    pub async fn new() -> Result<Self> {
        let orchestrator = ComputeOrchestrator::new().await?;
        Ok(Self {
            orchestrator,
            settings: settings::SimSettings::default(),
            info: SimInfo::default(),
            dem_path: String::new(),
            dem: dem::Dem::default(),
            normals: Vec::new(),
            slope: Vec::new(),
            cell_count: Vec::new(),
            max_velocity: Vec::new(),
            number_particles: 0,
            particles: Vec::new(),
            state: SimulationState::Uninitialized,
            gpu_cache: GpuCache::default(),
        })
    }
    pub fn get_state(&self) -> SimulationState {
        self.state
    }

    pub async fn create(dem_path: String, settings: settings::SimSettings) -> Result<Self> {
        let mut simulation = Simulation::new().await?;
        simulation.dem_path = dem_path.clone();
        simulation.dem = dem::Dem::new(&dem_path);
        simulation.settings = settings;
        simulation.settings.set_dem(&simulation.dem);
        simulation.state = SimulationState::Ready;
        debug!(
            "Created simulation with DEM path: {}, settings: {:?}",
            simulation.dem_path, simulation.settings
        );
        Ok(simulation)
    }

    pub async fn create_default(dem_path: String) -> Result<Self> {
        let sim_settings = settings::SimSettings::default();
        Self::create(dem_path, sim_settings).await
    }

    pub async fn run(&mut self) -> Result<()> {
        self.compute_normals().await?;
        let _ = self
            .load_release_areas(&self.dem_path.replace(".png", "releaseTexture.png"))
            .await?;
        self.gpu_cache.reset_simulation_result();
        self.initialize_particles().await?;
        self.compute_particles().await?;
        self.state = SimulationState::Finished;
        Ok(())
    }

    async fn compute_normals(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::Ready,
            "DEM and settings must be loaded before running normals shader"
        );
        self.gpu_cache.reset_all();
        self.orchestrator
            .run_normals(&self.settings, &self.dem)
            .await?;
        self.state = SimulationState::NormalsComputed;
        Ok(())
    }

    async fn load_release_areas(&mut self, release_areas_path: &String) -> Result<u32> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before loading release areas"
        );
        debug!("Loading release areas from path: {}", release_areas_path);
        let (data, _, _) = data_processor::read_png(&PathBuf::from(release_areas_path))
            .expect("Failed to read PNG");
        let number_release_cells = self
            .orchestrator
            .run_load_release_areas(&data, &self.settings)
            .await?;
        self.number_particles = number_release_cells * self.settings.released_particles_per_cell;
        self.state = SimulationState::ReleaseAreasComputed;
        info!("Number of release cells: {}", number_release_cells);
        Ok(number_release_cells)
    }

    async fn initialize_particles(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before initializing particles"
        );
        // set parameters that depend on the number of particles
        self.orchestrator
            .run_initialize_particles(self.number_particles)
            .await?;
        self.state = SimulationState::ParticlesInitialized;
        Ok(())
    }

    async fn compute_particles(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::ParticlesInitialized,
            "Particles must be initialized before running particle simulation"
        );
        self.orchestrator
            .run_compute_particles(&self.settings, self.number_particles)
            .await?;
        self.state = SimulationState::Running;
        Ok(())
    }

    async fn get_texture_data<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<TextureRgba<T>> {
        Ok(TextureRgba::from(
            self.orchestrator
                .read_texture(name)
                .await
                .expect("Failed to read texture"),
        ))
    }

    async fn get_texture_data_single_channel<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<Vec<T>> {
        self.orchestrator.read_texture_single_channel(name).await
    }

    async fn get_slope_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before reading normals texture"
        );
        if self.gpu_cache.slope.is_none() {
            self.gpu_cache.slope = Some(self.get_texture_data(TextureName::Slope).await?);
        }
        Ok(self.gpu_cache.slope.as_ref().unwrap())
    }

    pub async fn get_slope_angle(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_slope_texture().await?.r.clone())
    }

    pub async fn get_slope_aspect(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_slope_texture().await?.g.clone())
    }

    async fn get_normals_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before reading normals texture"
        );
        if self.gpu_cache.normals.is_none() {
            self.gpu_cache.normals = Some(self.get_texture_data(TextureName::Normals).await?);
        }
        Ok(self.gpu_cache.normals.as_ref().unwrap())
    }

    pub async fn get_normals_x(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_normals_texture().await?.r.clone())
    }

    pub async fn get_normals_y(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_normals_texture().await?.g.clone())
    }

    pub async fn get_normals_z(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_normals_texture().await?.b.clone())
    }

    pub async fn get_curvature(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_normals_texture().await?.a.clone())
    }

    pub async fn get_dem_texture(&self) -> Result<Vec<f32>> {
        self.get_texture_data_single_channel(TextureName::Dem).await
    }

    async fn get_release_areas_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before reading release areas texture"
        );
        if self.gpu_cache.release_areas.is_none() {
            self.gpu_cache.release_areas =
                Some(self.get_texture_data(TextureName::ReleaseAreas).await?);
        }
        Ok(self.gpu_cache.release_areas.as_ref().unwrap())
    }

    pub async fn get_release_areas(&mut self) -> Result<Vec<f32>> {
        Ok(self.get_release_areas_texture().await?.r.clone())
    }

    pub async fn get_max_velocity(&mut self) -> Result<&Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading max velocity"
        );
        if self.gpu_cache.max_velocity.is_none() {
            let data: Vec<u32> = self
                .orchestrator
                .read_buffer(BufferName::VelocityGrid)
                .await?;
            self.gpu_cache.max_velocity = Some(data.into_iter().map(|x| x as f32).collect());
        }
        Ok(self.gpu_cache.max_velocity.as_ref().unwrap())
    }

    pub async fn get_timestep_data(&mut self) -> Result<&TimestepData> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading timestep data"
        );
        if self.gpu_cache.timestep_data.is_none() {
            let data_aos = self
                .orchestrator
                .read_buffer(BufferName::TimestepData)
                .await?;

            self.gpu_cache.timestep_data = Some(TimestepData::from_aos(&data_aos));
        }
        Ok(self.gpu_cache.timestep_data.as_ref().unwrap())
    }

    pub async fn get_cell_count(&mut self) -> Result<&Vec<u32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading cell count grid"
        );
        if self.gpu_cache.cell_count.is_none() {
            self.gpu_cache.cell_count = Some(
                self.orchestrator
                    .read_buffer(BufferName::CellCountGrid)
                    .await?,
            );
        }
        Ok(self.gpu_cache.cell_count.as_ref().unwrap())
    }

    pub async fn get_particles(&mut self) -> Result<&Vec<Particle>> {
        assert!(
            self.state >= SimulationState::ParticlesInitialized,
            "Simulation must be finished before reading cell count grid"
        );
        if self.gpu_cache.particles.is_none() {
            self.gpu_cache.particles =
                Some(self.orchestrator.read_buffer(BufferName::Particles).await?);
        }
        Ok(self.gpu_cache.particles.as_ref().unwrap())
    }

    pub async fn get_compute_particles_debug(&self) -> Result<Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading cell count grid"
        );
        self.orchestrator
            .read_buffer(BufferName::OutDebugNormals)
            .await
    }
}

struct TextureRgba<T> {
    pub r: Vec<T>,
    pub g: Vec<T>,
    pub b: Vec<T>,
    pub a: Vec<T>,
}
impl<T> From<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> for TextureRgba<T> {
    fn from(channels: (Vec<T>, Vec<T>, Vec<T>, Vec<T>)) -> Self {
        Self {
            r: channels.0,
            g: channels.1,
            b: channels.2,
            a: channels.3,
        }
    }
}

#[derive(Default)]
struct GpuCache {
    pub particles: Option<Vec<Particle>>,
    pub max_velocity: Option<Vec<f32>>,
    pub cell_count: Option<Vec<u32>>,
    pub normals: Option<TextureRgba<f32>>,
    pub slope: Option<TextureRgba<f32>>,
    pub release_areas: Option<TextureRgba<f32>>,
    pub timestep_data: Option<TimestepData>,
}

impl GpuCache {
    pub fn reset_simulation_result(&mut self) {
        self.particles = None;
        self.max_velocity = None;
        self.cell_count = None;
        self.timestep_data = None;
    }

    pub fn reset_all(&mut self) {
        self.reset_simulation_result();
        self.normals = None;
        self.slope = None;
        self.release_areas = None;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub mass: f32,
    pub velocity: [f32; 3],
    pub snow_thickness: f32, // padding to align next field
    pub c: [f32; 4],         // 2x2 matrix: [xx, xy, yx, yy]
    pub stopped: u32,
    pub _pad0: [u32; 3], // 3 * 4 bytes padding
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
            snow_thickness: 0.0,
            c: [0.0; 4],
            stopped: 0,
            _pad0: [0; 3],
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

impl Default for SimInfo {
    fn default() -> Self {
        Self {
            timestep: 0,
            number_particles: 0,
            elevation_threshold: 0.0,
            max_velocity: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimestepDataAoS {
    pub velocity: [f32; 3], // 12 bytes
    pub dt: f32,            // 4 bytes

    pub acceleration_tangential: [f32; 3],    // 12 bytes
    pub acceleration_friction_magnitude: f32, // 4 bytes

    pub position: [f32; 3], // 12 bytes
    pub elevation: f32,     // 4 bytes

    pub normal: [f32; 3], // 12 bytes
    pub g_eff: f32,       // 4 bytes

    pub acceleration_normal: [f32; 3], // 12 bytes
    pub _pad0: f32,                    // 4 bytes (padding)

    pub uv: [f32; 2],    // 4 bytes
    pub _pad1: [f32; 2], // 12 bytes (padding to 96 bytes)
}

#[derive(Clone)]
pub struct TimestepData {
    pub velocity: Vec<[f32; 3]>,
    pub dt: Vec<f32>,
    pub acceleration_tangential: Vec<[f32; 3]>,
    pub acceleration_friction_magnitude: Vec<f32>,
    pub position: Vec<[f32; 3]>,
    pub elevation: Vec<f32>,
    pub normal: Vec<[f32; 3]>,
    pub g_eff: Vec<f32>,
    pub acceleration_normal: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
}

impl TimestepData {
    pub fn from_aos(aos_data: &[TimestepDataAoS]) -> Self {
        let len = aos_data.len();

        // Pre-allocate all vectors to the exact required size
        let mut soa = Self {
            velocity: Vec::with_capacity(len),
            dt: Vec::with_capacity(len),
            acceleration_tangential: Vec::with_capacity(len),
            acceleration_friction_magnitude: Vec::with_capacity(len),
            position: Vec::with_capacity(len),
            elevation: Vec::with_capacity(len),
            normal: Vec::with_capacity(len),
            g_eff: Vec::with_capacity(len),
            acceleration_normal: Vec::with_capacity(len),
            uv: Vec::with_capacity(len),
        };

        for item in aos_data {
            soa.velocity.push(item.velocity);
            soa.dt.push(item.dt);
            soa.acceleration_tangential
                .push(item.acceleration_tangential);
            soa.acceleration_friction_magnitude
                .push(item.acceleration_friction_magnitude);
            soa.position.push(item.position);
            soa.elevation.push(item.elevation);
            soa.normal.push(item.normal);
            soa.g_eff.push(item.g_eff);
            soa.acceleration_normal.push(item.acceleration_normal);
            soa.uv.push(item.uv);
        }

        soa
    }
}

pub struct WorkgroupSize {}
impl WorkgroupSize {
    const SIZE_1D: u32 = 64;
    const SIZE_2D: u32 = 16;
}

pub struct ComputeOrchestrator {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub buffers: ComputeBuffers,
    pub sampler: Sampler,
    pub max_texture_size: u32,
    pub max_storage_buffer_binding_size: u64,
    texture_size: Extent3d,
    shader_configs: HashMap<ShaderName, ComputeShaderConfig>,
    dispatch_number_workgroups_x_2d: u32,
    dispatch_number_workgroups_y_2d: u32,
    dispatch_number_workgroups_1d: u32,
}

impl ComputeOrchestrator {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter: wgpu::Adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let info = adapter.get_info();
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => info!("Using discrete GPU: {}", info.name),
            wgpu::DeviceType::IntegratedGpu => warn!(
                "Using integrated GPU: {}. If performance is poor, consider using a discrete GPU",
                info.name
            ),
            wgpu::DeviceType::VirtualGpu => {
                warn!("Using virtual GPU: {}, performance may be poor", info.name)
            }
            wgpu::DeviceType::Cpu => warn!(
                "Using CPU adapter: {}, performance will be very poor",
                info.name
            ),
            wgpu::DeviceType::Other => warn!(
                "Using unknown device type for adapter: {}, performance may be poor",
                info.name
            ),
        }
        info!("GPU Name    : {}", info.name);
        debug!("Driver      : {}", info.driver);
        debug!("Backend     : {:?}", info.backend);
        debug!("Device Type : {:?}", info.device_type);
        trace!("Adapter limits: {:?}", adapter.limits());

        let max_texture_size = adapter.limits().max_texture_dimension_2d;
        let max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        debug!(
            "Adapter limits: 
                                    - Max Compute Workgroup Size X: {:?}
                                    - Max Compute Invocations Per Workgroup: {:?} 
                                    - Max Storage Buffer Binding Size: {:.2} GB
                                    - Max Buffer Size: {:.2} GB
                                    - Max Texture Dimension 2D: {:?}",
            adapter.limits().max_compute_workgroup_size_x,
            adapter.limits().max_compute_invocations_per_workgroup,
            max_storage_buffer_binding_size as f64 / 1024.0 / 1024.0 / 1024.0,
            adapter.limits().max_buffer_size as f64 / 1024.0 / 1024.0 / 1024.0,
            max_texture_size
        );

        // let workgroup_size_2d = utils::highest_power_of_two(
        //     (adapter.limits().max_compute_workgroup_size_x as f64).sqrt() as u32,
        // );
        // debug!("Workgroup size 2D: {}", workgroup_size_2d);
        // let max_invocations = adapter.limits().max_compute_invocations_per_workgroup;
        // let workgroup_size_1d = max_invocations;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Compute Device"),
                required_features: Features::FLOAT32_FILTERABLE | Features::TIMESTAMP_QUERY,
                required_limits: Limits {
                    max_compute_workgroup_size_x: WorkgroupSize::SIZE_1D,
                    max_compute_workgroup_size_y: WorkgroupSize::SIZE_2D,
                    max_compute_workgroup_size_z: 1,
                    max_compute_invocations_per_workgroup: WorkgroupSize::SIZE_2D
                        * WorkgroupSize::SIZE_2D, // 256
                    max_storage_buffer_binding_size,
                    ..Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device and queue");

        let buffers = ComputeBuffers::new();
        let shader_configs = shaders::create_shader_configs(&device)?;
        let texture_size = Extent3d::default();
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
        let dispatch_number_workgroups_x_2d = 0;
        let dispatch_number_workgroups_y_2d = 0;
        let dispatch_number_workgroups_1d = 0;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            buffers,
            shader_configs,
            texture_size,
            sampler,
            max_texture_size,
            max_storage_buffer_binding_size,
            dispatch_number_workgroups_x_2d,
            dispatch_number_workgroups_y_2d,
            dispatch_number_workgroups_1d,
        })
    }

    #[allow(dead_code)]
    fn generate_shader_report(&self) -> String {
        generate_shader_report(&self.shader_configs)
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

    pub async fn run_shader(
        &self,
        shader_name: &ShaderName,
        resources: &[BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
        dispatch_number_workgroups_x: u32,
        dispatch_number_workgroups_y: u32,
        dispatch_number_workgroups_z: u32,
    ) -> Result<()> {
        assert_ne!(
            dispatch_number_workgroups_x, 0,
            "dispatch_number_workgroups_x must be greater than 0, check your settings"
        );
        assert_ne!(
            dispatch_number_workgroups_y, 0,
            "dispatch_number_workgroups_y must be greater than 0, check your settings"
        );
        assert_ne!(
            dispatch_number_workgroups_z, 0,
            "dispatch_number_workgroups_z must be greater than 0, check your settings"
        );
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
                dispatch_number_workgroups_x,
                dispatch_number_workgroups_y,
                dispatch_number_workgroups_z,
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
        &mut self,
        sim_settings: &settings::SimSettings,
        dem: &dem::Dem,
    ) -> Result<()> {
        assert!(
            sim_settings.grid_shape_x <= self.max_texture_size
                && sim_settings.grid_shape_y <= self.max_texture_size,
            "Grid shape ({}, {}) exceeds max texture size of {}. Consider reducing the grid shape or using a GPU with larger max texture size.",
            sim_settings.grid_shape_x,
            sim_settings.grid_shape_y,
            self.max_texture_size
        );
        self.texture_size = Extent3d {
            width: sim_settings.grid_shape_x,
            height: sim_settings.grid_shape_y,
            depth_or_array_layers: 1,
        };

        self.dispatch_number_workgroups_x_2d =
            sim_settings.grid_shape_x.div_ceil(WorkgroupSize::SIZE_2D);
        self.dispatch_number_workgroups_y_2d =
            sim_settings.grid_shape_y.div_ceil(WorkgroupSize::SIZE_2D);

        self.buffers = create_buffers_and_texture_descriptions(&self.device, self.texture_size);

        self.buffers.add_buffer_with_data(
            &self.device,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
        debug!("Running compute_normals shader...");
        let _ = self.buffers.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        );

        self.run_shader(
            &ShaderName::ComputeNormals,
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::Wind),
                self.get_view(TextureName::Normals),
                self.get_view(TextureName::Slope),
                self.get_buffer_binding(BufferName::OutDebugNormals),
            ],
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;
        Ok(())
    }

    pub async fn run_load_release_areas(
        &mut self, // `&mut self` because we're adding textures
        data: &[u8],
        sim_settings: &settings::SimSettings,
    ) -> Result<u32> {
        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.dispatch_number_workgroups_x_2d =
            sim_settings.grid_shape_x.div_ceil(WorkgroupSize::SIZE_2D);
        self.dispatch_number_workgroups_y_2d =
            sim_settings.grid_shape_y.div_ceil(WorkgroupSize::SIZE_2D);

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
        self.run_shader(
            &ShaderName::LoadReleaseAreas,
            &[
                self.get_view(TextureName::ReleaseAreasInput),
                self.get_view(TextureName::ReleaseAreas),
                self.get_buffer_binding(BufferName::NumberReleaseCells),
                self.get_buffer_binding(BufferName::OutDebugRelease),
            ],
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let number_release_cells: u32 = self
            .read_buffer::<u32>(BufferName::NumberReleaseCells)
            .await
            .expect("Failed to read number_release_cells buffer")[0];
        Ok(number_release_cells)
    }

    pub async fn run_initialize_particles(&mut self, number_release_particles: u32) -> Result<()> {
        let particle_buffer_size = number_release_particles as usize * Particle::BYTE_SIZE;
        assert!(
            particle_buffer_size as u64 <= self.max_storage_buffer_binding_size,
            "Particle buffer size {} bytes exceeds max storage buffer binding size of {} bytes. Consider reducing the number of particles or using a GPU with more memory.",
            particle_buffer_size,
            self.max_storage_buffer_binding_size
        );
        info!(
            "Initializing particles with number_release_particles: {}, particle_buffer_size: {:.2} MB ({:.1} % of max storage buffer binding size)",
            number_release_particles,
            particle_buffer_size as f64 / 1024.0 / 1024.0,
            (particle_buffer_size as f64 / self.max_storage_buffer_binding_size as f64) * 100.0
        );
        self.dispatch_number_workgroups_1d =
            number_release_particles.div_ceil(WorkgroupSize::SIZE_1D);
        debug!(
            "Running initialize particles shader with number_release_particles: {}, dispatch_number_workgroups_1d: {}",
            number_release_particles, self.dispatch_number_workgroups_1d
        );
        self.buffers.add_buffer_with_data(
            &self.device,
            BufferName::ParticleIndex,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        self.add_buffer(
            BufferName::Particles,
            particle_buffer_size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        self.run_shader(
            &ShaderName::InitializeParticles,
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_buffer_binding(BufferName::SimInfo),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::ReleaseAreas),
                self.get_sampler(),
                self.get_buffer_binding(BufferName::Particles),
                self.get_buffer_binding(BufferName::NumberReleaseParticles),
                self.get_buffer_binding(BufferName::CellCountGrid),
                self.get_buffer_binding(BufferName::MaxVelocityGrid),
            ],
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;
        Ok(())
    }

    pub async fn run_compute_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
    ) -> Result<()> {
        debug!("Start simulation");
        self.add_buffer(
            BufferName::TimestepData,
            size_of::<TimestepDataAoS>() * sim_settings.max_steps as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let sim_info: SimInfo = SimInfo {
            timestep: 0,
            number_particles: number_release_particles,
            elevation_threshold: 0.0,
            max_velocity: 0.0,
        };

        self.buffers.write_buffer(
            &self.queue,
            BufferName::SimInfo,
            bytemuck::bytes_of(&sim_info),
        )?;
        // Compute Particles Bind Group
        let compute_particles_config = self
            .shader_configs
            .get(&ShaderName::ComputeParticles)
            .expect("ComputeParticles shader config not found");

        let compute_particles_bindgroup = compute_particles_config.create_bind_group(
            &self.device,
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_buffer_binding(BufferName::SimInfo),
                self.get_view(TextureName::Dem),
                self.get_view(TextureName::Normals),
                self.get_sampler(),
                self.get_buffer_binding(BufferName::Particles),
                self.get_buffer_binding(BufferName::MaxVelocityGrid),
                self.get_buffer_binding(BufferName::CellCountGrid),
                self.get_buffer_binding(BufferName::VelocityGrid),
                self.get_buffer_binding(BufferName::TimestepData),
                self.get_buffer_binding(BufferName::OutDebugNormals),
            ],
        )?;

        let reset_max_velocity_config = self
            .shader_configs
            .get(&ShaderName::ResetMaxVelocity)
            .expect("ResetMaxVelocity shader config not found");

        // Reset Max Velocity Bind Group
        let reset_max_velocity_bind_group = reset_max_velocity_config.create_bind_group(
            &self.device,
            &[
                self.get_buffer_binding(BufferName::SimSettings),
                self.get_buffer_binding(BufferName::SimInfo),
                self.get_buffer_binding(BufferName::MaxVelocityGrid),
            ],
        )?;

        // Create command encoder
        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

        // Begin compute pass
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

            for _i in 0..sim_settings.max_steps {
                // --- computeParticles ---
                compute_pass.set_pipeline(&compute_particles_config.pipeline);
                compute_pass.set_bind_group(0, &compute_particles_bindgroup, &[]);
                compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                // --- resetMaxVelocity ---
                compute_pass.set_pipeline(&reset_max_velocity_config.pipeline);
                compute_pass.set_bind_group(0, &reset_max_velocity_bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        } // compute_pass dropped here (equivalent to end())

        // Submit commands
        self.queue.submit(Some(command_encoder.finish()));

        Ok(())
    }
    async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        self.buffers
            .read_texture(&self.device, &self.queue, name)
            .await
    }
    async fn read_texture_single_channel<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<Vec<T>> {
        self.buffers
            .read_texture_single_channel(&self.device, &self.queue, name)
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

    fn add_buffer(&mut self, name: BufferName, size_bytes: usize, usage: BufferUsages) {
        self.buffers
            .add_buffer(&self.device, name, size_bytes, usage);
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
    use super::*;
    use crate::settings::Settings;
    use data_processor::read_png;
    use pollster;
    use std::collections::HashSet;
    use std::mem;
    use std::path::Path;
    use utils::{Hist, HistFloat, MaxValue, MinValue};

    const INCLINED_PLANE_PATH: &str = "../../data/avaframe/avaInclinedPlane.png";
    const RELEASE_TEXTURE_PATH: &str = "../../data/avaframe/avaInclinedPlanereleaseTexture.png";
    const GAR_PATH: &str = "../../data/avaframe/avaGar.png";
    const GAR_RELEASE_TEXTURE_PATH: &str = "../../data/avaframe/avaGarreleaseTexture.png";

    #[test]
    fn test_init_logging_idempotent() {
        // Call it once
        init_logging();

        // Call it again - it should not panic or error because of .call_once()
        init_logging();
    }

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
        info!("Max: {:?}", data.max_value().unwrap());

        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .expect("Failed to create buffers and texture descriptions");
        let number_release_cells: u32 =
            pollster::block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
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
        assert_eq!(release_thickness.iter().filter(|&&x| x > 0.0).count(), 3245);
        assert!(
            release_thickness
                .iter()
                .all(|&x| x == 0.0 || (x - 1.0).abs() < 1e-6)
        );
        info!("Read number_release_cells: {:?}", number_release_cells);
    }
    #[test_log::test]
    fn test_load_release_areas_gar() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        let (sim_settings, _dem) = Settings::create_from_path(GAR_PATH);
        let (data, _, _) =
            read_png(&Path::new(GAR_RELEASE_TEXTURE_PATH)).expect("Failed to read PNG");
        info!("Max: {:?}", data.max_value().unwrap());

        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .expect("Failed to create buffers and texture descriptions");
        let number_release_cells: u32 =
            pollster::block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
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
        assert_eq!(number_release_cells, 1628);
        assert_eq!(release_thickness.iter().filter(|&&x| x > 0.0).count(), 1628);
        assert!(
            release_thickness
                .iter()
                .all(|&x| x == 0.0 || (x - 1.2).abs() < 1e-6)
        );
        info!("Read number_release_cells: {:?}", number_release_cells);
    }

    #[test_log::test]
    fn test_initialize_particles() {
        let mut orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        let (mut sim_settings, dem) = Settings::create_from_path(INCLINED_PLANE_PATH);
        sim_settings.released_particles_per_cell = 10;
        info!("Sim settings: {:?}", sim_settings);
        pollster::block_on(orchestrator.run_normals(&sim_settings, &dem))
            .expect("Failed to run normals shader");
        let (data, _, _) = read_png(&Path::new(RELEASE_TEXTURE_PATH)).expect("Failed to read PNG");
        let number_release_cells: u32 =
            pollster::block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
                .expect("Failed to run load_release_areas shader");
        pollster::block_on(orchestrator.run_initialize_particles(
            number_release_cells * sim_settings.released_particles_per_cell,
        ))
        .expect("Failed to run initialize_particles shader");
        let number_release_particles =
            pollster::block_on(orchestrator.read_buffer::<u32>(BufferName::NumberReleaseParticles))
                .expect("Failed to read particle index buffer")[0];
        info!("Number release particles: {}", number_release_particles);
        let particles =
            pollster::block_on(orchestrator.read_buffer::<Particle>(BufferName::Particles))
                .expect("Failed to read particles buffer");
        let particle = particles.first().expect("No particles found");
        for p in particles.iter() {
            assert!(p.position[0] > 100.0);
            assert!(p.position[0] < 400.0);
            assert!(p.position[1] < 1150.0);
            assert!(p.position[1] > 850.0);
            assert!(p.position[2] > 3000.0);
            assert!(p.position[2] < 3350.0);
        }
        assert_eq!(particle.mass, 500.0);
        let unique_values = particles
            .iter()
            .map(|p| {
                [
                    p.position[0].to_bits(),
                    p.position[0].to_bits(),
                    p.position[0].to_bits(),
                ]
            })
            .collect::<HashSet<_>>()
            .len();
        info!(
            "Unique values: {}, {}%",
            unique_values,
            unique_values as f32 / particles.len() as f32 * 100.0
        );
        assert!(
            unique_values as f32 / particles.len() as f32 > 0.98,
            "Duplicate position found in vector"
        );
        info!(
            "Read particles buffer: len: {}, first particle: {:?}",
            particles.len(),
            particles.last()
        );
        let cell_count_grid =
            pollster::block_on(orchestrator.read_buffer::<u32>(BufferName::CellCountGrid))
                .expect("Failed to read cell count grid");

        info!(
            "Read cell count grid: len: {:?}, max value: {:?}",
            cell_count_grid.hist(),
            cell_count_grid.max_value().unwrap()
        );
        assert_eq!(particles.iter().filter(|&&x| x.mass > 0.0).count(), 32450);
        assert!(cell_count_grid.max_value().unwrap() <= 12); // 10 particles per cell + some edge cases
        assert!(cell_count_grid.hist().get(&0).unwrap() > &398150); // most cells dont have particles
    }

    #[test_log::test]
    fn test_compute() {
        let mut sim: Simulation =
            pollster::block_on(Simulation::create_default(INCLINED_PLANE_PATH.to_string()))
                .expect("Failed to create simulation");
        pollster::block_on(sim.run()).expect("Failed to run simulation");
        let debug_buffer: Vec<f32> = pollster::block_on(sim.orchestrator.buffers.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::OutDebugNormals,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        log_debug_buffer(&debug_buffer);
        let cell_count =
            pollster::block_on(sim.get_cell_count()).expect("Failed to get cell count");
        info!("Cell count max: {:?}", cell_count.max_value().unwrap());
        let max_velocity =
            pollster::block_on(sim.get_max_velocity()).expect("Failed to get max velocity");
        info!("Max velocity: {:?}", max_velocity.max_value().unwrap());

        let sim_info: Vec<SimInfo> = pollster::block_on(sim.orchestrator.buffers.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::SimInfo,
        ))
        .expect("Failed to read sim info buffer");
        info!("Read sim info: {:?}", sim_info);

        let particles = pollster::block_on(sim.orchestrator.buffers.read_buffer::<Particle>(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::Particles,
        ))
        .expect("Failed to read particles buffer");
        // for particle in particles.iter() {
        //     if particle.velocity[0] != 0.0
        //         || particle.velocity[1] != 0.0
        //         || particle.velocity[2] != 0.0
        //     {
        //         info!(
        //             "Particle position: {:?}, velocity: {:?}, mass: {}",
        //             particle.position, particle.velocity, particle.mass
        //         );
        //     }
        // }
        let timestep_data =
            pollster::block_on(sim.orchestrator.buffers.read_buffer::<TimestepDataAoS>(
                &sim.orchestrator.device,
                &sim.orchestrator.queue,
                BufferName::TimestepData,
            ))
            .expect("Failed to read timestep data buffer");
        info!("Read timestep data: len: {}", timestep_data.len());
        for (i, data) in timestep_data.iter().step_by(3).take(40).enumerate() {
            info!("Timestep {}: {:?}", i, data);
        }
    }

    fn log_debug_buffer(buffer: &[f32]) {
        info!("Debug buffer length: {}", buffer.len());
        for (i, value) in buffer.iter().enumerate() {
            if *value != 0.0 {
                info!("{}: {}", i, value);
            }
        }
    }

    #[test_log::test]
    fn test_shader_report_generation() {
        let orchestrator = pollster::block_on(ComputeOrchestrator::new())
            .expect("Failed to create ComputeOrchestrator");
        orchestrator.generate_shader_report();
    }

    #[test]
    fn test_texture_rgba_from_tuple() {
        // 1. Prepare sample data
        let r = vec![1.0, 0.2, 0.3];
        let g = vec![0.4, 0.5, 0.6];
        let b = vec![0.7, 0.8, 0.9];
        let a = vec![1.0, 1.0, 1.0];

        // 2. Perform the conversion using .into()
        // Note: We explicitly type it to ensure the compiler uses our From impl
        let texture: TextureRgba<f32> = (r.clone(), g.clone(), b.clone(), a.clone()).into();

        // 3. Assert the data moved correctly
        assert_eq!(texture.r, r);
        assert_eq!(texture.g, g);
        assert_eq!(texture.b, b);
        assert_eq!(texture.a, a);
    }

    #[test]
    fn test_texture_rgba_generic_u8() {
        // Testing with u8 to ensure the generic <T> works as expected
        let r = vec![255, 0];
        let g = vec![128, 64];
        let b = vec![0, 255];
        let a = vec![255, 255];

        let texture = TextureRgba::from((r.clone(), g.clone(), b.clone(), a.clone()));

        assert_eq!(texture.r[0], 255);
        assert_eq!(texture.g[1], 64);
    }

    #[test]
    fn test_particle_initialization() {
        // Test both new() and Default
        let p1 = Particle::new();
        let p2 = Particle::default();

        // Check a few key fields
        assert_eq!(p1.position, [0.0; 3]);
        assert_eq!(p1.velocity, [0.0; 3]);
        assert_eq!(p1.stopped, 0);

        // Ensure new() and default() are identical
        assert_eq!(p1.position, p2.position);
        assert_eq!(p1.c, p2.c);
    }

    #[test]
    fn test_particle_memory_layout() {
        // This is CRITICAL for WebGPU.
        // We verify the size of the struct matches your constant.
        assert_eq!(
            mem::size_of::<Particle>(),
            Particle::BYTE_SIZE,
            "Particle struct size does not match BYTE_SIZE constant!"
        );

        // Verify that the size is exactly 64 bytes (16 * 4)
        assert_eq!(mem::size_of::<Particle>(), 64);
    }

    #[test]
    fn test_field_offsets() {
        // Optional: Verifies that fields are where you think they are.
        // WebGPU expects 'c' (the matrix) to start at byte 32 in this layout.
        let p = Particle::new();
        let base_ptr = &p as *const _ as usize;
        let c_ptr = &p.c as *const _ as usize;

        let offset_c = c_ptr - base_ptr;
        assert_eq!(
            offset_c, 32,
            "Field 'c' is not at the expected 32-byte offset!"
        );
    }
}
