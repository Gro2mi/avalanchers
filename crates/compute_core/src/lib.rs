use crate::buffers::{
    AtomicValues, BufferName, GpuResources, TextureName, create_buffers_and_texture_descriptions,
};
use crate::settings::SimFlags;
use crate::shaders::{ComputeShaderConfig, ShaderName, generate_shader_report};
use crate::utils::timer_checkpoint;
use anyhow::{Ok, Result, anyhow};
use std::cmp::min;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use wgpu::{
    Adapter, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    Queue, RequestAdapterOptions, TextureFormat, TextureUsages,
};

// use log::{debug, info, warn, error};
pub mod buffers;
pub mod dem;
pub mod settings;
pub mod shaders;
pub mod utils;
use dem::Dem;
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

pub struct TextureRgba<T> {
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
pub struct GpuCache {
    pub particles: Option<Vec<Particle>>,
    pub peak_velocity: Option<Vec<f32>>,
    pub peak_flow_thickness: Option<Vec<f32>>,
    pub cell_count: Option<Vec<u32>>,
    pub normals: Option<TextureRgba<f32>>,
    pub slope: Option<TextureRgba<f32>>,
    pub roughness: Option<TextureRgba<f32>>,
    pub release_areas: Option<TextureRgba<f32>>,
    pub timestep_data: Option<TimestepData>,
    pub read_count: usize,
}

impl GpuCache {
    pub fn reset_simulation_result(&mut self) {
        self.particles = None;
        self.peak_velocity = None;
        self.cell_count = None;
        self.timestep_data = None;
        self.peak_flow_thickness = None;
    }

    pub fn reset_all(&mut self) {
        self.reset_simulation_result();
        self.normals = None;
        self.slope = None;
        self.roughness = None;
        self.release_areas = None;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct Particle {
    pub position: [f32; 3],
    pub mass: f32,
    pub velocity: [f32; 3],
    pub stopped: u32,
    pub travel_length: f32,
    pub _pad: [f32; 3], // Padding to make the struct size a multiple of 16 bytes (for better GPU alignment)
}

impl Hash for Particle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash position bits
        for val in &self.position {
            val.to_bits().hash(state);
        }
        // Hash mass bits
        self.mass.to_bits().hash(state);
        // Hash velocity bits
        for val in &self.velocity {
            val.to_bits().hash(state);
        }
        // These are already hashable (integers)
        self.stopped.hash(state);
    }
}

// You MUST also implement PartialEq and Eq to match Hash logic
impl PartialEq for Particle {
    fn eq(&self, other: &Self) -> bool {
        self.position
            .iter()
            .zip(other.position.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.mass.to_bits() == other.mass.to_bits()
            && self
                .velocity
                .iter()
                .zip(other.velocity.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.stopped == other.stopped
    }
}

impl Eq for Particle {}

impl Particle {
    pub fn new() -> Self {
        Self {
            position: [0.0; 3],
            mass: 0.0,
            velocity: [0.0; 3],
            stopped: 0,
            travel_length: 0.0,
            _pad: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimInfo {
    pub timestep: u32,
    pub dt: f32,
    pub elapsed_time: f32,
    pub number_particles: u32,
    pub elevation_threshold: f32,
    pub max_velocity: f32,
    pub max_flow_thickness: f32,
    pub flags: u32,
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
    pub velocity_magnitude: Vec<f32>,
    pub acceleration_tangential_magnitude: Vec<f32>,
    pub time: Vec<f32>,
    pub step_distance: Vec<f32>,
    pub travel_distance: Vec<f32>,
    pub cfl: Vec<f32>,
}

impl TimestepData {
    pub fn from_aos(aos_data: &[TimestepDataAoS], cell_size: f32) -> Self {
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
            velocity_magnitude: Vec::with_capacity(len),
            acceleration_tangential_magnitude: Vec::with_capacity(len),
            time: Vec::with_capacity(len),
            step_distance: Vec::with_capacity(len),
            travel_distance: Vec::with_capacity(len),
            cfl: Vec::with_capacity(len),
        };

        for item in aos_data {
            let velocity_magnitude = magnitude(&item.velocity);
            if velocity_magnitude < 1e-5 {
                break;
            }
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
            soa.velocity_magnitude.push(velocity_magnitude);
            soa.acceleration_tangential_magnitude
                .push(magnitude(&item.acceleration_tangential));
        }
        // first time step
        soa.time.push(0.0);
        soa.step_distance.push(0.0);
        soa.travel_distance.push(0.0);
        soa.cfl.push(0.0);

        for n in 1..soa.position.len() {
            let prev_pos = soa.position[n - 1];
            let curr_pos = soa.position[n];

            let dist = magnitude_diff(&curr_pos, &prev_pos);

            soa.time.push(soa.time[n - 1] + soa.dt[n]);
            soa.step_distance.push(dist);
            soa.travel_distance.push(soa.travel_distance[n - 1] + dist);
            soa.cfl
                .push(soa.velocity_magnitude[n] * soa.dt[n] / cell_size);
        }

        soa
    }
}

fn magnitude(v: &[f32; 3]) -> f32 {
    (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt()
}

fn magnitude_diff(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

const WORKGROUP_SIZE_2D: u32 = 16;

pub struct ComputeOrchestrator {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub resources: GpuResources,
    pub max_texture_size: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_particles: u64,
    pub max_compute_invocations_per_workgroup: u32,
    pub batch_compute_steps: u32,
    texture_size: Extent3d,
    shader_configs: HashMap<ShaderName, ComputeShaderConfig>,
    dispatch_number_workgroups_x_2d: u32,
    dispatch_number_workgroups_y_2d: u32,
    dispatch_number_workgroups_1d: u32,
    has_float32_filterable: bool,
}

impl ComputeOrchestrator {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let mut adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        if adapter.is_err() {
            warn!("High-performance GPU not found, falling back to LowPower/Software.");
            adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await;
        }

        if adapter.is_err() {
            warn!("Low-performance GPU not found, falling back to Software.");
            adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .await;
        }

        let adapter = adapter.expect("Failed to find any suitable GPU adapter");
        timer_checkpoint("Get GPU adapter");

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
        let limits = adapter.limits();
        info!("GPU Name    : {}", info.name);
        debug!("Driver      : {}", info.driver);
        debug!("Backend     : {:?}", info.backend);
        debug!("Device Type : {:?}", info.device_type);
        trace!("Adapter limits: {:?}", limits);

        let max_texture_size = limits.max_texture_dimension_2d;
        let max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
        let max_buffer_size = limits.max_buffer_size;
        let max_compute_invocations_per_workgroup = min(
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
        );
        debug!(
            "Adapter limits: 
                                    - Max Compute Workgroup Size X: {:?}
                                    - Max Compute Invocations Per Workgroup: {:?} 
                                    - Max Storage Buffer Binding Size: {:.2} GB
                                    - Max Buffer Size: {:.2} GB
                                    - Max Texture Dimension 2D: {:?}
                                    - Max Compute Workgroups per Dimension: {:?}",
            limits.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup,
            max_storage_buffer_binding_size as f64 / 1024.0 / 1024.0 / 1024.0,
            max_buffer_size as f64 / 1024.0 / 1024.0 / 1024.0,
            max_texture_size,
            limits.max_compute_workgroups_per_dimension
        );

        let buffer_limit = max_storage_buffer_binding_size / std::mem::size_of::<Particle>() as u64;
        let compute_limit =
            limits.max_compute_workgroups_per_dimension * max_compute_invocations_per_workgroup;
        let max_particles = min(buffer_limit, compute_limit as u64);
        info!(
            "Maximum number of particles that can be simulated with current GPU: {} (limited by {})",
            max_particles,
            if max_particles == buffer_limit {
                "storage buffer binding size"
            } else {
                "compute shader dispatch limits"
            }
        );
        trace!(
            "Maximum number of cells that can be simulated with current GPU: {}, every {}th cell can have a single particle",
            max_texture_size * max_texture_size,
            (max_texture_size * max_texture_size) as f32 / max_particles as f32
        );

        let mut required_features = Features::empty();
        let mut has_float32_filterable = false;

        // Only request timestamps if the runner actually supports them
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
        {
            required_features |= wgpu::Features::FLOAT32_FILTERABLE;
            has_float32_filterable = true;
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Compute Device"),
                required_features,
                required_limits: Limits {
                    max_compute_workgroup_size_x: max_compute_invocations_per_workgroup,
                    max_compute_workgroup_size_y: WORKGROUP_SIZE_2D,
                    max_compute_workgroup_size_z: 1,
                    max_compute_invocations_per_workgroup,
                    max_storage_buffer_binding_size,
                    max_buffer_size,
                    max_storage_buffers_per_shader_stage: 10,
                    ..Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device and queue");
        device.set_device_lost_callback(move |reason, message| {
            error!("Device lost! Reason: {:?}, Message: {}", reason, message);
        });
        timer_checkpoint("Request GPU device");
        let buffers = GpuResources::new();
        let shader_configs = shaders::create_shader_configs(
            &device,
            max_compute_invocations_per_workgroup,
            has_float32_filterable,
        )?;
        timer_checkpoint("Create shaders");
        let texture_size = Extent3d::default();

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            resources: buffers,
            shader_configs,
            texture_size,
            max_texture_size,
            max_storage_buffer_binding_size,
            max_compute_invocations_per_workgroup,
            max_particles,
            dispatch_number_workgroups_x_2d: 0,
            dispatch_number_workgroups_y_2d: 0,
            dispatch_number_workgroups_1d: 0,
            has_float32_filterable,
            batch_compute_steps: 200,
        })
    }

    #[allow(dead_code)]
    fn generate_shader_report(&self) -> String {
        generate_shader_report(&self.shader_configs)
    }

    pub async fn run_shader(
        &self,
        shader_name: &ShaderName,
        // resources: &[BindingResource<'_>], // Pass actual resources (buffer bindings or texture views)
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

        let bind_group = config.create_bind_group(&self.device, &self.resources)?;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some(&format!("Compute Encoder for shader: {}", shader_name)),
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
        self.resources = create_buffers_and_texture_descriptions(
            &self.device,
            self.texture_size,
            self.has_float32_filterable,
        );
        Ok(())
    }

    pub async fn run_analyze_terrain(
        &mut self,
        sim_settings: &settings::SimSettings,
        dem: &Dem,
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
            sim_settings.grid_shape_x.div_ceil(WORKGROUP_SIZE_2D);
        self.dispatch_number_workgroups_y_2d =
            sim_settings.grid_shape_y.div_ceil(WORKGROUP_SIZE_2D);

        self.resources = create_buffers_and_texture_descriptions(
            &self.device,
            self.texture_size,
            self.has_float32_filterable,
        );

        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;

        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.resources
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

        self.run_shader(
            &ShaderName::AnalyzeTerrain,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;
        Ok(())
    }

    pub async fn run_compute_roughness(
        &mut self,
        sim_settings: &settings::SimSettings,
    ) -> Result<()> {
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        self.run_shader(
            &ShaderName::ComputeRoughness,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;
        Ok(())
    }

    pub async fn run_compute_release_areas(
        &mut self,
        sim_settings: &settings::SimSettings,
    ) -> Result<u32> {
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        self.run_shader(
            &ShaderName::ComputeReleaseAreas,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let number_release_cells: u32 = self
            .read_buffer::<buffers::AtomicValues>(BufferName::AtomicValues)
            .await
            .expect("Failed to read number_release_cells buffer")[0]
            .number_release_cells;

        Ok(number_release_cells)
    }

    pub async fn run_load_release_areas(
        &mut self, // `&mut self` because we're adding textures
        data: &[f32],
        sim_settings: &settings::SimSettings,
    ) -> Result<u32> {
        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        self.dispatch_number_workgroups_x_2d =
            sim_settings.grid_shape_x.div_ceil(WORKGROUP_SIZE_2D);
        self.dispatch_number_workgroups_y_2d =
            sim_settings.grid_shape_y.div_ceil(WORKGROUP_SIZE_2D);

        self.resources
            .add_texture_with_data(
                &self.device,
                &self.queue,
                data,
                TextureName::ReleaseAreasInput,
                self.texture_size,
                TextureFormat::R32Float,
                texture_usage_input,
            )
            .expect("Failed to add texture with data");
        self.run_shader(
            &ShaderName::LoadReleaseAreas,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let number_release_cells = self
            .read_buffer::<buffers::AtomicValues>(BufferName::AtomicValues)
            .await
            .expect("Failed to read number_release_cells buffer")[0]
            .number_release_cells;
        Ok(number_release_cells)
    }

    pub async fn run_initialize_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
    ) -> Result<()> {
        let particle_buffer_size =
            number_release_particles as usize * std::mem::size_of::<Particle>();
        assert!(
            number_release_particles as u64 <= self.max_particles,
            "Number of particles {} exceeds the limit of {}. Consider reducing the number of particles or using a GPU with more memory.",
            number_release_particles,
            self.max_particles
        );
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        info!(
            "Initializing particles with number_release_particles: {}, particle_buffer_size: {:.2} MB ({:.1} % of max storage buffer binding size)",
            number_release_particles,
            particle_buffer_size as f64 / 1024.0 / 1024.0,
            (particle_buffer_size as f64 / self.max_storage_buffer_binding_size as f64) * 100.0
        );
        self.dispatch_number_workgroups_1d =
            number_release_particles.div_ceil(self.max_compute_invocations_per_workgroup);
        debug!(
            "Running initialize particles shader with number_release_particles: {}, dispatch_number_workgroups_1d: {}",
            number_release_particles, self.dispatch_number_workgroups_1d
        );
        self.resources.add_buffer_with_data(
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
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let release_volume: u32 = self
            .read_buffer::<u32>(BufferName::AtomicValues)
            .await
            .expect("Failed to read release volume buffer")[4];
        info!("Estimated release volume: {}", release_volume);
        Ok(())
    }

    pub async fn run_compute_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        debug!("Start simulation");
        self.add_buffer(
            BufferName::TimestepData,
            size_of::<TimestepDataAoS>() * sim_settings.max_steps as usize * 3,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let sim_info: SimInfo = SimInfo {
            timestep: 1,
            number_particles: number_release_particles,
            // estimated timestep for a 60 degree slope
            dt: (2.0 * sim_settings.cfl * sim_settings.cell_size / (9.81 * 0.866) as f32).sqrt(),
            elevation_threshold: minimum_dem_elevation - 0.1,
            ..Default::default()
        };
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimInfo,
            bytemuck::bytes_of(&sim_info),
        )?;

        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;

        let update_sim_info_config = self
            .shader_configs
            .get(&ShaderName::UpdateSimInfo)
            .expect("UpdateSimInfo shader config not found");

        let update_sim_info_bindgroup =
            update_sim_info_config.create_bind_group(&self.device, &self.resources)?;

        let p2g_config = self
            .shader_configs
            .get(&ShaderName::P2G)
            .expect("P2G shader config not found");

        let p2g_bindgroup = p2g_config.create_bind_group(&self.device, &self.resources)?;

        let grid_physics_config = self
            .shader_configs
            .get(&ShaderName::GridPhysics)
            .expect("GridPhysics shader config not found");

        let grid_physics_bindgroup =
            grid_physics_config.create_bind_group(&self.device, &self.resources)?;

        // Compute Particles Bind Group
        let compute_particles_config = self
            .shader_configs
            .get(&ShaderName::ComputeParticles)
            .expect("ComputeParticles shader config not found");

        let compute_particles_bindgroup =
            compute_particles_config.create_bind_group(&self.device, &self.resources)?;

        let reset_grid_config = self
            .shader_configs
            .get(&ShaderName::ResetGrid)
            .expect("ResetGrid shader config not found");

        // Reset Grid Bind Group
        let reset_grid_bind_group =
            reset_grid_config.create_bind_group(&self.device, &self.resources)?;

        let mut current_step = 0;
        let mut atomic_values = AtomicValues::default();
        while current_step < sim_settings.max_steps {
            // Determine how many steps to run in this specific hardware batch
            let steps_to_run = std::cmp::min(
                self.batch_compute_steps,
                sim_settings.max_steps - current_step,
            );

            // 1. Create a fresh command encoder for this batch
            let mut command_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!(
                            "Compute Particles Compute Encoder - Batch Starting Step {}",
                            current_step
                        )),
                    });

            // 2. Open the compute pass and run the sub-steps
            {
                let mut compute_pass =
                    command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Particles Compute Pass Batch"),
                        timestamp_writes: None,
                    });

                for _i in 0..steps_to_run {
                    if SimFlags::from_u32(sim_settings.flags).is_particle_interaction_enabled() {
                        // --- P2G ---
                        compute_pass.set_pipeline(&p2g_config.pipeline);
                        compute_pass.set_bind_group(0, &p2g_bindgroup, &[]);
                        compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                        // --- Grid Physics ---
                        compute_pass.set_pipeline(&grid_physics_config.pipeline);
                        compute_pass.set_bind_group(0, &grid_physics_bindgroup, &[]);
                        compute_pass.dispatch_workgroups(
                            self.dispatch_number_workgroups_x_2d,
                            self.dispatch_number_workgroups_y_2d,
                            1,
                        );
                    }

                    // --- computeParticles ---
                    compute_pass.set_pipeline(&compute_particles_config.pipeline);
                    compute_pass.set_bind_group(0, &compute_particles_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                    // --- updateSimInfo ---
                    compute_pass.set_pipeline(&update_sim_info_config.pipeline);
                    compute_pass.set_bind_group(0, &update_sim_info_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(1, 1, 1);

                    // --- resetGrid ---
                    compute_pass.set_pipeline(&reset_grid_config.pipeline);
                    compute_pass.set_bind_group(0, &reset_grid_bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        self.dispatch_number_workgroups_x_2d,
                        self.dispatch_number_workgroups_y_2d,
                        1,
                    );
                }
            }

            // 3. Submit the batch to execution right now
            self.queue.submit(Some(command_encoder.finish()));
            current_step += steps_to_run;

            atomic_values = self
                .read_buffer::<AtomicValues>(BufferName::AtomicValues)
                .await
                .expect("Failed to read AtomicValues buffer")[0];
            if atomic_values.stopped_particles == number_release_particles {
                info!(
                    "Simulation finished early at step {} as all particles have stopped!",
                    current_step
                );
                break;
            }
            // else {
            //     trace!(
            //         "Step {}. Time: {:.4}, dt: {:.4}, Max velocity: {:.4}, Max flow thickness: {:.4}, stopped particles: {}, total particles: {}",
            //         current_step, sim_info.elapsed_time, sim_info.dt, sim_info.max_velocity, sim_info.max_flow_thickness, atomic_values.stopped_particles, number_release_particles
            //     );
            // }
        }
        let sim_info = self
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await
            .expect("Failed to read SimInfo buffer")[0];
        info!("{:#?}", sim_info);
        info!("{:#?}", atomic_values);
        if atomic_values.stopped_particles != number_release_particles {
            warn!(
                "Simulation reached max steps without all particles stopping. Consider increasing max_steps or checking for issues in the simulation."
            );
        }
        Ok(())
    }
    pub async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        self.resources
            .read_texture(&self.device, &self.queue, name)
            .await
    }
    pub fn write_texture<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        name: TextureName,
        data: &[T],
    ) -> Result<()> {
        self.resources.write_texture::<T>(&self.queue, name, data)
    }
    pub async fn read_texture_single_channel<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<Vec<T>> {
        self.resources
            .read_texture_single_channel(&self.device, &self.queue, name)
            .await
    }
    pub async fn read_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: BufferName,
    ) -> Result<Vec<T>> {
        self.resources
            .read_buffer(&self.device, &self.queue, name)
            .await
    }

    pub fn add_buffer(&mut self, name: BufferName, size_bytes: usize, usage: BufferUsages) {
        self.resources
            .add_buffer(&self.device, name, size_bytes, usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;
    use std::mem;

    #[test_log::test]
    fn test_shader_report_generation() {
        let orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
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
        assert_eq!(p1.velocity, p2.velocity);
        assert_eq!(p1.stopped, p2.stopped);
    }

    #[test]
    fn test_particle_memory_layout() {
        assert_eq!(mem::size_of::<Particle>(), 48);
    }

    #[test]
    fn test_field_offsets() {
        // Optional: Verifies that fields are where you think they are.
        // WebGPU expects 'c' (the matrix) to start at byte 32 in this layout.
        let p = Particle::new();
        let base_ptr = &p as *const _ as usize;
        let stopped_ptr = &p.stopped as *const _ as usize;

        let offset_c = stopped_ptr - base_ptr;
        assert_eq!(
            offset_c, 28,
            "Field 'c' is not at the expected 28-byte offset!"
        );
    }
}
