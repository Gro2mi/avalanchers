use crate::buffers::{
    AtomicValues, BufferName, GpuResources, TextureName, create_buffers_and_texture_descriptions,
};
use crate::shaders::{ComputeShaderConfig, ShaderName, generate_shader_report};
use crate::utils::timer_checkpoint;
use anyhow::{Result, anyhow};
use std::cmp::min;
use std::collections::HashMap;
use std::hash::Hash;
use wgpu::{
    Adapter, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    DeviceDescriptor, Extent3d, Features, Instance, Limits, Queue, TextureFormat, TextureUsages,
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

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SimInfoFlags: u32 {
        const OUT_OF_BOUNDS          = 1 << 0;
        const CFL_EXCEEDED           = 1 << 1;
        const IS_NAN                 = 1 << 2;
        const PARTICLE_OUT_OF_DEM    = 1 << 3;
        const STOPPED                = 1 << 31;
        const PARTICLES_STOPPED = 1 << 30;
        const NO_NEW_CELLS  = 1 << 29;
    }
}

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
    pub particles_position: Option<Vec<[f32; 2]>>,
    pub particles_mass: Option<Vec<f32>>,
    pub particles_velocity: Option<Vec<[f32; 2]>>,
    pub particles_stopped: Option<Vec<u32>>,
    pub particles_elevation: Option<Vec<f32>>,
    pub peak_velocity: Option<Vec<f32>>,
    pub peak_flow_thickness: Option<Vec<f32>>,
    pub terrain_geometry: Option<TextureRgba<f32>>,
    pub curvature: Option<TextureRgba<f32>>,
    pub slope_angle: Option<Vec<f32>>,
    pub slope_aspect: Option<Vec<f32>>,
    pub roughness: Option<Vec<f32>>,
    pub release_areas: Option<Vec<f32>>,
    pub timestep_data: Option<TimestepData>,
    pub read_count: usize,
}

impl GpuCache {
    pub fn reset_simulation_result(&mut self) {
        self.particles_position = None;
        self.particles_mass = None;
        self.particles_velocity = None;
        self.particles_stopped = None;
        self.particles_elevation = None;
        self.peak_velocity = None;
        self.timestep_data = None;
        self.peak_flow_thickness = None;
    }

    pub fn reset_all(&mut self) {
        self.reset_simulation_result();
        self.terrain_geometry = None;
        self.curvature = None;
        self.slope_angle = None;
        self.slope_aspect = None;
        self.roughness = None;
        self.release_areas = None;
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
    pub position: [f32; 3], // 12 bytes
    pub uv: [f32; 2],       // 8 bytes
    pub dt: f32,            // 4 bytes
    pub _pad1: [f32; 1],    // 4 bytes (padding to 32 bytes)
}

#[derive(Clone)]
pub struct TimestepData {
    pub velocity: Vec<[f32; 3]>,
    pub position: Vec<[f32; 3]>,
    pub dt: Vec<f32>,
    pub uv: Vec<[f32; 2]>,
    pub velocity_magnitude: Vec<f32>,
    pub time: Vec<f32>,
    pub step_distance2d: Vec<f32>,
    pub travel_distance2d: Vec<f32>,
    pub cfl: Vec<f32>,
}

impl TimestepData {
    pub fn from_aos(aos_data: &[TimestepDataAoS], cell_size: f32) -> Self {
        let len = aos_data.len();
        // Pre-allocate all vectors to the exact required size
        let mut soa = Self {
            velocity: Vec::with_capacity(len),
            dt: Vec::with_capacity(len),
            position: Vec::with_capacity(len),
            uv: Vec::with_capacity(len),
            velocity_magnitude: Vec::with_capacity(len),
            time: Vec::with_capacity(len),
            step_distance2d: Vec::with_capacity(len),
            travel_distance2d: Vec::with_capacity(len),
            cfl: Vec::with_capacity(len),
        };

        for item in aos_data {
            let velocity_magnitude = magnitude(&item.velocity);
            if velocity_magnitude < 1e-5 {
                break;
            }
            soa.velocity_magnitude.push(velocity_magnitude);
            soa.velocity.push(item.velocity);
            soa.dt.push(item.dt);
            soa.position.push(item.position);
            soa.uv.push(item.uv);
        }
        // first time step
        soa.time.push(0.0);
        soa.step_distance2d.push(0.0);
        soa.travel_distance2d.push(0.0);

        soa.cfl.push(0.0);

        for n in 1..soa.position.len() {
            let prev_pos = soa.position[n - 1];
            let curr_pos = soa.position[n];

            let dist = magnitude_diff(&curr_pos, &prev_pos);

            soa.time.push(soa.time[n - 1] + soa.dt[n]);
            soa.step_distance2d.push(dist);
            soa.travel_distance2d
                .push(soa.travel_distance2d[n - 1] + dist);
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
        // VULKAN is much faster to request
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let (instance, adapter) = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => (instance, adapter),
            Err(_) => {
                let fallback_instance =
                    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

                let adapter = match fallback_instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    Ok(adapter) => adapter,
                    Err(_) => match fallback_instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::LowPower,
                            compatible_surface: None,
                            force_fallback_adapter: false,
                        })
                        .await
                    {
                        Ok(adapter) => adapter,
                        Err(_) => fallback_instance
                            .request_adapter(&wgpu::RequestAdapterOptions {
                                power_preference: wgpu::PowerPreference::LowPower,
                                compatible_surface: None,
                                force_fallback_adapter: true,
                            })
                            .await
                            .expect("Failed to request adapter"),
                    },
                };

                (fallback_instance, adapter)
            }
        };
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
        let bytes_per_particle = 2 * std::mem::size_of::<[f32; 2]>() // position + velocity
             + 2 * std::mem::size_of::<f32>() // mass + elevation
             + std::mem::size_of::<u32>(); // stopped
        let buffer_limit = max_storage_buffer_binding_size / bytes_per_particle as u64;
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

        let mut required_features = Features::empty() | Features::SHADER_FLOAT32_ATOMIC;
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
                    max_storage_buffers_per_shader_stage: 13,
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
    fn generate_shader_report(
        &self,
        filename: Option<&str>,
        custom_order: &[ShaderName],
    ) -> String {
        generate_shader_report(filename, &self.shader_configs, Some(custom_order))
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
        match sim_settings.sim_model {
            0 => self.run_shader(
                &ShaderName::AnalyzeTerrain,
                self.dispatch_number_workgroups_x_2d,
                self.dispatch_number_workgroups_y_2d,
                1,
            ),
            1 => self.run_shader(
                &ShaderName::AnalyzeTerrainCurvilinear,
                self.dispatch_number_workgroups_x_2d,
                self.dispatch_number_workgroups_y_2d,
                1,
            ),
            2_u32..=u32::MAX => todo!(),
        }
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

    pub async fn run_initialize_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
    ) -> Result<()> {
        let particle_buffer_size_single_value = number_release_particles as usize * 4;
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
            "Initializing particles with number_release_particles: {}, particle_position_buffer_size: {:.2} MB ({:.1} % of max storage buffer binding size)",
            number_release_particles,
            particle_buffer_size_single_value as f64 * 2.0 / 1024.0 / 1024.0,
            (particle_buffer_size_single_value as f64 * 2.0
                / self.max_storage_buffer_binding_size as f64)
                * 100.0
        );
        self.dispatch_number_workgroups_1d =
            number_release_particles.div_ceil(self.max_compute_invocations_per_workgroup);
        debug!(
            "Running initialize particles shader with number_release_particles: {}, dispatch_number_workgroups_1d: {}",
            number_release_particles, self.dispatch_number_workgroups_1d
        );
        self.add_buffer_with_data(
            BufferName::ParticleIndex,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        self.add_buffer(
            BufferName::ParticlesPosition,
            particle_buffer_size_single_value * 2,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        self.add_buffer(
            BufferName::ParticlesMass,
            particle_buffer_size_single_value,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        self.add_buffer(
            BufferName::ParticlesElevation,
            particle_buffer_size_single_value,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let init_particles_velocity = vec![0f32; number_release_particles as usize * 2];
        self.add_buffer_with_data(
            BufferName::ParticlesVelocity,
            &init_particles_velocity,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let init_particles_stopped = vec![0u32; number_release_particles as usize];
        self.add_buffer_with_data(
            BufferName::ParticlesStopped,
            &init_particles_stopped,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        match sim_settings.sim_model {
            0 => {
                self.add_buffer(
                    BufferName::ParticlesVelocityZ,
                    particle_buffer_size_single_value,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                );
                self.add_buffer(
                    BufferName::GridForces,
                    particle_buffer_size_single_value * 2,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                );
            }
            1 => {
                self.add_buffer(
                    BufferName::ParticlesAffineMatrix,
                    particle_buffer_size_single_value * 4,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                );
            }
            2_u32..=u32::MAX => todo!(),
        }

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

        let mut sim_info: SimInfo = SimInfo {
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
                    // --- resetGrid ---
                    compute_pass.set_pipeline(&reset_grid_config.pipeline);
                    compute_pass.set_bind_group(0, &reset_grid_bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        self.dispatch_number_workgroups_x_2d,
                        self.dispatch_number_workgroups_y_2d,
                        1,
                    );
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

                    // --- computeParticles ---
                    compute_pass.set_pipeline(&compute_particles_config.pipeline);
                    compute_pass.set_bind_group(0, &compute_particles_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                    // --- updateSimInfo ---
                    compute_pass.set_pipeline(&update_sim_info_config.pipeline);
                    compute_pass.set_bind_group(0, &update_sim_info_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(1, 1, 1);
                }
            }

            // 3. Submit the batch to execution right now
            self.queue.submit(Some(command_encoder.finish()));
            current_step += steps_to_run;

            sim_info = self
                .read_buffer::<SimInfo>(BufferName::SimInfo)
                .await
                .expect("Failed to read SimInfo buffer")[0];
            // info!("{:#?},", sim_info);
            let flags = SimInfoFlags::from_bits_retain(sim_info.flags);
            if flags.contains(SimInfoFlags::STOPPED) {
                let reason = match flags {
                    _ if flags.contains(SimInfoFlags::PARTICLES_STOPPED) => {
                        "all particles have stopped moving"
                    }
                    _ if flags.contains(SimInfoFlags::NO_NEW_CELLS) => {
                        "no new cells were conquered by particles"
                    }
                    _ => "unknown reason",
                };
                info!(
                    "Simulation finished at step {} because {}.",
                    sim_info.timestep, reason
                );

                break;
            }
            if flags.contains(SimInfoFlags::NO_NEW_CELLS) {
                info!(
                    "Simulation finished early at step {} as no new cells were conquered by particles!",
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
        // info!("{:?}", new_cells);

        info!(
            "New cells conquered in the last 100 steps: {:?}",
            self.read_buffer::<u32>(BufferName::NewCellsRollingWindow)
                .await
                .expect("Failed to read AtomicValues buffer")
        );
        let atomic_values = self
            .read_buffer::<AtomicValues>(BufferName::AtomicValues)
            .await
            .expect("Failed to read AtomicValues buffer")[0];
        info!("{:#?}", sim_info);
        info!("{:#?}", atomic_values);
        if sim_info.flags < SimInfoFlags::PARTICLES_STOPPED.bits() {
            warn!(
                "Simulation reached max steps without all particles stopping. Consider increasing max_steps or checking for issues in the simulation."
            );
        }
        Ok(())
    }

    pub async fn run_sim(
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
        match sim_settings.sim_model {
            0 => {
                self.run_compute_particles(
                    sim_settings,
                    number_release_particles,
                    minimum_dem_elevation,
                )
                .await?
            }
            1 => self.run_mpm(sim_settings).await?,
            2_u32..=u32::MAX => todo!(),
        }

        // info!("{:?}", new_cells);

        info!(
            "New cells conquered in the last 100 steps: {:?}",
            self.read_buffer::<u32>(BufferName::NewCellsRollingWindow)
                .await
                .expect("Failed to read AtomicValues buffer")
        );
        let atomic_values = self
            .read_buffer::<AtomicValues>(BufferName::AtomicValues)
            .await
            .expect("Failed to read AtomicValues buffer")[0];
        let sim_info = self
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await
            .expect("Failed to read SimInfo buffer")[0];
        info!("{:#?}", sim_info);
        info!("{:#?}", atomic_values);
        if sim_info.flags < SimInfoFlags::PARTICLES_STOPPED.bits() {
            warn!(
                "Simulation reached max steps without all particles stopping. Consider increasing max_steps or checking for issues in the simulation."
            );
        }
        Ok(())
    }

    pub async fn run_mpm(&mut self, sim_settings: &settings::SimSettings) -> Result<()> {
        let update_sim_info_config = self
            .shader_configs
            .get(&ShaderName::UpdateSimInfo)
            .expect("UpdateSimInfo shader config not found");

        let update_sim_info_bindgroup =
            update_sim_info_config.create_bind_group(&self.device, &self.resources)?;

        let p2g_config = self
            .shader_configs
            .get(&ShaderName::P2GMPM)
            .expect("P2G shader config not found");

        let p2g_bindgroup = p2g_config.create_bind_group(&self.device, &self.resources)?;

        let grid_physics_config = self
            .shader_configs
            .get(&ShaderName::GridPhysicsMPM)
            .expect("GridPhysics shader config not found");

        let grid_physics_bindgroup =
            grid_physics_config.create_bind_group(&self.device, &self.resources)?;

        // Compute Particles Bind Group
        let g2p_config = self
            .shader_configs
            .get(&ShaderName::G2P)
            .expect("G2P shader config not found");

        let g2p_bindgroup = g2p_config.create_bind_group(&self.device, &self.resources)?;

        let reset_grid_config = self
            .shader_configs
            .get(&ShaderName::ResetGrid)
            .expect("ResetGrid shader config not found");

        // Reset Grid Bind Group
        let reset_grid_bind_group =
            reset_grid_config.create_bind_group(&self.device, &self.resources)?;
        let mut current_step = 0;
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
                    // --- resetGrid ---
                    compute_pass.set_pipeline(&reset_grid_config.pipeline);
                    compute_pass.set_bind_group(0, &reset_grid_bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        self.dispatch_number_workgroups_x_2d,
                        self.dispatch_number_workgroups_y_2d,
                        1,
                    );
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

                    // --- computeParticles ---
                    compute_pass.set_pipeline(&g2p_config.pipeline);
                    compute_pass.set_bind_group(0, &g2p_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                    // --- updateSimInfo ---
                    compute_pass.set_pipeline(&update_sim_info_config.pipeline);
                    compute_pass.set_bind_group(0, &update_sim_info_bindgroup, &[]);
                    compute_pass.dispatch_workgroups(1, 1, 1);
                }
            }

            // 3. Submit the batch to execution right now
            self.queue.submit(Some(command_encoder.finish()));
            current_step += steps_to_run;

            let sim_info = self
                .read_buffer::<SimInfo>(BufferName::SimInfo)
                .await
                .expect("Failed to read SimInfo buffer")[0];
            // info!("{:#?},", sim_info);
            let flags = SimInfoFlags::from_bits_retain(sim_info.flags);
            let mut reason = String::new();
            if flags.contains(SimInfoFlags::STOPPED) {
                if flags.contains(SimInfoFlags::PARTICLES_STOPPED) {
                    reason.push_str("all particles have stopped moving");
                } else if flags.contains(SimInfoFlags::NO_NEW_CELLS) {
                    reason.push_str("no new cells were conquered by particles");
                } else {
                    reason.push_str("unknown reason");
                }
                info!(
                    "Simulation finished at step {} because {}.",
                    sim_info.timestep, reason
                );
                break;
            }
            if flags.contains(SimInfoFlags::NO_NEW_CELLS) {
                info!(
                    "Simulation finished early at step {} as no new cells were conquered by particles!",
                    current_step
                );
                // break;
            }
            // else {
            //     trace!(
            //         "Step {}. Time: {:.4}, dt: {:.4}, Max velocity: {:.4}, Max flow thickness: {:.4}, stopped particles: {}, total particles: {}",
            //         current_step, sim_info.elapsed_time, sim_info.dt, sim_info.max_velocity, sim_info.max_flow_thickness, atomic_values.stopped_particles, number_release_particles
            //     );
            // }
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
    pub async fn write_buffer<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        name: BufferName,
        data: &[T],
    ) -> Result<()> {
        self.resources.write_buffer(&self.queue, name, data)
    }

    pub fn add_buffer(&mut self, name: BufferName, size_bytes: usize, usage: BufferUsages) {
        self.resources
            .add_buffer(&self.device, name, size_bytes, usage);
    }

    pub fn add_buffer_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        name: BufferName,
        data: &[T],
        usage: BufferUsages,
    ) {
        self.resources
            .add_buffer_with_data(&self.device, name, data, usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use pollster::block_on;

    #[test_log::test]
    fn test_shader_report_generation_sim_model_0() {
        let orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        orchestrator.generate_shader_report(
            Some("shader_report_sim_model_0.html"),
            &[
                ShaderName::AnalyzeTerrain,
                ShaderName::ComputeRoughness,
                ShaderName::ComputeReleaseAreas,
                ShaderName::InitializeParticles,
                ShaderName::ResetGrid,
                ShaderName::P2G,
                ShaderName::GridPhysics,
                ShaderName::ComputeParticles,
                ShaderName::UpdateSimInfo,
            ],
        );
    }

    #[test_log::test]
    fn test_shader_report_generation_sim_model_1() {
        let orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        orchestrator.generate_shader_report(
            Some("shader_report_sim_model_1.html"),
            &[
                ShaderName::AnalyzeTerrainCurvilinear,
                ShaderName::ComputeRoughness,
                ShaderName::ComputeReleaseAreas,
                ShaderName::InitializeParticles,
                ShaderName::ResetGrid,
                ShaderName::P2GMPM,
                ShaderName::GridPhysicsMPM,
                ShaderName::G2P,
                ShaderName::UpdateSimInfo,
            ],
        );
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
    #[test_log::test]
    fn test_shader_transforms() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
        struct CellTestData {
            idx_from_uv: u32,
            idx_from_xy: u32,
            idx_from_x_y: u32,
            error_code: i32, // -1 = Pass, 0 = didnt run, 1 = Index Flatten, 2 = Round-trip, 3 = Position

            cell_x: u32,
            cell_y: u32,
            rt_cell_x: u32,
            rt_cell_y: u32,

            mock_pos_x: f32,
            mock_pos_y: f32,
            computed_idx: u32,
            expected_idx: u32,
        }
        let mut orchestrator: ComputeOrchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let sim_settings = settings::SimSettings {
            grid_shape_x: 62,
            grid_shape_y: 66,
            world_size_x: 310.0,
            world_size_y: 330.0,
            cell_size: 5.0,
            ..Default::default()
        };
        orchestrator.add_buffer_with_data(
            BufferName::SimSettings,
            sim_settings.as_bytes(),
            BufferUsages::UNIFORM,
        );
        orchestrator.add_buffer(
            BufferName::TestOutput,
            std::mem::size_of::<CellTestData>()
                * sim_settings.grid_shape_x as usize
                * sim_settings.grid_shape_y as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        block_on(orchestrator.run_shader(&ShaderName::TestTransforms, 4, 5, 1)).expect("msg");

        let test_output =
            block_on(orchestrator.read_buffer::<CellTestData>(BufferName::TestOutput))
                .expect("msg");
        info!("Test output length: {}", test_output.len());
        assert_eq!(
            test_output.len(),
            sim_settings.grid_shape_x as usize * sim_settings.grid_shape_y as usize
        );
        for (i, cell_data) in test_output.iter().enumerate() {
            assert_eq!(
                -1, cell_data.error_code,
                "Error in cell index {}: {:?}",
                i, cell_data
            );
        }
        info!("{:#?}", test_output.iter().take(10).collect::<Vec<_>>());
    }
    #[test_log::test]
    fn test_shader_utils() {
        let mut orchestrator: ComputeOrchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let sim_settings = settings::SimSettings {
            ..Default::default()
        };
        orchestrator.add_buffer_with_data(
            BufferName::SimSettings,
            sim_settings.as_bytes(),
            BufferUsages::UNIFORM,
        );
        orchestrator.add_buffer(
            BufferName::TestOutput,
            20 * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        orchestrator.add_buffer(
            BufferName::AtomicValues,
            ((size_of::<AtomicValues>() - 1) / 16 + 1) * 16,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        block_on(orchestrator.run_shader(&ShaderName::TestUtils, 4, 5, 1)).expect("msg");

        let test_output =
            block_on(orchestrator.read_buffer::<f32>(BufferName::TestOutput)).expect("msg");
        info!("Test output length: {}", test_output.len());
        info!("{:#?}", test_output.iter().take(10).collect::<Vec<_>>());
        assert_eq!(test_output[0], 3.1415);
        assert_eq!(test_output[1], 42.0);
        assert_eq!(test_output[2], 42.0);

        // is_nan
        assert_eq!(test_output[3], 1.0);
        assert_eq!(test_output[4], 1.0);
        assert_eq!(test_output[5], 0.0);
        assert_eq!(test_output[6], 0.0);
        assert_eq!(test_output[7], 0.0);
        // is_inf
        assert_eq!(test_output[8], 1.0);
        assert_eq!(test_output[9], 1.0);
        assert_eq!(test_output[10], 0.0);
        assert_eq!(test_output[11], 0.0);
        assert_eq!(test_output[12], 0.0);
        // is_finite
        assert_eq!(test_output[13], 0.0);
        assert_eq!(test_output[14], 0.0);
        assert_eq!(test_output[15], 0.0);
        assert_eq!(test_output[16], 0.0);
        assert_eq!(test_output[17], 1.0);

        let atomic_values =
            block_on(orchestrator.read_buffer::<AtomicValues>(BufferName::AtomicValues))
                .expect("msg");
        info!("Atomic values: {:#?}", atomic_values);
        assert_eq!(atomic_values[0].grid_peak_flow_thickness, 2.71828);
        assert_eq!(atomic_values[0].expected_max_velocity, 1.618);
        assert_eq!(atomic_values[0].grid_peak_velocity, 1.4142);
        assert_eq!(atomic_values[0].travel_length, 1.732);
        assert_eq!(atomic_values[0].estimated_release_volume, 73);
        assert_eq!(atomic_values[0].number_release_cells, 37);
        assert_eq!(atomic_values[0].number_release_particles, 42);
        assert_eq!(atomic_values[0].stopped_particles, 99);
    }
    #[test_log::test]
    fn test_shader_sampling() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
        struct TestSamplingOutput {
            position_x: f32,
            position_y: f32,
            u: f32,
            v: f32,
            cell_x: u32,
            cell_y: u32,

            // Continuous sampled data (Filtered via sampler)
            dem_sampled: f32,
            dem_sampled_as_expected: i32,

            // Exact cell data (Unfiltered via textureLoad)
            dem_loaded: f32,
            dem_loaded_as_expected: i32,
        }
        let mut orchestrator: ComputeOrchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");

        let dem: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let cell_size = 5.0;
        let samples_per_cell = 100;
        let sim_settings = settings::SimSettings {
            grid_shape_x: dem.len() as u32,
            grid_shape_y: 1,
            world_size_x: cell_size * dem.len() as f32,
            world_size_y: cell_size,
            cell_size,
            sim_model: samples_per_cell, // for sampling steps per cell
            ..Default::default()
        };
        let steps = sim_settings.grid_shape_x * samples_per_cell as u32 + 1;
        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .unwrap();
        block_on(orchestrator.write_buffer(BufferName::SimSettings, sim_settings.as_bytes()))
            .expect("Failed to write simulation settings");

        orchestrator
            .resources
            .add_texture_with_data(
                &orchestrator.device,
                &orchestrator.queue,
                &dem,
                TextureName::Dem,
                Extent3d {
                    width: sim_settings.grid_shape_x,
                    height: sim_settings.grid_shape_y,
                    depth_or_array_layers: 1,
                },
                TextureFormat::R32Float,
                TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
            )
            .expect("Failed to add texture with data");

        orchestrator.add_buffer(
            BufferName::TestOutput,
            std::mem::size_of::<TestSamplingOutput>() * steps as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        block_on(orchestrator.run_shader(&ShaderName::TestSampling, steps.div_ceil(64), 1, 1))
            .expect("msg");

        let test_output =
            block_on(orchestrator.read_buffer::<TestSamplingOutput>(BufferName::TestOutput))
                .expect("msg");
        info!("Test output length: {}", test_output.len());
        assert_eq!(test_output.len(), steps as usize);
        info!(
            "{:<10} {:<10} {:<10} {:<10} {:<10}",
            "PositionX", "U", "Cell Idx", "Sampled", "Loaded"
        );
        let mut sampling_diffs = Vec::new();
        for (i, sampling_data) in test_output.iter().enumerate() {
            // assert_eq!(
            //     -1, cell_data.dem_sampled_as_expected,
            //     "Error in cell index {}: {:?}",
            //     i, cell_data
            // );
            if i % samples_per_cell as usize == 0 {
                info!("");
            }
            info!(
                "{:<10.2} {:<10.2} {:<10.2} {:<10.5} {:<10.2}",
                sampling_data.position_x,
                sampling_data.u,
                sampling_data.cell_x,
                sampling_data.dem_sampled,
                sampling_data.dem_loaded
            );
            if i >= (samples_per_cell / 2) as usize
                && i < (steps as usize - samples_per_cell as usize / 2)
            {
                let expected_elevation =
                    1 as f32 / samples_per_cell as f32 * (i as u32 - samples_per_cell / 2) as f32;
                let elevation_diff = (sampling_data.dem_sampled - expected_elevation).abs();
                sampling_diffs.push(elevation_diff);
                assert!(
                    elevation_diff < 5e-3,
                    "Unexpected sampled elevation at step {}: got {}, expected {}",
                    i,
                    sampling_data.dem_sampled,
                    expected_elevation
                );
            }
            if i < samples_per_cell as usize * dem.len() {
                assert_eq!(sampling_data.cell_x, i as u32 / samples_per_cell);
                assert_eq!(
                    sampling_data.dem_loaded,
                    (i as u32 / samples_per_cell) as f32
                );
            }
            assert!(
                (sampling_data.u - (i as f32 / samples_per_cell as f32 / dem.len() as f32)).abs()
                    < 1e-5
            );
        }

        info!(
            "Average elevation sampling error for elevation samples: {:.6}, min: {:.6}, max: {:.6}",
            sampling_diffs.iter().sum::<f32>() / sampling_diffs.len() as f32,
            sampling_diffs
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            sampling_diffs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        );
        // info!("{:#?}", test_output.iter().take(50).collect::<Vec<_>>());
    }

    #[test_log::test]
    fn test_shader_transfer() {
        let mut orchestrator: ComputeOrchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        // let position: [f32; 2] = [17.5, 22.5];
        let position: [f32; 2] = [17.1, 22.8];
        let velocity: [f32; 2] = [20.0, 30.0];
        let mass: f32 = 1000.0;
        let sim_settings = settings::SimSettings {
            grid_shape_x: 10,
            grid_shape_y: 10,
            world_size_x: 50.0,
            world_size_y: 50.0,
            cell_size: 5.0,
            ..Default::default()
        };
        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .unwrap();
        block_on(orchestrator.write_buffer(BufferName::SimSettings, sim_settings.as_bytes()))
            .expect("Failed to write simulation settings");
        orchestrator.add_buffer_with_data(
            BufferName::ParticlesPosition,
            bytemuck::bytes_of(&position),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        orchestrator.add_buffer_with_data(
            BufferName::ParticlesVelocity,
            bytemuck::bytes_of(&velocity),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        orchestrator.add_buffer_with_data(
            BufferName::ParticlesMass,
            bytemuck::bytes_of(&mass),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        orchestrator.add_buffer_with_data(
            BufferName::ParticlesAffineMatrix,
            bytemuck::bytes_of(&mass),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        orchestrator.add_buffer(
            BufferName::TestOutput,
            400 as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        block_on(orchestrator.run_shader(
            &ShaderName::TestTransfer,
            sim_settings.grid_shape_x.div_ceil(16),
            1,
            1,
        ))
        .expect("Failed to run transfer shader");
        let test_output =
            block_on(orchestrator.read_buffer::<f32>(BufferName::TestOutput)).expect("msg");
        info!("Test output length: {}", test_output.len());
        info!("{:#?}", test_output.iter().take(40).collect::<Vec<_>>());
        let mass_p2g: f32 = test_output.iter().skip(5).take(10).sum();
        info!(
            "Mass before: {} after p2g: {} relative error: {}",
            mass,
            mass_p2g,
            (mass_p2g - mass).abs() / mass
        );
        assert_eq!(mass, mass_p2g, "Mass transfer from particle to grid failed");
        let velocity_x: f32 = test_output[18];
        let velocity_y: f32 = test_output[19];
        let momentum: f32 = mass * (velocity_x * velocity_x + velocity_y * velocity_y).sqrt();
        let momentum_start = mass * (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        info!(
            "Velocity before: ({}, {}), after p2g: ({}, {}), momentum before: {}, after: {}, relative error: {}",
            velocity[0],
            velocity[1],
            velocity_x,
            velocity_y,
            momentum_start,
            momentum,
            (momentum - momentum_start).abs() / momentum_start
        );
        assert!(
            (momentum_start - momentum).abs() < 1e-2,
            "Momentum transfer from particle to grid failed"
        );
        assert!(
            (velocity[0] - velocity_x).abs() < 1e-4,
            "Velocity X transfer from particle to grid failed"
        );
        assert!(
            (velocity[1] - velocity_y).abs() < 1e-4,
            "Velocity Y transfer from particle to grid failed"
        );
        info!(
            "Mass Grid:\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}",
            test_output[5],
            test_output[6],
            test_output[7],
            test_output[8],
            test_output[9],
            test_output[10],
            test_output[11],
            test_output[12],
            test_output[13],
        );
        // expected mass distribution
        assert!(
            (test_output[5] - 16.28176).abs() < 1e-4,
            "test_output[5] failed"
        );
        assert!(
            (test_output[6] - 71.9805).abs() < 1e-4,
            "test_output[6] failed"
        );
        assert!(
            (test_output[7] - 8.537765).abs() < 1e-4,
            "test_output[7] failed"
        );
        assert!(
            (test_output[8] - 125.54444).abs() < 1e-4,
            "test_output[8] failed"
        );
        assert!(
            (test_output[9] - 555.0231).abs() < 1e-4,
            "test_output[9] failed"
        );
        assert!(
            (test_output[10] - 65.832504).abs() < 1e-4,
            "test_output[10] failed"
        );
        assert!(
            (test_output[11] - 26.373747).abs() < 1e-4,
            "test_output[11] failed"
        );
        assert!(
            (test_output[12] - 116.59646).abs() < 1e-4,
            "test_output[12] failed"
        );
        assert!(
            (test_output[13] - 13.829763).abs() < 1e-4,
            "test_output[13] failed"
        );
        assert_eq!(
            test_output[14], 0.0,
            "Particle influence outside of 3x3 grid should be zero"
        );
        assert_eq!(
            test_output[15], 0.0,
            "Particle influence outside of 3x3 grid should be zero"
        );

        assert_eq!(
            test_output[16] as u32,
            (position[0] as u32 / sim_settings.cell_size as u32) - 1,
            "Base node (lower-left) x coordinate wrong"
        );
        assert_eq!(
            test_output[17] as u32,
            (position[1] as u32 / sim_settings.cell_size as u32) - 1,
            "Base node (lower-left) y coordinate wrong"
        );

        info!(
            "Momentum Grid X:\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}",
            test_output[20],
            test_output[21],
            test_output[22],
            test_output[23],
            test_output[24],
            test_output[25],
            test_output[26],
            test_output[27],
            test_output[28],
        );

        // expected momentum distribution (grid x-momentum)
        assert!(
            (test_output[20] - 325.6352).abs() < 1e-4,
            "test_output[20] failed"
        );
        assert!(
            (test_output[21] - 1439.61).abs() < 1e-4,
            "test_output[21] failed"
        );
        assert!(
            (test_output[22] - 170.7553).abs() < 1e-4,
            "test_output[22] failed"
        );
        assert!(
            (test_output[23] - 2510.889).abs() < 1e-4,
            "test_output[23] failed"
        );
        assert!(
            (test_output[24] - 11100.462).abs() < 1e-4,
            "test_output[24] failed"
        );
        assert!(
            (test_output[25] - 1316.65).abs() < 1e-4,
            "test_output[25] failed"
        );
        assert!(
            (test_output[26] - 527.475).abs() < 1e-4,
            "test_output[26] failed"
        );
        assert!(
            (test_output[27] - 2331.9292).abs() < 1e-4,
            "test_output[27] failed"
        );
        assert!(
            (test_output[28] - 276.59528).abs() < 1e-4,
            "test_output[28] failed"
        );

        info!(
            "Momentum Grid Y:\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}\n {:8.2?} {:8.2?} {:8.2?}",
            test_output[30],
            test_output[31],
            test_output[32],
            test_output[33],
            test_output[34],
            test_output[35],
            test_output[36],
            test_output[37],
            test_output[38],
        );
        // expected momentum distribution (grid y-momentum)
        assert!(
            (test_output[30] - 488.4528).abs() < 1e-4,
            "test_output[30] failed"
        );
        assert!(
            (test_output[31] - 2159.415).abs() < 1e-4,
            "test_output[31] failed"
        );
        assert!(
            (test_output[32] - 256.13293).abs() < 1e-4,
            "test_output[32] failed"
        );
        assert!(
            (test_output[33] - 3766.3333).abs() < 1e-4,
            "test_output[33] failed"
        );
        assert!(
            (test_output[34] - 16650.691).abs() < 1e-4,
            "test_output[34] failed"
        );
        assert!(
            (test_output[35] - 1974.9751).abs() < 1e-4,
            "test_output[35] failed"
        );
        assert!(
            (test_output[36] - 791.2124).abs() < 1e-4,
            "test_output[36] failed"
        );
        assert!(
            (test_output[37] - 3497.8938).abs() < 1e-4,
            "test_output[37] failed"
        );
        assert!(
            (test_output[38] - 414.89288).abs() < 1e-4,
            "test_output[38] failed"
        );
    }
}
