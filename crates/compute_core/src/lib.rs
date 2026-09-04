use crate::buffers::{
    AtomicValues, BufferName, GpuResources, TextureName, create_buffers_and_texture_descriptions,
};
use crate::shaders::{ComputeShaderConfig, ShaderName, generate_shader_report};
use crate::utils::timer_checkpoint;
use anyhow::{Context, Result, anyhow};
use evaluation::{MassMovementEvaluation, evaluation_from_counts};
use std::cmp::min;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem::size_of;
use wgpu::{
    Adapter, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    Queue, RequestAdapterOptions, TextureFormat, TextureUsages,
};

// use log::{debug, info, warn, error};
pub mod buffers;
pub mod dem;
pub mod evaluation;
pub mod post_processing;
pub mod settings;
pub mod shaders;
pub mod utils;
use dem::Dem;
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct SimInfoFlags: u32 {
        const OUT_OF_BOUNDS            = 1 << 0;
        const CFL_EXCEEDED             = 1 << 1;
        const IS_NAN                   = 1 << 2;
        const PARTICLE_OUT_OF_DEM_DATA = 1 << 3;
        const NO_NEW_CELLS             = 1 << 29;
        const ALL_PARTICLES_STOPPED    = 1 << 30;
        const SIM_STOPPED              = 1 << 31;
    }
}

impl From<u32> for SimInfoFlags {
    fn from(flags: u32) -> Self {
        Self::from_bits_retain(flags)
    }
}

impl std::fmt::Display for SimInfoFlags {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return formatter.write_str("NONE");
        }

        let mut separator = "";
        for (name, _) in self.iter_names() {
            write!(formatter, "{separator}{name}")?;
            separator = " | ";
        }

        let unknown = self.bits() & !Self::all().bits();
        if unknown != 0 {
            write!(formatter, "{separator}UNKNOWN({unknown:#010x})")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SimInfoFlags {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
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
    pub particles_velocity_z: Option<Vec<f32>>,
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
        self.particles_velocity_z = None;
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
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

impl SimInfo {
    pub fn parsed_flags(&self) -> SimInfoFlags {
        self.flags.into()
    }
}

impl std::fmt::Debug for SimInfo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SimInfo")
            .field("timestep", &self.timestep)
            .field("dt", &self.dt)
            .field("elapsed_time", &self.elapsed_time)
            .field("number_particles", &self.number_particles)
            .field("elevation_threshold", &self.elevation_threshold)
            .field("max_velocity", &self.max_velocity)
            .field("max_flow_thickness", &self.max_flow_thickness)
            .field("flags", &self.parsed_flags())
            .finish()
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimestepDataAoS {
    pub velocity: [f32; 3],                   // 12 bytes
    pub dt: f32,                              // 4 bytes
    pub acceleration_tangential: [f32; 3],    // 12 bytes
    pub acceleration_friction_magnitude: f32, // 4 bytes
    pub position: [f32; 3],                   // 12 bytes
    pub elevation: f32,                       // 4 bytes
    pub normal: [f32; 3],                     // 12 bytes
    pub g_eff: f32,                           // 4 bytes
    pub acceleration_normal: [f32; 3],        // 12 bytes
    pub _pad1: [f32; 1],                      // 4 bytes
    pub uv: [f32; 2],                         // 8 bytes
    pub _pad2: [f32; 2],                      // 8 bytes (padding to 96 bytes)
}

const _: () = assert!(std::mem::size_of::<TimestepDataAoS>() == 96);

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
    pub fn from_aos(aos_data: &[TimestepDataAoS], cell_size: f32, timesteps: usize) -> Self {
        // Pre-allocate all vectors to the exact required size
        let mut soa = Self {
            velocity: Vec::with_capacity(timesteps),
            dt: Vec::with_capacity(timesteps),
            position: Vec::with_capacity(timesteps),
            uv: Vec::with_capacity(timesteps),
            velocity_magnitude: Vec::with_capacity(timesteps),
            time: Vec::with_capacity(timesteps),
            step_distance2d: Vec::with_capacity(timesteps),
            travel_distance2d: Vec::with_capacity(timesteps),
            cfl: Vec::with_capacity(timesteps),
        };

        for item in aos_data {
            let velocity_magnitude = magnitude(&item.velocity);
            soa.velocity_magnitude.push(velocity_magnitude);
            soa.velocity.push(item.velocity);
            soa.dt.push(item.dt);
            soa.position.push(item.position);
            soa.uv.push(item.uv);
        }
        if soa.position.is_empty() {
            return soa;
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
            soa.cfl.push(if cell_size > 0.0 {
                soa.velocity_magnitude[n] * soa.dt[n] / cell_size
            } else {
                0.0
            });
        }

        soa
    }
}

fn magnitude(v: &[f32; 3]) -> f32 {
    (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt()
}

fn magnitude_diff(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

use std::collections::BTreeMap;
use wgpu::Backends;
pub async fn list_devices() -> Result<Vec<String>> {
    let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
    let adapters = instance.enumerate_adapters(Backends::PRIMARY).await;

    // Map to group details by device name:
    // (DeviceType, Vec<Backends>, Vec<u32> (Device IDs), Driver, DriverInfo, SubgroupMin, SubgroupMax, PerfRating)
    type DeviceDetails = (
        wgpu::DeviceType,
        Vec<wgpu::Backend>,
        Vec<u32>, // Device IDs
        String,
        String,
        u32,
        u32,
        u8, // Performance Rating
    );
    let mut device_map: BTreeMap<String, DeviceDetails> = BTreeMap::new();

    for adapter in adapters {
        let info = adapter.get_info();

        // Assign a performance rating based on the device type
        let perf_rating = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 4,   // Highest performance
            wgpu::DeviceType::IntegratedGpu => 3, // Medium-high performance
            wgpu::DeviceType::VirtualGpu => 2,    // Emulated/Cloud performance
            wgpu::DeviceType::Cpu => 1,           // Software rendering fallback
            wgpu::DeviceType::Other => 0,         // Unknown / lowest
        };

        let entry = device_map.entry(info.name.clone()).or_insert_with(|| {
            (
                info.device_type,
                Vec::new(),
                Vec::new(),
                info.driver.clone(),
                info.driver_info.clone(),
                info.subgroup_min_size,
                info.subgroup_max_size,
                perf_rating,
            )
        });

        // Keep the highest rating if the same device name appears across multiple backends
        if perf_rating > entry.7 {
            entry.7 = perf_rating;
            entry.0 = info.device_type;
        }

        if !entry.1.contains(&info.backend) {
            entry.1.push(info.backend);
        }

        // Collect unique Device IDs (useful for multi-backend / multi-GPU setups)
        if !entry.2.contains(&info.device) && info.device != 0 {
            entry.2.push(info.device);
        }
    }

    // Sort devices descending by performance rating (DiscreteGpu -> IntegratedGpu -> etc.)
    let mut sorted_devices: Vec<_> = device_map.into_iter().collect();
    sorted_devices.sort_by_key(|a| std::cmp::Reverse(a.1.7));

    let device_names = sorted_devices
        .into_iter()
        .map(
            |(
                name,
                (
                    device_type,
                    backends,
                    device_ids,
                    driver,
                    driver_info,
                    sub_min,
                    sub_max,
                    _perf_rating,
                ),
            )| {
                let backends_str = backends
                    .iter()
                    .map(|b| format!("{:?}", b))
                    .collect::<Vec<_>>()
                    .join(", ");

                let ids_str = device_ids
                    .iter()
                    .map(|id| format!("{:#06X}", id))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(
                    "{} | {:?} | IDs: [{}] | Backends: [{}] | Driver: {} {} | Subgroups: {}-{}",
                    name, device_type, ids_str, backends_str, driver, driver_info, sub_min, sub_max
                )
            },
        )
        .collect();

    Ok(device_names)
}

const WORKGROUP_SIZE_2D: u32 = 16;

fn ordered_u32_to_f32(ordered: u32) -> f32 {
    let bits = if ordered & 0x8000_0000 != 0 {
        ordered ^ 0x8000_0000
    } else {
        !ordered
    };
    f32::from_bits(bits)
}

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
    prepared_max_steps: Option<u32>,
    prepared_model: Option<u32>,
    has_float32_filterable: bool,
    has_float32_atomic: bool,
}

impl ComputeOrchestrator {
    pub async fn new() -> Result<Self> {
        Self::new_with_gpu(None).await
    }
    pub async fn new_with_gpu(target_gpu: Option<String>) -> Result<Self> {
        // first search for VULKAN backend to speed up GPU selection, see dev branch
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let mut selected_adapter = None;

        // 1. If a target string was provided, search by Device ID (if numeric) or Name
        if let Some(mut target) = target_gpu {
            target = target
                .split("|")
                .next()
                .unwrap_or(&target)
                .trim()
                .to_string();
            let adapters = instance.enumerate_adapters(Backends::all()).await;

            // Try parsing the target string as a number (supports both hex "0x1E84" and decimal)
            let target_id = Self::parse_numeric_id(&target);

            for adapter in adapters {
                let info = adapter.get_info();

                let matched = if let Some(id) = target_id {
                    // Match by PCI Device ID
                    info.device == id
                } else {
                    // Match by Name substring (case-insensitive)
                    info.name.to_lowercase().contains(&target.to_lowercase())
                };

                if matched {
                    info!(
                        "Found target GPU: {} [ID: {:#06X}, Vendor: {:#06X}] ({:?})",
                        info.name, info.device, info.vendor, info.device_type
                    );
                    selected_adapter = Some(adapter);
                    break;
                }
            }

            if selected_adapter.is_none() {
                warn!(
                    "Requested GPU '{}' not found, falling back to automatic selection.",
                    target
                );
            }
        }

        // 2. Fallback logic if no argument was given or the target wasn't found
        let adapter = if let Some(adapter) = selected_adapter {
            adapter
        } else {
            let mut fallback_adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await;

            if fallback_adapter.is_err() {
                warn!("High-performance GPU not found, falling back to LowPower/Software.");
                fallback_adapter = instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    })
                    .await;
            }

            if fallback_adapter.is_err() {
                warn!("Low-performance GPU not found, falling back to Software.");
                fallback_adapter = instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                        apply_limit_buckets: false,
                    })
                    .await;
            }

            fallback_adapter
                .map_err(|error| anyhow!("Failed to find any suitable GPU adapter: {error}"))?
        };

        // let adapter = adapter.expect("Failed to find any suitable GPU adapter");
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
        // TODO estimate the limit based on the number of storage buffers used in the shaders, since each buffer has a limit of max_storage_buffer_binding_size
        let position_limit = max_storage_buffer_binding_size / size_of::<[f32; 2]>() as u64;
        let scalar_limit = max_storage_buffer_binding_size / size_of::<f32>() as u64;
        let affine_limit = max_storage_buffer_binding_size / size_of::<[[f32; 2]; 2]>() as u64;
        let max_particles = min(
            min(position_limit, scalar_limit),
            min(affine_limit, compute_limit as u64),
        );
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
            (max_texture_size as u64 * max_texture_size as u64) as f32 / max_particles as f32
        );

        let mut required_features = Features::empty();
        let mut has_float32_filterable = false;
        let mut has_float32_atomic = false;
        debug!("Adapter features: {:?}", adapter.features());

        // Only request timestamps if the runner actually supports them
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter
            .features()
            .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC)
        {
            required_features |= wgpu::Features::SHADER_FLOAT32_ATOMIC;
            has_float32_atomic = true;
        } else {
            warn!(
                "GPU does not support SHADER_FLOAT32_ATOMIC, the sim will be less accurate. Consider using a GPU that supports this feature for better results."
            );
        }
        if adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
        {
            required_features |= wgpu::Features::FLOAT32_FILTERABLE;
            has_float32_filterable = true;
        } else {
            warn!(
                "GPU does not support FLOAT32_FILTERABLE, the sim will be less accurate. Consider using a GPU that supports this feature for better results."
            );
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
                    max_storage_buffers_per_shader_stage: min(
                        13,
                        limits.max_storage_buffers_per_shader_stage,
                    ),
                    ..Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("Failed to create device and queue")?;
        device.set_device_lost_callback(move |reason, message| {
            error!("Device lost! Reason: {:?}, Message: {}", reason, message);
        });
        timer_checkpoint("Request GPU device");
        let buffers = GpuResources::new();
        let shader_configs = shaders::create_shader_configs(
            &device,
            max_compute_invocations_per_workgroup,
            has_float32_filterable,
            has_float32_atomic,
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
            prepared_max_steps: None,
            prepared_model: None,
            has_float32_filterable,
            batch_compute_steps: 200,
            has_float32_atomic,
        })
    }

    pub fn has_float32_atomic(&self) -> bool {
        self.has_float32_atomic
    }

    // Helper function to safely parse hex ("0x1e84") or decimal strings into a u32 Device ID
    fn parse_numeric_id(s: &str) -> Option<u32> {
        let s = s.trim();
        if s.starts_with("0x") || s.starts_with("0X") {
            u32::from_str_radix(&s[2..], 16).ok()
        } else {
            s.parse::<u32>().ok()
        }
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
        if dispatch_number_workgroups_x == 0
            || dispatch_number_workgroups_y == 0
            || dispatch_number_workgroups_z == 0
        {
            return Err(anyhow!(
                "Dispatch dimensions must be greater than zero: {}x{}x{}",
                dispatch_number_workgroups_x,
                dispatch_number_workgroups_y,
                dispatch_number_workgroups_z
            ));
        }
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
        )?;
        Ok(())
    }

    pub async fn run_analyze_terrain(
        &mut self,
        sim_settings: &settings::SimSettings,
        dem: &Dem,
    ) -> Result<()> {
        if sim_settings.grid_shape_x == 0 || sim_settings.grid_shape_y == 0 {
            return Err(anyhow!("Grid dimensions must be greater than zero"));
        }
        if sim_settings.grid_shape_x > self.max_texture_size
            || sim_settings.grid_shape_y > self.max_texture_size
        {
            return Err(anyhow!(
                "Grid shape ({}, {}) exceeds max texture size of {}",
                sim_settings.grid_shape_x,
                sim_settings.grid_shape_y,
                self.max_texture_size
            ));
        }
        if sim_settings.sim_model > 1 {
            return Err(anyhow!(
                "Unsupported simulation model: {}",
                sim_settings.sim_model
            ));
        }
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
        )?;

        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;

        let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        self.resources.add_texture_with_data(
            &self.device,
            &self.queue,
            dem.data1d.as_slice(),
            TextureName::Dem,
            self.texture_size,
            TextureFormat::R32Float,
            texture_usage_input,
        )?;
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
            2_u32..=u32::MAX => {
                return Err(anyhow!(
                    "Unsupported simulation model: {}",
                    sim_settings.sim_model
                ));
            }
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
        roi: &[bool],
    ) -> Result<u32> {
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        let mut roi_bits = vec![0u32; roi.len().div_ceil(32)];
        for (i, &flag) in roi.iter().enumerate() {
            if flag {
                roi_bits[i / 32] |= 1 << (i % 32);
            }
        }
        self.resources
            .write_buffer(&self.queue, BufferName::RegionOfInterest, &roi_bits)?;
        self.run_shader(
            &ShaderName::ComputeReleaseAreas,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let number_release_cells: u32 = self
            .read_buffer::<buffers::AtomicValues>(BufferName::AtomicValues)
            .await?
            .first()
            .ok_or_else(|| anyhow!("AtomicValues buffer was empty"))?
            .number_release_cells;

        Ok(number_release_cells)
    }

    pub async fn evaluate_gpu(
        &mut self,
        sim_settings: &settings::SimSettings,
    ) -> Result<MassMovementEvaluation> {
        if sim_settings.grid_shape_x == 0 || sim_settings.grid_shape_y == 0 {
            return Err(anyhow!("Evaluation grid must not be empty"));
        }
        self.resources.write_buffer(
            &self.queue,
            BufferName::SimSettings,
            sim_settings.as_bytes(),
        )?;
        self.resources.write_buffer(
            &self.queue,
            BufferName::EvaluationCounts,
            &[0, 0, 0, 0, u32::MAX, 0, u32::MAX, u32::MAX],
        )?;

        let dispatch_x = sim_settings.grid_shape_x.div_ceil(WORKGROUP_SIZE_2D);
        let dispatch_y = sim_settings.grid_shape_y.div_ceil(WORKGROUP_SIZE_2D);
        self.run_shader(&ShaderName::EvaluateMassMovement, dispatch_x, dispatch_y, 1)
            .await?;
        self.run_shader(
            &ShaderName::EvaluateMassMovementPoints,
            dispatch_x,
            dispatch_y,
            1,
        )
        .await?;

        let counts = self
            .read_buffer::<u32>(BufferName::EvaluationCounts)
            .await?;
        let mut evaluation = evaluation_from_counts(counts[0], counts[1], counts[2]);
        if counts[6] != u32::MAX && counts[7] != u32::MAX {
            let min_x = counts[6] % sim_settings.grid_shape_x;
            let min_y = counts[6] / sim_settings.grid_shape_x;
            let max_x = counts[7] % sim_settings.grid_shape_x;
            let max_y = counts[7] / sim_settings.grid_shape_x;
            let dx = (max_x as f64 - min_x as f64) * sim_settings.cell_size as f64;
            let dy = (max_y as f64 - min_y as f64) * sim_settings.cell_size as f64;
            let min_elevation = ordered_u32_to_f32(counts[4]) as f64;
            let max_elevation = ordered_u32_to_f32(counts[5]) as f64;
            let dz = max_elevation - min_elevation;
            evaluation.beeline_distance_3d = (dx * dx + dy * dy + dz * dz).sqrt();
        }
        Ok(evaluation)
    }

    pub async fn run_initialize_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
    ) -> Result<u32> {
        if sim_settings.sim_model > 1 {
            return Err(anyhow!(
                "Unsupported simulation model: {}",
                sim_settings.sim_model
            ));
        }
        if number_release_particles as u64 > self.max_particles {
            return Err(anyhow!(
                "Number of particles {} exceeds the limit of {}",
                number_release_particles,
                self.max_particles
            ));
        }
        let particle_count = usize::try_from(number_release_particles)
            .context("Particle count does not fit in usize")?;
        let particle_buffer_size_single_value = particle_count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow!("Particle buffer size overflow"))?;
        let grid_cell_count = usize::try_from(sim_settings.grid_shape_x)
            .ok()
            .and_then(|width| {
                usize::try_from(sim_settings.grid_shape_y)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| anyhow!("Grid buffer size overflow"))?;
        let grid_buffer_size_vec2 = grid_cell_count
            .checked_mul(size_of::<[f32; 2]>())
            .ok_or_else(|| anyhow!("Grid buffer size overflow"))?;
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
                    grid_buffer_size_vec2,
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
            2_u32..=u32::MAX => {
                return Err(anyhow!(
                    "Unsupported simulation model: {}",
                    sim_settings.sim_model
                ));
            }
        }

        self.run_shader(
            &ShaderName::InitializeParticles,
            self.dispatch_number_workgroups_x_2d,
            self.dispatch_number_workgroups_y_2d,
            1,
        )
        .await?;

        let estimated_release_volume: u32 = self
            .read_buffer::<AtomicValues>(BufferName::AtomicValues)
            .await?
            .first()
            .ok_or_else(|| anyhow!("AtomicValues buffer was empty"))?
            .estimated_release_volume;
        info!("Estimated release volume: {}", estimated_release_volume);
        Ok(estimated_release_volume)
    }

    async fn prepare_simulation(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        debug!("Start simulation");
        if sim_settings.max_steps == 0 {
            return Err(anyhow!("max_steps must be greater than zero"));
        }
        if number_release_particles == 0 {
            return Err(anyhow!(
                "number_release_particles must be greater than zero"
            ));
        }
        let timestep_count = usize::try_from(sim_settings.max_steps)
            .context("max_steps does not fit in usize")?
            .checked_mul(3)
            .ok_or_else(|| anyhow!("Timestep buffer size overflow"))?;
        let timestep_bytes = timestep_count
            .checked_mul(size_of::<TimestepDataAoS>())
            .ok_or_else(|| anyhow!("Timestep buffer size overflow"))?;
        self.add_buffer(
            BufferName::TimestepData,
            timestep_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let sim_info = SimInfo {
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

        self.prepared_max_steps = Some(sim_settings.max_steps);
        self.prepared_model = Some(sim_settings.sim_model);
        Ok(())
    }

    pub async fn prepare_compute_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        self.prepare_simulation(
            sim_settings,
            number_release_particles,
            minimum_dem_elevation,
        )
        .await
    }

    async fn step_simulation(
        &self,
        steps: u32,
        p2g_shader: ShaderName,
        grid_physics_shader: ShaderName,
        particle_update_shader: ShaderName,
    ) -> Result<SimInfo> {
        let prepared_model = self
            .prepared_model
            .ok_or_else(|| anyhow!("Simulation has not been prepared"))?;
        let expected_model = match p2g_shader {
            ShaderName::P2G => 0,
            ShaderName::P2GMPM => 1,
            _ => return Err(anyhow!("Invalid particle-to-grid shader: {p2g_shader}")),
        };
        if prepared_model != expected_model {
            return Err(anyhow!(
                "Simulation was prepared for model {prepared_model}, not model {expected_model}"
            ));
        }
        let current_info = self
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("SimInfo buffer was empty"))?;
        let max_steps = self
            .prepared_max_steps
            .ok_or_else(|| anyhow!("Simulation has not been prepared"))?;
        let completed_steps = current_info.timestep.saturating_sub(1);
        let remaining_steps = max_steps.saturating_sub(completed_steps);
        if steps > remaining_steps {
            return Err(anyhow!(
                "Requested {steps} steps, but only {remaining_steps} steps remain"
            ));
        }
        if steps == 0 {
            return Ok(current_info);
        }

        let update_sim_info_config = self
            .shader_configs
            .get(&ShaderName::UpdateSimInfo)
            .ok_or_else(|| anyhow!("UpdateSimInfo shader config not found"))?;

        let update_sim_info_bindgroup =
            update_sim_info_config.create_bind_group(&self.device, &self.resources)?;

        let p2g_config = self
            .shader_configs
            .get(&p2g_shader)
            .ok_or_else(|| anyhow!("{} shader config not found", p2g_shader))?;

        let p2g_bindgroup = p2g_config.create_bind_group(&self.device, &self.resources)?;

        let grid_physics_config = self
            .shader_configs
            .get(&grid_physics_shader)
            .ok_or_else(|| anyhow!("{} shader config not found", grid_physics_shader))?;

        let grid_physics_bindgroup =
            grid_physics_config.create_bind_group(&self.device, &self.resources)?;

        let particle_update_config = self
            .shader_configs
            .get(&particle_update_shader)
            .ok_or_else(|| anyhow!("{} shader config not found", particle_update_shader))?;

        let particle_update_bindgroup =
            particle_update_config.create_bind_group(&self.device, &self.resources)?;

        let reset_grid_config = self
            .shader_configs
            .get(&ShaderName::ResetGrid)
            .ok_or_else(|| anyhow!("ResetGrid shader config not found"))?;

        let reset_grid_bind_group =
            reset_grid_config.create_bind_group(&self.device, &self.resources)?;

        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Simulation Step Encoder"),
                });
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Simulation Step Pass"),
                    timestamp_writes: None,
                });

            for _ in 0..steps {
                compute_pass.set_pipeline(&reset_grid_config.pipeline);
                compute_pass.set_bind_group(0, &reset_grid_bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    self.dispatch_number_workgroups_x_2d,
                    self.dispatch_number_workgroups_y_2d,
                    1,
                );

                compute_pass.set_pipeline(&p2g_config.pipeline);
                compute_pass.set_bind_group(0, &p2g_bindgroup, &[]);
                compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                compute_pass.set_pipeline(&grid_physics_config.pipeline);
                compute_pass.set_bind_group(0, &grid_physics_bindgroup, &[]);
                compute_pass.dispatch_workgroups(
                    self.dispatch_number_workgroups_x_2d,
                    self.dispatch_number_workgroups_y_2d,
                    1,
                );

                compute_pass.set_pipeline(&particle_update_config.pipeline);
                compute_pass.set_bind_group(0, &particle_update_bindgroup, &[]);
                compute_pass.dispatch_workgroups(self.dispatch_number_workgroups_1d, 1, 1);

                compute_pass.set_pipeline(&update_sim_info_config.pipeline);
                compute_pass.set_bind_group(0, &update_sim_info_bindgroup, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }
        self.queue.submit(Some(command_encoder.finish()));

        self.read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("SimInfo buffer was empty"))
    }

    pub async fn step_compute_particles(&mut self, steps: u32) -> Result<SimInfo> {
        self.step_simulation(
            steps,
            ShaderName::P2G,
            ShaderName::GridPhysics,
            ShaderName::ComputeParticles,
        )
        .await
    }

    pub async fn run_compute_particles(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        if self.batch_compute_steps == 0 {
            return Err(anyhow!("batch_compute_steps must be greater than zero"));
        }
        self.prepare_compute_particles(
            sim_settings,
            number_release_particles,
            minimum_dem_elevation,
        )
        .await?;

        let mut steps_run = 0;
        while steps_run < sim_settings.max_steps {
            let steps = self
                .batch_compute_steps
                .min(sim_settings.max_steps - steps_run);
            let sim_info = self.step_compute_particles(steps).await?;
            steps_run += steps;
            let flags = sim_info.parsed_flags();
            if !flags.is_empty() {
                debug!("Flags after {} submitted steps: {:?}", steps_run, flags);
            }
            if flags.contains(SimInfoFlags::SIM_STOPPED) {
                break;
            }
        }
        Ok(())
    }

    pub async fn run_sim(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        match sim_settings.sim_model {
            0 => {
                self.run_compute_particles(
                    sim_settings,
                    number_release_particles,
                    minimum_dem_elevation,
                )
                .await?
            }
            1 => {
                self.run_mpm(
                    sim_settings,
                    number_release_particles,
                    minimum_dem_elevation,
                )
                .await?
            }
            2_u32..=u32::MAX => {
                return Err(anyhow!(
                    "Unsupported simulation model: {}",
                    sim_settings.sim_model
                ));
            }
        }

        let atomic_values = self
            .read_buffer::<AtomicValues>(BufferName::AtomicValues)
            .await?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("AtomicValues buffer was empty"))?;
        let sim_info = self
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("SimInfo buffer was empty"))?;
        info!("{:#?}", sim_info);
        info!("{:#?}", atomic_values);
        if !sim_info
            .parsed_flags()
            .contains(SimInfoFlags::ALL_PARTICLES_STOPPED)
        {
            warn!(
                "Simulation reached max steps without all particles stopping. Consider increasing max_steps or checking for issues in the simulation."
            );
        }
        Ok(())
    }

    pub async fn prepare_mpm(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        self.prepare_simulation(
            sim_settings,
            number_release_particles,
            minimum_dem_elevation,
        )
        .await
    }

    pub async fn step_mpm(&mut self, steps: u32) -> Result<SimInfo> {
        self.step_simulation(
            steps,
            ShaderName::P2GMPM,
            ShaderName::GridPhysicsMPM,
            ShaderName::G2P,
        )
        .await
    }

    pub async fn run_mpm(
        &mut self,
        sim_settings: &settings::SimSettings,
        number_release_particles: u32,
        minimum_dem_elevation: f32,
    ) -> Result<()> {
        if self.batch_compute_steps == 0 {
            return Err(anyhow!("batch_compute_steps must be greater than zero"));
        }
        self.prepare_mpm(
            sim_settings,
            number_release_particles,
            minimum_dem_elevation,
        )
        .await?;

        let mut steps_run = 0;
        while steps_run < sim_settings.max_steps {
            let steps = self
                .batch_compute_steps
                .min(sim_settings.max_steps - steps_run);
            let sim_info = self.step_mpm(steps).await?;
            steps_run += steps;
            let flags = sim_info.parsed_flags();
            if !flags.is_empty() {
                debug!("Flags after {} submitted steps: {:?}", steps_run, flags);
            }
            if flags.contains(SimInfoFlags::SIM_STOPPED) {
                break;
            }
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

    #[test]
    fn sim_info_flags_pretty_print_known_flags() {
        let flags = SimInfoFlags::from(
            SimInfoFlags::OUT_OF_BOUNDS.bits()
                | SimInfoFlags::PARTICLE_OUT_OF_DEM_DATA.bits()
                | SimInfoFlags::SIM_STOPPED.bits(),
        );

        assert_eq!(
            flags.to_string(),
            "OUT_OF_BOUNDS | PARTICLE_OUT_OF_DEM_DATA | SIM_STOPPED"
        );
        assert_eq!(format!("{flags:?}"), flags.to_string());
    }

    #[test]
    fn sim_info_flags_pretty_print_empty_and_unknown_flags() {
        assert_eq!(SimInfoFlags::from(0).to_string(), "NONE");
        assert_eq!(
            SimInfoFlags::from(1 << 12).to_string(),
            "UNKNOWN(0x00001000)"
        );
    }

    #[test]
    fn sim_info_debug_prints_parsed_flags() {
        let sim_info = SimInfo {
            flags: SimInfoFlags::CFL_EXCEEDED.bits() | SimInfoFlags::IS_NAN.bits(),
            ..SimInfo::default()
        };

        assert!(format!("{sim_info:#?}").contains("flags: CFL_EXCEEDED | IS_NAN"));
    }

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
    fn test_evaluate_gpu_matches_cpu() {
        let reference = vec![true, true, false, false, true, false];
        let simulated = vec![false, true, true, false, true, false];
        let expected = evaluation::evaluate_mass_movement_area(&reference, &simulated).unwrap();
        let settings = settings::SimSettings {
            grid_shape_x: 3,
            grid_shape_y: 2,
            peak_flow_thickness_threshold: 1.0,
            ..Default::default()
        };
        let mut orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        orchestrator
            .create_buffers_and_texture_descriptions(&settings)
            .expect("Failed to create GPU resources");
        orchestrator
            .resources
            .add_texture_with_data(
                &orchestrator.device,
                &orchestrator.queue,
                &[0.0f32, 5.0, 10.0, 0.0, 2.0, 0.0],
                TextureName::Dem,
                Extent3d {
                    width: 3,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                TextureFormat::R32Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            )
            .expect("Failed to upload DEM");
        block_on(orchestrator.write_buffer(BufferName::RegionOfInterest, &[0b010_011u32]))
            .expect("Failed to write ROI");
        block_on(orchestrator.write_buffer(
            BufferName::GridPeakFlowThickness,
            &[1.0f32, 2.0, 2.0, 0.0, 2.0, 0.0],
        ))
        .expect("Failed to write peak flow thicknesses");

        let actual = block_on(orchestrator.evaluate_gpu(&settings)).expect("GPU evaluation failed");

        assert_eq!(actual.alpha, expected.alpha);
        assert_eq!(actual.beta, expected.beta);
        assert_eq!(actual.gamma, expected.gamma);
        assert_eq!(actual.jaccard, expected.jaccard);
        let expected_distance = 66.0f64.sqrt();
        println!("Expected distance: {}", expected_distance);
        println!("Actual distance: {}", actual.beeline_distance_3d);
        // TODO not correctly implemented yet
        // assert!((actual.beeline_distance_3d - expected_distance).abs() < 1e-7);

        block_on(orchestrator.write_buffer(BufferName::GridPeakFlowThickness, &[1.0f32; 6]))
            .expect("Failed to reset peak flow thicknesses");
        let empty_simulation = block_on(orchestrator.evaluate_gpu(&settings))
            .expect("GPU evaluation without affected cells failed");
        assert_eq!(empty_simulation.beeline_distance_3d, 0.0);
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
        let (mass_factor, momentum_factor, rel_error_threshold) =
            if orchestrator.has_float32_atomic() {
                (1.0, 1.0, 1e-6)
            } else {
                (10.0, 0.01, 5e-3)
            };
        let mass_p2g: f32 = test_output.iter().skip(5).take(10).sum::<f32>() / mass_factor;
        info!(
            "Mass before: {} after p2g: {} relative error: {}",
            mass,
            mass_p2g,
            (mass_p2g - mass).abs() / mass
        );
        assert!(
            (mass - mass_p2g).abs() / mass < rel_error_threshold,
            "Mass transfer from particle to grid failed"
        );
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
            (momentum_start - momentum).abs() / momentum_start < rel_error_threshold,
            "Momentum transfer from particle to grid failed"
        );
        assert!(
            (velocity[0] - velocity_x).abs() / velocity_x < rel_error_threshold,
            "Velocity X transfer from particle to grid failed"
        );
        assert!(
            (velocity[1] - velocity_y).abs() / velocity_y < rel_error_threshold,
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
            (test_output[5] / mass_factor - 16.28176).abs() / 16.28176 < rel_error_threshold,
            "test_output[5] failed"
        );
        assert!(
            (test_output[6] / mass_factor - 71.9805).abs() / 71.9805 < rel_error_threshold,
            "test_output[6] failed"
        );
        assert!(
            (test_output[7] / mass_factor - 8.537765).abs() / 8.537765 < rel_error_threshold,
            "test_output[7] failed"
        );
        assert!(
            (test_output[8] / mass_factor - 125.54444).abs() / 125.54444 < rel_error_threshold,
            "test_output[8] failed"
        );
        assert!(
            (test_output[9] / mass_factor - 555.0231).abs() / 555.0231 < rel_error_threshold,
            "test_output[9] failed"
        );
        assert!(
            (test_output[10] / mass_factor - 65.832504).abs() / 65.832504 < rel_error_threshold,
            "test_output[10] failed"
        );
        assert!(
            (test_output[11] / mass_factor - 26.373747).abs() / 26.373747 < rel_error_threshold,
            "test_output[11] failed"
        );
        assert!(
            (test_output[12] / mass_factor - 116.59646).abs() / 116.59646 < rel_error_threshold,
            "test_output[12] failed"
        );
        assert!(
            (test_output[13] / mass_factor - 13.829763).abs() / 13.829763 < rel_error_threshold,
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

        if orchestrator.has_float32_atomic() {
            // expected momentum distribution (grid x-momentum)
            assert!(
                (test_output[20] / momentum_factor - 325.6352).abs() / 325.6352
                    < rel_error_threshold,
                "test_output[20] failed"
            );
            assert!(
                (test_output[21] / momentum_factor - 1439.61).abs() / 1439.61 < rel_error_threshold,
                "test_output[21] failed"
            );
            assert!(
                (test_output[22] / momentum_factor - 170.7553).abs() / 170.7553
                    < rel_error_threshold,
                "test_output[22] failed"
            );
            assert!(
                (test_output[23] / momentum_factor - 2510.889).abs() / 2510.889
                    < rel_error_threshold,
                "test_output[23] failed"
            );
            assert!(
                (test_output[24] / momentum_factor - 11100.462).abs() / 11100.462
                    < rel_error_threshold,
                "test_output[24] failed"
            );
            assert!(
                (test_output[25] / momentum_factor - 1316.65).abs() / 1316.65 < rel_error_threshold,
                "test_output[25] failed"
            );
            assert!(
                (test_output[26] / momentum_factor - 527.475).abs() / 527.475 < rel_error_threshold,
                "test_output[26] failed"
            );
            assert!(
                (test_output[27] / momentum_factor - 2331.9292).abs() / 2331.9292
                    < rel_error_threshold,
                "test_output[27] failed"
            );
            assert!(
                (test_output[28] / momentum_factor - 276.59528).abs() / 276.59528
                    < rel_error_threshold,
                "test_output[28] failed"
            );
        } else {
            // momentum transfer is not accurate without float32 atomics, but we can at least check that the maths is correct
            assert_eq!(test_output[20], 3.0, "test_output[20] failed");
            assert_eq!(test_output[21], 14.0, "test_output[21] failed");
            assert_eq!(test_output[22], 2.0, "test_output[22] failed");
            assert_eq!(test_output[23], 25.0, "test_output[23] failed");
            assert_eq!(test_output[24], 111.0, "test_output[24] failed");
            assert_eq!(test_output[25], 13.0, "test_output[25] failed");
            assert_eq!(test_output[26], 5.0, "test_output[26] failed");
            assert_eq!(test_output[27], 23.0, "test_output[27] failed");
            assert_eq!(test_output[28], 3.0, "test_output[28] failed");
        }

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
        if orchestrator.has_float32_atomic() {
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
        } else {
            // momentum transfer is not accurate without float32 atomics, but we can at least check that the maths is correct
            assert_eq!(test_output[30], 5.0, "test_output[30] failed");
            assert_eq!(test_output[31], 22.0, "test_output[31] failed");
            assert_eq!(test_output[32], 3.0, "test_output[32] failed");
            assert_eq!(test_output[33], 38.0, "test_output[33] failed");
            assert_eq!(test_output[34], 167.0, "test_output[34] failed");
            assert_eq!(test_output[35], 20.0, "test_output[35] failed");
            assert_eq!(test_output[36], 8.0, "test_output[36] failed");
            assert_eq!(test_output[37], 35.0, "test_output[37] failed");
            assert_eq!(test_output[38], 4.0, "test_output[38] failed");
        }
    }
}
