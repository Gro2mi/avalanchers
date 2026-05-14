use crate::settings::SimSettings;
use anyhow::{Result, anyhow};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::HashMap;
use tracing::warn;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, COPY_BYTES_PER_ROW_ALIGNMENT, CommandEncoderDescriptor,
    Device, Extent3d, MapMode, Origin3d, Queue, Sampler, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::SimInfo;
use crate::utils::split_channels;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct AtomicValues {
    pub grid_peak_velocity: u32,
    pub grid_peak_flow_thickness: u32,
    pub alpha: u32,
    pub travel_length: u32,
    pub release_volume: u32,
    pub number_release_cells: u32,
    pub number_release_particles: u32,
    pub stopped_particles: u32,
}

#[derive(Eq, Hash, PartialEq, Clone)]
pub enum BufferName {
    SimInfo,
    /// Index for initializing particles using atomicAdd in the shader
    ParticleIndex,
    // atomic grids
    GridCellCount,
    GridPeakVelocity,
    GridPeakFlowThickness,
    GridMass,
    GridForces,
    // settings/initialization dependent buffers
    SimSettings,
    Particles,
    /// timestep data of the 0 index particle
    TimestepData,
    // Debug buffers
    OutDebugNormals,
    OutDebugRelease,
    AtomicValues,
}

impl BufferName {
    pub fn to_str(&self) -> &'static str {
        match self {
            BufferName::OutDebugNormals => "out_debug_normals",
            BufferName::OutDebugRelease => "out_debug_release",
            BufferName::SimInfo => "sim_info",
            BufferName::SimSettings => "sim_settings",
            BufferName::GridCellCount => "cell_count_grid",
            BufferName::GridPeakVelocity => "max_velocity_grid",
            BufferName::ParticleIndex => "particle_index",
            BufferName::Particles => "particles",
            BufferName::TimestepData => "timestep_data",
            BufferName::GridMass => "grid_mass",
            BufferName::GridForces => "grid_forces",
            BufferName::GridPeakFlowThickness => "grid_peak_flow_thickness",
            BufferName::AtomicValues => "atomic_values",
        }
    }
}

impl std::fmt::Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

impl std::str::FromStr for BufferName {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "out_debug_normals" => Ok(BufferName::OutDebugNormals),
            "out_debug_release" => Ok(BufferName::OutDebugRelease),
            "sim_info" => Ok(BufferName::SimInfo),
            "sim_settings" => Ok(BufferName::SimSettings),
            "cell_count_grid" => Ok(BufferName::GridCellCount),
            "max_velocity_grid" => Ok(BufferName::GridPeakVelocity),
            "particle_index" => Ok(BufferName::ParticleIndex),
            "particles" => Ok(BufferName::Particles),
            "timestep_data" => Ok(BufferName::TimestepData),
            "grid_mass" => Ok(BufferName::GridMass),
            "grid_forces" => Ok(BufferName::GridForces),
            "grid_peak_flow_thickness" => Ok(BufferName::GridPeakFlowThickness),
            "atomic_values" => Ok(BufferName::AtomicValues),
            _ => Err(format!("Unknown buffer name: {}", name)),
        }
    }
}

#[derive(Eq, Hash, PartialEq, Clone)]
pub enum TextureName {
    Wind,
    Normals,
    Slope,
    Roughness,
    ReleaseAreas,
    Landcover,
    StagingBuffer,
    Dem,
    ReleaseAreasInput,
    CellCount,
    Curvature,
}

impl TextureName {
    pub fn to_str(&self) -> &'static str {
        match self {
            TextureName::Wind => "wind",
            TextureName::Normals => "normals",
            TextureName::Slope => "slope",
            TextureName::Roughness => "roughness",
            TextureName::ReleaseAreas => "release_areas",
            TextureName::Landcover => "landcover",
            TextureName::StagingBuffer => "staging_buffer",
            TextureName::Dem => "dem",
            TextureName::ReleaseAreasInput => "release_areas_input",
            TextureName::CellCount => "cell_count",
            TextureName::Curvature => "curvature",
        }
    }
}

impl std::fmt::Display for TextureName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}
impl std::str::FromStr for TextureName {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "wind" => Ok(TextureName::Wind),
            "normals" => Ok(TextureName::Normals),
            "slope" => Ok(TextureName::Slope),
            "roughness" => Ok(TextureName::Roughness),
            "release_areas" => Ok(TextureName::ReleaseAreas),
            "landcover" => Ok(TextureName::Landcover),
            "staging_buffer" => Ok(TextureName::StagingBuffer),
            "dem" => Ok(TextureName::Dem),
            "release_areas_input" => Ok(TextureName::ReleaseAreasInput),
            "cell_count" => Ok(TextureName::CellCount),
            "curvature" => Ok(TextureName::Curvature),
            _ => Err(format!("Unknown texture name: {}", name)),
        }
    }
}

// Helper function for alignment
fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

pub struct GpuResources {
    buffers: HashMap<BufferName, Buffer>,
    textures: HashMap<TextureName, Texture>,
    texture_views: HashMap<TextureName, TextureView>,
    samplers: HashMap<String, Sampler>,
    total_allocated_buffer_bytes: usize, // Track total allocated buffer size for debugging/monitoring
}

impl Default for GpuResources {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuResources {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
            samplers: HashMap::new(),
            total_allocated_buffer_bytes: 0,
        }
    }

    pub fn get_total_allocated_memory_mb(&self) -> f64 {
        self.total_allocated_buffer_bytes as f64 / (1024.0 * 1024.0)
    }

    fn poll(&self, device: &Device) {
        #[cfg(not(target_arch = "wasm32"))]
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("Failed to poll device");
        #[cfg(target_arch = "wasm32")]
        device
            .poll(wgpu::PollType::Poll)
            .expect("Failed to poll device");
    }

    pub fn get_sampler(&self, name: &str) -> Option<&Sampler> {
        self.samplers.get(name)
    }

    pub fn add_buffer(
        &mut self,
        device: &Device,
        name: BufferName,
        size_bytes: usize,
        usage: BufferUsages,
    ) {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{} Buffer", name)),
            size: size_bytes as u64,
            usage,
            mapped_at_creation: false,
        });
        self.buffers.insert(name, buffer);
        self.total_allocated_buffer_bytes += size_bytes;
    }

    pub fn add_buffer_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        device: &Device,
        name: BufferName,
        data: &[T],
        usage: BufferUsages,
    ) {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(name.to_str()),
            contents: &Self::prepare_buffer_contents(data),
            usage,
        });
        self.buffers.insert(name, buffer);
        self.total_allocated_buffer_bytes += std::mem::size_of_val(data);
    }

    pub fn prepare_buffer_contents<T: Pod>(original_data: &[T]) -> Cow<'_, [u8]> {
        // 1. Get the raw byte size of the input
        let original_bytes = bytemuck::cast_slice(original_data);
        let len = original_bytes.len();
        let remainder = len % 16;

        if remainder == 0 {
            // Return a borrowed reference to the original data (Zero allocation)
            Cow::Borrowed(original_bytes)
        } else {
            // Create an owned, padded version (Allocation only when needed)
            let padding = 16 - remainder;
            let mut padded_bytes = original_bytes.to_vec();
            padded_bytes.extend(std::iter::repeat_n(0, padding));
            Cow::Owned(padded_bytes)
        }
    }

    pub fn get_buffer(&self, name: &BufferName) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    pub fn get_buffer_mut(&mut self, name: BufferName) -> Option<&mut Buffer> {
        self.buffers.get_mut(&name)
    }

    pub async fn read_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        buffer_name: BufferName,
    ) -> Result<Vec<T>> {
        let src_buffer = self
            .get_buffer(&buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", &buffer_name))?;
        let size = src_buffer.size();
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("Copy {} Buffer Encoder", buffer_name)),
        });

        encoder.copy_buffer_to_buffer(src_buffer, 0, &staging_buffer, 0, size);

        queue.submit(Some(encoder.finish()));

        // Explicitly poll the device to ensure the copy and map_async are processed

        let buffer_slice = staging_buffer.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Poll again to ensure the map_async callback is processed
        self.poll(device);

        // Await the mapping result
        receiver
            .receive()
            .await
            .ok_or_else(|| anyhow!("Failed to receive map result"))??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }

    pub fn write_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        queue: &Queue,
        buffer_name: BufferName,
        data: &[T],
    ) -> Result<()> {
        let buffer = self
            .get_buffer(&buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found for writing", buffer_name))?;

        let expected_size = std::mem::size_of_val(data) as u64;
        if expected_size > buffer.size() {
            return Err(anyhow!(
                "Data size ({}) exceeds buffer '{}' capacity ({}).",
                expected_size,
                buffer_name,
                buffer.size()
            ));
        }
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// Adds a new empty texture and its default view.
    pub fn add_texture(
        &mut self,
        device: &Device,
        label: TextureName,
        texture_size: Extent3d,
        format: TextureFormat,
        texture_usage_input: TextureUsages,
    ) {
        let texture = device.create_texture(&texture_descriptor(
            &label,
            texture_size,
            format,
            texture_usage_input,
        ));
        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.insert(label.clone(), texture);
        self.texture_views.insert(label, view);
        self.total_allocated_buffer_bytes +=
            (texture_size.width * texture_size.height * texture_size.depth_or_array_layers)
                as usize
                * format.block_copy_size(None).unwrap_or(4) as usize; // Approximate size for tracking
    }

    /// Adds a new texture with initial data, handling 256-byte row alignment.
    /// Data is expected to be in a format compatible with `texture_format` (e.g., f32 for R32Float).
    /// `T` must be a POD type that can be cast to bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn add_texture_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        device: &Device,
        queue: &Queue,
        data: &[T],
        label: TextureName,
        texture_size: Extent3d,
        format: TextureFormat,
        texture_usage_input: TextureUsages,
    ) -> Result<()> {
        let name = label.to_str();
        let texture = device.create_texture(&texture_descriptor(
            &label,
            texture_size,
            format,
            texture_usage_input,
        ));
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bytes_per_pixel = format
            .block_copy_size(None)
            .expect("msg: Unsupported texture format for copying");
        let unpadded_bytes_per_row = texture_size.width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);

        let total_padded_size = padded_bytes_per_row * texture_size.height;

        // Create a staging buffer for the data
        let staging_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("Staging Buffer (for texture {})", name)),
            contents: &vec![0; total_padded_size as usize], // Initialize with zeros
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Copy user data into the staging buffer, adding padding
        let mut padded_data_bytes = vec![0u8; total_padded_size as usize];
        let data_bytes = bytemuck::cast_slice(data);

        for y in 0..texture_size.height as usize {
            let src_start = y * unpadded_bytes_per_row as usize;
            let src_end = src_start + unpadded_bytes_per_row as usize;
            let dest_start = y * padded_bytes_per_row as usize;
            let dest_end = dest_start + unpadded_bytes_per_row as usize; // Only copy actual data, padding is zeroed by default

            if src_end > data_bytes.len() {
                return Err(anyhow!(
                    "Provided data is too small for texture dimensions."
                ));
            }
            padded_data_bytes[dest_start..dest_end]
                .copy_from_slice(&data_bytes[src_start..src_end]);
        }

        queue.write_buffer(&staging_buffer, 0, &padded_data_bytes);

        // Copy from staging buffer to texture
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Texture Data Copy Encoder", name)),
        });

        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(texture_size.height),
                },
            },
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_size,
        );
        queue.submit(Some(encoder.finish()));

        self.textures.insert(label.clone(), texture);
        self.texture_views.insert(label, view);
        self.total_allocated_buffer_bytes +=
            (texture_size.width * texture_size.height * texture_size.depth_or_array_layers)
                as usize
                * format.block_copy_size(None).unwrap_or(4) as usize; // Approximate size for tracking
        Ok(())
    }

    pub fn get_texture(&self, name: &TextureName) -> Option<&Texture> {
        self.textures.get(name)
    }

    pub fn get_texture_view(&self, name: &TextureName) -> Option<&TextureView> {
        self.texture_views.get(name)
    }

    /// Reads data from a texture, handling 256-byte row alignment.
    /// Reads a texture from the GPU and returns a flat Vec<T> of the raw data.
    pub async fn read_texture_flat<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        texture_name: TextureName,
    ) -> Result<Vec<T>> {
        let texture = self
            .get_texture(&texture_name)
            .ok_or_else(|| anyhow!("Texture '{}' not found", &texture_name))?;

        let size = texture.size();
        let format = texture.format();

        // block_copy_size gives us bytes per pixel (e.g., 4 for R32Float, 16 for RGBA32Float)
        let bytes_per_pixel = format
            .block_copy_size(None)
            .ok_or_else(|| anyhow!("Unsupported texture format: {:?}", format))?;

        let unpadded_bytes_per_row = size.width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);
        let total_padded_size = (padded_bytes_per_row * size.height) as u64;

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Texture Staging Buffer"),
            size: total_padded_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });

        self.poll(device);
        receiver
            .receive()
            .await
            .ok_or_else(|| anyhow!("Failed to receive map result"))??;

        let padded_range = buffer_slice.get_mapped_range();
        let mut unpadded_data = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);

        for y in 0..size.height as usize {
            let start = y * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            unpadded_data.extend_from_slice(&padded_range[start..end]);
        }

        drop(padded_range);
        staging_buffer.unmap();

        // Convert raw bytes to Vec<T>
        Ok(bytemuck::cast_slice::<u8, T>(&unpadded_data).to_vec())
    }

    pub async fn read_texture_single_channel<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        name: TextureName,
    ) -> Result<Vec<T>> {
        // Simply returns the flat vector
        self.read_texture_flat::<T>(device, queue, name).await
    }

    pub async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        name: TextureName,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        let flat_data = self.read_texture_flat::<T>(device, queue, name).await?;

        // Use your existing split_channels logic
        Ok(split_channels::<T>(&flat_data))
    }

    /// Writes data to an existing texture, handling 256-byte row alignment.
    pub fn write_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        queue: &Queue,
        texture_name: TextureName,
        data: &[T],
    ) -> Result<()> {
        let texture = self
            .get_texture(&texture_name)
            .ok_or_else(|| anyhow!("Texture '{}' not found", texture_name))?;

        let size = texture.size();
        let format = texture.format();

        // Calculate how many bytes one row of pixels actually takes in memory
        let bytes_per_pixel = format
            .block_copy_size(None)
            .expect("Unsupported texture format");
        let bytes_per_row = size.width * bytes_per_pixel;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(size.height),
            },
            size,
        );

        Ok(())
    }
}

pub fn texture_descriptor(
    label: &TextureName,
    size: Extent3d,
    format: TextureFormat,
    usage: TextureUsages,
) -> TextureDescriptor<'_> {
    TextureDescriptor {
        label: Some(label.to_str()),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    }
}
pub const DEBUG_BUFFER_SIZE: usize = 100 * 4;
pub fn create_buffers_and_texture_descriptions(
    device: &Device,
    texture_size: Extent3d,
    has_float32_filterable: bool,
) -> GpuResources {
    let mut gpu_resources = GpuResources::default();
    let filter_mode = if has_float32_filterable {
        wgpu::FilterMode::Linear
    } else {
        warn!(
            "Device does not support FLOAT32_FILTERABLE, using Nearest filtering for float textures which may reduce accuray. Consider using a GPU that supports FLOAT32_FILTERABLE for better results."
        );
        wgpu::FilterMode::Nearest
    };
    gpu_resources.samplers.insert(
        "sampler".to_string(),
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }),
    );

    let texture_usage_default = TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;

    let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let texture_usage_output =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING;
    let buffer_usage_output = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let atomic_grid_size = (texture_size.width * texture_size.height * 4) as usize;

    gpu_resources.add_texture(
        device,
        TextureName::Wind,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_input,
    );
    gpu_resources.add_texture(
        device,
        TextureName::Normals,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_output,
    );
    gpu_resources.add_texture(
        device,
        TextureName::Slope,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_output,
    );
    gpu_resources.add_texture(
        device,
        TextureName::Curvature,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_output,
    );
    gpu_resources.add_texture(
        device,
        TextureName::Roughness,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    gpu_resources.add_texture(
        device,
        TextureName::ReleaseAreas,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    gpu_resources.add_texture(
        device,
        TextureName::Landcover,
        texture_size,
        TextureFormat::Rgba8Uint,
        texture_usage_input,
    );

    gpu_resources.add_buffer(
        device,
        BufferName::SimSettings,
        ((size_of::<SimSettings>() - 1) / 16 + 1) * 16, // Ensure 16-byte alignment
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::OutDebugNormals,
        DEBUG_BUFFER_SIZE,
        buffer_usage_output,
    );

    gpu_resources.add_buffer(
        device,
        BufferName::OutDebugRelease,
        DEBUG_BUFFER_SIZE,
        buffer_usage_output,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::SimInfo,
        size_of::<SimInfo>(),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridCellCount,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridPeakVelocity,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridMass,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridMass,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridForces,
        atomic_grid_size * 2,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::GridPeakFlowThickness,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    gpu_resources.add_buffer(
        device,
        BufferName::AtomicValues,
        ((size_of::<AtomicValues>() - 1) / 16 + 1) * 16,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );

    gpu_resources
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn buffer_name_to_str_covers_all_variants() {
        let cases = [
            (BufferName::OutDebugNormals, "out_debug_normals"),
            (BufferName::OutDebugRelease, "out_debug_release"),
            (BufferName::SimInfo, "sim_info"),
            (BufferName::SimSettings, "sim_settings"),
            (BufferName::GridCellCount, "cell_count_grid"),
            (BufferName::GridPeakVelocity, "max_velocity_grid"),
            (BufferName::GridMass, "grid_mass"),
            (BufferName::GridForces, "grid_forces"),
            (
                BufferName::GridPeakFlowThickness,
                "grid_peak_flow_thickness",
            ),
            (BufferName::AtomicValues, "atomic_values"),
            (BufferName::ParticleIndex, "particle_index"),
            (BufferName::Particles, "particles"),
            (BufferName::TimestepData, "timestep_data"),
        ];

        for (name, expected) in cases {
            assert_eq!(name.to_str(), expected);
            assert_eq!(name.to_string(), expected);
        }
    }

    #[test]
    fn texture_name_to_str_covers_all_variants() {
        let cases = [
            (TextureName::Wind, "wind"),
            (TextureName::Normals, "normals"),
            (TextureName::Slope, "slope"),
            (TextureName::Curvature, "curvature"),
            (TextureName::Roughness, "roughness"),
            (TextureName::ReleaseAreas, "release_areas"),
            (TextureName::Landcover, "landcover"),
            (TextureName::StagingBuffer, "staging_buffer"),
            (TextureName::Dem, "dem"),
            (TextureName::ReleaseAreasInput, "release_areas_input"),
            (TextureName::CellCount, "cell_count"),
        ];

        for (name, expected) in cases {
            assert_eq!(name.to_str(), expected);
            assert_eq!(name.to_string(), expected);
        }
    }

    #[test]
    fn align_up_behaves_as_expected() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn prepare_buffer_contents_returns_borrowed_when_already_16_byte_aligned() {
        let data = [1u32, 2, 3, 4]; // 16 bytes
        let prepared = GpuResources::prepare_buffer_contents(&data);

        assert!(matches!(prepared, Cow::Borrowed(_)));
        assert_eq!(prepared.len(), 16);
        assert_eq!(&prepared[..], bytemuck::cast_slice::<u32, u8>(&data));
    }

    #[test]
    fn prepare_buffer_contents_pads_when_not_16_byte_aligned() {
        let data = [1u32, 2, 3]; // 12 bytes
        let prepared = GpuResources::prepare_buffer_contents(&data);

        assert!(matches!(prepared, Cow::Owned(_)));
        assert_eq!(prepared.len(), 16);
        assert_eq!(&prepared[..12], bytemuck::cast_slice::<u32, u8>(&data));
        assert_eq!(&prepared[12..], &[0, 0, 0, 0]);
    }

    #[test]
    fn prepare_buffer_contents_is_always_multiple_of_16_and_prefix_is_original() {
        for len in 0usize..33 {
            let data: Vec<u8> = (0..len as u8).collect();
            let prepared = GpuResources::prepare_buffer_contents(&data);

            assert_eq!(prepared.len() % 16, 0);
            assert_eq!(&prepared[..data.len()], data.as_slice());
            for b in &prepared[data.len()..] {
                assert_eq!(*b, 0);
            }
        }
    }

    #[test]
    fn texture_descriptor_is_constructed_correctly() {
        let size = Extent3d {
            width: 64,
            height: 32,
            depth_or_array_layers: 1,
        };
        let desc = texture_descriptor(
            &TextureName::Normals,
            size,
            TextureFormat::Rgba32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );

        assert_eq!(desc.label, Some("normals"));
        assert_eq!(desc.size.width, 64);
        assert_eq!(desc.size.height, 32);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.dimension, TextureDimension::D2);
        assert_eq!(desc.format, TextureFormat::Rgba32Float);
        assert!(desc.usage.contains(TextureUsages::TEXTURE_BINDING));
        assert!(desc.usage.contains(TextureUsages::COPY_DST));
        assert_eq!(desc.mip_level_count, 1);
        assert_eq!(desc.sample_count, 1);
        assert!(desc.view_formats.is_empty());
    }

    #[test]
    fn compute_buffers_default_starts_empty() {
        let buffers = GpuResources::default();
        assert!(buffers.get_buffer(&BufferName::SimInfo).is_none());
        assert!(buffers.get_texture(&TextureName::Wind).is_none());
        assert!(buffers.get_texture_view(&TextureName::Wind).is_none());
    }

    #[test]
    fn debug_buffer_size_constant_is_expected() {
        assert_eq!(DEBUG_BUFFER_SIZE, 100 * 4);
        assert_eq!(DEBUG_BUFFER_SIZE, 400);
    }
}
