use anyhow::{Result, anyhow};
use std::collections::HashMap;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, COPY_BYTES_PER_ROW_ALIGNMENT, CommandEncoderDescriptor,
    Device, Extent3d, MapMode, Origin3d, Queue, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

#[derive(Eq, Hash, PartialEq, Clone)]
pub enum BufferName {
    OutDebugNormals,
    OutDebugRelease,
    NumberReleaseCells,
    NumberReleaseParticles,
    SimInfo,
    NumberParticles,
    CellCountGrid,
    VelocityGrid,
    MaxVelocityGrid,
    SimSettings,
    Particles,
    ParticleIndex,
    TimestepData,
}

impl BufferName {
    pub fn to_str(&self) -> &'static str {
        match self {
            BufferName::OutDebugNormals => "out_debug_normals",
            BufferName::OutDebugRelease => "out_debug_release",
            BufferName::NumberReleaseCells => "number_release_cells",
            BufferName::NumberReleaseParticles => "number_release_particles",
            BufferName::SimInfo => "sim_info",
            BufferName::NumberParticles => "number_particles",
            BufferName::CellCountGrid => "cell_count_grid",
            BufferName::VelocityGrid => "velocity_grid",
            BufferName::MaxVelocityGrid => "max_velocity_grid",
            BufferName::SimSettings => "sim_settings",
            BufferName::Particles => "particles",
            BufferName::ParticleIndex => "particle_index",
            BufferName::TimestepData => "timestep_data",
        }
    }

    pub fn to_string(&self) -> String {
        self.to_str().to_string()
    }
}
impl std::fmt::Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
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
        }
    }

    pub fn to_string(&self) -> String {
        self.to_str().to_string()
    }
}
impl std::fmt::Display for TextureName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

// Re-export features for conditional compilation
// #[cfg(feature = "native")]
// pub use pollster::block_on;
// #[cfg(feature = "wasm")]
// pub use wasm_bindgen_futures::spawn_local;

use crate::SimInfo;
use crate::utils::split_channels;

// Helper function for alignment
fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

// --- 1. Buffer and Texture Management (UPDATED) ---
pub struct ComputeBuffers {
    buffers: HashMap<BufferName, Buffer>,
    textures: HashMap<TextureName, Texture>,
    texture_views: HashMap<TextureName, TextureView>, // Store views as they are used in bind groups
}

impl Default for ComputeBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBuffers {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
        }
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
    }

    pub fn add_buffer_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        device: &Device,
        name: BufferName,
        data: &[T],
        usage: BufferUsages,
        fill: bool,
    ) {
        let mut bytes = bytemuck::cast_slice(data).to_vec();
        if fill {
            let remainder = bytes.len() % 16;
            if remainder != 0 {
                let pad = 16 - remainder;
                bytes.extend(std::iter::repeat_n(0u8, pad));
            }
        }
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("{} Buffer", name)),
            contents: &bytes,
            usage,
        });
        self.buffers.insert(name, buffer);
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
        Ok(())
    }

    pub fn get_texture(&self, name: &TextureName) -> Option<&Texture> {
        self.textures.get(name)
    }

    pub fn get_texture_view(&self, name: &TextureName) -> Option<&TextureView> {
        self.texture_views.get(name)
    }

    /// Reads data from a texture, handling 256-byte row alignment.
    /// `T` must be a POD type that can be cast from bytes.
    pub async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        texture_name: TextureName,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        let texture = self
            .get_texture(&texture_name)
            .ok_or_else(|| anyhow!("Texture '{}' not found for reading", &texture_name))?;

        let texture_desc = texture.as_image_copy();
        let bytes_per_pixel = texture_desc
            .texture
            .format()
            .block_copy_size(None)
            .expect("Unsupported texture format for copying");
        let width = texture_desc.texture.size().width;
        let height = texture_desc.texture.size().height;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);

        let total_padded_size = (padded_bytes_per_row * height) as u64;

        // Create a staging buffer to copy texture data into
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!(
                "{} Staging Buffer (for reading texture)",
                TextureName::StagingBuffer.to_str()
            )),
            size: total_padded_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!(
                "{} Texture Read Encoder",
                TextureName::StagingBuffer.to_str()
            )),
        });

        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(texture_desc.texture.size().height),
                },
            },
            texture_desc.texture.size(),
        );
        queue.submit(Some(encoder.finish()));

        // Map and read the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.poll(device);
        receiver
            .receive()
            .await
            .ok_or_else(|| anyhow!("Failed to receive map result"))??;

        let padded_data_bytes = buffer_slice.get_mapped_range();

        // Remove padding and convert to Vec<T>
        let mut unpadded_data_bytes = Vec::with_capacity(
            unpadded_bytes_per_row as usize * texture_desc.texture.size().height as usize,
        );
        for y in 0..texture_desc.texture.size().height as usize {
            let src_start = y * padded_bytes_per_row as usize;
            let src_end = src_start + unpadded_bytes_per_row as usize;
            unpadded_data_bytes.extend_from_slice(&padded_data_bytes[src_start..src_end]);
        }

        drop(padded_data_bytes); // Unmap
        staging_buffer.unmap();
        Ok(split_channels::<T>(bytemuck::cast_slice::<u8, T>(
            &unpadded_data_bytes,
        )))
    }

    /// Writes data to an existing texture, handling 256-byte row alignment.
    pub fn write_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        texture_name: TextureName,
        data: &[T],
    ) -> Result<()> {
        let texture = self
            .get_texture(&texture_name)
            .ok_or_else(|| anyhow!("Texture '{}' not found for writing", texture_name))?;

        let texture_desc = texture.as_image_copy();
        let bytes_per_pixel = texture_desc
            .texture
            .format()
            .block_copy_size(None)
            .expect("Unsupported texture format for copying");
        let unpadded_bytes_per_row = texture_desc.texture.size().width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);

        let total_padded_size = padded_bytes_per_row * texture_desc.texture.size().height;

        // Create a staging buffer for the data
        let staging_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{} Staging Buffer (for reading texture)",
                texture_name
            )),
            contents: &vec![0; total_padded_size as usize], // Initialize with zeros
            usage: BufferUsages::COPY_SRC,
        });

        // Copy user data into the staging buffer, adding padding
        let mut padded_data_bytes = vec![0u8; total_padded_size as usize];
        let data_bytes = bytemuck::cast_slice(data);

        for y in 0..texture_desc.texture.size().height as usize {
            let src_start = y * unpadded_bytes_per_row as usize;
            let src_end = src_start + unpadded_bytes_per_row as usize;
            let dest_start = y * padded_bytes_per_row as usize;
            let dest_end = dest_start + unpadded_bytes_per_row as usize;

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
            label: Some(&format!("{} Texture Data Write Encoder", texture_name)),
        });

        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(texture_desc.texture.size().height),
                },
            },
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_desc.texture.size(),
        );
        queue.submit(Some(encoder.finish()));
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
) -> ComputeBuffers {
    let mut compute_buffers = ComputeBuffers::new();
    let texture_usage_default = TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;

    let texture_usage_input = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let buffer_usage_output = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let atomic_grid_size = (texture_size.width * texture_size.height * 4) as usize;

    compute_buffers.add_texture(
        device,
        TextureName::Wind,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_input,
    );
    compute_buffers.add_texture(
        device,
        TextureName::Normals,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    compute_buffers.add_texture(
        device,
        TextureName::Slope,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    compute_buffers.add_texture(
        device,
        TextureName::Roughness,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    compute_buffers.add_texture(
        device,
        TextureName::ReleaseAreas,
        texture_size,
        TextureFormat::Rgba32Float,
        texture_usage_default,
    );
    compute_buffers.add_texture(
        device,
        TextureName::Landcover,
        texture_size,
        TextureFormat::Rgba8Uint,
        texture_usage_input,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::OutDebugNormals,
        DEBUG_BUFFER_SIZE,
        buffer_usage_output,
    );

    compute_buffers.add_buffer(
        device,
        BufferName::OutDebugRelease,
        DEBUG_BUFFER_SIZE,
        buffer_usage_output,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::NumberReleaseCells,
        4,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::NumberReleaseCells,
        4,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::SimInfo,
        size_of::<SimInfo>(),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::NumberParticles,
        4,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::CellCountGrid,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::VelocityGrid,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    compute_buffers.add_buffer(
        device,
        BufferName::MaxVelocityGrid,
        atomic_grid_size,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );

    compute_buffers
}
