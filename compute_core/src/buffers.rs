use crate::shaders;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, COPY_BYTES_PER_ROW_ALIGNMENT, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Extent3d, Features, Instance, InstanceDescriptor, Limits, MapMode, Origin3d,
    PipelineLayoutDescriptor, PowerPreference, Queue, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

// Re-export features for conditional compilation
#[cfg(feature = "native")]
pub use pollster::block_on;
#[cfg(feature = "wasm")]
pub use wasm_bindgen_futures::spawn_local;

// Helper function for alignment
fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) / alignment * alignment
}

// --- 1. Buffer and Texture Management (UPDATED) ---
pub struct ComputeBuffers {
    buffers: HashMap<String, Buffer>,
    textures: HashMap<String, Texture>,
    texture_views: HashMap<String, TextureView>, // Store views as they are used in bind groups
}

impl ComputeBuffers {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
        }
    }

    // --- Buffer Methods (from previous example) ---
    pub fn add_buffer(
        &mut self,
        device: &Device,
        name: String,
        size_bytes: u64,
        usage: BufferUsages,
    ) {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{} Buffer", name)),
            size: size_bytes,
            usage: usage,
            mapped_at_creation: false,
        });
        self.buffers.insert(name, buffer);
    }

    pub fn add_buffer_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        device: &Device,
        name: String,
        data: &[T],
        usage: BufferUsages,
    ) {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("{} Buffer", name)),
            contents: bytemuck::cast_slice(data),
            usage: usage,
        });
        self.buffers.insert(name, buffer);
    }

    pub fn get_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    pub fn get_buffer_mut(&mut self, name: &str) -> Option<&mut Buffer> {
        self.buffers.get_mut(name)
    }

    pub async fn read_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        buffer_name: &str,
    ) -> Result<Vec<T>> {
        let buffer = self
            .get_buffer(buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", buffer_name))?;

        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        #[cfg(feature = "native")]
        device.poll(wgpu::PollType::Wait)?;
        #[cfg(feature = "wasm")]
        device.poll(wgpu::PollType::Poll);

        receiver
            .receive()
            .await
            .ok_or_else(|| anyhow!("Failed to receive map result"))??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        buffer.unmap();
        Ok(result)
    }

    pub fn write_buffer<T: bytemuck::Pod + Send + Sync>(
        &self,
        queue: &Queue,
        buffer_name: &str,
        data: &[T],
    ) -> Result<()> {
        let buffer = self
            .get_buffer(buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found for writing", buffer_name))?;

        let expected_size = (data.len() * std::mem::size_of::<T>()) as u64;
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

    // --- Texture Methods (NEW) ---

    /// Adds a new empty texture and its default view.
    pub fn add_texture(&mut self, device: &Device, name: String, descriptor: &TextureDescriptor) {
        let texture = device.create_texture(descriptor);
        let view = texture.create_view(&TextureViewDescriptor::default());
        self.textures.insert(name.clone(), texture);
        self.texture_views.insert(name, view);
    }
    
    pub fn add_texture_data(&mut self, device: &Device, name: String, descriptor: &TextureDescriptor) {
        let texture = device.create_texture(descriptor);
        let view = texture.create_view(&TextureViewDescriptor::default());
        self.textures.insert(name.clone(), texture);
        self.texture_views.insert(name, view);
    }

    /// Adds a new texture with initial data, handling 256-byte row alignment.
    /// Data is expected to be in a format compatible with `texture_format` (e.g., f32 for R32Float).
    /// `T` must be a POD type that can be cast to bytes.
    pub fn add_texture_with_data<T: bytemuck::Pod + Send + Sync>(
        &mut self,
        device: &Device,
        queue: &Queue,
        name: String,
        data: &[T],
        texture_descriptor: &TextureDescriptor,
    ) -> Result<()> {
        let texture = device.create_texture(texture_descriptor);
        let view = texture.create_view(&TextureViewDescriptor::default());

        let bytes_per_pixel = texture_descriptor
            .format
            .block_copy_size(None)
            .expect("msg: Unsupported texture format for copying");
        let unpadded_bytes_per_row = texture_descriptor.size.width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);

        let total_padded_size = padded_bytes_per_row * texture_descriptor.size.height;

        // Create a staging buffer for the data
        let staging_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{} Staging Buffer (for texture {})",
                name,
                texture_descriptor.label.unwrap_or("unlabeled")
            )),
            contents: &vec![0; total_padded_size as usize], // Initialize with zeros
            usage: BufferUsages::COPY_SRC,
        });

        // Copy user data into the staging buffer, adding padding
        let mut padded_data_bytes = vec![0u8; total_padded_size as usize];
        let data_bytes = bytemuck::cast_slice(data);

        for y in 0..texture_descriptor.size.height as usize {
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
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(texture_descriptor.size.height),
                },
            },
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_descriptor.size,
        );
        queue.submit(Some(encoder.finish()));

        self.textures.insert(name.clone(), texture);
        self.texture_views.insert(name, view);
        Ok(())
    }

    pub fn get_texture(&self, name: &str) -> Option<&Texture> {
        self.textures.get(name)
    }

    pub fn get_texture_view(&self, name: &str) -> Option<&TextureView> {
        self.texture_views.get(name)
    }

    /// Reads data from a texture, handling 256-byte row alignment.
    /// `T` must be a POD type that can be cast from bytes.
    pub async fn read_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        texture_name: &str,
    ) -> Result<Vec<T>> {
        let texture = self
            .get_texture(texture_name)
            .ok_or_else(|| anyhow!("Texture '{}' not found for reading", texture_name))?;

        let texture_desc = texture.as_image_copy();
        let bytes_per_pixel = texture_desc
            .texture
            .format()
            .block_copy_size(None)
            .expect("Unsupported texture format for copying");
        let unpadded_bytes_per_row = texture_desc.texture.size().width * bytes_per_pixel;
        let padded_bytes_per_row = align_up(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);

        let total_padded_size = (padded_bytes_per_row * texture_desc.texture.size().height) as u64;

        // Create a staging buffer to copy texture data into
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!(
                "{} Staging Buffer (for reading texture)",
                texture_name
            )),
            size: total_padded_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Texture Read Encoder", texture_name)),
        });

        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
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

        #[cfg(feature = "native")]
        device.poll(wgpu::PollType::Wait)?;
        #[cfg(feature = "wasm")]
        device.poll(wgpu::PollType::Poll);

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

        Ok(bytemuck::cast_slice(&unpadded_data_bytes).to_vec())
    }

    /// Writes data to an existing texture, handling 256-byte row alignment.
    pub fn write_texture<T: bytemuck::Pod + Send + Sync>(
        &self,
        device: &Device,
        queue: &Queue,
        texture_name: &str,
        data: &[T],
    ) -> Result<()> {
        let texture = self
            .get_texture(texture_name)
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
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(texture_desc.texture.size().height),
                },
            },
            TexelCopyTextureInfo {
                texture: texture,
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

pub fn texture_descriptor(label: &str, size: Extent3d, format: TextureFormat, usage: TextureUsages) -> TextureDescriptor {
    TextureDescriptor {
            label: Some(label),
            size: size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: format,
            usage: usage,
            view_formats: &[],
        }
}
