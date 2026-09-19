//! Offscreen capture of a rendered frame. Native only: reading pixels back requires
//! blocking on the device, which the browser does not allow.

use crate::Renderer;
use anyhow::{Result, bail};

/// Renders one frame into an offscreen texture and returns tightly packed pixel bytes.
///
/// The renderer's colour format must be an 8-bit RGBA format, giving 4 bytes per pixel.
pub fn render_to_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let format = renderer.color_format();
    if !matches!(
        format,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    ) {
        bail!("capture requires an 8-bit RGBA colour format, got {format:?}");
    }

    let (width, height) = (width.max(1), height.max(1));
    renderer.resize(device, width, height);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Capture Target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer.render(device, queue, &view);

    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Capture Readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Capture Encoder"),
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        target.size(),
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    receiver.recv()??;

    let padded = slice
        .get_mapped_range()
        .map_err(|e| anyhow::anyhow!("failed to map the readback buffer: {e:?}"))?;
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bytes_per_row as usize;
        pixels.extend_from_slice(&padded[start..start + unpadded_bytes_per_row as usize]);
    }
    drop(padded);
    staging.unmap();

    Ok(pixels)
}
