//! Renders a DEM offscreen and writes it to a PNG. Useful for checking the output without a window.
//!
//! Usage: `cargo run -p render_core --example dem_snapshot -- [path/to/dem.asc] [out.png]`

use render_core::{GpuContext, Renderer, TerrainData};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 640;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dem_path = args.next();
    let out_path = args
        .next()
        .unwrap_or_else(|| "dem_snapshot.png".to_string());

    let terrain = match dem_path {
        Some(path) => {
            let dem = pollster::block_on(data_processor::load_dem(&path))?;
            println!(
                "Loaded DEM {}x{} at {} m resolution",
                dem.width, dem.height, dem.cell_size
            );
            TerrainData::new(
                dem.width as u32,
                dem.height as u32,
                dem.cell_size,
                dem.data1d,
            )?
        }
        None => {
            println!("No DEM path given, rendering a synthetic terrain");
            let (w, h) = (256u32, 256u32);
            let heights = (0..h)
                .flat_map(|y| {
                    (0..w).map(move |x| {
                        let fx = x as f32 / w as f32;
                        let fy = y as f32 / h as f32;
                        1500.0
                            + (fx * std::f32::consts::PI * 2.0).sin() * 180.0
                            + (fy * std::f32::consts::PI).cos() * 260.0
                    })
                })
                .collect();
            TerrainData::new(w, h, 20.0, heights)?
        }
    };

    let ctx = pollster::block_on(GpuContext::headless())?;
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Snapshot Target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = Renderer::new(&ctx.device, &ctx.queue, FORMAT, WIDTH, HEIGHT, &terrain);
    renderer.render(&ctx.device, &ctx.queue, &view);

    let bytes_per_row = WIDTH * 4;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snapshot Readback"),
        size: (bytes_per_row * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        target.size(),
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    receiver.recv()??;

    let data = slice
        .get_mapped_range()
        .map_err(|e| anyhow::anyhow!("failed to map readback buffer: {e:?}"))?;
    image::save_buffer(&out_path, &data, WIDTH, HEIGHT, image::ColorType::Rgba8)?;
    drop(data);
    staging.unmap();

    println!("Wrote {out_path}");
    Ok(())
}
