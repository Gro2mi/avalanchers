//! Renders a DEM offscreen and inspects the resulting pixels.
//!
//! Skipped when no GPU adapter is available, so it stays usable on machines without one.

use render_core::{GpuContext, OverlayRange, ParticleBuffers, Renderer, TerrainData};
use wgpu::util::DeviceExt;

const TARGET_SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn sloped_terrain(width: u32, height: u32) -> TerrainData {
    let heights = (0..height)
        .flat_map(|y| (0..width).map(move |x| 1000.0 + (x + y) as f32 * 1.5))
        .collect();
    TerrainData::new(width, height, 10.0, heights).unwrap()
}

fn read_rendered_pixels(ctx: &GpuContext, terrain: &TerrainData) -> Vec<[u8; 4]> {
    render_with(ctx, terrain, |_, _| {})
}

fn render_with(
    ctx: &GpuContext,
    terrain: &TerrainData,
    configure: impl FnOnce(&mut Renderer, &wgpu::Device),
) -> Vec<[u8; 4]> {
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Test Target"),
        size: wgpu::Extent3d {
            width: TARGET_SIZE,
            height: TARGET_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let error_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mut renderer = Renderer::new(
        &ctx.device,
        &ctx.queue,
        FORMAT,
        TARGET_SIZE,
        TARGET_SIZE,
        terrain,
    );
    configure(&mut renderer, &ctx.device);
    renderer.render(&ctx.device, &ctx.queue, &target_view);

    // TARGET_SIZE * 4 bytes is exactly the 256 byte row alignment wgpu requires.
    let bytes_per_row = TARGET_SIZE * 4;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Readback"),
        size: (bytes_per_row * TARGET_SIZE) as u64,
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
                rows_per_image: Some(TARGET_SIZE),
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
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("failed to poll device");
    receiver
        .recv()
        .expect("map callback dropped")
        .expect("failed to map readback buffer");

    if let Some(error) = pollster::block_on(error_scope.pop()) {
        panic!("GPU validation error while rendering the DEM: {error}");
    }

    let data = slice
        .get_mapped_range()
        .expect("failed to read mapped range");
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    staging.unmap();
    pixels
}

#[test]
fn renders_dem_surface_with_shading() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let terrain = sloped_terrain(32, 32);
    let pixels = read_rendered_pixels(&ctx, &terrain);
    assert_eq!(pixels.len(), (TARGET_SIZE * TARGET_SIZE) as usize);

    // The clear colour is a dark blue; anything else must have come from the terrain pass.
    let clear = [13u8, 18, 28, 255];
    let terrain_pixels = pixels.iter().filter(|p| **p != clear).count();
    assert!(
        terrain_pixels > pixels.len() / 10,
        "expected the DEM to cover a meaningful part of the frame, got {terrain_pixels} of {}",
        pixels.len()
    );

    let distinct_shades: std::collections::HashSet<_> =
        pixels.iter().filter(|p| **p != clear).collect();
    assert!(
        distinct_shades.len() > 1,
        "expected elevation shading to produce varying colours"
    );
}

#[test]
fn flat_and_sloped_terrain_render_differently() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let sloped = read_rendered_pixels(&ctx, &sloped_terrain(32, 32));
    let flat = read_rendered_pixels(
        &ctx,
        &TerrainData::new(32, 32, 10.0, vec![1500.0; 32 * 32]).unwrap(),
    );

    assert_ne!(
        sloped, flat,
        "terrain geometry should influence the rendered image"
    );
}

fn storage_buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Storage"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[test]
fn grid_overlay_tints_only_cells_above_the_threshold() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let terrain = sloped_terrain(32, 32);
    let plain = read_rendered_pixels(&ctx, &terrain);

    // Only the second half of the grid carries flow, so part of the terrain must stay bare.
    let field: Vec<f32> = (0..32 * 32)
        .map(|i| if i >= 32 * 16 { 8.0 } else { 0.0 })
        .collect();

    let overlaid = render_with(&ctx, &terrain, |renderer, device| {
        let buffer = storage_buffer(device, &field);
        renderer.set_grid_overlay(device, Some(&buffer), OverlayRange::new(0.0, 8.0));
    });

    assert_ne!(plain, overlaid, "the overlay should change the image");

    let changed = plain.iter().zip(&overlaid).filter(|(a, b)| a != b).count();
    let unchanged_terrain = plain
        .iter()
        .zip(&overlaid)
        .filter(|(a, b)| a == b && **a != [13u8, 18, 28, 255])
        .count();

    assert!(changed > 0, "flow cells should be tinted");
    assert!(
        unchanged_terrain > 0,
        "cells below the threshold should keep plain terrain shading"
    );
}

#[test]
fn disabling_the_overlay_restores_plain_terrain() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let terrain = sloped_terrain(32, 32);
    let plain = read_rendered_pixels(&ctx, &terrain);
    let field = vec![8.0f32; 32 * 32];

    let toggled = render_with(&ctx, &terrain, |renderer, device| {
        let buffer = storage_buffer(device, &field);
        renderer.set_grid_overlay(device, Some(&buffer), OverlayRange::new(0.0, 8.0));
        renderer.set_grid_overlay(device, None, OverlayRange::default());
    });

    assert_eq!(plain, toggled);
}

#[test]
fn particles_are_drawn_on_top_of_the_terrain() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let terrain = sloped_terrain(32, 32);
    let plain = read_rendered_pixels(&ctx, &terrain);

    // A diagonal line of particles following the sloped surface. The terrain height
    // at (i * 10, i * 10) is 1000 + (i + i) * 1.5, matching the cell centres.
    let count = 32u32;
    let positions: Vec<[f32; 2]> = (0..count)
        .map(|i| [i as f32 * 10.0, i as f32 * 10.0])
        .collect();
    let velocities: Vec<[f32; 2]> = (0..count).map(|i| [i as f32, 0.0]).collect();
    let stopped = vec![0u32; count as usize];
    let elevations: Vec<f32> = (0..count).map(|i| 1000.0 + i as f32 * 3.0).collect();

    let render_particles = |velocities_z: &[f32]| {
        render_with(&ctx, &terrain, |renderer, device| {
            let position = storage_buffer(device, &positions);
            let velocity = storage_buffer(device, &velocities);
            let velocity_z = storage_buffer(device, velocities_z);
            let stopped = storage_buffer(device, &stopped);
            let elevation = storage_buffer(device, &elevations);

            renderer.set_particles(
                device,
                Some(ParticleBuffers {
                    position: &position,
                    velocity: &velocity,
                    velocity_z: &velocity_z,
                    stopped: &stopped,
                    elevation: &elevation,
                }),
            );
            renderer.particles_mut().set_count(count);
            renderer.particles_mut().set_max_velocity(count as f32);
        })
    };

    // A vertical component raises the total speed, shifting particles up the colour
    // ramp; a zero-filled buffer (the MPM stand-in) keeps horizontal-speed colouring.
    let flat = render_particles(&vec![0.0f32; count as usize]);
    let plunging = render_particles(&vec![8.0f32; count as usize]);

    let changed = plain.iter().zip(&flat).filter(|(a, b)| a != b).count();
    assert!(
        changed > 20,
        "expected particles to cover pixels, only {changed} differed"
    );
    assert_ne!(
        flat, plunging,
        "vertical velocity should contribute to the speed colouring"
    );
}

#[test]
fn particles_without_buffers_draw_nothing() {
    let Ok(ctx) = pollster::block_on(GpuContext::headless()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let terrain = sloped_terrain(32, 32);
    let plain = read_rendered_pixels(&ctx, &terrain);
    let unattached = render_with(&ctx, &terrain, |renderer, _| {
        renderer.particles_mut().set_count(1000);
    });

    assert_eq!(plain, unattached);
}
