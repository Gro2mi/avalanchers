//! Colour bar legend for the active grid overlay.
//!
//! Draws the overlay's colour ramp as a vertical bar against the right edge of the frame,
//! with min/mid/max value labels, a notch at the overlay threshold, and the range's unit
//! rotated to read bottom-up beside the bar. The layout is rebuilt on the CPU whenever the
//! range, unit or viewport size changes; the GPU side is a single draw over a static font
//! atlas.

use crate::terrain::OverlayRange;

/// 5x7 pixel font. Rows run top to bottom, `#` marks a lit pixel. Covers the digits and
/// the lowercase alphabet so legend labels like "peak flow velocity (m/s)" render.
const GLYPHS: [(char, [&str; 7]); 42] = [
    (
        '0',
        [
            ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
        ],
    ),
    (
        '1',
        [
            "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
        ],
    ),
    (
        '2',
        [
            ".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####",
        ],
    ),
    (
        '3',
        [
            ".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###.",
        ],
    ),
    (
        '4',
        [
            "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#.",
        ],
    ),
    (
        '5',
        [
            "#####", "#....", "####.", "....#", "....#", "#...#", ".###.",
        ],
    ),
    (
        '6',
        [
            "..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###.",
        ],
    ),
    (
        '7',
        [
            "#####", "....#", "...#.", "..#..", "..#..", "..#..", "..#..",
        ],
    ),
    (
        '8',
        [
            ".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###.",
        ],
    ),
    (
        '9',
        [
            ".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##..",
        ],
    ),
    (
        '.',
        [
            ".....", ".....", ".....", ".....", ".....", ".##..", ".##..",
        ],
    ),
    (
        '-',
        [
            ".....", ".....", ".....", "#####", ".....", ".....", ".....",
        ],
    ),
    (
        '/',
        [
            "....#", "....#", "...#.", "..#..", ".#...", "#....", "#....",
        ],
    ),
    (
        ' ',
        [
            ".....", ".....", ".....", ".....", ".....", ".....", ".....",
        ],
    ),
    (
        '(',
        [
            "...#.", "..#..", "..#..", "..#..", "..#..", "..#..", "...#.",
        ],
    ),
    (
        ')',
        [
            ".#...", "..#..", "..#..", "..#..", "..#..", "..#..", ".#...",
        ],
    ),
    (
        'a',
        [
            ".....", ".....", ".###.", "....#", ".####", "#...#", ".####",
        ],
    ),
    (
        'b',
        [
            "#....", "#....", "####.", "#...#", "#...#", "#...#", "####.",
        ],
    ),
    (
        'c',
        [
            ".....", ".....", ".####", "#....", "#....", "#....", ".####",
        ],
    ),
    (
        'd',
        [
            "....#", "....#", ".###.", "#...#", "#...#", "#...#", ".####",
        ],
    ),
    (
        'e',
        [
            ".....", ".....", ".###.", "#...#", "#####", "#....", ".###.",
        ],
    ),
    (
        'f',
        [
            "..##.", ".#...", "#####", ".#...", ".#...", ".#...", ".#...",
        ],
    ),
    (
        'g',
        [
            ".....", ".###.", "#...#", "#...#", ".####", "....#", ".###.",
        ],
    ),
    (
        'h',
        [
            "#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#",
        ],
    ),
    (
        'i',
        [
            "..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###.",
        ],
    ),
    (
        'j',
        [
            "...##", ".....", "..##.", "...#.", "...#.", "...#.", "###..",
        ],
    ),
    (
        'k',
        [
            "#....", "#....", "#..#.", "#.#..", "##...", "#.#..", "#..#.",
        ],
    ),
    (
        'l',
        [
            ".#...", ".#...", ".#...", ".#...", ".#...", ".#...", "#####",
        ],
    ),
    (
        'm',
        [
            ".....", ".....", "#...#", "##.##", "#.#.#", "#.#.#", "#...#",
        ],
    ),
    (
        'n',
        [
            ".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#",
        ],
    ),
    (
        'o',
        [
            ".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.",
        ],
    ),
    (
        'p',
        [
            ".....", "####.", "#...#", "#...#", "####.", "#....", "#....",
        ],
    ),
    (
        'q',
        [
            ".....", ".###.", "#...#", "#...#", ".####", "....#", "....#",
        ],
    ),
    (
        'r',
        [
            ".....", ".....", "#.##.", "##..#", "#....", "#....", "#....",
        ],
    ),
    (
        's',
        [
            ".....", ".....", ".####", "#....", ".###.", "....#", "####.",
        ],
    ),
    (
        't',
        [
            ".#...", ".#...", "#####", ".#...", ".#...", ".#...", "..##.",
        ],
    ),
    (
        'u',
        [
            ".....", ".....", "#...#", "#...#", "#...#", "#...#", ".####",
        ],
    ),
    (
        'v',
        [
            ".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#..",
        ],
    ),
    (
        'w',
        [
            ".....", ".....", "#...#", "#...#", "#.#.#", "#.#.#", ".#.#.",
        ],
    ),
    (
        'x',
        [
            ".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#",
        ],
    ),
    (
        'y',
        [
            ".....", "#...#", "#...#", "#...#", ".####", "....#", ".###.",
        ],
    ),
    (
        'z',
        [
            ".....", ".....", "#####", "...#.", "..#..", ".#...", "#####",
        ],
    ),
];

const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;
const GLYPH_ADVANCE: u32 = 6;
/// A white texel at (0, 0) serves as the fill for solid quads; glyph cells follow it.
const ATLAS_W: u32 = 2 + GLYPHS.len() as u32 * GLYPH_ADVANCE;
const ATLAS_H: u32 = GLYPH_H + 1;
const MAX_VERTICES: usize = 4096;

const BAR_WIDTH: f32 = 12.0;
const TICK_LENGTH: f32 = 6.0;

const WHITE: [f32; 4] = [0.92, 0.93, 0.95, 0.9];
const BORDER: [f32; 4] = [0.92, 0.93, 0.95, 0.8];
const BACKDROP: [f32; 4] = [0.03, 0.05, 0.08, 0.6];

const WHITE_UV: [f32; 2] = [0.5 / ATLAS_W as f32, 0.5 / ATLAS_H as f32];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorbarVertex {
    /// Normalised device coordinates.
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

fn find_glyph(ch: char) -> Option<usize> {
    GLYPHS.iter().position(|&(c, _)| c == ch)
}

/// UV corners of a glyph cell, inset by a quarter texel so nearest sampling cannot
/// bleed into the neighbouring cell.
fn glyph_uv(index: usize) -> [[f32; 2]; 4] {
    let x0 = (2 + index as u32 * GLYPH_ADVANCE) as f32;
    let eps = 0.25;
    let u0 = (x0 + eps) / ATLAS_W as f32;
    let u1 = (x0 + GLYPH_W as f32 - eps) / ATLAS_W as f32;
    let v0 = eps / ATLAS_H as f32;
    let v1 = (GLYPH_H as f32 - eps) / ATLAS_H as f32;
    // top-left, top-right, bottom-left, bottom-right
    [[u0, v0], [u1, v0], [u0, v1], [u1, v1]]
}

/// The ramp from `overlay_color` in `terrain.wgsl`; keep the two in sync.
fn ramp_color(t: f32) -> [f32; 4] {
    const COLD: [f32; 3] = [0.13, 0.36, 0.86];
    const MID: [f32; 3] = [0.15, 0.82, 0.75];
    const WARM: [f32; 3] = [0.98, 0.82, 0.20];
    const HOT: [f32; 3] = [0.86, 0.14, 0.10];
    let mix = |a: [f32; 3], b: [f32; 3], k: f32| {
        [
            a[0] + (b[0] - a[0]) * k,
            a[1] + (b[1] - a[1]) * k,
            a[2] + (b[2] - a[2]) * k,
            1.0,
        ]
    };
    if t < 0.33 {
        mix(COLD, MID, t / 0.33)
    } else if t < 0.66 {
        mix(MID, WARM, (t - 0.33) / 0.33)
    } else {
        mix(WARM, HOT, (t - 0.66) / 0.34)
    }
}

fn font_scale(height: u32) -> u32 {
    (((height as f32 / 480.0).round() as i32).clamp(1, 3)) as u32
}

fn text_width(text: &str, scale: u32) -> f32 {
    let n = text.chars().count();
    if n == 0 {
        return 0.0;
    }
    (n as f32 * GLYPH_ADVANCE as f32 - 1.0) * scale as f32
}

fn format_value(v: f32) -> String {
    let a = v.abs();
    if a < 0.005 {
        "0".to_string()
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

struct QuadSink {
    vertices: Vec<ColorbarVertex>,
}

impl QuadSink {
    /// `corners`, `uv` and `colors` are ordered top-left, top-right, bottom-left,
    /// bottom-right in pixel coordinates.
    fn quad(&mut self, corners: [[f32; 2]; 4], uv: [[f32; 2]; 4], colors: [[f32; 4]; 4]) {
        for i in [0usize, 1, 2, 1, 3, 2] {
            self.vertices.push(ColorbarVertex {
                pos: corners[i],
                uv: uv[i],
                color: colors[i],
            });
        }
    }

    fn solid(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
        self.quad(
            [[x0, y0], [x1, y0], [x0, y1], [x1, y1]],
            [WHITE_UV; 4],
            [color; 4],
        );
    }

    /// Horizontal text with its top-left corner at (`x`, `y`). Returns the advance width.
    fn text(&mut self, text: &str, x: f32, y: f32, scale: u32, color: [f32; 4]) -> f32 {
        let s = scale as f32;
        let glyph_h = GLYPH_H as f32 * s;
        let mut pen = x;
        for ch in text.chars() {
            if let Some(index) = find_glyph(ch) {
                let x1 = pen + GLYPH_W as f32 * s;
                self.quad(
                    [[pen, y], [x1, y], [pen, y + glyph_h], [x1, y + glyph_h]],
                    glyph_uv(index),
                    [color; 4],
                );
            }
            pen += GLYPH_ADVANCE as f32 * s;
        }
        text_width(text, scale)
    }

    /// Text rotated to read bottom-up along the bar. (`left`, `bottom`) is the corner the
    /// first glyph's top-left maps to: the leftmost, lowest point of the block.
    fn text_rotated(&mut self, text: &str, left: f32, bottom: f32, scale: u32, color: [f32; 4]) {
        let s = scale as f32;
        // Glyph-local (lx, ly) with lx along the text and ly down the glyph maps to
        // (left + ly * s, bottom - lx * s): the text runs upwards, glyph tops face left.
        let point = |lx: f32, ly: f32| [left + ly * s, bottom - lx * s];
        for (k, ch) in text.chars().enumerate() {
            if let Some(index) = find_glyph(ch) {
                let lx0 = k as f32 * GLYPH_ADVANCE as f32;
                let lx1 = lx0 + GLYPH_W as f32;
                self.quad(
                    [
                        point(lx0, 0.0),
                        point(lx1, 0.0),
                        point(lx0, GLYPH_H as f32),
                        point(lx1, GLYPH_H as f32),
                    ],
                    glyph_uv(index),
                    [color; 4],
                );
            }
        }
    }
}

fn build_atlas() -> Vec<u8> {
    let mut data = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    data[0] = 255; // white fill texel
    for (i, (_, rows)) in GLYPHS.iter().enumerate() {
        let x0 = 2 + i as u32 * GLYPH_ADVANCE;
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    data[(y as u32 * ATLAS_W + x0 + x as u32) as usize] = 255;
                }
            }
        }
    }
    data
}

/// Lays out the whole bar in pixel coordinates, then converts positions to NDC.
fn build_vertices(width: u32, height: u32, range: &OverlayRange) -> Vec<ColorbarVertex> {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let scale = font_scale(height);
    let s = scale as f32;
    let glyph_h = GLYPH_H as f32 * s;

    // The legend reads "variable (unit)" along the bar, falling back to whichever
    // of the two is set.
    let legend = if !range.label.is_empty() && !range.unit.is_empty() {
        format!("{} ({})", range.label, range.unit)
    } else if !range.label.is_empty() {
        range.label.to_string()
    } else {
        range.unit.to_string()
    };
    let has_legend = !legend.is_empty();

    let margin = if has_legend {
        14.0 + GLYPH_H as f32 * s
    } else {
        12.0
    };
    let bar_x1 = (w - margin).max(BAR_WIDTH + 2.0);
    let bar_x0 = bar_x1 - BAR_WIDTH;
    let bar_h = (h * 0.45).clamp(96.0, 480.0).min(h - 16.0).max(8.0);
    let bar_y0 = (h - bar_h) * 0.5;
    let bar_y1 = bar_y0 + bar_h;
    let span = (range.max - range.min).abs().max(1e-6);
    let y_for = |v: f32| bar_y1 - (v - range.min) / span * bar_h;

    let mut sink = QuadSink {
        vertices: Vec::new(),
    };

    // The shader ramp is piecewise linear with stops at 1/3 and 2/3, so three segments
    // with per-vertex stop colours reproduce it exactly.
    for i in 0..3u32 {
        let t0 = f64::from(i) / 3.0;
        let t1 = f64::from(i + 1) / 3.0;
        let y_top = bar_y1 - t1 as f32 * bar_h;
        let y_bottom = bar_y1 - t0 as f32 * bar_h;
        let top = ramp_color(t1 as f32);
        let bottom = ramp_color(t0 as f32);
        sink.quad(
            [
                [bar_x0, y_top],
                [bar_x1, y_top],
                [bar_x0, y_bottom],
                [bar_x1, y_bottom],
            ],
            [WHITE_UV; 4],
            [top, top, bottom, bottom],
        );
    }

    // Ticks with value labels at both ends of the bar, plus the midpoint when it fits.
    let mut ticks = vec![range.max, range.min];
    if bar_h >= 90.0 {
        ticks.insert(1, range.min + (range.max - range.min) * 0.5);
    }
    for v in ticks {
        let y = y_for(v).clamp(bar_y0, bar_y1);
        sink.solid(
            bar_x0 - TICK_LENGTH - 2.0,
            y - 0.5,
            bar_x0 - 2.0,
            y + 0.5,
            WHITE,
        );
        let label = format_value(v);
        let lx1 = bar_x0 - TICK_LENGTH - 6.0;
        let lx0 = lx1 - text_width(&label, scale);
        sink.solid(
            lx0 - 4.0,
            y - glyph_h * 0.5 - 2.0,
            lx1 + 2.0,
            y + glyph_h * 0.5 + 2.0,
            BACKDROP,
        );
        sink.text(&label, lx0, y - glyph_h * 0.5, scale, WHITE);
    }

    // Notch marking where the overlay starts tinting the terrain.
    if range.threshold > range.min && range.threshold < range.max {
        let y = y_for(range.threshold).clamp(bar_y0 + 2.0, bar_y1 - 2.0);
        sink.solid(bar_x0, y - 1.0, bar_x1, y + 1.0, [1.0, 1.0, 1.0, 0.95]);
    }

    if has_legend {
        let legend_w = text_width(&legend, scale);
        let left = bar_x1 + 8.0;
        let bottom = bar_y0 + bar_h * 0.5 + legend_w * 0.5;
        sink.solid(
            left - 2.0,
            bottom - legend_w - 2.0,
            left + GLYPH_H as f32 * s + 2.0,
            bottom + 2.0,
            BACKDROP,
        );
        sink.text_rotated(&legend, left, bottom, scale, WHITE);
    }

    sink.solid(bar_x0 - 1.0, bar_y0 - 1.0, bar_x1 + 1.0, bar_y0, BORDER);
    sink.solid(bar_x0 - 1.0, bar_y1, bar_x1 + 1.0, bar_y1 + 1.0, BORDER);
    sink.solid(bar_x0 - 1.0, bar_y0, bar_x0, bar_y1, BORDER);
    sink.solid(bar_x1, bar_y0, bar_x1 + 1.0, bar_y1, BORDER);

    for v in sink.vertices.iter_mut() {
        v.pos = [v.pos[0] / w * 2.0 - 1.0, 1.0 - v.pos[1] / h * 2.0];
    }
    sink.vertices
}

/// Draws the overlay legend on top of the scene, inside the scene's render pass.
pub(crate) struct ColorbarRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<ColorbarVertex>,
    enabled: bool,
    range: OverlayRange,
    built_for: Option<(bool, OverlayRange, u32, u32)>,
}

impl ColorbarRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ATLAS_W,
            height: ATLAS_H,
            depth_or_array_layers: 1,
        };
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colorbar Font Atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &build_atlas(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_W),
                rows_per_image: Some(ATLAS_H),
            },
            size,
        );
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Colorbar Font Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Colorbar Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Colorbar Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Colorbar Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/colorbar.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Colorbar Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Colorbar Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<ColorbarVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                        },
                        wgpu::VertexAttribute {
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                        },
                        wgpu::VertexAttribute {
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // The scene pass carries a depth attachment, so the legend must match its
            // format while never testing or writing depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Colorbar Vertices"),
            size: (MAX_VERTICES * size_of::<ColorbarVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            vertex_buffer,
            vertices: Vec::new(),
            enabled: false,
            range: OverlayRange::default(),
            built_for: None,
        }
    }

    pub(crate) fn set_enabled_range(&mut self, enabled: bool, range: OverlayRange) {
        self.enabled = enabled;
        self.range = range;
    }

    pub(crate) fn set_range(&mut self, range: OverlayRange) {
        self.range = range;
    }

    /// Uploads new bar geometry when the state or viewport changed since the last build.
    pub(crate) fn prepare(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        let key = (self.enabled, self.range, width, height);
        if self.built_for == Some(key) {
            return;
        }
        self.built_for = Some(key);
        self.vertices = if self.enabled {
            build_vertices(width, height, &self.range)
        } else {
            Vec::new()
        };
        debug_assert!(self.vertices.len() <= MAX_VERTICES);
        self.vertices.truncate(MAX_VERTICES);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.vertices.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_covers_units_and_numbers() {
        for ch in "0123456789./- ()abcdefghijklmnopqrstuvwxyz".chars() {
            assert!(find_glyph(ch).is_some(), "missing glyph {ch:?}");
        }
        assert_eq!(find_glyph('Q'), None);
    }

    #[test]
    fn legend_text_renders_for_full_variable_names() {
        // Every character of every live_sim overlay label must exist in the font.
        for text in [
            "peak flow velocity (m/s)",
            "peak flow thickness (m)",
            "grid mass (kg)",
            "release areas (m)",
            "slope angle (deg)",
            "slope aspect (deg)",
            "roughness",
        ] {
            for ch in text.chars() {
                assert!(find_glyph(ch).is_some(), "missing glyph {ch:?} in {text:?}");
            }
        }
    }

    #[test]
    fn ramp_matches_the_terrain_shader_stops() {
        let assert_rgb = |got: [f32; 4], want: [f32; 3]| {
            for i in 0..3 {
                assert!((got[i] - want[i]).abs() < 1e-6);
            }
            assert_eq!(got[3], 1.0);
        };
        assert_rgb(ramp_color(0.0), [0.13, 0.36, 0.86]);
        assert_rgb(ramp_color(0.33), [0.15, 0.82, 0.75]);
        assert_rgb(ramp_color(0.66), [0.98, 0.82, 0.20]);
        assert_rgb(ramp_color(1.0), [0.86, 0.14, 0.10]);
        // Halfway into the first segment is the average of its stops.
        let mid = ramp_color(0.165);
        assert!((mid[0] - (0.13 + 0.15) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn formats_values_by_magnitude() {
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(0.01), "0.01");
        assert_eq!(format_value(-0.5), "-0.50");
        assert_eq!(format_value(8.0), "8.00");
        assert_eq!(format_value(30.0), "30.0");
        assert_eq!(format_value(360.0), "360");
        assert_eq!(format_value(5_000.0), "5000");
    }

    #[test]
    fn layout_produces_finite_quads_for_sane_inputs() {
        for (w, h) in [(1920u32, 1080u32), (64, 64), (2560, 1440), (1, 1)] {
            let range = OverlayRange::new(0.0, 30.0)
                .with_threshold(0.1)
                .with_label("peak flow velocity")
                .with_unit("m/s");
            let vertices = build_vertices(w, h, &range);
            assert!(!vertices.is_empty(), "no geometry for {w}x{h}");
            assert_eq!(vertices.len() % 6, 0, "each quad is two triangles");
            // Quads may leave the frame on degenerately small viewports; the GPU clips
            // them, but coordinates must stay finite and UVs in range.
            let in_frame = w >= 128 && h >= 128;
            for v in &vertices {
                assert!(v.pos[0].is_finite() && v.pos[1].is_finite());
                if in_frame {
                    assert!(v.pos[0] >= -1.5 && v.pos[0] <= 1.5, "x out of frame");
                    assert!(v.pos[1] >= -1.5 && v.pos[1] <= 1.5, "y out of frame");
                }
                assert!(v.uv[0] >= 0.0 && v.uv[0] <= 1.0);
                assert!(v.uv[1] >= 0.0 && v.uv[1] <= 1.0);
            }
        }
    }

    #[test]
    fn layout_places_max_above_min() {
        // In NDC, +1 is the top of the frame. The hot end (max) must sit above the cold end.
        let range = OverlayRange::new(0.0, 30.0);
        let vertices = build_vertices(800, 600, &range);
        let ys: Vec<f32> = vertices.iter().map(|v| v.pos[1]).collect();
        let top = ys.iter().cloned().fold(f32::MIN, f32::max);
        let bottom = ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!(top > bottom);
    }
}
