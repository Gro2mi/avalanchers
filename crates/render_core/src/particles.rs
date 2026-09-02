use crate::camera::OrbitCamera;

/// The simulation storage buffers needed to draw particles.
///
/// The buffers stay owned by the simulation; the renderer only borrows them to build a
/// bind group, so particle state never round-trips through the CPU.
#[derive(Clone, Copy)]
pub struct ParticleBuffers<'a> {
    /// `array<vec2<f32>>` of positions in the DEM plane, in metres.
    pub position: &'a wgpu::Buffer,
    /// `array<vec2<f32>>` of horizontal velocities, used for colouring.
    pub velocity: &'a wgpu::Buffer,
    /// `array<f32>` of vertical velocities, used for colouring. Models without one
    /// (MPM) can pass a zero-filled buffer to colour by horizontal speed only.
    pub velocity_z: &'a wgpu::Buffer,
    /// `array<u32>` where a non-zero entry marks a stopped particle.
    pub stopped: &'a wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleUniforms {
    view_proj: [[f32; 4]; 4],
    /// camera right vector, particle radius in `w`
    right: [f32; 4],
    /// camera up vector, vertical exaggeration in `w`
    up: [f32; 4],
    /// particle count, velocity at the top of the colour ramp
    params: [f32; 4],
    /// grid width, grid height, cell size, height offset above the surface
    grid: [f32; 4],
}

/// Draws particles as camera-facing discs, expanded on the GPU from the vertex index.
/// Particle height comes from the DEM, because the simulation only writes back the
/// horizontal position.
pub struct ParticleRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    heightmap: wgpu::TextureView,
    uniforms: ParticleUniforms,
    count: u32,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        heightmap: wgpu::TextureView,
        terrain: &crate::terrain::TerrainData,
    ) -> Self {
        let radius = terrain.cell_size();
        let uniforms = ParticleUniforms {
            view_proj: crate::math::IDENTITY,
            right: [1.0, 0.0, 0.0, radius],
            up: [0.0, 1.0, 0.0, terrain.vertical_exaggeration()],
            params: [0.0, 1.0, 0.0, 0.0],
            grid: [
                terrain.width() as f32,
                terrain.height() as f32,
                terrain.cell_size(),
                radius * 0.5,
            ],
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Uniforms"),
            size: size_of::<ParticleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(5),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particles.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: None,
            uniform_buffer,
            heightmap,
            uniforms,
            count: 0,
        }
    }

    /// Attaches simulation buffers. Passing `None` stops drawing particles.
    pub fn set_buffers(&mut self, device: &wgpu::Device, buffers: Option<ParticleBuffers<'_>>) {
        self.bind_group = buffers.map(|buffers| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.position.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.velocity.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.stopped.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.heightmap),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.velocity_z.as_entire_binding(),
                    },
                ],
            })
        });
    }

    /// Number of particles to draw. Must not exceed the length of the attached buffers.
    pub fn set_count(&mut self, count: u32) {
        self.count = count;
        self.uniforms.params[0] = count as f32;
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.uniforms.right[3] = radius;
        self.uniforms.grid[3] = radius * 0.5;
    }

    /// Velocity mapped to the top of the colour ramp, in m/s.
    pub fn set_max_velocity(&mut self, max_velocity: f32) {
        self.uniforms.params[1] = max_velocity.max(1e-6);
    }

    pub fn update_camera(&mut self, queue: &wgpu::Queue, camera: &OrbitCamera) {
        let (right, up) = camera.billboard_axes();
        self.uniforms.view_proj = camera.view_projection();
        self.uniforms.right = [right.x, right.y, right.z, self.uniforms.right[3]];
        self.uniforms.up = [up.x, up.y, up.z, self.uniforms.up[3]];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..self.count * 6, 0..1);
    }
}
