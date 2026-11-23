use super::*;

pub(super) struct PipelineBundle {
    pub(super) bg_pipeline: wgpu::RenderPipeline,
    pub(super) surface_indexed_opaque_pipeline: wgpu::RenderPipeline,
    pub(super) surface_indexed_alpha_pipeline: wgpu::RenderPipeline,
    pub(super) surface_edges_opaque_pipeline: wgpu::RenderPipeline,
    pub(super) surface_edges_alpha_pipeline: wgpu::RenderPipeline,
    pub(super) line_pipeline: wgpu::RenderPipeline,
    pub(super) point_pipeline: wgpu::RenderPipeline,
    pub(super) volume_pipeline: wgpu::RenderPipeline,
    pub(super) widget_tri_pipeline: wgpu::RenderPipeline,
    pub(super) widget_line_pipeline: wgpu::RenderPipeline,
    pub(super) gizmo_pipeline: wgpu::RenderPipeline,
}


pub(super) fn build_cached_geometry(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dataset: &DataSet,
    style: &ActorStyle,
) -> CachedGeometry {
    match style.representation {
        Representation::Surface => {
            let prep = mesh_prep::prepare_surface_indexed_with_style(dataset, style);
            if prep.indices.is_empty() || prep.geom_vertices.is_empty() {
                return CachedGeometry::Empty;
            }
            let geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-surface-geom-vb"),
                contents: bytemuck::cast_slice(&prep.geom_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-surface-scalar-vb"),
                contents: bytemuck::cast_slice(&prep.scalars),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-surface-ib"),
                contents: bytemuck::cast_slice(&prep.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            CachedGeometry::SurfaceIndexed {
                geom_buffer,
                scalar_buffer,
                index_buffer,
                vertex_count: prep.geom_vertices.len() as u32,
                index_count: prep.indices.len() as u32,
                scalar_source: prep.scalar_source,
                auto_range: prep.scalar_range,
            }
        }
        Representation::SurfaceWithEdges => {
            let prep = mesh_prep::prepare_surface_with_style(dataset, style);
            if prep.vertices.is_empty() {
                return CachedGeometry::Empty;
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-surface-edges-vb"),
                contents: bytemuck::cast_slice(&prep.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            CachedGeometry::SurfaceEdges {
                buffer,
                vertex_count: prep.vertices.len() as u32,
                auto_range: prep.scalar_range,
            }
        }
        Representation::Wireframe => {
            let prep = mesh_prep::prepare_wireframe_with_style(dataset, style);
            if prep.vertices.is_empty() {
                return CachedGeometry::Empty;
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-wireframe-vb"),
                contents: bytemuck::cast_slice(&prep.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            CachedGeometry::Lines {
                buffer,
                vertex_count: prep.vertices.len() as u32,
                auto_range: prep.scalar_range,
            }
        }
        Representation::Outline => {
            let verts = mesh_prep::prepare_outline(&style.dataset_bounds(dataset));
            if verts.is_empty() {
                return CachedGeometry::Empty;
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-outline-vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            CachedGeometry::Lines {
                buffer,
                vertex_count: verts.len() as u32,
                auto_range: (0.0, 1.0),
            }
        }
        Representation::Points | Representation::PointGaussian => {
            let prep = mesh_prep::prepare_points_with_style(dataset, style);
            if prep.instances.is_empty() {
                return CachedGeometry::Empty;
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-points-ib"),
                contents: bytemuck::cast_slice(&prep.instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
            CachedGeometry::Points {
                buffer,
                instance_count: prep.instances.len() as u32,
                auto_range: prep.scalar_range,
            }
        }
        Representation::Volume => {
            let Some(vol) = mesh_prep::prepare_volume_with_style(dataset, style) else {
                return CachedGeometry::Empty;
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("paraviz-volume-3d"),
                size: wgpu::Extent3d {
                    width: vol.dims[0],
                    height: vol.dims[1],
                    depth_or_array_layers: vol.dims[2],
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&vol.data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(vol.dims[0] * 4),
                    rows_per_image: Some(vol.dims[1]),
                },
                wgpu::Extent3d {
                    width: vol.dims[0],
                    height: vol.dims[1],
                    depth_or_array_layers: vol.dims[2],
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            CachedGeometry::Volume {
                dims: vol.dims,
                box_min: vol.box_min,
                box_max: vol.box_max,
                texture,
                view,
                auto_range: vol.scalar_range,
            }
        }
    }
}


pub(super) fn create_lut_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    label: &str,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: LUT_WIDTH as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    upload_lut_texture(queue, &tex, rgba);
    tex
}

pub(super) fn upload_lut_texture(queue: &wgpu::Queue, tex: &wgpu::Texture, rgba: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((LUT_WIDTH * 4) as u32),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: LUT_WIDTH as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

pub(super) fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
    texture_binding: bool,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    if texture_binding {
        usage |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paraviz-depth-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(super) fn create_render_targets(device: &wgpu::Device, width: u32, height: u32) -> RenderTargets {
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paraviz-offscreen-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (depth_texture, depth_view) = create_depth_texture(device, width, height, 1, false);
    RenderTargets {
        width,
        height,
        color_texture,
        color_view,
        depth_texture,
        depth_view,
    }
}

pub(super) fn create_msaa_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    depth_resolve_bgl: &wgpu::BindGroupLayout,
) -> MsaaTargets {
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paraviz-msaa4x-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (depth_texture, depth_view) = create_depth_texture(device, width, height, 4, true);
    let depth_resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paraviz-depth-resolve-bg"),
        layout: depth_resolve_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&depth_view),
        }],
    });
    MsaaTargets {
        width,
        height,
        format,
        color_texture,
        color_view,
        depth_texture,
        depth_view,
        depth_resolve_bg,
    }
}

pub(super) fn create_depth_resolve_pipeline(
    device: &wgpu::Device,
    depth_resolve_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-depth-resolve-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::DEPTH_RESOLVE_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paraviz-depth-resolve-layout"),
        bind_group_layouts: &[Some(depth_resolve_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-depth-resolve-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn create_edl_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    edl_tex_bgl: &wgpu::BindGroupLayout,
) -> EdlTargets {
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paraviz-edl-scene-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (depth_texture, depth_view) = create_depth_texture(device, width, height, 1, true);
    let tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paraviz-edl-tex-bg"),
        layout: edl_tex_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
        ],
    });
    EdlTargets {
        width,
        height,
        format,
        color_texture,
        color_view,
        depth_texture,
        depth_view,
        tex_bind_group,
    }
}

pub(super) fn create_edl_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    edl_tex_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let edl_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-edl-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::EDL_WGSL.into()),
    });
    let edl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paraviz-edl-layout"),
        bind_group_layouts: &[Some(uniform_bgl), Some(edl_tex_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-edl-pipeline"),
        layout: Some(&edl_layout),
        vertex: wgpu::VertexState {
            module: &edl_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &edl_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn create_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
    uniform_bgl: &wgpu::BindGroupLayout,
    lut_bgl: &wgpu::BindGroupLayout,
    volume_bgl: &wgpu::BindGroupLayout,
) -> PipelineBundle {
    let multisample = wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let bg_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-bg-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::BACKGROUND_WGSL.into()),
    });
    let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-surface-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::SURFACE_WGSL.into()),
    });
    let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-line-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::LINE_WGSL.into()),
    });
    let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-point-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::POINT_WGSL.into()),
    });
    let volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-volume-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::VOLUME_WGSL.into()),
    });
    let widget_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-widget-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::WIDGET_WGSL.into()),
    });
    let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paraviz-gizmo-shader"),
        source: wgpu::ShaderSource::Wgsl(shaders::GIZMO_WGSL.into()),
    });

    let uniform_only_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paraviz-uniform-only-layout"),
        bind_group_layouts: &[Some(uniform_bgl)],
        immediate_size: 0,
    });
    let actor_lut_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paraviz-actor-lut-layout"),
        bind_group_layouts: &[Some(uniform_bgl), Some(lut_bgl)],
        immediate_size: 0,
    });
    let volume_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paraviz-volume-layout"),
        bind_group_layouts: &[Some(uniform_bgl), Some(volume_bgl)],
        immediate_size: 0,
    });

    let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-bg-pipeline"),
        layout: Some(&uniform_only_layout),
        vertex: wgpu::VertexState {
            module: &bg_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &bg_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample,
        multiview_mask: None,
        cache: None,
    });

    let surface_geom_vbl = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SurfaceGeomVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 24,
                shader_location: 4,
            },
        ],
    };
    let surface_scalar_vbl = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: 0,
            shader_location: 2,
        }],
    };
    let surface_edges_vbl = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SurfaceVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 24,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 28,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 40,
                shader_location: 4,
            },
        ],
    };

    let make_surface_pipeline = |label: &str,
                                 entry_point: &'static str,
                                 buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                                 alpha_blend: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&actor_lut_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: Some(entry_point),
                buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &surface_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(if alpha_blend {
                        wgpu::BlendState::ALPHA_BLENDING
                    } else {
                        wgpu::BlendState::REPLACE
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(!alpha_blend),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample,
            multiview_mask: None,
            cache: None,
        })
    };

    let indexed_buffers = [Some(surface_geom_vbl), Some(surface_scalar_vbl)];
    let edges_buffers = [Some(surface_edges_vbl)];
    let surface_indexed_opaque_pipeline =
        make_surface_pipeline("paraviz-surface-indexed-opaque", "vs_indexed", &indexed_buffers, false);
    let surface_indexed_alpha_pipeline =
        make_surface_pipeline("paraviz-surface-indexed-alpha", "vs_indexed", &indexed_buffers, true);
    let surface_edges_opaque_pipeline =
        make_surface_pipeline("paraviz-surface-edges-opaque", "vs_main", &edges_buffers, false);
    let surface_edges_alpha_pipeline =
        make_surface_pipeline("paraviz-surface-edges-alpha", "vs_main", &edges_buffers, true);

    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-line-pipeline"),
        layout: Some(&actor_lut_layout),
        vertex: wgpu::VertexState {
            module: &line_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: (2 * std::mem::size_of::<LineVertex>()) as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 32,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 44,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 48,
                        shader_location: 5,
                    },
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &line_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample,
        multiview_mask: None,
        cache: None,
    });

    let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-point-pipeline"),
        layout: Some(&actor_lut_layout),
        vertex: wgpu::VertexState {
            module: &point_shader,
            entry_point: Some("vs_main"),
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 12,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 3,
                        },
                    ],
                }),
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &point_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample,
        multiview_mask: None,
        cache: None,
    });

    let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-volume-pipeline"),
        layout: Some(&volume_layout),
        vertex: wgpu::VertexState {
            module: &volume_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CubeVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &volume_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample,
        multiview_mask: None,
        cache: None,
    });

    let widget_tri_vbl = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<WidgetVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
    let widget_line_vbl = wgpu::VertexBufferLayout {
        array_stride: (2 * std::mem::size_of::<WidgetVertex>()) as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 28,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 40,
                shader_location: 3,
            },
        ],
    };

    let make_widget_pipeline =
        |label: &str, entry_point: &'static str, vbl: wgpu::VertexBufferLayout<'_>, depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&uniform_only_layout),
                vertex: wgpu::VertexState {
                    module: &widget_shader,
                    entry_point: Some(entry_point),
                    buffers: &[Some(vbl)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &widget_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample,
                multiview_mask: None,
                cache: None,
            })
        };

    let widget_tri_pipeline =
        make_widget_pipeline("paraviz-widget-tri", "vs_main", widget_tri_vbl, false);
    let widget_line_pipeline =
        make_widget_pipeline("paraviz-widget-line", "vs_line", widget_line_vbl, true);

    let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-gizmo-pipeline"),
        layout: Some(&uniform_only_layout),
        vertex: wgpu::VertexState {
            module: &gizmo_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GizmoVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 24,
                        shader_location: 2,
                    },
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &gizmo_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample,
        multiview_mask: None,
        cache: None,
    });

    PipelineBundle {
        bg_pipeline,
        surface_indexed_opaque_pipeline,
        surface_indexed_alpha_pipeline,
        surface_edges_opaque_pipeline,
        surface_edges_alpha_pipeline,
        line_pipeline,
        point_pipeline,
        volume_pipeline,
        widget_tri_pipeline,
        widget_line_pipeline,
        gizmo_pipeline,
    }
}
