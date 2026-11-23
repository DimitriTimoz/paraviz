//! WebGPU rendering engine with actor-level GPU caching, headless RGBA/PNG export,
//! Eye-Dome Lighting (EDL), 4x MSAA, 3-point ParaView Light Kit, 3D viewport widgets,
//! and multi-representation scientific visualization pipelines.

use crate::mesh_prep::{
    self, GizmoVertex, LineVertex, PointInstance, SurfaceGeomVertex, SurfaceScalarSource,
    SurfaceVertex, WidgetVertex,
};
use crate::scene::{ActorStyle, ColorBy, RenderScene, Representation, SceneActor};
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use paraviz_core::{ColorMap, DataSet, TransferFunction};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use wgpu::util::DeviceExt;

const LUT_WIDTH: usize = 256;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

mod pipelines;
mod uniforms;

use pipelines::*;
use uniforms::*;

/// Errors returned by [`WebGpuRenderer`].
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("No compatible WebGPU adapter found")]
    NoAdapter,

    #[error("Failed to request WebGPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("Invalid render target dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Failed to map GPU staging buffer: {0}")]
    BufferMap(String),

    #[error("WebGPU device poll error: {0}")]
    DevicePoll(String),

    #[error("Failed to encode PNG: {0}")]
    PngEncode(#[from] png::EncodingError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// WebGPU-based scientific renderer supporting offscreen and interactive viewport rendering.
pub struct WebGpuRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
    pipelines: HashMap<(wgpu::TextureFormat, u32), PipelineBundle>,
    edl_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    depth_resolve_bgl: wgpu::BindGroupLayout,
    depth_resolve_pipeline: wgpu::RenderPipeline,
    offscreen_targets: Option<RenderTargets>,
    external_depth: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    msaa_targets: Option<MsaaTargets>,
    edl_targets: Option<EdlTargets>,

    // Shared bind group layouts
    uniform_bgl: wgpu::BindGroupLayout,
    lut_bgl: wgpu::BindGroupLayout,
    volume_bgl: wgpu::BindGroupLayout,
    edl_tex_bgl: wgpu::BindGroupLayout,
    lut_sampler: wgpu::Sampler,

    // Shared static buffers & pass resources
    bg_uniform_buf: wgpu::Buffer,
    bg_bind_group: wgpu::BindGroup,
    edl_uniform_buf: wgpu::Buffer,
    edl_uniform_bind_group: wgpu::BindGroup,
    point_quad_buf: wgpu::Buffer,
    volume_cube_vb: wgpu::Buffer,
    volume_cube_ib: wgpu::Buffer,
    gizmo_vb: wgpu::Buffer,
    gizmo_vertex_count: u32,
    gizmo_uniform_buf: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,

    // Bounding-box overlay cache
    bbox_vb: Option<DynamicVertexBuffer>,
    bbox_vertex_count: u32,
    bbox_uniform_buf: wgpu::Buffer,
    bbox_bind_group: wgpu::BindGroup,
    bbox_lut_bind_group: wgpu::BindGroup,

    // 3D Viewport Widget overlay buffers
    widget_tri_vb: Option<DynamicVertexBuffer>,
    widget_tri_count: u32,
    widget_line_vb: Option<DynamicVertexBuffer>,
    widget_line_count: u32,
    widget_uniform_buf: wgpu::Buffer,
    widget_bind_group: wgpu::BindGroup,

    // Per-actor GPU resource cache
    actor_cache: HashMap<u64, CachedActor>,
}

impl WebGpuRenderer {
    /// Creates a headless WebGPU renderer targeting `Rgba8Unorm`.
    pub fn new_headless() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|_| RenderError::NoAdapter)?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("paraviz-headless-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))?;

        Ok(Self::from_device(
            Arc::new(device),
            Arc::new(queue),
            wgpu::TextureFormat::Rgba8Unorm,
        ))
    }

    /// Creates a renderer using an existing `wgpu::Device`, `wgpu::Queue`, and surface/target format.
    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paraviz-uniform-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let lut_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paraviz-lut-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let volume_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paraviz-volume-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let edl_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paraviz-edl-tex-bgl"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let depth_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paraviz-depth-resolve-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        let depth_resolve_pipeline = create_depth_resolve_pipeline(&device, &depth_resolve_bgl);

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("paraviz-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let mut pipelines = HashMap::new();
        pipelines.insert(
            (target_format, 1),
            create_pipelines(&device, target_format, 1, &uniform_bgl, &lut_bgl, &volume_bgl),
        );

        let bg_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-bg-uniform"),
            contents: bytemuck::bytes_of(&BgUniforms {
                bg_top: [0.13, 0.16, 0.22, 1.0],
                bg_bottom: [0.27, 0.32, 0.41, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-bg-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bg_uniform_buf.as_entire_binding(),
            }],
        });

        let edl_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-edl-uniform"),
            contents: bytemuck::bytes_of(&EdlUniforms {
                near: 0.01,
                far: 1000.0,
                edl_strength: 1.0,
                edl_radius: 2.0,
                orthographic: 0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let edl_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-edl-uniform-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: edl_uniform_buf.as_entire_binding(),
            }],
        });

        let quad_verts = [
            QuadVertex { uv: [-1.0, -1.0] },
            QuadVertex { uv: [1.0, -1.0] },
            QuadVertex { uv: [1.0, 1.0] },
            QuadVertex { uv: [-1.0, -1.0] },
            QuadVertex { uv: [1.0, 1.0] },
            QuadVertex { uv: [-1.0, 1.0] },
        ];
        let point_quad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-point-quad"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let cube_verts = [
            CubeVertex { pos: [0.0, 0.0, 0.0] },
            CubeVertex { pos: [1.0, 0.0, 0.0] },
            CubeVertex { pos: [1.0, 1.0, 0.0] },
            CubeVertex { pos: [0.0, 1.0, 0.0] },
            CubeVertex { pos: [0.0, 0.0, 1.0] },
            CubeVertex { pos: [1.0, 0.0, 1.0] },
            CubeVertex { pos: [1.0, 1.0, 1.0] },
            CubeVertex { pos: [0.0, 1.0, 1.0] },
        ];
        let cube_indices: [u16; 36] = [
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        let volume_cube_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-volume-cube-vb"),
            contents: bytemuck::cast_slice(&cube_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let volume_cube_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-volume-cube-ib"),
            contents: bytemuck::cast_slice(&cube_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let gizmo_verts = mesh_prep::build_axes_gizmo_mesh();
        let gizmo_vertex_count = gizmo_verts.len() as u32;
        let gizmo_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-gizmo-vb"),
            contents: bytemuck::cast_slice(&gizmo_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let gizmo_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-gizmo-uniform"),
            contents: bytemuck::bytes_of(&GizmoUniforms {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                light_dir: [0.3, 0.5, 0.8, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let gizmo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-gizmo-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gizmo_uniform_buf.as_entire_binding(),
            }],
        });

        let bbox_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-bbox-uniform"),
            contents: bytemuck::bytes_of(&LineUniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                solid_color: [0.85, 0.85, 0.85, 1.0],
                nan_color: [0.5, 0.5, 0.0, 1.0],
                scalar_min: 0.0,
                scalar_max: 1.0,
                use_colormap: 0,
                log_scale: 0,
                opacity: 1.0,
                line_width: 1.5,
                viewport_size: [1.0, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bbox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-bbox-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bbox_uniform_buf.as_entire_binding(),
            }],
        });
        let default_lut = ColorMap::default();
        let bbox_lut_tex = create_lut_texture(
            &device,
            &queue,
            &default_lut.to_rgba8_lut(LUT_WIDTH),
            "paraviz-bbox-lut",
        );
        let bbox_lut_view = bbox_lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bbox_lut_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-bbox-lut-bg"),
            layout: &lut_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bbox_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let widget_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paraviz-widget-uniform"),
            contents: bytemuck::bytes_of(&WidgetUniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                viewport_size: [1.0, 1.0],
                line_width: 2.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let widget_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paraviz-widget-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: widget_uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            device,
            queue,
            target_format,
            pipelines,
            edl_pipelines: HashMap::new(),
            depth_resolve_bgl,
            depth_resolve_pipeline,
            offscreen_targets: None,
            external_depth: None,
            msaa_targets: None,
            edl_targets: None,
            uniform_bgl,
            lut_bgl,
            volume_bgl,
            edl_tex_bgl,
            lut_sampler,
            bg_uniform_buf,
            bg_bind_group,
            edl_uniform_buf,
            edl_uniform_bind_group,
            point_quad_buf,
            volume_cube_vb,
            volume_cube_ib,
            gizmo_vb,
            gizmo_vertex_count,
            gizmo_uniform_buf,
            gizmo_bind_group,
            bbox_vb: None,
            bbox_vertex_count: 0,
            bbox_uniform_buf,
            bbox_bind_group,
            bbox_lut_bind_group,
            widget_tri_vb: None,
            widget_tri_count: 0,
            widget_line_vb: None,
            widget_line_count: 0,
            widget_uniform_buf,
            widget_bind_group,
            actor_cache: HashMap::new(),
        }
    }

    /// Returns the underlying WebGPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a cloned `Arc<wgpu::Device>` for sharing with compute engines.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Returns the underlying WebGPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Returns a cloned `Arc<wgpu::Queue>` for sharing with compute engines.
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }

    /// Renders `scene` into an internal `Rgba8Unorm` texture (`RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC`)
    /// and returns a reference to its `TextureView`.
    pub fn render_to_texture(
        &mut self,
        scene: &RenderScene,
        width: u32,
        height: u32,
    ) -> Result<&wgpu::TextureView, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }

        let recreate = match &self.offscreen_targets {
            Some(t) => t.width != width || t.height != height,
            None => true,
        };
        if recreate {
            self.offscreen_targets = Some(create_render_targets(&self.device, width, height));
        }

        let targets = self.offscreen_targets.take().unwrap();
        let res = self.render_into_views(
            scene,
            wgpu::TextureFormat::Rgba8Unorm,
            &targets.color_view,
            &targets.depth_view,
            width,
            height,
        );
        self.offscreen_targets = Some(targets);
        res?;

        Ok(&self.offscreen_targets.as_ref().unwrap().color_view)
    }

    /// Renders `scene` directly into a caller-supplied `TextureView` matching `target_format` from `from_device`.
    pub fn render_to_view(
        &mut self,
        scene: &RenderScene,
        color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        let need_depth = match &self.external_depth {
            Some((w, h, _, _)) => *w != width || *h != height,
            None => true,
        };
        if need_depth {
            let (tex, view) = create_depth_texture(&self.device, width, height, 1, false);
            self.external_depth = Some((width, height, tex, view));
        }
        let (_, _, depth_tex, depth_view) = self.external_depth.take().unwrap();
        let res = self.render_into_views(
            scene,
            self.target_format,
            color_view,
            &depth_view,
            width,
            height,
        );
        self.external_depth = Some((width, height, depth_tex, depth_view));
        res
    }

    /// Renders `scene` to a tightly-packed `Rgba8Unorm` byte buffer (`width * height * 4`).
    pub fn render_to_rgba(
        &mut self,
        scene: &RenderScene,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let _ = self.render_to_texture(scene, width, height)?;
        let targets = self.offscreen_targets.as_ref().unwrap();

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paraviz-readback-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paraviz-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::DevicePoll(format!("{e:?}")))?;
        rx.recv()
            .map_err(|e| RenderError::BufferMap(e.to_string()))?
            .map_err(|e| RenderError::BufferMap(format!("{e:?}")))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|e| RenderError::BufferMap(format!("{e:?}")))?;
        let mut rgba = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            rgba.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        staging.unmap();
        if scene.settings.show_scalar_bar
            || scene.settings.show_time_annotation
            || scene.settings.annotations.iter().any(|a| a.visible)
        {
            crate::legend::composite_viewport_overlays(scene, &mut rgba, width, height);
        }
        Ok(rgba)
    }

    /// Renders `scene` and writes an 8-bit RGBA PNG image to `path`.
    pub fn render_to_png(
        &mut self,
        scene: &RenderScene,
        width: u32,
        height: u32,
        path: impl AsRef<Path>,
    ) -> Result<(), RenderError> {
        let rgba = self.render_to_rgba(scene, width, height)?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut png_writer = encoder.write_header()?;
        png_writer.write_image_data(&rgba)?;
        Ok(())
    }

    fn ensure_pipeline_bundle(&mut self, format: wgpu::TextureFormat, sample_count: u32) {
        self.pipelines.entry((format, sample_count)).or_insert_with(|| {
            create_pipelines(
                &self.device,
                format,
                sample_count,
                &self.uniform_bgl,
                &self.lut_bgl,
                &self.volume_bgl,
            )
        });
    }

    fn ensure_edl_pipeline(&mut self, format: wgpu::TextureFormat) {
        self.edl_pipelines.entry(format).or_insert_with(|| {
            create_edl_pipeline(&self.device, format, &self.uniform_bgl, &self.edl_tex_bgl)
        });
    }

    fn ensure_msaa_targets(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let need = match &self.msaa_targets {
            Some(t) => t.width != width || t.height != height || t.format != format,
            None => true,
        };
        if need {
            self.msaa_targets = Some(create_msaa_targets(
                &self.device,
                width,
                height,
                format,
                &self.depth_resolve_bgl,
            ));
        }
    }

    fn ensure_edl_targets(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let need = match &self.edl_targets {
            Some(t) => t.width != width || t.height != height || t.format != format,
            None => true,
        };
        if need {
            self.edl_targets = Some(create_edl_targets(
                &self.device,
                width,
                height,
                format,
                &self.edl_tex_bgl,
            ));
        }
    }

    fn render_into_views(
        &mut self,
        scene: &RenderScene,
        format: wgpu::TextureFormat,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        let aspect = width as f32 / height as f32;
        let view_mat = scene.camera.view_matrix();
        let proj_mat = scene.camera.projection_matrix(aspect);
        let view_proj = proj_mat * view_mat;
        let inv_view_proj = view_proj.inverse();

        // Camera-relative 3-point ParaView Light Kit (Key, Fill, Rim/Back).
        let inv_view = view_mat.inverse();
        let key_world = inv_view
            .transform_vector3(Vec3::new(0.35, 0.55, 0.75))
            .normalize_or(Vec3::new(0.35, 0.55, 0.75));
        let fill_world = inv_view
            .transform_vector3(Vec3::new(-0.60, -0.35, 0.40))
            .normalize_or(Vec3::new(-0.60, -0.35, 0.40));
        let back_world = inv_view
            .transform_vector3(Vec3::new(-0.35, 0.65, -0.65))
            .normalize_or(Vec3::new(-0.35, 0.65, -0.65));

        // Update background uniforms.
        self.queue.write_buffer(
            &self.bg_uniform_buf,
            0,
            bytemuck::bytes_of(&BgUniforms {
                bg_top: [
                    scene.settings.bg_top[0],
                    scene.settings.bg_top[1],
                    scene.settings.bg_top[2],
                    1.0,
                ],
                bg_bottom: [
                    scene.settings.bg_bottom[0],
                    scene.settings.bg_bottom[1],
                    scene.settings.bg_bottom[2],
                    1.0,
                ],
            }),
        );

        // Sync actor GPU cache & write per-actor uniforms.
        let active_ids: HashSet<u64> = scene.actors.iter().map(|a| a.id).collect();
        self.actor_cache.retain(|id, _| active_ids.contains(id));

        for actor in &scene.actors {
            if !actor.style.visible {
                continue;
            }
            self.sync_actor_cache(actor);
            let cached = self.actor_cache.get(&actor.id).unwrap();
            let (smin, smax) = resolve_scalar_range(&actor.style, &cached.geometry);
            write_actor_uniforms(
                &self.queue,
                cached,
                &actor.style,
                view_proj,
                inv_view_proj,
                scene.camera.position,
                key_world,
                fill_world,
                back_world,
                [width as f32, height as f32],
                (smin as f32, smax as f32),
            );
        }

        // Optional global bounding box / Cube Axes Grid overlay (reusing persistent dynamic GPU buffer).
        if scene.settings.show_bounding_box || scene.settings.show_axes_grid {
            let bb = scene.bounds();
            let mut line_verts = mesh_prep::prepare_outline(&bb);
            if scene.settings.show_axes_grid && bb.is_valid() {
                let tick_len = (bb.diagonal() * 0.025).max(1e-4);
                let n_ticks = 5usize;
                for i in 0..=n_ticks {
                    let frac = i as f32 / n_ticks as f32;
                    // X-axis ticks along bottom-front edge
                    let x = bb.min.x + (bb.max.x - bb.min.x) * frac;
                    line_verts.push(mesh_prep::LineVertex { position: [x, bb.min.y, bb.min.z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                    line_verts.push(mesh_prep::LineVertex { position: [x, bb.min.y - tick_len, bb.min.z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                    // Y-axis ticks along left-front edge
                    let y = bb.min.y + (bb.max.y - bb.min.y) * frac;
                    line_verts.push(mesh_prep::LineVertex { position: [bb.min.x, y, bb.min.z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                    line_verts.push(mesh_prep::LineVertex { position: [bb.min.x - tick_len, y, bb.min.z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                    // Z-axis ticks along bottom-left edge
                    let z = bb.min.z + (bb.max.z - bb.min.z) * frac;
                    line_verts.push(mesh_prep::LineVertex { position: [bb.min.x, bb.min.y, z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                    line_verts.push(mesh_prep::LineVertex { position: [bb.min.x - tick_len, bb.min.y, z], scalar: 0.0, color_override: mesh_prep::NO_COLOR_OVERRIDE });
                }
            }
            self.bbox_vertex_count = line_verts.len() as u32;
            if !line_verts.is_empty() {
                DynamicVertexBuffer::upload(
                    &mut self.bbox_vb,
                    &self.device,
                    &self.queue,
                    "paraviz-scene-bbox-vb",
                    &line_verts,
                );
                self.queue.write_buffer(
                    &self.bbox_uniform_buf,
                    0,
                    bytemuck::bytes_of(&LineUniforms {
                        view_proj: view_proj.to_cols_array_2d(),
                        solid_color: [0.88, 0.88, 0.90, 1.0],
                        nan_color: [0.5, 0.5, 0.0, 1.0],
                        scalar_min: 0.0,
                        scalar_max: 1.0,
                        use_colormap: 0,
                        log_scale: 0,
                        opacity: 0.9,
                        line_width: 1.5,
                        viewport_size: [width as f32, height as f32],
                    }),
                );
            }
        } else {
            self.bbox_vertex_count = 0;
        }

        // Prepare 3D interactive viewport widget overlays + optional selection highlight crosshair.
        let mut all_widgets = scene.widgets.clone();
        if let Some(pos) = scene.selection_highlight {
            let r = (scene.bounds().diagonal() * 0.025).max(0.05);
            all_widgets.push(crate::scene::ViewportWidget::Sphere { center: pos, radius: r });
        }
        if !all_widgets.is_empty() {
            let prep = mesh_prep::prepare_widgets(&all_widgets);
            self.widget_tri_count = prep.triangles.len() as u32;
            self.widget_line_count = prep.lines.len() as u32;
            if !prep.triangles.is_empty() {
                DynamicVertexBuffer::upload(
                    &mut self.widget_tri_vb,
                    &self.device,
                    &self.queue,
                    "paraviz-widget-tri-vb",
                    &prep.triangles,
                );
            }
            if !prep.lines.is_empty() {
                DynamicVertexBuffer::upload(
                    &mut self.widget_line_vb,
                    &self.device,
                    &self.queue,
                    "paraviz-widget-line-vb",
                    &prep.lines,
                );
            }
            self.queue.write_buffer(
                &self.widget_uniform_buf,
                0,
                bytemuck::bytes_of(&WidgetUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    viewport_size: [width as f32, height as f32],
                    line_width: 2.0,
                    _pad: 0.0,
                }),
            );
        } else {
            self.widget_tri_count = 0;
            self.widget_line_count = 0;
        }

        // Update orientation axes gizmo uniform (rotation-only view matrix).
        let draw_gizmo = scene.settings.show_axes_gizmo && self.gizmo_vertex_count > 0;
        if draw_gizmo {
            let mut rot = view_mat;
            rot.w_axis = Vec4::new(0.0, 0.0, -3.2, 1.0);
            let gizmo_proj =
                glam::camera::rh::proj::directx::orthographic(-1.35, 1.35, -1.35, 1.35, 0.1, 10.0);
            let gizmo_mvp = gizmo_proj * rot;
            self.queue.write_buffer(
                &self.gizmo_uniform_buf,
                0,
                bytemuck::bytes_of(&GizmoUniforms {
                    mvp: gizmo_mvp.to_cols_array_2d(),
                    light_dir: [key_world.x, key_world.y, key_world.z, 0.0],
                }),
            );
        }

        // Determine sample count & render path (supports EDL + 4x MSAA simultaneously).
        let effective_samples = clamp_msaa_samples(scene.settings.msaa_samples);
        let edl_enabled = scene.settings.edl_enabled;

        self.ensure_pipeline_bundle(format, effective_samples);
        if edl_enabled && effective_samples == 4 && draw_gizmo {
            self.ensure_pipeline_bundle(format, 1);
        }
        if edl_enabled {
            self.ensure_edl_pipeline(format);
            self.ensure_edl_targets(width, height, format);
            let near = scene.camera.near.max(1e-5);
            let far = scene.camera.far.max(near + 1e-3);
            self.queue.write_buffer(
                &self.edl_uniform_buf,
                0,
                bytemuck::bytes_of(&EdlUniforms {
                    near,
                    far,
                    edl_strength: scene.settings.edl_strength.max(0.0),
                    edl_radius: 2.0,
                    orthographic: u32::from(scene.camera.orthographic),
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
            );
        }
        if effective_samples == 4 {
            self.ensure_msaa_targets(width, height, format);
        }

        let bundle = self.pipelines.get(&(format, effective_samples)).unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paraviz-render-encoder"),
            });

        // Choose main geometry pass targets.
        let (pass_color_view, pass_resolve_target, pass_depth_view, pass_color_store) =
            if edl_enabled && effective_samples == 4 {
                let msaa = self.msaa_targets.as_ref().unwrap();
                let edl = self.edl_targets.as_ref().unwrap();
                (
                    &msaa.color_view,
                    Some(&edl.color_view),
                    &msaa.depth_view,
                    wgpu::StoreOp::Discard,
                )
            } else if edl_enabled {
                let edl = self.edl_targets.as_ref().unwrap();
                (&edl.color_view, None, &edl.depth_view, wgpu::StoreOp::Store)
            } else if effective_samples == 4 {
                let msaa = self.msaa_targets.as_ref().unwrap();
                if draw_gizmo {
                    (&msaa.color_view, None, &msaa.depth_view, wgpu::StoreOp::Store)
                } else {
                    (
                        &msaa.color_view,
                        Some(color_view),
                        &msaa.depth_view,
                        wgpu::StoreOp::Discard,
                    )
                }
            } else {
                (color_view, None, depth_view, wgpu::StoreOp::Store)
            };

        let clear_color = if scene.settings.transparent_background {
            wgpu::Color::TRANSPARENT
        } else {
            wgpu::Color::BLACK
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pass_color_view,
                    depth_slice: None,
                    resolve_target: pass_resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: pass_color_store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: pass_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1. Background gradient (skipped when exporting transparent background)
            if !scene.settings.transparent_background {
                pass.set_pipeline(&bundle.bg_pipeline);
                pass.set_bind_group(0, &self.bg_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // 2. Opaque surfaces, wireframes, outlines, and points
            for actor in &scene.actors {
                if !actor.style.visible {
                    continue;
                }
                let Some(cached) = self.actor_cache.get(&actor.id) else {
                    continue;
                };
                match &cached.geometry {
                    CachedGeometry::SurfaceIndexed {
                        geom_buffer,
                        scalar_buffer,
                        index_buffer,
                        index_count,
                        ..
                    } if *index_count > 0 && actor.style.opacity >= 0.999 => {
                        pass.set_pipeline(&bundle.surface_indexed_opaque_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, geom_buffer.slice(..));
                        pass.set_vertex_buffer(1, scalar_buffer.slice(..));
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                    CachedGeometry::SurfaceEdges {
                        buffer,
                        vertex_count,
                        ..
                    } if *vertex_count > 0 && actor.style.opacity >= 0.999 => {
                        pass.set_pipeline(&bundle.surface_edges_opaque_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*vertex_count, 0..1);
                    }
                    CachedGeometry::Lines {
                        buffer,
                        vertex_count,
                        ..
                    } if *vertex_count >= 2 => {
                        pass.set_pipeline(&bundle.line_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..6, 0..(*vertex_count / 2));
                    }
                    CachedGeometry::Points {
                        buffer,
                        instance_count,
                        ..
                    } if *instance_count > 0 => {
                        pass.set_pipeline(&bundle.point_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, self.point_quad_buf.slice(..));
                        pass.set_vertex_buffer(1, buffer.slice(..));
                        pass.draw(0..6, 0..*instance_count);
                    }
                    _ => {}
                }
            }

            // 3. Global bounding-box / Cube Axes Grid overlay
            if (scene.settings.show_bounding_box || scene.settings.show_axes_grid)
                && self.bbox_vertex_count >= 2
                && let Some(vb) = &self.bbox_vb
            {
                pass.set_pipeline(&bundle.line_pipeline);
                pass.set_bind_group(0, &self.bbox_bind_group, &[]);
                pass.set_bind_group(1, &self.bbox_lut_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice_used());
                pass.draw(0..6, 0..(self.bbox_vertex_count / 2));
            }

            // 4. Translucent surfaces (sorted back-to-front by distance from camera)
            let mut translucent_indices: Vec<(usize, f32)> = scene
                .actors
                .iter()
                .enumerate()
                .filter(|(_, a)| a.style.visible && a.style.opacity < 0.999)
                .map(|(idx, a)| {
                    let c = a.style.dataset_bounds(&a.dataset).center();
                    let dist2 = (c - scene.camera.position).length_squared();
                    (idx, dist2)
                })
                .collect();
            translucent_indices.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (actor_idx, _) in translucent_indices {
                let actor = &scene.actors[actor_idx];
                let Some(cached) = self.actor_cache.get(&actor.id) else {
                    continue;
                };
                match &cached.geometry {
                    CachedGeometry::SurfaceIndexed {
                        geom_buffer,
                        scalar_buffer,
                        index_buffer,
                        index_count,
                        ..
                    } if *index_count > 0 => {
                        pass.set_pipeline(&bundle.surface_indexed_alpha_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, geom_buffer.slice(..));
                        pass.set_vertex_buffer(1, scalar_buffer.slice(..));
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                    CachedGeometry::SurfaceEdges {
                        buffer,
                        vertex_count,
                        ..
                    } if *vertex_count > 0 => {
                        pass.set_pipeline(&bundle.surface_edges_alpha_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*vertex_count, 0..1);
                    }
                    _ => {}
                }
            }

            // 5. Volume raymarching actors
            for actor in &scene.actors {
                if !actor.style.visible {
                    continue;
                }
                let Some(cached) = self.actor_cache.get(&actor.id) else {
                    continue;
                };
                if let CachedGeometry::Volume { .. } = &cached.geometry
                    && let Some(vol_bg) = &cached.volume_bind_group
                {
                    pass.set_pipeline(&bundle.volume_pipeline);
                    pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                    pass.set_bind_group(1, vol_bg, &[]);
                    pass.set_vertex_buffer(0, self.volume_cube_vb.slice(..));
                    pass.set_index_buffer(
                        self.volume_cube_ib.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    pass.draw_indexed(0..36, 0, 0..1);
                }
            }

            // 6. 3D Interactive Viewport Widget overlays (wide wireframe lines + translucent guides)
            if self.widget_line_count >= 2
                && let Some(vb) = &self.widget_line_vb
            {
                pass.set_pipeline(&bundle.widget_line_pipeline);
                pass.set_bind_group(0, &self.widget_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice_used());
                pass.draw(0..6, 0..(self.widget_line_count / 2));
            }
            if self.widget_tri_count > 0
                && let Some(vb) = &self.widget_tri_vb
            {
                pass.set_pipeline(&bundle.widget_tri_pipeline);
                pass.set_bind_group(0, &self.widget_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice_used());
                pass.draw(0..self.widget_tri_count, 0..1);
            }
        }

        // 6b. Resolve 4x MSAA depth into 1x EDL depth texture when both EDL and 4x MSAA are enabled.
        if edl_enabled && effective_samples == 4 {
            let msaa = self.msaa_targets.as_ref().unwrap();
            let edl_targets = self.edl_targets.as_ref().unwrap();
            let mut resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-depth-resolve-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &edl_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            resolve_pass.set_pipeline(&self.depth_resolve_pipeline);
            resolve_pass.set_bind_group(0, &msaa.depth_resolve_bg, &[]);
            resolve_pass.draw(0..3, 0..1);
        }

        // 7. Fullscreen Eye-Dome Lighting (EDL) post-process pass into final color_view
        if edl_enabled {
            let edl_pipeline = self.edl_pipelines.get(&format).unwrap();
            let edl_targets = self.edl_targets.as_ref().unwrap();
            let mut edl_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-edl-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            edl_pass.set_pipeline(edl_pipeline);
            edl_pass.set_bind_group(0, &self.edl_uniform_bind_group, &[]);
            edl_pass.set_bind_group(1, &edl_targets.tex_bind_group, &[]);
            edl_pass.draw(0..3, 0..1);
        }

        // 8. 3D Orientation Axes Gizmo in bottom-left corner
        if draw_gizmo {
            let gizmo_size = (width.min(height) as f32 * 0.16).clamp(48.0, 140.0);
            let margin = 10.0_f32.min((width as f32 - gizmo_size).max(0.0));
            let gy = (height as f32 - gizmo_size - margin).max(0.0);

            let (gizmo_pipeline, gizmo_color_view, gizmo_resolve, gizmo_depth_view, gizmo_store) =
                if !edl_enabled && effective_samples == 4 {
                    let msaa = self.msaa_targets.as_ref().unwrap();
                    (
                        &bundle.gizmo_pipeline,
                        &msaa.color_view,
                        Some(color_view),
                        &msaa.depth_view,
                        wgpu::StoreOp::Discard,
                    )
                } else {
                    let b1 = self.pipelines.get(&(format, 1)).unwrap();
                    (
                        &b1.gizmo_pipeline,
                        color_view,
                        None,
                        depth_view,
                        wgpu::StoreOp::Store,
                    )
                };

            let mut gizmo_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-gizmo-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: gizmo_color_view,
                    depth_slice: None,
                    resolve_target: gizmo_resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: gizmo_store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: gizmo_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            gizmo_pass.set_viewport(margin, gy, gizmo_size, gizmo_size, 0.0, 1.0);
            gizmo_pass.set_pipeline(gizmo_pipeline);
            gizmo_pass.set_bind_group(0, &self.gizmo_bind_group, &[]);
            gizmo_pass.set_vertex_buffer(0, self.gizmo_vb.slice(..));
            gizmo_pass.draw(0..self.gizmo_vertex_count, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn sync_actor_cache(&mut self, actor: &SceneActor) {
        let style_hash = actor_style_hash(&actor.style);
        let mut scalar_only_update = false;
        let needs_rebuild = match self.actor_cache.get(&actor.id) {
            Some(c) => {
                if !Arc::ptr_eq(&c.dataset, &actor.dataset)
                    || c.representation != actor.style.representation
                    || c.actor_style_hash != style_hash
                    || c.hidden_blocks != actor.style.hidden_blocks
                    || c.block_colors != actor.style.block_colors
                {
                    true
                } else if c.color_by != actor.style.color_by {
                    if matches!(c.geometry, CachedGeometry::SurfaceIndexed { .. }) {
                        scalar_only_update = true;
                        false
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            None => true,
        };

        if scalar_only_update {
            let cached = self.actor_cache.get_mut(&actor.id).unwrap();
            if let CachedGeometry::SurfaceIndexed {
                scalar_buffer,
                vertex_count,
                scalar_source,
                auto_range,
                ..
            } = &mut cached.geometry
                && let Some((new_scalars, new_range)) = mesh_prep::extract_indexed_surface_scalars(
                    &actor.dataset,
                    &actor.style.color_by,
                    scalar_source,
                    *vertex_count as usize,
                )
            {
                self.queue
                    .write_buffer(scalar_buffer, 0, bytemuck::cast_slice(&new_scalars));
                *auto_range = new_range;
                cached.color_by = actor.style.color_by.clone();
                if cached.colormap != actor.style.colormap {
                    let bytes = actor.style.colormap.to_rgba8_lut(LUT_WIDTH);
                    upload_lut_texture(&self.queue, &cached.lut_texture, &bytes);
                    cached.colormap = actor.style.colormap.clone();
                }
                if cached.volume_tf != actor.style.volume_tf {
                    let bytes = actor.style.volume_tf.to_rgba8_lut(LUT_WIDTH);
                    upload_lut_texture(&self.queue, &cached.tf_texture, &bytes);
                    cached.volume_tf = actor.style.volume_tf.clone();
                }
                return;
            }
        }

        if needs_rebuild || scalar_only_update {
            let geometry = build_cached_geometry(&self.device, &self.queue, &actor.dataset, &actor.style);
            let lut_texture = create_lut_texture(
                &self.device,
                &self.queue,
                &actor.style.colormap.to_rgba8_lut(LUT_WIDTH),
                "paraviz-actor-lut",
            );
            let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let lut_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("paraviz-actor-lut-bg"),
                layout: &self.lut_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                    },
                ],
            });

            let tf_texture = create_lut_texture(
                &self.device,
                &self.queue,
                &actor.style.volume_tf.to_rgba8_lut(LUT_WIDTH),
                "paraviz-actor-tf",
            );
            let volume_bind_group = if let CachedGeometry::Volume { view, .. } = &geometry {
                let tf_view = tf_texture.create_view(&wgpu::TextureViewDescriptor::default());
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("paraviz-actor-volume-bg"),
                    layout: &self.volume_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&tf_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                        },
                    ],
                }))
            } else {
                None
            };

            let uniform_size = std::mem::size_of::<SurfaceUniforms>() as u64;
            let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("paraviz-actor-uniform"),
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("paraviz-actor-uniform-bg"),
                layout: &self.uniform_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            self.actor_cache.insert(
                actor.id,
                CachedActor {
                    dataset: Arc::clone(&actor.dataset),
                    representation: actor.style.representation,
                    color_by: actor.style.color_by.clone(),
                    actor_style_hash: style_hash,
                    hidden_blocks: actor.style.hidden_blocks.clone(),
                    block_colors: actor.style.block_colors.clone(),
                    geometry,
                    colormap: actor.style.colormap.clone(),
                    lut_texture,
                    lut_bind_group,
                    volume_tf: actor.style.volume_tf.clone(),
                    tf_texture,
                    volume_bind_group,
                    uniform_buffer,
                    uniform_bind_group,
                },
            );
            return;
        }

        // Update LUTs in-place if only colormap or transfer function changed.
        let cached = self.actor_cache.get_mut(&actor.id).unwrap();
        if cached.colormap != actor.style.colormap {
            let bytes = actor.style.colormap.to_rgba8_lut(LUT_WIDTH);
            upload_lut_texture(&self.queue, &cached.lut_texture, &bytes);
            cached.colormap = actor.style.colormap.clone();
        }
        if cached.volume_tf != actor.style.volume_tf {
            let bytes = actor.style.volume_tf.to_rgba8_lut(LUT_WIDTH);
            upload_lut_texture(&self.queue, &cached.tf_texture, &bytes);
            cached.volume_tf = actor.style.volume_tf.clone();
        }
    }
}

