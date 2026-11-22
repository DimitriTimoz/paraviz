//! WebGPU rendering engine with actor-level GPU caching, headless RGBA/PNG export,
//! and multi-representation scientific visualization pipelines.

use crate::mesh_prep::{
    self, GizmoVertex, LineVertex, PointInstance, SurfaceVertex,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BgUniforms {
    bg_top: [f32; 4],
    bg_bottom: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SurfaceUniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    fill_light_dir: [f32; 4],
    solid_color: [f32; 4],
    edge_color: [f32; 4],
    nan_color: [f32; 4],
    scalar_min: f32,
    scalar_max: f32,
    use_colormap: u32,
    log_scale: u32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    opacity: f32,
    edge_width: f32,
    show_edges: u32,
    two_sided: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineUniforms {
    view_proj: [[f32; 4]; 4],
    solid_color: [f32; 4],
    nan_color: [f32; 4],
    scalar_min: f32,
    scalar_max: f32,
    use_colormap: u32,
    log_scale: u32,
    opacity: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PointUniforms {
    view_proj: [[f32; 4]; 4],
    solid_color: [f32; 4],
    nan_color: [f32; 4],
    viewport_size: [f32; 2],
    point_size: f32,
    opacity: f32,
    scalar_min: f32,
    scalar_max: f32,
    use_colormap: u32,
    log_scale: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VolumeUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    box_min: [f32; 4],
    box_max: [f32; 4],
    dims: [u32; 4],
    viewport_size: [f32; 2],
    _pad0: f32,
    _pad1: f32,
    scalar_min: f32,
    scalar_max: f32,
    step_world: f32,
    unit_distance: f32,
    shading: u32,
    log_scale: u32,
    opacity: f32,
    max_steps: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GizmoUniforms {
    mvp: [[f32; 4]; 4],
    light_dir: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeVertex {
    pos: [f32; 3],
}

struct PipelineBundle {
    format: wgpu::TextureFormat,
    bg_pipeline: wgpu::RenderPipeline,
    surface_opaque_pipeline: wgpu::RenderPipeline,
    surface_alpha_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    volume_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
}

struct RenderTargets {
    width: u32,
    height: u32,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

enum CachedGeometry {
    Surface {
        buffer: wgpu::Buffer,
        vertex_count: u32,
        auto_range: (f64, f64),
    },
    Lines {
        buffer: wgpu::Buffer,
        vertex_count: u32,
        auto_range: (f64, f64),
    },
    Points {
        buffer: wgpu::Buffer,
        instance_count: u32,
        auto_range: (f64, f64),
    },
    Volume {
        dims: [u32; 3],
        box_min: [f32; 3],
        box_max: [f32; 3],
        #[allow(dead_code)]
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        auto_range: (f64, f64),
    },
    Empty,
}

struct CachedActor {
    dataset: Arc<DataSet>,
    representation: Representation,
    color_by: ColorBy,
    geometry: CachedGeometry,
    colormap: ColorMap,
    lut_texture: wgpu::Texture,
    lut_bind_group: wgpu::BindGroup,
    volume_tf: TransferFunction,
    tf_texture: wgpu::Texture,
    volume_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

/// WebGPU-based scientific renderer supporting offscreen and interactive viewport rendering.
pub struct WebGpuRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: PipelineBundle,
    offscreen_pipelines: Option<PipelineBundle>,
    offscreen_targets: Option<RenderTargets>,
    external_depth: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,

    // Shared bind group layouts
    uniform_bgl: wgpu::BindGroupLayout,
    lut_bgl: wgpu::BindGroupLayout,
    volume_bgl: wgpu::BindGroupLayout,
    lut_sampler: wgpu::Sampler,

    // Shared static buffers & pass resources
    bg_uniform_buf: wgpu::Buffer,
    bg_bind_group: wgpu::BindGroup,
    point_quad_buf: wgpu::Buffer,
    volume_cube_vb: wgpu::Buffer,
    volume_cube_ib: wgpu::Buffer,
    gizmo_vb: wgpu::Buffer,
    gizmo_vertex_count: u32,
    gizmo_uniform_buf: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,

    // Bounding-box overlay cache
    bbox_vb: Option<wgpu::Buffer>,
    bbox_vertex_count: u32,
    bbox_uniform_buf: wgpu::Buffer,
    bbox_bind_group: wgpu::BindGroup,
    bbox_lut_bind_group: wgpu::BindGroup,

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

        let pipelines = create_pipelines(&device, target_format, &uniform_bgl, &lut_bgl, &volume_bgl);

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
                _pad: [0.0; 3],
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
        let bbox_lut_tex = create_lut_texture(&device, &queue, &default_lut.to_rgba8_lut(LUT_WIDTH), "paraviz-bbox-lut");
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

        Self {
            device,
            queue,
            pipelines,
            offscreen_pipelines: None,
            offscreen_targets: None,
            external_depth: None,
            uniform_bgl,
            lut_bgl,
            volume_bgl,
            lut_sampler,
            bg_uniform_buf,
            bg_bind_group,
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
            actor_cache: HashMap::new(),
        }
    }

    /// Returns the underlying WebGPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the underlying WebGPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
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

        if self.pipelines.format != wgpu::TextureFormat::Rgba8Unorm
            && self.offscreen_pipelines.is_none()
        {
            self.offscreen_pipelines = Some(create_pipelines(
                &self.device,
                wgpu::TextureFormat::Rgba8Unorm,
                &self.uniform_bgl,
                &self.lut_bgl,
                &self.volume_bgl,
            ));
        }

        let targets = self.offscreen_targets.take().unwrap();
        let use_offscreen_bundle = self.pipelines.format != wgpu::TextureFormat::Rgba8Unorm;
        let res = self.render_into_views(
            scene,
            &targets.color_view,
            &targets.depth_view,
            width,
            height,
            use_offscreen_bundle,
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
            let (tex, view) = create_depth_texture(&self.device, width, height);
            self.external_depth = Some((width, height, tex, view));
        }
        let (_, _, depth_tex, depth_view) = self.external_depth.take().unwrap();
        let res = self.render_into_views(scene, color_view, &depth_view, width, height, false);
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

    fn render_into_views(
        &mut self,
        scene: &RenderScene,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        use_offscreen_bundle: bool,
    ) -> Result<(), RenderError> {
        let aspect = width as f32 / height as f32;
        let view_mat = scene.camera.view_matrix();
        let proj_mat = scene.camera.projection_matrix(aspect);
        let view_proj = proj_mat * view_mat;
        let inv_view_proj = view_proj.inverse();

        // Camera-relative key and fill lights (ParaView light-kit style).
        let inv_view = view_mat.inverse();
        let key_world = inv_view
            .transform_vector3(Vec3::new(0.32, 0.48, 0.82))
            .normalize_or(Vec3::new(0.3, 0.5, 0.8));
        let fill_world = inv_view
            .transform_vector3(Vec3::new(-0.55, -0.30, 0.45))
            .normalize_or(Vec3::new(-0.5, -0.3, 0.5));

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
                [width as f32, height as f32],
                (smin as f32, smax as f32),
            );
        }

        // Optional global bounding box overlay.
        if scene.settings.show_bounding_box {
            let bb = scene.bounds();
            let line_verts = mesh_prep::prepare_outline(&bb);
            self.bbox_vertex_count = line_verts.len() as u32;
            if !line_verts.is_empty() {
                self.bbox_vb = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("paraviz-scene-bbox-vb"),
                        contents: bytemuck::cast_slice(&line_verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
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
                        _pad: [0.0; 3],
                    }),
                );
            }
        }

        // Update orientation axes gizmo uniform (rotation-only view matrix).
        if scene.settings.show_axes_gizmo {
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

        let bundle = if use_offscreen_bundle {
            self.offscreen_pipelines.as_ref().unwrap()
        } else {
            &self.pipelines
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paraviz-render-encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
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

            // 1. Background gradient
            pass.set_pipeline(&bundle.bg_pipeline);
            pass.set_bind_group(0, &self.bg_bind_group, &[]);
            pass.draw(0..3, 0..1);

            // 2. Opaque surfaces, wireframes, outlines, and points
            for actor in &scene.actors {
                if !actor.style.visible {
                    continue;
                }
                let Some(cached) = self.actor_cache.get(&actor.id) else {
                    continue;
                };
                match &cached.geometry {
                    CachedGeometry::Surface {
                        buffer,
                        vertex_count,
                        ..
                    } if *vertex_count > 0 && actor.style.opacity >= 0.999 => {
                        pass.set_pipeline(&bundle.surface_opaque_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*vertex_count, 0..1);
                    }
                    CachedGeometry::Lines {
                        buffer,
                        vertex_count,
                        ..
                    } if *vertex_count > 0 => {
                        pass.set_pipeline(&bundle.line_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*vertex_count, 0..1);
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

            // 3. Global bounding-box overlay
            if scene.settings.show_bounding_box && self.bbox_vertex_count > 0 {
                if let Some(vb) = &self.bbox_vb {
                    pass.set_pipeline(&bundle.line_pipeline);
                    pass.set_bind_group(0, &self.bbox_bind_group, &[]);
                    pass.set_bind_group(1, &self.bbox_lut_bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.draw(0..self.bbox_vertex_count, 0..1);
                }
            }

            // 4. Translucent surfaces
            for actor in &scene.actors {
                if !actor.style.visible || actor.style.opacity >= 0.999 {
                    continue;
                }
                let Some(cached) = self.actor_cache.get(&actor.id) else {
                    continue;
                };
                if let CachedGeometry::Surface {
                    buffer,
                    vertex_count,
                    ..
                } = &cached.geometry
                {
                    if *vertex_count > 0 {
                        pass.set_pipeline(&bundle.surface_alpha_pipeline);
                        pass.set_bind_group(0, &cached.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &cached.lut_bind_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*vertex_count, 0..1);
                    }
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
                if let CachedGeometry::Volume { .. } = &cached.geometry {
                    if let Some(vol_bg) = &cached.volume_bind_group {
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
            }
        }

        // 6. 3D Orientation Axes Gizmo in bottom-left corner
        if scene.settings.show_axes_gizmo && self.gizmo_vertex_count > 0 {
            let gizmo_size = (width.min(height) as f32 * 0.16).clamp(48.0, 140.0);
            let margin = 10.0_f32.min((width as f32 - gizmo_size).max(0.0));
            let gy = (height as f32 - gizmo_size - margin).max(0.0);

            let mut gizmo_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paraviz-gizmo-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
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
            gizmo_pass.set_pipeline(&bundle.gizmo_pipeline);
            gizmo_pass.set_bind_group(0, &self.gizmo_bind_group, &[]);
            gizmo_pass.set_vertex_buffer(0, self.gizmo_vb.slice(..));
            gizmo_pass.draw(0..self.gizmo_vertex_count, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn sync_actor_cache(&mut self, actor: &SceneActor) {
        let needs_rebuild = match self.actor_cache.get(&actor.id) {
            Some(c) => {
                !Arc::ptr_eq(&c.dataset, &actor.dataset)
                    || c.representation != actor.style.representation
                    || c.color_by != actor.style.color_by
            }
            None => true,
        };

        if needs_rebuild {
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

            // Allocate uniform buffer sized for the largest uniform struct (SurfaceUniforms = 272 bytes).
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

fn build_cached_geometry(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dataset: &DataSet,
    style: &ActorStyle,
) -> CachedGeometry {
    match style.representation {
        Representation::Surface | Representation::SurfaceWithEdges => {
            let prep = mesh_prep::prepare_surface(dataset, &style.color_by);
            if prep.vertices.is_empty() {
                return CachedGeometry::Empty;
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("paraviz-surface-vb"),
                contents: bytemuck::cast_slice(&prep.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            CachedGeometry::Surface {
                buffer,
                vertex_count: prep.vertices.len() as u32,
                auto_range: prep.scalar_range,
            }
        }
        Representation::Wireframe => {
            let prep = mesh_prep::prepare_wireframe(dataset, &style.color_by);
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
            let verts = mesh_prep::prepare_outline(&dataset.bounds());
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
        Representation::Points => {
            let prep = mesh_prep::prepare_points(dataset, &style.color_by);
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
            let Some(vol) = mesh_prep::prepare_volume(dataset, &style.color_by) else {
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

fn resolve_scalar_range(style: &ActorStyle, geom: &CachedGeometry) -> (f64, f64) {
    if style.auto_range {
        match geom {
            CachedGeometry::Surface { auto_range, .. }
            | CachedGeometry::Lines { auto_range, .. }
            | CachedGeometry::Points { auto_range, .. }
            | CachedGeometry::Volume { auto_range, .. } => *auto_range,
            CachedGeometry::Empty => style.scalar_range,
        }
    } else if style.representation == Representation::Volume {
        style.volume_tf.scalar_range
    } else {
        style.scalar_range
    }
}

#[allow(clippy::too_many_arguments)]
fn write_actor_uniforms(
    queue: &wgpu::Queue,
    cached: &CachedActor,
    style: &ActorStyle,
    view_proj: Mat4,
    inv_view_proj: Mat4,
    camera_pos: Vec3,
    light_dir: Vec3,
    fill_light_dir: Vec3,
    viewport_size: [f32; 2],
    scalar_range: (f32, f32),
) {
    let (solid_color, use_colormap) = match style.color_by {
        ColorBy::Solid(rgba) => (rgba, 0u32),
        ColorBy::PointScalar { .. } | ColorBy::CellScalar { .. } => ([1.0, 1.0, 1.0, 1.0], 1u32),
    };

    match &cached.geometry {
        CachedGeometry::Surface { .. } => {
            let data = SurfaceUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                model: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                fill_light_dir: [fill_light_dir.x, fill_light_dir.y, fill_light_dir.z, 0.0],
                solid_color,
                edge_color: style.edge_color,
                nan_color: style.colormap.nan_color,
                scalar_min: scalar_range.0,
                scalar_max: scalar_range.1,
                use_colormap,
                log_scale: u32::from(style.colormap.log_scale),
                ambient: style.ambient,
                diffuse: (1.0 - style.ambient).max(0.2),
                specular: style.specular,
                shininess: style.shininess,
                opacity: style.opacity,
                edge_width: style.edge_width,
                show_edges: u32::from(style.representation == Representation::SurfaceWithEdges),
                two_sided: u32::from(style.two_sided),
            };
            queue.write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&data));
        }
        CachedGeometry::Lines { .. } => {
            let is_outline = style.representation == Representation::Outline;
            let data = LineUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                solid_color: if is_outline {
                    [0.90, 0.90, 0.92, 1.0]
                } else {
                    solid_color
                },
                nan_color: style.colormap.nan_color,
                scalar_min: scalar_range.0,
                scalar_max: scalar_range.1,
                use_colormap: if is_outline { 0 } else { use_colormap },
                log_scale: u32::from(style.colormap.log_scale),
                opacity: style.opacity,
                _pad: [0.0; 3],
            };
            queue.write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&data));
        }
        CachedGeometry::Points { .. } => {
            let data = PointUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                solid_color,
                nan_color: style.colormap.nan_color,
                viewport_size,
                point_size: style.point_size,
                opacity: style.opacity,
                scalar_min: scalar_range.0,
                scalar_max: scalar_range.1,
                use_colormap,
                log_scale: u32::from(style.colormap.log_scale),
            };
            queue.write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&data));
        }
        CachedGeometry::Volume {
            dims,
            box_min,
            box_max,
            ..
        } => {
            let bmin = Vec3::from_array(*box_min);
            let bmax = Vec3::from_array(*box_max);
            let span = (bmax - bmin).max(Vec3::splat(1e-5));
            let denom = Vec3::new(
                (dims[0].saturating_sub(1)).max(1) as f32,
                (dims[1].saturating_sub(1)).max(1) as f32,
                (dims[2].saturating_sub(1)).max(1) as f32,
            );
            let voxel = span / denom;
            let min_voxel = voxel.x.min(voxel.y).min(voxel.z).max(1e-5);
            let step_world = min_voxel * style.volume_step_size.clamp(0.05, 4.0);
            let max_steps = ((span.length() / step_world).ceil() as u32 + 16).min(2048);

            let (vmin, vmax) = if style.auto_range {
                scalar_range
            } else {
                (
                    style.volume_tf.scalar_range.0 as f32,
                    style.volume_tf.scalar_range.1 as f32,
                )
            };

            let data = VolumeUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                inv_view_proj: inv_view_proj.to_cols_array_2d(),
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                box_min: [box_min[0], box_min[1], box_min[2], 1.0],
                box_max: [box_max[0], box_max[1], box_max[2], 1.0],
                dims: [dims[0], dims[1], dims[2], 0],
                viewport_size,
                _pad0: 0.0,
                _pad1: 0.0,
                scalar_min: vmin,
                scalar_max: vmax,
                step_world,
                unit_distance: style.volume_tf.scalar_opacity_unit_distance.max(1e-4),
                shading: u32::from(style.volume_shading),
                log_scale: u32::from(style.volume_tf.color_map.log_scale),
                opacity: style.opacity,
                max_steps,
            };
            queue.write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&data));
        }
        CachedGeometry::Empty => {}
    }
}

fn create_lut_texture(
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

fn upload_lut_texture(queue: &wgpu::Queue, tex: &wgpu::Texture, rgba: &[u8]) {
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

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paraviz-depth-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_render_targets(device: &wgpu::Device, width: u32, height: u32) -> RenderTargets {
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
    let (depth_texture, depth_view) = create_depth_texture(device, width, height);
    RenderTargets {
        width,
        height,
        color_texture,
        color_view,
        depth_texture,
        depth_view,
    }
}

fn create_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
    lut_bgl: &wgpu::BindGroupLayout,
    volume_bgl: &wgpu::BindGroupLayout,
) -> PipelineBundle {
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let surface_vbl = wgpu::VertexBufferLayout {
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
        ],
    };

    let make_surface_pipeline = |label: &str, alpha_blend: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&actor_lut_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(surface_vbl.clone())],
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
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    let surface_opaque_pipeline = make_surface_pipeline("paraviz-surface-opaque", false);
    let surface_alpha_pipeline = make_surface_pipeline("paraviz-surface-alpha", true);

    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("paraviz-line-pipeline"),
        layout: Some(&actor_lut_layout),
        vertex: wgpu::VertexState {
            module: &line_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
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
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
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
        multisample: wgpu::MultisampleState::default(),
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    PipelineBundle {
        format,
        bg_pipeline,
        surface_opaque_pipeline,
        surface_alpha_pipeline,
        line_pipeline,
        point_pipeline,
        volume_pipeline,
        gizmo_pipeline,
    }
}
