use super::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct BgUniforms {
    pub(super) bg_top: [f32; 4],
    pub(super) bg_bottom: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct SurfaceUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) model: [[f32; 4]; 4],
    pub(super) camera_pos: [f32; 4],
    pub(super) light_dir: [f32; 4],
    pub(super) fill_light_dir: [f32; 4],
    pub(super) back_light_dir: [f32; 4],
    pub(super) solid_color: [f32; 4],
    pub(super) edge_color: [f32; 4],
    pub(super) nan_color: [f32; 4],
    pub(super) scalar_min: f32,
    pub(super) scalar_max: f32,
    pub(super) use_colormap: u32,
    pub(super) log_scale: u32,
    pub(super) ambient: f32,
    pub(super) diffuse: f32,
    pub(super) specular: f32,
    pub(super) shininess: f32,
    pub(super) opacity: f32,
    pub(super) edge_width: f32,
    pub(super) show_edges: u32,
    pub(super) two_sided: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct LineUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) solid_color: [f32; 4],
    pub(super) nan_color: [f32; 4],
    pub(super) scalar_min: f32,
    pub(super) scalar_max: f32,
    pub(super) use_colormap: u32,
    pub(super) log_scale: u32,
    pub(super) opacity: f32,
    pub(super) line_width: f32,
    pub(super) viewport_size: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct PointUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) solid_color: [f32; 4],
    pub(super) nan_color: [f32; 4],
    pub(super) viewport_size: [f32; 2],
    pub(super) point_size: f32,
    pub(super) opacity: f32,
    pub(super) scalar_min: f32,
    pub(super) scalar_max: f32,
    pub(super) use_colormap: u32,
    pub(super) log_scale: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct VolumeUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) inv_view_proj: [[f32; 4]; 4],
    pub(super) camera_pos: [f32; 4],
    pub(super) light_dir: [f32; 4],
    pub(super) box_min: [f32; 4],
    pub(super) box_max: [f32; 4],
    pub(super) dims: [u32; 4],
    pub(super) viewport_size: [f32; 2],
    pub(super) _pad0: f32,
    pub(super) _pad1: f32,
    pub(super) scalar_min: f32,
    pub(super) scalar_max: f32,
    pub(super) step_world: f32,
    pub(super) unit_distance: f32,
    pub(super) shading: u32,
    pub(super) log_scale: u32,
    pub(super) opacity: f32,
    pub(super) max_steps: u32,
    pub(super) clip_origin: [f32; 4],
    pub(super) clip_normal: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct EdlUniforms {
    pub(super) near: f32,
    pub(super) far: f32,
    pub(super) edl_strength: f32,
    pub(super) edl_radius: f32,
    pub(super) orthographic: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct WidgetUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) viewport_size: [f32; 2],
    pub(super) line_width: f32,
    pub(super) _pad: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct GizmoUniforms {
    pub(super) mvp: [[f32; 4]; 4],
    pub(super) light_dir: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct QuadVertex {
    pub(super) uv: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct CubeVertex {
    pub(super) pos: [f32; 3],
}


pub(super) struct DynamicVertexBuffer {
    pub(super) buffer: wgpu::Buffer,
    pub(super) capacity_bytes: u64,
    pub(super) used_bytes: u64,
    pub(super) label: &'static str,
}

impl DynamicVertexBuffer {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, label: &'static str, initial_bytes: u64, data: &[u8]) -> Self {
        let capacity_bytes = initial_bytes.max(data.len() as u64).max(1024).next_power_of_two();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            queue.write_buffer(&buffer, 0, data);
        }
        Self {
            buffer,
            capacity_bytes,
            used_bytes: data.len() as u64,
            label,
        }
    }

    pub(super) fn upload<T: Pod>(
        slot: &mut Option<Self>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        items: &[T],
    ) {
        let bytes: &[u8] = bytemuck::cast_slice(items);
        let needed = bytes.len() as u64;
        if needed == 0 {
            if let Some(buf) = slot.as_mut() {
                buf.used_bytes = 0;
            }
            return;
        }
        match slot.as_mut() {
            Some(buf) if buf.capacity_bytes >= needed => {
                queue.write_buffer(&buf.buffer, 0, bytes);
                buf.used_bytes = needed;
            }
            Some(buf) => {
                *buf = Self::new(device, queue, buf.label, needed, bytes);
            }
            None => {
                *slot = Some(Self::new(device, queue, label, needed, bytes));
            }
        }
    }

    pub(super) fn slice_used(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..self.used_bytes)
    }
}

pub(super) struct RenderTargets {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color_texture: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) depth_texture: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
}

pub(super) struct MsaaTargets {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: wgpu::TextureFormat,
    #[allow(dead_code)]
    pub(super) color_texture: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) depth_texture: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) depth_resolve_bg: wgpu::BindGroup,
}

pub(super) struct EdlTargets {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: wgpu::TextureFormat,
    #[allow(dead_code)]
    pub(super) color_texture: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) depth_texture: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) tex_bind_group: wgpu::BindGroup,
}

pub(super) enum CachedGeometry {
    SurfaceIndexed {
        geom_buffer: wgpu::Buffer,
        scalar_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        vertex_count: u32,
        index_count: u32,
        scalar_source: SurfaceScalarSource,
        auto_range: (f64, f64),
    },
    SurfaceEdges {
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

pub(super) struct CachedActor {
    pub(super) dataset: Arc<DataSet>,
    pub(super) representation: Representation,
    pub(super) color_by: ColorBy,
    pub(super) geometry: CachedGeometry,
    pub(super) colormap: ColorMap,
    pub(super) lut_texture: wgpu::Texture,
    pub(super) lut_bind_group: wgpu::BindGroup,
    pub(super) volume_tf: TransferFunction,
    pub(super) tf_texture: wgpu::Texture,
    pub(super) volume_bind_group: Option<wgpu::BindGroup>,
    pub(super) uniform_buffer: wgpu::Buffer,
    pub(super) uniform_bind_group: wgpu::BindGroup,
}


pub(super) fn clamp_msaa_samples(samples: u32) -> u32 {
    if samples <= 1 {
        1
    } else {
        4
    }
}


pub(super) fn resolve_scalar_range(style: &ActorStyle, geom: &CachedGeometry) -> (f64, f64) {
    if style.auto_range {
        match geom {
            CachedGeometry::SurfaceIndexed { auto_range, .. }
            | CachedGeometry::SurfaceEdges { auto_range, .. }
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
pub(super) fn write_actor_uniforms(
    queue: &wgpu::Queue,
    cached: &CachedActor,
    style: &ActorStyle,
    view_proj: Mat4,
    inv_view_proj: Mat4,
    camera_pos: Vec3,
    light_dir: Vec3,
    fill_light_dir: Vec3,
    back_light_dir: Vec3,
    viewport_size: [f32; 2],
    scalar_range: (f32, f32),
) {
    let (solid_color, use_colormap) = match style.color_by {
        ColorBy::Solid(rgba) => (rgba, 0u32),
        ColorBy::PointScalar { .. } | ColorBy::CellScalar { .. } => ([1.0, 1.0, 1.0, 1.0], 1u32),
    };

    match &cached.geometry {
        CachedGeometry::SurfaceIndexed { .. } | CachedGeometry::SurfaceEdges { .. } => {
            let data = SurfaceUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                model: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                fill_light_dir: [fill_light_dir.x, fill_light_dir.y, fill_light_dir.z, 0.0],
                back_light_dir: [back_light_dir.x, back_light_dir.y, back_light_dir.z, 0.0],
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
                line_width: style.line_width,
                viewport_size,
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

            let (clip_origin, clip_normal) = match style.volume_clip_plane {
                Some((orig, norm)) => {
                    let n = glam::Vec3::from_array(norm).normalize_or(glam::Vec3::Z);
                    ([orig[0], orig[1], orig[2], 1.0], [n.x, n.y, n.z, 0.0])
                }
                None => ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
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
                clip_origin,
                clip_normal,
            };
            queue.write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&data));
        }
        CachedGeometry::Empty => {}
    }
}

