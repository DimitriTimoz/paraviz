//! WGSL shaders for background gradient, surface/edges (3-point ParaView Light Kit),
//! wireframe/outline, sphere-impostor point sprites, 3D volume raymarching,
//! Eye-Dome Lighting (EDL), 3D viewport widgets, and orientation axes gizmo.

/// Fullscreen vertical gradient background shader.
pub const BACKGROUND_WGSL: &str = r#"
struct BgUniforms {
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: BgUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv_y: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<f32, 3>(0.0, 0.0, 2.0);
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.99999, 1.0);
    out.uv_y = uvs[vid];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let t = clamp(in.uv_y, 0.0, 1.0);
    let col = mix(u.bg_bottom.rgb, u.bg_top.rgb, t);
    return vec4<f32>(col, 1.0);
}
"#;

/// Surface and SurfaceWithEdges shader with single-pass fwidth() barycentric wireframe,
/// two-sided 3-point ParaView Light Kit + headlight Blinn-Phong lighting,
/// 1D LUT colormapping, and NaN handling.
pub const SURFACE_WGSL: &str = r#"
struct SurfaceUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    fill_light_dir: vec4<f32>,
    back_light_dir: vec4<f32>,
    solid_color: vec4<f32>,
    edge_color: vec4<f32>,
    nan_color: vec4<f32>,
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
};

@group(0) @binding(0)
var<uniform> u: SurfaceUniforms;

@group(1) @binding(0)
var lut_tex: texture_2d<f32>;
@group(1) @binding(1)
var lut_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) barycentric: vec3<f32>,
};

struct VsInIndexed {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) barycentric: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let wp = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.normal = (u.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.scalar = in.scalar;
    out.barycentric = in.barycentric;
    return out;
}

@vertex
fn vs_indexed(in: VsInIndexed) -> VsOut {
    var out: VsOut;
    let wp = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.normal = (u.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.scalar = in.scalar;
    out.barycentric = vec3<f32>(1.0, 1.0, 1.0);
    return out;
}

fn map_scalar(s: f32) -> vec4<f32> {
    if (s != s || abs(s) > 1e37) {
        return u.nan_color;
    }
    var t: f32 = 0.5;
    let span = u.scalar_max - u.scalar_min;
    if (abs(span) >= 1e-12) {
        if (u.log_scale != 0u && u.scalar_min > 0.0 && u.scalar_max > 0.0 && s > 0.0) {
            t = (log(s) - log(u.scalar_min)) / (log(u.scalar_max) - log(u.scalar_min));
        } else {
            t = (s - u.scalar_min) / span;
        }
    }
    let tc = clamp(t, 0.0, 1.0);
    return textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(tc, 0.5), 0.0);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var base_rgba = u.solid_color;
    if (u.use_colormap != 0u) {
        base_rgba = map_scalar(in.scalar);
    }

    var n = in.normal;
    let nlen = length(n);
    if (nlen > 1e-6) {
        n = n / nlen;
    } else {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }

    let v = normalize(u.camera_pos.xyz - in.world_pos);
    if (u.two_sided != 0u && (!front_facing || dot(n, v) < 0.0)) {
        n = -n;
    }

    // ParaView 3-point Light Kit + Camera Headlight:
    // 1) Warm Key Light
    let l_key = normalize(u.light_dir.xyz);
    let diff_key = max(dot(n, l_key), 0.0) * 0.72;
    let key_tint = vec3<f32>(1.0, 0.98, 0.95);

    // 2) Cool Fill Light (opposite side)
    let l_fill = normalize(u.fill_light_dir.xyz);
    let diff_fill = max(dot(n, l_fill), 0.0) * 0.32;
    let fill_tint = vec3<f32>(0.84, 0.91, 1.0);

    // 3) Rim / Back Light (defines silhouette and separates backside from dark bg)
    let l_back = normalize(u.back_light_dir.xyz);
    let diff_back = max(dot(n, l_back), 0.0) * 0.30;
    let rim = pow(clamp(1.0 - abs(dot(n, v)), 0.0, 1.0), 2.5) * 0.16;
    let back_tint = vec3<f32>(0.88, 0.93, 1.0);

    // 4) Camera Headlight (guarantees facing surfaces never go dark)
    let diff_head = max(dot(n, v), 0.0) * 0.24;

    let light_contrib = key_tint * diff_key
        + fill_tint * diff_fill
        + back_tint * (diff_back + rim)
        + vec3<f32>(diff_head);

    let h_key = normalize(l_key + v);
    let spec_key = pow(max(dot(n, h_key), 0.0), max(u.shininess, 1.0));
    let spec_head = pow(max(dot(n, v), 0.0), max(u.shininess * 0.75, 1.0)) * 0.22;

    var lit_rgb = base_rgba.rgb * (vec3<f32>(u.ambient) + u.diffuse * light_contrib)
        + key_tint * (u.specular * spec_key)
        + vec3<f32>(u.specular * spec_head);

    if (u.show_edges != 0u) {
        let d = max(fwidth(in.barycentric), vec3<f32>(1e-6));
        let a3 = smoothstep(vec3<f32>(0.0), d * max(u.edge_width, 0.5), in.barycentric);
        let edge_factor = 1.0 - min(min(a3.x, a3.y), a3.z);
        lit_rgb = mix(lit_rgb, u.edge_color.rgb, edge_factor * u.edge_color.a);
    }

    return vec4<f32>(lit_rgb, base_rgba.a * u.opacity);
}
"#;

/// Wireframe and bounding-box screen-space quad-expanded wide line shader (6 vertices per segment instance).
pub const LINE_WGSL: &str = r#"
struct LineUniforms {
    view_proj: mat4x4<f32>,
    solid_color: vec4<f32>,
    nan_color: vec4<f32>,
    scalar_min: f32,
    scalar_max: f32,
    use_colormap: u32,
    log_scale: u32,
    opacity: f32,
    line_width: f32,
    viewport_size: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> u: LineUniforms;

@group(1) @binding(0)
var lut_tex: texture_2d<f32>;
@group(1) @binding(1)
var lut_sampler: sampler;

struct VsIn {
    @location(0) pos_a: vec3<f32>,
    @location(1) scalar_a: f32,
    @location(2) pos_b: vec3<f32>,
    @location(3) scalar_b: f32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) scalar: f32,
};

@vertex
fn vs_main(in: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    var clip_a = u.view_proj * vec4<f32>(in.pos_a, 1.0);
    var clip_b = u.view_proj * vec4<f32>(in.pos_b, 1.0);
    var sa = in.scalar_a;
    var sb = in.scalar_b;

    // Guard against endpoints behind the near plane (clip.z < 0 or clip.w <= 0).
    let eps: f32 = 1e-5;
    if ((clip_a.w <= eps || clip_a.z < 0.0) && (clip_b.w <= eps || clip_b.z < 0.0)) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.scalar = sa;
        return out;
    }
    if (clip_a.z < 0.0) {
        let t = clamp(clip_a.z / (clip_a.z - clip_b.z), 0.0, 1.0);
        clip_a = mix(clip_a, clip_b, t);
        sa = mix(sa, sb, t);
    } else if (clip_b.z < 0.0) {
        let t = clamp(clip_b.z / (clip_b.z - clip_a.z), 0.0, 1.0);
        clip_b = mix(clip_b, clip_a, t);
        sb = mix(sb, sa, t);
    }
    if (clip_a.w <= eps) {
        let t = clamp((eps - clip_a.w) / max(clip_b.w - clip_a.w, eps), 0.0, 1.0);
        clip_a = mix(clip_a, clip_b, t);
        sa = mix(sa, sb, t);
    } else if (clip_b.w <= eps) {
        let t = clamp((eps - clip_b.w) / max(clip_a.w - clip_b.w, eps), 0.0, 1.0);
        clip_b = mix(clip_b, clip_a, t);
        sb = mix(sb, sa, t);
    }

    let half_vp = 0.5 * max(u.viewport_size, vec2<f32>(1.0));
    let scr_a = (clip_a.xy / clip_a.w) * half_vp;
    let scr_b = (clip_b.xy / clip_b.w) * half_vp;
    let delta = scr_b - scr_a;
    let seg_len = length(delta);
    var dir = vec2<f32>(1.0, 0.0);
    if (seg_len > 1e-4) {
        dir = delta / seg_len;
    }
    let perp = vec2<f32>(-dir.y, dir.x);

    // 2 triangles forming quad: (0, -1), (1, -1), (1, 1), (0, -1), (1, 1), (0, 1)
    var end_ts = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let end_t = end_ts[vid];
    let side = sides[vid];

    let width_px = max(u.line_width, 1.0);
    let half_width = width_px * 0.5;
    let cap = select(0.0, half_width, width_px > 1.05);

    let base_clip = mix(clip_a, clip_b, end_t);
    let along = select(-cap, cap, end_t > 0.5);
    let offset_px = dir * along + perp * (side * half_width);
    let offset_ndc = offset_px / half_vp;

    out.clip_pos = vec4<f32>(
        base_clip.xy + offset_ndc * base_clip.w,
        base_clip.z,
        base_clip.w,
    );
    out.scalar = mix(sa, sb, end_t);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var rgba = u.solid_color;
    if (u.use_colormap != 0u) {
        let s = in.scalar;
        if (s != s || abs(s) > 1e37) {
            rgba = u.nan_color;
        } else {
            var t: f32 = 0.5;
            let span = u.scalar_max - u.scalar_min;
            if (abs(span) >= 1e-12) {
                if (u.log_scale != 0u && u.scalar_min > 0.0 && u.scalar_max > 0.0 && s > 0.0) {
                    t = (log(s) - log(u.scalar_min)) / (log(u.scalar_max) - log(u.scalar_min));
                } else {
                    t = (s - u.scalar_min) / span;
                }
            }
            rgba = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
        }
    }
    return vec4<f32>(rgba.rgb, rgba.a * u.opacity);
}
"#;

/// Instanced quad sphere-impostor point shader with anti-aliased silhouette and
/// 3-point ParaView Light Kit + headlight pseudo-sphere shading.
pub const POINT_WGSL: &str = r#"
struct PointUniforms {
    view_proj: mat4x4<f32>,
    solid_color: vec4<f32>,
    nan_color: vec4<f32>,
    viewport_size: vec2<f32>,
    point_size: f32,
    opacity: f32,
    scalar_min: f32,
    scalar_max: f32,
    use_colormap: u32,
    log_scale: u32,
};

@group(0) @binding(0)
var<uniform> u: PointUniforms;

@group(1) @binding(0)
var lut_tex: texture_2d<f32>;
@group(1) @binding(1)
var lut_sampler: sampler;

struct VsIn {
    @location(0) quad_uv: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) scalar: f32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let center_clip = u.view_proj * vec4<f32>(in.position, 1.0);
    let px = in.quad_uv * (max(u.point_size, 1.0) * 0.5);
    let ndc_offset = vec2<f32>(
        px.x * 2.0 / max(u.viewport_size.x, 1.0),
        px.y * 2.0 / max(u.viewport_size.y, 1.0),
    );
    out.clip_pos = vec4<f32>(
        center_clip.xy + ndc_offset * center_clip.w,
        center_clip.z,
        center_clip.w,
    );
    out.quad_uv = in.quad_uv;
    out.scalar = in.scalar;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }

    var base_rgba = u.solid_color;
    if (u.use_colormap != 0u) {
        let s = in.scalar;
        if (s != s || abs(s) > 1e37) {
            base_rgba = u.nan_color;
        } else {
            var t: f32 = 0.5;
            let span = u.scalar_max - u.scalar_min;
            if (abs(span) >= 1e-12) {
                if (u.log_scale != 0u && u.scalar_min > 0.0 && u.scalar_max > 0.0 && s > 0.0) {
                    t = (log(s) - log(u.scalar_min)) / (log(u.scalar_max) - log(u.scalar_min));
                } else {
                    t = (s - u.scalar_min) / span;
                }
            }
            base_rgba = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
        }
    }

    let r = sqrt(r2);
    let aa = max(fwidth(r), 0.04);
    let edge_alpha = 1.0 - smoothstep(1.0 - aa, 1.0, r);

    let nz = sqrt(max(1.0 - r2, 0.0));
    let n = vec3<f32>(in.quad_uv.x, in.quad_uv.y, nz);
    let v = vec3<f32>(0.0, 0.0, 1.0);

    // 3-point ParaView Light Kit + Headlight in view space
    let l_key = normalize(vec3<f32>(0.35, 0.55, 0.75));
    let l_fill = normalize(vec3<f32>(-0.60, -0.35, 0.40));
    let l_back = normalize(vec3<f32>(-0.35, 0.65, -0.65));

    let diff_key = max(dot(n, l_key), 0.0) * 0.72;
    let diff_fill = max(dot(n, l_fill), 0.0) * 0.30;
    let diff_back = max(dot(n, l_back), 0.0) * 0.25;
    let diff_head = nz * 0.24;
    let rim = pow(1.0 - nz, 2.5) * 0.16;

    let light_rgb = vec3<f32>(1.0, 0.98, 0.95) * diff_key
        + vec3<f32>(0.84, 0.91, 1.0) * diff_fill
        + vec3<f32>(0.88, 0.93, 1.0) * (diff_back + rim)
        + vec3<f32>(diff_head);

    let h_key = normalize(l_key + v);
    let spec = pow(max(dot(n, h_key), 0.0), 32.0);
    let shaded = base_rgba.rgb * (vec3<f32>(0.18) + 0.82 * light_rgb) + vec3<f32>(0.25 * spec);

    return vec4<f32>(shaded, base_rgba.a * u.opacity * edge_alpha);
}
"#;

/// 3D Volume raymarching shader:
/// - Ray-AABB intersection
/// - Interleaved Gradient Noise (IGN) ray jittering
/// - Manual trilinear sampling on R32Float 3D texture via textureLoad
/// - Central-difference gradient Phong shading
/// - 1D Transfer Function LUT lookup
/// - Beer-Lambert opacity accumulation & early ray termination
pub const VOLUME_WGSL: &str = r#"
struct VolumeUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    dims: vec4<u32>,
    viewport_size: vec2<f32>,
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
    clip_origin: vec4<f32>,
    clip_normal: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: VolumeUniforms;

@group(1) @binding(0)
var vol_tex: texture_3d<f32>;
@group(1) @binding(1)
var tf_tex: texture_2d<f32>;
@group(1) @binding(2)
var tf_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = mix(u.box_min.xyz, u.box_max.xyz, in.position);
    out.clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let eps = 1e-8;
    let sx = select(rd.x, eps, abs(rd.x) < eps);
    let sy = select(rd.y, eps, abs(rd.y) < eps);
    let sz = select(rd.z, eps, abs(rd.z) < eps);
    let inv_rd = vec3<f32>(1.0 / sx, 1.0 / sy, 1.0 / sz);
    let t0 = (bmin - ro) * inv_rd;
    let t1 = (bmax - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

fn sample_trilinear(world_pos: vec3<f32>) -> f32 {
    let dims_i = vec3<i32>(u.dims.xyz);
    let dims_f = vec3<f32>(u.dims.xyz);
    let extent = max(u.box_max.xyz - u.box_min.xyz, vec3<f32>(1e-6));
    let uvw = clamp((world_pos - u.box_min.xyz) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    let coord = uvw * max(dims_f - vec3<f32>(1.0), vec3<f32>(0.0));

    let i0 = clamp(vec3<i32>(floor(coord)), vec3<i32>(0), max(dims_i - vec3<i32>(1), vec3<i32>(0)));
    let i1 = min(i0 + vec3<i32>(1), max(dims_i - vec3<i32>(1), vec3<i32>(0)));
    let f = coord - vec3<f32>(i0);

    let v000 = textureLoad(vol_tex, vec3<i32>(i0.x, i0.y, i0.z), 0).r;
    let v100 = textureLoad(vol_tex, vec3<i32>(i1.x, i0.y, i0.z), 0).r;
    let v010 = textureLoad(vol_tex, vec3<i32>(i0.x, i1.y, i0.z), 0).r;
    let v110 = textureLoad(vol_tex, vec3<i32>(i1.x, i1.y, i0.z), 0).r;
    let v001 = textureLoad(vol_tex, vec3<i32>(i0.x, i0.y, i1.z), 0).r;
    let v101 = textureLoad(vol_tex, vec3<i32>(i1.x, i0.y, i1.z), 0).r;
    let v011 = textureLoad(vol_tex, vec3<i32>(i0.x, i1.y, i1.z), 0).r;
    let v111 = textureLoad(vol_tex, vec3<i32>(i1.x, i1.y, i1.z), 0).r;

    let c00 = mix(v000, v100, f.x);
    let c10 = mix(v010, v110, f.x);
    let c01 = mix(v001, v101, f.x);
    let c11 = mix(v011, v111, f.x);

    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);
    return mix(c0, c1, f.z);
}

fn sample_tf(val: f32) -> vec4<f32> {
    if (val != val || abs(val) > 1e37) {
        return vec4<f32>(0.0);
    }
    var t: f32 = 0.5;
    let span = u.scalar_max - u.scalar_min;
    if (abs(span) >= 1e-12) {
        if (u.log_scale != 0u && u.scalar_min > 0.0 && u.scalar_max > 0.0 && val > 0.0) {
            t = (log(val) - log(u.scalar_min)) / (log(u.scalar_max) - log(u.scalar_min));
        } else {
            t = (val - u.scalar_min) / span;
        }
    }
    return textureSampleLevel(tf_tex, tf_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ndc_x = (in.clip_pos.x / max(u.viewport_size.x, 1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.clip_pos.y / max(u.viewport_size.y, 1.0)) * 2.0;
    let ndc_xy = vec2<f32>(ndc_x, ndc_y);
    let near_h = u.inv_view_proj * vec4<f32>(ndc_xy, 0.0, 1.0);
    let far_h = u.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let ro = near_h.xyz / near_h.w;
    let p_far = far_h.xyz / far_h.w;
    let rd = normalize(p_far - ro);

    let hit = intersect_aabb(ro, rd, u.box_min.xyz, u.box_max.xyz);
    let t_enter = hit.x;
    let t_exit = hit.y;
    if (t_exit <= t_enter) {
        discard;
    }

    let step_w = max(u.step_world, 1e-5);
    let dims_f = max(vec3<f32>(u.dims.xyz) - vec3<f32>(1.0), vec3<f32>(1.0));
    let voxel_size = max(u.box_max.xyz - u.box_min.xyz, vec3<f32>(1e-5)) / dims_f;

    var t = t_enter + step_w * ign(in.clip_pos.xy);
    var accum_rgb = vec3<f32>(0.0);
    var accum_a: f32 = 0.0;

    let l = normalize(u.light_dir.xyz);
    let v = -rd;

    for (var step: u32 = 0u; step < u.max_steps; step = step + 1u) {
        if (t > t_exit || accum_a >= 0.99) {
            break;
        }
        let pos = ro + rd * t;
        if (u.clip_origin.w > 0.5) {
            if (dot(pos - u.clip_origin.xyz, u.clip_normal.xyz) < 0.0) {
                t = t + step_w;
                continue;
            }
        }
        let val = sample_trilinear(pos);
        let tf = sample_tf(val);
        let raw_a = clamp(tf.a * u.opacity, 0.0, 0.999);

        if (raw_a > 1e-4) {
            var rgb = tf.rgb;
            if (u.shading != 0u) {
                let gx = sample_trilinear(pos + vec3<f32>(voxel_size.x, 0.0, 0.0))
                       - sample_trilinear(pos - vec3<f32>(voxel_size.x, 0.0, 0.0));
                let gy = sample_trilinear(pos + vec3<f32>(0.0, voxel_size.y, 0.0))
                       - sample_trilinear(pos - vec3<f32>(0.0, voxel_size.y, 0.0));
                let gz = sample_trilinear(pos + vec3<f32>(0.0, 0.0, voxel_size.z))
                       - sample_trilinear(pos - vec3<f32>(0.0, 0.0, voxel_size.z));
                let grad = vec3<f32>(gx, gy, gz);
                let glen = length(grad);
                if (glen > 1e-5) {
                    var n = -grad / glen;
                    if (dot(n, v) < 0.0) {
                        n = -n;
                    }
                    let diff = max(dot(n, l), 0.0);
                    let h = normalize(l + v);
                    let spec = pow(max(dot(n, h), 0.0), 28.0);
                    rgb = rgb * (0.28 + 0.78 * diff) + vec3<f32>(0.22 * spec);
                }
            }

            // Beer-Lambert optical density integration normalized to scalar_opacity_unit_distance.
            let extinction = -log(max(1.0 - raw_a, 1e-4));
            let alpha_step = 1.0 - exp(-extinction * step_w / max(u.unit_distance, 1e-4));

            accum_rgb = accum_rgb + (1.0 - accum_a) * rgb * alpha_step;
            accum_a = accum_a + (1.0 - accum_a) * alpha_step;
        }

        t = t + step_w;
    }

    if (accum_a <= 1e-5) {
        discard;
    }
    return vec4<f32>(accum_rgb, accum_a);
}
"#;

/// Fullscreen Eye-Dome Lighting (EDL) post-processing shader sampling 8 neighbor depths.
pub const EDL_WGSL: &str = r#"
struct EdlUniforms {
    near: f32,
    far: f32,
    edl_strength: f32,
    edl_radius: f32,
    orthographic: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<uniform> u: EdlUniforms;

@group(1) @binding(0)
var color_tex: texture_2d<f32>;
@group(1) @binding(1)
var depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

fn linearize_depth(d: f32) -> f32 {
    if (u.orthographic != 0u) {
        return u.near + d * (u.far - u.near);
    }
    return (u.near * u.far) / max(u.far - d * (u.far - u.near), 1e-6);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let px = clamp(vec2<i32>(in.pos.xy), vec2<i32>(0), dims - vec2<i32>(1));
    let base_col = textureLoad(color_tex, px, 0);
    let d_center = textureLoad(depth_tex, px, 0);

    // Background pixels (depth cleared to 1.0) remain unshaded.
    if (d_center >= 0.99999) {
        return base_col;
    }

    let z_center = max(linearize_depth(d_center), 1e-5);
    let log2_depth = log2(z_center);

    var offsets = array<vec2<i32>, 8>(
        vec2<i32>(-1,  0),
        vec2<i32>( 1,  0),
        vec2<i32>( 0, -1),
        vec2<i32>( 0,  1),
        vec2<i32>(-1, -1),
        vec2<i32>( 1, -1),
        vec2<i32>(-1,  1),
        vec2<i32>( 1,  1),
    );

    let radius = i32(clamp(round(u.edl_radius), 1.0, 4.0));
    var response: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i = i + 1) {
        let n_px = clamp(px + offsets[i] * radius, vec2<i32>(0), dims - vec2<i32>(1));
        let d_n = textureLoad(depth_tex, n_px, 0);
        let z_n = max(linearize_depth(d_n), 1e-5);
        let log2_neighbor = log2(z_n);
        response = response + max(0.0, log2_depth - log2_neighbor);
    }

    let shade = exp(-max(u.edl_strength, 0.0) * response);
    return vec4<f32>(base_col.rgb * shade, base_col.a);
}
"#;

/// 4x MSAA Depth Resolve shader writing minimum sample depth into a 1x Depth32Float texture for EDL.
pub const DEPTH_RESOLVE_WGSL: &str = r#"
@group(0) @binding(0)
var msaa_depth: texture_depth_multisampled_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let dims = vec2<i32>(textureDimensions(msaa_depth));
    let px = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dims - vec2<i32>(1));
    let d0 = textureLoad(msaa_depth, px, 0);
    let d1 = textureLoad(msaa_depth, px, 1);
    let d2 = textureLoad(msaa_depth, px, 2);
    let d3 = textureLoad(msaa_depth, px, 3);
    return min(min(d0, d1), min(d2, d3));
}
"#;

/// 3D Viewport Widget overlay shader (per-vertex RGBA for crisp quad-expanded wide wireframes and translucent guides).
pub const WIDGET_WGSL: &str = r#"
struct WidgetUniforms {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> u: WidgetUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsLineIn {
    @location(0) pos_a: vec3<f32>,
    @location(1) col_a: vec4<f32>,
    @location(2) pos_b: vec3<f32>,
    @location(3) col_b: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@vertex
fn vs_line(in: VsLineIn, @builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    var clip_a = u.view_proj * vec4<f32>(in.pos_a, 1.0);
    var clip_b = u.view_proj * vec4<f32>(in.pos_b, 1.0);
    var ca = in.col_a;
    var cb = in.col_b;

    let eps: f32 = 1e-5;
    if ((clip_a.w <= eps || clip_a.z < 0.0) && (clip_b.w <= eps || clip_b.z < 0.0)) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = ca;
        return out;
    }
    if (clip_a.z < 0.0) {
        let t = clamp(clip_a.z / (clip_a.z - clip_b.z), 0.0, 1.0);
        clip_a = mix(clip_a, clip_b, t);
        ca = mix(ca, cb, t);
    } else if (clip_b.z < 0.0) {
        let t = clamp(clip_b.z / (clip_b.z - clip_a.z), 0.0, 1.0);
        clip_b = mix(clip_b, clip_a, t);
        cb = mix(cb, ca, t);
    }
    if (clip_a.w <= eps) {
        let t = clamp((eps - clip_a.w) / max(clip_b.w - clip_a.w, eps), 0.0, 1.0);
        clip_a = mix(clip_a, clip_b, t);
        ca = mix(ca, cb, t);
    } else if (clip_b.w <= eps) {
        let t = clamp((eps - clip_b.w) / max(clip_a.w - clip_b.w, eps), 0.0, 1.0);
        clip_b = mix(clip_b, clip_a, t);
        cb = mix(cb, ca, t);
    }

    let half_vp = 0.5 * max(u.viewport_size, vec2<f32>(1.0));
    let scr_a = (clip_a.xy / clip_a.w) * half_vp;
    let scr_b = (clip_b.xy / clip_b.w) * half_vp;
    let delta = scr_b - scr_a;
    let seg_len = length(delta);
    var dir = vec2<f32>(1.0, 0.0);
    if (seg_len > 1e-4) {
        dir = delta / seg_len;
    }
    let perp = vec2<f32>(-dir.y, dir.x);

    var end_ts = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let end_t = end_ts[vid];
    let side = sides[vid];

    let width_px = max(u.line_width, 1.0);
    let half_width = width_px * 0.5;
    let cap = select(0.0, half_width, width_px > 1.05);

    let base_clip = mix(clip_a, clip_b, end_t);
    let along = select(-cap, cap, end_t > 0.5);
    let offset_px = dir * along + perp * (side * half_width);
    let offset_ndc = offset_px / half_vp;

    out.clip_pos = vec4<f32>(
        base_clip.xy + offset_ndc * base_clip.w,
        base_clip.z,
        base_clip.w,
    );
    out.color = mix(ca, cb, end_t);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// 3D Orientation Axes Gizmo shader (bottom-left RGB XYZ triad).
pub const GIZMO_WGSL: &str = r#"
struct GizmoUniforms {
    mvp: mat4x4<f32>,
    light_dir: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: GizmoUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.mvp * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(u.light_dir.xyz);
    let diff = max(dot(n, l), 0.0);
    let rgb = in.color * (0.35 + 0.65 * diff);
    return vec4<f32>(rgb, 1.0);
}
"#;
