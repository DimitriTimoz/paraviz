//! 3D CPU raycast picking across visible [`SceneActor`]s using AABB pre-culling,
//! Möller–Trumbore ray-triangle intersection, and ray-point / ray-segment distance tests.

use crate::mesh_prep;
use crate::scene::{ActorStyle, ColorBy, RenderScene, Representation, SceneActor};
use glam::Vec3;
use paraviz_core::{BoundingBox, DataArray, DataSet, FieldData, PolyData};

/// Result of a 3D viewport pick query from [`pick_scene`].
#[derive(Debug, Clone, PartialEq)]
pub struct PickResult {
    pub actor_id: u64,
    pub actor_name: String,
    pub world_pos: [f32; 3],
    pub distance: f32,
    pub point_index: Option<usize>,
    pub cell_index: Option<usize>,
    pub scalar_value: Option<f64>,
}

/// Casts a ray from pixel `(px, py)` in a `width x height` viewport against all visible actors
/// in `scene`, returning the closest hit if any.
pub fn pick_scene(
    scene: &RenderScene,
    px: f32,
    py: f32,
    width: u32,
    height: u32,
) -> Option<PickResult> {
    if width == 0 || height == 0 {
        return None;
    }

    let (ray_origin, ray_dir) = scene.camera.screen_ray(px, py, width, height);
    let pixel_world_scale = if scene.camera.orthographic {
        (2.0 * scene.camera.ortho_scale.max(1e-5)) / height as f32
    } else {
        let half_fov = (scene.camera.fov_y_deg.to_radians() * 0.5).clamp(0.01, 1.5);
        2.0 * half_fov.tan() / height as f32
    };

    let mut best: Option<PickResult> = None;

    for actor in &scene.actors {
        if !actor.style.visible {
            continue;
        }
        let max_t = best.as_ref().map_or(f32::INFINITY, |b| b.distance);
        if let Some(hit) = pick_actor(actor, ray_origin, ray_dir, pixel_world_scale, scene.camera.orthographic, max_t)
            && best.as_ref().is_none_or(|b| hit.distance < b.distance)
        {
            best = Some(hit);
        }
    }

    best
}

fn pick_actor(
    actor: &SceneActor,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let bounds = actor.dataset.bounds();
    if !bounds.is_valid() {
        return None;
    }

    // Pad AABB for point/wireframe representations so grazing picks on boundary points still pass pre-cull.
    let aabb_pad = match actor.style.representation {
        Representation::Points => {
            let approx_dist = (bounds.center() - ray_origin).length().max(1.0);
            let scale = if orthographic { 1.0 } else { approx_dist };
            actor.style.point_size.max(4.0) * scale * pixel_world_scale + bounds.diagonal() * 0.02
        }
        Representation::Wireframe | Representation::Outline => {
            let approx_dist = (bounds.center() - ray_origin).length().max(1.0);
            let scale = if orthographic { 1.0 } else { approx_dist };
            actor.style.line_width.max(2.0) * 2.0 * scale * pixel_world_scale + bounds.diagonal() * 0.02
        }
        Representation::Surface | Representation::SurfaceWithEdges | Representation::Volume => 1e-4,
    };

    let padded = BoundingBox::new(
        bounds.min - Vec3::splat(aabb_pad),
        bounds.max + Vec3::splat(aabb_pad),
    );
    let (t_enter, t_exit) = ray_aabb_intersect(ray_origin, ray_dir, &padded)?;
    if t_exit < 0.0 || t_enter > max_t {
        return None;
    }

    match actor.style.representation {
        Representation::Surface | Representation::SurfaceWithEdges => {
            pick_surface(actor, ray_origin, ray_dir, max_t)
        }
        Representation::Points => {
            pick_points(actor, ray_origin, ray_dir, pixel_world_scale, orthographic, max_t)
        }
        Representation::Wireframe => {
            pick_wireframe(actor, ray_origin, ray_dir, pixel_world_scale, orthographic, max_t)
        }
        Representation::Outline => {
            let outline = mesh_prep::prepare_outline(&bounds);
            let mut best_t = max_t;
            let mut best_pos = None;
            for seg in outline.as_chunks::<2>().0 {
                let a = Vec3::from_array(seg[0].position);
                let b = Vec3::from_array(seg[1].position);
                if let Some((t, hit_pos)) =
                    ray_segment_hit(ray_origin, ray_dir, a, b, actor.style.line_width, pixel_world_scale, orthographic)
                    && t < best_t
                {
                    best_t = t;
                    best_pos = Some(hit_pos);
                }
            }
            best_pos.map(|p| PickResult {
                actor_id: actor.id,
                actor_name: actor.name.clone(),
                world_pos: p.to_array(),
                distance: best_t,
                point_index: None,
                cell_index: None,
                scalar_value: None,
            })
        }
        Representation::Volume => {
            let t = t_enter.max(0.0);
            if t >= max_t {
                return None;
            }
            let world_pos = ray_origin + ray_dir * t;
            let scalar_value = sample_volume_scalar(&actor.dataset, &actor.style, world_pos);
            Some(PickResult {
                actor_id: actor.id,
                actor_name: actor.name.clone(),
                world_pos: world_pos.to_array(),
                distance: t,
                point_index: None,
                cell_index: None,
                scalar_value,
            })
        }
    }
}

fn pick_surface(
    actor: &SceneActor,
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_t: f32,
) -> Option<PickResult> {
    if let DataSet::PolyData(poly) = actor.dataset.as_ref() {
        return pick_polydata_surface(actor, poly, ray_origin, ray_dir, max_t);
    }

    // Fallback for volumetric / grid / multiblock datasets: intersect extracted boundary surface triangles.
    let prep = mesh_prep::prepare_surface(&actor.dataset, &actor.style.color_by);
    let mut best_t = max_t;
    let mut best_pos = None;
    let mut best_scalar = None;

    for tri in prep.vertices.as_chunks::<3>().0 {
        let p0 = Vec3::from_array(tri[0].position);
        let p1 = Vec3::from_array(tri[1].position);
        let p2 = Vec3::from_array(tri[2].position);
        if let Some((t, u, v)) = ray_triangle_intersect(ray_origin, ray_dir, p0, p1, p2)
            && t < best_t
        {
            best_t = t;
            let w = 1.0 - u - v;
            best_pos = Some(p0 * w + p1 * u + p2 * v);
            let s = if !matches!(actor.style.color_by, ColorBy::Solid(_)) {
                Some((tri[0].scalar * w + tri[1].scalar * u + tri[2].scalar * v) as f64)
            } else {
                None
            };
            best_scalar = s;
        }
    }

    best_pos.map(|pos| PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        world_pos: pos.to_array(),
        distance: best_t,
        point_index: None,
        cell_index: None,
        scalar_value: best_scalar,
    })
}

struct PolyTriHit {
    pos: Vec3,
    point_index: usize,
    cell_index: usize,
    bary: [f32; 3],
    tri_idx: [usize; 3],
}

fn pick_polydata_surface(
    actor: &SceneActor,
    poly: &PolyData,
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_t: f32,
) -> Option<PickResult> {
    let mut best_t = max_t;
    let mut best_hit: Option<PolyTriHit> = None;

    let poly_base = poly.verts.len() + poly.lines.len();
    for (pi, p) in poly.polys.iter().enumerate() {
        if p.len() < 3 {
            continue;
        }
        let cell_idx = poly_base + pi;
        let i0 = p[0] as usize;
        if i0 >= poly.points.len() {
            continue;
        }
        let p0 = Vec3::from_array(poly.points[i0]);
        for k in 1..(p.len() - 1) {
            let i1 = p[k] as usize;
            let i2 = p[k + 1] as usize;
            if i1 >= poly.points.len() || i2 >= poly.points.len() {
                continue;
            }
            let p1 = Vec3::from_array(poly.points[i1]);
            let p2 = Vec3::from_array(poly.points[i2]);
            if let Some((t, u, v)) = ray_triangle_intersect(ray_origin, ray_dir, p0, p1, p2)
                && t < best_t
            {
                best_t = t;
                let w = 1.0 - u - v;
                let hit_pos = p0 * w + p1 * u + p2 * v;
                let nearest_pt = if w >= u && w >= v {
                    i0
                } else if u >= v {
                    i1
                } else {
                    i2
                };
                best_hit = Some(PolyTriHit {
                    pos: hit_pos,
                    point_index: nearest_pt,
                    cell_index: cell_idx,
                    bary: [w, u, v],
                    tri_idx: [i0, i1, i2],
                });
            }
        }
    }

    let strip_base = poly_base + poly.polys.len();
    for (si, s) in poly.strips.iter().enumerate() {
        if s.len() < 3 {
            continue;
        }
        let cell_idx = strip_base + si;
        for k in 0..(s.len() - 2) {
            let (i0, i1, i2) = if k % 2 == 0 {
                (s[k] as usize, s[k + 1] as usize, s[k + 2] as usize)
            } else {
                (s[k + 1] as usize, s[k] as usize, s[k + 2] as usize)
            };
            if i0 >= poly.points.len() || i1 >= poly.points.len() || i2 >= poly.points.len() {
                continue;
            }
            let p0 = Vec3::from_array(poly.points[i0]);
            let p1 = Vec3::from_array(poly.points[i1]);
            let p2 = Vec3::from_array(poly.points[i2]);
            if let Some((t, u, v)) = ray_triangle_intersect(ray_origin, ray_dir, p0, p1, p2)
                && t < best_t
            {
                best_t = t;
                let w = 1.0 - u - v;
                let hit_pos = p0 * w + p1 * u + p2 * v;
                let nearest_pt = if w >= u && w >= v {
                    i0
                } else if u >= v {
                    i1
                } else {
                    i2
                };
                best_hit = Some(PolyTriHit {
                    pos: hit_pos,
                    point_index: nearest_pt,
                    cell_index: cell_idx,
                    bary: [w, u, v],
                    tri_idx: [i0, i1, i2],
                });
            }
        }
    }

    let hit = best_hit?;
    let scalar_value = resolve_triangle_scalar(
        &poly.point_data,
        &poly.cell_data,
        &actor.style.color_by,
        hit.tri_idx,
        hit.bary,
        hit.cell_index,
    );

    Some(PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        world_pos: hit.pos.to_array(),
        distance: best_t,
        point_index: Some(hit.point_index),
        cell_index: Some(hit.cell_index),
        scalar_value,
    })
}

fn pick_points(
    actor: &SceneActor,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let prep = mesh_prep::prepare_points(&actor.dataset, &actor.style.color_by);
    let mut best_t = max_t;
    let mut best_hit: Option<(Vec3, usize, f32)> = None;

    for (idx, inst) in prep.instances.iter().enumerate() {
        let p = Vec3::from_array(inst.position);
        let to_p = p - ray_origin;
        let t = to_p.dot(ray_dir);
        if t < 0.0 || t >= best_t {
            continue;
        }
        let closest = ray_origin + ray_dir * t;
        let perp_dist = (p - closest).length();
        let scale = if orthographic { 1.0 } else { t };
        let radius = (actor.style.point_size.max(4.0) * 0.5) * scale * pixel_world_scale;
        if perp_dist <= radius {
            best_t = t;
            best_hit = Some((p, idx, inst.scalar));
        }
    }

    let (pos, idx, scalar) = best_hit?;
    let scalar_value = if !matches!(actor.style.color_by, ColorBy::Solid(_)) {
        Some(scalar as f64)
    } else {
        actor
            .dataset
            .point_data()
            .and_then(|pd| pd.active_scalars_array())
            .filter(|a| idx < a.num_tuples())
            .map(|a| a.get_magnitude(idx))
    };

    Some(PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        world_pos: pos.to_array(),
        distance: best_t,
        point_index: Some(idx),
        cell_index: None,
        scalar_value,
    })
}

fn pick_wireframe(
    actor: &SceneActor,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let prep = mesh_prep::prepare_wireframe(&actor.dataset, &actor.style.color_by);
    let mut best_t = max_t;
    let mut best_hit: Option<(Vec3, f32)> = None;

    for seg in prep.vertices.as_chunks::<2>().0 {
        let a = Vec3::from_array(seg[0].position);
        let b = Vec3::from_array(seg[1].position);
        if let Some((t, hit_pos)) =
            ray_segment_hit(ray_origin, ray_dir, a, b, actor.style.line_width, pixel_world_scale, orthographic)
            && t < best_t
        {
            best_t = t;
            let len_sq = (b - a).length_squared();
            let u = if len_sq > 1e-10 {
                ((hit_pos - a).dot(b - a) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let s = seg[0].scalar * (1.0 - u) + seg[1].scalar * u;
            best_hit = Some((hit_pos, s));
        }
    }

    let (pos, scalar) = best_hit?;
    let scalar_value = if !matches!(actor.style.color_by, ColorBy::Solid(_)) {
        Some(scalar as f64)
    } else {
        None
    };

    Some(PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        world_pos: pos.to_array(),
        distance: best_t,
        point_index: None,
        cell_index: None,
        scalar_value,
    })
}

fn resolve_array<'a>(field: &'a FieldData, name: &str) -> Option<&'a DataArray> {
    if name.is_empty() {
        field.active_scalars_array()
    } else {
        field.get(name)
    }
}

fn array_component(arr: &DataArray, tuple_idx: usize, component: Option<usize>) -> f64 {
    match component {
        Some(c) if c < arr.num_components => arr.data.get_f64(tuple_idx * arr.num_components + c),
        _ => arr.get_magnitude(tuple_idx),
    }
}

fn resolve_triangle_scalar(
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    tri_idx: [usize; 3],
    bary: [f32; 3],
    cell_idx: usize,
) -> Option<f64> {
    match color_by {
        ColorBy::PointScalar { array, component } => {
            let arr = resolve_array(point_data, array)?;
            let n = arr.num_tuples();
            if tri_idx[0] >= n || tri_idx[1] >= n || tri_idx[2] >= n {
                return None;
            }
            let s0 = array_component(arr, tri_idx[0], *component);
            let s1 = array_component(arr, tri_idx[1], *component);
            let s2 = array_component(arr, tri_idx[2], *component);
            Some(s0 * bary[0] as f64 + s1 * bary[1] as f64 + s2 * bary[2] as f64)
        }
        ColorBy::CellScalar { array, component } => {
            let arr = resolve_array(cell_data, array)?;
            if cell_idx >= arr.num_tuples() {
                return None;
            }
            Some(array_component(arr, cell_idx, *component))
        }
        ColorBy::Solid(_) => {
            if let Some(arr) = point_data.active_scalars_array() {
                let n = arr.num_tuples();
                if tri_idx[0] < n && tri_idx[1] < n && tri_idx[2] < n {
                    let s0 = array_component(arr, tri_idx[0], None);
                    let s1 = array_component(arr, tri_idx[1], None);
                    let s2 = array_component(arr, tri_idx[2], None);
                    return Some(s0 * bary[0] as f64 + s1 * bary[1] as f64 + s2 * bary[2] as f64);
                }
            }
            if let Some(arr) = cell_data.active_scalars_array()
                && cell_idx < arr.num_tuples()
            {
                return Some(array_component(arr, cell_idx, None));
            }
            None
        }
    }
}

fn sample_volume_scalar(dataset: &DataSet, style: &ActorStyle, world_pos: Vec3) -> Option<f64> {
    let img = match dataset {
        DataSet::ImageData(img) => img,
        DataSet::MultiBlock(mb) => mb.blocks.iter().find_map(|(_, b)| b.as_image_data())?,
        _ => return None,
    };
    let (name, comp) = match &style.color_by {
        ColorBy::PointScalar { array, component } => (array.as_str(), *component),
        ColorBy::Solid(_) | ColorBy::CellScalar { .. } => ("", None),
    };
    let arr = resolve_array(&img.point_data, name)?;
    let vals = img.sample_point_array(&arr.name, world_pos)?;
    match comp {
        Some(c) if c < vals.len() => Some(vals[c]),
        _ if vals.len() == 1 => Some(vals[0]),
        _ => Some(vals.iter().map(|v| v * v).sum::<f64>().sqrt()),
    }
}

/// Slab-method ray-AABB intersection returning `Some((t_enter, t_exit))` if `t_exit >= t_enter.max(0.0)`.
pub(crate) fn ray_aabb_intersect(
    ro: Vec3,
    rd: Vec3,
    bounds: &BoundingBox,
) -> Option<(f32, f32)> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    let bmin = bounds.min.to_array();
    let bmax = bounds.max.to_array();
    let ro_a = ro.to_array();
    let rd_a = rd.to_array();

    for axis in 0..3 {
        if rd_a[axis].abs() < 1e-8 {
            if ro_a[axis] < bmin[axis] || ro_a[axis] > bmax[axis] {
                return None;
            }
        } else {
            let inv = 1.0 / rd_a[axis];
            let mut t0 = (bmin[axis] - ro_a[axis]) * inv;
            let mut t1 = (bmax[axis] - ro_a[axis]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmax < tmin {
                return None;
            }
        }
    }

    if tmax < 0.0 {
        None
    } else {
        Some((tmin.max(0.0), tmax))
    }
}

/// Two-sided Möller–Trumbore ray-triangle intersection returning `(t, u, v)`.
pub(crate) fn ray_triangle_intersect(
    ro: Vec3,
    rd: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(f32, f32, f32)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = rd.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ro - v0;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = rd.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv_det;
    if t >= 0.0 {
        Some((t, u, v))
    } else {
        None
    }
}

fn ray_segment_hit(
    ro: Vec3,
    rd: Vec3,
    a: Vec3,
    b: Vec3,
    line_width: f32,
    pixel_world_scale: f32,
    orthographic: bool,
) -> Option<(f32, Vec3)> {
    let d1 = rd;
    let d2 = b - a;
    let r = ro - a;
    let a_dot = d1.dot(d1);
    let e_dot = d2.dot(d2);
    let f_dot = d2.dot(r);

    let (s, t_seg) = if e_dot <= 1e-10 {
        ((-d1.dot(r) / a_dot).max(0.0), 0.0)
    } else {
        let b_dot = d1.dot(d2);
        let c_dot = d1.dot(r);
        let denom = a_dot * e_dot - b_dot * b_dot;
        let mut s = if denom.abs() > 1e-10 {
            ((b_dot * f_dot - c_dot * e_dot) / denom).max(0.0)
        } else {
            0.0
        };
        let mut t = (b_dot * s + f_dot) / e_dot;
        if t < 0.0 {
            t = 0.0;
            s = (-c_dot / a_dot).max(0.0);
        } else if t > 1.0 {
            t = 1.0;
            s = ((b_dot - c_dot) / a_dot).max(0.0);
        }
        (s, t)
    };

    let p_ray = ro + d1 * s;
    let p_seg = a + d2 * t_seg;
    let dist = (p_ray - p_seg).length();
    let scale = if orthographic { 1.0 } else { s.max(1e-4) };
    let tol = line_width.max(2.0) * 1.5 * scale * pixel_world_scale;
    if dist <= tol {
        Some((s, p_seg))
    } else {
        None
    }
}
