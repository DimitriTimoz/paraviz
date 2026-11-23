//! 3D CPU raycast picking across visible [`SceneActor`]s using AABB pre-culling,
//! Möller–Trumbore ray-triangle intersection, and ray-point / ray-segment distance tests.

use crate::mesh_prep;
use crate::scene::{ActorStyle, ColorBy, RenderScene, Representation, SceneActor};
use glam::Vec3;
use paraviz_core::{BoundingBox, DataArray, DataSet, FieldData, PolyData};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, Weak};

const PARALLEL_TRI_THRESHOLD: usize = 2000;
const PICK_CHUNK_TRIS: usize = 64;
const MAX_PICK_CACHE_ENTRIES: usize = 32;

struct PickChunk {
    bounds: BoundingBox,
    tri_range: Range<usize>,
}

struct PickTriMesh {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    scalars: Option<Vec<f32>>,
    chunks: Vec<PickChunk>,
}

struct CachedPickEntry {
    actor_id: u64,
    dataset_weak: Weak<DataSet>,
    block_path: Option<String>,
    color_by: ColorBy,
    mesh: Arc<PickTriMesh>,
}

static PICK_SURFACE_CACHE: OnceLock<Mutex<Vec<CachedPickEntry>>> = OnceLock::new();

fn build_pick_trimesh(dataset: &DataSet, color_by: &ColorBy) -> PickTriMesh {
    let prep = mesh_prep::prepare_surface_indexed(dataset, color_by);
    let positions: Vec<[f32; 3]> = prep.geom_vertices.iter().map(|v| v.position).collect();
    let tris: Vec<[u32; 3]> = prep.indices.as_chunks::<3>().0.to_vec();
    let scalars = if matches!(color_by, ColorBy::Solid(_)) {
        None
    } else {
        Some(prep.scalars)
    };
    let mut chunks = Vec::with_capacity(tris.len().div_ceil(PICK_CHUNK_TRIS));
    let mut start = 0;
    while start < tris.len() {
        let end = (start + PICK_CHUNK_TRIS).min(tris.len());
        let mut bb = BoundingBox::EMPTY;
        for tri in &tris[start..end] {
            for &vi in tri {
                if let Some(&p) = positions.get(vi as usize) {
                    bb.expand_point(Vec3::from_array(p));
                }
            }
        }
        chunks.push(PickChunk {
            bounds: bb,
            tri_range: start..end,
        });
        start = end;
    }
    PickTriMesh {
        positions,
        tris,
        scalars,
        chunks,
    }
}

fn get_or_build_pick_trimesh(
    actor_id: u64,
    root_dataset: &Arc<DataSet>,
    leaf_dataset: &DataSet,
    block_path: Option<&str>,
    color_by: &ColorBy,
) -> Arc<PickTriMesh> {
    let cache = PICK_SURFACE_CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut entries) = cache.lock() {
        entries.retain(|e| e.dataset_weak.strong_count() > 0);
        if let Some(found) = entries.iter().find(|e| {
            e.actor_id == actor_id
                && e.block_path.as_deref() == block_path
                && e.color_by == *color_by
                && e.dataset_weak
                    .upgrade()
                    .is_some_and(|d| Arc::ptr_eq(&d, root_dataset))
        }) {
            return Arc::clone(&found.mesh);
        }
        let mesh = Arc::new(build_pick_trimesh(leaf_dataset, color_by));
        if entries.len() >= MAX_PICK_CACHE_ENTRIES {
            entries.remove(0);
        }
        entries.push(CachedPickEntry {
            actor_id,
            dataset_weak: Arc::downgrade(root_dataset),
            block_path: block_path.map(str::to_string),
            color_by: color_by.clone(),
            mesh: Arc::clone(&mesh),
        });
        mesh
    } else {
        Arc::new(build_pick_trimesh(leaf_dataset, color_by))
    }
}

/// Result of a 3D viewport pick query from [`pick_scene`].
#[derive(Debug, Clone, PartialEq)]
pub struct PickResult {
    pub actor_id: u64,
    pub actor_name: String,
    /// Slash-joined MultiBlock child path when picking into a `MultiBlockDataSet`, or `None`.
    pub block_path: Option<String>,
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

fn aabb_pad_for_style(style: &ActorStyle, bounds: &BoundingBox, ray_origin: Vec3, pixel_world_scale: f32, orthographic: bool) -> f32 {
    match style.representation {
        Representation::Points | Representation::PointGaussian => {
            let approx_dist = (bounds.center() - ray_origin).length().max(1.0);
            let scale = if orthographic { 1.0 } else { approx_dist };
            style.point_size.max(4.0) * scale * pixel_world_scale + bounds.diagonal() * 0.02
        }
        Representation::Wireframe | Representation::Outline => {
            let approx_dist = (bounds.center() - ray_origin).length().max(1.0);
            let scale = if orthographic { 1.0 } else { approx_dist };
            style.line_width.max(2.0) * 2.0 * scale * pixel_world_scale + bounds.diagonal() * 0.02
        }
        Representation::Surface | Representation::SurfaceWithEdges | Representation::Volume => 1e-4,
    }
}

fn pick_actor(
    actor: &SceneActor,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let bounds = actor.style.dataset_bounds(&actor.dataset);
    if !bounds.is_valid() {
        return None;
    }

    let aabb_pad = aabb_pad_for_style(&actor.style, &bounds, ray_origin, pixel_world_scale, orthographic);
    let padded = BoundingBox::new(
        bounds.min - Vec3::splat(aabb_pad),
        bounds.max + Vec3::splat(aabb_pad),
    );
    let (t_enter, t_exit) = ray_aabb_intersect(ray_origin, ray_dir, &padded)?;
    if t_exit < 0.0 || t_enter > max_t {
        return None;
    }

    if actor.style.representation == Representation::Outline {
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
        return best_pos.map(|p| PickResult {
            actor_id: actor.id,
            actor_name: actor.name.clone(),
            block_path: None,
            world_pos: p.to_array(),
            distance: best_t,
            point_index: None,
            cell_index: None,
            scalar_value: None,
        });
    }

    pick_dataset(
        actor,
        &actor.dataset,
        None,
        ray_origin,
        ray_dir,
        pixel_world_scale,
        orthographic,
        max_t,
    )
}

#[allow(clippy::too_many_arguments)]
fn pick_dataset(
    actor: &SceneActor,
    dataset: &DataSet,
    block_path: Option<&str>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    if let DataSet::MultiBlock(mb) = dataset {
        let mut best_t = max_t;
        let mut best: Option<PickResult> = None;
        for (name, child) in &mb.blocks {
            let child_path = match block_path {
                Some(p) if !p.is_empty() => format!("{p}/{name}"),
                _ => name.clone(),
            };
            if !actor.style.is_block_visible(&child_path, name) {
                continue;
            }
            if let Some(hit) = pick_dataset(
                actor,
                child,
                Some(&child_path),
                ray_origin,
                ray_dir,
                pixel_world_scale,
                orthographic,
                best_t,
            ) && hit.distance < best_t
            {
                best_t = hit.distance;
                best = Some(hit);
            }
        }
        return best;
    }

    let bounds = dataset.bounds();
    if !bounds.is_valid() {
        return None;
    }
    let aabb_pad = aabb_pad_for_style(&actor.style, &bounds, ray_origin, pixel_world_scale, orthographic);
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
            pick_surface(actor, dataset, block_path, ray_origin, ray_dir, max_t)
        }
        Representation::Points | Representation::PointGaussian => {
            pick_points(actor, dataset, block_path, ray_origin, ray_dir, pixel_world_scale, orthographic, max_t)
        }
        Representation::Wireframe => {
            pick_wireframe(actor, dataset, block_path, ray_origin, ray_dir, pixel_world_scale, orthographic, max_t)
        }
        Representation::Outline => None,
        Representation::Volume => {
            let DataSet::ImageData(img) = dataset else {
                return None;
            };
            let t = t_enter.max(0.0);
            if t >= max_t {
                return None;
            }
            let world_pos = ray_origin + ray_dir * t;
            let scalar_value = sample_volume_scalar(img, &actor.style, world_pos);
            Some(PickResult {
                actor_id: actor.id,
                actor_name: actor.name.clone(),
                block_path: block_path.map(str::to_string),
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
    dataset: &DataSet,
    block_path: Option<&str>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_t: f32,
) -> Option<PickResult> {
    if let DataSet::PolyData(poly) = dataset {
        return pick_polydata_surface(actor, poly, block_path, ray_origin, ray_dir, max_t);
    }

    // Cached indexed boundary surface with chunk AABB pre-filtering + Rayon parallel ray intersection.
    let mesh = get_or_build_pick_trimesh(
        actor.id,
        &actor.dataset,
        dataset,
        block_path,
        &actor.style.color_by,
    );
    if mesh.tris.is_empty() {
        return None;
    }

    let test_chunk = |chunk: &PickChunk, limit_t: f32| -> Option<(f32, usize, Vec3, Option<f64>)> {
        let (t_enter, _) = ray_aabb_intersect(ray_origin, ray_dir, &chunk.bounds)?;
        if t_enter >= limit_t {
            return None;
        }
        let mut best_t = limit_t;
        let mut best: Option<(f32, usize, Vec3, Option<f64>)> = None;
        for (local_idx, tri) in mesh.tris[chunk.tri_range.clone()].iter().enumerate() {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            if i0 >= mesh.positions.len() || i1 >= mesh.positions.len() || i2 >= mesh.positions.len() {
                continue;
            }
            let p0 = Vec3::from_array(mesh.positions[i0]);
            let p1 = Vec3::from_array(mesh.positions[i1]);
            let p2 = Vec3::from_array(mesh.positions[i2]);
            if let Some((t, u, v)) = ray_triangle_intersect(ray_origin, ray_dir, p0, p1, p2)
                && t < best_t
            {
                best_t = t;
                let w = 1.0 - u - v;
                let pos = p0 * w + p1 * u + p2 * v;
                let scalar = mesh
                    .scalars
                    .as_ref()
                    .map(|sc| (sc[i0] * w + sc[i1] * u + sc[i2] * v) as f64);
                best = Some((t, chunk.tri_range.start + local_idx, pos, scalar));
            }
        }
        best
    };

    let best_hit = if mesh.tris.len() > PARALLEL_TRI_THRESHOLD {
        mesh.chunks
            .par_iter()
            .filter_map(|chunk| test_chunk(chunk, max_t))
            .reduce_with(|a, b| {
                if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
                    a
                } else {
                    b
                }
            })
    } else {
        let mut best_t = max_t;
        let mut best: Option<(f32, usize, Vec3, Option<f64>)> = None;
        for chunk in &mesh.chunks {
            if let Some(hit) = test_chunk(chunk, best_t) {
                best_t = hit.0;
                best = Some(hit);
            }
        }
        best
    };

    best_hit.map(|(best_t, _, pos, best_scalar)| PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        block_path: block_path.map(str::to_string),
        world_pos: pos.to_array(),
        distance: best_t,
        point_index: None,
        cell_index: None,
        scalar_value: best_scalar,
    })
}

struct PolyTriHit {
    t: f32,
    pos: Vec3,
    point_index: usize,
    cell_index: usize,
    bary: [f32; 3],
    tri_idx: [usize; 3],
}

fn pick_polydata_surface(
    actor: &SceneActor,
    poly: &PolyData,
    block_path: Option<&str>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_t: f32,
) -> Option<PickResult> {
    let poly_base = poly.verts.len() + poly.lines.len();
    let strip_base = poly_base + poly.polys.len();
    let total_est_tris = poly.polys.len()
        + poly
            .strips
            .iter()
            .map(|s| s.len().saturating_sub(2))
            .sum::<usize>();

    let test_poly = |pi: usize, p: &Vec<u32>, limit_t: f32| -> Option<PolyTriHit> {
        if p.len() < 3 {
            return None;
        }
        let i0 = p[0] as usize;
        if i0 >= poly.points.len() {
            return None;
        }
        let cell_idx = poly_base + pi;
        let p0 = Vec3::from_array(poly.points[i0]);
        let mut best_t = limit_t;
        let mut best: Option<PolyTriHit> = None;
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
                best = Some(PolyTriHit {
                    t,
                    pos: hit_pos,
                    point_index: nearest_pt,
                    cell_index: cell_idx,
                    bary: [w, u, v],
                    tri_idx: [i0, i1, i2],
                });
            }
        }
        best
    };

    let test_strip = |si: usize, s: &Vec<u32>, limit_t: f32| -> Option<PolyTriHit> {
        if s.len() < 3 {
            return None;
        }
        let cell_idx = strip_base + si;
        let mut best_t = limit_t;
        let mut best: Option<PolyTriHit> = None;
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
                best = Some(PolyTriHit {
                    t,
                    pos: hit_pos,
                    point_index: nearest_pt,
                    cell_index: cell_idx,
                    bary: [w, u, v],
                    tri_idx: [i0, i1, i2],
                });
            }
        }
        best
    };

    let pick_closer = |a: PolyTriHit, b: PolyTriHit| -> PolyTriHit {
        if a.t < b.t || (a.t == b.t && a.cell_index < b.cell_index) {
            a
        } else {
            b
        }
    };

    let best_hit = if total_est_tris > PARALLEL_TRI_THRESHOLD {
        let poly_best = poly
            .polys
            .par_iter()
            .enumerate()
            .filter_map(|(pi, p)| test_poly(pi, p, max_t))
            .reduce_with(&pick_closer);
        let strip_best = poly
            .strips
            .par_iter()
            .enumerate()
            .filter_map(|(si, s)| test_strip(si, s, max_t))
            .reduce_with(&pick_closer);
        match (poly_best, strip_best) {
            (Some(a), Some(b)) => Some(pick_closer(a, b)),
            (a, b) => a.or(b),
        }
    } else {
        let mut best_t = max_t;
        let mut best: Option<PolyTriHit> = None;
        for (pi, p) in poly.polys.iter().enumerate() {
            if let Some(hit) = test_poly(pi, p, best_t) {
                best_t = hit.t;
                best = Some(hit);
            }
        }
        for (si, s) in poly.strips.iter().enumerate() {
            if let Some(hit) = test_strip(si, s, best_t) {
                best_t = hit.t;
                best = Some(hit);
            }
        }
        best
    };

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
        block_path: block_path.map(str::to_string),
        world_pos: hit.pos.to_array(),
        distance: hit.t,
        point_index: Some(hit.point_index),
        cell_index: Some(hit.cell_index),
        scalar_value,
    })
}

#[allow(clippy::too_many_arguments)]
fn pick_points(
    actor: &SceneActor,
    dataset: &DataSet,
    block_path: Option<&str>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let prep = mesh_prep::prepare_points(dataset, &actor.style.color_by);
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
        dataset
            .point_data()
            .and_then(|pd| pd.active_scalars_array())
            .filter(|a| idx < a.num_tuples())
            .map(|a| a.get_magnitude(idx))
    };

    Some(PickResult {
        actor_id: actor.id,
        actor_name: actor.name.clone(),
        block_path: block_path.map(str::to_string),
        world_pos: pos.to_array(),
        distance: best_t,
        point_index: Some(idx),
        cell_index: None,
        scalar_value,
    })
}

#[allow(clippy::too_many_arguments)]
fn pick_wireframe(
    actor: &SceneActor,
    dataset: &DataSet,
    block_path: Option<&str>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    pixel_world_scale: f32,
    orthographic: bool,
    max_t: f32,
) -> Option<PickResult> {
    let prep = mesh_prep::prepare_wireframe(dataset, &actor.style.color_by);
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
        block_path: block_path.map(str::to_string),
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

fn sample_volume_scalar(img: &paraviz_core::ImageData, style: &ActorStyle, world_pos: Vec3) -> Option<f64> {
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
