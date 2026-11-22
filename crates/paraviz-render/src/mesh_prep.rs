//! Self-contained mesh, line, point, outline, and 3D volume extraction from `paraviz-core` datasets.

use crate::scene::ColorBy;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use paraviz_core::{
    BoundingBox, CellType, DataArray, DataSet, FieldData, ImageData, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use std::collections::{HashMap, HashSet};

/// Non-indexed triangle vertex for Surface and SurfaceWithEdges rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub scalar: f32,
    pub barycentric: [f32; 3],
}

/// Line vertex for Wireframe and Outline rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub scalar: f32,
}

/// Per-point instance for sphere-impostor point rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub scalar: f32,
}

/// Prepared 3D volume payload for `R32Float` texture upload.
#[derive(Debug, Clone)]
pub struct PreparedVolume {
    pub dims: [u32; 3],
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    pub data: Vec<f32>,
    pub scalar_range: (f64, f64),
}

/// Prepared surface triangle list with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedSurface {
    pub vertices: Vec<SurfaceVertex>,
    pub scalar_range: (f64, f64),
}

/// Prepared wireframe line list with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedLines {
    pub vertices: Vec<LineVertex>,
    pub scalar_range: (f64, f64),
}

/// Prepared point cloud instances with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedPoints {
    pub instances: Vec<PointInstance>,
    pub scalar_range: (f64, f64),
}

/// Resolves a scalar `DataArray` from `FieldData` by name (or active scalars when `name` is empty).
fn resolve_array<'a>(field: &'a FieldData, name: &str) -> Option<&'a DataArray> {
    if name.is_empty() {
        field.active_scalars_array()
    } else {
        field.get(name)
    }
}

fn extract_scalars(field: &FieldData, name: &str, component: Option<usize>) -> Option<Vec<f32>> {
    resolve_array(field, name).map(|arr| arr.to_scalar_f32(component))
}

fn compute_finite_range(values: &[f32]) -> Option<(f64, f64)> {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            let vd = v as f64;
            mn = mn.min(vd);
            mx = mx.max(vd);
        }
    }
    if mn.is_finite() && mx.is_finite() {
        Some((mn, mx))
    } else {
        None
    }
}

fn merge_range(acc: &mut Option<(f64, f64)>, next: Option<(f64, f64)>) {
    if let Some((mn, mx)) = next {
        *acc = Some(match *acc {
            Some((amn, amx)) => (amn.min(mn), amx.max(mx)),
            None => (mn, mx),
        });
    }
}

/// Generates 12 bounding box edges (24 line vertices) for any [`BoundingBox`].
pub fn prepare_outline(bounds: &BoundingBox) -> Vec<LineVertex> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let mn = bounds.min;
    let mx = bounds.max;
    let corners = [
        [mn.x, mn.y, mn.z], // 0
        [mx.x, mn.y, mn.z], // 1
        [mx.x, mx.y, mn.z], // 2
        [mn.x, mx.y, mn.z], // 3
        [mn.x, mn.y, mx.z], // 4
        [mx.x, mn.y, mx.z], // 5
        [mx.x, mx.y, mx.z], // 6
        [mn.x, mx.y, mx.z], // 7
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    let mut out = Vec::with_capacity(24);
    for (a, b) in edges {
        out.push(LineVertex { position: corners[a], scalar: 0.0 });
        out.push(LineVertex { position: corners[b], scalar: 0.0 });
    }
    out
}

/// Extracts non-indexed surface triangles (with normals and barycentric coordinates) from any [`DataSet`].
pub fn prepare_surface(dataset: &DataSet, color_by: &ColorBy) -> PreparedSurface {
    let mut vertices = Vec::new();
    let mut range = None;
    append_surface_dataset(dataset, color_by, &mut vertices, &mut range);
    PreparedSurface {
        vertices,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

fn append_surface_dataset(
    dataset: &DataSet,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => append_surface_poly(poly, color_by, out, range),
        DataSet::ImageData(img) => append_surface_image(img, color_by, out, range),
        DataSet::RectilinearGrid(rect) => append_surface_rectilinear(rect, color_by, out, range),
        DataSet::StructuredGrid(grid) => append_surface_structured(grid, color_by, out, range),
        DataSet::UnstructuredGrid(ug) => append_surface_unstructured(ug, color_by, out, range),
        DataSet::MultiBlock(mb) => {
            for (_, block) in &mb.blocks {
                append_surface_dataset(block, color_by, out, range);
            }
        }
    }
}

fn scalar_lookup<'a>(
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
) -> (Option<Vec<f32>>, bool) {
    match color_by {
        ColorBy::Solid(_) => (None, false),
        ColorBy::PointScalar { array, component } => {
            (extract_scalars(point_data, array, *component), false)
        }
        ColorBy::CellScalar { array, component } => {
            (extract_scalars(cell_data, array, *component), true)
        }
    }
}

fn emit_triangle(
    points: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    scalars: Option<&[f32]>,
    is_cell_scalar: bool,
    cell_idx: usize,
    tri: [u32; 3],
    out: &mut Vec<SurfaceVertex>,
) {
    let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
    if i0 >= points.len() || i1 >= points.len() || i2 >= points.len() {
        return;
    }
    let p0 = Vec3::from_array(points[i0]);
    let p1 = Vec3::from_array(points[i1]);
    let p2 = Vec3::from_array(points[i2]);
    let face_n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();

    let bary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let indices = [i0, i1, i2];
    for (k, &idx) in indices.iter().enumerate() {
        let normal = normals.and_then(|ns| ns.get(idx).copied()).unwrap_or(face_n);
        let scalar = match scalars {
            Some(sc) if is_cell_scalar => sc.get(cell_idx).copied().unwrap_or(0.0),
            Some(sc) => sc.get(idx).copied().unwrap_or(0.0),
            None => 0.0,
        };
        out.push(SurfaceVertex {
            position: points[idx],
            normal,
            scalar,
            barycentric: bary[k],
        });
    }
}

fn compute_smooth_normals(points: &[[f32; 3]], tris: &[([u32; 3], usize)]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; points.len()];
    for &(tri, _) in tris {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        if a >= points.len() || b >= points.len() || c >= points.len() {
            continue;
        }
        let p0 = Vec3::from_array(points[a]);
        let p1 = Vec3::from_array(points[b]);
        let p2 = Vec3::from_array(points[c]);
        let n = (p1 - p0).cross(p2 - p0);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter()
        .map(|n| n.normalize_or(Vec3::Z).to_array())
        .collect()
}

fn append_surface_poly(
    poly: &PolyData,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    let mut tris_with_cell: Vec<([u32; 3], usize)> = Vec::with_capacity(poly.polys.len() * 2);
    let poly_base = poly.verts.len() + poly.lines.len();
    for (pi, p) in poly.polys.iter().enumerate() {
        if p.len() < 3 {
            continue;
        }
        let cell_idx = poly_base + pi;
        let v0 = p[0];
        for k in 1..(p.len() - 1) {
            tris_with_cell.push(([v0, p[k], p[k + 1]], cell_idx));
        }
    }
    let strip_base = poly_base + poly.polys.len();
    for (si, s) in poly.strips.iter().enumerate() {
        if s.len() < 3 {
            continue;
        }
        let cell_idx = strip_base + si;
        for k in 0..(s.len() - 2) {
            let tri = if k % 2 == 0 {
                [s[k], s[k + 1], s[k + 2]]
            } else {
                [s[k + 1], s[k], s[k + 2]]
            };
            tris_with_cell.push((tri, cell_idx));
        }
    }

    if tris_with_cell.is_empty() {
        return;
    }

    let provided_normals = poly
        .point_data
        .active_normals_array()
        .filter(|arr| arr.num_tuples() == poly.points.len())
        .map(|arr| (0..arr.num_tuples()).map(|i| arr.get_vec3(i)).collect::<Vec<_>>());
    let computed_normals;
    let normals_ref = match &provided_normals {
        Some(ns) => ns.as_slice(),
        None => {
            computed_normals = compute_smooth_normals(&poly.points, &tris_with_cell);
            &computed_normals
        }
    };

    let (scalars, is_cell) = scalar_lookup(&poly.point_data, &poly.cell_data, color_by);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    out.reserve(tris_with_cell.len() * 3);
    for (tri, cell_idx) in tris_with_cell {
        emit_triangle(
            &poly.points,
            Some(normals_ref),
            scalars.as_deref(),
            is_cell,
            cell_idx,
            tri,
            out,
        );
    }
}

fn emit_quad(
    p: [[f32; 3]; 4],
    s: [f32; 4],
    normal: [f32; 3],
    out: &mut Vec<SurfaceVertex>,
) {
    let tris = [(0, 1, 2), (0, 2, 3)];
    let bary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (a, b, c) in tris {
        out.push(SurfaceVertex { position: p[a], normal, scalar: s[a], barycentric: bary[0] });
        out.push(SurfaceVertex { position: p[b], normal, scalar: s[b], barycentric: bary[1] });
        out.push(SurfaceVertex { position: p[c], normal, scalar: s[c], barycentric: bary[2] });
    }
}

fn append_structured_boundary<FCoord, FPointIdx>(
    dims: [usize; 3],
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    coord: FCoord,
    point_idx: FPointIdx,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) where
    FCoord: Fn(usize, usize, usize) -> [f32; 3],
    FPointIdx: Fn(usize, usize, usize) -> usize,
{
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    let (scalars, is_cell) = scalar_lookup(point_data, cell_data, color_by);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    let cx = nx.saturating_sub(1).max(1);
    let cy = ny.saturating_sub(1).max(1);

    let pt_s = |i: usize, j: usize, k: usize, ci: usize, cj: usize, ck: usize| -> f32 {
        match &scalars {
            Some(sc) if is_cell => {
                let cidx = ci.min(cx - 1) + cx * (cj.min(cy - 1) + cy * ck);
                sc.get(cidx).copied().unwrap_or(0.0)
            }
            Some(sc) => sc.get(point_idx(i, j, k)).copied().unwrap_or(0.0),
            None => 0.0,
        }
    };

    let emit_face_quad = |q: [(usize, usize, usize); 4], ci: usize, cj: usize, ck: usize, out: &mut Vec<SurfaceVertex>| {
        let pts = [
            coord(q[0].0, q[0].1, q[0].2),
            coord(q[1].0, q[1].1, q[1].2),
            coord(q[2].0, q[2].1, q[2].2),
            coord(q[3].0, q[3].1, q[3].2),
        ];
        let p0 = Vec3::from_array(pts[0]);
        let p1 = Vec3::from_array(pts[1]);
        let p2 = Vec3::from_array(pts[2]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();
        let sv = [
            pt_s(q[0].0, q[0].1, q[0].2, ci, cj, ck),
            pt_s(q[1].0, q[1].1, q[1].2, ci, cj, ck),
            pt_s(q[2].0, q[2].1, q[2].2, ci, cj, ck),
            pt_s(q[3].0, q[3].1, q[3].2, ci, cj, ck),
        ];
        emit_quad(pts, sv, n, out);
    };

    // XY faces (k = 0 and k = nz - 1)
    if nx > 1 && ny > 1 {
        let k_planes: Vec<(usize, usize, bool)> = if nz > 1 {
            vec![(0, 0, true), (nz - 1, nz - 2, false)]
        } else {
            vec![(0, 0, false)]
        };
        for (k, ck, flip) in k_planes {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    let q = if flip {
                        [(i, j, k), (i, j + 1, k), (i + 1, j + 1, k), (i + 1, j, k)]
                    } else {
                        [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k)]
                    };
                    emit_face_quad(q, i, j, ck, out);
                }
            }
        }
    }
    // XZ faces (j = 0 and j = ny - 1)
    if nx > 1 && nz > 1 {
        let j_planes: Vec<(usize, usize, bool)> = if ny > 1 {
            vec![(0, 0, false), (ny - 1, ny - 2, true)]
        } else {
            vec![(0, 0, false)]
        };
        for (j, cj, flip) in j_planes {
            for k in 0..(nz - 1) {
                for i in 0..(nx - 1) {
                    let q = if flip {
                        [(i, j, k), (i + 1, j, k), (i + 1, j, k + 1), (i, j, k + 1)]
                    } else {
                        [(i, j, k), (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j, k)]
                    };
                    emit_face_quad(q, i, cj, k, out);
                }
            }
        }
    }
    // YZ faces (i = 0 and i = nx - 1)
    if ny > 1 && nz > 1 {
        let i_planes: Vec<(usize, usize, bool)> = if nx > 1 {
            vec![(0, 0, true), (nx - 1, nx - 2, false)]
        } else {
            vec![(0, 0, false)]
        };
        for (i, ci, flip) in i_planes {
            for k in 0..(nz - 1) {
                for j in 0..(ny - 1) {
                    let q = if flip {
                        [(i, j, k), (i, j, k + 1), (i, j + 1, k + 1), (i, j + 1, k)]
                    } else {
                        [(i, j, k), (i, j + 1, k), (i, j + 1, k + 1), (i, j, k + 1)]
                    };
                    emit_face_quad(q, ci, j, k, out);
                }
            }
        }
    }
}

fn append_surface_image(
    img: &ImageData,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    append_structured_boundary(
        img.dimensions,
        &img.point_data,
        &img.cell_data,
        color_by,
        |i, j, k| img.point_coords(i, j, k),
        |i, j, k| img.point_index(i, j, k),
        out,
        range,
    );
}

fn append_surface_rectilinear(
    rect: &RectilinearGrid,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    append_structured_boundary(
        rect.dimensions(),
        &rect.point_data,
        &rect.cell_data,
        color_by,
        |i, j, k| rect.point_coords(i, j, k),
        |i, j, k| rect.point_index(i, j, k),
        out,
        range,
    );
}

fn append_surface_structured(
    grid: &StructuredGrid,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    if grid.points.len() < grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2] {
        return;
    }
    append_structured_boundary(
        grid.dimensions,
        &grid.point_data,
        &grid.cell_data,
        color_by,
        |i, j, k| grid.point_coords(i, j, k),
        |i, j, k| grid.point_index(i, j, k),
        out,
        range,
    );
}

fn append_surface_unstructured(
    ug: &UnstructuredGrid,
    color_by: &ColorBy,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    let (scalars, is_cell) = scalar_lookup(&ug.point_data, &ug.cell_data, color_by);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    // Track 3D cell faces by sorted vertex tuple so interior shared faces cancel out.
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, usize)> = HashMap::new();
    let mut face_counts: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut surface_tris: Vec<([u32; 3], usize)> = Vec::new();

    for cell_idx in 0..ug.num_cells() {
        let ct = ug.cell_types[cell_idx];
        let pts = ug.cell_points(cell_idx);
        match ct.dimension() {
            2 => {
                if ct == CellType::TriangleStrip {
                    for k in 0..pts.len().saturating_sub(2) {
                        let tri = if k % 2 == 0 {
                            [pts[k], pts[k + 1], pts[k + 2]]
                        } else {
                            [pts[k + 1], pts[k], pts[k + 2]]
                        };
                        surface_tris.push((tri, cell_idx));
                    }
                } else {
                    for local_face in ct.faces(pts.len()) {
                        if local_face.len() < 3 {
                            continue;
                        }
                        let v0 = pts[local_face[0]];
                        for k in 1..(local_face.len() - 1) {
                            surface_tris.push(([v0, pts[local_face[k]], pts[local_face[k + 1]]], cell_idx));
                        }
                    }
                }
            }
            3 => {
                for local_face in ct.faces(pts.len()) {
                    let global: Vec<u32> = local_face.iter().map(|&li| pts[li]).collect();
                    let mut key = global.clone();
                    key.sort_unstable();
                    *face_counts.entry(key.clone()).or_insert(0) += 1;
                    face_map.entry(key).or_insert((global, cell_idx));
                }
            }
            _ => {}
        }
    }

    for (key, count) in face_counts {
        if count == 1 {
            if let Some((face, cell_idx)) = face_map.remove(&key) {
                if face.len() >= 3 {
                    let v0 = face[0];
                    for k in 1..(face.len() - 1) {
                        surface_tris.push(([v0, face[k], face[k + 1]], cell_idx));
                    }
                }
            }
        }
    }

    for (tri, cell_idx) in surface_tris {
        emit_triangle(
            &ug.points,
            None,
            scalars.as_deref(),
            is_cell,
            cell_idx,
            tri,
            out,
        );
    }
}

/// Extracts wireframe line segments from any [`DataSet`].
pub fn prepare_wireframe(dataset: &DataSet, color_by: &ColorBy) -> PreparedLines {
    let mut vertices = Vec::new();
    let mut range = None;
    append_wireframe_dataset(dataset, color_by, &mut vertices, &mut range);
    PreparedLines {
        vertices,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

fn append_wireframe_dataset(
    dataset: &DataSet,
    color_by: &ColorBy,
    out: &mut Vec<LineVertex>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            let (scalars, is_cell) = scalar_lookup(&poly.point_data, &poly.cell_data, color_by);
            if let Some(sc) = &scalars {
                merge_range(range, compute_finite_range(sc));
            }
            let mut seen = HashSet::new();
            let mut push_edge = |a: u32, b: u32, cell_idx: usize| {
                let (ia, ib) = (a as usize, b as usize);
                if ia >= poly.points.len() || ib >= poly.points.len() {
                    return;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    let sa = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ia).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let sb = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ib).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    out.push(LineVertex { position: poly.points[ia], scalar: sa });
                    out.push(LineVertex { position: poly.points[ib], scalar: sb });
                }
            };

            let line_base = poly.verts.len();
            for (li, line) in poly.lines.iter().enumerate() {
                for w in line.windows(2) {
                    push_edge(w[0], w[1], line_base + li);
                }
            }
            let poly_base = line_base + poly.lines.len();
            for (pi, p) in poly.polys.iter().enumerate() {
                if p.len() >= 2 {
                    for k in 0..p.len() {
                        push_edge(p[k], p[(k + 1) % p.len()], poly_base + pi);
                    }
                }
            }
            let strip_base = poly_base + poly.polys.len();
            for (si, s) in poly.strips.iter().enumerate() {
                if s.len() >= 3 {
                    for k in 0..(s.len() - 2) {
                        push_edge(s[k], s[k + 1], strip_base + si);
                        push_edge(s[k + 1], s[k + 2], strip_base + si);
                        push_edge(s[k], s[k + 2], strip_base + si);
                    }
                }
            }
        }
        DataSet::UnstructuredGrid(ug) => {
            let (scalars, is_cell) = scalar_lookup(&ug.point_data, &ug.cell_data, color_by);
            if let Some(sc) = &scalars {
                merge_range(range, compute_finite_range(sc));
            }
            let mut seen = HashSet::new();
            let mut push_edge = |a: u32, b: u32, cell_idx: usize| {
                let (ia, ib) = (a as usize, b as usize);
                if ia >= ug.points.len() || ib >= ug.points.len() {
                    return;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    let sa = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ia).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let sb = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ib).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    out.push(LineVertex { position: ug.points[ia], scalar: sa });
                    out.push(LineVertex { position: ug.points[ib], scalar: sb });
                }
            };
            for cell_idx in 0..ug.num_cells() {
                let ct = ug.cell_types[cell_idx];
                let pts = ug.cell_points(cell_idx);
                if ct == CellType::Line || ct == CellType::PolyLine {
                    for w in pts.windows(2) {
                        push_edge(w[0], w[1], cell_idx);
                    }
                } else if ct.dimension() >= 2 {
                    for f in ct.faces(pts.len()) {
                        for k in 0..f.len() {
                            push_edge(pts[f[k]], pts[f[(k + 1) % f.len()]], cell_idx);
                        }
                    }
                }
            }
        }
        DataSet::ImageData(_) | DataSet::RectilinearGrid(_) | DataSet::StructuredGrid(_) => {
            let surf = prepare_surface(dataset, color_by);
            merge_range(range, Some(surf.scalar_range));
            for tri in surf.vertices.chunks_exact(3) {
                for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                    out.push(LineVertex { position: tri[a].position, scalar: tri[a].scalar });
                    out.push(LineVertex { position: tri[b].position, scalar: tri[b].scalar });
                }
            }
        }
        DataSet::MultiBlock(mb) => {
            for (_, block) in &mb.blocks {
                append_wireframe_dataset(block, color_by, out, range);
            }
        }
    }
}

/// Extracts point cloud instances from any [`DataSet`].
pub fn prepare_points(dataset: &DataSet, color_by: &ColorBy) -> PreparedPoints {
    let mut instances = Vec::new();
    let mut range = None;
    append_points_dataset(dataset, color_by, &mut instances, &mut range);
    PreparedPoints {
        instances,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

fn append_points_slice(
    points: &[[f32; 3]],
    point_data: &FieldData,
    color_by: &ColorBy,
    out: &mut Vec<PointInstance>,
    range: &mut Option<(f64, f64)>,
) {
    let scalars = match color_by {
        ColorBy::PointScalar { array, component } => extract_scalars(point_data, array, *component),
        _ => None,
    };
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    out.reserve(points.len());
    for (i, &pos) in points.iter().enumerate() {
        let scalar = scalars.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(0.0);
        out.push(PointInstance { position: pos, scalar });
    }
}

fn append_points_dataset(
    dataset: &DataSet,
    color_by: &ColorBy,
    out: &mut Vec<PointInstance>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            append_points_slice(&poly.points, &poly.point_data, color_by, out, range);
        }
        DataSet::ImageData(img) => {
            let pts = img.all_points();
            append_points_slice(&pts, &img.point_data, color_by, out, range);
        }
        DataSet::RectilinearGrid(rect) => {
            let pts = rect.all_points();
            append_points_slice(&pts, &rect.point_data, color_by, out, range);
        }
        DataSet::StructuredGrid(grid) => {
            append_points_slice(&grid.points, &grid.point_data, color_by, out, range);
        }
        DataSet::UnstructuredGrid(ug) => {
            append_points_slice(&ug.points, &ug.point_data, color_by, out, range);
        }
        DataSet::MultiBlock(mb) => {
            for (_, block) in &mb.blocks {
                append_points_dataset(block, color_by, out, range);
            }
        }
    }
}

/// Extracts a 3D `R32Float` volume payload from an [`ImageData`] (or first ImageData block in a MultiBlock).
pub fn prepare_volume(dataset: &DataSet, color_by: &ColorBy) -> Option<PreparedVolume> {
    let img = find_image_data(dataset)?;
    let [nx, ny, nz] = img.dimensions;
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }

    let arr = match color_by {
        ColorBy::PointScalar { array, .. } => resolve_array(&img.point_data, array),
        ColorBy::CellScalar { array, .. } => resolve_array(&img.cell_data, array)
            .or_else(|| img.point_data.active_scalars_array()),
        ColorBy::Solid(_) => img.point_data.active_scalars_array(),
    }?;

    let component = match color_by {
        ColorBy::PointScalar { component, .. } | ColorBy::CellScalar { component, .. } => *component,
        ColorBy::Solid(_) => None,
    };

    let expected_pts = nx * ny * nz;
    let mut data = arr.to_scalar_f32(component);
    if data.len() < expected_pts {
        data.resize(expected_pts, 0.0);
    } else if data.len() > expected_pts {
        data.truncate(expected_pts);
    }

    let scalar_range = compute_finite_range(&data).unwrap_or((0.0, 1.0));
    let bb = img.bounds();
    let mut box_min = bb.min.to_array();
    let mut box_max = bb.max.to_array();
    for axis in 0..3 {
        if (box_max[axis] - box_min[axis]).abs() < 1e-5 {
            box_min[axis] -= 0.5;
            box_max[axis] += 0.5;
        }
    }

    Some(PreparedVolume {
        dims: [nx as u32, ny as u32, nz as u32],
        box_min,
        box_max,
        data,
        scalar_range,
    })
}

fn find_image_data(dataset: &DataSet) -> Option<&ImageData> {
    match dataset {
        DataSet::ImageData(img) => Some(img),
        DataSet::MultiBlock(mb) => mb.blocks.iter().find_map(|(_, d)| find_image_data(d)),
        _ => None,
    }
}

/// Vertex for the 3D orientation axes gizmo.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Builds a 3D RGB XYZ arrow triad centered at the origin (unit radius ~1.0).
pub fn build_axes_gizmo_mesh() -> Vec<GizmoVertex> {
    let mut verts = Vec::new();
    let axes = [
        (Vec3::X, Vec3::Y, Vec3::Z, [0.92, 0.22, 0.22]),
        (Vec3::Y, Vec3::Z, Vec3::X, [0.24, 0.85, 0.32]),
        (Vec3::Z, Vec3::X, Vec3::Y, [0.24, 0.52, 0.96]),
    ];
    let segments = 16usize;
    let shaft_r = 0.06;
    let shaft_len = 0.72;
    let head_r = 0.14;
    let tip_len = 1.0;

    for (dir, u, v, color) in axes {
        for s in 0..segments {
            let a0 = (s as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((s + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let r0 = u * a0.cos() + v * a0.sin();
            let r1 = u * a1.cos() + v * a1.sin();
            let n0 = r0.to_array();
            let n1 = r1.to_array();

            // Cylinder shaft
            let b0 = (r0 * shaft_r).to_array();
            let b1 = (r1 * shaft_r).to_array();
            let t0 = (dir * shaft_len + r0 * shaft_r).to_array();
            let t1 = (dir * shaft_len + r1 * shaft_r).to_array();

            verts.push(GizmoVertex { position: b0, normal: n0, color });
            verts.push(GizmoVertex { position: b1, normal: n1, color });
            verts.push(GizmoVertex { position: t1, normal: n1, color });

            verts.push(GizmoVertex { position: b0, normal: n0, color });
            verts.push(GizmoVertex { position: t1, normal: n1, color });
            verts.push(GizmoVertex { position: t0, normal: n0, color });

            // Cone head
            let h0 = (dir * shaft_len + r0 * head_r).to_array();
            let h1 = (dir * shaft_len + r1 * head_r).to_array();
            let tip = (dir * tip_len).to_array();
            let cone_n0 = (r0 * 0.9 + dir * 0.43).normalize().to_array();
            let cone_n1 = (r1 * 0.9 + dir * 0.43).normalize().to_array();
            let tip_n = ((r0 + r1) * 0.5 + dir * 0.45).normalize().to_array();

            verts.push(GizmoVertex { position: h0, normal: cone_n0, color });
            verts.push(GizmoVertex { position: h1, normal: cone_n1, color });
            verts.push(GizmoVertex { position: tip, normal: tip_n, color });
        }
    }

    // Small origin cube/octahedron in neutral gray
    let o_col = [0.75, 0.75, 0.78];
    let r = 0.11;
    let pts = [
        Vec3::new(r, 0.0, 0.0),
        Vec3::new(-r, 0.0, 0.0),
        Vec3::new(0.0, r, 0.0),
        Vec3::new(0.0, -r, 0.0),
        Vec3::new(0.0, 0.0, r),
        Vec3::new(0.0, 0.0, -r),
    ];
    let oct_faces = [
        (0, 2, 4), (4, 2, 1), (1, 2, 5), (5, 2, 0),
        (4, 3, 0), (1, 3, 4), (5, 3, 1), (0, 3, 5),
    ];
    for (a, b, c) in oct_faces {
        let n = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize().to_array();
        verts.push(GizmoVertex { position: pts[a].to_array(), normal: n, color: o_col });
        verts.push(GizmoVertex { position: pts[b].to_array(), normal: n, color: o_col });
        verts.push(GizmoVertex { position: pts[c].to_array(), normal: n, color: o_col });
    }

    verts
}
