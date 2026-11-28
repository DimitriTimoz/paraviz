//! Arbitrary plane slicing filter (`SliceFilter`, equivalent to `vtkCutter`).

use crate::contour::{contour_tetrahedron, contour_tetrahedron_pts, interp_edge};
use crate::util::{
    canonical_edge_key, compact_polydata_points, deduplicate_edge_triangles,
    interpolate_field_data_edges, merge_polydata, EdgeTri, FxHashMap, GridPoints,
};
use glam::Vec3;
use paraviz_core::{
    CellType, DataSet, FieldData, Filter, PipelineError, PolyData, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Cuts any dataset with an arbitrary plane `(origin, normal)`.
///
/// When `crinkle == false`, interpolates all `point_data` arrays smoothly onto the cut surface.
/// When `crinkle == true`, extracts the faces of whole cells intersecting the plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliceFilter {
    pub origin: [f32; 3],
    pub normal: [f32; 3],
    pub crinkle: bool,
}

impl Default for SliceFilter {
    fn default() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            crinkle: false,
        }
    }
}

impl SliceFilter {
    pub fn new(origin: [f32; 3], normal: [f32; 3]) -> Self {
        Self {
            origin,
            normal,
            crinkle: false,
        }
    }
}

impl Filter for SliceFilter {
    fn name(&self) -> &str {
        "SliceFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let normal = Vec3::from_array(self.normal).normalize_or(Vec3::Z);
        let origin = Vec3::from_array(self.origin);

        let pd = match input {
            DataSet::ImageData(img) => slice_structured(
                img.dimensions,
                GridPoints::Image(img),
                &img.point_data,
                &img.cell_data,
                CellType::Voxel,
                origin,
                normal,
                self.crinkle,
            ),
            DataSet::RectilinearGrid(rg) => slice_structured(
                rg.dimensions(),
                GridPoints::Rectilinear(rg),
                &rg.point_data,
                &rg.cell_data,
                CellType::Voxel,
                origin,
                normal,
                self.crinkle,
            ),
            DataSet::StructuredGrid(sg) => slice_structured(
                sg.dimensions,
                GridPoints::from_structured(&sg.points, sg.dimensions),
                &sg.point_data,
                &sg.cell_data,
                CellType::Hexahedron,
                origin,
                normal,
                self.crinkle,
            ),
            DataSet::UnstructuredGrid(ug) => slice_unstructured(ug, origin, normal, self.crinkle),
            DataSet::PolyData(pd) => slice_polydata(pd, origin, normal, self.crinkle),
            DataSet::MultiBlock(mb) => {
                let parts: Vec<PolyData> = mb
                    .blocks
                    .par_iter()
                    .map(|(_, block)| match self.apply(block)? {
                        DataSet::PolyData(pd) => Ok(pd),
                        _ => unreachable!(),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                merge_polydata(parts)
            }
        };

        Ok(DataSet::PolyData(pd))
    }
}

#[inline]
fn signed_dists(points: &[[f32; 3]], origin: Vec3, normal: Vec3) -> Vec<f64> {
    points
        .par_iter()
        .map(|&p| (Vec3::from_array(p) - origin).dot(normal) as f64)
        .collect()
}

fn grid_signed_dists(points: GridPoints<'_>, origin: Vec3, normal: Vec3) -> Vec<f64> {
    match points {
        GridPoints::Explicit { points, .. } => signed_dists(points, origin, normal),
        GridPoints::Image(img) => {
            let [nx, ny, nz] = img.dimensions;
            (0..(nx * ny * nz))
                .into_par_iter()
                .map(|idx| {
                    let i = idx % nx;
                    let rem = idx / nx;
                    let j = rem % ny;
                    let k = rem / ny;
                    let p = img.point_coords(i, j, k);
                    (Vec3::from_array(p) - origin).dot(normal) as f64
                })
                .collect()
        }
        GridPoints::Rectilinear(rg) => {
            let [nx, ny, nz] = rg.dimensions();
            (0..(nx * ny * nz))
                .into_par_iter()
                .map(|idx| {
                    let i = idx % nx;
                    let rem = idx / nx;
                    let j = rem % ny;
                    let k = rem / ny;
                    let p = rg.point_coords(i, j, k);
                    (Vec3::from_array(p) - origin).dot(normal) as f64
                })
                .collect()
        }
    }
}

/// Extracts the exact grid plane for axis-aligned slice normals (x/y/z).
/// Returns `None` when the plane is not aligned with a grid plane.
fn axis_aligned_slice(
    dims: [usize; 3],
    points: GridPoints<'_>,
    point_data: &FieldData,
    origin: Vec3,
    normal: Vec3,
) -> Option<PolyData> {
    let [nx, ny, nz] = dims;
    let n = normal.normalize_or(Vec3::Z);
    let ax = if n.x.abs() > 0.999 {
        0usize
    } else if n.y.abs() > 0.999 {
        1
    } else if n.z.abs() > 0.999 {
        2
    } else {
        return None;
    };
    let nax = [nx, ny, nz][ax];
    if nax < 2 {
        return None;
    }
    let o = [origin.x, origin.y, origin.z][ax];
    let coord = |i: usize| -> f32 {
        let p = match ax {
            0 => points.at_ijk(i, 0, 0),
            1 => points.at_ijk(0, i, 0),
            _ => points.at_ijk(0, 0, i),
        };
        p[ax]
    };
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for i in 0..nax {
        let d = (coord(i) - o).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let span = (coord(nax - 1) - coord(0)).abs().max(1.0);
    if best_d > 1e-4 * span {
        return None;
    }
    let (u, v) = match ax {
        0 => (ny, nz),
        1 => (nx, nz),
        _ => (nx, ny),
    };
    let src = |uu: usize, vv: usize| -> (usize, usize, usize) {
        match ax {
            0 => (best, uu, vv),
            1 => (uu, best, vv),
            _ => (uu, vv, best),
        }
    };
    let mut out_points = Vec::with_capacity(u * v);
    let mut src_indices = Vec::with_capacity(u * v);
    for vv in 0..v {
        for uu in 0..u {
            let (i, j, k) = src(uu, vv);
            out_points.push(points.at_ijk(i, j, k));
            src_indices.push(i + nx * (j + ny * k));
        }
    }
    let mut polys = Vec::with_capacity((u - 1) * (v - 1) * 2);
    for vv in 0..v.saturating_sub(1) {
        for uu in 0..u.saturating_sub(1) {
            let a = (vv * u + uu) as u32;
            let b = (vv * u + uu + 1) as u32;
            let c = ((vv + 1) * u + uu + 1) as u32;
            let d = ((vv + 1) * u + uu) as u32;
            polys.push(vec![a, b, c]);
            polys.push(vec![a, c, d]);
        }
    }
    let pdata = if point_data.is_empty() {
        FieldData::new()
    } else {
        point_data.gather(&src_indices)
    };
    Some(PolyData {
        points: out_points,
        polys,
        point_data: pdata,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn slice_structured(
    dims: [usize; 3],
    points: GridPoints<'_>,
    point_data: &FieldData,
    cell_data: &FieldData,
    cell_type: CellType,
    origin: Vec3,
    normal: Vec3,
    crinkle: bool,
) -> PolyData {
    let [nx, ny, nz] = dims;
    if nx < 2 || ny < 2 || nz < 2 || points.is_empty() {
        return PolyData::new();
    }
    // Axis-aligned planes on structured grids take the exact grid-plane fast
    // path (matches vtkCutter on ImageData: a clean triangulated grid slice,
    // no interpolated cut points, no generated normals).
    if !crinkle
        && let Some(pd) = axis_aligned_slice(dims, points, point_data, origin, normal)
    {
        return pd;
    }
    let dists = grid_signed_dists(points, origin, normal);
    let cx = nx - 1;
    let cy = ny - 1;
    let cz = nz - 1;
    let total_cells = cx * cy * cz;

    let corner_indices = |i: usize, j: usize, k: usize| -> [usize; 8] {
        let idx = |ii: usize, jj: usize, kk: usize| ii + nx * (jj + ny * kk);
        match cell_type {
            CellType::Voxel => [
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i, j + 1, k),
                idx(i + 1, j + 1, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i, j + 1, k + 1),
                idx(i + 1, j + 1, k + 1),
            ],
            _ => [
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j + 1, k),
                idx(i, j + 1, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i + 1, j + 1, k + 1),
                idx(i, j + 1, k + 1),
            ],
        }
    };

    let corner_coords = |i: usize, j: usize, k: usize| -> [[f32; 3]; 8] {
        match cell_type {
            CellType::Voxel => [
                points.at_ijk(i, j, k),
                points.at_ijk(i + 1, j, k),
                points.at_ijk(i, j + 1, k),
                points.at_ijk(i + 1, j + 1, k),
                points.at_ijk(i, j, k + 1),
                points.at_ijk(i + 1, j, k + 1),
                points.at_ijk(i, j + 1, k + 1),
                points.at_ijk(i + 1, j + 1, k + 1),
            ],
            _ => [
                points.at_ijk(i, j, k),
                points.at_ijk(i + 1, j, k),
                points.at_ijk(i + 1, j + 1, k),
                points.at_ijk(i, j + 1, k),
                points.at_ijk(i, j, k + 1),
                points.at_ijk(i + 1, j, k + 1),
                points.at_ijk(i + 1, j + 1, k + 1),
                points.at_ijk(i, j + 1, k + 1),
            ],
        }
    };

    if crinkle {
        let faces = cell_type.faces(8);
        let mut polys = Vec::new();
        let mut cell_ids = Vec::new();
        let mut pt_map: FxHashMap<usize, u32> = FxHashMap::default();
        let mut out_points = Vec::new();
        let mut old_pt_indices = Vec::new();

        for cell_idx in 0..total_cells {
            let k = cell_idx / (cx * cy);
            let rem = cell_idx % (cx * cy);
            let j = rem / cx;
            let i = rem % cx;
            let c = corner_indices(i, j, k);
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &vi in &c {
                mn = mn.min(dists[vi]);
                mx = mx.max(dists[vi]);
            }
            if mn <= 0.0 && mx >= 0.0 {
                let c_pts = corner_coords(i, j, k);
                for f in &faces {
                    let face_verts: Vec<u32> = f
                        .iter()
                        .map(|&li| {
                            let global_vi = c[li];
                            *pt_map.entry(global_vi).or_insert_with(|| {
                                let id = out_points.len() as u32;
                                out_points.push(c_pts[li]);
                                old_pt_indices.push(global_vi);
                                id
                            })
                        })
                        .collect();
                    polys.push(face_verts);
                    cell_ids.push(cell_idx);
                }
            }
        }
        let pdata = if point_data.is_empty() {
            FieldData::new()
        } else {
            point_data.gather(&old_pt_indices)
        };
        let cdata = if cell_data.is_empty() {
            FieldData::new()
        } else {
            cell_data.gather(&cell_ids)
        };
        let mut pd = PolyData {
            points: out_points,
            polys,
            point_data: pdata,
            cell_data: cdata,
            ..Default::default()
        };
        let _ = &mut pd;
        return pd;
    }

    let tets = cell_type.tetrahedra();
    let raw_tris: Vec<EdgeTri> = (0..total_cells)
        .into_par_iter()
        .flat_map_iter(|cell_idx| {
            let k = cell_idx / (cx * cy);
            let rem = cell_idx % (cx * cy);
            let j = rem / cx;
            let i = rem % cx;
            let c = corner_indices(i, j, k);
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &vi in &c {
                mn = mn.min(dists[vi]);
                mx = mx.max(dists[vi]);
            }
            if mn > 0.0 || mx < 0.0 {
                return Vec::new();
            }
            let c_pts = corner_coords(i, j, k);
            let mut local = Vec::new();
            for tet in &tets {
                let tv = [c[tet[0]], c[tet[1]], c[tet[2]], c[tet[3]]];
                let dv = [dists[tv[0]], dists[tv[1]], dists[tv[2]], dists[tv[3]]];
                let t_pts = [
                    c_pts[tet[0]],
                    c_pts[tet[1]],
                    c_pts[tet[2]],
                    c_pts[tet[3]],
                ];
                contour_tetrahedron_pts(t_pts, tv, dv, 0.0, &mut local);
            }
            local
        })
        .collect();

    let (pts, edges, tris) = deduplicate_edge_triangles(raw_tris, true);
    let pdata = interpolate_field_data_edges(point_data, &edges);
    let mut pd = PolyData::from_triangles(pts, tris);
    pd.point_data = pdata;
    pd
}

fn slice_unstructured(ug: &UnstructuredGrid, origin: Vec3, normal: Vec3, crinkle: bool) -> PolyData {
    let n_cells = ug.num_cells();
    if n_cells == 0 {
        return PolyData::new();
    }
    let dists = signed_dists(&ug.points, origin, normal);

    if crinkle {
        let mut polys = Vec::new();
        let mut lines = Vec::new();
        let mut cell_ids = Vec::new();
        for cell_idx in 0..n_cells {
            let ct = ug.cell_types[cell_idx];
            let pts = ug.cell_points(cell_idx);
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &vi in pts {
                let d = dists[vi as usize];
                mn = mn.min(d);
                mx = mx.max(d);
            }
            if mn <= 0.0 && mx >= 0.0 {
                if ct.dimension() >= 2 {
                    for f in ct.faces(pts.len()) {
                        polys.push(f.into_iter().map(|li| pts[li]).collect());
                        cell_ids.push(cell_idx);
                    }
                } else if ct.dimension() == 1 && pts.len() >= 2 {
                    lines.push(pts.to_vec());
                }
            }
        }
        let cdata = if !ug.cell_data.is_empty() && lines.is_empty() && !cell_ids.is_empty() {
            ug.cell_data.gather(&cell_ids)
        } else {
            FieldData::new()
        };
        let pd = compact_polydata_points(PolyData {
            points: ug.points.clone(),
            lines,
            polys,
            point_data: ug.point_data.clone(),
            cell_data: cdata,
            ..Default::default()
        });
        return pd;
    }

    let raw_tris: Vec<EdgeTri> = (0..n_cells)
        .into_par_iter()
        .flat_map_iter(|cell_idx| {
            let ct = ug.cell_types[cell_idx];
            let cpts = ug.cell_points(cell_idx);
            let mut local = Vec::new();
            for tet in ct.tetrahedra() {
                let tv = [
                    cpts[tet[0]] as usize,
                    cpts[tet[1]] as usize,
                    cpts[tet[2]] as usize,
                    cpts[tet[3]] as usize,
                ];
                let dv = [dists[tv[0]], dists[tv[1]], dists[tv[2]], dists[tv[3]]];
                contour_tetrahedron(&ug.points, tv, dv, 0.0, &mut local);
            }
            local
        })
        .collect();

    let (pts, edges, tris) = deduplicate_edge_triangles(raw_tris, true);
    let pdata = interpolate_field_data_edges(&ug.point_data, &edges);
    let mut pd = PolyData::from_triangles(pts, tris);
    pd.point_data = pdata;
    pd
}

fn slice_polydata(pd: &PolyData, origin: Vec3, normal: Vec3, crinkle: bool) -> PolyData {
    if pd.points.is_empty() {
        return PolyData::new();
    }
    let dists = signed_dists(&pd.points, origin, normal);

    if crinkle {
        let mut polys = Vec::new();
        for poly in &pd.polys {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &vi in poly {
                let d = dists[vi as usize];
                mn = mn.min(d);
                mx = mx.max(d);
            }
            if mn <= 0.0 && mx >= 0.0 {
                polys.push(poly.clone());
            }
        }
        return compact_polydata_points(PolyData {
            points: pd.points.clone(),
            polys,
            point_data: pd.point_data.clone(),
            ..Default::default()
        });
    }

    // Slice triangles/polygons by the plane into line segments.
    let tris = pd.triangulate();
    let mut edge_map: FxHashMap<u64, u32> = FxHashMap::default();
    let mut out_pts: Vec<[f32; 3]> = Vec::new();
    let mut out_edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut lines: Vec<Vec<u32>> = Vec::new();

    for tri in tris {
        let mut seg_verts: Vec<u32> = Vec::with_capacity(2);
        for e in 0..3 {
            let ia = tri[e] as usize;
            let ib = tri[(e + 1) % 3] as usize;
            let da = dists[ia];
            let db = dists[ib];
            if (da < 0.0 && db >= 0.0) || (db < 0.0 && da >= 0.0) || (da == 0.0 && db == 0.0) {
                if da == 0.0 && db == 0.0 {
                    // Edge lies on plane
                    let key_a = canonical_edge_key(ia as u32, ia as u32);
                    let key_b = canonical_edge_key(ib as u32, ib as u32);
                    let va = *edge_map.entry(key_a).or_insert_with(|| {
                        let id = out_pts.len() as u32;
                        out_pts.push(pd.points[ia]);
                        out_edges.push((ia, ia, 0.0));
                        id
                    });
                    let vb = *edge_map.entry(key_b).or_insert_with(|| {
                        let id = out_pts.len() as u32;
                        out_pts.push(pd.points[ib]);
                        out_edges.push((ib, ib, 0.0));
                        id
                    });
                    if va != vb {
                        lines.push(vec![va, vb]);
                    }
                    seg_verts.clear();
                    break;
                }
                let (t, pos) = interp_edge(&pd.points, ia, ib, da, db, 0.0);
                let key = canonical_edge_key(ia as u32, ib as u32);
                let vid = *edge_map.entry(key).or_insert_with(|| {
                    let id = out_pts.len() as u32;
                    out_pts.push(pos);
                    out_edges.push((ia, ib, t));
                    id
                });
                if !seg_verts.contains(&vid) {
                    seg_verts.push(vid);
                }
            }
        }
        if seg_verts.len() == 2 {
            lines.push(seg_verts);
        }
    }

    let pdata = interpolate_field_data_edges(&pd.point_data, &out_edges);
    PolyData {
        points: out_pts,
        lines,
        point_data: pdata,
        ..Default::default()
    }
}
