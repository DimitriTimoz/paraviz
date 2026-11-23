//! Arbitrary plane slicing filter (`SliceFilter`, equivalent to `vtkCutter`).

use crate::contour::{contour_tetrahedron, interp_edge};
use crate::util::{
    canonical_edge_key, compact_polydata_points, deduplicate_edge_triangles,
    interpolate_field_data_edges, merge_polydata, EdgeTri,
};
use glam::Vec3;
use paraviz_core::{
    CellType, DataSet, FieldData, Filter, PipelineError, PolyData, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
            normal: [0.0, 0.0, 1.0],
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
                &img.all_points(),
                &img.point_data,
                &img.cell_data,
                CellType::Voxel,
                origin,
                normal,
                self.crinkle,
            ),
            DataSet::RectilinearGrid(rg) => slice_structured(
                rg.dimensions(),
                &rg.all_points(),
                &rg.point_data,
                &rg.cell_data,
                CellType::Voxel,
                origin,
                normal,
                self.crinkle,
            ),
            DataSet::StructuredGrid(sg) => slice_structured(
                sg.dimensions,
                &sg.points,
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

#[allow(clippy::too_many_arguments)]
fn slice_structured(
    dims: [usize; 3],
    points: &[[f32; 3]],
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
    let dists = signed_dists(points, origin, normal);
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

    if crinkle {
        let faces = cell_type.faces(8);
        let mut polys = Vec::new();
        let mut cell_ids = Vec::new();
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
                for f in &faces {
                    polys.push(f.iter().map(|&li| c[li] as u32).collect());
                    cell_ids.push(cell_idx);
                }
            }
        }
        let cdata = if cell_data.is_empty() {
            FieldData::new()
        } else {
            cell_data.gather(&cell_ids)
        };
        let mut pd = compact_polydata_points(PolyData {
            points: points.to_vec(),
            polys,
            point_data: point_data.clone(),
            cell_data: cdata,
            ..Default::default()
        });
        if pd.point_data.active_normals_array().is_none() && !pd.polys.is_empty() {
            pd.compute_normals();
        }
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
            let mut local = Vec::new();
            for tet in &tets {
                let tv = [c[tet[0]], c[tet[1]], c[tet[2]], c[tet[3]]];
                let dv = [dists[tv[0]], dists[tv[1]], dists[tv[2]], dists[tv[3]]];
                contour_tetrahedron(points, tv, dv, 0.0, &mut local);
            }
            local
        })
        .collect();

    let (pts, edges, tris) = deduplicate_edge_triangles(raw_tris, true);
    let pdata = interpolate_field_data_edges(point_data, &edges);
    let mut pd = PolyData::from_triangles(pts, tris);
    pd.point_data = pdata;
    if !pd.polys.is_empty() {
        pd.compute_normals();
    }
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
        let mut pd = compact_polydata_points(PolyData {
            points: ug.points.clone(),
            lines,
            polys,
            point_data: ug.point_data.clone(),
            cell_data: cdata,
            ..Default::default()
        });
        if pd.point_data.active_normals_array().is_none() && !pd.polys.is_empty() {
            pd.compute_normals();
        }
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
    if !pd.polys.is_empty() {
        pd.compute_normals();
    }
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
    let mut edge_map: HashMap<u64, u32> = HashMap::new();
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
