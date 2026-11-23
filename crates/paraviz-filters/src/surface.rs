//! Surface extraction filter (`ExtractSurface`, equivalent to `vtkDataSetSurfaceFilter` / `vtkGeometryFilter`).

use crate::util::{compact_polydata_points, fx_map_with_capacity, merge_polydata, FxHashMap, GridPoints};
use paraviz_core::{
    CellType, DataSet, FieldData, Filter, ImageData, PipelineError, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Extracts the outer boundary surface of any dataset into a renderable [`PolyData`].
///
/// For 3D volumetric grids, shared interior faces are skipped so only boundary faces are emitted.
/// Smooth vertex normals are computed automatically if none exist.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtractSurface {
    /// Whether to compute smooth vertex normals when the input has no normals array.
    pub compute_normals: bool,
}

impl ExtractSurface {
    pub fn new() -> Self {
        Self {
            compute_normals: true,
        }
    }
}

impl Filter for ExtractSurface {
    fn name(&self) -> &str {
        "ExtractSurface"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let mut pd = extract_surface(input)?;
        if !self.compute_normals && input.point_data().and_then(|p| p.active_normals_array()).is_none()
        {
            pd.point_data.remove("Normals");
        }
        Ok(DataSet::PolyData(pd))
    }
}

/// Converts any [`DataSet`] into a renderable [`PolyData`] by extracting boundary faces
/// and computing smooth normals if none exist.
pub fn extract_surface(input: &DataSet) -> Result<PolyData, PipelineError> {
    let mut pd = match input {
        DataSet::PolyData(poly) => poly.clone(),
        DataSet::ImageData(img) => extract_image_surface(img),
        DataSet::RectilinearGrid(rg) => extract_rectilinear_surface(rg),
        DataSet::StructuredGrid(sg) => extract_structured_surface(sg),
        DataSet::UnstructuredGrid(ug) => extract_unstructured_surface(ug),
        DataSet::MultiBlock(mb) => {
            let parts: Vec<PolyData> = mb
                .blocks
                .par_iter()
                .map(|(_, block)| extract_surface(block))
                .collect::<Result<Vec<_>, _>>()?;
            merge_polydata(parts)
        }
    };

    if pd.point_data.active_normals_array().is_none() && !pd.polys.is_empty() {
        pd.compute_normals();
    }
    Ok(pd)
}

fn extract_structured_grid_topology(
    dims: [usize; 3],
    points: GridPoints<'_>,
    point_data: &FieldData,
    cell_data: &FieldData,
) -> PolyData {
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 || points.is_empty() {
        return PolyData::new();
    }

    let pt_idx = |i: usize, j: usize, k: usize| -> u32 { (i + nx * (j + ny * k)) as u32 };

    // Handle degenerate 0D/1D/2D structured grids cleanly.
    if nx == 1 && ny == 1 && nz == 1 {
        return PolyData {
            points: vec![points.at_ijk(0, 0, 0)],
            verts: vec![0],
            point_data: point_data.clone(),
            cell_data: cell_data.clone(),
            ..Default::default()
        };
    }
    if (nx == 1 && ny == 1) || (nx == 1 && nz == 1) || (ny == 1 && nz == 1) {
        let count = nx * ny * nz;
        let line: Vec<u32> = (0..count as u32).collect();
        return PolyData {
            points: points.to_vec_par(),
            lines: vec![line],
            point_data: point_data.clone(),
            ..Default::default()
        };
    }

    // 2D grid (one dimension == 1): all points belong to the boundary surface.
    if nx == 1 || ny == 1 || nz == 1 {
        let mut polys = Vec::new();
        let mut cell_indices = Vec::new();
        if nz == 1 {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    polys.push(vec![
                        pt_idx(i, j, 0),
                        pt_idx(i + 1, j, 0),
                        pt_idx(i + 1, j + 1, 0),
                        pt_idx(i, j + 1, 0),
                    ]);
                    cell_indices.push(i + (nx - 1) * j);
                }
            }
        } else if ny == 1 {
            for k in 0..(nz - 1) {
                for i in 0..(nx - 1) {
                    polys.push(vec![
                        pt_idx(i, 0, k),
                        pt_idx(i + 1, 0, k),
                        pt_idx(i + 1, 0, k + 1),
                        pt_idx(i, 0, k + 1),
                    ]);
                    cell_indices.push(i + (nx - 1) * k);
                }
            }
        } else {
            // nx == 1
            for k in 0..(nz - 1) {
                for j in 0..(ny - 1) {
                    polys.push(vec![
                        pt_idx(0, j, k),
                        pt_idx(0, j + 1, k),
                        pt_idx(0, j + 1, k + 1),
                        pt_idx(0, j, k + 1),
                    ]);
                    cell_indices.push(j + (ny - 1) * k);
                }
            }
        }
        let cdata = if cell_data.is_empty() {
            FieldData::new()
        } else {
            cell_data.gather(&cell_indices)
        };
        return PolyData {
            points: points.to_vec_par(),
            polys,
            point_data: point_data.clone(),
            cell_data: cdata,
            ..Default::default()
        };
    }

    // Full 3D grid: materialize only the outer 6 boundary face vertices directly
    // in ascending linear `(k, j, i)` order, skipping all interior points.
    let slice_full = nx * ny;
    let slice_ring = 2 * nx + 2 * (ny - 2);
    let boundary_count = 2 * slice_full + (nz - 2) * slice_ring;

    let mut boundary_points = Vec::with_capacity(boundary_count);
    let has_pdata = !point_data.is_empty();
    let mut old_pt_indices = if has_pdata {
        Vec::with_capacity(boundary_count)
    } else {
        Vec::new()
    };

    let mut push_boundary_pt = |i: usize, j: usize, k: usize| {
        boundary_points.push(points.at_ijk(i, j, k));
        if has_pdata {
            old_pt_indices.push(i + nx * (j + ny * k));
        }
    };

    for k in 0..nz {
        if k == 0 || k == nz - 1 {
            for j in 0..ny {
                for i in 0..nx {
                    push_boundary_pt(i, j, k);
                }
            }
        } else {
            for j in 0..ny {
                if j == 0 || j == ny - 1 {
                    for i in 0..nx {
                        push_boundary_pt(i, j, k);
                    }
                } else {
                    push_boundary_pt(0, j, k);
                    push_boundary_pt(nx - 1, j, k);
                }
            }
        }
    }

    // Maps any boundary `(i, j, k)` directly to its compacted index in `0..boundary_count`.
    let b_idx = |i: usize, j: usize, k: usize| -> u32 {
        let idx = if k == 0 {
            i + nx * j
        } else if k == nz - 1 {
            slice_full + (nz - 2) * slice_ring + i + nx * j
        } else {
            let base = slice_full + (k - 1) * slice_ring;
            if j == 0 {
                base + i
            } else if j == ny - 1 {
                base + nx + 2 * (ny - 2) + i
            } else {
                debug_assert!(i == 0 || i == nx - 1);
                base + nx + (j - 1) * 2 + usize::from(i == nx - 1)
            }
        };
        idx as u32
    };

    let cx = nx - 1;
    let cy = ny - 1;
    let cz = nz - 1;
    let cell_idx = |i: usize, j: usize, k: usize| -> usize { i + cx * (j + cy * k) };

    let face_count = 2 * (cx * cy + cx * cz + cy * cz);
    let mut polys = Vec::with_capacity(face_count);
    let mut cell_indices = Vec::with_capacity(face_count);

    // -X and +X faces
    for k in 0..cz {
        for j in 0..cy {
            polys.push(vec![
                b_idx(0, j, k),
                b_idx(0, j, k + 1),
                b_idx(0, j + 1, k + 1),
                b_idx(0, j + 1, k),
            ]);
            cell_indices.push(cell_idx(0, j, k));

            polys.push(vec![
                b_idx(nx - 1, j, k),
                b_idx(nx - 1, j + 1, k),
                b_idx(nx - 1, j + 1, k + 1),
                b_idx(nx - 1, j, k + 1),
            ]);
            cell_indices.push(cell_idx(cx - 1, j, k));
        }
    }
    // -Y and +Y faces
    for k in 0..cz {
        for i in 0..cx {
            polys.push(vec![
                b_idx(i, 0, k),
                b_idx(i + 1, 0, k),
                b_idx(i + 1, 0, k + 1),
                b_idx(i, 0, k + 1),
            ]);
            cell_indices.push(cell_idx(i, 0, k));

            polys.push(vec![
                b_idx(i, ny - 1, k),
                b_idx(i, ny - 1, k + 1),
                b_idx(i + 1, ny - 1, k + 1),
                b_idx(i + 1, ny - 1, k),
            ]);
            cell_indices.push(cell_idx(i, cy - 1, k));
        }
    }
    // -Z and +Z faces
    for j in 0..cy {
        for i in 0..cx {
            polys.push(vec![
                b_idx(i, j, 0),
                b_idx(i, j + 1, 0),
                b_idx(i + 1, j + 1, 0),
                b_idx(i + 1, j, 0),
            ]);
            cell_indices.push(cell_idx(i, j, 0));

            polys.push(vec![
                b_idx(i, j, nz - 1),
                b_idx(i + 1, j, nz - 1),
                b_idx(i + 1, j + 1, nz - 1),
                b_idx(i, j + 1, nz - 1),
            ]);
            cell_indices.push(cell_idx(i, j, cz - 1));
        }
    }

    let pdata = if has_pdata {
        point_data.gather(&old_pt_indices)
    } else {
        FieldData::new()
    };
    let cdata = if cell_data.is_empty() {
        FieldData::new()
    } else {
        cell_data.gather(&cell_indices)
    };

    PolyData {
        points: boundary_points,
        polys,
        point_data: pdata,
        cell_data: cdata,
        ..Default::default()
    }
}

fn extract_image_surface(img: &ImageData) -> PolyData {
    extract_structured_grid_topology(
        img.dimensions,
        GridPoints::Image(img),
        &img.point_data,
        &img.cell_data,
    )
}

fn extract_rectilinear_surface(rg: &RectilinearGrid) -> PolyData {
    extract_structured_grid_topology(
        rg.dimensions(),
        GridPoints::Rectilinear(rg),
        &rg.point_data,
        &rg.cell_data,
    )
}

fn extract_structured_surface(sg: &StructuredGrid) -> PolyData {
    extract_structured_grid_topology(
        sg.dimensions,
        GridPoints::from_structured(&sg.points, sg.dimensions),
        &sg.point_data,
        &sg.cell_data,
    )
}

/// Canonical sorted vertex key for polygonal faces (zero-allocation up to 4 vertices,
/// exact sorted `Vec<u32>` comparison for >4-vertex polygons).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FaceKey {
    Small([u32; 4], u8),
    Large(Vec<u32>),
}

impl FaceKey {
    #[inline]
    fn from_slice(verts: &[u32]) -> Self {
        if verts.len() <= 4 {
            let mut key = [u32::MAX; 4];
            let len = verts.len();
            key[..len].copy_from_slice(verts);
            key[..len].sort_unstable();
            Self::Small(key, len as u8)
        } else {
            let mut key = verts.to_vec();
            key.sort_unstable();
            Self::Large(key)
        }
    }
}

enum FaceEntry {
    Single {
        verts: Vec<u32>,
        cell_idx: usize,
    },
    Shared,
}

fn extract_unstructured_surface(ug: &UnstructuredGrid) -> PolyData {
    let n_cells = ug.num_cells();
    if n_cells == 0 {
        return PolyData::new();
    }

    // Extract faces of 3D cells in parallel.
    let face_candidates: Vec<(FaceKey, Vec<u32>, usize)> = (0..n_cells)
        .into_par_iter()
        .flat_map_iter(|cell_idx| {
            let ct = ug.cell_types[cell_idx];
            let pts = ug.cell_points(cell_idx);
            let mut local = Vec::new();
            if ct.dimension() == 3 {
                for f in ct.faces(pts.len()) {
                    let verts: Vec<u32> = f.into_iter().map(|li| pts[li]).collect();
                    let key = FaceKey::from_slice(&verts);
                    local.push((key, verts, cell_idx));
                }
            }
            local
        })
        .collect();

    let mut face_map: FxHashMap<FaceKey, FaceEntry> =
        fx_map_with_capacity(face_candidates.len());
    for (key, verts, cell_idx) in face_candidates {
        match face_map.get_mut(&key) {
            Some(entry) => {
                *entry = FaceEntry::Shared;
            }
            None => {
                face_map.insert(key, FaceEntry::Single { verts, cell_idx });
            }
        }
    }

    let mut polys = Vec::new();
    let mut poly_cell_indices = Vec::new();
    let mut lines = Vec::new();
    let mut verts = Vec::new();

    // Emit 0D/1D/2D cells in original order.
    for cell_idx in 0..n_cells {
        let ct = ug.cell_types[cell_idx];
        let pts = ug.cell_points(cell_idx);
        match ct.dimension() {
            0 => {
                verts.extend_from_slice(pts);
            }
            1 => {
                if pts.len() >= 2 {
                    lines.push(pts.to_vec());
                }
            }
            2 => {
                if ct == CellType::TriangleStrip {
                    for k in 0..pts.len().saturating_sub(2) {
                        if k % 2 == 0 {
                            polys.push(vec![pts[k], pts[k + 1], pts[k + 2]]);
                        } else {
                            polys.push(vec![pts[k + 1], pts[k], pts[k + 2]]);
                        }
                        poly_cell_indices.push(cell_idx);
                    }
                } else {
                    for f in ct.faces(pts.len()) {
                        polys.push(f.into_iter().map(|li| pts[li]).collect());
                        poly_cell_indices.push(cell_idx);
                    }
                }
            }
            _ => {}
        }
    }

    // Emit boundary faces from 3D cells (count == 1) in deterministic cell order.
    let mut boundary_faces: Vec<(usize, Vec<u32>)> = face_map
        .into_values()
        .filter_map(|entry| match entry {
            FaceEntry::Single { verts, cell_idx } => Some((cell_idx, verts)),
            FaceEntry::Shared => None,
        })
        .collect();
    boundary_faces.sort_by_key(|(cidx, _)| *cidx);
    for (cidx, fverts) in boundary_faces {
        polys.push(fverts);
        poly_cell_indices.push(cidx);
    }

    let cell_data = if !ug.cell_data.is_empty() && !poly_cell_indices.is_empty() && lines.is_empty() && verts.is_empty() {
        ug.cell_data.gather(&poly_cell_indices)
    } else {
        FieldData::new()
    };

    compact_polydata_points(PolyData {
        points: ug.points.clone(),
        verts,
        lines,
        polys,
        point_data: ug.point_data.clone(),
        cell_data,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use paraviz_core::DataArray;

    #[test]
    fn face_key_distinguishes_polygons_with_more_than_four_vertices() {
        // Two pentagons sharing the first 4 vertices [0, 1, 2, 3] must NOT collide.
        let k1 = FaceKey::from_slice(&[0, 1, 2, 3, 4]);
        let k2 = FaceKey::from_slice(&[0, 1, 2, 3, 5]);
        assert_ne!(k1, k2);

        // Same pentagon in reversed/rotated winding order must match.
        let k1_rev = FaceKey::from_slice(&[4, 3, 2, 1, 0]);
        assert_eq!(k1, k1_rev);

        // Quad vs pentagon sharing first 4 vertices must NOT collide.
        let q = FaceKey::from_slice(&[0, 1, 2, 3]);
        assert_ne!(q, k1);
    }

    #[test]
    fn extract_image_surface_materializes_only_outer_boundary_vertices() {
        let mut img = ImageData::new([4, 5, 6], [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]);
        let pts = img.all_points();
        let scalars: Vec<f32> = pts.iter().map(|p| p[0] + p[1] + p[2]).collect();
        img.point_data.set_scalars(DataArray::scalars_f32("s", scalars));

        let pd = extract_image_surface(&img);
        // 4*5*6 = 120 total points; 2*3*4 = 24 interior points => 96 boundary points.
        assert_eq!(pd.points.len(), 96);
        assert_eq!(pd.point_data.get("s").unwrap().num_tuples(), 96);
        // 2*(3*4 + 3*5 + 4*5) = 2*(12 + 15 + 20) = 94 boundary quad faces.
        assert_eq!(pd.polys.len(), 94);
        let s_arr = pd.point_data.get("s").unwrap();
        for (idx, p) in pd.points.iter().enumerate() {
            let expected = p[0] + p[1] + p[2];
            assert!((s_arr.get_scalar(idx) as f32 - expected).abs() < 1e-5);
        }
    }
}
