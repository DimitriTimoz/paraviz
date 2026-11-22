//! Surface extraction filter (`ExtractSurface`, equivalent to `vtkDataSetSurfaceFilter` / `vtkGeometryFilter`).

use crate::util::{compact_polydata_points, merge_polydata};
use paraviz_core::{
    CellType, DataSet, FieldData, Filter, ImageData, PipelineError, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    points: Vec<[f32; 3]>,
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
            points,
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
            points,
            lines: vec![line],
            point_data: point_data.clone(),
            ..Default::default()
        };
    }

    // 2D grid (one dimension == 1)
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
            points,
            polys,
            point_data: point_data.clone(),
            cell_data: cdata,
            ..Default::default()
        };
    }

    // Full 3D grid: extract only the 6 boundary faces (equivalent to skipping all shared interior faces).
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
                pt_idx(0, j, k),
                pt_idx(0, j, k + 1),
                pt_idx(0, j + 1, k + 1),
                pt_idx(0, j + 1, k),
            ]);
            cell_indices.push(cell_idx(0, j, k));

            polys.push(vec![
                pt_idx(nx - 1, j, k),
                pt_idx(nx - 1, j + 1, k),
                pt_idx(nx - 1, j + 1, k + 1),
                pt_idx(nx - 1, j, k + 1),
            ]);
            cell_indices.push(cell_idx(cx - 1, j, k));
        }
    }
    // -Y and +Y faces
    for k in 0..cz {
        for i in 0..cx {
            polys.push(vec![
                pt_idx(i, 0, k),
                pt_idx(i + 1, 0, k),
                pt_idx(i + 1, 0, k + 1),
                pt_idx(i, 0, k + 1),
            ]);
            cell_indices.push(cell_idx(i, 0, k));

            polys.push(vec![
                pt_idx(i, ny - 1, k),
                pt_idx(i, ny - 1, k + 1),
                pt_idx(i + 1, ny - 1, k + 1),
                pt_idx(i + 1, ny - 1, k),
            ]);
            cell_indices.push(cell_idx(i, cy - 1, k));
        }
    }
    // -Z and +Z faces
    for j in 0..cy {
        for i in 0..cx {
            polys.push(vec![
                pt_idx(i, j, 0),
                pt_idx(i, j + 1, 0),
                pt_idx(i + 1, j + 1, 0),
                pt_idx(i + 1, j, 0),
            ]);
            cell_indices.push(cell_idx(i, j, 0));

            polys.push(vec![
                pt_idx(i, j, nz - 1),
                pt_idx(i + 1, j, nz - 1),
                pt_idx(i + 1, j + 1, nz - 1),
                pt_idx(i, j + 1, nz - 1),
            ]);
            cell_indices.push(cell_idx(i, j, cz - 1));
        }
    }

    let cdata = if cell_data.is_empty() {
        FieldData::new()
    } else {
        cell_data.gather(&cell_indices)
    };

    compact_polydata_points(PolyData {
        points,
        polys,
        point_data: point_data.clone(),
        cell_data: cdata,
        ..Default::default()
    })
}

fn extract_image_surface(img: &ImageData) -> PolyData {
    extract_structured_grid_topology(
        img.dimensions,
        img.all_points(),
        &img.point_data,
        &img.cell_data,
    )
}

fn extract_rectilinear_surface(rg: &RectilinearGrid) -> PolyData {
    extract_structured_grid_topology(
        rg.dimensions(),
        rg.all_points(),
        &rg.point_data,
        &rg.cell_data,
    )
}

fn extract_structured_surface(sg: &StructuredGrid) -> PolyData {
    extract_structured_grid_topology(
        sg.dimensions,
        sg.points.clone(),
        &sg.point_data,
        &sg.cell_data,
    )
}

/// Canonical sorted vertex key for triangle/quad faces (zero-allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FaceKey4([u32; 4]);

impl FaceKey4 {
    #[inline]
    fn from_slice(verts: &[u32]) -> Self {
        let mut key = [u32::MAX; 4];
        let len = verts.len().min(4);
        key[..len].copy_from_slice(&verts[..len]);
        key[..len].sort_unstable();
        Self(key)
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
    let face_candidates: Vec<(FaceKey4, Vec<u32>, usize)> = (0..n_cells)
        .into_par_iter()
        .flat_map_iter(|cell_idx| {
            let ct = ug.cell_types[cell_idx];
            let pts = ug.cell_points(cell_idx);
            let mut local = Vec::new();
            if ct.dimension() == 3 {
                for f in ct.faces(pts.len()) {
                    let verts: Vec<u32> = f.into_iter().map(|li| pts[li]).collect();
                    let key = FaceKey4::from_slice(&verts);
                    local.push((key, verts, cell_idx));
                }
            }
            local
        })
        .collect();

    let mut face_map: HashMap<FaceKey4, FaceEntry> =
        HashMap::with_capacity(face_candidates.len());
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
