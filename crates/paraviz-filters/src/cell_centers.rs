//! Cell centroid extraction filter (`CellCentersFilter`, equivalent to `vtkCellCenters`).

use crate::util::structured_cells;
use glam::Vec3;
use paraviz_core::{DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Extracts the geometric centroid of every cell as a `PolyData` point cloud,
/// converting `cell_data` into `point_data`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CellCentersFilter {
    /// When `true`, emits standalone `verts` cells for each extracted centroid point.
    pub vertex_cells: bool,
}

impl CellCentersFilter {
    /// Creates a `CellCentersFilter`.
    pub fn new(vertex_cells: bool) -> Self {
        Self { vertex_cells }
    }
}

impl Filter for CellCentersFilter {
    fn name(&self) -> &str {
        "CellCentersFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let points = match input {
            DataSet::PolyData(pd) => compute_polydata_centers(pd),
            DataSet::UnstructuredGrid(ug) => (0..ug.num_cells())
                .into_par_iter()
                .map(|c| average_points(&ug.points, ug.cell_points(c)))
                .collect(),
            DataSet::ImageData(img) => {
                let [nx, ny, nz] = img.dimensions;
                if nx == 0 || ny == 0 || nz == 0 {
                    Vec::new()
                } else {
                    let cx = nx.saturating_sub(1).max(1);
                    let cy = ny.saturating_sub(1).max(1);
                    let cz = nz.saturating_sub(1).max(1);
                    let ox = if nx > 1 { 0.5 } else { 0.0 };
                    let oy = if ny > 1 { 0.5 } else { 0.0 };
                    let oz = if nz > 1 { 0.5 } else { 0.0 };
                    (0..(cx * cy * cz))
                        .into_par_iter()
                        .map(|idx| {
                            let i = idx % cx;
                            let rem = idx / cx;
                            let j = rem % cy;
                            let k = rem / cy;
                            [
                                (img.origin[0] + (i as f64 + ox) * img.spacing[0]) as f32,
                                (img.origin[1] + (j as f64 + oy) * img.spacing[1]) as f32,
                                (img.origin[2] + (k as f64 + oz) * img.spacing[2]) as f32,
                            ]
                        })
                        .collect()
                }
            }
            DataSet::RectilinearGrid(rg) => {
                let [nx, ny, nz] = rg.dimensions();
                if nx == 0 || ny == 0 || nz == 0 {
                    Vec::new()
                } else {
                    let cx = nx.saturating_sub(1).max(1);
                    let cy = ny.saturating_sub(1).max(1);
                    let cz = nz.saturating_sub(1).max(1);
                    (0..(cx * cy * cz))
                        .into_par_iter()
                        .map(|idx| {
                            let i = idx % cx;
                            let rem = idx / cx;
                            let j = rem % cy;
                            let k = rem / cy;
                            let x = if nx > 1 { 0.5 * (rg.x_coords[i] + rg.x_coords[i + 1]) } else { rg.x_coords[0] };
                            let y = if ny > 1 { 0.5 * (rg.y_coords[j] + rg.y_coords[j + 1]) } else { rg.y_coords[0] };
                            let z = if nz > 1 { 0.5 * (rg.z_coords[k] + rg.z_coords[k + 1]) } else { rg.z_coords[0] };
                            [x, y, z]
                        })
                        .collect()
                }
            }
            DataSet::StructuredGrid(sg) => {
                let cells = structured_cells(sg.dimensions, false);
                cells
                    .par_iter()
                    .map(|(_, verts)| average_points(&sg.points, verts))
                    .collect()
            }
            DataSet::MultiBlock(_) => unreachable!(),
        };

        let verts = if self.vertex_cells {
            (0..points.len() as u32).collect()
        } else {
            Vec::new()
        };

        Ok(DataSet::PolyData(PolyData {
            points,
            verts,
            point_data: input.cell_data().cloned().unwrap_or_default(),
            field_data: input.field_data().cloned().unwrap_or_default(),
            ..Default::default()
        }))
    }
}

#[inline]
fn average_points(src: &[[f32; 3]], indices: &[u32]) -> [f32; 3] {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for &vi in indices {
        if let Some(&p) = src.get(vi as usize) {
            sum += Vec3::from_array(p);
            count += 1;
        }
    }
    if count > 0 {
        (sum / count as f32).to_array()
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn compute_polydata_centers(pd: &PolyData) -> Vec<[f32; 3]> {
    let mut centers = Vec::with_capacity(pd.num_cells());
    for &v in &pd.verts {
        centers.push(average_points(&pd.points, &[v]));
    }
    for line in &pd.lines {
        centers.push(average_points(&pd.points, line));
    }
    for poly in &pd.polys {
        centers.push(average_points(&pd.points, poly));
    }
    for strip in &pd.strips {
        centers.push(average_points(&pd.points, strip));
    }
    centers
}
