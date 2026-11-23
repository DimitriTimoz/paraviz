//! Cell shrinking filter (`ShrinkFilter`, equivalent to `vtkShrinkFilter`).

use crate::util::dataset_to_unstructured;
use glam::Vec3;
use paraviz_core::{
    DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData, UnstructuredGrid,
};
use serde::{Deserialize, Serialize};

/// Shrinks each cell toward its geometric centroid by `shrink_factor` (`0.0..=1.0`),
/// producing unshared cell vertices while preserving `cell_data` and remapping `point_data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShrinkFilter {
    /// Scale factor around each cell's centroid (`0.0` collapses to centroid, `1.0` keeps original size).
    pub shrink_factor: f32,
}

impl Default for ShrinkFilter {
    fn default() -> Self {
        Self { shrink_factor: 0.5 }
    }
}

impl ShrinkFilter {
    /// Creates a `ShrinkFilter` with the given `shrink_factor` (`0.0..=1.0`).
    pub fn new(shrink_factor: f32) -> Self {
        Self { shrink_factor }
    }
}

impl Filter for ShrinkFilter {
    fn name(&self) -> &str {
        "ShrinkFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let factor = self.shrink_factor.clamp(0.0, 1.0);
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(shrink_polydata(pd, factor))),
            DataSet::UnstructuredGrid(ug) => {
                Ok(DataSet::UnstructuredGrid(shrink_unstructured(ug, factor)))
            }
            DataSet::ImageData(_) | DataSet::RectilinearGrid(_) | DataSet::StructuredGrid(_) => {
                let ug = dataset_to_unstructured(input)?;
                Ok(DataSet::UnstructuredGrid(shrink_unstructured(&ug, factor)))
            }
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
        }
    }
}

fn shrink_cell_vertices(
    src_points: &[[f32; 3]],
    cell_pts: &[u32],
    factor: f32,
    out_points: &mut Vec<[f32; 3]>,
    old_point_indices: &mut Vec<usize>,
) -> Vec<u32> {
    if cell_pts.is_empty() {
        return Vec::new();
    }
    let mut centroid = Vec3::ZERO;
    let mut valid_count = 0usize;
    for &vi in cell_pts {
        if let Some(&p) = src_points.get(vi as usize) {
            centroid += Vec3::from_array(p);
            valid_count += 1;
        }
    }
    if valid_count > 0 {
        centroid /= valid_count as f32;
    }

    let mut new_indices = Vec::with_capacity(cell_pts.len());
    for &vi in cell_pts {
        let idx = vi as usize;
        let p = src_points
            .get(idx)
            .map(|&a| Vec3::from_array(a))
            .unwrap_or(centroid);
        let shrunk = centroid + (p - centroid) * factor;
        let new_id = out_points.len() as u32;
        out_points.push(shrunk.to_array());
        old_point_indices.push(idx.min(src_points.len().saturating_sub(1)));
        new_indices.push(new_id);
    }
    new_indices
}

fn shrink_polydata(pd: &PolyData, factor: f32) -> PolyData {
    let total_refs = pd.verts.len()
        + pd.lines.iter().map(Vec::len).sum::<usize>()
        + pd.polys.iter().map(Vec::len).sum::<usize>()
        + pd.strips.iter().map(Vec::len).sum::<usize>();

    let mut out_points = Vec::with_capacity(total_refs);
    let mut old_point_indices = Vec::with_capacity(total_refs);
    let mut verts = Vec::with_capacity(pd.verts.len());
    let mut lines = Vec::with_capacity(pd.lines.len());
    let mut polys = Vec::with_capacity(pd.polys.len());
    let mut strips = Vec::with_capacity(pd.strips.len());

    for &v in &pd.verts {
        let new_v = shrink_cell_vertices(
            &pd.points,
            &[v],
            factor,
            &mut out_points,
            &mut old_point_indices,
        );
        verts.push(new_v[0]);
    }
    for line in &pd.lines {
        lines.push(shrink_cell_vertices(
            &pd.points,
            line,
            factor,
            &mut out_points,
            &mut old_point_indices,
        ));
    }
    for poly in &pd.polys {
        polys.push(shrink_cell_vertices(
            &pd.points,
            poly,
            factor,
            &mut out_points,
            &mut old_point_indices,
        ));
    }
    for strip in &pd.strips {
        strips.push(shrink_cell_vertices(
            &pd.points,
            strip,
            factor,
            &mut out_points,
            &mut old_point_indices,
        ));
    }

    let point_data = if pd.points.is_empty() {
        pd.point_data.clone()
    } else {
        pd.point_data.gather(&old_point_indices)
    };

    PolyData {
        points: out_points,
        verts,
        lines,
        polys,
        strips,
        point_data,
        cell_data: pd.cell_data.clone(),
        field_data: pd.field_data.clone(),
    }
}

fn shrink_unstructured(ug: &UnstructuredGrid, factor: f32) -> UnstructuredGrid {
    let n_cells = ug.num_cells();
    let mut out = UnstructuredGrid::new();
    out.points.reserve(ug.connectivity.len());
    let mut old_point_indices = Vec::with_capacity(ug.connectivity.len());

    for c in 0..n_cells {
        let pts = ug.cell_points(c);
        let new_pts = shrink_cell_vertices(
            &ug.points,
            pts,
            factor,
            &mut out.points,
            &mut old_point_indices,
        );
        out.push_cell(ug.cell_types[c], &new_pts);
    }

    out.point_data = if ug.points.is_empty() {
        ug.point_data.clone()
    } else {
        ug.point_data.gather(&old_point_indices)
    };
    out.cell_data = ug.cell_data.clone();
    out.field_data = ug.field_data.clone();
    out
}
