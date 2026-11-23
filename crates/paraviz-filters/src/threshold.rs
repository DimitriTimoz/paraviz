//! Scalar range threshold extraction filter (`ThresholdFilter`, equivalent to `vtkThreshold`).

use crate::util::{compact_polydata_points, compact_unstructured_points, GridPoints};
use paraviz_core::{
    AttributeAssociation, CellType, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet,
    PipelineError, PolyData, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Extracts cells whose scalar (or vector magnitude) falls within `[min, max]`.
///
/// Outputs a `PolyData` if the input is `PolyData`, or an `UnstructuredGrid` for volumetric grids,
/// compacting unused points and remapping both `point_data` and `cell_data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdFilter {
    pub field_name: String,
    pub association: AttributeAssociation,
    pub min: f64,
    pub max: f64,
    pub invert: bool,
    pub all_scalars: bool,
}

impl Default for ThresholdFilter {
    fn default() -> Self {
        Self {
            field_name: String::new(),
            association: AttributeAssociation::Point,
            min: 0.0,
            max: 1.0,
            invert: false,
            all_scalars: true,
        }
    }
}

impl ThresholdFilter {
    pub fn new(field_name: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            field_name: field_name.into(),
            min,
            max,
            ..Default::default()
        }
    }
}

impl Filter for ThresholdFilter {
    fn name(&self) -> &str {
        "ThresholdFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.threshold_polydata(pd)?)),
            DataSet::ImageData(img) => Ok(DataSet::UnstructuredGrid(self.threshold_structured(
                img.dimensions,
                GridPoints::Image(img),
                &img.point_data,
                &img.cell_data,
                CellType::Voxel,
            )?)),
            DataSet::RectilinearGrid(rg) => {
                Ok(DataSet::UnstructuredGrid(self.threshold_structured(
                    rg.dimensions(),
                    GridPoints::Rectilinear(rg),
                    &rg.point_data,
                    &rg.cell_data,
                    CellType::Voxel,
                )?))
            }
            DataSet::StructuredGrid(sg) => {
                Ok(DataSet::UnstructuredGrid(self.threshold_structured(
                    sg.dimensions,
                    GridPoints::from_structured(&sg.points, sg.dimensions),
                    &sg.point_data,
                    &sg.cell_data,
                    CellType::Hexahedron,
                )?))
            }
            DataSet::UnstructuredGrid(ug) => {
                Ok(DataSet::UnstructuredGrid(self.threshold_unstructured(ug)?))
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

impl ThresholdFilter {
    fn resolve_array<'a>(
        &self,
        point_data: &'a FieldData,
        cell_data: &'a FieldData,
    ) -> Result<&'a DataArray, PipelineError> {
        let fd = match self.association {
            AttributeAssociation::Point => point_data,
            AttributeAssociation::Cell => cell_data,
            AttributeAssociation::Field => {
                return Err(PipelineError::ExecutionFailed(
                    "ThresholdFilter requires Point or Cell association".into(),
                ))
            }
        };
        if self.field_name.is_empty() {
            fd.active_scalars_array().ok_or_else(|| {
                PipelineError::ExecutionFailed("No active scalar array found for ThresholdFilter".into())
            })
        } else {
            fd.get(&self.field_name).ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "Array '{}' not found for ThresholdFilter",
                    self.field_name
                ))
            })
        }
    }

    #[inline]
    fn in_range(&self, v: f64) -> bool {
        v >= self.min && v <= self.max
    }

    #[inline]
    fn cell_passes(&self, arr: &DataArray, cell_idx: usize, cell_pts: &[u32]) -> bool {
        let matches = match self.association {
            AttributeAssociation::Cell => self.in_range(arr.get_magnitude(cell_idx)),
            AttributeAssociation::Point => {
                if cell_pts.is_empty() {
                    false
                } else if self.all_scalars {
                    cell_pts
                        .iter()
                        .all(|&vi| self.in_range(arr.get_magnitude(vi as usize)))
                } else {
                    cell_pts
                        .iter()
                        .any(|&vi| self.in_range(arr.get_magnitude(vi as usize)))
                }
            }
            AttributeAssociation::Field => false,
        };
        if self.invert {
            !matches
        } else {
            matches
        }
    }

    fn threshold_polydata(&self, pd: &PolyData) -> Result<PolyData, PipelineError> {
        let arr = self.resolve_array(&pd.point_data, &pd.cell_data)?;
        let mut verts = Vec::new();
        let mut lines = Vec::new();
        let mut polys = Vec::new();
        let mut strips = Vec::new();
        let mut kept_cell_indices = Vec::new();
        let mut cell_idx = 0usize;

        for &v in &pd.verts {
            if self.cell_passes(arr, cell_idx, &[v]) {
                verts.push(v);
                kept_cell_indices.push(cell_idx);
            }
            cell_idx += 1;
        }
        for line in &pd.lines {
            if self.cell_passes(arr, cell_idx, line) {
                lines.push(line.clone());
                kept_cell_indices.push(cell_idx);
            }
            cell_idx += 1;
        }
        for poly in &pd.polys {
            if self.cell_passes(arr, cell_idx, poly) {
                polys.push(poly.clone());
                kept_cell_indices.push(cell_idx);
            }
            cell_idx += 1;
        }
        for strip in &pd.strips {
            if self.cell_passes(arr, cell_idx, strip) {
                strips.push(strip.clone());
                kept_cell_indices.push(cell_idx);
            }
            cell_idx += 1;
        }

        let cell_data = if pd.cell_data.is_empty() {
            FieldData::new()
        } else {
            pd.cell_data.gather(&kept_cell_indices)
        };

        Ok(compact_polydata_points(PolyData {
            points: pd.points.clone(),
            verts,
            lines,
            polys,
            strips,
            point_data: pd.point_data.clone(),
            cell_data,
            field_data: pd.field_data.clone(),
        }))
    }

    fn threshold_structured(
        &self,
        dims: [usize; 3],
        points: GridPoints<'_>,
        point_data: &FieldData,
        cell_data: &FieldData,
        cell_type: CellType,
    ) -> Result<UnstructuredGrid, PipelineError> {
        let arr = self.resolve_array(point_data, cell_data)?;
        let [nx, ny, nz] = dims;
        if nx < 2 || ny < 2 || nz < 2 || points.is_empty() {
            return Ok(UnstructuredGrid::new());
        }
        let cx = nx - 1;
        let cy = ny - 1;
        let cz = nz - 1;
        let total_cells = cx * cy * cz;

        let kept: Vec<(usize, [u32; 8])> = (0..total_cells)
            .into_par_iter()
            .filter_map(|cell_idx| {
                let k = cell_idx / (cx * cy);
                let rem = cell_idx % (cx * cy);
                let j = rem / cx;
                let i = rem % cx;
                let idx = |ii: usize, jj: usize, kk: usize| (ii + nx * (jj + ny * kk)) as u32;
                let c = match cell_type {
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
                };
                self.cell_passes(arr, cell_idx, &c)
                    .then_some((cell_idx, c))
            })
            .collect();

        if kept.is_empty() {
            return Ok(UnstructuredGrid::new());
        }

        let n_pts = nx * ny * nz;
        let mut remap = vec![u32::MAX; n_pts];
        let mut new_points = Vec::new();
        let mut old_indices = Vec::new();
        let mut ug = UnstructuredGrid::new();
        let mut kept_ids = Vec::with_capacity(kept.len());

        for (cidx, mut corners) in kept {
            for v in &mut corners {
                let orig = *v as usize;
                let mapped = remap[orig];
                if mapped == u32::MAX {
                    let id = new_points.len() as u32;
                    remap[orig] = id;
                    new_points.push(points.at_linear(orig));
                    old_indices.push(orig);
                    *v = id;
                } else {
                    *v = mapped;
                }
            }
            ug.push_cell(cell_type, &corners);
            kept_ids.push(cidx);
        }

        ug.points = new_points;
        ug.point_data = point_data.gather(&old_indices);
        if !cell_data.is_empty() {
            ug.cell_data = cell_data.gather(&kept_ids);
        }
        Ok(ug)
    }

    fn threshold_unstructured(&self, in_ug: &UnstructuredGrid) -> Result<UnstructuredGrid, PipelineError> {
        let arr = self.resolve_array(&in_ug.point_data, &in_ug.cell_data)?;
        let n_cells = in_ug.num_cells();

        let kept: Vec<usize> = (0..n_cells)
            .into_par_iter()
            .filter(|&cell_idx| {
                let pts = in_ug.cell_points(cell_idx);
                self.cell_passes(arr, cell_idx, pts)
            })
            .collect();

        let mut ug = UnstructuredGrid::new();
        ug.points = in_ug.points.clone();
        ug.point_data = in_ug.point_data.clone();
        for &cidx in &kept {
            ug.push_cell(in_ug.cell_types[cidx], in_ug.cell_points(cidx));
        }
        if !in_ug.cell_data.is_empty() {
            ug.cell_data = in_ug.cell_data.gather(&kept);
        }
        Ok(compact_unstructured_points(ug))
    }
}
