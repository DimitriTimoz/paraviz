//! Geometric warping filters (`WarpByVectorFilter` and `WarpByScalarFilter`).

use crate::util::GridPoints;
use glam::Vec3;
use paraviz_core::{
    DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData, StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Displaces point coordinates along a 3D vector array (`p' = p + scale_factor * v`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarpByVectorFilter {
    pub vector_field: String,
    pub scale_factor: f32,
}

impl Default for WarpByVectorFilter {
    fn default() -> Self {
        Self {
            vector_field: String::new(),
            scale_factor: 1.0,
        }
    }
}

impl WarpByVectorFilter {
    pub fn new(vector_field: impl Into<String>, scale_factor: f32) -> Self {
        Self {
            vector_field: vector_field.into(),
            scale_factor,
        }
    }
}

impl Filter for WarpByVectorFilter {
    fn name(&self) -> &str {
        "WarpByVectorFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        warp_dataset(input, &|points, point_data| {
            let arr = if self.vector_field.is_empty() {
                point_data.active_vectors_array().ok_or_else(|| {
                    PipelineError::ExecutionFailed(
                        "No active vector array found for WarpByVectorFilter".into(),
                    )
                })?
            } else {
                point_data.get(&self.vector_field).ok_or_else(|| {
                    PipelineError::ExecutionFailed(format!(
                        "Vector array '{}' not found in point_data",
                        self.vector_field
                    ))
                })?
            };
            let s = self.scale_factor;
            Ok((0..points.len())
                .into_par_iter()
                .map(|i| {
                    let p = points.at_linear(i);
                    let v = arr.get_vec3(i);
                    [p[0] + v[0] * s, p[1] + v[1] * s, p[2] + v[2] * s]
                })
                .collect())
        })
    }
}

/// Displaces point coordinates along a scalar field times a normal vector
/// (`p' = p + scale_factor * scalar * normal`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarpByScalarFilter {
    pub scalar_field: String,
    pub normal: [f32; 3],
    pub use_normal: bool,
    pub scale_factor: f32,
}

impl Default for WarpByScalarFilter {
    fn default() -> Self {
        Self {
            scalar_field: String::new(),
            normal: [0.0, 0.0, 1.0],
            use_normal: true,
            scale_factor: 1.0,
        }
    }
}

impl WarpByScalarFilter {
    pub fn new(scalar_field: impl Into<String>, scale_factor: f32) -> Self {
        Self {
            scalar_field: scalar_field.into(),
            scale_factor,
            ..Default::default()
        }
    }
}

impl Filter for WarpByScalarFilter {
    fn name(&self) -> &str {
        "WarpByScalarFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let fallback_n = Vec3::from_array(self.normal).normalize_or(Vec3::Z);
        warp_dataset(input, &|points, point_data| {
            let arr = if self.scalar_field.is_empty() {
                point_data.active_scalars_array().ok_or_else(|| {
                    PipelineError::ExecutionFailed(
                        "No active scalar array found for WarpByScalarFilter".into(),
                    )
                })?
            } else {
                point_data.get(&self.scalar_field).ok_or_else(|| {
                    PipelineError::ExecutionFailed(format!(
                        "Scalar array '{}' not found in point_data",
                        self.scalar_field
                    ))
                })?
            };
            let normals_arr = if self.use_normal {
                point_data.active_normals_array()
            } else {
                None
            };
            let s = self.scale_factor;
            Ok((0..points.len())
                .into_par_iter()
                .map(|i| {
                    let p = points.at_linear(i);
                    let val = arr.get_magnitude(i) as f32;
                    let n = normals_arr
                        .map(|na| Vec3::from_array(na.get_vec3(i)).normalize_or(fallback_n))
                        .unwrap_or(fallback_n);
                    let disp = n * (val * s);
                    [p[0] + disp.x, p[1] + disp.y, p[2] + disp.z]
                })
                .collect())
        })
    }
}

fn warp_dataset<F>(input: &DataSet, warp_fn: &F) -> Result<DataSet, PipelineError>
where
    F: Fn(GridPoints<'_>, &paraviz_core::FieldData) -> Result<Vec<[f32; 3]>, PipelineError> + Sync,
{
    match input {
        DataSet::PolyData(pd) => {
            let new_pts = warp_fn(
                GridPoints::Explicit {
                    points: &pd.points,
                    nx: pd.points.len(),
                    ny: 1,
                },
                &pd.point_data,
            )?;
            let mut out = PolyData {
                points: new_pts,
                ..pd.clone()
            };
            if out.point_data.active_normals_array().is_some() && !out.polys.is_empty() {
                out.compute_normals();
            }
            Ok(DataSet::PolyData(out))
        }
        DataSet::ImageData(img) => {
            let new_pts = warp_fn(GridPoints::Image(img), &img.point_data)?;
            Ok(DataSet::StructuredGrid(StructuredGrid {
                dimensions: img.dimensions,
                points: new_pts,
                point_data: img.point_data.clone(),
                cell_data: img.cell_data.clone(),
                field_data: img.field_data.clone(),
            }))
        }
        DataSet::RectilinearGrid(rg) => {
            let new_pts = warp_fn(GridPoints::Rectilinear(rg), &rg.point_data)?;
            Ok(DataSet::StructuredGrid(StructuredGrid {
                dimensions: rg.dimensions(),
                points: new_pts,
                point_data: rg.point_data.clone(),
                cell_data: rg.cell_data.clone(),
                field_data: rg.field_data.clone(),
            }))
        }
        DataSet::StructuredGrid(sg) => {
            let new_pts = warp_fn(
                GridPoints::from_structured(&sg.points, sg.dimensions),
                &sg.point_data,
            )?;
            Ok(DataSet::StructuredGrid(StructuredGrid {
                points: new_pts,
                ..sg.clone()
            }))
        }
        DataSet::UnstructuredGrid(ug) => {
            let new_pts = warp_fn(
                GridPoints::Explicit {
                    points: &ug.points,
                    nx: ug.points.len(),
                    ny: 1,
                },
                &ug.point_data,
            )?;
            Ok(DataSet::UnstructuredGrid(UnstructuredGrid {
                points: new_pts,
                ..ug.clone()
            }))
        }
        DataSet::MultiBlock(mb) => {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), warp_dataset(block, warp_fn)?);
            }
            Ok(DataSet::MultiBlock(out))
        }
    }
}
