//! Elevation scalar projection filter (`ElevationFilter`, equivalent to `vtkElevationFilter`).

use crate::util::GridPoints;
use glam::Vec3;
use paraviz_core::{DataArray, DataSet, Filter, MultiBlockDataSet, PipelineError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Computes a normalized `[0.0, 1.0]` scalar projection of each point along the line segment
/// `low_point -> high_point` and stores it in `point_data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElevationFilter {
    pub low_point: [f32; 3],
    pub high_point: [f32; 3],
    pub scalar_name: String,
}

impl Default for ElevationFilter {
    fn default() -> Self {
        Self {
            low_point: [0.0, 0.0, 0.0],
            high_point: [0.0, 0.0, 1.0],
            scalar_name: "Elevation".into(),
        }
    }
}

impl ElevationFilter {
    /// Creates an `ElevationFilter` projecting from `low_point` (0.0) to `high_point` (1.0).
    pub fn new(low_point: [f32; 3], high_point: [f32; 3]) -> Self {
        Self {
            low_point,
            high_point,
            ..Default::default()
        }
    }
}

impl Filter for ElevationFilter {
    fn name(&self) -> &str {
        "ElevationFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let low = Vec3::from_array(self.low_point);
        let high = Vec3::from_array(self.high_point);
        let diff = high - low;
        let len_sq = diff.length_squared();
        if len_sq <= 1e-20 {
            return Err(PipelineError::ExecutionFailed(
                "ElevationFilter low_point and high_point must be distinct".into(),
            ));
        }

        let pts = GridPoints::from_dataset(input)?;
        let inv_len_sq = 1.0 / len_sq;
        let scalars: Vec<f32> = (0..pts.len())
            .into_par_iter()
            .map(|idx| {
                let p = pts.at_linear(idx);
                let t = (Vec3::from_array(p) - low).dot(diff) * inv_len_sq;
                t.clamp(0.0, 1.0)
            })
            .collect();

        let name = if self.scalar_name.is_empty() {
            "Elevation".to_string()
        } else {
            self.scalar_name.clone()
        };

        let mut out = input.clone();
        out.point_data_mut()
            .ok_or_else(|| PipelineError::ExecutionFailed("Dataset has no point_data".into()))?
            .set_scalars(DataArray::scalars_f32(name, scalars));
        Ok(out)
    }
}
