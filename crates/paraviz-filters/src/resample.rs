//! Spatial resampling filters (`ResampleToImageFilter`, `ResampleWithDatasetFilter`).
//!
//! - `ResampleToImageFilter`: samples any volumetric dataset onto a regular [`ImageData`] grid,
//!   enabling GPU Volume Raymarching and GPU Marching Cubes on unstructured CFD/FEA grids.
//! - `ResampleWithDatasetFilter`: multi-input filter that samples `inputs[0]` (source data)
//!   at every point location of `inputs[1]` (destination geometry).

use crate::util::SpatialSampler;
use glam::Vec3;
use paraviz_core::{
    ArrayData, DataArray, DataSet, Filter, ImageData, MultiInputFilter, PipelineError,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Resamples an arbitrary volumetric dataset onto a uniform [`ImageData`] grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResampleToImageFilter {
    pub dimensions: [usize; 3],
    /// Optional explicit bounds `[xmin, xmax, ymin, ymax, zmin, zmax]`; `None` uses input bounds.
    pub custom_bounds: Option<[f64; 6]>,
}

impl Default for ResampleToImageFilter {
    fn default() -> Self {
        Self {
            dimensions: [64, 64, 64],
            custom_bounds: None,
        }
    }
}

impl ResampleToImageFilter {
    pub fn new(dimensions: [usize; 3]) -> Self {
        Self {
            dimensions: [
                dimensions[0].max(2),
                dimensions[1].max(2),
                dimensions[2].max(2),
            ],
            custom_bounds: None,
        }
    }
}

impl Filter for ResampleToImageFilter {
    fn name(&self) -> &str {
        "ResampleToImageFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let sampler = SpatialSampler::new(input)?;
        let bounds = input.bounds();
        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.custom_bounds.unwrap_or([
            bounds.min.x as f64,
            bounds.max.x as f64,
            bounds.min.y as f64,
            bounds.max.y as f64,
            bounds.min.z as f64,
            bounds.max.z as f64,
        ]);
        let [nx, ny, nz] = [
            self.dimensions[0].max(2),
            self.dimensions[1].max(2),
            self.dimensions[2].max(2),
        ];
        let origin = [xmin, ymin, zmin];
        let spacing = [
            ((xmax - xmin) / (nx - 1) as f64).max(1e-9),
            ((ymax - ymin) / (ny - 1) as f64).max(1e-9),
            ((zmax - zmin) / (nz - 1) as f64).max(1e-9),
        ];

        let mut img = ImageData::new([nx, ny, nz], origin, spacing);
        let n_total = nx * ny * nz;
        let fd = sampler.field_data();

        // Precompute valid mask + interpolated arrays in parallel
        let samples: Vec<Option<Vec<Vec<f64>>>> = (0..n_total)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let rem = idx / nx;
                let j = rem % ny;
                let k = rem / ny;
                let p = Vec3::from_array(img.point_coords(i, j, k));
                sampler.sample_all_fields(p)
            })
            .collect();

        let mut valid_mask = Vec::with_capacity(n_total);
        let mut arrays_out: Vec<(String, usize, Vec<f32>)> = fd
            .arrays()
            .iter()
            .map(|a| (a.name.clone(), a.num_components, vec![0.0f32; n_total * a.num_components]))
            .collect();

        for (idx, sample) in samples.into_iter().enumerate() {
            if let Some(vals) = sample {
                valid_mask.push(1u8);
                for (ai, tuple) in vals.into_iter().enumerate() {
                    let (_, nc, buf) = &mut arrays_out[ai];
                    for (c, v) in tuple.into_iter().enumerate().take(*nc) {
                        buf[idx * (*nc) + c] = v as f32;
                    }
                }
            } else {
                valid_mask.push(0u8);
            }
        }

        for (name, nc, buf) in arrays_out {
            let is_active = fd.active_scalars.as_deref() == Some(name.as_str());
            let arr = DataArray::new(name, nc, ArrayData::F32(buf));
            if is_active {
                img.point_data.set_scalars(arr);
            } else {
                img.point_data.add_array(arr);
            }
        }
        img.point_data
            .add_array(DataArray::new("vtkValidPointMask", 1, ArrayData::U8(valid_mask)));

        Ok(DataSet::ImageData(img))
    }
}

/// Multi-input filter: samples `inputs[0]` (source volumetric dataset) onto the point locations
/// of `inputs[1]` (destination geometry).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResampleWithDatasetFilter;

impl MultiInputFilter for ResampleWithDatasetFilter {
    fn name(&self) -> &str {
        "ResampleWithDatasetFilter"
    }

    fn apply_multi(&self, inputs: &[&DataSet]) -> Result<DataSet, PipelineError> {
        if inputs.len() < 2 {
            return Err(PipelineError::ExecutionFailed(
                "ResampleWithDatasetFilter requires 2 inputs: [source_data, destination_mesh]"
                    .into(),
            ));
        }
        let sampler = SpatialSampler::new(inputs[0])?;
        let dest = inputs[1];
        let mut out = dest.clone();

        let dest_pts: Vec<[f32; 3]> = match dest {
            DataSet::PolyData(pd) => pd.points.clone(),
            DataSet::UnstructuredGrid(ug) => ug.points.clone(),
            DataSet::StructuredGrid(sg) => sg.points.clone(),
            DataSet::ImageData(img) => img.all_points(),
            DataSet::RectilinearGrid(rg) => rg.all_points(),
            DataSet::MultiBlock(_) => {
                return Err(PipelineError::ExecutionFailed(
                    "ResampleWithDataset destination cannot be MultiBlock".into(),
                ));
            }
        };

        let n = dest_pts.len();
        let fd = sampler.field_data();
        let samples: Vec<Option<Vec<Vec<f64>>>> = dest_pts
            .par_iter()
            .map(|&p| sampler.sample_all_fields(Vec3::from_array(p)))
            .collect();

        let mut valid_mask = Vec::with_capacity(n);
        let mut arrays_out: Vec<(String, usize, Vec<f32>)> = fd
            .arrays()
            .iter()
            .map(|a| (a.name.clone(), a.num_components, vec![0.0f32; n * a.num_components]))
            .collect();

        for (idx, sample) in samples.into_iter().enumerate() {
            if let Some(vals) = sample {
                valid_mask.push(1u8);
                for (ai, tuple) in vals.into_iter().enumerate() {
                    let (_, nc, buf) = &mut arrays_out[ai];
                    for (c, v) in tuple.into_iter().enumerate().take(*nc) {
                        buf[idx * (*nc) + c] = v as f32;
                    }
                }
            } else {
                valid_mask.push(0u8);
            }
        }

        if let Some(out_pd) = out.point_data_mut() {
            for (name, nc, buf) in arrays_out {
                let is_active = fd.active_scalars.as_deref() == Some(name.as_str());
                let arr = DataArray::new(name, nc, ArrayData::F32(buf));
                if is_active {
                    out_pd.set_scalars(arr);
                } else {
                    out_pd.add_array(arr);
                }
            }
            out_pd.add_array(DataArray::new("vtkValidPointMask", 1, ArrayData::U8(valid_mask)));
        }

        Ok(out)
    }
}
