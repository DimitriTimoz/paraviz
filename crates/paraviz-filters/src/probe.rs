//! Attribute conversion and line-probing filters:
//! `CellDataToPointDataFilter`, `PointDataToCellDataFilter`, and `ProbeOverLineFilter`.

use crate::util::{interpolate_field_data_weighted, SpatialSampler};
use glam::Vec3;
use paraviz_core::{
    ArrayData, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError, PolyData,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Averages `cell_data` arrays onto incident points and stores them in `point_data`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CellDataToPointDataFilter {
    pub pass_cell_data: bool,
}

impl CellDataToPointDataFilter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for CellDataToPointDataFilter {
    fn name(&self) -> &str {
        "CellDataToPointDataFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let n_pts = input.num_points();
        let cdata = input.cell_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Dataset has no cell_data".into())
        })?;
        if cdata.is_empty() || n_pts == 0 {
            return Ok(input.clone());
        }

        // Build point -> incident cell indices
        let mut pt_cells: Vec<Vec<usize>> = vec![Vec::new(); n_pts];
        for (cell_idx, pts) in dataset_cell_point_lists(input).into_iter().enumerate() {
            for pi in pts {
                if pi < n_pts {
                    pt_cells[pi].push(cell_idx);
                }
            }
        }

        let mut out = input.clone();
        let out_pdata = out.point_data_mut().unwrap();
        for arr in cdata.arrays() {
            let nc = arr.num_components;
            let vals: Vec<f32> = (0..n_pts)
                .into_par_iter()
                .flat_map_iter(|pi| {
                    let cells = &pt_cells[pi];
                    let mut acc = vec![0.0f64; nc];
                    if !cells.is_empty() {
                        for &ci in cells {
                            let base = ci * nc;
                            for (c, slot) in acc.iter_mut().enumerate() {
                                *slot += arr.data.get_f64(base + c);
                            }
                        }
                        let inv = 1.0 / cells.len() as f64;
                        for v in &mut acc {
                            *v *= inv;
                        }
                    }
                    acc.into_iter().map(|v| v as f32)
                })
                .collect();
            out_pdata.add_array(DataArray::new(arr.name.clone(), nc, ArrayData::F32(vals)));
        }
        if out_pdata.active_scalars.is_none() {
            out_pdata.active_scalars = cdata.active_scalars.clone();
        }
        if out_pdata.active_vectors.is_none() {
            out_pdata.active_vectors = cdata.active_vectors.clone();
        }
        if !self.pass_cell_data {
            *out.cell_data_mut().unwrap() = FieldData::new();
        }
        Ok(out)
    }
}

/// Averages `point_data` arrays at each cell's vertices and stores them in `cell_data`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PointDataToCellDataFilter {
    pub pass_point_data: bool,
}

impl PointDataToCellDataFilter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for PointDataToCellDataFilter {
    fn name(&self) -> &str {
        "PointDataToCellDataFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let pdata = input.point_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Dataset has no point_data".into())
        })?;
        let cell_pts = dataset_cell_point_lists(input);
        if pdata.is_empty() || cell_pts.is_empty() {
            return Ok(input.clone());
        }

        let mut out = input.clone();
        let out_cdata = out.cell_data_mut().unwrap();
        for arr in pdata.arrays() {
            let nc = arr.num_components;
            let vals: Vec<f32> = cell_pts
                .par_iter()
                .flat_map_iter(|pts| {
                    let mut acc = vec![0.0f64; nc];
                    if !pts.is_empty() {
                        for &pi in pts {
                            let base = pi * nc;
                            for (c, slot) in acc.iter_mut().enumerate() {
                                *slot += arr.data.get_f64(base + c);
                            }
                        }
                        let inv = 1.0 / pts.len() as f64;
                        for v in &mut acc {
                            *v *= inv;
                        }
                    }
                    acc.into_iter().map(|v| v as f32)
                })
                .collect();
            out_cdata.add_array(DataArray::new(arr.name.clone(), nc, ArrayData::F32(vals)));
        }
        if out_cdata.active_scalars.is_none() {
            out_cdata.active_scalars = pdata.active_scalars.clone();
        }
        if out_cdata.active_vectors.is_none() {
            out_cdata.active_vectors = pdata.active_vectors.clone();
        }
        if !self.pass_point_data {
            *out.point_data_mut().unwrap() = FieldData::new();
        }
        Ok(out)
    }
}

fn dataset_cell_point_lists(ds: &DataSet) -> Vec<Vec<usize>> {
    match ds {
        DataSet::PolyData(pd) => {
            let mut out = Vec::with_capacity(pd.num_cells());
            for &v in &pd.verts {
                out.push(vec![v as usize]);
            }
            for line in &pd.lines {
                out.push(line.iter().map(|&v| v as usize).collect());
            }
            for poly in &pd.polys {
                out.push(poly.iter().map(|&v| v as usize).collect());
            }
            for strip in &pd.strips {
                out.push(strip.iter().map(|&v| v as usize).collect());
            }
            out
        }
        DataSet::ImageData(img) => structured_cell_point_lists(img.dimensions),
        DataSet::RectilinearGrid(rg) => structured_cell_point_lists(rg.dimensions()),
        DataSet::StructuredGrid(sg) => structured_cell_point_lists(sg.dimensions),
        DataSet::UnstructuredGrid(ug) => (0..ug.num_cells())
            .map(|c| ug.cell_points(c).iter().map(|&v| v as usize).collect())
            .collect(),
        DataSet::MultiBlock(_) => Vec::new(),
    }
}

fn structured_cell_point_lists(dims: [usize; 3]) -> Vec<Vec<usize>> {
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return Vec::new();
    }
    let cx = nx.saturating_sub(1).max(1);
    let cy = ny.saturating_sub(1).max(1);
    let cz = nz.saturating_sub(1).max(1);
    let mut out = Vec::with_capacity(cx * cy * cz);
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut pts = Vec::with_capacity(8);
                for dk in 0..=usize::from(nz > 1) {
                    for dj in 0..=usize::from(ny > 1) {
                        for di in 0..=usize::from(nx > 1) {
                            pts.push(idx(i + di, j + dj, k + dk));
                        }
                    }
                }
                out.push(pts);
            }
        }
    }
    out
}

/// Samples a volumetric dataset along a line segment `[point1, point2]` into a
/// [`PolyData`] polyline with `ArcLength`, `vtkValidPointMask`, and all interpolated arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeOverLineFilter {
    pub point1: [f32; 3],
    pub point2: [f32; 3],
    pub resolution: usize,
}

impl Default for ProbeOverLineFilter {
    fn default() -> Self {
        Self {
            point1: [-1.0, 0.0, 0.0],
            point2: [1.0, 0.0, 0.0],
            resolution: 100,
        }
    }
}

impl ProbeOverLineFilter {
    pub fn new(point1: [f32; 3], point2: [f32; 3], resolution: usize) -> Self {
        Self {
            point1,
            point2,
            resolution,
        }
    }
}

impl Filter for ProbeOverLineFilter {
    fn name(&self) -> &str {
        "ProbeOverLineFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let n = self.resolution.max(1);
        let p1 = Vec3::from_array(self.point1);
        let p2 = Vec3::from_array(self.point2);
        let total_len = (p2 - p1).length();

        let points: Vec<[f32; 3]> = (0..=n)
            .map(|i| p1.lerp(p2, i as f32 / n as f32).to_array())
            .collect();
        let arc_length: Vec<f32> = (0..=n)
            .map(|i| total_len * (i as f32 / n as f32))
            .collect();
        let line: Vec<u32> = (0..=n as u32).collect();

        let sampler = SpatialSampler::new(input)?;
        let weights: Vec<Option<Vec<(usize, f64)>>> = points
            .par_iter()
            .map(|&p| sampler.sample_weights(Vec3::from_array(p)))
            .collect();

        let valid_mask: Vec<u8> = weights
            .iter()
            .map(|w| if w.is_some() { 1 } else { 0 })
            .collect();

        let mut pdata = interpolate_field_data_weighted(sampler.field_data(), &weights);
        pdata.add_array(DataArray::scalars_f32("ArcLength", arc_length));
        pdata.add_array(DataArray::new(
            "vtkValidPointMask",
            1,
            ArrayData::U8(valid_mask),
        ));

        Ok(DataSet::PolyData(PolyData {
            points,
            lines: vec![line],
            point_data: pdata,
            ..Default::default()
        }))
    }
}


/// Interpolates a point-data component at a fixed world-space position.
/// For multi-block data, uses the first block with a finite sample.
pub fn sample_point_component(
    dataset: &DataSet,
    position: [f32; 3],
    field: &str,
    component: usize,
) -> Option<f64> {
    if let DataSet::MultiBlock(blocks) = dataset {
        return blocks.blocks.iter().find_map(|(_, block)| {
            sample_point_component(block, position, field, component)
        });
    }
    let array = dataset.point_data()?.get(field)?;
    if component >= array.num_components || array.num_tuples() != dataset.num_points() {
        return None;
    }
    let sampler = SpatialSampler::new(dataset).ok()?;
    let weights = sampler.sample_weights(Vec3::from_array(position))?;
    let mut value = 0.0;
    for (index, weight) in weights {
        if weight != 0.0 {
            value += array.data.get_f64(index * array.num_components + component) * weight;
        }
    }
    value.is_finite().then_some(value)
}
