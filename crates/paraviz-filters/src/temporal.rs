//! Temporal statistics and interpolation filters (`TemporalStatisticsFilter`, `interpolate_datasets`).
//!
//! Operates on multi-timestep [`paraviz_core::MultiBlockDataSet`] collections (such as those loaded from `.pvd`
//! time series) or slices of datasets.

use paraviz_core::{ArrayData, DataArray, DataSet, Filter, MultiBlockDataSet, PipelineError};
use serde::{Deserialize, Serialize};

/// Generates synthetic time steps (`[start_time .. end_time]`) over a static dataset,
/// producing a [`MultiBlockDataSet`] with `num_steps` blocks (`step_0`, `step_1`, ...).
///
/// Mirrors ParaView's requested `AddTimeSteps` filter (GitLab issue `#20241`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddTimeStepsFilter {
    pub num_steps: usize,
    pub start_time: f64,
    pub end_time: f64,
}

impl Default for AddTimeStepsFilter {
    fn default() -> Self {
        Self {
            num_steps: 10,
            start_time: 0.0,
            end_time: 1.0,
        }
    }
}

impl AddTimeStepsFilter {
    pub fn new(num_steps: usize, start_time: f64, end_time: f64) -> Self {
        Self {
            num_steps: num_steps.max(1),
            start_time,
            end_time,
        }
    }

    /// Returns the generated time step values.
    pub fn time_values(&self) -> Vec<f64> {
        let n = self.num_steps.max(1);
        if n == 1 {
            return vec![self.start_time];
        }
        (0..n)
            .map(|i| {
                let u = i as f64 / (n - 1) as f64;
                self.start_time + u * (self.end_time - self.start_time)
            })
            .collect()
    }
}

impl Filter for AddTimeStepsFilter {
    fn name(&self) -> &str {
        "AddTimeStepsFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let times = self.time_values();
        let mut mb = MultiBlockDataSet::new();
        for (idx, t) in times.into_iter().enumerate() {
            let mut step_ds = input.clone();
            if let Some(fd) = step_ds.field_data_mut() {
                fd.add_array(DataArray::new(
                    "TimeValue",
                    1,
                    ArrayData::F64(vec![t]),
                ));
            }
            mb.add_block(format!("step_{idx}"), step_ds);
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

/// Computes point-wise `min`, `max`, `average`, and `stddev` across all timesteps of a
/// [`paraviz_core::MultiBlockDataSet`] time series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalStatisticsFilter {
    pub compute_average: bool,
    pub compute_minimum: bool,
    pub compute_maximum: bool,
    pub compute_stddev: bool,
}

impl Default for TemporalStatisticsFilter {
    fn default() -> Self {
        Self {
            compute_average: true,
            compute_minimum: true,
            compute_maximum: true,
            compute_stddev: true,
        }
    }
}

impl Filter for TemporalStatisticsFilter {
    fn name(&self) -> &str {
        "TemporalStatisticsFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let DataSet::MultiBlock(mb) = input else {
            return Ok(input.clone());
        };
        if mb.blocks.is_empty() {
            return Err(PipelineError::ExecutionFailed(
                "TemporalStatisticsFilter received an empty MultiBlockDataSet".into(),
            ));
        }
        let steps: Vec<&DataSet> = mb.blocks.iter().map(|(_, d)| d).collect();
        self.compute_over_steps(&steps)
    }
}

impl TemporalStatisticsFilter {
    /// Computes temporal statistics over an explicit slice of same-topology datasets.
    pub fn compute_over_steps(&self, steps: &[&DataSet]) -> Result<DataSet, PipelineError> {
        let Some(first) = steps.first() else {
            return Err(PipelineError::ExecutionFailed("No time steps provided".into()));
        };
        let mut out = (*first).clone();
        let point_count = first.num_points();
        let fields: Vec<_> = steps.iter()
            .filter(|step| step.num_points() == point_count)
            .filter_map(|step| step.point_data())
            .collect();
        let mut seen = std::collections::HashSet::new();
        let mut new_arrays = Vec::new();
        for array in fields.iter().flat_map(|field| field.arrays()) {
            let components = array.num_components;
            if array.num_tuples() != point_count || !seen.insert(&array.name) {
                continue;
            }
            let len = array.data.len();
            let mut stats = vec![RunningStatistics::default(); len];
            for other in fields.iter().filter_map(|field| field.get(&array.name)) {
                if other.num_components != components || other.data.len() != len {
                    continue;
                }
                for (i, stat) in stats.iter_mut().enumerate() {
                    stat.add(other.data.get_f64(i));
                }
            }
            let outputs: [(bool, &str, StatisticValue); 4] = [
                (self.compute_average, "average", |stat| stat.mean),
                (self.compute_minimum, "minimum", |stat| stat.min),
                (self.compute_maximum, "maximum", |stat| stat.max),
                (self.compute_stddev, "stddev", |stat| (stat.m2 / stat.count as f64).max(0.0).sqrt()),
            ];
            for (enabled, suffix, value) in outputs {
                if enabled {
                    let values = stats.iter().map(|stat| {
                        if stat.count == 0 { f32::NAN } else { value(stat) as f32 }
                    }).collect();
                    new_arrays.push(DataArray::new(
                        format!("{}_{suffix}", array.name),
                        components,
                        ArrayData::F32(values),
                    ));
                }
            }
        }

        if let Some(out_pd) = out.point_data_mut() {
            for a in new_arrays {
                out_pd.add_array(a);
            }
        }
        Ok(out)
    }
}

type StatisticValue = fn(&RunningStatistics) -> f64;

#[derive(Clone)]
struct RunningStatistics {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Default for RunningStatistics {
    fn default() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY }
    }
}

impl RunningStatistics {
    fn add(&mut self, value: f64) {
        if value.is_finite() {
            self.count += 1;
            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            self.m2 += delta * (value - self.mean);
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
    }
}

/// Linearly interpolates point coordinates and point/cell arrays between two same-topology
/// datasets `a` (at `t = 0.0`) and `b` (at `t = 1.0`).
pub fn interpolate_datasets(a: &DataSet, b: &DataSet, t: f64) -> Result<DataSet, PipelineError> {
    let t = t.clamp(0.0, 1.0);
    let tf = t as f32;
    let mut out = a.clone();

    match (&mut out, b) {
        (DataSet::PolyData(pa), DataSet::PolyData(pb)) if pa.points.len() == pb.points.len() => {
            for (p, q) in pa.points.iter_mut().zip(&pb.points) {
                p[0] += (q[0] - p[0]) * tf;
                p[1] += (q[1] - p[1]) * tf;
                p[2] += (q[2] - p[2]) * tf;
            }
        }
        (DataSet::UnstructuredGrid(ua), DataSet::UnstructuredGrid(ub))
            if ua.points.len() == ub.points.len() =>
        {
            for (p, q) in ua.points.iter_mut().zip(&ub.points) {
                p[0] += (q[0] - p[0]) * tf;
                p[1] += (q[1] - p[1]) * tf;
                p[2] += (q[2] - p[2]) * tf;
            }
        }
        _ => {}
    }

    if let (Some(out_pd), Some(b_pd)) = (out.point_data_mut(), b.point_data()) {
        for arr in out_pd.arrays_mut() {
            if let Some(other) = b_pd.get(&arr.name)
                && other.data.len() == arr.data.len() {
                    let vals: Vec<f32> = (0..arr.data.len())
                        .map(|i| {
                            let va = arr.data.get_f64(i);
                            let vb = other.data.get_f64(i);
                            (va + (vb - va) * t) as f32
                        })
                        .collect();
                    arr.data = ArrayData::F32(vals);
                }
        }
    }

    Ok(out)
}
