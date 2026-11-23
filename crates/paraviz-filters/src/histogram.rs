//! Histogram and summary statistics utility (`compute_histogram` / `HistogramResult`).

use paraviz_core::{AttributeAssociation, DataArray, DataSet, FieldData, PipelineError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Histogram counts, bin edges, and summary statistics for a dataset attribute array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramResult {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<usize>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
}

fn collect_arrays<'a>(
    ds: &'a DataSet,
    array_name: &str,
    association: AttributeAssociation,
    out: &mut Vec<&'a DataArray>,
) -> Result<(), PipelineError> {
    if let DataSet::MultiBlock(mb) = ds {
        for (_, block) in &mb.blocks {
            collect_arrays(block, array_name, association, out)?;
        }
        return Ok(());
    }
    let fd: &FieldData = match association {
        AttributeAssociation::Point => ds.point_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Dataset has no point_data".into())
        })?,
        AttributeAssociation::Cell => ds.cell_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Dataset has no cell_data".into())
        })?,
        AttributeAssociation::Field => ds.field_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Dataset has no field_data".into())
        })?,
    };
    let arr = if array_name.is_empty() {
        fd.active_scalars_array().ok_or_else(|| {
            PipelineError::ExecutionFailed("No active scalar array found for histogram".into())
        })?
    } else {
        fd.get(array_name).ok_or_else(|| {
            PipelineError::ExecutionFailed(format!(
                "Array '{array_name}' not found for histogram"
            ))
        })?
    };
    out.push(arr);
    Ok(())
}

/// Computes an equal-width histogram and summary statistics (`min`, `max`, `mean`, population
/// `std_dev`) over a dataset's `Point`, `Cell`, or `Field` attribute array.
///
/// When `component` is `None`, tuple magnitudes are used (exact scalar for 1-component arrays).
pub fn compute_histogram(
    dataset: &DataSet,
    array_name: &str,
    association: AttributeAssociation,
    component: Option<usize>,
    bins: usize,
) -> Result<HistogramResult, PipelineError> {
    if bins == 0 {
        return Err(PipelineError::ExecutionFailed(
            "Histogram bins must be >= 1".into(),
        ));
    }

    let mut arrays = Vec::new();
    collect_arrays(dataset, array_name, association, &mut arrays)?;
    if arrays.is_empty() {
        return Err(PipelineError::ExecutionFailed(
            "No arrays available to compute histogram".into(),
        ));
    }

    let mut values = Vec::new();
    for arr in arrays {
        if let Some(c) = component
            && c >= arr.num_components
        {
            return Err(PipelineError::ExecutionFailed(format!(
                "Component {c} out of range for array '{}' with {} component(s)",
                arr.name, arr.num_components
            )));
        }
        let n = arr.num_tuples();
        let extracted: Vec<f64> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let v = match component {
                    Some(c) => arr.data.get_f64(i * arr.num_components + c),
                    None => arr.get_magnitude(i),
                };
                v.is_finite().then_some(v)
            })
            .collect();
        values.extend(extracted);
    }

    if values.is_empty() {
        return Err(PipelineError::ExecutionFailed(
            "Array contains no finite values for histogram".into(),
        ));
    }

    let n = values.len();
    let (min, max, sum) = values
        .par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY, 0.0f64),
            |(mn, mx, s), &v| (mn.min(v), mx.max(v), s + v),
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY, 0.0f64),
            |(mn_a, mx_a, s_a), (mn_b, mx_b, s_b)| (mn_a.min(mn_b), mx_a.max(mx_b), s_a + s_b),
        );

    let mean = sum / n as f64;
    let sq_sum: f64 = values
        .par_iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum();
    let std_dev = (sq_sum / n as f64).sqrt();

    let span = max - min;
    let bin_edges: Vec<f64> = (0..=bins)
        .map(|i| {
            if i == bins {
                max
            } else {
                min + span * (i as f64 / bins as f64)
            }
        })
        .collect();

    let counts = if span <= 0.0 {
        let mut c = vec![0usize; bins];
        c[0] = n;
        c
    } else {
        let inv_width = bins as f64 / span;
        values
            .par_iter()
            .fold(
                || vec![0usize; bins],
                |mut local, &v| {
                    let b = (((v - min) * inv_width).floor() as usize).min(bins - 1);
                    local[b] += 1;
                    local
                },
            )
            .reduce(
                || vec![0usize; bins],
                |mut a, b| {
                    for (slot, cnt) in a.iter_mut().zip(b) {
                        *slot += cnt;
                    }
                    a
                },
            )
    };

    Ok(HistogramResult {
        bin_edges,
        counts,
        min,
        max,
        mean,
        std_dev,
    })
}
