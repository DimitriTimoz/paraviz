//! Dataset merging filter (`AppendDatasetsFilter` / `append_datasets`,
//! equivalent to `vtkAppendFilter` and `vtkAppendPolyData`).

use crate::util::{dataset_to_unstructured, merge_polydata_refs, merge_unstructured_refs};
use paraviz_core::{DataSet, Filter, PipelineError, PolyData, UnstructuredGrid};
use serde::{Deserialize, Serialize};

/// Combines multiple `PolyData` meshes into a merged `PolyData`, or mixed/volumetric
/// datasets into a merged `UnstructuredGrid`, preserving common `point_data` and `cell_data`.
///
/// When used as a single-input [`Filter`] on a [`DataSet::MultiBlock`], all blocks are appended;
/// extra inputs can also be supplied via [`AppendDatasetsFilter::additional_inputs`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppendDatasetsFilter {
    pub additional_inputs: Vec<DataSet>,
}

impl AppendDatasetsFilter {
    /// Creates an `AppendDatasetsFilter` with no additional inputs (ideal for `MultiBlock` inputs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an `AppendDatasetsFilter` that appends `additional_inputs` onto the pipeline input.
    pub fn with_inputs(additional_inputs: Vec<DataSet>) -> Self {
        Self { additional_inputs }
    }
}

impl Filter for AppendDatasetsFilter {
    fn name(&self) -> &str {
        "AppendDatasetsFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let mut refs = Vec::with_capacity(1 + self.additional_inputs.len());
        refs.push(input);
        for extra in &self.additional_inputs {
            refs.push(extra);
        }
        append_datasets(&refs)
    }
}

fn flatten_datasets<'a>(inputs: &[&'a DataSet], out: &mut Vec<&'a DataSet>) {
    for &ds in inputs {
        match ds {
            DataSet::MultiBlock(mb) => {
                for (_, block) in &mb.blocks {
                    flatten_datasets(&[block], out);
                }
            }
            other => out.push(other),
        }
    }
}

/// Combines multiple [`DataSet`]s into a single [`DataSet`].
///
/// If all inputs are [`PolyData`], returns a merged [`DataSet::PolyData`].
/// Otherwise converts all inputs to [`UnstructuredGrid`] and returns a merged
/// [`DataSet::UnstructuredGrid`]. Common `point_data` and `cell_data` arrays are preserved.
pub fn append_datasets(inputs: &[&DataSet]) -> Result<DataSet, PipelineError> {
    let mut flat = Vec::new();
    flatten_datasets(inputs, &mut flat);

    if flat.is_empty() {
        return Ok(DataSet::PolyData(PolyData::new()));
    }

    if flat.iter().all(|ds| matches!(ds, DataSet::PolyData(_))) {
        let parts: Vec<&PolyData> = flat
            .iter()
            .filter_map(|ds| ds.as_poly_data())
            .collect();
        return Ok(DataSet::PolyData(merge_polydata_refs(&parts)));
    }

    let grids: Vec<UnstructuredGrid> = flat
        .iter()
        .map(|ds| dataset_to_unstructured(ds))
        .collect::<Result<Vec<_>, _>>()?;
    let refs: Vec<&UnstructuredGrid> = grids.iter().collect();
    Ok(DataSet::UnstructuredGrid(merge_unstructured_refs(&refs)))
}
