//! Extract Block filter (`ExtractBlockFilter`) for selecting blocks from a [`MultiBlockDataSet`].

use paraviz_core::{DataSet, Filter, MultiBlockDataSet, PipelineError};
use serde::{Deserialize, Serialize};

/// Extracts one or more blocks from a [`MultiBlockDataSet`] by index or name substring.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtractBlockFilter {
    /// Block indices to keep (empty = match all unless `name_filter` is set).
    pub block_indices: Vec<usize>,
    /// Optional case-insensitive block name substring filter.
    pub name_filter: String,
    /// When only 1 block matches, unwrap it from `MultiBlockDataSet` into a direct `DataSet`.
    pub unwrap_single: bool,
}

impl ExtractBlockFilter {
    pub fn by_index(index: usize) -> Self {
        Self {
            block_indices: vec![index],
            name_filter: String::new(),
            unwrap_single: true,
        }
    }

    pub fn by_name(substring: impl Into<String>) -> Self {
        Self {
            block_indices: Vec::new(),
            name_filter: substring.into(),
            unwrap_single: true,
        }
    }
}

impl Filter for ExtractBlockFilter {
    fn name(&self) -> &str {
        "ExtractBlockFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let DataSet::MultiBlock(mb) = input else {
            return Ok(input.clone());
        };
        let needle = self.name_filter.to_ascii_lowercase();
        let mut out = MultiBlockDataSet::new();
        for (idx, (name, block)) in mb.blocks.iter().enumerate() {
            let idx_ok = self.block_indices.is_empty() || self.block_indices.contains(&idx);
            let name_ok = needle.is_empty() || name.to_ascii_lowercase().contains(&needle);
            if idx_ok && name_ok {
                out.add_block(name.clone(), block.clone());
            }
        }
        if self.unwrap_single && out.blocks.len() == 1 {
            Ok(out.blocks.remove(0).1)
        } else {
            Ok(DataSet::MultiBlock(out))
        }
    }
}
