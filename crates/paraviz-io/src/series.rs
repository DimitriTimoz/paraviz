//! ParaView JSON File Series (`.series`) reader.
//!
//! Supports `.vtk.series`, `.vtu.series`, `.vtp.series`, `.vti.series`, `.vtr.series`,
//! `.vts.series`, `.vtm.series`, `.stl.series`, `.ply.series`, etc.:
//! ```json
//! {
//!   "file-series-version": "1.0",
//!   "files": [
//!     { "name": "step_000.vtu", "time": 0.0 },
//!     { "name": "step_001.vtu", "time": 0.5 }
//!   ]
//! }
//! ```

use crate::error::{IoError, Result};
use crate::limits::resolve_safe_path;
use paraviz_core::DataSet;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct SeriesFile {
    #[serde(default)]
    files: Vec<SeriesEntry>,
}

#[derive(Debug, Deserialize)]
struct SeriesEntry {
    name: String,
    #[serde(default)]
    time: Option<f64>,
}

/// Parses a ParaView `.series` JSON file into a sorted list of `(time, resolved_path)` pairs.
pub fn parse_series_index(path: impl AsRef<Path>) -> Result<Vec<(f64, PathBuf)>> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let text = fs::read_to_string(path)?;
    let parsed: SeriesFile = serde_json::from_str(&text)
        .map_err(|e| IoError::Parse(format!("Invalid ParaView .series JSON: {e}")))?;

    if parsed.files.is_empty() {
        return Err(IoError::Parse(format!(
            "ParaView .series file '{}' contains no entries",
            path.display()
        )));
    }

    let mut entries = Vec::with_capacity(parsed.files.len());
    for (idx, entry) in parsed.files.into_iter().enumerate() {
        let resolved = resolve_safe_path(base_dir, &entry.name)?;
        let time = entry.time.unwrap_or(idx as f64);
        entries.push((time, resolved));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(entries)
}

/// Returns all time steps listed in a ParaView `.series` file.
pub fn inspect_series_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    Ok(parse_series_index(path)?
        .into_iter()
        .map(|(t, _)| t)
        .collect())
}

/// Reads the first time step of a ParaView `.series` file.
pub fn read_series(path: impl AsRef<Path>) -> Result<DataSet> {
    let entries = parse_series_index(path)?;
    crate::read_dataset(&entries[0].1)
}

/// Reads the time step closest to `time` from a ParaView `.series` file.
pub fn read_series_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    let entries = parse_series_index(path)?;
    let (_, target) = entries
        .iter()
        .min_by(|a, b| (a.0 - time).abs().total_cmp(&(b.0 - time).abs()))
        .ok_or_else(|| IoError::Parse("Empty .series file".into()))?;
    crate::read_dataset(target)
}
