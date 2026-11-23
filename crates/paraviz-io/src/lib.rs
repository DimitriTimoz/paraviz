//! ParaViz scientific file format I/O.
//!
//! Provides readers, writers, and pipeline [`Source`] implementations for:
//! - Legacy VTK (`.vtk`, ASCII and Big-Endian Binary)
//! - VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.pvtp`, `.pvti`, `.pvtu`, ASCII / Base64 Binary / Appended / Zlib)
//! - Raw binary 3D volumes (`.raw`)
//! - Binary glTF 2.0 export (`.glb`)
//! - Mesh formats (`.stl` ASCII & Binary, `.obj`, `.ply` ASCII & Binary LE/BE)
//! - Delimited point clouds (`.csv`, `.tsv`)

pub mod csv;
pub mod error;
pub mod gltf;
pub mod mesh;
pub mod raw_volume;
pub mod vtk_legacy;
pub mod vtk_xml;

pub use error::{IoError, Result};
pub use gltf::write_glb;
pub use mesh::PlyFormat;
pub use raw_volume::{
    read_raw_volume, read_raw_volume_i16, read_raw_volume_u16, read_raw_volume_with_type,
    RawEndian, RawScalarType, RawVolumeDescriptor,
};
pub use vtk_legacy::VtkLegacyFormat;
pub use vtk_xml::{read_pvd_series, write_pvd_series, VtkXmlFormat};

use paraviz_core::{DataSet, PipelineError, PipelineInfo, Source};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Reads a scientific dataset from `path`, auto-detecting the format by extension and magic bytes.
pub fn read_dataset(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(IoError::MissingFile(path.to_path_buf()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let magic = read_magic_prefix(path, 512)?;
    let magic_lossy = String::from_utf8_lossy(&magic);
    let magic_str = magic_lossy.trim_start();

    match ext.as_str() {
        "vtk" => {
            if magic_str.starts_with('<') {
                vtk_xml::read_vtk_xml(path)
            } else {
                vtk_legacy::read_vtk_legacy(path)
            }
        }
        "vtp" | "vti" | "vtu" | "vtr" | "vts" | "vtm" | "pvd" | "pvtp" | "pvti" | "pvtu" => {
            vtk_xml::read_vtk_xml(path)
        }
        "stl" => Ok(DataSet::PolyData(mesh::read_stl(path)?)),
        "obj" => Ok(DataSet::PolyData(mesh::read_obj(path)?)),
        "ply" => Ok(DataSet::PolyData(mesh::read_ply(path)?)),
        "csv" | "tsv" => Ok(DataSet::PolyData(csv::read_csv(path)?)),
        _ => {
            if magic_str.starts_with("# vtk DataFile Version") {
                vtk_legacy::read_vtk_legacy(path)
            } else if magic_str.starts_with("<?xml") || magic_str.starts_with("<VTKFile") {
                vtk_xml::read_vtk_xml(path)
            } else if magic_str.starts_with("ply\n") || magic_str.starts_with("ply\r\n") {
                Ok(DataSet::PolyData(mesh::read_ply(path)?))
            } else if magic_str.starts_with("solid") && magic_str.contains("facet") {
                Ok(DataSet::PolyData(mesh::read_stl(path)?))
            } else {
                Err(IoError::UnsupportedFormat(format!(
                    "Cannot detect format for '{}'",
                    path.display()
                )))
            }
        }
    }
}

/// Writes `dataset` to `path`, dispatching by file extension.
pub fn write_dataset(path: impl AsRef<Path>, dataset: &DataSet) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "vtk" => vtk_legacy::write_vtk_legacy(path, dataset),
        "vtp" | "vti" | "vtu" | "vtr" | "vts" | "vtm" => vtk_xml::write_vtk_xml(path, dataset),
        "stl" => match dataset {
            DataSet::PolyData(p) => mesh::write_stl(path, p),
            other => Err(IoError::UnsupportedFormat(format!(
                "STL writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        "obj" => match dataset {
            DataSet::PolyData(p) => mesh::write_obj(path, p),
            other => Err(IoError::UnsupportedFormat(format!(
                "OBJ writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        "ply" => match dataset {
            DataSet::PolyData(p) => mesh::write_ply(path, p),
            other => Err(IoError::UnsupportedFormat(format!(
                "PLY writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        "csv" | "tsv" => match dataset {
            DataSet::PolyData(p) => csv::write_csv(path, p),
            other => Err(IoError::UnsupportedFormat(format!(
                "CSV writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        "glb" => match dataset {
            DataSet::PolyData(p) => gltf::write_glb(path, p, None, None),
            other => Err(IoError::UnsupportedFormat(format!(
                "GLB writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        _ => Err(IoError::UnsupportedFormat(format!(
            "Unsupported output extension '.{ext}' for '{}'",
            path.display()
        ))),
    }
}

fn read_magic_prefix(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; max_bytes];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

/// Pipeline [`Source`] that reads any single dataset file supported by `paraviz-io`.
#[derive(Debug, Clone)]
pub struct VtkFileSource {
    path: PathBuf,
    label: String,
}

impl VtkFileSource {
    /// Creates a new [`VtkFileSource`] pointing at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let label = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("VtkFileSource")
            .to_string();
        Self { path, label }
    }

    /// Returns the underlying file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Source for VtkFileSource {
    fn name(&self) -> &str {
        &self.label
    }

    fn request_information(&self) -> PipelineInfo {
        PipelineInfo::default()
    }

    fn execute(&self, _time: f64) -> std::result::Result<DataSet, PipelineError> {
        read_dataset(&self.path).map_err(|e| PipelineError::ExecutionFailed(e.to_string()))
    }
}

/// Pipeline [`Source`] that reads a `.pvd` ParaViewData time-series collection on demand.
#[derive(Debug, Clone)]
pub struct PvdSeriesSource {
    path: PathBuf,
    label: String,
    entries: Vec<(f64, PathBuf)>,
}

impl PvdSeriesSource {
    /// Opens a `.pvd` file and indexes its time steps without eagerly loading step datasets.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let entries = vtk_xml::parse_pvd_index(&content, base_dir)?;
        let label = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("PvdSeriesSource")
            .to_string();
        Ok(Self {
            path,
            label,
            entries,
        })
    }

    /// Returns the `.pvd` file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the indexed time steps in ascending order.
    pub fn time_steps(&self) -> Vec<f64> {
        self.entries.iter().map(|(t, _)| *t).collect()
    }
}

impl Source for PvdSeriesSource {
    fn name(&self) -> &str {
        &self.label
    }

    fn request_information(&self) -> PipelineInfo {
        PipelineInfo {
            time_steps: self.time_steps(),
            current_time: self.entries.first().map(|e| e.0).unwrap_or(0.0),
            bounds: None,
        }
    }

    fn execute(&self, time: f64) -> std::result::Result<DataSet, PipelineError> {
        let (_, target_path) = self
            .entries
            .iter()
            .min_by(|a, b| (a.0 - time).abs().total_cmp(&(b.0 - time).abs()))
            .ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "PVD series '{}' contains no time steps",
                    self.path.display()
                ))
            })?;
        read_dataset(target_path).map_err(|e| PipelineError::ExecutionFailed(e.to_string()))
    }
}
