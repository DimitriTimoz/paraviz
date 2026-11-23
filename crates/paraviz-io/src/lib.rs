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
pub mod ensight;
pub mod error;
pub mod gif;
pub mod gltf;
pub mod html_viewer;
pub mod medical_volume;
pub mod mesh;
pub mod nifti;
pub mod openfoam;
pub mod raw_volume;
pub mod vtk_legacy;
pub mod vtk_xml;
pub mod xdmf;

pub use ensight::read_ensight_case;
pub use error::{IoError, Result};
pub use gif::{encode_animated_gif_bytes, write_animated_gif};
pub use gltf::write_glb;
pub use html_viewer::{encode_html_viewer, write_html_viewer};
pub use medical_volume::{read_metaimage, read_nrrd, write_metaimage};
pub use mesh::PlyFormat;
pub use nifti::{read_nifti, write_nifti};
pub use openfoam::read_openfoam;
pub use raw_volume::{
    read_raw_volume, read_raw_volume_i16, read_raw_volume_u16, read_raw_volume_with_type,
    RawEndian, RawScalarType, RawVolumeDescriptor,
};
pub use vtk_legacy::VtkLegacyFormat;
pub use vtk_xml::{read_pvd_series, write_pvd_series, VtkXmlFormat};
pub use xdmf::{inspect_xdmf_time_steps, read_xdmf, read_xdmf_at_time};

use paraviz_core::{DataSet, PipelineError, PipelineInfo, Source};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Explicit or auto-detected scientific dataset reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DatasetReader {
    #[default]
    Auto,
    Xdmf,
    VtkXml,
    VtkLegacy,
    EnSight,
    OpenFoam,
    Stl,
    Obj,
    Ply,
    Csv,
    Nifti,
    Nrrd,
    MetaImage,
}

impl DatasetReader {
    /// All selectable reader options, starting with [`DatasetReader::Auto`].
    pub const ALL: &'static [DatasetReader] = &[
        Self::Auto,
        Self::Xdmf,
        Self::VtkXml,
        Self::VtkLegacy,
        Self::EnSight,
        Self::OpenFoam,
        Self::Stl,
        Self::Obj,
        Self::Ply,
        Self::Csv,
        Self::Nifti,
        Self::Nrrd,
        Self::MetaImage,
    ];

    /// Human-readable reader name shown in the UI and diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto-detect",
            Self::Xdmf => "XDMF Reader (.xdmf, .xmf)",
            Self::VtkXml => "VTK XML Reader (.vtp, .vtu, .vti, .vtr, .vts, .vtm)",
            Self::VtkLegacy => "VTK Legacy Reader (.vtk)",
            Self::EnSight => "EnSight Gold Reader (.case)",
            Self::OpenFoam => "OpenFOAM Reader (.foam, polyMesh)",
            Self::Stl => "STL Mesh Reader (.stl)",
            Self::Obj => "Wavefront OBJ Reader (.obj)",
            Self::Ply => "PLY Mesh Reader (.ply)",
            Self::Csv => "CSV / TSV Table Reader (.csv, .tsv)",
            Self::Nifti => "NIfTI Volume Reader (.nii, .nii.gz)",
            Self::Nrrd => "NRRD Volume Reader (.nrrd, .nhdr)",
            Self::MetaImage => "MetaImage Volume Reader (.mhd, .mha)",
        }
    }

    /// Short reader name for compact badges and status messages.
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Xdmf => "XDMF Reader",
            Self::VtkXml => "VTK XML Reader",
            Self::VtkLegacy => "VTK Legacy Reader",
            Self::EnSight => "EnSight Gold Reader",
            Self::OpenFoam => "OpenFOAM Reader",
            Self::Stl => "STL Reader",
            Self::Obj => "OBJ Reader",
            Self::Ply => "PLY Reader",
            Self::Csv => "CSV/TSV Reader",
            Self::Nifti => "NIfTI Reader",
            Self::Nrrd => "NRRD Reader",
            Self::MetaImage => "MetaImage Reader",
        }
    }

    /// Parses a reader name (case-insensitive) for CLI / API usage.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" | "auto-detect" => Some(Self::Auto),
            "xdmf" | "xmf" => Some(Self::Xdmf),
            "vtkxml" | "vtk-xml" | "vtk_xml" | "vtp" | "vtu" | "vti" | "vtr" | "vts" | "vtm" => {
                Some(Self::VtkXml)
            }
            "vtk" | "vtklegacy" | "vtk-legacy" | "vtk_legacy" => Some(Self::VtkLegacy),
            "ensight" | "case" => Some(Self::EnSight),
            "openfoam" | "foam" => Some(Self::OpenFoam),
            "stl" => Some(Self::Stl),
            "obj" => Some(Self::Obj),
            "ply" => Some(Self::Ply),
            "csv" | "tsv" => Some(Self::Csv),
            "nifti" | "nii" => Some(Self::Nifti),
            "nrrd" | "nhdr" => Some(Self::Nrrd),
            "metaimage" | "mhd" | "mha" => Some(Self::MetaImage),
            _ => None,
        }
    }
}

/// Detects which [`DatasetReader`] would be used for `path` without loading the full dataset.
pub fn detect_reader(path: impl AsRef<Path>) -> Result<DatasetReader> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(IoError::MissingFile(path.to_path_buf()));
    }
    if path.is_dir() && path.join("constant").join("polyMesh").is_dir() {
        return Ok(DatasetReader::OpenFoam);
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let magic = read_magic_prefix(path, 512)?;
    let magic_lossy = String::from_utf8_lossy(&magic);
    let magic_str = magic_lossy.trim_start();

    let full_lower = path.to_string_lossy().to_ascii_lowercase();
    if full_lower.ends_with(".nii.gz") || ext == "nii" {
        return Ok(DatasetReader::Nifti);
    }

    match ext.as_str() {
        "vtk" => {
            if magic_str.starts_with('<') {
                Ok(DatasetReader::VtkXml)
            } else {
                Ok(DatasetReader::VtkLegacy)
            }
        }
        "vtp" | "vti" | "vtu" | "vtr" | "vts" | "vtm" | "pvd" | "pvtp" | "pvti" | "pvtu"
        | "pvtr" | "pvts" => Ok(DatasetReader::VtkXml),
        "nrrd" | "nhdr" => Ok(DatasetReader::Nrrd),
        "mhd" | "mha" => Ok(DatasetReader::MetaImage),
        "case" => Ok(DatasetReader::EnSight),
        "xdmf" | "xmf" => Ok(DatasetReader::Xdmf),
        "foam" => Ok(DatasetReader::OpenFoam),
        "stl" => Ok(DatasetReader::Stl),
        "obj" => Ok(DatasetReader::Obj),
        "ply" => Ok(DatasetReader::Ply),
        "csv" | "tsv" => Ok(DatasetReader::Csv),
        _ => {
            if path.file_name().and_then(|s| s.to_str()) == Some("controlDict") {
                Ok(DatasetReader::OpenFoam)
            } else if magic_str.starts_with("# vtk DataFile Version") {
                Ok(DatasetReader::VtkLegacy)
            } else if magic_str.contains("<Xdmf") || magic_str.contains("<xdmf") {
                Ok(DatasetReader::Xdmf)
            } else if magic_str.starts_with("<?xml") || magic_str.starts_with("<VTKFile") {
                Ok(DatasetReader::VtkXml)
            } else if magic_str.starts_with("NRRD") {
                Ok(DatasetReader::Nrrd)
            } else if magic_str.starts_with("ply\n") || magic_str.starts_with("ply\r\n") {
                Ok(DatasetReader::Ply)
            } else if magic_str.starts_with("solid") && magic_str.contains("facet") {
                Ok(DatasetReader::Stl)
            } else {
                Err(IoError::UnsupportedFormat(format!(
                    "Cannot detect format for '{}'",
                    path.display()
                )))
            }
        }
    }
}

/// Reads a scientific dataset from `path` using `reader` (or auto-detecting if [`DatasetReader::Auto`]),
/// returning both the loaded [`DataSet`] and the concrete [`DatasetReader`] that was used.
pub fn read_dataset_with_reader(
    path: impl AsRef<Path>,
    reader: DatasetReader,
) -> Result<(DataSet, DatasetReader)> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(IoError::MissingFile(path.to_path_buf()));
    }
    let resolved = match reader {
        DatasetReader::Auto => detect_reader(path)?,
        explicit => explicit,
    };
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<DataSet> {
        match resolved {
            DatasetReader::Auto => unreachable!(),
            DatasetReader::Xdmf => xdmf::read_xdmf(path),
            DatasetReader::VtkXml => vtk_xml::read_vtk_xml(path),
            DatasetReader::VtkLegacy => vtk_legacy::read_vtk_legacy(path),
            DatasetReader::EnSight => ensight::read_ensight_case(path),
            DatasetReader::OpenFoam => openfoam::read_openfoam(path),
            DatasetReader::Stl => Ok(DataSet::PolyData(mesh::read_stl(path)?)),
            DatasetReader::Obj => Ok(DataSet::PolyData(mesh::read_obj(path)?)),
            DatasetReader::Ply => Ok(DataSet::PolyData(mesh::read_ply(path)?)),
            DatasetReader::Csv => Ok(DataSet::PolyData(csv::read_csv(path)?)),
            DatasetReader::Nifti => Ok(DataSet::ImageData(nifti::read_nifti(path)?)),
            DatasetReader::Nrrd => Ok(DataSet::ImageData(medical_volume::read_nrrd(path)?)),
            DatasetReader::MetaImage => {
                Ok(DataSet::ImageData(medical_volume::read_metaimage(path)?))
            }
        }
    }));
    let ds = match res {
        Ok(inner) => inner?,
        Err(_) => {
            return Err(IoError::Parse(format!(
                "{} failed unexpectedly while reading '{}' (the file may be incomplete or still being written)",
                resolved.short_name(),
                path.display()
            )));
        }
    };
    Ok((ds, resolved))
}

/// Reads a scientific dataset from `path`, auto-detecting the format by extension and magic bytes.
pub fn read_dataset(path: impl AsRef<Path>) -> Result<DataSet> {
    read_dataset_with_reader(path, DatasetReader::Auto).map(|(ds, _)| ds)
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
        "html" | "htm" => match dataset {
            DataSet::PolyData(p) => html_viewer::write_html_viewer(
                path,
                p,
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ParaViz"),
                None,
                None,
            ),
            other => Err(IoError::UnsupportedFormat(format!(
                "HTML viewer writer requires PolyData, got {}",
                other.type_name()
            ))),
        },
        "nii" | "gz" if path.to_string_lossy().to_ascii_lowercase().ends_with(".nii")
            || path.to_string_lossy().to_ascii_lowercase().ends_with(".nii.gz") =>
        {
            match dataset {
                DataSet::ImageData(img) => nifti::write_nifti(path, img),
                other => Err(IoError::UnsupportedFormat(format!(
                    "NIfTI writer requires ImageData, got {}",
                    other.type_name()
                ))),
            }
        }
        "mha" => match dataset {
            DataSet::ImageData(img) => medical_volume::write_metaimage(path, img),
            other => Err(IoError::UnsupportedFormat(format!(
                "MetaImage writer requires ImageData, got {}",
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

/// Pipeline [`Source`] that reads any single dataset file supported by `paraviz-io`
/// (including temporal XDMF collections on demand).
#[derive(Debug, Clone)]
pub struct VtkFileSource {
    path: PathBuf,
    label: String,
    reader: DatasetReader,
    time_steps: Vec<f64>,
}

impl VtkFileSource {
    /// Creates a new [`VtkFileSource`] pointing at `path` with auto-detected reader.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_reader(path, DatasetReader::Auto)
    }

    /// Creates a new [`VtkFileSource`] pointing at `path` using an explicit [`DatasetReader`].
    pub fn with_reader(path: impl Into<PathBuf>, reader: DatasetReader) -> Self {
        let path = path.into();
        let label = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("VtkFileSource")
            .to_string();
        let is_xdmf = reader == DatasetReader::Xdmf
            || (reader == DatasetReader::Auto && detect_reader(&path).ok() == Some(DatasetReader::Xdmf));
        let time_steps = if is_xdmf {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                xdmf::inspect_xdmf_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        Self {
            path,
            label,
            reader,
            time_steps,
        }
    }

    /// Returns the underlying file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the configured [`DatasetReader`] (`Auto` or explicit).
    pub fn reader(&self) -> DatasetReader {
        self.reader
    }
}

impl Source for VtkFileSource {
    fn name(&self) -> &str {
        &self.label
    }

    fn request_information(&self) -> PipelineInfo {
        PipelineInfo {
            time_steps: self.time_steps.clone(),
            current_time: self.time_steps.first().copied().unwrap_or(0.0),
            bounds: None,
        }
    }

    fn execute(&self, time: f64) -> std::result::Result<DataSet, PipelineError> {
        if !self.time_steps.is_empty() {
            let path = self.path.clone();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                xdmf::read_xdmf_at_time(&path, time)
            }));
            return match res {
                Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                Err(_) => Err(PipelineError::ExecutionFailed(format!(
                    "XDMF Reader failed unexpectedly while reading '{}'",
                    self.path.display()
                ))),
            };
        }
        read_dataset_with_reader(&self.path, self.reader)
            .map(|(ds, _)| ds)
            .map_err(|e| PipelineError::ExecutionFailed(e.to_string()))
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
