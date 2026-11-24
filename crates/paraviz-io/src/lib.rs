#![forbid(unsafe_code)]

//! ParaViz scientific file format I/O.
//!
//! Provides readers, writers, and pipeline [`Source`] implementations for:
//! - Legacy VTK (`.vtk`, ASCII and Big-Endian Binary)
//! - VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.pvtp`, `.pvti`, `.pvtu`, ASCII / Base64 Binary / Appended / Zlib)
//! - VTKHDF (`.vtkhdf`, HDF5-backed ImageData / PolyData / UnstructuredGrid / Composite + Steps)
//! - CGNS (`.cgns`, HDF5-backed Structured / Unstructured + FlowSolution)
//! - Exodus II (`.exo`, `.e`, `.ex2`, NetCDF Classic / 64-bit Offset / HDF5 NetCDF-4)
//! - Raw binary 3D volumes (`.raw`)
//! - Binary glTF 2.0 export (`.glb`)
//! - Mesh formats (`.stl` ASCII & Binary, `.obj`, `.ply` ASCII & Binary LE/BE)
//! - Delimited point clouds (`.csv`, `.tsv`)

pub mod abaqus;
pub mod cgns;
pub mod csv;
pub mod dicom;
pub mod ensight;
pub mod error;
pub mod exodus;
pub mod gif;
pub mod gltf;
pub mod gmsh;
pub(crate) mod hdf5_common;
pub mod html_viewer;
pub mod image2d;
pub mod las;
pub(crate) mod limits;
pub mod medical_volume;
pub mod mesh;
pub mod molecular;
pub mod netcdf_cf;
pub mod nifti;
pub mod off;
pub mod openfoam;
pub mod patch_hdf5;
pub mod plot3d;
pub mod raw_volume;
pub mod series;
pub mod tecplot;
pub mod vtk_legacy;
pub mod vtk_xml;
pub mod vtkhdf;
pub mod x3d;
pub mod xdmf;

pub use abaqus::read_abaqus;
pub use cgns::read_cgns;
pub use dicom::read_dicom;
pub use ensight::read_ensight_case;
pub use error::{IoError, Result};
pub use exodus::{inspect_exodus_time_steps, read_exodus, read_exodus_at_time};
pub use gif::{encode_animated_gif_bytes, write_animated_gif};
pub use gltf::{read_gltf, write_glb};
pub use gmsh::read_gmsh;
pub use html_viewer::{encode_html_viewer, write_html_viewer};
pub use image2d::{read_bmp, read_png, read_pnm};
pub use las::read_las;
pub use medical_volume::{read_metaimage, read_nrrd, write_metaimage};
pub use mesh::PlyFormat;
pub use molecular::{
    inspect_pdb_time_steps, inspect_xyz_time_steps, read_pdb, read_pdb_at_time, read_xyz,
    read_xyz_at_time,
};
pub use netcdf_cf::{inspect_netcdf_time_steps, read_netcdf_cf, read_netcdf_cf_at_time};
pub use nifti::{read_nifti, write_nifti};
pub use off::{read_off, write_off};
pub use openfoam::read_openfoam;
pub use plot3d::read_plot3d;
pub use raw_volume::{
    read_raw_volume, read_raw_volume_i16, read_raw_volume_u16, read_raw_volume_with_type,
    RawEndian, RawScalarType, RawVolumeDescriptor,
};
pub use series::{inspect_series_time_steps, read_series, read_series_at_time};
pub use tecplot::read_tecplot;
pub use vtk_legacy::VtkLegacyFormat;
pub use vtk_xml::{read_pvd_series, write_pvd_series, VtkXmlFormat};
pub use vtkhdf::{inspect_vtkhdf_time_steps, read_vtkhdf, read_vtkhdf_at_time};
pub use x3d::read_x3d;
pub use xdmf::{inspect_xdmf_time_steps, read_xdmf, read_xdmf_at_time};
pub use patch_hdf5::read_patch_hdf5;

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
    VtkHdf,
    Cgns,
    Exodus,
    PatchHdf5,
    Xdmf,
    VtkXml,
    VtkLegacy,
    Series,
    EnSight,
    OpenFoam,
    Gmsh,
    Plot3d,
    Tecplot,
    Abaqus,
    NetCdf,
    Stl,
    Obj,
    Ply,
    Off,
    Gltf,
    X3d,
    Las,
    Xyz,
    Pdb,
    Csv,
    Nifti,
    Nrrd,
    MetaImage,
    Dicom,
    Image2d,
}

impl DatasetReader {
    /// All selectable reader options, starting with [`DatasetReader::Auto`].
    pub const ALL: &'static [DatasetReader] = &[
        Self::Auto,
        Self::VtkHdf,
        Self::Cgns,
        Self::Exodus,
        Self::PatchHdf5,
        Self::Xdmf,
        Self::VtkXml,
        Self::VtkLegacy,
        Self::Series,
        Self::EnSight,
        Self::OpenFoam,
        Self::Gmsh,
        Self::Plot3d,
        Self::Tecplot,
        Self::Abaqus,
        Self::NetCdf,
        Self::Stl,
        Self::Obj,
        Self::Ply,
        Self::Off,
        Self::Gltf,
        Self::X3d,
        Self::Las,
        Self::Xyz,
        Self::Pdb,
        Self::Csv,
        Self::Nifti,
        Self::Nrrd,
        Self::MetaImage,
        Self::Dicom,
        Self::Image2d,
    ];

    /// Human-readable reader name shown in the UI and diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto-detect",
            Self::VtkHdf => "VTKHDF Reader (.vtkhdf)",
            Self::Cgns => "CGNS Reader (.cgns)",
            Self::Exodus => "Exodus II Reader (.exo, .e, .ex2)",
            Self::PatchHdf5 => "Patch HDF5 Reader (.h5)",
            Self::Xdmf => "XDMF Reader (.xdmf, .xmf)",
            Self::VtkXml => "VTK XML Reader (.vtp, .vtu, .vti, .vtr, .vts, .vtm)",
            Self::VtkLegacy => "VTK Legacy Reader (.vtk)",
            Self::Series => "ParaView File Series (.series)",
            Self::EnSight => "EnSight Gold Reader (.case)",
            Self::OpenFoam => "OpenFOAM Reader (.foam, polyMesh)",
            Self::Gmsh => "Gmsh Mesh Reader (.msh)",
            Self::Plot3d => "PLOT3D Reader (.xyz, .p3d)",
            Self::Tecplot => "Tecplot ASCII Reader (.dat, .tec, .plt)",
            Self::Abaqus => "Abaqus Input Reader (.inp)",
            Self::NetCdf => "NetCDF CF Reader (.nc, .cdf)",
            Self::Stl => "STL Mesh Reader (.stl)",
            Self::Obj => "Wavefront OBJ Reader (.obj)",
            Self::Ply => "PLY Mesh Reader (.ply)",
            Self::Off => "Geomview OFF Reader (.off)",
            Self::Gltf => "glTF 2.0 / GLB Reader (.gltf, .glb)",
            Self::X3d => "X3D Scene Reader (.x3d)",
            Self::Las => "LAS LiDAR Reader (.las)",
            Self::Xyz => "XYZ Molecular Reader (.xyz)",
            Self::Pdb => "PDB Molecular Reader (.pdb, .ent)",
            Self::Csv => "CSV / TSV Table Reader (.csv, .tsv)",
            Self::Nifti => "NIfTI Volume Reader (.nii, .nii.gz)",
            Self::Nrrd => "NRRD Volume Reader (.nrrd, .nhdr)",
            Self::MetaImage => "MetaImage Volume Reader (.mhd, .mha)",
            Self::Dicom => "DICOM Image Reader (.dcm)",
            Self::Image2d => "2D Image Reader (.png, .bmp, .ppm, .pgm, .pbm)",
        }
    }

    /// Short reader name for compact badges and status messages.
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::VtkHdf => "VTKHDF Reader",
            Self::Cgns => "CGNS Reader",
            Self::Exodus => "Exodus II Reader",
            Self::PatchHdf5 => "Patch HDF5 Reader",
            Self::Xdmf => "XDMF Reader",
            Self::VtkXml => "VTK XML Reader",
            Self::VtkLegacy => "VTK Legacy Reader",
            Self::Series => "Series Reader",
            Self::EnSight => "EnSight Gold Reader",
            Self::OpenFoam => "OpenFOAM Reader",
            Self::Gmsh => "Gmsh Reader",
            Self::Plot3d => "PLOT3D Reader",
            Self::Tecplot => "Tecplot Reader",
            Self::Abaqus => "Abaqus Reader",
            Self::NetCdf => "NetCDF CF Reader",
            Self::Stl => "STL Reader",
            Self::Obj => "OBJ Reader",
            Self::Ply => "PLY Reader",
            Self::Off => "OFF Reader",
            Self::Gltf => "glTF Reader",
            Self::X3d => "X3D Reader",
            Self::Las => "LAS Reader",
            Self::Xyz => "XYZ Molecular Reader",
            Self::Pdb => "PDB Reader",
            Self::Csv => "CSV/TSV Reader",
            Self::Nifti => "NIfTI Reader",
            Self::Nrrd => "NRRD Reader",
            Self::MetaImage => "MetaImage Reader",
            Self::Dicom => "DICOM Reader",
            Self::Image2d => "2D Image Reader",
        }
    }

    /// Parses a reader name (case-insensitive) for CLI / API usage.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" | "auto-detect" => Some(Self::Auto),
            "vtkhdf" | "vtk-hdf" | "vtk_hdf" => Some(Self::VtkHdf),
            "cgns" => Some(Self::Cgns),
            "exodus" | "exodus2" | "exodusii" | "exo" | "ex2" | "e" => Some(Self::Exodus),
            "patch" | "patch-hdf5" => Some(Self::PatchHdf5),
            "xdmf" | "xmf" => Some(Self::Xdmf),
            "vtkxml" | "vtk-xml" | "vtk_xml" | "vtp" | "vtu" | "vti" | "vtr" | "vts" | "vtm" => {
                Some(Self::VtkXml)
            }
            "vtk" | "vtklegacy" | "vtk-legacy" | "vtk_legacy" => Some(Self::VtkLegacy),
            "series" => Some(Self::Series),
            "ensight" | "case" => Some(Self::EnSight),
            "openfoam" | "foam" => Some(Self::OpenFoam),
            "gmsh" | "msh" => Some(Self::Gmsh),
            "plot3d" | "p3d" => Some(Self::Plot3d),
            "tecplot" | "tec" | "plt" => Some(Self::Tecplot),
            "abaqus" | "inp" => Some(Self::Abaqus),
            "netcdf" | "nc" | "cdf" => Some(Self::NetCdf),
            "stl" => Some(Self::Stl),
            "obj" => Some(Self::Obj),
            "ply" => Some(Self::Ply),
            "off" => Some(Self::Off),
            "gltf" | "glb" => Some(Self::Gltf),
            "x3d" => Some(Self::X3d),
            "las" => Some(Self::Las),
            "xyz" => Some(Self::Xyz),
            "pdb" | "ent" => Some(Self::Pdb),
            "csv" | "tsv" => Some(Self::Csv),
            "nifti" | "nii" => Some(Self::Nifti),
            "nrrd" | "nhdr" => Some(Self::Nrrd),
            "metaimage" | "mhd" | "mha" => Some(Self::MetaImage),
            "dicom" | "dcm" => Some(Self::Dicom),
            "image" | "png" | "bmp" | "ppm" | "pgm" | "pbm" => Some(Self::Image2d),
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
        "vtkhdf" => Ok(DatasetReader::VtkHdf),
        "cgns" => Ok(DatasetReader::Cgns),
        "exo" | "e" | "ex2" => Ok(DatasetReader::Exodus),
        "h5" | "hdf" | "hdf5" => detect_hdf5_reader(path),
        "nrrd" | "nhdr" => Ok(DatasetReader::Nrrd),
        "mhd" | "mha" => Ok(DatasetReader::MetaImage),
        "dcm" => Ok(DatasetReader::Dicom),
        "png" | "bmp" | "ppm" | "pgm" | "pbm" => Ok(DatasetReader::Image2d),
        "case" => Ok(DatasetReader::EnSight),
        "xdmf" | "xmf" => Ok(DatasetReader::Xdmf),
        "series" => Ok(DatasetReader::Series),
        "foam" => Ok(DatasetReader::OpenFoam),
        "msh" => Ok(DatasetReader::Gmsh),
        "p3d" => Ok(DatasetReader::Plot3d),
        "xyz" => {
            if molecular::looks_like_xyz_molecular(magic_str) {
                Ok(DatasetReader::Xyz)
            } else {
                Ok(DatasetReader::Plot3d)
            }
        }
        "pdb" | "ent" => Ok(DatasetReader::Pdb),
        "tec" | "plt" => Ok(DatasetReader::Tecplot),
        "dat" => {
            let up = magic_str.to_ascii_uppercase();
            if up.contains("VARIABLES") || up.contains("ZONE") || up.contains("TITLE") {
                Ok(DatasetReader::Tecplot)
            } else {
                Err(IoError::UnsupportedFormat(format!(
                    "Cannot detect format for '{}'",
                    path.display()
                )))
            }
        }
        "inp" => Ok(DatasetReader::Abaqus),
        "nc" | "cdf" => Ok(DatasetReader::NetCdf),
        "stl" => Ok(DatasetReader::Stl),
        "obj" => Ok(DatasetReader::Obj),
        "ply" => Ok(DatasetReader::Ply),
        "off" => Ok(DatasetReader::Off),
        "gltf" | "glb" => Ok(DatasetReader::Gltf),
        "x3d" => Ok(DatasetReader::X3d),
        "las" => Ok(DatasetReader::Las),
        "csv" | "tsv" => Ok(DatasetReader::Csv),
        _ => {
            if path.file_name().and_then(|s| s.to_str()) == Some("controlDict") {
                Ok(DatasetReader::OpenFoam)
            } else if magic.starts_with(b"CDF\x01") || magic.starts_with(b"CDF\x02") {
                Ok(DatasetReader::Exodus)
            } else if magic.starts_with(b"\x89HDF\r\n\x1a\n") {
                detect_hdf5_reader(path)
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

fn detect_hdf5_reader(path: &Path) -> Result<DatasetReader> {
    let file = hdf5_common::open_hdf5(path)?;
    let root = file
        .root_group()
        .map_err(|e| IoError::Parse(format!("Failed to inspect HDF5 root group: {e}")))?;

    if root.group("VTKHDF").is_ok() || hdf5_common::group_attr_string(&root, "Type").is_some() {
        return Ok(DatasetReader::VtkHdf);
    }
    if cgns::is_cgns_hdf5(&file) {
        return Ok(DatasetReader::Cgns);
    }
    if exodus::is_exodus_hdf5(&file) {
        return Ok(DatasetReader::Exodus);
    }
    if patch_hdf5::is_patch_hdf5(&file) {
        return Ok(DatasetReader::PatchHdf5);
    }

    Err(IoError::UnsupportedFormat(format!(
        "HDF5 file '{}' does not match VTKHDF, CGNS, Exodus II, or Patch HDF5 layout",
        path.display()
    )))
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
            DatasetReader::VtkHdf => vtkhdf::read_vtkhdf(path),
            DatasetReader::Cgns => cgns::read_cgns(path),
            DatasetReader::Exodus => exodus::read_exodus(path),
            DatasetReader::PatchHdf5 => patch_hdf5::read_patch_hdf5(path),
            DatasetReader::Xdmf => xdmf::read_xdmf(path),
            DatasetReader::VtkXml => vtk_xml::read_vtk_xml(path),
            DatasetReader::VtkLegacy => vtk_legacy::read_vtk_legacy(path),
            DatasetReader::Series => series::read_series(path),
            DatasetReader::EnSight => ensight::read_ensight_case(path),
            DatasetReader::OpenFoam => openfoam::read_openfoam(path),
            DatasetReader::Gmsh => gmsh::read_gmsh(path),
            DatasetReader::Plot3d => plot3d::read_plot3d(path),
            DatasetReader::Tecplot => tecplot::read_tecplot(path),
            DatasetReader::Abaqus => abaqus::read_abaqus(path),
            DatasetReader::NetCdf => netcdf_cf::read_netcdf_cf(path),
            DatasetReader::Stl => Ok(DataSet::PolyData(mesh::read_stl(path)?)),
            DatasetReader::Obj => Ok(DataSet::PolyData(mesh::read_obj(path)?)),
            DatasetReader::Ply => Ok(DataSet::PolyData(mesh::read_ply(path)?)),
            DatasetReader::Off => Ok(DataSet::PolyData(off::read_off(path)?)),
            DatasetReader::Gltf => gltf::read_gltf(path),
            DatasetReader::X3d => x3d::read_x3d(path),
            DatasetReader::Las => las::read_las(path),
            DatasetReader::Xyz => molecular::read_xyz(path),
            DatasetReader::Pdb => molecular::read_pdb(path),
            DatasetReader::Csv => Ok(DataSet::PolyData(csv::read_csv(path)?)),
            DatasetReader::Nifti => Ok(DataSet::ImageData(nifti::read_nifti(path)?)),
            DatasetReader::Nrrd => Ok(DataSet::ImageData(medical_volume::read_nrrd(path)?)),
            DatasetReader::MetaImage => {
                Ok(DataSet::ImageData(medical_volume::read_metaimage(path)?))
            }
            DatasetReader::Dicom => dicom::read_dicom(path),
            DatasetReader::Image2d => {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .unwrap_or_default();
                match ext.as_str() {
                    "png" => image2d::read_png(path),
                    "bmp" => image2d::read_bmp(path),
                    _ => image2d::read_pnm(path),
                }
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
        "off" => match dataset {
            DataSet::PolyData(p) => off::write_off(path, p),
            other => Err(IoError::UnsupportedFormat(format!(
                "OFF writer requires PolyData, got {}",
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
/// (including temporal XDMF, VTKHDF, and Exodus II collections on demand).
#[derive(Debug, Clone)]
pub struct VtkFileSource {
    path: PathBuf,
    label: String,
    reader: DatasetReader,
    resolved_reader: Option<DatasetReader>,
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
        let resolved_reader = match reader {
            DatasetReader::Auto => detect_reader(&path).ok(),
            explicit => Some(explicit),
        };
        let time_steps = match resolved_reader {
            Some(DatasetReader::Xdmf) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                xdmf::inspect_xdmf_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::VtkHdf) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                vtkhdf::inspect_vtkhdf_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::Exodus) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                exodus::inspect_exodus_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::Series) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                series::inspect_series_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::Xyz) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                molecular::inspect_xyz_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::Pdb) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                molecular::inspect_pdb_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            Some(DatasetReader::NetCdf) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                netcdf_cf::inspect_netcdf_time_steps(&path).unwrap_or_default()
            }))
            .unwrap_or_default(),
            _ => Vec::new(),
        };
        Self {
            path,
            label,
            reader,
            resolved_reader,
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
            let resolved = self
                .resolved_reader
                .or_else(|| detect_reader(&self.path).ok());
            let path = self.path.clone();
            match resolved {
                Some(DatasetReader::Xdmf) => {
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
                Some(DatasetReader::VtkHdf) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        vtkhdf::read_vtkhdf_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "VTKHDF Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                Some(DatasetReader::Exodus) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        exodus::read_exodus_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "Exodus II Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                Some(DatasetReader::Series) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        series::read_series_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "Series Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                Some(DatasetReader::Xyz) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        molecular::read_xyz_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "XYZ Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                Some(DatasetReader::Pdb) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        molecular::read_pdb_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "PDB Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                Some(DatasetReader::NetCdf) => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        netcdf_cf::read_netcdf_cf_at_time(&path, time)
                    }));
                    return match res {
                        Ok(inner) => inner.map_err(|e| PipelineError::ExecutionFailed(e.to_string())),
                        Err(_) => Err(PipelineError::ExecutionFailed(format!(
                            "NetCDF CF Reader failed unexpectedly while reading '{}'",
                            self.path.display()
                        ))),
                    };
                }
                _ => {}
            }
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
