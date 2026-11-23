//! VTK XML readers and writers (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.pvtp`, `.pvti`, `.pvtu`).
//!
//! Supports `ascii`, `binary` (inline base64, `UInt32`/`UInt64` headers), and `appended`
//! (`raw` `_` offset binary and `base64`), including `vtkZLibDataCompressor` compression.

use crate::error::{IoError, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, ImageData, MultiBlockDataSet, PolyData,
    RectilinearGrid, ScalarType, StructuredGrid, UnstructuredGrid,
};
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

mod reader;
mod writer;

pub use reader::read_vtk_xml_bytes;
pub use writer::{write_pvd_series, write_vtk_xml, write_vtk_xml_with_format, write_vtm};

/// Encoding mode for writing VTK XML files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VtkXmlFormat {
    Ascii,
    Binary,
    BinaryZlib,
    BinaryCompressed,
    Appended,
    AppendedCompressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    LittleEndian,
    BigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeaderType {
    UInt32,
    UInt64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayFormat {
    Ascii,
    Binary,
    Appended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XmlDataType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
}

impl XmlDataType {
    fn from_name(s: &str) -> Result<Self> {
        match s {
            "Int8" => Ok(Self::Int8),
            "UInt8" => Ok(Self::UInt8),
            "Int16" => Ok(Self::Int16),
            "UInt16" => Ok(Self::UInt16),
            "Int32" => Ok(Self::Int32),
            "UInt32" => Ok(Self::UInt32),
            "Int64" => Ok(Self::Int64),
            "UInt64" => Ok(Self::UInt64),
            "Float32" => Ok(Self::Float32),
            "Float64" => Ok(Self::Float64),
            other => Err(IoError::Parse(format!("Unsupported VTK XML type '{other}'"))),
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppendedEncoding {
    Raw,
    Base64,
}

struct ExtractedAppended<'a> {
    xml_bytes: Vec<u8>,
    encoding: AppendedEncoding,
    data: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XmlSection {
    None,
    PointData,
    CellData,
    FieldData,
    Points,
    Verts,
    Lines,
    Strips,
    Polys,
    Cells,
    Coordinates,
}

// ================================================================================================
// Readers
// ================================================================================================

/// Reads any VTK XML dataset file (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`).
pub fn read_vtk_xml(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    read_vtk_xml_bytes(&bytes, base_dir)
}

/// Reads a `.pvd` ParaViewData time-series collection into `(timestep, DataSet)` pairs.
pub fn read_pvd_series(path: impl AsRef<Path>) -> Result<Vec<(f64, DataSet)>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let entries = parse_pvd_index(&content, base_dir)?;
    let mut series = Vec::with_capacity(entries.len());
    for (time, file_path) in entries {
        if !file_path.exists() {
            return Err(IoError::MissingFile(file_path));
        }
        let ds = crate::read_dataset(&file_path)?;
        series.push((time, ds));
    }
    Ok(series)
}

/// Parses `.pvd` XML without loading the referenced step files (lightweight metadata).
pub(crate) fn parse_pvd_index(content: &str, base_dir: &Path) -> Result<Vec<(f64, PathBuf)>> {
    let mut reader = Reader::from_str(content);
    reader.config_mut().trim_text(true);
    let mut entries = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Start(e) | Event::Empty(e) => {
                if e.name().as_ref() == "DataSet" {
                    let mut timestep = 0.0f64;
                    let mut file = None;
                    for attr in e.attributes() {
                        let attr = attr?;
                        let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
                        match attr.key.as_ref() {
                            "timestep" => {
                                timestep = val.parse::<f64>().map_err(|_| {
                                    IoError::Parse(format!("Invalid PVD timestep '{val}'"))
                                })?;
                            }
                            "file" => file = Some(val.into_owned()),
                            _ => {}
                        }
                    }
                    if let Some(rel) = file {
                        entries.push((timestep, base_dir.join(rel)));
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(entries)
}

