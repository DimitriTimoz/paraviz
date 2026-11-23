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

/// Parses VTK XML bytes with `base_dir` for resolving external `.vtm` / `.pvd` references.
pub fn read_vtk_xml_bytes(bytes: &[u8], base_dir: &Path) -> Result<DataSet> {
    let extracted = extract_appended_data(bytes);
    let xml_str = std::str::from_utf8(&extracted.xml_bytes)
        .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in VTK XML: {e}")))?;

    let mut reader = Reader::from_str(xml_str);
    reader.config_mut().trim_text(true);

    let mut file_type = String::new();
    let mut byte_order = ByteOrder::LittleEndian;
    let mut header_type = HeaderType::UInt32;
    let mut compressed = false;

    type ImageMeta = ([usize; 3], [f64; 3], [f64; 3], [isize; 6]);
    type PieceCellsTuple = (Option<Vec<u32>>, Option<Vec<u32>>, Option<Vec<u8>>);

    let mut image_meta: Option<ImageMeta> = None;
    let mut whole_extent: Option<[isize; 6]> = None;

    let mut poly = PolyData::new();
    let mut image = ImageData::default();
    let mut rect = RectilinearGrid::default();
    let mut sgrid = StructuredGrid::default();
    let mut ugrid = UnstructuredGrid::new();
    let mut multiblock = MultiBlockDataSet::new();
    let mut pvd_entries: Vec<(f64, PathBuf)> = Vec::new();
    let mut parallel_pieces: Vec<ParallelPiece> = Vec::new();

    let mut section = XmlSection::None;
    let mut active_point_attrs: (Option<String>, Option<String>, Option<String>) =
        (None, None, None);
    let mut active_cell_attrs: (Option<String>, Option<String>, Option<String>) =
        (None, None, None);

    // Current DataArray being read
    struct PendingArray {
        name: String,
        dt: XmlDataType,
        num_comp: usize,
        format: ArrayFormat,
        offset: usize,
        text_buf: String,
    }
    let mut current_array: Option<PendingArray> = None;

    // Topology buffers per piece
    let mut piece_verts: (Option<Vec<u32>>, Option<Vec<u32>>) = (None, None);
    let mut piece_lines: (Option<Vec<u32>>, Option<Vec<u32>>) = (None, None);
    let mut piece_strips: (Option<Vec<u32>>, Option<Vec<u32>>) = (None, None);
    let mut piece_polys: (Option<Vec<u32>>, Option<Vec<u32>>) = (None, None);
    let mut piece_cells: PieceCellsTuple = (None, None, None);
    let mut coord_arrays: Vec<DataArray> = Vec::new();
    let mut block_name_stack: Vec<Option<String>> = Vec::new();

    let mut piece_point_base: u32 = 0;

    loop {
        let ev = reader.read_event()?;
        match ev {
            Event::Start(e) => {
                match e.name().as_ref() {
                    "VTKFile" => {
                        for attr in e.attributes() {
                            let attr = attr?;
                            let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
                            match attr.key.as_ref() {
                                "type" => file_type = val.into_owned(),
                                "byte_order" => {
                                    byte_order = if val == "BigEndian" {
                                        ByteOrder::BigEndian
                                    } else {
                                        ByteOrder::LittleEndian
                                    };
                                }
                                "header_type" => {
                                    header_type = if val == "UInt64" {
                                        HeaderType::UInt64
                                    } else {
                                        HeaderType::UInt32
                                    };
                                }
                                "compressor" if val.contains("ZLib") => {
                                    compressed = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    "ImageData" | "PImageData" => {
                        let meta = parse_image_meta(&e)?;
                        image_meta = Some(meta);
                        image.dimensions = meta.0;
                        image.origin = meta.1;
                        image.spacing = meta.2;
                    }
                    "RectilinearGrid" | "StructuredGrid" => {
                        for attr in e.attributes() {
                            let attr = attr?;
                            let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
                            if attr.key.as_ref() == "WholeExtent" {
                                whole_extent = Some(parse_extent(&val)?);
                            }
                        }
                    }
                    "Piece" => {
                        if is_parallel_file_type(&file_type) {
                            if let Some(pp) = parse_parallel_piece(&e)? {
                                parallel_pieces.push(pp);
                            }
                        } else {
                            piece_verts = (None, None);
                            piece_lines = (None, None);
                            piece_strips = (None, None);
                            piece_polys = (None, None);
                            piece_cells = (None, None, None);
                            coord_arrays.clear();
                            piece_point_base = match file_type.as_str() {
                                "PolyData" => poly.points.len() as u32,
                                "UnstructuredGrid" => ugrid.points.len() as u32,
                                _ => 0,
                            };
                            for attr in e.attributes() {
                                let attr = attr?;
                                let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
                                if attr.key.as_ref() == "Extent" {
                                    let ext = parse_extent(&val)?;
                                    if file_type == "ImageData" && image_meta.is_none() {
                                        image.dimensions = extent_to_dims(ext);
                                    } else if file_type == "StructuredGrid" && whole_extent.is_none()
                                    {
                                        whole_extent = Some(ext);
                                    }
                                }
                            }
                        }
                    }
                    "PointData" | "PPointData" => {
                        section = XmlSection::PointData;
                        let (s, v, n) = parse_active_attrs(&e)?;
                        if s.is_some() {
                            active_point_attrs.0 = s;
                        }
                        if v.is_some() {
                            active_point_attrs.1 = v;
                        }
                        if n.is_some() {
                            active_point_attrs.2 = n;
                        }
                    }
                    "CellData" | "PCellData" => {
                        section = XmlSection::CellData;
                        let (s, v, n) = parse_active_attrs(&e)?;
                        if s.is_some() {
                            active_cell_attrs.0 = s;
                        }
                        if v.is_some() {
                            active_cell_attrs.1 = v;
                        }
                        if n.is_some() {
                            active_cell_attrs.2 = n;
                        }
                    }
                    "FieldData" => section = XmlSection::FieldData,
                    "Points" => section = XmlSection::Points,
                    "Verts" => section = XmlSection::Verts,
                    "Lines" => section = XmlSection::Lines,
                    "Strips" => section = XmlSection::Strips,
                    "Polys" => section = XmlSection::Polys,
                    "Cells" => section = XmlSection::Cells,
                    "Coordinates" => section = XmlSection::Coordinates,
                    "Block" => {
                        let name = attr_str(&e, "name")?;
                        block_name_stack.push(name);
                    }
                    "DataSet" => {
                        handle_dataset_ref(
                            &e,
                            &file_type,
                            base_dir,
                            &block_name_stack,
                            &mut multiblock,
                            &mut pvd_entries,
                        )?;
                    }
                    "DataArray" => {
                        let pa = parse_data_array_start(&e)?;
                        current_array = Some(PendingArray {
                            name: pa.0,
                            dt: pa.1,
                            num_comp: pa.2,
                            format: pa.3,
                            offset: pa.4,
                            text_buf: String::new(),
                        });
                    }
                    _ => {}
                }
            }
            Event::Empty(e) => {
                match e.name().as_ref() {
                    "Piece" => {
                        if is_parallel_file_type(&file_type)
                            && let Some(pp) = parse_parallel_piece(&e)?
                        {
                            parallel_pieces.push(pp);
                        }
                    }
                    "PointData" | "PPointData" => {
                        let (s, v, n) = parse_active_attrs(&e)?;
                        if s.is_some() {
                            active_point_attrs.0 = s;
                        }
                        if v.is_some() {
                            active_point_attrs.1 = v;
                        }
                        if n.is_some() {
                            active_point_attrs.2 = n;
                        }
                    }
                    "CellData" | "PCellData" => {
                        let (s, v, n) = parse_active_attrs(&e)?;
                        if s.is_some() {
                            active_cell_attrs.0 = s;
                        }
                        if v.is_some() {
                            active_cell_attrs.1 = v;
                        }
                        if n.is_some() {
                            active_cell_attrs.2 = n;
                        }
                    }
                    "DataSet" => {
                        handle_dataset_ref(
                            &e,
                            &file_type,
                            base_dir,
                            &block_name_stack,
                            &mut multiblock,
                            &mut pvd_entries,
                        )?;
                    }
                    "DataArray" => {
                        let (name, dt, num_comp, format, offset) = parse_data_array_start(&e)?;
                        let arr = decode_data_array(
                            &name,
                            dt,
                            num_comp,
                            format,
                            offset,
                            "",
                            byte_order,
                            header_type,
                            compressed,
                            &extracted,
                        )?;
                        route_decoded_array(
                            arr,
                            section,
                            &file_type,
                            &mut poly,
                            &mut image,
                            &mut rect,
                            &mut sgrid,
                            &mut ugrid,
                            &mut piece_verts,
                            &mut piece_lines,
                            &mut piece_strips,
                            &mut piece_polys,
                            &mut piece_cells,
                            &mut coord_arrays,
                        )?;
                    }
                    _ => {}
                }
            }
            Event::Text(t) => {
                if let Some(pa) = &mut current_array {
                    pa.text_buf.push_str(&t);
                }
            }
            Event::End(e) => {
                match e.name().as_ref() {
                    "DataArray" => {
                        if let Some(pa) = current_array.take() {
                            let arr = decode_data_array(
                                &pa.name,
                                pa.dt,
                                pa.num_comp,
                                pa.format,
                                pa.offset,
                                &pa.text_buf,
                                byte_order,
                                header_type,
                                compressed,
                                &extracted,
                            )?;
                            route_decoded_array(
                                arr,
                                section,
                                &file_type,
                                &mut poly,
                                &mut image,
                                &mut rect,
                                &mut sgrid,
                                &mut ugrid,
                                &mut piece_verts,
                                &mut piece_lines,
                                &mut piece_strips,
                                &mut piece_polys,
                                &mut piece_cells,
                                &mut coord_arrays,
                            )?;
                        }
                    }
                    "PointData" | "CellData" | "FieldData" | "Points" | "Verts"
                    | "Lines" | "Strips" | "Polys" | "Cells" | "Coordinates" => {
                        section = XmlSection::None;
                    }
                    "Block" => {
                        block_name_stack.pop();
                    }
                    "Piece" => {
                        if file_type == "PolyData" {
                            if let (Some(conn), Some(offs)) =
                                (piece_verts.0.take(), piece_verts.1.take())
                            {
                                for cell in split_connectivity(&conn, &offs, piece_point_base) {
                                    poly.verts.extend(cell);
                                }
                            }
                            if let (Some(conn), Some(offs)) =
                                (piece_lines.0.take(), piece_lines.1.take())
                            {
                                poly.lines
                                    .extend(split_connectivity(&conn, &offs, piece_point_base));
                            }
                            if let (Some(conn), Some(offs)) =
                                (piece_strips.0.take(), piece_strips.1.take())
                            {
                                poly.strips
                                    .extend(split_connectivity(&conn, &offs, piece_point_base));
                            }
                            if let (Some(conn), Some(offs)) =
                                (piece_polys.0.take(), piece_polys.1.take())
                            {
                                poly.polys
                                    .extend(split_connectivity(&conn, &offs, piece_point_base));
                            }
                        } else if file_type == "UnstructuredGrid" {
                            if let (Some(conn), Some(offs), Some(types)) = (
                                piece_cells.0.take(),
                                piece_cells.1.take(),
                                piece_cells.2.take(),
                            ) {
                                for (i, cell) in
                                    split_connectivity(&conn, &offs, piece_point_base)
                                        .into_iter()
                                        .enumerate()
                                {
                                    let tid = *types.get(i).ok_or_else(|| {
                                        IoError::InvalidData(
                                            "VTU cell types shorter than offsets".into(),
                                        )
                                    })?;
                                    let ct = CellType::from_vtk_id(tid).ok_or_else(|| {
                                        IoError::UnsupportedFormat(format!(
                                            "Unsupported VTK cell type ID {tid}"
                                        ))
                                    })?;
                                    ugrid.push_cell(ct, &cell);
                                }
                            }
                        } else if file_type == "RectilinearGrid" {
                            if coord_arrays.len() >= 3 {
                                rect.x_coords = coord_arrays[0].data.to_f32_vec();
                                rect.y_coords = coord_arrays[1].data.to_f32_vec();
                                rect.z_coords = coord_arrays[2].data.to_f32_vec();
                            }
                        } else if file_type == "StructuredGrid"
                            && let Some(ext) = whole_extent
                        {
                            sgrid.dimensions = extent_to_dims(ext);
                        }
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }

    match file_type.as_str() {
        "PolyData" => {
            apply_active_attrs(&mut poly.point_data, active_point_attrs);
            apply_active_attrs(&mut poly.cell_data, active_cell_attrs);
            Ok(DataSet::PolyData(poly))
        }
        "ImageData" => {
            apply_active_attrs(&mut image.point_data, active_point_attrs);
            apply_active_attrs(&mut image.cell_data, active_cell_attrs);
            Ok(DataSet::ImageData(image))
        }
        "RectilinearGrid" => {
            apply_active_attrs(&mut rect.point_data, active_point_attrs);
            apply_active_attrs(&mut rect.cell_data, active_cell_attrs);
            Ok(DataSet::RectilinearGrid(rect))
        }
        "StructuredGrid" => {
            if let Some(ext) = whole_extent {
                sgrid.dimensions = extent_to_dims(ext);
            }
            apply_active_attrs(&mut sgrid.point_data, active_point_attrs);
            apply_active_attrs(&mut sgrid.cell_data, active_cell_attrs);
            Ok(DataSet::StructuredGrid(sgrid))
        }
        "UnstructuredGrid" => {
            apply_active_attrs(&mut ugrid.point_data, active_point_attrs);
            apply_active_attrs(&mut ugrid.cell_data, active_cell_attrs);
            Ok(DataSet::UnstructuredGrid(ugrid))
        }
        "vtkMultiBlockDataSet" => Ok(DataSet::MultiBlock(multiblock)),
        "Collection" => {
            pvd_entries.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut mb = MultiBlockDataSet::new();
            for (time, fp) in pvd_entries {
                if !fp.exists() {
                    return Err(IoError::MissingFile(fp));
                }
                let ds = crate::read_dataset(&fp)?;
                mb.add_block(format!("t={time}"), ds);
            }
            Ok(DataSet::MultiBlock(mb))
        }
        "PPolyData" => {
            read_pvtp(base_dir, &parallel_pieces, active_point_attrs, active_cell_attrs)
        }
        "PImageData" => read_pvti(
            base_dir,
            &parallel_pieces,
            image_meta,
            active_point_attrs,
            active_cell_attrs,
        ),
        "PUnstructuredGrid" => {
            read_pvtu(base_dir, &parallel_pieces, active_point_attrs, active_cell_attrs)
        }
        other => Err(IoError::UnsupportedFormat(format!(
            "Unsupported VTK XML type '{other}'"
        ))),
    }
}

struct ParallelPiece {
    source: String,
    extent: Option<[isize; 6]>,
}

fn is_parallel_file_type(ft: &str) -> bool {
    matches!(ft, "PPolyData" | "PImageData" | "PUnstructuredGrid")
}

type ImageMetaTuple = ([usize; 3], [f64; 3], [f64; 3], [isize; 6]);

fn parse_image_meta(e: &BytesStart<'_>) -> Result<ImageMetaTuple> {
    let mut ext = [0isize; 6];
    let mut origin = [0.0f64; 3];
    let mut spacing = [1.0f64; 3];
    for attr in e.attributes() {
        let attr = attr?;
        let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
        match attr.key.as_ref() {
            "WholeExtent" => ext = parse_extent(&val)?,
            "Origin" => origin = parse_vec3_f64(&val)?,
            "Spacing" => spacing = parse_vec3_f64(&val)?,
            _ => {}
        }
    }
    let dims = extent_to_dims(ext);
    let adj_origin = [
        origin[0] + ext[0] as f64 * spacing[0],
        origin[1] + ext[2] as f64 * spacing[1],
        origin[2] + ext[4] as f64 * spacing[2],
    ];
    Ok((dims, adj_origin, spacing, ext))
}

fn parse_parallel_piece(e: &BytesStart<'_>) -> Result<Option<ParallelPiece>> {
    let mut source = None;
    let mut extent = None;
    for attr in e.attributes() {
        let attr = attr?;
        let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
        match attr.key.as_ref() {
            "Source" | "source" => source = Some(val.into_owned()),
            "Extent" => extent = Some(parse_extent(&val)?),
            _ => {}
        }
    }
    Ok(source.map(|source| ParallelPiece { source, extent }))
}

fn load_parallel_pieces(base_dir: &Path, pieces: &[ParallelPiece]) -> Result<Vec<DataSet>> {
    let mut out = Vec::with_capacity(pieces.len());
    for p in pieces {
        let path = base_dir.join(&p.source);
        if !path.exists() {
            return Err(IoError::MissingFile(path));
        }
        out.push(crate::read_dataset(&path)?);
    }
    Ok(out)
}

fn read_pvtp(
    base_dir: &Path,
    pieces: &[ParallelPiece],
    active_point_attrs: (Option<String>, Option<String>, Option<String>),
    active_cell_attrs: (Option<String>, Option<String>, Option<String>),
) -> Result<DataSet> {
    let loaded = load_parallel_pieces(base_dir, pieces)?;
    let mut polys = Vec::with_capacity(loaded.len());
    for (idx, ds) in loaded.into_iter().enumerate() {
        match ds {
            DataSet::PolyData(p) => polys.push(p),
            other => {
                return Err(IoError::InvalidData(format!(
                    "PVTP piece {idx} expected PolyData, got {}",
                    other.type_name()
                )));
            }
        }
    }

    let mut merged = PolyData::new();
    for p in &polys {
        let base = merged.points.len() as u32;
        merged.points.extend_from_slice(&p.points);
        merged.verts.extend(p.verts.iter().map(|&v| v + base));
        merged
            .lines
            .extend(p.lines.iter().map(|c| c.iter().map(|&v| v + base).collect()));
        merged
            .strips
            .extend(p.strips.iter().map(|c| c.iter().map(|&v| v + base).collect()));
        merged
            .polys
            .extend(p.polys.iter().map(|c| c.iter().map(|&v| v + base).collect()));
    }

    let pt_fields: Vec<&FieldData> = polys.iter().map(|p| &p.point_data).collect();
    let cell_fields: Vec<&FieldData> = polys.iter().map(|p| &p.cell_data).collect();
    merged.point_data = merge_field_data(&pt_fields, active_point_attrs);
    merged.cell_data = merge_field_data(&cell_fields, active_cell_attrs);
    if let Some(first) = polys.first() {
        merged.field_data = first.field_data.clone();
    }

    Ok(DataSet::PolyData(merged))
}

fn read_pvtu(
    base_dir: &Path,
    pieces: &[ParallelPiece],
    active_point_attrs: (Option<String>, Option<String>, Option<String>),
    active_cell_attrs: (Option<String>, Option<String>, Option<String>),
) -> Result<DataSet> {
    let loaded = load_parallel_pieces(base_dir, pieces)?;
    let mut grids = Vec::with_capacity(loaded.len());
    for (idx, ds) in loaded.into_iter().enumerate() {
        match ds {
            DataSet::UnstructuredGrid(u) => grids.push(u),
            other => {
                return Err(IoError::InvalidData(format!(
                    "PVTU piece {idx} expected UnstructuredGrid, got {}",
                    other.type_name()
                )));
            }
        }
    }

    let mut merged = UnstructuredGrid::new();
    for ug in &grids {
        let base = merged.points.len() as u32;
        merged.points.extend_from_slice(&ug.points);
        for c in 0..ug.num_cells() {
            let shifted: Vec<u32> = ug.cell_points(c).iter().map(|&i| i + base).collect();
            merged.push_cell(ug.cell_types[c], &shifted);
        }
    }

    let pt_fields: Vec<&FieldData> = grids.iter().map(|g| &g.point_data).collect();
    let cell_fields: Vec<&FieldData> = grids.iter().map(|g| &g.cell_data).collect();
    merged.point_data = merge_field_data(&pt_fields, active_point_attrs);
    merged.cell_data = merge_field_data(&cell_fields, active_cell_attrs);
    if let Some(first) = grids.first() {
        merged.field_data = first.field_data.clone();
    }

    Ok(DataSet::UnstructuredGrid(merged))
}

fn read_pvti(
    base_dir: &Path,
    pieces: &[ParallelPiece],
    image_meta: Option<ImageMetaTuple>,
    active_point_attrs: (Option<String>, Option<String>, Option<String>),
    active_cell_attrs: (Option<String>, Option<String>, Option<String>),
) -> Result<DataSet> {
    let loaded = load_parallel_pieces(base_dir, pieces)?;
    if let Some((whole_dims, adj_origin, spacing, whole_ext)) = image_meta
        && let Some(merged) = try_merge_pvti_images(
            pieces,
            &loaded,
            whole_dims,
            adj_origin,
            spacing,
            whole_ext,
            &active_point_attrs,
            &active_cell_attrs,
        )
    {
        return Ok(DataSet::ImageData(merged));
    }

    let mut mb = MultiBlockDataSet::new();
    for (idx, ds) in loaded.into_iter().enumerate() {
        let label = pieces
            .get(idx)
            .map(|p| p.source.clone())
            .unwrap_or_else(|| format!("piece_{idx}"));
        mb.add_block(label, ds);
    }
    Ok(DataSet::MultiBlock(mb))
}

#[allow(clippy::too_many_arguments)]
fn try_merge_pvti_images(
    pieces: &[ParallelPiece],
    loaded: &[DataSet],
    whole_dims: [usize; 3],
    adj_origin: [f64; 3],
    spacing: [f64; 3],
    whole_ext: [isize; 6],
    active_point_attrs: &(Option<String>, Option<String>, Option<String>),
    active_cell_attrs: &(Option<String>, Option<String>, Option<String>),
) -> Option<ImageData> {
    if loaded.is_empty() || whole_dims.contains(&0) {
        return None;
    }

    let raw_origin = [
        adj_origin[0] - whole_ext[0] as f64 * spacing[0],
        adj_origin[1] - whole_ext[2] as f64 * spacing[1],
        adj_origin[2] - whole_ext[4] as f64 * spacing[2],
    ];

    let mut piece_images = Vec::with_capacity(loaded.len());
    let mut piece_exts = Vec::with_capacity(loaded.len());

    for (idx, ds) in loaded.iter().enumerate() {
        let DataSet::ImageData(img) = ds else {
            return None;
        };
        let ext = match pieces.get(idx).and_then(|p| p.extent) {
            Some(e) => e,
            None => {
                let mut e = [0isize; 6];
                for axis in 0..3 {
                    let sp = spacing[axis];
                    if sp.abs() < 1e-15 {
                        return None;
                    }
                    let start = ((img.origin[axis] - raw_origin[axis]) / sp).round() as isize;
                    e[2 * axis] = start;
                    e[2 * axis + 1] = start + img.dimensions[axis] as isize - 1;
                }
                e
            }
        };
        if ext[0] < whole_ext[0]
            || ext[1] > whole_ext[1]
            || ext[2] < whole_ext[2]
            || ext[3] > whole_ext[3]
            || ext[4] < whole_ext[4]
            || ext[5] > whole_ext[5]
        {
            return None;
        }
        piece_images.push(img);
        piece_exts.push(ext);
    }

    let mut merged = ImageData::new(whole_dims, adj_origin, spacing);
    let pt_fields: Vec<&FieldData> = piece_images.iter().map(|i| &i.point_data).collect();
    let pt_piece_dims: Vec<[usize; 3]> = piece_exts.iter().map(|&e| extent_to_dims(e)).collect();
    let pt_offsets: Vec<[usize; 3]> = piece_exts
        .iter()
        .map(|e| {
            [
                (e[0] - whole_ext[0]) as usize,
                (e[2] - whole_ext[2]) as usize,
                (e[4] - whole_ext[4]) as usize,
            ]
        })
        .collect();
    merged.point_data = scatter_image_field(
        &pt_fields,
        &pt_piece_dims,
        &pt_offsets,
        whole_dims,
        active_point_attrs.clone(),
    )?;

    let whole_cell_dims = [
        (whole_ext[1] - whole_ext[0]).max(1) as usize,
        (whole_ext[3] - whole_ext[2]).max(1) as usize,
        (whole_ext[5] - whole_ext[4]).max(1) as usize,
    ];
    let cell_fields: Vec<&FieldData> = piece_images.iter().map(|i| &i.cell_data).collect();
    let cell_piece_dims: Vec<[usize; 3]> = piece_exts
        .iter()
        .map(|e| {
            [
                (e[1] - e[0]).max(1) as usize,
                (e[3] - e[2]).max(1) as usize,
                (e[5] - e[4]).max(1) as usize,
            ]
        })
        .collect();
    merged.cell_data = scatter_image_field(
        &cell_fields,
        &cell_piece_dims,
        &pt_offsets,
        whole_cell_dims,
        active_cell_attrs.clone(),
    )?;

    if let Some(first) = piece_images.first() {
        merged.field_data = first.field_data.clone();
    }
    Some(merged)
}

fn scatter_image_field(
    fields: &[&FieldData],
    piece_dims: &[[usize; 3]],
    piece_offsets: &[[usize; 3]],
    whole_dims: [usize; 3],
    active_override: (Option<String>, Option<String>, Option<String>),
) -> Option<FieldData> {
    let mut names: Vec<String> = Vec::new();
    for f in fields {
        for a in f.arrays() {
            if !names.contains(&a.name) {
                names.push(a.name.clone());
            }
        }
    }

    let mut out = FieldData::new();
    if let Some(first) = fields.first() {
        out.active_scalars = first.active_scalars.clone();
        out.active_vectors = first.active_vectors.clone();
        out.active_normals = first.active_normals.clone();
    }
    apply_active_attrs(&mut out, active_override);

    let total_tuples = whole_dims[0] * whole_dims[1] * whole_dims[2];
    for name in names {
        let proto = fields.iter().find_map(|f| f.get(&name))?;
        let nc = proto.num_components;
        let mut buf = allocate_zeroed_array_data(proto.data.scalar_type(), total_tuples * nc);

        for (p_idx, f) in fields.iter().enumerate() {
            let Some(arr) = f.get(&name) else {
                continue;
            };
            let [pnx, pny, pnz] = piece_dims[p_idx];
            if arr.num_tuples() != pnx * pny * pnz || arr.num_components != nc {
                return None;
            }
            let [ox, oy, oz] = piece_offsets[p_idx];
            for lk in 0..pnz {
                for lj in 0..pny {
                    for li in 0..pnx {
                        let src_t = li + pnx * (lj + pny * lk);
                        let dst_t = (ox + li) + whole_dims[0] * ((oy + lj) + whole_dims[1] * (oz + lk));
                        copy_tuple(&mut buf, dst_t, &arr.data, src_t, nc);
                    }
                }
            }
        }
        out.add_array(DataArray::new(name, nc, buf));
    }

    Some(out)
}

fn merge_field_data(
    fields: &[&FieldData],
    active_override: (Option<String>, Option<String>, Option<String>),
) -> FieldData {
    let mut names: Vec<String> = Vec::new();
    for f in fields {
        for a in f.arrays() {
            if !names.contains(&a.name) {
                names.push(a.name.clone());
            }
        }
    }

    let mut out = FieldData::new();
    if let Some(first) = fields.first() {
        out.active_scalars = first.active_scalars.clone();
        out.active_vectors = first.active_vectors.clone();
        out.active_normals = first.active_normals.clone();
    }
    apply_active_attrs(&mut out, active_override);

    for name in names {
        let Some(proto) = fields.iter().find_map(|f| f.get(&name)) else {
            continue;
        };
        let nc = proto.num_components;
        let mut merged_data = allocate_zeroed_array_data(proto.data.scalar_type(), 0);
        for f in fields {
            if let Some(arr) = f.get(&name) {
                append_array_data(&mut merged_data, &arr.data);
            }
        }
        out.add_array(DataArray::new(name, nc, merged_data));
    }
    out
}

fn allocate_zeroed_array_data(st: ScalarType, len: usize) -> ArrayData {
    match st {
        ScalarType::F32 => ArrayData::F32(vec![0.0; len]),
        ScalarType::F64 => ArrayData::F64(vec![0.0; len]),
        ScalarType::I32 => ArrayData::I32(vec![0; len]),
        ScalarType::U32 => ArrayData::U32(vec![0; len]),
        ScalarType::I64 => ArrayData::I64(vec![0; len]),
        ScalarType::U8 => ArrayData::U8(vec![0; len]),
    }
}

fn append_array_data(dst: &mut ArrayData, src: &ArrayData) {
    match (dst, src) {
        (ArrayData::F32(d), ArrayData::F32(s)) => d.extend_from_slice(s),
        (ArrayData::F64(d), ArrayData::F64(s)) => d.extend_from_slice(s),
        (ArrayData::I32(d), ArrayData::I32(s)) => d.extend_from_slice(s),
        (ArrayData::U32(d), ArrayData::U32(s)) => d.extend_from_slice(s),
        (ArrayData::I64(d), ArrayData::I64(s)) => d.extend_from_slice(s),
        (ArrayData::U8(d), ArrayData::U8(s)) => d.extend_from_slice(s),
        (dst_any, src_any) => {
            let mut promoted = dst_any.to_f64_vec();
            promoted.extend(src_any.to_f64_vec());
            *dst_any = ArrayData::F64(promoted);
        }
    }
}

fn copy_tuple(dst: &mut ArrayData, dst_t: usize, src: &ArrayData, src_t: usize, nc: usize) {
    let db = dst_t * nc;
    let sb = src_t * nc;
    match (dst, src) {
        (ArrayData::F32(d), ArrayData::F32(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (ArrayData::F64(d), ArrayData::F64(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (ArrayData::I32(d), ArrayData::I32(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (ArrayData::U32(d), ArrayData::U32(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (ArrayData::I64(d), ArrayData::I64(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (ArrayData::U8(d), ArrayData::U8(s)) => d[db..db + nc].copy_from_slice(&s[sb..sb + nc]),
        (dst_any, src_any) => {
            let mut promoted = dst_any.to_f64_vec();
            for c in 0..nc {
                promoted[db + c] = src_any.get_f64(sb + c);
            }
            *dst_any = ArrayData::F64(promoted);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_dataset_ref(
    e: &BytesStart<'_>,
    file_type: &str,
    base_dir: &Path,
    block_name_stack: &[Option<String>],
    multiblock: &mut MultiBlockDataSet,
    pvd_entries: &mut Vec<(f64, PathBuf)>,
) -> Result<()> {
    let mut name = None;
    let mut file = None;
    let mut timestep = 0.0f64;
    let mut index = None;

    for attr in e.attributes() {
        let attr = attr?;
        let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
        match attr.key.as_ref() {
            "name" => name = Some(val.into_owned()),
            "file" => file = Some(val.into_owned()),
            "index" => index = Some(val.into_owned()),
            "timestep" => {
                timestep = val
                    .parse::<f64>()
                    .map_err(|_| IoError::Parse(format!("Invalid PVD timestep '{val}'")))?;
            }
            _ => {}
        }
    }

    let Some(rel_file) = file else {
        return Ok(());
    };
    let resolved = base_dir.join(&rel_file);

    if file_type == "vtkMultiBlockDataSet" {
        if !resolved.exists() {
            return Err(IoError::MissingFile(resolved));
        }
        let ds = crate::read_dataset(&resolved)?;
        let parent_name = block_name_stack
            .iter()
            .rev()
            .find_map(|o| o.as_ref())
            .cloned();
        let block_label = match (parent_name, name) {
            (Some(p), Some(n)) if !n.is_empty() && n != p => format!("{p}/{n}"),
            (_, Some(n)) if !n.is_empty() => n,
            (Some(p), _) if !p.is_empty() => p,
            _ => format!(
                "Block_{}",
                index.unwrap_or_else(|| multiblock.blocks.len().to_string())
            ),
        };
        multiblock.add_block(block_label, ds);
    } else if file_type == "Collection" {
        pvd_entries.push((timestep, resolved));
    }
    Ok(())
}

fn apply_active_attrs(
    field: &mut FieldData,
    active: (Option<String>, Option<String>, Option<String>),
) {
    if let Some(s) = active.0
        && !s.is_empty()
    {
        field.active_scalars = Some(s);
    }
    if let Some(v) = active.1
        && !v.is_empty()
    {
        field.active_vectors = Some(v);
    }
    if let Some(n) = active.2
        && !n.is_empty()
    {
        field.active_normals = Some(n);
    }
}

type PieceCells = (Option<Vec<u32>>, Option<Vec<u32>>, Option<Vec<u8>>);

#[allow(clippy::too_many_arguments)]
fn route_decoded_array(
    arr: DataArray,
    section: XmlSection,
    file_type: &str,
    poly: &mut PolyData,
    image: &mut ImageData,
    rect: &mut RectilinearGrid,
    sgrid: &mut StructuredGrid,
    ugrid: &mut UnstructuredGrid,
    piece_verts: &mut (Option<Vec<u32>>, Option<Vec<u32>>),
    piece_lines: &mut (Option<Vec<u32>>, Option<Vec<u32>>),
    piece_strips: &mut (Option<Vec<u32>>, Option<Vec<u32>>),
    piece_polys: &mut (Option<Vec<u32>>, Option<Vec<u32>>),
    piece_cells: &mut PieceCells,
    coord_arrays: &mut Vec<DataArray>,
) -> Result<()> {
    match section {
        XmlSection::PointData => match file_type {
            "PolyData" => poly.point_data.add_array(arr),
            "ImageData" => image.point_data.add_array(arr),
            "RectilinearGrid" => rect.point_data.add_array(arr),
            "StructuredGrid" => sgrid.point_data.add_array(arr),
            "UnstructuredGrid" => ugrid.point_data.add_array(arr),
            _ => {}
        },
        XmlSection::CellData => match file_type {
            "PolyData" => poly.cell_data.add_array(arr),
            "ImageData" => image.cell_data.add_array(arr),
            "RectilinearGrid" => rect.cell_data.add_array(arr),
            "StructuredGrid" => sgrid.cell_data.add_array(arr),
            "UnstructuredGrid" => ugrid.cell_data.add_array(arr),
            _ => {}
        },
        XmlSection::FieldData => match file_type {
            "PolyData" => poly.field_data.add_array(arr),
            "ImageData" => image.field_data.add_array(arr),
            "RectilinearGrid" => rect.field_data.add_array(arr),
            "StructuredGrid" => sgrid.field_data.add_array(arr),
            "UnstructuredGrid" => ugrid.field_data.add_array(arr),
            _ => {}
        },
        XmlSection::Points => {
            let flat = arr.data.to_f32_vec();
            let pts: Vec<[f32; 3]> = flat.as_chunks::<3>().0.to_vec();
            match file_type {
                "PolyData" => poly.points.extend(pts),
                "StructuredGrid" => sgrid.points.extend(pts),
                "UnstructuredGrid" => ugrid.points.extend(pts),
                _ => {}
            }
        }
        XmlSection::Verts => match arr.name.as_str() {
            "connectivity" => piece_verts.0 = Some(array_to_u32(&arr)),
            "offsets" => piece_verts.1 = Some(array_to_u32(&arr)),
            _ => {}
        },
        XmlSection::Lines => match arr.name.as_str() {
            "connectivity" => piece_lines.0 = Some(array_to_u32(&arr)),
            "offsets" => piece_lines.1 = Some(array_to_u32(&arr)),
            _ => {}
        },
        XmlSection::Strips => match arr.name.as_str() {
            "connectivity" => piece_strips.0 = Some(array_to_u32(&arr)),
            "offsets" => piece_strips.1 = Some(array_to_u32(&arr)),
            _ => {}
        },
        XmlSection::Polys => match arr.name.as_str() {
            "connectivity" => piece_polys.0 = Some(array_to_u32(&arr)),
            "offsets" => piece_polys.1 = Some(array_to_u32(&arr)),
            _ => {}
        },
        XmlSection::Cells => match arr.name.as_str() {
            "connectivity" => piece_cells.0 = Some(array_to_u32(&arr)),
            "offsets" => piece_cells.1 = Some(array_to_u32(&arr)),
            "types" => {
                piece_cells.2 = Some(
                    (0..arr.data.len())
                        .map(|i| arr.data.get_f64(i) as u8)
                        .collect(),
                )
            }
            _ => {}
        },
        XmlSection::Coordinates => coord_arrays.push(arr),
        XmlSection::None => {}
    }
    Ok(())
}

fn array_to_u32(arr: &DataArray) -> Vec<u32> {
    match &arr.data {
        ArrayData::U32(v) => v.clone(),
        ArrayData::I32(v) => v.iter().map(|&x| x as u32).collect(),
        ArrayData::I64(v) => v.iter().map(|&x| x as u32).collect(),
        ArrayData::U8(v) => v.iter().map(|&x| x as u32).collect(),
        ArrayData::F32(v) => v.iter().map(|&x| x as u32).collect(),
        ArrayData::F64(v) => v.iter().map(|&x| x as u32).collect(),
    }
}

fn split_connectivity(conn: &[u32], offsets: &[u32], point_base: u32) -> Vec<Vec<u32>> {
    let mut cells = Vec::with_capacity(offsets.len());
    let mut prev = 0usize;
    for &end_u32 in offsets {
        let end = (end_u32 as usize).min(conn.len());
        if end >= prev {
            cells.push(conn[prev..end].iter().map(|&i| i + point_base).collect());
        } else {
            cells.push(Vec::new());
        }
        prev = end;
    }
    cells
}

fn extract_appended_data(bytes: &[u8]) -> ExtractedAppended<'_> {
    let open_pos = find_subslice(bytes, b"<AppendedData");
    let Some(open_pos) = open_pos else {
        return ExtractedAppended {
            xml_bytes: bytes.to_vec(),
            encoding: AppendedEncoding::Raw,
            data: &[],
        };
    };
    let Some(gt_rel) = bytes[open_pos..].iter().position(|&b| b == b'>') else {
        return ExtractedAppended {
            xml_bytes: bytes.to_vec(),
            encoding: AppendedEncoding::Raw,
            data: &[],
        };
    };
    let tag_end = open_pos + gt_rel + 1;
    let tag_str = String::from_utf8_lossy(&bytes[open_pos..tag_end]);
    let encoding = if tag_str.contains("base64") {
        AppendedEncoding::Base64
    } else {
        AppendedEncoding::Raw
    };

    let Some(close_pos) = rfind_subslice(bytes, b"</AppendedData>") else {
        return ExtractedAppended {
            xml_bytes: bytes.to_vec(),
            encoding,
            data: &[],
        };
    };

    let data_start = bytes[tag_end..close_pos]
        .iter()
        .position(|&b| b == b'_')
        .map(|rel| tag_end + rel + 1)
        .unwrap_or(tag_end);

    let appended_slice = if data_start <= close_pos {
        &bytes[data_start..close_pos]
    } else {
        &[]
    };

    let mut clean_xml = Vec::with_capacity(bytes.len() - (close_pos - tag_end));
    clean_xml.extend_from_slice(&bytes[..tag_end]);
    clean_xml.extend_from_slice(&bytes[close_pos..]);

    ExtractedAppended {
        xml_bytes: clean_xml,
        encoding,
        data: appended_slice,
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn rfind_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

fn attr_str(e: &BytesStart<'_>, key: &str) -> Result<Option<String>> {
    for attr in e.attributes() {
        let attr = attr?;
        if attr.key.as_ref() == key {
            return Ok(Some(
                attr.normalized_value(XmlVersion::Implicit1_0)?.into_owned(),
            ));
        }
    }
    Ok(None)
}

fn parse_active_attrs(
    e: &BytesStart<'_>,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    let mut scalars = None;
    let mut vectors = None;
    let mut normals = None;
    for attr in e.attributes() {
        let attr = attr?;
        let val = attr.normalized_value(XmlVersion::Implicit1_0)?.into_owned();
        match attr.key.as_ref() {
            "Scalars" => scalars = Some(val),
            "Vectors" => vectors = Some(val),
            "Normals" => normals = Some(val),
            _ => {}
        }
    }
    Ok((scalars, vectors, normals))
}

fn parse_data_array_start(
    e: &BytesStart<'_>,
) -> Result<(String, XmlDataType, usize, ArrayFormat, usize)> {
    let mut name = String::new();
    let mut dt = XmlDataType::Float32;
    let mut num_comp = 1usize;
    let mut format = ArrayFormat::Ascii;
    let mut offset = 0usize;

    for attr in e.attributes() {
        let attr = attr?;
        let val = attr.normalized_value(XmlVersion::Implicit1_0)?;
        match attr.key.as_ref() {
            "Name" => name = val.into_owned(),
            "type" => dt = XmlDataType::from_name(&val)?,
            "NumberOfComponents" => {
                num_comp = val.parse::<usize>().map_err(|_| {
                    IoError::Parse(format!("Invalid NumberOfComponents '{val}'"))
                })?;
            }
            "format" => {
                format = match val.as_ref() {
                    "ascii" => ArrayFormat::Ascii,
                    "binary" => ArrayFormat::Binary,
                    "appended" => ArrayFormat::Appended,
                    other => {
                        return Err(IoError::UnsupportedFormat(format!(
                            "Unsupported VTK XML DataArray format '{other}'"
                        )))
                    }
                };
            }
            "offset" => {
                offset = val
                    .parse::<usize>()
                    .map_err(|_| IoError::Parse(format!("Invalid appended offset '{val}'")))?;
            }
            _ => {}
        }
    }

    Ok((name, dt, num_comp.max(1), format, offset))
}

#[allow(clippy::too_many_arguments)]
fn decode_data_array(
    name: &str,
    dt: XmlDataType,
    num_comp: usize,
    format: ArrayFormat,
    offset: usize,
    text: &str,
    byte_order: ByteOrder,
    header_type: HeaderType,
    compressed: bool,
    extracted: &ExtractedAppended<'_>,
) -> Result<DataArray> {
    let data = match format {
        ArrayFormat::Ascii => decode_ascii_values(text, dt)?,
        ArrayFormat::Binary => {
            let decoded = decode_vtk_base64_stream(text)?;
            let raw = decode_vtk_binary_block(&decoded, byte_order, header_type, compressed)?;
            bytes_to_array_data(&raw, dt, byte_order)?
        }
        ArrayFormat::Appended => match extracted.encoding {
            AppendedEncoding::Raw => {
                if offset > extracted.data.len() {
                    return Err(IoError::InvalidData(format!(
                        "Appended offset {offset} out of bounds (len {})",
                        extracted.data.len()
                    )));
                }
                let raw = decode_vtk_binary_block(
                    &extracted.data[offset..],
                    byte_order,
                    header_type,
                    compressed,
                )?;
                bytes_to_array_data(&raw, dt, byte_order)?
            }
            AppendedEncoding::Base64 => {
                if offset > extracted.data.len() {
                    return Err(IoError::InvalidData(format!(
                        "Appended base64 offset {offset} out of bounds (len {})",
                        extracted.data.len()
                    )));
                }
                let raw = decode_appended_base64_block(
                    &extracted.data[offset..],
                    byte_order,
                    header_type,
                    compressed,
                )?;
                bytes_to_array_data(&raw, dt, byte_order)?
            }
        },
    };
    Ok(DataArray::new(name, num_comp, data))
}

/// Decodes VTK inline base64 text, handling both a single continuous base64 stream and
/// concatenated base64 chunks (where `=` padding appears mid-stream between header and payload).
fn decode_vtk_base64_stream(text: &str) -> Result<Vec<u8>> {
    let cleaned: String = text.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }
    let bytes = cleaned.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let quad = &bytes[i..i + 4];
        if quad.contains(&b'=') {
            let chunk = &cleaned[start..i + 4];
            out.extend_from_slice(&BASE64.decode(chunk)?);
            start = i + 4;
        }
        i += 4;
    }
    if start < bytes.len() {
        out.extend_from_slice(&BASE64.decode(&cleaned[start..])?);
    }
    Ok(out)
}

fn decode_appended_base64_block(
    slice: &[u8],
    byte_order: ByteOrder,
    header_type: HeaderType,
    compressed: bool,
) -> Result<Vec<u8>> {
    let s = std::str::from_utf8(slice)
        .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in base64 AppendedData: {e}")))?;
    // In base64 AppendedData, VTK writes base64(header) followed by base64(payload),
    // possibly followed immediately by the next array's base64(header) without separator.
    let word_sz: usize = match header_type {
        HeaderType::UInt32 => 4,
        HeaderType::UInt64 => 8,
    };

    let b64_only: Vec<u8> = s
        .bytes()
        .filter(|b| b.is_ascii_alphanumeric() || *b == b'+' || *b == b'/' || *b == b'=')
        .collect();

    if !compressed {
        let hdr_b64_len = word_sz.div_ceil(3) * 4;
        if b64_only.len() < hdr_b64_len {
            return Err(IoError::InvalidData(
                "Truncated base64 appended header".into(),
            ));
        }
        let hdr_slice = &b64_only[..hdr_b64_len];
        let hdr_bytes = BASE64.decode(hdr_slice)?;
        let payload_len = read_header_word(&hdr_bytes, 0, byte_order, header_type)?;
        if hdr_slice.contains(&b'=') {
            let pay_b64_len = payload_len.div_ceil(3) * 4;
            if hdr_b64_len + pay_b64_len > b64_only.len() {
                return Err(IoError::InvalidData(
                    "Truncated base64 appended payload".into(),
                ));
            }
            let payload = BASE64.decode(&b64_only[hdr_b64_len..hdr_b64_len + pay_b64_len])?;
            Ok(payload[..payload_len.min(payload.len())].to_vec())
        } else {
            let total_bytes = word_sz + payload_len;
            let total_b64_len = total_bytes.div_ceil(3) * 4;
            if total_b64_len > b64_only.len() {
                return Err(IoError::InvalidData(
                    "Truncated base64 appended stream".into(),
                ));
            }
            let all = BASE64.decode(&b64_only[..total_b64_len])?;
            Ok(all[word_sz..(word_sz + payload_len).min(all.len())].to_vec())
        }
    } else {
        // Compressed header: 3 words + num_blocks words
        let fixed_hdr_b64_len = (3 * word_sz).div_ceil(3) * 4;
        if b64_only.len() < fixed_hdr_b64_len {
            return Err(IoError::InvalidData(
                "Truncated compressed base64 appended header".into(),
            ));
        }
        let fixed_hdr = BASE64.decode(&b64_only[..fixed_hdr_b64_len])?;
        let num_blocks = read_header_word(&fixed_hdr, 0, byte_order, header_type)?;
        let total_hdr_bytes = (3 + num_blocks) * word_sz;
        let total_hdr_b64_len = total_hdr_bytes.div_ceil(3) * 4;
        if total_hdr_b64_len > b64_only.len() {
            return Err(IoError::InvalidData(
                "Truncated compressed base64 block table".into(),
            ));
        }
        let full_hdr_slice = &b64_only[..total_hdr_b64_len];
        let full_hdr = BASE64.decode(full_hdr_slice)?;
        let mut comp_total = 0usize;
        for b in 0..num_blocks {
            comp_total += read_header_word(&full_hdr, (3 + b) * word_sz, byte_order, header_type)?;
        }
        if full_hdr_slice.contains(&b'=') {
            let pay_b64_len = comp_total.div_ceil(3) * 4;
            let mut combined = Vec::with_capacity(total_hdr_bytes + comp_total);
            combined.extend_from_slice(&full_hdr[..total_hdr_bytes]);
            if pay_b64_len > 0 {
                let pay =
                    BASE64.decode(&b64_only[total_hdr_b64_len..total_hdr_b64_len + pay_b64_len])?;
                combined.extend_from_slice(&pay[..comp_total.min(pay.len())]);
            }
            decode_vtk_binary_block(&combined, byte_order, header_type, true)
        } else {
            let total_b64_len = (total_hdr_bytes + comp_total).div_ceil(3) * 4;
            if total_b64_len > b64_only.len() {
                return Err(IoError::InvalidData(
                    "Truncated compressed base64 appended stream".into(),
                ));
            }
            let combined = BASE64.decode(&b64_only[..total_b64_len])?;
            decode_vtk_binary_block(&combined, byte_order, header_type, true)
        }
    }
}

fn decode_vtk_binary_block(
    bytes: &[u8],
    byte_order: ByteOrder,
    header_type: HeaderType,
    compressed: bool,
) -> Result<Vec<u8>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let word_sz: usize = match header_type {
        HeaderType::UInt32 => 4,
        HeaderType::UInt64 => 8,
    };
    if !compressed {
        if bytes.len() < word_sz {
            return Err(IoError::InvalidData("VTK binary block missing header".into()));
        }
        let payload_len = read_header_word(bytes, 0, byte_order, header_type)?;
        if word_sz + payload_len > bytes.len() {
            return Err(IoError::InvalidData(format!(
                "VTK binary payload truncated: need {}, got {}",
                word_sz + payload_len,
                bytes.len()
            )));
        }
        Ok(bytes[word_sz..word_sz + payload_len].to_vec())
    } else {
        if bytes.len() < 3 * word_sz {
            return Err(IoError::InvalidData(
                "VTK compressed block missing 3-word header".into(),
            ));
        }
        let num_blocks = read_header_word(bytes, 0, byte_order, header_type)?;
        let _block_size = read_header_word(bytes, word_sz, byte_order, header_type)?;
        let _last_block_size = read_header_word(bytes, 2 * word_sz, byte_order, header_type)?;
        let header_end = (3 + num_blocks) * word_sz;
        if bytes.len() < header_end {
            return Err(IoError::InvalidData(
                "VTK compressed block header truncated".into(),
            ));
        }
        let mut out = Vec::new();
        let mut cursor = header_end;
        for b in 0..num_blocks {
            let c_len = read_header_word(bytes, (3 + b) * word_sz, byte_order, header_type)?;
            if cursor + c_len > bytes.len() {
                return Err(IoError::InvalidData(
                    "VTK zlib compressed block truncated".into(),
                ));
            }
            let mut decoder = ZlibDecoder::new(&bytes[cursor..cursor + c_len]);
            decoder.read_to_end(&mut out)?;
            cursor += c_len;
        }
        Ok(out)
    }
}

fn read_header_word(
    bytes: &[u8],
    offset: usize,
    byte_order: ByteOrder,
    header_type: HeaderType,
) -> Result<usize> {
    match header_type {
        HeaderType::UInt32 => {
            if offset + 4 > bytes.len() {
                return Err(IoError::InvalidData("Truncated UInt32 header word".into()));
            }
            let s = &bytes[offset..offset + 4];
            let v = match byte_order {
                ByteOrder::LittleEndian => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                ByteOrder::BigEndian => u32::from_be_bytes([s[0], s[1], s[2], s[3]]),
            };
            Ok(v as usize)
        }
        HeaderType::UInt64 => {
            if offset + 8 > bytes.len() {
                return Err(IoError::InvalidData("Truncated UInt64 header word".into()));
            }
            let s = &bytes[offset..offset + 8];
            let v = match byte_order {
                ByteOrder::LittleEndian => {
                    u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                }
                ByteOrder::BigEndian => {
                    u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                }
            };
            Ok(v as usize)
        }
    }
}

fn bytes_to_array_data(raw: &[u8], dt: XmlDataType, bo: ByteOrder) -> Result<ArrayData> {
    let sz = dt.byte_size();
    if !raw.len().is_multiple_of(sz) {
        return Err(IoError::InvalidData(format!(
            "VTK binary byte count {} not divisible by element size {sz}",
            raw.len()
        )));
    }
    let le = bo == ByteOrder::LittleEndian;
    Ok(match dt {
        XmlDataType::UInt8 => ArrayData::U8(raw.to_vec()),
        XmlDataType::Int8 => ArrayData::I32(raw.iter().map(|&b| (b as i8) as i32).collect()),
        XmlDataType::Int16 => ArrayData::I32(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i16::from_le_bytes(c) as i32
                    } else {
                        i16::from_be_bytes(c) as i32
                    }
                })
                .collect(),
        ),
        XmlDataType::UInt16 => ArrayData::U32(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        u16::from_le_bytes(c) as u32
                    } else {
                        u16::from_be_bytes(c) as u32
                    }
                })
                .collect(),
        ),
        XmlDataType::Int32 => ArrayData::I32(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i32::from_le_bytes(c)
                    } else {
                        i32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        XmlDataType::UInt32 => ArrayData::U32(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        u32::from_le_bytes(c)
                    } else {
                        u32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        XmlDataType::Int64 => ArrayData::I64(
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i64::from_le_bytes(c)
                    } else {
                        i64::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        XmlDataType::UInt64 => ArrayData::I64(
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        u64::from_le_bytes(c) as i64
                    } else {
                        u64::from_be_bytes(c) as i64
                    }
                })
                .collect(),
        ),
        XmlDataType::Float32 => ArrayData::F32(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        f32::from_le_bytes(c)
                    } else {
                        f32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        XmlDataType::Float64 => ArrayData::F64(
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        f64::from_le_bytes(c)
                    } else {
                        f64::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
    })
}

fn decode_ascii_values(text: &str, dt: XmlDataType) -> Result<ArrayData> {
    let parse_err = |tok: &str| IoError::Parse(format!("Invalid {dt:?} token '{tok}'"));
    Ok(match dt {
        XmlDataType::UInt8 => ArrayData::U8(
            text.split_whitespace()
                .map(|t| t.parse::<u8>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::Int8 | XmlDataType::Int16 | XmlDataType::Int32 => ArrayData::I32(
            text.split_whitespace()
                .map(|t| t.parse::<i32>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::UInt16 | XmlDataType::UInt32 => ArrayData::U32(
            text.split_whitespace()
                .map(|t| t.parse::<u32>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::Int64 => ArrayData::I64(
            text.split_whitespace()
                .map(|t| t.parse::<i64>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::UInt64 => ArrayData::I64(
            text.split_whitespace()
                .map(|t| t.parse::<u64>().map(|u| u as i64).map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::Float32 => ArrayData::F32(
            text.split_whitespace()
                .map(|t| t.parse::<f32>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
        XmlDataType::Float64 => ArrayData::F64(
            text.split_whitespace()
                .map(|t| t.parse::<f64>().map_err(|_| parse_err(t)))
                .collect::<Result<_>>()?,
        ),
    })
}

fn parse_extent(s: &str) -> Result<[isize; 6]> {
    let vals: Vec<isize> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<isize>()
                .map_err(|_| IoError::Parse(format!("Invalid extent token '{t}'")))
        })
        .collect::<Result<_>>()?;
    if vals.len() != 6 {
        return Err(IoError::Parse(format!(
            "Expected 6 extent integers, got '{}'",
            s
        )));
    }
    Ok([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]])
}

fn extent_to_dims(ext: [isize; 6]) -> [usize; 3] {
    [
        (ext[1] - ext[0] + 1).max(0) as usize,
        (ext[3] - ext[2] + 1).max(0) as usize,
        (ext[5] - ext[4] + 1).max(0) as usize,
    ]
}

fn parse_vec3_f64(s: &str) -> Result<[f64; 3]> {
    let vals: Vec<f64> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| IoError::Parse(format!("Invalid f64 token '{t}'")))
        })
        .collect::<Result<_>>()?;
    if vals.len() != 3 {
        return Err(IoError::Parse(format!("Expected 3 floats, got '{}'", s)));
    }
    Ok([vals[0], vals[1], vals[2]])
}

// ================================================================================================
// Writers
// ================================================================================================

/// Writes a [`DataSet`] to a VTK XML file (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`)
/// using inline base64 binary encoding.
pub fn write_vtk_xml(path: impl AsRef<Path>, dataset: &DataSet) -> Result<()> {
    write_vtk_xml_with_format(path, dataset, VtkXmlFormat::Binary)
}

/// Writes a [`DataSet`] to a VTK XML file with an explicit [`VtkXmlFormat`].
pub fn write_vtk_xml_with_format(
    path: impl AsRef<Path>,
    dataset: &DataSet,
    format: VtkXmlFormat,
) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    if let DataSet::MultiBlock(mb) = dataset {
        return write_vtm(path, mb, format);
    }

    let mut appended_buf: Vec<u8> = Vec::new();
    let is_compressed = matches!(
        format,
        VtkXmlFormat::BinaryZlib | VtkXmlFormat::BinaryCompressed | VtkXmlFormat::AppendedCompressed
    );
    let is_appended = matches!(
        format,
        VtkXmlFormat::Appended | VtkXmlFormat::AppendedCompressed
    );

    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let compressor_attr = if is_compressed {
        " compressor=\"vtkZLibDataCompressor\""
    } else {
        ""
    };

    writeln!(w, "<?xml version=\"1.0\"?>")?;

    match dataset {
        DataSet::PolyData(poly) => {
            if !ext.is_empty() && ext != "vtp" {
                return Err(IoError::UnsupportedFormat(format!(
                    "Cannot write PolyData as .{ext}"
                )));
            }
            writeln!(
                w,
                "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor_attr}>"
            )?;
            writeln!(w, "  <PolyData>")?;
            writeln!(
                w,
                "    <Piece NumberOfPoints=\"{}\" NumberOfVerts=\"{}\" NumberOfLines=\"{}\" NumberOfStrips=\"{}\" NumberOfPolys=\"{}\">",
                poly.points.len(),
                poly.verts.len(),
                poly.lines.len(),
                poly.strips.len(),
                poly.polys.len()
            )?;
            write_field_section(&mut w, "PointData", &poly.point_data, format, &mut appended_buf)?;
            write_field_section(&mut w, "CellData", &poly.cell_data, format, &mut appended_buf)?;
            write_points_section(&mut w, &poly.points, format, &mut appended_buf)?;

            let vert_cells: Vec<Vec<u32>> = poly.verts.iter().map(|&v| vec![v]).collect();
            write_cell_group(&mut w, "Verts", &vert_cells, format, &mut appended_buf)?;
            write_cell_group(&mut w, "Lines", &poly.lines, format, &mut appended_buf)?;
            write_cell_group(&mut w, "Strips", &poly.strips, format, &mut appended_buf)?;
            write_cell_group(&mut w, "Polys", &poly.polys, format, &mut appended_buf)?;
            writeln!(w, "    </Piece>")?;
            write_field_section(&mut w, "FieldData", &poly.field_data, format, &mut appended_buf)?;
            writeln!(w, "  </PolyData>")?;
        }
        DataSet::ImageData(img) => {
            if !ext.is_empty() && ext != "vti" {
                return Err(IoError::UnsupportedFormat(format!(
                    "Cannot write ImageData as .{ext}"
                )));
            }
            let x1 = img.dimensions[0].saturating_sub(1);
            let y1 = img.dimensions[1].saturating_sub(1);
            let z1 = img.dimensions[2].saturating_sub(1);
            let extent = format!("0 {x1} 0 {y1} 0 {z1}");
            writeln!(
                w,
                "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor_attr}>"
            )?;
            writeln!(
                w,
                "  <ImageData WholeExtent=\"{extent}\" Origin=\"{} {} {}\" Spacing=\"{} {} {}\">",
                img.origin[0],
                img.origin[1],
                img.origin[2],
                img.spacing[0],
                img.spacing[1],
                img.spacing[2]
            )?;
            writeln!(w, "    <Piece Extent=\"{extent}\">")?;
            write_field_section(&mut w, "PointData", &img.point_data, format, &mut appended_buf)?;
            write_field_section(&mut w, "CellData", &img.cell_data, format, &mut appended_buf)?;
            writeln!(w, "    </Piece>")?;
            write_field_section(&mut w, "FieldData", &img.field_data, format, &mut appended_buf)?;
            writeln!(w, "  </ImageData>")?;
        }
        DataSet::RectilinearGrid(rg) => {
            if !ext.is_empty() && ext != "vtr" {
                return Err(IoError::UnsupportedFormat(format!(
                    "Cannot write RectilinearGrid as .{ext}"
                )));
            }
            let x1 = rg.x_coords.len().saturating_sub(1);
            let y1 = rg.y_coords.len().saturating_sub(1);
            let z1 = rg.z_coords.len().saturating_sub(1);
            let extent = format!("0 {x1} 0 {y1} 0 {z1}");
            writeln!(
                w,
                "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor_attr}>"
            )?;
            writeln!(w, "  <RectilinearGrid WholeExtent=\"{extent}\">")?;
            writeln!(w, "    <Piece Extent=\"{extent}\">")?;
            write_field_section(&mut w, "PointData", &rg.point_data, format, &mut appended_buf)?;
            write_field_section(&mut w, "CellData", &rg.cell_data, format, &mut appended_buf)?;
            writeln!(w, "      <Coordinates>")?;
            let xa = DataArray::scalars_f32("XCoordinates", rg.x_coords.clone());
            let ya = DataArray::scalars_f32("YCoordinates", rg.y_coords.clone());
            let za = DataArray::scalars_f32("ZCoordinates", rg.z_coords.clone());
            write_xml_data_array(&mut w, "        ", &xa, format, &mut appended_buf)?;
            write_xml_data_array(&mut w, "        ", &ya, format, &mut appended_buf)?;
            write_xml_data_array(&mut w, "        ", &za, format, &mut appended_buf)?;
            writeln!(w, "      </Coordinates>")?;
            writeln!(w, "    </Piece>")?;
            write_field_section(&mut w, "FieldData", &rg.field_data, format, &mut appended_buf)?;
            writeln!(w, "  </RectilinearGrid>")?;
        }
        DataSet::StructuredGrid(sg) => {
            if !ext.is_empty() && ext != "vts" {
                return Err(IoError::UnsupportedFormat(format!(
                    "Cannot write StructuredGrid as .{ext}"
                )));
            }
            let x1 = sg.dimensions[0].saturating_sub(1);
            let y1 = sg.dimensions[1].saturating_sub(1);
            let z1 = sg.dimensions[2].saturating_sub(1);
            let extent = format!("0 {x1} 0 {y1} 0 {z1}");
            writeln!(
                w,
                "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor_attr}>"
            )?;
            writeln!(w, "  <StructuredGrid WholeExtent=\"{extent}\">")?;
            writeln!(w, "    <Piece Extent=\"{extent}\">")?;
            write_field_section(&mut w, "PointData", &sg.point_data, format, &mut appended_buf)?;
            write_field_section(&mut w, "CellData", &sg.cell_data, format, &mut appended_buf)?;
            write_points_section(&mut w, &sg.points, format, &mut appended_buf)?;
            writeln!(w, "    </Piece>")?;
            write_field_section(&mut w, "FieldData", &sg.field_data, format, &mut appended_buf)?;
            writeln!(w, "  </StructuredGrid>")?;
        }
        DataSet::UnstructuredGrid(ug) => {
            if !ext.is_empty() && ext != "vtu" {
                return Err(IoError::UnsupportedFormat(format!(
                    "Cannot write UnstructuredGrid as .{ext}"
                )));
            }
            writeln!(
                w,
                "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor_attr}>"
            )?;
            writeln!(w, "  <UnstructuredGrid>")?;
            writeln!(
                w,
                "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
                ug.points.len(),
                ug.num_cells()
            )?;
            write_field_section(&mut w, "PointData", &ug.point_data, format, &mut appended_buf)?;
            write_field_section(&mut w, "CellData", &ug.cell_data, format, &mut appended_buf)?;
            write_points_section(&mut w, &ug.points, format, &mut appended_buf)?;
            writeln!(w, "      <Cells>")?;
            let conn_i32: Vec<i32> = ug.connectivity.iter().map(|&v| v as i32).collect();
            let offs_i32: Vec<i32> = ug
                .offsets
                .iter()
                .skip(1)
                .map(|&v| v as i32)
                .collect();
            let types_u8: Vec<u8> = ug.cell_types.iter().map(|c| c.vtk_id()).collect();
            let conn_arr = DataArray::new("connectivity", 1, ArrayData::I32(conn_i32));
            let offs_arr = DataArray::new("offsets", 1, ArrayData::I32(offs_i32));
            let types_arr = DataArray::new("types", 1, ArrayData::U8(types_u8));
            write_xml_data_array(&mut w, "        ", &conn_arr, format, &mut appended_buf)?;
            write_xml_data_array(&mut w, "        ", &offs_arr, format, &mut appended_buf)?;
            write_xml_data_array(&mut w, "        ", &types_arr, format, &mut appended_buf)?;
            writeln!(w, "      </Cells>")?;
            writeln!(w, "    </Piece>")?;
            write_field_section(&mut w, "FieldData", &ug.field_data, format, &mut appended_buf)?;
            writeln!(w, "  </UnstructuredGrid>")?;
        }
        DataSet::MultiBlock(_) => unreachable!(),
    }

    if is_appended {
        writeln!(w, "  <AppendedData encoding=\"raw\">")?;
        w.write_all(b"_")?;
        w.write_all(&appended_buf)?;
        writeln!(w)?;
        writeln!(w, "  </AppendedData>")?;
    }

    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}

/// Writes a [`MultiBlockDataSet`] to a `.vtm` file and sibling block files.
pub fn write_vtm(
    path: impl AsRef<Path>,
    mb: &MultiBlockDataSet,
    format: VtkXmlFormat,
) -> Result<()> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("multiblock");

    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "<?xml version=\"1.0\"?>")?;
    writeln!(
        w,
        "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(w, "  <vtkMultiBlockDataSet>")?;

    for (idx, (name, ds)) in mb.blocks.iter().enumerate() {
        let block_ext = dataset_xml_extension(ds)?;
        let rel_name = format!("{stem}_block_{idx}.{block_ext}");
        let block_path = parent.join(&rel_name);
        write_vtk_xml_with_format(&block_path, ds, format)?;
        writeln!(
            w,
            "    <DataSet index=\"{idx}\" name=\"{}\" file=\"{rel_name}\"/>",
            escape_xml_attr(name)
        )?;
    }

    writeln!(w, "  </vtkMultiBlockDataSet>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}

/// Writes a `.pvd` ParaViewData time-series collection and its step dataset files.
pub fn write_pvd_series(path: impl AsRef<Path>, series: &[(f64, DataSet)]) -> Result<()> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("series");

    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "<?xml version=\"1.0\"?>")?;
    writeln!(
        w,
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(w, "  <Collection>")?;

    for (idx, (time, ds)) in series.iter().enumerate() {
        let step_ext = dataset_xml_extension(ds)?;
        let rel_name = format!("{stem}_t{idx}.{step_ext}");
        let step_path = parent.join(&rel_name);
        write_vtk_xml(&step_path, ds)?;
        writeln!(
            w,
            "    <DataSet timestep=\"{time}\" group=\"\" part=\"0\" file=\"{rel_name}\"/>"
        )?;
    }

    writeln!(w, "  </Collection>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}

fn dataset_xml_extension(ds: &DataSet) -> Result<&'static str> {
    match ds {
        DataSet::PolyData(_) => Ok("vtp"),
        DataSet::ImageData(_) => Ok("vti"),
        DataSet::UnstructuredGrid(_) => Ok("vtu"),
        DataSet::RectilinearGrid(_) => Ok("vtr"),
        DataSet::StructuredGrid(_) => Ok("vts"),
        DataSet::MultiBlock(_) => Ok("vtm"),
    }
}

fn write_field_section(
    w: &mut impl Write,
    tag: &str,
    field: &FieldData,
    format: VtkXmlFormat,
    appended_buf: &mut Vec<u8>,
) -> Result<()> {
    if field.is_empty() {
        return Ok(());
    }
    let mut attrs = String::new();
    if let Some(s) = &field.active_scalars {
        attrs.push_str(&format!(" Scalars=\"{}\"", escape_xml_attr(s)));
    }
    if let Some(v) = &field.active_vectors {
        attrs.push_str(&format!(" Vectors=\"{}\"", escape_xml_attr(v)));
    }
    if let Some(n) = &field.active_normals {
        attrs.push_str(&format!(" Normals=\"{}\"", escape_xml_attr(n)));
    }
    writeln!(w, "      <{tag}{attrs}>")?;
    for arr in field.arrays() {
        write_xml_data_array(w, "        ", arr, format, appended_buf)?;
    }
    writeln!(w, "      </{tag}>")?;
    Ok(())
}

fn write_points_section(
    w: &mut impl Write,
    points: &[[f32; 3]],
    format: VtkXmlFormat,
    appended_buf: &mut Vec<u8>,
) -> Result<()> {
    let mut flat = Vec::with_capacity(points.len() * 3);
    for p in points {
        flat.extend_from_slice(p);
    }
    let arr = DataArray::new("Points", 3, ArrayData::F32(flat));
    writeln!(w, "      <Points>")?;
    write_xml_data_array(w, "        ", &arr, format, appended_buf)?;
    writeln!(w, "      </Points>")?;
    Ok(())
}

fn write_cell_group(
    w: &mut impl Write,
    tag: &str,
    cells: &[Vec<u32>],
    format: VtkXmlFormat,
    appended_buf: &mut Vec<u8>,
) -> Result<()> {
    let mut conn = Vec::new();
    let mut offs = Vec::with_capacity(cells.len());
    for cell in cells {
        for &idx in cell {
            conn.push(idx as i32);
        }
        offs.push(conn.len() as i32);
    }
    let conn_arr = DataArray::new("connectivity", 1, ArrayData::I32(conn));
    let offs_arr = DataArray::new("offsets", 1, ArrayData::I32(offs));
    writeln!(w, "      <{tag}>")?;
    write_xml_data_array(w, "        ", &conn_arr, format, appended_buf)?;
    write_xml_data_array(w, "        ", &offs_arr, format, appended_buf)?;
    writeln!(w, "      </{tag}>")?;
    Ok(())
}

fn write_xml_data_array(
    w: &mut impl Write,
    indent: &str,
    arr: &DataArray,
    format: VtkXmlFormat,
    appended_buf: &mut Vec<u8>,
) -> Result<()> {
    let vtk_ty = scalar_type_xml_name(arr.data.scalar_type());
    let name = escape_xml_attr(&arr.name);
    let nc = arr.num_components;

    match format {
        VtkXmlFormat::Ascii => {
            writeln!(
                w,
                "{indent}<DataArray type=\"{vtk_ty}\" Name=\"{name}\" NumberOfComponents=\"{nc}\" format=\"ascii\">"
            )?;
            write!(w, "{indent}  ")?;
            write_ascii_array(w, arr)?;
            writeln!(w)?;
            writeln!(w, "{indent}</DataArray>")?;
        }
        VtkXmlFormat::Binary | VtkXmlFormat::BinaryZlib | VtkXmlFormat::BinaryCompressed => {
            let compressed = matches!(
                format,
                VtkXmlFormat::BinaryZlib | VtkXmlFormat::BinaryCompressed
            );
            let block = encode_vtk_binary_block(&array_data_to_le_bytes(&arr.data), compressed)?;
            let b64 = BASE64.encode(&block);
            writeln!(
                w,
                "{indent}<DataArray type=\"{vtk_ty}\" Name=\"{name}\" NumberOfComponents=\"{nc}\" format=\"binary\">"
            )?;
            writeln!(w, "{indent}  {b64}")?;
            writeln!(w, "{indent}</DataArray>")?;
        }
        VtkXmlFormat::Appended | VtkXmlFormat::AppendedCompressed => {
            let compressed = format == VtkXmlFormat::AppendedCompressed;
            let offset = appended_buf.len();
            let block = encode_vtk_binary_block(&array_data_to_le_bytes(&arr.data), compressed)?;
            appended_buf.extend_from_slice(&block);
            writeln!(
                w,
                "{indent}<DataArray type=\"{vtk_ty}\" Name=\"{name}\" NumberOfComponents=\"{nc}\" format=\"appended\" offset=\"{offset}\"/>"
            )?;
        }
    }
    Ok(())
}

fn encode_vtk_binary_block(raw: &[u8], compressed: bool) -> Result<Vec<u8>> {
    if !compressed {
        let mut out = Vec::with_capacity(8 + raw.len());
        out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        out.extend_from_slice(raw);
        return Ok(out);
    }

    const BLOCK_SIZE: usize = 32768;
    let chunks: Vec<&[u8]> = if raw.is_empty() {
        vec![&[]]
    } else {
        raw.chunks(BLOCK_SIZE).collect()
    };
    let num_blocks = chunks.len();
    let last_block_size = chunks.last().map(|c| c.len()).unwrap_or(0);

    let mut compressed_chunks = Vec::with_capacity(num_blocks);
    for chunk in chunks {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(chunk)?;
        compressed_chunks.push(encoder.finish()?);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&(num_blocks as u64).to_le_bytes());
    out.extend_from_slice(&(BLOCK_SIZE as u64).to_le_bytes());
    out.extend_from_slice(&(last_block_size as u64).to_le_bytes());
    for cc in &compressed_chunks {
        out.extend_from_slice(&(cc.len() as u64).to_le_bytes());
    }
    for cc in &compressed_chunks {
        out.extend_from_slice(cc);
    }
    Ok(out)
}

fn array_data_to_le_bytes(data: &ArrayData) -> Vec<u8> {
    match data {
        ArrayData::U8(v) => v.clone(),
        ArrayData::I32(v) => {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        ArrayData::U32(v) => {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        ArrayData::I64(v) => {
            let mut out = Vec::with_capacity(v.len() * 8);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        ArrayData::F32(v) => {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        ArrayData::F64(v) => {
            let mut out = Vec::with_capacity(v.len() * 8);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
    }
}

fn write_ascii_array(w: &mut impl Write, arr: &DataArray) -> Result<()> {
    match &arr.data {
        ArrayData::U8(v) => write_joined(w, v),
        ArrayData::I32(v) => write_joined(w, v),
        ArrayData::U32(v) => write_joined(w, v),
        ArrayData::I64(v) => write_joined(w, v),
        ArrayData::F32(v) => write_joined(w, v),
        ArrayData::F64(v) => write_joined(w, v),
    }
}

fn write_joined<T: std::fmt::Display>(w: &mut impl Write, vals: &[T]) -> Result<()> {
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            write!(w, " ")?;
        }
        write!(w, "{v}")?;
    }
    Ok(())
}

fn scalar_type_xml_name(st: ScalarType) -> &'static str {
    st.vtk_name()
}

fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
