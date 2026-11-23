//! Legacy VTK (`.vtk`) ASCII and Binary (big-endian) reader and writer.

use crate::error::{IoError, Result};
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, ImageData, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Legacy VTK encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VtkLegacyFormat {
    Ascii,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtkDataType {
    Bit,
    UnsignedChar,
    Char,
    UnsignedShort,
    Short,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    Int64,
    UInt64,
    Float,
    Double,
}

impl VtkDataType {
    fn from_token(tok: &str) -> Result<Self> {
        match tok.to_ascii_lowercase().as_str() {
            "bit" => Ok(Self::Bit),
            "unsigned_char" => Ok(Self::UnsignedChar),
            "char" => Ok(Self::Char),
            "unsigned_short" => Ok(Self::UnsignedShort),
            "short" => Ok(Self::Short),
            "int" => Ok(Self::Int),
            "unsigned_int" => Ok(Self::UnsignedInt),
            "long" => Ok(Self::Long),
            "unsigned_long" => Ok(Self::UnsignedLong),
            "vtktypeint64" | "int64" => Ok(Self::Int64),
            "vtktypeuint64" | "uint64" => Ok(Self::UInt64),
            "float" => Ok(Self::Float),
            "double" => Ok(Self::Double),
            other => Err(IoError::Parse(format!(
                "Unsupported VTK legacy data type '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttrSection {
    None,
    PointData(usize),
    CellData(usize),
}

enum PendingDataSet {
    None,
    StructuredPoints(ImageData),
    RectilinearGrid(RectilinearGrid),
    StructuredGrid(StructuredGrid),
    PolyData(PolyData),
    UnstructuredGrid(UnstructuredGrid),
}

impl PendingDataSet {
    fn num_points(&self) -> usize {
        match self {
            Self::None => 0,
            Self::StructuredPoints(d) => d.num_points(),
            Self::RectilinearGrid(d) => d.num_points(),
            Self::StructuredGrid(d) => d.num_points(),
            Self::PolyData(d) => d.num_points(),
            Self::UnstructuredGrid(d) => d.num_points(),
        }
    }

    fn num_cells(&self) -> usize {
        match self {
            Self::None => 0,
            Self::StructuredPoints(d) => d.num_cells(),
            Self::RectilinearGrid(d) => d.num_cells(),
            Self::StructuredGrid(d) => d.num_cells(),
            Self::PolyData(d) => d.num_cells(),
            Self::UnstructuredGrid(d) => d.num_cells(),
        }
    }

    fn target_field_mut<'a>(
        &'a mut self,
        section: AttrSection,
        top_field: &'a mut FieldData,
    ) -> &'a mut FieldData {
        match section {
            AttrSection::PointData(_) => match self {
                Self::StructuredPoints(d) => &mut d.point_data,
                Self::RectilinearGrid(d) => &mut d.point_data,
                Self::StructuredGrid(d) => &mut d.point_data,
                Self::PolyData(d) => &mut d.point_data,
                Self::UnstructuredGrid(d) => &mut d.point_data,
                Self::None => top_field,
            },
            AttrSection::CellData(_) => match self {
                Self::StructuredPoints(d) => &mut d.cell_data,
                Self::RectilinearGrid(d) => &mut d.cell_data,
                Self::StructuredGrid(d) => &mut d.cell_data,
                Self::PolyData(d) => &mut d.cell_data,
                Self::UnstructuredGrid(d) => &mut d.cell_data,
                Self::None => top_field,
            },
            AttrSection::None => match self {
                Self::StructuredPoints(d) => &mut d.field_data,
                Self::RectilinearGrid(d) => &mut d.field_data,
                Self::StructuredGrid(d) => &mut d.field_data,
                Self::PolyData(d) => &mut d.field_data,
                Self::UnstructuredGrid(d) => &mut d.field_data,
                Self::None => top_field,
            },
        }
    }
}

/// Reads a Legacy VTK (`.vtk`) file (ASCII or Binary) into a [`DataSet`].
pub fn read_vtk_legacy(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_vtk_legacy_bytes(&bytes)
}

/// Parses Legacy VTK bytes (ASCII or Binary) into a [`DataSet`].
pub fn read_vtk_legacy_bytes(bytes: &[u8]) -> Result<DataSet> {
    let mut cur = LegacyCursor::new(bytes);

    let header = cur
        .next_line_trimmed()
        .ok_or_else(|| IoError::InvalidData("Empty VTK file".into()))?;
    if !header.starts_with("# vtk DataFile Version") {
        return Err(IoError::Parse(format!(
            "Invalid VTK legacy header: '{header}'"
        )));
    }

    // Line 2: title (may be empty)
    let _title = cur
        .read_line_raw()
        .ok_or_else(|| IoError::InvalidData("VTK file truncated after header".into()))?;

    // Line 3: ASCII or BINARY
    let fmt_line = cur
        .next_non_empty_line()
        .ok_or_else(|| IoError::InvalidData("VTK file missing ASCII/BINARY line".into()))?;
    let format = match fmt_line.trim().to_ascii_uppercase().as_str() {
        "ASCII" => VtkLegacyFormat::Ascii,
        "BINARY" => VtkLegacyFormat::Binary,
        other => {
            return Err(IoError::Parse(format!(
                "Expected ASCII or BINARY in VTK file, got '{other}'"
            )))
        }
    };

    let mut ds = PendingDataSet::None;
    let mut top_field_data = FieldData::new();
    let mut section = AttrSection::None;

    while let Some(line) = cur.next_non_empty_line() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let kw = parts[0].to_ascii_uppercase();
        match kw.as_str() {
            "DATASET" => {
                let kind = parts
                    .get(1)
                    .ok_or_else(|| IoError::Parse("DATASET missing type".into()))?
                    .to_ascii_uppercase();
                ds = match kind.as_str() {
                    "STRUCTURED_POINTS" => PendingDataSet::StructuredPoints(ImageData::default()),
                    "RECTILINEAR_GRID" => {
                        PendingDataSet::RectilinearGrid(RectilinearGrid::default())
                    }
                    "STRUCTURED_GRID" => PendingDataSet::StructuredGrid(StructuredGrid::default()),
                    "POLYDATA" => PendingDataSet::PolyData(PolyData::new()),
                    "UNSTRUCTURED_GRID" => {
                        PendingDataSet::UnstructuredGrid(UnstructuredGrid::new())
                    }
                    other => {
                        return Err(IoError::UnsupportedFormat(format!(
                            "Unsupported VTK legacy DATASET '{other}'"
                        )))
                    }
                };
            }
            "DIMENSIONS" => {
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Invalid DIMENSIONS line: '{line}'")));
                }
                let nx = parse_usize(parts[1])?;
                let ny = parse_usize(parts[2])?;
                let nz = parse_usize(parts[3])?;
                match &mut ds {
                    PendingDataSet::StructuredPoints(img) => img.dimensions = [nx, ny, nz],
                    PendingDataSet::StructuredGrid(sg) => sg.dimensions = [nx, ny, nz],
                    PendingDataSet::RectilinearGrid(_) => {}
                    _ => {
                        return Err(IoError::Parse(
                            "DIMENSIONS encountered for unsupported dataset type".into(),
                        ))
                    }
                }
            }
            "ORIGIN" => {
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Invalid ORIGIN line: '{line}'")));
                }
                let origin = [parse_f64(parts[1])?, parse_f64(parts[2])?, parse_f64(parts[3])?];
                if let PendingDataSet::StructuredPoints(img) = &mut ds {
                    img.origin = origin;
                }
            }
            "SPACING" | "ASPECT_RATIOS" => {
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Invalid SPACING line: '{line}'")));
                }
                let spacing = [parse_f64(parts[1])?, parse_f64(parts[2])?, parse_f64(parts[3])?];
                if let PendingDataSet::StructuredPoints(img) = &mut ds {
                    img.spacing = spacing;
                }
            }
            "X_COORDINATES" | "Y_COORDINATES" | "Z_COORDINATES" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid {kw} line: '{line}'")));
                }
                let n = parse_usize(parts[1])?;
                let dt = VtkDataType::from_token(parts[2])?;
                let arr = cur.read_typed_array("coords", 1, n, dt, format)?;
                let coords = arr.data.to_f32_vec();
                if let PendingDataSet::RectilinearGrid(rg) = &mut ds {
                    match kw.as_str() {
                        "X_COORDINATES" => rg.x_coords = coords,
                        "Y_COORDINATES" => rg.y_coords = coords,
                        "Z_COORDINATES" => rg.z_coords = coords,
                        _ => {}
                    }
                }
            }
            "POINTS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid POINTS line: '{line}'")));
                }
                let n = parse_usize(parts[1])?;
                let dt = VtkDataType::from_token(parts[2])?;
                let arr = cur.read_typed_array("points", 3, n, dt, format)?;
                let vals = arr.data.to_f32_vec();
                let mut pts = Vec::with_capacity(n);
                for chunk in vals.as_chunks::<3>().0 {
                    pts.push(*chunk);
                }
                match &mut ds {
                    PendingDataSet::PolyData(p) => p.points = pts,
                    PendingDataSet::StructuredGrid(sg) => sg.points = pts,
                    PendingDataSet::UnstructuredGrid(ug) => ug.points = pts,
                    _ => {
                        return Err(IoError::Parse(
                            "POINTS encountered for unsupported dataset type".into(),
                        ))
                    }
                }
            }
            "VERTICES" | "LINES" | "POLYGONS" | "TRIANGLE_STRIPS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid {kw} line: '{line}'")));
                }
                let n = parse_usize(parts[1])?;
                let size = parse_usize(parts[2])?;
                let cells = read_cell_lists(&mut cur, n, size, format)?;
                if let PendingDataSet::PolyData(p) = &mut ds {
                    match kw.as_str() {
                        "VERTICES" => {
                            for c in cells {
                                p.verts.extend(c);
                            }
                        }
                        "LINES" => p.lines = cells,
                        "POLYGONS" => p.polys = cells,
                        "TRIANGLE_STRIPS" => p.strips = cells,
                        _ => {}
                    }
                }
            }
            "CELLS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid CELLS line: '{line}'")));
                }
                let n = parse_usize(parts[1])?;
                let size = parse_usize(parts[2])?;
                let cells = read_cell_lists(&mut cur, n, size, format)?;
                if let PendingDataSet::UnstructuredGrid(ug) = &mut ds {
                    ug.connectivity.clear();
                    ug.offsets = Vec::with_capacity(cells.len() + 1);
                    ug.offsets.push(0);
                    for c in cells {
                        ug.connectivity.extend_from_slice(&c);
                        ug.offsets.push(ug.connectivity.len() as u32);
                    }
                }
            }
            "CELL_TYPES" => {
                if parts.len() < 2 {
                    return Err(IoError::Parse(format!("Invalid CELL_TYPES line: '{line}'")));
                }
                let n = parse_usize(parts[1])?;
                let arr = cur.read_typed_array("cell_types", 1, n, VtkDataType::Int, format)?;
                let mut types = Vec::with_capacity(n);
                for i in 0..n {
                    let id = arr.data.get_f64(i) as i64;
                    let ct = CellType::from_vtk_id(id as u8).ok_or_else(|| {
                        IoError::UnsupportedFormat(format!("Unsupported VTK cell type ID {id}"))
                    })?;
                    types.push(ct);
                }
                if let PendingDataSet::UnstructuredGrid(ug) = &mut ds {
                    ug.cell_types = types;
                }
            }
            "POINT_DATA" => {
                let n = parts
                    .get(1)
                    .map(|s| parse_usize(s))
                    .transpose()?
                    .unwrap_or_else(|| ds.num_points());
                section = AttrSection::PointData(n);
            }
            "CELL_DATA" => {
                let n = parts
                    .get(1)
                    .map(|s| parse_usize(s))
                    .transpose()?
                    .unwrap_or_else(|| ds.num_cells());
                section = AttrSection::CellData(n);
            }
            "FIELD" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid FIELD line: '{line}'")));
                }
                let num_arrays = parse_usize(parts[2])?;
                for _ in 0..num_arrays {
                    let arr_line = cur.next_non_empty_line().ok_or_else(|| {
                        IoError::InvalidData("Unexpected EOF inside FIELD block".into())
                    })?;
                    let ap: Vec<&str> = arr_line.split_whitespace().collect();
                    if ap.len() < 4 {
                        return Err(IoError::Parse(format!(
                            "Invalid FIELD array line: '{arr_line}'"
                        )));
                    }
                    let name = ap[0].to_string();
                    let num_comp = parse_usize(ap[1])?;
                    let num_tuples = parse_usize(ap[2])?;
                    let dt = VtkDataType::from_token(ap[3])?;
                    let arr = cur.read_typed_array(&name, num_comp, num_tuples, dt, format)?;
                    ds.target_field_mut(section, &mut top_field_data)
                        .add_array(arr);
                }
            }
            "SCALARS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid SCALARS line: '{line}'")));
                }
                let name = parts[1].to_string();
                let dt = VtkDataType::from_token(parts[2])?;
                let num_comp = parts.get(3).map(|s| parse_usize(s)).transpose()?.unwrap_or(1);

                // Consume optional LOOKUP_TABLE line
                let saved = cur.pos;
                if let Some(next_line) = cur.next_non_empty_line()
                    && !next_line
                        .split_whitespace()
                        .next()
                        .map(|t| t.eq_ignore_ascii_case("LOOKUP_TABLE"))
                        .unwrap_or(false)
                {
                    cur.pos = saved;
                }

                let num_tuples = section_tuple_count(&ds, section);
                let arr = cur.read_typed_array(&name, num_comp, num_tuples, dt, format)?;
                ds.target_field_mut(section, &mut top_field_data)
                    .set_scalars(arr);
            }
            "COLOR_SCALARS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!(
                        "Invalid COLOR_SCALARS line: '{line}'"
                    )));
                }
                let name = parts[1].to_string();
                let num_comp = parse_usize(parts[2])?;
                let num_tuples = section_tuple_count(&ds, section);
                let arr = match format {
                    VtkLegacyFormat::Ascii => {
                        let tmp = cur.read_typed_array(
                            &name,
                            num_comp,
                            num_tuples,
                            VtkDataType::Float,
                            format,
                        )?;
                        let u8_vals: Vec<u8> = tmp
                            .data
                            .to_f64_vec()
                            .into_iter()
                            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                            .collect();
                        DataArray::colors_u8(name, num_comp, u8_vals)
                    }
                    VtkLegacyFormat::Binary => cur.read_typed_array(
                        &name,
                        num_comp,
                        num_tuples,
                        VtkDataType::UnsignedChar,
                        format,
                    )?,
                };
                ds.target_field_mut(section, &mut top_field_data)
                    .add_array(arr);
            }
            "VECTORS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid VECTORS line: '{line}'")));
                }
                let name = parts[1].to_string();
                let dt = VtkDataType::from_token(parts[2])?;
                let num_tuples = section_tuple_count(&ds, section);
                let arr = cur.read_typed_array(&name, 3, num_tuples, dt, format)?;
                ds.target_field_mut(section, &mut top_field_data)
                    .set_vectors(arr);
            }
            "NORMALS" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid NORMALS line: '{line}'")));
                }
                let name = parts[1].to_string();
                let dt = VtkDataType::from_token(parts[2])?;
                let num_tuples = section_tuple_count(&ds, section);
                let arr = cur.read_typed_array(&name, 3, num_tuples, dt, format)?;
                ds.target_field_mut(section, &mut top_field_data)
                    .set_normals(arr);
            }
            "TENSORS" | "TENSORS6" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid {kw} line: '{line}'")));
                }
                let name = parts[1].to_string();
                let dt = VtkDataType::from_token(parts[2])?;
                let num_comp = if kw == "TENSORS6" { 6 } else { 9 };
                let num_tuples = section_tuple_count(&ds, section);
                let arr = cur.read_typed_array(&name, num_comp, num_tuples, dt, format)?;
                ds.target_field_mut(section, &mut top_field_data)
                    .add_array(arr);
            }
            "TEXTURE_COORDINATES" => {
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!(
                        "Invalid TEXTURE_COORDINATES line: '{line}'"
                    )));
                }
                let name = parts[1].to_string();
                let dim = parse_usize(parts[2])?;
                let dt = VtkDataType::from_token(parts[3])?;
                let num_tuples = section_tuple_count(&ds, section);
                let arr = cur.read_typed_array(&name, dim, num_tuples, dt, format)?;
                ds.target_field_mut(section, &mut top_field_data)
                    .add_array(arr);
            }
            "LOOKUP_TABLE" => {
                if parts.len() >= 3
                    && let Ok(n) = parts[2].parse::<usize>()
                {
                    match format {
                        VtkLegacyFormat::Ascii => {
                            let _ = cur.read_typed_array(
                                "_lut",
                                4,
                                n,
                                VtkDataType::Float,
                                format,
                            )?;
                        }
                        VtkLegacyFormat::Binary => {
                            let _ = cur.read_typed_array(
                                "_lut",
                                4,
                                n,
                                VtkDataType::UnsignedChar,
                                format,
                            )?;
                        }
                    }
                }
            }
            "METADATA" => {
                while let Some(ml) = cur.read_line_raw() {
                    if ml.trim().is_empty() {
                        break;
                    }
                }
            }
            other => {
                return Err(IoError::Parse(format!(
                    "Unexpected keyword '{other}' in VTK legacy file"
                )));
            }
        }
    }

    let mut out = match ds {
        PendingDataSet::StructuredPoints(img) => DataSet::ImageData(img),
        PendingDataSet::RectilinearGrid(rg) => DataSet::RectilinearGrid(rg),
        PendingDataSet::StructuredGrid(sg) => DataSet::StructuredGrid(sg),
        PendingDataSet::PolyData(p) => DataSet::PolyData(p),
        PendingDataSet::UnstructuredGrid(ug) => DataSet::UnstructuredGrid(ug),
        PendingDataSet::None => {
            return Err(IoError::InvalidData(
                "VTK legacy file did not define a DATASET".into(),
            ))
        }
    };

    if !top_field_data.is_empty() {
        let dst = match &mut out {
            DataSet::PolyData(d) => &mut d.field_data,
            DataSet::ImageData(d) => &mut d.field_data,
            DataSet::RectilinearGrid(d) => &mut d.field_data,
            DataSet::StructuredGrid(d) => &mut d.field_data,
            DataSet::UnstructuredGrid(d) => &mut d.field_data,
            DataSet::MultiBlock(_) => unreachable!(),
        };
        for arr in top_field_data.arrays() {
            dst.add_array(arr.clone());
        }
    }

    Ok(out)
}

fn section_tuple_count(ds: &PendingDataSet, section: AttrSection) -> usize {
    match section {
        AttrSection::PointData(n) => {
            if n > 0 {
                n
            } else {
                ds.num_points()
            }
        }
        AttrSection::CellData(n) => {
            if n > 0 {
                n
            } else {
                ds.num_cells()
            }
        }
        AttrSection::None => 0,
    }
}

fn read_cell_lists(
    cur: &mut LegacyCursor<'_>,
    num_cells: usize,
    total_ints: usize,
    format: VtkLegacyFormat,
) -> Result<Vec<Vec<u32>>> {
    let saved = cur.pos;
    if let Some(peek) = cur.next_non_empty_line() {
        let p: Vec<&str> = peek.split_whitespace().collect();
        if p.first()
            .map(|s| s.eq_ignore_ascii_case("OFFSETS"))
            .unwrap_or(false)
        {
            let off_dt = p
                .get(1)
                .map(|s| VtkDataType::from_token(s))
                .transpose()?
                .unwrap_or(VtkDataType::Int64);
            let offsets_arr =
                cur.read_typed_array("offsets", 1, num_cells + 1, off_dt, format)?;
            let conn_line = cur.next_non_empty_line().ok_or_else(|| {
                IoError::InvalidData("Expected CONNECTIVITY after OFFSETS".into())
            })?;
            let cp: Vec<&str> = conn_line.split_whitespace().collect();
            let conn_dt = cp
                .get(1)
                .map(|s| VtkDataType::from_token(s))
                .transpose()?
                .unwrap_or(VtkDataType::Int64);
            let conn_len = offsets_arr.data.get_f64(num_cells) as usize;
            let conn_arr =
                cur.read_typed_array("connectivity", 1, conn_len, conn_dt, format)?;
            let mut cells = Vec::with_capacity(num_cells);
            for i in 0..num_cells {
                let start = offsets_arr.data.get_f64(i) as usize;
                let end = offsets_arr.data.get_f64(i + 1) as usize;
                let mut cell = Vec::with_capacity(end.saturating_sub(start));
                for idx in start..end {
                    cell.push(conn_arr.data.get_f64(idx) as u32);
                }
                cells.push(cell);
            }
            return Ok(cells);
        }
    }
    cur.pos = saved;

    let raw = cur.read_typed_array("cells", 1, total_ints, VtkDataType::Int, format)?;
    let vals = raw.data;
    let mut cells = Vec::with_capacity(num_cells);
    let mut i = 0usize;
    for _ in 0..num_cells {
        if i >= vals.len() {
            return Err(IoError::InvalidData("Truncated VTK cell list".into()));
        }
        let k = vals.get_f64(i) as usize;
        i += 1;
        if i + k > vals.len() {
            return Err(IoError::InvalidData(
                "Truncated VTK cell vertex indices".into(),
            ));
        }
        let mut cell = Vec::with_capacity(k);
        for j in 0..k {
            cell.push(vals.get_f64(i + j) as u32);
        }
        i += k;
        cells.push(cell);
    }
    Ok(cells)
}

struct LegacyCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> LegacyCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_line_raw(&mut self) -> Option<String> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        if let Some(rel) = self.bytes[start..].iter().position(|&b| b == b'\n') {
            self.pos = start + rel + 1;
            let mut slice = &self.bytes[start..start + rel];
            if slice.ends_with(b"\r") {
                slice = &slice[..slice.len() - 1];
            }
            Some(String::from_utf8_lossy(slice).into_owned())
        } else {
            self.pos = self.bytes.len();
            let mut slice = &self.bytes[start..];
            if slice.ends_with(b"\r") {
                slice = &slice[..slice.len() - 1];
            }
            Some(String::from_utf8_lossy(slice).into_owned())
        }
    }

    fn next_line_trimmed(&mut self) -> Option<String> {
        self.read_line_raw().map(|s| s.trim().to_string())
    }

    fn next_non_empty_line(&mut self) -> Option<String> {
        while let Some(line) = self.read_line_raw() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        None
    }

    fn next_ascii_token(&mut self) -> Result<String> {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(IoError::InvalidData(
                "Unexpected EOF reading VTK ASCII numeric token".into(),
            ));
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && !self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn consume_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return Err(IoError::InvalidData(format!(
                "Unexpected EOF reading {n} binary VTK bytes at offset {}",
                self.pos
            )));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_typed_array(
        &mut self,
        name: &str,
        num_comp: usize,
        num_tuples: usize,
        dt: VtkDataType,
        format: VtkLegacyFormat,
    ) -> Result<DataArray> {
        let count = num_comp.saturating_mul(num_tuples);
        let data = match format {
            VtkLegacyFormat::Ascii => self.read_ascii_array_data(count, dt)?,
            VtkLegacyFormat::Binary => self.read_binary_array_data(count, dt)?,
        };
        Ok(DataArray::new(name, num_comp.max(1), data))
    }

    fn read_ascii_array_data(&mut self, count: usize, dt: VtkDataType) -> Result<ArrayData> {
        match dt {
            VtkDataType::Bit | VtkDataType::UnsignedChar => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<u8>().map_err(|_| {
                        IoError::Parse(format!("Invalid u8 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::U8(v))
            }
            VtkDataType::Char | VtkDataType::Short | VtkDataType::Int => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<i32>().map_err(|_| {
                        IoError::Parse(format!("Invalid i32 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::I32(v))
            }
            VtkDataType::UnsignedShort | VtkDataType::UnsignedInt => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<u32>().map_err(|_| {
                        IoError::Parse(format!("Invalid u32 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::U32(v))
            }
            VtkDataType::Long | VtkDataType::Int64 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<i64>().map_err(|_| {
                        IoError::Parse(format!("Invalid i64 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::I64(v))
            }
            VtkDataType::UnsignedLong | VtkDataType::UInt64 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    let u = tok.parse::<u64>().map_err(|_| {
                        IoError::Parse(format!("Invalid u64 token '{tok}'"))
                    })?;
                    v.push(u as i64);
                }
                Ok(ArrayData::I64(v))
            }
            VtkDataType::Float => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<f32>().map_err(|_| {
                        IoError::Parse(format!("Invalid f32 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::F32(v))
            }
            VtkDataType::Double => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = self.next_ascii_token()?;
                    v.push(tok.parse::<f64>().map_err(|_| {
                        IoError::Parse(format!("Invalid f64 token '{tok}'"))
                    })?);
                }
                Ok(ArrayData::F64(v))
            }
        }
    }

    fn read_binary_array_data(&mut self, count: usize, dt: VtkDataType) -> Result<ArrayData> {
        let data = match dt {
            VtkDataType::Bit => {
                let num_bytes = count.div_ceil(8);
                let raw = self.consume_bytes(num_bytes)?;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let byte = raw[i / 8];
                    let bit = (byte >> (7 - (i % 8))) & 1;
                    out.push(bit);
                }
                ArrayData::U8(out)
            }
            VtkDataType::UnsignedChar => {
                let raw = self.consume_bytes(count)?;
                ArrayData::U8(raw.to_vec())
            }
            VtkDataType::Char => {
                let raw = self.consume_bytes(count)?;
                ArrayData::I32(raw.iter().map(|&b| (b as i8) as i32).collect())
            }
            VtkDataType::Short => {
                let raw = self.consume_bytes(count * 2)?;
                ArrayData::I32(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&c| i16::from_be_bytes(c) as i32)
                        .collect(),
                )
            }
            VtkDataType::UnsignedShort => {
                let raw = self.consume_bytes(count * 2)?;
                ArrayData::U32(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&c| u16::from_be_bytes(c) as u32)
                        .collect(),
                )
            }
            VtkDataType::Int => {
                let raw = self.consume_bytes(count * 4)?;
                ArrayData::I32(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&c| i32::from_be_bytes(c))
                        .collect(),
                )
            }
            VtkDataType::UnsignedInt => {
                let raw = self.consume_bytes(count * 4)?;
                ArrayData::U32(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&c| u32::from_be_bytes(c))
                        .collect(),
                )
            }
            VtkDataType::Long | VtkDataType::Int64 => {
                let raw = self.consume_bytes(count * 8)?;
                ArrayData::I64(
                    raw.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|&c| i64::from_be_bytes(c))
                        .collect(),
                )
            }
            VtkDataType::UnsignedLong | VtkDataType::UInt64 => {
                let raw = self.consume_bytes(count * 8)?;
                ArrayData::I64(
                    raw.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|&c| u64::from_be_bytes(c) as i64)
                        .collect(),
                )
            }
            VtkDataType::Float => {
                let raw = self.consume_bytes(count * 4)?;
                ArrayData::F32(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&c| f32::from_be_bytes(c))
                        .collect(),
                )
            }
            VtkDataType::Double => {
                let raw = self.consume_bytes(count * 8)?;
                ArrayData::F64(
                    raw.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|&c| f64::from_be_bytes(c))
                        .collect(),
                )
            }
        };
        Ok(data)
    }
}

// ================================================================================================
// Writer
// ================================================================================================

/// Writes a [`DataSet`] to a Legacy VTK (`.vtk`) file in ASCII format.
pub fn write_vtk_legacy(path: impl AsRef<Path>, dataset: &DataSet) -> Result<()> {
    write_vtk_legacy_with_format(path, dataset, VtkLegacyFormat::Ascii)
}

/// Writes a [`DataSet`] to a Legacy VTK (`.vtk`) file with explicit ASCII or Binary encoding.
pub fn write_vtk_legacy_with_format(
    path: impl AsRef<Path>,
    dataset: &DataSet,
    format: VtkLegacyFormat,
) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "ParaViz output")?;
    writeln!(
        w,
        "{}",
        match format {
            VtkLegacyFormat::Ascii => "ASCII",
            VtkLegacyFormat::Binary => "BINARY",
        }
    )?;

    match dataset {
        DataSet::ImageData(img) => {
            writeln!(w, "DATASET STRUCTURED_POINTS")?;
            writeln!(
                w,
                "DIMENSIONS {} {} {}",
                img.dimensions[0], img.dimensions[1], img.dimensions[2]
            )?;
            writeln!(
                w,
                "ORIGIN {} {} {}",
                img.origin[0], img.origin[1], img.origin[2]
            )?;
            writeln!(
                w,
                "SPACING {} {} {}",
                img.spacing[0], img.spacing[1], img.spacing[2]
            )?;
            write_field_data_top(&mut w, &img.field_data, format)?;
            write_attribute_section(
                &mut w,
                "POINT_DATA",
                img.num_points(),
                &img.point_data,
                format,
            )?;
            write_attribute_section(
                &mut w,
                "CELL_DATA",
                img.num_cells(),
                &img.cell_data,
                format,
            )?;
        }
        DataSet::RectilinearGrid(rg) => {
            writeln!(w, "DATASET RECTILINEAR_GRID")?;
            writeln!(
                w,
                "DIMENSIONS {} {} {}",
                rg.x_coords.len(),
                rg.y_coords.len(),
                rg.z_coords.len()
            )?;
            writeln!(w, "X_COORDINATES {} float", rg.x_coords.len())?;
            write_f32_slice(&mut w, &rg.x_coords, 1, format)?;
            writeln!(w, "Y_COORDINATES {} float", rg.y_coords.len())?;
            write_f32_slice(&mut w, &rg.y_coords, 1, format)?;
            writeln!(w, "Z_COORDINATES {} float", rg.z_coords.len())?;
            write_f32_slice(&mut w, &rg.z_coords, 1, format)?;
            write_field_data_top(&mut w, &rg.field_data, format)?;
            write_attribute_section(
                &mut w,
                "POINT_DATA",
                rg.num_points(),
                &rg.point_data,
                format,
            )?;
            write_attribute_section(
                &mut w,
                "CELL_DATA",
                rg.num_cells(),
                &rg.cell_data,
                format,
            )?;
        }
        DataSet::StructuredGrid(sg) => {
            writeln!(w, "DATASET STRUCTURED_GRID")?;
            writeln!(
                w,
                "DIMENSIONS {} {} {}",
                sg.dimensions[0], sg.dimensions[1], sg.dimensions[2]
            )?;
            writeln!(w, "POINTS {} float", sg.points.len())?;
            write_points(&mut w, &sg.points, format)?;
            write_field_data_top(&mut w, &sg.field_data, format)?;
            write_attribute_section(
                &mut w,
                "POINT_DATA",
                sg.num_points(),
                &sg.point_data,
                format,
            )?;
            write_attribute_section(
                &mut w,
                "CELL_DATA",
                sg.num_cells(),
                &sg.cell_data,
                format,
            )?;
        }
        DataSet::PolyData(poly) => {
            writeln!(w, "DATASET POLYDATA")?;
            writeln!(w, "POINTS {} float", poly.points.len())?;
            write_points(&mut w, &poly.points, format)?;
            if !poly.verts.is_empty() {
                writeln!(w, "VERTICES {} {}", poly.verts.len(), poly.verts.len() * 2)?;
                let vert_cells: Vec<Vec<u32>> = poly.verts.iter().map(|&v| vec![v]).collect();
                write_cell_lists(&mut w, &vert_cells, format)?;
            }
            if !poly.lines.is_empty() {
                let total: usize = poly.lines.iter().map(|l| 1 + l.len()).sum();
                writeln!(w, "LINES {} {}", poly.lines.len(), total)?;
                write_cell_lists(&mut w, &poly.lines, format)?;
            }
            if !poly.polys.is_empty() {
                let total: usize = poly.polys.iter().map(|p| 1 + p.len()).sum();
                writeln!(w, "POLYGONS {} {}", poly.polys.len(), total)?;
                write_cell_lists(&mut w, &poly.polys, format)?;
            }
            if !poly.strips.is_empty() {
                let total: usize = poly.strips.iter().map(|s| 1 + s.len()).sum();
                writeln!(w, "TRIANGLE_STRIPS {} {}", poly.strips.len(), total)?;
                write_cell_lists(&mut w, &poly.strips, format)?;
            }
            write_field_data_top(&mut w, &poly.field_data, format)?;
            write_attribute_section(
                &mut w,
                "POINT_DATA",
                poly.num_points(),
                &poly.point_data,
                format,
            )?;
            write_attribute_section(
                &mut w,
                "CELL_DATA",
                poly.num_cells(),
                &poly.cell_data,
                format,
            )?;
        }
        DataSet::UnstructuredGrid(ug) => {
            writeln!(w, "DATASET UNSTRUCTURED_GRID")?;
            writeln!(w, "POINTS {} float", ug.points.len())?;
            write_points(&mut w, &ug.points, format)?;
            let cells: Vec<Vec<u32>> = (0..ug.num_cells())
                .map(|i| ug.cell_points(i).to_vec())
                .collect();
            let total: usize = cells.iter().map(|c| 1 + c.len()).sum();
            writeln!(w, "CELLS {} {}", cells.len(), total)?;
            write_cell_lists(&mut w, &cells, format)?;
            writeln!(w, "CELL_TYPES {}", ug.cell_types.len())?;
            let ct_ids: Vec<i32> = ug.cell_types.iter().map(|ct| ct.vtk_id() as i32).collect();
            write_i32_slice(&mut w, &ct_ids, 1, format)?;
            write_field_data_top(&mut w, &ug.field_data, format)?;
            write_attribute_section(
                &mut w,
                "POINT_DATA",
                ug.num_points(),
                &ug.point_data,
                format,
            )?;
            write_attribute_section(
                &mut w,
                "CELL_DATA",
                ug.num_cells(),
                &ug.cell_data,
                format,
            )?;
        }
        DataSet::MultiBlock(_) => {
            return Err(IoError::UnsupportedFormat(
                "Legacy .vtk format does not support MultiBlockDataSet; use .vtm".into(),
            ))
        }
    }

    w.flush()?;
    Ok(())
}

fn write_points(w: &mut impl Write, points: &[[f32; 3]], format: VtkLegacyFormat) -> Result<()> {
    let mut flat = Vec::with_capacity(points.len() * 3);
    for p in points {
        flat.extend_from_slice(p);
    }
    write_f32_slice(w, &flat, 3, format)
}

fn write_cell_lists(
    w: &mut impl Write,
    cells: &[Vec<u32>],
    format: VtkLegacyFormat,
) -> Result<()> {
    match format {
        VtkLegacyFormat::Ascii => {
            for c in cells {
                write!(w, "{}", c.len())?;
                for &idx in c {
                    write!(w, " {idx}")?;
                }
                writeln!(w)?;
            }
        }
        VtkLegacyFormat::Binary => {
            for c in cells {
                w.write_all(&(c.len() as i32).to_be_bytes())?;
                for &idx in c {
                    w.write_all(&(idx as i32).to_be_bytes())?;
                }
            }
            writeln!(w)?;
        }
    }
    Ok(())
}

fn write_f32_slice(
    w: &mut impl Write,
    vals: &[f32],
    tuple_width: usize,
    format: VtkLegacyFormat,
) -> Result<()> {
    match format {
        VtkLegacyFormat::Ascii => {
            for (i, v) in vals.iter().enumerate() {
                if i > 0 && i % tuple_width == 0 {
                    writeln!(w)?;
                } else if i > 0 {
                    write!(w, " ")?;
                }
                write!(w, "{v}")?;
            }
            writeln!(w)?;
        }
        VtkLegacyFormat::Binary => {
            for &v in vals {
                w.write_all(&v.to_be_bytes())?;
            }
            writeln!(w)?;
        }
    }
    Ok(())
}

fn write_i32_slice(
    w: &mut impl Write,
    vals: &[i32],
    tuple_width: usize,
    format: VtkLegacyFormat,
) -> Result<()> {
    match format {
        VtkLegacyFormat::Ascii => {
            for (i, v) in vals.iter().enumerate() {
                if i > 0 && i % tuple_width == 0 {
                    writeln!(w)?;
                } else if i > 0 {
                    write!(w, " ")?;
                }
                write!(w, "{v}")?;
            }
            writeln!(w)?;
        }
        VtkLegacyFormat::Binary => {
            for &v in vals {
                w.write_all(&v.to_be_bytes())?;
            }
            writeln!(w)?;
        }
    }
    Ok(())
}

fn write_field_data_top(
    w: &mut impl Write,
    field: &FieldData,
    format: VtkLegacyFormat,
) -> Result<()> {
    if field.is_empty() {
        return Ok(());
    }
    writeln!(w, "FIELD FieldData {}", field.len())?;
    for arr in field.arrays() {
        write_field_array(w, arr, format)?;
    }
    Ok(())
}

fn write_attribute_section(
    w: &mut impl Write,
    section_kw: &str,
    num_tuples: usize,
    field: &FieldData,
    format: VtkLegacyFormat,
) -> Result<()> {
    if field.is_empty() {
        return Ok(());
    }
    writeln!(w, "{section_kw} {num_tuples}")?;

    let mut written: Vec<String> = Vec::new();

    if let Some(scalars) = field.active_scalars_array() {
        writeln!(
            w,
            "SCALARS {} {} {}",
            sanitize_vtk_name(&scalars.name),
            scalars.data.scalar_type().vtk_legacy_name(),
            scalars.num_components
        )?;
        writeln!(w, "LOOKUP_TABLE default")?;
        write_array_data(w, scalars, format)?;
        written.push(scalars.name.clone());
    }

    if let Some(vectors) = field.active_vectors_array()
        && vectors.num_components == 3
        && !written.contains(&vectors.name)
    {
        writeln!(
            w,
            "VECTORS {} {}",
            sanitize_vtk_name(&vectors.name),
            vectors.data.scalar_type().vtk_legacy_name()
        )?;
        write_array_data(w, vectors, format)?;
        written.push(vectors.name.clone());
    }

    if let Some(normals) = field.active_normals_array()
        && normals.num_components == 3
        && !written.contains(&normals.name)
    {
        writeln!(
            w,
            "NORMALS {} {}",
            sanitize_vtk_name(&normals.name),
            normals.data.scalar_type().vtk_legacy_name()
        )?;
        write_array_data(w, normals, format)?;
        written.push(normals.name.clone());
    }

    let remaining: Vec<&DataArray> = field
        .arrays()
        .iter()
        .filter(|a| !written.contains(&a.name))
        .collect();
    if !remaining.is_empty() {
        writeln!(w, "FIELD FieldData {}", remaining.len())?;
        for arr in remaining {
            write_field_array(w, arr, format)?;
        }
    }

    Ok(())
}

fn write_field_array(
    w: &mut impl Write,
    arr: &DataArray,
    format: VtkLegacyFormat,
) -> Result<()> {
    writeln!(
        w,
        "{} {} {} {}",
        sanitize_vtk_name(&arr.name),
        arr.num_components,
        arr.num_tuples(),
        arr.data.scalar_type().vtk_legacy_name()
    )?;
    write_array_data(w, arr, format)
}

fn write_array_data(w: &mut impl Write, arr: &DataArray, format: VtkLegacyFormat) -> Result<()> {
    let nc = arr.num_components.max(1);
    match format {
        VtkLegacyFormat::Ascii => {
            match &arr.data {
                ArrayData::F32(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::F64(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::I8(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::I16(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::I32(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::I64(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::U8(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::U16(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::U32(v) => write_ascii_vals(w, v, nc)?,
                ArrayData::U64(v) => write_ascii_vals(w, v, nc)?,
            }
            writeln!(w)?;
        }
        VtkLegacyFormat::Binary => {
            match &arr.data {
                ArrayData::F32(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::F64(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::I8(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::I16(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::I32(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::I64(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::U8(v) => {
                    w.write_all(v)?;
                }
                ArrayData::U16(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::U32(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
                ArrayData::U64(v) => {
                    for &x in v {
                        w.write_all(&x.to_be_bytes())?;
                    }
                }
            }
            writeln!(w)?;
        }
    }
    Ok(())
}

fn write_ascii_vals<T: std::fmt::Display>(
    w: &mut impl Write,
    vals: &[T],
    tuple_width: usize,
) -> Result<()> {
    for (i, v) in vals.iter().enumerate() {
        if i > 0 && i % tuple_width == 0 {
            writeln!(w)?;
        } else if i > 0 {
            write!(w, " ")?;
        }
        write!(w, "{v}")?;
    }
    Ok(())
}

fn sanitize_vtk_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect();
    if s.is_empty() {
        "array".to_string()
    } else {
        s
    }
}

fn parse_usize(s: &str) -> Result<usize> {
    s.parse::<usize>()
        .map_err(|_| IoError::Parse(format!("Invalid integer '{s}'")))
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .map_err(|_| IoError::Parse(format!("Invalid float '{s}'")))
}
