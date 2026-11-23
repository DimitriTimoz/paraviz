use super::*;

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
        ArrayData::I8(v) => v.iter().map(|&x| x as u8).collect(),
        ArrayData::U8(v) => v.clone(),
        ArrayData::I16(v) => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        ArrayData::U16(v) => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
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
        ArrayData::U64(v) => {
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
        ArrayData::I8(v) => write_joined(w, v),
        ArrayData::U8(v) => write_joined(w, v),
        ArrayData::I16(v) => write_joined(w, v),
        ArrayData::U16(v) => write_joined(w, v),
        ArrayData::I32(v) => write_joined(w, v),
        ArrayData::U32(v) => write_joined(w, v),
        ArrayData::I64(v) => write_joined(w, v),
        ArrayData::U64(v) => write_joined(w, v),
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
