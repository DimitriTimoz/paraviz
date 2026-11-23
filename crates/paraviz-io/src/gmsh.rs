//! Gmsh (`.msh`) mesh and solution reader supporting MSH v2.2 (ASCII & Binary) and v4.1 (ASCII & Binary).
//!
//! Reads `$Nodes`, `$Elements`, `$NodeData`, and `$ElementData` into an [`UnstructuredGrid`].

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, CellType, DataArray, DataSet, UnstructuredGrid};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reads a Gmsh `.msh` file (v2.x or v4.x, ASCII or Binary) into a [`DataSet`].
pub fn read_gmsh(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_gmsh_bytes(&bytes)
}

/// Parses Gmsh `.msh` bytes into a [`DataSet`].
pub fn read_gmsh_bytes(bytes: &[u8]) -> Result<DataSet> {
    let mut cursor = 0usize;
    let mut version = 2.2f64;
    let mut is_binary = false;

    let mut node_map: HashMap<u64, u32> = HashMap::new();
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut cells: Vec<(CellType, Vec<u32>)> = Vec::new();
    let mut physical_tags: Vec<i32> = Vec::new();
    let mut entity_tags: Vec<i32> = Vec::new();

    let mut grid = UnstructuredGrid::new();

    while let Some(section) = next_line_str(bytes, &mut cursor) {
        let sec = section.trim();
        if sec.is_empty() {
            continue;
        }
        match sec {
            "$MeshFormat" => {
                let fmt_line = next_line_str(bytes, &mut cursor)
                    .ok_or_else(|| IoError::Parse("Unexpected EOF in Gmsh $MeshFormat".into()))?;
                let parts: Vec<&str> = fmt_line.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(IoError::Parse(format!("Invalid Gmsh $MeshFormat '{fmt_line}'")));
                }
                version = parts[0]
                    .parse()
                    .map_err(|_| IoError::Parse(format!("Invalid Gmsh version '{}'", parts[0])))?;
                let file_type: i32 = parts[1].parse().unwrap_or(0);
                is_binary = file_type == 1;
                if is_binary {
                    // 4-byte endianness check integer (1) + newline
                    if cursor + 4 > bytes.len() {
                        return Err(IoError::Parse("Truncated Gmsh binary endianness check".into()));
                    }
                    cursor += 4;
                    skip_optional_newline(bytes, &mut cursor);
                }
                skip_until_end_tag(bytes, &mut cursor, "$EndMeshFormat")?;
            }
            "$Nodes" => {
                if version >= 4.0 {
                    parse_nodes_v4(bytes, &mut cursor, is_binary, &mut node_map, &mut points)?;
                } else {
                    parse_nodes_v2(bytes, &mut cursor, is_binary, &mut node_map, &mut points)?;
                }
                skip_until_end_tag(bytes, &mut cursor, "$EndNodes")?;
            }
            "$Elements" => {
                if version >= 4.0 {
                    parse_elements_v4(
                        bytes,
                        &mut cursor,
                        is_binary,
                        &node_map,
                        &mut cells,
                        &mut physical_tags,
                        &mut entity_tags,
                    )?;
                } else {
                    parse_elements_v2(
                        bytes,
                        &mut cursor,
                        is_binary,
                        &node_map,
                        &mut cells,
                        &mut physical_tags,
                        &mut entity_tags,
                    )?;
                }
                skip_until_end_tag(bytes, &mut cursor, "$EndElements")?;
            }
            "$NodeData" => {
                if !is_binary {
                    if let Some(arr) = parse_field_data_ascii(bytes, &mut cursor, points.len(), Some(&node_map))? {
                        grid.point_data.add_array(arr);
                    }
                }
                skip_until_end_tag(bytes, &mut cursor, "$EndNodeData")?;
            }
            "$ElementData" => {
                if !is_binary {
                    if let Some(arr) = parse_field_data_ascii(bytes, &mut cursor, cells.len(), None)? {
                        grid.cell_data.add_array(arr);
                    }
                }
                skip_until_end_tag(bytes, &mut cursor, "$EndElementData")?;
            }
            _ if sec.starts_with('$') => {
                let end_tag = format!("$End{}", &sec[1..]);
                skip_until_end_tag(bytes, &mut cursor, &end_tag)?;
            }
            _ => {}
        }
    }

    if points.is_empty() {
        return Err(IoError::Parse("Gmsh file contains no nodes".into()));
    }

    grid.points = points;
    for (ct, conn) in cells {
        grid.push_cell(ct, &conn);
    }
    if !physical_tags.is_empty() && physical_tags.len() == grid.num_cells() {
        grid.cell_data.add_array(DataArray::new(
            "PhysicalTag",
            1,
            ArrayData::I32(physical_tags),
        ));
    }
    if !entity_tags.is_empty() && entity_tags.len() == grid.num_cells() {
        grid.cell_data.add_array(DataArray::new(
            "EntityTag",
            1,
            ArrayData::I32(entity_tags),
        ));
    }
    Ok(DataSet::UnstructuredGrid(grid))
}

fn gmsh_elem_to_vtk(gmsh_type: i32) -> Option<(CellType, usize)> {
    match gmsh_type {
        1 => Some((CellType::Line, 2)),
        2 => Some((CellType::Triangle, 3)),
        3 => Some((CellType::Quad, 4)),
        4 => Some((CellType::Tetra, 4)),
        5 => Some((CellType::Hexahedron, 8)),
        6 => Some((CellType::Wedge, 6)),
        7 => Some((CellType::Pyramid, 5)),
        8 => Some((CellType::QuadraticEdge, 3)),
        9 => Some((CellType::QuadraticTriangle, 6)),
        10 => Some((CellType::QuadraticQuad, 9)),
        11 => Some((CellType::QuadraticTetra, 10)),
        12 => Some((CellType::QuadraticHexahedron, 27)),
        15 => Some((CellType::Vertex, 1)),
        16 => Some((CellType::QuadraticQuad, 8)),
        17 => Some((CellType::QuadraticHexahedron, 20)),
        _ => None,
    }
}

fn parse_nodes_v2(
    bytes: &[u8],
    cursor: &mut usize,
    is_binary: bool,
    node_map: &mut HashMap<u64, u32>,
    points: &mut Vec<[f32; 3]>,
) -> Result<()> {
    let count_line = next_line_str(bytes, cursor)
        .ok_or_else(|| IoError::Parse("Unexpected EOF in Gmsh v2 $Nodes".into()))?;
    let n: usize = count_line
        .trim()
        .parse()
        .map_err(|_| IoError::Parse(format!("Invalid Gmsh node count '{count_line}'")))?;
    points.reserve(safe_prealloc_capacity(n, 12, MAX_ALLOC_BYTES)?);

    if is_binary {
        for _ in 0..n {
            let id = read_i32_le(bytes, cursor)? as u64;
            let x = read_f64_le(bytes, cursor)? as f32;
            let y = read_f64_le(bytes, cursor)? as f32;
            let z = read_f64_le(bytes, cursor)? as f32;
            let idx = points.len() as u32;
            node_map.insert(id, idx);
            points.push([x, y, z]);
        }
        skip_optional_newline(bytes, cursor);
    } else {
        for _ in 0..n {
            let line = next_line_str(bytes, cursor)
                .ok_or_else(|| IoError::Parse("Truncated Gmsh v2 $Nodes".into()))?;
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() < 4 {
                return Err(IoError::Parse(format!("Invalid Gmsh node line '{line}'")));
            }
            let id: u64 = p[0]
                .parse()
                .map_err(|_| IoError::Parse(format!("Invalid Gmsh node id '{}'", p[0])))?;
            let x: f32 = p[1].parse().unwrap_or(0.0);
            let y: f32 = p[2].parse().unwrap_or(0.0);
            let z: f32 = p[3].parse().unwrap_or(0.0);
            let idx = points.len() as u32;
            node_map.insert(id, idx);
            points.push([x, y, z]);
        }
    }
    Ok(())
}

fn parse_nodes_v4(
    bytes: &[u8],
    cursor: &mut usize,
    is_binary: bool,
    node_map: &mut HashMap<u64, u32>,
    points: &mut Vec<[f32; 3]>,
) -> Result<()> {
    if is_binary {
        let num_blocks = read_u64_le(bytes, cursor)? as usize;
        let total_nodes = read_u64_le(bytes, cursor)? as usize;
        let _min_tag = read_u64_le(bytes, cursor)?;
        let _max_tag = read_u64_le(bytes, cursor)?;
        points.reserve(safe_prealloc_capacity(total_nodes, 12, MAX_ALLOC_BYTES)?);
        for _ in 0..num_blocks {
            let _dim = read_i32_le(bytes, cursor)?;
            let _tag = read_i32_le(bytes, cursor)?;
            let _parametric = read_i32_le(bytes, cursor)?;
            let n_in_block = read_u64_le(bytes, cursor)? as usize;
            let mut tags = Vec::with_capacity(safe_prealloc_capacity(n_in_block, 8, MAX_ALLOC_BYTES)?);
            for _ in 0..n_in_block {
                tags.push(read_u64_le(bytes, cursor)?);
            }
            for tag in tags {
                let x = read_f64_le(bytes, cursor)? as f32;
                let y = read_f64_le(bytes, cursor)? as f32;
                let z = read_f64_le(bytes, cursor)? as f32;
                let idx = points.len() as u32;
                node_map.insert(tag, idx);
                points.push([x, y, z]);
            }
        }
        skip_optional_newline(bytes, cursor);
    } else {
        let hdr = next_line_str(bytes, cursor)
            .ok_or_else(|| IoError::Parse("Unexpected EOF in Gmsh v4 $Nodes".into()))?;
        let hp: Vec<&str> = hdr.split_whitespace().collect();
        if hp.len() < 2 {
            return Err(IoError::Parse(format!("Invalid Gmsh v4 $Nodes header '{hdr}'")));
        }
        let num_blocks: usize = hp[0].parse().unwrap_or(0);
        let total_nodes: usize = hp[1].parse().unwrap_or(0);
        points.reserve(safe_prealloc_capacity(total_nodes, 12, MAX_ALLOC_BYTES)?);

        for _ in 0..num_blocks {
            let bh = next_line_str(bytes, cursor)
                .ok_or_else(|| IoError::Parse("Truncated Gmsh v4 node block".into()))?;
            let bp: Vec<&str> = bh.split_whitespace().collect();
            let n_in_block: usize = bp.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let mut tags = Vec::with_capacity(safe_prealloc_capacity(n_in_block, 8, MAX_ALLOC_BYTES)?);
            for _ in 0..n_in_block {
                let tl = next_line_str(bytes, cursor)
                    .ok_or_else(|| IoError::Parse("Truncated Gmsh v4 node tags".into()))?;
                let tag: u64 = tl.trim().parse().unwrap_or(0);
                tags.push(tag);
            }
            for tag in tags {
                let cl = next_line_str(bytes, cursor)
                    .ok_or_else(|| IoError::Parse("Truncated Gmsh v4 node coords".into()))?;
                let cp: Vec<&str> = cl.split_whitespace().collect();
                let x: f32 = cp.first().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = cp.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = cp.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let idx = points.len() as u32;
                node_map.insert(tag, idx);
                points.push([x, y, z]);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_elements_v2(
    bytes: &[u8],
    cursor: &mut usize,
    is_binary: bool,
    node_map: &HashMap<u64, u32>,
    cells: &mut Vec<(CellType, Vec<u32>)>,
    physical_tags: &mut Vec<i32>,
    entity_tags: &mut Vec<i32>,
) -> Result<()> {
    let count_line = next_line_str(bytes, cursor)
        .ok_or_else(|| IoError::Parse("Unexpected EOF in Gmsh v2 $Elements".into()))?;
    let total: usize = count_line.trim().parse().unwrap_or(0);

    if is_binary {
        let mut read_so_far = 0usize;
        while read_so_far < total {
            let elem_type = read_i32_le(bytes, cursor)?;
            let n_elems = read_i32_le(bytes, cursor)? as usize;
            let n_tags = read_i32_le(bytes, cursor)? as usize;
            let mapped = gmsh_elem_to_vtk(elem_type);
            let n_nodes = mapped.map(|(_, c)| c).unwrap_or(0);
            for _ in 0..n_elems {
                let _elem_id = read_i32_le(bytes, cursor)?;
                let mut phys = 0i32;
                let mut ent = 0i32;
                for ti in 0..n_tags {
                    let t = read_i32_le(bytes, cursor)?;
                    if ti == 0 {
                        phys = t;
                    } else if ti == 1 {
                        ent = t;
                    }
                }
                let mut conn = Vec::with_capacity(n_nodes);
                let mut ok = mapped.is_some();
                for _ in 0..n_nodes {
                    let nid = read_i32_le(bytes, cursor)? as u64;
                    if let Some(&idx) = node_map.get(&nid) {
                        conn.push(idx);
                    } else {
                        ok = false;
                    }
                }
                if ok {
                    if let Some((ct, _)) = mapped {
                        cells.push((ct, conn));
                        physical_tags.push(phys);
                        entity_tags.push(ent);
                    }
                }
                read_so_far += 1;
            }
        }
        skip_optional_newline(bytes, cursor);
    } else {
        for _ in 0..total {
            let line = next_line_str(bytes, cursor)
                .ok_or_else(|| IoError::Parse("Truncated Gmsh v2 $Elements".into()))?;
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() < 4 {
                continue;
            }
            let elem_type: i32 = p[1].parse().unwrap_or(0);
            let n_tags: usize = p[2].parse().unwrap_or(0);
            let Some((ct, n_nodes)) = gmsh_elem_to_vtk(elem_type) else {
                continue;
            };
            let phys = if n_tags >= 1 {
                p.get(3).and_then(|s| s.parse().ok()).unwrap_or(0)
            } else {
                0
            };
            let ent = if n_tags >= 2 {
                p.get(4).and_then(|s| s.parse().ok()).unwrap_or(0)
            } else {
                0
            };
            let node_start = 3 + n_tags;
            if p.len() < node_start + n_nodes {
                continue;
            }
            let mut conn = Vec::with_capacity(n_nodes);
            let mut ok = true;
            for tok in &p[node_start..node_start + n_nodes] {
                if let Ok(nid) = tok.parse::<u64>() {
                    if let Some(&idx) = node_map.get(&nid) {
                        conn.push(idx);
                        continue;
                    }
                }
                ok = false;
                break;
            }
            if ok {
                cells.push((ct, conn));
                physical_tags.push(phys);
                entity_tags.push(ent);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_elements_v4(
    bytes: &[u8],
    cursor: &mut usize,
    is_binary: bool,
    node_map: &HashMap<u64, u32>,
    cells: &mut Vec<(CellType, Vec<u32>)>,
    physical_tags: &mut Vec<i32>,
    entity_tags: &mut Vec<i32>,
) -> Result<()> {
    if is_binary {
        let num_blocks = read_u64_le(bytes, cursor)? as usize;
        let _total = read_u64_le(bytes, cursor)?;
        let _min_tag = read_u64_le(bytes, cursor)?;
        let _max_tag = read_u64_le(bytes, cursor)?;
        for _ in 0..num_blocks {
            let _dim = read_i32_le(bytes, cursor)?;
            let ent_tag = read_i32_le(bytes, cursor)?;
            let elem_type = read_i32_le(bytes, cursor)?;
            let n_in_block = read_u64_le(bytes, cursor)? as usize;
            let Some((ct, n_nodes)) = gmsh_elem_to_vtk(elem_type) else {
                return Err(IoError::UnsupportedFormat(format!(
                    "Unsupported binary Gmsh v4 element type {elem_type}"
                )));
            };
            for _ in 0..n_in_block {
                let _elem_tag = read_u64_le(bytes, cursor)?;
                let mut conn = Vec::with_capacity(n_nodes);
                let mut ok = true;
                for _ in 0..n_nodes {
                    let nid = read_u64_le(bytes, cursor)?;
                    if let Some(&idx) = node_map.get(&nid) {
                        conn.push(idx);
                    } else {
                        ok = false;
                    }
                }
                if ok {
                    cells.push((ct, conn));
                    physical_tags.push(ent_tag);
                    entity_tags.push(ent_tag);
                }
            }
        }
        skip_optional_newline(bytes, cursor);
    } else {
        let hdr = next_line_str(bytes, cursor)
            .ok_or_else(|| IoError::Parse("Unexpected EOF in Gmsh v4 $Elements".into()))?;
        let hp: Vec<&str> = hdr.split_whitespace().collect();
        let num_blocks: usize = hp.first().and_then(|s| s.parse().ok()).unwrap_or(0);

        for _ in 0..num_blocks {
            let bh = next_line_str(bytes, cursor)
                .ok_or_else(|| IoError::Parse("Truncated Gmsh v4 element block".into()))?;
            let bp: Vec<&str> = bh.split_whitespace().collect();
            let ent_tag: i32 = bp.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let elem_type: i32 = bp.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            let n_in_block: usize = bp.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let mapped = gmsh_elem_to_vtk(elem_type);
            for _ in 0..n_in_block {
                let el = next_line_str(bytes, cursor)
                    .ok_or_else(|| IoError::Parse("Truncated Gmsh v4 element line".into()))?;
                let Some((ct, n_nodes)) = mapped else {
                    continue;
                };
                let ep: Vec<&str> = el.split_whitespace().collect();
                if ep.len() < 1 + n_nodes {
                    continue;
                }
                let mut conn = Vec::with_capacity(n_nodes);
                let mut ok = true;
                for tok in &ep[1..1 + n_nodes] {
                    if let Ok(nid) = tok.parse::<u64>() {
                        if let Some(&idx) = node_map.get(&nid) {
                            conn.push(idx);
                            continue;
                        }
                    }
                    ok = false;
                    break;
                }
                if ok {
                    cells.push((ct, conn));
                    physical_tags.push(ent_tag);
                    entity_tags.push(ent_tag);
                }
            }
        }
    }
    Ok(())
}

fn parse_field_data_ascii(
    bytes: &[u8],
    cursor: &mut usize,
    expected_len: usize,
    node_map: Option<&HashMap<u64, u32>>,
) -> Result<Option<DataArray>> {
    let n_str: usize = next_line_str(bytes, cursor)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let mut name = String::from("Field");
    for i in 0..n_str {
        let s = next_line_str(bytes, cursor).unwrap_or_default();
        if i == 0 {
            name = s.trim().trim_matches('"').to_string();
        }
    }
    let n_real: usize = next_line_str(bytes, cursor)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    for _ in 0..n_real {
        let _ = next_line_str(bytes, cursor);
    }
    let n_int: usize = next_line_str(bytes, cursor)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let mut n_comp = 1usize;
    let mut n_vals = 0usize;
    for i in 0..n_int {
        let s = next_line_str(bytes, cursor).unwrap_or_default();
        let v: usize = s.trim().parse().unwrap_or(0);
        if i == 1 {
            n_comp = v.max(1);
        } else if i == 2 {
            n_vals = v;
        }
    }
    if expected_len == 0 || n_vals == 0 {
        return Ok(None);
    }

    let mut data = vec![0.0f32; expected_len * n_comp];
    for idx in 0..n_vals {
        let line = next_line_str(bytes, cursor).unwrap_or_default();
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 1 + n_comp {
            continue;
        }
        let target_idx = if let Some(map) = node_map {
            let tag: u64 = p[0].parse().unwrap_or(0);
            map.get(&tag).copied().map(|v| v as usize)
        } else if idx < expected_len {
            Some(idx)
        } else {
            None
        };
        if let Some(ti) = target_idx {
            if ti < expected_len {
                for c in 0..n_comp {
                    data[ti * n_comp + c] = p[1 + c].parse().unwrap_or(0.0);
                }
            }
        }
    }
    Ok(Some(DataArray::new(name, n_comp, ArrayData::F32(data))))
}

fn next_line_str<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a str> {
    if *cursor >= bytes.len() {
        return None;
    }
    let start = *cursor;
    while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
        *cursor += 1;
    }
    let mut end = *cursor;
    if end > start && bytes[end - 1] == b'\r' {
        end -= 1;
    }
    if *cursor < bytes.len() && bytes[*cursor] == b'\n' {
        *cursor += 1;
    }
    Some(std::str::from_utf8(&bytes[start..end]).unwrap_or(""))
}

fn skip_optional_newline(bytes: &[u8], cursor: &mut usize) {
    if *cursor < bytes.len() && bytes[*cursor] == b'\r' {
        *cursor += 1;
    }
    if *cursor < bytes.len() && bytes[*cursor] == b'\n' {
        *cursor += 1;
    }
}

fn skip_until_end_tag(bytes: &[u8], cursor: &mut usize, end_tag: &str) -> Result<()> {
    while let Some(line) = next_line_str(bytes, cursor) {
        if line.trim() == end_tag {
            return Ok(());
        }
    }
    Ok(())
}

fn read_i32_le(bytes: &[u8], cursor: &mut usize) -> Result<i32> {
    if *cursor + 4 > bytes.len() {
        return Err(IoError::Parse("Unexpected EOF reading Gmsh i32".into()));
    }
    let v = i32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(IoError::Parse("Unexpected EOF reading Gmsh u64".into()));
    }
    let v = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f64_le(bytes: &[u8], cursor: &mut usize) -> Result<f64> {
    if *cursor + 8 > bytes.len() {
        return Err(IoError::Parse("Unexpected EOF reading Gmsh f64".into()));
    }
    let v = f64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}
