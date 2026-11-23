//! Abaqus Input Deck (`.inp`) FEA mesh reader.
//!
//! Parses `*NODE`, `*ELEMENT` (C3D4, C3D6, C3D8, C3D10, C3D20, S3, S4, CPE3/4, CPS3/4, T3D2, B31),
//! and `*NSET` / `*ELSET` set tags into an [`UnstructuredGrid`].

use crate::error::{IoError, Result};
use paraviz_core::{ArrayData, CellType, DataArray, DataSet, UnstructuredGrid};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reads an Abaqus `.inp` mesh file into a [`DataSet`].
pub fn read_abaqus(path: impl AsRef<Path>) -> Result<DataSet> {
    let text = fs::read_to_string(path)?;
    read_abaqus_str(&text)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AbaqusSection {
    None,
    Node,
    Element(CellType),
    Elset(i32),
}

/// Parses an Abaqus `.inp` string into a [`DataSet`].
pub fn read_abaqus_str(text: &str) -> Result<DataSet> {
    let mut node_map: HashMap<u64, u32> = HashMap::new();
    let mut elem_map: HashMap<u64, usize> = HashMap::new();
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut cells: Vec<(CellType, Vec<u32>)> = Vec::new();
    let mut elset_ids: Vec<i32> = Vec::new();
    let mut next_elset_id = 1i32;

    let mut section = AbaqusSection::None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("**") {
            continue;
        }
        if line.starts_with('*') {
            let upper = line.to_ascii_uppercase();
            if upper.starts_with("*NODE") && !upper.starts_with("*NODE ") || upper == "*NODE" {
                section = AbaqusSection::Node;
            } else if upper.starts_with("*NODE,") {
                section = AbaqusSection::Node;
            } else if upper.starts_with("*ELEMENT") {
                let etype = extract_param(&upper, "TYPE").unwrap_or("");
                if let Some(ct) = abaqus_type_to_vtk(etype) {
                    section = AbaqusSection::Element(ct);
                } else {
                    section = AbaqusSection::None;
                }
            } else if upper.starts_with("*ELSET") {
                let id = next_elset_id;
                next_elset_id += 1;
                section = AbaqusSection::Elset(id);
            } else {
                section = AbaqusSection::None;
            }
            continue;
        }

        match section {
            AbaqusSection::Node => {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let id: u64 = parts[0]
                        .parse()
                        .map_err(|_| IoError::Parse(format!("Invalid Abaqus node id '{}'", parts[0])))?;
                    let x: f32 = parts[1].parse().unwrap_or(0.0);
                    let y: f32 = parts[2].parse().unwrap_or(0.0);
                    let z: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let idx = points.len() as u32;
                    node_map.insert(id, idx);
                    points.push([x, y, z]);
                }
            }
            AbaqusSection::Element(ct) => {
                let parts: Vec<&str> = line
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                if parts.len() < 2 {
                    continue;
                }
                let eid: u64 = parts[0].parse().unwrap_or(0);
                let mut conn = Vec::with_capacity(parts.len() - 1);
                let mut ok = true;
                for tok in &parts[1..] {
                    if let Ok(nid) = tok.parse::<u64>() {
                        if let Some(&idx) = node_map.get(&nid) {
                            conn.push(idx);
                            continue;
                        }
                    }
                    ok = false;
                    break;
                }
                if ok && !conn.is_empty() {
                    elem_map.insert(eid, cells.len());
                    cells.push((ct, conn));
                    elset_ids.push(0);
                }
            }
            AbaqusSection::Elset(set_id) => {
                for tok in line.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    if let Ok(eid) = tok.parse::<u64>() {
                        if let Some(&cidx) = elem_map.get(&eid) {
                            if cidx < elset_ids.len() {
                                elset_ids[cidx] = set_id;
                            }
                        }
                    }
                }
            }
            AbaqusSection::None => {}
        }
    }

    if points.is_empty() {
        return Err(IoError::Parse("Abaqus .inp file contains no *NODE records".into()));
    }

    let mut grid = UnstructuredGrid::new();
    grid.points = points;
    for (ct, conn) in cells {
        grid.push_cell(ct, &conn);
    }
    if elset_ids.iter().any(|&id| id != 0) {
        grid.cell_data
            .add_array(DataArray::new("ElsetId", 1, ArrayData::I32(elset_ids)));
    }
    Ok(DataSet::UnstructuredGrid(grid))
}

fn extract_param<'a>(upper_line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=");
    for part in upper_line.split(',') {
        let p = part.trim();
        if let Some(rest) = p.strip_prefix(&needle) {
            return Some(rest.trim());
        }
    }
    None
}

fn abaqus_type_to_vtk(etype: &str) -> Option<CellType> {
    if etype.starts_with("C3D8") || etype.starts_with("DC3D8") {
        Some(CellType::Hexahedron)
    } else if etype.starts_with("C3D20") {
        Some(CellType::QuadraticHexahedron)
    } else if etype.starts_with("C3D4") || etype.starts_with("DC3D4") {
        Some(CellType::Tetra)
    } else if etype.starts_with("C3D10") {
        Some(CellType::QuadraticTetra)
    } else if etype.starts_with("C3D6") {
        Some(CellType::Wedge)
    } else if etype.starts_with("C3D5") {
        Some(CellType::Pyramid)
    } else if etype.starts_with("S4")
        || etype.starts_with("CPE4")
        || etype.starts_with("CPS4")
        || etype.starts_with("CAX4")
        || etype.starts_with("M3D4")
    {
        Some(CellType::Quad)
    } else if etype.starts_with("S3")
        || etype.starts_with("CPE3")
        || etype.starts_with("CPS3")
        || etype.starts_with("CAX3")
        || etype.starts_with("M3D3")
    {
        Some(CellType::Triangle)
    } else if etype.starts_with("T3D2") || etype.starts_with("B31") || etype.starts_with("T2D2") {
        Some(CellType::Line)
    } else {
        None
    }
}
