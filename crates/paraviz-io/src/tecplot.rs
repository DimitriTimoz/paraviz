//! Tecplot ASCII (`.dat`, `.plt`, `.tec`) reader.
//!
//! Supports `TITLE`, `VARIABLES`, and multiple `ZONE` blocks:
//! - Ordered `I=`, `J=`, `K=` (`StructuredGrid`)
//! - Finite-element `NODES=`/`N=`, `ELEMENTS=`/`E=`, `ZONETYPE=`
//!   (`FETRIANGLE`, `FEQUADRILATERAL`, `FETETRAHEDRON`, `FEBRICK`, `FELINESEG`)
//! - `DATAPACKING=POINT` and `DATAPACKING=BLOCK`

use crate::error::{IoError, Result};
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, MultiBlockDataSet, StructuredGrid, UnstructuredGrid,
};
use std::fs;
use std::path::Path;

/// Reads a Tecplot ASCII file (`.dat`, `.tec`, `.plt`) into a [`DataSet`].
pub fn read_tecplot(path: impl AsRef<Path>) -> Result<DataSet> {
    let text = fs::read_to_string(path)?;
    read_tecplot_str(&text)
}

/// Parses a Tecplot ASCII string into a [`DataSet`].
pub fn read_tecplot_str(text: impl AsRef<str>) -> Result<DataSet> {
    let text = text.as_ref();
    let mut variables: Vec<String> = Vec::new();
    let mut blocks: Vec<(String, DataSet)> = Vec::new();

    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;

    // Parse header until first ZONE
    while idx < lines.len() {
        let raw = strip_comment(lines[idx]).trim();
        if raw.is_empty() {
            idx += 1;
            continue;
        }
        let upper = raw.to_ascii_uppercase();
        if upper.starts_with("ZONE") {
            break;
        }
        if upper.starts_with("VARIABLES") {
            let mut combined = raw.to_string();
            while idx + 1 < lines.len() {
                let next_u = strip_comment(lines[idx + 1]).trim().to_ascii_uppercase();
                if next_u.starts_with("ZONE") || next_u.starts_with("TITLE") {
                    break;
                }
                if next_u.starts_with('"') {
                    idx += 1;
                    combined.push(' ');
                    combined.push_str(strip_comment(lines[idx]).trim());
                } else {
                    break;
                }
            }
            variables = parse_variables_list(&combined);
        }
        idx += 1;
    }

    if variables.is_empty() {
        variables = vec!["X".into(), "Y".into(), "Z".into()];
    }

    while idx < lines.len() {
        let raw = strip_comment(lines[idx]).trim();
        if raw.is_empty() || !raw.to_ascii_uppercase().starts_with("ZONE") {
            idx += 1;
            continue;
        }
        // Gather full ZONE header (may span multiple lines until numeric data starts)
        let mut zone_hdr = raw.to_string();
        idx += 1;
        while idx < lines.len() {
            let next = strip_comment(lines[idx]).trim();
            if next.is_empty() {
                idx += 1;
                continue;
            }
            let first_tok = next.split_whitespace().next().unwrap_or("");
            if first_tok.parse::<f64>().is_ok()
                || next.to_ascii_uppercase().starts_with("ZONE")
            {
                break;
            }
            zone_hdr.push(' ');
            zone_hdr.push_str(next);
            idx += 1;
        }

        // Gather numeric data lines until next ZONE
        let mut data_tokens: Vec<&str> = Vec::new();
        while idx < lines.len() {
            let line = strip_comment(lines[idx]).trim();
            if line.to_ascii_uppercase().starts_with("ZONE") {
                break;
            }
            for tok in line.split(|c: char| c.is_whitespace() || c == ',') {
                if !tok.is_empty() {
                    data_tokens.push(tok);
                }
            }
            idx += 1;
        }

        let (zone_name, ds) = parse_zone(&zone_hdr, &variables, &data_tokens, blocks.len())?;
        blocks.push((zone_name, ds));
    }

    if blocks.is_empty() {
        return Err(IoError::Parse("Tecplot file contains no ZONE blocks".into()));
    }
    if blocks.len() == 1 {
        Ok(blocks.into_iter().next().unwrap().1)
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, ds) in blocks {
            mb.add_block(name, ds);
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn parse_zone(
    hdr: &str,
    variables: &[String],
    tokens: &[&str],
    zone_idx: usize,
) -> Result<(String, DataSet)> {
    let zone_name = extract_kv_string(hdr, "T")
        .unwrap_or_else(|| format!("Zone_{zone_idx}"));
    let ni = extract_kv_usize(hdr, "I");
    let nj = extract_kv_usize(hdr, "J");
    let nk = extract_kv_usize(hdr, "K");
    let n_nodes = extract_kv_usize(hdr, "NODES").or_else(|| extract_kv_usize(hdr, "N"));
    let n_elems = extract_kv_usize(hdr, "ELEMENTS").or_else(|| extract_kv_usize(hdr, "E"));
    let zone_type = extract_kv_string(hdr, "ZONETYPE")
        .or_else(|| extract_kv_string(hdr, "ET"))
        .unwrap_or_default()
        .to_ascii_uppercase();
    let is_block = extract_kv_string(hdr, "DATAPACKING")
        .or_else(|| extract_kv_string(hdr, "F"))
        .map(|s| s.to_ascii_uppercase().contains("BLOCK"))
        .unwrap_or(false);

    let n_vars = variables.len();
    if n_vars == 0 {
        return Err(IoError::Parse("Tecplot ZONE has 0 variables".into()));
    }
    let x_idx = find_var_idx(variables, &["X"]).unwrap_or(0);
    let y_idx = find_var_idx(variables, &["Y"]).unwrap_or(1.min(n_vars.saturating_sub(1)));
    let z_idx = find_var_idx(variables, &["Z"]);

    if let (Some(nodes), Some(elems)) = (n_nodes, n_elems) {
        let mut pos = 0usize;
        let mut cols = parse_tecplot_columns(tokens, &mut pos, n_vars, nodes, is_block);

        let points: Vec<[f32; 3]> = (0..nodes)
            .map(|i| {
                [
                    cols[x_idx][i],
                    cols[y_idx][i],
                    z_idx.map(|zi| cols[zi][i]).unwrap_or(0.0),
                ]
            })
            .collect();

        let (ct, n_per_cell) = match zone_type.as_str() {
            "FELINESEG" | "LINESEG" => (CellType::Line, 2),
            "FETRIANGLE" | "TRIANGLE" => (CellType::Triangle, 3),
            "FEQUADRILATERAL" | "QUADRILATERAL" | "QUAD" => (CellType::Quad, 4),
            "FETETRAHEDRON" | "TETRAHEDRON" | "TETRA" => (CellType::Tetra, 4),
            "FEBRICK" | "BRICK" => (CellType::Hexahedron, 8),
            _ => (CellType::Triangle, 3),
        };

        let mut grid = UnstructuredGrid::new();
        grid.points = points;
        for _ in 0..elems {
            if pos + n_per_cell > tokens.len() {
                break;
            }
            let mut conn = Vec::with_capacity(n_per_cell);
            for k in 0..n_per_cell {
                let v1: u32 = tokens[pos + k].parse().unwrap_or(1);
                conn.push(v1.saturating_sub(1));
            }
            pos += n_per_cell;
            grid.push_cell(ct, &conn);
        }
        for (vi, vname) in variables.iter().enumerate() {
            if vi == x_idx || vi == y_idx || Some(vi) == z_idx {
                continue;
            }
            grid.point_data.add_array(DataArray::new(
                vname.clone(),
                1,
                ArrayData::F32(std::mem::take(&mut cols[vi])),
            ));
        }
        Ok((zone_name, DataSet::UnstructuredGrid(grid)))
    } else {
        let i_dim = ni.unwrap_or(1).max(1);
        let j_dim = nj.unwrap_or(1).max(1);
        let k_dim = nk.unwrap_or(1).max(1);
        let total = i_dim * j_dim * k_dim;
        let mut pos = 0usize;
        let mut cols = parse_tecplot_columns(tokens, &mut pos, n_vars, total, is_block);
        let points: Vec<[f32; 3]> = (0..total)
            .map(|i| {
                [
                    cols[x_idx][i],
                    cols[y_idx][i],
                    z_idx.map(|zi| cols[zi][i]).unwrap_or(0.0),
                ]
            })
            .collect();
        let mut grid = StructuredGrid {
            dimensions: [i_dim, j_dim, k_dim],
            points,
            ..Default::default()
        };
        for (vi, vname) in variables.iter().enumerate() {
            if vi == x_idx || vi == y_idx || Some(vi) == z_idx {
                continue;
            }
            grid.point_data.add_array(DataArray::new(
                vname.clone(),
                1,
                ArrayData::F32(std::mem::take(&mut cols[vi])),
            ));
        }
        Ok((zone_name, DataSet::StructuredGrid(grid)))
    }
}

fn parse_tecplot_columns(
    tokens: &[&str],
    pos: &mut usize,
    n_vars: usize,
    n_rows: usize,
    is_block: bool,
) -> Vec<Vec<f32>> {
    let mut cols = vec![vec![0.0f32; n_rows]; n_vars];
    if is_block {
        for col in &mut cols {
            for cell in col.iter_mut() {
                if *pos < tokens.len() {
                    *cell = tokens[*pos].parse().unwrap_or(0.0);
                    *pos += 1;
                }
            }
        }
    } else {
        for i in 0..n_rows {
            for col in &mut cols {
                if *pos < tokens.len() {
                    col[i] = tokens[*pos].parse().unwrap_or(0.0);
                    *pos += 1;
                }
            }
        }
    }
    cols
}

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or(line)
}

fn parse_variables_list(line: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let after_eq = line.split_once('=').map(|(_, r)| r).unwrap_or(line);
    let mut in_quote = false;
    let mut cur = String::new();
    for ch in after_eq.chars() {
        if ch == '"' {
            if in_quote {
                if !cur.trim().is_empty() {
                    vars.push(cur.trim().to_string());
                }
                cur.clear();
                in_quote = false;
            } else {
                in_quote = true;
            }
        } else if in_quote {
            cur.push(ch);
        } else if ch == ',' || ch.is_whitespace() {
            if !cur.trim().is_empty() {
                vars.push(cur.trim().to_string());
            }
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        vars.push(cur.trim().to_string());
    }
    vars
}

fn extract_kv_string(hdr: &str, key: &str) -> Option<String> {
    let upper = hdr.to_ascii_uppercase();
    let needle = format!("{key}=");
    let mut search_from = 0usize;
    while let Some(pos) = upper[search_from..].find(&needle) {
        let abs = search_from + pos;
        let prev_ok = abs == 0
            || !hdr.as_bytes()[abs - 1].is_ascii_alphanumeric();
        if prev_ok {
            let rest = hdr[abs + needle.len()..].trim_start();
            if let Some(stripped) = rest.strip_prefix('"') {
                if let Some(end) = stripped.find('"') {
                    return Some(stripped[..end].to_string());
                }
            } else {
                let val: String = rest
                    .chars()
                    .take_while(|c| !c.is_whitespace() && *c != ',')
                    .collect();
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }
        search_from = abs + needle.len();
    }
    None
}

fn extract_kv_usize(hdr: &str, key: &str) -> Option<usize> {
    extract_kv_string(hdr, key).and_then(|s| s.parse::<usize>().ok())
}

fn find_var_idx(vars: &[String], candidates: &[&str]) -> Option<usize> {
    vars.iter().position(|v| {
        let vu = v.trim().to_ascii_uppercase();
        candidates.iter().any(|c| vu == *c)
    })
}
