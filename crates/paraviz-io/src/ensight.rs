//! EnSight Gold (`.case`) scientific CFD/CAE reader producing [`UnstructuredGrid`] / [`MultiBlockDataSet`].

use crate::error::{IoError, Result};
use paraviz_core::{CellType, DataArray, DataSet, MultiBlockDataSet, UnstructuredGrid};
use std::fs;
use std::path::Path;

/// Reads an EnSight Gold `.case` file (ASCII geometry + per-node/per-element scalar & vector variables).
pub fn read_ensight_case(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let text = fs::read_to_string(path)?;

    let mut geo_file: Option<String> = None;
    let mut node_scalars: Vec<(String, String)> = Vec::new();
    let mut node_vectors: Vec<(String, String)> = Vec::new();
    let mut elem_scalars: Vec<(String, String)> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("model:") {
            let fname = rest.split_whitespace().last().unwrap_or("").trim();
            if !fname.is_empty() {
                geo_file = Some(fname.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("scalar per node:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                node_scalars.push((
                    parts[parts.len() - 2].to_string(),
                    parts[parts.len() - 1].to_string(),
                ));
            }
        } else if let Some(rest) = line.strip_prefix("vector per node:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                node_vectors.push((
                    parts[parts.len() - 2].to_string(),
                    parts[parts.len() - 1].to_string(),
                ));
            }
        } else if let Some(rest) = line.strip_prefix("scalar per element:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                elem_scalars.push((
                    parts[parts.len() - 2].to_string(),
                    parts[parts.len() - 1].to_string(),
                ));
            }
        }
    }

    let geo_name = geo_file.ok_or_else(|| {
        IoError::InvalidData("EnSight .case file missing 'model:' geometry entry".into())
    })?;
    let geo_path = crate::limits::resolve_safe_path(base_dir, &geo_name)?;
    let mut parts = read_ensight_ascii_geo(&geo_path)?;

    // Attach per-node scalar variables
    for (var_name, var_file) in node_scalars {
        let var_path = crate::limits::resolve_safe_path(base_dir, &var_file)?;
        if let Ok(part_vals) = read_ensight_ascii_var(&var_path, 1) {
            for (idx, (_, grid)) in parts.iter_mut().enumerate() {
                if let Some(vals) = part_vals.get(idx) {
                    if vals.len() == grid.points.len() {
                        grid.point_data
                            .add_array(DataArray::scalars_f32(&var_name, vals.clone()));
                    }
                }
            }
        }
    }
    // Attach per-node vector variables
    for (var_name, var_file) in node_vectors {
        let var_path = crate::limits::resolve_safe_path(base_dir, &var_file)?;
        if let Ok(part_vals) = read_ensight_ascii_var(&var_path, 3) {
            for (idx, (_, grid)) in parts.iter_mut().enumerate() {
                if let Some(vals) = part_vals.get(idx) {
                    let n = grid.points.len();
                    if vals.len() == n * 3 {
                        let vecs: Vec<[f32; 3]> = (0..n)
                            .map(|i| [vals[i], vals[n + i], vals[2 * n + i]])
                            .collect();
                        grid.point_data
                            .add_array(DataArray::vectors_f32(&var_name, vecs));
                    }
                }
            }
        }
    }
    // Attach per-element scalar variables
    for (var_name, var_file) in elem_scalars {
        let var_path = crate::limits::resolve_safe_path(base_dir, &var_file)?;
        if let Ok(part_vals) = read_ensight_ascii_var(&var_path, 1) {
            for (idx, (_, grid)) in parts.iter_mut().enumerate() {
                if let Some(vals) = part_vals.get(idx) {
                    if vals.len() == grid.num_cells() {
                        grid.cell_data
                            .add_array(DataArray::scalars_f32(&var_name, vals.clone()));
                    }
                }
            }
        }
    }

    if parts.is_empty() {
        return Err(IoError::InvalidData(
            "EnSight geometry contained no parts".into(),
        ));
    }
    if parts.len() == 1 {
        Ok(DataSet::UnstructuredGrid(parts.remove(0).1))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, grid) in parts {
            mb.add_block(name, DataSet::UnstructuredGrid(grid));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn read_ensight_ascii_geo(path: &Path) -> Result<Vec<(String, UnstructuredGrid)>> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let mut parts = Vec::new();
    let mut i = 0usize;

    while i < lines.len() {
        if lines[i] == "part" {
            i += 1;
            let part_id = lines.get(i).copied().unwrap_or("1");
            i += 1;
            let part_desc = lines.get(i).copied().unwrap_or(part_id).to_string();
            i += 1;
            let mut grid = UnstructuredGrid::new();
            while i < lines.len() && lines[i] != "part" {
                match lines[i] {
                    "coordinates" => {
                        i += 1;
                        let n: usize = lines
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| IoError::Parse("Invalid EnSight node count".into()))?;
                        i += 1;
                        let mut xs = Vec::with_capacity(n);
                        let mut ys = Vec::with_capacity(n);
                        let mut zs = Vec::with_capacity(n);
                        for _ in 0..n {
                            xs.push(lines.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0));
                            i += 1;
                        }
                        for _ in 0..n {
                            ys.push(lines.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0));
                            i += 1;
                        }
                        for _ in 0..n {
                            zs.push(lines.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0));
                            i += 1;
                        }
                        grid.points = (0..n).map(|k| [xs[k], ys[k], zs[k]]).collect();
                    }
                    "tria3" | "quad4" | "tetra4" | "hexa8" | "penta6" | "pyramid5" | "bar2" => {
                        let (ct, k) = match lines[i] {
                            "bar2" => (CellType::Line, 2),
                            "tria3" => (CellType::Triangle, 3),
                            "quad4" => (CellType::Quad, 4),
                            "tetra4" => (CellType::Tetra, 4),
                            "pyramid5" => (CellType::Pyramid, 5),
                            "penta6" => (CellType::Wedge, 6),
                            "hexa8" => (CellType::Hexahedron, 8),
                            _ => unreachable!(),
                        };
                        i += 1;
                        let ne: usize = lines.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
                        i += 1;
                        for _ in 0..ne {
                            if let Some(row) = lines.get(i) {
                                let ids: Vec<u32> = row
                                    .split_whitespace()
                                    .filter_map(|t| t.parse::<u32>().ok())
                                    .map(|id| id.saturating_sub(1))
                                    .collect();
                                if ids.len() >= k {
                                    grid.push_cell(ct, &ids[..k]);
                                }
                            }
                            i += 1;
                        }
                    }
                    _ => i += 1,
                }
            }
            parts.push((part_desc, grid));
        } else {
            i += 1;
        }
    }
    Ok(parts)
}

fn read_ensight_ascii_var(path: &Path, _components: usize) -> Result<Vec<Vec<f32>>> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i] == "part" {
            i += 2; // skip "part" and part_id
            let mut vals = Vec::new();
            while i < lines.len() && lines[i] != "part" {
                for tok in lines[i].split_whitespace() {
                    if let Ok(v) = tok.parse::<f32>() {
                        vals.push(v);
                    }
                }
                i += 1;
            }
            out.push(vals);
        } else {
            i += 1;
        }
    }
    Ok(out)
}
