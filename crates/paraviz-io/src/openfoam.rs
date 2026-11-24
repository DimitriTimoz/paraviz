//! OpenFOAM ASCII `polyMesh` reader producing [`PolyData`] (boundary faces + internal mesh faces).

use crate::error::{IoError, Result};
use paraviz_core::{DataArray, DataSet, PolyData};
use std::fs;
use std::path::Path;

/// Reads an OpenFOAM case directory (or `system/controlDict` / `constant/polyMesh/points` path)
/// into a [`PolyData`] mesh with boundary faces and any ASCII scalar/vector fields found in the
/// latest time directory.
pub fn read_openfoam(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let case_dir = if path.is_file() {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "controlDict" {
            path.parent()
                .and_then(|p| p.parent())
                .unwrap_or(path)
                .to_path_buf()
        } else if name == "points" {
            path.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .unwrap_or(path)
                .to_path_buf()
        } else {
            path.parent().unwrap_or(path).to_path_buf()
        }
    } else {
        path.to_path_buf()
    };

    let poly_dir = case_dir.join("constant").join("polyMesh");
    let points_path = poly_dir.join("points");
    let faces_path = poly_dir.join("faces");
    if !points_path.exists() || !faces_path.exists() {
        return Err(IoError::MissingFile(points_path));
    }

    let points = parse_foam_points(&fs::read_to_string(&points_path)?)?;
    let faces = parse_foam_faces(&fs::read_to_string(&faces_path)?)?;

    let mut poly = PolyData {
        points,
        polys: faces,
        ..Default::default()
    };
    poly.compute_normals();

    // Optional: attach simple point-count or cell-count scalar field from `0/` if present
    let t0 = case_dir.join("0");
    if t0.is_dir()
        && let Ok(entries) = fs::read_dir(&t0) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file()
                    && let Ok(txt) = fs::read_to_string(&p)
                        && let Some(vals) = parse_foam_nonuniform_scalar_list(&txt) {
                            let fname = p
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("field")
                                .to_string();
                            if vals.len() == poly.polys.len() {
                                poly.cell_data
                                    .add_array(DataArray::scalars_f32(fname, vals));
                            } else if vals.len() == poly.points.len() {
                                poly.point_data
                                    .add_array(DataArray::scalars_f32(fname, vals));
                            }
                        }
            }
        }

    Ok(DataSet::PolyData(poly))
}

fn strip_foam_header_and_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_block_comment = false;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if in_block_comment {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            in_block_comment = true;
            i += 2;
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    // Strip FoamFile { ... } header block if present
    if let Some(pos) = out.find("FoamFile")
        && let Some(end_brace) = out[pos..].find('}') {
            return out[pos + end_brace + 1..].to_string();
        }
    out
}

fn parse_foam_points(text: &str) -> Result<Vec<[f32; 3]>> {
    let body = strip_foam_header_and_comments(text);
    let start = body
        .find('(')
        .ok_or_else(|| IoError::Parse("OpenFOAM points missing '('".into()))?;
    let end = body
        .rfind(')')
        .ok_or_else(|| IoError::Parse("OpenFOAM points missing ')'".into()))?;
    let inner = &body[start + 1..end];
    let mut pts = Vec::new();
    let mut i = 0;
    let bytes = inner.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'('
            && let Some(close) = inner[i + 1..].find(')') {
                let coords: Vec<f32> = inner[i + 1..i + 1 + close]
                    .split_whitespace()
                    .filter_map(|t| t.parse::<f32>().ok())
                    .collect();
                if coords.len() >= 3 {
                    pts.push([coords[0], coords[1], coords[2]]);
                }
                i += close + 2;
                continue;
            }
        i += 1;
    }
    Ok(pts)
}

fn parse_foam_faces(text: &str) -> Result<Vec<Vec<u32>>> {
    let body = strip_foam_header_and_comments(text);
    let start = body
        .find('(')
        .ok_or_else(|| IoError::Parse("OpenFOAM faces missing '('".into()))?;
    let end = body
        .rfind(')')
        .ok_or_else(|| IoError::Parse("OpenFOAM faces missing ')'".into()))?;
    let inner = &body[start + 1..end];
    let mut faces = Vec::new();
    let mut i = 0;
    let bytes = inner.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'('
            && let Some(close) = inner[i + 1..].find(')') {
                let ids: Vec<u32> = inner[i + 1..i + 1 + close]
                    .split_whitespace()
                    .filter_map(|t| t.parse::<u32>().ok())
                    .collect();
                if ids.len() >= 3 {
                    faces.push(ids);
                }
                i += close + 2;
                continue;
            }
        i += 1;
    }
    Ok(faces)
}

fn parse_foam_nonuniform_scalar_list(text: &str) -> Option<Vec<f32>> {
    let body = strip_foam_header_and_comments(text);
    let marker = body.find("List<scalar>")?;
    let rest = &body[marker..];
    let open = rest.find('(')?;
    let close = rest[open + 1..].find(')')?;
    let vals: Vec<f32> = rest[open + 1..open + 1 + close]
        .split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals)
    }
}
