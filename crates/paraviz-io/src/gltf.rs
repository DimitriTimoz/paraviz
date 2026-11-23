//! Binary glTF 2.0 (`.glb`) exporter for [`PolyData`] surface meshes.

use crate::error::{IoError, Result};
use glam::Vec3;
use paraviz_core::{ColorMap, PolyData};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

const COMPONENT_UNSIGNED_INT: u32 = 5125;
const COMPONENT_FLOAT: u32 = 5126;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

const MODE_POINTS: u32 = 0;
const MODE_LINES: u32 = 1;
const MODE_TRIANGLES: u32 = 4;

/// Writes a [`PolyData`] mesh as a self-contained binary glTF 2.0 (`.glb`) file.
///
/// Includes `POSITION`, `NORMAL`, optional `COLOR_0` vertex colors mapped via `color_map`,
/// and `INDICES`.
pub fn write_glb(
    path: impl AsRef<Path>,
    poly: &PolyData,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Result<()> {
    let bytes = encode_glb_bytes(poly, color_map, scalar_array)?;
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Encodes a [`PolyData`] mesh into binary glTF 2.0 (`.glb`) bytes.
pub fn encode_glb_bytes(
    poly: &PolyData,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Result<Vec<u8>> {
    if poly.points.is_empty() {
        return Err(IoError::InvalidData(
            "Cannot export empty PolyData (0 points) to GLB".into(),
        ));
    }

    let normals = resolve_normals(poly);
    let colors = resolve_vertex_colors(poly, color_map, scalar_array)?;
    let (primitive_mode, indices) = collect_indices(poly);

    let mut bin = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut attributes = Map::new();

    // 1. POSITION
    let (pos_min, pos_max) = position_bounds(&poly.points);
    let pos_offset = bin.len();
    for p in &poly.points {
        for &c in p {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let pos_len = bin.len() - pos_offset;
    pad_to_4(&mut bin, 0x00);

    let pos_bv = push_buffer_view(&mut buffer_views, pos_offset, pos_len, TARGET_ARRAY_BUFFER);
    let pos_acc = accessors.len();
    accessors.push(json!({
        "bufferView": pos_bv,
        "componentType": COMPONENT_FLOAT,
        "count": poly.points.len(),
        "type": "VEC3",
        "min": pos_min,
        "max": pos_max,
    }));
    attributes.insert("POSITION".to_string(), json!(pos_acc));

    // 2. NORMAL
    let norm_offset = bin.len();
    for n in &normals {
        for &c in n {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let norm_len = bin.len() - norm_offset;
    pad_to_4(&mut bin, 0x00);

    let norm_bv = push_buffer_view(&mut buffer_views, norm_offset, norm_len, TARGET_ARRAY_BUFFER);
    let norm_acc = accessors.len();
    accessors.push(json!({
        "bufferView": norm_bv,
        "componentType": COMPONENT_FLOAT,
        "count": normals.len(),
        "type": "VEC3",
    }));
    attributes.insert("NORMAL".to_string(), json!(norm_acc));

    // 3. Optional COLOR_0
    if let Some(rgba_list) = colors {
        let col_offset = bin.len();
        for rgba in &rgba_list {
            for &c in rgba {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let col_len = bin.len() - col_offset;
        pad_to_4(&mut bin, 0x00);

        let col_bv = push_buffer_view(&mut buffer_views, col_offset, col_len, TARGET_ARRAY_BUFFER);
        let col_acc = accessors.len();
        accessors.push(json!({
            "bufferView": col_bv,
            "componentType": COMPONENT_FLOAT,
            "count": rgba_list.len(),
            "type": "VEC4",
        }));
        attributes.insert("COLOR_0".to_string(), json!(col_acc));
    }

    // 4. INDICES
    let idx_offset = bin.len();
    for &idx in &indices {
        bin.extend_from_slice(&idx.to_le_bytes());
    }
    let idx_len = bin.len() - idx_offset;
    pad_to_4(&mut bin, 0x00);

    let idx_bv = push_buffer_view(
        &mut buffer_views,
        idx_offset,
        idx_len,
        TARGET_ELEMENT_ARRAY_BUFFER,
    );
    let idx_acc = accessors.len();
    let idx_min = indices.iter().copied().min().unwrap_or(0);
    let idx_max = indices.iter().copied().max().unwrap_or(0);
    accessors.push(json!({
        "bufferView": idx_bv,
        "componentType": COMPONENT_UNSIGNED_INT,
        "count": indices.len(),
        "type": "SCALAR",
        "min": [idx_min],
        "max": [idx_max],
    }));

    let gltf_json = json!({
        "asset": {
            "version": "2.0",
            "generator": "paraviz-io"
        },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "primitives": [{
                "attributes": Value::Object(attributes),
                "indices": idx_acc,
                "mode": primitive_mode,
            }]
        }],
        "buffers": [{
            "byteLength": bin.len(),
        }],
        "bufferViews": buffer_views,
        "accessors": accessors,
    });

    let mut json_bytes = serde_json::to_vec(&gltf_json)
        .map_err(|e| IoError::InvalidData(format!("Failed to serialize glTF JSON: {e}")))?;
    pad_to_4(&mut json_bytes, b' ');

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let total_u32 = u32::try_from(total_len)
        .map_err(|_| IoError::InvalidData("GLB output exceeds 4 GiB limit".into()))?;

    let mut out = Vec::with_capacity(total_len);
    // 12-byte GLB header
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&total_u32.to_le_bytes());
    // Chunk 0: JSON
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    // Chunk 1: BIN
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin);

    Ok(out)
}

fn push_buffer_view(
    buffer_views: &mut Vec<Value>,
    byte_offset: usize,
    byte_length: usize,
    target: u32,
) -> usize {
    let idx = buffer_views.len();
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": byte_length,
        "target": target,
    }));
    idx
}

fn pad_to_4(buf: &mut Vec<u8>, pad_byte: u8) {
    while !buf.len().is_multiple_of(4) {
        buf.push(pad_byte);
    }
}

fn position_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    (min, max)
}

fn resolve_normals(poly: &PolyData) -> Vec<[f32; 3]> {
    if let Some(arr) = poly
        .point_data
        .active_normals_array()
        .filter(|a| a.num_components == 3 && a.num_tuples() == poly.points.len())
    {
        return (0..poly.points.len())
            .map(|i| {
                let n = Vec3::from_array(arr.get_vec3(i));
                if n.length_squared() > 1e-16 && n.is_finite() {
                    n.normalize().to_array()
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect();
    }

    let mut acc = vec![Vec3::ZERO; poly.points.len()];
    for tri in poly.triangulate() {
        let p0 = Vec3::from_array(poly.points[tri[0] as usize]);
        let p1 = Vec3::from_array(poly.points[tri[1] as usize]);
        let p2 = Vec3::from_array(poly.points[tri[2] as usize]);
        let n = (p1 - p0).cross(p2 - p0);
        for &idx in &tri {
            if let Some(slot) = acc.get_mut(idx as usize) {
                *slot += n;
            }
        }
    }

    acc.into_iter()
        .map(|n| {
            let len = n.length();
            if len > 1e-12 && n.is_finite() {
                (n / len).to_array()
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

fn resolve_vertex_colors(
    poly: &PolyData,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Result<Option<Vec<[f32; 4]>>> {
    let Some(cmap) = color_map else {
        return Ok(None);
    };

    let arr = match scalar_array {
        Some(name) => Some(poly.point_data.get(name).ok_or_else(|| {
            IoError::InvalidData(format!(
                "Scalar array '{name}' not found in PolyData point_data"
            ))
        })?),
        None => poly.point_data.active_scalars_array(),
    };

    let Some(arr) = arr else {
        return Ok(None);
    };

    if arr.num_tuples() != poly.points.len() {
        return Err(IoError::InvalidData(format!(
            "Scalar array '{}' tuple count ({}) does not match point count ({})",
            arr.name,
            arr.num_tuples(),
            poly.points.len()
        )));
    }

    let comp = if arr.num_components == 1 {
        Some(0)
    } else {
        None
    };
    let range = arr.range(comp);
    let colors = (0..poly.points.len())
        .map(|i| {
            let val = if arr.num_components == 1 {
                arr.get_scalar(i)
            } else {
                arr.get_magnitude(i)
            };
            cmap.map_value(val, range)
        })
        .collect();
    Ok(Some(colors))
}

fn collect_indices(poly: &PolyData) -> (u32, Vec<u32>) {
    let tris = poly.triangulate();
    if !tris.is_empty() {
        let mut flat = Vec::with_capacity(tris.len() * 3);
        for [a, b, c] in tris {
            flat.push(a);
            flat.push(b);
            flat.push(c);
        }
        return (MODE_TRIANGLES, flat);
    }

    if !poly.lines.is_empty() {
        let mut segs = Vec::new();
        for line in &poly.lines {
            for win in line.windows(2) {
                segs.push(win[0]);
                segs.push(win[1]);
            }
        }
        if !segs.is_empty() {
            return (MODE_LINES, segs);
        }
    }

    if !poly.verts.is_empty() {
        return (MODE_POINTS, poly.verts.clone());
    }

    (MODE_POINTS, (0..poly.points.len() as u32).collect())
}
