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

// ================================================================================================
// glTF 2.0 / GLB Reader
// ================================================================================================

/// Reads a glTF 2.0 (`.gltf` or `.glb`) file into a [`paraviz_core::DataSet`].
pub fn read_gltf(path: impl AsRef<Path>) -> Result<paraviz_core::DataSet> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    if bytes.len() >= 12 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == GLB_MAGIC
    {
        read_glb_bytes(&bytes)
    } else {
        let json_str = std::str::from_utf8(&bytes)
            .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in glTF JSON: {e}")))?;
        read_gltf_str(json_str, base_dir, None)
    }
}

/// Parses binary glTF 2.0 (`.glb`) bytes into a [`paraviz_core::DataSet`].
pub fn read_glb_bytes(bytes: &[u8]) -> Result<paraviz_core::DataSet> {
    if bytes.len() < 12 {
        return Err(IoError::Parse("GLB file shorter than 12-byte header".into()));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != GLB_MAGIC {
        return Err(IoError::Parse("Invalid GLB magic header".into()));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != GLB_VERSION {
        return Err(IoError::UnsupportedFormat(format!(
            "Unsupported GLB version {version} (expected 2)"
        )));
    }
    let total_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let end = total_len.min(bytes.len());

    let mut offset = 12usize;
    let mut json_slice: Option<&[u8]> = None;
    let mut bin_slice: Option<&[u8]> = None;

    while offset + 8 <= end {
        let chunk_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let chunk_type = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;
        if offset + chunk_len > end {
            return Err(IoError::Parse("Truncated GLB chunk".into()));
        }
        let payload = &bytes[offset..offset + chunk_len];
        if chunk_type == CHUNK_JSON && json_slice.is_none() {
            json_slice = Some(payload);
        } else if chunk_type == CHUNK_BIN && bin_slice.is_none() {
            bin_slice = Some(payload);
        }
        offset += chunk_len;
    }

    let json_bytes = json_slice.ok_or_else(|| IoError::Parse("GLB missing JSON chunk".into()))?;
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in GLB JSON chunk: {e}")))?;
    read_gltf_str(json_str, Path::new("."), bin_slice)
}

/// Parses a glTF 2.0 JSON string (with optional GLB binary chunk) into a [`paraviz_core::DataSet`].
pub fn read_gltf_str(
    json_str: &str,
    base_dir: &Path,
    glb_bin: Option<&[u8]>,
) -> Result<paraviz_core::DataSet> {
    use base64::Engine as _;
    use glam::{Mat4, Quat};
    use paraviz_core::{DataSet, MultiBlockDataSet};

    let root: Value = serde_json::from_str(json_str)
        .map_err(|e| IoError::Parse(format!("Invalid glTF JSON: {e}")))?;

    let buffers_json = root
        .get("buffers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(buffers_json.len());
    for (idx, buf) in buffers_json.iter().enumerate() {
        if let Some(uri) = buf.get("uri").and_then(|v| v.as_str()) {
            if let Some(b64) = uri
                .strip_prefix("data:application/octet-stream;base64,")
                .or_else(|| uri.strip_prefix("data:application/gltf-buffer;base64,"))
            {
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(b64.trim())
                    .map_err(|e| IoError::Parse(format!("Invalid glTF base64 buffer: {e}")))?;
                buffers.push(decoded);
            } else {
                let resolved = crate::limits::resolve_safe_path(base_dir, uri)?;
                buffers.push(fs::read(resolved)?);
            }
        } else if idx == 0 {
            if let Some(bin) = glb_bin {
                buffers.push(bin.to_vec());
            } else {
                return Err(IoError::Parse("glTF buffer 0 has no URI and no GLB BIN chunk".into()));
            }
        } else {
            return Err(IoError::Parse(format!("glTF buffer {idx} missing URI")));
        }
    }

    let buffer_views = root
        .get("bufferViews")
        .and_then(|v| v.as_array())
        .ok_or_else(|| IoError::Parse("glTF missing bufferViews".into()))?;
    let accessors = root
        .get("accessors")
        .and_then(|v| v.as_array())
        .ok_or_else(|| IoError::Parse("glTF missing accessors".into()))?;
    let meshes = root
        .get("meshes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| IoError::Parse("glTF missing meshes".into()))?;

    // Compute world transform for each mesh if nodes/scenes exist
    let mut mesh_transforms: Vec<(usize, Mat4, String)> = Vec::new();
    if let Some(nodes) = root.get("nodes").and_then(|v| v.as_array()) {
        let mut root_nodes: Vec<usize> = Vec::new();
        if let Some(scenes) = root.get("scenes").and_then(|v| v.as_array()) {
            let scene_idx = root.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            if let Some(scene) = scenes.get(scene_idx)
                && let Some(ns) = scene.get("nodes").and_then(|v| v.as_array()) {
                    for n in ns {
                        if let Some(id) = n.as_u64() {
                            root_nodes.push(id as usize);
                        }
                    }
                }
        }
        if root_nodes.is_empty() {
            // Fallback: treat nodes not referenced as children as roots
            let mut is_child = vec![false; nodes.len()];
            for n in nodes {
                if let Some(children) = n.get("children").and_then(|v| v.as_array()) {
                    for c in children {
                        if let Some(cid) = c.as_u64()
                            && (cid as usize) < is_child.len() {
                                is_child[cid as usize] = true;
                            }
                    }
                }
            }
            for (i, &c) in is_child.iter().enumerate() {
                if !c {
                    root_nodes.push(i);
                }
            }
        }

        fn visit_node(
            idx: usize,
            parent_mat: Mat4,
            nodes: &[Value],
            out: &mut Vec<(usize, Mat4, String)>,
            depth: usize,
        ) {
            if depth > 64 || idx >= nodes.len() {
                return;
            }
            let node = &nodes[idx];
            let local = if let Some(m) = node.get("matrix").and_then(|v| v.as_array()) {
                if m.len() == 16 {
                    let mut cols = [0.0f32; 16];
                    for (i, v) in m.iter().enumerate() {
                        cols[i] = v.as_f64().unwrap_or(0.0) as f32;
                    }
                    Mat4::from_cols_array(&cols)
                } else {
                    Mat4::IDENTITY
                }
            } else {
                let t = node
                    .get("translation")
                    .and_then(|v| v.as_array())
                    .filter(|a| a.len() == 3)
                    .map(|a| {
                        Vec3::new(
                            a[0].as_f64().unwrap_or(0.0) as f32,
                            a[1].as_f64().unwrap_or(0.0) as f32,
                            a[2].as_f64().unwrap_or(0.0) as f32,
                        )
                    })
                    .unwrap_or(Vec3::ZERO);
                let r = node
                    .get("rotation")
                    .and_then(|v| v.as_array())
                    .filter(|a| a.len() == 4)
                    .map(|a| {
                        Quat::from_xyzw(
                            a[0].as_f64().unwrap_or(0.0) as f32,
                            a[1].as_f64().unwrap_or(0.0) as f32,
                            a[2].as_f64().unwrap_or(0.0) as f32,
                            a[3].as_f64().unwrap_or(1.0) as f32,
                        )
                        .normalize()
                    })
                    .unwrap_or(Quat::IDENTITY);
                let s = node
                    .get("scale")
                    .and_then(|v| v.as_array())
                    .filter(|a| a.len() == 3)
                    .map(|a| {
                        Vec3::new(
                            a[0].as_f64().unwrap_or(1.0) as f32,
                            a[1].as_f64().unwrap_or(1.0) as f32,
                            a[2].as_f64().unwrap_or(1.0) as f32,
                        )
                    })
                    .unwrap_or(Vec3::ONE);
                Mat4::from_scale_rotation_translation(s, r, t)
            };
            let world = parent_mat * local;
            if let Some(mesh_idx) = node.get("mesh").and_then(|v| v.as_u64()) {
                let name = node
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| format!("Node_{idx}"));
                out.push((mesh_idx as usize, world, name));
            }
            if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
                for c in children {
                    if let Some(cid) = c.as_u64() {
                        visit_node(cid as usize, world, nodes, out, depth + 1);
                    }
                }
            }
        }

        for r in root_nodes {
            visit_node(r, Mat4::IDENTITY, nodes, &mut mesh_transforms, 0);
        }
    }

    if mesh_transforms.is_empty() {
        for (i, m) in meshes.iter().enumerate() {
            let name = m
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("Mesh_{i}"));
            mesh_transforms.push((i, Mat4::IDENTITY, name));
        }
    }

    let mut blocks: Vec<(String, PolyData)> = Vec::new();
    for (mesh_idx, world, base_name) in mesh_transforms {
        let Some(mesh) = meshes.get(mesh_idx) else {
            continue;
        };
        let Some(primitives) = mesh.get("primitives").and_then(|v| v.as_array()) else {
            continue;
        };
        for (pi, prim) in primitives.iter().enumerate() {
            let poly = parse_gltf_primitive(prim, world, accessors, buffer_views, &buffers)?;
            let name = if primitives.len() == 1 {
                base_name.clone()
            } else {
                format!("{base_name}_prim{pi}")
            };
            blocks.push((name, poly));
        }
    }

    if blocks.is_empty() {
        return Err(IoError::Parse("glTF contains no mesh primitives".into()));
    }
    if blocks.len() == 1 {
        Ok(DataSet::PolyData(blocks.into_iter().next().unwrap().1))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, poly) in blocks {
            mb.add_block(name, DataSet::PolyData(poly));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn gltf_read_accessor_f32(
    acc_idx: usize,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<(Vec<f32>, usize, usize)> {
    let acc = accessors
        .get(acc_idx)
        .ok_or_else(|| IoError::Parse(format!("glTF accessor {acc_idx} out of bounds")))?;
    let count = acc
        .get("count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IoError::Parse("glTF accessor missing count".into()))? as usize;
    let comp_type = acc
        .get("componentType")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IoError::Parse("glTF accessor missing componentType".into()))?
        as u32;
    let acc_type = acc
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| IoError::Parse("glTF accessor missing type".into()))?;
    let normalized = acc
        .get("normalized")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let n_comp = match acc_type {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        other => {
            return Err(IoError::Parse(format!(
                "Unsupported glTF accessor type '{other}'"
            )));
        }
    };

    let elem_size = match comp_type {
        5120 | 5121 => 1usize,
        5122 | 5123 => 2usize,
        5125 | 5126 => 4usize,
        other => {
            return Err(IoError::Parse(format!(
                "Unsupported glTF componentType {other}"
            )));
        }
    };

    let bv_idx = acc
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IoError::Parse("Sparse/bufferless glTF accessors unsupported".into()))?
        as usize;
    let bv = buffer_views
        .get(bv_idx)
        .ok_or_else(|| IoError::Parse(format!("glTF bufferView {bv_idx} out of bounds")))?;
    let buf_idx = bv
        .get("buffer")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IoError::Parse("glTF bufferView missing buffer".into()))?
        as usize;
    let buf = buffers
        .get(buf_idx)
        .ok_or_else(|| IoError::Parse(format!("glTF buffer {buf_idx} out of bounds")))?;

    let bv_offset = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let acc_offset = acc.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let base = bv_offset + acc_offset;
    let stride = bv
        .get("byteStride")
        .and_then(|v| v.as_u64())
        .map(|s| s as usize)
        .unwrap_or(n_comp * elem_size);

    let mut out = Vec::with_capacity(
        crate::limits::safe_prealloc_capacity(count * n_comp, 4, crate::limits::MAX_ALLOC_BYTES)?,
    );
    for i in 0..count {
        let row = base + i * stride;
        if row + n_comp * elem_size > buf.len() {
            return Err(IoError::Parse("glTF accessor reads past end of buffer".into()));
        }
        for c in 0..n_comp {
            let p = row + c * elem_size;
            let val = match comp_type {
                5126 => f32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]),
                5125 => u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as f32,
                5123 => {
                    let v = u16::from_le_bytes([buf[p], buf[p + 1]]) as f32;
                    if normalized { v / 65535.0 } else { v }
                }
                5122 => {
                    let v = i16::from_le_bytes([buf[p], buf[p + 1]]) as f32;
                    if normalized { (v / 32767.0).max(-1.0) } else { v }
                }
                5121 => {
                    let v = buf[p] as f32;
                    if normalized { v / 255.0 } else { v }
                }
                5120 => {
                    let v = (buf[p] as i8) as f32;
                    if normalized { (v / 127.0).max(-1.0) } else { v }
                }
                _ => unreachable!(),
            };
            out.push(val);
        }
    }
    Ok((out, count, n_comp))
}

fn gltf_read_indices(
    acc_idx: usize,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<Vec<u32>> {
    let (vals, _, _) = gltf_read_accessor_f32(acc_idx, accessors, buffer_views, buffers)?;
    Ok(vals.into_iter().map(|v| v as u32).collect())
}

fn parse_gltf_primitive(
    prim: &Value,
    world: glam::Mat4,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<PolyData> {
    use paraviz_core::{ArrayData, DataArray};

    let attrs = prim
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| IoError::Parse("glTF primitive missing attributes".into()))?;
    let pos_idx = attrs
        .get("POSITION")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IoError::Parse("glTF primitive missing POSITION".into()))?
        as usize;
    let (pos_raw, n_pts, n_comp) =
        gltf_read_accessor_f32(pos_idx, accessors, buffer_views, buffers)?;
    if n_comp != 3 {
        return Err(IoError::Parse("glTF POSITION must be VEC3".into()));
    }

    let mut poly = PolyData::new();
    poly.points = pos_raw
        .as_chunks::<3>().0.iter()
        .map(|c| world.transform_point3(Vec3::new(c[0], c[1], c[2])).to_array())
        .collect();

    if let Some(norm_idx) = attrs.get("NORMAL").and_then(|v| v.as_u64()) {
        let (norm_raw, count, nc) =
            gltf_read_accessor_f32(norm_idx as usize, accessors, buffer_views, buffers)?;
        if count == n_pts && nc == 3 {
            let flat: Vec<f32> = norm_raw
                .as_chunks::<3>().0.iter()
                .flat_map(|c| {
                    let n = world
                        .transform_vector3(Vec3::new(c[0], c[1], c[2]))
                        .normalize_or_zero();
                    [n.x, n.y, n.z]
                })
                .collect();
            poly.point_data
                .add_array(DataArray::new("Normals", 3, ArrayData::F32(flat)));
        }
    }

    if let Some(uv_idx) = attrs.get("TEXCOORD_0").and_then(|v| v.as_u64()) {
        let (uv_raw, count, nc) =
            gltf_read_accessor_f32(uv_idx as usize, accessors, buffer_views, buffers)?;
        if count == n_pts && nc == 2 {
            poly.point_data
                .add_array(DataArray::new("TCoords", 2, ArrayData::F32(uv_raw)));
        }
    }

    if let Some(col_idx) = attrs.get("COLOR_0").and_then(|v| v.as_u64()) {
        let (col_raw, count, nc) =
            gltf_read_accessor_f32(col_idx as usize, accessors, buffer_views, buffers)?;
        if count == n_pts && (nc == 3 || nc == 4) {
            let mut rgba = Vec::with_capacity(n_pts * 4);
            for chunk in col_raw.chunks_exact(nc) {
                rgba.push((chunk[0].clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba.push((chunk[1].clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba.push((chunk[2].clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba.push(if nc == 4 {
                    (chunk[3].clamp(0.0, 1.0) * 255.0).round() as u8
                } else {
                    255
                });
            }
            poly.point_data
                .add_array(DataArray::new("Colors", 4, ArrayData::U8(rgba)));
        }
    }

    let mode = prim.get("mode").and_then(|v| v.as_u64()).unwrap_or(4) as u32;
    let indices: Vec<u32> = if let Some(idx_acc) = prim.get("indices").and_then(|v| v.as_u64()) {
        gltf_read_indices(idx_acc as usize, accessors, buffer_views, buffers)?
    } else {
        (0..n_pts as u32).collect()
    };

    for &i in &indices {
        if (i as usize) >= n_pts {
            return Err(IoError::InvalidData(format!(
                "glTF index {i} out of bounds ({n_pts})"
            )));
        }
    }

    match mode {
        0 => poly.verts = indices,
        1 => {
            for pair in indices.as_chunks::<2>().0 {
                poly.lines.push(vec![pair[0], pair[1]]);
            }
        }
        2 => {
            if indices.len() >= 2 {
                let mut line = indices.clone();
                line.push(indices[0]);
                poly.lines.push(line);
            }
        }
        3 => {
            if indices.len() >= 2 {
                poly.lines.push(indices);
            }
        }
        4 => {
            for tri in indices.as_chunks::<3>().0 {
                poly.polys.push(vec![tri[0], tri[1], tri[2]]);
            }
        }
        5 => {
            for i in 0..indices.len().saturating_sub(2) {
                if i % 2 == 0 {
                    poly.polys.push(vec![indices[i], indices[i + 1], indices[i + 2]]);
                } else {
                    poly.polys.push(vec![indices[i + 1], indices[i], indices[i + 2]]);
                }
            }
        }
        6 => {
            for i in 1..indices.len().saturating_sub(1) {
                poly.polys.push(vec![indices[0], indices[i], indices[i + 1]]);
            }
        }
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported glTF primitive mode {other}"
            )));
        }
    }

    Ok(poly)
}
