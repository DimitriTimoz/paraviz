//! Mesh file format readers and writers: STL, Wavefront OBJ, and PLY.

use crate::error::{IoError, Result};
use glam::Vec3;
use paraviz_core::{ArrayData, DataArray, PolyData, ScalarType};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

// ================================================================================================
// STL (ASCII & Binary)
// ================================================================================================

/// Reads an ASCII or Binary STL file into a [`PolyData`].
///
/// Identical vertex positions are welded by exact bit pattern, facet normals are stored in
/// `cell_data` (computed from triangle geometry when absent/zero), and smooth vertex normals are
/// computed in `point_data`.
pub fn read_stl(path: impl AsRef<Path>) -> Result<PolyData> {
    let bytes = fs::read(path)?;
    read_stl_bytes(&bytes)
}

/// Parses ASCII or Binary STL bytes into a [`PolyData`].
pub fn read_stl_bytes(bytes: &[u8]) -> Result<PolyData> {
    if is_binary_stl(bytes) {
        read_stl_binary_bytes(bytes)
    } else {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in ASCII STL: {e}")))?;
        read_stl_ascii_str(text)
    }
}

fn is_binary_stl(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected_len = 84usize.saturating_add(tri_count.saturating_mul(50));
    let head = &bytes[..bytes.len().min(512)];
    let trimmed = std::str::from_utf8(head)
        .ok()
        .map(|s| s.trim_start())
        .unwrap_or("");
    let looks_ascii = trimmed.starts_with("solid") && trimmed.contains("facet");
    if looks_ascii && bytes.len() != expected_len {
        return false;
    }
    if bytes.len() == expected_len && !looks_ascii {
        return true;
    }
    !looks_ascii
}

fn read_stl_binary_bytes(bytes: &[u8]) -> Result<PolyData> {
    if bytes.len() < 84 {
        return Err(IoError::InvalidData("Binary STL shorter than 84 bytes".into()));
    }
    let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let needed = 84usize
        .checked_add(tri_count.saturating_mul(50))
        .ok_or_else(|| IoError::InvalidData("STL triangle count overflow".into()))?;
    if bytes.len() < needed {
        return Err(IoError::InvalidData(format!(
            "Binary STL truncated: expected {needed} bytes, got {}",
            bytes.len()
        )));
    }

    let mut builder = StlMeshBuilder::with_capacity(tri_count);
    for t in 0..tri_count {
        let base = 84 + t * 50;
        let nx = read_f32_le(&bytes[base..base + 4]);
        let ny = read_f32_le(&bytes[base + 4..base + 8]);
        let nz = read_f32_le(&bytes[base + 8..base + 12]);
        let v0 = [
            read_f32_le(&bytes[base + 12..base + 16]),
            read_f32_le(&bytes[base + 16..base + 20]),
            read_f32_le(&bytes[base + 20..base + 24]),
        ];
        let v1 = [
            read_f32_le(&bytes[base + 24..base + 28]),
            read_f32_le(&bytes[base + 28..base + 32]),
            read_f32_le(&bytes[base + 32..base + 36]),
        ];
        let v2 = [
            read_f32_le(&bytes[base + 36..base + 40]),
            read_f32_le(&bytes[base + 40..base + 44]),
            read_f32_le(&bytes[base + 44..base + 48]),
        ];
        builder.push_triangle([nx, ny, nz], v0, v1, v2);
    }

    Ok(builder.finish())
}

fn read_stl_ascii_str(text: &str) -> Result<PolyData> {
    let mut builder = StlMeshBuilder::with_capacity(text.len() / 128);
    let mut cur_normal = [0.0f32; 3];
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(3);
    let mut in_facet = false;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("facet") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 && parts[1].eq_ignore_ascii_case("normal") {
                cur_normal = [
                    parse_f32(parts[2])?,
                    parse_f32(parts[3])?,
                    parse_f32(parts[4])?,
                ];
            } else {
                cur_normal = [0.0; 3];
            }
            verts.clear();
            in_facet = true;
        } else if lower.starts_with("vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(IoError::Parse(format!("Invalid STL vertex line: '{line}'")));
            }
            verts.push([
                parse_f32(parts[1])?,
                parse_f32(parts[2])?,
                parse_f32(parts[3])?,
            ]);
        } else if lower.starts_with("endfacet") {
            if in_facet && verts.len() >= 3 {
                builder.push_triangle(cur_normal, verts[0], verts[1], verts[2]);
            }
            in_facet = false;
            verts.clear();
        }
    }

    Ok(builder.finish())
}

struct StlMeshBuilder {
    points: Vec<[f32; 3]>,
    point_map: HashMap<[u32; 3], u32>,
    polys: Vec<Vec<u32>>,
}

impl StlMeshBuilder {
    fn with_capacity(tris: usize) -> Self {
        Self {
            points: Vec::with_capacity(tris / 2 + 4),
            point_map: HashMap::with_capacity(tris / 2 + 4),
            polys: Vec::with_capacity(tris),
        }
    }

    fn intern(&mut self, p: [f32; 3]) -> u32 {
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        if let Some(&idx) = self.point_map.get(&key) {
            idx
        } else {
            let idx = self.points.len() as u32;
            self.points.push(p);
            self.point_map.insert(key, idx);
            idx
        }
    }

    fn push_triangle(&mut self, _normal: [f32; 3], v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) {
        let i0 = self.intern(v0);
        let i1 = self.intern(v1);
        let i2 = self.intern(v2);
        self.polys.push(vec![i0, i1, i2]);
    }

    fn finish(self) -> PolyData {
        let mut poly = PolyData {
            points: self.points,
            polys: self.polys,
            ..Default::default()
        };
        // vtkSTLReader exposes a per-cell solid id, not normals arrays.
        let ncells = poly.polys.len();
        poly.cell_data.add_array(DataArray::new(
            "STLSolidLabeling",
            1,
            ArrayData::I32(vec![0; ncells]),
        ));
        poly
    }
}

/// Writes a [`PolyData`] surface mesh as a Binary STL file (triangulating polygons/strips).
pub fn write_stl(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    write_stl_binary(path, poly)
}

/// Writes a [`PolyData`] surface mesh as a Binary STL file.
pub fn write_stl_binary(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    let tris = collect_stl_triangles(poly);
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let mut header = [0u8; 80];
    let banner = b"ParaViz Binary STL";
    header[..banner.len()].copy_from_slice(banner);
    w.write_all(&header)?;
    w.write_all(&(tris.len() as u32).to_le_bytes())?;

    for t in tris {
        for val in t.normal.into_iter().chain(t.v0).chain(t.v1).chain(t.v2) {
            w.write_all(&val.to_le_bytes())?;
        }
        w.write_all(&0u16.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

/// Writes a [`PolyData`] surface mesh as an ASCII STL file.
pub fn write_stl_ascii(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    let tris = collect_stl_triangles(poly);
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "solid paraviz")?;
    for t in tris {
        writeln!(
            w,
            "  facet normal {} {} {}",
            t.normal[0], t.normal[1], t.normal[2]
        )?;
        writeln!(w, "    outer loop")?;
        writeln!(w, "      vertex {} {} {}", t.v0[0], t.v0[1], t.v0[2])?;
        writeln!(w, "      vertex {} {} {}", t.v1[0], t.v1[1], t.v1[2])?;
        writeln!(w, "      vertex {} {} {}", t.v2[0], t.v2[1], t.v2[2])?;
        writeln!(w, "    endloop")?;
        writeln!(w, "  endfacet")?;
    }
    writeln!(w, "endsolid paraviz")?;
    w.flush()?;
    Ok(())
}

struct StlTri {
    normal: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
}

fn collect_stl_triangles(poly: &PolyData) -> Vec<StlTri> {
    let cell_normals = poly.cell_data.active_normals_array();
    let mut out = Vec::new();

    for (cell_idx, polygon) in poly.polys.iter().enumerate() {
        if polygon.len() < 3 {
            continue;
        }
        let cell_n = cell_normals
            .filter(|a| cell_idx < a.num_tuples())
            .map(|a| a.get_vec3(cell_idx));

        let i0 = polygon[0] as usize;
        for k in 1..(polygon.len() - 1) {
            let i1 = polygon[k] as usize;
            let i2 = polygon[k + 1] as usize;
            let v0 = poly.points[i0];
            let v1 = poly.points[i1];
            let v2 = poly.points[i2];
            let normal = cell_n.unwrap_or_else(|| triangle_normal(v0, v1, v2));
            out.push(StlTri { normal, v0, v1, v2 });
        }
    }

    for strip in &poly.strips {
        if strip.len() < 3 {
            continue;
        }
        for k in 0..(strip.len() - 2) {
            let (a, b, c) = if k % 2 == 0 {
                (strip[k], strip[k + 1], strip[k + 2])
            } else {
                (strip[k + 1], strip[k], strip[k + 2])
            };
            let v0 = poly.points[a as usize];
            let v1 = poly.points[b as usize];
            let v2 = poly.points[c as usize];
            out.push(StlTri {
                normal: triangle_normal(v0, v1, v2),
                v0,
                v1,
                v2,
            });
        }
    }

    out
}

fn triangle_normal(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [f32; 3] {
    let n = (Vec3::from_array(v1) - Vec3::from_array(v0))
        .cross(Vec3::from_array(v2) - Vec3::from_array(v0));
    if n.length_squared() > 1e-20 {
        n.normalize().to_array()
    } else {
        [0.0, 0.0, 1.0]
    }
}

// ================================================================================================
// Wavefront OBJ
// ================================================================================================

/// Reads a Wavefront `.obj` file into a [`PolyData`].
///
/// Supports `v`, `vn`, `vt`, `f` (triangles, quads, n-gons, `v`, `v/vt`, `v//vn`, `v/vt/vn`,
/// and negative relative indices), `l` (polylines), and `p` (vertices).
pub fn read_obj(path: impl AsRef<Path>) -> Result<PolyData> {
    let content = fs::read_to_string(path)?;
    read_obj_str(&content)
}

/// Parses Wavefront `.obj` text into a [`PolyData`].
pub fn read_obj_str(content: &str) -> Result<PolyData> {
    let mut raw_positions: Vec<[f32; 3]> = Vec::new();
    let mut raw_texcoords: Vec<[f32; 2]> = Vec::new();
    let mut raw_normals: Vec<[f32; 3]> = Vec::new();

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct ObjCorner {
        v: usize,
        vt: Option<usize>,
        vn: Option<usize>,
    }

    enum FaceOrLine {
        Face(Vec<ObjCorner>),
        Line(Vec<usize>),
        Point(usize),
    }

    let mut primitives = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else { continue };
        match tag {
            "v" => {
                let x = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ v missing x".into()))?)?;
                let y = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ v missing y".into()))?)?;
                let z = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ v missing z".into()))?)?;
                raw_positions.push([x, y, z]);
            }
            "vt" => {
                let u = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ vt missing u".into()))?)?;
                let v = parts.next().map(parse_f32).transpose()?.unwrap_or(0.0);
                raw_texcoords.push([u, v]);
            }
            "vn" => {
                let nx = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ vn missing x".into()))?)?;
                let ny = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ vn missing y".into()))?)?;
                let nz = parse_f32(parts.next().ok_or_else(|| IoError::Parse("OBJ vn missing z".into()))?)?;
                raw_normals.push([nx, ny, nz]);
            }
            "f" => {
                let mut corners = Vec::new();
                for tok in parts {
                    let mut segs = tok.split('/');
                    let v_tok = segs.next().unwrap_or("");
                    let v = resolve_obj_index(v_tok, raw_positions.len())?;
                    let vt = match segs.next() {
                        Some(s) if !s.is_empty() => Some(resolve_obj_index(s, raw_texcoords.len())?),
                        _ => None,
                    };
                    let vn = match segs.next() {
                        Some(s) if !s.is_empty() => Some(resolve_obj_index(s, raw_normals.len())?),
                        _ => None,
                    };
                    corners.push(ObjCorner { v, vt, vn });
                }
                if corners.len() >= 3 {
                    primitives.push(FaceOrLine::Face(corners));
                }
            }
            "l" => {
                let mut line_pts = Vec::new();
                for tok in parts {
                    let v_tok = tok.split('/').next().unwrap_or("");
                    line_pts.push(resolve_obj_index(v_tok, raw_positions.len())?);
                }
                if line_pts.len() >= 2 {
                    primitives.push(FaceOrLine::Line(line_pts));
                }
            }
            "p" => {
                for tok in parts {
                    let v_tok = tok.split('/').next().unwrap_or("");
                    primitives.push(FaceOrLine::Point(resolve_obj_index(v_tok, raw_positions.len())?));
                }
            }
            _ => {}
        }
    }

    let has_faces = primitives
        .iter()
        .any(|p| matches!(p, FaceOrLine::Face(_)));

    let mut poly = PolyData::new();

    if !has_faces {
        poly.points = raw_positions;
        for prim in primitives {
            match prim {
                FaceOrLine::Line(l) => poly.lines.push(l.into_iter().map(|i| i as u32).collect()),
                FaceOrLine::Point(p) => poly.verts.push(p as u32),
                FaceOrLine::Face(_) => {}
            }
        }
        return Ok(poly);
    }

    // Track emitted vertex attributes per raw position so unsplit meshes preserve 0..N indexing
    // while corners with differing (vt, vn) cleanly split into distinct vertices.
    let mut corner_map: HashMap<(usize, Option<usize>, Option<usize>), u32> = HashMap::new();
    let mut first_slot: Vec<Option<(Option<usize>, Option<usize>)>> = vec![None; raw_positions.len()];
    let mut out_points = raw_positions.clone();
    let any_vt = primitives.iter().any(|p| {
        matches!(p, FaceOrLine::Face(cs) if cs.iter().any(|c| c.vt.is_some()))
    });
    let any_vn = primitives.iter().any(|p| {
        matches!(p, FaceOrLine::Face(cs) if cs.iter().any(|c| c.vn.is_some()))
    });

    let mut out_tcoords = if any_vt {
        vec![[0.0f32; 2]; out_points.len()]
    } else {
        Vec::new()
    };
    let mut out_normals = if any_vn {
        vec![[0.0f32, 0.0, 1.0]; out_points.len()]
    } else {
        Vec::new()
    };

    for prim in primitives {
        match prim {
            FaceOrLine::Face(corners) => {
                let mut face_indices = Vec::with_capacity(corners.len());
                for c in corners {
                    let key = (c.v, c.vt, c.vn);
                    let idx = if let Some(&existing) = corner_map.get(&key) {
                        existing
                    } else if first_slot[c.v].is_none() {
                        first_slot[c.v] = Some((c.vt, c.vn));
                        if any_vt && let Some(vt_i) = c.vt {
                            out_tcoords[c.v] = raw_texcoords[vt_i];
                        }
                        if any_vn && let Some(vn_i) = c.vn {
                            out_normals[c.v] = raw_normals[vn_i];
                        }
                        let idx = c.v as u32;
                        corner_map.insert(key, idx);
                        idx
                    } else {
                        let idx = out_points.len() as u32;
                        out_points.push(raw_positions[c.v]);
                        if any_vt {
                            out_tcoords.push(c.vt.map(|i| raw_texcoords[i]).unwrap_or([0.0, 0.0]));
                        }
                        if any_vn {
                            out_normals
                                .push(c.vn.map(|i| raw_normals[i]).unwrap_or([0.0, 0.0, 1.0]));
                        }
                        corner_map.insert(key, idx);
                        idx
                    };
                    face_indices.push(idx);
                }
                poly.polys.push(face_indices);
            }
            FaceOrLine::Line(l) => {
                poly.lines.push(l.into_iter().map(|i| i as u32).collect());
            }
            FaceOrLine::Point(p) => {
                poly.verts.push(p as u32);
            }
        }
    }

    poly.points = out_points;
    if any_vt {
        let mut flat = Vec::with_capacity(out_tcoords.len() * 2);
        for uv in out_tcoords {
            flat.extend_from_slice(&uv);
        }
        poly.point_data
            .add_array(DataArray::new("TCoords", 2, ArrayData::F32(flat)));
    }
    if any_vn {
        poly.point_data
            .set_normals(DataArray::vectors_f32("Normals", out_normals));
    }

    Ok(poly)
}

fn resolve_obj_index(tok: &str, len: usize) -> Result<usize> {
    let raw: isize = tok
        .parse()
        .map_err(|_| IoError::Parse(format!("Invalid OBJ index '{tok}'")))?;
    let idx = if raw > 0 {
        raw - 1
    } else if raw < 0 {
        len as isize + raw
    } else {
        return Err(IoError::Parse("OBJ indices are 1-based, got 0".into()));
    };
    if idx < 0 || (idx as usize) >= len {
        return Err(IoError::InvalidData(format!(
            "OBJ index {raw} out of bounds for length {len}"
        )));
    }
    Ok(idx as usize)
}

/// Writes a [`PolyData`] to a Wavefront `.obj` file.
pub fn write_obj(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# ParaViz Wavefront OBJ")?;
    for p in &poly.points {
        writeln!(w, "v {} {} {}", p[0], p[1], p[2])?;
    }

    let tcoords = poly
        .point_data
        .get("TCoords")
        .filter(|a| a.num_components >= 2 && a.num_tuples() == poly.points.len());
    if let Some(tc) = tcoords {
        for i in 0..tc.num_tuples() {
            let u = tc.data.get_f64(i * tc.num_components);
            let v = tc.data.get_f64(i * tc.num_components + 1);
            writeln!(w, "vt {u} {v}")?;
        }
    }

    let normals = poly
        .point_data
        .active_normals_array()
        .filter(|a| a.num_components == 3 && a.num_tuples() == poly.points.len());
    if let Some(ns) = normals {
        for i in 0..ns.num_tuples() {
            let n = ns.get_vec3(i);
            writeln!(w, "vn {} {} {}", n[0], n[1], n[2])?;
        }
    }

    if !poly.verts.is_empty() {
        write!(w, "p")?;
        for &v in &poly.verts {
            write!(w, " {}", v + 1)?;
        }
        writeln!(w)?;
    }

    for line in &poly.lines {
        write!(w, "l")?;
        for &v in line {
            write!(w, " {}", v + 1)?;
        }
        writeln!(w)?;
    }

    let has_vt = tcoords.is_some();
    let has_vn = normals.is_some();
    for face in &poly.polys {
        write!(w, "f")?;
        for &v in face {
            let idx = v + 1;
            match (has_vt, has_vn) {
                (false, false) => write!(w, " {idx}")?,
                (true, false) => write!(w, " {idx}/{idx}")?,
                (false, true) => write!(w, " {idx}//{idx}")?,
                (true, true) => write!(w, " {idx}/{idx}/{idx}")?,
            }
        }
        writeln!(w)?;
    }

    w.flush()?;
    Ok(())
}

// ================================================================================================
// PLY (ASCII, binary_little_endian, binary_big_endian)
// ================================================================================================

/// PLY encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyType {
    fn from_name(s: &str) -> Result<Self> {
        match s {
            "char" | "int8" => Ok(Self::I8),
            "uchar" | "uint8" => Ok(Self::U8),
            "short" | "int16" => Ok(Self::I16),
            "ushort" | "uint16" => Ok(Self::U16),
            "int" | "int32" => Ok(Self::I32),
            "uint" | "uint32" => Ok(Self::U32),
            "float" | "float32" => Ok(Self::F32),
            "double" | "float64" => Ok(Self::F64),
            _ => Err(IoError::Parse(format!("Unsupported PLY property type '{s}'"))),
        }
    }

    fn ply_name(self) -> &'static str {
        match self {
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
enum PlyPropertyKind {
    Scalar(PlyType),
    List { count: PlyType, elem: PlyType },
}

#[derive(Debug, Clone)]
struct PlyProperty {
    name: String,
    kind: PlyPropertyKind,
}

#[derive(Debug, Clone)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

#[derive(Debug, Clone, Copy)]
enum PlyValue {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
}

impl PlyValue {
    fn as_f32(self) -> f32 {
        match self {
            Self::I8(v) => v as f32,
            Self::U8(v) => v as f32,
            Self::I16(v) => v as f32,
            Self::U16(v) => v as f32,
            Self::I32(v) => v as f32,
            Self::U32(v) => v as f32,
            Self::F32(v) => v,
            Self::F64(v) => v as f32,
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            Self::I8(v) => v as f64,
            Self::U8(v) => v as f64,
            Self::I16(v) => v as f64,
            Self::U16(v) => v as f64,
            Self::I32(v) => v as f64,
            Self::U32(v) => v as f64,
            Self::F32(v) => v as f64,
            Self::F64(v) => v,
        }
    }

    fn as_usize(self) -> usize {
        match self {
            Self::I8(v) => v.max(0) as usize,
            Self::U8(v) => v as usize,
            Self::I16(v) => v.max(0) as usize,
            Self::U16(v) => v as usize,
            Self::I32(v) => v.max(0) as usize,
            Self::U32(v) => v as usize,
            Self::F32(v) => v.max(0.0) as usize,
            Self::F64(v) => v.max(0.0) as usize,
        }
    }

    fn as_u8_color(self) -> u8 {
        match self {
            Self::U8(v) => v,
            Self::I8(v) => v.max(0) as u8,
            Self::U16(v) => (v.min(255)) as u8,
            Self::I16(v) => v.clamp(0, 255) as u8,
            Self::U32(v) => (v.min(255)) as u8,
            Self::I32(v) => v.clamp(0, 255) as u8,
            Self::F32(v) => (v.clamp(0.0, 1.0) * 255.0).round() as u8,
            Self::F64(v) => (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

/// Reads a PLY file (ASCII, `binary_little_endian`, or `binary_big_endian`) into a [`PolyData`].
pub fn read_ply(path: impl AsRef<Path>) -> Result<PolyData> {
    let bytes = fs::read(path)?;
    read_ply_bytes(&bytes)
}

/// Parses PLY bytes into a [`PolyData`].
pub fn read_ply_bytes(bytes: &[u8]) -> Result<PolyData> {
    let (format, elements, body_offset) = parse_ply_header(bytes)?;

    let mut poly = PolyData::new();
    let body = &bytes[body_offset..];

    match format {
        PlyFormat::Ascii => {
            let text = std::str::from_utf8(body)
                .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in ASCII PLY body: {e}")))?;
            let mut tokens = text.split_whitespace();
            for elem in &elements {
                read_ply_element_ascii(elem, &mut tokens, &mut poly)?;
            }
        }
        PlyFormat::BinaryLittleEndian => {
            let mut pos = 0usize;
            for elem in &elements {
                read_ply_element_binary(elem, body, &mut pos, true, &mut poly)?;
            }
        }
        PlyFormat::BinaryBigEndian => {
            let mut pos = 0usize;
            for elem in &elements {
                read_ply_element_binary(elem, body, &mut pos, false, &mut poly)?;
            }
        }
    }

    Ok(poly)
}

fn parse_ply_header(bytes: &[u8]) -> Result<(PlyFormat, Vec<PlyElement>, usize)> {
    let mut pos = 0usize;
    let mut format = None;
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut saw_ply = false;

    loop {
        let line_start = pos;
        let line_end = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|off| pos + off)
            .ok_or_else(|| IoError::Parse("PLY header missing 'end_header'".into()))?;
        pos = line_end + 1;

        let line = std::str::from_utf8(&bytes[line_start..line_end])
            .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in PLY header: {e}")))?
            .trim();
        if line.is_empty() {
            continue;
        }
        if line == "ply" {
            saw_ply = true;
            continue;
        }
        if line == "end_header" {
            if !saw_ply {
                return Err(IoError::Parse("Missing 'ply' magic header".into()));
            }
            let fmt = format.ok_or_else(|| IoError::Parse("PLY header missing 'format'".into()))?;
            return Ok((fmt, elements, pos));
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "comment" | "obj_info" => {}
            "format" => {
                if parts.len() < 2 {
                    return Err(IoError::Parse(format!("Invalid PLY format line: '{line}'")));
                }
                format = Some(match parts[1] {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    other => {
                        return Err(IoError::UnsupportedFormat(format!(
                            "Unsupported PLY format '{other}'"
                        )))
                    }
                });
            }
            "element" => {
                if parts.len() < 3 {
                    return Err(IoError::Parse(format!("Invalid PLY element line: '{line}'")));
                }
                let count: usize = parts[2]
                    .parse()
                    .map_err(|_| IoError::Parse(format!("Invalid PLY element count '{}'", parts[2])))?;
                elements.push(PlyElement {
                    name: parts[1].to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            "property" => {
                let elem = elements.last_mut().ok_or_else(|| {
                    IoError::Parse("PLY 'property' appeared before any 'element'".into())
                })?;
                if parts.len() >= 5 && parts[1] == "list" {
                    let count = PlyType::from_name(parts[2])?;
                    let item = PlyType::from_name(parts[3])?;
                    elem.properties.push(PlyProperty {
                        name: parts[4].to_string(),
                        kind: PlyPropertyKind::List { count, elem: item },
                    });
                } else if parts.len() >= 3 {
                    let ty = PlyType::from_name(parts[1])?;
                    elem.properties.push(PlyProperty {
                        name: parts[2].to_string(),
                        kind: PlyPropertyKind::Scalar(ty),
                    });
                } else {
                    return Err(IoError::Parse(format!("Invalid PLY property line: '{line}'")));
                }
            }
            _ => {}
        }
    }
}

fn read_ply_element_ascii<'a>(
    elem: &PlyElement,
    tokens: &mut impl Iterator<Item = &'a str>,
    poly: &mut PolyData,
) -> Result<()> {
    let row_cap = crate::limits::safe_prealloc_capacity(
        elem.count,
        std::mem::size_of::<Vec<Option<PlyValue>>>(),
        crate::limits::MAX_ALLOC_BYTES,
    )?;
    let mut rows_scalar: Vec<Vec<Option<PlyValue>>> = Vec::with_capacity(row_cap);
    let mut rows_list: Vec<Vec<Option<Vec<PlyValue>>>> = Vec::with_capacity(row_cap);

    for _ in 0..elem.count {
        let mut s_row = vec![None; elem.properties.len()];
        let mut l_row = vec![None; elem.properties.len()];
        for (p_idx, prop) in elem.properties.iter().enumerate() {
            match prop.kind {
                PlyPropertyKind::Scalar(ty) => {
                    let tok = tokens
                        .next()
                        .ok_or_else(|| IoError::InvalidData("Unexpected EOF in PLY ASCII".into()))?;
                    s_row[p_idx] = Some(parse_ply_ascii_value(tok, ty)?);
                }
                PlyPropertyKind::List { count, elem: ety } => {
                    let count_tok = tokens
                        .next()
                        .ok_or_else(|| IoError::InvalidData("Unexpected EOF in PLY list".into()))?;
                    let n = parse_ply_ascii_value(count_tok, count)?.as_usize();
                    let list_cap = crate::limits::safe_prealloc_capacity(
                        n,
                        std::mem::size_of::<PlyValue>(),
                        crate::limits::MAX_ALLOC_BYTES,
                    )?;
                    let mut list = Vec::with_capacity(list_cap);
                    for _ in 0..n {
                        let tok = tokens.next().ok_or_else(|| {
                            IoError::InvalidData("Unexpected EOF in PLY list items".into())
                        })?;
                        list.push(parse_ply_ascii_value(tok, ety)?);
                    }
                    l_row[p_idx] = Some(list);
                }
            }
        }
        rows_scalar.push(s_row);
        rows_list.push(l_row);
    }

    populate_ply_element(elem, &rows_scalar, &rows_list, poly)
}

fn read_ply_element_binary(
    elem: &PlyElement,
    body: &[u8],
    pos: &mut usize,
    little_endian: bool,
    poly: &mut PolyData,
) -> Result<()> {
    let row_cap = crate::limits::safe_prealloc_capacity(
        elem.count,
        std::mem::size_of::<Vec<Option<PlyValue>>>(),
        crate::limits::MAX_ALLOC_BYTES,
    )?;
    let mut rows_scalar: Vec<Vec<Option<PlyValue>>> = Vec::with_capacity(row_cap);
    let mut rows_list: Vec<Vec<Option<Vec<PlyValue>>>> = Vec::with_capacity(row_cap);

    for _ in 0..elem.count {
        let mut s_row = vec![None; elem.properties.len()];
        let mut l_row = vec![None; elem.properties.len()];
        for (p_idx, prop) in elem.properties.iter().enumerate() {
            match prop.kind {
                PlyPropertyKind::Scalar(ty) => {
                    s_row[p_idx] = Some(read_ply_bin_value(body, pos, ty, little_endian)?);
                }
                PlyPropertyKind::List { count, elem: ety } => {
                    let n = read_ply_bin_value(body, pos, count, little_endian)?.as_usize();
                    let list_cap = crate::limits::safe_prealloc_capacity(
                        n,
                        std::mem::size_of::<PlyValue>(),
                        crate::limits::MAX_ALLOC_BYTES,
                    )?;
                    let mut list = Vec::with_capacity(list_cap);
                    for _ in 0..n {
                        list.push(read_ply_bin_value(body, pos, ety, little_endian)?);
                    }
                    l_row[p_idx] = Some(list);
                }
            }
        }
        rows_scalar.push(s_row);
        rows_list.push(l_row);
    }

    populate_ply_element(elem, &rows_scalar, &rows_list, poly)
}

fn populate_ply_element(
    elem: &PlyElement,
    rows_scalar: &[Vec<Option<PlyValue>>],
    rows_list: &[Vec<Option<Vec<PlyValue>>>],
    poly: &mut PolyData,
) -> Result<()> {
    let prop_idx = |names: &[&str]| -> Option<usize> {
        elem.properties
            .iter()
            .position(|p| names.iter().any(|n| p.name == *n))
    };

    if elem.name == "vertex" {
        let ix = prop_idx(&["x"]);
        let iy = prop_idx(&["y"]);
        let iz = prop_idx(&["z"]);
        if let (Some(ix), Some(iy), Some(iz)) = (ix, iy, iz) {
            for row in rows_scalar {
                let x = row[ix].map(|v| v.as_f32()).unwrap_or(0.0);
                let y = row[iy].map(|v| v.as_f32()).unwrap_or(0.0);
                let z = row[iz].map(|v| v.as_f32()).unwrap_or(0.0);
                poly.points.push([x, y, z]);
            }
        }

        let inx = prop_idx(&["nx"]);
        let iny = prop_idx(&["ny"]);
        let inz = prop_idx(&["nz"]);
        if let (Some(inx), Some(iny), Some(inz)) = (inx, iny, inz) {
            let normals = rows_scalar
                .iter()
                .map(|row| {
                    [
                        row[inx].map(|v| v.as_f32()).unwrap_or(0.0),
                        row[iny].map(|v| v.as_f32()).unwrap_or(0.0),
                        row[inz].map(|v| v.as_f32()).unwrap_or(1.0),
                    ]
                })
                .collect();
            poly.point_data
                .set_normals(DataArray::vectors_f32("Normals", normals));
        }

        let ir = prop_idx(&["red", "r"]);
        let ig = prop_idx(&["green", "g"]);
        let ib = prop_idx(&["blue", "b"]);
        let ia = prop_idx(&["alpha", "a"]);
        if let (Some(ir), Some(ig), Some(ib)) = (ir, ig, ib) {
            let has_alpha = ia.is_some();
            let nc = if has_alpha { 4 } else { 3 };
            let mut colors = Vec::with_capacity(rows_scalar.len() * nc);
            for row in rows_scalar {
                colors.push(row[ir].map(|v| v.as_u8_color()).unwrap_or(0));
                colors.push(row[ig].map(|v| v.as_u8_color()).unwrap_or(0));
                colors.push(row[ib].map(|v| v.as_u8_color()).unwrap_or(0));
                if let Some(ia_idx) = ia {
                    colors.push(row[ia_idx].map(|v| v.as_u8_color()).unwrap_or(255));
                }
            }
            poly.point_data
                .add_array(DataArray::colors_u8("Colors", nc, colors));
        }

        let special: Vec<Option<usize>> = vec![ix, iy, iz, inx, iny, inz, ir, ig, ib, ia];
        for (p_idx, prop) in elem.properties.iter().enumerate() {
            if special.contains(&Some(p_idx)) {
                continue;
            }
            if let PlyPropertyKind::Scalar(ty) = prop.kind {
                let arr = collect_ply_scalar_array(&prop.name, ty, p_idx, rows_scalar);
                poly.point_data.add_array(arr);
            }
        }
    } else if elem.name == "face" {
        let iv = prop_idx(&["vertex_indices", "vertex_index"]);
        if let Some(iv) = iv {
            for row in rows_list {
                if let Some(list) = &row[iv] {
                    let face: Vec<u32> = list.iter().map(|v| v.as_usize() as u32).collect();
                    if face.len() >= 3 {
                        poly.polys.push(face);
                    }
                }
            }
        }
        for (p_idx, prop) in elem.properties.iter().enumerate() {
            if Some(p_idx) == iv {
                continue;
            }
            if let PlyPropertyKind::Scalar(ty) = prop.kind {
                let arr = collect_ply_scalar_array(&prop.name, ty, p_idx, rows_scalar);
                poly.cell_data.add_array(arr);
            }
        }
    } else if elem.name == "edge"
        && let (Some(i1), Some(i2)) = (prop_idx(&["vertex1"]), prop_idx(&["vertex2"]))
    {
        for row in rows_scalar {
            if let (Some(v1), Some(v2)) = (row[i1], row[i2]) {
                poly.lines.push(vec![v1.as_usize() as u32, v2.as_usize() as u32]);
            }
        }
    }

    Ok(())
}

fn collect_ply_scalar_array(
    name: &str,
    ty: PlyType,
    p_idx: usize,
    rows: &[Vec<Option<PlyValue>>],
) -> DataArray {
    let data = match ty {
        PlyType::U8 => ArrayData::U8(
            rows.iter()
                .map(|r| match r[p_idx] {
                    Some(PlyValue::U8(v)) => v,
                    Some(other) => other.as_usize() as u8,
                    None => 0,
                })
                .collect(),
        ),
        PlyType::I8 | PlyType::I16 | PlyType::I32 => ArrayData::I32(
            rows.iter()
                .map(|r| match r[p_idx] {
                    Some(PlyValue::I8(v)) => v as i32,
                    Some(PlyValue::I16(v)) => v as i32,
                    Some(PlyValue::I32(v)) => v,
                    Some(other) => other.as_f64() as i32,
                    None => 0,
                })
                .collect(),
        ),
        PlyType::U16 | PlyType::U32 => ArrayData::U32(
            rows.iter()
                .map(|r| match r[p_idx] {
                    Some(PlyValue::U16(v)) => v as u32,
                    Some(PlyValue::U32(v)) => v,
                    Some(other) => other.as_usize() as u32,
                    None => 0,
                })
                .collect(),
        ),
        PlyType::F32 => ArrayData::F32(
            rows.iter()
                .map(|r| r[p_idx].map(|v| v.as_f32()).unwrap_or(0.0))
                .collect(),
        ),
        PlyType::F64 => ArrayData::F64(
            rows.iter()
                .map(|r| r[p_idx].map(|v| v.as_f64()).unwrap_or(0.0))
                .collect(),
        ),
    };
    DataArray::new(name, 1, data)
}

fn parse_ply_ascii_value(tok: &str, ty: PlyType) -> Result<PlyValue> {
    let parse_err = || IoError::Parse(format!("Invalid PLY value '{tok}' for {ty:?}"));
    Ok(match ty {
        PlyType::I8 => PlyValue::I8(tok.parse().map_err(|_| parse_err())?),
        PlyType::U8 => PlyValue::U8(tok.parse().map_err(|_| parse_err())?),
        PlyType::I16 => PlyValue::I16(tok.parse().map_err(|_| parse_err())?),
        PlyType::U16 => PlyValue::U16(tok.parse().map_err(|_| parse_err())?),
        PlyType::I32 => PlyValue::I32(tok.parse().map_err(|_| parse_err())?),
        PlyType::U32 => PlyValue::U32(tok.parse().map_err(|_| parse_err())?),
        PlyType::F32 => PlyValue::F32(tok.parse().map_err(|_| parse_err())?),
        PlyType::F64 => PlyValue::F64(tok.parse().map_err(|_| parse_err())?),
    })
}

fn read_ply_bin_value(
    body: &[u8],
    pos: &mut usize,
    ty: PlyType,
    little_endian: bool,
) -> Result<PlyValue> {
    let sz = ty.byte_size();
    if *pos + sz > body.len() {
        return Err(IoError::InvalidData("Unexpected EOF in binary PLY".into()));
    }
    let s = &body[*pos..*pos + sz];
    *pos += sz;
    Ok(match ty {
        PlyType::I8 => PlyValue::I8(s[0] as i8),
        PlyType::U8 => PlyValue::U8(s[0]),
        PlyType::I16 => PlyValue::I16(if little_endian {
            i16::from_le_bytes([s[0], s[1]])
        } else {
            i16::from_be_bytes([s[0], s[1]])
        }),
        PlyType::U16 => PlyValue::U16(if little_endian {
            u16::from_le_bytes([s[0], s[1]])
        } else {
            u16::from_be_bytes([s[0], s[1]])
        }),
        PlyType::I32 => PlyValue::I32(if little_endian {
            i32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            i32::from_be_bytes([s[0], s[1], s[2], s[3]])
        }),
        PlyType::U32 => PlyValue::U32(if little_endian {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        }),
        PlyType::F32 => PlyValue::F32(if little_endian {
            f32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            f32::from_be_bytes([s[0], s[1], s[2], s[3]])
        }),
        PlyType::F64 => PlyValue::F64(if little_endian {
            f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
        } else {
            f64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
        }),
    })
}

/// Writes a [`PolyData`] to a PLY file (`binary_little_endian` by default).
pub fn write_ply(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    write_ply_with_format(path, poly, PlyFormat::BinaryLittleEndian)
}

/// Writes a [`PolyData`] to a PLY file with an explicit [`PlyFormat`].
pub fn write_ply_with_format(
    path: impl AsRef<Path>,
    poly: &PolyData,
    format: PlyFormat,
) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let fmt_str = match format {
        PlyFormat::Ascii => "ascii",
        PlyFormat::BinaryLittleEndian => "binary_little_endian",
        PlyFormat::BinaryBigEndian => "binary_big_endian",
    };

    let normals = poly
        .point_data
        .active_normals_array()
        .filter(|a| a.num_components == 3 && a.num_tuples() == poly.points.len());
    let colors = poly
        .point_data
        .get("Colors")
        .or_else(|| poly.point_data.get("RGBA"))
        .or_else(|| poly.point_data.get("RGB"))
        .filter(|a| {
            matches!(a.data, ArrayData::U8(_))
                && (a.num_components == 3 || a.num_components == 4)
                && a.num_tuples() == poly.points.len()
        });

    let normals_name = normals.map(|a| a.name.as_str());
    let colors_name = colors.map(|a| a.name.as_str());
    let extra_point_arrays: Vec<&DataArray> = poly
        .point_data
        .arrays()
        .iter()
        .filter(|a| {
            Some(a.name.as_str()) != normals_name
                && Some(a.name.as_str()) != colors_name
                && a.num_components == 1
                && a.num_tuples() == poly.points.len()
        })
        .collect();

    let extra_cell_arrays: Vec<&DataArray> = poly
        .cell_data
        .arrays()
        .iter()
        .filter(|a| a.num_components == 1 && a.num_tuples() == poly.polys.len())
        .collect();

    writeln!(w, "ply")?;
    writeln!(w, "format {fmt_str} 1.0")?;
    writeln!(w, "comment Generated by ParaViz")?;
    writeln!(w, "element vertex {}", poly.points.len())?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    if normals.is_some() {
        writeln!(w, "property float nx")?;
        writeln!(w, "property float ny")?;
        writeln!(w, "property float nz")?;
    }
    if let Some(c) = colors {
        writeln!(w, "property uchar red")?;
        writeln!(w, "property uchar green")?;
        writeln!(w, "property uchar blue")?;
        if c.num_components == 4 {
            writeln!(w, "property uchar alpha")?;
        }
    }
    for arr in &extra_point_arrays {
        writeln!(
            w,
            "property {} {}",
            scalar_type_to_ply(arr.data.scalar_type()).ply_name(),
            arr.name
        )?;
    }

    if !poly.polys.is_empty() {
        writeln!(w, "element face {}", poly.polys.len())?;
        writeln!(w, "property list uchar int vertex_indices")?;
        for arr in &extra_cell_arrays {
            writeln!(
                w,
                "property {} {}",
                scalar_type_to_ply(arr.data.scalar_type()).ply_name(),
                arr.name
            )?;
        }
    }
    writeln!(w, "end_header")?;

    for (i, p) in poly.points.iter().enumerate() {
        write_ply_f32(&mut w, format, p[0])?;
        write_ply_space(&mut w, format)?;
        write_ply_f32(&mut w, format, p[1])?;
        write_ply_space(&mut w, format)?;
        write_ply_f32(&mut w, format, p[2])?;

        if let Some(ns) = normals {
            let n = ns.get_vec3(i);
            for comp in n {
                write_ply_space(&mut w, format)?;
                write_ply_f32(&mut w, format, comp)?;
            }
        }
        if let Some(cs) = colors
            && let ArrayData::U8(buf) = &cs.data
        {
            let base = i * cs.num_components;
            for c in 0..cs.num_components {
                write_ply_space(&mut w, format)?;
                write_ply_u8(&mut w, format, buf[base + c])?;
            }
        }
        for arr in &extra_point_arrays {
            write_ply_space(&mut w, format)?;
            write_ply_scalar_at(&mut w, format, arr, i)?;
        }
        if format == PlyFormat::Ascii {
            writeln!(w)?;
        }
    }

    for (cell_idx, face) in poly.polys.iter().enumerate() {
        write_ply_u8(&mut w, format, face.len() as u8)?;
        for &v in face {
            write_ply_space(&mut w, format)?;
            write_ply_i32(&mut w, format, v as i32)?;
        }
        for arr in &extra_cell_arrays {
            write_ply_space(&mut w, format)?;
            write_ply_scalar_at(&mut w, format, arr, cell_idx)?;
        }
        if format == PlyFormat::Ascii {
            writeln!(w)?;
        }
    }

    w.flush()?;
    Ok(())
}

fn scalar_type_to_ply(st: ScalarType) -> PlyType {
    match st {
        ScalarType::I8 => PlyType::I8,
        ScalarType::U8 => PlyType::U8,
        ScalarType::I16 => PlyType::I16,
        ScalarType::U16 => PlyType::U16,
        ScalarType::I32 => PlyType::I32,
        ScalarType::U32 => PlyType::U32,
        ScalarType::I64 | ScalarType::U64 => PlyType::F64,
        ScalarType::F32 => PlyType::F32,
        ScalarType::F64 => PlyType::F64,
    }
}

fn write_ply_space(w: &mut impl Write, fmt: PlyFormat) -> Result<()> {
    if fmt == PlyFormat::Ascii {
        w.write_all(b" ")?;
    }
    Ok(())
}

fn write_ply_u8(w: &mut impl Write, fmt: PlyFormat, v: u8) -> Result<()> {
    match fmt {
        PlyFormat::Ascii => write!(w, "{v}")?,
        PlyFormat::BinaryLittleEndian | PlyFormat::BinaryBigEndian => w.write_all(&[v])?,
    }
    Ok(())
}

fn write_ply_i32(w: &mut impl Write, fmt: PlyFormat, v: i32) -> Result<()> {
    match fmt {
        PlyFormat::Ascii => write!(w, "{v}")?,
        PlyFormat::BinaryLittleEndian => w.write_all(&v.to_le_bytes())?,
        PlyFormat::BinaryBigEndian => w.write_all(&v.to_be_bytes())?,
    }
    Ok(())
}

fn write_ply_u32(w: &mut impl Write, fmt: PlyFormat, v: u32) -> Result<()> {
    match fmt {
        PlyFormat::Ascii => write!(w, "{v}")?,
        PlyFormat::BinaryLittleEndian => w.write_all(&v.to_le_bytes())?,
        PlyFormat::BinaryBigEndian => w.write_all(&v.to_be_bytes())?,
    }
    Ok(())
}

fn write_ply_f32(w: &mut impl Write, fmt: PlyFormat, v: f32) -> Result<()> {
    match fmt {
        PlyFormat::Ascii => write!(w, "{v}")?,
        PlyFormat::BinaryLittleEndian => w.write_all(&v.to_le_bytes())?,
        PlyFormat::BinaryBigEndian => w.write_all(&v.to_be_bytes())?,
    }
    Ok(())
}

fn write_ply_f64(w: &mut impl Write, fmt: PlyFormat, v: f64) -> Result<()> {
    match fmt {
        PlyFormat::Ascii => write!(w, "{v}")?,
        PlyFormat::BinaryLittleEndian => w.write_all(&v.to_le_bytes())?,
        PlyFormat::BinaryBigEndian => w.write_all(&v.to_be_bytes())?,
    }
    Ok(())
}

fn write_ply_scalar_at(w: &mut impl Write, fmt: PlyFormat, arr: &DataArray, idx: usize) -> Result<()> {
    match &arr.data {
        ArrayData::I8(v) => match fmt {
            PlyFormat::Ascii => write!(w, "{}", v[idx]).map_err(Into::into),
            PlyFormat::BinaryLittleEndian | PlyFormat::BinaryBigEndian => {
                w.write_all(&v[idx].to_le_bytes()).map_err(Into::into)
            }
        },
        ArrayData::U8(v) => write_ply_u8(w, fmt, v[idx]),
        ArrayData::I16(v) => match fmt {
            PlyFormat::Ascii => write!(w, "{}", v[idx]).map_err(Into::into),
            PlyFormat::BinaryLittleEndian => w.write_all(&v[idx].to_le_bytes()).map_err(Into::into),
            PlyFormat::BinaryBigEndian => w.write_all(&v[idx].to_be_bytes()).map_err(Into::into),
        },
        ArrayData::U16(v) => match fmt {
            PlyFormat::Ascii => write!(w, "{}", v[idx]).map_err(Into::into),
            PlyFormat::BinaryLittleEndian => w.write_all(&v[idx].to_le_bytes()).map_err(Into::into),
            PlyFormat::BinaryBigEndian => w.write_all(&v[idx].to_be_bytes()).map_err(Into::into),
        },
        ArrayData::I32(v) => write_ply_i32(w, fmt, v[idx]),
        ArrayData::U32(v) => write_ply_u32(w, fmt, v[idx]),
        ArrayData::I64(v) => write_ply_f64(w, fmt, v[idx] as f64),
        ArrayData::U64(v) => write_ply_f64(w, fmt, v[idx] as f64),
        ArrayData::F32(v) => write_ply_f32(w, fmt, v[idx]),
        ArrayData::F64(v) => write_ply_f64(w, fmt, v[idx]),
    }
}

fn read_f32_le(s: &[u8]) -> f32 {
    f32::from_le_bytes([s[0], s[1], s[2], s[3]])
}

fn parse_f32(s: &str) -> Result<f32> {
    s.parse::<f32>()
        .map_err(|_| IoError::Parse(format!("Invalid float '{s}'")))
}
