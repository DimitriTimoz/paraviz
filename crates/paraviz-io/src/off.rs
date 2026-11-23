//! Geomview OFF (`.off`) ASCII polygonal mesh reader and writer.
//!
//! Supports `OFF`, `COFF` (per-vertex RGBA), `NOFF` (per-vertex normals), and `CNOFF` headers.

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, PolyData};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Reads an ASCII Geomview OFF file (`.off`) into a [`PolyData`].
pub fn read_off(path: impl AsRef<Path>) -> Result<PolyData> {
    let text = fs::read_to_string(path)?;
    read_off_str(&text)
}

/// Parses an ASCII Geomview OFF string into a [`PolyData`].
pub fn read_off_str(text: &str) -> Result<PolyData> {
    let mut tokens = OffTokens::new(text);

    let first = tokens
        .next()
        .ok_or_else(|| IoError::Parse("Empty OFF file".into()))?;

    let mut has_color = false;
    let mut has_normal = false;
    let mut has_texcoord = false;

    let mut header = first;
    // Optional ST / C / N prefixes before OFF, e.g. OFF, COFF, NOFF, CNOFF, STOFF
    let (v_count, f_count) = if header.ends_with("OFF") {
        let prefix = &header[..header.len() - 3];
        if prefix.contains('C') {
            has_color = true;
        }
        if prefix.contains('N') {
            has_normal = true;
        }
        if prefix.contains("ST") {
            has_texcoord = true;
        }
        let nv: usize = tokens.next_parse("vertex count")?;
        let nf: usize = tokens.next_parse("face count")?;
        let _ne: usize = tokens.next_parse("edge count")?;
        (nv, nf)
    } else {
        // Some OFF files omit the "OFF" keyword and start directly with `n_verts n_faces n_edges`.
        let nv: usize = header
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid OFF header token '{header}'")))?;
        let nf: usize = tokens.next_parse("face count")?;
        let _ne: usize = tokens.next_parse("edge count")?;
        (nv, nf)
    };
    let _ = &mut header;

    let mut points = Vec::with_capacity(safe_prealloc_capacity(v_count, 12, MAX_ALLOC_BYTES)?);
    let mut normals = if has_normal {
        Vec::with_capacity(safe_prealloc_capacity(v_count, 12, MAX_ALLOC_BYTES)?)
    } else {
        Vec::new()
    };
    let mut colors = if has_color {
        Vec::with_capacity(safe_prealloc_capacity(v_count, 4, MAX_ALLOC_BYTES)?)
    } else {
        Vec::new()
    };

    for i in 0..v_count {
        let x: f32 = tokens
            .next_parse("vertex x")
            .map_err(|e| IoError::Parse(format!("OFF vertex {i}: {e}")))?;
        let y: f32 = tokens
            .next_parse("vertex y")
            .map_err(|e| IoError::Parse(format!("OFF vertex {i}: {e}")))?;
        let z: f32 = tokens
            .next_parse("vertex z")
            .map_err(|e| IoError::Parse(format!("OFF vertex {i}: {e}")))?;
        points.push([x, y, z]);

        if has_normal {
            let nx: f32 = tokens.next_parse("normal x")?;
            let ny: f32 = tokens.next_parse("normal y")?;
            let nz: f32 = tokens.next_parse("normal z")?;
            normals.push([nx, ny, nz]);
        }
        if has_color {
            let r = parse_off_color_component(tokens.next_token("color r")?)?;
            let g = parse_off_color_component(tokens.next_token("color g")?)?;
            let b = parse_off_color_component(tokens.next_token("color b")?)?;
            let a = parse_off_color_component(tokens.next_token("color a")?)?;
            colors.push([r, g, b, a]);
        }
        if has_texcoord {
            let _u: f32 = tokens.next_parse("texcoord u")?;
            let _v: f32 = tokens.next_parse("texcoord v")?;
        }
    }

    let mut polys: Vec<Vec<u32>> = Vec::with_capacity(safe_prealloc_capacity(f_count, 24, MAX_ALLOC_BYTES)?);
    let mut face_colors: Vec<[u8; 4]> = Vec::new();

    for fi in 0..f_count {
        // Faces are line-oriented because optional face colors appear on the same line after indices.
        let line_tokens = tokens
            .next_line_tokens()
            .ok_or_else(|| IoError::Parse(format!("Unexpected EOF reading OFF face {fi}")))?;
        if line_tokens.is_empty() {
            return Err(IoError::Parse(format!("Empty OFF face line at {fi}")));
        }
        let n: usize = line_tokens[0]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid face vertex count '{}'", line_tokens[0])))?;
        if n < 1 || line_tokens.len() < 1 + n {
            return Err(IoError::Parse(format!(
                "OFF face {fi} declares {n} vertices but line has {} tokens",
                line_tokens.len().saturating_sub(1)
            )));
        }
        let mut poly = Vec::with_capacity(n);
        for tok in &line_tokens[1..=n] {
            let idx: u32 = tok
                .parse()
                .map_err(|_| IoError::Parse(format!("Invalid face vertex index '{tok}'")))?;
            if idx as usize >= points.len() {
                return Err(IoError::InvalidData(format!(
                    "OFF face {fi} vertex index {idx} out of bounds ({})",
                    points.len()
                )));
            }
            poly.push(idx);
        }
        polys.push(poly);

        let rest = &line_tokens[1 + n..];
        if rest.len() >= 3 {
            if face_colors.is_empty() && fi > 0 {
                face_colors.resize(fi, [255, 255, 255, 255]);
            }
            let r = parse_off_color_component(rest[0])?;
            let g = parse_off_color_component(rest[1])?;
            let b = parse_off_color_component(rest[2])?;
            let a = if rest.len() >= 4 {
                parse_off_color_component(rest[3])?
            } else {
                255
            };
            face_colors.push([r, g, b, a]);
        } else if !face_colors.is_empty() {
            face_colors.push([255, 255, 255, 255]);
        }
    }

    let mut poly = PolyData::new();
    poly.points = points;
    poly.polys = polys;
    if has_normal {
        let flat: Vec<f32> = normals.into_iter().flat_map(|n| n.into_iter()).collect();
        poly.point_data
            .add_array(DataArray::new("Normals", 3, ArrayData::F32(flat)));
    }
    if has_color {
        let flat: Vec<u8> = colors.into_iter().flat_map(|c| c.into_iter()).collect();
        poly.point_data
            .add_array(DataArray::new("Colors", 4, ArrayData::U8(flat)));
    }
    if !face_colors.is_empty() {
        let flat: Vec<u8> = face_colors.into_iter().flat_map(|c| c.into_iter()).collect();
        poly.cell_data
            .add_array(DataArray::new("Colors", 4, ArrayData::U8(flat)));
    }
    Ok(poly)
}

/// Writes a [`PolyData`] mesh to an ASCII Geomview OFF file (`.off`).
pub fn write_off(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let colors = poly.point_data.get("Colors").and_then(|arr| {
        if arr.num_components == 4 && arr.num_tuples() == poly.points.len() {
            Some(arr)
        } else {
            None
        }
    });

    if colors.is_some() {
        writeln!(w, "COFF")?;
    } else {
        writeln!(w, "OFF")?;
    }
    writeln!(w, "{} {} 0", poly.points.len(), poly.polys.len())?;

    for (i, p) in poly.points.iter().enumerate() {
        if let Some(carr) = colors {
            let r = carr.data.get_f64(i * 4) as u8;
            let g = carr.data.get_f64(i * 4 + 1) as u8;
            let b = carr.data.get_f64(i * 4 + 2) as u8;
            let a = carr.data.get_f64(i * 4 + 3) as u8;
            writeln!(w, "{} {} {} {} {} {} {}", p[0], p[1], p[2], r, g, b, a)?;
        } else {
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }
    }

    for face in &poly.polys {
        write!(w, "{}", face.len())?;
        for &idx in face {
            write!(w, " {idx}")?;
        }
        writeln!(w)?;
    }
    w.flush()?;
    Ok(())
}

fn parse_off_color_component(tok: &str) -> Result<u8> {
    if tok.contains('.') || tok.contains('e') || tok.contains('E') {
        let v: f32 = tok
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid OFF float color '{tok}'")))?;
        Ok((v.clamp(0.0, 1.0) * 255.0).round() as u8)
    } else {
        let v: i32 = tok
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid OFF int color '{tok}'")))?;
        Ok(v.clamp(0, 255) as u8)
    }
}

struct OffTokens<'a> {
    lines: std::str::Lines<'a>,
    cur_line: Vec<&'a str>,
    cur_idx: usize,
}

impl<'a> OffTokens<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines(),
            cur_line: Vec::new(),
            cur_idx: 0,
        }
    }

    fn refill(&mut self) -> bool {
        while self.cur_idx >= self.cur_line.len() {
            let Some(raw) = self.lines.next() else {
                return false;
            };
            let clean = raw.split('#').next().unwrap_or("").trim();
            if clean.is_empty() {
                continue;
            }
            self.cur_line = clean.split_whitespace().collect();
            self.cur_idx = 0;
        }
        true
    }

    fn next(&mut self) -> Option<&'a str> {
        if !self.refill() {
            return None;
        }
        let tok = self.cur_line[self.cur_idx];
        self.cur_idx += 1;
        Some(tok)
    }

    fn next_token(&mut self, what: &str) -> Result<&'a str> {
        self.next()
            .ok_or_else(|| IoError::Parse(format!("Unexpected EOF reading OFF {what}")))
    }

    fn next_parse<T: std::str::FromStr>(&mut self, what: &str) -> Result<T> {
        let tok = self.next_token(what)?;
        tok.parse::<T>()
            .map_err(|_| IoError::Parse(format!("Invalid OFF {what} '{tok}'")))
    }

    fn next_line_tokens(&mut self) -> Option<Vec<&'a str>> {
        if !self.refill() {
            return None;
        }
        let rest = self.cur_line[self.cur_idx..].to_vec();
        self.cur_idx = self.cur_line.len();
        Some(rest)
    }
}
