//! Delimited CSV/TSV point-cloud reader and writer.

use crate::error::{IoError, Result};
use paraviz_core::{DataArray, PolyData};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Reads a delimited CSV/TSV point-cloud file into a [`PolyData`] with vertex cells.
///
/// Automatically detects the delimiter (`,`, `\t`, `;`) and coordinate columns
/// (`x`/`y`/`z` or `Points:0`/`Points:1`/`Points:2`, case-insensitive). All other
/// numeric columns are loaded into `point_data` as `Float64` scalar arrays.
pub fn read_csv(path: impl AsRef<Path>) -> Result<PolyData> {
    let content = fs::read_to_string(path)?;
    read_csv_str(&content)
}

/// Parses delimited CSV/TSV point-cloud text into a [`PolyData`].
pub fn read_csv_str(content: &str) -> Result<PolyData> {
    let mut lines = content
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        });

    let header_line = lines
        .next()
        .ok_or_else(|| IoError::InvalidData("CSV file is empty".into()))?;
    let delimiter = detect_delimiter(header_line);
    let headers = split_csv_line(header_line, delimiter);
    if headers.is_empty() {
        return Err(IoError::InvalidData("CSV header row is empty".into()));
    }

    let mut x_col = None;
    let mut y_col = None;
    let mut z_col = None;

    for (idx, h) in headers.iter().enumerate() {
        let lower = h.trim().to_ascii_lowercase();
        match lower.as_str() {
            "x" | "points:0" | "points_0" | "pos_x" if x_col.is_none() => x_col = Some(idx),
            "y" | "points:1" | "points_1" | "pos_y" if y_col.is_none() => y_col = Some(idx),
            "z" | "points:2" | "points_2" | "pos_z" if z_col.is_none() => z_col = Some(idx),
            _ => {}
        }
    }

    let x_idx = x_col.ok_or_else(|| {
        IoError::InvalidData(
            "CSV header must contain an x/Points:0 coordinate column".into(),
        )
    })?;
    let y_idx = y_col.ok_or_else(|| {
        IoError::InvalidData(
            "CSV header must contain a y/Points:1 coordinate column".into(),
        )
    })?;

    let num_cols = headers.len();
    let mut extra_cols: Vec<(usize, String, Vec<f64>, bool)> = headers
        .iter()
        .enumerate()
        .filter(|&(i, _)| Some(i) != x_col && Some(i) != y_col && Some(i) != z_col)
        .map(|(i, name)| (i, name.clone(), Vec::new(), true))
        .collect();

    let mut points = Vec::new();

    for (row_num, line) in lines.enumerate() {
        let fields = split_csv_line(line, delimiter);
        if fields.len() < num_cols && fields.iter().all(|f| f.is_empty()) {
            continue;
        }
        let get_f64 = |col: usize| -> Result<f64> {
            let s = fields.get(col).map(|s| s.trim()).unwrap_or("");
            s.parse::<f64>().map_err(|_| {
                IoError::Parse(format!(
                    "Invalid numeric coordinate at row {}, column {}: '{}'",
                    row_num + 2,
                    col + 1,
                    s
                ))
            })
        };

        let x = get_f64(x_idx)? as f32;
        let y = get_f64(y_idx)? as f32;
        let z = match z_col {
            Some(zi) => get_f64(zi)? as f32,
            None => 0.0,
        };
        points.push([x, y, z]);

        for (col, _, vals, is_numeric) in &mut extra_cols {
            if !*is_numeric {
                continue;
            }
            let raw = fields.get(*col).map(|s| s.trim()).unwrap_or("");
            if raw.is_empty() {
                vals.push(0.0);
            } else if let Ok(v) = raw.parse::<f64>() {
                vals.push(v);
            } else {
                *is_numeric = false;
            }
        }
    }

    let verts = (0..points.len() as u32).collect();
    let mut poly = PolyData {
        points,
        verts,
        ..Default::default()
    };

    for (_, name, vals, is_numeric) in extra_cols {
        if is_numeric && vals.len() == poly.points.len() && !name.is_empty() {
            poly.point_data.add_array(DataArray::scalars_f64(name, vals));
        }
    }

    Ok(poly)
}

/// Writes a [`PolyData`] point cloud and its `point_data` arrays to a CSV file.
pub fn write_csv(path: impl AsRef<Path>, poly: &PolyData) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    write!(w, "x,y,z")?;
    for arr in poly.point_data.arrays() {
        if arr.num_components == 1 {
            write!(w, ",{}", escape_csv_field(&arr.name))?;
        } else {
            for c in 0..arr.num_components {
                write!(w, ",{}", escape_csv_field(&format!("{}:{c}", arr.name)))?;
            }
        }
    }
    writeln!(w)?;

    for (i, p) in poly.points.iter().enumerate() {
        write!(w, "{},{},{}", p[0], p[1], p[2])?;
        for arr in poly.point_data.arrays() {
            if i < arr.num_tuples() {
                for c in 0..arr.num_components {
                    let v = arr.data.get_f64(i * arr.num_components + c);
                    write!(w, ",{v}")?;
                }
            } else {
                for _ in 0..arr.num_components {
                    write!(w, ",0")?;
                }
            }
        }
        writeln!(w)?;
    }

    w.flush()?;
    Ok(())
}

fn detect_delimiter(header: &str) -> char {
    let mut commas = 0usize;
    let mut tabs = 0usize;
    let mut semis = 0usize;
    let mut in_quotes = false;
    for ch in header.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => commas += 1,
            '\t' if !in_quotes => tabs += 1,
            ';' if !in_quotes => semis += 1,
            _ => {}
        }
    }
    if tabs > commas && tabs >= semis {
        '\t'
    } else if semis > commas && semis > tabs {
        ';'
    } else {
        ','
    }
}

fn split_csv_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                cur.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
        } else if ch == delimiter && !in_quotes {
            fields.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    fields.push(cur.trim().to_string());
    fields
}

fn escape_csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
