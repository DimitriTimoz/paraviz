//! Molecular / particle dataset readers: XYZ (`.xyz`) and Protein Data Bank (`.pdb`).
//!
//! Both formats support multi-frame trajectories as temporal time steps.

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, PolyData};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ================================================================================================
// XYZ Molecular Format
// ================================================================================================

/// Returns `true` if `text` starts with a valid XYZ molecular header (`<int>\n<comment>\n<sym> <x> <y> <z>`).
pub fn looks_like_xyz_molecular(text: &str) -> bool {
    let mut lines = text.lines();
    let Some(first) = lines.next() else {
        return false;
    };
    let Ok(count) = first.trim().parse::<usize>() else {
        return false;
    };
    if count == 0 {
        return false;
    }
    // Comment line
    if lines.next().is_none() {
        return false;
    }
    let Some(atom_line) = lines.next() else {
        return false;
    };
    let parts: Vec<&str> = atom_line.split_whitespace().collect();
    if parts.len() < 4 {
        return false;
    }
    // Element symbol (letters or atomic number) followed by 3 floats
    let sym_ok = parts[0].chars().all(|c| c.is_ascii_alphanumeric());
    sym_ok
        && parts[1].parse::<f64>().is_ok()
        && parts[2].parse::<f64>().is_ok()
        && parts[3].parse::<f64>().is_ok()
}

/// Inspects the number of frames in an `.xyz` molecular trajectory as `0.0, 1.0, ...`
/// without parsing atom coordinates.
pub fn inspect_xyz_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut cursor = 0usize;
    let mut frame_count = 0usize;
    while cursor < lines.len() {
        while cursor < lines.len() && lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        if cursor >= lines.len() {
            break;
        }
        let Ok(n_atoms) = lines[cursor].trim().parse::<usize>() else {
            break;
        };
        if cursor + 1 + n_atoms >= lines.len() {
            break;
        }
        cursor += 2 + n_atoms;
        frame_count += 1;
    }
    if frame_count <= 1 {
        Ok(Vec::new())
    } else {
        Ok((0..frame_count).map(|i| i as f64).collect())
    }
}

/// Reads the first frame of an `.xyz` molecular file.
pub fn read_xyz(path: impl AsRef<Path>) -> Result<DataSet> {
    read_xyz_at_time(path, 0.0)
}

/// Reads the frame closest to `time` from an `.xyz` molecular file.
pub fn read_xyz_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    let text = fs::read_to_string(path)?;
    let target_idx = time.round().max(0.0) as usize;
    let poly = parse_xyz_frame_at(&text, target_idx)?;
    Ok(DataSet::PolyData(poly))
}

fn parse_xyz_frame_at(text: &str, target_idx: usize) -> Result<PolyData> {
    let lines: Vec<&str> = text.lines().collect();
    let mut cursor = 0usize;
    let mut frame_idx = 0usize;
    let mut last_span: Option<(usize, usize)> = None;

    while cursor < lines.len() {
        while cursor < lines.len() && lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        if cursor >= lines.len() {
            break;
        }
        let n_atoms: usize = lines[cursor].trim().parse().map_err(|_| {
            IoError::Parse(format!(
                "Invalid XYZ atom count at line {}: '{}'",
                cursor + 1,
                lines[cursor]
            ))
        })?;
        cursor += 1;
        if cursor >= lines.len() {
            return Err(IoError::Parse("XYZ file missing comment line".into()));
        }
        cursor += 1; // skip comment line
        if cursor + n_atoms > lines.len() {
            return Err(IoError::Parse(format!(
                "XYZ frame truncated: expected {n_atoms} atom lines"
            )));
        }
        last_span = Some((cursor, n_atoms));
        if frame_idx == target_idx {
            break;
        }
        cursor += n_atoms;
        frame_idx += 1;
    }

    let (start, n_atoms) =
        last_span.ok_or_else(|| IoError::Parse("XYZ file contains no frames".into()))?;
    let mut points = Vec::with_capacity(safe_prealloc_capacity(n_atoms, 12, MAX_ALLOC_BYTES)?);
    let mut verts = Vec::with_capacity(safe_prealloc_capacity(n_atoms, 4, MAX_ALLOC_BYTES)?);
    let mut atomic_numbers = Vec::with_capacity(safe_prealloc_capacity(n_atoms, 4, MAX_ALLOC_BYTES)?);
    let mut radii = Vec::with_capacity(safe_prealloc_capacity(n_atoms, 4, MAX_ALLOC_BYTES)?);

    for i in 0..n_atoms {
        let line = lines[start + i];
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(IoError::Parse(format!(
                "Invalid XYZ atom line {}: '{}'",
                start + i + 1,
                line
            )));
        }
        let x: f32 = parts[1]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid XYZ x '{}'", parts[1])))?;
        let y: f32 = parts[2]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid XYZ y '{}'", parts[2])))?;
        let z: f32 = parts[3]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid XYZ z '{}'", parts[3])))?;

        let (z_num, rad) = element_properties(parts[0]);
        points.push([x, y, z]);
        verts.push(i as u32);
        atomic_numbers.push(z_num);
        radii.push(rad);
    }

    let mut poly = PolyData::new();
    poly.points = points;
    poly.verts = verts;
    poly.point_data.add_array(DataArray::new(
        "AtomicNumber",
        1,
        ArrayData::I32(atomic_numbers),
    ));
    poly.point_data
        .add_array(DataArray::new("Radius", 1, ArrayData::F32(radii)));
    Ok(poly)
}

// ================================================================================================
// Protein Data Bank (PDB) Format
// ================================================================================================

/// Inspects `MODEL` time steps in a `.pdb` file (empty if single-model).
pub fn inspect_pdb_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path)?;
    let frames = parse_pdb_models(&text)?;
    if frames.len() <= 1 {
        Ok(Vec::new())
    } else {
        Ok((0..frames.len()).map(|i| i as f64).collect())
    }
}

/// Reads a `.pdb` file (first `MODEL`) into a [`DataSet`].
pub fn read_pdb(path: impl AsRef<Path>) -> Result<DataSet> {
    read_pdb_at_time(path, 0.0)
}

/// Reads the `MODEL` closest to `time` from a `.pdb` file.
pub fn read_pdb_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    let text = fs::read_to_string(path)?;
    let frames = parse_pdb_models(&text)?;
    let idx = (time.round().max(0.0) as usize).min(frames.len().saturating_sub(1));
    Ok(DataSet::PolyData(frames[idx].clone()))
}

struct PdbModel {
    points: Vec<[f32; 3]>,
    serial_to_idx: HashMap<i32, u32>,
    atomic_numbers: Vec<i32>,
    radii: Vec<f32>,
    occupancies: Vec<f32>,
    temp_factors: Vec<f32>,
    residue_seq: Vec<i32>,
    is_hetatm: Vec<u8>,
    bonds: Vec<Vec<u32>>,
}

impl PdbModel {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            serial_to_idx: HashMap::new(),
            atomic_numbers: Vec::new(),
            radii: Vec::new(),
            occupancies: Vec::new(),
            temp_factors: Vec::new(),
            residue_seq: Vec::new(),
            is_hetatm: Vec::new(),
            bonds: Vec::new(),
        }
    }

    fn into_polydata(self) -> PolyData {
        let mut poly = PolyData::new();
        let n = self.points.len();
        poly.points = self.points;
        poly.verts = (0..n as u32).collect();
        poly.lines = self.bonds;
        poly.point_data.add_array(DataArray::new(
            "AtomicNumber",
            1,
            ArrayData::I32(self.atomic_numbers),
        ));
        poly.point_data
            .add_array(DataArray::new("Radius", 1, ArrayData::F32(self.radii)));
        poly.point_data.add_array(DataArray::new(
            "Occupancy",
            1,
            ArrayData::F32(self.occupancies),
        ));
        poly.point_data.add_array(DataArray::new(
            "BFactor",
            1,
            ArrayData::F32(self.temp_factors),
        ));
        poly.point_data.add_array(DataArray::new(
            "ResidueSeq",
            1,
            ArrayData::I32(self.residue_seq),
        ));
        poly.point_data.add_array(DataArray::new(
            "IsHetAtm",
            1,
            ArrayData::U8(self.is_hetatm),
        ));
        poly
    }
}

fn parse_pdb_models(text: &str) -> Result<Vec<PolyData>> {
    let mut models: Vec<PdbModel> = Vec::new();
    let mut current = PdbModel::new();
    let mut global_bonds: Vec<(i32, i32)> = Vec::new();
    let mut saw_model = false;

    for line in text.lines() {
        let record = line.get(0..6).unwrap_or(line).trim();
        match record {
            "MODEL" => {
                if saw_model && !current.points.is_empty() {
                    models.push(current);
                    current = PdbModel::new();
                }
                saw_model = true;
            }
            "ENDMDL" => {
                if !current.points.is_empty() {
                    models.push(current);
                    current = PdbModel::new();
                }
            }
            "ATOM" | "HETATM" => {
                let x = parse_pdb_float(line, 30, 38)?;
                let y = parse_pdb_float(line, 38, 46)?;
                let z = parse_pdb_float(line, 46, 54)?;
                let serial = parse_pdb_slice(line, 6, 11)
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or((current.points.len() + 1) as i32);
                let res_seq = parse_pdb_slice(line, 22, 26)
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(0);
                let occ = parse_pdb_slice(line, 54, 60)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(1.0);
                let bfactor = parse_pdb_slice(line, 60, 66)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0);

                let elem = parse_pdb_slice(line, 76, 78)
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .or_else(|| parse_pdb_slice(line, 12, 16).map(|s| s.trim()))
                    .unwrap_or("C");
                let (z_num, rad) = element_properties(elem);

                let idx = current.points.len() as u32;
                current.serial_to_idx.insert(serial, idx);
                current.points.push([x, y, z]);
                current.atomic_numbers.push(z_num);
                current.radii.push(rad);
                current.occupancies.push(occ);
                current.temp_factors.push(bfactor);
                current.residue_seq.push(res_seq);
                current.is_hetatm.push(u8::from(record == "HETATM"));
            }
            "CONECT" => {
                let mut nums = line[6..].split_whitespace().filter_map(|s| s.parse::<i32>().ok());
                if let Some(src) = nums.next() {
                    for dst in nums {
                        if src < dst {
                            global_bonds.push((src, dst));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !current.points.is_empty() {
        models.push(current);
    }
    if models.is_empty() {
        return Err(IoError::Parse("PDB file contains no ATOM/HETATM records".into()));
    }

    for model in &mut models {
        for &(a, b) in &global_bonds {
            if let (Some(&ia), Some(&ib)) =
                (model.serial_to_idx.get(&a), model.serial_to_idx.get(&b))
            {
                model.bonds.push(vec![ia, ib]);
            }
        }
    }

    Ok(models.into_iter().map(PdbModel::into_polydata).collect())
}

fn parse_pdb_slice(line: &str, start: usize, end: usize) -> Option<&str> {
    if line.len() <= start {
        return None;
    }
    let clamped_end = end.min(line.len());
    line.get(start..clamped_end)
}

fn parse_pdb_float(line: &str, start: usize, end: usize) -> Result<f32> {
    let s = parse_pdb_slice(line, start, end)
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| IoError::Parse(format!("PDB record missing coordinate in cols {start}..{end}")))?;
    s.parse::<f32>()
        .map_err(|_| IoError::Parse(format!("Invalid PDB coordinate '{s}'")))
}

fn element_properties(symbol: &str) -> (i32, f32) {
    let s = symbol.trim_matches(|c: char| !c.is_ascii_alphabetic());
    let upper = s.to_ascii_uppercase();
    // Handle common 2-letter and 1-letter symbols
    match upper.as_str() {
        "H" | "D" => (1, 0.31),
        "HE" => (2, 0.28),
        "LI" => (3, 1.28),
        "BE" => (4, 0.96),
        "B" => (5, 0.84),
        "C" => (6, 0.76),
        "N" => (7, 0.71),
        "O" => (8, 0.66),
        "F" => (9, 0.57),
        "NE" => (10, 0.58),
        "NA" => (11, 1.66),
        "MG" => (12, 1.41),
        "AL" => (13, 1.21),
        "SI" => (14, 1.11),
        "P" => (15, 1.07),
        "S" => (16, 1.05),
        "CL" => (17, 1.02),
        "AR" => (18, 1.06),
        "K" => (19, 2.03),
        "CA" => (20, 1.76),
        "FE" => (26, 1.32),
        "CU" => (29, 1.32),
        "ZN" => (30, 1.22),
        "BR" => (35, 1.20),
        "I" => (53, 1.39),
        _ => {
            if let Ok(z) = symbol.trim().parse::<i32>() {
                (z, 0.75)
            } else if upper.starts_with('C') {
                (6, 0.76)
            } else if upper.starts_with('N') {
                (7, 0.71)
            } else if upper.starts_with('O') {
                (8, 0.66)
            } else if upper.starts_with('H') {
                (1, 0.31)
            } else if upper.starts_with('S') {
                (16, 1.05)
            } else if upper.starts_with('P') {
                (15, 1.07)
            } else {
                (0, 0.75)
            }
        }
    }
}
