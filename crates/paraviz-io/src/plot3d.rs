//! NASA PLOT3D structured multi-block CFD grid (`.xyz`, `.p3d`) and solution (`.q`) reader.
//!
//! Supports:
//! - ASCII and Binary (Fortran unformatted 4-byte record markers, Little or Big Endian)
//! - Single-grid and multi-grid (`ngrid`) files
//! - 2D and 3D structured grids, optional `IBlank` array
//! - Companion `.q` solution file auto-loading (`Density`, `Momentum`, `StagnationEnergy`, `Velocity`)

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, MultiBlockDataSet, StructuredGrid};
use std::fs;
use std::path::Path;

/// Reads a PLOT3D grid (`.xyz` / `.p3d`) and optional companion `.q` solution file.
pub fn read_plot3d(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let mut grids = if is_ascii_plot3d(&bytes) {
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| IoError::Parse(format!("Invalid UTF-8 in PLOT3D ASCII file: {e}")))?;
        parse_plot3d_xyz_ascii(text)?
    } else {
        parse_plot3d_xyz_binary(&bytes)?
    };

    // Look for companion `.q` solution file with same stem
    let q_path = path.with_extension("q");
    if q_path.is_file() && q_path != path
        && let Ok(q_bytes) = fs::read(&q_path) {
            let _ = attach_plot3d_q(&mut grids, &q_bytes);
        }

    if grids.len() == 1 {
        Ok(DataSet::StructuredGrid(grids.into_iter().next().unwrap()))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (i, g) in grids.into_iter().enumerate() {
            mb.add_block(format!("Block_{i}"), DataSet::StructuredGrid(g));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn is_ascii_plot3d(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(256)];
    !head.is_empty()
        && head
            .iter()
            .all(|&b| b.is_ascii_whitespace() || b.is_ascii_graphic())
}

fn parse_plot3d_xyz_ascii(text: &str) -> Result<Vec<StructuredGrid>> {
    let mut tokens = text.split_whitespace();
    let first: usize = tokens
        .next()
        .ok_or_else(|| IoError::Parse("Empty PLOT3D ASCII file".into()))?
        .parse()
        .map_err(|_| IoError::Parse("Invalid PLOT3D header integer".into()))?;
    let second: usize = tokens
        .next()
        .ok_or_else(|| IoError::Parse("Truncated PLOT3D header".into()))?
        .parse()
        .map_err(|_| IoError::Parse("Invalid PLOT3D second integer".into()))?;
    let third: usize = tokens
        .next()
        .ok_or_else(|| IoError::Parse("Truncated PLOT3D header".into()))?
        .parse()
        .map_err(|_| IoError::Parse("Invalid PLOT3D third integer".into()))?;

    let all_rest: Vec<&str> = tokens.collect();

    // Heuristic: single-grid starts with `ni nj nk`, multi-grid starts with `ngrid` followed by `ni nj nk` per block
    let (dims_list, coord_start) = if (1..=1024).contains(&first) && all_rest.len() > first * 3 {
        // Check if `first` is `ngrid`: `second, third, all_rest[0]` would be `ni, nj, nk` of grid 0
        let total_if_single = first * second * third;
        if all_rest.len() == total_if_single * 3 || all_rest.len() == total_if_single * 4 {
            (vec![[first, second, third]], 0usize)
        } else {
            let ngrid = first;
            let mut dims = Vec::with_capacity(ngrid);
            dims.push([second, third, all_rest[0].parse::<usize>().unwrap_or(1)]);
            let mut idx = 1usize;
            for _ in 1..ngrid {
                if idx + 2 >= all_rest.len() {
                    return Err(IoError::Parse("Truncated PLOT3D multi-grid dimensions".into()));
                }
                let ni = all_rest[idx].parse::<usize>().unwrap_or(1);
                let nj = all_rest[idx + 1].parse::<usize>().unwrap_or(1);
                let nk = all_rest[idx + 2].parse::<usize>().unwrap_or(1);
                dims.push([ni, nj, nk]);
                idx += 3;
            }
            (dims, idx)
        }
    } else {
        (vec![[first, second, third]], 0usize)
    };

    let mut cur = coord_start;
    let mut grids = Vec::with_capacity(dims_list.len());
    for [ni, nj, nk] in dims_list {
        let n_pts = ni * nj * nk;
        if cur + n_pts * 3 > all_rest.len() {
            return Err(IoError::Parse("Truncated PLOT3D ASCII coordinates".into()));
        }
        let mut xs = Vec::with_capacity(safe_prealloc_capacity(n_pts, 4, MAX_ALLOC_BYTES)?);
        let mut ys = Vec::with_capacity(safe_prealloc_capacity(n_pts, 4, MAX_ALLOC_BYTES)?);
        let mut zs = Vec::with_capacity(safe_prealloc_capacity(n_pts, 4, MAX_ALLOC_BYTES)?);
        for _ in 0..n_pts {
            xs.push(all_rest[cur].parse::<f32>().unwrap_or(0.0));
            cur += 1;
        }
        for _ in 0..n_pts {
            ys.push(all_rest[cur].parse::<f32>().unwrap_or(0.0));
            cur += 1;
        }
        for _ in 0..n_pts {
            zs.push(all_rest[cur].parse::<f32>().unwrap_or(0.0));
            cur += 1;
        }
        let points: Vec<[f32; 3]> = (0..n_pts).map(|i| [xs[i], ys[i], zs[i]]).collect();
        let mut g = StructuredGrid {
            dimensions: [ni, nj, nk],
            points,
            ..Default::default()
        };
        if cur + n_pts <= all_rest.len() && all_rest[cur].parse::<i32>().is_ok() {
            let iblank: Vec<i32> = (0..n_pts)
                .map(|i| all_rest[cur + i].parse::<i32>().unwrap_or(1))
                .collect();
            cur += n_pts;
            g.point_data
                .add_array(DataArray::new("IBlank", 1, ArrayData::I32(iblank)));
        }
        grids.push(g);
    }
    Ok(grids)
}

fn parse_plot3d_xyz_binary(bytes: &[u8]) -> Result<Vec<StructuredGrid>> {
    if bytes.len() < 12 {
        return Err(IoError::Parse("PLOT3D binary file too short".into()));
    }
    // Detect Fortran 4-byte record marker & endianness
    let le_rec = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let be_rec = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;

    let (records, big_endian) = if le_rec > 0 && le_rec < bytes.len() && 4 + le_rec + 4 <= bytes.len()
        && u32::from_le_bytes(bytes[4 + le_rec..8 + le_rec].try_into().unwrap()) as usize == le_rec
    {
        (split_fortran_records(bytes, false)?, false)
    } else if be_rec > 0 && be_rec < bytes.len() && 4 + be_rec + 4 <= bytes.len()
        && u32::from_be_bytes(bytes[4 + be_rec..8 + be_rec].try_into().unwrap()) as usize == be_rec
    {
        (split_fortran_records(bytes, true)?, true)
    } else {
        return Err(IoError::Parse(
            "PLOT3D binary file missing valid Fortran unformatted record markers".into(),
        ));
    };

    if records.is_empty() {
        return Err(IoError::Parse("PLOT3D binary file has no records".into()));
    }

    let (dims_list, next_rec) = if records[0].len() == 4 {
        let ngrid = read_i32(records[0], 0, big_endian) as usize;
        if records.len() < 2 || records[1].len() < ngrid * 12 {
            return Err(IoError::Parse("Truncated PLOT3D multi-grid dimension record".into()));
        }
        let mut dims = Vec::with_capacity(ngrid);
        for i in 0..ngrid {
            let ni = read_i32(records[1], i * 12, big_endian) as usize;
            let nj = read_i32(records[1], i * 12 + 4, big_endian) as usize;
            let nk = read_i32(records[1], i * 12 + 8, big_endian) as usize;
            dims.push([ni, nj, nk]);
        }
        (dims, 2usize)
    } else if records[0].len() == 12 {
        let ni = read_i32(records[0], 0, big_endian) as usize;
        let nj = read_i32(records[0], 4, big_endian) as usize;
        let nk = read_i32(records[0], 8, big_endian) as usize;
        (vec![[ni, nj, nk]], 1usize)
    } else if records[0].len() == 8 {
        let ni = read_i32(records[0], 0, big_endian) as usize;
        let nj = read_i32(records[0], 4, big_endian) as usize;
        (vec![[ni, nj, 1]], 1usize)
    } else {
        return Err(IoError::Parse(format!(
            "Unexpected PLOT3D header record length {}",
            records[0].len()
        )));
    };

    let mut grids = Vec::with_capacity(dims_list.len());
    for (gi, &[ni, nj, nk]) in dims_list.iter().enumerate() {
        let rec = records.get(next_rec + gi).ok_or_else(|| {
            IoError::Parse(format!("Missing PLOT3D coordinate record for grid {gi}"))
        })?;
        let n_pts = ni * nj * nk;
        let has_f64 = rec.len() >= n_pts * 24;
        let elem_sz = if has_f64 { 8 } else { 4 };
        if rec.len() < n_pts * 3 * elem_sz {
            return Err(IoError::Parse("Truncated PLOT3D coordinate record".into()));
        }
        let mut points = Vec::with_capacity(safe_prealloc_capacity(n_pts, 12, MAX_ALLOC_BYTES)?);
        for i in 0..n_pts {
            let x = read_float(rec, i * elem_sz, has_f64, big_endian);
            let y = read_float(rec, (n_pts + i) * elem_sz, has_f64, big_endian);
            let z = read_float(rec, (2 * n_pts + i) * elem_sz, has_f64, big_endian);
            points.push([x, y, z]);
        }
        let mut g = StructuredGrid {
            dimensions: [ni, nj, nk],
            points,
            ..Default::default()
        };
        if rec.len() >= n_pts * 3 * elem_sz + n_pts * 4 {
            let base = n_pts * 3 * elem_sz;
            let iblank: Vec<i32> = (0..n_pts)
                .map(|i| read_i32(rec, base + i * 4, big_endian))
                .collect();
            g.point_data
                .add_array(DataArray::new("IBlank", 1, ArrayData::I32(iblank)));
        }
        grids.push(g);
    }
    Ok(grids)
}

fn attach_plot3d_q(grids: &mut [StructuredGrid], q_bytes: &[u8]) -> Result<()> {
    if is_ascii_plot3d(q_bytes) {
        return Ok(());
    }
    let le_rec = if q_bytes.len() >= 4 {
        u32::from_le_bytes(q_bytes[0..4].try_into().unwrap()) as usize
    } else {
        return Ok(());
    };
    let big_endian = !(le_rec > 0
        && le_rec < q_bytes.len()
        && 4 + le_rec + 4 <= q_bytes.len()
        && u32::from_le_bytes(q_bytes[4 + le_rec..8 + le_rec].try_into().unwrap()) as usize
            == le_rec);
    let records = split_fortran_records(q_bytes, big_endian)?;
    let data_start = if records.first().map(|r| r.len()) == Some(4) {
        2usize
    } else {
        1usize
    };
    for (gi, grid) in grids.iter_mut().enumerate() {
        // Each grid in .q has a 4-float condition record followed by the Q field record
        let q_rec_idx = data_start + gi * 2 + 1;
        let Some(rec) = records.get(q_rec_idx) else {
            continue;
        };
        let n_pts = grid.points.len();
        let has_f64 = rec.len() >= n_pts * 5 * 8;
        let elem_sz = if has_f64 { 8 } else { 4 };
        if rec.len() < n_pts * 5 * elem_sz {
            continue;
        }
        let mut rho = Vec::with_capacity(n_pts);
        let mut mom = Vec::with_capacity(n_pts * 3);
        let mut vel = Vec::with_capacity(n_pts * 3);
        let mut energy = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            let r = read_float(rec, i * elem_sz, has_f64, big_endian);
            let mx = read_float(rec, (n_pts + i) * elem_sz, has_f64, big_endian);
            let my = read_float(rec, (2 * n_pts + i) * elem_sz, has_f64, big_endian);
            let mz = read_float(rec, (3 * n_pts + i) * elem_sz, has_f64, big_endian);
            let e = read_float(rec, (4 * n_pts + i) * elem_sz, has_f64, big_endian);
            rho.push(r);
            mom.extend_from_slice(&[mx, my, mz]);
            let inv_r = if r.abs() > 1e-12 { 1.0 / r } else { 0.0 };
            vel.extend_from_slice(&[mx * inv_r, my * inv_r, mz * inv_r]);
            energy.push(e);
        }
        grid.point_data
            .add_array(DataArray::new("Density", 1, ArrayData::F32(rho)));
        grid.point_data
            .add_array(DataArray::new("Momentum", 3, ArrayData::F32(mom)));
        grid.point_data
            .add_array(DataArray::new("Velocity", 3, ArrayData::F32(vel)));
        grid.point_data
            .add_array(DataArray::new("StagnationEnergy", 1, ArrayData::F32(energy)));
    }
    Ok(())
}

fn split_fortran_records(bytes: &[u8], big_endian: bool) -> Result<Vec<&[u8]>> {
    let mut records = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let len = if big_endian {
            u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize
        };
        pos += 4;
        if pos + len + 4 > bytes.len() {
            return Err(IoError::Parse("Truncated Fortran record in PLOT3D file".into()));
        }
        records.push(&bytes[pos..pos + len]);
        pos += len + 4;
    }
    Ok(records)
}

fn read_i32(buf: &[u8], off: usize, big_endian: bool) -> i32 {
    let b: [u8; 4] = buf[off..off + 4].try_into().unwrap();
    if big_endian {
        i32::from_be_bytes(b)
    } else {
        i32::from_le_bytes(b)
    }
}

fn read_float(buf: &[u8], off: usize, is_f64: bool, big_endian: bool) -> f32 {
    if is_f64 {
        let b: [u8; 8] = buf[off..off + 8].try_into().unwrap();
        if big_endian {
            f64::from_be_bytes(b) as f32
        } else {
            f64::from_le_bytes(b) as f32
        }
    } else {
        let b: [u8; 4] = buf[off..off + 4].try_into().unwrap();
        if big_endian {
            f32::from_be_bytes(b)
        } else {
            f32::from_le_bytes(b)
        }
    }
}
