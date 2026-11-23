//! NetCDF CF / COARDS (`.nc`, `.cdf`) scientific grid reader.
//!
//! Supports NetCDF-3 Classic (`CDF\x01`), 64-bit Offset (`CDF\x02`), and NetCDF-4 (HDF5).
//! Maps 1D/2D/3D/4D variables with coordinate axes into [`RectilinearGrid`] or [`ImageData`],
//! with temporal step support when a `time` / `t` dimension is present.

use crate::error::{IoError, Result};
use crate::hdf5_common::{
    attr_as_string, open_hdf5, read_dataset_f32, read_dataset_f64, slice_axis0,
};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, ImageData, RectilinearGrid};
use std::fs;
use std::path::Path;

/// Inspects time steps in a NetCDF CF file (if a `time` / `t` dimension + coordinate variable exists).
pub fn inspect_netcdf_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    let path = path.as_ref();
    let bytes = read_prefix(path, 8)?;
    if bytes.starts_with(b"\x89HDF\r\n\x1a\n") {
        let file = open_hdf5(path)?;
        let root = file
            .root_group()
            .map_err(|e| IoError::Parse(format!("HDF5 root error: {e}")))?;
        for name in ["time", "TIME", "t", "T"] {
            if let Ok(ds) = root.dataset(name) {
                if ds.shape().len() == 1 {
                    return read_dataset_f64(&ds, None);
                }
            }
        }
        Ok(Vec::new())
    } else if bytes.starts_with(b"CDF\x01") || bytes.starts_with(b"CDF\x02") {
        let full = fs::read(path)?;
        let nc = parse_nc3(&full)?;
        for name in ["time", "TIME", "t", "T"] {
            if let Some(var) = nc.vars.iter().find(|v| v.name == name && v.dim_ids.len() == 1) {
                return nc3_read_var_f64(&full, &nc, var, None);
            }
        }
        Ok(Vec::new())
    } else {
        Ok(Vec::new())
    }
}

/// Reads the first time step of a NetCDF CF file into a [`DataSet`].
pub fn read_netcdf_cf(path: impl AsRef<Path>) -> Result<DataSet> {
    read_netcdf_cf_at_time(path, 0.0)
}

/// Reads the time step closest to `time` from a NetCDF CF file into a [`DataSet`].
pub fn read_netcdf_cf_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    let path = path.as_ref();
    let prefix = read_prefix(path, 8)?;
    if prefix.starts_with(b"\x89HDF\r\n\x1a\n") {
        read_nc4_hdf5(path, time)
    } else if prefix.starts_with(b"CDF\x01") || prefix.starts_with(b"CDF\x02") {
        let full = fs::read(path)?;
        read_nc3_cf(&full, time)
    } else {
        Err(IoError::UnsupportedFormat(format!(
            "'{}' is neither NetCDF-3 (CDF) nor NetCDF-4 (HDF5)",
            path.display()
        )))
    }
}

fn read_prefix(path: &Path, n: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf)?;
    buf.truncate(got);
    Ok(buf)
}

// ================================================================================================
// NetCDF-4 (HDF5) CF Reader
// ================================================================================================

fn read_nc4_hdf5(path: &Path, target_time: f64) -> Result<DataSet> {
    let file = open_hdf5(path)?;
    let root = file
        .root_group()
        .map_err(|e| IoError::Parse(format!("HDF5 root error: {e}")))?;
    let datasets = root
        .datasets()
        .map_err(|e| IoError::Parse(format!("Failed to list NetCDF-4 datasets: {e}")))?;

    let time_vals: Vec<f64> = ["time", "TIME", "t", "T"]
        .iter()
        .find_map(|&n| root.dataset(n).ok())
        .and_then(|ds| read_dataset_f64(&ds, None).ok())
        .unwrap_or_default();
    let time_idx = closest_time_idx(&time_vals, target_time);

    let x_axis = find_hdf5_axis(&root, &["x", "X", "lon", "longitude", "xc"]);
    let y_axis = find_hdf5_axis(&root, &["y", "Y", "lat", "latitude", "yc"]);
    let z_axis = find_hdf5_axis(&root, &["z", "Z", "lev", "level", "depth", "altitude", "zc"]);

    // Pick primary 2D/3D/4D data variables
    let mut arrays: Vec<DataArray> = Vec::new();
    let mut dims_zyx: Option<[usize; 3]> = None;

    for ds in &datasets {
        let name = ds.name();
        if is_coord_name(name) {
            continue;
        }
        let shape = ds.shape();
        if shape.len() < 2 || shape.len() > 4 {
            continue;
        }
        let has_time = shape.len() == 4 || (!time_vals.is_empty() && shape[0] as usize == time_vals.len());
        let (spatial_shape, slice) = if has_time && shape.len() >= 3 {
            let ti = time_idx.min((shape[0] as usize).saturating_sub(1)) as u64;
            let s = slice_axis0(&shape, ti, ti + 1)?;
            (&shape[1..], Some(s))
        } else {
            (&shape[..], None)
        };

        let cur_zyx = match spatial_shape.len() {
            2 => [1usize, spatial_shape[0] as usize, spatial_shape[1] as usize],
            3 => [
                spatial_shape[0] as usize,
                spatial_shape[1] as usize,
                spatial_shape[2] as usize,
            ],
            _ => continue,
        };

        if let Some(prev) = dims_zyx {
            if prev != cur_zyx {
                continue;
            }
        } else {
            dims_zyx = Some(cur_zyx);
        }

        if let Ok(vals) = read_dataset_f32(ds, slice.as_ref()) {
            arrays.push(DataArray::new(name.to_string(), 1, ArrayData::F32(vals)));
        }
    }

    let [nz, ny, nx] = dims_zyx.ok_or_else(|| {
        IoError::UnsupportedFormat("NetCDF-4 file contains no 2D/3D/4D gridded variables".into())
    })?;

    build_cf_grid([nx, ny, nz], x_axis, y_axis, z_axis, arrays)
}

fn find_hdf5_axis(root: &hdf5_reader::group::Group, names: &[&str]) -> Option<Vec<f64>> {
    for &n in names {
        if let Ok(ds) = root.dataset(n) {
            if ds.shape().len() == 1 {
                if let Ok(v) = read_dataset_f64(&ds, None) {
                    if !v.is_empty() {
                        return Some(v);
                    }
                }
            }
        }
    }
    None
}

// ================================================================================================
// NetCDF-3 Classic / 64-bit Offset CF Reader
// ================================================================================================

struct Nc3Dim {
    _name: String,
    len: usize,
}

struct Nc3Var {
    name: String,
    dim_ids: Vec<usize>,
    nc_type: u32,
    vsize: usize,
    begin: usize,
}

struct Nc3Meta {
    is_64bit: bool,
    numrecs: usize,
    dims: Vec<Nc3Dim>,
    vars: Vec<Nc3Var>,
    rec_stride: usize,
}

fn read_nc3_cf(bytes: &[u8], target_time: f64) -> Result<DataSet> {
    let nc = parse_nc3(bytes)?;

    let time_vals: Vec<f64> = ["time", "TIME", "t", "T"]
        .iter()
        .find_map(|&n| nc.vars.iter().find(|v| v.name == n && v.dim_ids.len() == 1))
        .and_then(|v| nc3_read_var_f64(bytes, &nc, v, None).ok())
        .unwrap_or_default();
    let time_idx = closest_time_idx(&time_vals, target_time);

    let x_axis = find_nc3_axis(bytes, &nc, &["x", "X", "lon", "longitude", "xc"]);
    let y_axis = find_nc3_axis(bytes, &nc, &["y", "Y", "lat", "latitude", "yc"]);
    let z_axis = find_nc3_axis(bytes, &nc, &["z", "Z", "lev", "level", "depth", "altitude", "zc"]);

    let mut arrays: Vec<DataArray> = Vec::new();
    let mut dims_zyx: Option<[usize; 3]> = None;

    for var in &nc.vars {
        if is_coord_name(&var.name) || var.dim_ids.len() < 2 || var.dim_ids.len() > 4 {
            continue;
        }
        let shape: Vec<usize> = var
            .dim_ids
            .iter()
            .map(|&did| {
                let d = &nc.dims[did];
                if d.len == 0 { nc.numrecs.max(1) } else { d.len }
            })
            .collect();

        let is_rec_var = nc.dims[var.dim_ids[0]].len == 0;
        let has_time = shape.len() == 4
            || is_rec_var
            || (!time_vals.is_empty() && shape[0] == time_vals.len());

        let (spatial_shape, slice_rec) = if has_time && shape.len() >= 3 {
            let ti = time_idx.min(shape[0].saturating_sub(1));
            (&shape[1..], Some(ti))
        } else {
            (&shape[..], None)
        };

        let cur_zyx = match spatial_shape.len() {
            2 => [1usize, spatial_shape[0], spatial_shape[1]],
            3 => [spatial_shape[0], spatial_shape[1], spatial_shape[2]],
            _ => continue,
        };

        if let Some(prev) = dims_zyx {
            if prev != cur_zyx {
                continue;
            }
        } else {
            dims_zyx = Some(cur_zyx);
        }

        if let Ok(vals_f64) = nc3_read_var_f64(bytes, &nc, var, slice_rec) {
            let vals_f32: Vec<f32> = vals_f64.into_iter().map(|v| v as f32).collect();
            arrays.push(DataArray::new(
                var.name.clone(),
                1,
                ArrayData::F32(vals_f32),
            ));
        }
    }

    let [nz, ny, nx] = dims_zyx.ok_or_else(|| {
        IoError::UnsupportedFormat("NetCDF-3 file contains no 2D/3D/4D gridded variables".into())
    })?;

    build_cf_grid([nx, ny, nz], x_axis, y_axis, z_axis, arrays)
}

fn find_nc3_axis(bytes: &[u8], nc: &Nc3Meta, names: &[&str]) -> Option<Vec<f64>> {
    for &n in names {
        if let Some(var) = nc.vars.iter().find(|v| v.name == n && v.dim_ids.len() == 1) {
            if let Ok(v) = nc3_read_var_f64(bytes, nc, var, None) {
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn parse_nc3(bytes: &[u8]) -> Result<Nc3Meta> {
    if bytes.len() < 8 {
        return Err(IoError::Parse("NetCDF-3 file too short".into()));
    }
    let is_64bit = match &bytes[0..4] {
        b"CDF\x01" => false,
        b"CDF\x02" => true,
        _ => return Err(IoError::Parse("Invalid NetCDF-3 magic".into())),
    };
    let mut pos = 4usize;
    let numrecs = read_be_u32(bytes, &mut pos)? as usize;

    let mut dims = Vec::new();
    let tag = read_be_u32(bytes, &mut pos)?;
    if tag == 10 {
        // NC_DIMENSION
        let count = read_be_u32(bytes, &mut pos)? as usize;
        for _ in 0..count {
            let name = read_nc3_string(bytes, &mut pos)?;
            let len = read_be_u32(bytes, &mut pos)? as usize;
            dims.push(Nc3Dim { _name: name, len });
        }
    } else if tag != 0 {
        return Err(IoError::Parse("Invalid NetCDF-3 dim_list tag".into()));
    }

    // Global attributes
    let gatt_tag = read_be_u32(bytes, &mut pos)?;
    if gatt_tag == 12 {
        skip_nc3_atts(bytes, &mut pos)?;
    }

    // Variables
    let mut vars = Vec::new();
    let var_tag = read_be_u32(bytes, &mut pos)?;
    if var_tag == 11 {
        let count = read_be_u32(bytes, &mut pos)? as usize;
        for _ in 0..count {
            let name = read_nc3_string(bytes, &mut pos)?;
            let ndims = read_be_u32(bytes, &mut pos)? as usize;
            let mut dim_ids = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                dim_ids.push(read_be_u32(bytes, &mut pos)? as usize);
            }
            let vatt_tag = read_be_u32(bytes, &mut pos)?;
            if vatt_tag == 12 {
                skip_nc3_atts(bytes, &mut pos)?;
            }
            let nc_type = read_be_u32(bytes, &mut pos)?;
            let vsize = read_be_u32(bytes, &mut pos)? as usize;
            let begin = if is_64bit {
                read_be_u64(bytes, &mut pos)? as usize
            } else {
                read_be_u32(bytes, &mut pos)? as usize
            };
            vars.push(Nc3Var {
                name,
                dim_ids,
                nc_type,
                vsize,
                begin,
            });
        }
    }

    let rec_stride: usize = vars
        .iter()
        .filter(|v| !v.dim_ids.is_empty() && dims[v.dim_ids[0]].len == 0)
        .map(|v| v.vsize)
        .sum();

    Ok(Nc3Meta {
        is_64bit,
        numrecs,
        dims,
        vars,
        rec_stride,
    })
}

fn nc3_read_var_f64(
    bytes: &[u8],
    nc: &Nc3Meta,
    var: &Nc3Var,
    rec_idx: Option<usize>,
) -> Result<Vec<f64>> {
    let elem_sz = match var.nc_type {
        1 | 2 => 1usize,
        3 => 2usize,
        4 | 5 => 4usize,
        6 => 8usize,
        _ => return Err(IoError::UnsupportedFormat("Unsupported NetCDF-3 type".into())),
    };
    let is_rec = !var.dim_ids.is_empty() && nc.dims[var.dim_ids[0]].len == 0;
    let count: usize = var
        .dim_ids
        .iter()
        .enumerate()
        .map(|(i, &did)| {
            let l = nc.dims[did].len;
            if l == 0 {
                if i == 0 && rec_idx.is_some() { 1 } else { nc.numrecs.max(1) }
            } else {
                l
            }
        })
        .product();

    let mut out = Vec::with_capacity(safe_prealloc_capacity(count, 8, MAX_ALLOC_BYTES)?);
    if is_rec && rec_idx.is_none() {
        let per_rec = count / nc.numrecs.max(1);
        for r in 0..nc.numrecs.max(1) {
            let base = var.begin + r * nc.rec_stride;
            read_nc3_slice(bytes, base, per_rec, var.nc_type, elem_sz, &mut out)?;
        }
    } else if is_rec {
        let r = rec_idx.unwrap_or(0);
        let base = var.begin + r * nc.rec_stride;
        read_nc3_slice(bytes, base, count, var.nc_type, elem_sz, &mut out)?;
    } else {
        let base = if let Some(r) = rec_idx {
            let per_step = var.vsize / nc.dims[var.dim_ids[0]].len.max(1);
            var.begin + r * per_step
        } else {
            var.begin
        };
        let n_read = if rec_idx.is_some() {
            count / nc.dims[var.dim_ids[0]].len.max(1)
        } else {
            count
        };
        read_nc3_slice(bytes, base, n_read, var.nc_type, elem_sz, &mut out)?;
    }
    let _ = nc.is_64bit;
    Ok(out)
}

fn read_nc3_slice(
    bytes: &[u8],
    base: usize,
    count: usize,
    nc_type: u32,
    elem_sz: usize,
    out: &mut Vec<f64>,
) -> Result<()> {
    if base + count * elem_sz > bytes.len() {
        return Err(IoError::Parse("Truncated NetCDF-3 variable data".into()));
    }
    for i in 0..count {
        let p = base + i * elem_sz;
        let v = match nc_type {
            1 => (bytes[p] as i8) as f64,
            2 => bytes[p] as f64,
            3 => i16::from_be_bytes([bytes[p], bytes[p + 1]]) as f64,
            4 => i32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64,
            5 => f32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64,
            6 => f64::from_be_bytes(bytes[p..p + 8].try_into().unwrap()),
            _ => 0.0,
        };
        out.push(v);
    }
    Ok(())
}

fn skip_nc3_atts(bytes: &[u8], pos: &mut usize) -> Result<()> {
    let count = read_be_u32(bytes, pos)? as usize;
    for _ in 0..count {
        let _name = read_nc3_string(bytes, pos)?;
        let nc_type = read_be_u32(bytes, pos)?;
        let nelems = read_be_u32(bytes, pos)? as usize;
        let elem_sz = match nc_type {
            1 | 2 => 1,
            3 => 2,
            4 | 5 => 4,
            6 => 8,
            _ => 1,
        };
        let padded = ((nelems * elem_sz) + 3) & !3;
        *pos += padded;
    }
    Ok(())
}

fn read_nc3_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_be_u32(bytes, pos)? as usize;
    let padded = (len + 3) & !3;
    if *pos + padded > bytes.len() {
        return Err(IoError::Parse("Truncated NetCDF-3 string".into()));
    }
    let s = String::from_utf8_lossy(&bytes[*pos..*pos + len]).into_owned();
    *pos += padded;
    Ok(s)
}

fn read_be_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() {
        return Err(IoError::Parse("Unexpected EOF in NetCDF-3 header".into()));
    }
    let v = u32::from_be_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_be_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos + 8 > bytes.len() {
        return Err(IoError::Parse("Unexpected EOF in NetCDF-3 64-bit header".into()));
    }
    let v = u64::from_be_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

// ================================================================================================
// Shared Helpers
// ================================================================================================

fn is_coord_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "x" | "y" | "z" | "t" | "time" | "lon" | "lat" | "longitude" | "latitude" | "lev"
            | "level" | "depth" | "altitude" | "xc" | "yc" | "zc"
    )
}

fn closest_time_idx(times: &[f64], target: f64) -> usize {
    if times.is_empty() {
        return 0;
    }
    times
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - target).abs().total_cmp(&(*b - target).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn build_cf_grid(
    [nx, ny, nz]: [usize; 3],
    x_axis: Option<Vec<f64>>,
    y_axis: Option<Vec<f64>>,
    z_axis: Option<Vec<f64>>,
    arrays: Vec<DataArray>,
) -> Result<DataSet> {
    let x_ok = x_axis.as_ref().is_some_and(|v| v.len() == nx);
    let y_ok = y_axis.as_ref().is_some_and(|v| v.len() == ny);
    let z_ok = z_axis.as_ref().is_some_and(|v| v.len() == nz);

    if x_ok || y_ok || z_ok {
        let xc = if x_ok {
            x_axis.unwrap().into_iter().map(|v| v as f32).collect()
        } else {
            (0..nx).map(|i| i as f32).collect()
        };
        let yc = if y_ok {
            y_axis.unwrap().into_iter().map(|v| v as f32).collect()
        } else {
            (0..ny).map(|i| i as f32).collect()
        };
        let zc = if z_ok {
            z_axis.unwrap().into_iter().map(|v| v as f32).collect()
        } else {
            (0..nz).map(|i| i as f32).collect()
        };
        let mut rg = RectilinearGrid {
            x_coords: xc,
            y_coords: yc,
            z_coords: zc,
            ..Default::default()
        };
        for arr in arrays {
            rg.point_data.add_array(arr);
        }
        Ok(DataSet::RectilinearGrid(rg))
    } else {
        let mut img = ImageData::new([nx, ny, nz], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        for arr in arrays {
            img.point_data.add_array(arr);
        }
        Ok(DataSet::ImageData(img))
    }
}

#[allow(dead_code)]
fn _unused(_: &hdf5_reader::Attribute) {
    let _ = attr_as_string;
}
