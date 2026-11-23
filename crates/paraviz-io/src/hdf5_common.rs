//! Shared safe HDF5 reading helpers backed by `hdf5-reader` and `crate::limits`.

use crate::error::{IoError, Result};
use crate::limits::{check_alloc_bytes, safe_prealloc_capacity, MAX_ALLOC_BYTES};
use hdf5_reader::{
    group::Group, Attribute, Dataset, Datatype, Hdf5File, OpenOptions, SliceInfo, SliceInfoElem,
};
use paraviz_core::{ArrayData, DataArray};
use std::path::Path;

/// Opens an HDF5 file with `MAX_ALLOC_BYTES` read budget enforced.
pub(crate) fn open_hdf5(path: &Path) -> Result<Hdf5File> {
    let opts = OpenOptions::default().with_max_read_bytes(MAX_ALLOC_BYTES);
    Hdf5File::open_with_options(path, opts).map_err(|e| {
        IoError::Parse(format!("Failed to open HDF5 file '{}': {e}", path.display()))
    })
}

/// Opens an in-memory HDF5 byte slice with `MAX_ALLOC_BYTES` read budget enforced.
#[allow(dead_code)]
pub(crate) fn open_hdf5_bytes(bytes: &[u8]) -> Result<Hdf5File> {
    let opts = OpenOptions::default().with_max_read_bytes(MAX_ALLOC_BYTES);
    Hdf5File::from_bytes_with_options(bytes, opts)
        .map_err(|e| IoError::Parse(format!("Failed to parse HDF5 bytes: {e}")))
}

/// Extracts a trimmed string from an HDF5 attribute, supporting fixed/vlen strings and byte arrays.
pub(crate) fn attr_as_string(attr: &Attribute) -> Option<String> {
    if let Ok(s) = attr.read_string() {
        return Some(trim_nul(&s));
    }
    match &attr.datatype {
        Datatype::FixedPoint { size: 1, signed: false, .. } => {
            let bytes = attr.read_1d::<u8>().ok()?;
            Some(decode_nul_terminated_bytes(&bytes))
        }
        Datatype::FixedPoint { size: 1, signed: true, .. } => {
            let bytes = attr.read_1d::<i8>().ok()?;
            let ubytes: Vec<u8> = bytes.into_iter().map(|b| b as u8).collect();
            Some(decode_nul_terminated_bytes(&ubytes))
        }
        _ => None,
    }
}

/// Reads a group attribute as a string if present.
pub(crate) fn group_attr_string(group: &Group, name: &str) -> Option<String> {
    group
        .attribute(name)
        .ok()
        .and_then(|a| attr_as_string(&a))
}

/// Reads a numeric attribute (scalar or 1D) into `Vec<i64>`.
pub(crate) fn attr_as_i64_vec(attr: &Attribute) -> Result<Vec<i64>> {
    match &attr.datatype {
        Datatype::FixedPoint { size: 1, signed: true, .. } => {
            Ok(attr_vec_i8(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 1, signed: false, .. } => {
            Ok(attr_vec_u8(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 2, signed: true, .. } => {
            Ok(attr_vec_t::<i16>(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 2, signed: false, .. } => {
            Ok(attr_vec_t::<u16>(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 4, signed: true, .. } => {
            Ok(attr_vec_t::<i32>(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 4, signed: false, .. } => {
            Ok(attr_vec_t::<u32>(attr)?.into_iter().map(|v| v as i64).collect())
        }
        Datatype::FixedPoint { size: 8, signed: true, .. } => attr_vec_t::<i64>(attr),
        Datatype::FixedPoint { size: 8, signed: false, .. } => Ok(attr_vec_t::<u64>(attr)?
            .into_iter()
            .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
            .collect()),
        other => Err(IoError::UnsupportedFormat(format!(
            "Attribute '{}' has non-integer datatype {other:?}",
            attr.name
        ))),
    }
}

/// Reads a numeric attribute (scalar or 1D) into `Vec<f64>`.
pub(crate) fn attr_as_f64_vec(attr: &Attribute) -> Result<Vec<f64>> {
    match &attr.datatype {
        Datatype::FloatingPoint { size: 4, .. } => {
            Ok(attr_vec_t::<f32>(attr)?.into_iter().map(|v| v as f64).collect())
        }
        Datatype::FloatingPoint { size: 8, .. } => attr_vec_t::<f64>(attr),
        Datatype::FixedPoint { .. } => Ok(attr_as_i64_vec(attr)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        other => Err(IoError::UnsupportedFormat(format!(
            "Attribute '{}' has non-numeric datatype {other:?}",
            attr.name
        ))),
    }
}

fn attr_vec_i8(attr: &Attribute) -> Result<Vec<i8>> {
    attr_vec_t::<i8>(attr)
}

fn attr_vec_u8(attr: &Attribute) -> Result<Vec<u8>> {
    attr_vec_t::<u8>(attr)
}

fn attr_vec_t<T: hdf5_reader::H5Type>(attr: &Attribute) -> Result<Vec<T>> {
    if attr.shape.is_empty() {
        let v = attr
            .read_scalar::<T>()
            .map_err(|e| IoError::Parse(format!("HDF5 attribute '{}' read failed: {e}", attr.name)))?;
        Ok(vec![v])
    } else {
        attr.read_1d::<T>()
            .map_err(|e| IoError::Parse(format!("HDF5 attribute '{}' read failed: {e}", attr.name)))
    }
}

pub(crate) fn trim_nul(s: &str) -> String {
    let trimmed = match s.find('\0') {
        Some(idx) => &s[..idx],
        None => s,
    };
    trimmed.trim().to_string()
}

pub(crate) fn decode_nul_terminated_bytes(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

/// Computes total element count of an HDF5 dataset shape with overflow checking.
pub(crate) fn shape_element_count(shape: &[u64]) -> Result<usize> {
    let mut total = 1usize;
    for &dim in shape {
        let d = usize::try_from(dim)
            .map_err(|_| IoError::InvalidData(format!("HDF5 dimension {dim} overflows usize")))?;
        total = total
            .checked_mul(d)
            .ok_or_else(|| IoError::InvalidData("HDF5 dataset element count overflows usize".into()))?;
    }
    Ok(total)
}

/// Computes total element count that a slice selection will produce.
pub(crate) fn selection_element_count(shape: &[u64], slice: Option<&SliceInfo>) -> Result<usize> {
    let Some(sel) = slice else {
        return shape_element_count(shape);
    };
    if sel.selections.len() != shape.len() {
        return Err(IoError::InvalidData(format!(
            "HDF5 slice rank {} does not match dataset rank {}",
            sel.selections.len(),
            shape.len()
        )));
    }
    let mut total = 1usize;
    for (dim_idx, (elem, &dim_len)) in sel.selections.iter().zip(shape.iter()).enumerate() {
        let count_u64 = match elem {
            SliceInfoElem::Index(idx) => {
                if *idx >= dim_len {
                    return Err(IoError::InvalidData(format!(
                        "HDF5 slice index {idx} out of bounds for dim {dim_idx} (len {dim_len})"
                    )));
                }
                1
            }
            SliceInfoElem::Slice { start, end, step } => {
                if *step == 0 {
                    return Err(IoError::InvalidData("HDF5 slice step cannot be 0".into()));
                }
                let clamped_end = (*end).min(dim_len);
                if *start > clamped_end {
                    return Err(IoError::InvalidData(format!(
                        "HDF5 slice start {start} exceeds end {clamped_end} for dim {dim_idx}"
                    )));
                }
                (clamped_end - *start).div_ceil(*step)
            }
        };
        let count = usize::try_from(count_u64).map_err(|_| {
            IoError::InvalidData(format!("HDF5 slice element count {count_u64} overflows usize"))
        })?;
        total = total
            .checked_mul(count)
            .ok_or_else(|| IoError::InvalidData("HDF5 slice element count overflows usize".into()))?;
    }
    Ok(total)
}

/// Builds a hyperslab slicing `[start..end]` along axis 0 and full extent along remaining axes.
pub(crate) fn slice_axis0(shape: &[u64], start: u64, end: u64) -> Result<SliceInfo> {
    if shape.is_empty() {
        return Err(IoError::InvalidData(
            "Cannot slice axis 0 of a scalar HDF5 dataset".into(),
        ));
    }
    if start > end || end > shape[0] {
        return Err(IoError::InvalidData(format!(
            "HDF5 axis-0 slice [{start}..{end}] out of bounds for extent {}",
            shape[0]
        )));
    }
    let mut selections = Vec::with_capacity(shape.len());
    selections.push(SliceInfoElem::Slice {
        start,
        end,
        step: 1,
    });
    for &d in &shape[1..] {
        selections.push(SliceInfoElem::Slice {
            start: 0,
            end: d,
            step: 1,
        });
    }
    Ok(SliceInfo { selections })
}

/// Builds a hyperslab indexing a single element `idx` along axis 0 (reducing rank by 1).
pub(crate) fn index_axis0(shape: &[u64], idx: u64) -> Result<SliceInfo> {
    if shape.is_empty() {
        return Err(IoError::InvalidData(
            "Cannot index axis 0 of a scalar HDF5 dataset".into(),
        ));
    }
    if idx >= shape[0] {
        return Err(IoError::InvalidData(format!(
            "HDF5 axis-0 index {idx} out of bounds for extent {}",
            shape[0]
        )));
    }
    let mut selections = Vec::with_capacity(shape.len());
    selections.push(SliceInfoElem::Index(idx));
    for &d in &shape[1..] {
        selections.push(SliceInfoElem::Slice {
            start: 0,
            end: d,
            step: 1,
        });
    }
    Ok(SliceInfo { selections })
}

fn read_typed_vec<T: hdf5_reader::H5Type + Copy>(
    ds: &Dataset,
    slice: Option<&SliceInfo>,
) -> Result<Vec<T>> {
    let count = selection_element_count(ds.shape(), slice)?;
    check_alloc_bytes(count, std::mem::size_of::<T>(), MAX_ALLOC_BYTES)?;
    let arr = match slice {
        Some(sel) => ds.read_slice::<T>(sel),
        None => ds.read_array::<T>(),
    }
    .map_err(|e| IoError::Parse(format!("HDF5 dataset '{}' read failed: {e}", ds.name())))?;
    let (vec, offset) = arr.into_raw_vec_and_offset();
    if offset == Some(0) {
        Ok(vec)
    } else {
        Ok(vec.into_iter().skip(offset.unwrap_or(0)).take(count).collect())
    }
}

/// Reads any numeric HDF5 dataset (or slice) into `Vec<f64>`.
pub(crate) fn read_dataset_f64(ds: &Dataset, slice: Option<&SliceInfo>) -> Result<Vec<f64>> {
    match ds.dtype() {
        Datatype::FloatingPoint { size: 8, .. } => read_typed_vec::<f64>(ds, slice),
        Datatype::FloatingPoint { size: 4, .. } => Ok(read_typed_vec::<f32>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 8, signed: true, .. } => Ok(read_typed_vec::<i64>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 8, signed: false, .. } => Ok(read_typed_vec::<u64>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 4, signed: true, .. } => Ok(read_typed_vec::<i32>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 4, signed: false, .. } => Ok(read_typed_vec::<u32>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 2, signed: true, .. } => Ok(read_typed_vec::<i16>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 2, signed: false, .. } => Ok(read_typed_vec::<u16>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 1, signed: true, .. } => Ok(read_typed_vec::<i8>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Datatype::FixedPoint { size: 1, signed: false, .. } => Ok(read_typed_vec::<u8>(ds, slice)?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        other => Err(IoError::UnsupportedFormat(format!(
            "HDF5 dataset '{}' has unsupported numeric datatype {other:?}",
            ds.name()
        ))),
    }
}

/// Reads any numeric HDF5 dataset (or slice) into `Vec<f32>`.
pub(crate) fn read_dataset_f32(ds: &Dataset, slice: Option<&SliceInfo>) -> Result<Vec<f32>> {
    match ds.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => read_typed_vec::<f32>(ds, slice),
        _ => Ok(read_dataset_f64(ds, slice)?
            .into_iter()
            .map(|v| v as f32)
            .collect()),
    }
}

/// Reads an integer HDF5 dataset (or slice) into `Vec<i64>`.
pub(crate) fn read_dataset_i64(ds: &Dataset, slice: Option<&SliceInfo>) -> Result<Vec<i64>> {
    match ds.dtype() {
        Datatype::FixedPoint { size: 8, signed: true, .. } => read_typed_vec::<i64>(ds, slice),
        Datatype::FixedPoint { size: 8, signed: false, .. } => {
            let raw = read_typed_vec::<u64>(ds, slice)?;
            raw.into_iter()
                .map(|v| {
                    i64::try_from(v).map_err(|_| {
                        IoError::InvalidData(format!(
                            "Integer value {v} in dataset '{}' overflows i64",
                            ds.name()
                        ))
                    })
                })
                .collect()
        }
        Datatype::FixedPoint { size: 4, signed: true, .. } => Ok(read_typed_vec::<i32>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Datatype::FixedPoint { size: 4, signed: false, .. } => Ok(read_typed_vec::<u32>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Datatype::FixedPoint { size: 2, signed: true, .. } => Ok(read_typed_vec::<i16>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Datatype::FixedPoint { size: 2, signed: false, .. } => Ok(read_typed_vec::<u16>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Datatype::FixedPoint { size: 1, signed: true, .. } => Ok(read_typed_vec::<i8>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Datatype::FixedPoint { size: 1, signed: false, .. } => Ok(read_typed_vec::<u8>(ds, slice)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        other => Err(IoError::UnsupportedFormat(format!(
            "HDF5 dataset '{}' has non-integer datatype {other:?}",
            ds.name()
        ))),
    }
}

/// Reads an unsigned integer HDF5 dataset (or slice) into `Vec<u32>`, rejecting negative values.
pub(crate) fn read_dataset_u32(ds: &Dataset, slice: Option<&SliceInfo>) -> Result<Vec<u32>> {
    match ds.dtype() {
        Datatype::FixedPoint { size: 4, signed: false, .. } => read_typed_vec::<u32>(ds, slice),
        Datatype::FixedPoint { size: 1, signed: false, .. } => Ok(read_typed_vec::<u8>(ds, slice)?
            .into_iter()
            .map(|v| v as u32)
            .collect()),
        Datatype::FixedPoint { size: 2, signed: false, .. } => Ok(read_typed_vec::<u16>(ds, slice)?
            .into_iter()
            .map(|v| v as u32)
            .collect()),
        _ => {
            let vals = read_dataset_i64(ds, slice)?;
            let cap = safe_prealloc_capacity(vals.len(), std::mem::size_of::<u32>(), MAX_ALLOC_BYTES)?;
            let mut out = Vec::with_capacity(cap);
            for v in vals {
                let u = u32::try_from(v).map_err(|_| {
                    IoError::InvalidData(format!(
                        "Index/count {v} in dataset '{}' out of u32 range",
                        ds.name()
                    ))
                })?;
                out.push(u);
            }
            Ok(out)
        }
    }
}

/// Reads an HDF5 dataset (or slice) into `Vec<u8>`.
pub(crate) fn read_dataset_u8(ds: &Dataset, slice: Option<&SliceInfo>) -> Result<Vec<u8>> {
    match ds.dtype() {
        Datatype::FixedPoint { size: 1, signed: false, .. } => read_typed_vec::<u8>(ds, slice),
        Datatype::FixedPoint { size: 1, signed: true, .. } => Ok(read_typed_vec::<i8>(ds, slice)?
            .into_iter()
            .map(|v| v as u8)
            .collect()),
        _ => {
            let vals = read_dataset_i64(ds, slice)?;
            let cap = safe_prealloc_capacity(vals.len(), 1, MAX_ALLOC_BYTES)?;
            let mut out = Vec::with_capacity(cap);
            for v in vals {
                let b = u8::try_from(v).map_err(|_| {
                    IoError::InvalidData(format!(
                        "Byte value {v} in dataset '{}' out of u8 range",
                        ds.name()
                    ))
                })?;
                out.push(b);
            }
            Ok(out)
        }
    }
}

/// Reads an HDF5 numeric dataset (or slice) into a native-typed [`DataArray`].
pub(crate) fn read_dataset_as_data_array(
    ds: &Dataset,
    name: &str,
    num_components: usize,
    slice: Option<&SliceInfo>,
) -> Result<DataArray> {
    if num_components == 0 {
        return Err(IoError::InvalidData(format!(
            "DataArray '{name}' must have at least 1 component"
        )));
    }
    let data = match ds.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => {
            ArrayData::F32(read_typed_vec::<f32>(ds, slice)?)
        }
        Datatype::FloatingPoint { size: 8, .. } => {
            ArrayData::F64(read_typed_vec::<f64>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 1, signed: true, .. } => {
            ArrayData::I8(read_typed_vec::<i8>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 1, signed: false, .. } => {
            ArrayData::U8(read_typed_vec::<u8>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 2, signed: true, .. } => {
            ArrayData::I16(read_typed_vec::<i16>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 2, signed: false, .. } => {
            ArrayData::U16(read_typed_vec::<u16>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 4, signed: true, .. } => {
            ArrayData::I32(read_typed_vec::<i32>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 4, signed: false, .. } => {
            ArrayData::U32(read_typed_vec::<u32>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 8, signed: true, .. } => {
            ArrayData::I64(read_typed_vec::<i64>(ds, slice)?)
        }
        Datatype::FixedPoint { size: 8, signed: false, .. } => {
            ArrayData::U64(read_typed_vec::<u64>(ds, slice)?)
        }
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "HDF5 dataset '{}' has non-numeric datatype {other:?}",
                ds.name()
            )));
        }
    };

    if !data.len().is_multiple_of(num_components) {
        return Err(IoError::InvalidData(format!(
            "HDF5 dataset '{name}' length ({}) is not divisible by num_components ({num_components})",
            data.len()
        )));
    }

    Ok(DataArray::new(name, num_components, data))
}

/// Reads an HDF5 string dataset (or 2D fixed char matrix `[N, len]`) into `Vec<String>`.
pub(crate) fn read_dataset_strings(ds: &Dataset) -> Result<Vec<String>> {
    match ds.dtype() {
        Datatype::String { .. } | Datatype::VarLen { .. } => {
            let count = shape_element_count(ds.shape())?;
            let elem_size = ds.raw_element_size().unwrap_or(8).max(1);
            check_alloc_bytes(count, elem_size, MAX_ALLOC_BYTES)?;
            ds.read_strings()
                .map(|v| v.into_iter().map(|s| trim_nul(&s)).collect())
                .map_err(|e| IoError::Parse(format!("HDF5 string dataset '{}' read failed: {e}", ds.name())))
        }
        Datatype::FixedPoint { size: 1, .. } => {
            let shape = ds.shape();
            let raw = read_dataset_u8(ds, None)?;
            match shape.len() {
                0 | 1 => Ok(vec![decode_nul_terminated_bytes(&raw)]),
                2 => {
                    let rows = usize::try_from(shape[0]).map_err(|_| {
                        IoError::InvalidData("HDF5 string table rows overflow usize".into())
                    })?;
                    let cols = usize::try_from(shape[1]).map_err(|_| {
                        IoError::InvalidData("HDF5 string table cols overflow usize".into())
                    })?;
                    let cap = safe_prealloc_capacity(rows, std::mem::size_of::<String>(), MAX_ALLOC_BYTES)?;
                    let mut out = Vec::with_capacity(cap);
                    for r in 0..rows {
                        let start = r * cols;
                        let end = (start + cols).min(raw.len());
                        if start >= raw.len() {
                            out.push(String::new());
                        } else {
                            out.push(decode_nul_terminated_bytes(&raw[start..end]));
                        }
                    }
                    Ok(out)
                }
                rank => Err(IoError::InvalidData(format!(
                    "Unsupported rank {rank} for char dataset '{}'",
                    ds.name()
                ))),
            }
        }
        other => Err(IoError::UnsupportedFormat(format!(
            "HDF5 dataset '{}' is not a string or char dataset ({other:?})",
            ds.name()
        ))),
    }
}

/// Converts a flat coordinate buffer (3D `[x,y,z,...]` or 2D `[x,y,...]` if `dims_per_pt == 2`)
/// into `Vec<[f32; 3]>` with bounded pre-allocation.
pub(crate) fn pack_points_f32(flat: &[f32], dims_per_pt: usize) -> Result<Vec<[f32; 3]>> {
    if dims_per_pt != 2 && dims_per_pt != 3 {
        return Err(IoError::InvalidData(format!(
            "Expected 2 or 3 coordinate components per point, got {dims_per_pt}"
        )));
    }
    if !flat.len().is_multiple_of(dims_per_pt) {
        return Err(IoError::InvalidData(format!(
            "Coordinate buffer length {} is not divisible by {dims_per_pt}",
            flat.len()
        )));
    }
    let count = flat.len() / dims_per_pt;
    let cap = safe_prealloc_capacity(count, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
    let mut pts = Vec::with_capacity(cap);
    if dims_per_pt == 3 {
        for chunk in flat.as_chunks::<3>().0 {
            pts.push([chunk[0], chunk[1], chunk[2]]);
        }
    } else {
        for chunk in flat.as_chunks::<2>().0 {
            pts.push([chunk[0], chunk[1], 0.0]);
        }
    }
    Ok(pts)
}
