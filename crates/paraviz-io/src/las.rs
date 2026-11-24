//! ASPRS LAS LiDAR point cloud (`.las`) reader (versions 1.0 – 1.4, uncompressed).
//!
//! Reads XYZ coordinates (with header scale & offset), `Intensity`, `Classification`,
//! `ReturnNumber`, and `Colors` (RGBA) when the point format includes RGB.

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, PolyData};
use std::fs;
use std::path::Path;

/// Reads an uncompressed ASPRS `.las` LiDAR file into a [`DataSet`].
pub fn read_las(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_las_bytes(&bytes)
}

/// Parses uncompressed ASPRS `.las` bytes into a [`DataSet`].
pub fn read_las_bytes(bytes: &[u8]) -> Result<DataSet> {
    if bytes.len() < 227 {
        return Err(IoError::Parse("LAS file shorter than 227-byte public header".into()));
    }
    if &bytes[0..4] != b"LASF" {
        return Err(IoError::Parse("Invalid LAS magic header (expected 'LASF')".into()));
    }

    let ver_major = bytes[24];
    let ver_minor = bytes[25];
    let offset_to_points = u32::from_le_bytes(bytes[96..100].try_into().unwrap()) as usize;
    let point_format = bytes[104] & 0x3F;
    let is_compressed = (bytes[104] & 0x80) != 0 || (bytes[104] & 0x40) != 0;
    if is_compressed {
        return Err(IoError::UnsupportedFormat(
            "LAZ compressed LAS files are not supported".into(),
        ));
    }
    let record_len = u16::from_le_bytes(bytes[105..107].try_into().unwrap()) as usize;
    let legacy_count = u32::from_le_bytes(bytes[107..111].try_into().unwrap()) as u64;
    let point_count = if ver_major >= 1 && ver_minor >= 4 && bytes.len() >= 255 && legacy_count == 0
    {
        u64::from_le_bytes(bytes[247..255].try_into().unwrap()) as usize
    } else {
        legacy_count as usize
    };

    let x_scale = f64::from_le_bytes(bytes[131..139].try_into().unwrap());
    let y_scale = f64::from_le_bytes(bytes[139..147].try_into().unwrap());
    let z_scale = f64::from_le_bytes(bytes[147..155].try_into().unwrap());
    let x_offset = f64::from_le_bytes(bytes[155..163].try_into().unwrap());
    let y_offset = f64::from_le_bytes(bytes[163..171].try_into().unwrap());
    let z_offset = f64::from_le_bytes(bytes[171..179].try_into().unwrap());

    if record_len < 20 {
        return Err(IoError::Parse(format!(
            "Invalid LAS point record length {record_len}"
        )));
    }
    let needed = offset_to_points
        .checked_add(point_count.checked_mul(record_len).ok_or_else(|| {
            IoError::InvalidData("LAS point count overflows size".into())
        })?)
        .ok_or_else(|| IoError::InvalidData("LAS offset overflows size".into()))?;
    if needed > bytes.len() {
        return Err(IoError::Parse("Truncated LAS point records".into()));
    }

    let mut points = Vec::with_capacity(safe_prealloc_capacity(point_count, 12, MAX_ALLOC_BYTES)?);
    let mut verts = Vec::with_capacity(safe_prealloc_capacity(point_count, 4, MAX_ALLOC_BYTES)?);
    let mut intensity = Vec::with_capacity(safe_prealloc_capacity(point_count, 2, MAX_ALLOC_BYTES)?);
    let mut classification = Vec::with_capacity(safe_prealloc_capacity(point_count, 1, MAX_ALLOC_BYTES)?);

    let rgb_offset: Option<usize> = match point_format {
        2 => Some(20),
        3 | 5 => Some(28),
        7 => Some(30),
        8 | 10 => Some(30),
        _ => None,
    };
    let mut colors = if rgb_offset.is_some() {
        Vec::with_capacity(safe_prealloc_capacity(point_count, 4, MAX_ALLOC_BYTES)?)
    } else {
        Vec::new()
    };

    for i in 0..point_count {
        let rec = &bytes[offset_to_points + i * record_len..offset_to_points + (i + 1) * record_len];
        let ix = i32::from_le_bytes(rec[0..4].try_into().unwrap()) as f64;
        let iy = i32::from_le_bytes(rec[4..8].try_into().unwrap()) as f64;
        let iz = i32::from_le_bytes(rec[8..12].try_into().unwrap()) as f64;
        let x = (ix * x_scale + x_offset) as f32;
        let y = (iy * y_scale + y_offset) as f32;
        let z = (iz * z_scale + z_offset) as f32;
        points.push([x, y, z]);
        verts.push(i as u32);

        let inten = u16::from_le_bytes(rec[12..14].try_into().unwrap());
        intensity.push(inten);

        let cls = if point_format >= 6 && rec.len() > 16 {
            rec[16]
        } else {
            rec[15] & 0x1F
        };
        classification.push(cls);

        if let Some(ro) = rgb_offset
            && ro + 6 <= rec.len() {
                let r16 = u16::from_le_bytes(rec[ro..ro + 2].try_into().unwrap());
                let g16 = u16::from_le_bytes(rec[ro + 2..ro + 4].try_into().unwrap());
                let b16 = u16::from_le_bytes(rec[ro + 4..ro + 6].try_into().unwrap());
                // Scale 16-bit LAS colors to 8-bit
                colors.push((r16 >> 8) as u8);
                colors.push((g16 >> 8) as u8);
                colors.push((b16 >> 8) as u8);
                colors.push(255u8);
            }
    }

    let mut poly = PolyData::new();
    poly.points = points;
    poly.verts = verts;
    poly.point_data
        .add_array(DataArray::new("Intensity", 1, ArrayData::U16(intensity)));
    poly.point_data.add_array(DataArray::new(
        "Classification",
        1,
        ArrayData::U8(classification),
    ));
    if colors.len() == point_count * 4 {
        poly.point_data
            .add_array(DataArray::new("Colors", 4, ArrayData::U8(colors)));
    }
    Ok(DataSet::PolyData(poly))
}
