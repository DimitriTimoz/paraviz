//! NIfTI-1 (`.nii`, `.nii.gz`) medical volumetric image reader and writer producing [`ImageData`].

use crate::error::{IoError, Result};
use crate::limits::{decompress_gzip_limited, read_to_end_limited, MAX_DECOMPRESSED_BYTES};
use flate2::write::GzEncoder;
use flate2::Compression;
use paraviz_core::{ArrayData, DataArray, ImageData};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Reads a NIfTI-1 file (`.nii` or gzip-compressed `.nii.gz`) into an [`ImageData`].
pub fn read_nifti(path: impl AsRef<Path>) -> Result<ImageData> {
    let path = path.as_ref();
    let f = File::open(path)?;
    let raw = read_to_end_limited(f, MAX_DECOMPRESSED_BYTES)?;

    let bytes = if raw.len() >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
        decompress_gzip_limited(&raw, MAX_DECOMPRESSED_BYTES)?
    } else {
        raw
    };

    read_nifti_bytes(&bytes)
}

/// Parses uncompressed NIfTI-1 bytes into an [`ImageData`].
pub fn read_nifti_bytes(bytes: &[u8]) -> Result<ImageData> {
    if bytes.len() < 348 {
        return Err(IoError::InvalidData(
            "NIfTI file shorter than 348-byte header".into(),
        ));
    }

    let sizeof_le = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let sizeof_be = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let little_endian = if sizeof_le == 348 {
        true
    } else if sizeof_be == 348 {
        false
    } else {
        return Err(IoError::InvalidData(format!(
            "Invalid NIfTI-1 sizeof_hdr (le={sizeof_le}, be={sizeof_be}, expected 348)"
        )));
    };

    let read_i16 = |off: usize| -> i16 {
        let s = [bytes[off], bytes[off + 1]];
        if little_endian {
            i16::from_le_bytes(s)
        } else {
            i16::from_be_bytes(s)
        }
    };
    let read_f32 = |off: usize| -> f32 {
        let s = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        if little_endian {
            f32::from_le_bytes(s)
        } else {
            f32::from_be_bytes(s)
        }
    };

    let ndim = read_i16(40).max(1) as usize;
    let nx = read_i16(42).max(1) as usize;
    let ny = if ndim >= 2 {
        read_i16(44).max(1) as usize
    } else {
        1
    };
    let nz = if ndim >= 3 {
        read_i16(46).max(1) as usize
    } else {
        1
    };
    let datatype = read_i16(70);
    let sx = read_f32(80).abs();
    let sy = read_f32(84).abs();
    let sz = read_f32(88).abs();
    let spacing = [
        if sx > 0.0 { sx as f64 } else { 1.0 },
        if sy > 0.0 { sy as f64 } else { 1.0 },
        if sz > 0.0 { sz as f64 } else { 1.0 },
    ];
    let vox_offset = (read_f32(108) as usize).max(352);
    let scl_slope = read_f32(112);
    let scl_inter = read_f32(116);
    let qoffset_x = read_f32(268) as f64;
    let qoffset_y = read_f32(272) as f64;
    let qoffset_z = read_f32(276) as f64;
    let origin = [qoffset_x, qoffset_y, qoffset_z];

    if vox_offset > bytes.len() {
        return Err(IoError::InvalidData(format!(
            "NIfTI vox_offset ({vox_offset}) exceeds file size ({})",
            bytes.len()
        )));
    }
    let payload = &bytes[vox_offset..];
    let count = nx * ny * nz;
    let apply_scale = scl_slope != 0.0 && ((scl_slope - 1.0).abs() > 1e-6 || scl_inter.abs() > 1e-6);

    let array_data = match datatype {
        2 => {
            // UINT8
            if payload.len() < count {
                return Err(IoError::InvalidData("Truncated NIfTI uint8 payload".into()));
            }
            if apply_scale {
                ArrayData::F32(
                    payload[..count]
                        .iter()
                        .map(|&v| v as f32 * scl_slope + scl_inter)
                        .collect(),
                )
            } else {
                ArrayData::U8(payload[..count].to_vec())
            }
        }
        256 => {
            // INT8
            if payload.len() < count {
                return Err(IoError::InvalidData("Truncated NIfTI int8 payload".into()));
            }
            ArrayData::I8(payload[..count].iter().map(|&b| b as i8).collect())
        }
        4 => {
            // INT16
            if payload.len() < count * 2 {
                return Err(IoError::InvalidData("Truncated NIfTI int16 payload".into()));
            }
            let vals: Vec<i16> = payload[..count * 2]
                .chunks_exact(2)
                .map(|c| {
                    if little_endian {
                        i16::from_le_bytes([c[0], c[1]])
                    } else {
                        i16::from_be_bytes([c[0], c[1]])
                    }
                })
                .collect();
            if apply_scale {
                ArrayData::F32(
                    vals.into_iter()
                        .map(|v| v as f32 * scl_slope + scl_inter)
                        .collect(),
                )
            } else {
                ArrayData::I16(vals)
            }
        }
        512 => {
            // UINT16
            if payload.len() < count * 2 {
                return Err(IoError::InvalidData("Truncated NIfTI uint16 payload".into()));
            }
            let vals: Vec<u16> = payload[..count * 2]
                .chunks_exact(2)
                .map(|c| {
                    if little_endian {
                        u16::from_le_bytes([c[0], c[1]])
                    } else {
                        u16::from_be_bytes([c[0], c[1]])
                    }
                })
                .collect();
            if apply_scale {
                ArrayData::F32(
                    vals.into_iter()
                        .map(|v| v as f32 * scl_slope + scl_inter)
                        .collect(),
                )
            } else {
                ArrayData::U16(vals)
            }
        }
        8 => {
            // INT32
            if payload.len() < count * 4 {
                return Err(IoError::InvalidData("Truncated NIfTI int32 payload".into()));
            }
            let vals: Vec<i32> = payload[..count * 4]
                .chunks_exact(4)
                .map(|c| {
                    if little_endian {
                        i32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        i32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    }
                })
                .collect();
            ArrayData::I32(vals)
        }
        16 => {
            // FLOAT32
            if payload.len() < count * 4 {
                return Err(IoError::InvalidData("Truncated NIfTI float32 payload".into()));
            }
            let vals: Vec<f32> = payload[..count * 4]
                .chunks_exact(4)
                .map(|c| {
                    let v = if little_endian {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        f32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    };
                    if apply_scale {
                        v * scl_slope + scl_inter
                    } else {
                        v
                    }
                })
                .collect();
            ArrayData::F32(vals)
        }
        64 => {
            // FLOAT64
            if payload.len() < count * 8 {
                return Err(IoError::InvalidData("Truncated NIfTI float64 payload".into()));
            }
            let vals: Vec<f64> = payload[..count * 8]
                .chunks_exact(8)
                .map(|c| {
                    if little_endian {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    } else {
                        f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                })
                .collect();
            ArrayData::F64(vals)
        }
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported NIfTI datatype code {other}"
            )));
        }
    };

    let mut img = ImageData::new([nx, ny, nz], origin, spacing);
    img.point_data
        .set_scalars(DataArray::new("intensity", 1, array_data));
    Ok(img)
}

/// Writes an [`ImageData`] as a NIfTI-1 file (`.nii` or `.nii.gz` if filename ends with `.gz`).
pub fn write_nifti(path: impl AsRef<Path>, img: &ImageData) -> Result<()> {
    let path = path.as_ref();
    let arr = img
        .point_data
        .active_scalars
        .as_deref()
        .and_then(|n| img.point_data.get(n))
        .or_else(|| img.point_data.arrays().first())
        .ok_or_else(|| IoError::InvalidData("ImageData has no point scalar array".into()))?;
    let scalars = arr.to_scalar_f32(None);

    let mut hdr = vec![0u8; 352];
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());
    hdr[40..42].copy_from_slice(&3i16.to_le_bytes());
    hdr[42..44].copy_from_slice(&(img.dimensions[0] as i16).to_le_bytes());
    hdr[44..46].copy_from_slice(&(img.dimensions[1] as i16).to_le_bytes());
    hdr[46..48].copy_from_slice(&(img.dimensions[2] as i16).to_le_bytes());
    hdr[70..72].copy_from_slice(&16i16.to_le_bytes()); // FLOAT32
    hdr[72..74].copy_from_slice(&32i16.to_le_bytes()); // bitpix
    hdr[80..84].copy_from_slice(&(img.spacing[0] as f32).to_le_bytes());
    hdr[84..88].copy_from_slice(&(img.spacing[1] as f32).to_le_bytes());
    hdr[88..92].copy_from_slice(&(img.spacing[2] as f32).to_le_bytes());
    hdr[108..112].copy_from_slice(&352.0f32.to_le_bytes()); // vox_offset
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes()); // scl_slope
    hdr[268..272].copy_from_slice(&(img.origin[0] as f32).to_le_bytes());
    hdr[272..276].copy_from_slice(&(img.origin[1] as f32).to_le_bytes());
    hdr[276..280].copy_from_slice(&(img.origin[2] as f32).to_le_bytes());
    hdr[344..348].copy_from_slice(b"n+1\0");

    let mut payload = Vec::with_capacity(scalars.len() * 4);
    for v in scalars {
        payload.extend_from_slice(&v.to_le_bytes());
    }

    let is_gz = path
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".gz");
    let f = File::create(path)?;
    if is_gz {
        let mut enc = GzEncoder::new(f, Compression::default());
        enc.write_all(&hdr)?;
        enc.write_all(&payload)?;
        enc.finish()?;
    } else {
        let mut f = f;
        f.write_all(&hdr)?;
        f.write_all(&payload)?;
    }
    Ok(())
}
