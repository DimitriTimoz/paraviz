//! Uncompressed DICOM (`.dcm`) medical image slice reader.
//!
//! Supports Explicit VR Little-Endian and Implicit VR Little-Endian DICOM files,
//! extracting `Rows`, `Columns`, `PixelSpacing`, `SliceThickness`, `BitsAllocated`,
//! `PixelRepresentation`, `RescaleSlope`, `RescaleIntercept`, and `PixelData`.

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, ImageData};
use std::fs;
use std::path::Path;

/// Reads an uncompressed DICOM `.dcm` file into an [`ImageData`].
pub fn read_dicom(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_dicom_bytes(&bytes)
}

/// Parses uncompressed DICOM `.dcm` bytes into an [`ImageData`].
pub fn read_dicom_bytes(bytes: &[u8]) -> Result<DataSet> {
    if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
        return Err(IoError::Parse(
            "Invalid DICOM file (missing 128-byte preamble + 'DICM' magic)".into(),
        ));
    }

    let mut pos = 132usize;
    let mut rows: usize = 0;
    let mut cols: usize = 0;
    let mut bits_allocated: u16 = 16;
    let mut pixel_rep: u16 = 0; // 0 = unsigned, 1 = signed
    let mut samples_per_pixel: usize = 1;
    let mut spacing = [1.0f64, 1.0, 1.0];
    let mut rescale_slope = 1.0f32;
    let mut rescale_intercept = 0.0f32;
    let mut explicit_vr = true;
    let mut pixel_data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
        let elem = u16::from_le_bytes([bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        // Group 0x0002 is always Explicit VR Little-Endian; afterwards check if VR is ASCII
        let is_explicit = if group == 0x0002 {
            true
        } else {
            if pos == 136 || (group > 0x0002 && pos + 2 <= bytes.len()) {
                let b0 = bytes[pos];
                let b1 = bytes[pos + 1];
                if b0.is_ascii_uppercase() && b1.is_ascii_uppercase() {
                    explicit_vr = true;
                } else if group > 0x0002 && !matches!(bytes[pos..pos + 2], [b'O', b'B'] | [b'O', b'W']) {
                    // Keep prior explicit_vr unless clearly implicit
                }
            }
            explicit_vr
        };

        let (vr, val_len) = if is_explicit && pos + 2 <= bytes.len() && bytes[pos].is_ascii_uppercase() && bytes[pos + 1].is_ascii_uppercase() {
            let vr = [bytes[pos], bytes[pos + 1]];
            pos += 2;
            let long_vr = matches!(
                &vr,
                b"OB" | b"OD" | b"OF" | b"OL" | b"OW" | b"SQ" | b"UC" | b"UN" | b"UR" | b"UT"
            );
            if long_vr {
                if pos + 6 > bytes.len() {
                    break;
                }
                pos += 2; // reserved
                let l = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                (Some(vr), l)
            } else {
                if pos + 2 > bytes.len() {
                    break;
                }
                let l = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                (Some(vr), l)
            }
        } else {
            explicit_vr = false;
            if pos + 4 > bytes.len() {
                break;
            }
            let l = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            (None, l)
        };

        if val_len == 0xFFFF_FFFF {
            // Undefined length (e.g., encapsulated pixel data or SQ)
            if group == 0x7FE0 && elem == 0x0010 {
                return Err(IoError::UnsupportedFormat(
                    "Compressed/encapsulated DICOM PixelData is not supported".into(),
                ));
            }
            // Skip until Sequence Delimitation Item (FFFE,E0DD)
            while pos + 8 <= bytes.len() {
                if bytes[pos..pos + 4] == [0xFE, 0xFF, 0xDD, 0xE0] {
                    pos += 8;
                    break;
                }
                pos += 2;
            }
            continue;
        }

        if pos + val_len > bytes.len() {
            break;
        }
        let val = &bytes[pos..pos + val_len];
        pos += val_len;

        match (group, elem) {
            (0x0002, 0x0010) => {
                let uid = String::from_utf8_lossy(val).trim_matches(['\0', ' ']).to_string();
                if uid == "1.2.840.10008.1.2" {
                    explicit_vr = false;
                }
            }
            (0x0018, 0x0050) => {
                if let Ok(v) = String::from_utf8_lossy(val).trim().parse::<f64>() {
                    spacing[2] = v;
                }
            }
            (0x0028, 0x0002) if val.len() >= 2 => {
                samples_per_pixel = u16::from_le_bytes([val[0], val[1]]) as usize;
            }
            (0x0028, 0x0010) if val.len() >= 2 => {
                rows = u16::from_le_bytes([val[0], val[1]]) as usize;
            }
            (0x0028, 0x0011) if val.len() >= 2 => {
                cols = u16::from_le_bytes([val[0], val[1]]) as usize;
            }
            (0x0028, 0x0030) => {
                let s = String::from_utf8_lossy(val);
                let mut parts = s.trim_matches(['\0', ' ']).split('\\');
                if let (Some(sy), Some(sx)) = (parts.next(), parts.next())
                    && let (Ok(vy), Ok(vx)) = (sy.trim().parse::<f64>(), sx.trim().parse::<f64>()) {
                        spacing[0] = vx;
                        spacing[1] = vy;
                    }
            }
            (0x0028, 0x0100) if val.len() >= 2 => {
                bits_allocated = u16::from_le_bytes([val[0], val[1]]);
            }
            (0x0028, 0x0103) if val.len() >= 2 => {
                pixel_rep = u16::from_le_bytes([val[0], val[1]]);
            }
            (0x0028, 0x1052) => {
                if let Ok(v) = String::from_utf8_lossy(val).trim_matches(['\0', ' ']).parse::<f32>() {
                    rescale_intercept = v;
                }
            }
            (0x0028, 0x1053) => {
                if let Ok(v) = String::from_utf8_lossy(val).trim_matches(['\0', ' ']).parse::<f32>() {
                    rescale_slope = v;
                }
            }
            (0x7FE0, 0x0010) => {
                pixel_data = Some(val);
                let _ = vr;
                break;
            }
            _ => {}
        }
    }

    let raw_pixels = pixel_data
        .ok_or_else(|| IoError::Parse("DICOM file missing PixelData (7FE0,0010)".into()))?;
    if rows == 0 || cols == 0 {
        return Err(IoError::InvalidData("DICOM file has zero Rows or Columns".into()));
    }
    let n_vox = rows * cols * samples_per_pixel;

    let array_data = match bits_allocated {
        8 => {
            if raw_pixels.len() < n_vox {
                return Err(IoError::Parse("Truncated 8-bit DICOM PixelData".into()));
            }
            if rescale_slope != 1.0 || rescale_intercept != 0.0 {
                let mut out = Vec::with_capacity(safe_prealloc_capacity(n_vox, 4, MAX_ALLOC_BYTES)?);
                for &b in &raw_pixels[..n_vox] {
                    out.push(b as f32 * rescale_slope + rescale_intercept);
                }
                ArrayData::F32(out)
            } else {
                ArrayData::U8(raw_pixels[..n_vox].to_vec())
            }
        }
        16 => {
            if raw_pixels.len() < n_vox * 2 {
                return Err(IoError::Parse("Truncated 16-bit DICOM PixelData".into()));
            }
            if rescale_slope != 1.0 || rescale_intercept != 0.0 {
                let mut out = Vec::with_capacity(safe_prealloc_capacity(n_vox, 4, MAX_ALLOC_BYTES)?);
                for chunk in raw_pixels[..n_vox * 2].as_chunks::<2>().0 {
                    let raw = if pixel_rep == 1 {
                        i16::from_le_bytes([chunk[0], chunk[1]]) as f32
                    } else {
                        u16::from_le_bytes([chunk[0], chunk[1]]) as f32
                    };
                    out.push(raw * rescale_slope + rescale_intercept);
                }
                ArrayData::F32(out)
            } else if pixel_rep == 1 {
                let out: Vec<i16> = raw_pixels[..n_vox * 2]
                    .as_chunks::<2>().0.iter()
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                ArrayData::I16(out)
            } else {
                let out: Vec<u16> = raw_pixels[..n_vox * 2]
                    .as_chunks::<2>().0.iter()
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                ArrayData::U16(out)
            }
        }
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported DICOM BitsAllocated={other}"
            )));
        }
    };

    let mut img = ImageData::new([cols, rows, 1], [0.0, 0.0, 0.0], spacing);
    img.point_data
        .add_array(DataArray::new("Scalars", samples_per_pixel, array_data));
    Ok(DataSet::ImageData(img))
}
