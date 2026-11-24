//! NRRD (`.nrrd`, `.nhdr`) and MetaImage (`.mhd`, `.mha`) volumetric medical image readers and writers.

use crate::error::{IoError, Result};
use paraviz_core::{ArrayData, DataArray, ImageData};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Reads a MetaImage `.mhd` or `.mha` volumetric file into an [`ImageData`].
pub fn read_metaimage(path: impl AsRef<Path>) -> Result<ImageData> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    // Find end of ASCII header: either `ElementDataFile = LOCAL\n` or EOF for `.mhd`
    let mut cursor = 0usize;
    let mut dims = [1usize; 3];
    let mut spacing = [1.0f64; 3];
    let mut origin = [0.0f64; 3];
    let mut element_type = String::from("MET_FLOAT");
    let mut data_file = String::new();
    let mut compressed = false;
    let mut big_endian = false;

    while cursor < bytes.len() {
        let line_end = bytes[cursor..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| cursor + p + 1)
            .unwrap_or(bytes.len());
        let line = String::from_utf8_lossy(&bytes[cursor..line_end])
            .trim()
            .to_string();
        cursor = line_end;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            let val = v.trim();
            match key {
                "DimSize" => {
                    let nums: Vec<usize> = val
                        .split_whitespace()
                        .filter_map(|t| t.parse::<usize>().ok())
                        .collect();
                    if nums.len() >= 3 {
                        dims = [nums[0], nums[1], nums[2]];
                    } else if nums.len() == 2 {
                        dims = [nums[0], nums[1], 1];
                    }
                }
                "ElementSpacing" | "ElementSize" => {
                    let nums: Vec<f64> = val
                        .split_whitespace()
                        .filter_map(|t| t.parse::<f64>().ok())
                        .collect();
                    if nums.len() >= 3 {
                        spacing = [nums[0], nums[1], nums[2]];
                    }
                }
                "Offset" | "Origin" | "Position" => {
                    let nums: Vec<f64> = val
                        .split_whitespace()
                        .filter_map(|t| t.parse::<f64>().ok())
                        .collect();
                    if nums.len() >= 3 {
                        origin = [nums[0], nums[1], nums[2]];
                    }
                }
                "ElementType" => element_type = val.to_string(),
                "CompressedData" => compressed = val.eq_ignore_ascii_case("true"),
                "BinaryDataByteOrderMSB" | "ElementByteOrderMSB" => {
                    big_endian = val.eq_ignore_ascii_case("true")
                }
                "ElementDataFile" => {
                    data_file = val.to_string();
                    if data_file.eq_ignore_ascii_case("LOCAL") {
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    let raw_payload = if data_file.eq_ignore_ascii_case("LOCAL") {
        bytes[cursor..].to_vec()
    } else if !data_file.is_empty() {
        let safe_path = crate::limits::resolve_safe_path(base_dir, &data_file)?;
        fs::read(safe_path)?
    } else {
        return Err(IoError::InvalidData(
            "MetaImage header missing ElementDataFile".into(),
        ));
    };

    let payload = if compressed {
        match crate::limits::decompress_zlib_limited(
            &raw_payload,
            crate::limits::MAX_DECOMPRESSED_BYTES,
        ) {
            Ok(out) => out,
            Err(_) => crate::limits::decompress_gzip_limited(
                &raw_payload,
                crate::limits::MAX_DECOMPRESSED_BYTES,
            )?,
        }
    } else {
        raw_payload
    };

    let count = dims[0] * dims[1] * dims[2];
    let arr_data = decode_typed_payload(&payload, &element_type, count, !big_endian)?;
    let mut img = ImageData::new(dims, origin, spacing);
    img.point_data
        .set_scalars(DataArray::new("MetaImage", 1, arr_data));
    Ok(img)
}

/// Writes an [`ImageData`] as a single-file `.mha` MetaImage.
pub fn write_metaimage(path: impl AsRef<Path>, img: &ImageData) -> Result<()> {
    let arr = img
        .point_data
        .active_scalars
        .as_deref()
        .and_then(|n| img.point_data.get(n))
        .or_else(|| img.point_data.arrays().first())
        .ok_or_else(|| IoError::InvalidData("ImageData has no point scalars".into()))?;
    let scalars = arr.to_scalar_f32(None);
    let mut f = fs::File::create(path)?;
    writeln!(f, "ObjectType = Image")?;
    writeln!(f, "NDims = 3")?;
    writeln!(f, "BinaryData = True")?;
    writeln!(f, "BinaryDataByteOrderMSB = False")?;
    writeln!(f, "CompressedData = False")?;
    writeln!(
        f,
        "Offset = {} {} {}",
        img.origin[0], img.origin[1], img.origin[2]
    )?;
    writeln!(
        f,
        "ElementSpacing = {} {} {}",
        img.spacing[0], img.spacing[1], img.spacing[2]
    )?;
    writeln!(
        f,
        "DimSize = {} {} {}",
        img.dimensions[0], img.dimensions[1], img.dimensions[2]
    )?;
    writeln!(f, "ElementType = MET_FLOAT")?;
    writeln!(f, "ElementDataFile = LOCAL")?;
    for v in scalars {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Reads an NRRD (`.nrrd` or `.nhdr`) volumetric file into an [`ImageData`].
pub fn read_nrrd(path: impl AsRef<Path>) -> Result<ImageData> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    // Split header and attached body at blank line (\n\n or \r\n\r\n)
    let mut header_end = bytes.len();
    let mut body_start = bytes.len();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            header_end = i;
            body_start = i + 2;
            break;
        }
        if i + 3 < bytes.len()
            && bytes[i] == b'\r'
            && bytes[i + 1] == b'\n'
            && bytes[i + 2] == b'\r'
            && bytes[i + 3] == b'\n'
        {
            header_end = i;
            body_start = i + 4;
            break;
        }
    }

    let header_str = String::from_utf8_lossy(&bytes[..header_end]);
    let mut dims = [1usize; 3];
    let mut spacing = [1.0f64; 3];
    let mut origin = [0.0f64; 3];
    let mut nrrd_type = String::from("float");
    let mut encoding = String::from("raw");
    let mut little_endian = true;
    let mut data_file: Option<String> = None;

    for line in header_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("NRRD") {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim();
            match key.as_str() {
                "type" => nrrd_type = val.to_ascii_lowercase(),
                "sizes" => {
                    let nums: Vec<usize> = val
                        .split_whitespace()
                        .filter_map(|t| t.parse::<usize>().ok())
                        .collect();
                    if nums.len() >= 3 {
                        dims = [nums[0], nums[1], nums[2]];
                    } else if nums.len() == 2 {
                        dims = [nums[0], nums[1], 1];
                    }
                }
                "spacings" => {
                    let nums: Vec<f64> = val
                        .split_whitespace()
                        .filter_map(|t| t.parse::<f64>().ok())
                        .collect();
                    for i in 0..nums.len().min(3) {
                        if nums[i].is_finite() && nums[i] > 0.0 {
                            spacing[i] = nums[i];
                        }
                    }
                }
                "space origin" => {
                    let cleaned = val.trim_matches(|c| c == '(' || c == ')');
                    let nums: Vec<f64> = cleaned
                        .split(',')
                        .filter_map(|t| t.trim().parse::<f64>().ok())
                        .collect();
                    if nums.len() >= 3 {
                        origin = [nums[0], nums[1], nums[2]];
                    }
                }
                "encoding" => encoding = val.to_ascii_lowercase(),
                "endian" => little_endian = !val.eq_ignore_ascii_case("big"),
                "data file" | "datafile" => data_file = Some(val.to_string()),
                _ => {}
            }
        }
    }

    let raw_payload = if let Some(df) = data_file {
        let safe_path = crate::limits::resolve_safe_path(base_dir, &df)?;
        fs::read(safe_path)?
    } else {
        bytes[body_start..].to_vec()
    };

    let payload = match encoding.as_str() {
        "raw" => raw_payload,
        "gzip" | "gz" => crate::limits::decompress_gzip_limited(
            &raw_payload,
            crate::limits::MAX_DECOMPRESSED_BYTES,
        )?,
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported NRRD encoding '{other}'"
            )));
        }
    };

    let count = dims[0] * dims[1] * dims[2];
    let arr_data = decode_typed_payload(&payload, &nrrd_type, count, little_endian)?;
    let mut img = ImageData::new(dims, origin, spacing);
    img.point_data
        .set_scalars(DataArray::new("NRRD", 1, arr_data));
    Ok(img)
}

fn decode_typed_payload(
    payload: &[u8],
    type_name: &str,
    count: usize,
    little_endian: bool,
) -> Result<ArrayData> {
    let t = type_name.to_ascii_lowercase();
    match t.as_str() {
        "met_uchar" | "uchar" | "unsigned char" | "uint8" | "uint8_t" => {
            if payload.len() < count {
                return Err(IoError::InvalidData("Truncated u8 volume payload".into()));
            }
            Ok(ArrayData::U8(payload[..count].to_vec()))
        }
        "met_char" | "char" | "signed char" | "int8" | "int8_t" => {
            if payload.len() < count {
                return Err(IoError::InvalidData("Truncated i8 volume payload".into()));
            }
            Ok(ArrayData::I8(
                payload[..count].iter().map(|&b| b as i8).collect(),
            ))
        }
        "met_short" | "short" | "signed short" | "int16" | "int16_t" => {
            if payload.len() < count * 2 {
                return Err(IoError::InvalidData("Truncated i16 volume payload".into()));
            }
            Ok(ArrayData::I16(
                payload[..count * 2]
                    .as_chunks::<2>().0.iter()
                    .map(|c| {
                        if little_endian {
                            i16::from_le_bytes([c[0], c[1]])
                        } else {
                            i16::from_be_bytes([c[0], c[1]])
                        }
                    })
                    .collect(),
            ))
        }
        "met_ushort" | "ushort" | "unsigned short" | "uint16" | "uint16_t" => {
            if payload.len() < count * 2 {
                return Err(IoError::InvalidData("Truncated u16 volume payload".into()));
            }
            Ok(ArrayData::U16(
                payload[..count * 2]
                    .as_chunks::<2>().0.iter()
                    .map(|c| {
                        if little_endian {
                            u16::from_le_bytes([c[0], c[1]])
                        } else {
                            u16::from_be_bytes([c[0], c[1]])
                        }
                    })
                    .collect(),
            ))
        }
        "met_int" | "int" | "signed int" | "int32" | "int32_t" => {
            if payload.len() < count * 4 {
                return Err(IoError::InvalidData("Truncated i32 volume payload".into()));
            }
            Ok(ArrayData::I32(
                payload[..count * 4]
                    .as_chunks::<4>().0.iter()
                    .map(|c| {
                        if little_endian {
                            i32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        } else {
                            i32::from_be_bytes([c[0], c[1], c[2], c[3]])
                        }
                    })
                    .collect(),
            ))
        }
        "met_float" | "float" | "float32" => {
            if payload.len() < count * 4 {
                return Err(IoError::InvalidData("Truncated f32 volume payload".into()));
            }
            Ok(ArrayData::F32(
                payload[..count * 4]
                    .as_chunks::<4>().0.iter()
                    .map(|c| {
                        if little_endian {
                            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        } else {
                            f32::from_be_bytes([c[0], c[1], c[2], c[3]])
                        }
                    })
                    .collect(),
            ))
        }
        "met_double" | "double" | "float64" => {
            if payload.len() < count * 8 {
                return Err(IoError::InvalidData("Truncated f64 volume payload".into()));
            }
            Ok(ArrayData::F64(
                payload[..count * 8]
                    .as_chunks::<8>().0.iter()
                    .map(|c| {
                        if little_endian {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        } else {
                            f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        }
                    })
                    .collect(),
            ))
        }
        other => Err(IoError::UnsupportedFormat(format!(
            "Unsupported volume scalar type '{other}'"
        ))),
    }
}
