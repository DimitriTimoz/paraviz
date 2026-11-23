//! 2D scientific/raster image readers: PNG (`.png`), BMP (`.bmp`), and Netpbm (`.ppm`, `.pgm`, `.pbm`).
//!
//! Loads 2D images as [`ImageData`] with dimensions `[width, height, 1]`.

use crate::error::{IoError, Result};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use paraviz_core::{ArrayData, DataArray, DataSet, ImageData};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

/// Reads a PNG (`.png`) file into a 2D [`ImageData`].
pub fn read_png(path: impl AsRef<Path>) -> Result<DataSet> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| IoError::Parse(format!("Invalid PNG header: {e}")))?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| IoError::Parse(format!("Failed to decode PNG frame: {e}")))?;
    let bytes = &buf[..info.buffer_size()];
    let w = info.width as usize;
    let h = info.height as usize;

    let mut img = ImageData::new([w, h, 1], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let n_comp = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 3,
    };

    // Convert 16-bit PNGs to U16 and 8-bit to U8; compute luminance scalar as well
    if info.bit_depth == png::BitDepth::Sixteen {
        let vals: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        let name = if n_comp == 1 { "Scalars" } else { "PNGImage" };
        img.point_data
            .add_array(DataArray::new(name, n_comp, ArrayData::U16(vals)));
    } else {
        let name = if n_comp >= 3 { "Colors" } else { "Scalars" };
        img.point_data
            .add_array(DataArray::new(name, n_comp, ArrayData::U8(bytes.to_vec())));
    }

    Ok(DataSet::ImageData(img))
}

/// Reads an uncompressed Windows BMP (`.bmp`, 8-bit/24-bit/32-bit) into a 2D [`ImageData`].
pub fn read_bmp(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_bmp_bytes(&bytes)
}

/// Parses uncompressed BMP bytes into a 2D [`ImageData`].
pub fn read_bmp_bytes(bytes: &[u8]) -> Result<DataSet> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(IoError::Parse("Invalid BMP header".into()));
    }
    let pixel_offset = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
    let width = i32::from_le_bytes(bytes[18..22].try_into().unwrap());
    let height_signed = i32::from_le_bytes(bytes[22..26].try_into().unwrap());
    let bpp = u16::from_le_bytes(bytes[28..30].try_into().unwrap());
    let compression = u32::from_le_bytes(bytes[30..34].try_into().unwrap());

    if width <= 0 || height_signed == 0 {
        return Err(IoError::InvalidData("Invalid BMP dimensions".into()));
    }
    if compression != 0 {
        return Err(IoError::UnsupportedFormat(format!(
            "Compressed BMP (method {compression}) is not supported"
        )));
    }
    let w = width as usize;
    let h = height_signed.unsigned_abs() as usize;
    let top_down = height_signed < 0;

    let n_comp = match bpp {
        8 => 1,
        24 => 3,
        32 => 4,
        _ => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported BMP bit depth {bpp}"
            )));
        }
    };
    let row_stride = (w * (bpp as usize / 8) + 3) & !3;
    if pixel_offset + row_stride * h > bytes.len() {
        return Err(IoError::Parse("Truncated BMP pixel data".into()));
    }

    let mut out = Vec::with_capacity(safe_prealloc_capacity(w * h, n_comp, MAX_ALLOC_BYTES)?);
    for y in 0..h {
        let src_y = if top_down { y } else { h - 1 - y };
        let row = &bytes[pixel_offset + src_y * row_stride..pixel_offset + (src_y + 1) * row_stride];
        match bpp {
            8 => out.extend_from_slice(&row[..w]),
            24 => {
                for x in 0..w {
                    out.push(row[x * 3 + 2]); // R
                    out.push(row[x * 3 + 1]); // G
                    out.push(row[x * 3]);     // B
                }
            }
            32 => {
                for x in 0..w {
                    out.push(row[x * 4 + 2]); // R
                    out.push(row[x * 4 + 1]); // G
                    out.push(row[x * 4]);     // B
                    out.push(row[x * 4 + 3]); // A
                }
            }
            _ => unreachable!(),
        }
    }

    let mut img = ImageData::new([w, h, 1], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let name = if n_comp >= 3 { "Colors" } else { "Scalars" };
    img.point_data
        .add_array(DataArray::new(name, n_comp, ArrayData::U8(out)));
    Ok(DataSet::ImageData(img))
}

/// Reads a Netpbm image (`.ppm`, `.pgm`, `.pbm` — P1/P2/P3 ASCII or P5/P6 Binary) into a 2D [`ImageData`].
pub fn read_pnm(path: impl AsRef<Path>) -> Result<DataSet> {
    let bytes = fs::read(path)?;
    read_pnm_bytes(&bytes)
}

/// Parses Netpbm bytes into a 2D [`ImageData`].
pub fn read_pnm_bytes(bytes: &[u8]) -> Result<DataSet> {
    let mut pos = 0usize;
    let magic = next_pnm_token(bytes, &mut pos)
        .ok_or_else(|| IoError::Parse("Empty PNM file".into()))?;
    let w: usize = next_pnm_token(bytes, &mut pos)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| IoError::Parse("Missing PNM width".into()))?;
    let h: usize = next_pnm_token(bytes, &mut pos)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| IoError::Parse("Missing PNM height".into()))?;

    let is_bitmap = magic == "P1" || magic == "P4";
    let _maxval: usize = if is_bitmap {
        1
    } else {
        next_pnm_token(bytes, &mut pos)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| IoError::Parse("Missing PNM maxval".into()))?
    };

    let (n_comp, data) = match magic.as_str() {
        "P1" | "P2" => {
            let mut out = Vec::with_capacity(safe_prealloc_capacity(w * h, 1, MAX_ALLOC_BYTES)?);
            for _ in 0..(w * h) {
                let v: u8 = next_pnm_token(bytes, &mut pos)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                out.push(if is_bitmap { if v == 0 { 255 } else { 0 } } else { v });
            }
            (1, out)
        }
        "P3" => {
            let mut out = Vec::with_capacity(safe_prealloc_capacity(w * h, 3, MAX_ALLOC_BYTES)?);
            for _ in 0..(w * h * 3) {
                let v: u8 = next_pnm_token(bytes, &mut pos)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                out.push(v);
            }
            (3, out)
        }
        "P5" => {
            if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let need = w * h;
            if pos + need > bytes.len() {
                return Err(IoError::Parse("Truncated P5 PGM data".into()));
            }
            (1, bytes[pos..pos + need].to_vec())
        }
        "P6" => {
            if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let need = w * h * 3;
            if pos + need > bytes.len() {
                return Err(IoError::Parse("Truncated P6 PPM data".into()));
            }
            (3, bytes[pos..pos + need].to_vec())
        }
        other => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported Netpbm magic '{other}'"
            )));
        }
    };

    let mut img = ImageData::new([w, h, 1], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let name = if n_comp == 3 { "Colors" } else { "Scalars" };
    img.point_data
        .add_array(DataArray::new(name, n_comp, ArrayData::U8(data)));
    Ok(DataSet::ImageData(img))
}

fn next_pnm_token(bytes: &[u8], pos: &mut usize) -> Option<String> {
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
        } else if b.is_ascii_whitespace() {
            *pos += 1;
        } else {
            break;
        }
    }
    if *pos >= bytes.len() {
        return None;
    }
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() && bytes[*pos] != b'#' {
        *pos += 1;
    }
    Some(String::from_utf8_lossy(&bytes[start..*pos]).into_owned())
}
