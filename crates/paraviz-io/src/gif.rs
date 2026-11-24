//! Pure-Rust animated GIF89a encoder (with Netscape 2.0 loop extension and LZW compression)
//! for exporting ParaViz turntable and time-series animations (ParaView Discourse GIF wishlist).

use crate::error::{IoError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const TRANSPARENT_IDX: u8 = 255;

/// Writes an animated GIF89a file from a sequence of tightly-packed `Rgba8` frames.
///
/// `delay_cs` is the per-frame delay in centiseconds (`1/100` s, e.g., `5` = 20 FPS).
pub fn write_animated_gif(
    path: impl AsRef<Path>,
    frames_rgba: &[Vec<u8>],
    width: u32,
    height: u32,
    delay_cs: u16,
) -> Result<()> {
    let bytes = encode_animated_gif_bytes(frames_rgba, width, height, delay_cs)?;
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Encodes a sequence of `Rgba8` frames into animated GIF89a bytes.
pub fn encode_animated_gif_bytes(
    frames_rgba: &[Vec<u8>],
    width: u32,
    height: u32,
    delay_cs: u16,
) -> Result<Vec<u8>> {
    if frames_rgba.is_empty() {
        return Err(IoError::InvalidData("Cannot encode empty GIF frame list".into()));
    }
    let w16 = u16::try_from(width)
        .map_err(|_| IoError::InvalidData("GIF width exceeds 65535".into()))?;
    let h16 = u16::try_from(height)
        .map_err(|_| IoError::InvalidData("GIF height exceeds 65535".into()))?;
    let expected_len = (width as usize) * (height as usize) * 4;
    for (i, frame) in frames_rgba.iter().enumerate() {
        if frame.len() != expected_len {
            return Err(IoError::InvalidData(format!(
                "GIF frame {i} byte length {} != expected {}",
                frame.len(),
                expected_len
            )));
        }
    }

    let palette = build_global_palette();
    let has_transparency = frames_rgba
        .iter()
        .any(|f| f.as_chunks::<4>().0.iter().any(|px| px[3] < 128));

    let mut out = Vec::with_capacity(frames_rgba.len() * (width as usize * height as usize / 4));
    // Header
    out.extend_from_slice(b"GIF89a");
    // Logical Screen Descriptor
    out.extend_from_slice(&w16.to_le_bytes());
    out.extend_from_slice(&h16.to_le_bytes());
    // GCT flag (1) | color res (7 -> 0b111) | sort (0) | GCT size (7 -> 256 colors)
    out.push(0xF7);
    out.push(0); // Background color index
    out.push(0); // Pixel aspect ratio

    // Global Color Table (256 * 3 bytes)
    for rgb in &palette {
        out.extend_from_slice(rgb);
    }

    // Netscape 2.0 Application Extension for infinite looping
    if frames_rgba.len() > 1 {
        out.push(0x21);
        out.push(0xFF);
        out.push(11);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.push(3);
        out.push(1);
        out.extend_from_slice(&0u16.to_le_bytes()); // loop count 0 = infinite
        out.push(0);
    }

    let disposal = if has_transparency { 2u8 } else { 0u8 };
    let gce_flags = (disposal << 2) | if has_transparency { 1 } else { 0 };

    for frame in frames_rgba {
        // Graphic Control Extension
        out.push(0x21);
        out.push(0xF9);
        out.push(4);
        out.push(gce_flags);
        out.extend_from_slice(&delay_cs.to_le_bytes());
        out.push(if has_transparency { TRANSPARENT_IDX } else { 0 });
        out.push(0);

        // Image Descriptor
        out.push(0x2C);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&w16.to_le_bytes());
        out.extend_from_slice(&h16.to_le_bytes());
        out.push(0); // No local color table, not interlaced

        let indices = quantize_frame(frame);
        write_lzw_image_data(&mut out, 8, &indices);
    }

    // GIF Trailer
    out.push(0x3B);
    Ok(out)
}

/// 256-entry palette:
/// - 0..216: 6x6x6 RGB cube
/// - 216..255: 39 grayscale ramp entries (leaving 255 for optional transparency)
fn build_global_palette() -> [[u8; 3]; 256] {
    let mut pal = [[0u8; 3]; 256];
    let levels = [0u8, 51, 102, 153, 204, 255];
    let mut idx = 0usize;
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                pal[idx] = [r, g, b];
                idx += 1;
            }
        }
    }
    for i in 0..39 {
        let v = ((i * 255) / 38) as u8;
        pal[216 + i] = [v, v, v];
    }
    pal[255] = [0, 0, 0];
    pal
}

#[inline]
fn quantize_frame(rgba: &[u8]) -> Vec<u8> {
    let mut indices = Vec::with_capacity(rgba.len() / 4);
    for px in rgba.as_chunks::<4>().0 {
        if px[3] < 128 {
            indices.push(TRANSPARENT_IDX);
            continue;
        }
        let r = px[0];
        let g = px[1];
        let b = px[2];
        // Use grayscale ramp when near-achromatic for smooth shading
        let max_c = r.max(g).max(b);
        let min_c = r.min(g).min(b);
        if max_c - min_c <= 8 {
            let luma = ((r as u16 * 77 + g as u16 * 150 + b as u16 * 29) >> 8) as usize;
            let g_idx = (luma * 38 + 127) / 255;
            indices.push((216 + g_idx.min(38)) as u8);
        } else {
            let ri = ((r as u16 * 5 + 127) / 255) as u8;
            let gi = ((g as u16 * 5 + 127) / 255) as u8;
            let bi = ((b as u16 * 5 + 127) / 255) as u8;
            indices.push(ri * 36 + gi * 6 + bi);
        }
    }
    indices
}

fn write_lzw_image_data(out: &mut Vec<u8>, min_code_size: u8, pixels: &[u8]) {
    out.push(min_code_size);
    let clear_code: u16 = 1 << min_code_size;
    let eoi_code: u16 = clear_code + 1;

    let mut sub_blocks = Vec::new();
    let mut bit_buf: u32 = 0;
    let mut bit_count: u8 = 0;

    let emit_bits = |code: u16,
                     code_size: u8,
                     bit_buf: &mut u32,
                     bit_count: &mut u8,
                     buf: &mut Vec<u8>| {
        *bit_buf |= (code as u32) << *bit_count;
        *bit_count += code_size;
        while *bit_count >= 8 {
            buf.push((*bit_buf & 0xFF) as u8);
            *bit_buf >>= 8;
            *bit_count -= 8;
        }
    };

    let mut code_size = min_code_size + 1;
    let mut next_code = eoi_code + 1;
    let mut dict: HashMap<u32, u16> = HashMap::with_capacity(4096);

    emit_bits(clear_code, code_size, &mut bit_buf, &mut bit_count, &mut sub_blocks);

    if let Some((&first, rest)) = pixels.split_first() {
        let mut prefix = first as u16;
        for &p in rest {
            let key = ((prefix as u32) << 8) | (p as u32);
            if let Some(&found) = dict.get(&key) {
                prefix = found;
            } else {
                emit_bits(prefix, code_size, &mut bit_buf, &mut bit_count, &mut sub_blocks);
                if next_code < 4096 {
                    dict.insert(key, next_code);
                    if next_code == (1u16 << code_size) && code_size < 12 {
                        code_size += 1;
                    }
                    next_code += 1;
                } else {
                    emit_bits(clear_code, code_size, &mut bit_buf, &mut bit_count, &mut sub_blocks);
                    dict.clear();
                    code_size = min_code_size + 1;
                    next_code = eoi_code + 1;
                }
                prefix = p as u16;
            }
        }
        emit_bits(prefix, code_size, &mut bit_buf, &mut bit_count, &mut sub_blocks);
    }

    emit_bits(eoi_code, code_size, &mut bit_buf, &mut bit_count, &mut sub_blocks);
    if bit_count > 0 {
        sub_blocks.push((bit_buf & 0xFF) as u8);
    }

    for chunk in sub_blocks.chunks(255) {
        out.push(chunk.len() as u8);
        out.extend_from_slice(chunk);
    }
    out.push(0); // Block terminator
}
