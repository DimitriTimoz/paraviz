//! Security limits, bounded decompression, safe path resolution, and checked allocation helpers.

use crate::error::{IoError, Result};
use flate2::read::{GzDecoder, ZlibDecoder};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

/// Maximum decompressed payload size accepted for a single compressed block/stream (512 MiB).
pub const MAX_DECOMPRESSED_BYTES: usize = 512 * 1024 * 1024;

/// Maximum total allocation size for a single parsed array/buffer (1 GiB).
pub const MAX_ALLOC_BYTES: usize = 1024 * 1024 * 1024;

/// Maximum elements to pre-allocate with `Vec::with_capacity` before reading actual data.
pub const MAX_PREALLOC_ELEMS: usize = 1_000_000;

/// Maximum XML nesting depth to prevent stack/memory exhaustion on deeply nested documents.
pub const MAX_XML_DEPTH: usize = 256;

/// Resolves `rel_path` strictly inside `base_dir`, rejecting absolute paths, `..` traversal,
/// and symlinks that escape `base_dir`.
pub fn resolve_safe_path(base_dir: &Path, rel_path: &str) -> Result<PathBuf> {
    let trimmed = rel_path.trim();
    if trimmed.is_empty() || trimmed.contains('\0') {
        return Err(IoError::InvalidData(
            "Referenced relative path is empty or contains NUL bytes".into(),
        ));
    }

    let rel = Path::new(trimmed);
    if rel.is_absolute() {
        return Err(IoError::InvalidData(format!(
            "Disallowed absolute path reference '{trimmed}'"
        )));
    }

    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(IoError::InvalidData(format!(
                    "Disallowed path traversal in reference '{trimmed}'"
                )));
            }
        }
    }

    let base = if base_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base_dir
    };
    let joined = base.join(rel);

    if joined.exists() && base.exists() {
        let canon_base = std::fs::canonicalize(base)?;
        let canon_target = std::fs::canonicalize(&joined)?;
        if !canon_target.starts_with(&canon_base) {
            return Err(IoError::InvalidData(format!(
                "Referenced path '{trimmed}' escapes base directory"
            )));
        }
        return Ok(canon_target);
    }

    Ok(joined)
}

/// Reads from `reader` up to `max_bytes`, failing with `InvalidData` if more bytes are produced.
pub fn read_to_end_limited<R: Read>(reader: R, max_bytes: usize) -> io::Result<Vec<u8>> {
    let limit = (max_bytes as u64).saturating_add(1);
    let mut buf = Vec::new();
    reader.take(limit).read_to_end(&mut buf)?;
    if buf.len() > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Decompressed payload exceeds maximum allowed size of {max_bytes} bytes"),
        ));
    }
    Ok(buf)
}

/// Decompresses a zlib stream with an upper bound of `max_bytes`.
pub fn decompress_zlib_limited(data: &[u8], max_bytes: usize) -> Result<Vec<u8>> {
    let dec = ZlibDecoder::new(data);
    Ok(read_to_end_limited(dec, max_bytes)?)
}

/// Decompresses a gzip stream with an upper bound of `max_bytes`.
pub fn decompress_gzip_limited(data: &[u8], max_bytes: usize) -> Result<Vec<u8>> {
    let dec = GzDecoder::new(data);
    Ok(read_to_end_limited(dec, max_bytes)?)
}

struct LimitedWriter<'a> {
    buf: &'a mut Vec<u8>,
    start_len: usize,
    max_bytes: usize,
}

impl<'a> Write for LimitedWriter<'a> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let written = self.buf.len() - self.start_len;
        if written.saturating_add(data.len()) > self.max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Decompressed LZMA/XZ payload exceeds maximum allowed size of {} bytes",
                    self.max_bytes
                ),
            ));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Decompresses an XZ or raw LZMA stream into `out`, enforcing `max_bytes` on the output.
pub fn decompress_lzma_or_xz_limited(
    data: &[u8],
    out: &mut Vec<u8>,
    max_bytes: usize,
) -> Result<()> {
    let start_len = out.len();
    {
        let mut cur = io::Cursor::new(data);
        let mut writer = LimitedWriter {
            buf: out,
            start_len,
            max_bytes,
        };
        if lzma_rs::xz_decompress(&mut cur, &mut writer).is_ok() {
            return Ok(());
        }
    }
    out.truncate(start_len);
    let mut cur2 = io::Cursor::new(data);
    let mut writer2 = LimitedWriter {
        buf: out,
        start_len,
        max_bytes,
    };
    lzma_rs::lzma_decompress(&mut cur2, &mut writer2)
        .map_err(|e| IoError::InvalidData(format!("VTK LZMA decompress error: {e}")))
}

/// Validates that `count * elem_size` does not overflow `usize` and does not exceed `max_bytes`.
pub fn check_alloc_bytes(count: usize, elem_size: usize, max_bytes: usize) -> Result<usize> {
    let total = count.checked_mul(elem_size).ok_or_else(|| {
        IoError::InvalidData(format!(
            "Allocation size overflow: {count} * {elem_size} bytes"
        ))
    })?;
    if total > max_bytes {
        return Err(IoError::InvalidData(format!(
            "Requested allocation ({total} bytes) exceeds maximum allowed ({max_bytes} bytes)"
        )));
    }
    Ok(total)
}

/// Validates `count * elem_size <= max_bytes` and returns a safe initial `Vec::with_capacity`
/// capped at [`MAX_PREALLOC_ELEMS`] so malformed headers cannot trigger huge upfront allocations.
pub fn safe_prealloc_capacity(
    count: usize,
    elem_size: usize,
    max_bytes: usize,
) -> Result<usize> {
    check_alloc_bytes(count, elem_size, max_bytes)?;
    Ok(count.min(MAX_PREALLOC_ELEMS))
}
