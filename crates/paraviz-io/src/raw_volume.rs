//! Raw binary 3D volume reader (`.raw`, `.bin`, `.vol`) producing [`ImageData`].

use crate::error::{IoError, Result};
use paraviz_core::{ArrayData, DataArray, ImageData, ScalarType};
use std::fs;
use std::path::Path;

/// Byte endianness for raw binary volume files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RawEndian {
    #[default]
    Little,
    Big,
}

/// Extended raw scalar storage format including 16-bit integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawScalarType {
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    F32,
    F64,
}

impl From<ScalarType> for RawScalarType {
    fn from(st: ScalarType) -> Self {
        match st {
            ScalarType::U8 => Self::U8,
            ScalarType::I32 => Self::I32,
            ScalarType::U32 => Self::U32,
            ScalarType::I64 => Self::I64,
            ScalarType::F32 => Self::F32,
            ScalarType::F64 => Self::F64,
        }
    }
}

impl RawScalarType {
    fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

/// Metadata describing how to interpret a headerless (or fixed-header) raw binary 3D volume.
#[derive(Debug, Clone, PartialEq)]
pub struct RawVolumeDescriptor {
    pub dimensions: [usize; 3],
    pub scalar_type: ScalarType,
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub endian: RawEndian,
    pub header_bytes: usize,
    pub array_name: String,
}

impl Default for RawVolumeDescriptor {
    fn default() -> Self {
        Self {
            dimensions: [1, 1, 1],
            scalar_type: ScalarType::U8,
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            endian: RawEndian::Little,
            header_bytes: 0,
            array_name: "Scalars".to_string(),
        }
    }
}

/// Reads a raw binary 3D volume into an [`ImageData`] using `desc.scalar_type`.
pub fn read_raw_volume(path: impl AsRef<Path>, desc: &RawVolumeDescriptor) -> Result<ImageData> {
    read_raw_volume_with_type(path, desc, desc.scalar_type.into())
}

/// Reads a signed 16-bit integer raw binary volume into [`ImageData`] (stored as `I32`).
pub fn read_raw_volume_i16(path: impl AsRef<Path>, desc: &RawVolumeDescriptor) -> Result<ImageData> {
    read_raw_volume_with_type(path, desc, RawScalarType::I16)
}

/// Reads an unsigned 16-bit integer raw binary volume into [`ImageData`] (stored as `U32`).
pub fn read_raw_volume_u16(path: impl AsRef<Path>, desc: &RawVolumeDescriptor) -> Result<ImageData> {
    read_raw_volume_with_type(path, desc, RawScalarType::U16)
}

/// Reads a raw binary 3D volume using an explicit [`RawScalarType`] (supporting 16-bit integers).
pub fn read_raw_volume_with_type(
    path: impl AsRef<Path>,
    desc: &RawVolumeDescriptor,
    raw_type: RawScalarType,
) -> Result<ImageData> {
    let bytes = fs::read(path)?;
    read_raw_volume_bytes(&bytes, desc, raw_type)
}

/// Parses raw binary volume bytes into an [`ImageData`].
pub fn read_raw_volume_bytes(
    bytes: &[u8],
    desc: &RawVolumeDescriptor,
    raw_type: RawScalarType,
) -> Result<ImageData> {
    let [nx, ny, nz] = desc.dimensions;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(IoError::InvalidData(format!(
            "Raw volume dimensions must be non-zero, got {:?}",
            desc.dimensions
        )));
    }
    let num_voxels = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| IoError::InvalidData("Raw volume dimensions overflow".into()))?;
    let elem_size = raw_type.byte_size();
    let payload_bytes = num_voxels
        .checked_mul(elem_size)
        .ok_or_else(|| IoError::InvalidData("Raw volume byte size overflow".into()))?;
    let required_bytes = desc
        .header_bytes
        .checked_add(payload_bytes)
        .ok_or_else(|| IoError::InvalidData("Raw volume total byte size overflow".into()))?;

    if bytes.len() < required_bytes {
        return Err(IoError::InvalidData(format!(
            "Raw volume truncated: expected at least {required_bytes} bytes, got {}",
            bytes.len()
        )));
    }

    let payload = &bytes[desc.header_bytes..required_bytes];
    let array_data = decode_raw_payload(payload, raw_type, desc.endian);

    let mut image = ImageData::new(desc.dimensions, desc.origin, desc.spacing);
    let name = if desc.array_name.is_empty() {
        "Scalars"
    } else {
        desc.array_name.as_str()
    };
    image
        .point_data
        .set_scalars(DataArray::new(name, 1, array_data));
    Ok(image)
}

fn decode_raw_payload(payload: &[u8], raw_type: RawScalarType, endian: RawEndian) -> ArrayData {
    let le = endian == RawEndian::Little;
    match raw_type {
        RawScalarType::U8 => ArrayData::U8(payload.to_vec()),
        RawScalarType::I16 => ArrayData::I32(
            payload
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i16::from_le_bytes(c) as i32
                    } else {
                        i16::from_be_bytes(c) as i32
                    }
                })
                .collect(),
        ),
        RawScalarType::U16 => ArrayData::U32(
            payload
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        u16::from_le_bytes(c) as u32
                    } else {
                        u16::from_be_bytes(c) as u32
                    }
                })
                .collect(),
        ),
        RawScalarType::I32 => ArrayData::I32(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i32::from_le_bytes(c)
                    } else {
                        i32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        RawScalarType::U32 => ArrayData::U32(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        u32::from_le_bytes(c)
                    } else {
                        u32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        RawScalarType::I64 => ArrayData::I64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        i64::from_le_bytes(c)
                    } else {
                        i64::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        RawScalarType::F32 => ArrayData::F32(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        f32::from_le_bytes(c)
                    } else {
                        f32::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
        RawScalarType::F64 => ArrayData::F64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| {
                    if le {
                        f64::from_le_bytes(c)
                    } else {
                        f64::from_be_bytes(c)
                    }
                })
                .collect(),
        ),
    }
}
