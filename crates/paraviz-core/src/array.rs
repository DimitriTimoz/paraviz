//! Typed multi-component scientific data arrays (equivalent to `vtkDataArray`).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Scalar storage type of a [`DataArray`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl ScalarType {
    /// Returns the VTK legacy/XML type name.
    pub fn vtk_name(self) -> &'static str {
        match self {
            Self::F32 => "Float32",
            Self::F64 => "Float64",
            Self::I8 => "Int8",
            Self::I16 => "Int16",
            Self::I32 => "Int32",
            Self::I64 => "Int64",
            Self::U8 => "UInt8",
            Self::U16 => "UInt16",
            Self::U32 => "UInt32",
            Self::U64 => "UInt64",
        }
    }

    /// Returns the legacy VTK ASCII scalar type token.
    pub fn vtk_legacy_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F64 => "double",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "vtktypeint64",
            Self::U8 => "unsigned_char",
            Self::U16 => "unsigned_short",
            Self::U32 => "unsigned_int",
            Self::U64 => "vtktypeuint64",
        }
    }

    /// Byte width of one scalar element.
    pub fn byte_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

/// Underlying typed buffer of a [`DataArray`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrayData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl ArrayData {
    /// Number of raw scalar elements (tuples * components).
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
        }
    }

    /// Returns `true` if the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`ScalarType`] of this buffer.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::I8(_) => ScalarType::I8,
            Self::I16(_) => ScalarType::I16,
            Self::I32(_) => ScalarType::I32,
            Self::I64(_) => ScalarType::I64,
            Self::U8(_) => ScalarType::U8,
            Self::U16(_) => ScalarType::U16,
            Self::U32(_) => ScalarType::U32,
            Self::U64(_) => ScalarType::U64,
        }
    }

    /// Reads the scalar element at flat index `idx` as `f64`.
    #[inline]
    pub fn get_f64(&self, idx: usize) -> f64 {
        match self {
            Self::F32(v) => v[idx] as f64,
            Self::F64(v) => v[idx],
            Self::I8(v) => v[idx] as f64,
            Self::I16(v) => v[idx] as f64,
            Self::I32(v) => v[idx] as f64,
            Self::I64(v) => v[idx] as f64,
            Self::U8(v) => v[idx] as f64,
            Self::U16(v) => v[idx] as f64,
            Self::U32(v) => v[idx] as f64,
            Self::U64(v) => v[idx] as f64,
        }
    }

    /// Converts all elements to `Vec<f32>`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32(v) => v.clone(),
            Self::F64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I16(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U16(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Converts all elements to `Vec<f64>`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::F64(v) => v.clone(),
            Self::I8(v) => v.iter().map(|&x| x as f64).collect(),
            Self::I16(v) => v.iter().map(|&x| x as f64).collect(),
            Self::I32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U8(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U16(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
}

/// Named, multi-component scientific data array (scalars, vectors, tensors, normals, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataArray {
    pub name: String,
    pub num_components: usize,
    pub data: ArrayData,
}

impl DataArray {
    /// Creates a new `DataArray` from raw [`ArrayData`].
    pub fn new(name: impl Into<String>, num_components: usize, data: ArrayData) -> Self {
        assert!(num_components >= 1, "num_components must be >= 1");
        assert_eq!(
            data.len() % num_components,
            0,
            "data length ({}) must be divisible by num_components ({})",
            data.len(),
            num_components
        );
        Self {
            name: name.into(),
            num_components,
            data,
        }
    }

    /// Creates a single-component `f32` scalar array.
    pub fn scalars_f32(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self::new(name, 1, ArrayData::F32(values))
    }

    /// Creates a single-component `f64` scalar array.
    pub fn scalars_f64(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self::new(name, 1, ArrayData::F64(values))
    }

    /// Creates a 3-component `f32` vector array from `[f32; 3]` tuples.
    pub fn vectors_f32(name: impl Into<String>, vectors: Vec<[f32; 3]>) -> Self {
        let mut flat = Vec::with_capacity(vectors.len() * 3);
        for v in vectors {
            flat.extend_from_slice(&v);
        }
        Self::new(name, 3, ArrayData::F32(flat))
    }

    /// Creates a `u8` RGBA or RGB color array.
    pub fn colors_u8(name: impl Into<String>, num_components: usize, values: Vec<u8>) -> Self {
        Self::new(name, num_components, ArrayData::U8(values))
    }

    /// Number of tuples (points or cells) in this array.
    #[inline]
    pub fn num_tuples(&self) -> usize {
        self.data.len() / self.num_components
    }

    /// Returns `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns scalar value for `tuple_idx` (component 0) as `f64`.
    #[inline]
    pub fn get_scalar(&self, tuple_idx: usize) -> f64 {
        self.data.get_f64(tuple_idx * self.num_components)
    }

    /// Returns scalar magnitude for `tuple_idx` (exact value for 1-component, L2 norm for multi-component).
    #[inline]
    pub fn get_magnitude(&self, tuple_idx: usize) -> f64 {
        let base = tuple_idx * self.num_components;
        if self.num_components == 1 {
            self.data.get_f64(base)
        } else {
            let mut sum_sq = 0.0;
            for c in 0..self.num_components {
                let v = self.data.get_f64(base + c);
                sum_sq += v * v;
            }
            sum_sq.sqrt()
        }
    }

    /// Reads tuple `tuple_idx` as a `Vec<f64>`.
    pub fn get_tuple(&self, tuple_idx: usize) -> Vec<f64> {
        let base = tuple_idx * self.num_components;
        (0..self.num_components)
            .map(|c| self.data.get_f64(base + c))
            .collect()
    }

    /// Reads 3D vector tuple as `[f32; 3]`. Pads with 0 if `num_components < 3`.
    #[inline]
    pub fn get_vec3(&self, tuple_idx: usize) -> [f32; 3] {
        let base = tuple_idx * self.num_components;
        let x = self.data.get_f64(base) as f32;
        let y = if self.num_components > 1 {
            self.data.get_f64(base + 1) as f32
        } else {
            0.0
        };
        let z = if self.num_components > 2 {
            self.data.get_f64(base + 2) as f32
        } else {
            0.0
        };
        [x, y, z]
    }

    /// Computes `(min, max)` range for component `component` (or magnitude when `component == None`).
    pub fn range(&self, component: Option<usize>) -> (f64, f64) {
        let n = self.num_tuples();
        if n == 0 {
            return (0.0, 1.0);
        }
        (0..n)
            .into_par_iter()
            .map(|i| {
                let val = match component {
                    Some(c) if c < self.num_components => {
                        self.data.get_f64(i * self.num_components + c)
                    }
                    _ => self.get_magnitude(i),
                };
                if val.is_finite() {
                    (val, val)
                } else {
                    (f64::INFINITY, f64::NEG_INFINITY)
                }
            })
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
            )
    }

    /// Extracts scalar magnitudes (or single scalar component) across all tuples as `Vec<f32>`.
    pub fn to_scalar_f32(&self, component: Option<usize>) -> Vec<f32> {
        let n = self.num_tuples();
        (0..n)
            .into_par_iter()
            .map(|i| match component {
                Some(c) if c < self.num_components => {
                    self.data.get_f64(i * self.num_components + c) as f32
                }
                _ => self.get_magnitude(i) as f32,
            })
            .collect()
    }

    /// Samples a subset of tuple indices, producing a new `DataArray`.
    pub fn gather(&self, indices: &[usize]) -> Self {
        let nc = self.num_components;
        macro_rules! gather_typed {
            ($v:expr, $variant:ident) => {{
                let mut out = Vec::with_capacity(indices.len() * nc);
                for &idx in indices {
                    out.extend_from_slice(&$v[idx * nc..(idx + 1) * nc]);
                }
                ArrayData::$variant(out)
            }};
        }
        let data = match &self.data {
            ArrayData::F32(v) => gather_typed!(v, F32),
            ArrayData::F64(v) => gather_typed!(v, F64),
            ArrayData::I8(v) => gather_typed!(v, I8),
            ArrayData::I16(v) => gather_typed!(v, I16),
            ArrayData::I32(v) => gather_typed!(v, I32),
            ArrayData::I64(v) => gather_typed!(v, I64),
            ArrayData::U8(v) => gather_typed!(v, U8),
            ArrayData::U16(v) => gather_typed!(v, U16),
            ArrayData::U32(v) => gather_typed!(v, U32),
            ArrayData::U64(v) => gather_typed!(v, U64),
        };
        Self::new(self.name.clone(), nc, data)
    }

    /// Linearly interpolates between tuples `a` and `b` at factor `t` (`0.0..=1.0`), returning `Vec<f64>`.
    pub fn lerp_tuple(&self, a: usize, b: usize, t: f64) -> Vec<f64> {
        let base_a = a * self.num_components;
        let base_b = b * self.num_components;
        (0..self.num_components)
            .map(|c| {
                let va = self.data.get_f64(base_a + c);
                let vb = self.data.get_f64(base_b + c);
                va + (vb - va) * t
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_range_and_magnitude() {
        let vecs = DataArray::vectors_f32("vel", vec![[3.0, 4.0, 0.0], [0.0, 0.0, 10.0]]);
        assert_eq!(vecs.num_tuples(), 2);
        assert!((vecs.get_magnitude(0) - 5.0).abs() < 1e-6);
        assert!((vecs.get_magnitude(1) - 10.0).abs() < 1e-6);
        let (min, max) = vecs.range(None);
        assert!((min - 5.0).abs() < 1e-6);
        assert!((max - 10.0).abs() < 1e-6);
        let (xmin, xmax) = vecs.range(Some(0));
        assert!((xmin - 0.0).abs() < 1e-6);
        assert!((xmax - 3.0).abs() < 1e-6);
    }
}
