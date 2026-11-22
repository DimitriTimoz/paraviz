//! Core scientific data model, colormaps, transfer functions, and demand-driven pipeline
//! for ParaViz (a Rust + WebGPU reimplementation of Kitware ParaView / VTK).

pub mod array;
pub mod cell;
pub mod colormap;
pub mod dataset;
pub mod field;
pub mod pipeline;

pub use array::{ArrayData, DataArray, ScalarType};
pub use cell::CellType;
pub use colormap::{ColorMap, ColorMapPreset, OpacityPoint, TransferFunction};
pub use dataset::{
    BoundingBox, DataSet, ImageData, MultiBlockDataSet, PolyData, RectilinearGrid, StructuredGrid,
    UnstructuredGrid,
};
pub use field::{AttributeAssociation, FieldData};
pub use pipeline::{
    BoxedFilter, BoxedSource, Filter, NodeId, PipelineError, PipelineGraph, PipelineInfo, Source,
};

use thiserror::Error;

/// Top-level error type for `paraviz-core`.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Array '{name}' not found")]
    ArrayNotFound { name: String },

    #[error("Component mismatch for array '{name}': expected {expected}, got {actual}")]
    ComponentMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("Length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("Unsupported dataset type for operation: {0}")]
    UnsupportedDataset(String),

    #[error("Invalid topology or dimensions: {0}")]
    InvalidGeometry(String),

    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),
}

pub type Result<T> = std::result::Result<T, CoreError>;
