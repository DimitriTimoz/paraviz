//! ParaViz: Kitware ParaView rewritten in Rust using WebGPU (`wgpu`).
//!
//! # Architecture
//! - [`core`] (`paraviz-core`): Scientific data model (`DataArray`, `FieldData`, `CellType`,
//!   `PolyData`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`,
//!   `MultiBlockDataSet`), colormaps, transfer functions, and demand-driven `PipelineGraph`.
//! - [`sources`] (`paraviz-sources`): Procedural geometric & analytical sources (`SphereSource`,
//!   `CubeSource`, `ConeSource`, `CylinderSource`, `PlaneSource`, `LineSource`, `ArrowSource`,
//!   `DiskSource`, `TorusSource`, `WaveletSource`, `VectorFieldSource`, `SuperquadricSource`,
//!   `GyroidSource`, `PointCloudSource`).
//! - [`filters`] (`paraviz-filters`): Parallel (`rayon`) scientific visualization filters
//!   (`ExtractSurface`, `ContourFilter`, `SliceFilter`, `ClipFilter`, `ThresholdFilter`,
//!   `StreamTracerFilter`, `TubeFilter`, `GlyphFilter`, `WarpByVectorFilter`,
//!   `WarpByScalarFilter`, `GradientFilter`, `CalculatorFilter`, `SmoothFilter`,
//!   `TransformFilter`, `ProbeOverLineFilter`).
//! - [`io`] (`paraviz-io`): Legacy VTK (`.vtk`), VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`,
//!   `.vtm`, `.pvd`), `.stl`, `.obj`, `.ply`, and `.csv` readers/writers.
//! - [`render`] (`paraviz-render`): WebGPU (`wgpu`) rendering engine supporting Surface,
//!   single-pass barycentric SurfaceWithEdges, Wireframe, Points (sphere impostors),
//!   3D Volume raymarching, Outline, and 3D Orientation Axes Gizmo.
//! - [`session`] (`ParaViz`): Ergonomic high-level Rust API for building pipelines, styling
//!   actors, saving/loading state, generating Rust traces, and headless PNG/RGBA rendering.
//!
//! # Quick Start
//! ```no_run
//! use paraviz::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut pv = ParaViz::new();
//!     let wav = pv.add_source("Wavelet", WaveletSource::default());
//!     let iso = pv.add_filter("Contour", wav, ContourFilter::new("RTData", vec![120.0, 180.0]))?;
//!     pv.show(iso)
//!         .representation(Representation::SurfaceWithEdges)
//!         .colormap(ColorMapPreset::Viridis)
//!         .color_by_point("RTData");
//!     pv.camera_preset(CameraPreset::Isometric)?;
//!     pv.render_png("wavelet.png", 1280, 720)?;
//!     Ok(())
//! }
//! ```

pub use paraviz_core as core;
pub use paraviz_filters as filters;
pub use paraviz_io as io;
pub use paraviz_render as render;
pub use paraviz_sources as sources;

pub mod demos;
pub mod session;

pub use demos::DemoPreset;
pub use session::{
    DisplayHandle, FilterConfig, InMemorySource, ParaViz, PipelineState, SessionError, SessionNode,
    SourceConfig,
};

/// Convenient prelude re-exporting common types for ParaViz Rust scripts and applications.
pub mod prelude {
    pub use crate::core::{
        AttributeAssociation, BoundingBox, CellType, ColorMap, ColorMapPreset, DataArray, DataSet,
        FieldData, Filter, ImageData, MultiBlockDataSet, NodeId, OpacityPoint, PipelineGraph,
        PolyData, RectilinearGrid, Source, StructuredGrid, TransferFunction, UnstructuredGrid,
    };
    pub use crate::demos::DemoPreset;
    pub use crate::filters::{
        CalculatorFilter, CellDataToPointDataFilter, ClipFilter, ContourFilter, ExtractSurface,
        GlyphFilter, GlyphShape, GradientFilter, ImplicitFunction, IntegrationDirection,
        Integrator, PointDataToCellDataFilter, ProbeOverLineFilter, ScaleMode, SeedSource,
        SliceFilter, SmoothFilter, SmoothingMethod, StreamTracerFilter, ThresholdFilter,
        TransformFilter, TubeFilter, WarpByScalarFilter, WarpByVectorFilter,
    };
    pub use crate::io::{
        read_dataset, read_pvd_series, write_dataset, write_pvd_series, PvdSeriesSource,
        VtkFileSource,
    };
    pub use crate::render::{
        ActorStyle, Camera, CameraPreset, ColorBy, RenderScene, Representation, SceneActor,
        SceneSettings, WebGpuRenderer,
    };
    pub use crate::session::{FilterConfig, ParaViz, SessionError, SourceConfig};
    pub use crate::sources::{
        ArrowSource, ConeSource, CubeSource, CylinderSource, DiskSource, GyroidSource, LineSource,
        PlaneSource, PointCloudSource, SphereSource, SuperquadricSource, TorusSource,
        VectorFieldKind, VectorFieldSource, WaveletSource,
    };
}
