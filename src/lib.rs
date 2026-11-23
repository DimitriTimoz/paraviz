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
//!   `IsoVolumeFilter`, `StreamTracerFilter`, `TubeFilter`, `GlyphFilter`, `WarpByVectorFilter`,
//!   `WarpByScalarFilter`, `GradientFilter`, `CalculatorFilter`, `SmoothFilter`, `DecimateFilter`,
//!   `Delaunay2DFilter`, `Delaunay3DFilter`, `ResampleToImageFilter`, `ResampleWithDatasetFilter`,
//!   `IntegrateVariablesFilter`, `ExtractBlockFilter`, `TemporalStatisticsFilter`,
//!   `TransformFilter`, `ProbeOverLineFilter`, `FeatureEdgesFilter`, `ConnectivityFilter`,
//!   `ShrinkFilter`, `CellCentersFilter`, `ReflectFilter`, `GenerateNormalsFilter`,
//!   `ElevationFilter`, `TriangulateFilter`, `ExtractEdgesFilter`, `CleanPolyDataFilter`,
//!   `AppendDatasetsFilter`).
//! - [`io`] (`paraviz-io`): Legacy VTK (`.vtk`), VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`,
//!   `.vtm`, `.pvd`, `.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts`), `.stl`, `.obj`, `.ply`, `.csv`/`.tsv`,
//!   binary glTF 2.0 (`.glb`), raw 3D volumes (`.raw`), NIfTI-1 (`.nii`, `.nii.gz`),
//!   NRRD (`.nrrd`, `.nhdr`), MetaImage (`.mhd`, `.mha`), EnSight Gold (`.case`), XDMF (`.xdmf`),
//!   and OpenFOAM `polyMesh` readers/writers.
//! - [`render`] (`paraviz-render`): WebGPU (`wgpu`) rendering engine supporting `Surface`,
//!   single-pass barycentric `SurfaceWithEdges`, `Wireframe`, `Points` (sphere impostors),
//!   `PointGaussian`, 3D `Volume` raymarching (`R32Float` + IGN jittering), `Outline`,
//!   4x MSAA, Eye-Dome Lighting (`EDL`), 3D raycast picking (`pick_scene`), GPU Compute Marching
//!   Cubes (`GpuContourEngine`), 3D `ViewportWidget` overlays, and 3D Orientation Axes Gizmo.
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
        FieldData, Filter, ImageData, MultiBlockDataSet, MultiInputFilter, NodeId, OpacityPoint, PipelineGraph,
        PolyData, RectilinearGrid, Source, StructuredGrid, TransferFunction, UnstructuredGrid,
    };
    pub use crate::demos::DemoPreset;
    pub use crate::filters::{
        append_datasets, compute_histogram, interpolate_datasets, AddTimeStepsFilter,
        AppendDatasetsFilter, CalculatorFilter, CellCentersFilter, CellDataToPointDataFilter,
        CleanPolyDataFilter, ClipFilter, ConnectivityFilter, ContourFilter, DecimateFilter,
        Delaunay2DFilter, Delaunay3DFilter, ElevationFilter, ExtractBlockFilter,
        ExtractEdgesFilter, ExtractSurface, FeatureEdgesFilter, GenerateNormalsFilter, GlyphFilter,
        GlyphShape, GradientFilter, HistogramResult, ImplicitFunction, IntegrateVariablesFilter,
        IntegrationDirection, Integrator, IsoVolumeFilter, PointDataToCellDataFilter,
        ProbeOverLineFilter, ReflectFilter, ReflectPlane, ResampleToImageFilter,
        ResampleWithDatasetFilter, ScaleMode, SeedSource, ShrinkFilter, SliceFilter, SmoothFilter,
        SmoothingMethod, StreamTracerFilter, StreamTracerWithCustomSource,
        TemporalStatisticsFilter, ThresholdFilter, TransformFilter, TriangulateFilter, TubeFilter,
        WarpByScalarFilter, WarpByVectorFilter,
    };
    pub use crate::io::{
        encode_animated_gif_bytes, encode_html_viewer, inspect_exodus_time_steps,
        inspect_vtkhdf_time_steps, read_cgns, read_dataset, read_ensight_case, read_exodus,
        read_exodus_at_time, read_metaimage, read_nifti, read_nrrd, read_openfoam,
        read_pvd_series, read_raw_volume, read_vtkhdf, read_vtkhdf_at_time, read_xdmf,
        write_animated_gif, write_dataset, write_glb, write_html_viewer, write_metaimage,
        write_nifti, write_pvd_series, DatasetReader, PvdSeriesSource, RawEndian, RawScalarType,
        RawVolumeDescriptor, VtkFileSource,
    };
    pub use crate::render::{
        pick_scene, ActorStyle, Camera, CameraPreset, ColorBy, GpuContourEngine, PickResult,
        RenderScene, Representation, ScalarBarConfig, SceneActor, SceneSettings,
        ViewportAnnotation, ViewportWidget, WebGpuRenderer,
    };
    pub use crate::session::{FilterConfig, ParaViz, SessionError, SourceConfig};
    pub use crate::sources::{
        ArrowSource, ConeSource, CubeSource, CylinderSource, DiskSource, GyroidSource, LineSource,
        PlaneSource, PointCloudSource, SphereSource, SuperquadricSource, TorusSource,
        VectorFieldKind, VectorFieldSource, WaveletSource,
    };
}
