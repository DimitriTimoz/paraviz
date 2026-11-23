//! ParaViz parallel scientific visualization filters.
//!
//! Includes surface extraction, Marching Cubes/Tetrahedra contouring, plane slicing,
//! implicit-function clipping, scalar thresholding, RK4/RK45 streamline tracing,
//! parallel-transport tubes, 3D glyphs, vector/scalar warping, gradients/vorticity/Q-criterion,
//! expression calculator, Laplacian/Taubin smoothing, affine transforms, probing,
//! feature edges, connectivity, shrinking, cell centers, reflection, normals generation,
//! elevation scalars, dataset appending, and histograms.

pub mod append;
pub mod calculator;
pub mod cell_centers;
pub mod clean;
pub mod clip;
pub mod connectivity;
pub mod contour;
pub mod elevation;
pub mod extract_edges;
pub mod feature_edges;
pub mod glyph;
pub mod gradient;
pub mod histogram;
pub mod normals;
pub mod probe;
pub mod reflect;
pub mod shrink;
pub mod slice;
pub mod smooth;
pub mod streamlines;
pub mod surface;
pub mod threshold;
pub mod transform;
pub mod triangulate;
pub mod tube;
mod util;
pub mod warp;

pub use append::{append_datasets, AppendDatasetsFilter};
pub use calculator::CalculatorFilter;
pub use cell_centers::CellCentersFilter;
pub use clean::CleanPolyDataFilter;
pub use clip::{ClipFilter, ImplicitFunction};
pub use connectivity::ConnectivityFilter;
pub use contour::ContourFilter;
pub use elevation::ElevationFilter;
pub use extract_edges::ExtractEdgesFilter;
pub use feature_edges::FeatureEdgesFilter;
pub use glyph::{GlyphFilter, GlyphShape, ScaleMode};
pub use gradient::GradientFilter;
pub use histogram::{compute_histogram, HistogramResult};
pub use normals::GenerateNormalsFilter;
pub use probe::{CellDataToPointDataFilter, PointDataToCellDataFilter, ProbeOverLineFilter};
pub use reflect::{ReflectFilter, ReflectPlane};
pub use shrink::ShrinkFilter;
pub use slice::SliceFilter;
pub use smooth::{SmoothFilter, SmoothingMethod};
pub use streamlines::{IntegrationDirection, Integrator, SeedSource, StreamTracerFilter};
pub use surface::{extract_surface, ExtractSurface};
pub use threshold::ThresholdFilter;
pub use transform::TransformFilter;
pub use triangulate::TriangulateFilter;
pub use tube::TubeFilter;
pub use warp::{WarpByScalarFilter, WarpByVectorFilter};
