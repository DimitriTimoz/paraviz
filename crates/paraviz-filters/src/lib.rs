//! ParaViz parallel scientific visualization filters.
//!
//! Includes surface extraction, Marching Cubes/Tetrahedra contouring, plane slicing,
//! implicit-function clipping, scalar thresholding, RK4/RK45 streamline tracing,
//! parallel-transport tubes, 3D glyphs, vector/scalar warping, gradients/vorticity/Q-criterion,
//! expression calculator, Laplacian/Taubin smoothing, affine transforms, and probing.

pub mod calculator;
pub mod clip;
pub mod contour;
pub mod glyph;
pub mod gradient;
pub mod probe;
pub mod slice;
pub mod smooth;
pub mod streamlines;
pub mod surface;
pub mod threshold;
pub mod transform;
pub mod tube;
mod util;
pub mod warp;

pub use calculator::CalculatorFilter;
pub use clip::{ClipFilter, ImplicitFunction};
pub use contour::ContourFilter;
pub use glyph::{GlyphFilter, GlyphShape, ScaleMode};
pub use gradient::GradientFilter;
pub use probe::{CellDataToPointDataFilter, PointDataToCellDataFilter, ProbeOverLineFilter};
pub use slice::SliceFilter;
pub use smooth::{SmoothFilter, SmoothingMethod};
pub use streamlines::{IntegrationDirection, Integrator, SeedSource, StreamTracerFilter};
pub use surface::{extract_surface, ExtractSurface};
pub use threshold::ThresholdFilter;
pub use transform::TransformFilter;
pub use tube::TubeFilter;
pub use warp::{WarpByScalarFilter, WarpByVectorFilter};
