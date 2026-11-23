//! Self-contained mesh, line, point, outline, and 3D volume extraction from `paraviz-core` datasets.

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::scene::{ActorStyle, ColorBy, ViewportWidget};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use paraviz_core::{
    BoundingBox, CellType, DataArray, DataSet, FieldData, ImageData, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 4096;

/// Sentinel `color_override` indicating per-actor uniform / scalar colormap should be used.
pub const NO_COLOR_OVERRIDE: [f32; 4] = [-1.0, -1.0, -1.0, -1.0];
/// Neutral fallback solid color when a block has no matching scalar array.
pub const DEFAULT_FALLBACK_SOLID: [f32; 4] = [0.82, 0.84, 0.88, 1.0];

mod surface;
mod wireframe_points;

pub use surface::{
    extract_indexed_surface_scalars, prepare_surface, prepare_surface_indexed,
    prepare_surface_indexed_with_style, prepare_surface_with_style,
};
pub use wireframe_points::{
    build_axes_gizmo_mesh, prepare_points, prepare_points_with_style, prepare_volume,
    prepare_volume_with_style, prepare_widgets, prepare_wireframe, prepare_wireframe_with_style,
    GizmoVertex, PreparedWidgets, WidgetVertex,
};

/// Non-indexed triangle vertex for SurfaceWithEdges rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub scalar: f32,
    pub barycentric: [f32; 3],
    pub color_override: [f32; 4],
}

/// Geometry-only vertex (position + normal + optional per-block color override) for indexed Surface rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceGeomVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color_override: [f32; 4],
}

/// Describes how an indexed surface's vertices map back to dataset point scalars for fast re-coloring.
#[derive(Debug, Clone)]
pub enum SurfaceScalarSource {
    /// GPU vertex `i` maps directly to `PolyData.points[i]`.
    PointDirect,
    /// GPU vertex `v` maps to dataset point index `point_indices[v]`.
    PointMapped(Vec<u32>),
    /// Cell scalars or multi-block datasets require rebuilding geometry when `color_by` changes.
    RebuildRequired,
}

/// Indexed surface mesh with split geometry and scalar streams plus `u32` triangle indices.
#[derive(Debug, Clone)]
pub struct PreparedIndexedSurface {
    pub geom_vertices: Vec<SurfaceGeomVertex>,
    pub scalars: Vec<f32>,
    pub indices: Vec<u32>,
    pub scalar_source: SurfaceScalarSource,
    pub scalar_range: (f64, f64),
}

/// Line vertex for Wireframe and Outline rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub scalar: f32,
    pub color_override: [f32; 4],
}

/// Per-point instance for sphere-impostor point rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub scalar: f32,
    pub color_override: [f32; 4],
}

/// Prepared 3D volume payload for `R32Float` texture upload.
#[derive(Debug, Clone)]
pub struct PreparedVolume {
    pub dims: [u32; 3],
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    pub data: Vec<f32>,
    pub scalar_range: (f64, f64),
}

/// Prepared surface triangle list with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedSurface {
    pub vertices: Vec<SurfaceVertex>,
    pub scalar_range: (f64, f64),
}

/// Prepared wireframe line list with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedLines {
    pub vertices: Vec<LineVertex>,
    pub scalar_range: (f64, f64),
}

/// Prepared point cloud instances with auto-computed scalar range.
#[derive(Debug, Clone)]
pub struct PreparedPoints {
    pub instances: Vec<PointInstance>,
    pub scalar_range: (f64, f64),
}

/// Resolves a scalar `DataArray` from `FieldData` by name (or active scalars when `name` is empty).
pub(super) fn resolve_array<'a>(field: &'a FieldData, name: &str) -> Option<&'a DataArray> {
    if name.is_empty() {
        field.active_scalars_array()
    } else {
        field.get(name)
    }
}

pub(super) fn extract_scalars(field: &FieldData, name: &str, component: Option<usize>) -> Option<Vec<f32>> {
    resolve_array(field, name).map(|arr| arr.to_scalar_f32(component))
}

pub(super) fn join_block_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else if name.is_empty() {
        parent.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

pub(super) fn resolve_block_coloring(
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
) -> (Option<Vec<f32>>, bool, [f32; 4]) {
    if let Some(rgba) = block_override {
        return (None, false, [rgba[0], rgba[1], rgba[2], rgba[3].max(0.0)]);
    }
    let (scalars, is_cell) = surface::scalar_lookup(point_data, cell_data, color_by);
    if scalars.is_none() && !matches!(color_by, ColorBy::Solid(_)) {
        (None, false, DEFAULT_FALLBACK_SOLID)
    } else {
        (scalars, is_cell, NO_COLOR_OVERRIDE)
    }
}

pub(super) fn compute_finite_range(values: &[f32]) -> Option<(f64, f64)> {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            let vd = v as f64;
            mn = mn.min(vd);
            mx = mx.max(vd);
        }
    }
    if mn.is_finite() && mx.is_finite() {
        Some((mn, mx))
    } else {
        None
    }
}

pub(super) fn merge_range(acc: &mut Option<(f64, f64)>, next: Option<(f64, f64)>) {
    if let Some((mn, mx)) = next {
        *acc = Some(match *acc {
            Some((amn, amx)) => (amn.min(mn), amx.max(mx)),
            None => (mn, mx),
        });
    }
}

/// Generates 12 bounding box edges (24 line vertices) for any [`BoundingBox`].
pub fn prepare_outline(bounds: &BoundingBox) -> Vec<LineVertex> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let mn = bounds.min;
    let mx = bounds.max;
    let corners = [
        [mn.x, mn.y, mn.z], // 0
        [mx.x, mn.y, mn.z], // 1
        [mx.x, mx.y, mn.z], // 2
        [mn.x, mx.y, mn.z], // 3
        [mn.x, mn.y, mx.z], // 4
        [mx.x, mn.y, mx.z], // 5
        [mx.x, mx.y, mx.z], // 6
        [mn.x, mx.y, mx.z], // 7
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    let mut out = Vec::with_capacity(24);
    for (a, b) in edges {
        out.push(LineVertex {
            position: corners[a],
            scalar: 0.0,
            color_override: NO_COLOR_OVERRIDE,
        });
        out.push(LineVertex {
            position: corners[b],
            scalar: 0.0,
            color_override: NO_COLOR_OVERRIDE,
        });
    }
    out
}
