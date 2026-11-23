//! Self-contained mesh, line, point, outline, and 3D volume extraction from `paraviz-core` datasets.

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::scene::{ColorBy, ViewportWidget};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use paraviz_core::{
    BoundingBox, CellType, DataArray, DataSet, FieldData, ImageData, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 4096;

mod surface;
mod wireframe_points;

pub use surface::{extract_indexed_surface_scalars, prepare_surface, prepare_surface_indexed};
pub use wireframe_points::{build_axes_gizmo_mesh, prepare_points, prepare_volume, prepare_widgets, prepare_wireframe, GizmoVertex, PreparedWidgets, WidgetVertex};

/// Non-indexed triangle vertex for SurfaceWithEdges rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub scalar: f32,
    pub barycentric: [f32; 3],
}

/// Geometry-only vertex (position + normal) for indexed Surface rendering with split scalar slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceGeomVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
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
}

/// Per-point instance for sphere-impostor point rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub scalar: f32,
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
        out.push(LineVertex { position: corners[a], scalar: 0.0 });
        out.push(LineVertex { position: corners[b], scalar: 0.0 });
    }
    out
}
