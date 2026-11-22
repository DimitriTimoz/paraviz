//! Polyline-to-tube filter (`TubeFilter`, equivalent to `vtkTubeFilter`) using
//! parallel-transport (Bishop) frames to prevent twisting or pinching.

use crate::util::merge_polydata;
use glam::{Quat, Vec3};
use paraviz_core::{
    ArrayData, DataArray, DataSet, FieldData, Filter, PipelineError, PolyData,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

/// Converts [`PolyData`] polylines into smooth 3D shaded cylindrical tubes using
/// parallel-transport (Bishop) frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TubeFilter {
    pub radius: f32,
    pub vary_radius_by_scalar: Option<String>,
    pub radius_factor: f32,
    pub sides: usize,
    pub capping: bool,
}

impl Default for TubeFilter {
    fn default() -> Self {
        Self {
            radius: 0.02,
            vary_radius_by_scalar: None,
            radius_factor: 3.0,
            sides: 12,
            capping: true,
        }
    }
}

impl TubeFilter {
    pub fn new(radius: f32, sides: usize) -> Self {
        Self {
            radius,
            sides,
            ..Default::default()
        }
    }
}

impl Filter for TubeFilter {
    fn name(&self) -> &str {
        "TubeFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let pd = input.as_poly_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("TubeFilter requires a PolyData input".into())
        })?;

        if pd.lines.is_empty() {
            return Ok(DataSet::PolyData(PolyData::new()));
        }

        let sides = self.sides.max(3);
        let scalar_range = if let Some(ref name) = self.vary_radius_by_scalar {
            let arr = pd.point_data.get(name).ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "Scalar array '{name}' not found in point_data for TubeFilter"
                ))
            })?;
            let (mn, mx) = arr.range(None);
            Some((arr, mn, mx))
        } else {
            None
        };

        let parts: Vec<PolyData> = pd
            .lines
            .par_iter()
            .filter_map(|line| {
                build_tube_for_polyline(
                    pd,
                    line,
                    self.radius,
                    self.radius_factor,
                    sides,
                    self.capping,
                    scalar_range,
                )
            })
            .collect();

        Ok(DataSet::PolyData(merge_polydata(parts)))
    }
}

fn build_tube_for_polyline(
    pd: &PolyData,
    line: &[u32],
    base_radius: f32,
    radius_factor: f32,
    sides: usize,
    capping: bool,
    scalar_range: Option<(&DataArray, f64, f64)>,
) -> Option<PolyData> {
    // Filter out consecutive duplicate points to keep tangents well-defined.
    let mut clean_indices: Vec<usize> = Vec::with_capacity(line.len());
    for &idx in line {
        let i = idx as usize;
        if i >= pd.points.len() {
            continue;
        }
        if let Some(&prev) = clean_indices.last() {
            let d = (Vec3::from_array(pd.points[i]) - Vec3::from_array(pd.points[prev])).length();
            if d < 1e-7 {
                continue;
            }
        }
        clean_indices.push(i);
    }
    let n = clean_indices.len();
    if n < 2 {
        return None;
    }

    let pts: Vec<Vec3> = clean_indices
        .iter()
        .map(|&i| Vec3::from_array(pd.points[i]))
        .collect();

    // Compute smooth tangents.
    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let t = if i == 0 {
            pts[1] - pts[0]
        } else if i == n - 1 {
            pts[n - 1] - pts[n - 2]
        } else {
            (pts[i + 1] - pts[i]).normalize_or_zero() + (pts[i] - pts[i - 1]).normalize_or_zero()
        };
        tangents.push(t.normalize_or(Vec3::X));
    }

    // Initial Bishop frame (N0, B0) orthogonal to T0.
    let t0 = tangents[0];
    let ref_axis = if t0.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let mut normal = (ref_axis - t0 * t0.dot(ref_axis)).normalize();
    let mut normals_frame = Vec::with_capacity(n);
    let mut binormals_frame = Vec::with_capacity(n);
    normals_frame.push(normal);
    binormals_frame.push(t0.cross(normal).normalize());

    // Parallel transport along the polyline.
    for i in 1..n {
        let prev_t = tangents[i - 1];
        let cur_t = tangents[i];
        let rot = Quat::from_rotation_arc(prev_t, cur_t);
        normal = rot * normal;
        normal = (normal - cur_t * cur_t.dot(normal)).normalize_or(normal);
        let binormal = cur_t.cross(normal).normalize();
        normals_frame.push(normal);
        binormals_frame.push(binormal);
    }

    let radii: Vec<f32> = clean_indices
        .iter()
        .map(|&idx| match scalar_range {
            Some((arr, mn, mx)) if (mx - mn).abs() > 1e-12 => {
                let s = arr.get_magnitude(idx);
                let t = ((s - mn) / (mx - mn)).clamp(0.0, 1.0) as f32;
                base_radius * (1.0 + (radius_factor - 1.0) * t)
            }
            _ => base_radius,
        })
        .collect();

    let mut out_points = Vec::with_capacity(n * sides + if capping { 2 + 2 * sides } else { 0 });
    let mut out_normals = Vec::with_capacity(out_points.capacity());
    let mut src_point_map = Vec::with_capacity(out_points.capacity());
    let mut polys = Vec::with_capacity((n - 1) * sides * 2 + if capping { 2 * sides } else { 0 });

    for i in 0..n {
        let c = pts[i];
        let n_vec = normals_frame[i];
        let b_vec = binormals_frame[i];
        let r = radii[i];
        let src_idx = clean_indices[i];

        for s in 0..sides {
            let ang = (s as f32 / sides as f32) * TAU;
            let radial = n_vec * ang.cos() + b_vec * ang.sin();
            out_points.push((c + radial * r).to_array());
            out_normals.push(radial.to_array());
            src_point_map.push(src_idx);
        }
    }

    for i in 0..(n - 1) {
        let ring0 = (i * sides) as u32;
        let ring1 = ((i + 1) * sides) as u32;
        for s in 0..sides {
            let s_next = (s + 1) % sides;
            let a = ring0 + s as u32;
            let b = ring0 + s_next as u32;
            let c = ring1 + s_next as u32;
            let d = ring1 + s as u32;
            polys.push(vec![a, b, c]);
            polys.push(vec![a, c, d]);
        }
    }

    if capping {
        // Start cap (-T0 normal)
        let start_center = out_points.len() as u32;
        out_points.push(pts[0].to_array());
        out_normals.push((-tangents[0]).to_array());
        src_point_map.push(clean_indices[0]);

        let start_ring_base = out_points.len() as u32;
        for s in 0..sides {
            let p = out_points[s];
            out_points.push(p);
            out_normals.push((-tangents[0]).to_array());
            src_point_map.push(clean_indices[0]);
        }
        for s in 0..sides {
            let s_next = (s + 1) % sides;
            polys.push(vec![
                start_center,
                start_ring_base + s_next as u32,
                start_ring_base + s as u32,
            ]);
        }

        // End cap (+Tn-1 normal)
        let end_center = out_points.len() as u32;
        out_points.push(pts[n - 1].to_array());
        out_normals.push(tangents[n - 1].to_array());
        src_point_map.push(clean_indices[n - 1]);

        let last_ring_start = (n - 1) * sides;
        let end_ring_base = out_points.len() as u32;
        for s in 0..sides {
            let p = out_points[last_ring_start + s];
            out_points.push(p);
            out_normals.push(tangents[n - 1].to_array());
            src_point_map.push(clean_indices[n - 1]);
        }
        for s in 0..sides {
            let s_next = (s + 1) % sides;
            polys.push(vec![
                end_center,
                end_ring_base + s as u32,
                end_ring_base + s_next as u32,
            ]);
        }
    }

    let mut pdata: FieldData = pd.point_data.gather(&src_point_map);
    pdata.set_normals(DataArray::vectors_f32("Normals", out_normals));

    // Preserve active scalar if it existed
    if let Some(ref s) = pd.point_data.active_scalars {
        pdata.active_scalars = Some(s.clone());
    }
    if pdata.is_empty() {
        let _ = ArrayData::F32(Vec::new());
    }

    Some(PolyData {
        points: out_points,
        polys,
        point_data: pdata,
        ..Default::default()
    })
}
