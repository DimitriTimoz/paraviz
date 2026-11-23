//! Feature edge extraction filter (`FeatureEdgesFilter`, equivalent to `vtkFeatureEdges`).

use crate::surface::extract_surface;
use crate::util::{canonical_edge_key, compact_polydata_points, polygon_area_normal, FxHashMap};
use glam::Vec3;
use paraviz_core::{DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData};
use serde::{Deserialize, Serialize};

/// Extracts boundary, sharp dihedral (> `feature_angle_deg`), and non-manifold edges
/// from `PolyData` or surface-extracted volumetric grids into `PolyData` line segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureEdgesFilter {
    pub boundary_edges: bool,
    pub feature_edges: bool,
    pub non_manifold_edges: bool,
    pub feature_angle_deg: f32,
}

impl Default for FeatureEdgesFilter {
    fn default() -> Self {
        Self {
            boundary_edges: true,
            feature_edges: true,
            non_manifold_edges: true,
            feature_angle_deg: 30.0,
        }
    }
}

impl FeatureEdgesFilter {
    /// Creates a `FeatureEdgesFilter` with default VTK settings (all edge types, 30° feature angle).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for FeatureEdgesFilter {
    fn name(&self) -> &str {
        "FeatureEdgesFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.extract_polydata(pd))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => {
                let surf = extract_surface(input)?;
                Ok(DataSet::PolyData(self.extract_polydata(&surf)))
            }
        }
    }
}

impl FeatureEdgesFilter {
    fn extract_polydata(&self, pd: &PolyData) -> PolyData {
        let n_pts = pd.points.len();
        if n_pts == 0 || (pd.polys.is_empty() && pd.strips.is_empty()) {
            return PolyData::new();
        }

        let mut edge_faces: FxHashMap<u64, (u32, u32, Vec<Vec3>)> = FxHashMap::default();
        let mut add_face = |poly: &[u32]| {
            let m = poly.len();
            if m < 3 || poly.iter().any(|&v| (v as usize) >= n_pts) {
                return;
            }
            let normal = polygon_area_normal(&pd.points, poly).normalize_or_zero();
            for k in 0..m {
                let a = poly[k];
                let b = poly[(k + 1) % m];
                if a == b {
                    continue;
                }
                let key = canonical_edge_key(a, b);
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_faces
                    .entry(key)
                    .or_insert_with(|| (lo, hi, Vec::new()))
                    .2
                    .push(normal);
            }
        };

        for poly in &pd.polys {
            add_face(poly);
        }
        for strip in &pd.strips {
            for k in 0..strip.len().saturating_sub(2) {
                let tri = if k % 2 == 0 {
                    [strip[k], strip[k + 1], strip[k + 2]]
                } else {
                    [strip[k + 1], strip[k], strip[k + 2]]
                };
                add_face(&tri);
            }
        }

        let mut selected: Vec<(u32, u32)> = Vec::new();
        for (_, (a, b, normals)) in edge_faces {
            let keep = match normals.len() {
                0 => false,
                1 => self.boundary_edges,
                2 => {
                    if !self.feature_edges {
                        false
                    } else {
                        let n0 = normals[0];
                        let n1 = normals[1];
                        if n0.length_squared() > 0.0 && n1.length_squared() > 0.0 {
                            let angle = n0.dot(n1).clamp(-1.0, 1.0).acos().to_degrees();
                            angle > self.feature_angle_deg
                        } else {
                            false
                        }
                    }
                }
                _ => self.non_manifold_edges,
            };
            if keep {
                selected.push((a, b));
            }
        }
        selected.sort_unstable();

        let lines: Vec<Vec<u32>> = selected.into_iter().map(|(a, b)| vec![a, b]).collect();
        compact_polydata_points(PolyData {
            points: pd.points.clone(),
            lines,
            point_data: pd.point_data.clone(),
            field_data: pd.field_data.clone(),
            ..Default::default()
        })
    }
}
