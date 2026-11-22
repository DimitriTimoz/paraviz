//! Surface mesh smoothing filter (`SmoothFilter`) supporting Laplacian and
//! Taubin non-shrinking smoothing on `PolyData`.

use crate::util::canonical_edge_key;
use glam::Vec3;
use paraviz_core::{DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Mesh smoothing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SmoothingMethod {
    Laplacian,
    #[default]
    Taubin,
}

/// Smooths a [`PolyData`] surface using Laplacian or Taubin (λ|μ non-shrinking) smoothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmoothFilter {
    pub method: SmoothingMethod,
    pub iterations: usize,
    pub relaxation_factor: f32,
    pub pass_band: f32,
    pub preserve_boundary: bool,
}

impl Default for SmoothFilter {
    fn default() -> Self {
        Self {
            method: SmoothingMethod::Taubin,
            iterations: 20,
            relaxation_factor: 0.5,
            pass_band: 0.1,
            preserve_boundary: true,
        }
    }
}

impl SmoothFilter {
    pub fn new(method: SmoothingMethod, iterations: usize) -> Self {
        Self {
            method,
            iterations,
            ..Default::default()
        }
    }
}

impl Filter for SmoothFilter {
    fn name(&self) -> &str {
        "SmoothFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.smooth_polydata(pd))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => Err(PipelineError::ExecutionFailed(
                "SmoothFilter requires a PolyData input".into(),
            )),
        }
    }
}

impl SmoothFilter {
    fn smooth_polydata(&self, pd: &PolyData) -> PolyData {
        let n = pd.points.len();
        if n == 0 || self.iterations == 0 {
            return pd.clone();
        }

        // Build adjacency list and detect boundary vertices from polygons/strips.
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut edge_counts: HashMap<u64, usize> = HashMap::new();

        let mut add_poly_edges = |poly: &[u32]| {
            if poly.len() < 2 {
                return;
            }
            for k in 0..poly.len() {
                let a = poly[k] as usize;
                let b = poly[(k + 1) % poly.len()] as usize;
                if a < n && b < n && a != b {
                    neighbors[a].insert(b);
                    neighbors[b].insert(a);
                    *edge_counts
                        .entry(canonical_edge_key(a as u32, b as u32))
                        .or_insert(0) += 1;
                }
            }
        };

        for poly in &pd.polys {
            add_poly_edges(poly);
        }
        for strip in &pd.strips {
            for k in 0..strip.len().saturating_sub(2) {
                add_poly_edges(&[strip[k], strip[k + 1], strip[k + 2]]);
            }
        }
        if pd.polys.is_empty() && pd.strips.is_empty() {
            for line in &pd.lines {
                for w in line.windows(2) {
                    let a = w[0] as usize;
                    let b = w[1] as usize;
                    if a < n && b < n && a != b {
                        neighbors[a].insert(b);
                        neighbors[b].insert(a);
                    }
                }
            }
        }

        let mut is_boundary = vec![false; n];
        if self.preserve_boundary {
            for (&key, &count) in &edge_counts {
                if count == 1 {
                    let a = (key >> 32) as usize;
                    let b = (key & 0xFFFF_FFFF) as usize;
                    if a < n {
                        is_boundary[a] = true;
                    }
                    if b < n {
                        is_boundary[b] = true;
                    }
                }
            }
        }

        let adj: Vec<Vec<usize>> = neighbors
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();

        let mut pts = pd.points.clone();
        let lambda = self.relaxation_factor.clamp(0.01, 0.99);
        let mu = {
            let denom = self.pass_band * lambda - 1.0;
            if denom.abs() > 1e-6 {
                lambda / denom
            } else {
                -lambda * 1.05
            }
        };

        let laplacian_step = |cur: &[[f32; 3]], factor: f32| -> Vec<[f32; 3]> {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if is_boundary[i] || adj[i].is_empty() {
                        return cur[i];
                    }
                    let pi = Vec3::from_array(cur[i]);
                    let mut avg = Vec3::ZERO;
                    for &nb in &adj[i] {
                        avg += Vec3::from_array(cur[nb]);
                    }
                    avg /= adj[i].len() as f32;
                    (pi + (avg - pi) * factor).to_array()
                })
                .collect()
        };

        match self.method {
            SmoothingMethod::Laplacian => {
                for _ in 0..self.iterations {
                    pts = laplacian_step(&pts, lambda);
                }
            }
            SmoothingMethod::Taubin => {
                for _ in 0..self.iterations {
                    pts = laplacian_step(&pts, lambda);
                    pts = laplacian_step(&pts, mu);
                }
            }
        }

        let mut out = PolyData {
            points: pts,
            ..pd.clone()
        };
        if out.point_data.active_normals_array().is_some() && !out.polys.is_empty() {
            out.compute_normals();
        }
        out
    }
}
