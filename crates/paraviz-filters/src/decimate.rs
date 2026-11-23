//! Quadric Error Metric (QEM) mesh decimation filter (`DecimateFilter`, equivalent to `vtkQuadricDecimation`).

use crate::surface::extract_surface;
use crate::util::FxHashMap;
use glam::{DVec3, Vec3};
use paraviz_core::{DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Garland-Heckbert Quadric Error Metric surface simplification filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecimateFilter {
    /// Target fraction of triangles to remove (`0.0` = no change, `0.5` = remove 50%, `0.9` = keep 10%).
    pub target_reduction: f32,
    /// Preserve open boundary edges from collapsing excessively.
    pub preserve_boundaries: bool,
}

impl Default for DecimateFilter {
    fn default() -> Self {
        Self {
            target_reduction: 0.5,
            preserve_boundaries: true,
        }
    }
}

impl DecimateFilter {
    pub fn new(target_reduction: f32) -> Self {
        Self {
            target_reduction: target_reduction.clamp(0.0, 0.99),
            preserve_boundaries: true,
        }
    }
}

impl Filter for DecimateFilter {
    fn name(&self) -> &str {
        "DecimateFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let poly = match input {
            DataSet::PolyData(pd) => pd.clone(),
            other => extract_surface(other)?,
        };
        Ok(DataSet::PolyData(decimate_polydata(
            &poly,
            self.target_reduction,
            self.preserve_boundaries,
        )))
    }
}

/// Symmetric 4x4 error quadric matrix stored as 10 upper-triangular `f64` values.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    fn add(&self, o: &Self) -> Self {
        let mut out = [0.0; 10];
        for i in 0..10 {
            out[i] = self.0[i] + o.0[i];
        }
        Self(out)
    }

    fn evaluate(&self, v: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (v.x, v.y, v.z);
        (q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9])
            .max(0.0)
    }

    fn optimal_position(&self, p0: DVec3, p1: DVec3) -> (DVec3, f64) {
        let q = &self.0;
        // Try solving 3x3 linear system A * v = -b
        let det = q[0] * (q[4] * q[7] - q[5] * q[5])
            - q[1] * (q[1] * q[7] - q[5] * q[2])
            + q[2] * (q[1] * q[5] - q[4] * q[2]);
        if det.abs() > 1e-12 {
            let inv = 1.0 / det;
            let c00 = (q[4] * q[7] - q[5] * q[5]) * inv;
            let c01 = -(q[1] * q[7] - q[2] * q[5]) * inv;
            let c02 = (q[1] * q[5] - q[2] * q[4]) * inv;
            let c11 = (q[0] * q[7] - q[2] * q[2]) * inv;
            let c12 = -(q[0] * q[5] - q[1] * q[2]) * inv;
            let c22 = (q[0] * q[4] - q[1] * q[1]) * inv;
            let bx = -q[3];
            let by = -q[6];
            let bz = -q[8];
            let cand = DVec3::new(
                c00 * bx + c01 * by + c02 * bz,
                c01 * bx + c11 * by + c12 * bz,
                c02 * bx + c12 * by + c22 * bz,
            );
            if cand.is_finite() {
                let err = self.evaluate(cand);
                return (cand, err);
            }
        }
        // Fallback: best of endpoints and midpoint
        let mid = (p0 + p1) * 0.5;
        let e0 = self.evaluate(p0);
        let e1 = self.evaluate(p1);
        let em = self.evaluate(mid);
        if em <= e0 && em <= e1 {
            (mid, em)
        } else if e0 <= e1 {
            (p0, e0)
        } else {
            (p1, e1)
        }
    }
}

#[derive(Clone, Copy)]
struct HeapItem {
    cost: f64,
    u: usize,
    v: usize,
    ver_u: u32,
    ver_v: u32,
    target: [f32; 3],
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost) // min-heap
    }
}

fn decimate_polydata(poly: &PolyData, target_reduction: f32, preserve_boundaries: bool) -> PolyData {
    let tris = poly.triangulate();
    let initial_tris = tris.len();
    if initial_tris <= 4 || target_reduction <= 0.0 {
        return poly.clone();
    }
    let target_tris = ((initial_tris as f32) * (1.0 - target_reduction.clamp(0.0, 0.99)))
        .round()
        .max(4.0) as usize;

    let n_pts = poly.points.len();
    let mut positions = poly.points.clone();
    let mut quadrics = vec![Quadric::default(); n_pts];
    let mut vert_tris: Vec<Vec<usize>> = vec![Vec::new(); n_pts];
    let mut tri_list: Vec<[usize; 3]> = tris
        .iter()
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    let mut tri_alive = vec![true; initial_tris];

    let mut edge_counts: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    for (ti, &[a, b, c]) in tri_list.iter().enumerate() {
        vert_tris[a].push(ti);
        vert_tris[b].push(ti);
        vert_tris[c].push(ti);

        let pa = Vec3::from_array(positions[a]).as_dvec3();
        let pb = Vec3::from_array(positions[b]).as_dvec3();
        let pc = Vec3::from_array(positions[c]).as_dvec3();
        let n = (pb - pa).cross(pc - pa);
        let len = n.length();
        if len > 1e-12 {
            let nn = n / len;
            let d = -nn.dot(pa);
            let q = Quadric::from_plane(nn.x, nn.y, nn.z, d);
            quadrics[a] = quadrics[a].add(&q);
            quadrics[b] = quadrics[b].add(&q);
            quadrics[c] = quadrics[c].add(&q);
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut is_boundary = vec![false; n_pts];
    if preserve_boundaries {
        for (&(u, v), &cnt) in &edge_counts {
            if cnt == 1 {
                is_boundary[u] = true;
                is_boundary[v] = true;
            }
        }
    }

    let mut heap = BinaryHeap::with_capacity(edge_counts.len());
    let mut version = vec![0u32; n_pts];
    let mut alive_vert = vec![true; n_pts];

    let push_edge = |heap: &mut BinaryHeap<HeapItem>,
                     u: usize,
                     v: usize,
                     positions: &[[f32; 3]],
                     quadrics: &[Quadric],
                     version: &[u32],
                     is_boundary: &[bool]| {
        if u == v || (preserve_boundaries && (is_boundary[u] || is_boundary[v])) {
            return;
        }
        let q = quadrics[u].add(&quadrics[v]);
        let p0 = Vec3::from_array(positions[u]).as_dvec3();
        let p1 = Vec3::from_array(positions[v]).as_dvec3();
        let (opt, cost) = q.optimal_position(p0, p1);
        heap.push(HeapItem {
            cost,
            u,
            v,
            ver_u: version[u],
            ver_v: version[v],
            target: opt.as_vec3().to_array(),
        });
    };

    for &(u, v) in edge_counts.keys() {
        push_edge(
            &mut heap,
            u,
            v,
            &positions,
            &quadrics,
            &version,
            &is_boundary,
        );
    }

    let mut current_tris = initial_tris;
    while current_tris > target_tris {
        let Some(item) = heap.pop() else {
            break;
        };
        let (u, v) = (item.u, item.v);
        if !alive_vert[u]
            || !alive_vert[v]
            || version[u] != item.ver_u
            || version[v] != item.ver_v
        {
            continue;
        }

        // Collapse v -> u
        alive_vert[v] = false;
        positions[u] = item.target;
        quadrics[u] = quadrics[u].add(&quadrics[v]);
        version[u] = version[u].wrapping_add(1);

        let moved_tris = std::mem::take(&mut vert_tris[v]);
        for ti in moved_tris {
            if !tri_alive[ti] {
                continue;
            }
            let t = &mut tri_list[ti];
            for idx in t.iter_mut() {
                if *idx == v {
                    *idx = u;
                }
            }
            if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                tri_alive[ti] = false;
                current_tris = current_tris.saturating_sub(1);
            } else {
                vert_tris[u].push(ti);
            }
        }
        vert_tris[u].retain(|&ti| tri_alive[ti]);

        // Re-evaluate incident edges around u
        let mut neighbors = Vec::new();
        for &ti in &vert_tris[u] {
            for &w in &tri_list[ti] {
                if w != u && alive_vert[w] && !neighbors.contains(&w) {
                    neighbors.push(w);
                }
            }
        }
        for w in neighbors {
            push_edge(
                &mut heap,
                u,
                w,
                &positions,
                &quadrics,
                &version,
                &is_boundary,
            );
        }
    }

    // Compact surviving vertices and triangles
    let mut old_to_new = vec![u32::MAX; n_pts];
    let mut new_points = Vec::new();
    let mut kept_old_indices = Vec::new();
    let mut new_polys = Vec::with_capacity(current_tris);

    for (ti, &alive) in tri_alive.iter().enumerate() {
        if !alive {
            continue;
        }
        let t = tri_list[ti];
        let mut poly_out = Vec::with_capacity(3);
        for old_v in t {
            let mapped = if old_to_new[old_v] == u32::MAX {
                let id = new_points.len() as u32;
                old_to_new[old_v] = id;
                new_points.push(positions[old_v]);
                kept_old_indices.push(old_v);
                id
            } else {
                old_to_new[old_v]
            };
            poly_out.push(mapped);
        }
        new_polys.push(poly_out);
    }

    let mut out = PolyData {
        points: new_points,
        polys: new_polys,
        point_data: poly.point_data.gather(&kept_old_indices),
        ..Default::default()
    };
    out.compute_normals();
    out
}
