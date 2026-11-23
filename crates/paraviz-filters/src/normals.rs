//! Surface normal generation and sharp-edge vertex splitting filter
//! (`GenerateNormalsFilter`, equivalent to `vtkPolyDataNormals`).

use crate::util::{canonical_edge_key, polygon_area_normal};
use glam::Vec3;
use paraviz_core::{
    DataArray, DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Splits vertices across sharp edges exceeding `feature_angle_deg` and computes
/// smooth area-weighted vertex normals on [`PolyData`] surfaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateNormalsFilter {
    /// Dihedral angle threshold in degrees above which edges are considered sharp.
    pub feature_angle_deg: f32,
    /// Split shared vertices across sharp edges.
    pub splitting: bool,
    /// Reverse polygon winding and negate computed normals.
    pub flip_normals: bool,
    /// Enforce consistent winding across connected manifold polygon components.
    pub consistency: bool,
}

impl Default for GenerateNormalsFilter {
    fn default() -> Self {
        Self {
            feature_angle_deg: 30.0,
            splitting: true,
            flip_normals: false,
            consistency: true,
        }
    }
}

impl GenerateNormalsFilter {
    /// Creates a `GenerateNormalsFilter` with default VTK settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for GenerateNormalsFilter {
    fn name(&self) -> &str {
        "GenerateNormalsFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.apply_polydata(pd))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => Err(PipelineError::ExecutionFailed(
                "GenerateNormalsFilter requires a PolyData input".into(),
            )),
        }
    }
}

impl GenerateNormalsFilter {
    fn apply_polydata(&self, pd: &PolyData) -> PolyData {
        let n_pts = pd.points.len();
        if n_pts == 0 {
            return pd.clone();
        }

        // Convert triangle strips into explicit triangles if present so normals cover them too.
        let mut polys = pd.polys.clone();
        let mut strips_converted = false;
        if !pd.strips.is_empty() {
            strips_converted = true;
            for strip in &pd.strips {
                for k in 0..strip.len().saturating_sub(2) {
                    if k % 2 == 0 {
                        polys.push(vec![strip[k], strip[k + 1], strip[k + 2]]);
                    } else {
                        polys.push(vec![strip[k + 1], strip[k], strip[k + 2]]);
                    }
                }
            }
        }

        if polys.is_empty() {
            return pd.clone();
        }

        if self.consistency {
            enforce_consistent_winding(&mut polys);
        }

        if self.flip_normals {
            for poly in &mut polys {
                poly.reverse();
            }
        }

        let face_area_normals: Vec<Vec3> = polys
            .iter()
            .map(|poly| {
                if poly.iter().any(|&vi| (vi as usize) >= n_pts) {
                    Vec3::ZERO
                } else {
                    polygon_area_normal(&pd.points, poly)
                }
            })
            .collect();

        if !self.splitting {
            let mut acc = vec![Vec3::ZERO; n_pts];
            for (poly, &fnorm) in polys.iter().zip(face_area_normals.iter()) {
                for &vi in poly {
                    if let Some(slot) = acc.get_mut(vi as usize) {
                        *slot += fnorm;
                    }
                }
            }
            let normals: Vec<[f32; 3]> = acc
                .into_iter()
                .map(|n| n.normalize_or(Vec3::Z).to_array())
                .collect();

            let mut out = PolyData {
                points: pd.points.clone(),
                verts: pd.verts.clone(),
                lines: pd.lines.clone(),
                polys,
                strips: if strips_converted {
                    Vec::new()
                } else {
                    pd.strips.clone()
                },
                point_data: pd.point_data.clone(),
                cell_data: pd.cell_data.clone(),
                field_data: pd.field_data.clone(),
            };
            out.point_data
                .set_normals(DataArray::vectors_f32("Normals", normals));
            return out;
        }

        // Sharp-edge vertex splitting:
        // Group incident polygons at each vertex into smooth fan clusters connected across
        // shared edges whose dihedral angle <= feature_angle_deg.
        let cos_thresh = self.feature_angle_deg.to_radians().cos();
        let unit_normals: Vec<Vec3> = face_area_normals
            .iter()
            .map(|n| n.normalize_or_zero())
            .collect();

        let mut edge_to_polys: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut vert_to_polys: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_pts];
        for (pidx, poly) in polys.iter().enumerate() {
            let m = poly.len();
            if m < 3 || poly.iter().any(|&vi| (vi as usize) >= n_pts) {
                continue;
            }
            for (cidx, &vi) in poly.iter().enumerate() {
                vert_to_polys[vi as usize].push((pidx, cidx));
                let next = poly[(cidx + 1) % m];
                if vi != next {
                    edge_to_polys
                        .entry(canonical_edge_key(vi, next))
                        .or_default()
                        .push(pidx);
                }
            }
        }

        // Map each (poly_idx, corner_idx) -> new vertex index.
        let mut new_points: Vec<[f32; 3]> = Vec::with_capacity(n_pts);
        let mut old_point_indices: Vec<usize> = Vec::with_capacity(n_pts);
        let mut out_normals: Vec<[f32; 3]> = Vec::with_capacity(n_pts);
        let mut vertex_remap: Vec<u32> = vec![u32::MAX; n_pts];

        let mut new_polys: Vec<Vec<u32>> = polys.clone();

        for (v_idx, occurrences) in vert_to_polys.iter().enumerate() {
            if occurrences.is_empty() {
                continue;
            }
            let k = occurrences.len();
            // Build local adjacency among incident occurrences of vertex `v_idx`.
            let mut visited = vec![false; k];
            let mut poly_to_occ: HashMap<usize, usize> = HashMap::with_capacity(k);
            for (oi, &(pidx, _)) in occurrences.iter().enumerate() {
                poly_to_occ.insert(pidx, oi);
            }

            for start in 0..k {
                if visited[start] {
                    continue;
                }
                visited[start] = true;
                let mut cluster = vec![start];
                let mut queue = VecDeque::new();
                queue.push_back(start);

                while let Some(cur_oi) = queue.pop_front() {
                    let (pidx, cidx) = occurrences[cur_oi];
                    let poly = &polys[pidx];
                    let m = poly.len();
                    let prev_v = poly[(cidx + m - 1) % m];
                    let next_v = poly[(cidx + 1) % m];
                    let cur_n = unit_normals[pidx];

                    for &other_v in &[prev_v, next_v] {
                        if other_v as usize == v_idx {
                            continue;
                        }
                        let key = canonical_edge_key(v_idx as u32, other_v);
                        if let Some(adj_polys) = edge_to_polys.get(&key) {
                            for &nb_pidx in adj_polys {
                                if nb_pidx == pidx {
                                    continue;
                                }
                                if let Some(&nb_oi) = poly_to_occ.get(&nb_pidx)
                                    && !visited[nb_oi]
                                {
                                    let nb_n = unit_normals[nb_pidx];
                                    let smooth = if cur_n.length_squared() > 0.0
                                        && nb_n.length_squared() > 0.0
                                    {
                                        cur_n.dot(nb_n) >= cos_thresh
                                    } else {
                                        true
                                    };
                                    if smooth {
                                        visited[nb_oi] = true;
                                        cluster.push(nb_oi);
                                        queue.push_back(nb_oi);
                                    }
                                }
                            }
                        }
                    }
                }

                let mut norm_acc = Vec3::ZERO;
                for &oi in &cluster {
                    let (pidx, _) = occurrences[oi];
                    norm_acc += face_area_normals[pidx];
                }
                let final_norm = norm_acc.normalize_or(Vec3::Z).to_array();

                let new_vid = new_points.len() as u32;
                if vertex_remap[v_idx] == u32::MAX {
                    vertex_remap[v_idx] = new_vid;
                }
                new_points.push(pd.points[v_idx]);
                old_point_indices.push(v_idx);
                out_normals.push(final_norm);

                for &oi in &cluster {
                    let (pidx, cidx) = occurrences[oi];
                    new_polys[pidx][cidx] = new_vid;
                }
            }
        }

        // Preserve any vertices referenced only by verts/lines.
        let mut new_verts = Vec::with_capacity(pd.verts.len());
        for &v in &pd.verts {
            let vi = v as usize;
            if vi < n_pts {
                if vertex_remap[vi] == u32::MAX {
                    vertex_remap[vi] = new_points.len() as u32;
                    new_points.push(pd.points[vi]);
                    old_point_indices.push(vi);
                    out_normals.push([0.0, 0.0, 1.0]);
                }
                new_verts.push(vertex_remap[vi]);
            }
        }
        let mut new_lines = Vec::with_capacity(pd.lines.len());
        for line in &pd.lines {
            let mut mapped = Vec::with_capacity(line.len());
            for &v in line {
                let vi = v as usize;
                if vi < n_pts {
                    if vertex_remap[vi] == u32::MAX {
                        vertex_remap[vi] = new_points.len() as u32;
                        new_points.push(pd.points[vi]);
                        old_point_indices.push(vi);
                        out_normals.push([0.0, 0.0, 1.0]);
                    }
                    mapped.push(vertex_remap[vi]);
                }
            }
            new_lines.push(mapped);
        }

        let mut point_data = pd.point_data.gather(&old_point_indices);
        point_data.set_normals(DataArray::vectors_f32("Normals", out_normals));

        PolyData {
            points: new_points,
            verts: new_verts,
            lines: new_lines,
            polys: new_polys,
            strips: if strips_converted {
                Vec::new()
            } else {
                pd.strips.clone()
            },
            point_data,
            cell_data: if strips_converted {
                paraviz_core::FieldData::new()
            } else {
                pd.cell_data.clone()
            },
            field_data: pd.field_data.clone(),
        }
    }
}

/// Ensures consistent polygon winding across manifold edges within each connected face component.
fn enforce_consistent_winding(polys: &mut [Vec<u32>]) {
    let n = polys.len();
    if n <= 1 {
        return;
    }

    let mut edge_map: HashMap<u64, Vec<(usize, u32, u32)>> = HashMap::new();
    for (pidx, poly) in polys.iter().enumerate() {
        let m = poly.len();
        if m < 3 {
            continue;
        }
        for k in 0..m {
            let a = poly[k];
            let b = poly[(k + 1) % m];
            if a != b {
                edge_map
                    .entry(canonical_edge_key(a, b))
                    .or_default()
                    .push((pidx, a, b));
            }
        }
    }

    let mut adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for entries in edge_map.values() {
        if entries.len() != 2 {
            continue;
        }
        let (p0, a0, _b0) = entries[0];
        let (p1, a1, _b1) = entries[1];
        if p0 == p1 {
            continue;
        }
        // Consistent winding across shared edge requires opposite traversal directions:
        // (a0 -> b0) in p0 and (b0 -> a0) in p1, i.e., a0 != a1.
        let same_direction = a0 == a1;
        adj[p0].push((p1, same_direction));
        adj[p1].push((p0, same_direction));
    }

    let mut visited = vec![false; n];
    let mut flipped = vec![false; n];
    let mut queue = VecDeque::new();

    for start in 0..n {
        if visited[start] || polys[start].len() < 3 {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            let cur_flipped = flipped[cur];
            for &(nb, raw_same_dir) in &adj[cur] {
                if !visited[nb] {
                    visited[nb] = true;
                    // If raw_same_dir is true, nb must have opposite flip state to cur.
                    flipped[nb] = if raw_same_dir { !cur_flipped } else { cur_flipped };
                    queue.push_back(nb);
                }
            }
        }
    }

    for (poly, &do_flip) in polys.iter_mut().zip(flipped.iter()) {
        if do_flip {
            poly.reverse();
        }
    }
}
