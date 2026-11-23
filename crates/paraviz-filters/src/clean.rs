//! Coincident point merging and degenerate cell removal filter
//! (`CleanPolyDataFilter`, equivalent to `vtkCleanPolyData`).

use crate::util::{compact_polydata_points, FxHashMap};
use glam::Vec3;
use paraviz_core::{
    DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError, PolyData,
};
use serde::{Deserialize, Serialize};

/// Merges coincident points within `tolerance`, remaps cell vertex indices,
/// removes degenerate cells, and compacts unused points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanPolyDataFilter {
    /// Spatial merge tolerance in world units (`<= 0.0` merges exact bit-identical coordinates).
    pub tolerance: f32,
}

impl Default for CleanPolyDataFilter {
    fn default() -> Self {
        Self { tolerance: 1e-5 }
    }
}

impl CleanPolyDataFilter {
    /// Creates a `CleanPolyDataFilter` with the given spatial merge tolerance.
    pub fn new(tolerance: f32) -> Self {
        Self { tolerance }
    }
}

impl Filter for CleanPolyDataFilter {
    fn name(&self) -> &str {
        "CleanPolyDataFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.clean_polydata(pd))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => Err(PipelineError::ExecutionFailed(
                "CleanPolyDataFilter requires a PolyData input".into(),
            )),
        }
    }
}

impl CleanPolyDataFilter {
    fn clean_polydata(&self, pd: &PolyData) -> PolyData {
        if pd.points.is_empty() {
            return pd.clone();
        }

        let tol = self.tolerance.max(0.0);
        let tol_sq = tol * tol;
        let cell_size = if tol > 0.0 { tol } else { 1e-6 };
        let inv_cell = 1.0 / cell_size;

        let mut grid: FxHashMap<(i64, i64, i64), Vec<u32>> = FxHashMap::default();
        let mut new_points: Vec<[f32; 3]> = Vec::with_capacity(pd.points.len());
        let mut kept_old_pts: Vec<usize> = Vec::with_capacity(pd.points.len());
        let mut old_to_new: Vec<u32> = Vec::with_capacity(pd.points.len());

        for (old_idx, &p) in pd.points.iter().enumerate() {
            let pv = Vec3::from_array(p);
            let gx = (p[0] * inv_cell).round() as i64;
            let gy = (p[1] * inv_cell).round() as i64;
            let gz = (p[2] * inv_cell).round() as i64;

            let mut matched: Option<u32> = None;
            'outer: for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(bucket) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                            for &cand in bucket {
                                let q = Vec3::from_array(new_points[cand as usize]);
                                if (tol == 0.0 && p == new_points[cand as usize])
                                    || (tol > 0.0 && (pv - q).length_squared() <= tol_sq)
                                {
                                    matched = Some(cand);
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }

            let new_id = if let Some(m) = matched {
                m
            } else {
                let id = new_points.len() as u32;
                new_points.push(p);
                kept_old_pts.push(old_idx);
                grid.entry((gx, gy, gz)).or_default().push(id);
                id
            };
            old_to_new.push(new_id);
        }

        let remap_dedup = |indices: &[u32]| -> Vec<u32> {
            let mut out = Vec::with_capacity(indices.len());
            for &vi in indices {
                if let Some(&mapped) = old_to_new.get(vi as usize)
                    && out.last().copied() != Some(mapped)
                {
                    out.push(mapped);
                }
            }
            if out.len() >= 2 && out.first() == out.last() {
                out.pop();
            }
            out
        };

        let mut verts = Vec::new();
        let mut lines = Vec::new();
        let mut polys = Vec::new();
        let mut strips = Vec::new();
        let mut kept_cells = Vec::new();
        let mut cidx = 0usize;

        for &v in &pd.verts {
            if let Some(&mapped) = old_to_new.get(v as usize) {
                verts.push(mapped);
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for line in &pd.lines {
            let mut mapped = Vec::with_capacity(line.len());
            for &vi in line {
                if let Some(&m) = old_to_new.get(vi as usize)
                    && mapped.last().copied() != Some(m)
                {
                    mapped.push(m);
                }
            }
            if mapped.len() >= 2 {
                lines.push(mapped);
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for poly in &pd.polys {
            let mapped = remap_dedup(poly);
            if mapped.len() >= 3 {
                polys.push(mapped);
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for strip in &pd.strips {
            let mut mapped = Vec::with_capacity(strip.len());
            for &vi in strip {
                if let Some(&m) = old_to_new.get(vi as usize)
                    && mapped.last().copied() != Some(m)
                {
                    mapped.push(m);
                }
            }
            if mapped.len() >= 3 {
                strips.push(mapped);
                kept_cells.push(cidx);
            }
            cidx += 1;
        }

        let point_data = if pd.point_data.is_empty() {
            FieldData::new()
        } else {
            pd.point_data.gather(&kept_old_pts)
        };
        let cell_data = if pd.cell_data.is_empty() {
            FieldData::new()
        } else {
            pd.cell_data.gather(&kept_cells)
        };

        compact_polydata_points(PolyData {
            points: new_points,
            verts,
            lines,
            polys,
            strips,
            point_data,
            cell_data,
            field_data: pd.field_data.clone(),
        })
    }
}
