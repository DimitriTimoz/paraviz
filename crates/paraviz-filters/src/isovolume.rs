//! IsoVolume filter (`IsoVolumeFilter`, equivalent to `vtkIsoVolume`).
//!
//! Smoothly clips a volumetric dataset to the scalar band `[min_value, max_value]`,
//! producing an [`UnstructuredGrid`] with smooth isosurface boundaries.

use crate::util::{dataset_to_unstructured, FxHashMap};
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError,
    UnstructuredGrid,
};
use serde::{Deserialize, Serialize};

/// Extracts the smooth volumetric sub-domain where `min_value <= scalar <= max_value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsoVolumeFilter {
    pub field_name: String,
    pub min_value: f64,
    pub max_value: f64,
}

impl Default for IsoVolumeFilter {
    fn default() -> Self {
        Self {
            field_name: String::new(),
            min_value: 0.0,
            max_value: 1.0,
        }
    }
}

impl IsoVolumeFilter {
    pub fn new(field_name: impl Into<String>, min_value: f64, max_value: f64) -> Self {
        let (lo, hi) = if min_value <= max_value {
            (min_value, max_value)
        } else {
            (max_value, min_value)
        };
        Self {
            field_name: field_name.into(),
            min_value: lo,
            max_value: hi,
        }
    }
}

impl Filter for IsoVolumeFilter {
    fn name(&self) -> &str {
        "IsoVolumeFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let ug = dataset_to_unstructured(input)?;
        let arr_name = if self.field_name.is_empty() {
            ug.point_data
                .active_scalars
                .clone()
                .or_else(|| ug.point_data.arrays().first().map(|a| a.name.clone()))
                .ok_or_else(|| {
                    PipelineError::ExecutionFailed("No point scalar array for IsoVolumeFilter".into())
                })?
        } else {
            self.field_name.clone()
        };

        let step1 = clip_unstructured_by_scalar(&ug, &arr_name, self.min_value, true)?;
        let step2 = clip_unstructured_by_scalar(&step1, &arr_name, self.max_value, false)?;
        Ok(DataSet::UnstructuredGrid(step2))
    }
}

/// Smoothly clips an `UnstructuredGrid` by a point scalar threshold:
/// `keep_above = true` keeps `s >= iso`; `keep_above = false` keeps `s <= iso`.
fn clip_unstructured_by_scalar(
    ug: &UnstructuredGrid,
    arr_name: &str,
    iso: f64,
    keep_above: bool,
) -> Result<UnstructuredGrid, PipelineError> {
    let scalar_arr = ug.point_data.get(arr_name).ok_or_else(|| {
        PipelineError::ExecutionFailed(format!("Scalar array '{arr_name}' not found"))
    })?;
    let scalars: Vec<f64> = (0..ug.num_points())
        .map(|i| {
            let s = scalar_arr.get_magnitude(i);
            if keep_above { s - iso } else { iso - s }
        })
        .collect();

    let mut out_points: Vec<[f32; 3]> = Vec::new();
    let mut pt_map: FxHashMap<usize, u32> = FxHashMap::default();
    let mut edge_map: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    let mut interp_sources: Vec<(usize, usize, f64)> = Vec::new();
    let mut out_tets: Vec<[u32; 4]> = Vec::new();

    let get_orig = |idx: usize,
                    out_points: &mut Vec<[f32; 3]>,
                    pt_map: &mut FxHashMap<usize, u32>,
                    interp_sources: &mut Vec<(usize, usize, f64)>|
     -> u32 {
        *pt_map.entry(idx).or_insert_with(|| {
            let id = out_points.len() as u32;
            out_points.push(ug.points[idx]);
            interp_sources.push((idx, idx, 0.0));
            id
        })
    };

    let get_edge = |a: usize,
                    b: usize,
                    out_points: &mut Vec<[f32; 3]>,
                    edge_map: &mut FxHashMap<(usize, usize), u32>,
                    interp_sources: &mut Vec<(usize, usize, f64)>|
     -> u32 {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        *edge_map.entry((lo, hi)).or_insert_with(|| {
            let va = scalars[lo];
            let vb = scalars[hi];
            let denom = vb - va;
            let t = if denom.abs() > 1e-12 {
                (-va / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let tf = t as f32;
            let pa = ug.points[lo];
            let pb = ug.points[hi];
            let p = [
                pa[0] + (pb[0] - pa[0]) * tf,
                pa[1] + (pb[1] - pa[1]) * tf,
                pa[2] + (pb[2] - pa[2]) * tf,
            ];
            let id = out_points.len() as u32;
            out_points.push(p);
            interp_sources.push((lo, hi, t));
            id
        })
    };

    for ci in 0..ug.num_cells() {
        let ct = ug.cell_types[ci];
        let cell_pts = ug.cell_points(ci);
        for [la, lb, lc, ld] in ct.tetrahedra() {
            if la >= cell_pts.len()
                || lb >= cell_pts.len()
                || lc >= cell_pts.len()
                || ld >= cell_pts.len()
            {
                continue;
            }
            let v = [
                cell_pts[la] as usize,
                cell_pts[lb] as usize,
                cell_pts[lc] as usize,
                cell_pts[ld] as usize,
            ];
            let mut inside = Vec::with_capacity(4);
            let mut outside = Vec::with_capacity(4);
            for &vi in &v {
                if scalars[vi] >= 0.0 {
                    inside.push(vi);
                } else {
                    outside.push(vi);
                }
            }
            match inside.len() {
                0 => {}
                4 => {
                    let a = get_orig(v[0], &mut out_points, &mut pt_map, &mut interp_sources);
                    let b = get_orig(v[1], &mut out_points, &mut pt_map, &mut interp_sources);
                    let c = get_orig(v[2], &mut out_points, &mut pt_map, &mut interp_sources);
                    let d = get_orig(v[3], &mut out_points, &mut pt_map, &mut interp_sources);
                    out_tets.push([a, b, c, d]);
                }
                1 => {
                    let i0 = inside[0];
                    let a = get_orig(i0, &mut out_points, &mut pt_map, &mut interp_sources);
                    let e0 = get_edge(i0, outside[0], &mut out_points, &mut edge_map, &mut interp_sources);
                    let e1 = get_edge(i0, outside[1], &mut out_points, &mut edge_map, &mut interp_sources);
                    let e2 = get_edge(i0, outside[2], &mut out_points, &mut edge_map, &mut interp_sources);
                    out_tets.push([a, e0, e1, e2]);
                }
                2 => {
                    let (i0, i1) = (inside[0], inside[1]);
                    let (o0, o1) = (outside[0], outside[1]);
                    let a0 = get_orig(i0, &mut out_points, &mut pt_map, &mut interp_sources);
                    let a1 = get_orig(i1, &mut out_points, &mut pt_map, &mut interp_sources);
                    let e00 = get_edge(i0, o0, &mut out_points, &mut edge_map, &mut interp_sources);
                    let e01 = get_edge(i0, o1, &mut out_points, &mut edge_map, &mut interp_sources);
                    let e10 = get_edge(i1, o0, &mut out_points, &mut edge_map, &mut interp_sources);
                    let e11 = get_edge(i1, o1, &mut out_points, &mut edge_map, &mut interp_sources);
                    out_tets.push([a0, a1, e00, e01]);
                    out_tets.push([a1, e00, e10, e01]);
                    out_tets.push([a1, e10, e11, e01]);
                }
                3 => {
                    let (i0, i1, i2) = (inside[0], inside[1], inside[2]);
                    let o0 = outside[0];
                    let a0 = get_orig(i0, &mut out_points, &mut pt_map, &mut interp_sources);
                    let a1 = get_orig(i1, &mut out_points, &mut pt_map, &mut interp_sources);
                    let a2 = get_orig(i2, &mut out_points, &mut pt_map, &mut interp_sources);
                    let e0 = get_edge(i0, o0, &mut out_points, &mut edge_map, &mut interp_sources);
                    let e1 = get_edge(i1, o0, &mut out_points, &mut edge_map, &mut interp_sources);
                    let e2 = get_edge(i2, o0, &mut out_points, &mut edge_map, &mut interp_sources);
                    out_tets.push([a0, a1, a2, e0]);
                    out_tets.push([a1, a2, e0, e1]);
                    out_tets.push([a2, e0, e1, e2]);
                }
                _ => {}
            }
        }
    }

    let mut out_pd = FieldData::new();
    for arr in ug.point_data.arrays() {
        let nc = arr.num_components;
        let mut buf = Vec::with_capacity(interp_sources.len() * nc);
        for &(lo, hi, t) in &interp_sources {
            for c in 0..nc {
                let va = arr.data.get_f64(lo * nc + c);
                let vb = arr.data.get_f64(hi * nc + c);
                buf.push((va + (vb - va) * t) as f32);
            }
        }
        let new_arr = DataArray::new(arr.name.clone(), nc, ArrayData::F32(buf));
        if ug.point_data.active_scalars.as_deref() == Some(arr.name.as_str()) {
            out_pd.set_scalars(new_arr);
        } else {
            out_pd.add_array(new_arr);
        }
    }

    let mut out = UnstructuredGrid::new();
    out.points = out_points;
    out.point_data = out_pd;
    for t in out_tets {
        out.push_cell(CellType::Tetra, &t);
    }
    Ok(out)
}
