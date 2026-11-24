//! Integrate Variables filter (`IntegrateVariablesFilter`, equivalent to `vtkIntegrateAttributes`).
//!
//! Computes total arc length (1D), surface area (2D), or volume (3D) along with
//! geometry-weighted integrals and averages of all point and cell data arrays.

use glam::Vec3;
use paraviz_core::{
    DataArray, DataSet, FieldData, Filter, ImageData, PipelineError, PolyData, RectilinearGrid,
    StructuredGrid, UnstructuredGrid,
};
use serde::{Deserialize, Serialize};

/// Integrates all point and cell attributes over the geometry (length, area, or volume).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub struct IntegrateVariablesFilter {
    /// When `true`, divides integrated values by total length/area/volume to produce spatial averages.
    pub divide_cell_data_by_volume: bool,
}


impl IntegrateVariablesFilter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for IntegrateVariablesFilter {
    fn name(&self) -> &str {
        "IntegrateVariablesFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let mut total_measure = 0.0f64;
        let mut dim_label = "Volume";
        let mut pt_integrals: Vec<(String, usize, Vec<f64>)> = Vec::new();
        let mut cell_integrals: Vec<(String, usize, Vec<f64>)> = Vec::new();
        let mut centroid_accum = [0.0f64; 3];

        integrate_dataset(
            input,
            &mut total_measure,
            &mut dim_label,
            &mut centroid_accum,
            &mut pt_integrals,
            &mut cell_integrals,
        );

        let center = if total_measure > 1e-18 {
            [
                (centroid_accum[0] / total_measure) as f32,
                (centroid_accum[1] / total_measure) as f32,
                (centroid_accum[2] / total_measure) as f32,
            ]
        } else {
            input.bounds().center().to_array()
        };

        let denom = if self.divide_cell_data_by_volume && total_measure > 1e-18 {
            total_measure
        } else {
            1.0
        };

        let mut out = PolyData::new();
        out.points.push(center);
        out.verts.push(0);

        out.cell_data
            .set_scalars(DataArray::scalars_f64(dim_label, vec![total_measure]));

        for (name, nc, sums) in pt_integrals {
            let vals: Vec<f64> = sums.into_iter().map(|s| s / denom).collect();
            out.point_data
                .add_array(DataArray::new(name, nc, paraviz_core::ArrayData::F64(vals)));
        }
        for (name, nc, sums) in cell_integrals {
            let vals: Vec<f64> = sums.into_iter().map(|s| s / denom).collect();
            out.cell_data
                .add_array(DataArray::new(name, nc, paraviz_core::ArrayData::F64(vals)));
        }

        Ok(DataSet::PolyData(out))
    }
}

fn accumulate_field_weighted(
    acc: &mut Vec<(String, usize, Vec<f64>)>,
    fd: &FieldData,
    tuple_indices: &[usize],
    weight: f64,
) {
    if tuple_indices.is_empty() || weight == 0.0 {
        return;
    }
    let inv_k = 1.0 / (tuple_indices.len() as f64);
    for arr in fd.arrays() {
        let nc = arr.num_components;
        let entry = if let Some(pos) = acc.iter().position(|(n, c, _)| n == &arr.name && *c == nc) {
            &mut acc[pos].2
        } else {
            acc.push((arr.name.clone(), nc, vec![0.0; nc]));
            &mut acc.last_mut().unwrap().2
        };
        for (c, slot) in entry.iter_mut().enumerate().take(nc) {
            let mut avg = 0.0;
            for &ti in tuple_indices {
                avg += arr.data.get_f64(ti * nc + c);
            }
            *slot += avg * inv_k * weight;
        }
    }
}

fn tet_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    ((b - a).dot((c - a).cross(d - a)).abs() as f64) / 6.0
}

fn integrate_dataset(
    ds: &DataSet,
    total_measure: &mut f64,
    dim_label: &mut &'static str,
    centroid_accum: &mut [f64; 3],
    pt_acc: &mut Vec<(String, usize, Vec<f64>)>,
    cell_acc: &mut Vec<(String, usize, Vec<f64>)>,
) {
    match ds {
        DataSet::MultiBlock(mb) => {
            for (_, b) in &mb.blocks {
                integrate_dataset(b, total_measure, dim_label, centroid_accum, pt_acc, cell_acc);
            }
        }
        DataSet::PolyData(pd) => {
            if !pd.polys.is_empty() || !pd.strips.is_empty() {
                *dim_label = "Area";
                let base_cell = pd.verts.len() + pd.lines.len();
                for (pi, poly) in pd.polys.iter().enumerate() {
                    if poly.len() < 3 {
                        continue;
                    }
                    let p0 = Vec3::from_array(pd.points[poly[0] as usize]);
                    let mut poly_area = 0.0f64;
                    let mut poly_c = Vec3::ZERO;
                    for k in 1..(poly.len() - 1) {
                        let p1 = Vec3::from_array(pd.points[poly[k] as usize]);
                        let p2 = Vec3::from_array(pd.points[poly[k + 1] as usize]);
                        let a = 0.5 * ((p1 - p0).cross(p2 - p0).length() as f64);
                        poly_area += a;
                        poly_c += (p0 + p1 + p2) * (a as f32 / 3.0);
                    }
                    *total_measure += poly_area;
                    centroid_accum[0] += poly_c.x as f64;
                    centroid_accum[1] += poly_c.y as f64;
                    centroid_accum[2] += poly_c.z as f64;
                    let pts_idx: Vec<usize> = poly.iter().map(|&i| i as usize).collect();
                    accumulate_field_weighted(pt_acc, &pd.point_data, &pts_idx, poly_area);
                    let ci = base_cell + pi;
                    if pd.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
                        accumulate_field_weighted(cell_acc, &pd.cell_data, &[ci], poly_area);
                    }
                }
            } else if !pd.lines.is_empty() {
                *dim_label = "Length";
                let base_cell = pd.verts.len();
                for (li, line) in pd.lines.iter().enumerate() {
                    if line.len() < 2 {
                        continue;
                    }
                    let mut len = 0.0f64;
                    for seg in line.windows(2) {
                        let a = Vec3::from_array(pd.points[seg[0] as usize]);
                        let b = Vec3::from_array(pd.points[seg[1] as usize]);
                        let d = a.distance(b) as f64;
                        len += d;
                        let mid = (a + b) * 0.5;
                        centroid_accum[0] += mid.x as f64 * d;
                        centroid_accum[1] += mid.y as f64 * d;
                        centroid_accum[2] += mid.z as f64 * d;
                    }
                    *total_measure += len;
                    let pts_idx: Vec<usize> = line.iter().map(|&i| i as usize).collect();
                    accumulate_field_weighted(pt_acc, &pd.point_data, &pts_idx, len);
                    let ci = base_cell + li;
                    if pd.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
                        accumulate_field_weighted(cell_acc, &pd.cell_data, &[ci], len);
                    }
                }
            }
        }
        DataSet::ImageData(img) => {
            integrate_image(img, total_measure, dim_label, centroid_accum, pt_acc, cell_acc);
        }
        DataSet::RectilinearGrid(rg) => {
            integrate_rectilinear(rg, total_measure, dim_label, centroid_accum, pt_acc, cell_acc);
        }
        DataSet::StructuredGrid(sg) => {
            integrate_structured(sg, total_measure, dim_label, centroid_accum, pt_acc, cell_acc);
        }
        DataSet::UnstructuredGrid(ug) => {
            integrate_unstructured(ug, total_measure, dim_label, centroid_accum, pt_acc, cell_acc);
        }
    }
}

fn integrate_image(
    img: &ImageData,
    total_measure: &mut f64,
    dim_label: &mut &'static str,
    centroid_accum: &mut [f64; 3],
    pt_acc: &mut Vec<(String, usize, Vec<f64>)>,
    cell_acc: &mut Vec<(String, usize, Vec<f64>)>,
) {
    *dim_label = "Volume";
    let [nx, ny, nz] = img.dimensions;
    if nx < 2 || ny < 2 || nz < 2 {
        return;
    }
    let cell_vol = (img.spacing[0] * img.spacing[1] * img.spacing[2]).abs();
    let cx = nx - 1;
    let cy = ny - 1;
    let cz = nz - 1;
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let ci = i + j * cx + k * cx * cy;
                let c = img.point_coords(i, j, k);
                let mx = c[0] as f64 + 0.5 * img.spacing[0];
                let my = c[1] as f64 + 0.5 * img.spacing[1];
                let mz = c[2] as f64 + 0.5 * img.spacing[2];
                *total_measure += cell_vol;
                centroid_accum[0] += mx * cell_vol;
                centroid_accum[1] += my * cell_vol;
                centroid_accum[2] += mz * cell_vol;
                let corners = [
                    img.point_index(i, j, k),
                    img.point_index(i + 1, j, k),
                    img.point_index(i + 1, j + 1, k),
                    img.point_index(i, j + 1, k),
                    img.point_index(i, j, k + 1),
                    img.point_index(i + 1, j, k + 1),
                    img.point_index(i + 1, j + 1, k + 1),
                    img.point_index(i, j + 1, k + 1),
                ];
                accumulate_field_weighted(pt_acc, &img.point_data, &corners, cell_vol);
                if img.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
                    accumulate_field_weighted(cell_acc, &img.cell_data, &[ci], cell_vol);
                }
            }
        }
    }
}

fn integrate_rectilinear(
    rg: &RectilinearGrid,
    total_measure: &mut f64,
    dim_label: &mut &'static str,
    centroid_accum: &mut [f64; 3],
    pt_acc: &mut Vec<(String, usize, Vec<f64>)>,
    cell_acc: &mut Vec<(String, usize, Vec<f64>)>,
) {
    *dim_label = "Volume";
    let [nx, ny, nz] = rg.dimensions();
    if nx < 2 || ny < 2 || nz < 2 {
        return;
    }
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    for k in 0..cz {
        let dz = (rg.z_coords[k + 1] - rg.z_coords[k]).abs() as f64;
        let mz = 0.5 * (rg.z_coords[k + 1] + rg.z_coords[k]) as f64;
        for j in 0..cy {
            let dy = (rg.y_coords[j + 1] - rg.y_coords[j]).abs() as f64;
            let my = 0.5 * (rg.y_coords[j + 1] + rg.y_coords[j]) as f64;
            for i in 0..cx {
                let dx = (rg.x_coords[i + 1] - rg.x_coords[i]).abs() as f64;
                let mx = 0.5 * (rg.x_coords[i + 1] + rg.x_coords[i]) as f64;
                let vol = dx * dy * dz;
                let ci = i + j * cx + k * cx * cy;
                *total_measure += vol;
                centroid_accum[0] += mx * vol;
                centroid_accum[1] += my * vol;
                centroid_accum[2] += mz * vol;
                let corners = [
                    rg.point_index(i, j, k),
                    rg.point_index(i + 1, j, k),
                    rg.point_index(i + 1, j + 1, k),
                    rg.point_index(i, j + 1, k),
                    rg.point_index(i, j, k + 1),
                    rg.point_index(i + 1, j, k + 1),
                    rg.point_index(i + 1, j + 1, k + 1),
                    rg.point_index(i, j + 1, k + 1),
                ];
                accumulate_field_weighted(pt_acc, &rg.point_data, &corners, vol);
                if rg.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
                    accumulate_field_weighted(cell_acc, &rg.cell_data, &[ci], vol);
                }
            }
        }
    }
}

fn integrate_structured(
    sg: &StructuredGrid,
    total_measure: &mut f64,
    dim_label: &mut &'static str,
    centroid_accum: &mut [f64; 3],
    pt_acc: &mut Vec<(String, usize, Vec<f64>)>,
    cell_acc: &mut Vec<(String, usize, Vec<f64>)>,
) {
    *dim_label = "Volume";
    let [nx, ny, nz] = sg.dimensions;
    if nx < 2 || ny < 2 || nz < 2 {
        return;
    }
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let ci = i + j * cx + k * cx * cy;
                let corners = [
                    sg.point_index(i, j, k),
                    sg.point_index(i + 1, j, k),
                    sg.point_index(i + 1, j + 1, k),
                    sg.point_index(i, j + 1, k),
                    sg.point_index(i, j, k + 1),
                    sg.point_index(i + 1, j, k + 1),
                    sg.point_index(i + 1, j + 1, k + 1),
                    sg.point_index(i, j + 1, k + 1),
                ];
                let tets = paraviz_core::CellType::Hexahedron
                    .tetrahedra_with_parity((i + j + k) % 2 == 1);
                let mut vol = 0.0;
                let mut c_sum = Vec3::ZERO;
                for [a, b, c, d] in tets {
                    let pa = Vec3::from_array(sg.points[corners[a]]);
                    let pb = Vec3::from_array(sg.points[corners[b]]);
                    let pc = Vec3::from_array(sg.points[corners[c]]);
                    let pd = Vec3::from_array(sg.points[corners[d]]);
                    let tv = tet_volume(pa, pb, pc, pd);
                    vol += tv;
                    c_sum += (pa + pb + pc + pd) * (tv as f32 * 0.25);
                }
                *total_measure += vol;
                centroid_accum[0] += c_sum.x as f64;
                centroid_accum[1] += c_sum.y as f64;
                centroid_accum[2] += c_sum.z as f64;
                accumulate_field_weighted(pt_acc, &sg.point_data, &corners, vol);
                if sg.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
                    accumulate_field_weighted(cell_acc, &sg.cell_data, &[ci], vol);
                }
            }
        }
    }
}

fn integrate_unstructured(
    ug: &UnstructuredGrid,
    total_measure: &mut f64,
    dim_label: &mut &'static str,
    centroid_accum: &mut [f64; 3],
    pt_acc: &mut Vec<(String, usize, Vec<f64>)>,
    cell_acc: &mut Vec<(String, usize, Vec<f64>)>,
) {
    let mut max_dim = 0u8;
    for &ct in &ug.cell_types {
        max_dim = max_dim.max(ct.dimension());
    }
    *dim_label = match max_dim {
        3 => "Volume",
        2 => "Area",
        _ => "Length",
    };

    for ci in 0..ug.num_cells() {
        let ct = ug.cell_types[ci];
        let pts = ug.cell_points(ci);
        let pts_usize: Vec<usize> = pts.iter().map(|&i| i as usize).collect();
        let measure = if ct.dimension() == 3 {
            let mut vol = 0.0;
            for [a, b, c, d] in ct.tetrahedra() {
                if a < pts.len() && b < pts.len() && c < pts.len() && d < pts.len() {
                    let pa = Vec3::from_array(ug.points[pts[a] as usize]);
                    let pb = Vec3::from_array(ug.points[pts[b] as usize]);
                    let pc = Vec3::from_array(ug.points[pts[c] as usize]);
                    let pd = Vec3::from_array(ug.points[pts[d] as usize]);
                    let tv = tet_volume(pa, pb, pc, pd);
                    vol += tv;
                    let mid = (pa + pb + pc + pd) * 0.25;
                    centroid_accum[0] += mid.x as f64 * tv;
                    centroid_accum[1] += mid.y as f64 * tv;
                    centroid_accum[2] += mid.z as f64 * tv;
                }
            }
            vol
        } else if ct.dimension() == 2 && pts.len() >= 3 {
            let p0 = Vec3::from_array(ug.points[pts[0] as usize]);
            let mut area = 0.0;
            for k in 1..(pts.len() - 1) {
                let p1 = Vec3::from_array(ug.points[pts[k] as usize]);
                let p2 = Vec3::from_array(ug.points[pts[k + 1] as usize]);
                let a = 0.5 * ((p1 - p0).cross(p2 - p0).length() as f64);
                area += a;
                let mid = (p0 + p1 + p2) / 3.0;
                centroid_accum[0] += mid.x as f64 * a;
                centroid_accum[1] += mid.y as f64 * a;
                centroid_accum[2] += mid.z as f64 * a;
            }
            area
        } else if ct.dimension() == 1 && pts.len() >= 2 {
            let mut len = 0.0;
            for seg in pts.windows(2) {
                let a = Vec3::from_array(ug.points[seg[0] as usize]);
                let b = Vec3::from_array(ug.points[seg[1] as usize]);
                let d = a.distance(b) as f64;
                len += d;
                let mid = (a + b) * 0.5;
                centroid_accum[0] += mid.x as f64 * d;
                centroid_accum[1] += mid.y as f64 * d;
                centroid_accum[2] += mid.z as f64 * d;
            }
            len
        } else {
            0.0
        };

        *total_measure += measure;
        accumulate_field_weighted(pt_acc, &ug.point_data, &pts_usize, measure);
        if ug.cell_data.arrays().first().is_some_and(|a| ci < a.num_tuples()) {
            accumulate_field_weighted(cell_acc, &ug.cell_data, &[ci], measure);
        }
    }
}
