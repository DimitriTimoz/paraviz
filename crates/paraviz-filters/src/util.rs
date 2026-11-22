//! Internal shared utilities for geometry extraction, array interpolation,
//! point compaction, and spatial sampling across dataset types.

use glam::Vec3;
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, ImageData, PipelineError, PolyData,
    RectilinearGrid, StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Extracts the point coordinates of any concrete single-block `DataSet`.
pub(crate) fn dataset_points(ds: &DataSet) -> Result<Vec<[f32; 3]>, PipelineError> {
    match ds {
        DataSet::PolyData(pd) => Ok(pd.points.clone()),
        DataSet::ImageData(img) => Ok(img.all_points()),
        DataSet::RectilinearGrid(rg) => Ok(rg.all_points()),
        DataSet::StructuredGrid(sg) => Ok(sg.points.clone()),
        DataSet::UnstructuredGrid(ug) => Ok(ug.points.clone()),
        DataSet::MultiBlock(_) => Err(PipelineError::ExecutionFailed(
            "Expected single-block dataset, got MultiBlock".into(),
        )),
    }
}

/// Linearly interpolates a `FieldData` across edge samples `(pt_a, pt_b, t)`.
pub(crate) fn interpolate_field_data_edges(
    src: &FieldData,
    edges: &[(usize, usize, f64)],
) -> FieldData {
    let arrays: Vec<DataArray> = src
        .arrays()
        .iter()
        .map(|arr| interpolate_array_edges(arr, edges))
        .collect();
    let mut out = FieldData::new();
    for arr in arrays {
        out.add_array(arr);
    }
    out.active_scalars = src.active_scalars.clone();
    out.active_vectors = src.active_vectors.clone();
    out.active_normals = src.active_normals.clone();
    out
}

/// Linearly interpolates a single `DataArray` across edge samples `(pt_a, pt_b, t)`.
pub(crate) fn interpolate_array_edges(
    arr: &DataArray,
    edges: &[(usize, usize, f64)],
) -> DataArray {
    let nc = arr.num_components;
    let is_f64 = matches!(arr.data, ArrayData::F64(_));
    if is_f64 {
        let vals: Vec<f64> = edges
            .par_iter()
            .flat_map_iter(|&(a, b, t)| {
                let ba = a * nc;
                let bb = b * nc;
                (0..nc).map(move |c| {
                    let va = arr.data.get_f64(ba + c);
                    let vb = arr.data.get_f64(bb + c);
                    va + (vb - va) * t
                })
            })
            .collect();
        DataArray::new(arr.name.clone(), nc, ArrayData::F64(vals))
    } else {
        let vals: Vec<f32> = edges
            .par_iter()
            .flat_map_iter(|&(a, b, t)| {
                let ba = a * nc;
                let bb = b * nc;
                (0..nc).map(move |c| {
                    let va = arr.data.get_f64(ba + c);
                    let vb = arr.data.get_f64(bb + c);
                    (va + (vb - va) * t) as f32
                })
            })
            .collect();
        DataArray::new(arr.name.clone(), nc, ArrayData::F32(vals))
    }
}

/// Interpolates a `FieldData` using arbitrary weighted point tuples per output point.
pub(crate) fn interpolate_field_data_weighted(
    src: &FieldData,
    samples: &[Option<Vec<(usize, f64)>>],
) -> FieldData {
    let mut out = FieldData::new();
    for arr in src.arrays() {
        let nc = arr.num_components;
        let is_f64 = matches!(arr.data, ArrayData::F64(_));
        if is_f64 {
            let vals: Vec<f64> = samples
                .par_iter()
                .flat_map_iter(|sample| {
                    let mut acc = vec![0.0f64; nc];
                    if let Some(weights) = sample {
                        for &(idx, w) in weights {
                            let base = idx * nc;
                            for c in 0..nc {
                                acc[c] += arr.data.get_f64(base + c) * w;
                            }
                        }
                    }
                    acc.into_iter()
                })
                .collect();
            out.add_array(DataArray::new(arr.name.clone(), nc, ArrayData::F64(vals)));
        } else {
            let vals: Vec<f32> = samples
                .par_iter()
                .flat_map_iter(|sample| {
                    let mut acc = vec![0.0f64; nc];
                    if let Some(weights) = sample {
                        for &(idx, w) in weights {
                            let base = idx * nc;
                            for c in 0..nc {
                                acc[c] += arr.data.get_f64(base + c) * w;
                            }
                        }
                    }
                    acc.into_iter().map(|v| v as f32)
                })
                .collect();
            out.add_array(DataArray::new(arr.name.clone(), nc, ArrayData::F32(vals)));
        }
    }
    out.active_scalars = src.active_scalars.clone();
    out.active_vectors = src.active_vectors.clone();
    out.active_normals = src.active_normals.clone();
    out
}

/// Merges multiple `PolyData` meshes into a single `PolyData`, offsetting vertex indices
/// and concatenating matching `point_data` and `cell_data` arrays.
pub(crate) fn merge_polydata(parts: Vec<PolyData>) -> PolyData {
    if parts.is_empty() {
        return PolyData::new();
    }
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }

    let total_points: usize = parts.iter().map(|p| p.points.len()).sum();
    let mut points = Vec::with_capacity(total_points);
    let mut verts = Vec::new();
    let mut lines = Vec::new();
    let mut polys = Vec::new();
    let mut strips = Vec::new();

    // Find common point_data array names present in all non-empty parts.
    let non_empty_parts: Vec<&PolyData> = parts.iter().filter(|p| !p.points.is_empty()).collect();
    let common_point_arrays: Vec<String> = if let Some(first) = non_empty_parts.first() {
        first
            .point_data
            .arrays()
            .iter()
            .filter(|a| {
                non_empty_parts.iter().all(|p| {
                    p.point_data
                        .get(&a.name)
                        .is_some_and(|other| other.num_components == a.num_components)
                })
            })
            .map(|a| a.name.clone())
            .collect()
    } else {
        Vec::new()
    };

    let mut point_buffers: Vec<(String, usize, Vec<f32>)> = common_point_arrays
        .iter()
        .map(|name| {
            let nc = non_empty_parts[0].point_data.get(name).unwrap().num_components;
            (name.clone(), nc, Vec::with_capacity(total_points * nc))
        })
        .collect();

    for part in &parts {
        let offset = points.len() as u32;
        points.extend_from_slice(&part.points);
        for &v in &part.verts {
            verts.push(v + offset);
        }
        for line in &part.lines {
            lines.push(line.iter().map(|&v| v + offset).collect());
        }
        for poly in &part.polys {
            polys.push(poly.iter().map(|&v| v + offset).collect());
        }
        for strip in &part.strips {
            strips.push(strip.iter().map(|&v| v + offset).collect());
        }
        if !part.points.is_empty() {
            for (name, nc, buf) in &mut point_buffers {
                let arr = part.point_data.get(name).unwrap();
                for t in 0..part.points.len() {
                    let base = t * *nc;
                    for c in 0..*nc {
                        buf.push(arr.data.get_f64(base + c) as f32);
                    }
                }
            }
        }
    }

    let mut out = PolyData {
        points,
        verts,
        lines,
        polys,
        strips,
        ..Default::default()
    };
    for (name, nc, buf) in point_buffers {
        out.point_data
            .add_array(DataArray::new(name, nc, ArrayData::F32(buf)));
    }
    if let Some(first) = non_empty_parts.first() {
        if let Some(ref s) = first.point_data.active_scalars {
            if out.point_data.get(s).is_some() {
                out.point_data.active_scalars = Some(s.clone());
            }
        }
        if let Some(ref v) = first.point_data.active_vectors {
            if out.point_data.get(v).is_some() {
                out.point_data.active_vectors = Some(v.clone());
            }
        }
        if let Some(ref n) = first.point_data.active_normals {
            if out.point_data.get(n).is_some() {
                out.point_data.active_normals = Some(n.clone());
            }
        }
    }
    out
}

/// Compacts a `PolyData` to only reference used points and remaps `point_data`.
pub(crate) fn compact_polydata_points(mut pd: PolyData) -> PolyData {
    let n = pd.points.len();
    let mut used = vec![false; n];
    for &v in &pd.verts {
        if (v as usize) < n {
            used[v as usize] = true;
        }
    }
    for line in &pd.lines {
        for &v in line {
            if (v as usize) < n {
                used[v as usize] = true;
            }
        }
    }
    for poly in &pd.polys {
        for &v in poly {
            if (v as usize) < n {
                used[v as usize] = true;
            }
        }
    }
    for strip in &pd.strips {
        for &v in strip {
            if (v as usize) < n {
                used[v as usize] = true;
            }
        }
    }

    let mut remap = vec![u32::MAX; n];
    let mut old_indices = Vec::new();
    let mut new_points = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            remap[i] = new_points.len() as u32;
            new_points.push(pd.points[i]);
            old_indices.push(i);
        }
    }

    if old_indices.len() == n {
        return pd;
    }

    for v in &mut pd.verts {
        *v = remap[*v as usize];
    }
    for line in &mut pd.lines {
        for v in line {
            *v = remap[*v as usize];
        }
    }
    for poly in &mut pd.polys {
        for v in poly {
            *v = remap[*v as usize];
        }
    }
    for strip in &mut pd.strips {
        for v in strip {
            *v = remap[*v as usize];
        }
    }
    pd.points = new_points;
    pd.point_data = pd.point_data.gather(&old_indices);
    pd
}

/// Compacts an `UnstructuredGrid` to only keep used points and remaps `point_data`.
pub(crate) fn compact_unstructured_points(mut ug: UnstructuredGrid) -> UnstructuredGrid {
    let n = ug.points.len();
    let mut used = vec![false; n];
    for &v in &ug.connectivity {
        if (v as usize) < n {
            used[v as usize] = true;
        }
    }
    let mut remap = vec![u32::MAX; n];
    let mut old_indices = Vec::new();
    let mut new_points = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            remap[i] = new_points.len() as u32;
            new_points.push(ug.points[i]);
            old_indices.push(i);
        }
    }
    if old_indices.len() == n {
        return ug;
    }
    for v in &mut ug.connectivity {
        *v = remap[*v as usize];
    }
    ug.points = new_points;
    ug.point_data = ug.point_data.gather(&old_indices);
    ug
}

/// Canonical edge key for two vertex indices (order-independent).
#[inline]
pub(crate) fn canonical_edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

/// Spatial locator and point-data sampler supporting `ImageData`, `RectilinearGrid`,
/// `StructuredGrid`, and `UnstructuredGrid`.
pub(crate) enum SpatialSampler<'a> {
    Image(&'a ImageData),
    Rectilinear(&'a RectilinearGrid),
    Structured {
        grid: &'a StructuredGrid,
        accel: TetAccel,
    },
    Unstructured {
        grid: &'a UnstructuredGrid,
        accel: TetAccel,
    },
}

pub(crate) struct TetAccel {
    bounds_min: Vec3,
    inv_cell_size: Vec3,
    dims: [usize; 3],
    buckets: Vec<Vec<usize>>,
    tets: Vec<[u32; 4]>,
    points: Vec<[f32; 3]>,
}

impl TetAccel {
    fn from_structured(sg: &StructuredGrid) -> Self {
        let [nx, ny, nz] = sg.dimensions;
        let mut tets = Vec::new();
        if nx >= 2 && ny >= 2 && nz >= 2 {
            let hex_tets = CellType::Hexahedron.tetrahedra();
            tets.reserve((nx - 1) * (ny - 1) * (nz - 1) * 5);
            for k in 0..(nz - 1) {
                for j in 0..(ny - 1) {
                    for i in 0..(nx - 1) {
                        let c = [
                            sg.point_index(i, j, k) as u32,
                            sg.point_index(i + 1, j, k) as u32,
                            sg.point_index(i + 1, j + 1, k) as u32,
                            sg.point_index(i, j + 1, k) as u32,
                            sg.point_index(i, j, k + 1) as u32,
                            sg.point_index(i + 1, j, k + 1) as u32,
                            sg.point_index(i + 1, j + 1, k + 1) as u32,
                            sg.point_index(i, j + 1, k + 1) as u32,
                        ];
                        for t in &hex_tets {
                            tets.push([c[t[0]], c[t[1]], c[t[2]], c[t[3]]]);
                        }
                    }
                }
            }
        }
        Self::build(sg.points.clone(), tets)
    }

    fn from_unstructured(ug: &UnstructuredGrid) -> Self {
        let mut tets = Vec::new();
        for cell_idx in 0..ug.num_cells() {
            let ct = ug.cell_types[cell_idx];
            let pts = ug.cell_points(cell_idx);
            for t in ct.tetrahedra() {
                tets.push([pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]]);
            }
        }
        Self::build(ug.points.clone(), tets)
    }

    fn build(points: Vec<[f32; 3]>, tets: Vec<[u32; 4]>) -> Self {
        let mut bmin = Vec3::splat(f32::INFINITY);
        let mut bmax = Vec3::splat(f32::NEG_INFINITY);
        for &p in &points {
            let v = Vec3::from_array(p);
            bmin = bmin.min(v);
            bmax = bmax.max(v);
        }
        if tets.is_empty() || !bmin.is_finite() {
            return Self {
                bounds_min: Vec3::ZERO,
                inv_cell_size: Vec3::ZERO,
                dims: [1, 1, 1],
                buckets: vec![Vec::new()],
                tets,
                points,
            };
        }

        let extent = (bmax - bmin).max(Vec3::splat(1e-5));
        let n_per_axis = ((tets.len() as f32).cbrt().ceil() as usize).clamp(4, 32);
        let dims = [n_per_axis, n_per_axis, n_per_axis];
        let inv_cell_size = Vec3::new(
            dims[0] as f32 / extent.x,
            dims[1] as f32 / extent.y,
            dims[2] as f32 / extent.z,
        );
        let mut buckets = vec![Vec::new(); dims[0] * dims[1] * dims[2]];

        for (tid, tet) in tets.iter().enumerate() {
            let mut tmin = Vec3::splat(f32::INFINITY);
            let mut tmax = Vec3::splat(f32::NEG_INFINITY);
            for &vi in tet {
                let p = Vec3::from_array(points[vi as usize]);
                tmin = tmin.min(p);
                tmax = tmax.max(p);
            }
            let i0 = (((tmin.x - bmin.x) * inv_cell_size.x).floor() as isize)
                .clamp(0, dims[0] as isize - 1) as usize;
            let j0 = (((tmin.y - bmin.y) * inv_cell_size.y).floor() as isize)
                .clamp(0, dims[1] as isize - 1) as usize;
            let k0 = (((tmin.z - bmin.z) * inv_cell_size.z).floor() as isize)
                .clamp(0, dims[2] as isize - 1) as usize;
            let i1 = (((tmax.x - bmin.x) * inv_cell_size.x).floor() as isize)
                .clamp(0, dims[0] as isize - 1) as usize;
            let j1 = (((tmax.y - bmin.y) * inv_cell_size.y).floor() as isize)
                .clamp(0, dims[1] as isize - 1) as usize;
            let k1 = (((tmax.z - bmin.z) * inv_cell_size.z).floor() as isize)
                .clamp(0, dims[2] as isize - 1) as usize;

            for k in k0..=k1 {
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        buckets[i + dims[0] * (j + dims[1] * k)].push(tid);
                    }
                }
            }
        }

        Self {
            bounds_min: bmin,
            inv_cell_size,
            dims,
            buckets,
            tets,
            points,
        }
    }

    fn locate(&self, pos: Vec3) -> Option<Vec<(usize, f64)>> {
        if self.tets.is_empty() {
            return None;
        }
        let fx = (pos.x - self.bounds_min.x) * self.inv_cell_size.x;
        let fy = (pos.y - self.bounds_min.y) * self.inv_cell_size.y;
        let fz = (pos.z - self.bounds_min.z) * self.inv_cell_size.z;
        if fx < -1e-4
            || fy < -1e-4
            || fz < -1e-4
            || fx > self.dims[0] as f32 + 1e-4
            || fy > self.dims[1] as f32 + 1e-4
            || fz > self.dims[2] as f32 + 1e-4
        {
            return None;
        }
        let ix = (fx.floor() as isize).clamp(0, self.dims[0] as isize - 1) as usize;
        let iy = (fy.floor() as isize).clamp(0, self.dims[1] as isize - 1) as usize;
        let iz = (fz.floor() as isize).clamp(0, self.dims[2] as isize - 1) as usize;

        let bucket = &self.buckets[ix + self.dims[0] * (iy + self.dims[1] * iz)];
        let mut best: Option<(f64, Vec<(usize, f64)>)> = None;

        for &tid in bucket {
            let [a, b, c, d] = self.tets[tid];
            let p0 = Vec3::from_array(self.points[a as usize]);
            let p1 = Vec3::from_array(self.points[b as usize]);
            let p2 = Vec3::from_array(self.points[c as usize]);
            let p3 = Vec3::from_array(self.points[d as usize]);

            if let Some(w) = barycentric_tet(pos, p0, p1, p2, p3) {
                let min_w = w.into_iter().fold(f64::INFINITY, f64::min);
                if min_w >= -1e-5 {
                    let weights = vec![
                        (a as usize, w[0]),
                        (b as usize, w[1]),
                        (c as usize, w[2]),
                        (d as usize, w[3]),
                    ];
                    if min_w >= 0.0 {
                        return Some(weights);
                    }
                    if best.as_ref().is_none_or(|(bw, _)| min_w > *bw) {
                        best = Some((min_w, weights));
                    }
                }
            }
        }
        best.map(|(_, w)| w)
    }
}

#[inline]
fn barycentric_tet(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Option<[f64; 4]> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = d - a;
    let det = v0.dot(v1.cross(v2)) as f64;
    if det.abs() < 1e-18 {
        return None;
    }
    let vp = p - a;
    let inv_det = 1.0 / det;
    let w1 = (vp.dot(v1.cross(v2)) as f64) * inv_det;
    let w2 = (v0.dot(vp.cross(v2)) as f64) * inv_det;
    let w3 = (v0.dot(v1.cross(vp)) as f64) * inv_det;
    let w0 = 1.0 - w1 - w2 - w3;
    Some([w0, w1, w2, w3])
}

impl<'a> SpatialSampler<'a> {
    pub(crate) fn new(ds: &'a DataSet) -> Result<Self, PipelineError> {
        match ds {
            DataSet::ImageData(img) => Ok(Self::Image(img)),
            DataSet::RectilinearGrid(rg) => Ok(Self::Rectilinear(rg)),
            DataSet::StructuredGrid(sg) => Ok(Self::Structured {
                grid: sg,
                accel: TetAccel::from_structured(sg),
            }),
            DataSet::UnstructuredGrid(ug) => Ok(Self::Unstructured {
                grid: ug,
                accel: TetAccel::from_unstructured(ug),
            }),
            _ => Err(PipelineError::ExecutionFailed(format!(
                "Spatial sampling is not supported on {}",
                ds.type_name()
            ))),
        }
    }

    pub(crate) fn field_data(&self) -> &'a FieldData {
        match self {
            Self::Image(img) => &img.point_data,
            Self::Rectilinear(rg) => &rg.point_data,
            Self::Structured { grid, .. } => &grid.point_data,
            Self::Unstructured { grid, .. } => &grid.point_data,
        }
    }

    /// Locates the sample weights `(point_index, weight)` at world position `pos`.
    pub(crate) fn sample_weights(&self, pos: Vec3) -> Option<Vec<(usize, f64)>> {
        match self {
            Self::Image(img) => sample_image_weights(img, pos),
            Self::Rectilinear(rg) => sample_rectilinear_weights(rg, pos),
            Self::Structured { accel, .. } => accel.locate(pos),
            Self::Unstructured { accel, .. } => accel.locate(pos),
        }
    }

    /// Samples a 3D vector array at `pos`.
    pub(crate) fn sample_vec3(&self, array: &DataArray, pos: Vec3) -> Option<Vec3> {
        let weights = self.sample_weights(pos)?;
        let nc = array.num_components;
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut z = 0.0f64;
        for (idx, w) in weights {
            let base = idx * nc;
            x += array.data.get_f64(base) * w;
            if nc > 1 {
                y += array.data.get_f64(base + 1) * w;
            }
            if nc > 2 {
                z += array.data.get_f64(base + 2) * w;
            }
        }
        Some(Vec3::new(x as f32, y as f32, z as f32))
    }
}

fn trilinear_weights(
    idx: [usize; 8],
    tx: f64,
    ty: f64,
    tz: f64,
) -> Vec<(usize, f64)> {
    let wx0 = 1.0 - tx;
    let wy0 = 1.0 - ty;
    let wz0 = 1.0 - tz;
    vec![
        (idx[0], wx0 * wy0 * wz0),
        (idx[1], tx * wy0 * wz0),
        (idx[2], wx0 * ty * wz0),
        (idx[3], tx * ty * wz0),
        (idx[4], wx0 * wy0 * tz),
        (idx[5], tx * wy0 * tz),
        (idx[6], wx0 * ty * tz),
        (idx[7], tx * ty * tz),
    ]
}

fn sample_image_weights(img: &ImageData, pos: Vec3) -> Option<Vec<(usize, f64)>> {
    let [nx, ny, nz] = img.dimensions;
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }
    let fx = if nx > 1 {
        (pos.x as f64 - img.origin[0]) / img.spacing[0]
    } else if (pos.x as f64 - img.origin[0]).abs() <= 1e-5 {
        0.0
    } else {
        return None;
    };
    let fy = if ny > 1 {
        (pos.y as f64 - img.origin[1]) / img.spacing[1]
    } else if (pos.y as f64 - img.origin[1]).abs() <= 1e-5 {
        0.0
    } else {
        return None;
    };
    let fz = if nz > 1 {
        (pos.z as f64 - img.origin[2]) / img.spacing[2]
    } else if (pos.z as f64 - img.origin[2]).abs() <= 1e-5 {
        0.0
    } else {
        return None;
    };

    let eps = 1e-6;
    if fx < -eps
        || fy < -eps
        || fz < -eps
        || fx > (nx - 1) as f64 + eps
        || fy > (ny - 1) as f64 + eps
        || fz > (nz - 1) as f64 + eps
    {
        return None;
    }
    let fx = fx.clamp(0.0, (nx - 1) as f64);
    let fy = fy.clamp(0.0, (ny - 1) as f64);
    let fz = fz.clamp(0.0, (nz - 1) as f64);

    let i0 = (fx.floor() as usize).min(nx.saturating_sub(2));
    let j0 = (fy.floor() as usize).min(ny.saturating_sub(2));
    let k0 = (fz.floor() as usize).min(nz.saturating_sub(2));
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let k1 = (k0 + 1).min(nz - 1);

    let tx = if nx > 1 { fx - i0 as f64 } else { 0.0 };
    let ty = if ny > 1 { fy - j0 as f64 } else { 0.0 };
    let tz = if nz > 1 { fz - k0 as f64 } else { 0.0 };

    let idx = [
        img.point_index(i0, j0, k0),
        img.point_index(i1, j0, k0),
        img.point_index(i0, j1, k0),
        img.point_index(i1, j1, k0),
        img.point_index(i0, j0, k1),
        img.point_index(i1, j0, k1),
        img.point_index(i0, j1, k1),
        img.point_index(i1, j1, k1),
    ];
    Some(trilinear_weights(idx, tx, ty, tz))
}

fn find_rect_coord(coords: &[f32], val: f32) -> Option<(usize, usize, f64)> {
    if coords.is_empty() {
        return None;
    }
    if coords.len() == 1 {
        return ((val - coords[0]).abs() <= 1e-5).then_some((0, 0, 0.0));
    }
    let n = coords.len();
    let ascending = coords[n - 1] >= coords[0];
    let (cmin, cmax) = if ascending {
        (coords[0], coords[n - 1])
    } else {
        (coords[n - 1], coords[0])
    };
    let eps = 1e-5;
    if val < cmin - eps || val > cmax + eps {
        return None;
    }
    let v = val.clamp(cmin, cmax);
    let i0 = if ascending {
        let pos = coords.partition_point(|&c| c <= v);
        pos.saturating_sub(1).min(n - 2)
    } else {
        let pos = coords.partition_point(|&c| c >= v);
        pos.saturating_sub(1).min(n - 2)
    };
    let i1 = i0 + 1;
    let denom = (coords[i1] - coords[i0]) as f64;
    let t = if denom.abs() > 1e-12 {
        ((v - coords[i0]) as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((i0, i1, t))
}

fn sample_rectilinear_weights(rg: &RectilinearGrid, pos: Vec3) -> Option<Vec<(usize, f64)>> {
    let (i0, i1, tx) = find_rect_coord(&rg.x_coords, pos.x)?;
    let (j0, j1, ty) = find_rect_coord(&rg.y_coords, pos.y)?;
    let (k0, k1, tz) = find_rect_coord(&rg.z_coords, pos.z)?;
    let idx = [
        rg.point_index(i0, j0, k0),
        rg.point_index(i1, j0, k0),
        rg.point_index(i0, j1, k0),
        rg.point_index(i1, j1, k0),
        rg.point_index(i0, j0, k1),
        rg.point_index(i1, j0, k1),
        rg.point_index(i0, j1, k1),
        rg.point_index(i1, j1, k1),
    ];
    Some(trilinear_weights(idx, tx, ty, tz))
}

/// Deduplicates vertices keyed by `u64` edge keys in deterministic order, returning
/// `(dedup_points, dedup_edges, tris)`.
pub(crate) fn deduplicate_edge_triangles(
    raw_tris: Vec<[(u64, [f32; 3], (usize, usize, f64)); 3]>,
    merge_vertices: bool,
) -> (Vec<[f32; 3]>, Vec<(usize, usize, f64)>, Vec<[u32; 3]>) {
    if !merge_vertices {
        let mut points = Vec::with_capacity(raw_tris.len() * 3);
        let mut edges = Vec::with_capacity(raw_tris.len() * 3);
        let mut tris = Vec::with_capacity(raw_tris.len());
        for tri in raw_tris {
            let base = points.len() as u32;
            for (_, p, e) in tri {
                points.push(p);
                edges.push(e);
            }
            tris.push([base, base + 1, base + 2]);
        }
        return (points, edges, tris);
    }

    let mut map: HashMap<u64, u32> = HashMap::with_capacity(raw_tris.len() * 2);
    let mut points = Vec::with_capacity(raw_tris.len() * 2);
    let mut edges = Vec::with_capacity(raw_tris.len() * 2);
    let mut tris = Vec::with_capacity(raw_tris.len());

    for tri in raw_tris {
        let mut out_idx = [0u32; 3];
        for (k, (key, p, e)) in tri.into_iter().enumerate() {
            let idx = *map.entry(key).or_insert_with(|| {
                let id = points.len() as u32;
                points.push(p);
                edges.push(e);
                id
            });
            out_idx[k] = idx;
        }
        if out_idx[0] != out_idx[1] && out_idx[1] != out_idx[2] && out_idx[2] != out_idx[0] {
            tris.push(out_idx);
        }
    }
    (points, edges, tris)
}
