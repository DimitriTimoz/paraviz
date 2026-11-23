//! Implicit-function mesh clipping (`ClipFilter` + `ImplicitFunction`, equivalent to `vtkClipDataSet`).

use crate::contour::interp_edge_pos;
use crate::util::{
    canonical_edge_key, compact_polydata_points, compact_unstructured_points,
    interpolate_field_data_edges, merge_polydata, FxHashMap, GridPoints,
};
use glam::Vec3;
use paraviz_core::{
    CellType, DataSet, FieldData, Filter, PipelineError, PolyData, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Analytical implicit scalar function `f(p)` whose zero-contour defines the clip boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImplicitFunction {
    /// Signed distance `(p - origin) · normal.normalize()`.
    Plane { origin: [f32; 3], normal: [f32; 3] },
    /// Signed distance `|p - center| - radius` (negative inside, positive outside).
    Sphere { center: [f32; 3], radius: f32 },
    /// Signed distance to axis-aligned box `[min, max]` (negative inside, positive outside).
    Box { min: [f32; 3], max: [f32; 3] },
}

impl ImplicitFunction {
    /// Evaluates the signed distance at world position `p`.
    #[inline]
    pub fn evaluate(&self, p: [f32; 3]) -> f64 {
        let v = Vec3::from_array(p);
        match *self {
            Self::Plane { origin, normal } => {
                let n = Vec3::from_array(normal).normalize_or(Vec3::Z);
                (v - Vec3::from_array(origin)).dot(n) as f64
            }
            Self::Sphere { center, radius } => {
                ((v - Vec3::from_array(center)).length() - radius) as f64
            }
            Self::Box { min, max } => {
                let bmin = Vec3::from_array(min).min(Vec3::from_array(max));
                let bmax = Vec3::from_array(min).max(Vec3::from_array(max));
                let c = (bmin + bmax) * 0.5;
                let h = (bmax - bmin) * 0.5;
                let q = (v - c).abs() - h;
                let outside = q.max(Vec3::ZERO).length();
                let inside = q.x.max(q.y).max(q.z).min(0.0);
                (outside + inside) as f64
            }
        }
    }
}

/// Smoothly (or crinkle) clips cells along the zero-contour of an [`ImplicitFunction`].
///
/// By default (`invert: false`), keeps the region where `f(p) >= 0`.
/// When `invert: true`, keeps the region where `f(p) <= 0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipFilter {
    pub function: ImplicitFunction,
    pub invert: bool,
    pub crinkle: bool,
}

impl Default for ClipFilter {
    fn default() -> Self {
        Self {
            function: ImplicitFunction::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            invert: false,
            crinkle: false,
        }
    }
}

impl ClipFilter {
    pub fn new(function: ImplicitFunction) -> Self {
        Self {
            function,
            invert: false,
            crinkle: false,
        }
    }
}

impl Filter for ClipFilter {
    fn name(&self) -> &str {
        "ClipFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let sign = if self.invert { -1.0 } else { 1.0 };
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(clip_polydata(
                pd,
                &self.function,
                sign,
                self.crinkle,
            ))),
            DataSet::ImageData(img) => Ok(DataSet::UnstructuredGrid(clip_structured(
                img.dimensions,
                GridPoints::Image(img),
                &img.point_data,
                &img.cell_data,
                CellType::Voxel,
                &self.function,
                sign,
                self.crinkle,
            ))),
            DataSet::RectilinearGrid(rg) => Ok(DataSet::UnstructuredGrid(clip_structured(
                rg.dimensions(),
                GridPoints::Rectilinear(rg),
                &rg.point_data,
                &rg.cell_data,
                CellType::Voxel,
                &self.function,
                sign,
                self.crinkle,
            ))),
            DataSet::StructuredGrid(sg) => Ok(DataSet::UnstructuredGrid(clip_structured(
                sg.dimensions,
                GridPoints::from_structured(&sg.points, sg.dimensions),
                &sg.point_data,
                &sg.cell_data,
                CellType::Hexahedron,
                &self.function,
                sign,
                self.crinkle,
            ))),
            DataSet::UnstructuredGrid(ug) => Ok(DataSet::UnstructuredGrid(clip_unstructured(
                ug,
                &self.function,
                sign,
                self.crinkle,
            ))),
            DataSet::MultiBlock(mb) => {
                let blocks = mb
                    .blocks
                    .par_iter()
                    .map(|(name, block)| Ok((name.clone(), self.apply(block)?)))
                    .collect::<Result<Vec<_>, PipelineError>>()?;
                // If all blocks are PolyData, still preserve MultiBlock structure cleanly.
                let mut out = paraviz_core::MultiBlockDataSet::new();
                for (name, ds) in blocks {
                    out.add_block(name, ds);
                }
                Ok(DataSet::MultiBlock(out))
            }
        }
    }
}

fn eval_signed(points: &[[f32; 3]], func: &ImplicitFunction, sign: f64) -> Vec<f64> {
    points
        .par_iter()
        .map(|&p| func.evaluate(p) * sign)
        .collect()
}

fn eval_signed_grid(points: GridPoints<'_>, func: &ImplicitFunction, sign: f64) -> Vec<f64> {
    match points {
        GridPoints::Explicit { points, .. } => eval_signed(points, func, sign),
        GridPoints::Image(img) => {
            let [nx, ny, nz] = img.dimensions;
            (0..(nx * ny * nz))
                .into_par_iter()
                .map(|idx| {
                    let i = idx % nx;
                    let rem = idx / nx;
                    let j = rem % ny;
                    let k = rem / ny;
                    func.evaluate(img.point_coords(i, j, k)) * sign
                })
                .collect()
        }
        GridPoints::Rectilinear(rg) => {
            let [nx, ny, nz] = rg.dimensions();
            (0..(nx * ny * nz))
                .into_par_iter()
                .map(|idx| {
                    let i = idx % nx;
                    let rem = idx / nx;
                    let j = rem % ny;
                    let k = rem / ny;
                    func.evaluate(rg.point_coords(i, j, k)) * sign
                })
                .collect()
        }
    }
}

struct EdgeCache<'a> {
    orig_points: GridPoints<'a>,
    vals: &'a [f64],
    out_points: Vec<[f32; 3]>,
    out_edges: Vec<(usize, usize, f64)>,
    edge_map: FxHashMap<u64, u32>,
    vert_remap: Vec<u32>,
}

impl<'a> EdgeCache<'a> {
    fn new(orig_points: &'a [[f32; 3]], vals: &'a [f64]) -> Self {
        let n = orig_points.len();
        let mut out_edges = Vec::with_capacity(n);
        for i in 0..n {
            out_edges.push((i, i, 0.0));
        }
        Self {
            orig_points: GridPoints::Explicit {
                points: orig_points,
                nx: n,
                ny: 1,
            },
            vals,
            out_points: orig_points.to_vec(),
            out_edges,
            edge_map: FxHashMap::default(),
            vert_remap: Vec::new(),
        }
    }

    fn new_lazy(orig_points: GridPoints<'a>, vals: &'a [f64]) -> Self {
        let n = orig_points.len();
        Self {
            orig_points,
            vals,
            out_points: Vec::new(),
            out_edges: Vec::new(),
            edge_map: FxHashMap::default(),
            vert_remap: vec![u32::MAX; n],
        }
    }

    #[inline]
    fn vertex_id(&mut self, orig_idx: usize) -> u32 {
        if self.vert_remap.is_empty() {
            return orig_idx as u32;
        }
        let slot = &mut self.vert_remap[orig_idx];
        if *slot != u32::MAX {
            return *slot;
        }
        let id = self.out_points.len() as u32;
        self.out_points.push(self.orig_points.at_linear(orig_idx));
        self.out_edges.push((orig_idx, orig_idx, 0.0));
        *slot = id;
        id
    }

    fn get_or_insert_intersection(&mut self, ia: usize, ib: usize) -> u32 {
        let key = canonical_edge_key(ia as u32, ib as u32);
        if let Some(&id) = self.edge_map.get(&key) {
            return id;
        }
        let pa = self.orig_points.at_linear(ia);
        let pb = self.orig_points.at_linear(ib);
        let (t, pos) = interp_edge_pos(pa, pb, self.vals[ia], self.vals[ib], 0.0);
        let id = self.out_points.len() as u32;
        self.out_points.push(pos);
        self.out_edges.push((ia, ib, t));
        self.edge_map.insert(key, id);
        id
    }
}

fn clip_polydata(
    pd: &PolyData,
    func: &ImplicitFunction,
    sign: f64,
    crinkle: bool,
) -> PolyData {
    if pd.points.is_empty() {
        return PolyData::new();
    }
    let vals = eval_signed(&pd.points, func, sign);

    if crinkle {
        let mut polys = Vec::new();
        for poly in &pd.polys {
            if poly.iter().any(|&v| vals[v as usize] >= 0.0) {
                polys.push(poly.clone());
            }
        }
        let mut lines = Vec::new();
        for line in &pd.lines {
            if line.iter().any(|&v| vals[v as usize] >= 0.0) {
                lines.push(line.clone());
            }
        }
        let verts: Vec<u32> = pd
            .verts
            .iter()
            .copied()
            .filter(|&v| vals[v as usize] >= 0.0)
            .collect();
        return compact_polydata_points(PolyData {
            points: pd.points.clone(),
            verts,
            lines,
            polys,
            point_data: pd.point_data.clone(),
            ..Default::default()
        });
    }

    let mut cache = EdgeCache::new(&pd.points, &vals);
    let mut out_polys: Vec<Vec<u32>> = Vec::new();
    let mut out_lines: Vec<Vec<u32>> = Vec::new();
    let out_verts: Vec<u32> = pd
        .verts
        .iter()
        .copied()
        .filter(|&v| vals[v as usize] >= 0.0)
        .collect();

    // Sutherland-Hodgman polygon clipping against f(p) >= 0.
    let tris_or_polys: Vec<Vec<u32>> = if pd.strips.is_empty() {
        pd.polys.clone()
    } else {
        let mut combined = pd.polys.clone();
        for strip in &pd.strips {
            for k in 0..strip.len().saturating_sub(2) {
                if k % 2 == 0 {
                    combined.push(vec![strip[k], strip[k + 1], strip[k + 2]]);
                } else {
                    combined.push(vec![strip[k + 1], strip[k], strip[k + 2]]);
                }
            }
        }
        combined
    };

    for poly in &tris_or_polys {
        if poly.len() < 3 {
            continue;
        }
        let mut output = Vec::with_capacity(poly.len() + 2);
        let n = poly.len();
        for i in 0..n {
            let cur = poly[i] as usize;
            let nxt = poly[(i + 1) % n] as usize;
            let dc = vals[cur];
            let dn = vals[nxt];
            let c_in = dc >= 0.0;
            let n_in = dn >= 0.0;
            if c_in {
                output.push(cur as u32);
                if !n_in {
                    output.push(cache.get_or_insert_intersection(cur, nxt));
                }
            } else if n_in {
                output.push(cache.get_or_insert_intersection(cur, nxt));
            }
        }
        if output.len() >= 3 {
            // Remove consecutive duplicates if intersection hit an exact vertex.
            output.dedup();
            if output.len() >= 2 && output.first() == output.last() {
                output.pop();
            }
            if output.len() >= 3 {
                out_polys.push(output);
            }
        }
    }

    for line in &pd.lines {
        if line.len() < 2 {
            continue;
        }
        let mut cur_seg: Vec<u32> = Vec::new();
        for w in line.windows(2) {
            let ia = w[0] as usize;
            let ib = w[1] as usize;
            let da = vals[ia];
            let db = vals[ib];
            match (da >= 0.0, db >= 0.0) {
                (true, true) => {
                    if cur_seg.is_empty() {
                        cur_seg.push(ia as u32);
                    }
                    cur_seg.push(ib as u32);
                }
                (true, false) => {
                    if cur_seg.is_empty() {
                        cur_seg.push(ia as u32);
                    }
                    let mid = cache.get_or_insert_intersection(ia, ib);
                    cur_seg.push(mid);
                    out_lines.push(std::mem::take(&mut cur_seg));
                }
                (false, true) => {
                    let mid = cache.get_or_insert_intersection(ia, ib);
                    cur_seg.push(mid);
                    cur_seg.push(ib as u32);
                }
                (false, false) => {}
            }
        }
        if cur_seg.len() >= 2 {
            out_lines.push(cur_seg);
        }
    }

    let pdata = interpolate_field_data_edges(&pd.point_data, &cache.out_edges);
    let mut out = compact_polydata_points(PolyData {
        points: cache.out_points,
        verts: out_verts,
        lines: out_lines,
        polys: out_polys,
        point_data: pdata,
        ..Default::default()
    });
    if pd.point_data.active_normals_array().is_some() && !out.polys.is_empty() {
        out.compute_normals();
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn clip_structured(
    dims: [usize; 3],
    points: GridPoints<'_>,
    point_data: &FieldData,
    cell_data: &FieldData,
    cell_type: CellType,
    func: &ImplicitFunction,
    sign: f64,
    crinkle: bool,
) -> UnstructuredGrid {
    let [nx, ny, nz] = dims;
    if nx < 2 || ny < 2 || nz < 2 || points.is_empty() {
        return UnstructuredGrid::new();
    }
    let vals = eval_signed_grid(points, func, sign);
    let cx = nx - 1;
    let cy = ny - 1;
    let cz = nz - 1;
    let total_cells = cx * cy * cz;

    let mut ug = UnstructuredGrid::new();
    let mut cache = EdgeCache::new_lazy(points, &vals);
    let mut out_cell_indices = Vec::new();
    let tets = cell_type.tetrahedra();

    for cell_idx in 0..total_cells {
        let k = cell_idx / (cx * cy);
        let rem = cell_idx % (cx * cy);
        let j = rem / cx;
        let i = rem % cx;
        let idx = |ii: usize, jj: usize, kk: usize| (ii + nx * (jj + ny * kk)) as u32;
        let c = match cell_type {
            CellType::Voxel => [
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i, j + 1, k),
                idx(i + 1, j + 1, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i, j + 1, k + 1),
                idx(i + 1, j + 1, k + 1),
            ],
            _ => [
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j + 1, k),
                idx(i, j + 1, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i + 1, j + 1, k + 1),
                idx(i, j + 1, k + 1),
            ],
        };

        let mut all_in = true;
        let mut any_in = false;
        for &vi in &c {
            if vals[vi as usize] >= 0.0 {
                any_in = true;
            } else {
                all_in = false;
            }
        }

        if !any_in {
            continue;
        }
        if all_in || crinkle {
            let remapped = [
                cache.vertex_id(c[0] as usize),
                cache.vertex_id(c[1] as usize),
                cache.vertex_id(c[2] as usize),
                cache.vertex_id(c[3] as usize),
                cache.vertex_id(c[4] as usize),
                cache.vertex_id(c[5] as usize),
                cache.vertex_id(c[6] as usize),
                cache.vertex_id(c[7] as usize),
            ];
            ug.push_cell(cell_type, &remapped);
            out_cell_indices.push(cell_idx);
            continue;
        }

        for tet in &tets {
            let tv = [
                c[tet[0]] as usize,
                c[tet[1]] as usize,
                c[tet[2]] as usize,
                c[tet[3]] as usize,
            ];
            let added = clip_tet(tv, &vals, &mut cache, &mut ug);
            for _ in 0..added {
                out_cell_indices.push(cell_idx);
            }
        }
    }

    ug.points = cache.out_points;
    ug.point_data = interpolate_field_data_edges(point_data, &cache.out_edges);
    if !cell_data.is_empty() && !out_cell_indices.is_empty() {
        ug.cell_data = cell_data.gather(&out_cell_indices);
    }
    compact_unstructured_points(ug)
}

fn clip_unstructured(
    in_ug: &UnstructuredGrid,
    func: &ImplicitFunction,
    sign: f64,
    crinkle: bool,
) -> UnstructuredGrid {
    let n_cells = in_ug.num_cells();
    if n_cells == 0 {
        return UnstructuredGrid::new();
    }
    let vals = eval_signed(&in_ug.points, func, sign);
    let mut ug = UnstructuredGrid::new();
    let mut cache = EdgeCache::new(&in_ug.points, &vals);
    let mut out_cell_indices = Vec::new();

    for cell_idx in 0..n_cells {
        let ct = in_ug.cell_types[cell_idx];
        let pts = in_ug.cell_points(cell_idx);
        let mut all_in = true;
        let mut any_in = false;
        for &vi in pts {
            if vals[vi as usize] >= 0.0 {
                any_in = true;
            } else {
                all_in = false;
            }
        }
        if !any_in {
            continue;
        }
        if all_in || crinkle {
            ug.push_cell(ct, pts);
            out_cell_indices.push(cell_idx);
            continue;
        }

        if ct.dimension() == 3 {
            for tet in ct.tetrahedra() {
                let tv = [
                    pts[tet[0]] as usize,
                    pts[tet[1]] as usize,
                    pts[tet[2]] as usize,
                    pts[tet[3]] as usize,
                ];
                let added = clip_tet(tv, &vals, &mut cache, &mut ug);
                for _ in 0..added {
                    out_cell_indices.push(cell_idx);
                }
            }
        } else if ct.dimension() == 2 {
            // Clip 2D polygon
            let mut poly = Vec::new();
            let n = pts.len();
            for i in 0..n {
                let c = pts[i] as usize;
                let nx = pts[(i + 1) % n] as usize;
                let cin = vals[c] >= 0.0;
                let nin = vals[nx] >= 0.0;
                if cin {
                    poly.push(c as u32);
                    if !nin {
                        poly.push(cache.get_or_insert_intersection(c, nx));
                    }
                } else if nin {
                    poly.push(cache.get_or_insert_intersection(c, nx));
                }
            }
            poly.dedup();
            if poly.len() >= 3 {
                let out_ct = match poly.len() {
                    3 => CellType::Triangle,
                    4 => CellType::Quad,
                    _ => CellType::Polygon,
                };
                ug.push_cell(out_ct, &poly);
                out_cell_indices.push(cell_idx);
            }
        }
    }

    ug.points = cache.out_points;
    ug.point_data = interpolate_field_data_edges(&in_ug.point_data, &cache.out_edges);
    if !in_ug.cell_data.is_empty() && !out_cell_indices.is_empty() {
        ug.cell_data = in_ug.cell_data.gather(&out_cell_indices);
    }
    compact_unstructured_points(ug)
}

/// Pushes a tetrahedron `[a, b, c, d]` ensuring its orientation matches `expected_positive`.
fn push_oriented_tet(
    ug: &mut UnstructuredGrid,
    cache: &EdgeCache<'_>,
    verts: [u32; 4],
    expected_positive: bool,
) {
    let p0 = Vec3::from_array(cache.out_points[verts[0] as usize]);
    let p1 = Vec3::from_array(cache.out_points[verts[1] as usize]);
    let p2 = Vec3::from_array(cache.out_points[verts[2] as usize]);
    let p3 = Vec3::from_array(cache.out_points[verts[3] as usize]);
    let det = (p1 - p0).cross(p2 - p0).dot(p3 - p0);
    if det.abs() < 1e-18 {
        return;
    }
    if (det > 0.0) == expected_positive {
        ug.push_cell(CellType::Tetra, &verts);
    } else {
        ug.push_cell(CellType::Tetra, &[verts[0], verts[2], verts[1], verts[3]]);
    }
}

/// Smoothly clips a tetrahedron against `vals[v] >= 0.0`, returning how many tets were emitted.
fn clip_tet(
    tv: [usize; 4],
    vals: &[f64],
    cache: &mut EdgeCache<'_>,
    ug: &mut UnstructuredGrid,
) -> usize {
    let mut inside = [0usize; 4];
    let mut outside = [0usize; 4];
    let mut n_in = 0usize;
    let mut n_out = 0usize;
    for &v in &tv {
        if vals[v] >= 0.0 {
            inside[n_in] = v;
            n_in += 1;
        } else {
            outside[n_out] = v;
            n_out += 1;
        }
    }

    if n_in == 0 {
        return 0;
    }
    if n_in == 4 {
        let v0 = cache.vertex_id(tv[0]);
        let v1 = cache.vertex_id(tv[1]);
        let v2 = cache.vertex_id(tv[2]);
        let v3 = cache.vertex_id(tv[3]);
        ug.push_cell(CellType::Tetra, &[v0, v1, v2, v3]);
        return 1;
    }

    let p0 = Vec3::from_array(cache.orig_points.at_linear(tv[0]));
    let p1 = Vec3::from_array(cache.orig_points.at_linear(tv[1]));
    let p2 = Vec3::from_array(cache.orig_points.at_linear(tv[2]));
    let p3 = Vec3::from_array(cache.orig_points.at_linear(tv[3]));
    let pos_orient = (p1 - p0).cross(p2 - p0).dot(p3 - p0) >= 0.0;

    let before = ug.num_cells();
    match n_in {
        1 => {
            let a = inside[0];
            let ab = cache.get_or_insert_intersection(a, outside[0]);
            let ac = cache.get_or_insert_intersection(a, outside[1]);
            let ad = cache.get_or_insert_intersection(a, outside[2]);
            let va = cache.vertex_id(a);
            push_oriented_tet(ug, cache, [va, ab, ac, ad], pos_orient);
        }
        2 => {
            let a = inside[0];
            let b = inside[1];
            let c = outside[0];
            let d = outside[1];
            let ac = cache.get_or_insert_intersection(a, c);
            let ad = cache.get_or_insert_intersection(a, d);
            let bc = cache.get_or_insert_intersection(b, c);
            let bd = cache.get_or_insert_intersection(b, d);
            let va = cache.vertex_id(a);
            let vb = cache.vertex_id(b);
            // Decompose wedge [a, b, ac, ad, bc, bd] into 3 tets.
            push_oriented_tet(ug, cache, [va, vb, ac, ad], pos_orient);
            push_oriented_tet(ug, cache, [vb, ac, ad, bd], pos_orient);
            push_oriented_tet(ug, cache, [vb, ac, bd, bc], pos_orient);
        }
        3 => {
            let a = inside[0];
            let b = inside[1];
            let c = inside[2];
            let d = outside[0];
            let ad = cache.get_or_insert_intersection(a, d);
            let bd = cache.get_or_insert_intersection(b, d);
            let cd = cache.get_or_insert_intersection(c, d);
            let va = cache.vertex_id(a);
            let vb = cache.vertex_id(b);
            let vc = cache.vertex_id(c);
            // Decompose truncated tet (prism [a, b, c, ad, bd, cd]) into 3 tets.
            push_oriented_tet(ug, cache, [va, vb, vc, ad], pos_orient);
            push_oriented_tet(ug, cache, [vb, vc, ad, bd], pos_orient);
            push_oriented_tet(ug, cache, [vc, ad, bd, cd], pos_orient);
        }
        _ => {}
    }
    ug.num_cells() - before
}

// Keep unused import warning away if merge_polydata is used only in clip tests/future
#[allow(dead_code)]
fn _use_merge(p: Vec<PolyData>) -> PolyData {
    merge_polydata(p)
}
