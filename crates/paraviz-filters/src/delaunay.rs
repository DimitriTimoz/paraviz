//! Delaunay 2D & 3D triangulation filters (`Delaunay2DFilter`, `Delaunay3DFilter`).
//!
//! Uses the Bowyer-Watson incremental algorithm to reconstruct triangulated surfaces
//! (`PolyData`) or tetrahedral meshes (`UnstructuredGrid`) from raw point clouds.

use glam::{DVec2, DVec3};
use paraviz_core::{
    CellType, DataSet, Filter, PipelineError, PolyData, UnstructuredGrid,
};
use serde::{Deserialize, Serialize};

/// 2D Delaunay triangulation projected onto the XY plane (preserving original Z coordinates).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delaunay2DFilter {
    /// Optional circumradius cutoff (`0.0` = keep all triangles).
    pub alpha: f32,
}

impl Default for Delaunay2DFilter {
    fn default() -> Self {
        Self { alpha: 0.0 }
    }
}

impl Filter for Delaunay2DFilter {
    fn name(&self) -> &str {
        "Delaunay2DFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let (points, point_data) = extract_points_and_data(input)?;
        if points.len() < 3 {
            return Err(PipelineError::ExecutionFailed(
                "Delaunay2DFilter requires at least 3 points".into(),
            ));
        }
        let tris = bowyer_watson_2d(&points, self.alpha as f64);
        let mut pd = PolyData::from_triangles(points, tris);
        pd.point_data = point_data;
        pd.compute_normals();
        Ok(DataSet::PolyData(pd))
    }
}

/// 3D Delaunay tetrahedralization producing an [`UnstructuredGrid`] of tetrahedra.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delaunay3DFilter {
    /// Optional circumradius cutoff (`0.0` = keep all tetrahedra).
    pub alpha: f32,
}

impl Default for Delaunay3DFilter {
    fn default() -> Self {
        Self { alpha: 0.0 }
    }
}

impl Filter for Delaunay3DFilter {
    fn name(&self) -> &str {
        "Delaunay3DFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let (points, point_data) = extract_points_and_data(input)?;
        if points.len() < 4 {
            return Err(PipelineError::ExecutionFailed(
                "Delaunay3DFilter requires at least 4 points".into(),
            ));
        }
        let tets = bowyer_watson_3d(&points, self.alpha as f64);
        let mut ug = UnstructuredGrid::new();
        ug.points = points;
        ug.point_data = point_data;
        for t in tets {
            ug.push_cell(CellType::Tetra, &t);
        }
        Ok(DataSet::UnstructuredGrid(ug))
    }
}

fn extract_points_and_data(
    ds: &DataSet,
) -> Result<(Vec<[f32; 3]>, paraviz_core::FieldData), PipelineError> {
    match ds {
        DataSet::PolyData(pd) => Ok((pd.points.clone(), pd.point_data.clone())),
        DataSet::UnstructuredGrid(ug) => Ok((ug.points.clone(), ug.point_data.clone())),
        DataSet::StructuredGrid(sg) => Ok((sg.points.clone(), sg.point_data.clone())),
        DataSet::ImageData(img) => Ok((img.all_points(), img.point_data.clone())),
        DataSet::RectilinearGrid(rg) => Ok((rg.all_points(), rg.point_data.clone())),
        DataSet::MultiBlock(_) => Err(PipelineError::ExecutionFailed(
            "Delaunay requires a single dataset (not MultiBlock)".into(),
        )),
    }
}

fn bowyer_watson_2d(points: &[[f32; 3]], alpha: f64) -> Vec<[u32; 3]> {
    let n = points.len();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut pts: Vec<DVec2> = Vec::with_capacity(n + 3);
    for p in points {
        let x = p[0] as f64;
        let y = p[1] as f64;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        pts.push(DVec2::new(x, y));
    }
    let dx = (max_x - min_x).max(1.0);
    let dy = (max_y - min_y).max(1.0);
    let dmax = dx.max(dy);
    let mid_x = 0.5 * (min_x + max_x);
    let mid_y = 0.5 * (min_y + max_y);

    // Super-triangle vertices at indices n, n+1, n+2
    pts.push(DVec2::new(mid_x - 20.0 * dmax, mid_y - dmax));
    pts.push(DVec2::new(mid_x, mid_y + 20.0 * dmax));
    pts.push(DVec2::new(mid_x + 20.0 * dmax, mid_y - dmax));

    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    for i in 0..n {
        let p = pts[i];
        let mut bad = Vec::new();
        for (ti, &[a, b, c]) in triangles.iter().enumerate() {
            if in_circumcircle_2d(p, pts[a], pts[b], pts[c]) {
                bad.push(ti);
            }
        }
        let mut polygon_edges: Vec<(usize, usize)> = Vec::new();
        for &ti in &bad {
            let [a, b, c] = triangles[ti];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                if let Some(pos) = polygon_edges
                    .iter()
                    .position(|&(eu, ev)| (eu == u && ev == v) || (eu == v && ev == u))
                {
                    polygon_edges.remove(pos);
                } else {
                    polygon_edges.push((u, v));
                }
            }
        }
        for &ti in bad.iter().rev() {
            triangles.swap_remove(ti);
        }
        for (u, v) in polygon_edges {
            triangles.push([u, v, i]);
        }
    }

    triangles
        .into_iter()
        .filter(|&[a, b, c]| a < n && b < n && c < n)
        .filter(|&[a, b, c]| {
            if alpha <= 0.0 {
                true
            } else {
                circumradius_2d(pts[a], pts[b], pts[c]) <= alpha
            }
        })
        .map(|[a, b, c]| {
            // Ensure CCW winding
            let cross = (pts[b] - pts[a]).perp_dot(pts[c] - pts[a]);
            if cross >= 0.0 {
                [a as u32, b as u32, c as u32]
            } else {
                [a as u32, c as u32, b as u32]
            }
        })
        .collect()
}

fn in_circumcircle_2d(p: DVec2, a: DVec2, b: DVec2, c: DVec2) -> bool {
    let ax = a.x - p.x;
    let ay = a.y - p.y;
    let bx = b.x - p.x;
    let by = b.y - p.y;
    let cx = c.x - p.x;
    let cy = c.y - p.y;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    let orient = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if orient >= 0.0 {
        det > 1e-12
    } else {
        det < -1e-12
    }
}

fn circumradius_2d(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);
    let area2 = ((b - a).perp_dot(c - a)).abs();
    if area2 < 1e-15 {
        f64::INFINITY
    } else {
        (ab * bc * ca) / (2.0 * area2)
    }
}

fn bowyer_watson_3d(points: &[[f32; 3]], alpha: f64) -> Vec<[u32; 4]> {
    let n = points.len();
    let mut min_p = DVec3::splat(f64::INFINITY);
    let mut max_p = DVec3::splat(f64::NEG_INFINITY);
    let mut pts: Vec<DVec3> = Vec::with_capacity(n + 4);
    for p in points {
        let v = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        min_p = min_p.min(v);
        max_p = max_p.max(v);
        pts.push(v);
    }
    let span = (max_p - min_p).max_element().max(1.0);
    let mid = (min_p + max_p) * 0.5;
    let r = span * 25.0;

    // Super-tetrahedron vertices n..n+4
    pts.push(mid + DVec3::new(-r, -r, -r));
    pts.push(mid + DVec3::new(3.0 * r, -r, -r));
    pts.push(mid + DVec3::new(-r, 3.0 * r, -r));
    pts.push(mid + DVec3::new(-r, -r, 3.0 * r));

    let mut tets: Vec<[usize; 4]> = vec![[n, n + 1, n + 2, n + 3]];

    for i in 0..n {
        let p = pts[i];
        let mut bad = Vec::new();
        for (ti, &[a, b, c, d]) in tets.iter().enumerate() {
            if let Some((center, rad_sq)) = circumsphere_3d(pts[a], pts[b], pts[c], pts[d])
                && p.distance_squared(center) < rad_sq + 1e-10 {
                    bad.push(ti);
                }
        }
        let mut boundary_faces: Vec<[usize; 3]> = Vec::new();
        for &ti in &bad {
            let [a, b, c, d] = tets[ti];
            for f in [[a, b, c], [a, b, d], [a, c, d], [b, c, d]] {
                let mut key = f;
                key.sort_unstable();
                if let Some(pos) = boundary_faces.iter().position(|existing| {
                    let mut ek = *existing;
                    ek.sort_unstable();
                    ek == key
                }) {
                    boundary_faces.remove(pos);
                } else {
                    boundary_faces.push(f);
                }
            }
        }
        for &ti in bad.iter().rev() {
            tets.swap_remove(ti);
        }
        for [a, b, c] in boundary_faces {
            tets.push([a, b, c, i]);
        }
    }

    tets.into_iter()
        .filter(|&[a, b, c, d]| a < n && b < n && c < n && d < n)
        .filter(|&[a, b, c, d]| {
            if alpha <= 0.0 {
                true
            } else {
                circumsphere_3d(pts[a], pts[b], pts[c], pts[d])
                    .is_some_and(|(_, rad_sq)| rad_sq.sqrt() <= alpha)
            }
        })
        .map(|[a, b, c, d]| [a as u32, b as u32, c as u32, d as u32])
        .collect()
}

fn circumsphere_3d(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> Option<(DVec3, f64)> {
    let ba = b - a;
    let ca = c - a;
    let da = d - a;
    let denom = 2.0 * ba.dot(ca.cross(da));
    if denom.abs() < 1e-15 {
        return None;
    }
    let num = ca.cross(da) * ba.length_squared()
        + da.cross(ba) * ca.length_squared()
        + ba.cross(ca) * da.length_squared();
    let offset = num / denom;
    let center = a + offset;
    Some((center, offset.length_squared()))
}
