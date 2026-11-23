use super::*;

/// Extracts indexed surface triangles (position+normal, split scalar stream, and `u32` indices)
/// from any [`DataSet`] for `Representation::Surface` without barycentric overhead.
pub fn prepare_surface_indexed(dataset: &DataSet, color_by: &ColorBy) -> PreparedIndexedSurface {
    let style = ActorStyle {
        color_by: color_by.clone(),
        ..Default::default()
    };
    prepare_surface_indexed_with_style(dataset, &style)
}

/// Style-aware variant of [`prepare_surface_indexed`] honoring MultiBlock `hidden_blocks` and `block_colors`.
pub fn prepare_surface_indexed_with_style(
    dataset: &DataSet,
    style: &ActorStyle,
) -> PreparedIndexedSurface {
    let mut geom_vertices = Vec::new();
    let mut scalars = Vec::new();
    let mut indices = Vec::new();
    let mut range = None;
    let is_multiblock = matches!(dataset, DataSet::MultiBlock(_));
    let has_block_overrides = !style.hidden_blocks.is_empty() || !style.block_colors.is_empty();
    let mut point_map: Option<Vec<u32>> = if !is_multiblock
        && !has_block_overrides
        && !matches!(dataset, DataSet::PolyData(_))
        && !matches!(style.color_by, ColorBy::CellScalar { .. })
    {
        Some(Vec::new())
    } else {
        None
    };

    append_indexed_dataset(
        dataset,
        style,
        "",
        None,
        &mut geom_vertices,
        &mut scalars,
        &mut indices,
        &mut point_map,
        &mut range,
    );

    let scalar_source = if is_multiblock
        || has_block_overrides
        || matches!(style.color_by, ColorBy::CellScalar { .. })
    {
        SurfaceScalarSource::RebuildRequired
    } else if matches!(dataset, DataSet::PolyData(_)) {
        SurfaceScalarSource::PointDirect
    } else {
        match point_map {
            Some(map) => SurfaceScalarSource::PointMapped(map),
            None => SurfaceScalarSource::RebuildRequired,
        }
    };

    PreparedIndexedSurface {
        geom_vertices,
        scalars,
        indices,
        scalar_source,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

/// Re-extracts only the per-vertex scalar float stream for an already-uploaded indexed surface,
/// avoiding position/normal/index buffer rebuilds when only `ColorBy` point scalar or solid changes.
pub fn extract_indexed_surface_scalars(
    dataset: &DataSet,
    color_by: &ColorBy,
    source: &SurfaceScalarSource,
    vertex_count: usize,
) -> Option<(Vec<f32>, (f64, f64))> {
    if matches!(color_by, ColorBy::CellScalar { .. }) {
        return None;
    }
    let pd = dataset.point_data()?;
    match source {
        SurfaceScalarSource::PointDirect => {
            if let ColorBy::PointScalar { array, component } = color_by {
                let mut sc = extract_scalars(pd, array, *component)?;
                let range = compute_finite_range(&sc).unwrap_or((0.0, 1.0));
                sc.resize(vertex_count, 0.0);
                Some((sc, range))
            } else {
                Some((vec![0.0; vertex_count], (0.0, 1.0)))
            }
        }
        SurfaceScalarSource::PointMapped(map) => {
            if let ColorBy::PointScalar { array, component } = color_by {
                let sc = extract_scalars(pd, array, *component)?;
                let range = compute_finite_range(&sc).unwrap_or((0.0, 1.0));
                let mapped: Vec<f32> = if map.len() >= PARALLEL_THRESHOLD {
                    map.par_iter()
                        .map(|&pi| sc.get(pi as usize).copied().unwrap_or(0.0))
                        .collect()
                } else {
                    map.iter()
                        .map(|&pi| sc.get(pi as usize).copied().unwrap_or(0.0))
                        .collect()
                };
                Some((mapped, range))
            } else {
                Some((vec![0.0; vertex_count], (0.0, 1.0)))
            }
        }
        SurfaceScalarSource::RebuildRequired => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_indexed_dataset(
    dataset: &DataSet,
    style: &ActorStyle,
    block_path: &str,
    inherited_override: Option<[f32; 4]>,
    geom: &mut Vec<SurfaceGeomVertex>,
    scalars: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    point_map: &mut Option<Vec<u32>>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            append_indexed_poly(
                poly,
                &style.color_by,
                inherited_override,
                geom,
                scalars,
                indices,
                range,
            );
        }
        DataSet::ImageData(img) => {
            append_indexed_structured(
                img.dimensions,
                &img.point_data,
                &img.cell_data,
                &style.color_by,
                inherited_override,
                |i, j, k| img.point_coords(i, j, k),
                |i, j, k| img.point_index(i, j, k),
                true,
                geom,
                scalars,
                indices,
                point_map,
                range,
            );
        }
        DataSet::RectilinearGrid(rect) => {
            append_indexed_structured(
                rect.dimensions(),
                &rect.point_data,
                &rect.cell_data,
                &style.color_by,
                inherited_override,
                |i, j, k| rect.point_coords(i, j, k),
                |i, j, k| rect.point_index(i, j, k),
                true,
                geom,
                scalars,
                indices,
                point_map,
                range,
            );
        }
        DataSet::StructuredGrid(grid) => {
            if grid.points.len() >= grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2] {
                append_indexed_structured(
                    grid.dimensions,
                    &grid.point_data,
                    &grid.cell_data,
                    &style.color_by,
                    inherited_override,
                    |i, j, k| grid.point_coords(i, j, k),
                    |i, j, k| grid.point_index(i, j, k),
                    false,
                    geom,
                    scalars,
                    indices,
                    point_map,
                    range,
                );
            }
        }
        DataSet::UnstructuredGrid(ug) => {
            append_indexed_unstructured(
                ug,
                &style.color_by,
                inherited_override,
                geom,
                scalars,
                indices,
                point_map,
                range,
            );
        }
        DataSet::MultiBlock(mb) => {
            for (name, block) in &mb.blocks {
                let child_path = join_block_path(block_path, name);
                if !style.is_block_visible(&child_path, name) {
                    continue;
                }
                let child_override = style
                    .block_color_override(&child_path, name)
                    .or(inherited_override);
                append_indexed_dataset(
                    block,
                    style,
                    &child_path,
                    child_override,
                    geom,
                    scalars,
                    indices,
                    point_map,
                    range,
                );
            }
        }
    }
}

fn collect_poly_tris_with_cell(poly: &PolyData) -> Vec<([u32; 3], usize)> {
    let mut tris_with_cell: Vec<([u32; 3], usize)> = Vec::with_capacity(poly.polys.len() * 2);
    let poly_base = poly.verts.len() + poly.lines.len();
    for (pi, p) in poly.polys.iter().enumerate() {
        if p.len() < 3 {
            continue;
        }
        let cell_idx = poly_base + pi;
        let v0 = p[0];
        for k in 1..(p.len() - 1) {
            tris_with_cell.push(([v0, p[k], p[k + 1]], cell_idx));
        }
    }
    let strip_base = poly_base + poly.polys.len();
    for (si, s) in poly.strips.iter().enumerate() {
        if s.len() < 3 {
            continue;
        }
        let cell_idx = strip_base + si;
        for k in 0..(s.len() - 2) {
            let tri = if k % 2 == 0 {
                [s[k], s[k + 1], s[k + 2]]
            } else {
                [s[k + 1], s[k], s[k + 2]]
            };
            tris_with_cell.push((tri, cell_idx));
        }
    }
    tris_with_cell
}

pub(super) fn resolve_poly_normals(poly: &PolyData, tris_with_cell: &[([u32; 3], usize)]) -> Vec<[f32; 3]> {
    if let Some(arr) = poly
        .point_data
        .active_normals_array()
        .filter(|arr| arr.num_tuples() == poly.points.len())
    {
        let n = arr.num_tuples();
        if n >= PARALLEL_THRESHOLD {
            (0..n).into_par_iter().map(|i| arr.get_vec3(i)).collect()
        } else {
            (0..n).map(|i| arr.get_vec3(i)).collect()
        }
    } else {
        compute_smooth_normals(&poly.points, tris_with_cell)
    }
}

fn append_indexed_poly(
    poly: &PolyData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    geom: &mut Vec<SurfaceGeomVertex>,
    scalars_out: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    range: &mut Option<(f64, f64)>,
) {
    let tris_with_cell = collect_poly_tris_with_cell(poly);
    if tris_with_cell.is_empty() {
        return;
    }

    let normals = resolve_poly_normals(poly, &tris_with_cell);
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(&poly.point_data, &poly.cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    let n_pts = poly.points.len();
    if !is_cell {
        let base_v = geom.len() as u32;
        if n_pts >= PARALLEL_THRESHOLD {
            let new_geom: Vec<SurfaceGeomVertex> = poly
                .points
                .par_iter()
                .zip(normals.par_iter())
                .map(|(&position, &normal)| SurfaceGeomVertex {
                    position,
                    normal,
                    color_override,
                })
                .collect();
            geom.extend(new_geom);
        } else {
            geom.reserve(n_pts);
            for (&position, &normal) in poly.points.iter().zip(normals.iter()) {
                geom.push(SurfaceGeomVertex {
                    position,
                    normal,
                    color_override,
                });
            }
        }
        if let Some(sc) = &scalars {
            scalars_out.reserve(n_pts);
            let copy_len = sc.len().min(n_pts);
            scalars_out.extend_from_slice(&sc[..copy_len]);
            scalars_out.resize(scalars_out.len() + (n_pts - copy_len), 0.0);
        } else {
            scalars_out.resize(scalars_out.len() + n_pts, 0.0);
        }
        indices.reserve(tris_with_cell.len() * 3);
        for (tri, _) in tris_with_cell {
            let [a, b, c] = tri;
            if (a as usize) < n_pts && (b as usize) < n_pts && (c as usize) < n_pts {
                indices.push(base_v + a);
                indices.push(base_v + b);
                indices.push(base_v + c);
            }
        }
        return;
    }

    // CellScalar: share vertices within each polygon cell while splitting across cell boundaries.
    let poly_base = poly.verts.len() + poly.lines.len();
    for (pi, p) in poly.polys.iter().enumerate() {
        if p.len() < 3 {
            continue;
        }
        let cell_idx = poly_base + pi;
        let cell_s = scalars.as_ref().and_then(|sc| sc.get(cell_idx).copied()).unwrap_or(0.0);
        if p.iter().any(|&idx| (idx as usize) >= n_pts) {
            continue;
        }
        let base_v = geom.len() as u32;
        for &idx in p {
            let i = idx as usize;
            geom.push(SurfaceGeomVertex {
                position: poly.points[i],
                normal: normals[i],
                color_override,
            });
            scalars_out.push(cell_s);
        }
        for k in 1..(p.len() - 1) {
            indices.push(base_v);
            indices.push(base_v + k as u32);
            indices.push(base_v + (k + 1) as u32);
        }
    }
    let strip_base = poly_base + poly.polys.len();
    for (si, s) in poly.strips.iter().enumerate() {
        if s.len() < 3 {
            continue;
        }
        let cell_idx = strip_base + si;
        let cell_s = scalars.as_ref().and_then(|sc| sc.get(cell_idx).copied()).unwrap_or(0.0);
        if s.iter().any(|&idx| (idx as usize) >= n_pts) {
            continue;
        }
        let base_v = geom.len() as u32;
        for &idx in s {
            let i = idx as usize;
            geom.push(SurfaceGeomVertex {
                position: poly.points[i],
                normal: normals[i],
                color_override,
            });
            scalars_out.push(cell_s);
        }
        for k in 0..(s.len() - 2) {
            if k % 2 == 0 {
                indices.push(base_v + k as u32);
                indices.push(base_v + (k + 1) as u32);
                indices.push(base_v + (k + 2) as u32);
            } else {
                indices.push(base_v + (k + 1) as u32);
                indices.push(base_v + k as u32);
                indices.push(base_v + (k + 2) as u32);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_indexed_structured<FCoord, FPointIdx>(
    dims: [usize; 3],
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    coord: FCoord,
    point_idx: FPointIdx,
    planar_faces: bool,
    geom: &mut Vec<SurfaceGeomVertex>,
    scalars_out: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    point_map: &mut Option<Vec<u32>>,
    range: &mut Option<(f64, f64)>,
) where
    FCoord: Fn(usize, usize, usize) -> [f32; 3],
    FPointIdx: Fn(usize, usize, usize) -> usize,
{
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(point_data, cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    let cx = nx.saturating_sub(1).max(1);
    let cy = ny.saturating_sub(1).max(1);

    // For flat axis-aligned grids (ImageData / RectilinearGrid) with point/solid coloring,
    // share vertices across each of the 6 flat boundary planes (~4x fewer vertices).
    if planar_faces && !is_cell {
        let emit_plane = |nu: usize,
                          nv: usize,
                          flip_winding: bool,
                          pt_at: &dyn Fn(usize, usize) -> (usize, usize, usize),
                          geom: &mut Vec<SurfaceGeomVertex>,
                          scalars_out: &mut Vec<f32>,
                          indices: &mut Vec<u32>,
                          point_map: &mut Option<Vec<u32>>| {
            if nu < 2 || nv < 2 {
                return;
            }
            let (i0, j0, k0) = pt_at(0, 0);
            let (i1, j1, k1) = if flip_winding { pt_at(0, 1) } else { pt_at(1, 0) };
            let (i2, j2, k2) = pt_at(1, 1);
            let p0 = Vec3::from_array(coord(i0, j0, k0));
            let p1 = Vec3::from_array(coord(i1, j1, k1));
            let p2 = Vec3::from_array(coord(i2, j2, k2));
            let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();

            let base_v = geom.len() as u32;
            geom.reserve(nu * nv);
            scalars_out.reserve(nu * nv);
            if let Some(map) = point_map.as_mut() {
                map.reserve(nu * nv);
            }
            for v in 0..nv {
                for u in 0..nu {
                    let (i, j, k) = pt_at(u, v);
                    let pidx = point_idx(i, j, k);
                    geom.push(SurfaceGeomVertex {
                        position: coord(i, j, k),
                        normal: n,
                        color_override,
                    });
                    let s = scalars
                        .as_ref()
                        .and_then(|sc| sc.get(pidx).copied())
                        .unwrap_or(0.0);
                    scalars_out.push(s);
                    if let Some(map) = point_map.as_mut() {
                        map.push(pidx as u32);
                    }
                }
            }
            indices.reserve((nu - 1) * (nv - 1) * 6);
            for v in 0..(nv - 1) {
                for u in 0..(nu - 1) {
                    let a = base_v + (v * nu + u) as u32;
                    let b = base_v + (v * nu + u + 1) as u32;
                    let c = base_v + ((v + 1) * nu + u + 1) as u32;
                    let d = base_v + ((v + 1) * nu + u) as u32;
                    if flip_winding {
                        indices.extend_from_slice(&[a, d, c, a, c, b]);
                    } else {
                        indices.extend_from_slice(&[a, b, c, a, c, d]);
                    }
                }
            }
        };

        if nx > 1 && ny > 1 {
            if nz > 1 {
                emit_plane(nx, ny, true, &|u, v| (u, v, 0), geom, scalars_out, indices, point_map);
                emit_plane(nx, ny, false, &|u, v| (u, v, nz - 1), geom, scalars_out, indices, point_map);
            } else {
                emit_plane(nx, ny, false, &|u, v| (u, v, 0), geom, scalars_out, indices, point_map);
            }
        }
        if nx > 1 && nz > 1 {
            if ny > 1 {
                emit_plane(nx, nz, true, &|u, v| (u, 0, v), geom, scalars_out, indices, point_map);
                emit_plane(nx, nz, false, &|u, v| (u, ny - 1, v), geom, scalars_out, indices, point_map);
            } else {
                emit_plane(nx, nz, true, &|u, v| (u, 0, v), geom, scalars_out, indices, point_map);
            }
        }
        if ny > 1 && nz > 1 {
            if nx > 1 {
                emit_plane(ny, nz, false, &|u, v| (0, u, v), geom, scalars_out, indices, point_map);
                emit_plane(ny, nz, true, &|u, v| (nx - 1, u, v), geom, scalars_out, indices, point_map);
            } else {
                emit_plane(ny, nz, true, &|u, v| (0, u, v), geom, scalars_out, indices, point_map);
            }
        }
        return;
    }

    // Per-quad indexed emission (4 vertices + 6 indices per boundary quad) for curvilinear grids or CellScalar.
    let mut emit_quad_indexed = |q: [(usize, usize, usize); 4], ci: usize, cj: usize, ck: usize| {
        let pts = [
            coord(q[0].0, q[0].1, q[0].2),
            coord(q[1].0, q[1].1, q[1].2),
            coord(q[2].0, q[2].1, q[2].2),
            coord(q[3].0, q[3].1, q[3].2),
        ];
        let p0 = Vec3::from_array(pts[0]);
        let p1 = Vec3::from_array(pts[1]);
        let p2 = Vec3::from_array(pts[2]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();
        let base_v = geom.len() as u32;
        let cidx = ci.min(cx - 1) + cx * (cj.min(cy - 1) + cy * ck);
        for &(i, j, k) in &q {
            let pidx = point_idx(i, j, k);
            geom.push(SurfaceGeomVertex {
                position: coord(i, j, k),
                normal: n,
                color_override,
            });
            let s = match &scalars {
                Some(sc) if is_cell => sc.get(cidx).copied().unwrap_or(0.0),
                Some(sc) => sc.get(pidx).copied().unwrap_or(0.0),
                None => 0.0,
            };
            scalars_out.push(s);
            if let Some(map) = point_map.as_mut() {
                map.push(pidx as u32);
            }
        }
        indices.extend_from_slice(&[base_v, base_v + 1, base_v + 2, base_v, base_v + 2, base_v + 3]);
    };

    for_each_structured_boundary_quad(dims, |q, ci, cj, ck| {
        emit_quad_indexed(q, ci, cj, ck);
    });
}

#[allow(clippy::too_many_arguments)]
fn append_indexed_unstructured(
    ug: &UnstructuredGrid,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    geom: &mut Vec<SurfaceGeomVertex>,
    scalars_out: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    point_map: &mut Option<Vec<u32>>,
    range: &mut Option<(f64, f64)>,
) {
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(&ug.point_data, &ug.cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    let boundary_faces = extract_unstructured_boundary_faces(ug);
    let n_pts = ug.points.len();
    for (face, cell_idx) in boundary_faces {
        let fverts = face.as_slice();
        if fverts.len() < 3 || fverts.iter().any(|&v| (v as usize) >= n_pts) {
            continue;
        }
        let p0 = Vec3::from_array(ug.points[fverts[0] as usize]);
        let p1 = Vec3::from_array(ug.points[fverts[1] as usize]);
        let p2 = Vec3::from_array(ug.points[fverts[2] as usize]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();
        let base_v = geom.len() as u32;
        for &vi in fverts {
            let idx = vi as usize;
            geom.push(SurfaceGeomVertex {
                position: ug.points[idx],
                normal: n,
                color_override,
            });
            let s = match &scalars {
                Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                Some(sc) => sc.get(idx).copied().unwrap_or(0.0),
                None => 0.0,
            };
            scalars_out.push(s);
            if let Some(map) = point_map.as_mut() {
                map.push(vi);
            }
        }
        for k in 1..(fverts.len() - 1) {
            indices.push(base_v);
            indices.push(base_v + k as u32);
            indices.push(base_v + (k + 1) as u32);
        }
    }
}

/// Extracts non-indexed surface triangles (with normals and barycentric coordinates) from any [`DataSet`].
pub fn prepare_surface(dataset: &DataSet, color_by: &ColorBy) -> PreparedSurface {
    let style = ActorStyle {
        color_by: color_by.clone(),
        ..Default::default()
    };
    prepare_surface_with_style(dataset, &style)
}

/// Style-aware variant of [`prepare_surface`] honoring MultiBlock `hidden_blocks` and `block_colors`.
pub fn prepare_surface_with_style(dataset: &DataSet, style: &ActorStyle) -> PreparedSurface {
    let mut vertices = Vec::new();
    let mut range = None;
    append_surface_dataset(dataset, style, "", None, &mut vertices, &mut range);
    PreparedSurface {
        vertices,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

fn append_surface_dataset(
    dataset: &DataSet,
    style: &ActorStyle,
    block_path: &str,
    inherited_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            append_surface_poly(poly, &style.color_by, inherited_override, out, range);
        }
        DataSet::ImageData(img) => {
            append_surface_image(img, &style.color_by, inherited_override, out, range);
        }
        DataSet::RectilinearGrid(rect) => {
            append_surface_rectilinear(rect, &style.color_by, inherited_override, out, range);
        }
        DataSet::StructuredGrid(grid) => {
            append_surface_structured(grid, &style.color_by, inherited_override, out, range);
        }
        DataSet::UnstructuredGrid(ug) => {
            append_surface_unstructured(ug, &style.color_by, inherited_override, out, range);
        }
        DataSet::MultiBlock(mb) => {
            for (name, block) in &mb.blocks {
                let child_path = join_block_path(block_path, name);
                if !style.is_block_visible(&child_path, name) {
                    continue;
                }
                let child_override = style
                    .block_color_override(&child_path, name)
                    .or(inherited_override);
                append_surface_dataset(block, style, &child_path, child_override, out, range);
            }
        }
    }
}

pub(super) fn scalar_lookup(
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
) -> (Option<Vec<f32>>, bool) {
    match color_by {
        ColorBy::Solid(_) => (None, false),
        ColorBy::PointScalar { array, component } => {
            (extract_scalars(point_data, array, *component), false)
        }
        ColorBy::CellScalar { array, component } => {
            (extract_scalars(cell_data, array, *component), true)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_triangle(
    points: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    scalars: Option<&[f32]>,
    is_cell_scalar: bool,
    cell_idx: usize,
    tri: [u32; 3],
    color_override: [f32; 4],
    out: &mut Vec<SurfaceVertex>,
) {
    let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
    if i0 >= points.len() || i1 >= points.len() || i2 >= points.len() {
        return;
    }
    let p0 = Vec3::from_array(points[i0]);
    let p1 = Vec3::from_array(points[i1]);
    let p2 = Vec3::from_array(points[i2]);
    let face_n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();

    let bary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let indices = [i0, i1, i2];
    for (k, &idx) in indices.iter().enumerate() {
        let normal = normals.and_then(|ns| ns.get(idx).copied()).unwrap_or(face_n);
        let scalar = match scalars {
            Some(sc) if is_cell_scalar => sc.get(cell_idx).copied().unwrap_or(0.0),
            Some(sc) => sc.get(idx).copied().unwrap_or(0.0),
            None => 0.0,
        };
        out.push(SurfaceVertex {
            position: points[idx],
            normal,
            scalar,
            barycentric: bary[k],
            color_override,
        });
    }
}

fn compute_smooth_normals(points: &[[f32; 3]], tris: &[([u32; 3], usize)]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; points.len()];
    if tris.len() >= PARALLEL_THRESHOLD {
        let face_normals: Vec<(usize, usize, usize, Vec3)> = tris
            .par_iter()
            .filter_map(|&(tri, _)| {
                let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                if a >= points.len() || b >= points.len() || c >= points.len() {
                    return None;
                }
                let p0 = Vec3::from_array(points[a]);
                let p1 = Vec3::from_array(points[b]);
                let p2 = Vec3::from_array(points[c]);
                Some((a, b, c, (p1 - p0).cross(p2 - p0)))
            })
            .collect();
        for (a, b, c, n) in face_normals {
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
    } else {
        for &(tri, _) in tris {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            if a >= points.len() || b >= points.len() || c >= points.len() {
                continue;
            }
            let p0 = Vec3::from_array(points[a]);
            let p1 = Vec3::from_array(points[b]);
            let p2 = Vec3::from_array(points[c]);
            let n = (p1 - p0).cross(p2 - p0);
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
    }
    if acc.len() >= PARALLEL_THRESHOLD {
        acc.into_par_iter()
            .map(|n| n.normalize_or(Vec3::Z).to_array())
            .collect()
    } else {
        acc.into_iter()
            .map(|n| n.normalize_or(Vec3::Z).to_array())
            .collect()
    }
}

fn append_surface_poly(
    poly: &PolyData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    let tris_with_cell = collect_poly_tris_with_cell(poly);
    if tris_with_cell.is_empty() {
        return;
    }

    let normals = resolve_poly_normals(poly, &tris_with_cell);
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(&poly.point_data, &poly.cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    if tris_with_cell.len() >= PARALLEL_THRESHOLD {
        let bary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let points = &poly.points;
        let normals_slice = normals.as_slice();
        let scalars_slice = scalars.as_deref();
        let built: Vec<SurfaceVertex> = tris_with_cell
            .par_iter()
            .flat_map_iter(|&(tri, cell_idx)| {
                let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                let valid = i0 < points.len() && i1 < points.len() && i2 < points.len();
                let mut verts = [SurfaceVertex::zeroed(); 3];
                let count = if valid {
                    let p0 = Vec3::from_array(points[i0]);
                    let p1 = Vec3::from_array(points[i1]);
                    let p2 = Vec3::from_array(points[i2]);
                    let face_n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();
                    let indices = [i0, i1, i2];
                    for k in 0..3 {
                        let idx = indices[k];
                        let normal = normals_slice.get(idx).copied().unwrap_or(face_n);
                        let scalar = match scalars_slice {
                            Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                            Some(sc) => sc.get(idx).copied().unwrap_or(0.0),
                            None => 0.0,
                        };
                        verts[k] = SurfaceVertex {
                            position: points[idx],
                            normal,
                            scalar,
                            barycentric: bary[k],
                            color_override,
                        };
                    }
                    3
                } else {
                    0
                };
                verts.into_iter().take(count)
            })
            .collect();
        out.extend(built);
    } else {
        out.reserve(tris_with_cell.len() * 3);
        for (tri, cell_idx) in tris_with_cell {
            emit_triangle(
                &poly.points,
                Some(&normals),
                scalars.as_deref(),
                is_cell,
                cell_idx,
                tri,
                color_override,
                out,
            );
        }
    }
}

fn emit_quad(
    p: [[f32; 3]; 4],
    s: [f32; 4],
    normal: [f32; 3],
    color_override: [f32; 4],
    out: &mut Vec<SurfaceVertex>,
) {
    let tris = [(0, 1, 2), (0, 2, 3)];
    let bary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (a, b, c) in tris {
        out.push(SurfaceVertex {
            position: p[a],
            normal,
            scalar: s[a],
            barycentric: bary[0],
            color_override,
        });
        out.push(SurfaceVertex {
            position: p[b],
            normal,
            scalar: s[b],
            barycentric: bary[1],
            color_override,
        });
        out.push(SurfaceVertex {
            position: p[c],
            normal,
            scalar: s[c],
            barycentric: bary[2],
            color_override,
        });
    }
}

pub(super) fn for_each_structured_boundary_quad<F>(dims: [usize; 3], mut emit: F)
where
    F: FnMut([(usize, usize, usize); 4], usize, usize, usize),
{
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    // XY faces (k = 0 and k = nz - 1)
    if nx > 1 && ny > 1 {
        let k_planes: &[(usize, usize, bool)] = if nz > 1 {
            &[(0, 0, true), (nz - 1, nz - 2, false)]
        } else {
            &[(0, 0, false)]
        };
        for &(k, ck, flip) in k_planes {
            for j in 0..(ny - 1) {
                for i in 0..(nx - 1) {
                    let q = if flip {
                        [(i, j, k), (i, j + 1, k), (i + 1, j + 1, k), (i + 1, j, k)]
                    } else {
                        [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k)]
                    };
                    emit(q, i, j, ck);
                }
            }
        }
    }
    // XZ faces (j = 0 and j = ny - 1)
    if nx > 1 && nz > 1 {
        let j_planes: &[(usize, usize, bool)] = if ny > 1 {
            &[(0, 0, false), (ny - 1, ny - 2, true)]
        } else {
            &[(0, 0, false)]
        };
        for &(j, cj, flip) in j_planes {
            for k in 0..(nz - 1) {
                for i in 0..(nx - 1) {
                    let q = if flip {
                        [(i, j, k), (i + 1, j, k), (i + 1, j, k + 1), (i, j, k + 1)]
                    } else {
                        [(i, j, k), (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j, k)]
                    };
                    emit(q, i, cj, k);
                }
            }
        }
    }
    // YZ faces (i = 0 and i = nx - 1)
    if ny > 1 && nz > 1 {
        let i_planes: &[(usize, usize, bool)] = if nx > 1 {
            &[(0, 0, true), (nx - 1, nx - 2, false)]
        } else {
            &[(0, 0, false)]
        };
        for &(i, ci, flip) in i_planes {
            for k in 0..(nz - 1) {
                for j in 0..(ny - 1) {
                    let q = if flip {
                        [(i, j, k), (i, j, k + 1), (i, j + 1, k + 1), (i, j + 1, k)]
                    } else {
                        [(i, j, k), (i, j + 1, k), (i, j + 1, k + 1), (i, j, k + 1)]
                    };
                    emit(q, ci, j, k);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_structured_boundary<FCoord, FPointIdx>(
    dims: [usize; 3],
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    coord: FCoord,
    point_idx: FPointIdx,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) where
    FCoord: Fn(usize, usize, usize) -> [f32; 3],
    FPointIdx: Fn(usize, usize, usize) -> usize,
{
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(point_data, cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    let cx = nx.saturating_sub(1).max(1);
    let cy = ny.saturating_sub(1).max(1);

    let pt_s = |i: usize, j: usize, k: usize, ci: usize, cj: usize, ck: usize| -> f32 {
        match &scalars {
            Some(sc) if is_cell => {
                let cidx = ci.min(cx - 1) + cx * (cj.min(cy - 1) + cy * ck);
                sc.get(cidx).copied().unwrap_or(0.0)
            }
            Some(sc) => sc.get(point_idx(i, j, k)).copied().unwrap_or(0.0),
            None => 0.0,
        }
    };

    for_each_structured_boundary_quad(dims, |q, ci, cj, ck| {
        let pts = [
            coord(q[0].0, q[0].1, q[0].2),
            coord(q[1].0, q[1].1, q[1].2),
            coord(q[2].0, q[2].1, q[2].2),
            coord(q[3].0, q[3].1, q[3].2),
        ];
        let p0 = Vec3::from_array(pts[0]);
        let p1 = Vec3::from_array(pts[1]);
        let p2 = Vec3::from_array(pts[2]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z).to_array();
        let sv = [
            pt_s(q[0].0, q[0].1, q[0].2, ci, cj, ck),
            pt_s(q[1].0, q[1].1, q[1].2, ci, cj, ck),
            pt_s(q[2].0, q[2].1, q[2].2, ci, cj, ck),
            pt_s(q[3].0, q[3].1, q[3].2, ci, cj, ck),
        ];
        emit_quad(pts, sv, n, color_override, out);
    });
}

fn append_surface_image(
    img: &ImageData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    append_structured_boundary(
        img.dimensions,
        &img.point_data,
        &img.cell_data,
        color_by,
        block_override,
        |i, j, k| img.point_coords(i, j, k),
        |i, j, k| img.point_index(i, j, k),
        out,
        range,
    );
}

fn append_surface_rectilinear(
    rect: &RectilinearGrid,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    append_structured_boundary(
        rect.dimensions(),
        &rect.point_data,
        &rect.cell_data,
        color_by,
        block_override,
        |i, j, k| rect.point_coords(i, j, k),
        |i, j, k| rect.point_index(i, j, k),
        out,
        range,
    );
}

fn append_surface_structured(
    grid: &StructuredGrid,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    if grid.points.len() < grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2] {
        return;
    }
    append_structured_boundary(
        grid.dimensions,
        &grid.point_data,
        &grid.cell_data,
        color_by,
        block_override,
        |i, j, k| grid.point_coords(i, j, k),
        |i, j, k| grid.point_index(i, j, k),
        out,
        range,
    );
}

/// Canonical sorted vertex key for boundary face deduplication (zero-alloc for <=4 verts).
#[derive(Clone, PartialEq, Eq, Hash)]
enum FaceKey {
    Small([u32; 4], u8),
    Large(Vec<u32>),
}

impl FaceKey {
    #[inline]
    fn from_small(face: &SmallFace) -> Self {
        match face {
            SmallFace::Small(arr, len) => {
                let mut sorted = *arr;
                sorted[..(*len as usize)].sort_unstable();
                Self::Small(sorted, *len)
            }
            SmallFace::Large(v) => {
                let mut sorted = v.clone();
                sorted.sort_unstable();
                Self::Large(sorted)
            }
        }
    }
}

/// Ordered boundary face vertex indices (stack-allocated for triangles/quads).
enum SmallFace {
    Small([u32; 4], u8),
    Large(Vec<u32>),
}

impl SmallFace {
    #[inline]
    fn from_local(pts: &[u32], local: &[usize]) -> Self {
        if local.len() <= 4 {
            let mut arr = [0u32; 4];
            for (k, &li) in local.iter().enumerate() {
                arr[k] = pts[li];
            }
            Self::Small(arr, local.len() as u8)
        } else {
            Self::Large(local.iter().map(|&li| pts[li]).collect())
        }
    }

    #[inline]
    fn from_tri(tri: [u32; 3]) -> Self {
        Self::Small([tri[0], tri[1], tri[2], 0], 3)
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Small(arr, len) => &arr[..(*len as usize)],
            Self::Large(v) => v.as_slice(),
        }
    }
}

enum FaceEntry {
    Single { face: SmallFace, cell_idx: usize },
    Shared,
}

fn extract_unstructured_boundary_faces(ug: &UnstructuredGrid) -> Vec<(SmallFace, usize)> {
    let n_cells = ug.num_cells();
    if n_cells == 0 {
        return Vec::new();
    }

    let mut out_faces: Vec<(SmallFace, usize)> = Vec::new();

    // Extract 3D cell faces (parallel for non-trivial grids) and deduplicate via single FxHashMap.
    if n_cells >= 256 {
        let candidates: Vec<(FaceKey, SmallFace, usize)> = (0..n_cells)
            .into_par_iter()
            .flat_map_iter(|cell_idx| {
                let ct = ug.cell_types[cell_idx];
                let pts = ug.cell_points(cell_idx);
                let mut local = Vec::new();
                if ct.dimension() == 3 {
                    for f in ct.faces(pts.len()) {
                        if f.len() >= 3 {
                            let face = SmallFace::from_local(pts, &f);
                            let key = FaceKey::from_small(&face);
                            local.push((key, face, cell_idx));
                        }
                    }
                }
                local
            })
            .collect();

        let mut face_map: FxHashMap<FaceKey, FaceEntry> =
            FxHashMap::with_capacity_and_hasher(candidates.len(), Default::default());
        for (key, face, cell_idx) in candidates {
            match face_map.get_mut(&key) {
                Some(entry) => *entry = FaceEntry::Shared,
                None => {
                    face_map.insert(key, FaceEntry::Single { face, cell_idx });
                }
            }
        }
        for cell_idx in 0..n_cells {
            let ct = ug.cell_types[cell_idx];
            if ct.dimension() == 2 {
                let pts = ug.cell_points(cell_idx);
                if ct == CellType::TriangleStrip {
                    for k in 0..pts.len().saturating_sub(2) {
                        let tri = if k % 2 == 0 {
                            [pts[k], pts[k + 1], pts[k + 2]]
                        } else {
                            [pts[k + 1], pts[k], pts[k + 2]]
                        };
                        out_faces.push((SmallFace::from_tri(tri), cell_idx));
                    }
                } else {
                    for f in ct.faces(pts.len()) {
                        if f.len() >= 3 {
                            out_faces.push((SmallFace::from_local(pts, &f), cell_idx));
                        }
                    }
                }
            }
        }
        let mut boundary_3d: Vec<(SmallFace, usize)> = face_map
            .into_values()
            .filter_map(|entry| match entry {
                FaceEntry::Single { face, cell_idx } => Some((face, cell_idx)),
                FaceEntry::Shared => None,
            })
            .collect();
        boundary_3d.sort_unstable_by_key(|&(_, cell_idx)| cell_idx);
        out_faces.extend(boundary_3d);
    } else {
        let mut face_map: FxHashMap<FaceKey, FaceEntry> =
            FxHashMap::with_capacity_and_hasher(n_cells * 4, Default::default());
        for cell_idx in 0..n_cells {
            let ct = ug.cell_types[cell_idx];
            let pts = ug.cell_points(cell_idx);
            match ct.dimension() {
                2 => {
                    if ct == CellType::TriangleStrip {
                        for k in 0..pts.len().saturating_sub(2) {
                            let tri = if k % 2 == 0 {
                                [pts[k], pts[k + 1], pts[k + 2]]
                            } else {
                                [pts[k + 1], pts[k], pts[k + 2]]
                            };
                            out_faces.push((SmallFace::from_tri(tri), cell_idx));
                        }
                    } else {
                        for f in ct.faces(pts.len()) {
                            if f.len() >= 3 {
                                out_faces.push((SmallFace::from_local(pts, &f), cell_idx));
                            }
                        }
                    }
                }
                3 => {
                    for f in ct.faces(pts.len()) {
                        if f.len() >= 3 {
                            let face = SmallFace::from_local(pts, &f);
                            let key = FaceKey::from_small(&face);
                            match face_map.get_mut(&key) {
                                Some(entry) => *entry = FaceEntry::Shared,
                                None => {
                                    face_map.insert(key, FaceEntry::Single { face, cell_idx });
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let mut boundary_3d: Vec<(SmallFace, usize)> = face_map
            .into_values()
            .filter_map(|entry| match entry {
                FaceEntry::Single { face, cell_idx } => Some((face, cell_idx)),
                FaceEntry::Shared => None,
            })
            .collect();
        boundary_3d.sort_unstable_by_key(|&(_, cell_idx)| cell_idx);
        out_faces.extend(boundary_3d);
    }
    out_faces
}

fn append_surface_unstructured(
    ug: &UnstructuredGrid,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<SurfaceVertex>,
    range: &mut Option<(f64, f64)>,
) {
    let (scalars, is_cell, color_override) =
        resolve_block_coloring(&ug.point_data, &ug.cell_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }

    let boundary_faces = extract_unstructured_boundary_faces(ug);
    let mut surface_tris: Vec<([u32; 3], usize)> = Vec::with_capacity(boundary_faces.len() * 2);
    for (face, cell_idx) in boundary_faces {
        let f = face.as_slice();
        if f.len() < 3 {
            continue;
        }
        let v0 = f[0];
        for k in 1..(f.len() - 1) {
            surface_tris.push(([v0, f[k], f[k + 1]], cell_idx));
        }
    }

    out.reserve(surface_tris.len() * 3);
    for (tri, cell_idx) in surface_tris {
        emit_triangle(
            &ug.points,
            None,
            scalars.as_deref(),
            is_cell,
            cell_idx,
            tri,
            color_override,
            out,
        );
    }
}

