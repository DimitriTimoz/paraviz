use super::*;
use surface::for_each_structured_boundary_quad;

/// Extracts wireframe line segments from any [`DataSet`].
pub fn prepare_wireframe(dataset: &DataSet, color_by: &ColorBy) -> PreparedLines {
    let style = ActorStyle {
        color_by: color_by.clone(),
        ..Default::default()
    };
    prepare_wireframe_with_style(dataset, &style)
}

/// Style-aware variant of [`prepare_wireframe`] honoring MultiBlock `hidden_blocks` and `block_colors`.
pub fn prepare_wireframe_with_style(dataset: &DataSet, style: &ActorStyle) -> PreparedLines {
    let mut vertices = Vec::new();
    let mut range = None;
    append_wireframe_dataset(dataset, style, "", None, &mut vertices, &mut range);
    PreparedLines {
        vertices,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

#[inline]
fn pack_edge_key(a: u32, b: u32) -> u64 {
    if a < b {
        ((a as u64) << 32) | (b as u64)
    } else {
        ((b as u64) << 32) | (a as u64)
    }
}

#[allow(clippy::too_many_arguments)]
fn append_structured_wireframe<FCoord, FPointIdx>(
    dims: [usize; 3],
    point_data: &FieldData,
    cell_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    coord: FCoord,
    point_idx: FPointIdx,
    out: &mut Vec<LineVertex>,
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
    let mut seen: FxHashSet<u64> = FxHashSet::default();

    for_each_structured_boundary_quad(dims, |q, ci, cj, ck| {
        let cidx = ci.min(cx - 1) + cx * (cj.min(cy - 1) + cy * ck);
        for k in 0..4 {
            let va = q[k];
            let vb = q[(k + 1) % 4];
            let ia = point_idx(va.0, va.1, va.2);
            let ib = point_idx(vb.0, vb.1, vb.2);
            if seen.insert(pack_edge_key(ia as u32, ib as u32)) {
                let sa = match &scalars {
                    Some(sc) if is_cell => sc.get(cidx).copied().unwrap_or(0.0),
                    Some(sc) => sc.get(ia).copied().unwrap_or(0.0),
                    None => 0.0,
                };
                let sb = match &scalars {
                    Some(sc) if is_cell => sc.get(cidx).copied().unwrap_or(0.0),
                    Some(sc) => sc.get(ib).copied().unwrap_or(0.0),
                    None => 0.0,
                };
                out.push(LineVertex {
                    position: coord(va.0, va.1, va.2),
                    scalar: sa,
                    color_override,
                });
                out.push(LineVertex {
                    position: coord(vb.0, vb.1, vb.2),
                    scalar: sb,
                    color_override,
                });
            }
        }
    });
}

fn append_wireframe_dataset(
    dataset: &DataSet,
    style: &ActorStyle,
    block_path: &str,
    inherited_override: Option<[f32; 4]>,
    out: &mut Vec<LineVertex>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            let (scalars, is_cell, color_override) = resolve_block_coloring(
                &poly.point_data,
                &poly.cell_data,
                &style.color_by,
                inherited_override,
            );
            if let Some(sc) = &scalars {
                merge_range(range, compute_finite_range(sc));
            }
            let mut seen: FxHashSet<u64> =
                FxHashSet::with_capacity_and_hasher(poly.polys.len() * 3, Default::default());
            let mut push_edge = |a: u32, b: u32, cell_idx: usize| {
                let (ia, ib) = (a as usize, b as usize);
                if ia >= poly.points.len() || ib >= poly.points.len() {
                    return;
                }
                if seen.insert(pack_edge_key(a, b)) {
                    let sa = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ia).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let sb = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ib).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    out.push(LineVertex {
                        position: poly.points[ia],
                        scalar: sa,
                        color_override,
                    });
                    out.push(LineVertex {
                        position: poly.points[ib],
                        scalar: sb,
                        color_override,
                    });
                }
            };

            let line_base = poly.verts.len();
            for (li, line) in poly.lines.iter().enumerate() {
                for w in line.windows(2) {
                    push_edge(w[0], w[1], line_base + li);
                }
            }
            let poly_base = line_base + poly.lines.len();
            for (pi, p) in poly.polys.iter().enumerate() {
                if p.len() >= 2 {
                    for k in 0..p.len() {
                        push_edge(p[k], p[(k + 1) % p.len()], poly_base + pi);
                    }
                }
            }
            let strip_base = poly_base + poly.polys.len();
            for (si, s) in poly.strips.iter().enumerate() {
                if s.len() >= 3 {
                    for k in 0..(s.len() - 2) {
                        push_edge(s[k], s[k + 1], strip_base + si);
                        push_edge(s[k + 1], s[k + 2], strip_base + si);
                        push_edge(s[k], s[k + 2], strip_base + si);
                    }
                }
            }
        }
        DataSet::UnstructuredGrid(ug) => {
            let (scalars, is_cell, color_override) = resolve_block_coloring(
                &ug.point_data,
                &ug.cell_data,
                &style.color_by,
                inherited_override,
            );
            if let Some(sc) = &scalars {
                merge_range(range, compute_finite_range(sc));
            }
            let mut seen: FxHashSet<u64> =
                FxHashSet::with_capacity_and_hasher(ug.num_cells() * 4, Default::default());
            let mut push_edge = |a: u32, b: u32, cell_idx: usize| {
                let (ia, ib) = (a as usize, b as usize);
                if ia >= ug.points.len() || ib >= ug.points.len() {
                    return;
                }
                if seen.insert(pack_edge_key(a, b)) {
                    let sa = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ia).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let sb = match &scalars {
                        Some(sc) if is_cell => sc.get(cell_idx).copied().unwrap_or(0.0),
                        Some(sc) => sc.get(ib).copied().unwrap_or(0.0),
                        None => 0.0,
                    };
                    out.push(LineVertex {
                        position: ug.points[ia],
                        scalar: sa,
                        color_override,
                    });
                    out.push(LineVertex {
                        position: ug.points[ib],
                        scalar: sb,
                        color_override,
                    });
                }
            };
            for cell_idx in 0..ug.num_cells() {
                let ct = ug.cell_types[cell_idx];
                let pts = ug.cell_points(cell_idx);
                if ct == CellType::Line || ct == CellType::PolyLine {
                    for w in pts.windows(2) {
                        push_edge(w[0], w[1], cell_idx);
                    }
                } else if ct.dimension() >= 2 {
                    for f in ct.faces(pts.len()) {
                        for k in 0..f.len() {
                            push_edge(pts[f[k]], pts[f[(k + 1) % f.len()]], cell_idx);
                        }
                    }
                }
            }
        }
        DataSet::ImageData(img) => {
            append_structured_wireframe(
                img.dimensions,
                &img.point_data,
                &img.cell_data,
                &style.color_by,
                inherited_override,
                |i, j, k| img.point_coords(i, j, k),
                |i, j, k| img.point_index(i, j, k),
                out,
                range,
            );
        }
        DataSet::RectilinearGrid(rect) => {
            append_structured_wireframe(
                rect.dimensions(),
                &rect.point_data,
                &rect.cell_data,
                &style.color_by,
                inherited_override,
                |i, j, k| rect.point_coords(i, j, k),
                |i, j, k| rect.point_index(i, j, k),
                out,
                range,
            );
        }
        DataSet::StructuredGrid(grid) => {
            if grid.points.len() >= grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2] {
                append_structured_wireframe(
                    grid.dimensions,
                    &grid.point_data,
                    &grid.cell_data,
                    &style.color_by,
                    inherited_override,
                    |i, j, k| grid.point_coords(i, j, k),
                    |i, j, k| grid.point_index(i, j, k),
                    out,
                    range,
                );
            }
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
                append_wireframe_dataset(block, style, &child_path, child_override, out, range);
            }
        }
    }
}

/// Extracts point cloud instances from any [`DataSet`].
pub fn prepare_points(dataset: &DataSet, color_by: &ColorBy) -> PreparedPoints {
    let style = ActorStyle {
        color_by: color_by.clone(),
        ..Default::default()
    };
    prepare_points_with_style(dataset, &style)
}

/// Style-aware variant of [`prepare_points`] honoring MultiBlock `hidden_blocks` and `block_colors`.
pub fn prepare_points_with_style(dataset: &DataSet, style: &ActorStyle) -> PreparedPoints {
    let mut instances = Vec::new();
    let mut range = None;
    append_points_dataset(dataset, style, "", None, &mut instances, &mut range);
    PreparedPoints {
        instances,
        scalar_range: range.unwrap_or((0.0, 1.0)),
    }
}

fn resolve_point_block_coloring(
    point_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
) -> (Option<Vec<f32>>, [f32; 4]) {
    if let Some(rgba) = block_override {
        return (None, [rgba[0], rgba[1], rgba[2], rgba[3].max(0.0)]);
    }
    match color_by {
        ColorBy::PointScalar { array, component } => {
            let sc = extract_scalars(point_data, array, *component);
            if sc.is_some() {
                (sc, NO_COLOR_OVERRIDE)
            } else {
                (None, DEFAULT_FALLBACK_SOLID)
            }
        }
        ColorBy::CellScalar { .. } => (None, DEFAULT_FALLBACK_SOLID),
        ColorBy::Solid(_) => (None, NO_COLOR_OVERRIDE),
    }
}

fn append_points_slice(
    points: &[[f32; 3]],
    point_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    out: &mut Vec<PointInstance>,
    range: &mut Option<(f64, f64)>,
) {
    let (scalars, color_override) =
        resolve_point_block_coloring(point_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    if points.len() >= PARALLEL_THRESHOLD {
        let sc_ref = scalars.as_deref();
        let built: Vec<PointInstance> = points
            .par_iter()
            .enumerate()
            .map(|(i, &position)| PointInstance {
                position,
                scalar: sc_ref.and_then(|s| s.get(i).copied()).unwrap_or(0.0),
                color_override,
            })
            .collect();
        out.extend(built);
    } else {
        out.reserve(points.len());
        for (i, &pos) in points.iter().enumerate() {
            let scalar = scalars.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(0.0);
            out.push(PointInstance {
                position: pos,
                scalar,
                color_override,
            });
        }
    }
}

fn append_points_grid<FCoord>(
    dims: [usize; 3],
    point_data: &FieldData,
    color_by: &ColorBy,
    block_override: Option<[f32; 4]>,
    coord: FCoord,
    out: &mut Vec<PointInstance>,
    range: &mut Option<(f64, f64)>,
) where
    FCoord: Fn(usize, usize, usize) -> [f32; 3] + Sync,
{
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;
    if total == 0 {
        return;
    }
    let (scalars, color_override) =
        resolve_point_block_coloring(point_data, color_by, block_override);
    if let Some(sc) = &scalars {
        merge_range(range, compute_finite_range(sc));
    }
    if total >= PARALLEL_THRESHOLD {
        let sc_ref = scalars.as_deref();
        let built: Vec<PointInstance> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let rem = idx / nx;
                let j = rem % ny;
                let k = rem / ny;
                PointInstance {
                    position: coord(i, j, k),
                    scalar: sc_ref.and_then(|s| s.get(idx).copied()).unwrap_or(0.0),
                    color_override,
                }
            })
            .collect();
        out.extend(built);
    } else {
        out.reserve(total);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + nx * (j + ny * k);
                    let scalar = scalars.as_ref().and_then(|s| s.get(idx).copied()).unwrap_or(0.0);
                    out.push(PointInstance {
                        position: coord(i, j, k),
                        scalar,
                        color_override,
                    });
                }
            }
        }
    }
}

fn append_points_dataset(
    dataset: &DataSet,
    style: &ActorStyle,
    block_path: &str,
    inherited_override: Option<[f32; 4]>,
    out: &mut Vec<PointInstance>,
    range: &mut Option<(f64, f64)>,
) {
    match dataset {
        DataSet::PolyData(poly) => {
            append_points_slice(
                &poly.points,
                &poly.point_data,
                &style.color_by,
                inherited_override,
                out,
                range,
            );
        }
        DataSet::ImageData(img) => {
            append_points_grid(
                img.dimensions,
                &img.point_data,
                &style.color_by,
                inherited_override,
                |i, j, k| img.point_coords(i, j, k),
                out,
                range,
            );
        }
        DataSet::RectilinearGrid(rect) => {
            append_points_grid(
                rect.dimensions(),
                &rect.point_data,
                &style.color_by,
                inherited_override,
                |i, j, k| rect.point_coords(i, j, k),
                out,
                range,
            );
        }
        DataSet::StructuredGrid(grid) => {
            append_points_slice(
                &grid.points,
                &grid.point_data,
                &style.color_by,
                inherited_override,
                out,
                range,
            );
        }
        DataSet::UnstructuredGrid(ug) => {
            append_points_slice(
                &ug.points,
                &ug.point_data,
                &style.color_by,
                inherited_override,
                out,
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
                append_points_dataset(block, style, &child_path, child_override, out, range);
            }
        }
    }
}

/// Extracts a 3D `R32Float` volume payload from an [`ImageData`] (or first ImageData block in a MultiBlock).
pub fn prepare_volume(dataset: &DataSet, color_by: &ColorBy) -> Option<PreparedVolume> {
    let style = ActorStyle {
        color_by: color_by.clone(),
        ..Default::default()
    };
    prepare_volume_with_style(dataset, &style)
}

/// Style-aware variant of [`prepare_volume`] honoring MultiBlock `hidden_blocks`.
pub fn prepare_volume_with_style(dataset: &DataSet, style: &ActorStyle) -> Option<PreparedVolume> {
    let img = find_image_data(dataset, style, "")?;
    let [nx, ny, nz] = img.dimensions;
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }

    let arr = match &style.color_by {
        ColorBy::PointScalar { array, .. } => resolve_array(&img.point_data, array),
        ColorBy::CellScalar { array, .. } => resolve_array(&img.cell_data, array)
            .or_else(|| img.point_data.active_scalars_array()),
        ColorBy::Solid(_) => img.point_data.active_scalars_array(),
    }?;

    let component = match &style.color_by {
        ColorBy::PointScalar { component, .. } | ColorBy::CellScalar { component, .. } => *component,
        ColorBy::Solid(_) => None,
    };

    let expected_pts = nx * ny * nz;
    let mut data = arr.to_scalar_f32(component);
    if data.len() < expected_pts {
        data.resize(expected_pts, 0.0);
    } else if data.len() > expected_pts {
        data.truncate(expected_pts);
    }

    let scalar_range = compute_finite_range(&data).unwrap_or((0.0, 1.0));
    let bb = img.bounds();
    let mut box_min = bb.min.to_array();
    let mut box_max = bb.max.to_array();
    for axis in 0..3 {
        if (box_max[axis] - box_min[axis]).abs() < 1e-5 {
            box_min[axis] -= 0.5;
            box_max[axis] += 0.5;
        }
    }

    Some(PreparedVolume {
        dims: [nx as u32, ny as u32, nz as u32],
        box_min,
        box_max,
        data,
        scalar_range,
    })
}

fn find_image_data<'a>(
    dataset: &'a DataSet,
    style: &ActorStyle,
    block_path: &str,
) -> Option<&'a ImageData> {
    match dataset {
        DataSet::ImageData(img) => Some(img),
        DataSet::MultiBlock(mb) => mb.blocks.iter().find_map(|(name, d)| {
            let child_path = join_block_path(block_path, name);
            if !style.is_block_visible(&child_path, name) {
                return None;
            }
            find_image_data(d, style, &child_path)
        }),
        _ => None,
    }
}

/// Vertex for the 3D orientation axes gizmo.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Builds a 3D RGB XYZ arrow triad centered at the origin (unit radius ~1.0).
pub fn build_axes_gizmo_mesh() -> Vec<GizmoVertex> {
    let mut verts = Vec::new();
    let axes = [
        (Vec3::X, Vec3::Y, Vec3::Z, [0.92, 0.22, 0.22]),
        (Vec3::Y, Vec3::Z, Vec3::X, [0.24, 0.85, 0.32]),
        (Vec3::Z, Vec3::X, Vec3::Y, [0.24, 0.52, 0.96]),
    ];
    let segments = 16usize;
    let shaft_r = 0.06;
    let shaft_len = 0.72;
    let head_r = 0.14;
    let tip_len = 1.0;

    for (dir, u, v, color) in axes {
        for s in 0..segments {
            let a0 = (s as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((s + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let r0 = u * a0.cos() + v * a0.sin();
            let r1 = u * a1.cos() + v * a1.sin();
            let n0 = r0.to_array();
            let n1 = r1.to_array();

            // Cylinder shaft
            let b0 = (r0 * shaft_r).to_array();
            let b1 = (r1 * shaft_r).to_array();
            let t0 = (dir * shaft_len + r0 * shaft_r).to_array();
            let t1 = (dir * shaft_len + r1 * shaft_r).to_array();

            verts.push(GizmoVertex { position: b0, normal: n0, color });
            verts.push(GizmoVertex { position: b1, normal: n1, color });
            verts.push(GizmoVertex { position: t1, normal: n1, color });

            verts.push(GizmoVertex { position: b0, normal: n0, color });
            verts.push(GizmoVertex { position: t1, normal: n1, color });
            verts.push(GizmoVertex { position: t0, normal: n0, color });

            // Cone head
            let h0 = (dir * shaft_len + r0 * head_r).to_array();
            let h1 = (dir * shaft_len + r1 * head_r).to_array();
            let tip = (dir * tip_len).to_array();
            let cone_n0 = (r0 * 0.9 + dir * 0.43).normalize().to_array();
            let cone_n1 = (r1 * 0.9 + dir * 0.43).normalize().to_array();
            let tip_n = ((r0 + r1) * 0.5 + dir * 0.45).normalize().to_array();

            verts.push(GizmoVertex { position: h0, normal: cone_n0, color });
            verts.push(GizmoVertex { position: h1, normal: cone_n1, color });
            verts.push(GizmoVertex { position: tip, normal: tip_n, color });
        }
    }

    // Small origin cube/octahedron in neutral gray
    let o_col = [0.75, 0.75, 0.78];
    let r = 0.11;
    let pts = [
        Vec3::new(r, 0.0, 0.0),
        Vec3::new(-r, 0.0, 0.0),
        Vec3::new(0.0, r, 0.0),
        Vec3::new(0.0, -r, 0.0),
        Vec3::new(0.0, 0.0, r),
        Vec3::new(0.0, 0.0, -r),
    ];
    let oct_faces = [
        (0, 2, 4), (4, 2, 1), (1, 2, 5), (5, 2, 0),
        (4, 3, 0), (1, 3, 4), (5, 3, 1), (0, 3, 5),
    ];
    for (a, b, c) in oct_faces {
        let n = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize().to_array();
        verts.push(GizmoVertex { position: pts[a].to_array(), normal: n, color: o_col });
        verts.push(GizmoVertex { position: pts[b].to_array(), normal: n, color: o_col });
        verts.push(GizmoVertex { position: pts[c].to_array(), normal: n, color: o_col });
    }

    verts
}

/// Per-vertex payload for 3D interactive viewport widget overlays.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WidgetVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Prepared line and translucent triangle geometry for [`ViewportWidget`] overlays.
#[derive(Debug, Clone, Default)]
pub struct PreparedWidgets {
    pub lines: Vec<WidgetVertex>,
    pub triangles: Vec<WidgetVertex>,
}

/// Builds wireframe and translucent triangle geometry for all [`ViewportWidget`]s.
pub fn prepare_widgets(widgets: &[ViewportWidget]) -> PreparedWidgets {
    let mut out = PreparedWidgets::default();
    for widget in widgets {
        match widget {
            ViewportWidget::Plane {
                origin,
                normal,
                size,
            } => {
                let o = Vec3::from_array(*origin);
                let n = Vec3::from_array(*normal).normalize_or(Vec3::Y);
                let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let u = helper.cross(n).normalize_or(Vec3::X);
                let v = n.cross(u).normalize_or(Vec3::Z);
                let h = size.max(1e-4) * 0.5;

                let c0 = (o - u * h - v * h).to_array();
                let c1 = (o + u * h - v * h).to_array();
                let c2 = (o + u * h + v * h).to_array();
                let c3 = (o - u * h + v * h).to_array();

                let fill_col = [0.22, 0.78, 0.96, 0.18];
                for p in [c0, c1, c2, c0, c2, c3] {
                    out.triangles.push(WidgetVertex {
                        position: p,
                        color: fill_col,
                    });
                }

                let edge_col = [0.28, 0.88, 1.0, 0.95];
                for (a, b) in [(c0, c1), (c1, c2), (c2, c3), (c3, c0)] {
                    out.lines.push(WidgetVertex {
                        position: a,
                        color: edge_col,
                    });
                    out.lines.push(WidgetVertex {
                        position: b,
                        color: edge_col,
                    });
                }

                // Normal vector arrow from plane origin
                let arrow_col = [1.0, 0.84, 0.24, 1.0];
                let tip = o + n * (h * 0.85);
                let head_base = o + n * (h * 0.65);
                let head_r = h * 0.10;
                out.lines.push(WidgetVertex {
                    position: o.to_array(),
                    color: arrow_col,
                });
                out.lines.push(WidgetVertex {
                    position: tip.to_array(),
                    color: arrow_col,
                });
                for dir in [u, -u, v, -v] {
                    out.lines.push(WidgetVertex {
                        position: tip.to_array(),
                        color: arrow_col,
                    });
                    out.lines.push(WidgetVertex {
                        position: (head_base + dir * head_r).to_array(),
                        color: arrow_col,
                    });
                }
            }
            ViewportWidget::Line { p1, p2 } => {
                let a = Vec3::from_array(*p1);
                let b = Vec3::from_array(*p2);
                let line_col = [1.0, 0.82, 0.20, 1.0];
                out.lines.push(WidgetVertex {
                    position: *p1,
                    color: line_col,
                });
                out.lines.push(WidgetVertex {
                    position: *p2,
                    color: line_col,
                });

                let handle = ((b - a).length() * 0.04).clamp(0.02, 0.25);
                let handle_col = [1.0, 0.45, 0.20, 1.0];
                for center in [a, b] {
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        out.lines.push(WidgetVertex {
                            position: (center - axis * handle).to_array(),
                            color: handle_col,
                        });
                        out.lines.push(WidgetVertex {
                            position: (center + axis * handle).to_array(),
                            color: handle_col,
                        });
                    }
                }
            }
            ViewportWidget::Box { min, max } => {
                let mn = Vec3::from_array(*min).min(Vec3::from_array(*max));
                let mx = Vec3::from_array(*min).max(Vec3::from_array(*max));
                let c = [
                    [mn.x, mn.y, mn.z],
                    [mx.x, mn.y, mn.z],
                    [mx.x, mx.y, mn.z],
                    [mn.x, mx.y, mn.z],
                    [mn.x, mn.y, mx.z],
                    [mx.x, mn.y, mx.z],
                    [mx.x, mx.y, mx.z],
                    [mn.x, mx.y, mx.z],
                ];
                let edge_col = [0.28, 0.94, 0.72, 0.95];
                let edges = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ];
                for (a, b) in edges {
                    out.lines.push(WidgetVertex {
                        position: c[a],
                        color: edge_col,
                    });
                    out.lines.push(WidgetVertex {
                        position: c[b],
                        color: edge_col,
                    });
                }

                let fill_col = [0.28, 0.94, 0.72, 0.08];
                let quads = [
                    [0, 3, 2, 1],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                ];
                for q in quads {
                    for idx in [q[0], q[1], q[2], q[0], q[2], q[3]] {
                        out.triangles.push(WidgetVertex {
                            position: c[idx],
                            color: fill_col,
                        });
                    }
                }
            }
            ViewportWidget::Sphere { center, radius } => {
                let c = Vec3::from_array(*center);
                let r = radius.max(1e-4);
                let ring_col = [1.0, 0.56, 0.22, 0.95];
                let segs = 48usize;
                let planes = [(Vec3::X, Vec3::Y), (Vec3::X, Vec3::Z), (Vec3::Y, Vec3::Z)];
                for (u, v) in planes {
                    for s in 0..segs {
                        let a0 = (s as f32 / segs as f32) * std::f32::consts::TAU;
                        let a1 = ((s + 1) as f32 / segs as f32) * std::f32::consts::TAU;
                        let p0 = c + (u * a0.cos() + v * a0.sin()) * r;
                        let p1 = c + (u * a1.cos() + v * a1.sin()) * r;
                        out.lines.push(WidgetVertex {
                            position: p0.to_array(),
                            color: ring_col,
                        });
                        out.lines.push(WidgetVertex {
                            position: p1.to_array(),
                            color: ring_col,
                        });
                    }
                }
                let cross_r = r * 0.15;
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    out.lines.push(WidgetVertex {
                        position: (c - axis * cross_r).to_array(),
                        color: ring_col,
                    });
                    out.lines.push(WidgetVertex {
                        position: (c + axis * cross_r).to_array(),
                        color: ring_col,
                    });
                }
            }
        }
    }
    out
}
