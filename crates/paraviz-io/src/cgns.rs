//! CGNS (`.cgns`, HDF5-backed) CFD dataset reader producing [`StructuredGrid`],
//! [`UnstructuredGrid`], or [`MultiBlockDataSet`].

use crate::error::{IoError, Result};
use crate::hdf5_common::{
    decode_nul_terminated_bytes, group_attr_string, open_hdf5, read_dataset_as_data_array,
    read_dataset_f32, read_dataset_i64, read_dataset_strings, read_dataset_u8,
};
use crate::limits::{check_alloc_bytes, safe_prealloc_capacity, MAX_ALLOC_BYTES};
use hdf5_reader::{group::Group, Dataset, Datatype};
use paraviz_core::{
    CellType, DataSet, FieldData, MultiBlockDataSet, StructuredGrid, UnstructuredGrid,
};
use std::path::Path;

/// Reads an HDF5-backed `.cgns` file into a [`DataSet`].
pub fn read_cgns(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let file = open_hdf5(path)?;
    let root = file
        .root_group()
        .map_err(|e| IoError::Parse(format!("Failed to open CGNS root group: {e}")))?;

    let bases = find_cgns_bases(&root)?;
    if bases.is_empty() {
        return Err(IoError::Parse(format!(
            "CGNS file '{}' contains no CGNSBase_t groups",
            path.display()
        )));
    }

    let multi_base = bases.len() > 1;
    let mut zones_out: Vec<(String, DataSet)> = Vec::new();

    for base in bases {
        let base_name = base.name().to_string();
        let zones = find_cgns_zones(&base)?;
        for zone in zones {
            let zone_name = if multi_base {
                format!("{base_name}/{}", zone.name())
            } else {
                zone.name().to_string()
            };
            let ds = read_cgns_zone(&zone)?;
            zones_out.push((zone_name, ds));
        }
    }

    if zones_out.is_empty() {
        return Err(IoError::Parse(format!(
            "CGNS file '{}' contains no Zone_t groups",
            path.display()
        )));
    }

    if zones_out.len() == 1 {
        Ok(zones_out.remove(0).1)
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, ds) in zones_out {
            mb.add_block(name, ds);
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn find_cgns_bases(root: &Group) -> Result<Vec<Group>> {
    let groups = root
        .groups()
        .map_err(|e| IoError::Parse(format!("Failed to list CGNS root groups: {e}")))?;
    let mut bases = Vec::new();
    for g in groups {
        let label = group_attr_string(&g, "label").unwrap_or_default();
        if label == "CGNSBase_t"
            || (label.is_empty() && g.name().to_ascii_lowercase().starts_with("base"))
        {
            bases.push(g);
        }
    }
    Ok(bases)
}

fn find_cgns_zones(base: &Group) -> Result<Vec<Group>> {
    let groups = base.groups().map_err(|e| {
        IoError::Parse(format!("Failed to list CGNS base '{}': {e}", base.name()))
    })?;
    let mut zones = Vec::new();
    for g in groups {
        let label = group_attr_string(&g, "label").unwrap_or_default();
        if label == "Zone_t"
            || (label.is_empty() && g.name().to_ascii_lowercase().starts_with("zone"))
        {
            zones.push(g);
        }
    }
    Ok(zones)
}

fn read_cgns_zone(zone: &Group) -> Result<DataSet> {
    let zone_data = zone
        .dataset(" data")
        .ok()
        .map(|ds| read_dataset_i64(&ds, None))
        .transpose()?;

    let zone_type = find_zone_type(zone)?;
    let has_elements = !find_children_by_label(zone, "Elements_t", &[])?.is_empty();

    let is_structured = match zone_type.as_deref() {
        Some(t) if t.eq_ignore_ascii_case("Structured") => true,
        Some(t) if t.eq_ignore_ascii_case("Unstructured") => false,
        Some(other) => {
            return Err(IoError::UnsupportedFormat(format!(
                "Unsupported CGNS ZoneType '{other}' in zone '{}'",
                zone.name()
            )));
        }
        None => !has_elements,
    };

    if is_structured {
        read_structured_zone(zone, zone_data.as_deref()).map(DataSet::StructuredGrid)
    } else {
        read_unstructured_zone(zone, zone_data.as_deref()).map(DataSet::UnstructuredGrid)
    }
}

fn find_zone_type(zone: &Group) -> Result<Option<String>> {
    if let Some(zt_grp) = find_first_child_by_label(zone, "ZoneType_t", &["ZoneType"])?
        && let Some(s) = read_cgns_group_string(&zt_grp)
    {
        return Ok(Some(s));
    }
    if let Ok(ds) = zone.dataset("ZoneType") {
        return Ok(read_dataset_single_string(&ds));
    }
    Ok(None)
}

fn read_structured_zone(zone: &Group, zone_data: Option<&[i64]>) -> Result<StructuredGrid> {
    let grid_grp = find_first_child_by_label(
        zone,
        "GridCoordinates_t",
        &["GridCoordinates", "GridCoordinates_t"],
    )?
    .ok_or_else(|| {
        IoError::Parse(format!(
            "CGNS structured zone '{}' missing GridCoordinates_t",
            zone.name()
        ))
    })?;

    let cx_ds = read_cgns_child_dataset(&grid_grp, "CoordinateX").ok_or_else(|| {
        IoError::Parse(format!(
            "CGNS zone '{}' GridCoordinates missing CoordinateX",
            zone.name()
        ))
    })?;
    let cx_shape = cx_ds.shape().to_vec();
    let cx = read_dataset_f32(&cx_ds, None)?;
    let cy = read_cgns_child_dataset(&grid_grp, "CoordinateY")
        .map(|ds| read_dataset_f32(&ds, None))
        .transpose()?;
    let cz = read_cgns_child_dataset(&grid_grp, "CoordinateZ")
        .map(|ds| read_dataset_f32(&ds, None))
        .transpose()?;

    let dims = infer_structured_dims(&cx_shape, zone_data, cx.len())?;
    let n_pts = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .ok_or_else(|| IoError::InvalidData("CGNS structured grid dimensions overflow usize".into()))?;

    if cx.len() != n_pts {
        return Err(IoError::InvalidData(format!(
            "CGNS zone '{}' CoordinateX length ({}) does not match grid dimensions {:?} ({n_pts})",
            zone.name(),
            cx.len(),
            dims
        )));
    }
    if let Some(ref y) = cy
        && y.len() != n_pts
    {
        return Err(IoError::InvalidData(format!(
            "CGNS zone '{}' CoordinateY length ({}) != {n_pts}",
            zone.name(),
            y.len()
        )));
    }
    if let Some(ref z) = cz
        && z.len() != n_pts
    {
        return Err(IoError::InvalidData(format!(
            "CGNS zone '{}' CoordinateZ length ({}) != {n_pts}",
            zone.name(),
            z.len()
        )));
    }

    let cap = safe_prealloc_capacity(n_pts, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
    let mut points = Vec::with_capacity(cap);
    for idx in 0..n_pts {
        points.push([
            cx[idx],
            cy.as_ref().map_or(0.0, |v| v[idx]),
            cz.as_ref().map_or(0.0, |v| v[idx]),
        ]);
    }

    let mut sg = StructuredGrid {
        dimensions: dims,
        points,
        ..Default::default()
    };

    attach_flow_solutions(
        zone,
        sg.num_points(),
        sg.num_cells(),
        &mut sg.point_data,
        &mut sg.cell_data,
    )?;

    Ok(sg)
}

fn infer_structured_dims(
    coord_shape: &[u64],
    zone_data: Option<&[i64]>,
    total_pts: usize,
) -> Result<[usize; 3]> {
    // In CGNS/HDF5 (ADFH), HDF5 dimensions are stored slowest-to-fastest [nk, nj, ni]
    // so that `i` varies fastest in memory, matching `StructuredGrid`.
    match coord_shape.len() {
        3 => {
            let ni = usize::try_from(coord_shape[2]).unwrap_or(0);
            let nj = usize::try_from(coord_shape[1]).unwrap_or(0);
            let nk = usize::try_from(coord_shape[0]).unwrap_or(0);
            return Ok([ni, nj, nk]);
        }
        2 => {
            let ni = usize::try_from(coord_shape[1]).unwrap_or(0);
            let nj = usize::try_from(coord_shape[0]).unwrap_or(0);
            return Ok([ni, nj, 1]);
        }
        _ => {}
    }

    if let Some(zd) = zone_data {
        if zd.len() >= 9 {
            let ni = usize::try_from(zd[0].max(0)).unwrap_or(0);
            let nj = usize::try_from(zd[1].max(0)).unwrap_or(0);
            let nk = usize::try_from(zd[2].max(0)).unwrap_or(0);
            if ni * nj * nk == total_pts {
                return Ok([ni, nj, nk]);
            }
        }
        if zd.len() >= 6 {
            let ni = usize::try_from(zd[0].max(0)).unwrap_or(0);
            let nj = usize::try_from(zd[1].max(0)).unwrap_or(0);
            if ni * nj == total_pts {
                return Ok([ni, nj, 1]);
            }
        }
    }

    Ok([total_pts, 1, 1])
}

struct CgnsElementSection {
    range_start: i64,
    range_end: i64,
    elem_type_code: i64,
    connectivity: Vec<i64>,
    start_offsets: Option<Vec<i64>>,
}

fn read_unstructured_zone(zone: &Group, zone_data: Option<&[i64]>) -> Result<UnstructuredGrid> {
    let grid_grp = find_first_child_by_label(
        zone,
        "GridCoordinates_t",
        &["GridCoordinates", "GridCoordinates_t"],
    )?
    .ok_or_else(|| {
        IoError::Parse(format!(
            "CGNS unstructured zone '{}' missing GridCoordinates_t",
            zone.name()
        ))
    })?;

    let cx_ds = read_cgns_child_dataset(&grid_grp, "CoordinateX").ok_or_else(|| {
        IoError::Parse(format!(
            "CGNS zone '{}' GridCoordinates missing CoordinateX",
            zone.name()
        ))
    })?;
    let cx = read_dataset_f32(&cx_ds, None)?;
    let cy = read_cgns_child_dataset(&grid_grp, "CoordinateY")
        .map(|ds| read_dataset_f32(&ds, None))
        .transpose()?;
    let cz = read_cgns_child_dataset(&grid_grp, "CoordinateZ")
        .map(|ds| read_dataset_f32(&ds, None))
        .transpose()?;

    let n_pts = cx.len();
    if let Some(ref y) = cy
        && y.len() != n_pts
    {
        return Err(IoError::InvalidData(format!(
            "CGNS zone '{}' CoordinateY length ({}) != CoordinateX length ({n_pts})",
            zone.name(),
            y.len()
        )));
    }
    if let Some(ref z) = cz
        && z.len() != n_pts
    {
        return Err(IoError::InvalidData(format!(
            "CGNS zone '{}' CoordinateZ length ({}) != CoordinateX length ({n_pts})",
            zone.name(),
            z.len()
        )));
    }

    let cap = safe_prealloc_capacity(n_pts, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
    let mut points = Vec::with_capacity(cap);
    for idx in 0..n_pts {
        points.push([
            cx[idx],
            cy.as_ref().map_or(0.0, |v| v[idx]),
            cz.as_ref().map_or(0.0, |v| v[idx]),
        ]);
    }

    let elem_groups = find_children_by_label(zone, "Elements_t", &["Elements"])?;
    let mut sections = Vec::with_capacity(elem_groups.len());

    for eg in elem_groups {
        let Some(data_ds) = eg.dataset(" data").ok() else {
            continue;
        };
        let meta = read_dataset_i64(&data_ds, None)?;
        let elem_type_code = meta.first().copied().unwrap_or(0);

        let (range_start, range_end) = if let Some(range_ds) = read_cgns_child_dataset(&eg, "ElementRange") {
            let r = read_dataset_i64(&range_ds, None)?;
            if r.len() >= 2 {
                (r[0], r[1])
            } else {
                (1, 0)
            }
        } else {
            (1, 0)
        };

        let conn_ds = read_cgns_child_dataset(&eg, "ElementConnectivity").ok_or_else(|| {
            IoError::Parse(format!(
                "CGNS Elements_t '{}' missing ElementConnectivity",
                eg.name()
            ))
        })?;
        let connectivity = read_dataset_i64(&conn_ds, None)?;
        let start_offsets = read_cgns_child_dataset(&eg, "ElementStartOffset")
            .map(|ds| read_dataset_i64(&ds, None))
            .transpose()?;

        sections.push(CgnsElementSection {
            range_start,
            range_end,
            elem_type_code,
            connectivity,
            start_offsets,
        });
    }

    sections.sort_by_key(|s| s.range_start);

    // If zone_data declares NCell and there are extra boundary sections beyond NCell
    // while a CellCenter solution matches NCell, keep only primary cells 1..=NCell.
    let declared_cells = zone_data
        .and_then(|zd| zd.get(1).copied())
        .filter(|&c| c > 0)
        .map(|c| c as usize);
    let has_matching_cell_solution = declared_cells
        .map(|nc| zone_has_cell_solution_of_len(zone, nc))
        .unwrap_or(false);

    let mut ug = UnstructuredGrid::new();
    ug.points = points;

    for sec in &sections {
        if let Some(nc) = declared_cells
            && has_matching_cell_solution
            && sec.range_end > 0
            && sec.range_start > nc as i64
        {
            continue;
        }
        append_cgns_section(&mut ug, sec, n_pts)?;
    }

    attach_flow_solutions(
        zone,
        ug.num_points(),
        ug.num_cells(),
        &mut ug.point_data,
        &mut ug.cell_data,
    )?;

    Ok(ug)
}

fn append_cgns_section(
    ug: &mut UnstructuredGrid,
    sec: &CgnsElementSection,
    num_points: usize,
) -> Result<()> {
    let conn = &sec.connectivity;
    let convert_idx = |raw: i64| -> Result<u32> {
        if raw < 1 {
            return Err(IoError::InvalidData(format!(
                "CGNS 1-based vertex index {raw} must be >= 1"
            )));
        }
        let zero_based = (raw - 1) as usize;
        if zero_based >= num_points {
            return Err(IoError::InvalidData(format!(
                "CGNS vertex index {raw} out of bounds ({num_points} points)"
            )));
        }
        u32::try_from(zero_based)
            .map_err(|_| IoError::InvalidData("CGNS vertex index overflows u32".into()))
    };

    if sec.elem_type_code == 20 {
        // MIXED: each element starts with its ElementType_t code
        if let Some(offs) = &sec.start_offsets {
            if offs.len() < 2 {
                return Ok(());
            }
            for w in offs.windows(2) {
                let s = usize::try_from(w[0].max(0)).unwrap_or(0);
                let e = usize::try_from(w[1].max(0)).unwrap_or(0);
                if s >= e || e > conn.len() {
                    return Err(IoError::InvalidData(
                        "CGNS MIXED ElementStartOffset out of bounds".into(),
                    ));
                }
                let code = conn[s];
                let ctype = if code == 22 {
                    CellType::Polygon
                } else {
                    cgns_element_type_to_vtk(code)?.0
                };
                let mut verts = Vec::with_capacity(e - s - 1);
                for &idx in &conn[s + 1..e] {
                    verts.push(convert_idx(idx)?);
                }
                ug.push_cell(ctype, &verts);
            }
        } else {
            let mut pos = 0usize;
            while pos < conn.len() {
                let code = conn[pos];
                pos += 1;
                let (ctype, n_nodes) = if code == 22 {
                    let count = usize::try_from(*conn.get(pos).ok_or_else(|| {
                        IoError::InvalidData("CGNS MIXED NGON_n missing vertex count".into())
                    })?)
                    .map_err(|_| IoError::InvalidData("Invalid NGON_n vertex count".into()))?;
                    pos += 1;
                    (CellType::Polygon, count)
                } else {
                    cgns_element_type_to_vtk(code)?
                };
                if pos + n_nodes > conn.len() {
                    return Err(IoError::InvalidData(
                        "CGNS MIXED ElementConnectivity truncated".into(),
                    ));
                }
                let mut verts = Vec::with_capacity(n_nodes);
                for &idx in &conn[pos..pos + n_nodes] {
                    verts.push(convert_idx(idx)?);
                }
                ug.push_cell(ctype, &verts);
                pos += n_nodes;
            }
        }
        return Ok(());
    }

    if sec.elem_type_code == 22 {
        // NGON_n
        if let Some(offs) = &sec.start_offsets {
            for w in offs.windows(2) {
                let s = usize::try_from(w[0].max(0)).unwrap_or(0);
                let e = usize::try_from(w[1].max(0)).unwrap_or(0);
                if s > e || e > conn.len() {
                    return Err(IoError::InvalidData(
                        "CGNS NGON_n ElementStartOffset out of bounds".into(),
                    ));
                }
                let mut verts = Vec::with_capacity(e - s);
                for &idx in &conn[s..e] {
                    verts.push(convert_idx(idx)?);
                }
                ug.push_cell(CellType::Polygon, &verts);
            }
        } else {
            let mut pos = 0usize;
            while pos < conn.len() {
                let count = usize::try_from(conn[pos]).map_err(|_| {
                    IoError::InvalidData("CGNS NGON_n negative vertex count".into())
                })?;
                pos += 1;
                if pos + count > conn.len() {
                    return Err(IoError::InvalidData(
                        "CGNS NGON_n ElementConnectivity truncated".into(),
                    ));
                }
                let mut verts = Vec::with_capacity(count);
                for &idx in &conn[pos..pos + count] {
                    verts.push(convert_idx(idx)?);
                }
                ug.push_cell(CellType::Polygon, &verts);
                pos += count;
            }
        }
        return Ok(());
    }

    let (ctype, nodes_per_cell) = cgns_element_type_to_vtk(sec.elem_type_code)?;
    if !conn.len().is_multiple_of(nodes_per_cell) {
        return Err(IoError::InvalidData(format!(
            "CGNS ElementConnectivity length ({}) is not a multiple of {nodes_per_cell}",
            conn.len()
        )));
    }
    let n_cells = conn.len() / nodes_per_cell;
    check_alloc_bytes(n_cells, std::mem::size_of::<CellType>(), MAX_ALLOC_BYTES)?;

    let mut verts = Vec::with_capacity(nodes_per_cell);
    for chunk in conn.chunks_exact(nodes_per_cell) {
        verts.clear();
        for &idx in chunk {
            verts.push(convert_idx(idx)?);
        }
        ug.push_cell(ctype, &verts);
    }

    Ok(())
}

fn cgns_element_type_to_vtk(code: i64) -> Result<(CellType, usize)> {
    match code {
        2 => Ok((CellType::Vertex, 1)),          // NODE
        3 => Ok((CellType::Line, 2)),            // BAR_2
        4 => Ok((CellType::QuadraticEdge, 3)),   // BAR_3
        5 => Ok((CellType::Triangle, 3)),        // TRI_3
        6 => Ok((CellType::QuadraticTriangle, 6)), // TRI_6
        7 => Ok((CellType::Quad, 4)),            // QUAD_4
        8 => Ok((CellType::QuadraticQuad, 8)),   // QUAD_8
        9 => Ok((CellType::BiQuadraticQuad, 9)), // QUAD_9
        10 => Ok((CellType::Tetra, 4)),          // TETRA_4
        11 => Ok((CellType::QuadraticTetra, 10)), // TETRA_10
        12 => Ok((CellType::Pyramid, 5)),        // PYRA_5
        13 => Ok((CellType::QuadraticPyramid, 14)), // PYRA_14
        24 => Ok((CellType::QuadraticPyramid, 13)), // PYRA_13
        14 => Ok((CellType::Wedge, 6)),          // PENTA_6
        15 => Ok((CellType::QuadraticWedge, 15)), // PENTA_15
        17 => Ok((CellType::Hexahedron, 8)),     // HEXA_8
        18 => Ok((CellType::QuadraticHexahedron, 20)), // HEXA_20
        19 => Ok((CellType::TriQuadraticHexahedron, 27)), // HEXA_27
        other => Err(IoError::UnsupportedFormat(format!(
            "Unsupported CGNS ElementType_t code {other}"
        ))),
    }
}

fn zone_has_cell_solution_of_len(zone: &Group, expected_cells: usize) -> bool {
    let Ok(sols) = find_children_by_label(zone, "FlowSolution_t", &["FlowSolution"]) else {
        return false;
    };
    for sol in sols {
        let loc = find_grid_location(&sol).ok().flatten();
        if loc.as_deref().is_some_and(|l| l.eq_ignore_ascii_case("CellCenter"))
            && let Ok(arrays) = collect_flow_solution_datasets(&sol)
        {
            for (_, ds) in arrays {
                let count: u64 = ds.shape().iter().product();
                if count as usize == expected_cells {
                    return true;
                }
            }
        }
    }
    false
}

fn attach_flow_solutions(
    zone: &Group,
    num_points: usize,
    num_cells: usize,
    point_data: &mut FieldData,
    cell_data: &mut FieldData,
) -> Result<()> {
    let sols = find_children_by_label(zone, "FlowSolution_t", &["FlowSolution"])?;
    for sol in sols {
        let grid_loc = find_grid_location(&sol)?;
        let arrays = collect_flow_solution_datasets(&sol)?;
        for (name, ds) in arrays {
            if !matches!(
                ds.dtype(),
                Datatype::FloatingPoint { .. } | Datatype::FixedPoint { .. }
            ) {
                continue;
            }
            let arr = read_dataset_as_data_array(&ds, &name, 1, None)?;
            let len = arr.num_tuples();

            let target_is_cell = match grid_loc.as_deref() {
                Some(loc) if loc.eq_ignore_ascii_case("CellCenter") => true,
                Some(loc) if loc.eq_ignore_ascii_case("Vertex") => false,
                _ => len == num_cells && num_cells != num_points,
            };

            if target_is_cell {
                if len == num_cells || num_cells == 0 {
                    cell_data.add_array(arr);
                }
            } else if len == num_points || num_points == 0 {
                point_data.add_array(arr);
            }
        }
    }
    Ok(())
}

fn find_grid_location(sol: &Group) -> Result<Option<String>> {
    if let Some(gl_grp) = find_first_child_by_label(sol, "GridLocation_t", &["GridLocation"])?
        && let Some(s) = read_cgns_group_string(&gl_grp)
    {
        return Ok(Some(s));
    }
    if let Ok(ds) = sol.dataset("GridLocation") {
        return Ok(read_dataset_single_string(&ds));
    }
    Ok(None)
}

fn collect_flow_solution_datasets(sol: &Group) -> Result<Vec<(String, Dataset)>> {
    let (groups, datasets) = sol
        .members()
        .map_err(|e| IoError::Parse(format!("Failed to list CGNS FlowSolution '{}': {e}", sol.name())))?;
    let mut out = Vec::new();
    for ds in datasets {
        if ds.name() == " data" || ds.name() == "GridLocation" {
            continue;
        }
        out.push((ds.name().to_string(), ds));
    }
    for g in groups {
        let label = group_attr_string(&g, "label").unwrap_or_default();
        if matches!(
            label.as_str(),
            "GridLocation_t"
                | "PointRange_t"
                | "PointList_t"
                | "Rind_t"
                | "DataClass_t"
                | "DimensionalUnits_t"
                | "Descriptor_t"
        ) || g.name() == "GridLocation"
        {
            continue;
        }
        if (label == "DataArray_t" || label.is_empty())
            && let Ok(data_ds) = g.dataset(" data")
        {
            out.push((g.name().to_string(), data_ds));
        }
    }
    Ok(out)
}

fn find_first_child_by_label(
    parent: &Group,
    target_label: &str,
    fallback_names: &[&str],
) -> Result<Option<Group>> {
    let mut matches = find_children_by_label(parent, target_label, fallback_names)?;
    if matches.is_empty() {
        Ok(None)
    } else {
        Ok(Some(matches.remove(0)))
    }
}

fn find_children_by_label(
    parent: &Group,
    target_label: &str,
    fallback_names: &[&str],
) -> Result<Vec<Group>> {
    let groups = parent
        .groups()
        .map_err(|e| IoError::Parse(format!("Failed to list CGNS group '{}': {e}", parent.name())))?;
    let mut out = Vec::new();
    for g in groups {
        let label = group_attr_string(&g, "label").unwrap_or_default();
        let name_matches = label.is_empty()
            && fallback_names
                .iter()
                .any(|prefix| g.name().eq_ignore_ascii_case(prefix) || g.name().starts_with(prefix));
        if label == target_label || name_matches {
            out.push(g);
        }
    }
    Ok(out)
}

fn read_cgns_child_dataset(parent: &Group, name: &str) -> Option<Dataset> {
    if let Ok(g) = parent.group(name)
        && let Ok(ds) = g.dataset(" data")
    {
        return Some(ds);
    }
    parent.dataset(name).ok()
}

fn read_cgns_group_string(group: &Group) -> Option<String> {
    let ds = group.dataset(" data").ok()?;
    read_dataset_single_string(&ds)
}

fn read_dataset_single_string(ds: &Dataset) -> Option<String> {
    match ds.dtype() {
        Datatype::String { .. } | Datatype::VarLen { .. } => {
            read_dataset_strings(ds).ok()?.into_iter().next()
        }
        Datatype::FixedPoint { size: 1, .. } => {
            let bytes = read_dataset_u8(ds, None).ok()?;
            Some(decode_nul_terminated_bytes(&bytes))
        }
        _ => None,
    }
}

/// Returns `true` if the HDF5 file has a CGNS signature (`/CGNSLibraryVersion` or `CGNSBase_t`).
pub(crate) fn is_cgns_hdf5(file: &hdf5_reader::Hdf5File) -> bool {
    let Ok(root) = file.root_group() else {
        return false;
    };
    if root.dataset("CGNSLibraryVersion").is_ok()
        || root.group("CGNSLibraryVersion").is_ok()
    {
        return true;
    }
    if let Ok(groups) = root.groups() {
        for g in groups {
            if group_attr_string(&g, "label").as_deref() == Some("CGNSBase_t") {
                return true;
            }
        }
    }
    false
}
