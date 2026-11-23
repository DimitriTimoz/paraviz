//! VTKHDF (`.vtkhdf`) HDF5 scientific dataset reader.
//!
//! Supports `ImageData`, `PolyData`, `UnstructuredGrid`, `PartitionedDataSet`,
//! `PartitionedDataSetCollection`, and `MultiBlockDataSet`, including temporal `Steps`.

use crate::error::{IoError, Result};
use crate::hdf5_common::{
    attr_as_f64_vec, attr_as_i64_vec, group_attr_string, open_hdf5, pack_points_f32,
    read_dataset_as_data_array, read_dataset_f32, read_dataset_f64, read_dataset_i64,
    read_dataset_u32, read_dataset_u8, selection_element_count, shape_element_count, slice_axis0,
    index_axis0,
};
use crate::limits::{safe_prealloc_capacity, MAX_ALLOC_BYTES};
use hdf5_reader::{group::Group, Dataset, SliceInfo};
use paraviz_core::{
    CellType, DataSet, FieldData, ImageData, MultiBlockDataSet, PolyData, UnstructuredGrid,
};
use std::path::Path;

const MAX_COMPOSITE_DEPTH: usize = 32;

/// Reads a `.vtkhdf` file into a [`DataSet`] (first time step if temporal).
pub fn read_vtkhdf(path: impl AsRef<Path>) -> Result<DataSet> {
    read_vtkhdf_impl(path.as_ref(), None)
}

/// Reads the time step closest to `time` from a temporal `.vtkhdf` file (or the static dataset).
pub fn read_vtkhdf_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    read_vtkhdf_impl(path.as_ref(), Some(time))
}

/// Inspects a `.vtkhdf` file and returns its temporal time step values (empty if static).
pub fn inspect_vtkhdf_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    let file = open_hdf5(path.as_ref())?;
    let root = open_vtkhdf_root(&file)?;
    inspect_group_time_steps(&root, 0)
}

fn open_vtkhdf_root(file: &hdf5_reader::Hdf5File) -> Result<Group> {
    if let Ok(g) = file.group("VTKHDF") {
        return Ok(g);
    }
    let root = file
        .root_group()
        .map_err(|e| IoError::Parse(format!("Failed to open HDF5 root group: {e}")))?;
    if group_attr_string(&root, "Type").is_some() {
        return Ok(root);
    }
    Err(IoError::Parse(
        "VTKHDF file is missing root '/VTKHDF' group".into(),
    ))
}

fn inspect_group_time_steps(group: &Group, depth: usize) -> Result<Vec<f64>> {
    if depth > MAX_COMPOSITE_DEPTH {
        return Ok(Vec::new());
    }
    if let Ok(steps_grp) = group.group("Steps")
        && let Ok(values_ds) = steps_grp.dataset("Values")
    {
        return read_dataset_f64(&values_ds, None);
    }
    if let Ok(children) = group.groups() {
        for child in children {
            let name = child.name();
            if name == "Steps" || name == "PointData" || name == "CellData" || name == "FieldData" {
                continue;
            }
            let steps = inspect_group_time_steps(&child, depth + 1)?;
            if !steps.is_empty() {
                return Ok(steps);
            }
        }
    }
    Ok(Vec::new())
}

fn read_vtkhdf_impl(path: &Path, target_time: Option<f64>) -> Result<DataSet> {
    let file = open_hdf5(path)?;
    let root = open_vtkhdf_root(&file)?;
    read_vtkhdf_group(&root, target_time, 0)
}

fn read_vtkhdf_group(group: &Group, target_time: Option<f64>, depth: usize) -> Result<DataSet> {
    if depth > MAX_COMPOSITE_DEPTH {
        return Err(IoError::InvalidData(format!(
            "VTKHDF composite hierarchy exceeded maximum depth of {MAX_COMPOSITE_DEPTH}"
        )));
    }

    let vtk_type = group_attr_string(group, "Type").ok_or_else(|| {
        IoError::Parse(format!(
            "VTKHDF group '{}' is missing 'Type' attribute",
            group.name()
        ))
    })?;

    match vtk_type.as_str() {
        "ImageData" => read_image_data(group, target_time).map(DataSet::ImageData),
        "PolyData" => read_poly_data(group, target_time).map(DataSet::PolyData),
        "UnstructuredGrid" => read_unstructured_grid(group, target_time).map(DataSet::UnstructuredGrid),
        "PartitionedDataSetCollection" | "MultiBlockDataSet" | "PartitionedDataSet" => {
            read_composite(group, target_time, depth).map(DataSet::MultiBlock)
        }
        other => Err(IoError::UnsupportedFormat(format!(
            "Unsupported VTKHDF dataset type '{other}'"
        ))),
    }
}

struct StepsInfo {
    group: Group,
    step_idx: usize,
    n_steps: usize,
}

fn resolve_steps(group: &Group, target_time: Option<f64>) -> Result<Option<StepsInfo>> {
    let Ok(steps_grp) = group.group("Steps") else {
        return Ok(None);
    };
    let values = if let Ok(ds) = steps_grp.dataset("Values") {
        read_dataset_f64(&ds, None)?
    } else {
        return Ok(None);
    };
    if values.is_empty() {
        return Ok(None);
    }
    let step_idx = match target_time {
        None => 0,
        Some(t) => values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - t).abs().total_cmp(&(*b - t).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0),
    };
    Ok(Some(StepsInfo {
        group: steps_grp,
        step_idx,
        n_steps: values.len(),
    }))
}

fn read_image_data(group: &Group, target_time: Option<f64>) -> Result<ImageData> {
    let whole_extent_attr = group.attribute("WholeExtent").map_err(|_| {
        IoError::Parse("VTKHDF ImageData is missing 'WholeExtent' attribute".into())
    })?;
    let ext = attr_as_i64_vec(&whole_extent_attr)?;
    if ext.len() != 6 {
        return Err(IoError::InvalidData(format!(
            "VTKHDF ImageData WholeExtent must have 6 integers, got {}",
            ext.len()
        )));
    }
    let nx = usize::try_from((ext[1] - ext[0] + 1).max(0))
        .map_err(|_| IoError::InvalidData("VTKHDF ImageData X dimension overflows usize".into()))?;
    let ny = usize::try_from((ext[3] - ext[2] + 1).max(0))
        .map_err(|_| IoError::InvalidData("VTKHDF ImageData Y dimension overflows usize".into()))?;
    let nz = usize::try_from((ext[5] - ext[4] + 1).max(0))
        .map_err(|_| IoError::InvalidData("VTKHDF ImageData Z dimension overflows usize".into()))?;

    let origin = if let Ok(attr) = group.attribute("Origin") {
        let v = attr_as_f64_vec(&attr)?;
        if v.len() >= 3 {
            [v[0], v[1], v[2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    } else {
        [0.0, 0.0, 0.0]
    };

    let spacing = if let Ok(attr) = group.attribute("Spacing") {
        let v = attr_as_f64_vec(&attr)?;
        if v.len() >= 3 {
            [v[0], v[1], v[2]]
        } else {
            [1.0, 1.0, 1.0]
        }
    } else {
        [1.0, 1.0, 1.0]
    };

    let mut img = ImageData::new([nx, ny, nz], origin, spacing);
    let steps = resolve_steps(group, target_time)?;

    let cz = nz.saturating_sub(1).max(1);
    img.point_data = read_image_attr_group(
        group,
        "PointData",
        img.num_points(),
        nz,
        steps.as_ref(),
        "PointDataOffsets",
    )?;
    img.cell_data = read_image_attr_group(
        group,
        "CellData",
        img.num_cells(),
        cz,
        steps.as_ref(),
        "CellDataOffsets",
    )?;
    img.field_data = read_field_attr_group(group, "FieldData", steps.as_ref(), "FieldDataOffsets")?;

    Ok(img)
}

fn read_image_attr_group(
    parent: &Group,
    group_name: &str,
    expected_tuples: usize,
    slab_z: usize,
    steps: Option<&StepsInfo>,
    offsets_group_name: &str,
) -> Result<FieldData> {
    let Ok(grp) = parent.group(group_name) else {
        return Ok(FieldData::new());
    };
    let datasets = grp
        .datasets()
        .map_err(|e| IoError::Parse(format!("Failed to list VTKHDF '{group_name}': {e}")))?;

    let mut field = FieldData::new();
    for ds in datasets {
        let slice = resolve_image_step_slice(&ds, slab_z, expected_tuples, steps, offsets_group_name)?;
        let total_elems = selection_element_count(ds.shape(), slice.as_ref())?;
        let n_comp = infer_num_components(total_elems, expected_tuples, ds.shape(), true);
        let arr = read_dataset_as_data_array(&ds, ds.name(), n_comp, slice.as_ref())?;
        field.add_array(arr);
    }
    Ok(field)
}

fn resolve_image_step_slice(
    ds: &Dataset,
    slab_z: usize,
    expected_tuples: usize,
    steps: Option<&StepsInfo>,
    offsets_group_name: &str,
) -> Result<Option<SliceInfo>> {
    let Some(st) = steps else {
        return Ok(None);
    };
    let shape = ds.shape();
    if shape.is_empty() {
        return Ok(None);
    }

    if let Ok(off_grp) = st.group.group(offsets_group_name)
        && let Ok(off_ds) = off_grp.dataset(ds.name())
    {
        let offsets = read_dataset_i64(&off_ds, None)?;
        if st.step_idx < offsets.len() {
            let start = u64::try_from(offsets[st.step_idx].max(0)).unwrap_or(0);
            // Case 1: Leading time axis [NSteps, nz, ny, nx, (nc)]
            if shape.len() >= 4
                && shape[0] == st.n_steps as u64
                && start == st.step_idx as u64
                && (shape[1] as usize == slab_z || shape[1] as usize == slab_z + 1)
            {
                return index_axis0(shape, start).map(Some);
            }
            // Case 2: Concatenated along axis 0
            let end = if st.step_idx + 1 < offsets.len() {
                u64::try_from(offsets[st.step_idx + 1].max(0)).unwrap_or(shape[0])
            } else {
                let total = shape_element_count(shape)?;
                if total == expected_tuples * (total / expected_tuples.max(1)).max(1)
                    && shape[0] == slab_z as u64
                {
                    shape[0]
                } else {
                    (start.saturating_add(slab_z as u64)).min(shape[0])
                }
            };
            return slice_axis0(shape, start, end.min(shape[0])).map(Some);
        }
    }

    // Fallback: if shape[0] == NSteps and trailing dims match grid
    if shape.len() >= 4 && shape[0] == st.n_steps as u64 && shape[1] as usize == slab_z {
        return index_axis0(shape, st.step_idx as u64).map(Some);
    }

    Ok(None)
}

fn read_poly_data(group: &Group, target_time: Option<f64>) -> Result<PolyData> {
    let steps = resolve_steps(group, target_time)?;
    let (part_off, n_parts) = resolve_part_range(group, steps.as_ref())?;

    let (pt_start, total_pts, part_pt_counts) = if let Ok(npts_ds) = group.dataset("NumberOfPoints") {
        let counts = read_dataset_i64(&npts_ds, None)?;
        let part_slice = slice_part_counts(&counts, part_off, n_parts)?;
        let total: usize = part_slice.iter().sum();
        let pt_start = if let Some(st) = &steps {
            read_step_offset_scalar(&st.group, "PointOffsets", st.step_idx)?
                .unwrap_or_else(|| counts[..part_off.min(counts.len())].iter().map(|&v| v.max(0) as u64).sum())
        } else {
            0
        };
        (pt_start, total, part_slice)
    } else {
        let pts_ds = group
            .dataset("Points")
            .map_err(|_| IoError::Parse("VTKHDF PolyData missing 'Points' dataset".into()))?;
        let n = usize::try_from(pts_ds.shape().first().copied().unwrap_or(0)).unwrap_or(0);
        (0, n, vec![n])
    };
    let pts_ds = group
        .dataset("Points")
        .map_err(|_| IoError::Parse("VTKHDF PolyData missing 'Points' dataset".into()))?;
    let points = read_points_slice(&pts_ds, pt_start, total_pts)?;

    let mut poly = PolyData::new();
    poly.points = points;

    let mut total_cell_start_for_step: u64 = 0;

    for (prim_idx, prim_name) in ["Vertices", "Lines", "Polygons", "Strips"]
        .into_iter()
        .enumerate()
    {
        let Ok(prim_grp) = group.group(prim_name) else {
            continue;
        };
        let Some(prim_data) = read_poly_primitive(
            &prim_grp,
            prim_idx,
            steps.as_ref(),
            part_off,
            n_parts,
            &part_pt_counts,
            total_pts,
        )?
        else {
            continue;
        };
        total_cell_start_for_step = total_cell_start_for_step.saturating_add(prim_data.cell_start);
        match prim_idx {
            0 => {
                for cell in prim_data.cells {
                    for v in cell {
                        poly.verts.push(v);
                    }
                }
            }
            1 => poly.lines = prim_data.cells,
            2 => poly.polys = prim_data.cells,
            3 => poly.strips = prim_data.cells,
            _ => unreachable!(),
        }
    }

    poly.point_data = read_mesh_attr_group(
        group,
        "PointData",
        total_pts,
        pt_start,
        steps.as_ref(),
        "PointDataOffsets",
    )?;
    poly.cell_data = read_mesh_attr_group(
        group,
        "CellData",
        poly.num_cells(),
        total_cell_start_for_step,
        steps.as_ref(),
        "CellDataOffsets",
    )?;
    poly.field_data = read_field_attr_group(group, "FieldData", steps.as_ref(), "FieldDataOffsets")?;

    Ok(poly)
}

struct PrimResult {
    cells: Vec<Vec<u32>>,
    cell_start: u64,
}

fn read_poly_primitive(
    prim_grp: &Group,
    prim_idx: usize,
    steps: Option<&StepsInfo>,
    part_off: usize,
    n_parts: usize,
    part_pt_counts: &[usize],
    total_pts: usize,
) -> Result<Option<PrimResult>> {
    let Ok(ncells_ds) = prim_grp.dataset("NumberOfCells") else {
        return Ok(None);
    };
    let all_cell_counts = read_dataset_i64(&ncells_ds, None)?;
    let part_cell_counts = slice_part_counts(&all_cell_counts, part_off, n_parts)?;
    let total_cells: usize = part_cell_counts.iter().sum();

    let cell_start = if let Some(st) = steps {
        read_step_offset_col(&st.group, "CellOffsets", st.step_idx, prim_idx)?
            .unwrap_or_else(|| all_cell_counts[..part_off.min(all_cell_counts.len())].iter().map(|&v| v.max(0) as u64).sum())
    } else {
        0
    };

    if total_cells == 0 {
        return Ok(Some(PrimResult {
            cells: Vec::new(),
            cell_start,
        }));
    }

    let conn_ds = prim_grp
        .dataset("Connectivity")
        .map_err(|_| IoError::Parse(format!("VTKHDF '{}' missing 'Connectivity'", prim_grp.name())))?;
    let off_ds = prim_grp
        .dataset("Offsets")
        .map_err(|_| IoError::Parse(format!("VTKHDF '{}' missing 'Offsets'", prim_grp.name())))?;

    let all_conn_counts = if let Ok(nconn_ds) = prim_grp.dataset("NumberOfConnectivityIds") {
        Some(read_dataset_i64(&nconn_ds, None)?)
    } else {
        None
    };

    let (conn_start, total_conn_ids) = if let Some(counts) = &all_conn_counts {
        let part_conn = slice_part_counts(counts, part_off, n_parts)?;
        let total: usize = part_conn.iter().sum();
        let start = if let Some(st) = steps {
            read_step_offset_col(&st.group, "ConnectivityIdOffsets", st.step_idx, prim_idx)?
                .unwrap_or_else(|| counts[..part_off.min(counts.len())].iter().map(|&v| v.max(0) as u64).sum())
        } else {
            0
        };
        (start, total)
    } else {
        let start = if let Some(st) = steps {
            read_step_offset_col(&st.group, "ConnectivityIdOffsets", st.step_idx, prim_idx)?.unwrap_or(0)
        } else {
            0
        };
        let avail = conn_ds.shape().first().copied().unwrap_or(0).saturating_sub(start);
        (start, usize::try_from(avail).unwrap_or(0))
    };

    let off_len_total = off_ds.shape().first().copied().unwrap_or(0);
    let total_all_cells: u64 = all_cell_counts.iter().map(|&v| v.max(0) as u64).sum();
    let total_all_parts = all_cell_counts.len() as u64;
    let has_per_part_terminator = off_len_total >= total_all_cells.saturating_add(total_all_parts);

    let off_start = if has_per_part_terminator {
        cell_start.saturating_add(part_off as u64)
    } else {
        cell_start
    };
    let off_count = if has_per_part_terminator {
        total_cells + n_parts
    } else {
        total_cells + 1
    };

    let off_slice = slice_axis0(
        off_ds.shape(),
        off_start,
        (off_start + off_count as u64).min(off_len_total),
    )?;
    let offsets = read_dataset_i64(&off_ds, Some(&off_slice))?;

    let conn_shape_len = conn_ds.shape().first().copied().unwrap_or(0);
    let conn_end = (conn_start + total_conn_ids as u64).min(conn_shape_len);
    let conn_slice = slice_axis0(conn_ds.shape(), conn_start, conn_end)?;
    let conn = read_dataset_u32(&conn_ds, Some(&conn_slice))?;

    let cap = safe_prealloc_capacity(total_cells, std::mem::size_of::<Vec<u32>>(), MAX_ALLOC_BYTES)?;
    let mut cells = Vec::with_capacity(cap);

    let mut pt_base: u32 = 0;
    let mut off_cursor: usize = 0;
    let mut conn_cursor: usize = 0;

    for (p, &p_cells) in part_cell_counts.iter().enumerate().take(n_parts) {
        let p_pts = part_pt_counts.get(p).copied().unwrap_or(total_pts);
        if p_cells > 0 {
            if off_cursor + p_cells >= offsets.len() {
                return Err(IoError::InvalidData(format!(
                    "VTKHDF '{}' Offsets array too short for part {p}",
                    prim_grp.name()
                )));
            }
            let base_off = offsets[off_cursor];
            for i in 0..p_cells {
                let s = offsets[off_cursor + i] - base_off;
                let e = offsets[off_cursor + i + 1] - base_off;
                if s < 0 || e < s {
                    return Err(IoError::InvalidData(format!(
                        "VTKHDF '{}' contains invalid negative/decreasing cell offset",
                        prim_grp.name()
                    )));
                }
                let cs = conn_cursor + s as usize;
                let ce = conn_cursor + e as usize;
                if ce > conn.len() {
                    return Err(IoError::InvalidData(format!(
                        "VTKHDF '{}' cell connectivity range [{cs}..{ce}] exceeds buffer ({})",
                        prim_grp.name(),
                        conn.len()
                    )));
                }
                let mut cell_verts = Vec::with_capacity(ce - cs);
                for &v in &conn[cs..ce] {
                    if (v as usize) >= p_pts || (pt_base as usize + v as usize) >= total_pts {
                        return Err(IoError::InvalidData(format!(
                            "VTKHDF '{}' point index {v} out of range ({p_pts} points)",
                            prim_grp.name()
                        )));
                    }
                    cell_verts.push(pt_base + v);
                }
                cells.push(cell_verts);
            }
            let last_rel = (offsets[off_cursor + p_cells] - base_off).max(0) as usize;
            conn_cursor += last_rel;
            off_cursor += if has_per_part_terminator { p_cells + 1 } else { p_cells };
        } else if has_per_part_terminator && off_cursor < offsets.len() {
            off_cursor += 1;
        }
        pt_base = pt_base.saturating_add(p_pts as u32);
    }

    Ok(Some(PrimResult { cells, cell_start }))
}

fn read_unstructured_grid(group: &Group, target_time: Option<f64>) -> Result<UnstructuredGrid> {
    let steps = resolve_steps(group, target_time)?;
    let (part_off, n_parts) = resolve_part_range(group, steps.as_ref())?;

    let (pt_start, total_pts, part_pt_counts) = if let Ok(npts_ds) = group.dataset("NumberOfPoints") {
        let counts = read_dataset_i64(&npts_ds, None)?;
        let part_slice = slice_part_counts(&counts, part_off, n_parts)?;
        let total: usize = part_slice.iter().sum();
        let start = if let Some(st) = &steps {
            read_step_offset_scalar(&st.group, "PointOffsets", st.step_idx)?
                .unwrap_or_else(|| counts[..part_off.min(counts.len())].iter().map(|&v| v.max(0) as u64).sum())
        } else {
            0
        };
        (start, total, part_slice)
    } else {
        let pts_ds = group
            .dataset("Points")
            .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Points'".into()))?;
        let n = usize::try_from(pts_ds.shape().first().copied().unwrap_or(0)).unwrap_or(0);
        (0, n, vec![n])
    };

    let pts_ds = group
        .dataset("Points")
        .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Points'".into()))?;
    let points = read_points_slice(&pts_ds, pt_start, total_pts)?;

    let mut ug = UnstructuredGrid::new();
    ug.points = points;

    let (cell_start, total_cells, part_cell_counts, all_cell_counts) =
        if let Ok(ncells_ds) = group.dataset("NumberOfCells") {
            let counts = read_dataset_i64(&ncells_ds, None)?;
            let part_slice = slice_part_counts(&counts, part_off, n_parts)?;
            let total: usize = part_slice.iter().sum();
            let start = if let Some(st) = &steps {
                read_step_offset_scalar(&st.group, "CellOffsets", st.step_idx)?
                    .unwrap_or_else(|| counts[..part_off.min(counts.len())].iter().map(|&v| v.max(0) as u64).sum())
            } else {
                0
            };
            (start, total, part_slice, counts)
        } else {
            let types_ds = group
                .dataset("Types")
                .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Types'".into()))?;
            let n = usize::try_from(types_ds.shape().first().copied().unwrap_or(0)).unwrap_or(0);
            (0, n, vec![n], vec![n as i64])
        };

    if total_cells > 0 {
        let types_ds = group
            .dataset("Types")
            .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Types'".into()))?;
        let off_ds = group
            .dataset("Offsets")
            .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Offsets'".into()))?;
        let conn_ds = group
            .dataset("Connectivity")
            .map_err(|_| IoError::Parse("VTKHDF UnstructuredGrid missing 'Connectivity'".into()))?;

        let types_len = types_ds.shape().first().copied().unwrap_or(0);
        let types_slice = slice_axis0(
            types_ds.shape(),
            cell_start,
            (cell_start + total_cells as u64).min(types_len),
        )?;
        let cell_type_ids = read_dataset_u8(&types_ds, Some(&types_slice))?;

        let all_conn_counts = if let Ok(nconn_ds) = group.dataset("NumberOfConnectivityIds") {
            Some(read_dataset_i64(&nconn_ds, None)?)
        } else {
            None
        };
        let (conn_start, total_conn_ids) = if let Some(counts) = &all_conn_counts {
            let part_conn = slice_part_counts(counts, part_off, n_parts)?;
            let total: usize = part_conn.iter().sum();
            let start = if let Some(st) = &steps {
                read_step_offset_scalar(&st.group, "ConnectivityIdOffsets", st.step_idx)?
                    .unwrap_or_else(|| counts[..part_off.min(counts.len())].iter().map(|&v| v.max(0) as u64).sum())
            } else {
                0
            };
            (start, total)
        } else {
            let start = if let Some(st) = &steps {
                read_step_offset_scalar(&st.group, "ConnectivityIdOffsets", st.step_idx)?.unwrap_or(0)
            } else {
                0
            };
            let avail = conn_ds.shape().first().copied().unwrap_or(0).saturating_sub(start);
            (start, usize::try_from(avail).unwrap_or(0))
        };

        let off_len_total = off_ds.shape().first().copied().unwrap_or(0);
        let total_all_cells: u64 = all_cell_counts.iter().map(|&v| v.max(0) as u64).sum();
        let total_all_parts = all_cell_counts.len() as u64;
        let has_per_part_terminator = off_len_total >= total_all_cells.saturating_add(total_all_parts);

        let off_start = if has_per_part_terminator {
            cell_start.saturating_add(part_off as u64)
        } else {
            cell_start
        };
        let off_count = if has_per_part_terminator {
            total_cells + n_parts
        } else {
            total_cells + 1
        };
        let off_slice = slice_axis0(
            off_ds.shape(),
            off_start,
            (off_start + off_count as u64).min(off_len_total),
        )?;
        let offsets = read_dataset_i64(&off_ds, Some(&off_slice))?;

        let conn_len_total = conn_ds.shape().first().copied().unwrap_or(0);
        let conn_slice = slice_axis0(
            conn_ds.shape(),
            conn_start,
            (conn_start + total_conn_ids as u64).min(conn_len_total),
        )?;
        let conn = read_dataset_u32(&conn_ds, Some(&conn_slice))?;

        let cell_cap = safe_prealloc_capacity(total_cells, std::mem::size_of::<CellType>(), MAX_ALLOC_BYTES)?;
        let conn_cap = safe_prealloc_capacity(conn.len(), std::mem::size_of::<u32>(), MAX_ALLOC_BYTES)?;
        ug.cell_types.reserve(cell_cap);
        ug.offsets.reserve(cell_cap);
        ug.connectivity.reserve(conn_cap);

        let mut pt_base: u32 = 0;
        let mut off_cursor: usize = 0;
        let mut conn_cursor: usize = 0;
        let mut type_cursor: usize = 0;

        for (p, &p_cells) in part_cell_counts.iter().enumerate().take(n_parts) {
            let p_pts = part_pt_counts.get(p).copied().unwrap_or(total_pts);
            if p_cells > 0 {
                if off_cursor + p_cells >= offsets.len() {
                    return Err(IoError::InvalidData(format!(
                        "VTKHDF UnstructuredGrid Offsets too short for part {p}"
                    )));
                }
                if type_cursor + p_cells > cell_type_ids.len() {
                    return Err(IoError::InvalidData(format!(
                        "VTKHDF UnstructuredGrid Types too short for part {p}"
                    )));
                }
                let base_off = offsets[off_cursor];
                for i in 0..p_cells {
                    let tid = cell_type_ids[type_cursor + i];
                    let ctype = CellType::from_vtk_id(tid).ok_or_else(|| {
                        IoError::UnsupportedFormat(format!(
                            "Unsupported VTKHDF cell type ID {tid}"
                        ))
                    })?;
                    let s = offsets[off_cursor + i] - base_off;
                    let e = offsets[off_cursor + i + 1] - base_off;
                    if s < 0 || e < s {
                        return Err(IoError::InvalidData(
                            "VTKHDF UnstructuredGrid invalid negative/decreasing cell offset".into(),
                        ));
                    }
                    let cs = conn_cursor + s as usize;
                    let ce = conn_cursor + e as usize;
                    if ce > conn.len() {
                        return Err(IoError::InvalidData(format!(
                            "VTKHDF UnstructuredGrid connectivity range [{cs}..{ce}] exceeds buffer ({})",
                            conn.len()
                        )));
                    }
                    let mut verts = Vec::with_capacity(ce - cs);
                    for &v in &conn[cs..ce] {
                        if (v as usize) >= p_pts || (pt_base as usize + v as usize) >= total_pts {
                            return Err(IoError::InvalidData(format!(
                                "VTKHDF UnstructuredGrid point index {v} out of range ({p_pts} points)"
                            )));
                        }
                        verts.push(pt_base + v);
                    }
                    ug.push_cell(ctype, &verts);
                }
                let last_rel = (offsets[off_cursor + p_cells] - base_off).max(0) as usize;
                conn_cursor += last_rel;
                type_cursor += p_cells;
                off_cursor += if has_per_part_terminator { p_cells + 1 } else { p_cells };
            } else if has_per_part_terminator && off_cursor < offsets.len() {
                off_cursor += 1;
            }
            pt_base = pt_base.saturating_add(p_pts as u32);
        }
    }

    ug.point_data = read_mesh_attr_group(
        group,
        "PointData",
        total_pts,
        pt_start,
        steps.as_ref(),
        "PointDataOffsets",
    )?;
    ug.cell_data = read_mesh_attr_group(
        group,
        "CellData",
        ug.num_cells(),
        cell_start,
        steps.as_ref(),
        "CellDataOffsets",
    )?;
    ug.field_data = read_field_attr_group(group, "FieldData", steps.as_ref(), "FieldDataOffsets")?;

    Ok(ug)
}

fn read_composite(
    group: &Group,
    target_time: Option<f64>,
    depth: usize,
) -> Result<MultiBlockDataSet> {
    let mut mb = MultiBlockDataSet::new();
    let children = group
        .groups()
        .map_err(|e| IoError::Parse(format!("Failed to list VTKHDF composite children: {e}")))?;

    let mut direct_blocks = Vec::new();
    let mut assembly_group = None;

    for child in children {
        let name = child.name();
        if name == "Steps" {
            continue;
        }
        if name == "Assembly" {
            assembly_group = Some(child);
            continue;
        }
        direct_blocks.push(child);
    }

    if !direct_blocks.is_empty() {
        for child in direct_blocks {
            let block_name = child.name().to_string();
            if group_attr_string(&child, "Type").is_some() {
                let ds = read_vtkhdf_group(&child, target_time, depth + 1)?;
                mb.add_block(block_name, ds);
            } else {
                // PartitionedDataSet child groups without explicit Type attribute
                let sub = read_composite_or_partitions(&child, target_time, depth + 1)?;
                mb.add_block(block_name, sub);
            }
        }
    } else if let Some(asm) = assembly_group {
        let ds = read_composite_or_partitions(&asm, target_time, depth + 1)?;
        if let DataSet::MultiBlock(inner) = ds {
            return Ok(inner);
        }
        mb.add_block("Assembly", ds);
    }

    Ok(mb)
}

fn read_composite_or_partitions(
    group: &Group,
    target_time: Option<f64>,
    depth: usize,
) -> Result<DataSet> {
    if depth > MAX_COMPOSITE_DEPTH {
        return Err(IoError::InvalidData(format!(
            "VTKHDF composite hierarchy exceeded maximum depth of {MAX_COMPOSITE_DEPTH}"
        )));
    }
    if group_attr_string(group, "Type").is_some() {
        return read_vtkhdf_group(group, target_time, depth);
    }
    let mut mb = MultiBlockDataSet::new();
    for child in group
        .groups()
        .map_err(|e| IoError::Parse(format!("Failed to list VTKHDF group '{}': {e}", group.name())))?
    {
        let name = child.name().to_string();
        if name == "Steps" {
            continue;
        }
        let ds = read_composite_or_partitions(&child, target_time, depth + 1)?;
        mb.add_block(name, ds);
    }
    if mb.blocks.len() == 1 {
        Ok(mb.blocks.remove(0).1)
    } else {
        Ok(DataSet::MultiBlock(mb))
    }
}

fn resolve_part_range(group: &Group, steps: Option<&StepsInfo>) -> Result<(usize, usize)> {
    let Some(st) = steps else {
        let n_parts = if let Ok(npts_ds) = group.dataset("NumberOfPoints") {
            let counts = read_dataset_i64(&npts_ds, None)?;
            counts.len().max(1)
        } else {
            1
        };
        return Ok((0, n_parts));
    };

    let part_off = read_step_offset_scalar(&st.group, "PartOffsets", st.step_idx)?
        .map(|v| v as usize)
        .unwrap_or(st.step_idx);

    let n_parts = if let Ok(np_ds) = st.group.dataset("NumberOfParts") {
        let vals = read_dataset_i64(&np_ds, None)?;
        vals.get(st.step_idx).copied().unwrap_or(1).max(0) as usize
    } else if let Some(next_off) = read_step_offset_scalar(&st.group, "PartOffsets", st.step_idx + 1)? {
        (next_off as usize).saturating_sub(part_off).max(1)
    } else {
        1
    };

    Ok((part_off, n_parts))
}

fn slice_part_counts(counts: &[i64], part_off: usize, n_parts: usize) -> Result<Vec<usize>> {
    if counts.is_empty() {
        return Ok(vec![0; n_parts]);
    }
    if part_off + n_parts > counts.len() {
        return Err(IoError::InvalidData(format!(
            "VTKHDF part range [{part_off}..{}] exceeds array length {}",
            part_off + n_parts,
            counts.len()
        )));
    }
    let mut out = Vec::with_capacity(n_parts);
    for &c in &counts[part_off..part_off + n_parts] {
        let u = usize::try_from(c).map_err(|_| {
            IoError::InvalidData(format!("VTKHDF negative element count {c}"))
        })?;
        out.push(u);
    }
    Ok(out)
}

fn read_step_offset_scalar(steps_grp: &Group, ds_name: &str, step_idx: usize) -> Result<Option<u64>> {
    let Ok(ds) = steps_grp.dataset(ds_name) else {
        return Ok(None);
    };
    let vals = read_dataset_i64(&ds, None)?;
    Ok(vals.get(step_idx).map(|&v| v.max(0) as u64))
}

fn read_step_offset_col(
    steps_grp: &Group,
    ds_name: &str,
    step_idx: usize,
    col: usize,
) -> Result<Option<u64>> {
    let Ok(ds) = steps_grp.dataset(ds_name) else {
        return Ok(None);
    };
    let shape = ds.shape();
    let vals = read_dataset_i64(&ds, None)?;
    if shape.len() == 2 {
        let cols = shape[1] as usize;
        let idx = step_idx * cols + col;
        Ok(vals.get(idx).map(|&v| v.max(0) as u64))
    } else {
        Ok(vals.get(step_idx).map(|&v| v.max(0) as u64))
    }
}

fn read_points_slice(ds: &Dataset, pt_start: u64, n_pts: usize) -> Result<Vec<[f32; 3]>> {
    if n_pts == 0 {
        return Ok(Vec::new());
    }
    let shape = ds.shape();
    let (flat, dims_per_pt) = match shape.len() {
        2 => {
            let dims = usize::try_from(shape[1]).unwrap_or(3);
            let end = (pt_start + n_pts as u64).min(shape[0]);
            let slice = slice_axis0(shape, pt_start, end)?;
            (read_dataset_f32(ds, Some(&slice))?, dims)
        }
        1 => {
            let total = shape[0];
            let dims = 3usize;
            let start = pt_start.saturating_mul(3);
            let end = start.saturating_add((n_pts as u64).saturating_mul(3)).min(total);
            let slice = slice_axis0(shape, start, end)?;
            (read_dataset_f32(ds, Some(&slice))?, dims)
        }
        other => {
            return Err(IoError::InvalidData(format!(
                "VTKHDF Points dataset has unsupported rank {other}"
            )));
        }
    };
    pack_points_f32(&flat, dims_per_pt)
}

fn read_mesh_attr_group(
    parent: &Group,
    group_name: &str,
    expected_tuples: usize,
    default_start: u64,
    steps: Option<&StepsInfo>,
    offsets_group_name: &str,
) -> Result<FieldData> {
    let Ok(grp) = parent.group(group_name) else {
        return Ok(FieldData::new());
    };
    let datasets = grp
        .datasets()
        .map_err(|e| IoError::Parse(format!("Failed to list VTKHDF '{group_name}': {e}")))?;

    let mut field = FieldData::new();
    for ds in datasets {
        let shape = ds.shape();
        if shape.is_empty() {
            continue;
        }
        let slice = if let Some(st) = steps {
            let mut explicit_start: Option<(u64, Option<u64>)> = None;
            if let Ok(off_grp) = st.group.group(offsets_group_name)
                && let Ok(off_ds) = off_grp.dataset(ds.name())
            {
                let offs = read_dataset_i64(&off_ds, None)?;
                if st.step_idx < offs.len() {
                    let s = offs[st.step_idx].max(0) as u64;
                    let e = offs.get(st.step_idx + 1).map(|&v| v.max(0) as u64);
                    explicit_start = Some((s, e));
                }
            }
            if let Some((s, maybe_e)) = explicit_start {
                let end = maybe_e
                    .unwrap_or_else(|| s.saturating_add(expected_tuples as u64))
                    .min(shape[0]);
                Some(slice_axis0(shape, s, end)?)
            } else if shape[0] as usize == expected_tuples {
                // Static array in a temporal file
                None
            } else {
                let end = default_start
                    .saturating_add(expected_tuples as u64)
                    .min(shape[0]);
                Some(slice_axis0(shape, default_start.min(shape[0]), end)?)
            }
        } else {
            None
        };

        let total_elems = selection_element_count(shape, slice.as_ref())?;
        let n_comp = infer_num_components(total_elems, expected_tuples, shape, false);
        let arr = read_dataset_as_data_array(&ds, ds.name(), n_comp, slice.as_ref())?;
        field.add_array(arr);
    }
    Ok(field)
}

fn read_field_attr_group(
    parent: &Group,
    group_name: &str,
    steps: Option<&StepsInfo>,
    offsets_group_name: &str,
) -> Result<FieldData> {
    let Ok(grp) = parent.group(group_name) else {
        return Ok(FieldData::new());
    };
    let datasets = grp
        .datasets()
        .map_err(|e| IoError::Parse(format!("Failed to list VTKHDF '{group_name}': {e}")))?;

    let mut field = FieldData::new();
    for ds in datasets {
        let shape = ds.shape();
        if !matches!(
            ds.dtype(),
            hdf5_reader::Datatype::FloatingPoint { .. } | hdf5_reader::Datatype::FixedPoint { .. }
        ) {
            continue;
        }
        let slice = if let Some(st) = steps
            && !shape.is_empty()
            && let Ok(off_grp) = st.group.group(offsets_group_name)
            && let Ok(off_ds) = off_grp.dataset(ds.name())
        {
            let offs = read_dataset_i64(&off_ds, None)?;
            if st.step_idx < offs.len() {
                let s = offs[st.step_idx].max(0) as u64;
                let e = offs
                    .get(st.step_idx + 1)
                    .map(|&v| v.max(0) as u64)
                    .unwrap_or(shape[0])
                    .min(shape[0]);
                Some(slice_axis0(shape, s.min(shape[0]), e)?)
            } else {
                None
            }
        } else {
            None
        };
        let n_comp = if shape.len() >= 2 {
            usize::try_from(*shape.last().unwrap_or(&1)).unwrap_or(1).max(1)
        } else {
            1
        };
        if let Ok(arr) = read_dataset_as_data_array(&ds, ds.name(), n_comp, slice.as_ref()) {
            field.add_array(arr);
        }
    }
    Ok(field)
}

fn infer_num_components(
    total_elems: usize,
    expected_tuples: usize,
    shape: &[u64],
    is_image: bool,
) -> usize {
    if total_elems == 0 {
        return 1;
    }
    if expected_tuples > 0 && total_elems.is_multiple_of(expected_tuples) {
        return (total_elems / expected_tuples).max(1);
    }
    if is_image && shape.len() >= 4 {
        return usize::try_from(*shape.last().unwrap_or(&1)).unwrap_or(1).max(1);
    }
    if !is_image && shape.len() >= 2 {
        return usize::try_from(*shape.last().unwrap_or(&1)).unwrap_or(1).max(1);
    }
    1
}
