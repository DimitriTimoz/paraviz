//! Patch-based CFD HDF5 (`.h5`) reader.
//!
//! Reads HDF5 files structured with `/Patch_*` groups containing `XYZ` (points) and
//! `ELEM2NODE` (XDMF-style mixed element connectivity) plus node-centered (`NNODE`)
//! and cell-centered (`NFACES` / `NELEM`) field datasets.

use crate::error::{IoError, Result};
use crate::hdf5_common::{open_hdf5, read_dataset_as_data_array, read_dataset_f32, read_dataset_i64};
use hdf5_reader::Hdf5File;
use paraviz_core::{CellType, DataSet, MultiBlockDataSet, UnstructuredGrid};
use std::path::Path;

/// Returns `true` if `file` has a `/Patch_*` group containing `XYZ` and `ELEM2NODE`.
pub fn is_patch_hdf5(file: &Hdf5File) -> bool {
    let Ok(root) = file.root_group() else {
        return false;
    };
    let Ok(groups) = root.groups() else {
        return false;
    };
    for g in groups {
        if g.dataset("XYZ").is_ok() && g.dataset("ELEM2NODE").is_ok() {
            return true;
        }
    }
    false
}

/// Reads a patch-based `.h5` file into a [`DataSet`].
pub fn read_patch_hdf5(path: impl AsRef<Path>) -> Result<DataSet> {
    let file = open_hdf5(path.as_ref())?;
    let root = file
        .root_group()
        .map_err(|e| IoError::Parse(format!("Failed to open HDF5 root: {e}")))?;
    let mut groups = root
        .groups()
        .map_err(|e| IoError::Parse(format!("Failed to list HDF5 groups: {e}")))?;
    groups.sort_by(|a, b| a.name().cmp(b.name()));

    let mut blocks: Vec<(String, UnstructuredGrid)> = Vec::new();

    for grp in groups {
        let gname = grp.name().to_string();
        let (Ok(xyz_ds), Ok(conn_ds)) = (grp.dataset("XYZ"), grp.dataset("ELEM2NODE")) else {
            continue;
        };

        let xyz = read_dataset_f32(&xyz_ds, None)?;
        if xyz.len() % 3 != 0 {
            return Err(IoError::InvalidData(format!(
                "Patch HDF5 group '{gname}/XYZ' length {} is not divisible by 3",
                xyz.len()
            )));
        }
        let n_pts = xyz.len() / 3;
        let points: Vec<[f32; 3]> = xyz
            .as_chunks::<3>().0.iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let raw_conn = read_dataset_i64(&conn_ds, None)?;
        let mut grid = UnstructuredGrid::new();
        grid.points = points;
        decode_xdmf_mixed_into_grid(&raw_conn, n_pts, &mut grid)?;
        let n_cells = grid.num_cells();

        if let Ok(datasets) = grp.datasets() {
            for ds in datasets {
                let dname = ds.name();
                if matches!(
                    dname,
                    "XYZ" | "ELEM2NODE" | "ELEM2NODE_SIZE" | "NFACES" | "NNODE" | "NELEM"
                ) {
                    continue;
                }
                let shape = ds.shape();
                let n_rows = shape.first().copied().unwrap_or(0) as usize;
                let n_comp = if shape.len() >= 2 {
                    shape[1] as usize
                } else {
                    1
                };
                if n_rows == n_pts && n_pts > 0 {
                    if let Ok(arr) = read_dataset_as_data_array(&ds, dname, n_comp, None) {
                        grid.point_data.add_array(arr);
                    }
                } else if n_rows == n_cells && n_cells > 0
                    && let Ok(arr) = read_dataset_as_data_array(&ds, dname, n_comp, None) {
                        grid.cell_data.add_array(arr);
                    }
            }
        }

        blocks.push((gname, grid));
    }

    if blocks.is_empty() {
        return Err(IoError::UnsupportedFormat(
            "HDF5 file contains no Patch groups with XYZ and ELEM2NODE".into(),
        ));
    }
    if blocks.len() == 1 {
        Ok(DataSet::UnstructuredGrid(blocks.into_iter().next().unwrap().1))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, g) in blocks {
            mb.add_block(name, DataSet::UnstructuredGrid(g));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn decode_xdmf_mixed_into_grid(
    raw: &[i64],
    n_pts: usize,
    grid: &mut UnstructuredGrid,
) -> Result<()> {
    grid.connectivity.reserve(raw.len());
    grid.cell_types.reserve(raw.len() / 4);
    grid.offsets.reserve(raw.len() / 4);
    let mut i = 0usize;
    while i < raw.len() {
        let code = raw[i];
        i += 1;
        let (ct, count) = match code {
            1 => (CellType::Vertex, 1usize),
            2 => (CellType::Line, 2usize),
            4 => (CellType::Triangle, 3usize),
            5 => (CellType::Quad, 4usize),
            6 => (CellType::Tetra, 4usize),
            7 => (CellType::Pyramid, 5usize),
            8 => (CellType::Wedge, 6usize),
            9 => (CellType::Hexahedron, 8usize),
            _ => {
                return Err(IoError::UnsupportedFormat(format!(
                    "Unsupported XDMF/Patch element type code {code}"
                )));
            }
        };
        if i + count > raw.len() {
            return Err(IoError::Parse(
                "Truncated ELEM2NODE connectivity in Patch HDF5".into(),
            ));
        }
        for &v in &raw[i..i + count] {
            let idx = u32::try_from(v).map_err(|_| {
                IoError::InvalidData(format!("Negative node index {v} in ELEM2NODE"))
            })?;
            if (idx as usize) >= n_pts {
                return Err(IoError::InvalidData(format!(
                    "ELEM2NODE index {idx} out of bounds ({n_pts})"
                )));
            }
            grid.connectivity.push(idx);
        }
        grid.offsets.push(grid.connectivity.len() as u32);
        grid.cell_types.push(ct);
        i += count;
    }
    Ok(())
}
