//! Exodus II (`.exo`, `.e`, `.ex2`) scientific finite-element reader.
//!
//! Supports both:
//! - NetCDF Classic (`CDF\x01`) and 64-bit Offset (`CDF\x02`) via a pure-Rust bounds-checked parser
//! - NetCDF-4 / HDF5 (`\x89HDF\r\n\x1a\n`) via `hdf5-reader`

use crate::error::{IoError, Result};
use crate::hdf5_common::{
    attr_as_string, decode_nul_terminated_bytes, open_hdf5, open_hdf5_bytes,
    read_dataset_as_data_array, read_dataset_f32, read_dataset_f64, read_dataset_i64,
    read_dataset_strings,
};
use crate::limits::{check_alloc_bytes, safe_prealloc_capacity, MAX_ALLOC_BYTES};
use hdf5_reader::{Hdf5File, SliceInfo, SliceInfoElem};
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, MultiBlockDataSet, UnstructuredGrid,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const CDF1_MAGIC: &[u8; 4] = b"CDF\x01";
const CDF2_MAGIC: &[u8; 4] = b"CDF\x02";

/// Reads an Exodus II file (`.exo`, `.e`, `.ex2`) into a [`DataSet`] (first time step if temporal).
pub fn read_exodus(path: impl AsRef<Path>) -> Result<DataSet> {
    read_exodus_impl(path.as_ref(), None)
}

/// Reads the time step closest to `time` from an Exodus II file.
pub fn read_exodus_at_time(path: impl AsRef<Path>, time: f64) -> Result<DataSet> {
    read_exodus_impl(path.as_ref(), Some(time))
}

/// Inspects an Exodus II file and returns its `time_whole` time step values (empty if static).
pub fn inspect_exodus_time_steps(path: impl AsRef<Path>) -> Result<Vec<f64>> {
    let path = path.as_ref();
    let magic = read_prefix(path, 8)?;
    if magic.starts_with(HDF5_MAGIC) {
        let file = open_hdf5(path)?;
        let backend = Hdf5ExodusBackend::new(file)?;
        read_time_whole(&backend)
    } else if magic.starts_with(CDF1_MAGIC) || magic.starts_with(CDF2_MAGIC) {
        let f = File::open(path)?;
        let backend = NetCdfClassicBackend::open(BufReader::new(f))?;
        read_time_whole(&backend)
    } else {
        Err(IoError::UnsupportedFormat(format!(
            "Unrecognized Exodus II file header in '{}'",
            path.display()
        )))
    }
}

/// Parses Exodus II bytes (NetCDF Classic, 64-bit Offset, or HDF5) into a [`DataSet`].
pub fn read_exodus_bytes(bytes: &[u8]) -> Result<DataSet> {
    read_exodus_bytes_at_time(bytes, None)
}

/// Parses Exodus II bytes at the time step closest to `time`.
pub fn read_exodus_bytes_at_time(bytes: &[u8], target_time: Option<f64>) -> Result<DataSet> {
    if bytes.starts_with(HDF5_MAGIC) {
        let file = open_hdf5_bytes(bytes)?;
        let backend = Hdf5ExodusBackend::new(file)?;
        build_exodus_dataset(&backend, target_time)
    } else if bytes.starts_with(CDF1_MAGIC) || bytes.starts_with(CDF2_MAGIC) {
        let backend = NetCdfClassicBackend::open(Cursor::new(bytes))?;
        build_exodus_dataset(&backend, target_time)
    } else {
        Err(IoError::UnsupportedFormat(
            "Unrecognized Exodus II header bytes (expected CDF\\x01, CDF\\x02, or HDF5)".into(),
        ))
    }
}

fn read_exodus_impl(path: &Path, target_time: Option<f64>) -> Result<DataSet> {
    let magic = read_prefix(path, 8)?;
    if magic.starts_with(HDF5_MAGIC) {
        let file = open_hdf5(path)?;
        let backend = Hdf5ExodusBackend::new(file)?;
        build_exodus_dataset(&backend, target_time)
    } else if magic.starts_with(CDF1_MAGIC) || magic.starts_with(CDF2_MAGIC) {
        let f = File::open(path)?;
        let backend = NetCdfClassicBackend::open(BufReader::new(f))?;
        build_exodus_dataset(&backend, target_time)
    } else {
        Err(IoError::UnsupportedFormat(format!(
            "Unrecognized Exodus II file header in '{}'",
            path.display()
        )))
    }
}

fn read_prefix(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; max_bytes];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

// ============================================================================
// Shared Exodus II Domain Extraction
// ============================================================================

trait ExodusBackend {
    fn has_var(&self, name: &str) -> bool;
    fn var_names(&self) -> Vec<String>;
    fn var_shape(&self, name: &str) -> Option<Vec<usize>>;
    fn var_attr_string(&self, var_name: &str, attr_name: &str) -> Option<String>;
    fn read_var_f64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f64>>;
    fn read_var_f32(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f32>>;
    fn read_var_i64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<i64>>;
    fn read_var_data_array(
        &self,
        name: &str,
        out_name: &str,
        step_idx: Option<usize>,
        sub_idx: Option<usize>,
    ) -> Result<DataArray>;
    fn read_var_strings(&self, name: &str) -> Result<Vec<String>>;
}

fn read_time_whole(backend: &dyn ExodusBackend) -> Result<Vec<f64>> {
    if !backend.has_var("time_whole") {
        return Ok(Vec::new());
    }
    backend.read_var_f64("time_whole", None)
}

fn build_exodus_dataset(backend: &dyn ExodusBackend, target_time: Option<f64>) -> Result<DataSet> {
    let time_steps = read_time_whole(backend)?;
    let step_idx = if time_steps.is_empty() {
        0
    } else {
        match target_time {
            None => 0,
            Some(t) => time_steps
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - t).abs().total_cmp(&(*b - t).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0),
        }
    };

    let points = read_exodus_coordinates(backend)?;
    let num_nodes = points.len();

    let mut point_data = read_nodal_variables(backend, num_nodes, step_idx)?;
    let mut field_data = FieldData::new();
    if let Some(t) = time_steps.get(step_idx).copied() {
        field_data.add_array(DataArray::scalars_f64("TimeValue", vec![t]));
    }
    read_global_variables(backend, step_idx, &mut field_data)?;

    // Discover connect1..connectN sorted numerically
    let mut block_vars: Vec<(usize, String)> = Vec::new();
    for name in backend.var_names() {
        if let Some(suffix) = name.strip_prefix("connect")
            && let Ok(idx) = suffix.parse::<usize>()
            && idx >= 1
        {
            block_vars.push((idx, name));
        }
    }
    block_vars.sort_by_key(|(idx, _)| *idx);

    if block_vars.is_empty() {
        let mut ug = UnstructuredGrid::new();
        ug.points = points;
        ug.point_data = point_data;
        ug.field_data = field_data;
        return Ok(DataSet::UnstructuredGrid(ug));
    }

    let eb_names = if backend.has_var("eb_names") {
        backend.read_var_strings("eb_names").unwrap_or_default()
    } else {
        Vec::new()
    };

    let elem_var_names = if backend.has_var("name_elem_var") {
        backend.read_var_strings("name_elem_var").unwrap_or_default()
    } else {
        Vec::new()
    };

    // Determine num_elem_var from name_elem_var or vals_elem_var*eb* scan
    let mut num_elem_vars = elem_var_names.len();
    if num_elem_vars == 0 {
        for vname in backend.var_names() {
            if let Some(rest) = vname.strip_prefix("vals_elem_var")
                && let Some((v_str, _)) = rest.split_once("eb")
                && let Ok(v_idx) = v_str.parse::<usize>()
            {
                num_elem_vars = num_elem_vars.max(v_idx);
            }
        }
    }

    let mut blocks: Vec<(String, UnstructuredGrid)> = Vec::with_capacity(block_vars.len());

    for (pos, (blk_num, conn_name)) in block_vars.iter().enumerate() {
        let shape = backend.var_shape(conn_name).ok_or_else(|| {
            IoError::Parse(format!("Exodus II missing shape for '{conn_name}'"))
        })?;
        if shape.len() != 2 {
            return Err(IoError::InvalidData(format!(
                "Exodus II '{conn_name}' expected 2D shape [num_el, num_nod_per_el], got {shape:?}"
            )));
        }
        let num_el = shape[0];
        let nodes_per_el = shape[1];
        if nodes_per_el == 0 {
            return Err(IoError::InvalidData(format!(
                "Exodus II '{conn_name}' has 0 nodes per element"
            )));
        }

        let elem_type_str = backend.var_attr_string(conn_name, "elem_type");
        let cell_type = map_exodus_elem_type(elem_type_str.as_deref(), nodes_per_el)?;
        let conn = backend.read_var_i64(conn_name, None)?;
        if conn.len() != num_el * nodes_per_el {
            return Err(IoError::InvalidData(format!(
                "Exodus II '{conn_name}' connectivity length ({}) != {num_el} * {nodes_per_el}",
                conn.len()
            )));
        }

        let mut ug = UnstructuredGrid::new();
        ug.points = points.clone();
        ug.point_data = point_data.clone();
        ug.field_data = field_data.clone();

        let cell_cap = safe_prealloc_capacity(num_el, std::mem::size_of::<CellType>(), MAX_ALLOC_BYTES)?;
        let conn_cap = safe_prealloc_capacity(conn.len(), std::mem::size_of::<u32>(), MAX_ALLOC_BYTES)?;
        ug.cell_types.reserve(cell_cap);
        ug.offsets.reserve(cell_cap);
        ug.connectivity.reserve(conn_cap);

        let mut verts = Vec::with_capacity(nodes_per_el);
        for chunk in conn.chunks_exact(nodes_per_el) {
            verts.clear();
            for &raw_id in chunk {
                if raw_id < 1 {
                    return Err(IoError::InvalidData(format!(
                        "Exodus II '{conn_name}' contains invalid 1-based node index {raw_id}"
                    )));
                }
                let idx = (raw_id - 1) as usize;
                if idx >= num_nodes {
                    return Err(IoError::InvalidData(format!(
                        "Exodus II '{conn_name}' node index {raw_id} out of range ({num_nodes} nodes)"
                    )));
                }
                verts.push(idx as u32);
            }
            ug.push_cell(cell_type, &verts);
        }

        // Attach element variables for this block (vals_elem_var{v}eb{blk_num})
        for v_idx in 1..=num_elem_vars {
            let var_key = format!("vals_elem_var{v_idx}eb{blk_num}");
            if !backend.has_var(&var_key) {
                continue;
            }
            let out_name = elem_var_names
                .get(v_idx - 1)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("elem_var_{v_idx}"));
            let arr = backend.read_var_data_array(&var_key, &out_name, Some(step_idx), None)?;
            if arr.num_tuples() == num_el {
                ug.cell_data.add_array(arr);
            }
        }

        let blk_name = eb_names
            .get(pos)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("Block_{blk_num}"));

        blocks.push((blk_name, ug));
    }

    if blocks.len() == 1 {
        let (_, ug) = blocks.remove(0);
        // Avoid keeping an unnecessary extra clone of point_data in caller
        let _ = &mut point_data;
        Ok(DataSet::UnstructuredGrid(ug))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, ug) in blocks {
            mb.add_block(name, DataSet::UnstructuredGrid(ug));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

fn read_exodus_coordinates(backend: &dyn ExodusBackend) -> Result<Vec<[f32; 3]>> {
    if backend.has_var("coordx") {
        let cx = backend.read_var_f32("coordx", None)?;
        let cy = if backend.has_var("coordy") {
            Some(backend.read_var_f32("coordy", None)?)
        } else {
            None
        };
        let cz = if backend.has_var("coordz") {
            Some(backend.read_var_f32("coordz", None)?)
        } else {
            None
        };
        let n = cx.len();
        if let Some(ref y) = cy
            && y.len() != n
        {
            return Err(IoError::InvalidData(format!(
                "Exodus II coordy length ({}) != coordx length ({n})",
                y.len()
            )));
        }
        if let Some(ref z) = cz
            && z.len() != n
        {
            return Err(IoError::InvalidData(format!(
                "Exodus II coordz length ({}) != coordx length ({n})",
                z.len()
            )));
        }
        let cap = safe_prealloc_capacity(n, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
        let mut pts = Vec::with_capacity(cap);
        for i in 0..n {
            pts.push([
                cx[i],
                cy.as_ref().map_or(0.0, |v| v[i]),
                cz.as_ref().map_or(0.0, |v| v[i]),
            ]);
        }
        return Ok(pts);
    }

    if backend.has_var("coord") {
        let shape = backend
            .var_shape("coord")
            .ok_or_else(|| IoError::Parse("Exodus II missing shape for 'coord'".into()))?;
        let flat = backend.read_var_f32("coord", None)?;
        if shape.len() != 2 {
            return Err(IoError::InvalidData(format!(
                "Exodus II 'coord' expected 2D shape, got {shape:?}"
            )));
        }
        let (rows, cols) = (shape[0], shape[1]);
        if rows <= 3 && rows > 0 && cols > 3 {
            // Standard Exodus [num_dim, num_nodes]
            let n = cols;
            let cap = safe_prealloc_capacity(n, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
            let mut pts = Vec::with_capacity(cap);
            for i in 0..n {
                let x = flat[i];
                let y = if rows >= 2 { flat[n + i] } else { 0.0 };
                let z = if rows >= 3 { flat[2 * n + i] } else { 0.0 };
                pts.push([x, y, z]);
            }
            return Ok(pts);
        } else if cols <= 3 && cols > 0 {
            // [num_nodes, num_dim] (or small <=3 nodes with [num_dim, num_nodes])
            let n = rows;
            let cap = safe_prealloc_capacity(n, std::mem::size_of::<[f32; 3]>(), MAX_ALLOC_BYTES)?;
            let mut pts = Vec::with_capacity(cap);
            for i in 0..n {
                let base = i * cols;
                let x = flat[base];
                let y = if cols >= 2 { flat[base + 1] } else { 0.0 };
                let z = if cols >= 3 { flat[base + 2] } else { 0.0 };
                pts.push([x, y, z]);
            }
            return Ok(pts);
        } else {
            return Err(IoError::InvalidData(format!(
                "Exodus II 'coord' has unexpected shape {shape:?}"
            )));
        }
    }

    Err(IoError::Parse(
        "Exodus II file contains neither 'coordx' nor 'coord' coordinates".into(),
    ))
}

fn read_nodal_variables(
    backend: &dyn ExodusBackend,
    num_nodes: usize,
    step_idx: usize,
) -> Result<FieldData> {
    let mut point_data = FieldData::new();
    let names = if backend.has_var("name_nod_var") {
        backend.read_var_strings("name_nod_var").unwrap_or_default()
    } else {
        Vec::new()
    };

    // Case 1: Combined 3D `vals_nod_var` of shape [num_time_steps, num_nod_var, num_nodes]
    if backend.has_var("vals_nod_var")
        && let Some(shape) = backend.var_shape("vals_nod_var")
        && shape.len() == 3
    {
        let n_vars = shape[1];
        for k in 0..n_vars {
            let vname = names
                .get(k)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("nod_var_{}", k + 1));
            let arr = backend.read_var_data_array("vals_nod_var", &vname, Some(step_idx), Some(k))?;
            if arr.num_tuples() == num_nodes {
                point_data.add_array(arr);
            }
        }
        return Ok(point_data);
    }

    // Case 2: Individual `vals_nod_var{k}` variables
    let mut nod_vars: Vec<(usize, String)> = Vec::new();
    for vname in backend.var_names() {
        if let Some(suffix) = vname.strip_prefix("vals_nod_var")
            && let Ok(k) = suffix.parse::<usize>()
            && k >= 1
        {
            nod_vars.push((k, vname));
        }
    }
    nod_vars.sort_by_key(|(k, _)| *k);

    for (k, key) in nod_vars {
        let out_name = names
            .get(k - 1)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("nod_var_{k}"));
        let arr = backend.read_var_data_array(&key, &out_name, Some(step_idx), None)?;
        if arr.num_tuples() == num_nodes {
            point_data.add_array(arr);
        }
    }

    Ok(point_data)
}

fn read_global_variables(
    backend: &dyn ExodusBackend,
    step_idx: usize,
    field_data: &mut FieldData,
) -> Result<()> {
    if !backend.has_var("vals_glo_var") {
        return Ok(());
    }
    let names = if backend.has_var("name_glo_var") {
        backend.read_var_strings("name_glo_var").unwrap_or_default()
    } else {
        Vec::new()
    };
    let arr = backend.read_var_data_array("vals_glo_var", "GlobalVariables", Some(step_idx), None)?;
    if names.len() == arr.num_tuples() && !names.is_empty() {
        for (i, gname) in names.into_iter().enumerate() {
            if !gname.is_empty() {
                let val = arr.data.get_f64(i);
                field_data.add_array(DataArray::scalars_f64(gname, vec![val]));
            }
        }
    } else {
        field_data.add_array(arr);
    }
    Ok(())
}

fn map_exodus_elem_type(elem_type: Option<&str>, nodes_per_el: usize) -> Result<CellType> {
    if let Some(raw) = elem_type {
        let up = raw.trim().to_ascii_uppercase();
        match up.as_str() {
            "SPHERE" | "CIRCLE" | "NODE" => return Ok(CellType::Vertex),
            "BAR" | "BAR2" | "TRUSS" | "BEAM" | "BEAM2" | "EDGE" | "EDGE2" => {
                return Ok(CellType::Line)
            }
            "BAR3" | "BEAM3" | "TRUSS3" | "EDGE3" => return Ok(CellType::QuadraticEdge),
            "TRI" | "TRI3" | "TRIANGLE" => return Ok(CellType::Triangle),
            "TRI6" => return Ok(CellType::QuadraticTriangle),
            "QUAD" | "QUAD4" | "SHELL" | "SHELL4" => return Ok(CellType::Quad),
            "QUAD8" | "SHELL8" => return Ok(CellType::QuadraticQuad),
            "QUAD9" | "SHELL9" => return Ok(CellType::BiQuadraticQuad),
            "TET" | "TETRA" | "TET4" | "TETRA4" => return Ok(CellType::Tetra),
            "TET10" | "TETRA10" => return Ok(CellType::QuadraticTetra),
            "HEX" | "HEX8" | "HEXAHEDRON" => return Ok(CellType::Hexahedron),
            "HEX20" => return Ok(CellType::QuadraticHexahedron),
            "HEX27" => return Ok(CellType::TriQuadraticHexahedron),
            "WEDGE" | "WEDGE6" | "PRISM" => return Ok(CellType::Wedge),
            "WEDGE15" => return Ok(CellType::QuadraticWedge),
            "PYRAMID" | "PYRAMID5" | "PYR5" => return Ok(CellType::Pyramid),
            "PYRAMID13" => return Ok(CellType::QuadraticPyramid),
            _ => {}
        }
    }

    match nodes_per_el {
        1 => Ok(CellType::Vertex),
        2 => Ok(CellType::Line),
        3 => Ok(CellType::Triangle),
        4 => Ok(CellType::Tetra),
        5 => Ok(CellType::Pyramid),
        6 => Ok(CellType::Wedge),
        8 => Ok(CellType::Hexahedron),
        other => Err(IoError::UnsupportedFormat(format!(
            "Unsupported Exodus II element type {:?} ({other} nodes per element)",
            elem_type.unwrap_or("<unspecified>")
        ))),
    }
}

// ============================================================================
// Backend 1: HDF5 / NetCDF-4
// ============================================================================

struct Hdf5ExodusBackend {
    file: Hdf5File,
    dataset_names: Vec<String>,
}

impl Hdf5ExodusBackend {
    fn new(file: Hdf5File) -> Result<Self> {
        let root = file
            .root_group()
            .map_err(|e| IoError::Parse(format!("Failed to open HDF5 root group: {e}")))?;
        let (_, dataset_names) = root
            .member_names()
            .map_err(|e| IoError::Parse(format!("Failed to list HDF5 datasets: {e}")))?;
        Ok(Self {
            file,
            dataset_names,
        })
    }

    fn build_slice(
        shape: &[u64],
        step_idx: Option<usize>,
        sub_idx: Option<usize>,
    ) -> Result<Option<SliceInfo>> {
        match (step_idx, sub_idx, shape.len()) {
            (Some(s), Some(sub), rank) if rank >= 3 => {
                let s = u64::try_from(s).unwrap_or(0);
                let sub = u64::try_from(sub).unwrap_or(0);
                if s >= shape[0] || sub >= shape[1] {
                    return Err(IoError::InvalidData(format!(
                        "Exodus II slice [{s}, {sub}] out of bounds for shape {shape:?}"
                    )));
                }
                let mut selections = Vec::with_capacity(rank);
                selections.push(SliceInfoElem::Index(s));
                selections.push(SliceInfoElem::Index(sub));
                for &d in &shape[2..] {
                    selections.push(SliceInfoElem::Slice {
                        start: 0,
                        end: d,
                        step: 1,
                    });
                }
                Ok(Some(SliceInfo { selections }))
            }
            (Some(s), None, rank) if rank >= 2 => {
                let s = u64::try_from(s).unwrap_or(0);
                if s >= shape[0] {
                    return Err(IoError::InvalidData(format!(
                        "Exodus II time step index {s} out of bounds for shape {shape:?}"
                    )));
                }
                let mut selections = Vec::with_capacity(rank);
                selections.push(SliceInfoElem::Index(s));
                for &d in &shape[1..] {
                    selections.push(SliceInfoElem::Slice {
                        start: 0,
                        end: d,
                        step: 1,
                    });
                }
                Ok(Some(SliceInfo { selections }))
            }
            _ => Ok(None),
        }
    }
}

impl ExodusBackend for Hdf5ExodusBackend {
    fn has_var(&self, name: &str) -> bool {
        self.dataset_names.iter().any(|n| n == name)
    }

    fn var_names(&self) -> Vec<String> {
        self.dataset_names.clone()
    }

    fn var_shape(&self, name: &str) -> Option<Vec<usize>> {
        let ds = self.file.dataset(name).ok()?;
        Some(ds.shape().iter().map(|&d| d as usize).collect())
    }

    fn var_attr_string(&self, var_name: &str, attr_name: &str) -> Option<String> {
        let ds = self.file.dataset(var_name).ok()?;
        let attr = ds.attribute(attr_name).ok()?;
        attr_as_string(&attr)
    }

    fn read_var_f64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f64>> {
        let ds = self
            .file
            .dataset(name)
            .map_err(|e| IoError::Parse(format!("HDF5 Exodus missing dataset '{name}': {e}")))?;
        let slice = Self::build_slice(ds.shape(), step_idx, None)?;
        read_dataset_f64(&ds, slice.as_ref())
    }

    fn read_var_f32(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f32>> {
        let ds = self
            .file
            .dataset(name)
            .map_err(|e| IoError::Parse(format!("HDF5 Exodus missing dataset '{name}': {e}")))?;
        let slice = Self::build_slice(ds.shape(), step_idx, None)?;
        read_dataset_f32(&ds, slice.as_ref())
    }

    fn read_var_i64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<i64>> {
        let ds = self
            .file
            .dataset(name)
            .map_err(|e| IoError::Parse(format!("HDF5 Exodus missing dataset '{name}': {e}")))?;
        let slice = Self::build_slice(ds.shape(), step_idx, None)?;
        read_dataset_i64(&ds, slice.as_ref())
    }

    fn read_var_data_array(
        &self,
        name: &str,
        out_name: &str,
        step_idx: Option<usize>,
        sub_idx: Option<usize>,
    ) -> Result<DataArray> {
        let ds = self
            .file
            .dataset(name)
            .map_err(|e| IoError::Parse(format!("HDF5 Exodus missing dataset '{name}': {e}")))?;
        let slice = Self::build_slice(ds.shape(), step_idx, sub_idx)?;
        read_dataset_as_data_array(&ds, out_name, 1, slice.as_ref())
    }

    fn read_var_strings(&self, name: &str) -> Result<Vec<String>> {
        let ds = self
            .file
            .dataset(name)
            .map_err(|e| IoError::Parse(format!("HDF5 Exodus missing dataset '{name}': {e}")))?;
        read_dataset_strings(&ds)
    }
}

// ============================================================================
// Backend 2: Pure-Rust Zero-Unsafe NetCDF Classic (CDF-1) & 64-bit Offset (CDF-2)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NcType {
    Byte = 1,
    Char = 2,
    Short = 3,
    Int = 4,
    Float = 5,
    Double = 6,
}

impl NcType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(Self::Byte),
            2 => Ok(Self::Char),
            3 => Ok(Self::Short),
            4 => Ok(Self::Int),
            5 => Ok(Self::Float),
            6 => Ok(Self::Double),
            other => Err(IoError::InvalidData(format!(
                "Invalid NetCDF datatype tag {other}"
            ))),
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Byte | Self::Char => 1,
            Self::Short => 2,
            Self::Int | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Debug, Clone)]
struct NcDim {
    _name: String,
    length: usize, // 0 == unlimited (record) dimension
}

#[derive(Debug, Clone)]
struct NcAttr {
    name: String,
    nc_type: NcType,
    raw: Vec<u8>,
}

impl NcAttr {
    fn as_string(&self) -> Option<String> {
        match self.nc_type {
            NcType::Char | NcType::Byte => Some(decode_nul_terminated_bytes(&self.raw)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct NcVar {
    name: String,
    dimids: Vec<usize>,
    attrs: Vec<NcAttr>,
    nc_type: NcType,
    vsize: u64,
    begin: u64,
    is_record: bool,
}

struct NetCdfClassicBackend<R: Read + Seek> {
    reader: RefCell<R>,
    total_len: u64,
    numrecs: usize,
    recsize: u64,
    dims: Vec<NcDim>,
    vars: HashMap<String, NcVar>,
    var_order: Vec<String>,
}

impl<R: Read + Seek> NetCdfClassicBackend<R> {
    fn open(mut reader: R) -> Result<Self> {
        let total_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let is_cdf2 = if &magic == CDF1_MAGIC {
            false
        } else if &magic == CDF2_MAGIC {
            true
        } else {
            return Err(IoError::UnsupportedFormat(
                "Not a NetCDF Classic (CDF\\x01) or 64-bit Offset (CDF\\x02) file".into(),
            ));
        };

        let raw_numrecs = read_u32_be(&mut reader)?;
        let numrecs = if raw_numrecs == u32::MAX {
            0
        } else {
            raw_numrecs as usize
        };

        let dims = read_nc_dims(&mut reader, total_len)?;
        let _global_attrs = read_nc_attrs(&mut reader, total_len)?;
        let var_list = read_nc_vars(&mut reader, &dims, is_cdf2, total_len)?;

        let mut recsize = 0u64;
        for v in &var_list {
            if v.is_record {
                recsize = recsize.checked_add(v.vsize).ok_or_else(|| {
                    IoError::InvalidData("NetCDF record stride overflows u64".into())
                })?;
            }
        }

        let mut vars = HashMap::with_capacity(var_list.len());
        let mut var_order = Vec::with_capacity(var_list.len());
        for v in var_list {
            var_order.push(v.name.clone());
            vars.insert(v.name.clone(), v);
        }

        Ok(Self {
            reader: RefCell::new(reader),
            total_len,
            numrecs,
            recsize,
            dims,
            vars,
            var_order,
        })
    }

    fn var_shape_internal(&self, var: &NcVar) -> Vec<usize> {
        var.dimids
            .iter()
            .enumerate()
            .map(|(i, &did)| {
                let dlen = self.dims[did].length;
                if i == 0 && dlen == 0 {
                    self.numrecs
                } else {
                    dlen
                }
            })
            .collect()
    }

    fn read_var_raw(
        &self,
        name: &str,
        step_idx: Option<usize>,
        sub_idx: Option<usize>,
    ) -> Result<(NcType, usize, Vec<u8>)> {
        let var = self.vars.get(name).ok_or_else(|| {
            IoError::Parse(format!("NetCDF Exodus missing variable '{name}'"))
        })?;
        let shape = self.var_shape_internal(var);
        let type_size = var.nc_type.byte_size();

        let (offset, elem_count) = match (step_idx, sub_idx, shape.len()) {
            (Some(s), Some(sub), rank) if rank >= 3 => {
                if s >= shape[0] || sub >= shape[1] {
                    return Err(IoError::InvalidData(format!(
                        "NetCDF variable '{name}' slice [{s}, {sub}] out of bounds for {shape:?}"
                    )));
                }
                let inner_elems = checked_shape_product(&shape[2..])?;
                let inner_bytes = checked_alloc_u64(inner_elems, type_size)?;
                let sub_offset = (sub as u64).checked_mul(inner_bytes).ok_or_else(|| {
                    IoError::InvalidData("NetCDF offset overflow".into())
                })?;
                let base = if var.is_record {
                    let rec_off = (s as u64).checked_mul(self.recsize).ok_or_else(|| {
                        IoError::InvalidData("NetCDF record offset overflow".into())
                    })?;
                    var.begin.checked_add(rec_off)
                } else {
                    let slab = (shape[1] as u64).checked_mul(inner_bytes).ok_or_else(|| {
                        IoError::InvalidData("NetCDF offset overflow".into())
                    })?;
                    let step_off = (s as u64).checked_mul(slab).ok_or_else(|| {
                        IoError::InvalidData("NetCDF offset overflow".into())
                    })?;
                    var.begin.checked_add(step_off)
                }
                .and_then(|b| b.checked_add(sub_offset))
                .ok_or_else(|| IoError::InvalidData("NetCDF file offset overflow".into()))?;
                (base, inner_elems)
            }
            (Some(s), None, rank) if rank >= 2 => {
                if s >= shape[0] {
                    return Err(IoError::InvalidData(format!(
                        "NetCDF variable '{name}' time step {s} out of bounds for {shape:?}"
                    )));
                }
                let inner_elems = checked_shape_product(&shape[1..])?;
                let base = if var.is_record {
                    let rec_off = (s as u64).checked_mul(self.recsize).ok_or_else(|| {
                        IoError::InvalidData("NetCDF record offset overflow".into())
                    })?;
                    var.begin.checked_add(rec_off)
                } else {
                    let inner_bytes = checked_alloc_u64(inner_elems, type_size)?;
                    let step_off = (s as u64).checked_mul(inner_bytes).ok_or_else(|| {
                        IoError::InvalidData("NetCDF offset overflow".into())
                    })?;
                    var.begin.checked_add(step_off)
                }
                .ok_or_else(|| IoError::InvalidData("NetCDF file offset overflow".into()))?;
                (base, inner_elems)
            }
            _ => {
                if var.is_record && !shape.is_empty() {
                    // Reading all records of a record variable (e.g. `time_whole`)
                    let total_elems = checked_shape_product(&shape)?;
                    let per_rec_elems = checked_shape_product(&shape[1..])?;
                    let per_rec_bytes = check_alloc_bytes(per_rec_elems, type_size, MAX_ALLOC_BYTES)?;
                    check_alloc_bytes(total_elems, type_size, MAX_ALLOC_BYTES)?;
                    let mut out = Vec::with_capacity(
                        safe_prealloc_capacity(total_elems, type_size, MAX_ALLOC_BYTES)? * type_size,
                    );
                    let mut reader = self.reader.borrow_mut();
                    for r in 0..shape[0] {
                        let off = var
                            .begin
                            .checked_add((r as u64).checked_mul(self.recsize).ok_or_else(|| {
                                IoError::InvalidData("NetCDF record offset overflow".into())
                            })?)
                            .ok_or_else(|| IoError::InvalidData("NetCDF offset overflow".into()))?;
                        let end = off.checked_add(per_rec_bytes as u64).ok_or_else(|| {
                            IoError::InvalidData("NetCDF record end offset overflow".into())
                        })?;
                        if end > self.total_len {
                            return Err(IoError::InvalidData(format!(
                                "NetCDF variable '{name}' record {r} exceeds file size ({} > {})",
                                end, self.total_len
                            )));
                        }
                        reader.seek(SeekFrom::Start(off))?;
                        let mut rec_buf = vec![0u8; per_rec_bytes];
                        reader.read_exact(&mut rec_buf)?;
                        out.extend_from_slice(&rec_buf);
                    }
                    return Ok((var.nc_type, total_elems, out));
                }
                let elems = checked_shape_product(&shape)?;
                (var.begin, elems)
            }
        };

        let byte_len = check_alloc_bytes(elem_count, type_size, MAX_ALLOC_BYTES)?;
        let end = offset
            .checked_add(byte_len as u64)
            .ok_or_else(|| IoError::InvalidData("NetCDF variable end offset overflow".into()))?;
        if end > self.total_len {
            return Err(IoError::InvalidData(format!(
                "NetCDF variable '{name}' data range [{offset}..{end}] exceeds file size ({})",
                self.total_len
            )));
        }

        let mut buf = vec![0u8; byte_len];
        let mut reader = self.reader.borrow_mut();
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(&mut buf)?;
        Ok((var.nc_type, elem_count, buf))
    }
}

impl<R: Read + Seek> ExodusBackend for NetCdfClassicBackend<R> {
    fn has_var(&self, name: &str) -> bool {
        self.vars.contains_key(name)
    }

    fn var_names(&self) -> Vec<String> {
        self.var_order.clone()
    }

    fn var_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.vars.get(name).map(|v| self.var_shape_internal(v))
    }

    fn var_attr_string(&self, var_name: &str, attr_name: &str) -> Option<String> {
        let var = self.vars.get(var_name)?;
        var.attrs
            .iter()
            .find(|a| a.name.eq_ignore_ascii_case(attr_name))
            .and_then(|a| a.as_string())
    }

    fn read_var_f64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f64>> {
        let (nc_type, count, raw) = self.read_var_raw(name, step_idx, None)?;
        let cap = safe_prealloc_capacity(count, std::mem::size_of::<f64>(), MAX_ALLOC_BYTES)?;
        let mut out = Vec::with_capacity(cap);
        match nc_type {
            NcType::Double => {
                for &c in raw.as_chunks::<8>().0 {
                    out.push(f64::from_be_bytes(c));
                }
            }
            NcType::Float => {
                for &c in raw.as_chunks::<4>().0 {
                    out.push(f32::from_be_bytes(c) as f64);
                }
            }
            NcType::Int => {
                for &c in raw.as_chunks::<4>().0 {
                    out.push(i32::from_be_bytes(c) as f64);
                }
            }
            NcType::Short => {
                for &c in raw.as_chunks::<2>().0 {
                    out.push(i16::from_be_bytes(c) as f64);
                }
            }
            NcType::Byte | NcType::Char => {
                for &b in &raw {
                    out.push((b as i8) as f64);
                }
            }
        }
        Ok(out)
    }

    fn read_var_f32(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<f32>> {
        let (nc_type, count, raw) = self.read_var_raw(name, step_idx, None)?;
        let cap = safe_prealloc_capacity(count, std::mem::size_of::<f32>(), MAX_ALLOC_BYTES)?;
        let mut out = Vec::with_capacity(cap);
        match nc_type {
            NcType::Float => {
                for &c in raw.as_chunks::<4>().0 {
                    out.push(f32::from_be_bytes(c));
                }
            }
            NcType::Double => {
                for &c in raw.as_chunks::<8>().0 {
                    out.push(f64::from_be_bytes(c) as f32);
                }
            }
            _ => {
                return Ok(self
                    .read_var_f64(name, step_idx)?
                    .into_iter()
                    .map(|v| v as f32)
                    .collect());
            }
        }
        Ok(out)
    }

    fn read_var_i64(&self, name: &str, step_idx: Option<usize>) -> Result<Vec<i64>> {
        let (nc_type, count, raw) = self.read_var_raw(name, step_idx, None)?;
        let cap = safe_prealloc_capacity(count, std::mem::size_of::<i64>(), MAX_ALLOC_BYTES)?;
        let mut out = Vec::with_capacity(cap);
        match nc_type {
            NcType::Int => {
                for &c in raw.as_chunks::<4>().0 {
                    out.push(i32::from_be_bytes(c) as i64);
                }
            }
            NcType::Short => {
                for &c in raw.as_chunks::<2>().0 {
                    out.push(i16::from_be_bytes(c) as i64);
                }
            }
            NcType::Byte | NcType::Char => {
                for &b in &raw {
                    out.push((b as i8) as i64);
                }
            }
            other => {
                return Err(IoError::UnsupportedFormat(format!(
                    "NetCDF variable '{name}' expected integer type, got {other:?}"
                )));
            }
        }
        Ok(out)
    }

    fn read_var_data_array(
        &self,
        name: &str,
        out_name: &str,
        step_idx: Option<usize>,
        sub_idx: Option<usize>,
    ) -> Result<DataArray> {
        let (nc_type, count, raw) = self.read_var_raw(name, step_idx, sub_idx)?;
        let data = match nc_type {
            NcType::Double => {
                let cap = safe_prealloc_capacity(count, 8, MAX_ALLOC_BYTES)?;
                let mut v = Vec::with_capacity(cap);
                for &c in raw.as_chunks::<8>().0 {
                    v.push(f64::from_be_bytes(c));
                }
                ArrayData::F64(v)
            }
            NcType::Float => {
                let cap = safe_prealloc_capacity(count, 4, MAX_ALLOC_BYTES)?;
                let mut v = Vec::with_capacity(cap);
                for &c in raw.as_chunks::<4>().0 {
                    v.push(f32::from_be_bytes(c));
                }
                ArrayData::F32(v)
            }
            NcType::Int => {
                let cap = safe_prealloc_capacity(count, 4, MAX_ALLOC_BYTES)?;
                let mut v = Vec::with_capacity(cap);
                for &c in raw.as_chunks::<4>().0 {
                    v.push(i32::from_be_bytes(c));
                }
                ArrayData::I32(v)
            }
            NcType::Short => {
                let cap = safe_prealloc_capacity(count, 2, MAX_ALLOC_BYTES)?;
                let mut v = Vec::with_capacity(cap);
                for &c in raw.as_chunks::<2>().0 {
                    v.push(i16::from_be_bytes(c));
                }
                ArrayData::I16(v)
            }
            NcType::Byte => {
                let v: Vec<i8> = raw.into_iter().map(|b| b as i8).collect();
                ArrayData::I8(v)
            }
            NcType::Char => ArrayData::U8(raw),
        };
        Ok(DataArray::new(out_name, 1, data))
    }

    fn read_var_strings(&self, name: &str) -> Result<Vec<String>> {
        let shape = self
            .var_shape(name)
            .ok_or_else(|| IoError::Parse(format!("NetCDF Exodus missing string var '{name}'")))?;
        let (_, _, raw) = self.read_var_raw(name, None, None)?;
        match shape.len() {
            0 | 1 => Ok(vec![decode_nul_terminated_bytes(&raw)]),
            2 => {
                let rows = shape[0];
                let cols = shape[1];
                let cap = safe_prealloc_capacity(rows, std::mem::size_of::<String>(), MAX_ALLOC_BYTES)?;
                let mut out = Vec::with_capacity(cap);
                for r in 0..rows {
                    let s = r * cols;
                    let e = (s + cols).min(raw.len());
                    if s >= raw.len() {
                        out.push(String::new());
                    } else {
                        out.push(decode_nul_terminated_bytes(&raw[s..e]));
                    }
                }
                Ok(out)
            }
            rank => Err(IoError::InvalidData(format!(
                "NetCDF string variable '{name}' has unsupported rank {rank}"
            ))),
        }
    }
}

fn checked_shape_product(dims: &[usize]) -> Result<usize> {
    let mut prod = 1usize;
    for &d in dims {
        prod = prod
            .checked_mul(d)
            .ok_or_else(|| IoError::InvalidData("NetCDF shape element count overflows usize".into()))?;
    }
    Ok(prod)
}

fn checked_alloc_u64(count: usize, elem_size: usize) -> Result<u64> {
    let bytes = check_alloc_bytes(count, elem_size, MAX_ALLOC_BYTES)?;
    Ok(bytes as u64)
}

fn read_u32_be<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_be_bytes(b))
}

fn read_u64_be<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_be_bytes(b))
}

fn read_nc_string<R: Read + Seek>(r: &mut R, total_len: u64) -> Result<String> {
    let len = read_u32_be(r)? as usize;
    check_alloc_bytes(len, 1, MAX_ALLOC_BYTES)?;
    let pos = r.stream_position()?;
    let padded = (len + 3) & !3;
    if pos.saturating_add(padded as u64) > total_len {
        return Err(IoError::InvalidData(
            "NetCDF string length exceeds file size".into(),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    let pad = padded - len;
    if pad > 0 {
        let mut discard = [0u8; 3];
        r.read_exact(&mut discard[..pad])?;
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_nc_dims<R: Read + Seek>(r: &mut R, total_len: u64) -> Result<Vec<NcDim>> {
    let tag = read_u32_be(r)?;
    let nelems = read_u32_be(r)? as usize;
    if tag == 0 {
        if nelems != 0 {
            return Err(IoError::InvalidData(
                "NetCDF ABSENT dim_list has non-zero count".into(),
            ));
        }
        return Ok(Vec::new());
    }
    if tag != 10 {
        return Err(IoError::InvalidData(format!(
            "Expected NetCDF NC_DIMENSION tag (10), got {tag}"
        )));
    }
    // Each dim takes at least 8 bytes (name len + length)
    let pos = r.stream_position()?;
    if (nelems as u64).saturating_mul(8) > total_len.saturating_sub(pos) {
        return Err(IoError::InvalidData(
            "NetCDF dimension count exceeds remaining file bytes".into(),
        ));
    }
    let cap = safe_prealloc_capacity(nelems, std::mem::size_of::<NcDim>(), MAX_ALLOC_BYTES)?;
    let mut dims = Vec::with_capacity(cap);
    for _ in 0..nelems {
        let name = read_nc_string(r, total_len)?;
        let length = read_u32_be(r)? as usize;
        dims.push(NcDim { _name: name, length });
    }
    Ok(dims)
}

fn read_nc_attrs<R: Read + Seek>(r: &mut R, total_len: u64) -> Result<Vec<NcAttr>> {
    let tag = read_u32_be(r)?;
    let nelems = read_u32_be(r)? as usize;
    if tag == 0 {
        if nelems != 0 {
            return Err(IoError::InvalidData(
                "NetCDF ABSENT attr_list has non-zero count".into(),
            ));
        }
        return Ok(Vec::new());
    }
    if tag != 12 {
        return Err(IoError::InvalidData(format!(
            "Expected NetCDF NC_ATTRIBUTE tag (12), got {tag}"
        )));
    }
    let pos = r.stream_position()?;
    if (nelems as u64).saturating_mul(12) > total_len.saturating_sub(pos) {
        return Err(IoError::InvalidData(
            "NetCDF attribute count exceeds remaining file bytes".into(),
        ));
    }
    let cap = safe_prealloc_capacity(nelems, std::mem::size_of::<NcAttr>(), MAX_ALLOC_BYTES)?;
    let mut attrs = Vec::with_capacity(cap);
    for _ in 0..nelems {
        let name = read_nc_string(r, total_len)?;
        let nc_type = NcType::from_u32(read_u32_be(r)?)?;
        let count = read_u32_be(r)? as usize;
        let byte_len = check_alloc_bytes(count, nc_type.byte_size(), MAX_ALLOC_BYTES)?;
        let padded = (byte_len + 3) & !3;
        let cur_pos = r.stream_position()?;
        if cur_pos.saturating_add(padded as u64) > total_len {
            return Err(IoError::InvalidData(format!(
                "NetCDF attribute '{name}' payload exceeds file size"
            )));
        }
        let mut raw = vec![0u8; byte_len];
        r.read_exact(&mut raw)?;
        let pad = padded - byte_len;
        if pad > 0 {
            let mut discard = [0u8; 3];
            r.read_exact(&mut discard[..pad])?;
        }
        attrs.push(NcAttr { name, nc_type, raw });
    }
    Ok(attrs)
}

fn read_nc_vars<R: Read + Seek>(
    r: &mut R,
    dims: &[NcDim],
    is_cdf2: bool,
    total_len: u64,
) -> Result<Vec<NcVar>> {
    let tag = read_u32_be(r)?;
    let nelems = read_u32_be(r)? as usize;
    if tag == 0 {
        if nelems != 0 {
            return Err(IoError::InvalidData(
                "NetCDF ABSENT var_list has non-zero count".into(),
            ));
        }
        return Ok(Vec::new());
    }
    if tag != 11 {
        return Err(IoError::InvalidData(format!(
            "Expected NetCDF NC_VARIABLE tag (11), got {tag}"
        )));
    }
    let pos = r.stream_position()?;
    if (nelems as u64).saturating_mul(20) > total_len.saturating_sub(pos) {
        return Err(IoError::InvalidData(
            "NetCDF variable count exceeds remaining file bytes".into(),
        ));
    }
    let cap = safe_prealloc_capacity(nelems, std::mem::size_of::<NcVar>(), MAX_ALLOC_BYTES)?;
    let mut vars = Vec::with_capacity(cap);
    for _ in 0..nelems {
        let name = read_nc_string(r, total_len)?;
        let ndims = read_u32_be(r)? as usize;
        let cur_pos = r.stream_position()?;
        if (ndims as u64).saturating_mul(4) > total_len.saturating_sub(cur_pos) {
            return Err(IoError::InvalidData(format!(
                "NetCDF variable '{name}' ndims ({ndims}) exceeds file size"
            )));
        }
        let dim_cap = safe_prealloc_capacity(ndims, std::mem::size_of::<usize>(), MAX_ALLOC_BYTES)?;
        let mut dimids = Vec::with_capacity(dim_cap);
        for _ in 0..ndims {
            let did = read_u32_be(r)? as usize;
            if did >= dims.len() {
                return Err(IoError::InvalidData(format!(
                    "NetCDF variable '{name}' references invalid dimid {did}"
                )));
            }
            dimids.push(did);
        }
        let attrs = read_nc_attrs(r, total_len)?;
        let nc_type = NcType::from_u32(read_u32_be(r)?)?;
        let vsize = read_u32_be(r)? as u64;
        let begin = if is_cdf2 {
            read_u64_be(r)?
        } else {
            read_u32_be(r)? as u64
        };
        let is_record = !dimids.is_empty() && dims[dimids[0]].length == 0;
        vars.push(NcVar {
            name,
            dimids,
            attrs,
            nc_type,
            vsize,
            begin,
            is_record,
        });
    }
    Ok(vars)
}

/// Returns `true` if the HDF5 file contains Exodus II signature datasets.
pub(crate) fn is_exodus_hdf5(file: &Hdf5File) -> bool {
    let Ok(root) = file.root_group() else {
        return false;
    };
    let Ok((_, datasets)) = root.member_names() else {
        return false;
    };
    datasets
        .iter()
        .any(|d| d == "coordx" || d == "coord" || d == "connect1" || d == "eb_prop1")
}
