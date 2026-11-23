//! Axis-aligned reflection filter (`ReflectFilter`, equivalent to `vtkReflectionFilter`).

use crate::util::{dataset_to_unstructured, merge_polydata_refs, merge_unstructured_refs};
use paraviz_core::{
    CellType, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError, PolyData,
    UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Mirror plane selection for [`ReflectFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ReflectPlane {
    #[default]
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
    CenterX,
    CenterY,
    CenterZ,
}

impl ReflectPlane {
    fn axis_and_coord(self, ds: &DataSet) -> (usize, f32) {
        let bb = ds.bounds();
        let (min, max, center) = if bb.is_valid() {
            (bb.min, bb.max, bb.center())
        } else {
            (glam::Vec3::ZERO, glam::Vec3::ZERO, glam::Vec3::ZERO)
        };
        match self {
            Self::XMin => (0, min.x),
            Self::XMax => (0, max.x),
            Self::CenterX => (0, center.x),
            Self::YMin => (1, min.y),
            Self::YMax => (1, max.y),
            Self::CenterY => (1, center.y),
            Self::ZMin => (2, min.z),
            Self::ZMax => (2, max.z),
            Self::CenterZ => (2, center.z),
        }
    }
}

/// Mirrors a dataset across a bounding-box axis plane, flipping cell winding to preserve
/// outward normals and reflecting normal/vector components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectFilter {
    pub plane: ReflectPlane,
    /// When `true`, combines the original dataset with its mirrored reflection.
    pub copy_input: bool,
}

impl Default for ReflectFilter {
    fn default() -> Self {
        Self {
            plane: ReflectPlane::XMin,
            copy_input: true,
        }
    }
}

impl ReflectFilter {
    /// Creates a `ReflectFilter` for the given `plane` and `copy_input` setting.
    pub fn new(plane: ReflectPlane, copy_input: bool) -> Self {
        Self { plane, copy_input }
    }
}

impl Filter for ReflectFilter {
    fn name(&self) -> &str {
        "ReflectFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let (axis, coord) = self.plane.axis_and_coord(input);
        match input {
            DataSet::PolyData(pd) => {
                let reflected = reflect_polydata(pd, axis, coord);
                if self.copy_input {
                    Ok(DataSet::PolyData(merge_polydata_refs(&[pd, &reflected])))
                } else {
                    Ok(DataSet::PolyData(reflected))
                }
            }
            DataSet::UnstructuredGrid(ug) => {
                let reflected = reflect_unstructured(ug, axis, coord);
                if self.copy_input {
                    Ok(DataSet::UnstructuredGrid(merge_unstructured_refs(&[
                        ug,
                        &reflected,
                    ])))
                } else {
                    Ok(DataSet::UnstructuredGrid(reflected))
                }
            }
            DataSet::ImageData(_) | DataSet::RectilinearGrid(_) | DataSet::StructuredGrid(_) => {
                let ug = dataset_to_unstructured(input)?;
                let reflected = reflect_unstructured(&ug, axis, coord);
                if self.copy_input {
                    Ok(DataSet::UnstructuredGrid(merge_unstructured_refs(&[
                        &ug,
                        &reflected,
                    ])))
                } else {
                    Ok(DataSet::UnstructuredGrid(reflected))
                }
            }
            DataSet::MultiBlock(_) => unreachable!(),
        }
    }
}

#[inline]
fn reflect_points(points: &[[f32; 3]], axis: usize, coord: f32) -> Vec<[f32; 3]> {
    let two_c = 2.0 * coord;
    points
        .par_iter()
        .map(|&p| {
            let mut out = p;
            out[axis] = two_c - out[axis];
            out
        })
        .collect()
}

fn reflect_field_vectors(fd: &FieldData, axis: usize) -> FieldData {
    let mut out = fd.clone();

    let flip_array = |arr: &DataArray| -> DataArray {
        let nc = arr.num_components;
        let name = arr.name.clone();
        let vals: Vec<[f32; 3]> = (0..arr.num_tuples())
            .into_par_iter()
            .map(|i| {
                let mut v = arr.get_vec3(i);
                v[axis] = -v[axis];
                v
            })
            .collect();
        if nc == 3 {
            DataArray::vectors_f32(name, vals)
        } else {
            let mut flat = Vec::with_capacity(arr.num_tuples() * nc);
            for (i, v) in vals.into_iter().enumerate() {
                for (c, &vc) in v.iter().enumerate().take(nc.min(3)) {
                    if c == axis {
                        flat.push(-arr.data.get_f64(i * nc + c) as f32);
                    } else {
                        flat.push(vc);
                    }
                }
                for c in 3..nc {
                    flat.push(arr.data.get_f64(i * nc + c) as f32);
                }
            }
            DataArray::new(name, nc, paraviz_core::ArrayData::F32(flat))
        }
    };

    let norm_name = fd.active_normals_array().map(|a| a.name.clone());
    if let Some(n_arr) = fd.active_normals_array() {
        out.set_normals(flip_array(n_arr));
    }
    if let Some(v_arr) = fd.active_vectors_array()
        && norm_name.as_deref() != Some(v_arr.name.as_str())
    {
        out.set_vectors(flip_array(v_arr));
    }
    out
}

fn reflect_polydata(pd: &PolyData, axis: usize, coord: f32) -> PolyData {
    let polys: Vec<Vec<u32>> = pd
        .polys
        .iter()
        .map(|poly| {
            let mut r = poly.clone();
            r.reverse();
            r
        })
        .collect();
    let strips: Vec<Vec<u32>> = pd
        .strips
        .iter()
        .map(|strip| {
            let mut r = strip.clone();
            r.reverse();
            r
        })
        .collect();

    PolyData {
        points: reflect_points(&pd.points, axis, coord),
        verts: pd.verts.clone(),
        lines: pd.lines.clone(),
        polys,
        strips,
        point_data: reflect_field_vectors(&pd.point_data, axis),
        cell_data: reflect_field_vectors(&pd.cell_data, axis),
        field_data: pd.field_data.clone(),
    }
}

fn flip_cell_winding(ct: CellType, pts: &[u32]) -> Vec<u32> {
    match ct {
        CellType::Triangle | CellType::Quad | CellType::Polygon | CellType::TriangleStrip => {
            let mut out = pts.to_vec();
            out.reverse();
            out
        }
        CellType::Pixel if pts.len() == 4 => vec![pts[0], pts[2], pts[1], pts[3]],
        CellType::Tetra if pts.len() == 4 => vec![pts[1], pts[0], pts[2], pts[3]],
        CellType::Hexahedron | CellType::Voxel if pts.len() == 8 => vec![
            pts[4], pts[5], pts[6], pts[7], pts[0], pts[1], pts[2], pts[3],
        ],
        CellType::Wedge if pts.len() == 6 => {
            vec![pts[3], pts[4], pts[5], pts[0], pts[1], pts[2]]
        }
        CellType::Pyramid if pts.len() == 5 => {
            vec![pts[0], pts[3], pts[2], pts[1], pts[4]]
        }
        _ => pts.to_vec(),
    }
}

fn reflect_unstructured(ug: &UnstructuredGrid, axis: usize, coord: f32) -> UnstructuredGrid {
    let mut out = UnstructuredGrid::new();
    out.points = reflect_points(&ug.points, axis, coord);
    for c in 0..ug.num_cells() {
        let ct = ug.cell_types[c];
        let flipped = flip_cell_winding(ct, ug.cell_points(c));
        out.push_cell(ct, &flipped);
    }
    out.point_data = reflect_field_vectors(&ug.point_data, axis);
    out.cell_data = reflect_field_vectors(&ug.cell_data, axis);
    out.field_data = ug.field_data.clone();
    out
}
