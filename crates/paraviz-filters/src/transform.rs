//! Affine 3D geometric transformation filter (`TransformFilter`, equivalent to `vtkTransformFilter`).

use glam::{EulerRot, Mat3, Mat4, Vec3};
use paraviz_core::{
    DataArray, DataSet, FieldData, Filter, ImageData, MultiBlockDataSet, PipelineError, PolyData,
    StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Applies a 3D scale, Euler XYZ rotation (in degrees), and translation to dataset points,
/// normals, and active vector arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFilter {
    pub translation: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for TransformFilter {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl TransformFilter {
    pub fn new(translation: [f32; 3], rotation_deg: [f32; 3], scale: [f32; 3]) -> Self {
        Self {
            translation,
            rotation_deg,
            scale,
        }
    }

    /// Builds the 4x4 affine transformation matrix (scale -> XYZ rotation -> translation).
    pub fn matrix(&self) -> Mat4 {
        let rot = Mat4::from_euler(
            EulerRot::XYZ,
            self.rotation_deg[0].to_radians(),
            self.rotation_deg[1].to_radians(),
            self.rotation_deg[2].to_radians(),
        );
        let scale = Mat4::from_scale(Vec3::from_array(self.scale));
        let trans = Mat4::from_translation(Vec3::from_array(self.translation));
        trans * rot * scale
    }
}

impl Filter for TransformFilter {
    fn name(&self) -> &str {
        "TransformFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let mat = self.matrix();
        let linear = Mat3::from_mat4(mat);
        let det = linear.determinant();
        let normal_mat = if det.abs() > 1e-12 {
            linear.inverse().transpose()
        } else {
            Mat3::IDENTITY
        };

        let transform_pts = |pts: &[[f32; 3]]| -> Vec<[f32; 3]> {
            pts.par_iter()
                .map(|&p| mat.transform_point3(Vec3::from_array(p)).to_array())
                .collect()
        };

        let transform_field_data = |fd: &FieldData| -> FieldData {
            let mut out = fd.clone();
            if let Some(n_arr) = fd.active_normals_array() {
                let name = n_arr.name.clone();
                let new_n: Vec<[f32; 3]> = (0..n_arr.num_tuples())
                    .into_par_iter()
                    .map(|i| {
                        let v = Vec3::from_array(n_arr.get_vec3(i));
                        (normal_mat * v).normalize_or(Vec3::Z).to_array()
                    })
                    .collect();
                out.set_normals(DataArray::vectors_f32(name, new_n));
            }
            if let Some(v_arr) = fd.active_vectors_array() {
                let name = v_arr.name.clone();
                let new_v: Vec<[f32; 3]> = (0..v_arr.num_tuples())
                    .into_par_iter()
                    .map(|i| {
                        let v = Vec3::from_array(v_arr.get_vec3(i));
                        (linear * v).to_array()
                    })
                    .collect();
                out.set_vectors(DataArray::vectors_f32(name, new_v));
            }
            out
        };

        let is_axis_aligned = self.rotation_deg == [0.0, 0.0, 0.0]
            && self.scale[0] > 0.0
            && self.scale[1] > 0.0
            && self.scale[2] > 0.0;

        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(PolyData {
                points: transform_pts(&pd.points),
                point_data: transform_field_data(&pd.point_data),
                ..pd.clone()
            })),
            DataSet::ImageData(img) if is_axis_aligned => Ok(DataSet::ImageData(ImageData {
                dimensions: img.dimensions,
                origin: [
                    img.origin[0] * self.scale[0] as f64 + self.translation[0] as f64,
                    img.origin[1] * self.scale[1] as f64 + self.translation[1] as f64,
                    img.origin[2] * self.scale[2] as f64 + self.translation[2] as f64,
                ],
                spacing: [
                    img.spacing[0] * self.scale[0] as f64,
                    img.spacing[1] * self.scale[1] as f64,
                    img.spacing[2] * self.scale[2] as f64,
                ],
                point_data: transform_field_data(&img.point_data),
                cell_data: img.cell_data.clone(),
                field_data: img.field_data.clone(),
            })),
            DataSet::ImageData(img) => {
                let [nx, ny, nz] = img.dimensions;
                let pts: Vec<[f32; 3]> = (0..(nx * ny * nz))
                    .into_par_iter()
                    .map(|idx| {
                        let i = idx % nx;
                        let rem = idx / nx;
                        let j = rem % ny;
                        let k = rem / ny;
                        mat.transform_point3(Vec3::from_array(img.point_coords(i, j, k)))
                            .to_array()
                    })
                    .collect();
                Ok(DataSet::StructuredGrid(StructuredGrid {
                    dimensions: img.dimensions,
                    points: pts,
                    point_data: transform_field_data(&img.point_data),
                    cell_data: img.cell_data.clone(),
                    field_data: img.field_data.clone(),
                }))
            }
            DataSet::RectilinearGrid(rg) => {
                let [nx, ny, nz] = rg.dimensions();
                let pts: Vec<[f32; 3]> = (0..(nx * ny * nz))
                    .into_par_iter()
                    .map(|idx| {
                        let i = idx % nx;
                        let rem = idx / nx;
                        let j = rem % ny;
                        let k = rem / ny;
                        mat.transform_point3(Vec3::from_array(rg.point_coords(i, j, k)))
                            .to_array()
                    })
                    .collect();
                Ok(DataSet::StructuredGrid(StructuredGrid {
                    dimensions: rg.dimensions(),
                    points: pts,
                    point_data: transform_field_data(&rg.point_data),
                    cell_data: rg.cell_data.clone(),
                    field_data: rg.field_data.clone(),
                }))
            }
            DataSet::StructuredGrid(sg) => Ok(DataSet::StructuredGrid(StructuredGrid {
                points: transform_pts(&sg.points),
                point_data: transform_field_data(&sg.point_data),
                ..sg.clone()
            })),
            DataSet::UnstructuredGrid(ug) => Ok(DataSet::UnstructuredGrid(UnstructuredGrid {
                points: transform_pts(&ug.points),
                point_data: transform_field_data(&ug.point_data),
                ..ug.clone()
            })),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
        }
    }
}
