//! 3D glyph placement filter (`GlyphFilter`, equivalent to `vtkGlyph3D`).

use crate::util::dataset_points;
use glam::{Quat, Vec3};
use paraviz_core::{
    ArrayData, DataArray, DataSet, FieldData, Filter, PipelineError, PolyData,
};
use paraviz_sources::{ArrowSource, ConeSource, CubeSource, SphereSource};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Geometric shape placed at sampled dataset points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GlyphShape {
    #[default]
    Arrow,
    Cone,
    Sphere,
    Cube,
}

/// How each glyph instance is scaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ScaleMode {
    Scalar,
    #[default]
    VectorMagnitude,
    Constant,
}

/// Places oriented, scaled 3D glyph geometry at sampled points using `paraviz-sources`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlyphFilter {
    pub shape: GlyphShape,
    pub vector_field: Option<String>,
    pub scalar_field: Option<String>,
    pub scale_mode: ScaleMode,
    pub scale_factor: f32,
    pub max_points: usize,
}

impl Default for GlyphFilter {
    fn default() -> Self {
        Self {
            shape: GlyphShape::Arrow,
            vector_field: None,
            scalar_field: None,
            scale_mode: ScaleMode::VectorMagnitude,
            scale_factor: 0.25,
            max_points: 1000,
        }
    }
}

impl GlyphFilter {
    pub fn new(shape: GlyphShape) -> Self {
        Self {
            shape,
            ..Default::default()
        }
    }
}

impl Filter for GlyphFilter {
    fn name(&self) -> &str {
        "GlyphFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let parts: Vec<PolyData> = mb
                .blocks
                .par_iter()
                .map(|(_, b)| match self.apply(b)? {
                    DataSet::PolyData(pd) => Ok(pd),
                    _ => unreachable!(),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(DataSet::PolyData(crate::util::merge_polydata(parts)));
        }

        let in_points = dataset_points(input)?;
        if in_points.is_empty() {
            return Ok(DataSet::PolyData(PolyData::new()));
        }
        let pd_data = input.point_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("Input dataset has no point_data".into())
        })?;

        let vec_arr = match &self.vector_field {
            Some(name) => Some(pd_data.get(name).ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "Vector array '{name}' not found for GlyphFilter"
                ))
            })?),
            None => pd_data.active_vectors_array(),
        };

        let sca_arr = match &self.scalar_field {
            Some(name) => Some(pd_data.get(name).ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "Scalar array '{name}' not found for GlyphFilter"
                ))
            })?),
            None => pd_data.active_scalars_array(),
        };

        let template = build_template_glyph(self.shape);
        let tpl_tris = template.triangulate();
        let tpl_normals = template
            .point_data
            .active_normals_array()
            .map(|a| (0..template.num_points()).map(|i| a.get_vec3(i)).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; template.num_points()]);
        let tpl_npts = template.num_points();

        // Uniform strided sampling up to max_points
        let total = in_points.len();
        let sample_indices: Vec<usize> = if self.max_points > 0 && total > self.max_points {
            let step = total as f64 / self.max_points as f64;
            (0..self.max_points)
                .map(|k| ((k as f64 * step).floor() as usize).min(total - 1))
                .collect()
        } else {
            (0..total).collect()
        };

        let instances: Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>, usize)> = sample_indices
            .par_iter()
            .map(|&pt_idx| {
                let origin = Vec3::from_array(in_points[pt_idx]);
                let vec = vec_arr.map(|a| Vec3::from_array(a.get_vec3(pt_idx)));
                let rot = match vec {
                    Some(v) if v.length_squared() > 1e-12 => {
                        Quat::from_rotation_arc(Vec3::X, v.normalize())
                    }
                    _ => Quat::IDENTITY,
                };
                let scale = match self.scale_mode {
                    ScaleMode::Constant => self.scale_factor,
                    ScaleMode::Scalar => {
                        let s = sca_arr.map(|a| a.get_magnitude(pt_idx) as f32).unwrap_or(1.0);
                        s * self.scale_factor
                    }
                    ScaleMode::VectorMagnitude => {
                        let mag = vec.map(|v| v.length()).unwrap_or(1.0);
                        mag * self.scale_factor
                    }
                };

                let mut g_pts = Vec::with_capacity(tpl_npts);
                let mut g_nrms = Vec::with_capacity(tpl_npts);
                for k in 0..tpl_npts {
                    let p = Vec3::from_array(template.points[k]) * scale;
                    let n = Vec3::from_array(tpl_normals[k]);
                    g_pts.push((origin + rot * p).to_array());
                    g_nrms.push((rot * n).normalize_or(Vec3::Z).to_array());
                }
                (g_pts, g_nrms, pt_idx)
            })
            .collect();

        let n_inst = instances.len();
        let mut out_points = Vec::with_capacity(n_inst * tpl_npts);
        let mut out_normals = Vec::with_capacity(n_inst * tpl_npts);
        let mut out_polys = Vec::with_capacity(n_inst * tpl_tris.len());
        let mut src_indices = Vec::with_capacity(n_inst * tpl_npts);

        for (g_pts, g_nrms, src_idx) in instances {
            let base = out_points.len() as u32;
            out_points.extend(g_pts);
            out_normals.extend(g_nrms);
            for _ in 0..tpl_npts {
                src_indices.push(src_idx);
            }
            for tri in &tpl_tris {
                out_polys.push(vec![base + tri[0], base + tri[1], base + tri[2]]);
            }
        }

        // Propagate input point_data arrays (excluding Normals which come from rotated glyph normals).
        let mut out_pdata = FieldData::new();
        for arr in pd_data.arrays() {
            if arr.name == "Normals" {
                continue;
            }
            out_pdata.add_array(arr.gather(&src_indices));
        }
        out_pdata.active_scalars = pd_data.active_scalars.clone();
        out_pdata.active_vectors = pd_data.active_vectors.clone();
        out_pdata.set_normals(DataArray::vectors_f32("Normals", out_normals));
        if out_pdata.is_empty() {
            let _ = ArrayData::F32(Vec::new());
        }

        Ok(DataSet::PolyData(PolyData {
            points: out_points,
            polys: out_polys,
            point_data: out_pdata,
            ..Default::default()
        }))
    }
}

fn build_template_glyph(shape: GlyphShape) -> PolyData {
    match shape {
        GlyphShape::Arrow => ArrowSource::default().build(),
        GlyphShape::Cone => ConeSource::default().build(),
        GlyphShape::Sphere => SphereSource {
            theta_resolution: 16,
            phi_resolution: 10,
            ..Default::default()
        }
        .build(),
        GlyphShape::Cube => CubeSource::default().build(),
    }
}
