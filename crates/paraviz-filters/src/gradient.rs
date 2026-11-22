//! Vector/scalar gradient, divergence, vorticity, and Q-criterion filter (`GradientFilter`).

use glam::Vec3;
use paraviz_core::{
    ArrayData, DataArray, DataSet, FieldData, Filter, ImageData, MultiBlockDataSet, PipelineError,
    RectilinearGrid, StructuredGrid, UnstructuredGrid,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Computes spatial derivatives (`Gradient`, `Divergence`, `Vorticity`, `QCriterion`)
/// of a scalar or 3D vector field on volumetric grids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientFilter {
    pub field_name: String,
    pub compute_gradient: bool,
    pub compute_divergence: bool,
    pub compute_vorticity: bool,
    pub compute_q_criterion: bool,
    pub gradient_name: String,
    pub divergence_name: String,
    pub vorticity_name: String,
    pub q_criterion_name: String,
}

impl Default for GradientFilter {
    fn default() -> Self {
        Self {
            field_name: String::new(),
            compute_gradient: true,
            compute_divergence: false,
            compute_vorticity: false,
            compute_q_criterion: false,
            gradient_name: "Gradient".into(),
            divergence_name: "Divergence".into(),
            vorticity_name: "Vorticity".into(),
            q_criterion_name: "QCriterion".into(),
        }
    }
}

impl GradientFilter {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            ..Default::default()
        }
    }

    pub fn with_all_vector_quantities(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            compute_gradient: true,
            compute_divergence: true,
            compute_vorticity: true,
            compute_q_criterion: true,
            ..Default::default()
        }
    }
}

impl Filter for GradientFilter {
    fn name(&self) -> &str {
        "GradientFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::ImageData(img) => {
                let mut out = img.clone();
                let arr = self.resolve_array(&img.point_data)?;
                let res = compute_image_jacobians(img, &arr)?;
                self.attach_outputs(&mut out.point_data, res)?;
                Ok(DataSet::ImageData(out))
            }
            DataSet::RectilinearGrid(rg) => {
                let mut out = rg.clone();
                let arr = self.resolve_array(&rg.point_data)?;
                let res = compute_rectilinear_jacobians(rg, &arr)?;
                self.attach_outputs(&mut out.point_data, res)?;
                Ok(DataSet::RectilinearGrid(out))
            }
            DataSet::StructuredGrid(sg) => {
                let mut out = sg.clone();
                let arr = self.resolve_array(&sg.point_data)?;
                let res = compute_structured_jacobians(sg, &arr)?;
                self.attach_outputs(&mut out.point_data, res)?;
                Ok(DataSet::StructuredGrid(out))
            }
            DataSet::UnstructuredGrid(ug) => {
                let mut out = ug.clone();
                let arr = self.resolve_array(&ug.point_data)?;
                let res = compute_unstructured_jacobians(ug, &arr)?;
                self.attach_outputs(&mut out.point_data, res)?;
                Ok(DataSet::UnstructuredGrid(out))
            }
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            DataSet::PolyData(_) => Err(PipelineError::ExecutionFailed(
                "GradientFilter requires ImageData, RectilinearGrid, StructuredGrid, or UnstructuredGrid".into(),
            )),
        }
    }
}

enum JacobianResult {
    Scalar(Vec<[f32; 3]>),
    Vector(Vec<[f64; 9]>),
}

impl GradientFilter {
    fn resolve_array(&self, fd: &FieldData) -> Result<DataArray, PipelineError> {
        let arr = if self.field_name.is_empty() {
            if self.compute_divergence || self.compute_vorticity || self.compute_q_criterion {
                fd.active_vectors_array()
                    .or_else(|| fd.active_scalars_array())
            } else {
                fd.active_scalars_array()
                    .or_else(|| fd.active_vectors_array())
            }
            .ok_or_else(|| {
                PipelineError::ExecutionFailed("No active array found for GradientFilter".into())
            })?
        } else {
            fd.get(&self.field_name).ok_or_else(|| {
                PipelineError::ExecutionFailed(format!(
                    "Array '{}' not found in point_data for GradientFilter",
                    self.field_name
                ))
            })?
        };
        if arr.num_components != 1 && arr.num_components < 3 {
            return Err(PipelineError::ExecutionFailed(format!(
                "GradientFilter supports 1-component scalar or >=3-component vector arrays, got {} components",
                arr.num_components
            )));
        }
        Ok(arr.clone())
    }

    fn attach_outputs(
        &self,
        pd: &mut FieldData,
        res: JacobianResult,
    ) -> Result<(), PipelineError> {
        match res {
            JacobianResult::Scalar(grads) => {
                if self.compute_divergence || self.compute_vorticity || self.compute_q_criterion {
                    return Err(PipelineError::ExecutionFailed(
                        "Divergence, Vorticity, and QCriterion require a 3-component vector field"
                            .into(),
                    ));
                }
                if self.compute_gradient {
                    pd.add_array(DataArray::vectors_f32(&self.gradient_name, grads));
                }
            }
            JacobianResult::Vector(jacs) => {
                if self.compute_gradient {
                    let flat: Vec<f32> = jacs
                        .par_iter()
                        .flat_map_iter(|j| j.iter().map(|&v| v as f32))
                        .collect();
                    pd.add_array(DataArray::new(
                        &self.gradient_name,
                        9,
                        ArrayData::F32(flat),
                    ));
                }
                if self.compute_divergence {
                    let div: Vec<f32> = jacs
                        .par_iter()
                        .map(|j| (j[0] + j[4] + j[8]) as f32)
                        .collect();
                    pd.add_array(DataArray::scalars_f32(&self.divergence_name, div));
                }
                if self.compute_vorticity {
                    let vort: Vec<[f32; 3]> = jacs
                        .par_iter()
                        .map(|j| {
                            [
                                (j[7] - j[5]) as f32, // dw/dy - dv/dz
                                (j[2] - j[6]) as f32, // du/dz - dw/dx
                                (j[3] - j[1]) as f32, // dv/dx - du/dy
                            ]
                        })
                        .collect();
                    pd.add_array(DataArray::vectors_f32(&self.vorticity_name, vort));
                }
                if self.compute_q_criterion {
                    let qcrit: Vec<f32> = jacs
                        .par_iter()
                        .map(|j| {
                            let s00 = j[0];
                            let s11 = j[4];
                            let s22 = j[8];
                            let s01 = 0.5 * (j[1] + j[3]);
                            let s02 = 0.5 * (j[2] + j[6]);
                            let s12 = 0.5 * (j[5] + j[7]);
                            let s_norm2 = s00 * s00
                                + s11 * s11
                                + s22 * s22
                                + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12);

                            let o01 = 0.5 * (j[1] - j[3]);
                            let o02 = 0.5 * (j[2] - j[6]);
                            let o12 = 0.5 * (j[5] - j[7]);
                            let o_norm2 = 2.0 * (o01 * o01 + o02 * o02 + o12 * o12);

                            (0.5 * (o_norm2 - s_norm2)) as f32
                        })
                        .collect();
                    pd.add_array(DataArray::scalars_f32(&self.q_criterion_name, qcrit));
                }
            }
        }
        Ok(())
    }
}

fn compute_image_jacobians(
    img: &ImageData,
    arr: &DataArray,
) -> Result<JacobianResult, PipelineError> {
    let [nx, ny, nz] = img.dimensions;
    let total = img.num_points();
    let nc = arr.num_components;
    let [dx, dy, dz] = img.spacing;

    let fd_deriv = |i: usize, j: usize, k: usize, comp: usize| -> [f64; 3] {
        let val = |ii: usize, jj: usize, kk: usize| -> f64 {
            arr.data.get_f64(img.point_index(ii, jj, kk) * nc + comp)
        };
        let gx = if nx <= 1 || dx.abs() < 1e-12 {
            0.0
        } else if i == 0 {
            (val(1, j, k) - val(0, j, k)) / dx
        } else if i == nx - 1 {
            (val(nx - 1, j, k) - val(nx - 2, j, k)) / dx
        } else {
            (val(i + 1, j, k) - val(i - 1, j, k)) / (2.0 * dx)
        };
        let gy = if ny <= 1 || dy.abs() < 1e-12 {
            0.0
        } else if j == 0 {
            (val(i, 1, k) - val(i, 0, k)) / dy
        } else if j == ny - 1 {
            (val(i, ny - 1, k) - val(i, ny - 2, k)) / dy
        } else {
            (val(i, j + 1, k) - val(i, j - 1, k)) / (2.0 * dy)
        };
        let gz = if nz <= 1 || dz.abs() < 1e-12 {
            0.0
        } else if k == 0 {
            (val(i, j, 1) - val(i, j, 0)) / dz
        } else if k == nz - 1 {
            (val(i, j, nz - 1) - val(i, j, nz - 2)) / dz
        } else {
            (val(i, j, k + 1) - val(i, j, k - 1)) / (2.0 * dz)
        };
        [gx, gy, gz]
    };

    if nc == 1 {
        let grads: Vec<[f32; 3]> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let g = fd_deriv(i, j, k, 0);
                [g[0] as f32, g[1] as f32, g[2] as f32]
            })
            .collect();
        Ok(JacobianResult::Scalar(grads))
    } else {
        let jacs: Vec<[f64; 9]> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let r0 = fd_deriv(i, j, k, 0);
                let r1 = fd_deriv(i, j, k, 1);
                let r2 = fd_deriv(i, j, k, 2);
                [
                    r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
                ]
            })
            .collect();
        Ok(JacobianResult::Vector(jacs))
    }
}

fn compute_rectilinear_jacobians(
    rg: &RectilinearGrid,
    arr: &DataArray,
) -> Result<JacobianResult, PipelineError> {
    let [nx, ny, nz] = rg.dimensions();
    let total = rg.num_points();
    let nc = arr.num_components;

    let deriv_1d = |v_prev: f64, v_cur: f64, v_next: f64, c_prev: f64, c_cur: f64, c_next: f64, idx: usize, len: usize| -> f64 {
        if len <= 1 {
            return 0.0;
        }
        if idx == 0 {
            let d = c_next - c_cur;
            if d.abs() > 1e-12 { (v_next - v_cur) / d } else { 0.0 }
        } else if idx == len - 1 {
            let d = c_cur - c_prev;
            if d.abs() > 1e-12 { (v_cur - v_prev) / d } else { 0.0 }
        } else {
            let d = c_next - c_prev;
            if d.abs() > 1e-12 { (v_next - v_prev) / d } else { 0.0 }
        }
    };

    let fd_deriv = |i: usize, j: usize, k: usize, comp: usize| -> [f64; 3] {
        let val = |ii: usize, jj: usize, kk: usize| -> f64 {
            arr.data.get_f64(rg.point_index(ii, jj, kk) * nc + comp)
        };
        let gx = deriv_1d(
            val(i.saturating_sub(1), j, k),
            val(i, j, k),
            val((i + 1).min(nx - 1), j, k),
            rg.x_coords[i.saturating_sub(1)] as f64,
            rg.x_coords[i] as f64,
            rg.x_coords[(i + 1).min(nx - 1)] as f64,
            i,
            nx,
        );
        let gy = deriv_1d(
            val(i, j.saturating_sub(1), k),
            val(i, j, k),
            val(i, (j + 1).min(ny - 1), k),
            rg.y_coords[j.saturating_sub(1)] as f64,
            rg.y_coords[j] as f64,
            rg.y_coords[(j + 1).min(ny - 1)] as f64,
            j,
            ny,
        );
        let gz = deriv_1d(
            val(i, j, k.saturating_sub(1)),
            val(i, j, k),
            val(i, j, (k + 1).min(nz - 1)),
            rg.z_coords[k.saturating_sub(1)] as f64,
            rg.z_coords[k] as f64,
            rg.z_coords[(k + 1).min(nz - 1)] as f64,
            k,
            nz,
        );
        [gx, gy, gz]
    };

    if nc == 1 {
        let grads: Vec<[f32; 3]> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let g = fd_deriv(i, j, k, 0);
                [g[0] as f32, g[1] as f32, g[2] as f32]
            })
            .collect();
        Ok(JacobianResult::Scalar(grads))
    } else {
        let jacs: Vec<[f64; 9]> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let r0 = fd_deriv(i, j, k, 0);
                let r1 = fd_deriv(i, j, k, 1);
                let r2 = fd_deriv(i, j, k, 2);
                [
                    r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
                ]
            })
            .collect();
        Ok(JacobianResult::Vector(jacs))
    }
}

fn compute_structured_jacobians(
    sg: &StructuredGrid,
    arr: &DataArray,
) -> Result<JacobianResult, PipelineError> {
    let [nx, ny, nz] = sg.dimensions;
    let n = sg.num_points();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let u = sg.point_index(i, j, k);
                if i > 0 {
                    adj[u].push(sg.point_index(i - 1, j, k));
                }
                if i + 1 < nx {
                    adj[u].push(sg.point_index(i + 1, j, k));
                }
                if j > 0 {
                    adj[u].push(sg.point_index(i, j - 1, k));
                }
                if j + 1 < ny {
                    adj[u].push(sg.point_index(i, j + 1, k));
                }
                if k > 0 {
                    adj[u].push(sg.point_index(i, j, k - 1));
                }
                if k + 1 < nz {
                    adj[u].push(sg.point_index(i, j, k + 1));
                }
            }
        }
    }
    least_squares_jacobians(&sg.points, &adj, arr)
}

fn compute_unstructured_jacobians(
    ug: &UnstructuredGrid,
    arr: &DataArray,
) -> Result<JacobianResult, PipelineError> {
    let n = ug.num_points();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for cell_idx in 0..ug.num_cells() {
        let pts = ug.cell_points(cell_idx);
        for &a in pts {
            let ai = a as usize;
            for &b in pts {
                let bi = b as usize;
                if ai != bi && !adj[ai].contains(&bi) {
                    adj[ai].push(bi);
                }
            }
        }
    }
    least_squares_jacobians(&ug.points, &adj, arr)
}

/// Weighted least-squares gradient estimation at each vertex from its neighborhood.
fn least_squares_jacobians(
    points: &[[f32; 3]],
    adj: &[Vec<usize>],
    arr: &DataArray,
) -> Result<JacobianResult, PipelineError> {
    let n = points.len();
    let nc = arr.num_components;

    let solve_vertex = |i: usize, comp: usize| -> [f64; 3] {
        let nbs = &adj[i];
        if nbs.is_empty() {
            return [0.0; 3];
        }
        let pi = Vec3::from_array(points[i]);
        let fi = arr.data.get_f64(i * nc + comp);

        let mut ata = [[0.0f64; 3]; 3];
        let mut atb = [0.0f64; 3];
        for &j in nbs {
            let d = Vec3::from_array(points[j]) - pi;
            let len2 = d.length_squared() as f64;
            if len2 < 1e-20 {
                continue;
            }
            let w = 1.0 / len2;
            let df = arr.data.get_f64(j * nc + comp) - fi;
            let dv = [d.x as f64, d.y as f64, d.z as f64];
            for r in 0..3 {
                atb[r] += w * dv[r] * df;
                for c in 0..3 {
                    ata[r][c] += w * dv[r] * dv[c];
                }
            }
        }
        solve_3x3_regularized(ata, atb)
    };

    if nc == 1 {
        let grads: Vec<[f32; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let g = solve_vertex(i, 0);
                [g[0] as f32, g[1] as f32, g[2] as f32]
            })
            .collect();
        Ok(JacobianResult::Scalar(grads))
    } else {
        let jacs: Vec<[f64; 9]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let r0 = solve_vertex(i, 0);
                let r1 = solve_vertex(i, 1);
                let r2 = solve_vertex(i, 2);
                [
                    r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
                ]
            })
            .collect();
        Ok(JacobianResult::Vector(jacs))
    }
}

/// Solves `A * x = b` for symmetric positive semi-definite 3x3 matrix `A` with
/// tiny Tikhonov regularization for 1D/2D degenerate stencils.
fn solve_3x3_regularized(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    let trace = a[0][0] + a[1][1] + a[2][2];
    if trace <= 1e-24 {
        return [0.0; 3];
    }
    let eps = trace * 1e-12;
    a[0][0] += eps;
    a[1][1] += eps;
    a[2][2] += eps;

    // Gaussian elimination with partial pivoting
    for col in 0..3 {
        let mut pivot = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..3 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                pivot = row;
            }
        }
        if max_val < 1e-20 {
            continue;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let inv_diag = 1.0 / a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] * inv_diag;
            for c in col..3 {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; 3];
    for row in (0..3).rev() {
        let mut sum = b[row];
        for c in (row + 1)..3 {
            sum -= a[row][c] * x[c];
        }
        x[row] = if a[row][row].abs() > 1e-20 {
            sum / a[row][row]
        } else {
            0.0
        };
    }
    x
}
