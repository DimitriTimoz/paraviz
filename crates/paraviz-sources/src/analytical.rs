//! Analytical scientific sources (`WaveletSource`, `VectorFieldSource`, `SuperquadricSource`).

use paraviz_core::{DataArray, DataSet, ImageData, PipelineError, PipelineInfo, PolyData, Source};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

/// VTK's classic 3D analytical wavelet volume source (`vtkRTAnalyticSource`),
/// producing an [`ImageData`] with an `"RTData"` scalar field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveletSource {
    /// Whole extent `[xmin, xmax, ymin, ymax, zmin, zmax]` (inclusive).
    pub whole_extent: [i32; 6],
    pub center: [f64; 3],
    pub maximum: f64,
    pub std_dev: f64,
    pub x_freq: f64,
    pub y_freq: f64,
    pub z_freq: f64,
    pub x_mag: f64,
    pub y_mag: f64,
    pub z_mag: f64,
}

impl Default for WaveletSource {
    fn default() -> Self {
        Self {
            whole_extent: [-10, 10, -10, 10, -10, 10],
            center: [0.0, 0.0, 0.0],
            maximum: 255.0,
            std_dev: 0.5,
            x_freq: 60.0,
            y_freq: 30.0,
            z_freq: 40.0,
            x_mag: 10.0,
            y_mag: 18.0,
            z_mag: 5.0,
        }
    }
}

impl WaveletSource {
    pub fn with_extent(half_extent: i32) -> Self {
        Self {
            whole_extent: [
                -half_extent,
                half_extent,
                -half_extent,
                half_extent,
                -half_extent,
                half_extent,
            ],
            ..Default::default()
        }
    }

    pub fn build(&self, time: f64) -> ImageData {
        let [x0, x1, y0, y1, z0, z1] = self.whole_extent;
        let nx = (x1 - x0 + 1).max(1) as usize;
        let ny = (y1 - y0 + 1).max(1) as usize;
        let nz = (z1 - z0 + 1).max(1) as usize;

        let origin = [x0 as f64, y0 as f64, z0 as f64];
        let spacing = [1.0, 1.0, 1.0];

        let scale_x = if nx > 1 { 1.0 / (nx - 1) as f64 } else { 1.0 };
        let scale_y = if ny > 1 { 1.0 / (ny - 1) as f64 } else { 1.0 };
        let scale_z = if nz > 1 { 1.0 / (nz - 1) as f64 } else { 1.0 };
        let temp2 = 1.0 / (2.0 * self.std_dev * self.std_dev);
        let phase = time * 2.0 * std::f64::consts::PI;

        let total = nx * ny * nz;
        let values: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;

                let x = x0 as f64 + i as f64;
                let y = y0 as f64 + j as f64;
                let z = z0 as f64 + k as f64;

                let dx = (x - self.center[0]) * scale_x;
                let dy = (y - self.center[1]) * scale_y;
                let dz = (z - self.center[2]) * scale_z;
                let r2 = dx * dx + dy * dy + dz * dz;

                let gauss = self.maximum * (-r2 * temp2).exp();
                let sx = self.x_mag * (self.x_freq * dx + phase).sin();
                let cy = self.y_mag * (self.y_freq * dy).cos();
                let sz = self.z_mag * (self.z_freq * dz).sin();
                (gauss + sx + cy + sz) as f32
            })
            .collect();

        let mut img = ImageData::new([nx, ny, nz], origin, spacing);
        img.point_data
            .set_scalars(DataArray::scalars_f32("RTData", values));
        img
    }
}

impl Source for WaveletSource {
    fn name(&self) -> &str {
        "WaveletSource"
    }

    fn request_information(&self) -> PipelineInfo {
        PipelineInfo {
            time_steps: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            current_time: 0.0,
            bounds: Some(self.build(0.0).bounds()),
        }
    }

    fn execute(&self, time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::ImageData(self.build(time)))
    }
}

/// Type of analytical 3D vector field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VectorFieldKind {
    #[default]
    SwirlingVortex,
    Dipole,
    HelicalTorus,
}

/// Analytical 3D vector field source on an [`ImageData`] grid, producing `"Velocity"` (vec3)
/// and `"Speed"` (scalar) arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorFieldSource {
    pub dimensions: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub kind: VectorFieldKind,
}

impl Default for VectorFieldSource {
    fn default() -> Self {
        Self {
            dimensions: [25, 25, 25],
            origin: [-2.0, -2.0, -2.0],
            spacing: [4.0 / 24.0, 4.0 / 24.0, 4.0 / 24.0],
            kind: VectorFieldKind::SwirlingVortex,
        }
    }
}

impl VectorFieldSource {
    pub fn build(&self, time: f64) -> ImageData {
        let [nx, ny, nz] = self.dimensions;
        let total = nx * ny * nz;
        let phase = time as f32 * 2.0 * PI;

        let pairs: Vec<([f32; 3], f32)> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let x = (self.origin[0] + i as f64 * self.spacing[0]) as f32;
                let y = (self.origin[1] + j as f64 * self.spacing[1]) as f32;
                let z = (self.origin[2] + k as f64 * self.spacing[2]) as f32;

                let v = match self.kind {
                    VectorFieldKind::SwirlingVortex => {
                        let r2 = x * x + y * y + 0.15;
                        let vx = -y / r2 + 0.25 * (z + phase).sin();
                        let vy = x / r2 + 0.25 * (z + phase).cos();
                        let vz = 0.6 * (-0.3 * (x * x + y * y)).exp() - 0.1 * z;
                        [vx, vy, vz]
                    }
                    VectorFieldKind::Dipole => {
                        let r2 = (x * x + y * y + z * z + 0.25).powf(2.5);
                        let vx = (3.0 * x * z) / r2;
                        let vy = (3.0 * y * z) / r2;
                        let vz = (3.0 * z * z - (x * x + y * y + z * z)) / r2;
                        [vx, vy, vz]
                    }
                    VectorFieldKind::HelicalTorus => {
                        let r = (x * x + y * y).sqrt().max(1e-4);
                        let d = r - 1.1;
                        let vx = -y / r - 0.5 * z * (x / r);
                        let vy = x / r - 0.5 * z * (y / r);
                        let vz = 0.5 * d + 0.15 * phase.sin();
                        [vx, vy, vz]
                    }
                };
                let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                (v, speed)
            })
            .collect();

        let mut vecs = Vec::with_capacity(total);
        let mut speeds = Vec::with_capacity(total);
        for (v, s) in pairs {
            vecs.push(v);
            speeds.push(s);
        }

        let mut img = ImageData::new(self.dimensions, self.origin, self.spacing);
        img.point_data
            .set_vectors(DataArray::vectors_f32("Velocity", vecs));
        img.point_data
            .set_scalars(DataArray::scalars_f32("Speed", speeds));
        img
    }
}

impl Source for VectorFieldSource {
    fn name(&self) -> &str {
        "VectorFieldSource"
    }

    fn execute(&self, time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::ImageData(self.build(time)))
    }
}

/// Parametric Superquadric (toroidal or ellipsoidal) surface source (equivalent to `vtkSuperquadricSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperquadricSource {
    pub toroidal: bool,
    pub thickness: f32,
    pub phi_roundness: f32,
    pub theta_roundness: f32,
    pub size: f32,
    pub phi_resolution: usize,
    pub theta_resolution: usize,
}

impl Default for SuperquadricSource {
    fn default() -> Self {
        Self {
            toroidal: true,
            thickness: 0.35,
            phi_roundness: 0.7,
            theta_roundness: 0.7,
            size: 0.6,
            phi_resolution: 32,
            theta_resolution: 32,
        }
    }
}

fn spow(v: f32, p: f32) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        v.signum() * v.abs().powf(p)
    }
}

impl SuperquadricSource {
    pub fn build(&self) -> PolyData {
        let np = self.phi_resolution.max(4);
        let nt = self.theta_resolution.max(4);
        let mut points = Vec::with_capacity((np + 1) * (nt + 1));
        let mut scalars = Vec::with_capacity((np + 1) * (nt + 1));
        let mut polys = Vec::with_capacity(np * nt * 2);

        let r_major = 1.0 / (self.thickness.clamp(0.05, 0.99));
        for j in 0..=np {
            let v = j as f32 / np as f32;
            let phi = if self.toroidal {
                (v - 0.5) * TAU
            } else {
                (v - 0.5) * PI
            };
            for i in 0..=nt {
                let u = i as f32 / nt as f32;
                let theta = (u - 0.5) * TAU;
                let (x, y, z) = if self.toroidal {
                    let c_phi = spow(phi.cos(), self.phi_roundness);
                    let s_phi = spow(phi.sin(), self.phi_roundness);
                    let c_th = spow(theta.cos(), self.theta_roundness);
                    let s_th = spow(theta.sin(), self.theta_roundness);
                    let fac = (r_major + c_phi) * self.size * 0.5 / r_major;
                    (fac * c_th, fac * s_th, s_phi * self.size * 0.5 / r_major)
                } else {
                    let c_phi = spow(phi.cos(), self.phi_roundness);
                    let s_phi = spow(phi.sin(), self.phi_roundness);
                    let c_th = spow(theta.cos(), self.theta_roundness);
                    let s_th = spow(theta.sin(), self.theta_roundness);
                    (
                        self.size * c_phi * c_th,
                        self.size * c_phi * s_th,
                        self.size * s_phi,
                    )
                };
                points.push([x, y, z]);
                scalars.push((x * x + y * y + z * z).sqrt());
            }
        }

        let row = nt + 1;
        for j in 0..np {
            for i in 0..nt {
                let a = (j * row + i) as u32;
                let b = (j * row + i + 1) as u32;
                let c = ((j + 1) * row + i + 1) as u32;
                let d = ((j + 1) * row + i) as u32;
                polys.push(vec![a, b, c]);
                polys.push(vec![a, c, d]);
            }
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.compute_normals();
        pd.point_data
            .set_scalars(DataArray::scalars_f32("Radial", scalars));
        pd
    }
}

impl Source for SuperquadricSource {
    fn name(&self) -> &str {
        "SuperquadricSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural 3D gyroid triply-periodic minimal surface volume (`ImageData`),
/// ideal for Marching Cubes isosurface extraction and volume raymarching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GyroidSource {
    pub dimensions: [usize; 3],
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub frequency: f64,
}

impl Default for GyroidSource {
    fn default() -> Self {
        Self {
            dimensions: [32, 32, 32],
            bounds_min: [-std::f64::consts::PI, -std::f64::consts::PI, -std::f64::consts::PI],
            bounds_max: [std::f64::consts::PI, std::f64::consts::PI, std::f64::consts::PI],
            frequency: 1.5,
        }
    }
}

impl GyroidSource {
    pub fn build(&self, time: f64) -> ImageData {
        let [nx, ny, nz] = self.dimensions;
        let nx = nx.max(2);
        let ny = ny.max(2);
        let nz = nz.max(2);
        let sx = (self.bounds_max[0] - self.bounds_min[0]) / (nx - 1) as f64;
        let sy = (self.bounds_max[1] - self.bounds_min[1]) / (ny - 1) as f64;
        let sz = (self.bounds_max[2] - self.bounds_min[2]) / (nz - 1) as f64;
        let phase = time * std::f64::consts::TAU;
        let freq = self.frequency;

        let total = nx * ny * nz;
        let values: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let k = idx / (nx * ny);
                let rem = idx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;
                let x = (self.bounds_min[0] + i as f64 * sx) * freq + phase;
                let y = (self.bounds_min[1] + j as f64 * sy) * freq;
                let z = (self.bounds_min[2] + k as f64 * sz) * freq;
                (x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()) as f32
            })
            .collect();

        let mut img = ImageData::new([nx, ny, nz], self.bounds_min, [sx, sy, sz]);
        img.point_data
            .set_scalars(DataArray::scalars_f32("Gyroid", values));
        img
    }
}

impl Source for GyroidSource {
    fn name(&self) -> &str {
        "GyroidSource"
    }

    fn execute(&self, time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::ImageData(self.build(time)))
    }
}

/// Procedural 3D point cloud source (equivalent to `vtkPointSource`), generating
/// a Fibonacci-sphere or volumetric spiral point distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointCloudSource {
    pub center: [f32; 3],
    pub radius: f32,
    pub num_points: usize,
    pub volumetric: bool,
}

impl Default for PointCloudSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            num_points: 1000,
            volumetric: true,
        }
    }
}

impl PointCloudSource {
    pub fn build(&self) -> PolyData {
        let n = self.num_points.max(1);
        let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
        let mut points = Vec::with_capacity(n);
        let mut verts = Vec::with_capacity(n);
        let mut radial = Vec::with_capacity(n);

        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let y = 1.0 - 2.0 * t;
            let r_xy = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden_angle * i as f32;
            let shell = if self.volumetric { t.cbrt() } else { 1.0 } * self.radius;
            let px = self.center[0] + r_xy * theta.cos() * shell;
            let py = self.center[1] + y * shell;
            let pz = self.center[2] + r_xy * theta.sin() * shell;
            points.push([px, py, pz]);
            verts.push(i as u32);
            radial.push(shell);
        }

        let mut pd = PolyData {
            points,
            verts,
            ..Default::default()
        };
        pd.point_data
            .set_scalars(DataArray::scalars_f32("Radius", radial));
        pd
    }
}

impl Source for PointCloudSource {
    fn name(&self) -> &str {
        "PointCloudSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}
