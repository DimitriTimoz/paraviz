//! Procedural polygonal geometric sources (`SphereSource`, `CubeSource`, `ConeSource`,
//! `CylinderSource`, `PlaneSource`, `LineSource`, `ArrowSource`).

use glam::{Quat, Vec3};
use paraviz_core::{DataArray, DataSet, PipelineError, PolyData, Source};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

/// Procedural UV sphere source (equivalent to `vtkSphereSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphereSource {
    pub center: [f32; 3],
    pub radius: f32,
    pub theta_resolution: usize,
    pub phi_resolution: usize,
}

impl Default for SphereSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius: 0.5,
            theta_resolution: 32,
            phi_resolution: 20,
        }
    }
}

impl SphereSource {
    pub fn build(&self) -> PolyData {
        let n_theta = self.theta_resolution.max(3);
        let n_phi = self.phi_resolution.max(3);
        let c = Vec3::from_array(self.center);

        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut elevation = Vec::new();
        let mut polys = Vec::new();

        for j in 0..=n_phi {
            let v = j as f32 / n_phi as f32;
            let phi = v * PI;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            for i in 0..=n_theta {
                let u = i as f32 / n_theta as f32;
                let theta = u * TAU;
                let n = Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin());
                let p = c + n * self.radius;
                points.push(p.to_array());
                normals.push(n.to_array());
                elevation.push(p.y);
            }
        }

        let row = n_theta + 1;
        for j in 0..n_phi {
            for i in 0..n_theta {
                let a = (j * row + i) as u32;
                let b = (j * row + i + 1) as u32;
                let c_idx = ((j + 1) * row + i + 1) as u32;
                let d = ((j + 1) * row + i) as u32;
                if j > 0 {
                    polys.push(vec![a, b, c_idx]);
                }
                if j < n_phi - 1 {
                    polys.push(vec![a, c_idx, d]);
                }
            }
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.point_data
            .set_normals(DataArray::vectors_f32("Normals", normals));
        pd.point_data
            .set_scalars(DataArray::scalars_f32("Elevation", elevation));
        pd
    }
}

impl Source for SphereSource {
    fn name(&self) -> &str {
        "SphereSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural axis-aligned box / cube source (equivalent to `vtkCubeSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CubeSource {
    pub center: [f32; 3],
    pub lengths: [f32; 3],
}

impl Default for CubeSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            lengths: [1.0, 1.0, 1.0],
        }
    }
}

impl CubeSource {
    pub fn build(&self) -> PolyData {
        let hx = self.lengths[0] * 0.5;
        let hy = self.lengths[1] * 0.5;
        let hz = self.lengths[2] * 0.5;
        let c = Vec3::from_array(self.center);

        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([1.0, 0.0, 0.0], [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]]),
            ([-1.0, 0.0, 0.0], [[-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, -hz]]),
            ([0.0, 1.0, 0.0], [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]]),
            ([0.0, -1.0, 0.0], [[-hx, -hy, hz], [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz]]),
            ([0.0, 0.0, 1.0], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]),
            ([0.0, 0.0, -1.0], [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]]),
        ];

        let mut points = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut face_ids = Vec::with_capacity(24);
        let mut polys = Vec::with_capacity(12);

        for (fid, (n, corners)) in faces.iter().enumerate() {
            let base = points.len() as u32;
            for corner in corners {
                points.push((c + Vec3::from_array(*corner)).to_array());
                normals.push(*n);
                face_ids.push(fid as f32);
            }
            polys.push(vec![base, base + 1, base + 2]);
            polys.push(vec![base, base + 2, base + 3]);
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.point_data
            .set_normals(DataArray::vectors_f32("Normals", normals));
        pd.point_data
            .set_scalars(DataArray::scalars_f32("FaceId", face_ids));
        pd
    }
}

impl Source for CubeSource {
    fn name(&self) -> &str {
        "CubeSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural cone source (equivalent to `vtkConeSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConeSource {
    pub center: [f32; 3],
    pub direction: [f32; 3],
    pub height: f32,
    pub radius: f32,
    pub resolution: usize,
    pub capping: bool,
}

impl Default for ConeSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
            height: 1.0,
            radius: 0.5,
            resolution: 32,
            capping: true,
        }
    }
}

impl ConeSource {
    pub fn build(&self) -> PolyData {
        let res = self.resolution.max(3);
        let dir = Vec3::from_array(self.direction).normalize_or(Vec3::X);
        let rot = Quat::from_rotation_arc(Vec3::X, dir);
        let center = Vec3::from_array(self.center);

        let apex = center + rot * Vec3::new(self.height * 0.5, 0.0, 0.0);
        let base_center = center + rot * Vec3::new(-self.height * 0.5, 0.0, 0.0);

        let mut points = Vec::new();
        let mut polys = Vec::new();

        let apex_idx = 0u32;
        points.push(apex.to_array());

        for i in 0..res {
            let ang = (i as f32 / res as f32) * TAU;
            let local = Vec3::new(-self.height * 0.5, self.radius * ang.cos(), self.radius * ang.sin());
            points.push((center + rot * local).to_array());
        }

        for i in 0..res {
            let b0 = 1 + i as u32;
            let b1 = 1 + ((i + 1) % res) as u32;
            polys.push(vec![apex_idx, b0, b1]);
        }

        if self.capping {
            let cap_center_idx = points.len() as u32;
            points.push(base_center.to_array());
            for i in 0..res {
                let b0 = 1 + i as u32;
                let b1 = 1 + ((i + 1) % res) as u32;
                polys.push(vec![cap_center_idx, b1, b0]);
            }
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.compute_normals();
        let elev: Vec<f32> = pd.points.iter().map(|p| p[0]).collect();
        pd.point_data
            .set_scalars(DataArray::scalars_f32("AxisCoord", elev));
        pd
    }
}

impl Source for ConeSource {
    fn name(&self) -> &str {
        "ConeSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural cylinder source aligned along Y axis by default (equivalent to `vtkCylinderSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CylinderSource {
    pub center: [f32; 3],
    pub height: f32,
    pub radius: f32,
    pub resolution: usize,
    pub capping: bool,
}

impl Default for CylinderSource {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            height: 1.0,
            radius: 0.5,
            resolution: 32,
            capping: true,
        }
    }
}

impl CylinderSource {
    pub fn build(&self) -> PolyData {
        let res = self.resolution.max(3);
        let c = Vec3::from_array(self.center);
        let hy = self.height * 0.5;

        let mut points = Vec::new();
        let mut polys = Vec::new();

        for i in 0..res {
            let ang = (i as f32 / res as f32) * TAU;
            let x = self.radius * ang.cos();
            let z = self.radius * ang.sin();
            points.push((c + Vec3::new(x, -hy, z)).to_array());
            points.push((c + Vec3::new(x, hy, z)).to_array());
        }

        for i in 0..res {
            let next = (i + 1) % res;
            let b0 = (2 * i) as u32;
            let t0 = (2 * i + 1) as u32;
            let b1 = (2 * next) as u32;
            let t1 = (2 * next + 1) as u32;
            polys.push(vec![b0, b1, t1]);
            polys.push(vec![b0, t1, t0]);
        }

        if self.capping {
            let bot_c = points.len() as u32;
            points.push((c + Vec3::new(0.0, -hy, 0.0)).to_array());
            let top_c = points.len() as u32;
            points.push((c + Vec3::new(0.0, hy, 0.0)).to_array());
            for i in 0..res {
                let next = (i + 1) % res;
                let b0 = (2 * i) as u32;
                let t0 = (2 * i + 1) as u32;
                let b1 = (2 * next) as u32;
                let t1 = (2 * next + 1) as u32;
                polys.push(vec![bot_c, b1, b0]);
                polys.push(vec![top_c, t0, t1]);
            }
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.compute_normals();
        let elev: Vec<f32> = pd.points.iter().map(|p| p[1]).collect();
        pd.point_data
            .set_scalars(DataArray::scalars_f32("Elevation", elev));
        pd
    }
}

impl Source for CylinderSource {
    fn name(&self) -> &str {
        "CylinderSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural subdivided plane source (equivalent to `vtkPlaneSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaneSource {
    pub origin: [f32; 3],
    pub point1: [f32; 3],
    pub point2: [f32; 3],
    pub x_resolution: usize,
    pub y_resolution: usize,
}

impl Default for PlaneSource {
    fn default() -> Self {
        Self {
            origin: [-0.5, -0.5, 0.0],
            point1: [0.5, -0.5, 0.0],
            point2: [-0.5, 0.5, 0.0],
            x_resolution: 10,
            y_resolution: 10,
        }
    }
}

impl PlaneSource {
    pub fn build(&self) -> PolyData {
        let nx = self.x_resolution.max(1);
        let ny = self.y_resolution.max(1);
        let o = Vec3::from_array(self.origin);
        let u_vec = Vec3::from_array(self.point1) - o;
        let v_vec = Vec3::from_array(self.point2) - o;
        let normal = u_vec.cross(v_vec).normalize_or(Vec3::Z).to_array();

        let mut points = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut normals = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut u_coord = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut polys = Vec::with_capacity(nx * ny * 2);

        for j in 0..=ny {
            let v = j as f32 / ny as f32;
            for i in 0..=nx {
                let u = i as f32 / nx as f32;
                let p = o + u_vec * u + v_vec * v;
                points.push(p.to_array());
                normals.push(normal);
                u_coord.push((u * u + v * v).sqrt());
            }
        }

        let row = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
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
        pd.point_data
            .set_normals(DataArray::vectors_f32("Normals", normals));
        pd.point_data
            .set_scalars(DataArray::scalars_f32("RadialDist", u_coord));
        pd
    }
}

impl Source for PlaneSource {
    fn name(&self) -> &str {
        "PlaneSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural line segment source (equivalent to `vtkLineSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineSource {
    pub point1: [f32; 3],
    pub point2: [f32; 3],
    pub resolution: usize,
}

impl Default for LineSource {
    fn default() -> Self {
        Self {
            point1: [-0.5, 0.0, 0.0],
            point2: [0.5, 0.0, 0.0],
            resolution: 16,
        }
    }
}

impl LineSource {
    pub fn build(&self) -> PolyData {
        let n = self.resolution.max(1);
        let p1 = Vec3::from_array(self.point1);
        let p2 = Vec3::from_array(self.point2);
        let mut points = Vec::with_capacity(n + 1);
        let mut line = Vec::with_capacity(n + 1);
        let mut arc = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            points.push(p1.lerp(p2, t).to_array());
            line.push(i as u32);
            arc.push(t);
        }
        let mut pd = PolyData {
            points,
            lines: vec![line],
            ..Default::default()
        };
        pd.point_data
            .set_scalars(DataArray::scalars_f32("ArcLength", arc));
        pd
    }
}

impl Source for LineSource {
    fn name(&self) -> &str {
        "LineSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}

/// Procedural 3D arrow source pointing along +X from origin `(0,0,0)` to `(1,0,0)`
/// (equivalent to `vtkArrowSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrowSource {
    pub tip_length: f32,
    pub tip_radius: f32,
    pub tip_resolution: usize,
    pub shaft_radius: f32,
    pub shaft_resolution: usize,
}

impl Default for ArrowSource {
    fn default() -> Self {
        Self {
            tip_length: 0.35,
            tip_radius: 0.10,
            tip_resolution: 12,
            shaft_radius: 0.035,
            shaft_resolution: 12,
        }
    }
}

impl ArrowSource {
    pub fn build(&self) -> PolyData {
        let shaft_len = (1.0 - self.tip_length).max(0.05);
        let s_res = self.shaft_resolution.max(4);
        let t_res = self.tip_resolution.max(4);

        let mut points = Vec::new();
        let mut polys = Vec::new();

        // Shaft cylinder from x = 0 to x = shaft_len
        for i in 0..s_res {
            let ang = (i as f32 / s_res as f32) * TAU;
            let y = self.shaft_radius * ang.cos();
            let z = self.shaft_radius * ang.sin();
            points.push([0.0, y, z]);
            points.push([shaft_len, y, z]);
        }
        for i in 0..s_res {
            let next = (i + 1) % s_res;
            let a = (2 * i) as u32;
            let b = (2 * i + 1) as u32;
            let c = (2 * next + 1) as u32;
            let d = (2 * next) as u32;
            polys.push(vec![a, d, c]);
            polys.push(vec![a, c, b]);
        }

        // Tip cone from x = shaft_len to x = 1.0
        let tip_base_start = points.len() as u32;
        for i in 0..t_res {
            let ang = (i as f32 / t_res as f32) * TAU;
            let y = self.tip_radius * ang.cos();
            let z = self.tip_radius * ang.sin();
            points.push([shaft_len, y, z]);
        }
        let apex_idx = points.len() as u32;
        points.push([1.0, 0.0, 0.0]);
        let cap_idx = points.len() as u32;
        points.push([shaft_len, 0.0, 0.0]);

        for i in 0..t_res {
            let next = (i + 1) % t_res;
            let b0 = tip_base_start + i as u32;
            let b1 = tip_base_start + next as u32;
            polys.push(vec![apex_idx, b1, b0]);
            polys.push(vec![cap_idx, b0, b1]);
        }

        let mut pd = PolyData {
            points,
            polys,
            ..Default::default()
        };
        pd.compute_normals();
        pd
    }
}

impl Source for ArrowSource {
    fn name(&self) -> &str {
        "ArrowSource"
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok(DataSet::PolyData(self.build()))
    }
}
