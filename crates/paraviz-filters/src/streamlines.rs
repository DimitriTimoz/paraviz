//! Streamline integration filter (`StreamTracerFilter`, equivalent to `vtkStreamTracer`).

use crate::probe::CellDataToPointDataFilter;
use crate::util::{interpolate_field_data_weighted, SpatialSampler};
use glam::Vec3;
use paraviz_core::{
    ArrayData, DataArray, DataSet, Filter, MultiInputFilter, PipelineError, PolyData,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Numerical ODE integration scheme for streamlines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Integrator {
    /// Classical 4th-order Runge-Kutta.
    #[default]
    Rk4,
    /// Adaptive Dormand-Prince 5(4) Runge-Kutta.
    Rk45,
}

/// Seed geometry specification for streamline generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeedSource {
    Line {
        p1: [f32; 3],
        p2: [f32; 3],
        count: usize,
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
        count: usize,
    },
    Points(Vec<[f32; 3]>),
}

impl Default for SeedSource {
    fn default() -> Self {
        Self::Line {
            p1: [-1.0, 0.0, 0.0],
            p2: [1.0, 0.0, 0.0],
            count: 10,
        }
    }
}

impl SeedSource {
    /// Materializes the seed positions as world-space coordinates.
    pub fn seed_points(&self) -> Vec<[f32; 3]> {
        match self {
            Self::Line { p1, p2, count } => {
                let n = (*count).max(1);
                if n == 1 {
                    return vec![[
                        0.5 * (p1[0] + p2[0]),
                        0.5 * (p1[1] + p2[1]),
                        0.5 * (p1[2] + p2[2]),
                    ]];
                }
                let a = Vec3::from_array(*p1);
                let b = Vec3::from_array(*p2);
                (0..n)
                    .map(|i| a.lerp(b, i as f32 / (n - 1) as f32).to_array())
                    .collect()
            }
            Self::Sphere {
                center,
                radius,
                count,
            } => {
                let n = (*count).max(1);
                let c = Vec3::from_array(*center);
                if n == 1 {
                    return vec![(c + Vec3::new(*radius, 0.0, 0.0)).to_array()];
                }
                // Fibonacci sphere distribution
                let golden_angle = PI * (3.0 - 5.0f32.sqrt());
                (0..n)
                    .map(|i| {
                        let y = 1.0 - (2.0 * i as f32) / (n - 1) as f32;
                        let r_xy = (1.0 - y * y).max(0.0).sqrt();
                        let theta = golden_angle * i as f32;
                        let dir = Vec3::new(theta.cos() * r_xy, y, theta.sin() * r_xy);
                        (c + dir * *radius).to_array()
                    })
                    .collect()
            }
            Self::Points(pts) => pts.clone(),
        }
    }
}

/// Direction to integrate from each seed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntegrationDirection {
    #[default]
    Forward,
    Backward,
    Both,
}

/// Integrates streamlines through a 3D vector field using RK4 or adaptive RK45.
///
/// Outputs a [`PolyData`] of polylines with `IntegrationTime`, `ArcLength`, `Speed`,
/// and all interpolated `point_data` arrays from the input dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamTracerFilter {
    pub vector_field: String,
    pub seeds: SeedSource,
    pub direction: IntegrationDirection,
    pub integrator: Integrator,
    pub step_size: f32,
    pub max_steps: usize,
    pub max_length: f32,
    pub terminal_speed: f32,
}

impl Default for StreamTracerFilter {
    fn default() -> Self {
        Self {
            vector_field: String::new(),
            seeds: SeedSource::default(),
            direction: IntegrationDirection::Both,
            integrator: Integrator::Rk4,
            step_size: 0.05,
            max_steps: 500,
            max_length: 10.0,
            terminal_speed: 1e-6,
        }
    }
}

impl StreamTracerFilter {
    pub fn new(vector_field: impl Into<String>, seeds: SeedSource) -> Self {
        Self {
            vector_field: vector_field.into(),
            seeds,
            ..Default::default()
        }
    }
}

impl Filter for StreamTracerFilter {
    fn name(&self) -> &str {
        "StreamTracerFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        let seed_pts = self.seeds.seed_points();
        trace_streamlines(input, self, &seed_pts)
    }
}

/// Multi-input streamline tracer (`vtkStreamTracerWithCustomSource` equivalent).
///
/// Uses `inputs[0]` as the vector-field dataset (supporting both point and cell vector arrays)
/// and extracts seed coordinates from every point of `inputs[1]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamTracerWithCustomSource {
    pub base: StreamTracerFilter,
}

impl StreamTracerWithCustomSource {
    pub fn new(base: StreamTracerFilter) -> Self {
        Self { base }
    }
}

impl MultiInputFilter for StreamTracerWithCustomSource {
    fn name(&self) -> &str {
        "StreamTracerWithCustomSource"
    }

    fn apply_multi(&self, inputs: &[&DataSet]) -> Result<DataSet, PipelineError> {
        if inputs.len() < 2 {
            return Err(PipelineError::ExecutionFailed(
                "StreamTracerWithCustomSource requires 2 inputs (vector field, seed geometry)"
                    .into(),
            ));
        }
        let seeds = dataset_all_points(inputs[1]);
        if seeds.is_empty() {
            return Err(PipelineError::ExecutionFailed(
                "Seed dataset contains no points for StreamTracerWithCustomSource".into(),
            ));
        }
        trace_streamlines(inputs[0], &self.base, &seeds)
    }
}

fn dataset_all_points(ds: &DataSet) -> Vec<[f32; 3]> {
    match ds {
        DataSet::PolyData(pd) => pd.points.clone(),
        DataSet::ImageData(img) => img.all_points(),
        DataSet::RectilinearGrid(rg) => rg.all_points(),
        DataSet::StructuredGrid(sg) => sg.points.clone(),
        DataSet::UnstructuredGrid(ug) => ug.points.clone(),
        DataSet::MultiBlock(mb) => mb.blocks.iter().flat_map(|(_, b)| dataset_all_points(b)).collect(),
    }
}

fn has_point_vector_field(ds: &DataSet, field_name: &str) -> bool {
    let Some(pd) = ds.point_data() else {
        return false;
    };
    if field_name.is_empty() {
        pd.active_vectors_array().is_some_and(|a| a.num_components >= 3)
    } else {
        pd.get(field_name).is_some_and(|a| a.num_components >= 3)
    }
}

fn has_cell_vector_field(ds: &DataSet, field_name: &str) -> bool {
    let Some(cd) = ds.cell_data() else {
        return false;
    };
    if field_name.is_empty() {
        cd.active_vectors_array().is_some_and(|a| a.num_components >= 3)
            || cd.arrays().iter().any(|a| a.num_components >= 3)
    } else {
        cd.get(field_name).is_some_and(|a| a.num_components >= 3)
    }
}

fn trace_streamlines(
    input: &DataSet,
    cfg: &StreamTracerFilter,
    seed_pts: &[[f32; 3]],
) -> Result<DataSet, PipelineError> {
    let converted_cow;
    let working = if !has_point_vector_field(input, &cfg.vector_field)
        && has_cell_vector_field(input, &cfg.vector_field)
    {
        converted_cow = CellDataToPointDataFilter { pass_cell_data: true }.apply(input)?;
        &converted_cow
    } else {
        input
    };

    let sampler = SpatialSampler::new(working)?;
    let fd = sampler.field_data();
    let vec_arr = if cfg.vector_field.is_empty() {
        fd.active_vectors_array()
            .or_else(|| fd.arrays().iter().find(|a| a.num_components >= 3))
            .ok_or_else(|| {
                PipelineError::ExecutionFailed(
                    "No active vector array found for StreamTracerFilter".into(),
                )
            })?
    } else {
        fd.get(&cfg.vector_field).ok_or_else(|| {
            PipelineError::ExecutionFailed(format!(
                "Vector array '{}' not found in point_data or cell_data",
                cfg.vector_field
            ))
        })?
    };
    if vec_arr.num_components < 3 {
        return Err(PipelineError::ExecutionFailed(format!(
            "Vector field '{}' has {} components, expected >= 3",
            vec_arr.name, vec_arr.num_components
        )));
    }

    let step_size = cfg.step_size.max(1e-6);
    let tol = (working.bounds().diagonal() * 1e-4).max(1e-5);

        // Trace each seed in parallel.
        struct TraceResult {
            points: Vec<[f32; 3]>,
            time: Vec<f32>,
            arc: Vec<f32>,
            speed: Vec<f32>,
            weights: Vec<Option<Vec<(usize, f64)>>>,
        }

        let traces: Vec<TraceResult> = seed_pts
            .par_iter()
            .filter_map(|&seed| {
                let s_pos = Vec3::from_array(seed);
                let w0 = sampler.sample_weights(s_pos)?;
                let v0 = sampler.sample_vec3(vec_arr, s_pos)?;
                let sp0 = v0.length();
                if sp0 <= cfg.terminal_speed {
                    return None;
                }

                let integrate_one_way = |sign: f32| {
                    let mut pts = Vec::new();
                    let mut times = Vec::new();
                    let mut speeds = Vec::new();
                    let mut wts = Vec::new();

                    let mut pos = s_pos;
                    let mut cur_speed = sp0;
                    let mut cur_time = 0.0f32;
                    let mut cur_arc = 0.0f32;
                    let mut h = step_size;

                    for _ in 0..cfg.max_steps {
                        if cur_arc >= cfg.max_length || cur_speed <= cfg.terminal_speed {
                            break;
                        }
                        let rem_len = cfg.max_length - cur_arc;
                        let step_h = h.min(rem_len);

                        let dir_at = |p: Vec3| -> Option<(Vec3, f32)> {
                            let v = sampler.sample_vec3(vec_arr, p)?;
                            let sp = v.length();
                            if sp <= cfg.terminal_speed {
                                None
                            } else {
                                Some(((v / sp) * sign, sp))
                            }
                        };

                        let (next_pos, next_h) = match cfg.integrator {
                            Integrator::Rk4 => {
                                let Some(p_next) = step_rk4(pos, step_h, &dir_at) else {
                                    break;
                                };
                                (p_next, step_h)
                            }
                            Integrator::Rk45 => {
                                let Some((p_next, adapted_h)) =
                                    step_rk45(pos, step_h, step_size, tol, &dir_at)
                                else {
                                    break;
                                };
                                (p_next, adapted_h)
                            }
                        };

                        let Some(w_next) = sampler.sample_weights(next_pos) else {
                            break;
                        };
                        let Some(v_next) = sampler.sample_vec3(vec_arr, next_pos) else {
                            break;
                        };
                        let next_speed = v_next.length();
                        let ds = (next_pos - pos).length();
                        if ds < 1e-8 {
                            break;
                        }
                        let avg_speed = 0.5 * (cur_speed + next_speed).max(cfg.terminal_speed);
                        cur_arc += ds;
                        cur_time += sign * (ds / avg_speed);
                        pos = next_pos;
                        cur_speed = next_speed;
                        h = next_h;

                        pts.push(pos.to_array());
                        times.push(cur_time);
                        speeds.push(cur_speed);
                        wts.push(Some(w_next));
                    }
                    (pts, times, speeds, wts)
                };

                let mut points = Vec::new();
                let mut time = Vec::new();
                let mut speed = Vec::new();
                let mut weights = Vec::new();

                match cfg.direction {
                    IntegrationDirection::Forward => {
                        points.push(seed);
                        time.push(0.0);
                        speed.push(sp0);
                        weights.push(Some(w0));
                        let (fp, ft, fs, fw) = integrate_one_way(1.0);
                        points.extend(fp);
                        time.extend(ft);
                        speed.extend(fs);
                        weights.extend(fw);
                    }
                    IntegrationDirection::Backward => {
                        points.push(seed);
                        time.push(0.0);
                        speed.push(sp0);
                        weights.push(Some(w0));
                        let (bp, bt, bs, bw) = integrate_one_way(-1.0);
                        points.extend(bp);
                        time.extend(bt);
                        speed.extend(bs);
                        weights.extend(bw);
                    }
                    IntegrationDirection::Both => {
                        let (mut bp, mut bt, mut bs, mut bw) = integrate_one_way(-1.0);
                        bp.reverse();
                        bt.reverse();
                        bs.reverse();
                        bw.reverse();
                        points.extend(bp);
                        time.extend(bt);
                        speed.extend(bs);
                        weights.extend(bw);

                        points.push(seed);
                        time.push(0.0);
                        speed.push(sp0);
                        weights.push(Some(w0));

                        let (fp, ft, fs, fw) = integrate_one_way(1.0);
                        points.extend(fp);
                        time.extend(ft);
                        speed.extend(fs);
                        weights.extend(fw);
                    }
                }

                if points.len() < 2 {
                    return None;
                }

                let mut arc = Vec::with_capacity(points.len());
                let mut accum = 0.0f32;
                arc.push(0.0);
                for w in points.windows(2) {
                    accum += (Vec3::from_array(w[1]) - Vec3::from_array(w[0])).length();
                    arc.push(accum);
                }

                Some(TraceResult {
                    points,
                    time,
                    arc,
                    speed,
                    weights,
                })
            })
            .collect();

        let total_pts: usize = traces.iter().map(|t| t.points.len()).sum();
        let mut all_points = Vec::with_capacity(total_pts);
        let mut all_lines = Vec::with_capacity(traces.len());
        let mut all_time = Vec::with_capacity(total_pts);
        let mut all_arc = Vec::with_capacity(total_pts);
        let mut all_speed = Vec::with_capacity(total_pts);
        let mut all_weights: Vec<Option<Vec<(usize, f64)>>> = Vec::with_capacity(total_pts);

        for t in traces {
            let start = all_points.len() as u32;
            let count = t.points.len() as u32;
            all_points.extend(t.points);
            all_time.extend(t.time);
            all_arc.extend(t.arc);
            all_speed.extend(t.speed);
            all_weights.extend(t.weights);
            all_lines.push((start..(start + count)).collect());
        }

        let mut pd = PolyData {
            points: all_points,
            lines: all_lines,
            ..Default::default()
        };
        let mut pdata = interpolate_field_data_weighted(fd, &all_weights);
        pdata.add_array(DataArray::scalars_f32("IntegrationTime", all_time));
        pdata.add_array(DataArray::scalars_f32("ArcLength", all_arc));
        pdata.add_array(DataArray::new(
            "Speed",
            1,
            ArrayData::F32(all_speed),
        ));
        pdata.active_scalars = Some("Speed".into());
        pd.point_data = pdata;

        Ok(DataSet::PolyData(pd))
}

fn step_rk4<F>(pos: Vec3, h: f32, dir_at: &F) -> Option<Vec3>
where
    F: Fn(Vec3) -> Option<(Vec3, f32)>,
{
    let (k1, _) = dir_at(pos)?;
    let (k2, _) = dir_at(pos + k1 * (0.5 * h))?;
    let (k3, _) = dir_at(pos + k2 * (0.5 * h))?;
    let (k4, _) = dir_at(pos + k3 * h)?;
    Some(pos + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0))
}

/// Dormand-Prince RK45 adaptive step.
fn step_rk45<F>(
    pos: Vec3,
    mut h: f32,
    base_h: f32,
    tol: f32,
    dir_at: &F,
) -> Option<(Vec3, f32)>
where
    F: Fn(Vec3) -> Option<(Vec3, f32)>,
{
    let min_h = base_h * 0.05;
    let max_h = base_h * 4.0;
    let (k1, _) = dir_at(pos)?;

    for _ in 0..8 {
        let Some((k2, _)) = dir_at(pos + k1 * (h * (1.0 / 5.0))) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };
        let Some((k3, _)) = dir_at(pos + (k1 * (3.0 / 40.0) + k2 * (9.0 / 40.0)) * h) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };
        let Some((k4, _)) = dir_at(
            pos + (k1 * (44.0 / 45.0) - k2 * (56.0 / 15.0) + k3 * (32.0 / 9.0)) * h,
        ) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };
        let Some((k5, _)) = dir_at(
            pos + (k1 * (19372.0 / 6561.0) - k2 * (25360.0 / 2187.0)
                + k3 * (64448.0 / 6561.0)
                - k4 * (212.0 / 729.0))
                * h,
        ) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };
        let Some((k6, _)) = dir_at(
            pos + (k1 * (9017.0 / 3168.0) - k2 * (355.0 / 33.0)
                + k3 * (46732.0 / 5247.0)
                + k4 * (49.0 / 176.0)
                - k5 * (5103.0 / 18656.0))
                * h,
        ) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };

        let y5 = pos
            + (k1 * (35.0 / 384.0)
                + k3 * (500.0 / 1113.0)
                + k4 * (125.0 / 192.0)
                - k5 * (2187.0 / 6784.0)
                + k6 * (11.0 / 84.0))
                * h;

        let Some((k7, _)) = dir_at(y5) else {
            h *= 0.5;
            if h < min_h {
                return None;
            }
            continue;
        };

        let y4 = pos
            + (k1 * (5179.0 / 57600.0)
                + k3 * (7571.0 / 16695.0)
                + k4 * (393.0 / 640.0)
                - k5 * (92097.0 / 339200.0)
                + k6 * (187.0 / 2100.0)
                + k7 * (1.0 / 40.0))
                * h;

        let err = (y5 - y4).length();
        if err <= tol || h <= min_h {
            let next_h = if err > 1e-12 {
                (0.9 * h * (tol / err).powf(0.2)).clamp(min_h, max_h)
            } else {
                (h * 1.5).clamp(min_h, max_h)
            };
            return Some((y5, next_h));
        }

        h = (0.9 * h * (tol / err).powf(0.25)).max(min_h);
    }

    step_rk4(pos, h, dir_at).map(|p| (p, h))
}
