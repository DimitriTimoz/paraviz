//! Samples one point-data component across the time steps of a pipeline node.

use crate::session::SessionError;
use paraviz_core::{NodeId, PipelineError, PipelineGraph};
use paraviz_filters::sample_point_component;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

/// A fixed probe position and point-data component. Components are zero-based.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalProbeRequest {
    pub position: [f32; 3],
    pub field: String,
    pub component: usize,
}

/// An independent evaluation of a node's upstream pipeline.
pub struct TemporalProbe {
    pub(crate) graph: PipelineGraph,
    pub(crate) node: NodeId,
    pub(crate) times: Vec<f64>,
    pub(crate) request: TemporalProbeRequest,
}

/// A measurement at one advertised time. A missing sample has no numeric value.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalSample {
    pub time: f64,
    pub value: Option<f64>,
}

/// Time-ordered measurements at the requested position.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalProbeResult {
    pub request: TemporalProbeRequest,
    pub samples: Vec<TemporalSample>,
}

impl TemporalProbe {
    /// Evaluates every time step without changing the parent session's time or caches.
    pub fn run(mut self) -> Result<TemporalProbeResult, SessionError> {
        let mut samples = Vec::with_capacity(self.times.len());
        for time in self.times {
            let dataset = self.graph.update(self.node, time).map_err(|err| {
                PipelineError::ExecutionFailed(format!("Temporal probe at time {time}: {err}"))
            })?;
            let value = sample_point_component(
                &dataset,
                self.request.position,
                &self.request.field,
                self.request.component,
            );
            samples.push(TemporalSample { time, value });
        }
        Ok(TemporalProbeResult {
            request: self.request,
            samples,
        })
    }
}

impl TemporalProbeResult {
    /// Encodes all time steps as CSV. Missing samples have an empty value and `missing` status.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("time,value,status,field,component,x,y,z\n");
        let field = self.request.field.replace('"', "\"\"");
        let component = self.request.component;
        let [x, y, z] = self.request.position;
        for sample in &self.samples {
            let (value, status) = match sample.value {
                Some(value) => (value.to_string(), "ok"),
                None => (String::new(), "missing"),
            };
            writeln!(
                csv,
                "{},{value},{status},\"{field}\",{component},{x},{y},{z}",
                sample.time
            )
            .expect("writing to a String cannot fail");
        }
        csv
    }
}
