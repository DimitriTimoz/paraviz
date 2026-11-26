//! A self-contained dataset source for shareable projects.

use paraviz_core::{DataSet, PipelineError, PipelineInfo, Source};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Dataset materialized from an external reader at each of its advertised times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedSource {
    frames: Arc<[EmbeddedFrame]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct EmbeddedFrame {
    pub(crate) time: Option<f64>,
    pub(crate) dataset: Arc<DataSet>,
}

impl PartialEq for EmbeddedSource {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frames, &other.frames) || self.frames == other.frames
    }
}

impl EmbeddedSource {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new(frames: Vec<EmbeddedFrame>) -> Self {
        Self {
            frames: frames.into(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), PipelineError> {
        if let [EmbeddedFrame { time: None, .. }] = &*self.frames {
            return Ok(());
        }
        let mut previous = f64::NEG_INFINITY;
        if self.frames.is_empty()
            || !self.frames.iter().all(|frame| {
                let Some(time) = frame.time else { return false };
                let valid = time.is_finite() && time > previous;
                previous = time;
                valid
            })
        {
            return Err(PipelineError::ExecutionFailed(
                "Embedded data must have one static frame or strictly increasing finite time steps"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Source for EmbeddedSource {
    fn name(&self) -> &str {
        "Embedded Dataset"
    }

    fn request_information(&self) -> PipelineInfo {
        let time_steps: Vec<_> = self.frames.iter().filter_map(|frame| frame.time).collect();
        PipelineInfo {
            current_time: time_steps.first().copied().unwrap_or(0.0),
            time_steps,
            ..Default::default()
        }
    }

    fn execute(&self, time: f64) -> Result<DataSet, PipelineError> {
        Ok((*self.execute_arc(time)?).clone())
    }

    fn execute_arc(&self, time: f64) -> Result<Arc<DataSet>, PipelineError> {
        let frame = self
            .frames
            .iter()
            .min_by(|left, right| {
                let left = left.time.map_or(0.0, |t| (time - t).abs());
                let right = right.time.map_or(0.0, |t| (time - t).abs());
                left.total_cmp(&right)
            })
            .ok_or_else(|| PipelineError::ExecutionFailed("Embedded source is empty".into()))?;
        Ok(frame.dataset.clone())
    }
}
