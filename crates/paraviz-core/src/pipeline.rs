//! Demand-driven visualization pipeline (equivalent to VTK's `vtkAlgorithm` + `vtkExecutive`).

use crate::dataset::{BoundingBox, DataSet};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Unique handle to a node inside a [`PipelineGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// Errors emitted during pipeline registration or execution.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Pipeline node {0:?} does not exist")]
    MissingNode(NodeId),

    #[error("Node {node:?} expects {expected} input(s), got {actual}")]
    InputCountMismatch {
        node: NodeId,
        expected: usize,
        actual: usize,
    },

    #[error("Cycle detected in pipeline graph at {0:?}")]
    CycleDetected(NodeId),

    #[error("Filter execution failed: {0}")]
    ExecutionFailed(String),
}

/// Lightweight metadata propagated during the `RequestInformation` pass.
#[derive(Debug, Clone, Default)]
pub struct PipelineInfo {
    pub time_steps: Vec<f64>,
    pub current_time: f64,
    pub bounds: Option<BoundingBox>,
}

/// A producer algorithm with zero inputs (readers and procedural sources).
pub trait Source: Send + Sync {
    /// Human-readable algorithm name.
    fn name(&self) -> &str;

    /// Lightweight metadata request (does not need to allocate heavy buffers).
    fn request_information(&self) -> PipelineInfo {
        PipelineInfo::default()
    }

    /// Executes the source for the requested time step.
    fn execute(&self, time: f64) -> Result<DataSet, PipelineError>;
}

/// A transformation algorithm taking one dataset and producing a transformed dataset.
pub trait Filter: Send + Sync {
    /// Human-readable algorithm name.
    fn name(&self) -> &str;

    /// Executes the filter on the given input dataset.
    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError>;
}

pub type BoxedSource = Box<dyn Source>;
pub type BoxedFilter = Box<dyn Filter>;

enum NodeKind {
    Source(BoxedSource),
    Filter {
        filter: BoxedFilter,
        input: NodeId,
    },
}

struct PipelineNodeEntry {
    label: String,
    kind: NodeKind,
    dirty: bool,
    cached_output: Option<Arc<DataSet>>,
    cached_time: f64,
}

/// Demand-driven DAG executive with dirty propagation and output caching.
#[derive(Default)]
pub struct PipelineGraph {
    next_id: u64,
    nodes: HashMap<NodeId, PipelineNodeEntry>,
    dependents: HashMap<NodeId, Vec<NodeId>>,
}

impl PipelineGraph {
    /// Creates an empty pipeline graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a source node and returns its [`NodeId`].
    pub fn add_source(&mut self, label: impl Into<String>, source: BoxedSource) -> NodeId {
        self.next_id += 1;
        let id = NodeId(self.next_id);
        self.nodes.insert(
            id,
            PipelineNodeEntry {
                label: label.into(),
                kind: NodeKind::Source(source),
                dirty: true,
                cached_output: None,
                cached_time: 0.0,
            },
        );
        id
    }

    /// Registers a filter node connected to `input`.
    pub fn add_filter(
        &mut self,
        label: impl Into<String>,
        input: NodeId,
        filter: BoxedFilter,
    ) -> Result<NodeId, PipelineError> {
        if !self.nodes.contains_key(&input) {
            return Err(PipelineError::MissingNode(input));
        }
        self.next_id += 1;
        let id = NodeId(self.next_id);
        self.nodes.insert(
            id,
            PipelineNodeEntry {
                label: label.into(),
                kind: NodeKind::Filter { filter, input },
                dirty: true,
                cached_output: None,
                cached_time: 0.0,
            },
        );
        self.dependents.entry(input).or_default().push(id);
        Ok(id)
    }

    /// Replaces a source implementation and invalidates downstream caches.
    pub fn update_source(&mut self, id: NodeId, source: BoxedSource) -> Result<(), PipelineError> {
        let entry = self
            .nodes
            .get_mut(&id)
            .ok_or(PipelineError::MissingNode(id))?;
        entry.kind = NodeKind::Source(source);
        self.mark_dirty(id);
        Ok(())
    }

    /// Replaces a filter implementation and invalidates downstream caches.
    pub fn update_filter(&mut self, id: NodeId, filter: BoxedFilter) -> Result<(), PipelineError> {
        let entry = self
            .nodes
            .get_mut(&id)
            .ok_or(PipelineError::MissingNode(id))?;
        let input = match &entry.kind {
            NodeKind::Filter { input, .. } => *input,
            NodeKind::Source(_) => {
                return Err(PipelineError::ExecutionFailed(
                    "Cannot update source node with a filter".into(),
                ))
            }
        };
        entry.kind = NodeKind::Filter { filter, input };
        self.mark_dirty(id);
        Ok(())
    }

    /// Marks `id` and all transitive downstream dependents as dirty.
    pub fn mark_dirty(&mut self, id: NodeId) {
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            if let Some(entry) = self.nodes.get_mut(&cur) {
                entry.dirty = true;
                entry.cached_output = None;
            }
            if let Some(children) = self.dependents.get(&cur) {
                stack.extend(children.iter().copied());
            }
        }
    }

    /// Removes a node (and recursively detaches/removes any dependents).
    pub fn remove_node(&mut self, id: NodeId) -> Vec<NodeId> {
        let mut to_remove = vec![id];
        let mut i = 0;
        while i < to_remove.len() {
            let cur = to_remove[i];
            if let Some(children) = self.dependents.get(&cur) {
                to_remove.extend(children.iter().copied());
            }
            i += 1;
        }
        for rem in &to_remove {
            self.nodes.remove(rem);
            self.dependents.remove(rem);
        }
        for children in self.dependents.values_mut() {
            children.retain(|c| !to_remove.contains(c));
        }
        to_remove
    }

    /// Demand-driven execution: evaluates only dirty upstream nodes at `time` and returns cached output.
    pub fn update(&mut self, id: NodeId, time: f64) -> Result<Arc<DataSet>, PipelineError> {
        let mut visiting = Vec::new();
        self.update_recursive(id, time, &mut visiting)
    }

    fn update_recursive(
        &mut self,
        id: NodeId,
        time: f64,
        visiting: &mut Vec<NodeId>,
    ) -> Result<Arc<DataSet>, PipelineError> {
        if visiting.contains(&id) {
            return Err(PipelineError::CycleDetected(id));
        }
        visiting.push(id);

        let (needs_eval, input_id) = {
            let entry = self.nodes.get(&id).ok_or(PipelineError::MissingNode(id))?;
            let time_changed = (entry.cached_time - time).abs() > 1e-12;
            let needs_eval = entry.dirty || entry.cached_output.is_none() || time_changed;
            let input_id = match &entry.kind {
                NodeKind::Source(_) => None,
                NodeKind::Filter { input, .. } => Some(*input),
            };
            (needs_eval, input_id)
        };

        let upstream = if let Some(in_id) = input_id {
            Some(self.update_recursive(in_id, time, visiting)?)
        } else {
            None
        };

        let result = if !needs_eval {
            self.nodes
                .get(&id)
                .and_then(|e| e.cached_output.clone())
                .ok_or(PipelineError::MissingNode(id))?
        } else {
            let entry = self.nodes.get(&id).ok_or(PipelineError::MissingNode(id))?;
            let ds = match &entry.kind {
                NodeKind::Source(src) => src.execute(time)?,
                NodeKind::Filter { filter, .. } => {
                    let input_ds = upstream
                        .as_ref()
                        .ok_or(PipelineError::MissingNode(id))?;
                    filter.apply(input_ds)?
                }
            };
            let arc = Arc::new(ds);
            let entry_mut = self.nodes.get_mut(&id).unwrap();
            entry_mut.cached_output = Some(arc.clone());
            entry_mut.dirty = false;
            entry_mut.cached_time = time;
            arc
        };

        visiting.pop();
        Ok(result)
    }

    /// Returns the human-readable label for a node.
    pub fn label(&self, id: NodeId) -> Option<&str> {
        self.nodes.get(&id).map(|e| e.label.as_str())
    }

    /// Returns all registered node IDs sorted deterministically.
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Returns the parent node ID if `id` is a filter node.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        match &self.nodes.get(&id)?.kind {
            NodeKind::Source(_) => None,
            NodeKind::Filter { input, .. } => Some(*input),
        }
    }
}
