//! High-level stateful ParaView-style pipeline, scene, and rendering session API.

use paraviz_core::{
    ColorMap, ColorMapPreset, DataSet, Filter, NodeId, PipelineError, PipelineGraph,
    PipelineInfo, Source, TransferFunction,
};
use paraviz_filters::{
    CalculatorFilter, CellDataToPointDataFilter, ClipFilter, ContourFilter, ExtractSurface,
    GlyphFilter, GradientFilter, PointDataToCellDataFilter, ProbeOverLineFilter, SliceFilter,
    SmoothFilter, StreamTracerFilter, ThresholdFilter, TransformFilter, TubeFilter,
    WarpByScalarFilter, WarpByVectorFilter,
};
use paraviz_io::{read_dataset, write_dataset, IoError, PvdSeriesSource, VtkFileSource};
use paraviz_render::{
    ActorStyle, Camera, CameraPreset, ColorBy, RenderError, RenderScene, Representation,
    SceneActor, SceneSettings, WebGpuRenderer,
};
use paraviz_sources::{
    ArrowSource, ConeSource, CubeSource, CylinderSource, DiskSource, GyroidSource, LineSource,
    PlaneSource, PointCloudSource, SphereSource, SuperquadricSource, TorusSource,
    VectorFieldSource, WaveletSource,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// Unified error type for high-level [`ParaViz`] operations.
#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    #[error("WebGPU render error: {0}")]
    Render(#[from] RenderError),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Node {0:?} not found in session")]
    MissingNode(NodeId),
}

/// Wrapper turning an in-memory [`DataSet`] into a pipeline [`Source`].
#[derive(Debug, Clone)]
pub struct InMemorySource {
    pub label: String,
    pub dataset: Arc<DataSet>,
}

impl Source for InMemorySource {
    fn name(&self) -> &str {
        &self.label
    }

    fn request_information(&self) -> PipelineInfo {
        PipelineInfo {
            bounds: Some(self.dataset.bounds()),
            ..Default::default()
        }
    }

    fn execute(&self, _time: f64) -> Result<DataSet, PipelineError> {
        Ok((*self.dataset).clone())
    }
}

/// Serializable source configuration mirroring ParaView's Sources & Readers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceConfig {
    Sphere(SphereSource),
    Cube(CubeSource),
    Cone(ConeSource),
    Cylinder(CylinderSource),
    Plane(PlaneSource),
    Line(LineSource),
    Arrow(ArrowSource),
    Disk(DiskSource),
    Torus(TorusSource),
    Wavelet(WaveletSource),
    VectorField(VectorFieldSource),
    Superquadric(SuperquadricSource),
    Gyroid(GyroidSource),
    PointCloud(PointCloudSource),
    File(PathBuf),
    PvdSeries(PathBuf),
}

impl SourceConfig {
    /// Human-readable default label for the pipeline browser.
    pub fn default_label(&self) -> String {
        match self {
            Self::Sphere(_) => "Sphere".into(),
            Self::Cube(_) => "Cube".into(),
            Self::Cone(_) => "Cone".into(),
            Self::Cylinder(_) => "Cylinder".into(),
            Self::Plane(_) => "Plane".into(),
            Self::Line(_) => "Line".into(),
            Self::Arrow(_) => "Arrow".into(),
            Self::Disk(_) => "Disk".into(),
            Self::Torus(_) => "Torus".into(),
            Self::Wavelet(_) => "Wavelet".into(),
            Self::VectorField(_) => "VectorField".into(),
            Self::Superquadric(_) => "Superquadric".into(),
            Self::Gyroid(_) => "Gyroid".into(),
            Self::PointCloud(_) => "PointCloud".into(),
            Self::File(p) | Self::PvdSeries(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("Dataset")
                .to_string(),
        }
    }

    /// Instantiates a boxed [`Source`] from this configuration.
    pub fn build_source(&self) -> Result<Box<dyn Source>, SessionError> {
        let src: Box<dyn Source> = match self {
            Self::Sphere(s) => Box::new(s.clone()),
            Self::Cube(s) => Box::new(s.clone()),
            Self::Cone(s) => Box::new(s.clone()),
            Self::Cylinder(s) => Box::new(s.clone()),
            Self::Plane(s) => Box::new(s.clone()),
            Self::Line(s) => Box::new(s.clone()),
            Self::Arrow(s) => Box::new(s.clone()),
            Self::Disk(s) => Box::new(s.clone()),
            Self::Torus(s) => Box::new(s.clone()),
            Self::Wavelet(s) => Box::new(s.clone()),
            Self::VectorField(s) => Box::new(s.clone()),
            Self::Superquadric(s) => Box::new(s.clone()),
            Self::Gyroid(s) => Box::new(s.clone()),
            Self::PointCloud(s) => Box::new(s.clone()),
            Self::File(p) => Box::new(VtkFileSource::new(p)),
            Self::PvdSeries(p) => Box::new(PvdSeriesSource::new(p)?),
        };
        Ok(src)
    }
}

/// Serializable filter configuration mirroring ParaView's Filters menu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterConfig {
    ExtractSurface(ExtractSurface),
    Contour(ContourFilter),
    Slice(SliceFilter),
    Clip(ClipFilter),
    Threshold(ThresholdFilter),
    StreamTracer(StreamTracerFilter),
    Tube(TubeFilter),
    Glyph(GlyphFilter),
    WarpByVector(WarpByVectorFilter),
    WarpByScalar(WarpByScalarFilter),
    Gradient(GradientFilter),
    Calculator(CalculatorFilter),
    Smooth(SmoothFilter),
    Transform(TransformFilter),
    CellDataToPointData(CellDataToPointDataFilter),
    PointDataToCellData(PointDataToCellDataFilter),
    ProbeOverLine(ProbeOverLineFilter),
}

impl FilterConfig {
    /// Human-readable default label for the pipeline browser.
    pub fn default_label(&self) -> &'static str {
        match self {
            Self::ExtractSurface(_) => "ExtractSurface",
            Self::Contour(_) => "Contour",
            Self::Slice(_) => "Slice",
            Self::Clip(_) => "Clip",
            Self::Threshold(_) => "Threshold",
            Self::StreamTracer(_) => "StreamTracer",
            Self::Tube(_) => "Tube",
            Self::Glyph(_) => "Glyph",
            Self::WarpByVector(_) => "WarpByVector",
            Self::WarpByScalar(_) => "WarpByScalar",
            Self::Gradient(_) => "Gradient",
            Self::Calculator(_) => "Calculator",
            Self::Smooth(_) => "Smooth",
            Self::Transform(_) => "Transform",
            Self::CellDataToPointData(_) => "CellDataToPointData",
            Self::PointDataToCellData(_) => "PointDataToCellData",
            Self::ProbeOverLine(_) => "ProbeOverLine",
        }
    }

    /// Instantiates a boxed [`Filter`] from this configuration.
    pub fn build_filter(&self) -> Box<dyn Filter> {
        match self {
            Self::ExtractSurface(f) => Box::new(f.clone()),
            Self::Contour(f) => Box::new(f.clone()),
            Self::Slice(f) => Box::new(f.clone()),
            Self::Clip(f) => Box::new(f.clone()),
            Self::Threshold(f) => Box::new(f.clone()),
            Self::StreamTracer(f) => Box::new(f.clone()),
            Self::Tube(f) => Box::new(f.clone()),
            Self::Glyph(f) => Box::new(f.clone()),
            Self::WarpByVector(f) => Box::new(f.clone()),
            Self::WarpByScalar(f) => Box::new(f.clone()),
            Self::Gradient(f) => Box::new(f.clone()),
            Self::Calculator(f) => Box::new(f.clone()),
            Self::Smooth(f) => Box::new(f.clone()),
            Self::Transform(f) => Box::new(f.clone()),
            Self::CellDataToPointData(f) => Box::new(f.clone()),
            Self::PointDataToCellData(f) => Box::new(f.clone()),
            Self::ProbeOverLine(f) => Box::new(f.clone()),
        }
    }
}

/// Kind of a node stored in [`ParaViz`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKindConfig {
    Source(Option<SourceConfig>),
    Filter {
        input: NodeIdSerialized,
        config: Option<FilterConfig>,
    },
}

/// Helper alias for serializing node indices cleanly.
pub type NodeIdSerialized = u64;

/// Metadata and display state for a single pipeline node.
#[derive(Debug, Clone)]
pub struct SessionNode {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    pub source_config: Option<SourceConfig>,
    pub filter_config: Option<FilterConfig>,
    pub style: ActorStyle,
    pub cached_dataset: Option<Arc<DataSet>>,
    pub time_steps: Vec<f64>,
    auto_styled: bool,
}

impl SessionNode {
    /// Returns the most recently evaluated [`DataSet`] for this node, if available.
    pub fn dataset(&self) -> Option<&Arc<DataSet>> {
        self.cached_dataset.as_ref()
    }
}

/// Fluent display-style builder returned by [`ParaViz::show`].
pub struct DisplayHandle<'a> {
    node: &'a mut SessionNode,
}

impl<'a> DisplayHandle<'a> {
    pub fn visible(self, visible: bool) -> Self {
        self.node.style.visible = visible;
        self
    }

    pub fn representation(self, repr: Representation) -> Self {
        self.node.style.representation = repr;
        self
    }

    pub fn colormap(self, preset: ColorMapPreset) -> Self {
        let cm = ColorMap::from_preset(preset);
        self.node.style.colormap = cm.clone();
        self.node.style.volume_tf.color_map = cm;
        self
    }

    pub fn custom_colormap(self, colormap: ColorMap) -> Self {
        self.node.style.colormap = colormap.clone();
        self.node.style.volume_tf.color_map = colormap;
        self
    }

    pub fn color_by_point(self, array: impl Into<String>) -> Self {
        self.node.style.color_by = ColorBy::PointScalar {
            array: array.into(),
            component: None,
        };
        self.node.auto_styled = true;
        self
    }

    pub fn color_by_point_component(self, array: impl Into<String>, component: usize) -> Self {
        self.node.style.color_by = ColorBy::PointScalar {
            array: array.into(),
            component: Some(component),
        };
        self.node.auto_styled = true;
        self
    }

    pub fn color_by_cell(self, array: impl Into<String>) -> Self {
        self.node.style.color_by = ColorBy::CellScalar {
            array: array.into(),
            component: None,
        };
        self.node.auto_styled = true;
        self
    }

    pub fn solid_color(self, rgba: [f32; 4]) -> Self {
        self.node.style.color_by = ColorBy::Solid(rgba);
        self.node.auto_styled = true;
        self
    }

    pub fn scalar_range(self, min: f64, max: f64) -> Self {
        self.node.style.auto_range = false;
        self.node.style.scalar_range = (min, max);
        self.node.style.volume_tf.scalar_range = (min, max);
        self
    }

    pub fn auto_range(self, enabled: bool) -> Self {
        self.node.style.auto_range = enabled;
        self
    }

    pub fn opacity(self, opacity: f32) -> Self {
        self.node.style.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn point_size(self, size: f32) -> Self {
        self.node.style.point_size = size.max(1.0);
        self
    }

    pub fn line_width(self, width: f32) -> Self {
        self.node.style.line_width = width.max(0.5);
        self
    }

    pub fn edge_color(self, rgba: [f32; 4]) -> Self {
        self.node.style.edge_color = rgba;
        self
    }

    pub fn edge_width(self, width: f32) -> Self {
        self.node.style.edge_width = width.max(0.25);
        self
    }

    pub fn volume_tf(self, tf: TransferFunction) -> Self {
        self.node.style.volume_tf = tf;
        self
    }

    pub fn volume_step_size(self, step: f32) -> Self {
        self.node.style.volume_step_size = step.max(0.05);
        self
    }
}

/// Serializable snapshot of a [`ParaViz`] pipeline, styles, and camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineState {
    pub nodes: Vec<SerializedNode>,
    pub camera: Camera,
    pub settings: SceneSettings,
    pub time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    pub id: u64,
    pub name: String,
    pub parent: Option<u64>,
    pub source: Option<SourceConfig>,
    pub filter: Option<FilterConfig>,
    pub style: ActorStyle,
}

/// High-level ParaView session managing a demand-driven pipeline, visual actors,
/// camera, time steps, I/O, and lazy WebGPU headless rendering.
pub struct ParaViz {
    graph: PipelineGraph,
    nodes: Vec<SessionNode>,
    pub camera: Camera,
    pub settings: SceneSettings,
    pub time: f64,
    renderer: Option<WebGpuRenderer>,
}

impl Default for ParaViz {
    fn default() -> Self {
        Self::new()
    }
}

impl ParaViz {
    /// Creates an empty [`ParaViz`] session (WebGPU is initialized lazily on first render).
    pub fn new() -> Self {
        Self {
            graph: PipelineGraph::new(),
            nodes: Vec::new(),
            camera: Camera::default(),
            settings: SceneSettings::default(),
            time: 0.0,
            renderer: None,
        }
    }

    /// Creates a [`ParaViz`] session and eagerly initializes the headless WebGPU renderer.
    pub fn headless() -> Result<Self, SessionError> {
        let mut s = Self::new();
        s.renderer = Some(WebGpuRenderer::new_headless()?);
        Ok(s)
    }

    /// Adds a custom [`Source`] node to the pipeline.
    pub fn add_source(&mut self, label: impl Into<String>, source: impl Source + 'static) -> NodeId {
        let name = label.into();
        let info = source.request_information();
        let id = self.graph.add_source(name.clone(), Box::new(source));
        self.nodes.push(SessionNode {
            id,
            name,
            parent: None,
            source_config: None,
            filter_config: None,
            style: ActorStyle::default(),
            cached_dataset: None,
            time_steps: info.time_steps,
            auto_styled: false,
        });
        id
    }

    /// Adds a configurable [`SourceConfig`] node to the pipeline.
    pub fn add_source_cfg(
        &mut self,
        label: impl Into<String>,
        config: SourceConfig,
    ) -> Result<NodeId, SessionError> {
        let src = config.build_source()?;
        let info = src.request_information();
        let name = label.into();
        let id = self.graph.add_source(name.clone(), src);
        let mut style = ActorStyle::default();
        if matches!(config, SourceConfig::PointCloud(_)) {
            style.representation = Representation::Points;
        } else if matches!(config, SourceConfig::Wavelet(_) | SourceConfig::Gyroid(_)) {
            style.representation = Representation::Surface;
        }
        self.nodes.push(SessionNode {
            id,
            name,
            parent: None,
            source_config: Some(config),
            filter_config: None,
            style,
            cached_dataset: None,
            time_steps: info.time_steps,
            auto_styled: false,
        });
        Ok(id)
    }

    /// Wraps an in-memory [`DataSet`] as a pipeline source node.
    pub fn add_dataset(&mut self, label: impl Into<String>, dataset: DataSet) -> NodeId {
        let name = label.into();
        let src = InMemorySource {
            label: name.clone(),
            dataset: Arc::new(dataset),
        };
        self.add_source(name, src)
    }

    /// Opens a scientific file (`.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.stl`, `.obj`, `.ply`, `.csv`).
    pub fn open(&mut self, path: impl AsRef<Path>) -> Result<NodeId, SessionError> {
        let path = path.as_ref().to_path_buf();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let cfg = if ext == "pvd" {
            SourceConfig::PvdSeries(path)
        } else {
            SourceConfig::File(path)
        };
        let label = cfg.default_label();
        self.add_source_cfg(label, cfg)
    }

    /// Adds a custom [`Filter`] node connected to `input`.
    pub fn add_filter(
        &mut self,
        label: impl Into<String>,
        input: NodeId,
        filter: impl Filter + 'static,
    ) -> Result<NodeId, SessionError> {
        let name = label.into();
        let id = self.graph.add_filter(name.clone(), input, Box::new(filter))?;
        self.nodes.push(SessionNode {
            id,
            name,
            parent: Some(input),
            source_config: None,
            filter_config: None,
            style: ActorStyle::default(),
            cached_dataset: None,
            time_steps: Vec::new(),
            auto_styled: false,
        });
        Ok(id)
    }

    /// Adds a configurable [`FilterConfig`] node connected to `input`.
    pub fn add_filter_cfg(
        &mut self,
        label: impl Into<String>,
        input: NodeId,
        config: FilterConfig,
    ) -> Result<NodeId, SessionError> {
        let flt = config.build_filter();
        let name = label.into();
        let id = self.graph.add_filter(name.clone(), input, flt)?;
        let mut style = ActorStyle::default();
        if matches!(config, FilterConfig::StreamTracer(_)) {
            style.representation = Representation::Wireframe;
        }
        self.nodes.push(SessionNode {
            id,
            name,
            parent: Some(input),
            source_config: None,
            filter_config: Some(config),
            style,
            cached_dataset: None,
            time_steps: Vec::new(),
            auto_styled: false,
        });
        Ok(id)
    }

    /// Updates an existing configurable source node and invalidates downstream caches.
    pub fn update_source_cfg(
        &mut self,
        id: NodeId,
        config: SourceConfig,
    ) -> Result<(), SessionError> {
        let src = config.build_source()?;
        let info = src.request_information();
        self.graph.update_source(id, src)?;
        let node = self.node_mut(id)?;
        node.source_config = Some(config);
        node.time_steps = info.time_steps;
        self.invalidate_subtree(id);
        Ok(())
    }

    /// Updates an existing configurable filter node and invalidates downstream caches.
    pub fn update_filter_cfg(
        &mut self,
        id: NodeId,
        config: FilterConfig,
    ) -> Result<(), SessionError> {
        let flt = config.build_filter();
        self.graph.update_filter(id, flt)?;
        let node = self.node_mut(id)?;
        node.filter_config = Some(config);
        self.invalidate_subtree(id);
        Ok(())
    }

    fn invalidate_subtree(&mut self, root: NodeId) {
        let mut stack = vec![root];
        while let Some(cur) = stack.pop() {
            if let Some(n) = self.nodes.iter_mut().find(|n| n.id == cur) {
                n.cached_dataset = None;
            }
            for n in &self.nodes {
                if n.parent == Some(cur) {
                    stack.push(n.id);
                }
            }
        }
    }

    /// Removes `id` and any downstream dependent nodes.
    pub fn remove_node(&mut self, id: NodeId) -> Vec<NodeId> {
        let removed = self.graph.remove_node(id);
        self.nodes.retain(|n| !removed.contains(&n.id));
        removed
    }

    /// Makes `id` visible and returns a fluent [`DisplayHandle`] to configure its style.
    pub fn show(&mut self, id: NodeId) -> DisplayHandle<'_> {
        let node = self
            .nodes
            .iter_mut()
            .find(|n| n.id == id)
            .expect("NodeId not found in ParaViz::show");
        node.style.visible = true;
        DisplayHandle { node }
    }

    /// Hides `id` in the 3D scene without removing it from the pipeline.
    pub fn hide(&mut self, id: NodeId) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            node.style.visible = false;
        }
    }

    /// Sets the active simulation time step.
    pub fn set_time(&mut self, time: f64) {
        if (self.time - time).abs() > 1e-12 {
            self.time = time;
            for n in &mut self.nodes {
                n.cached_dataset = None;
            }
        }
    }

    /// Collects all unique time steps advertised by pipeline sources.
    pub fn all_time_steps(&self) -> Vec<f64> {
        let mut ts: Vec<f64> = self
            .nodes
            .iter()
            .flat_map(|n| n.time_steps.iter().copied())
            .collect();
        ts.sort_by(|a, b| a.total_cmp(b));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        ts
    }

    /// Evaluates a single node on demand at the current `time`.
    pub fn evaluate(&mut self, id: NodeId) -> Result<Arc<DataSet>, SessionError> {
        let ds = self.graph.update(id, self.time)?;
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            if !node.auto_styled {
                auto_select_color_by(&mut node.style, &ds);
                node.auto_styled = true;
            }
            node.cached_dataset = Some(ds.clone());
        }
        Ok(ds)
    }

    /// Evaluates all visible nodes in the pipeline.
    pub fn update_visible(&mut self) -> Result<(), SessionError> {
        let visible_ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|n| n.style.visible)
            .map(|n| n.id)
            .collect();
        for id in visible_ids {
            let _ = self.evaluate(id)?;
        }
        Ok(())
    }

    /// Evaluates all nodes in the pipeline.
    pub fn update_all(&mut self) -> Result<(), SessionError> {
        let ids: Vec<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        for id in ids {
            let _ = self.evaluate(id)?;
        }
        Ok(())
    }

    /// Builds a [`RenderScene`] from all evaluated visible nodes.
    pub fn build_scene(&mut self) -> Result<RenderScene, SessionError> {
        self.update_visible()?;
        let mut scene = RenderScene {
            camera: self.camera.clone(),
            actors: Vec::new(),
            settings: self.settings.clone(),
        };
        for node in &self.nodes {
            if !node.style.visible {
                continue;
            }
            if let Some(ds) = &node.cached_dataset {
                scene.add_actor(SceneActor {
                    id: node.id.0,
                    name: node.name.clone(),
                    dataset: ds.clone(),
                    style: node.style.clone(),
                });
            }
        }
        Ok(scene)
    }

    /// Fits the camera to frame all visible datasets.
    pub fn reset_camera(&mut self) -> Result<(), SessionError> {
        let scene = self.build_scene()?;
        let bounds = scene.bounds();
        self.camera.fit_bounds(&bounds);
        Ok(())
    }

    /// Moves the camera to a standard [`CameraPreset`] framing all visible datasets.
    pub fn camera_preset(&mut self, preset: CameraPreset) -> Result<(), SessionError> {
        let scene = self.build_scene()?;
        let bounds = scene.bounds();
        self.camera.apply_preset(preset, &bounds);
        Ok(())
    }

    fn ensure_renderer(&mut self) -> Result<&mut WebGpuRenderer, SessionError> {
        if self.renderer.is_none() {
            self.renderer = Some(WebGpuRenderer::new_headless()?);
        }
        Ok(self.renderer.as_mut().unwrap())
    }

    /// Renders the current scene headlessly with WebGPU into an `Rgba8Unorm` byte buffer.
    pub fn render_rgba(&mut self, width: u32, height: u32) -> Result<Vec<u8>, SessionError> {
        let scene = self.build_scene()?;
        let renderer = self.ensure_renderer()?;
        Ok(renderer.render_to_rgba(&scene, width, height)?)
    }

    /// Renders the current scene headlessly with WebGPU and writes a PNG file to `path`.
    pub fn render_png(
        &mut self,
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
    ) -> Result<(), SessionError> {
        let scene = self.build_scene()?;
        let renderer = self.ensure_renderer()?;
        renderer.render_to_png(&scene, width, height, path)?;
        Ok(())
    }

    /// Exports the evaluated dataset of `id` to disk using `paraviz-io`.
    pub fn export_dataset(
        &mut self,
        id: NodeId,
        path: impl AsRef<Path>,
    ) -> Result<(), SessionError> {
        let ds = self.evaluate(id)?;
        write_dataset(path, &ds)?;
        Ok(())
    }

    /// Returns an immutable slice of all pipeline nodes.
    pub fn nodes(&self) -> &[SessionNode] {
        &self.nodes
    }

    /// Returns a reference to a node by [`NodeId`].
    pub fn node(&self, id: NodeId) -> Result<&SessionNode, SessionError> {
        self.nodes
            .iter()
            .find(|n| n.id == id)
            .ok_or(SessionError::MissingNode(id))
    }

    /// Returns a mutable reference to a node by [`NodeId`].
    pub fn node_mut(&mut self, id: NodeId) -> Result<&mut SessionNode, SessionError> {
        self.nodes
            .iter_mut()
            .find(|n| n.id == id)
            .ok_or(SessionError::MissingNode(id))
    }

    /// Serializes the configurable pipeline nodes, styles, camera, and settings to JSON.
    pub fn save_state_json(&self) -> Result<String, SessionError> {
        let state = PipelineState {
            nodes: self
                .nodes
                .iter()
                .map(|n| SerializedNode {
                    id: n.id.0,
                    name: n.name.clone(),
                    parent: n.parent.map(|p| p.0),
                    source: n.source_config.clone(),
                    filter: n.filter_config.clone(),
                    style: n.style.clone(),
                })
                .collect(),
            camera: self.camera.clone(),
            settings: self.settings.clone(),
            time: self.time,
        };
        Ok(serde_json::to_string_pretty(&state)?)
    }

    /// Reconstructs a [`ParaViz`] session from a JSON state produced by [`Self::save_state_json`].
    pub fn load_state_json(json: &str) -> Result<Self, SessionError> {
        let state: PipelineState = serde_json::from_str(json)?;
        let mut pv = Self::new();
        pv.camera = state.camera;
        pv.settings = state.settings;
        pv.time = state.time;

        let mut id_map: HashMap<u64, NodeId> = HashMap::new();
        for sn in state.nodes {
            let new_id = if let Some(src) = sn.source {
                pv.add_source_cfg(sn.name, src)?
            } else if let (Some(old_parent), Some(flt)) = (sn.parent, sn.filter) {
                let mapped_parent = *id_map.get(&old_parent).ok_or(SessionError::MissingNode(NodeId(old_parent)))?;
                pv.add_filter_cfg(sn.name, mapped_parent, flt)?
            } else {
                continue;
            };
            if let Ok(node) = pv.node_mut(new_id) {
                node.style = sn.style;
                node.auto_styled = true;
            }
            id_map.insert(sn.id, new_id);
        }
        Ok(pv)
    }

    /// Generates an executable Rust code snippet reproducing the current pipeline via the `paraviz` API
    /// (equivalent to ParaView's Python Trace).
    pub fn to_rust_trace(&self) -> String {
        let mut out = String::from(
            "use paraviz::prelude::*;\n\nfn main() -> Result<(), Box<dyn std::error::Error>> {\n    let mut pv = ParaViz::new();\n",
        );
        let mut var_names: HashMap<NodeId, String> = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            let var = format!("node_{}", idx);
            if let Some(src) = &node.source_config {
                out.push_str(&format!(
                    "    let {var} = pv.add_source_cfg({:?}, {:?})?;\n",
                    node.name, src
                ));
            } else if let (Some(parent), Some(flt)) = (node.parent, &node.filter_config) {
                let pvar = var_names.get(&parent).cloned().unwrap_or_else(|| "node_0".into());
                out.push_str(&format!(
                    "    let {var} = pv.add_filter_cfg({:?}, {pvar}, {:?})?;\n",
                    node.name, flt
                ));
            } else {
                out.push_str(&format!("    // Custom node {:?}\n", node.name));
                continue;
            }
            if node.style.visible {
                out.push_str(&format!(
                    "    pv.show({var}).representation(Representation::{:?}).opacity({:.2});\n",
                    node.style.representation, node.style.opacity
                ));
            } else {
                out.push_str(&format!("    pv.hide({var});\n"));
            }
            var_names.insert(node.id, var);
        }
        out.push_str("    pv.reset_camera()?;\n    pv.render_png(\"output.png\", 1280, 720)?;\n    Ok(())\n}\n");
        out
    }

    /// Convenience helper matching [`read_dataset`] without adding to a pipeline.
    pub fn read_file(path: impl AsRef<Path>) -> Result<DataSet, SessionError> {
        Ok(read_dataset(path)?)
    }
}

fn auto_select_color_by(style: &mut ActorStyle, dataset: &DataSet) {
    if let Some(pd) = dataset.point_data() {
        if let Some(arr) = pd.active_scalars_array() {
            style.color_by = ColorBy::PointScalar {
                array: arr.name.clone(),
                component: None,
            };
            let rng = arr.range(None);
            style.scalar_range = rng;
            style.volume_tf.scalar_range = rng;
            return;
        }
        if let Some(arr) = pd.active_vectors_array() {
            style.color_by = ColorBy::PointScalar {
                array: arr.name.clone(),
                component: None,
            };
            let rng = arr.range(None);
            style.scalar_range = rng;
            style.volume_tf.scalar_range = rng;
            return;
        }
    }
    if let Some(cd) = dataset.cell_data() {
        if let Some(arr) = cd.active_scalars_array() {
            style.color_by = ColorBy::CellScalar {
                array: arr.name.clone(),
                component: None,
            };
            let rng = arr.range(None);
            style.scalar_range = rng;
            style.volume_tf.scalar_range = rng;
        }
    }
}
