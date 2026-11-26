//! High-level stateful ParaView-style pipeline, scene, and rendering session API.

use paraviz_core::{
    ColorMap, ColorMapPreset, DataSet, Filter, NodeId, PipelineError, PipelineGraph,
    PipelineInfo, Source, TransferFunction,
};
use paraviz_filters::{
    AddTimeStepsFilter, AppendDatasetsFilter, CalculatorFilter, CellCentersFilter,
    CellDataToPointDataFilter, CleanPolyDataFilter, ClipFilter, ConnectivityFilter, ContourFilter,
    DecimateFilter, Delaunay2DFilter, Delaunay3DFilter, ElevationFilter, ExtractBlockFilter,
    ExtractEdgesFilter, ExtractSurface, FeatureEdgesFilter, GenerateNormalsFilter, GlyphFilter,
    GradientFilter, IntegrateVariablesFilter, IsoVolumeFilter, PointDataToCellDataFilter,
    ProbeOverLineFilter, ReflectFilter, ResampleToImageFilter, ResampleWithDatasetFilter,
    ShrinkFilter, SliceFilter, SmoothFilter, StreamTracerFilter, StreamTracerWithCustomSource,
    TemporalStatisticsFilter, ThresholdFilter, TransformFilter, TriangulateFilter, TubeFilter,
    WarpByScalarFilter, WarpByVectorFilter,
};
use paraviz_io::{
    read_dataset, read_raw_volume, write_animated_gif, write_dataset, write_glb,
    write_html_viewer, DatasetReader, IoError, PvdSeriesSource, RawVolumeDescriptor, VtkFileSource,
};
use paraviz_render::{
    pick_scene, ActorStyle, Camera, CameraPreset, ColorBy, GpuContourEngine, PickResult,
    RenderError, RenderScene, Representation, SceneActor, SceneSettings, ViewportWidget,
    WebGpuRenderer,
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

    #[error("Cannot save node '{0}': evaluate this custom node first")]
    UnsavedCustomNode(String),

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

    fn execute_arc(&self, _time: f64) -> Result<Arc<DataSet>, PipelineError> {
        Ok(Arc::clone(&self.dataset))
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
    FileWithReader {
        path: PathBuf,
        reader: DatasetReader,
    },
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
            Self::File(p) | Self::FileWithReader { path: p, .. } | Self::PvdSeries(p) => p
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
            Self::FileWithReader { path, reader } => {
                Box::new(VtkFileSource::with_reader(path, *reader))
            }
            Self::PvdSeries(p) => Box::new(PvdSeriesSource::new(p)?),
        };
        Ok(src)
    }

    /// Deserializes a [`SourceConfig`] from JSON (used by saved states and Rust traces).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
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
    FeatureEdges(FeatureEdgesFilter),
    Connectivity(ConnectivityFilter),
    Shrink(ShrinkFilter),
    CellCenters(CellCentersFilter),
    Reflect(ReflectFilter),
    GenerateNormals(GenerateNormalsFilter),
    Elevation(ElevationFilter),
    Triangulate(TriangulateFilter),
    ExtractEdges(ExtractEdgesFilter),
    CleanPolyData(CleanPolyDataFilter),
    AppendDatasets(AppendDatasetsFilter),
    IntegrateVariables(IntegrateVariablesFilter),
    Decimate(DecimateFilter),
    Delaunay2D(Delaunay2DFilter),
    Delaunay3D(Delaunay3DFilter),
    IsoVolume(IsoVolumeFilter),
    ResampleToImage(ResampleToImageFilter),
    ExtractBlock(ExtractBlockFilter),
    TemporalStatistics(TemporalStatisticsFilter),
    AddTimeSteps(AddTimeStepsFilter),
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
            Self::FeatureEdges(_) => "FeatureEdges",
            Self::Connectivity(_) => "Connectivity",
            Self::Shrink(_) => "Shrink",
            Self::CellCenters(_) => "CellCenters",
            Self::Reflect(_) => "Reflect",
            Self::GenerateNormals(_) => "GenerateNormals",
            Self::Elevation(_) => "Elevation",
            Self::Triangulate(_) => "Triangulate",
            Self::ExtractEdges(_) => "ExtractEdges",
            Self::CleanPolyData(_) => "CleanPolyData",
            Self::AppendDatasets(_) => "AppendDatasets",
            Self::IntegrateVariables(_) => "IntegrateVariables",
            Self::Decimate(_) => "Decimate",
            Self::Delaunay2D(_) => "Delaunay2D",
            Self::Delaunay3D(_) => "Delaunay3D",
            Self::IsoVolume(_) => "IsoVolume",
            Self::ResampleToImage(_) => "ResampleToImage",
            Self::ExtractBlock(_) => "ExtractBlock",
            Self::TemporalStatistics(_) => "TemporalStatistics",
            Self::AddTimeSteps(_) => "AddTimeSteps",
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
            Self::FeatureEdges(f) => Box::new(f.clone()),
            Self::Connectivity(f) => Box::new(f.clone()),
            Self::Shrink(f) => Box::new(f.clone()),
            Self::CellCenters(f) => Box::new(f.clone()),
            Self::Reflect(f) => Box::new(f.clone()),
            Self::GenerateNormals(f) => Box::new(f.clone()),
            Self::Elevation(f) => Box::new(f.clone()),
            Self::Triangulate(f) => Box::new(f.clone()),
            Self::ExtractEdges(f) => Box::new(f.clone()),
            Self::CleanPolyData(f) => Box::new(f.clone()),
            Self::AppendDatasets(f) => Box::new(f.clone()),
            Self::IntegrateVariables(f) => Box::new(f.clone()),
            Self::Decimate(f) => Box::new(f.clone()),
            Self::Delaunay2D(f) => Box::new(f.clone()),
            Self::Delaunay3D(f) => Box::new(f.clone()),
            Self::IsoVolume(f) => Box::new(f.clone()),
            Self::ResampleToImage(f) => Box::new(f.clone()),
            Self::ExtractBlock(f) => Box::new(f.clone()),
            Self::TemporalStatistics(f) => Box::new(f.clone()),
            Self::AddTimeSteps(f) => Box::new(f.clone()),
        }
    }

    /// Deserializes a [`FilterConfig`] from JSON (used by saved states and Rust traces).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
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
    in_memory_dataset: Option<Arc<DataSet>>,
    multi_filter_config: Option<MultiFilterConfig>,
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

    /// Hides a named or slash-pathed block inside a `MultiBlockDataSet`.
    pub fn hide_block(self, block: impl Into<String>) -> Self {
        self.node.style.hidden_blocks.insert(block.into());
        self
    }

    /// Shows a previously hidden block inside a `MultiBlockDataSet`.
    pub fn show_block(self, block: &str) -> Self {
        self.node.style.hidden_blocks.remove(block);
        self
    }

    /// Sets a solid RGBA color override on a named or slash-pathed block in a `MultiBlockDataSet`.
    pub fn block_color(self, block: impl Into<String>, rgba: [f32; 4]) -> Self {
        self.node.style.block_colors.insert(block.into(), rgba);
        self
    }

    /// Removes any per-block color override for `block`.
    pub fn clear_block_color(self, block: &str) -> Self {
        self.node.style.block_colors.remove(block);
        self
    }
}

/// Serializable configuration for the built-in filters with multiple live inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiFilterConfig {
    AppendDatasets,
    ResampleWithDataset,
    StreamTracerWithCustomSource(StreamTracerFilter),
}

impl MultiFilterConfig {
    fn build_filter(&self) -> Box<dyn paraviz_core::MultiInputFilter> {
        match self {
            Self::AppendDatasets => Box::new(AppendDatasetsFilter::new()),
            Self::ResampleWithDataset => Box::new(ResampleWithDatasetFilter),
            Self::StreamTracerWithCustomSource(config) => {
                Box::new(StreamTracerWithCustomSource::new(config.clone()))
            }
        }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_dataset: Option<DataSet>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_filter: Option<MultiFilterConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<u64>,
    pub style: ActorStyle,
}

/// High-level ParaView session managing a demand-driven pipeline, visual actors,
/// camera, time steps, I/O, and lazy WebGPU headless rendering.
pub struct ParaViz {
    graph: PipelineGraph,
    nodes: Vec<SessionNode>,
    pub camera: Camera,
    pub settings: SceneSettings,
    pub widgets: Vec<ViewportWidget>,
    pub selection_highlight: Option<[f32; 3]>,
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
            widgets: Vec::new(),
            selection_highlight: None,
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
            in_memory_dataset: None,
            multi_filter_config: None,
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
            in_memory_dataset: None,
            multi_filter_config: None,
        });
        Ok(id)
    }

    /// Wraps an in-memory [`DataSet`] as a pipeline source node.
    pub fn add_dataset(&mut self, label: impl Into<String>, dataset: DataSet) -> NodeId {
        let name = label.into();
        let dataset = Arc::new(dataset);
        let src = InMemorySource {
            label: name.clone(),
            dataset: Arc::clone(&dataset),
        };
        let id = self.add_source(name, src);
        self.nodes.last_mut().unwrap().in_memory_dataset = Some(dataset);
        id
    }

    /// Opens a scientific file (`.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.xdmf`, `.xmf`, `.stl`, `.obj`, `.ply`, `.csv`, etc.).
    pub fn open(&mut self, path: impl AsRef<Path>) -> Result<NodeId, SessionError> {
        self.open_with_reader(path, DatasetReader::Auto)
    }

    /// Opens a scientific file using an explicit [`DatasetReader`] (or [`DatasetReader::Auto`]).
    pub fn open_with_reader(
        &mut self,
        path: impl AsRef<Path>,
        reader: DatasetReader,
    ) -> Result<NodeId, SessionError> {
        let path = path.as_ref().to_path_buf();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let cfg = if ext == "pvd" && reader == DatasetReader::Auto {
            SourceConfig::PvdSeries(path)
        } else if reader == DatasetReader::Auto {
            SourceConfig::File(path)
        } else {
            SourceConfig::FileWithReader { path, reader }
        };
        let label = cfg.default_label();
        self.add_source_cfg(label, cfg)
    }

    /// Registers a file source whose dataset was already loaded (for example on a background worker thread)
    /// without re-reading from disk on the first evaluation.
    pub fn open_preloaded(
        &mut self,
        path: impl AsRef<Path>,
        reader: DatasetReader,
        dataset: DataSet,
        time_steps: Vec<f64>,
    ) -> Result<NodeId, SessionError> {
        let path = path.as_ref().to_path_buf();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let cfg = if ext == "pvd" && reader == DatasetReader::Auto {
            SourceConfig::PvdSeries(path)
        } else if reader == DatasetReader::Auto {
            SourceConfig::File(path)
        } else {
            SourceConfig::FileWithReader { path, reader }
        };
        let label = cfg.default_label();
        let src = cfg.build_source()?;
        let id = self.graph.add_source(label.clone(), src);
        let arc = Arc::new(dataset);
        self.graph.seed_cached_output(id, arc.clone(), self.time)?;
        let mut style = ActorStyle::default();
        auto_select_color_by(&mut style, &arc);
        if matches!(arc.as_ref(), DataSet::ImageData(_)) {
            style.representation = Representation::Outline;
        }
        self.nodes.push(SessionNode {
            id,
            name: label,
            parent: None,
            source_config: Some(cfg),
            filter_config: None,
            style,
            cached_dataset: Some(arc),
            time_steps,
            auto_styled: true,
            in_memory_dataset: None,
            multi_filter_config: None,
        });
        Ok(id)
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
            in_memory_dataset: None,
            multi_filter_config: None,
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
        let time_steps = match &config {
            FilterConfig::AddTimeSteps(f) => f.time_values(),
            _ => Vec::new(),
        };
        self.nodes.push(SessionNode {
            id,
            name,
            parent: Some(input),
            source_config: None,
            filter_config: Some(config),
            style,
            cached_dataset: None,
            time_steps,
            auto_styled: false,
            in_memory_dataset: None,
            multi_filter_config: None,
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
        node.parent = None;
        node.filter_config = None;
        node.in_memory_dataset = None;
        node.multi_filter_config = None;
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
        node.time_steps = match &config {
            FilterConfig::AddTimeSteps(f) => f.time_values(),
            _ => Vec::new(),
        };
        node.filter_config = Some(config);
        self.invalidate_subtree(id);
        Ok(())
    }

    fn invalidate_subtree(&mut self, root: NodeId) {
        let mut invalidated = std::collections::HashSet::from([root]);
        // Session nodes are stored in topological order.
        for node in &mut self.nodes {
            if invalidated.contains(&node.id)
                || self.graph.inputs(node.id).iter().any(|id| invalidated.contains(id))
            {
                node.cached_dataset = None;
                invalidated.insert(node.id);
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

    /// Seeds a pre-evaluated dataset (for example from a background worker) for `id` at `time`,
    /// avoiding synchronous re-execution on the main UI thread.
    pub fn seed_evaluated(
        &mut self,
        id: NodeId,
        dataset: Arc<DataSet>,
        time: f64,
    ) -> Result<(), SessionError> {
        self.graph.seed_cached_output(id, dataset.clone(), time)?;
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            if !node.auto_styled {
                auto_select_color_by(&mut node.style, &dataset);
                node.auto_styled = true;
            }
            node.cached_dataset = Some(dataset);
        }
        Ok(())
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
        let mut settings = self.settings.clone();
        if settings.show_time_annotation && settings.time_label.is_none() {
            settings.time_label = Some(format!("Time: {:.4}", self.time));
        }
        let mut scene = RenderScene {
            camera: self.camera.clone(),
            actors: Vec::new(),
            settings,
            widgets: self.widgets.clone(),
            selection_highlight: self.selection_highlight,
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

    /// Renders a 360-degree turntable animation and writes an animated GIF89a file to `path`.
    pub fn render_turntable_gif(
        &mut self,
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        frames: u32,
        delay_ms: u16,
    ) -> Result<(), SessionError> {
        let count = frames.max(1);
        let saved_cam = self.camera.clone();
        let step_rad = std::f32::consts::TAU / (count as f32);
        let mut rgba_frames = Vec::with_capacity(count as usize);
        for _ in 0..count {
            rgba_frames.push(self.render_rgba(width, height)?);
            self.camera.orbit(step_rad, 0.0);
        }
        self.camera = saved_cam;
        let delay_cs = (delay_ms / 10).max(1);
        write_animated_gif(path, &rgba_frames, width, height, delay_cs)?;
        Ok(())
    }

    /// Renders all advertised pipeline time steps and writes an animated GIF89a file to `path`.
    pub fn render_timeseries_gif(
        &mut self,
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        delay_ms: u16,
    ) -> Result<(), SessionError> {
        let steps = self.all_time_steps();
        let saved_time = self.time;
        let mut rgba_frames = if steps.is_empty() {
            vec![self.render_rgba(width, height)?]
        } else {
            let mut out = Vec::with_capacity(steps.len());
            for t in steps {
                self.set_time(t);
                out.push(self.render_rgba(width, height)?);
            }
            out
        };
        if rgba_frames.is_empty() {
            rgba_frames.push(self.render_rgba(width, height)?);
        }
        self.set_time(saved_time);
        let delay_cs = (delay_ms / 10).max(1);
        write_animated_gif(path, &rgba_frames, width, height, delay_cs)?;
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

    fn prepare_surface_export(
        &mut self,
        id: NodeId,
        use_node_colormap: bool,
    ) -> Result<(String, paraviz_core::PolyData, Option<ColorMap>, Option<String>), SessionError>
    {
        let ds = self.evaluate(id)?;
        let node = self.node(id)?;
        let name = node.name.clone();
        let style = node.style.clone();
        let poly = match ds.as_ref() {
            DataSet::PolyData(p) => p.clone(),
            other => paraviz_filters::extract_surface(other)?,
        };
        let scalar_name = match &style.color_by {
            ColorBy::PointScalar { array, .. } => Some(array.clone()),
            _ => None,
        };
        let cmap = if use_node_colormap {
            Some(style.colormap)
        } else {
            None
        };
        Ok((name, poly, cmap, scalar_name))
    }

    /// Exports a `PolyData` node (or extracted surface) to a binary glTF 2.0 (`.glb`) file
    /// with optional scalar vertex colors.
    pub fn export_glb(
        &mut self,
        id: NodeId,
        path: impl AsRef<Path>,
        use_node_colormap: bool,
    ) -> Result<(), SessionError> {
        let (_, poly, cmap, scalar_name) = self.prepare_surface_export(id, use_node_colormap)?;
        write_glb(path, &poly, cmap.as_ref(), scalar_name.as_deref())?;
        Ok(())
    }

    /// Exports a `PolyData` node (or extracted surface) to a self-contained interactive
    /// WebGL2 HTML5 3D viewer (`.html`).
    pub fn export_html(
        &mut self,
        id: NodeId,
        path: impl AsRef<Path>,
        use_node_colormap: bool,
    ) -> Result<(), SessionError> {
        let (title, poly, cmap, scalar_name) = self.prepare_surface_export(id, use_node_colormap)?;
        write_html_viewer(path, &poly, &title, cmap.as_ref(), scalar_name.as_deref())?;
        Ok(())
    }

    /// Loads a raw binary 3D volumetric file as an `ImageData` source node.
    pub fn open_raw_volume(
        &mut self,
        label: impl Into<String>,
        path: impl AsRef<Path>,
        desc: &RawVolumeDescriptor,
    ) -> Result<NodeId, SessionError> {
        let img = read_raw_volume(path, desc)?;
        Ok(self.add_dataset(label, DataSet::ImageData(img)))
    }

    /// Casts a 3D pick ray from pixel `(px, py)` in a `width x height` viewport against visible actors.
    pub fn pick(
        &mut self,
        px: f32,
        py: f32,
        width: u32,
        height: u32,
    ) -> Result<Option<PickResult>, SessionError> {
        let scene = self.build_scene()?;
        Ok(pick_scene(&scene, px, py, width, height))
    }

    /// Extracts an isosurface directly on the GPU via WebGPU Compute Marching Cubes (`GpuContourEngine`)
    /// and adds the resulting `PolyData` to the session as a new node.
    pub fn gpu_contour(
        &mut self,
        label: impl Into<String>,
        input: NodeId,
        field_name: Option<&str>,
        iso_value: f64,
    ) -> Result<NodeId, SessionError> {
        let ds = self.evaluate(input)?;
        let img = ds.as_image_data().ok_or_else(|| {
            PipelineError::ExecutionFailed("GpuContourEngine requires an ImageData dataset".into())
        })?;
        let renderer = self.ensure_renderer()?;
        let engine = GpuContourEngine::from_device(
            Arc::new(renderer.device().clone()),
            Arc::new(renderer.queue().clone()),
        );
        let poly = engine.contour_imagedata(img, field_name.unwrap_or(""), iso_value)?;
        Ok(self.add_dataset(label, DataSet::PolyData(poly)))
    }

    /// Registers a multi-input filter node (e.g. `ResampleWithDataset`, live `AppendDatasets`).
    pub fn add_multi_filter(
        &mut self,
        label: impl Into<String>,
        inputs: Vec<NodeId>,
        filter: Box<dyn paraviz_core::MultiInputFilter>,
    ) -> Result<NodeId, SessionError> {
        let name = label.into();
        let primary_parent = inputs.first().copied();
        let id = self.graph.add_multi_filter(name.clone(), inputs, filter)?;
        self.nodes.push(SessionNode {
            id,
            name,
            parent: primary_parent,
            source_config: None,
            filter_config: None,
            style: ActorStyle::default(),
            cached_dataset: None,
            auto_styled: false,
            in_memory_dataset: None,
            multi_filter_config: None,
            time_steps: Vec::new(),
        });
        Ok(id)
    }

    fn add_multi_filter_cfg(
        &mut self,
        label: impl Into<String>,
        inputs: Vec<NodeId>,
        config: MultiFilterConfig,
    ) -> Result<NodeId, SessionError> {
        let id = self.add_multi_filter(label, inputs, config.build_filter())?;
        self.node_mut(id)?.multi_filter_config = Some(config);
        Ok(id)
    }

    /// Samples `source_node`'s volumetric fields onto `destination_node`'s geometry points.
    pub fn resample_with_dataset(
        &mut self,
        source_node: NodeId,
        destination_node: NodeId,
    ) -> Result<NodeId, SessionError> {
        self.add_multi_filter_cfg(
            "ResampleWithDataset",
            vec![source_node, destination_node],
            MultiFilterConfig::ResampleWithDataset,
        )
    }

    /// Traces streamlines through `field_node`'s vector field using `seed_node`'s points as seeds.
    pub fn stream_tracer_with_custom_source(
        &mut self,
        field_node: NodeId,
        seed_node: NodeId,
        config: StreamTracerFilter,
    ) -> Result<NodeId, SessionError> {
        let id = self.add_multi_filter_cfg(
            "StreamTracerWithCustomSource",
            vec![field_node, seed_node],
            MultiFilterConfig::StreamTracerWithCustomSource(config),
        )?;
        if let Ok(n) = self.node_mut(id) {
            n.style.representation = Representation::Wireframe;
        }
        Ok(id)
    }

    /// Appends multiple live pipeline nodes together into a single combined mesh/grid.
    pub fn append_nodes(&mut self, inputs: Vec<NodeId>) -> Result<NodeId, SessionError> {
        self.add_multi_filter_cfg(
            "AppendDatasets",
            inputs,
            MultiFilterConfig::AppendDatasets,
        )
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

    /// Serializes the configurable pipeline nodes, custom in-memory datasets, styles, camera, and settings to JSON.
    pub fn save_state_json(&self) -> Result<String, SessionError> {
        let state = PipelineState {
            nodes: self
                .nodes
                .iter()
                .map(|n| {
                    let custom_dataset = if n.source_config.is_none()
                        && n.filter_config.is_none()
                        && n.multi_filter_config.is_none()
                    {
                        let dataset = n.in_memory_dataset.clone()
                            .or_else(|| n.cached_dataset.clone())
                            .or_else(|| self.graph.cached_output(n.id))
                            .ok_or_else(|| SessionError::UnsavedCustomNode(n.name.clone()))?;
                        Some((*dataset).clone())
                    } else {
                        None
                    };
                    let inputs = if n.multi_filter_config.is_some() {
                        self.graph.inputs(n.id).into_iter().map(|id| id.0).collect()
                    } else {
                        Vec::new()
                    };
                    Ok(SerializedNode {
                        id: n.id.0,
                        name: n.name.clone(),
                        parent: n.parent.map(|p| p.0),
                        source: n.source_config.clone(),
                        filter: n.filter_config.clone(),
                        custom_dataset,
                        multi_filter: n.multi_filter_config.clone(),
                        inputs,
                        style: n.style.clone(),
                    })
                })
                .collect::<Result<_, SessionError>>()?,
            camera: self.camera.clone(),
            settings: self.settings.clone(),
            time: self.time,
        };
        Ok(serde_json::to_string_pretty(&state)?)
    }

    /// Reconstructs a [`ParaViz`] session from a JSON state produced by [`Self::save_state_json`].
    pub fn load_state_json(json: &str) -> Result<Self, SessionError> {
        const MAX_SESSION_NODES: usize = 10_000;
        const MAX_PIPELINE_DEPTH: usize = 256;

        let state: PipelineState = serde_json::from_str(json)?;
        if state.nodes.len() > MAX_SESSION_NODES {
            return Err(SessionError::Pipeline(
                paraviz_core::PipelineError::ExecutionFailed(format!(
                    "Session exceeds maximum node count of {MAX_SESSION_NODES}"
                )),
            ));
        }
        let mut pv = Self::new();
        pv.camera = state.camera;
        pv.settings = state.settings;
        pv.time = state.time;

        let mut id_map: HashMap<u64, NodeId> = HashMap::new();
        let mut depth_map: HashMap<u64, usize> = HashMap::new();
        for sn in state.nodes {
            let resolve_inputs = |ids: &[u64]| -> Result<(Vec<NodeId>, usize), SessionError> {
                let mut inputs = Vec::with_capacity(ids.len());
                let mut depth = 0;
                for old_id in ids {
                    inputs.push(*id_map.get(old_id)
                        .ok_or(SessionError::MissingNode(NodeId(*old_id)))?);
                    depth = depth.max(depth_map[old_id] + 1);
                }
                if depth > MAX_PIPELINE_DEPTH {
                    return Err(PipelineError::ExecutionFailed(format!(
                        "Session pipeline depth exceeds limit of {MAX_PIPELINE_DEPTH}"
                    )).into());
                }
                Ok((inputs, depth))
            };
            let (new_id, depth) = if let Some(src) = sn.source {
                (pv.add_source_cfg(sn.name, src)?, 0)
            } else if let Some(config) = sn.multi_filter {
                let (inputs, depth) = resolve_inputs(&sn.inputs)?;
                (pv.add_multi_filter_cfg(sn.name, inputs, config)?, depth)
            } else if let (Some(old_parent), Some(flt)) = (sn.parent, sn.filter) {
                let (inputs, depth) = resolve_inputs(&[old_parent])?;
                (pv.add_filter_cfg(sn.name, inputs[0], flt)?, depth)
            } else if let Some(ds) = sn.custom_dataset {
                (pv.add_dataset(sn.name, ds), 0)
            } else {
                continue;
            };
            if let Ok(node) = pv.node_mut(new_id) {
                node.style = sn.style;
                node.auto_styled = true;
            }
            id_map.insert(sn.id, new_id);
            depth_map.insert(sn.id, depth);
        }
        Ok(pv)
    }

    /// Generates an executable Rust code snippet reproducing the current pipeline via the `paraviz` API
    /// (equivalent to ParaView's Python Trace).
    pub fn to_rust_trace(&self) -> String {
        fn raw_str(s: &str) -> String {
            let mut hashes = "#".to_string();
            while s.contains(&format!("\"{hashes}")) {
                hashes.push('#');
            }
            format!("r{hashes}\"{s}\"{hashes}")
        }

        let mut out = String::from(
            "use paraviz::prelude::*;\n\nfn main() -> Result<(), Box<dyn std::error::Error>> {\n    let mut pv = ParaViz::new();\n",
        );
        let mut var_names: HashMap<NodeId, String> = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            let var = format!("node_{}", idx);
            if let Some(src) = &node.source_config {
                let json = serde_json::to_string(src).unwrap_or_default();
                out.push_str(&format!(
                    "    let {var} = pv.add_source_cfg({:?}, SourceConfig::from_json({})?)?;\n",
                    node.name,
                    raw_str(&json)
                ));
            } else if let (Some(parent), Some(flt)) = (node.parent, &node.filter_config) {
                let pvar = var_names.get(&parent).cloned().unwrap_or_else(|| "node_0".into());
                let json = serde_json::to_string(flt).unwrap_or_default();
                out.push_str(&format!(
                    "    let {var} = pv.add_filter_cfg({:?}, {pvar}, FilterConfig::from_json({})?)?;\n",
                    node.name,
                    raw_str(&json)
                ));
            } else {
                out.push_str(&format!("    // In-memory custom node {:?} (recreate via add_dataset/gpu_contour)\n", node.name));
                continue;
            }
            if node.style.visible {
                out.push_str(&format!(
                    "    pv.show({var})\n        .representation(Representation::{:?})\n",
                    node.style.representation
                ));
                if let Some(preset) = ColorMapPreset::ALL
                    .iter()
                    .find(|p| p.name() == node.style.colormap.name)
                {
                    out.push_str(&format!("        .colormap(ColorMapPreset::{preset:?})\n"));
                }
                match &node.style.color_by {
                    ColorBy::PointScalar { array, component: None } => {
                        out.push_str(&format!("        .color_by_point({array:?})\n"));
                    }
                    ColorBy::PointScalar {
                        array,
                        component: Some(c),
                    } => {
                        out.push_str(&format!(
                            "        .color_by_point_component({array:?}, {c})\n"
                        ));
                    }
                    ColorBy::CellScalar { array, .. } => {
                        out.push_str(&format!("        .color_by_cell({array:?})\n"));
                    }
                    ColorBy::Solid(rgba) => {
                        out.push_str(&format!(
                            "        .solid_color([{:.4}, {:.4}, {:.4}, {:.4}])\n",
                            rgba[0], rgba[1], rgba[2], rgba[3]
                        ));
                    }
                }
                if !node.style.auto_range {
                    out.push_str(&format!(
                        "        .scalar_range({:.6}, {:.6})\n",
                        node.style.scalar_range.0, node.style.scalar_range.1
                    ));
                }
                out.push_str(&format!(
                    "        .opacity({:.3})\n        .line_width({:.2})\n        .point_size({:.2});\n",
                    node.style.opacity, node.style.line_width, node.style.point_size
                ));
            } else {
                out.push_str(&format!("    pv.hide({var});\n"));
            }
            var_names.insert(node.id, var);
        }
        out.push_str(&format!(
            "    pv.camera.position = [{:.4}, {:.4}, {:.4}].into();\n",
            self.camera.position.x, self.camera.position.y, self.camera.position.z
        ));
        out.push_str(&format!(
            "    pv.camera.target = [{:.4}, {:.4}, {:.4}].into();\n",
            self.camera.target.x, self.camera.target.y, self.camera.target.z
        ));
        out.push_str(&format!(
            "    pv.camera.up = [{:.4}, {:.4}, {:.4}].into();\n",
            self.camera.up.x, self.camera.up.y, self.camera.up.z
        ));
        out.push_str(&format!(
            "    pv.camera.fov_y_deg = {:.2};\n",
            self.camera.fov_y_deg
        ));
        out.push_str(&format!(
            "    pv.settings.edl_enabled = {};\n    pv.settings.msaa_samples = {};\n",
            self.settings.edl_enabled, self.settings.msaa_samples
        ));
        out.push_str("    pv.render_png(\"output.png\", 1280, 720)?;\n    Ok(())\n}\n");
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
    if let Some(cd) = dataset.cell_data()
        && let Some(arr) = cd.active_scalars_array()
    {
        style.color_by = ColorBy::CellScalar {
            array: arr.name.clone(),
            component: None,
        };
        let rng = arr.range(None);
        style.scalar_range = rng;
        style.volume_tf.scalar_range = rng;
    }
}
