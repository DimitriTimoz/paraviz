//! Native Python extension for ParaViz, exposed as `paraviz._paraviz`.
//!
//! On top of the typed convenience methods, this exposes the *generic*
//! configuration surface (`SourceConfig` / `FilterConfig` as JSON) that the
//! pure-Python `paraview.simple` compatibility shim drives.

use ::paraviz::core::array::ArrayData;
use ::paraviz::core::colormap::ColorStop;
use ::paraviz::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn parse_reader(s: &str) -> Option<DatasetReader> {
    Some(match s.to_ascii_lowercase().replace(['-', ' '], "").as_str() {
        "auto" => DatasetReader::Auto,
        "vtkhdf" => DatasetReader::VtkHdf,
        "cgns" => DatasetReader::Cgns,
        "exodus" | "exodaii" => DatasetReader::Exodus,
        "patchhdf5" | "hdf5" | "hdf" => DatasetReader::PatchHdf5,
        "xdmf" | "xmf" => DatasetReader::Xdmf,
        "vtkxml" | "xml" | "vtu" | "vti" | "vtp" => DatasetReader::VtkXml,
        "vtklegacy" | "legacy" | "vtk" => DatasetReader::VtkLegacy,
        "series" | "pvd" => DatasetReader::Series,
        "ensight" | "ensightgold" => DatasetReader::EnSight,
        "openfoam" | "foam" => DatasetReader::OpenFoam,
        "gmsh" | "msh" => DatasetReader::Gmsh,
        "plot3d" => DatasetReader::Plot3d,
        "tecplot" | "tec" => DatasetReader::Tecplot,
        "abaqus" | "inp" => DatasetReader::Abaqus,
        "netcdf" | "nc" | "cdf" => DatasetReader::NetCdf,
        "stl" => DatasetReader::Stl,
        "obj" => DatasetReader::Obj,
        "ply" => DatasetReader::Ply,
        "off" => DatasetReader::Off,
        "gltf" | "glb" => DatasetReader::Gltf,
        "x3d" => DatasetReader::X3d,
        "las" => DatasetReader::Las,
        "xyz" => DatasetReader::Xyz,
        "pdb" | "ent" => DatasetReader::Pdb,
        "csv" | "tsv" => DatasetReader::Csv,
        "nifti" | "nii" => DatasetReader::Nifti,
        "nrrd" => DatasetReader::Nrrd,
        "metaimage" | "mhd" | "mha" => DatasetReader::MetaImage,
        "dicom" | "dcm" => DatasetReader::Dicom,
        "image2d" | "png" | "bmp" => DatasetReader::Image2d,
        _ => return None,
    })
}

fn parse_representation(s: &str) -> Option<Representation> {
    Some(match s.to_ascii_lowercase().replace(['_', '-'], " ").trim() {
        "surface" => Representation::Surface,
        "surface with edges" | "edges" => Representation::SurfaceWithEdges,
        "wireframe" => Representation::Wireframe,
        "points" => Representation::Points,
        "point gaussian" | "gaussian" | "pointgaussian" => Representation::PointGaussian,
        "volume" => Representation::Volume,
        "outline" => Representation::Outline,
        _ => return None,
    })
}

fn parse_colormap(s: &str) -> Option<ColorMapPreset> {
    Some(match s.to_ascii_lowercase().replace(['_', '-', ' ', '(', ')'], "").as_str() {
        "coolwarm" | "cool2warm" | "cooltowarm" | "cool2warmmatplotlib" => ColorMapPreset::CoolToWarm,
        "viridis" | "viridismatplotlib" => ColorMapPreset::Viridis,
        "plasma" | "plasmamatplotlib" => ColorMapPreset::Plasma,
        "inferno" | "infernomatplotlib" => ColorMapPreset::Inferno,
        "magma" | "magmamatplotlib" => ColorMapPreset::Magma,
        "turbo" => ColorMapPreset::Turbo,
        "fast" => ColorMapPreset::Fast,
        "rainbow" => ColorMapPreset::Rainbow,
        "grayscale" | "grey" | "gray" => ColorMapPreset::Grayscale,
        "blueredyellow" | "bluered" => ColorMapPreset::BlueRedYellow,
        "cividis" => ColorMapPreset::Cividis,
        _ => return None,
    })
}

fn parse_association(s: &str) -> AttributeAssociation {
    match s.to_ascii_lowercase().as_str() {
        "cells" | "cell" | "cell data" | "celldata" => AttributeAssociation::Cell,
        "field" | "field data" => AttributeAssociation::Field,
        _ => AttributeAssociation::Point,
    }
}

/// numpy dtype name for an [`ArrayData`] buffer.
fn array_dtype(a: &ArrayData) -> &'static str {
    match a {
        ArrayData::F32(_) => "float32",
        ArrayData::F64(_) => "float64",
        ArrayData::I8(_) => "int8",
        ArrayData::I16(_) => "int16",
        ArrayData::I32(_) => "int32",
        ArrayData::I64(_) => "int64",
        ArrayData::U8(_) => "uint8",
        ArrayData::U16(_) => "uint16",
        ArrayData::U32(_) => "uint32",
        ArrayData::U64(_) => "uint64",
    }
}

/// Raw little-endian bytes of an [`ArrayData`] buffer.
fn array_bytes(a: &ArrayData) -> Vec<u8> {
    match a {
        ArrayData::F32(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::F64(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::I8(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::I16(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::I32(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::I64(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::U8(v) => v.clone(),
        ArrayData::U16(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::U32(v) => bytemuck::cast_slice(v).to_vec(),
        ArrayData::U64(v) => bytemuck::cast_slice(v).to_vec(),
    }
}

fn field_json(fd: &FieldData) -> serde_json::Value {
    let arrays: Vec<serde_json::Value> = fd
        .arrays()
        .iter()
        .map(|a| {
            let (rmin, rmax) = a.range(None);
            serde_json::json!({
                "name": a.name,
                "components": a.num_components,
                "dtype": array_dtype(&a.data),
                "num_tuples": a.data.len() / a.num_components.max(1),
                "range": [rmin, rmax],
            })
        })
        .collect();
    serde_json::json!({
        "arrays": arrays,
        "active_scalars": fd.active_scalars,
        "active_vectors": fd.active_vectors,
        "active_normals": fd.active_normals,
    })
}

#[pyclass(name = "Session")]
struct PySession {
    inner: ParaViz,
}

/// Evaluates a node and, for multi-block datasets, extracts one block.
fn resolve_dataset(
    inner: &mut ParaViz,
    id: u64,
    block: Option<usize>,
) -> PyResult<Arc<DataSet>> {
    let ds = inner.evaluate(NodeId(id)).map_err(to_py_err)?;
    if let (DataSet::MultiBlock(mb), Some(b)) = (ds.as_ref(), block) {
        let (_, blk) = mb
            .blocks
            .get(b)
            .ok_or_else(|| PyRuntimeError::new_err(format!("block index {b} out of range")))?;
        return Ok(Arc::new(blk.clone()));
    }
    Ok(ds)
}

#[pymethods]
impl PySession {
    #[new]
    fn new() -> Self {
        Self {
            inner: ParaViz::new(),
        }
    }

    /// Creates a session with an eagerly initialized headless WebGPU renderer.
    #[staticmethod]
    fn headless() -> PyResult<Self> {
        Ok(Self {
            inner: ParaViz::headless().map_err(to_py_err)?,
        })
    }

    /// ParaViz version string.
    #[staticmethod]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Whether a headless WebGPU adapter is available on this machine.
    #[staticmethod]
    fn gpu_available() -> bool {
        WebGpuRenderer::new_headless().is_ok()
    }

    /// Serialized default `SourceConfig` for `key` (e.g. `"Sphere"`, `"File"`).
    #[staticmethod]
    fn default_source_cfg_json(key: &str) -> PyResult<String> {
        let cfg = match key {
            "Sphere" => SourceConfig::Sphere(SphereSource::default()),
            "Cube" => SourceConfig::Cube(CubeSource::default()),
            "Cone" => SourceConfig::Cone(ConeSource::default()),
            "Cylinder" => SourceConfig::Cylinder(CylinderSource::default()),
            "Plane" => SourceConfig::Plane(PlaneSource::default()),
            "Line" => SourceConfig::Line(LineSource::default()),
            "Arrow" => SourceConfig::Arrow(ArrowSource::default()),
            "Disk" => SourceConfig::Disk(DiskSource::default()),
            "Torus" => SourceConfig::Torus(TorusSource::default()),
            "Wavelet" => SourceConfig::Wavelet(WaveletSource::default()),
            "VectorField" => SourceConfig::VectorField(VectorFieldSource::default()),
            "Superquadric" => SourceConfig::Superquadric(SuperquadricSource::default()),
            "Gyroid" => SourceConfig::Gyroid(GyroidSource::default()),
            "PointCloud" => SourceConfig::PointCloud(PointCloudSource::default()),
            "File" => SourceConfig::File(std::path::PathBuf::new()),
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown source config '{key}'"
                )))
            }
        };
        serde_json::to_string(&cfg).map_err(to_py_err)
    }

    /// Serialized default `FilterConfig` for `key` (e.g. `"Contour"`).
    #[staticmethod]
    fn default_filter_cfg_json(key: &str) -> PyResult<String> {
        let cfg = match key {
            "ExtractSurface" => FilterConfig::ExtractSurface(ExtractSurface::default()),
            "Contour" => FilterConfig::Contour(ContourFilter::default()),
            "Slice" => FilterConfig::Slice(SliceFilter::default()),
            "Clip" => FilterConfig::Clip(ClipFilter::default()),
            "Threshold" => FilterConfig::Threshold(ThresholdFilter::default()),
            "StreamTracer" => FilterConfig::StreamTracer(StreamTracerFilter::default()),
            "Tube" => FilterConfig::Tube(TubeFilter::default()),
            "Glyph" => FilterConfig::Glyph(GlyphFilter::default()),
            "WarpByVector" => FilterConfig::WarpByVector(WarpByVectorFilter::default()),
            "WarpByScalar" => FilterConfig::WarpByScalar(WarpByScalarFilter::default()),
            "Gradient" => FilterConfig::Gradient(GradientFilter::default()),
            "Calculator" => FilterConfig::Calculator(CalculatorFilter::default()),
            "Smooth" => FilterConfig::Smooth(SmoothFilter::default()),
            "Transform" => FilterConfig::Transform(TransformFilter::default()),
            "CellDataToPointData" => {
                FilterConfig::CellDataToPointData(CellDataToPointDataFilter::default())
            }
            "PointDataToCellData" => {
                FilterConfig::PointDataToCellData(PointDataToCellDataFilter::default())
            }
            "ProbeOverLine" => FilterConfig::ProbeOverLine(ProbeOverLineFilter::default()),
            "FeatureEdges" => FilterConfig::FeatureEdges(FeatureEdgesFilter::default()),
            "Connectivity" => FilterConfig::Connectivity(ConnectivityFilter::default()),
            "Shrink" => FilterConfig::Shrink(ShrinkFilter::default()),
            "CellCenters" => FilterConfig::CellCenters(CellCentersFilter::default()),
            "Reflect" => FilterConfig::Reflect(ReflectFilter::default()),
            "GenerateNormals" => FilterConfig::GenerateNormals(GenerateNormalsFilter::default()),
            "Elevation" => FilterConfig::Elevation(ElevationFilter::default()),
            "Triangulate" => FilterConfig::Triangulate(TriangulateFilter::default()),
            "ExtractEdges" => FilterConfig::ExtractEdges(ExtractEdgesFilter::default()),
            "CleanPolyData" => FilterConfig::CleanPolyData(CleanPolyDataFilter::default()),
            "AppendDatasets" => FilterConfig::AppendDatasets(AppendDatasetsFilter::default()),
            "IntegrateVariables" => {
                FilterConfig::IntegrateVariables(IntegrateVariablesFilter::default())
            }
            "Decimate" => FilterConfig::Decimate(DecimateFilter::default()),
            "Delaunay2D" => FilterConfig::Delaunay2D(Delaunay2DFilter::default()),
            "Delaunay3D" => FilterConfig::Delaunay3D(Delaunay3DFilter::default()),
            "IsoVolume" => FilterConfig::IsoVolume(IsoVolumeFilter::default()),
            "ResampleToImage" => FilterConfig::ResampleToImage(ResampleToImageFilter::default()),
            "ExtractBlock" => FilterConfig::ExtractBlock(ExtractBlockFilter::default()),
            "TemporalStatistics" => {
                FilterConfig::TemporalStatistics(TemporalStatisticsFilter::default())
            }
            "AddTimeSteps" => FilterConfig::AddTimeSteps(AddTimeStepsFilter::default()),
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown filter config '{key}'"
                )))
            }
        };
        serde_json::to_string(&cfg).map_err(to_py_err)
    }

    // ------------------------------------------------------------------
    // Generic pipeline surface (drives the `paraview.simple` shim)
    // ------------------------------------------------------------------

    /// Adds a source from a serialized `SourceConfig` JSON (e.g. `{"Sphere": {...}}`).
    fn add_source_cfg_json(&mut self, label: &str, json: &str) -> PyResult<u64> {
        let cfg = SourceConfig::from_json(json).map_err(to_py_err)?;
        self.inner
            .add_source_cfg(label, cfg)
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    /// Adds a single-input filter from a serialized `FilterConfig` JSON.
    fn add_filter_cfg_json(&mut self, label: &str, input: u64, json: &str) -> PyResult<u64> {
        let cfg = FilterConfig::from_json(json).map_err(to_py_err)?;
        self.inner
            .add_filter_cfg(label, NodeId(input), cfg)
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    /// Appends several inputs into one multi-input node (ParaView `AppendGeometry`-style).
    fn append_nodes(&mut self, label: &str, inputs: Vec<u64>) -> PyResult<u64> {
        let ids: Vec<NodeId> = inputs.into_iter().map(NodeId).collect();
        let id = self.inner.append_nodes(ids).map_err(to_py_err)?;
        self.inner
            .node_mut(id)
            .map_err(to_py_err)?
            .name = label.to_string();
        Ok(id.0)
    }

    /// Resamples `src` onto the point set of `dst` (ParaView `ResampleWithDataset`).
    fn resample_nodes(&mut self, src: u64, dst: u64) -> PyResult<u64> {
        self.inner
            .resample_with_dataset(NodeId(src), NodeId(dst))
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn source_cfg_json(&self, id: u64) -> PyResult<String> {
        let node = self.inner.node(NodeId(id)).map_err(to_py_err)?;
        let cfg = node.source_config.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!("node {id} is not a configurable source"))
        })?;
        serde_json::to_string(cfg).map_err(to_py_err)
    }

    fn filter_cfg_json(&self, id: u64) -> PyResult<String> {
        let node = self.inner.node(NodeId(id)).map_err(to_py_err)?;
        let cfg = node.filter_config.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!("node {id} is not a configurable filter"))
        })?;
        serde_json::to_string(cfg).map_err(to_py_err)
    }

    fn set_source_cfg_json(&mut self, id: u64, json: &str) -> PyResult<()> {
        let cfg = SourceConfig::from_json(json).map_err(to_py_err)?;
        self.inner
            .update_source_cfg(NodeId(id), cfg)
            .map_err(to_py_err)
    }

    fn set_filter_cfg_json(&mut self, id: u64, json: &str) -> PyResult<()> {
        let cfg = FilterConfig::from_json(json).map_err(to_py_err)?;
        self.inner
            .update_filter_cfg(NodeId(id), cfg)
            .map_err(to_py_err)
    }

    /// Lists all pipeline nodes as `(id, name, kind, parent_id)` tuples (`kind` is "source"/"filter", `parent_id` 0 for sources).
    fn nodes(&self) -> Vec<(u64, String, String, u64)> {
        self.inner
            .nodes()
            .iter()
            .map(|n| {
                let kind = if n.filter_config.is_some() {
                    "filter"
                } else {
                    "source"
                };
                (
                    n.id.0,
                    n.name.clone(),
                    kind.to_string(),
                    n.parent.map(|p| p.0).unwrap_or(0),
                )
            })
            .collect()
    }

    /// Removes a node and its downstream dependents; returns removed ids.
    fn remove_node(&mut self, id: u64) -> Vec<u64> {
        self.inner
            .remove_node(NodeId(id))
            .into_iter()
            .map(|n| n.0)
            .collect()
    }

    fn rename_node(&mut self, id: u64, name: &str) -> PyResult<()> {
        self.inner
            .node_mut(NodeId(id))
            .map_err(to_py_err)?
            .name = name.to_string();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Opening files
    // ------------------------------------------------------------------

    fn open(&mut self, path: &str) -> PyResult<u64> {
        self.inner.open(path).map(|id| id.0).map_err(to_py_err)
    }

    /// Opens a file with an explicit reader name (e.g. "VtkXml", "OpenFoam", "auto").
    fn open_with_reader(&mut self, path: &str, reader: &str) -> PyResult<u64> {
        let reader = parse_reader(reader).ok_or_else(|| {
            PyRuntimeError::new_err(format!("unknown reader '{reader}'"))
        })?;
        self.inner
            .open_with_reader(path, reader)
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    // ------------------------------------------------------------------
    // Pipeline evaluation & time
    // ------------------------------------------------------------------

    fn update_pipeline_node(&mut self, id: u64) -> PyResult<()> {
        self.inner
            .evaluate(NodeId(id))
            .map(|_| ())
            .map_err(to_py_err)
    }

    fn update_visible(&mut self) -> PyResult<()> {
        self.inner.update_visible().map_err(to_py_err)
    }

    fn update_all(&mut self) -> PyResult<()> {
        self.inner.update_all().map_err(to_py_err)
    }

    fn set_time(&mut self, time: f64) {
        self.inner.set_time(time);
    }

    fn time_steps(&self) -> Vec<f64> {
        self.inner.all_time_steps()
    }

    // ------------------------------------------------------------------
    // Display style (per-node `ActorStyle`)
    // ------------------------------------------------------------------

    fn set_visible(&mut self, id: u64, visible: bool) {
        if visible {
            self.inner.show(NodeId(id));
        } else {
            self.inner.hide(NodeId(id));
        }
    }

    fn set_representation(&mut self, id: u64, repr: &str) -> PyResult<()> {
        let repr = parse_representation(repr).ok_or_else(|| {
            PyRuntimeError::new_err(format!("unknown representation '{repr}'"))
        })?;
        self.inner.show(NodeId(id)).representation(repr);
        Ok(())
    }

    fn set_colormap(&mut self, id: u64, preset: &str) -> PyResult<()> {
        let preset = parse_colormap(preset).ok_or_else(|| {
            PyRuntimeError::new_err(format!("unknown colormap '{preset}'"))
        })?;
        self.inner.show(NodeId(id)).colormap(preset);
        Ok(())
    }

    /// Sets a custom color map from a flat ParaView `RGBPoints` list `[t, r, g, b, ...]` (0..1).
    fn set_custom_colormap(&mut self, id: u64, rgb_points: Vec<f64>) -> PyResult<()> {
        if rgb_points.len() % 4 != 0 || rgb_points.is_empty() {
            return Err(PyRuntimeError::new_err(
                "RGBPoints must be a non-empty flat [t, r, g, b, ...] list",
            ));
        }
        let stops = rgb_points
            .chunks(4)
            .map(|c| ColorStop {
                t: c[0] as f32,
                rgb: [c[1] as f32, c[2] as f32, c[3] as f32],
            })
            .collect();
        let mut map = ColorMap::from_preset(ColorMapPreset::CoolToWarm);
        map.name = "Custom".to_string();
        map.stops = stops;
        self.inner.show(NodeId(id)).custom_colormap(map);
        Ok(())
    }

    /// Colors a node by an attribute array: `assoc` is "POINTS"/"CELLS", `component` None = magnitude/scalar.
    fn set_color_by(
        &mut self,
        id: u64,
        assoc: &str,
        array: &str,
        component: Option<usize>,
    ) {
        let h = self.inner.show(NodeId(id));
        match parse_association(assoc) {
            AttributeAssociation::Cell => {
                if let Some(c) = component {
                    // no cell-component handle on DisplayHandle; fall back to the array
                    let _ = c;
                    h.color_by_cell(array);
                } else {
                    h.color_by_cell(array);
                }
            }
            _ => match component {
                Some(c) => {
                    h.color_by_point_component(array, c);
                }
                None => {
                    h.color_by_point(array);
                }
            },
        }
    }

    fn set_solid_color(&mut self, id: u64, rgba: Vec<f64>) {
        let rgba = [
            rgba.first().copied().unwrap_or(1.0) as f32,
            rgba.get(1).copied().unwrap_or(1.0) as f32,
            rgba.get(2).copied().unwrap_or(1.0) as f32,
            rgba.get(3).copied().unwrap_or(1.0) as f32,
        ];
        self.inner.show(NodeId(id)).solid_color(rgba);
    }

    fn set_opacity(&mut self, id: u64, opacity: f64) {
        self.inner.show(NodeId(id)).opacity(opacity as f32);
    }

    fn set_line_width(&mut self, id: u64, width: f64) {
        self.inner.show(NodeId(id)).line_width(width as f32);
    }

    fn set_point_size(&mut self, id: u64, size: f64) {
        self.inner.show(NodeId(id)).point_size(size as f32);
    }

    fn set_edge_color(&mut self, id: u64, rgba: Vec<f64>) {
        let rgba = [
            rgba.first().copied().unwrap_or(0.0) as f32,
            rgba.get(1).copied().unwrap_or(0.0) as f32,
            rgba.get(2).copied().unwrap_or(0.0) as f32,
            rgba.get(3).copied().unwrap_or(1.0) as f32,
        ];
        self.inner.show(NodeId(id)).edge_color(rgba);
    }

    fn set_edge_width(&mut self, id: u64, width: f64) {
        self.inner.show(NodeId(id)).edge_width(width as f32);
    }

    fn set_scalar_range(&mut self, id: u64, min: f64, max: f64) {
        self.inner.show(NodeId(id)).scalar_range(min, max);
    }

    fn set_auto_range(&mut self, id: u64, enabled: bool) {
        self.inner.show(NodeId(id)).auto_range(enabled);
    }

    /// Serializes the node's `ActorStyle` as JSON (for `GetDisplayProperties` reads).
    fn style_json(&self, id: u64) -> PyResult<String> {
        let node = self.inner.node(NodeId(id)).map_err(to_py_err)?;
        serde_json::to_string(&node.style).map_err(to_py_err)
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    fn render_png(&mut self, path: &str, width: u32, height: u32) -> PyResult<()> {
        self.inner
            .render_png(path, width, height)
            .map_err(to_py_err)
    }

    /// Renders headlessly and returns the RGBA pixels as `bytes`.
    fn render_rgba<'py>(
        &mut self,
        width: u32,
        height: u32,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let rgba = self
            .inner
            .render_rgba(width, height)
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &rgba))
    }

    fn reset_camera(&mut self) -> PyResult<()> {
        self.inner.reset_camera().map_err(to_py_err)
    }

    fn render_turntable_gif(
        &mut self,
        path: &str,
        width: u32,
        height: u32,
        frames: u32,
        delay_ms: u16,
    ) -> PyResult<()> {
        self.inner
            .render_turntable_gif(path, width, height, frames, delay_ms)
            .map_err(to_py_err)
    }

    fn render_timeseries_gif(
        &mut self,
        path: &str,
        width: u32,
        height: u32,
        delay_ms: u16,
    ) -> PyResult<()> {
        self.inner
            .render_timeseries_gif(path, width, height, delay_ms)
            .map_err(to_py_err)
    }

    // ------------------------------------------------------------------
    // Data information & export
    // ------------------------------------------------------------------

    /// Evaluates the node and returns a JSON summary (type, counts, bounds, arrays, time steps).
    fn data_info_json(&mut self, id: u64) -> PyResult<String> {
        let ds = self
            .inner
            .evaluate(NodeId(id))
            .map_err(to_py_err)?;
        let node = self.inner.node(NodeId(id)).map_err(to_py_err)?;
        let time_steps = node.time_steps.clone();

        let mut point_arrays = Vec::new();
        let mut cell_arrays = Vec::new();
        let mut active_scalars = None;
        let mut active_vectors = None;
        let collect = |fd: &FieldData, out: &mut Vec<serde_json::Value>| {
            for arr in fd.arrays() {
                let (rmin, rmax) = arr.range(None);
                out.push(serde_json::json!({
                    "name": arr.name,
                    "components": arr.num_components,
                    "range": [rmin, rmax],
                }));
            }
        };
        match ds.as_ref() {
            DataSet::MultiBlock(mb) => {
                for (_, block) in &mb.blocks {
                    if let Some(fd) = block.point_data() {
                        if active_scalars.is_none() {
                            active_scalars = fd.active_scalars.clone();
                        }
                        if active_vectors.is_none() {
                            active_vectors = fd.active_vectors.clone();
                        }
                        collect(fd, &mut point_arrays);
                    }
                    if let Some(fd) = block.cell_data() {
                        collect(fd, &mut cell_arrays);
                    }
                }
            }
            other => {
                if let Some(fd) = other.point_data() {
                    active_scalars = fd.active_scalars.clone();
                    active_vectors = fd.active_vectors.clone();
                    collect(fd, &mut point_arrays);
                }
                if let Some(fd) = other.cell_data() {
                    collect(fd, &mut cell_arrays);
                }
            }
        }
        let b = ds.bounds();
        let info = serde_json::json!({
            "type": ds.type_name(),
            "num_points": ds.num_points(),
            "num_cells": ds.num_cells(),
            "bounds": [b.min.x, b.max.x, b.min.y, b.max.y, b.min.z, b.max.z],
            "point_arrays": point_arrays,
            "cell_arrays": cell_arrays,
            "active_scalars": active_scalars,
            "active_vectors": active_vectors,
            "time_steps": time_steps,
        });
        serde_json::to_string(&info).map_err(to_py_err)
    }

    /// Exports the evaluated dataset of a node to `path` (format inferred from extension).
    fn export_dataset(&mut self, id: u64, path: &str) -> PyResult<()> {
        self.inner
            .export_dataset(NodeId(id), path)
            .map_err(to_py_err)
    }

    /// Exports an A/B comparison viewer: nodes `id_a` and `id_b` in one self-contained
    /// `.html` file (shared camera, wipe slider, common scalar legend).
    fn export_html_ab(&mut self, id_a: u64, id_b: u64, path: &str, colors: bool) -> PyResult<()> {
        self.inner
            .export_html_ab(NodeId(id_a), NodeId(id_b), path, colors)
            .map_err(to_py_err)
    }

    // ------------------------------------------------------------------
    // Fetch (client-side dataset access, `paraview.simple.Fetch`)
    // ------------------------------------------------------------------

    /// JSON metadata of the evaluated dataset (arrays with dtype/range, bounds, blocks, image geometry).
    fn fetch_meta_json(&mut self, id: u64, block: Option<usize>) -> PyResult<String> {
        let ds = resolve_dataset(&mut self.inner, id, block)?;
        let b = ds.bounds();
        let mut meta = serde_json::json!({
            "type": ds.type_name(),
            "num_points": ds.num_points(),
            "num_cells": ds.num_cells(),
            "bounds": [b.min.x, b.max.x, b.min.y, b.max.y, b.min.z, b.max.z],
        });
        match ds.as_ref() {
            DataSet::ImageData(img) => {
                meta["dimensions"] = serde_json::json!(img.dimensions);
                meta["origin"] = serde_json::json!(img.origin);
                meta["spacing"] = serde_json::json!(img.spacing);
            }
            DataSet::StructuredGrid(sg) => {
                meta["dimensions"] = serde_json::json!(sg.dimensions);
            }
            DataSet::MultiBlock(mb) => {
                meta["blocks"] = serde_json::json!(mb.blocks.iter().map(|(n, _)| n).collect::<Vec<_>>());
            }
            _ => {}
        }
        if let Some(fd) = ds.point_data() {
            meta["point_data"] = field_json(fd);
        }
        if let Some(fd) = ds.cell_data() {
            meta["cell_data"] = field_json(fd);
        }
        serde_json::to_string(&meta).map_err(to_py_err)
    }

    /// Returns `(dtype, num_components, raw_bytes)` for one attribute array.
    fn get_array_bytes<'py>(
        &mut self,
        id: u64,
        assoc: &str,
        name: &str,
        block: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<(String, usize, Bound<'py, PyBytes>)> {
        let ds = resolve_dataset(&mut self.inner, id, block)?;
        let fd = match parse_association(assoc) {
            AttributeAssociation::Cell => ds.cell_data(),
            _ => ds.point_data(),
        };
        let arr = fd
            .and_then(|fd| fd.get(name))
            .ok_or_else(|| PyRuntimeError::new_err(format!("no {assoc} array '{name}'")))
            .map_err(to_py_err)?;
        let dtype = array_dtype(&arr.data).to_string();
        let ncomp = arr.num_components;
        let bytes = PyBytes::new(py, &array_bytes(&arr.data));
        Ok((dtype, ncomp, bytes))
    }

    /// Returns `(num_points, raw_f32_xyz_bytes)` for the point coordinates.
    fn get_points_bytes<'py>(
        &mut self,
        id: u64,
        block: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<(usize, Bound<'py, PyBytes>)> {
        let ds = resolve_dataset(&mut self.inner, id, block)?;
        let points: Vec<[f32; 3]> = match ds.as_ref() {
            DataSet::PolyData(d) => d.points.clone(),
            DataSet::UnstructuredGrid(d) => d.points.clone(),
            DataSet::StructuredGrid(d) => d.points.clone(),
            // Implicit-geometry grids: synthesize the point cloud like vtkRectilinearGrid does.
            DataSet::ImageData(img) => {
                let [nx, ny, nz] = img.dimensions;
                let mut pts = Vec::with_capacity(nx * ny * nz);
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            pts.push([
                                (img.origin[0] + i as f64 * img.spacing[0]) as f32,
                                (img.origin[1] + j as f64 * img.spacing[1]) as f32,
                                (img.origin[2] + k as f64 * img.spacing[2]) as f32,
                            ]);
                        }
                    }
                }
                pts
            }
            DataSet::RectilinearGrid(rg) => {
                let mut pts = Vec::with_capacity(rg.x_coords.len() * rg.y_coords.len() * rg.z_coords.len());
                for &z in &rg.z_coords {
                    for &y in &rg.y_coords {
                        for &x in &rg.x_coords {
                            pts.push([x, y, z]);
                        }
                    }
                }
                pts
            }
            DataSet::MultiBlock(_) => Vec::new(),
        };
        let n = points.len();
        Ok((n, PyBytes::new(py, bytemuck::cast_slice(&points))))
    }

    /// Returns `(offsets_u32, connectivity_u32, types_u8)` in VTK cell-array layout.
    fn get_cells_bytes<'py>(
        &mut self,
        id: u64,
        block: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let ds = resolve_dataset(&mut self.inner, id, block)?;
        let (offsets, conn, types): (Vec<u32>, Vec<u32>, Vec<u8>) = match ds.as_ref() {
            DataSet::UnstructuredGrid(d) => (
                d.offsets.clone(),
                d.connectivity.clone(),
                d.cell_types.iter().map(|t| *t as u8).collect(),
            ),
            DataSet::PolyData(d) => {
                let mut offsets = Vec::new();
                let mut conn = Vec::new();
                let mut types = Vec::new();
                let mut push = |cell: &[u32], ty: u8| {
                    offsets.push(conn.len() as u32);
                    conn.extend_from_slice(cell);
                    types.push(ty);
                };
                for v in &d.verts {
                    push(&[*v], 2); // VTK_POLY_VERTEX
                }
                for l in &d.lines {
                    push(l, 4); // VTK_POLY_LINE
                }
                for p in &d.polys {
                    push(p, 7); // VTK_POLYGON
                }
                for s in &d.strips {
                    push(s, 6); // VTK_TRIANGLE_STRIP
                }
                (offsets, conn, types)
            }
            // Structured/implicit grids: cells are implicit (see fetch_meta_json dimensions).
            _ => (Vec::new(), Vec::new(), Vec::new()),
        };
        Ok((
            PyBytes::new(py, bytemuck::cast_slice(&offsets)),
            PyBytes::new(py, bytemuck::cast_slice(&conn)),
            PyBytes::new(py, bytemuck::cast_slice(&types)),
        ))
    }

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------

    fn save_state_json(&self) -> PyResult<String> {
        self.inner.save_state_json().map_err(to_py_err)
    }

    fn load_state_json(&mut self, json: &str) -> PyResult<()> {
        self.inner = ParaViz::load_state_json(json).map_err(to_py_err)?;
        Ok(())
    }

    fn to_rust_trace(&self) -> String {
        self.inner.to_rust_trace()
    }

    // ------------------------------------------------------------------
    // Legacy typed helpers (kept for backward compatibility)
    // ------------------------------------------------------------------

    fn wavelet(&mut self, label: &str) -> u64 {
        self.inner.add_source(label, WaveletSource::default()).0
    }

    fn sphere(&mut self, label: &str, radius: f32) -> u64 {
        let mut s = SphereSource::default();
        s.radius = radius;
        self.inner.add_source(label, s).0
    }

    fn contour(&mut self, label: &str, input: u64, field: &str, values: Vec<f64>) -> PyResult<u64> {
        self.inner
            .add_filter(label, NodeId(input), ContourFilter::new(field, values))
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn decimate(&mut self, label: &str, input: u64, target_reduction: f32) -> PyResult<u64> {
        self.inner
            .add_filter(label, NodeId(input), DecimateFilter::new(target_reduction))
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn isovolume(
        &mut self,
        label: &str,
        input: u64,
        field: &str,
        min_val: f64,
        max_val: f64,
    ) -> PyResult<u64> {
        self.inner
            .add_filter(
                label,
                NodeId(input),
                IsoVolumeFilter::new(field, min_val, max_val),
            )
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn integrate_variables(&mut self, label: &str, input: u64) -> PyResult<u64> {
        self.inner
            .add_filter(label, NodeId(input), IntegrateVariablesFilter::new())
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn resample_with_dataset(&mut self, source: u64, dest: u64) -> PyResult<u64> {
        self.inner
            .resample_with_dataset(NodeId(source), NodeId(dest))
            .map(|id| id.0)
            .map_err(to_py_err)
    }

    fn show(&mut self, node: u64, repr: Option<&str>, colormap: Option<&str>, color_by: Option<&str>) {
        let h = self.inner.show(NodeId(node));
        let h = match repr.and_then(parse_representation) {
            Some(r) => h.representation(r),
            None => h.representation(Representation::Surface),
        };
        let h = match colormap.and_then(parse_colormap) {
            Some(c) => h.colormap(c),
            None => h.colormap(ColorMapPreset::CoolToWarm),
        };
        if let Some(arr) = color_by {
            h.color_by_point(arr);
        }
    }

    fn hide(&mut self, node: u64) {
        self.inner.hide(NodeId(node));
    }
}

#[pymodule]
fn _paraviz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySession>()?;
    Ok(())
}
