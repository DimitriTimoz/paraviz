//! Python bindings for ParaViz (`import paraviz`), mirroring `paraview.simple`.

use ::paraviz::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "Session")]
struct PySession {
    inner: ParaViz,
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl PySession {
    #[new]
    fn new() -> Self {
        Self {
            inner: ParaViz::new(),
        }
    }

    fn open(&mut self, path: &str) -> PyResult<u64> {
        self.inner.open(path).map(|id| id.0).map_err(to_py_err)
    }

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
        let h = match repr {
            Some("edges") => h.representation(Representation::SurfaceWithEdges),
            Some("wireframe") => h.representation(Representation::Wireframe),
            Some("points") => h.representation(Representation::Points),
            Some("gaussian") => h.representation(Representation::PointGaussian),
            Some("volume") => h.representation(Representation::Volume),
            Some("outline") => h.representation(Representation::Outline),
            _ => h.representation(Representation::Surface),
        };
        let h = match colormap {
            Some("viridis") => h.colormap(ColorMapPreset::Viridis),
            Some("plasma") => h.colormap(ColorMapPreset::Plasma),
            Some("inferno") => h.colormap(ColorMapPreset::Inferno),
            Some("turbo") => h.colormap(ColorMapPreset::Turbo),
            _ => h.colormap(ColorMapPreset::CoolToWarm),
        };
        if let Some(arr) = color_by {
            h.color_by_point(arr);
        }
    }

    fn hide(&mut self, node: u64) {
        self.inner.hide(NodeId(node));
    }

    fn reset_camera(&mut self) -> PyResult<()> {
        self.inner.reset_camera().map_err(to_py_err)
    }

    fn render_png(&mut self, path: &str, width: u32, height: u32) -> PyResult<()> {
        self.inner.render_png(path, width, height).map_err(to_py_err)
    }

    fn save_state_json(&self) -> PyResult<String> {
        self.inner.save_state_json().map_err(to_py_err)
    }

    fn to_rust_trace(&self) -> String {
        self.inner.to_rust_trace()
    }
}

#[pymodule]
fn paraviz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySession>()?;
    Ok(())
}
