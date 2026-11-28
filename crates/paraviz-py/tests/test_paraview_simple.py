"""Compatibility tests: classic `paraview.simple` scripts running unmodified on ParaViz.

Run with:  python -m pytest tests/ -q   (inside the maturin venv)
"""

import json
import os

import pytest

import paraviz
from paraview.simple import *  # noqa: F401,F403 — exactly like a ParaView script
import paraview.simple as pvs


@pytest.fixture(autouse=True)
def fresh_session():
    Disconnect()
    yield
    Disconnect()


def _sess():
    return pvs._get_session()


# ---------------------------------------------------------------------------
# Classic ParaView script, end to end
# ---------------------------------------------------------------------------


def test_classic_paraview_script(tmp_path):
    # 1. produce a dataset file through the compat API itself
    w = Wavelet()
    vti = str(tmp_path / "wavelet.vti")
    SaveData(vti, proxy=w)
    assert os.path.getsize(vti) > 0

    # 2. the classic reader -> filter -> show -> render script, unmodified
    reader = XMLImageDataReader(FileName=[vti])
    reader.UpdatePipeline()

    info = reader.GetDataInformation()
    assert info.GetNumberOfPoints() > 0
    arr = info.PointData.GetArray("RTData")
    assert arr is not None and arr.GetNumberOfComponents() == 1
    rmin, rmax = arr.GetRange()
    assert rmin < rmax

    contour = Contour(Input=reader)
    contour.SelectInputScalars = ["POINTS", "RTData"]
    contour.ContourValue = [0.5 * (rmin + rmax)]
    contour.ComputeNormals = True
    contour.UpdatePipeline()

    Show(contour, representation="Surface With Edges")
    Hide(reader)
    ColorBy(contour, ("POINTS", "RTData"))

    lut = GetColorTransferFunction("RTData")
    lut.ApplyPreset("Viridis (matplotlib)", True)

    view = GetRenderView()
    view.ViewSize = [320, 240]
    Render()

    png = str(tmp_path / "out.png")
    SaveScreenshot(png)
    assert os.path.getsize(png) > 1000

    # pipeline introspection
    sources = GetSources()
    filters = GetFilters()
    assert any("wavelet" in k.lower() for k in sources)
    assert any("Contour" in k for k in filters)

    # config round-trip check: iso value landed in the Rust FilterConfig
    cfg = json.loads(_sess().filter_cfg_json(contour._node_id))
    assert cfg["Contour"]["iso_values"] == [0.5 * (rmin + rmax)]
    assert cfg["Contour"]["field_name"] == "RTData"
    assert cfg["Contour"]["compute_normals"] is True

    # display properties reflect Show/ColorBy
    dp = GetDisplayProperties(contour)
    assert dp.Visibility is True
    assert dp.Representation == "Surface With Edges"
    assert dp.ColorArrayName == ["POINTS", "RTData"]

    # reader hidden
    style = json.loads(_sess().style_json(reader._node_id))
    assert style["visible"] is False


def test_gpu_render_available():
    assert paraviz.Session.gpu_available() is True
    w = Wavelet()
    Show(w)
    Render()
    rgba = _sess().render_rgba(64, 64)
    assert len(rgba) == 64 * 64 * 4


# ---------------------------------------------------------------------------
# Property mapping details
# ---------------------------------------------------------------------------


def test_source_properties():
    s = Sphere()
    s.Center = [1.0, 2.0, 3.0]
    s.Radius = 2.5
    s.ThetaResolution = 64
    cfg = json.loads(_sess().source_cfg_json(s._node_id))
    assert cfg["Sphere"]["center"] == [1.0, 2.0, 3.0]
    assert cfg["Sphere"]["radius"] == 2.5
    assert cfg["Sphere"]["theta_resolution"] == 64
    # read-back through the ParaView attribute
    assert s.Radius == 2.5


def test_clip_plane_properties():
    s = Sphere()
    clip = Clip(Input=s)
    clip.ClipType = "Plane"
    clip.Origin = [0.0, 0.0, 0.1]
    clip.Normal = [0.0, 0.0, 1.0]
    clip.InsideOut = True
    cfg = json.loads(_sess().filter_cfg_json(clip._node_id))
    assert cfg["Clip"]["function"] == {"Plane": {"origin": [0.0, 0.0, 0.1], "normal": [0.0, 0.0, 1.0]}}
    assert cfg["Clip"]["invert"] is True


def test_clip_sphere_properties():
    s = Sphere()
    clip = Clip(Input=s)
    clip.ClipType = "Sphere"
    clip.Center = [0.0, 0.0, 0.0]
    clip.Radius = 0.3
    cfg = json.loads(_sess().filter_cfg_json(clip._node_id))
    assert cfg["Clip"]["function"] == {"Sphere": {"center": [0.0, 0.0, 0.0], "radius": 0.3}}


def test_threshold_properties():
    w = Wavelet()
    t = Threshold(Input=w)
    t.SelectInputScalars = ["CELLS", "RTData"]
    t.ThresholdRange = [0.1, 0.9]
    cfg = json.loads(_sess().filter_cfg_json(t._node_id))
    assert cfg["Threshold"]["field_name"] == "RTData"
    assert cfg["Threshold"]["association"] == "Cell"
    assert cfg["Threshold"]["min"] == 0.1
    assert cfg["Threshold"]["max"] == 0.9


def test_stream_tracer_properties():
    w = Wavelet()
    st = StreamTracer(Input=w)
    st.SeedType = "Line"
    st.Point1 = [-1.0, 0.0, 0.0]
    st.Point2 = [1.0, 0.0, 0.0]
    st.Resolution = 20
    st.IntegrationDirection = "FORWARD"
    st.IntegratorType = "RK45"
    st.MaximumSteps = 1000
    cfg = json.loads(_sess().filter_cfg_json(st._node_id))
    assert cfg["StreamTracer"]["seeds"] == {"Line": {"p1": [-1.0, 0.0, 0.0], "p2": [1.0, 0.0, 0.0], "count": 20}}
    assert cfg["StreamTracer"]["direction"] == "Forward"
    assert cfg["StreamTracer"]["integrator"] == "Rk45"
    assert cfg["StreamTracer"]["max_steps"] == 1000


def test_calculator_and_glyph_properties():
    w = Wavelet()
    calc = Calculator(Input=w)
    calc.Function = "RTData*2"
    calc.ResultArrayName = "double"
    calc.AttributeType = "Point Data"
    cfg = json.loads(_sess().filter_cfg_json(calc._node_id))
    assert cfg["Calculator"] == {
        "expression": "RTData*2",
        "result_name": "double",
        "association": "Point",
    }

    g = Glyph(Input=w)
    g.GlyphType = "Arrow"
    g.ScaleMode = "Data Scaling Off"
    g.ScaleFactor = 0.1
    cfg = json.loads(_sess().filter_cfg_json(g._node_id))
    assert cfg["Glyph"]["shape"] == "Arrow"
    assert cfg["Glyph"]["scale_mode"] == "Constant"
    assert cfg["Glyph"]["scale_factor"] == 0.1


def test_unknown_reader_fallback(tmp_path):
    w = Wavelet()
    vti = str(tmp_path / "w.vti")
    SaveData(vti, proxy=w)
    # a reader class name paraview.simple would not know -> auto file reader
    r = pvs.SomeExoticReader(FileName=[vti])
    r.UpdatePipeline()
    info = r.GetDataInformation()
    assert info.PointData.GetArray("RTData") is not None


def test_append_geometry():
    a = Sphere()
    b = Cone()
    app = AppendGeometry(Input=[a, b])
    app.UpdatePipeline()
    info = app.GetDataInformation()
    assert info.GetNumberOfPoints() > 0


def test_delete_and_rename():
    s = Sphere()
    RenameSource("MySphere", s)
    names = [n[1] for n in _sess().nodes()]
    assert "MySphere" in names
    Delete(s)
    assert all(n[0] != s._node_id for n in _sess().nodes())


def test_animation_scene_time():
    w = Wavelet()
    scene = GetAnimationScene()
    scene.AnimationTime = 0.0
    scene.GoToLast()
    steps = _sess().time_steps()
    assert isinstance(steps, list)


def test_lut_rgb_points():
    w = Wavelet()
    Show(w)
    lut = GetColorTransferFunction("RTData")
    lut.RGBPoints = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    style = json.loads(_sess().style_json(w._node_id))
    stops = style["colormap"]["stops"]
    assert stops[0]["t"] == 0.0 and stops[-1]["t"] == 1.0
    assert lut.RGBPoints[0] == 0.0 and lut.RGBPoints[-1] == 1.0


def test_state_json_contains_pipeline():
    w = Wavelet()
    c = Contour(Input=w)
    c.ContourValue = [123.0]
    state = _sess().save_state_json()
    assert "Contour" in state and "123.0" in state


def test_import_star_exports():
    for name in ("XMLUnstructuredGridReader", "Sphere", "Contour", "Clip", "Render",
                 "Show", "Hide", "ColorBy", "SaveScreenshot", "GetRenderView", "Fetch"):
        assert name in pvs.__all__


# ---------------------------------------------------------------------------
# Fetch: client-side data access (paraview.servermanager-style)
# ---------------------------------------------------------------------------


def test_fetch_source_arrays_and_points():
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    w = Wavelet()
    data = Fetch(w)
    assert data.GetNumberOfPoints() > 0

    rt = data.PointData["RTData"]
    assert rt.GetName() == "RTData"
    assert rt.GetNumberOfComponents() == 1
    a = vtk_to_numpy(rt)
    assert isinstance(a, np.ndarray)
    assert a.ndim == 1 and a.shape[0] == data.GetNumberOfPoints()
    rmin, rmax = rt.GetRange()
    assert rmin <= float(a.min()) and float(a.max()) <= rmax

    pts = vtk_to_numpy(data.GetPoints().GetData())
    assert pts.shape == (data.GetNumberOfPoints(), 3)
    b = data.GetBounds()
    assert pts[:, 0].min() >= b[0] - 1e-4 and pts[:, 0].max() <= b[3] + 1e-4

    # np.asarray works directly on the array view
    assert np.asarray(rt).shape == a.shape


def test_fetch_from_file_reader(tmp_path):
    import numpy as np
    from vtkmodules.util.numpy_support import vtk_to_numpy  # VTK 9.x import path

    w = Wavelet()
    vti = str(tmp_path / "w.vti")
    SaveData(vti, proxy=w)

    r = XMLImageDataReader(FileName=[vti])
    data = Fetch(r)
    assert data.GetNumberOfPoints() > 0
    a = vtk_to_numpy(data.PointData["RTData"])
    assert a.shape[0] == data.GetNumberOfPoints()
    # image data exposes its implicit geometry through extent/origin/spacing
    assert len(data.GetDimensions()) == 3
    assert len(data.GetOrigin()) == 3
    assert len(data.GetSpacing()) == 3


def test_fetch_filter_output_and_cells():
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    s = Sphere()
    s.Radius = 1.0
    surf = ExtractSurface(Input=s)
    data = Fetch(surf)
    cells = data.GetCells()
    assert cells.GetNumberOfCells() > 0
    conn = cells.connectivity
    types = cells.types
    assert len(conn) > 0 and len(types) == cells.GetNumberOfCells()
    pts = vtk_to_numpy(data.GetPoints().GetData())
    assert pts.max() <= 1.0 + 1e-4  # sphere radius 1


def test_fetch_time_step():
    w = Wavelet()
    scene = GetAnimationScene()
    scene.AnimationTime = 0.0
    data = Fetch(w, time=0.0)
    assert data.GetNumberOfPoints() > 0


def test_servermanager_module():
    from paraview.servermanager import Fetch as SMFetch

    w = Wavelet()
    data = SMFetch(w)
    assert data.PointData.GetArray("RTData").GetNumberOfTuples() == data.GetNumberOfPoints()


def test_numpy_to_vtk_roundtrip():
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    arr = numpy_to_vtk(a, deep=True)
    assert arr.GetNumberOfComponents() == 3
    assert arr.GetNumberOfTuples() == 4
    back = vtk_to_numpy(arr)
    assert np.array_equal(back, a)


def test_export_html_ab_single_file(tmp_path):
    """A/B compare export: one self-contained .html with slider + common legend."""
    w = Wavelet()
    w.WholeExtent = [-4, 4, -4, 4, -4, 4]
    a = Contour(Input=w)
    a.ContourValue = [120.0]
    b = Contour(Input=w)
    b.ContourValue = [200.0]
    a.UpdatePipeline()
    b.UpdatePipeline()

    out = tmp_path / "ab.html"
    _sess().export_html_ab(a._node_id, b._node_id, str(out), True)
    html = out.read_text()
    assert html.count("<canvas") == 1
    assert 'id="abSlider"' in html
    assert 'id="legendBar"' in html
    assert '"labelA":"Contour"' in html
    assert '"array":"RTData"' in html
