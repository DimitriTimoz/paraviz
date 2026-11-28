"""Robustness corpus: realistic end-to-end ParaView-style workflows, state
round-trips, cascades, exports, and error behavior — all under the shim.
"""

import json
import os
import subprocess
import sys

import pytest

import paraviz
import paraview.simple as pvs
from paraview.simple import *


@pytest.fixture(autouse=True)
def fresh():
    Disconnect()
    yield
    Disconnect()


def _sess():
    return pvs._get_session()


def test_full_workflow_pipeline(tmp_path):
    """Reader -> Contour -> Tube-like chain -> Clip -> Show/Hide -> render -> save."""
    w = Wavelet()
    w.WholeExtent = [-4, 4, -4, 4, -4, 4]
    w.UpdatePipeline()

    c = Contour(Input=w)
    c.SelectInputScalars = ["POINTS", "RTData"]
    c.ContourValue = [80.0, 160.0]
    c.UpdatePipeline()

    clip = Clip(Input=c)
    clip.ClipType = "Plane"
    clip.Origin = [0.0, 0.0, 0.0]
    clip.Normal = [1.0, 0.0, 0.0]
    clip.UpdatePipeline()

    Show(clip, representation="Surface With Edges")
    Hide(w)
    ColorBy(clip, ("POINTS", "RTData"))
    dp = GetDisplayProperties(clip)
    dp.Opacity = 0.8
    assert abs(dp.Opacity - 0.8) < 1e-6
    dp.Representation = "Wireframe"
    assert dp.Representation == "Wireframe"

    view = GetRenderView()
    view.ViewSize = [256, 256]
    Render()
    png = str(tmp_path / "wf.png")
    SaveScreenshot(png)
    assert os.path.getsize(png) > 1000

    # export the filter output
    out = str(tmp_path / "clip.vtp")
    SaveData(out, proxy=clip)
    assert os.path.getsize(out) > 0

    # read it back through a reader (round trip through disk)
    r = XMLPolyDataReader(FileName=[out])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() > 0
    assert d.PointData["RTData"].GetNumberOfTuples() == d.GetNumberOfPoints()


def test_state_save_load_roundtrip():
    w = Wavelet()
    c = Contour(Input=w)
    c.ContourValue = [111.0]
    state = _sess().save_state_json()
    assert "111.0" in state

    Disconnect()
    s2 = _sess()
    s2.load_state_json(state)
    nodes = s2.nodes()
    assert any("Contour" in n[1] for n in nodes)
    cfg = json.loads(s2.filter_cfg_json([n[0] for n in nodes if "Contour" in n[1]][0]))
    assert cfg["Contour"]["iso_values"] == [111.0]


def test_delete_cascades_downstream():
    w = Wavelet()
    c = Contour(Input=w)
    c.ContourValue = [100.0]
    n_before = len(_sess().nodes())
    Delete(w)
    n_after = _sess().nodes()
    assert len(n_after) == n_before - 2  # source + dependent filter removed
    assert all("Contour" not in n[1] for n in n_after)


def test_views_management():
    v1 = GetRenderView()
    v2 = CreateView("RenderView")
    v2.ViewSize = [100, 100]
    views = GetViews()
    assert len(views) >= 2
    assert GetActiveView() is v1
    RenderAllViews()


def test_animation_time_steps():
    w = Wavelet()  # 5 time steps: 0, .25, .5, .75, 1
    scene = GetAnimationScene()
    scene.GoToLast()
    steps = _sess().time_steps()
    assert len(steps) == 5
    d = Fetch(w, time=steps[-1])
    assert d.GetNumberOfPoints() > 0


def test_save_animation_gif(tmp_path):
    w = Wavelet()
    w.WholeExtent = [-2, 2, -2, 2, -2, 2]
    Show(w)
    view = GetRenderView()
    view.ViewSize = [128, 128]
    gif = str(tmp_path / "anim.gif")
    SaveAnimation(gif, view=view, frames=8)
    assert os.path.getsize(gif) > 500
    with open(gif, "rb") as f:
        assert f.read(6) in (b"GIF89a", b"GIF87a")


def test_export_html_viewer(tmp_path):
    s = Sphere()
    s.Radius = 1.0
    s.UpdatePipeline()
    out = str(tmp_path / "viewer.html")
    SaveData(out, proxy=s)
    with open(out) as f:
        content = f.read()
    assert "<html" in content.lower()


def test_lut_presets_all_apply():
    w = Wavelet()
    w.WholeExtent = [-2, 2, -2, 2, -2, 2]
    Show(w)
    lut = GetColorTransferFunction("RTData")
    for preset in ["Viridis (matplotlib)", "Plasma (matplotlib)", "Cool to Warm", "Grayscale", "Rainbow"]:
        lut.ApplyPreset(preset, True)
        style = json.loads(_sess().style_json(w._node_id))
        assert style["colormap"]["stops"], preset


def test_disconnect_resets_pipeline():
    Wavelet()
    assert len(_sess().nodes()) == 1
    Disconnect()
    assert len(_sess().nodes()) == 0
    assert pvs.GetSources() == {}


def test_get_sources_filters_reflect_pipeline():
    w = Wavelet()
    c = Contour(Input=w)
    c.ContourValue = [1.0]
    srcs = GetSources()
    flts = GetFilters()
    assert any(type(p).__name__ == "Wavelet" for p in srcs.values())
    assert any(type(p).__name__ == "Contour" for p in flts.values())


def test_differential_cases_run_clean(tmp_path):
    """Every differential case script must execute under the shim."""
    here = os.path.dirname(os.path.abspath(__file__))
    cases_dir = os.path.join(here, "differential", "cases")
    for f in sorted(os.listdir(cases_dir)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        out = str(tmp_path / (f[:-3] + ".json"))
        r = subprocess.run(
            [sys.executable, os.path.join(here, "differential", "run_case.py"), f[:-3], out],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, f"{f}:\n{r.stdout}\n{r.stderr}"
        with open(out) as fh:
            trace = json.load(fh)
        assert trace["fetches"], f"{f}: no fetch trace"
