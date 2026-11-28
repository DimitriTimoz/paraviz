"""Golden property-matrix tests: every registered proxy class, every mapped
ParaView property -> assert it lands in the Rust config JSON (or is accepted).
Plus error-parity and evaluable-filter tests.
"""

import json

import pytest

import paraviz
import paraview.simple as pvs


def _sess():
    return pvs._get_session()


@pytest.fixture(autouse=True)
def fresh():
    pvs.Disconnect()
    yield
    pvs.Disconnect()


def _sample(name, entry):
    n = name.lower()
    if entry is pvs._IGNORE:
        return 1
    if "scalars" in n or "vectors" in n:
        return ["POINTS", "RTData"] if "vector" not in n else ["POINTS", "Velocity"]
    if "extent" in n:
        return [0, 4, 0, 4, 0, 4]
    if "dimension" in n:
        return [4, 4, 4]
    if "indices" in n:
        return [1, 2]
    if "range" in n or "between" in n:
        return [0.2, 0.8]
    if any(k in n for k in ("factor", "angle", "reduction", "opacity", "width", "size", "tolerance",
                            "alpha", "value", "threshold", "radius", "length", "height", "speed",
                            "error", "propagation", "deviation", "maximum", "minimum", "step",
                            "magnitude", "frequency", "deviation")):
        return 2.5
    if any(k in n for k in ("resolution", "steps", "sides", "iterations", "count", "points")):
        return 7
    if any(k in n for k in ("center", "origin", "point", "normal", "direction", "lowpoint", "highpoint", "translate", "rotate", "bounds")):
        return [1.0, 2.0, 3.0]
    if "scale" in n and "mode" not in n:
        return [1.0, 1.0, 1.0]
    if any(k in n for k in ("type", "mode", "name", "function", "expression", "result", "plane", "filter")):
        return "Sample"
    if any(k in n for k in ("invert", "capping", "normal", "scalar", "edge", "largest", "unwrap", "crinkle",
                            "split", "flip", "consistency", "topology", "average", "std", "visible",
                            "preserve", "vertex", "vary", "compute")):
        return True
    return 2.5


def _navigate(cfg, path):
    cur = cfg
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and p < len(cur):
            cur = cur[p]
        else:
            return None, False
    return cur, True


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(pvs.READERS))
def test_reader_classes_create_and_map_filename(name):
    cls = getattr(pvs, name)
    r = cls(FileName=["/tmp/whatever.vtu"])
    cfg = json.loads(_sess().source_cfg_json(r._node_id))
    key = next(iter(cfg))
    assert key in ("File", "FileWithReader", "PvdSeries")
    if key == "FileWithReader":
        assert cfg[key]["path"] == "/tmp/whatever.vtu"
        assert cfg[key]["reader"] == pvs.READERS[name]
    assert r.FileName == ["/tmp/whatever.vtu"]
    assert r._node_id in pvs._proxies


def test_unknown_reader_fallback_class():
    cls = pvs.SomeMadeUpReader  # module __getattr__
    r = cls(FileName=["/tmp/x.stl"])
    cfg = json.loads(_sess().source_cfg_json(r._node_id))
    assert cfg == {"File": "/tmp/x.stl"}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(pvs.SOURCES))
def test_source_property_matrix(name):
    spec_cls = pvs._CLASSES[name]
    config_key, prop_map = pvs.SOURCES[name]
    s = spec_cls()
    assert _sess().source_cfg_json(s._node_id)
    for prop, entry in prop_map.items():
        value = _sample(prop, entry)
        setattr(s, prop, value)
        if entry is pvs._IGNORE:
            assert s._props[prop] == value
            continue
        cfg = json.loads(_sess().source_cfg_json(s._node_id))
        inner = cfg[config_key]
        if isinstance(entry, tuple) and isinstance(entry[0], tuple):
            path, conv = entry
            try:
                predicted = conv(value, {})
            except Exception:
                predicted = None
            actual, found = _navigate(inner, path)
            assert found, f"{name}.{prop}: path {path} not in config {inner}"
            if predicted is not None:
                assert actual == predicted, f"{name}.{prop}: {actual} != {predicted}"
        # config must stay valid & re-appliable
        _sess().set_source_cfg_json(s._node_id, json.dumps(cfg))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

FIELD_INPUT = "wavelet"
GEOM_INPUT = "sphere"


def _make_input(kind):
    if kind == FIELD_INPUT:
        w = pvs.Wavelet()
        w.WholeExtent = [-2, 2, -2, 2, -2, 2]
        return w
    s = pvs.Sphere()
    s.Radius = 1.0
    return s


@pytest.mark.parametrize("name", sorted(pvs.FILTERS))
def test_filter_property_matrix(name):
    config_key, prop_map = pvs.FILTERS[name]
    inp = _make_input(FIELD_INPUT if config_key in (
        "Contour", "Threshold", "IsoVolume", "Calculator", "Gradient", "WarpByScalar", "Elevation",
    ) else GEOM_INPUT)
    f = pvs._CLASSES[name](Input=inp)
    for prop, entry in prop_map.items():
        value = _sample(prop, entry)
        if prop in ("SelectInputScalars",) and config_key in ("Contour", "Threshold", "IsoVolume", "Gradient"):
            value = ["POINTS", "RTData"]
        if prop in ("SelectInputVectors",) and config_key in ("StreamTracer", "WarpByVector", "Glyph"):
            value = ["POINTS", "Velocity"]
        setattr(f, prop, value)
        cfg = json.loads(_sess().filter_cfg_json(f._node_id))
        inner = cfg[config_key]
        if isinstance(entry, tuple) and isinstance(entry[0], tuple):
            path, conv = entry
            try:
                predicted = conv(value, {})
            except Exception:
                predicted = None
            actual, found = _navigate(inner, path)
            assert found, f"{name}.{prop}: path {path} not in {inner}"
            if predicted is not None:
                assert actual == predicted, f"{name}.{prop}: {actual} != {predicted}"
        _sess().set_filter_cfg_json(f._node_id, json.dumps(cfg))


# Filters that must evaluate end-to-end on a real input.
EVALUABLE = [
    ("Contour", FIELD_INPUT, {"SelectInputScalars": ["POINTS", "RTData"], "ContourValue": [100.0]}),
    ("Clip", GEOM_INPUT, {"ClipType": "Plane", "Origin": [0.0, 0.0, 0.0], "Normal": [1.0, 0.0, 0.0]}),
    ("Slice", FIELD_INPUT, {"Origin": [0.0, 0.0, 0.0], "Normal": [0.0, 0.0, 1.0]}),
    ("Threshold", FIELD_INPUT, {"SelectInputScalars": ["POINTS", "RTData"], "ThresholdRange": [50.0, 200.0]}),
    ("Calculator", FIELD_INPUT, {"Function": "RTData/255", "ResultArrayName": "n"}),
    ("Elevation", GEOM_INPUT, {"LowPoint": [-1.0, 0.0, 0.0], "HighPoint": [1.0, 0.0, 0.0]}),
    ("ExtractSurface", FIELD_INPUT, {}),
    ("DecimatePro", GEOM_INPUT, {"TargetReduction": 0.5}),
    ("IsoVolume", FIELD_INPUT, {"SelectInputScalars": ["POINTS", "RTData"], "LowerThreshold": 50.0, "UpperThreshold": 200.0}),
    ("IntegrateVariables", GEOM_INPUT, {}),
    ("WarpByScalar", FIELD_INPUT, {"SelectInputScalars": ["POINTS", "RTData"], "ScaleFactor": 0.01}),
    ("Gradient", FIELD_INPUT, {"SelectInputScalars": ["POINTS", "RTData"]}),
    ("CellDataToPointData", GEOM_INPUT, {}),
    ("PointDataToCellData", GEOM_INPUT, {}),
    ("FeatureEdges", GEOM_INPUT, {"FeatureEdges": True}),
    ("Connectivity", GEOM_INPUT, {"ExtractLargestRegion": True}),
    ("Shrink", GEOM_INPUT, {"ShrinkFactor": 0.5}),
    ("CellCenters", GEOM_INPUT, {}),
    ("Reflect", GEOM_INPUT, {"Plane": "X Min"}),
    ("Normals", GEOM_INPUT, {"FeatureAngle": 30.0}),
    ("Triangulate", GEOM_INPUT, {}),
    ("Clean", GEOM_INPUT, {}),
    ("ExtractEdges", GEOM_INPUT, {}),
    ("ProbeOverLine", FIELD_INPUT, {"Point1": [-4.0, 0.0, 0.0], "Point2": [4.0, 0.0, 0.0], "Resolution": 33}),
    ("Smooth", GEOM_INPUT, {"NumberOfIterations": 5}),
    ("Transform", GEOM_INPUT, {"Translate": [1.0, 0.0, 0.0]}),
    ("ResampleToImage", FIELD_INPUT, {"SamplingDimensions": [9, 9, 9]}),
]


@pytest.mark.parametrize("name,input_kind,props", EVALUABLE)
def test_filter_evaluates(name, input_kind, props):
    inp = _make_input(input_kind)
    inp.UpdatePipeline()
    f = pvs._CLASSES[name](Input=inp)
    for k, v in props.items():
        setattr(f, k, v)
    f.UpdatePipeline()
    data = pvs.Fetch(f)
    assert data.GetNumberOfPoints() + data.GetNumberOfCells() > 0


def test_append_and_resample():
    a = pvs.Sphere()
    b = pvs.Cone()
    app = pvs.AppendGeometry(Input=[a, b])
    app.UpdatePipeline()
    assert pvs.Fetch(app).GetNumberOfPoints() > 0

    w = pvs.Wavelet()
    w.WholeExtent = [-2, 2, -2, 2, -2, 2]
    r = pvs.ResampleWithDataset(Input=w, Source=a)
    r.UpdatePipeline()
    assert pvs.Fetch(r).GetNumberOfPoints() > 0


# ---------------------------------------------------------------------------
# Error parity with ParaView semantics
# ---------------------------------------------------------------------------


def test_missing_input_raises():
    with pytest.raises(TypeError):
        pvs.Contour()


def test_fetch_unknown_array_raises():
    w = pvs.Wavelet()
    with pytest.raises(KeyError):
        pvs.Fetch(w).PointData["nope"]


def test_unknown_module_attr_raises():
    with pytest.raises(AttributeError):
        pvs.NoSuchThing


def test_unknown_property_is_accepted():
    s = pvs.Sphere()
    s.SomeFutureProperty = 42  # ParaView would accept any registered prop; we accept & store
    assert s.SomeFutureProperty == 42


def test_vector_filters_evaluate():
    sess = _sess()
    cfg = json.loads(sess.default_source_cfg_json("VectorField"))
    nid = sess.add_source_cfg_json("vf", json.dumps(cfg))
    vf = pvs._shell_proxy(nid, "vf", "source")
    vf.UpdatePipeline()
    for name, props in [
        ("StreamTracer", {"SelectInputVectors": ["POINTS", "Velocity"], "SeedType": "Line",
                          "Point1": [-1.0, 0.0, 0.0], "Point2": [1.0, 0.0, 0.0], "Resolution": 5}),
        ("WarpByVector", {"SelectInputVectors": ["POINTS", "Velocity"], "ScaleFactor": 0.1}),
        ("Glyph", {"SelectInputVectors": ["POINTS", "Velocity"], "ScaleMode": "Data Scaling Off"}),
    ]:
        f = pvs._CLASSES[name](Input=vf)
        for k, v in props.items():
            setattr(f, k, v)
        f.UpdatePipeline()
        assert pvs.Fetch(f).GetNumberOfPoints() > 0, name
