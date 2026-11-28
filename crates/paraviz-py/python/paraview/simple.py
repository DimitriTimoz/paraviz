"""`paraview.simple` compatibility layer backed by ParaViz (Rust + WebGPU).

Scripts written for ParaView's `paraview.simple` run unmodified:

    from paraview.simple import *
    reader = XMLUnstructuredGridReader(FileName=['mesh.vtu'])
    Show(reader)
    c = Contour(Input=reader)
    c.SelectInputScalars = ['POINTS', 'p']
    c.ContourValue = [0.5]
    Show(c, representation='Surface With Edges')
    Hide(reader)
    ColorBy(c, ('POINTS', 'p'))
    Render()
    SaveScreenshot('out.png')

Only the `paraview.simple` scripting subset is emulated; `paraview.servermanager`
and the raw `vtk` module are out of scope.
"""

import json as _json
import re as _re

import paraviz as _paraviz
from ._dataset import DataObject

# ---------------------------------------------------------------------------
# Session singleton & registries
# ---------------------------------------------------------------------------

_session = None
_proxies = {}  # node_id -> proxy object
_views = []
_default_view = None
_animation_scene = None


def _get_session():
    global _session
    if _session is None:
        _session = _paraviz.Session()
    return _session


# ---------------------------------------------------------------------------
# Value converters (ParaView conventions -> paraviz serde JSON)
# ---------------------------------------------------------------------------

_IGNORE = object()  # sentinel: accept & store the property, don't map it


def _camel_to_snake(name):
    return _re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _f(v, _inner=None):
    return float(v)


def _b(v, _inner=None):
    return bool(v)


def _i(v, _inner=None):
    return int(v)


def _vec(v, n=None):
    v = [float(x) for x in v]
    if n is not None:
        if len(v) < n:
            v = v + [0.0] * (n - len(v))
        v = v[:n]
    return v


def _vec3(v, _inner=None):
    return _vec(v, 3)


def _str(v, _inner=None):
    return str(v)


def _assoc_of(v):
    """('POINTS', 'RTData') -> 'POINTS'; 'RTData' -> 'POINTS'."""
    if isinstance(v, (list, tuple)) and v:
        return str(v[0]).upper()
    return "POINTS"


def _scalar_name(v):
    """('POINTS', 'RTData') -> 'RTData'; ('POINTS','VEL','Magnitude') -> 'VEL'; 'RTData' -> 'RTData'."""
    if isinstance(v, (list, tuple)):
        if len(v) >= 2:
            return str(v[1])
        return str(v[0])
    return str(v)


def _component_of(v):
    """('POINTS', 'VEL', 'Magnitude') -> None; ('POINTS','VEL',2) -> 2."""
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        c = v[2]
        if isinstance(c, str):
            return None if c.lower() in ("magnitude", "") else None
        return int(c)
    return None


def _iso_list(v):
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def _assoc_field(v):
    """('CELLS', 'x') -> 'Cell'; default 'Point'."""
    a = _assoc_of(v)
    return "Cell" if a == "CELLS" else "Point"


def _deep_set(d, path, value):
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1]] = value


# ---------------------------------------------------------------------------
# Proxy base class
# ---------------------------------------------------------------------------


class _Proxy:
    """Base class for all pipeline proxies (readers, sources, filters)."""

    _kind = "source"  # 'source' | 'filter' | 'file'
    _config_key = None  # SourceConfig/FilterConfig variant name, or None for file readers
    _prop_map = {}  # ParaView property -> (path, conv) | conv | _IGNORE | callable(self, value, inner)

    def __init__(self, *args, **kwargs):
        sess = _get_session()
        kwargs = dict(kwargs)
        register_name = kwargs.pop("registerName", None)
        input_kw = kwargs.pop("Input", None)
        source_kw = kwargs.pop("Source", None)

        input_ids = []
        for a in args:
            if isinstance(a, _Proxy):
                input_ids.append(a._node_id)
            elif isinstance(a, (list, tuple)) and all(isinstance(x, _Proxy) for x in a):
                input_ids.extend(x._node_id for x in a)
        if isinstance(input_kw, _Proxy):
            input_ids.insert(0, input_kw._node_id)
        elif isinstance(input_kw, (list, tuple)):
            input_ids = [x._node_id for x in input_kw if isinstance(x, _Proxy)] + input_ids

        label = register_name or self.__class__.__name__

        if self._kind == "file":
            node = sess.add_source_cfg_json(label, _json.dumps({"File": ""}))
        elif self._kind == "resample":
            if not input_ids:
                raise TypeError(f"{self.__class__.__name__}() requires Input=")
            src = source_kw._node_id if isinstance(source_kw, _Proxy) else None
            if src is None:
                raise TypeError(f"{self.__class__.__name__}() requires Source=")
            node = sess.resample_nodes(input_ids[0], src)
        elif self._kind == "append":
            if not input_ids:
                raise TypeError(f"{self.__class__.__name__}() requires Input=[...]")
            node = sess.append_nodes(label, input_ids)
        elif self._kind == "filter":
            if not input_ids:
                raise TypeError(
                    f"{self.__class__.__name__}() missing required argument: 'Input'"
                )
            cfg = _json.loads(sess.default_filter_cfg_json(self._config_key))
            node = sess.add_filter_cfg_json(label, input_ids[0], _json.dumps(cfg))
        else:  # source
            cfg = _json.loads(sess.default_source_cfg_json(self._config_key))
            node = sess.add_source_cfg_json(label, _json.dumps(cfg))

        object.__setattr__(self, "_node_id", node)
        object.__setattr__(self, "_props", {})
        _proxies[node] = self

        for k, v in kwargs.items():
            setattr(self, k, v)

    # -- internal config access -------------------------------------------

    def _get_cfg(self):
        sess = _get_session()
        if self._kind == "filter":
            return _json.loads(sess.filter_cfg_json(self._node_id))
        return _json.loads(sess.source_cfg_json(self._node_id))

    def _set_cfg(self, cfg):
        sess = _get_session()
        if self._kind == "filter":
            sess.set_filter_cfg_json(self._node_id, _json.dumps(cfg))
        else:
            sess.set_source_cfg_json(self._node_id, _json.dumps(cfg))

    def _inner(self):
        cfg = self._get_cfg()
        return cfg[self._config_key] if self._config_key else cfg

    def _set_file(self, value):
        files = [str(f) for f in (value if isinstance(value, (list, tuple)) else [value])]
        files = [f for f in files if f]
        if not files:
            cfg = {"File": ""}
        else:
            first = files[0]
            low = first.lower()
            if len(files) > 1 and (low.endswith(".pvd") or low.endswith(".series")):
                cfg = {"PvdSeries": first}
            elif low.endswith(".pvd") and getattr(self, "_reader_variant", None) == "Series":
                cfg = {"PvdSeries": first}
            elif getattr(self, "_reader_variant", None):
                cfg = {"FileWithReader": {"path": first, "reader": self._reader_variant}}
            else:
                cfg = {"File": first}
        _get_session().set_source_cfg_json(self._node_id, _json.dumps(cfg))
        self._props["FileName"] = value

    # -- property routing ---------------------------------------------------

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        pm = type(self)._prop_map
        if self._kind == "file" and name in ("FileName", "FileNames"):
            self._set_file(value)
            return
        if name in pm:
            self._apply_prop(name, value)
        else:
            # Unknown property: accept and store (permissive), and also try a
            # direct snake_case mapping into the config when it exists.
            object.__setattr__(self, name, value)
            self._props[name] = value
            try:
                inner = self._inner()
                key = _camel_to_snake(name)
                if self._config_key and isinstance(inner, dict) and key in inner:
                    inner[key] = value
                    cfg = self._get_cfg()
                    cfg[self._config_key] = inner
                    self._set_cfg(cfg)
            except Exception:
                pass

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        props = object.__getattribute__(self, "_props")
        if name in props:
            return props[name]
        pm = type(self)._prop_map
        if name in pm:
            entry = pm[name]
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], tuple):
                path = entry[0]
                try:
                    inner = self._inner()
                    for p in path:
                        inner = inner[p]
                    return inner
                except Exception:
                    return None
            return None
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _apply_prop(self, name, value):
        pm = type(self)._prop_map
        entry = pm[name]
        self._props[name] = value
        if entry is _IGNORE:
            return
        cfg = self._get_cfg()
        inner = cfg[self._config_key] if self._config_key else cfg
        if callable(entry) and not isinstance(entry, tuple):
            entry(self, value, inner)
        else:
            path, conv = entry
            v = conv(value, inner) if conv else value
            _deep_set(inner, path, v)
        if self._config_key:
            cfg[self._config_key] = inner
        self._set_cfg(cfg)

    # -- ParaView proxy API -------------------------------------------------

    def UpdatePipeline(self, time=None):
        sess = _get_session()
        if time is not None:
            sess.set_time(float(time))
        sess.update_pipeline_node(self._node_id)

    def UpdatePipelineInformation(self):
        pass

    def GetDataInformation(self):
        return DataInformation(_json.loads(_get_session().data_info_json(self._node_id)))

    @property
    def _name(self):
        for nid, name, kind, parent in _get_session().nodes():
            if nid == self._node_id:
                return name
        return type(self).__name__

    def ListProperties(self):
        return list(type(self)._prop_map.keys())

    def __repr__(self):
        return f"<{type(self).__name__} proxy '{self._name}' (node {self._node_id})>"


# ---------------------------------------------------------------------------
# DataInformation (vtkDataInformation subset)
# ---------------------------------------------------------------------------


class ArrayInformation:
    def __init__(self, d):
        self._d = d

    def GetName(self):
        return self._d["name"]

    def GetNumberOfComponents(self):
        return self._d["components"]

    def GetRange(self, component=0):
        return list(self._d["range"])


class FieldInformation:
    def __init__(self, arrays):
        self._arrays = {a["name"]: a for a in arrays}

    def GetNumberOfArrays(self):
        return len(self._arrays)

    def GetArray(self, name):
        a = self._arrays.get(name)
        return ArrayInformation(a) if a else None

    def GetArrayName(self, i):
        return list(self._arrays.keys())[i]


class DataInformation:
    def __init__(self, info):
        self._info = info
        self.PointData = FieldInformation(info.get("point_arrays", []))
        self.CellData = FieldInformation(info.get("cell_arrays", []))

    def GetBounds(self):
        return list(self._info["bounds"])

    def GetNumberOfPoints(self):
        return self._info["num_points"]

    def GetNumberOfCells(self):
        return self._info["num_cells"]

    def GetTimeSteps(self):
        return list(self._info.get("time_steps", []))

    def GetArray(self, name):  # search both point and cell data
        a = self.PointData.GetArray(name) or self.CellData.GetArray(name)
        return a


# ---------------------------------------------------------------------------
# Reader / source / filter registries
# ---------------------------------------------------------------------------

# ParaView reader class name -> DatasetReader variant (None = auto-detect by extension)
READERS = {
    "XMLUnstructuredGridReader": "VtkXml",
    "XMLPolyDataReader": "VtkXml",
    "XMLImageDataReader": "VtkXml",
    "XMLRectilinearGridReader": "VtkXml",
    "XMLStructuredGridReader": "VtkXml",
    "XMLMultiBlockDataReader": "VtkXml",
    "XDMFReader": "Xdmf",
    "LegacyVTKReader": "VtkLegacy",
    "OpenFOAMReader": "OpenFoam",
    "CGNSReader": "Cgns",
    "EnSightReader": "EnSight",
    "EnSightGoldReader": "EnSight",
    "ExodusIIReader": "Exodus",
    "STLReader": "Stl",
    "OBJReader": "Obj",
    "PLYReader": "Ply",
    "OFFReader": "Off",
    "glTFReader": "Gltf",
    "GLTFReader": "Gltf",
    "CSVReader": "Csv",
    "DICOMImageReader": "Dicom",
    "NIFTIImageReader": "Nifti",
    "NIFTIReader": "Nifti",
    "NrrdReader": "Nrrd",
    "MetaImageReader": "MetaImage",
    "LASReader": "Las",
    "GmshReader": "Gmsh",
    "AbaqusReader": "Abaqus",
    "TecplotReader": "Tecplot",
    "PLOT3DReader": "Plot3d",
    "NetCDFReader": "NetCdf",
    "PVDReader": "Series",
    "VTKHDFReader": "VtkHdf",
    "HDF5Reader": "PatchHdf5",
    "XYZReader": "Xyz",
    "PDBReader": "Pdb",
    "X3DReader": "X3d",
    "ImageReader": "Image2d",
    "PNGReader": "Image2d",
}

# ParaView source class name -> SourceConfig variant
SOURCES = {
    "Sphere": ("Sphere", {
        "Center": (("center",), _vec3), "Radius": (("radius",), _f),
        "ThetaResolution": (("theta_resolution",), _i), "PhiResolution": (("phi_resolution",), _i),
        "EndTheta": _IGNORE, "EndPhi": _IGNORE, "StartTheta": _IGNORE, "StartPhi": _IGNORE,
        "LatLongTessellation": _IGNORE, "GenerateNormals": _IGNORE, "Synchronize": _IGNORE,
    }),
    "Box": ("Cube", {
        "Center": (("center",), _vec3),
        "XLength": (("lengths", 0), _f), "YLength": (("lengths", 1), _f), "ZLength": (("lengths", 2), _f),
    }),
    "Cone": ("Cone", {
        "Center": (("center",), _vec3), "Direction": (("direction",), _vec3),
        "Height": (("height",), _f), "Radius": (("radius",), _f),
        "Resolution": (("resolution",), _i), "Capping": (("capping",), _b),
        "Angle": _IGNORE, "ResolveCoincidentTopology": _IGNORE,
    }),
    "Cylinder": ("Cylinder", {
        "Center": (("center",), _vec3), "Height": (("height",), _f), "Radius": (("radius",), _f),
        "Resolution": (("resolution",), _i), "Capping": (("capping",), _b), "Direction": _IGNORE,
    }),
    "Plane": ("Plane", {
        "Origin": (("origin",), _vec3), "Point1": (("point1",), _vec3), "Point2": (("point2",), _vec3),
        "XResolution": (("x_resolution",), _i), "YResolution": (("y_resolution",), _i),
        "Normal": _IGNORE, "Center": _IGNORE, "XRes": (("x_resolution",), _i), "YRes": (("y_resolution",), _i),
    }),
    "Line": ("Line", {
        "Point1": (("point1",), _vec3), "Point2": (("point2",), _vec3),
        "Resolution": (("resolution",), _i), "Origin": _IGNORE,
    }),
    "Arrow": ("Arrow", {
        "TipLength": (("tip_length",), _f), "TipRadius": (("tip_radius",), _f),
        "TipResolution": (("tip_resolution",), _i), "ShaftRadius": (("shaft_radius",), _f),
        "ShaftResolution": (("shaft_resolution",), _i),
        "Origin": _IGNORE, "Direction": _IGNORE, "Scale": _IGNORE, "TipScale": _IGNORE, "ShaftScale": _IGNORE,
    }),
    "Disk": ("Disk", {
        "InnerRadius": (("inner_radius",), _f), "OuterRadius": (("outer_radius",), _f),
        "RadialResolution": (("radial_resolution",), _i), "CircumferentialResolution": (("circum_resolution",), _i),
        "CircumResolution": (("circum_resolution",), _i),
        "Center": _IGNORE, "Normal": _IGNORE, "CircumferentialLines": _IGNORE,
    }),
    "Torus": ("Torus", {
        "Center": (("center",), _vec3), "MajorRadius": (("major_radius",), _f),
        "MinorRadius": (("minor_radius",), _f),
        "MajorSegments": (("major_resolution",), _i), "MinorSegments": (("minor_resolution",), _i),
    }),
    "Wavelet": ("Wavelet", {
        "WholeExtent": (("whole_extent",), lambda v, _i: [int(x) for x in v]),
        "Center": (("center",), lambda v, _i: [float(x) for x in v]),
        "Maximum": (("maximum",), _f), "StandardDeviation": (("std_dev",), _f),
        "XFrequency": (("x_freq",), _f), "YFrequency": (("y_freq",), _f), "ZFrequency": (("z_freq",), _f),
        "XMagnitude": (("x_mag",), _f), "YMagnitude": (("y_mag",), _f), "ZMagnitude": (("z_mag",), _f),
        # ParaView 6.x short names
        "XFreq": (("x_freq",), _f), "YFreq": (("y_freq",), _f), "ZFreq": (("z_freq",), _f),
        "XMag": (("x_mag",), _f), "YMag": (("y_mag",), _f), "ZMag": (("z_mag",), _f),
        "SampleRate": _IGNORE, "SubsampleRate": _IGNORE,
    }),
    "PointSource": ("PointCloud", {
        "Center": (("center",), _vec3), "Radius": (("radius",), _f),
        "NumberOfPoints": (("num_points",), _i), "Volumetric": (("volumetric",), _b),
    }),
    "Superquadric": ("Superquadric", {
        "Toroidal": (("toroidal",), _b), "Thickness": (("thickness",), _f),
        "PhiRoundness": (("phi_roundness",), _f), "ThetaRoundness": (("theta_roundness",), _f),
        "Size": (("size",), _f), "PhiResolution": (("phi_resolution",), _i),
        "ThetaResolution": (("theta_resolution",), _i), "Center": _IGNORE, "Scale": _IGNORE,
    }),
}


def _clip_type(self, value, inner):
    kind = str(value)
    fn_ = inner.get("function", {})
    plane = fn_.get("Plane", {"origin": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]})
    sphere = fn_.get("Sphere", {"center": [0.0, 0.0, 0.0], "radius": 0.5})
    box = fn_.get("Box", {"min": [-1.0] * 3, "max": [1.0] * 3})
    if kind == "Sphere":
        inner["function"] = {"Sphere": sphere}
    elif kind == "Box":
        inner["function"] = {"Box": box}
    else:
        inner["function"] = {"Plane": plane}


def _clip_plane_field(field):
    def handler(self, value, inner):
        fn_ = inner.get("function", {})
        plane = dict(fn_.get("Plane", {"origin": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}))
        plane[field] = _vec3(value)
        inner["function"] = {"Plane": plane}

    return handler


def _clip_sphere_field(field):
    def handler(self, value, inner):
        fn_ = inner.get("function", {})
        sphere = dict(fn_.get("Sphere", {"center": [0.0, 0.0, 0.0], "radius": 0.5}))
        sphere[field] = _vec3(value) if field == "center" else _f(value)
        inner["function"] = {"Sphere": sphere}

    return handler


def _clip_bounds(self, value, inner):
    v = _vec(value, 6)
    inner["function"] = {"Box": {"min": v[:3], "max": v[3:]}}


def _threshold_scalars(self, value, inner):
    inner["field_name"] = _scalar_name(value)
    inner["association"] = _assoc_field(value)


def _threshold_range(self, value, inner):
    v = [float(x) for x in value]
    if len(v) >= 2:
        inner["min"], inner["max"] = v[0], v[1]


def _seed_update(self, value, inner):
    seed_type = self._props.get("SeedType", "Line")
    center = self._props.get("Center", [0.0, 0.0, 0.0])
    p1 = self._props.get("Point1", [-1.0, 0.0, 0.0])
    p2 = self._props.get("Point2", [1.0, 0.0, 0.0])
    resolution = int(self._props.get("Resolution", 10))
    radius = _f(self._props.get("Radius", 0.5))
    if seed_type == "Sphere":
        inner["seeds"] = {"Sphere": {"center": _vec3(center), "radius": radius, "count": resolution}}
    elif seed_type == "Point":
        inner["seeds"] = {"Points": [_vec3(center)]}
    else:
        inner["seeds"] = {"Line": {"p1": _vec3(p1), "p2": _vec3(p2), "count": resolution}}


_DIRECTION_MAP = {"FORWARD": "Forward", "BACKWARD": "Backward", "BOTH": "Both"}
_INTEGRATOR_MAP = {"RK2": "Rk4", "RK4": "Rk4", "RK45": "Rk45"}
_GLYPH_TYPE_MAP = {"ARROW": "Arrow", "2D ARROW": "Arrow", "CONE": "Cone", "SPHERE": "Sphere", "CUBE": "Cube"}
_SCALE_MODE_MAP = {
    "SCALE BY VECTOR": "VectorMagnitude",
    "SCALE BY VECTOR MAGNITUDE": "VectorMagnitude",
    "SCALE BY SCALAR": "Scalar",
    "DATA SCALING OFF": "Constant",
}
_REFLECT_PLANE_MAP = {
    "X MIN": "XMin", "X MAX": "XMax", "Y MIN": "YMin", "Y MAX": "YMax",
    "Z MIN": "ZMin", "Z MAX": "ZMax", "X CENTER": "CenterX", "Y CENTER": "CenterY", "Z CENTER": "CenterZ",
}
_ATTRIBUTE_TYPE_MAP = {"POINT DATA": "Point", "CELL DATA": "Cell", "POINT": "Point", "CELL": "Cell"}


def _enum_map(m):
    def conv(v, _inner):
        return m.get(str(v).upper(), m.get(str(v), list(m.values())[0]))

    return conv


# ParaView filter class name -> (FilterConfig variant, property map)
FILTERS = {
    "Contour": ("Contour", {
        "SelectInputScalars": (("field_name",), lambda v, _i: _scalar_name(v)),
        "ContourValue": (("iso_values",), lambda v, _i: _iso_list(v)),
        "IsoValue": (("iso_values",), lambda v, _i: _iso_list(v)),
        # ParaView >= 5.10 names
        "ContourBy": (("field_name",), lambda v, _i: _scalar_name(v)),
        "Isosurfaces": (("iso_values",), lambda v, _i: _iso_list(v)),
        "ComputeNormals": (("compute_normals",), _b),
        "ComputeScalars": _IGNORE, "ComputeGradients": _IGNORE, "MergeVertices": (("merge_vertices",), _b),
        "GenerateTriangles": _IGNORE, "InputScalars": (("field_name",), lambda v, _i: _scalar_name(v)),
    }),
    "Clip": ("Clip", {
        "ClipType": _clip_type,
        "Origin": _clip_plane_field("origin"),
        "Normal": _clip_plane_field("normal"),
        "Center": _clip_sphere_field("center"),
        "Radius": _clip_sphere_field("radius"),
        "Bounds": _clip_bounds,
        "InsideOut": (("invert",), _b), "Invert": (("invert",), _b),
        "CrinkleClip": (("crinkle",), _b), "Crinkleclip": (("crinkle",), _b),  # ParaView 6.x spelling
        "ClipFunction": _IGNORE, "Value": _IGNORE, "Scalars": _IGNORE,
        "UseValueAsOffset": _IGNORE, "GenerateClippedOutput": _IGNORE, "Exact": _IGNORE,
    }),
    "Slice": ("Slice", {
        "SliceType": _IGNORE,  # paraviz Slice is plane-based
        "Origin": (("origin",), _vec3), "Normal": (("normal",), _vec3),
        "CrinkleSlice": (("crinkle",), _b), "Crinkleslice": (("crinkle",), _b),  # ParaView 6.x spelling
        "SliceOffsetValues": _IGNORE,
        "HyperTreeGridSlicer": _IGNORE, "GenerateTriangles": _IGNORE,
    }),
    "Threshold": ("Threshold", {
        "SelectInputScalars": _threshold_scalars,
        "ThresholdRange": _threshold_range,
        # ParaView >= 5.10 names
        "Scalars": _threshold_scalars,
        "LowerThreshold": (("min",), _f),
        "UpperThreshold": (("max",), _f),
        "AllScalars": (("all_scalars",), _b), "Invert": (("invert",), _b),
        "Method": _IGNORE, "ThresholdMethod": _IGNORE, "UseAllArrays": _IGNORE, "Component": _IGNORE,
        "ThresholdBetween": _threshold_range, "SelectedComponent": _IGNORE, "ComponentMode": _IGNORE,
    }),
    "StreamTracer": ("StreamTracer", {
        "SelectInputVectors": (("vector_field",), lambda v, _i: _scalar_name(v)),
        "Vectors": (("vector_field",), lambda v, _i: _scalar_name(v)),  # ParaView >= 5.10
        "SeedType": _seed_update, "Center": _seed_update, "Point1": _seed_update,
        "Point2": _seed_update, "Resolution": _seed_update, "Radius": _seed_update,
        "IntegrationDirection": (("direction",), _enum_map(_DIRECTION_MAP)),
        "IntegratorType": (("integrator",), _enum_map(_INTEGRATOR_MAP)),
        "InitialStepLength": (("step_size",), _f), "MinimumStepLength": _IGNORE,
        "MaximumStepLength": _IGNORE, "MaximumSteps": (("max_steps",), _i),
        "MaximumPropagation": (("max_length",), _f),
        "MaximumStreamlineLength": (("max_length",), _f),  # ParaView >= 5.10
        "TerminalSpeed": (("terminal_speed",), _f),
        "MaximumError": _IGNORE, "ComputeVorticity": _IGNORE, "RotationScale": _IGNORE,
        "SurfaceStreamlines": _IGNORE, "InputVectors": (("vector_field",), lambda v, _i: _scalar_name(v)),
    }),
    "Tube": ("Tube", {
        "Radius": (("radius",), _f), "RadiusFactor": (("radius_factor",), _f),
        "NumberofSides": (("sides",), _i), "Capping": (("capping",), _b),
        "VaryRadius": _IGNORE, "VaryRadiusByScalar": (("vary_radius_by_scalar",), lambda v, _i: _scalar_name(v) if v else None),
        "VaryRadiusByVector": _IGNORE, "Sides": (("sides",), _i), "CappingOn": (("capping",), _b),
    }),
    "Glyph": ("Glyph", {
        "GlyphType": (("shape",), _enum_map(_GLYPH_TYPE_MAP)),
        "SelectInputVectors": (("vector_field",), lambda v, _i: _scalar_name(v) if v else None),
        "SelectInputScalars": (("scalar_field",), lambda v, _i: _scalar_name(v) if v else None),
        "ScaleMode": (("scale_mode",), _enum_map(_SCALE_MODE_MAP)),
        "VectorScaleMode": (("scale_mode",), _enum_map(_SCALE_MODE_MAP)),  # ParaView >= 5.10
        "ScaleFactor": (("scale_factor",), _f), "MaximumNumberOfPoints": (("max_points",), _i),
        "MaximumNumberOfSamplePoints": (("max_points",), _i),  # ParaView >= 5.10
        "Orient": _IGNORE, "OrientationMode": _IGNORE, "ScaleMode2": _IGNORE, "VectorMode": _IGNORE,
        "ClampScaling": _IGNORE, "Range": _IGNORE, "GlyphMode": _IGNORE, "SetScaleFactor": _IGNORE,
        "Stride": _IGNORE, "ScaleArray": _IGNORE, "OrientationArray": _IGNORE,
    }),
    "Calculator": ("Calculator", {
        "Function": (("expression",), _str), "ResultArrayName": (("result_name",), _str),
        "AttributeType": (("association",), _enum_map(_ATTRIBUTE_TYPE_MAP)),
        "Result": (("result_name",), _str), "CoordinateResults": _IGNORE, "ReplaceInvalidResults": _IGNORE,
    }),
    "ExtractSurface": ("ExtractSurface", {
        "ComputeNormals": (("compute_normals",), _b), "PassPointData": _IGNORE, "PassCellData": _IGNORE,
    }),
    "DecimatePro": ("Decimate", {
        "TargetReduction": (("target_reduction",), _f), "PreserveTopology": (("preserve_boundaries",), _b),
        "FeatureAngle": _IGNORE, "Splitting": _IGNORE, "PreSplitMesh": _IGNORE, "BoundaryVertexDeletion": _IGNORE,
    }),    "IsoVolume": ("IsoVolume", {
        "SelectInputScalars": (("field_name",), lambda v, _i: _scalar_name(v)),
        "LowerThreshold": (("min_value",), _f), "UpperThreshold": (("max_value",), _f),
        "Invert": _IGNORE, "Component": _IGNORE,
    }),
    "IntegrateVariables": ("IntegrateVariables", {}),
    "ProbeOverLine": ("ProbeOverLine", {
        "Point1": (("point1",), _vec3), "Point2": (("point2",), _vec3), "Resolution": (("resolution",), _i),
        "ProbeType": _IGNORE, "PassCellArrays": _IGNORE, "PassPointArrays": _IGNORE, "PassFieldArrays": _IGNORE,
    }),
    "WarpByVector": ("WarpByVector", {
        "SelectInputVectors": (("vector_field",), lambda v, _i: _scalar_name(v)),
        "ScaleFactor": (("scale_factor",), _f), "NormalizeResult": _IGNORE, "UseNormal": _IGNORE,
    }),
    "WarpByScalar": ("WarpByScalar", {
        "SelectInputScalars": (("scalar_field",), lambda v, _i: _scalar_name(v)),
        "UseNormal": (("use_normal",), _b), "Normal": (("normal",), _vec3), "ScaleFactor": (("scale_factor",), _f),
    }),
    "Gradient": ("Gradient", {
        "SelectInputScalars": (("field_name",), lambda v, _i: _scalar_name(v)),
        "ComputeGradient": (("compute_gradient",), _b),
        "ComputeDivergence": (("compute_divergence",), _b),
        "ComputeVorticity": (("compute_vorticity",), _b),
        "ComputeQCriterion": (("compute_q_criterion",), _b),
        "ComputeLambda2": (("compute_lambda2",), _b),
        "ResultArrayName": (("gradient_name",), _str),
        "FasterApproximation": _IGNORE, "ContributingCellOption": _IGNORE, "DivergenceArrayName": _IGNORE,
        "VorticityArrayName": _IGNORE, "QCriterionArrayName": _IGNORE, "Lambda2ArrayName": _IGNORE,
    }),
    "Smooth": ("Smooth", {
        "NumberOfIterations": (("iterations",), _i), "RelaxationFactor": (("relaxation_factor",), _f),
        "FeatureAngle": (("pass_band",), _f), "PreserveBoundary": (("preserve_boundary",), _b),
        "BoundarySmoothing": _IGNORE, "Convergence": _IGNORE, "Method": _IGNORE, "EdgeAngle": _IGNORE,
        "FeatureEdgeSmoothing": _IGNORE, "GenerateErrorScalars": _IGNORE,
    }),
    "Transform": ("Transform", {
        "Translate": (("translation",), _vec3), "Rotate": (("rotation_deg",), _vec3), "Scale": (("scale",), _vec3),
    }),
    "CellDataToPointData": ("CellDataToPointData", {"PassCellData": (("pass_cell_data",), _b)}),
    "PointDataToCellData": ("PointDataToCellData", {"PassPointData": (("pass_point_data",), _b)}),
    "FeatureEdges": ("FeatureEdges", {
        "BoundaryEdges": (("boundary_edges",), _b), "FeatureEdges": (("feature_edges",), _b),
        "NonManifoldEdges": (("non_manifold_edges",), _b), "FeatureAngle": (("feature_angle_deg",), _f),
        "ManifoldEdges": _IGNORE, "Coloring": _IGNORE, "Locator": _IGNORE,
    }),
    "Connectivity": ("Connectivity", {
        "ExtractLargestRegion": (("extract_largest",), _b), "ExtractionMode": _IGNORE,
        "ColorRegions": _IGNORE, "RegionIdsArrayName": _IGNORE, "ClosestPoint": _IGNORE,
        "ScalarConnectivity": _IGNORE, "ScalarRange": _IGNORE,
    }),
    "Shrink": ("Shrink", {"ShrinkFactor": (("shrink_factor",), _f)}),
    "CellCenters": ("CellCenters", {"VertexCells": (("vertex_cells",), _b)}),
    "Reflect": ("Reflect", {
        "Plane": (("plane",), _enum_map(_REFLECT_PLANE_MAP)), "CopyInput": (("copy_input",), _b),
        "Center": _IGNORE, "Normal": _IGNORE, "Transform": _IGNORE,
    }),
    "Normals": ("GenerateNormals", {
        "FeatureAngle": (("feature_angle_deg",), _f), "Splitting": (("splitting",), _b),
        "FlipNormals": (("flip_normals",), _b), "Consistency": (("consistency",), _b),
        "ComputePointNormals": _IGNORE, "ComputeCellNormals": _IGNORE, "AutoOrientNormals": _IGNORE,
        "NonManifoldTraversal": _IGNORE, "SplittingOff": _IGNORE,
    }),
    "Elevation": ("Elevation", {
        "LowPoint": (("low_point",), _vec3), "HighPoint": (("high_point",), _vec3),
        "ScalarRange": _IGNORE, "SetScaleFactor": _IGNORE,
    }),
    "Triangulate": ("Triangulate", {"PassLines": (("pass_verts_and_lines",), _b), "PassVerts": (("pass_verts_and_lines",), _b)}),
    "Clean": ("CleanPolyData", {"Tolerance": (("tolerance",), _f), "ConvertLinesToPoints": _IGNORE, "ConvertPolysToLines": _IGNORE, "ConvertStripsToPolys": _IGNORE, "PointMerging": _IGNORE, "AbsoluteTolerance": _IGNORE, "PieceInvariant": _IGNORE}),
    "ExtractEdges": ("ExtractEdges", {}),
    "Delaunay2D": ("Delaunay2D", {"Alpha": (("alpha",), _f), "Tolerance": _IGNORE, "ProjectionPlaneMode": _IGNORE, "Offset": _IGNORE, "BoundingTriangulation": _IGNORE}),
    "Delaunay3D": ("Delaunay3D", {"Alpha": (("alpha",), _f), "Tolerance": _IGNORE, "Offset": _IGNORE, "BoundingTriangulation": _IGNORE}),
    "ExtractBlock": ("ExtractBlock", {
        "BlockIndices": (("block_indices",), lambda v, _i: [int(x) for x in v]),
        "SelectBlockIndices": (("block_indices",), lambda v, _i: [int(x) for x in v]),
        "NameFilter": (("name_filter",), _str), "UnwrapSingleBlock": (("unwrap_single",), _b),
        "SelectBlocks": _IGNORE, "PruneOutput": _IGNORE,
    }),
    "ResampleToImage": ("ResampleToImage", {
        "SamplingDimensions": (("dimensions",), lambda v, _i: [int(x) for x in v]),
        "Dimensions": (("dimensions",), lambda v, _i: [int(x) for x in v]),
        "UseInputBounds": (("custom_bounds",), lambda v, _i: None if v else _vec(v, 6)),
        "CustomBounds": (("custom_bounds",), lambda v, _i: _vec(v, 6)),
        "ComputeGhostCells": _IGNORE,
    }),
    "TemporalStatistics": ("TemporalStatistics", {
        "ComputeAverage": (("compute_average",), _b), "ComputeMinimum": (("compute_minimum",), _b),
        "ComputeMaximum": (("compute_maximum",), _b), "ComputeStandardDeviation": (("compute_stddev",), _b),
    }),
}

# Special multi-input / two-input filters
APPEND_FILTERS = {"AppendGeometry": "AppendDatasets", "AppendDatasets": "AppendDatasets"}
RESAMPLE_FILTERS = {"ResampleWithDataset": "ResampleWithDataset"}


def _make_reader_class(name, variant):
    return type(
        name,
        (_Proxy,),
        {
            "_kind": "file",
            "_config_key": None,
            "_reader_variant": variant,
            "_prop_map": {
                "FileName": _IGNORE,  # routed in __setattr__
                "FileNames": _IGNORE,
                "FilePrefix": _IGNORE,
                "Path": _IGNORE,
                "TimestepValues": _IGNORE,
                "Refresh": _IGNORE,
            },
            "__doc__": f"ParaView-compatible reader proxy (paraviz). Maps to DatasetReader::{variant or 'Auto'}.",
        },
    )


def _make_source_class(name, spec):
    config_key, prop_map = spec
    return type(
        name,
        (_Proxy,),
        {
            "_kind": "source",
            "_config_key": config_key,
            "_prop_map": dict(prop_map),
            "__doc__": f"ParaView-compatible source proxy (paraviz SourceConfig::{config_key}).",
        },
    )


def _make_filter_class(name, spec):
    config_key, prop_map = spec
    return type(
        name,
        (_Proxy,),
        {
            "_kind": "filter",
            "_config_key": config_key,
            "_prop_map": dict(prop_map),
            "__doc__": f"ParaView-compatible filter proxy (paraviz FilterConfig::{config_key}).",
        },
    )


def _make_append_class(name):
    return type(name, (_Proxy,), {"_kind": "append", "_config_key": None, "_prop_map": {}})


def _make_resample_class(name):
    return type(name, (_Proxy,), {"_kind": "resample", "_config_key": None, "_prop_map": {}})


_CLASSES = {}
for _name, _variant in READERS.items():
    _CLASSES[_name] = _make_reader_class(_name, _variant)
for _name, _spec in SOURCES.items():
    _CLASSES[_name] = _make_source_class(_name, _spec)
for _name, _spec in FILTERS.items():
    _CLASSES[_name] = _make_filter_class(_name, _spec)
for _name in APPEND_FILTERS:
    _CLASSES[_name] = _make_append_class(_name)
_CLASSES["Decimate"] = _CLASSES["DecimatePro"]  # ParaView 6.x name
for _name in RESAMPLE_FILTERS:
    _CLASSES[_name] = _make_resample_class(_name)

globals().update(_CLASSES)


def __getattr__(name):
    # Any unknown XxxReader class falls back to an auto-detecting file reader.
    if name.endswith("Reader"):
        cls = _make_reader_class(name, None)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'paraview.simple' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

_REPRESENTATIONS = {
    "surface": "Surface",
    "surface with edges": "Surface With Edges",
    "wireframe": "Wireframe",
    "points": "Points",
    "volume": "Volume",
    "outline": "Outline",
    "point gaussian": "Point Gaussian",
}

_RUST_REPR_NAMES = {
    "Surface": "Surface",
    "SurfaceWithEdges": "Surface With Edges",
    "Wireframe": "Wireframe",
    "Points": "Points",
    "PointGaussian": "Point Gaussian",
    "Volume": "Volume",
    "Outline": "Outline",
}


class RenderView:
    """Minimal `RenderView` proxy. `ViewSize` drives Render/SaveScreenshot."""

    def __init__(self, view_type="RenderView"):
        self.ViewType = view_type
        self.ViewSize = [1280, 720]
        self.Background = [0.32, 0.34, 0.43]
        self.CameraPosition = None
        self.CameraFocalPoint = None
        self.CameraViewUp = None
        self.CameraParallelScale = None
        self.UseLightKit = None
        self.LightSwitch = None
        self.ViewTime = 0.0
        self.OrientationAxesVisibility = None
        self.StereoRender = None

    def __repr__(self):
        return f"<RenderView {self.ViewSize[0]}x{self.ViewSize[1]}>"


def GetRenderView():
    global _default_view
    if _default_view is None:
        _default_view = RenderView()
        _views.append(_default_view)
    return _default_view


def GetActiveView():
    return GetRenderView()


def GetViews():
    if not _views:
        GetRenderView()
    return list(_views)


def CreateView(view_type="RenderView", **_kw):
    v = RenderView(view_type)
    _views.append(v)
    return v


def _view_size(view, image_resolution):
    if image_resolution:
        return [int(image_resolution[0]), int(image_resolution[1])]
    if view is not None:
        return [int(view.ViewSize[0]), int(view.ViewSize[1])]
    return [int(GetRenderView().ViewSize[0]), int(GetRenderView().ViewSize[1])]


def Render(view=None):
    sess = _get_session()
    sess.update_visible()
    w, h = _view_size(view, None)
    if sess.gpu_available():
        sess.render_rgba(w, h)  # offscreen draw; result discarded like ParaView's Render()
    return None


def RenderAllViews():
    for v in GetViews():
        Render(v)


def Interact(view=None):
    return None


def ResetCamera(view=None):
    _get_session().reset_camera()


def SaveScreenshot(filename, view=None, ImageResolution=None, **_kw):
    sess = _get_session()
    sess.update_visible()
    w, h = _view_size(view, ImageResolution)
    sess.render_png(str(filename), w, h)


def SaveAnimation(filename, view=None, ImageResolution=None, **kw):
    # Animated export: reuse the turntable GIF pipeline when filename ends with .gif
    sess = _get_session()
    if str(filename).lower().endswith(".gif"):
        w, h = _view_size(view, ImageResolution)
        sess.render_turntable_gif(str(filename), w, h, int(kw.get("frames", 36)), int(kw.get("delay_ms", 100)))
    else:
        SaveScreenshot(filename, view=view, ImageResolution=ImageResolution)


# ---------------------------------------------------------------------------
# Representation / display properties
# ---------------------------------------------------------------------------


def _proxy_id(obj):
    if isinstance(obj, _Proxy):
        return obj._node_id
    if isinstance(obj, DisplayProperties):
        return obj._node_id
    raise TypeError(f"expected a pipeline proxy, got {type(obj).__name__}")


def Show(proxy=None, view=None, **kwargs):
    if proxy is None:
        raise TypeError("Show() missing required argument: proxy")
    sess = _get_session()
    nid = _proxy_id(proxy)
    sess.set_visible(nid, True)
    # ParaView <= 5.x kwarg `representation`, ParaView 6.x kwarg `representationType`
    repr_ = kwargs.pop("representationType", kwargs.get("representation", "Surface"))
    if repr_ is not None:
        sess.set_representation(nid, _REPRESENTATIONS.get(str(repr_).lower(), str(repr_)))
    # ParaView colors newly shown pipelines by the active point scalars.
    try:
        style = _json.loads(sess.style_json(nid))
        if isinstance(style.get("color_by"), dict) and "Solid" in style["color_by"]:
            info = _json.loads(sess.data_info_json(nid))
            active = info.get("active_scalars")
            if active:
                sess.set_color_by(nid, "POINTS", active, None)
    except Exception:
        pass
    return GetDisplayProperties(proxy, view)


def Hide(proxy, view=None):
    _get_session().set_visible(_proxy_id(proxy), False)


def HideAll(view=None):
    sess = _get_session()
    for nid, _name, _kind, _parent in sess.nodes():
        sess.set_visible(nid, False)


class DisplayProperties:
    """`GetDisplayProperties(proxy)` — maps to the node's ActorStyle."""

    def __init__(self, proxy, view=None):
        self._node_id = _proxy_id(proxy)
        self._proxy = proxy if isinstance(proxy, _Proxy) else None

    def _style(self):
        return _json.loads(_get_session().style_json(self._node_id))

    # -- setters/getters ----------------------------------------------------
    @property
    def Visibility(self):
        return self._style().get("visible", False)

    @Visibility.setter
    def Visibility(self, v):
        _get_session().set_visible(self._node_id, bool(v))

    @property
    def Representation(self):
        r = self._style().get("representation", "Surface")
        return _RUST_REPR_NAMES.get(str(r), _REPRESENTATIONS.get(str(r).lower(), r))

    @Representation.setter
    def Representation(self, v):
        _get_session().set_representation(self._node_id, _REPRESENTATIONS.get(str(v).lower(), str(v)))

    @property
    def ColorArrayName(self):
        cb = self._style().get("color_by", {})
        if "PointScalar" in cb:
            return ["POINTS", cb["PointScalar"]["array"]]
        if "CellScalar" in cb:
            return ["CELLS", cb["CellScalar"]["array"]]
        return None

    @ColorArrayName.setter
    def ColorArrayName(self, v):
        if v is None:
            _get_session().set_solid_color(self._node_id, [1.0, 1.0, 1.0, 1.0])
        else:
            _get_session().set_color_by(self._node_id, _assoc_of(v), _scalar_name(v), _component_of(v))

    @property
    def Opacity(self):
        return self._style().get("opacity", 1.0)

    @Opacity.setter
    def Opacity(self, v):
        _get_session().set_opacity(self._node_id, float(v))

    @property
    def LineWidth(self):
        return self._style().get("line_width", 1.0)

    @LineWidth.setter
    def LineWidth(self, v):
        _get_session().set_line_width(self._node_id, float(v))

    @property
    def PointSize(self):
        return self._style().get("point_size", 1.0)

    @PointSize.setter
    def PointSize(self, v):
        _get_session().set_point_size(self._node_id, float(v))

    @property
    def DiffuseColor(self):
        cb = self._style().get("color_by", {})
        if "Solid" in cb:
            return list(cb["Solid"])
        return [1.0, 1.0, 1.0, 1.0]

    @DiffuseColor.setter
    def DiffuseColor(self, v):
        _get_session().set_solid_color(self._node_id, [float(x) for x in v])

    @property
    def EdgeColor(self):
        return list(self._style().get("edge_color", [0.0, 0.0, 0.0, 1.0]))

    @EdgeColor.setter
    def EdgeColor(self, v):
        _get_session().set_edge_color(self._node_id, [float(x) for x in v])

    @property
    def EdgeVisibility(self):
        r = str(self._style().get("representation", ""))
        return "edges" in r.lower()

    @EdgeVisibility.setter
    def EdgeVisibility(self, v):
        if v:
            _get_session().set_representation(self._node_id, "Surface With Edges")

    def __repr__(self):
        return f"<DisplayProperties node {self._node_id}>"


def GetDisplayProperties(proxy, view=None):
    return DisplayProperties(proxy, view)


def ColorBy(rep, value):
    """ColorBy(proxy, ('POINTS', 'RTData')) or ColorBy(proxy, None) for solid color."""
    sess = _get_session()
    nid = _proxy_id(rep)
    if value is None:
        sess.set_solid_color(nid, [1.0, 1.0, 1.0, 1.0])
    else:
        sess.set_color_by(nid, _assoc_of(value), _scalar_name(value), _component_of(value))


# ---------------------------------------------------------------------------
# Color transfer functions (LUTs)
# ---------------------------------------------------------------------------


class ColorTransferFunction:
    """`GetColorTransferFunction('RTData')` — preset + RGBPoints subset."""

    def __init__(self, array_name, proxy=None):
        self._array = array_name
        self._proxy = proxy
        self._preset = None
        self._rgb_points = None

    def _targets(self):
        sess = _get_session()
        if self._proxy is not None:
            return [self._proxy._node_id]
        out = []
        for nid, _name, _kind, _parent in sess.nodes():
            try:
                style = _json.loads(sess.style_json(nid))
                cb = style.get("color_by", {})
                arr = None
                if "PointScalar" in cb:
                    arr = cb["PointScalar"]["array"]
                elif "CellScalar" in cb:
                    arr = cb["CellScalar"]["array"]
                if arr == self._array and style.get("visible"):
                    out.append(nid)
            except Exception:
                pass
        return out

    def ApplyPreset(self, preset_name, rescale_transfer_function=False):
        sess = _get_session()
        preset = str(preset_name)
        self._preset = preset
        for nid in self._targets():
            sess.set_colormap(nid, preset)
            if rescale_transfer_function:
                try:
                    info = _json.loads(sess.data_info_json(nid))
                    for arr in info.get("point_arrays", []) + info.get("cell_arrays", []):
                        if arr["name"] == self._array:
                            sess.set_scalar_range(nid, arr["range"][0], arr["range"][1])
                            break
                except Exception:
                    pass

    @property
    def RGBPoints(self):
        if self._rgb_points is not None:
            return list(self._rgb_points)
        for nid in self._targets():
            style = _json.loads(_get_session().style_json(nid))
            stops = style.get("colormap", {}).get("stops", [])
            pts = []
            for s in stops:
                pts.extend([s["t"], s["rgb"][0], s["rgb"][1], s["rgb"][2]])
            return pts
        return []

    @RGBPoints.setter
    def RGBPoints(self, value):
        sess = _get_session()
        self._rgb_points = [float(x) for x in value]
        for nid in self._targets():
            sess.set_custom_colormap(nid, self._rgb_points)

    def RescaleTransferFunction(self, min_val, max_val):
        sess = _get_session()
        for nid in self._targets():
            sess.set_scalar_range(nid, float(min_val), float(max_val))

    def InvertTransferFunction(self):
        pts = self.RGBPoints
        if pts:
            n = len(pts) // 4
            out = []
            for i in range(n):
                t, r, g, b = pts[4 * i: 4 * i + 4]
                out.extend([1.0 - t, r, g, b])
            self.RGBPoints = out

    def __repr__(self):
        return f"<ColorTransferFunction '{self._array}'>"


def GetColorTransferFunction(array_name, proxy=None):
    return ColorTransferFunction(_scalar_name(array_name), proxy)


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------


def _shell_proxy(node_id, name, kind):
    if node_id in _proxies:
        return _proxies[node_id]
    p = _Proxy.__new__(_Proxy)
    object.__setattr__(p, "_node_id", node_id)
    object.__setattr__(p, "_props", {})
    p._kind = kind
    p._config_key = None
    _proxies[node_id] = p
    return p


def GetSources():
    # ParaView 6.x: GetSources() returns every proxy in the session
    # (sources and filters alike); GetFilters() was removed.
    return {
        name: _shell_proxy(nid, name, kind)
        for nid, name, kind, _parent in _get_session().nodes()
    }


def GetFilters():
    return {
        name: _shell_proxy(nid, name, kind)
        for nid, name, kind, _parent in _get_session().nodes()
        if kind == "filter"
    }


def GetRepresentations():
    return dict(_REPRESENTATIONS)


def Delete(proxy=None):
    if proxy is None:
        raise TypeError("Delete() missing required argument: proxy")
    nid = _proxy_id(proxy)
    removed = _get_session().remove_node(nid)
    for r in removed:
        _proxies.pop(r, None)


def RenameSource(new_name, proxy):
    _get_session().rename_node(_proxy_id(proxy), str(new_name))


def SaveData(filename, proxy=None, **kw):
    sess = _get_session()
    if proxy is None:
        sources = [n for n in sess.nodes() if n[2] == "source"]
        if not sources:
            raise RuntimeError("SaveData() with no proxy and no sources in the pipeline")
        nid = sources[-1][0]
    else:
        nid = _proxy_id(proxy)
    sess.export_dataset(nid, str(filename))


def Fetch(proxy, time=None, **kw):
    """`paraview.simple.Fetch` (ParaView <= 5.x) / `FetchData` (6.x): evaluates the
    pipeline and returns a client-side `DataObject` (PointData/CellData arrays,
    points, cells) — the same role as `paraview.servermanager.Fetch` in ParaView 6.

        data = Fetch(reader)
        from vtk.util.numpy_support import vtk_to_numpy
        a = vtk_to_numpy(data.PointData['RTData'])
    """
    sess = _get_session()
    if time is not None:
        sess.set_time(float(time))
    return DataObject(sess, _proxy_id(proxy))


# ParaView 6.x name for the same operation.
FetchData = Fetch


# ---------------------------------------------------------------------------
# Animation / time
# ---------------------------------------------------------------------------


class AnimationScene:
    def __init__(self):
        self._animation_time = 0.0
        self.PlayMode = "Snap To TimeSteps"

    @property
    def AnimationTime(self):
        return self._animation_time

    @AnimationTime.setter
    def AnimationTime(self, t):
        self._animation_time = float(t)
        _get_session().set_time(float(t))

    def Play(self):
        pass

    def Pause(self):
        pass

    def GoToFirst(self):
        steps = _get_session().time_steps()
        if steps:
            self.AnimationTime = steps[0]

    def GoToLast(self):
        steps = _get_session().time_steps()
        if steps:
            self.AnimationTime = steps[-1]

    def UpdateAnimationUsingDataTime(self):
        pass


def GetAnimationScene():
    global _animation_scene
    if _animation_scene is None:
        _animation_scene = AnimationScene()
    return _animation_scene


# ---------------------------------------------------------------------------
# Connection / misc (local mode)
# ---------------------------------------------------------------------------


def Connect(host=None, port=None, **_kw):
    """Local mode: no remote server. Returns None like a connected session."""
    _get_session()
    return None


def Disconnect(*_a, **_kw):
    global _session, _proxies, _views, _default_view, _animation_scene
    _session = None
    _proxies = {}
    _views = []
    _default_view = None
    _animation_scene = None
    return None


class _Version:
    Major = 5
    Minor = 13
    Patch = 0
    Suffix = "-paraviz"

    def __str__(self):
        return f"{self.Major}.{self.Minor}.{self.Patch}{self.Suffix}"


def GetParaViewVersion():
    return _Version()


def GetParaViewStateVersion():
    return "5.13"


def LoadPlugin(*_a, **_kw):
    return True


def ImportFile(*_a, **_kw):
    raise NotImplementedError("ImportFile is not supported by the paraviz shim; use a *Reader proxy")


__all__ = (
    sorted(_CLASSES.keys())
    + [
        "RenderView", "AnimationScene", "DisplayProperties", "ColorTransferFunction",
        "DataInformation", "ArrayInformation", "FieldInformation",
        "Connect", "Disconnect", "GetParaViewVersion", "GetParaViewStateVersion",
        "GetRenderView", "GetActiveView", "GetViews", "CreateView",
        "Render", "RenderAllViews", "Interact", "ResetCamera",
        "SaveScreenshot", "SaveAnimation", "SaveData",
        "Show", "Hide", "HideAll", "ColorBy",
        "GetDisplayProperties", "GetColorTransferFunction", "GetRepresentations",
        "GetSources", "GetFilters", "Delete", "RenameSource",
        "GetAnimationScene", "LoadPlugin", "ImportFile", "Fetch", "FetchData",
    ]
)
