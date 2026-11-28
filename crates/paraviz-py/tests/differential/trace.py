"""Shared trace helpers for differential tests.

A trace is a JSON document capturing the *semantic* result of a ParaView script:
pipeline structure (proxy class names), property read-backs, and Fetch() data
statistics. The same case script runs unchanged under real ParaView
(`paraview.simple`) and under the paraviz shim — only the shared API subset is
used here.
"""

import ast
import json
import math
import os


def begin():
    return {"pipeline": {}, "props": {}, "fetches": {}, "render": None, "misc": {}}


def _norm(v):
    if isinstance(v, (list, tuple)):
        out = [_norm(x) for x in v]
        # real ParaView stringifies single-valued properties as "['x']" / "[1.0]"
        if len(out) == 1:
            return out[0]
        return out
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return round(v, 5)
    if isinstance(v, (int, bool)) or v is None:
        return v
    if isinstance(v, str):
        s = v.strip()
        # real ParaView stringifies non-JSON property values (VectorProperty,
        # enum proxies) — recover the underlying literal when possible
        if s[:1] in ("[", "(", "'", '"'):
            try:
                return _norm(ast.literal_eval(s))
            except Exception:
                return v
        return v
    if hasattr(v, "__iter__"):
        try:
            return _norm(list(v))
        except Exception:
            return str(v)
    return str(v)


def trace_props(trace, label, proxy, names):
    trace["props"][label] = {n: _norm(getattr(proxy, n)) for n in names}


def trace_pipeline(trace):
    import paraview.simple as pvs

    entry = {}
    # ParaView 6.x: GetSources() returns every proxy in the session and
    # GetFilters() no longer exists. Record sources only, so both sides agree.
    if hasattr(pvs, "GetSources"):
        entry["sources"] = sorted(type(p).__name__ for p in pvs.GetSources().values())
    trace["pipeline"] = entry


def _bounds(d):
    try:
        return list(d.GetBounds())
    except TypeError:  # VTK 9 requires an output argument
        b = [0.0] * 6
        d.GetBounds(b)
        return list(b)


def _fetch_entry(d):
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    out = {
        "num_points": int(d.GetNumberOfPoints()),
        "num_cells": int(d.GetNumberOfCells()),
        "bounds": [round(float(b), 4) for b in _bounds(d)],
        "arrays": {},
    }
    for assoc, fd in (("POINTS", d.GetPointData()), ("CELLS", d.GetCellData())):
        n = fd.GetNumberOfArrays()
        for i in range(n):
            name = fd.GetArrayName(i)
            arr = fd.GetArray(i)
            a = np.asarray(vtk_to_numpy(arr), dtype=np.float64).reshape(-1)
            out["arrays"][f"{assoc}:{name}"] = {
                "components": int(arr.GetNumberOfComponents()),
                "num_tuples": int(arr.GetNumberOfTuples()),
                "range": [round(float(a.min()), 4), round(float(a.max()), 4)],
                "mean": round(float(a.mean()), 4),
            }
    return out


def set_reader_file(reader, path):
    """Sets the file property with whichever name the ParaView version uses."""
    if hasattr(reader, "FileName"):
        reader.FileName = [path]
    else:
        reader.FileNames = [path]


def _fetch_fn():
    try:
        from paraview.simple import Fetch  # ParaView <= 5.x

        return Fetch
    except ImportError:
        from paraview.servermanager import Fetch  # ParaView 6.x

        return Fetch


def trace_fetch(trace, key, proxy, time=None):
    Fetch = _fetch_fn()
    if time is not None:
        proxy.UpdatePipeline(time)
    d = Fetch(proxy)
    entry = {}
    nb = d.GetNumberOfBlocks() if hasattr(d, "GetNumberOfBlocks") else 0
    if nb:
        entry["num_blocks"] = int(nb)
        entry["blocks"] = [_fetch_entry(d.GetBlock(i)) for i in range(nb)]
    else:
        entry.update(_fetch_entry(d))
    trace["fetches"][key] = entry


def trace_render(trace, view, png_path):
    from paraview.simple import Render, SaveScreenshot

    try:
        Render(view)
        SaveScreenshot(png_path, view)
        trace["render"] = {"ok": True, "size": os.path.getsize(png_path)}
    except Exception as e:  # headless GPU may be missing in one env
        trace["render"] = {"ok": False, "error": type(e).__name__}


def finish(trace, out_path):
    trace["misc"]["out_dir"] = None  # volatile
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=1, sort_keys=True, default=str)
