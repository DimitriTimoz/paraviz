"""Differential case: Elevation filter on a Plane (deterministic scalar)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {"fetches.elev.arrays.POINTS:Elevation.mean": 1e-3,
       "fetches.elev.arrays.POINTS:Elevation.range": 1e-3}


def main(out_path):
    t = T.begin()
    p = Plane()
    p.Origin = [-1.0, -2.0, 0.0]
    p.Point1 = [3.0, -2.0, 0.0]
    p.Point2 = [-1.0, 4.0, 0.0]
    p.XResolution = 8
    p.YResolution = 6
    p.UpdatePipeline()

    e = Elevation(Input=p)
    e.LowPoint = [-1.0, 0.0, 0.0]
    e.HighPoint = [1.0, 0.0, 0.0]
    e.UpdatePipeline()
    T.trace_props(t, "elev", e, ["LowPoint", "HighPoint"])
    T.trace_fetch(t, "elev", e)

    Show(e)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "elev.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
