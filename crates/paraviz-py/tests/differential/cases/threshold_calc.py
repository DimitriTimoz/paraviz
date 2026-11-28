"""Differential case: Plane -> Calculator -> Threshold (deterministic point count)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {}


def main(out_path):
    t = T.begin()
    p = Plane()
    p.Origin = [0.0, 0.0, 0.0]
    p.Point1 = [2.0, 0.0, 0.0]
    p.Point2 = [0.0, 3.0, 0.0]
    p.XResolution = 5
    p.YResolution = 7
    p.UpdatePipeline()

    calc = Calculator(Input=p)
    calc.Function = "coordsX"
    calc.ResultArrayName = "x"
    calc.UpdatePipeline()

    th = Threshold(Input=calc)
    th.Scalars = ["POINTS", "x"]
    th.LowerThreshold = 0.5
    th.UpperThreshold = 1.5
    th.UpdatePipeline()
    T.trace_props(t, "threshold", th, ["LowerThreshold", "UpperThreshold"])
    T.trace_fetch(t, "plane", p)
    T.trace_fetch(t, "threshold", th)

    Show(th)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "th.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
