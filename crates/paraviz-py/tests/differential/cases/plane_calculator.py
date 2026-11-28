"""Differential case: Plane source + Calculator — fully deterministic."""
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
    calc.Function = "coordsX*2 + coordsY - 0.5*coordsZ"
    calc.ResultArrayName = "f"
    calc.AttributeType = "Point Data"
    calc.UpdatePipeline()

    T.trace_props(t, "plane", p, ["Origin", "Point1", "Point2", "XResolution", "YResolution"])
    T.trace_props(t, "calc", calc, ["Function", "ResultArrayName", "AttributeType"])
    T.trace_fetch(t, "plane", p)
    T.trace_fetch(t, "calc", calc)

    Show(calc, representationType="Surface")
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "plane.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
