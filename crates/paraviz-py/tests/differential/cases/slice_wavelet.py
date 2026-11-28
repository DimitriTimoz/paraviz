"""Differential case: Wavelet -> Slice (plane). Counts are soft, props strict."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size", "props.slice.SliceType"]  # PV6 enum getter returns a proxy dump
TOL = {
    "fetches.slice.num_points": 0.5,
    "fetches.slice.num_cells": 0.5,
    "fetches.slice.bounds": 0.05,
    "fetches.slice.arrays.POINTS:RTData.range": 0.1,
    "fetches.slice.arrays.POINTS:RTData.mean": 0.1,
}


def main(out_path):
    t = T.begin()
    w = Wavelet()
    w.WholeExtent = [-4, 4, -4, 4, -4, 4]
    w.UpdatePipeline()

    sl = Slice(Input=w)
    sl.SliceType = "Plane"
    sl.UpdatePipeline()
    T.trace_props(t, "slice", sl, ["SliceType"])
    T.trace_fetch(t, "slice", sl)

    rep = Show(sl)
    ColorBy(rep, ("POINTS", "RTData"))
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "slice.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
