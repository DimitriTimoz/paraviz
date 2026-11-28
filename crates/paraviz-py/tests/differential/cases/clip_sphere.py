"""Differential case: Sphere -> Clip (plane). Geometry counts are soft."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size", "props.clip.ClipType"]  # PV6 enum getter returns a proxy dump
TOL = {
    "fetches.clip.num_points": 0.5,
    "fetches.clip.num_cells": 0.5,
    "fetches.clip.bounds": 0.05,
    "fetches.clip.arrays.POINTS:Normals.range": 0.2,
    "fetches.clip.arrays.POINTS:Normals.mean": 2.0,
}


def main(out_path):
    t = T.begin()
    s = Sphere()
    s.Center = [0.0, 0.0, 0.0]
    s.Radius = 2.0
    s.ThetaResolution = 16
    s.PhiResolution = 12
    s.UpdatePipeline()
    T.trace_props(t, "sphere", s, ["Center", "Radius", "ThetaResolution", "PhiResolution"])

    clip = Clip(Input=s)
    clip.ClipType = "Plane"
    clip.Invert = False
    clip.UpdatePipeline()
    T.trace_props(t, "clip", clip, ["ClipType", "Invert"])
    T.trace_fetch(t, "sphere", s)
    T.trace_fetch(t, "clip", clip)

    Show(clip, representationType="Surface With Edges")
    Hide(s)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "clip.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
