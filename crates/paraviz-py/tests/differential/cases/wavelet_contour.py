"""Differential case: Wavelet (matches vtkWaveletSource at t=0) + Contour.

RTData statistics must match real ParaView closely; contour geometry counts are
implementation-dependent (marching cubes), hence soft.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {
    "fetches.wavelet.arrays.POINTS:RTData.range": 2e-3,
    "fetches.wavelet.arrays.POINTS:RTData.mean": 2e-3,
    "fetches.contour.num_points": 0.5,
    "fetches.contour.num_cells": 0.5,
    "fetches.contour.arrays.POINTS:RTData.range": 0.1,
    "fetches.contour.arrays.POINTS:RTData.mean": 0.1,
    "fetches.contour.arrays.POINTS:Normals.mean": 0.3,
    "fetches.contour.arrays.POINTS:Normals.range": 0.05,
    "fetches.contour.bounds": 0.05,
}


def main(out_path):
    t = T.begin()
    w = Wavelet()
    w.WholeExtent = [-4, 4, -4, 4, -4, 4]
    w.Center = [0.0, 0.0, 0.0]
    w.Maximum = 255.0
    w.UpdatePipeline()
    T.trace_props(t, "wavelet", w, ["WholeExtent", "Center", "Maximum"])
    T.trace_fetch(t, "wavelet", w)

    c = Contour(Input=w)
    c.ContourBy = ["POINTS", "RTData"]
    c.Isosurfaces = [100.0]
    c.ComputeNormals = True
    c.UpdatePipeline()
    T.trace_props(t, "contour", c, ["Isosurfaces", "ComputeNormals"])
    T.trace_fetch(t, "contour", c)

    rep = Show(c, representationType="Surface With Edges")
    Hide(w)
    ColorBy(rep, ("POINTS", "RTData"))
    GetColorTransferFunction("RTData").ApplyPreset("Viridis (matplotlib)", True)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(os.path.dirname(out_path), "wavelet.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
