"""Differential case: default signatures of every procedural source.

Captures default property values, point/cell counts, bounds and array
signatures for each source, under real ParaView and the shim alike.
Counts/bounds for arrow topology are soft; array names and defaults are strict.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {
    "fetches.arrow.num_points": 0.5,
    "fetches.arrow.num_cells": 3.0,
    "fetches.arrow.bounds": 0.05,
}

SOURCES = {
    "sphere": ("Sphere", ["Center", "Radius", "ThetaResolution", "PhiResolution"]),
    "cylinder": ("Cylinder", ["Center", "Height", "Radius", "Resolution", "Capping"]),
    "cone": ("Cone", ["Center", "Height", "Radius", "Resolution", "Capping"]),
    "plane": ("Plane", ["Origin", "Point1", "Point2", "XResolution", "YResolution"]),
    "disk": ("Disk", ["InnerRadius", "OuterRadius", "RadialResolution", "CircumferentialResolution"]),
    "line": ("Line", ["Point1", "Point2", "Resolution"]),
    "arrow": ("Arrow", ["TipLength", "TipRadius", "TipResolution", "ShaftRadius", "ShaftResolution"]),
    "pointsource": ("PointSource", ["Center", "Radius", "NumberOfPoints"]),
    "wavelet": ("Wavelet", ["WholeExtent", "Maximum", "StandardDeviation", "XFreq", "YFreq", "ZFreq", "XMag", "YMag", "ZMag"]),
}


def main(out_path):
    import paraview.simple as pvs

    t = T.begin()
    for key, (name, props) in SOURCES.items():
        mk = getattr(pvs, name)
        s = mk()
        s.UpdatePipeline()
        T.trace_props(t, key, s, props)
        T.trace_fetch(t, key, s)
    T.trace_pipeline(t)
    T.finish(t, out_path)
