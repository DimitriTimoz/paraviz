"""Differential case: ASCII STL -> STLReader (3 points, 1 triangle)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {}

STL = """solid tri
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 2 0 0
  vertex 0 3 0
 endloop
endfacet
endsolid tri
"""


def main(out_path):
    d = os.path.dirname(out_path)
    stl = os.path.join(d, "tri.stl")
    with open(stl, "w") as f:
        f.write(STL)

    t = T.begin()
    r = STLReader()
    T.set_reader_file(r, stl)
    r.UpdatePipeline()
    T.trace_fetch(t, "stl", r)

    Show(r)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(d, "stl.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
