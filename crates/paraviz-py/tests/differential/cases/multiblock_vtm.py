"""Differential case: hand-written .vtm multi-block -> XMLMultiBlockDataReader."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {}

VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>
      </Polys>
      <PointData><DataArray type="Float32" Name="{arr}" NumberOfComponents="1" format="ascii">{vals}</DataArray></PointData>
    </Piece>
  </PolyData>
</VTKFile>
"""

VTM = """<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">
  <vtkMultiBlockDataSet>
    <DataSet index="0" file="blk0.vtp"/>
    <DataSet index="1" file="blk1.vtp"/>
  </vtkMultiBlockDataSet>
</VTKFile>
"""


def main(out_path):
    d = os.path.dirname(out_path)
    with open(os.path.join(d, "blk0.vtp"), "w") as f:
        f.write(VTP.format(pts="0 0 0 1 0 0 0 1 0", arr="s", vals="5 6 7"))
    with open(os.path.join(d, "blk1.vtp"), "w") as f:
        f.write(VTP.format(pts="0 0 1 1 0 1 0 1 1", arr="t", vals="9 8 7"))
    vtm = os.path.join(d, "demo.vtm")
    with open(vtm, "w") as f:
        f.write(VTM)

    t = T.begin()
    r = XMLMultiBlockDataReader(FileName=[vtm])
    r.UpdatePipeline()
    T.trace_props(t, "reader", r, ["FileName"])
    T.trace_fetch(t, "mb", r)

    Show(r)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(d, "mb.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
