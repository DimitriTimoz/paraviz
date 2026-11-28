"""Differential case: hand-written .pvd time series -> PVDReader, fetch at t=0 and t=1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace as T
from paraview.simple import *

IGNORE = ["render.size"]
TOL = {}

VTU = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0 0 0 1</DataArray></Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData Scalars="p"><DataArray type="Float32" Name="p" NumberOfComponents="1" format="ascii">{vals}</DataArray></PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

PVD = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
    <DataSet timestep="0.0" group="" part="0" file="step0.vtu"/>
    <DataSet timestep="1.0" group="" part="0" file="step1.vtu"/>
  </Collection>
</VTKFile>
"""


def main(out_path):
    d = os.path.dirname(out_path)
    with open(os.path.join(d, "step0.vtu"), "w") as f:
        f.write(VTU.format(vals="1 2 3 4"))
    with open(os.path.join(d, "step1.vtu"), "w") as f:
        f.write(VTU.format(vals="10 20 30 40"))
    pvd = os.path.join(d, "demo.pvd")
    with open(pvd, "w") as f:
        f.write(PVD)

    t = T.begin()
    r = PVDReader(FileName=[pvd])
    r.UpdatePipeline()
    T.trace_props(t, "reader", r, ["FileName"])
    T.trace_fetch(t, "t0", r, time=0.0)
    T.trace_fetch(t, "t1", r, time=1.0)

    Show(r)
    view = GetRenderView()
    view.ViewSize = [320, 240]
    T.trace_render(t, view, os.path.join(d, "pvd.png"))
    T.trace_pipeline(t)
    T.finish(t, out_path)
