#!/usr/bin/env python3
"""Stock ParaView data-fetching script, running unmodified on ParaViz.

Typical ParaView pattern:

    from paraview.simple import *
    from vtk.util.numpy_support import vtk_to_numpy
    import numpy as np

    reader = XMLImageDataReader(FileName=['wavelet.vti'])
    data = Fetch(reader)
    rt = vtk_to_numpy(data.PointData['RTData'])
    print(rt.min(), rt.max(), rt.mean())

Run inside the paraviz venv:
    python ../../examples/paraview_fetch_demo.py
"""

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from paraview.simple import *

# produce a dataset file, then fetch it back through a reader (classic pattern)
w = Wavelet()
SaveData("/tmp/wavelet_fetch.vti", proxy=w)

reader = XMLImageDataReader(FileName=["/tmp/wavelet_fetch.vti"])
reader.UpdatePipeline()

data = Fetch(reader)  # <- paraview.servermanager-style fetch, client-side
print("dataset:", data)
print("points:", data.GetNumberOfPoints(), "bounds:", data.GetBounds())

rt = vtk_to_numpy(data.PointData["RTData"])
print("RTData shape:", rt.shape, "dtype:", rt.dtype)
print("RTData min/max/mean: %.4f %.4f %.4f" % (rt.min(), rt.max(), rt.mean()))

pts = vtk_to_numpy(data.GetPoints().GetData())
print("points shape:", pts.shape, "x range:", pts[:, 0].min(), pts[:, 0].max())

# same through a filter output
c = Contour(Input=reader)
c.SelectInputScalars = ["POINTS", "RTData"]
c.ContourValue = [float(rt.mean())]
c.UpdatePipeline()
cdata = Fetch(c)
iso = vtk_to_numpy(cdata.PointData["RTData"])
print("contour points:", cdata.GetNumberOfPoints(), "iso value:", iso[0])
