#!/usr/bin/env python3
"""A stock ParaView `paraview.simple` script, running unmodified on ParaViz.

Run inside the venv where `paraviz` (paraviz-py) is installed:

    cd crates/paraviz-py && maturin develop && python ../../examples/paraview_compat_demo.py out.png
"""

import sys

from paraview.simple import *

out = sys.argv[1] if len(sys.argv) > 1 else "paraviz_demo.png"

# --- begin stock ParaView script -------------------------------------------
wavelet = Wavelet()
wavelet.WholeExtent = [0, 10, 0, 10, 0, 10]
wavelet.XFrequency = wavelet.YFrequency = wavelet.ZFrequency = 3.0
wavelet.UpdatePipeline()

reader = wavelet  # in ParaView this would be e.g. XMLUnstructuredGridReader(FileName=[...])

contour = Contour(Input=reader)
contour.SelectInputScalars = ["POINTS", "RTData"]
contour.ContourValue = [120.0, 175.0]
contour.UpdatePipeline()

Show(contour, representation="Surface With Edges")
Hide(reader)
ColorBy(contour, ("POINTS", "RTData"))

lut = GetColorTransferFunction("RTData")
lut.ApplyPreset("Viridis (matplotlib)", True)

view = GetRenderView()
view.ViewSize = [1280, 720]
ResetCamera()
Render()
SaveScreenshot(out)
# --- end stock ParaView script ----------------------------------------------

print("wrote", out)
print("sources:", list(GetSources()))
print("filters:", list(GetFilters()))
