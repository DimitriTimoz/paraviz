# ParaView Python API compatibility (`paraview.simple`)

The `paraviz` Python wheel (built from [`crates/paraviz-py`](../crates/paraviz-py)) bundles a
**drop-in `paraview` package**. Once installed, stock ParaView Python scripts run unmodified:

```bash
pip install paraviz          # or: cd crates/paraviz-py && maturin develop
```

```python
from paraview.simple import *   # same import as with real ParaView

reader = XMLUnstructuredGridReader(FileName=["mesh.vtu"])
reader.UpdatePipeline()

c = Contour(Input=reader)
c.SelectInputScalars = ["POINTS", "p"]
c.ContourValue = [0.5]
Show(c, representation="Surface With Edges")
Hide(reader)
ColorBy(c, ("POINTS", "p"))
GetColorTransferFunction("p").ApplyPreset("Viridis (matplotlib)", True)

view = GetRenderView()
view.ViewSize = [1280, 720]
Render()
SaveScreenshot("out.png")
```

See [`examples/paraview_compat_demo.py`](../examples/paraview_compat_demo.py) and the test
suite [`crates/paraviz-py/tests/test_paraview_simple.py`](../crates/paraviz-py/tests/test_paraview_simple.py).

## How it works

`paraview.simple` is a thin pure-Python layer (`crates/paraviz-py/python/paraview/simple.py`)
on top of the native `paraviz._paraviz` extension. Proxies (`Reader`, `Source`, `Filter`
classes) map ParaView property names onto ParaViz's serde-serializable
`SourceConfig` / `FilterConfig`, so `proxy.SomeProperty = value` updates the Rust pipeline
directly, and `UpdatePipeline()` / `Render()` evaluate and render through WebGPU.

## Supported surface

- **Readers**: all `*Reader` classes for formats ParaViz reads (VTK XML/legacy/series/VTKHDF,
  XDMF, OpenFOAM, CGNS, EnSight, Exodus II, Gmsh, PLOT3D, Tecplot, Abaqus, NetCDF, STL, OBJ,
  PLY, OFF, glTF, X3D, LAS, XYZ, PDB, CSV, NIfTI, NRRD, MetaImage, DICOM, 2D images).
  Unknown `*Reader` names fall back to auto-detection by file extension.
  `FileName` accepts a string or a list (`.pvd`/`.series` lists map to the series reader).
- **Sources**: `Sphere`, `Box`, `Cone`, `Cylinder`, `Plane`, `Line`, `Arrow`, `Disk`, `Torus`,
  `Wavelet`, `PointSource`, `Superquadric` with ParaView property names
  (`Center`, `Radius`, `Origin`, `Normal`, `WholeExtent`, ...).
- **Filters**: `Contour`, `Clip`, `Slice`, `Threshold`, `StreamTracer`, `Tube`, `Glyph`,
  `Calculator`, `ExtractSurface`, `DecimatePro`, `IsoVolume`, `IntegrateVariables`,
  `AppendGeometry`, `ResampleWithDataset`, `ResampleToImage`, `ProbeOverLine`, `WarpByVector`,
  `WarpByScalar`, `Gradient`, `Smooth`, `Transform`, `CellDataToPointData`,
  `PointDataToCellData`, `FeatureEdges`, `Connectivity`, `Shrink`, `CellCenters`, `Reflect`,
  `Normals`, `Elevation`, `Triangulate`, `Clean`, `ExtractEdges`, `Delaunay2D`, `Delaunay3D`,
  `ExtractBlock`, `TemporalStatistics`.
- **Representation**: `Show` / `Hide` / `HideAll`, `GetDisplayProperties` (`Representation`,
  `ColorArrayName`, `Opacity`, `LineWidth`, `PointSize`, `DiffuseColor`, `EdgeColor`,
  `Visibility`), `ColorBy` (incl. `('POINTS','VEL','Magnitude')`), `GetColorTransferFunction`
  (`ApplyPreset`, `RGBPoints`, `RescaleTransferFunction`, `InvertTransferFunction`).
- **Views**: `GetRenderView`, `GetActiveView`, `GetViews`, `CreateView`, `Render`,
  `RenderAllViews`, `ResetCamera`, `SaveScreenshot`, `SaveAnimation` (GIF), `ViewSize`.
- **Pipeline management**: `GetSources`, `GetFilters`, `Delete`, `RenameSource`, `SaveData`,
  `proxy.UpdatePipeline()`, `proxy.GetDataInformation()` (point/cell arrays, ranges, bounds,
  counts), state save/restore via the underlying `paraviz.Session`.
- **Data fetch**: `Fetch(proxy, time=None)` returns a client-side `DataObject` — the same role
  as `paraview.servermanager.Fetch` in real ParaView. It exposes `PointData` / `CellData`
  (`data.PointData['RTData']`, `GetArray(...)`, `keys()`), `GetPoints().GetData()`,
  `GetCells()` (offsets/connectivity/types), `GetNumberOfPoints/Cells()`, `GetBounds()`,
  and `GetDimensions()/GetOrigin()/GetSpacing()` for image data, plus `GetBlock(i)` for
  multi-block datasets. A minimal `vtk` / `vtkmodules` top-level shim is bundled so
  `from vtk.util.numpy_support import vtk_to_numpy` works:

  ```python
  from paraview.simple import *
  from vtk.util.numpy_support import vtk_to_numpy
  import numpy as np

  reader = XMLImageDataReader(FileName=["wavelet.vti"])
  data = Fetch(reader)
  rt = vtk_to_numpy(data.PointData["RTData"])   # numpy array, shape (n_points,)
  pts = vtk_to_numpy(data.GetPoints().GetData())  # shape (n_points, 3)
  ```

  `numpy_to_vtk(arr)` is also provided, and array views support `np.asarray(arr)` directly.
  See [`examples/paraview_fetch_demo.py`](../examples/paraview_fetch_demo.py).
- **Time**: `GetAnimationScene()` with `AnimationTime`, `Play`, `Pause`, `GoToFirst`,
  `GoToLast`.
- **Misc**: `Connect`/`Disconnect` (local mode), `GetParaViewVersion`, `LoadPlugin` (no-op).

## Limitations (vs. real ParaView)

- Only the `paraview.simple` scripting subset is emulated (plus `Fetch` and a minimal
  `paraview.servermanager` re-export). Full `paraview.servermanager` proxy introspection
  and the raw `vtk` object model are not: the bundled `vtk`/`vtkmodules` shims only provide
  `util.numpy_support` (`vtk_to_numpy` / `numpy_to_vtk`) over `Fetch()` results.
- Point coordinates of `ImageData` / `RectilinearGrid` are synthesized on `Fetch`
  (like VTK's implicit points); cells of structured/implicit grids are not materialized —
  use `GetDimensions()` / `GetSpacing()`.
- Distributed MPI rendering, OSPRay/OptiX ray tracing, and proprietary formats
  (ANSYS `.cas/.dat`, STEP/IGES) remain out of scope — see
  [Differences from Classic ParaView](differences-from-paraview.md).
- Properties that have no ParaViz equivalent are accepted and stored but ignored
  (marked `_IGNORE` in the property maps), so scripts never crash on them.
- `Render()` requires a WebGPU adapter (native GPU or software Vulkan). Check with
  `paraviz.Session.gpu_available()`.
- The wheel installs top-level `paraview`, `vtk` and `vtkmodules` packages: use a dedicated
  virtualenv if real ParaView/VTK is also installed, or uninstall one of them.

## Differential testing against real ParaView

Equivalence with real ParaView is verified by **differential tests**, not just unit tests:
the same script runs unchanged under real ParaView (`pvpython`) and under this shim, and both
runs dump a JSON *trace* (pipeline proxy classes, property read-backs, and `Fetch()` data
statistics: point/cell counts, bounds, per-array components/range/mean). The traces are then
diffed key-by-key with per-case tolerances.

- Harness: [`crates/paraviz-py/tests/differential/`](../crates/paraviz-py/tests/differential/)
  (`trace.py` shared recorder, `run_case.py` runner, `cases/*.py` case scripts).
- Cases cover: `Plane`+`Calculator`, `Wavelet`+`Contour`, `Threshold`, `Elevation`, `Clip`
  (sphere), `Slice` (wavelet), `.vtm` multiblock, `.pvd` time series (`Fetch` at t=0 and t=1),
  ASCII `STL` reader, and a `sources_signature` case that pins the default properties, array
  names, counts and bounds of every procedural source (`Sphere`, `Cylinder`, `Cone`, `Plane`,
  `Disk`, `Line`, `Arrow`, `PointSource`, `Wavelet`).
- Run: `cd crates/paraviz-py && . .venv/bin/activate && python -m pytest tests/differential -q`
  (requires a real ParaView with `pvpython` on the machine; otherwise the comparison tests
  skip and only the shim-side runs execute). Real ParaView 6.x is detected automatically;
  `paraview.servermanager.Fetch` (PV6) and `paraview.simple.Fetch` (PV5) are both supported.
- Geometry that is genuinely implementation-defined (marching-cubes triangulation counts,
  contour normal estimation, arrow tessellation) uses explicit soft tolerances in each case's
  `TOL` dict; everything else (array names, bounds, counts of exact sources, property
  read-backs) is compared strictly.
