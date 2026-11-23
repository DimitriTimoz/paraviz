# Differences from Classic ParaView

ParaViz reimplements the core concepts of **Kitware ParaView** (data model, demand-driven pipeline, scientific filters, file I/O, color mapping, multi-pane desktop studio, and headless CLI) in **Rust + WebGPU (`wgpu`) + `egui`**. Because it is a clean-slate Rust/WebGPU architecture rather than a C++/Qt/OpenGL wrapper around VTK, several behaviors and capabilities differ intentionally—and some specialized ParaView subsystems remain outside the current scope.

---

## 1. Improvements & Technical Differences Over ParaView

| Area | Classic ParaView (VTK + OpenGL2 + Qt) | ParaViz (Rust + WebGPU + `egui`) |
| --- | --- | --- |
| **`SurfaceWithEdges` Rendering** | Typically relies on polygon-offset / two-pass line overlay, which can Z-fight or produce dashed artifacts at grazing angles. | **Single-pass barycentric shader**: computes screen-space `fwidth()` derivatives directly in WGSL for crisp anti-aliased edges with zero Z-fighting. |
| **3D Volume Raymarching** | Hardware 3D texture filtering; can show wood-grain slicing artifacts at coarse steps. | **`R32Float` 3D textures** with manual WGSL trilinear filtering (zero optional GPU features), **Interleaved Gradient Noise (IGN)** ray jittering, gradient Phong shading, and a live GPU **3D Volume Clip Plane**. |
| **Properties Workflow** | Manual "Apply" step required before seeing changes. | **Auto-Apply** enabled by default with explicit **Apply / Discard** available for heavy meshes, backed by a **background worker thread + Cancel** on desktop. |
| **Contour / Marching Cubes** | CPU Marching Cubes (GPU compute contour is non-standard in desktop pipeline). | **Parallel `rayon` Marching Cubes** with canonical edge-key vertex welding **plus** WebGPU Compute **`GpuContourEngine`**. |
| **Streamline Tubes** | Classic `vtkTubeFilter` uses Frenet frames, which can twist/pinch at inflection points. | `TubeFilter` uses **parallel-transport (Bishop) frames** for twist-free smooth tubes. |
| **Stream Tracer Vector Location** | `StreamTracer` / `StreamTracerWithCustomSource` historically had friction when vectors lived only in `CellData` (GitLab `#23130`). | Automatically converts `cell_data` vectors via `CellDataToPointDataFilter` when `point_data` lacks them. |
| **Copy / Paste Filters** | Long-standing feature request (`#21123`) in ParaView. | Built-in **Copy Filter**, **Copy Filter Chain**, and **Paste Filter / Chain** (`Ctrl/Cmd+C` / `Ctrl/Cmd+V`). |
| **Multi-View Layout** | Split render/chart/spreadsheet tabs via complex Qt dock trees. | Integrated **`1x1` / `1x2` / `2x1` / `2x2`** panes switchable between **`3D View`**, **`XY Chart`**, and **`Spreadsheet`**, with per-pane node visibility and **Link Cameras**. |
| **Scripting & Trace** | **Python Trace** (`paraview.simple` / `pvpython`) and XML `.pvsm` states. | Live **Rust Trace** (`to_rust_trace()`), human-readable **JSON state**, and standalone `paraviz-py` PyO3 bindings. |
| **Portable Sharing** | Sharing interactive 3D scenes requires external `vtk.js` / Glance setups. | Built-in **pure-Rust GIF89a** turntables/timeseries, **`.glb`** with baked colors, **transparent PNG/GIF (`alpha=0`)**, and standalone **WebGL2 `.html`** viewers. |
| **Deployment & Memory Safety** | Large C++/Qt/Python distribution. | Single self-contained binary (plus browser WASM target) with `#![forbid(unsafe_code)]` parsers and bounded decompression/allocations. |

---

## 2. UI & Workflow Mapping (ParaView → ParaViz)

| Classic ParaView Concept | ParaViz Equivalent |
| --- | --- |
| `paraview.simple` (`Sphere()`, `Show()`, `Render()`) | `ParaViz` session API (`add_source`, `show`, `render_png`) or `paraviz-py` (`Session`) |
| `pvbatch` | `paraviz` headless CLI (`demo`, `render`, `info`, `convert`) |
| Pipeline Browser | Left Dock **Pipeline Browser** (`All` / `None` / `Solo`, `Copy` / `Paste`, `Dup`, `Rename`) |
| MultiBlock Inspector | **Properties / Display → MultiBlock Patches** (`All`/`None`/`Invert`/`Solo`, search, per-patch visibility & RGBA colors) |
| Properties Panel (`Apply` / `Reset`) | **Properties** tab (**Auto-Apply**, **Apply**, **Discard**) with background async worker + **Cancel** |
| Color Map Editor / Transfer Function | **Display** tab (11 presets, `Invert`, `Log Scale`, `Fit Data`, `±Sym`, interactive histogram opacity curve & clip plane) |
| Render View Settings & Color Legend | **Scene** tab (skybox presets, FOV/Ortho, Bookmarks, MSAA 4x, EDL, Cube Axes Grid, movable/resizable **Scalar Color Bar**, **Time:** badge & **Annotations**) |
| Split View / Link Camera | Top bar **`1x1` / `1x2` / `2x1` / `2x2`** + **`Link Cameras`** + per-pane `3D View` / `XY Chart` / `Spreadsheet` & `Vis` override |
| Tools → Start Trace (`Python`) | Bottom Drawer **Rust Trace** tab (`pv.to_rust_trace()`) |
| File → Save State (`.pvsm`) | File → **Save State JSON...** (`.json`) |

### Viewport Mode Shortcuts
- **`1`** = Orbit mode (Left-drag orbit, Shift+Left / Middle-drag pan, Right-drag / Scroll zoom)
- **`2`** = Pick mode (3D raycast pick of point/cell ID, MultiBlock `Block:`, world position, scalar value)
- **`3`** = Ruler mode (2-point 3D surface distance measurement)
- **`R`** / Double-click = Reset active pane camera; **`F`** = Focus selected node
- **`W`** = Toggle 3D Filter Widgets; **`E`** = EDL; **`M`** = 4x MSAA; **`G`** = Axes Gizmo; **`B`** = Bounding Box; **`T`** = Turntable; **`Tab`** = Toggle Left Dock; **`Space`** = Play/Pause time steps; **`F1`** = Shortcuts

---

## 3. Current Scope & Out-of-Scope Subsystems

ParaViz covers single-workstation interactive exploration, multi-block CFD/FEA/medical inspection, and batch CLI rendering, and does **not** currently target:

- **Distributed MPI / Client-Server (`pvserver`)**: Runs on a single workstation/server (parallelized across CPU cores with `rayon` and GPU via `wgpu`) without MPI domain decomposition or remote `pvserver` streaming.
- **CPU/GPU Path Tracing**: No OSPRay or OptiX ray-traced global illumination; rendering uses rasterization, screen-space EDL, and WebGPU raymarched volumes.
- **Specialized CAD / Niche Legacy Readers**: Silo, Plot3D, and STEP/IGES CAD B-Rep geometry kernels are not currently included (see [File Formats & I/O](file-formats.md) for all 20+ supported readers and writers including VTKHDF, CGNS, Exodus II, XDMF, OpenFOAM, EnSight, NIfTI, NRRD, and MetaImage).
- **In-GUI Python Interpreter / Programmable Filter**: Custom pipeline stages are authored in Rust (`Source`/`Filter`/`MultiInputFilter`) or via standalone `paraviz-py`, while runtime expressions use `CalculatorFilter`.
- **In-Situ Co-Processing**: No Catalyst or Ascent live simulation coupling yet.
