# Differences from Classic ParaView

ParaViz reimplements the core concepts of **Kitware ParaView** (data model, demand-driven pipeline, scientific filters, file I/O, color mapping, and desktop studio) in **Rust + WebGPU (`wgpu`) + `egui`**. Because it is a clean-slate Rust/WebGPU architecture rather than a C++/Qt/OpenGL wrapper around VTK, several behaviors and capabilities differ intentionally—and some advanced ParaView features are outside the current scope.

---

## 1. Improvements & Technical Differences Over ParaView

| Area | Classic ParaView (VTK + OpenGL2 + Qt) | ParaViz (Rust + WebGPU + `egui`) |
| --- | --- | --- |
| **`SurfaceWithEdges` Rendering** | Typically relies on polygon-offset / two-pass line overlay, which can produce Z-fighting or dashed-line artifacts at grazing angles. | **Single-pass barycentric shader**: computes screen-space `fwidth()` derivatives on per-triangle barycentric coordinates directly in the WGSL fragment shader for crisp, anti-aliased edges with zero Z-fighting. |
| **3D Volume Raymarching** | Uses hardware 3D texture filtering (often requiring driver extensions or 8/16-bit quantization) and can show wood-grain slicing artifacts at coarse step sizes. | Uses full **`R32Float` 3D textures** with manual trilinear interpolation in WGSL (requires zero optional GPU features), **Interleaved Gradient Noise (IGN)** ray jittering to eliminate banding, central-difference gradient Phong shading, and a live GPU **3D Volume Clip Plane**. |
| **Properties Workflow** | Classic "forgot to click Apply" friction when adding/tweaking sources or filters. | **Auto-Apply toggle** enabled by default for instant live updates, while keeping explicit **Apply / Discard** mode available for heavy pipelines. |
| **Contour / Marching Cubes** | CPU Marching Cubes can produce duplicated vertices or slower single-threaded passes depending on filter variant; GPU contour is not standard in desktop pipeline. | **Parallel `rayon` Marching Cubes** with spatial edge-key vertex welding (~5–6× less vertex memory and smooth normals) **plus** an optional **WebGPU Compute Marching Cubes** engine (`GpuContourEngine`). |
| **Streamline Tubes** | Classic `vtkTubeFilter` uses Frenet frames, which can twist or pinch around straight inflection points. | `TubeFilter` uses **parallel-transport (Bishop) frames** for twist-free smooth tubes. |
| **Stream Tracer Vector Location** | `StreamTracer` / `StreamTracerWithCustomSource` historically had friction when vectors lived only in `CellData` (GitLab `#23130`). | Automatically detects `cell_data` vector arrays when `point_data` lacks them and converts via `CellDataToPointDataFilter` seamlessly. |
| **Copy / Paste Filters** | Long-standing feature request (`#21123`) in ParaView. | Built-in **Copy Filter**, **Copy Filter Chain**, and **Paste Filter / Chain** via buttons, right-click menu, and `Ctrl/Cmd+C` / `Ctrl/Cmd+V`. |
| **Scripting & Trace** | **Python Trace** (`paraview.simple` / `pvpython`) and XML `.pvsm` state files. | **Rust Trace** (`to_rust_trace()`) emitting compilable Rust API code, human-readable **JSON state** (`save_state_json`), and optional standalone PyO3 Python bindings (`crates/paraviz-py`). |
| **Exports & Sharing** | Exports PNG/AVI/VTK/X3D/glTF; sharing interactive 3D views requires external `vtk.js` / ParaView Glance tooling. | Built-in **pure-Rust animated GIF89a** (turntable & time series), **binary glTF 2.0 (`.glb`)** with baked colormap vertex colors, **transparent background (`alpha = 0`)** PNG/GIF renders, headless **Scalar Color Bar** compositor, and **single-file interactive WebGL2 `.html`** viewer export. |
| **Deployment & Footprint** | Large multi-hundred-MB C++/Qt/Python distribution. | Single self-contained native binary (Linux AppImage / `.deb`, macOS Universal `.dmg`, Windows `.zip`) with memory safety across file parsers and filters. |

---

## 2. UI & Workflow Mapping (ParaView → ParaViz)

| Classic ParaView Concept | ParaViz Equivalent |
| --- | --- |
| `paraview.simple` (`Sphere()`, `Show()`, `Render()`) | `ParaViz` session API (`add_source`, `show`, `render_png`) or `paraviz-py` (`Session`) |
| `pvbatch` | `paraviz` headless CLI (`demo`, `render`, `info`, `convert`) |
| Pipeline Browser | Left Dock **Pipeline Browser** (with `All` / `None` / `Solo`, `Copy` / `Paste`, `Dup`, `Rename`) |
| Properties Panel (`Apply` / `Reset`) | **Properties** tab (with **Auto-Apply** toggle, **Apply**, **Discard**) |
| Color Map Editor / Transfer Function | **Display** tab (11 presets, `Invert`, `Log Scale`, `Fit Data`, `±Sym`, interactive histogram opacity curve) |
| Render View Settings & Camera Controls | **Scene** tab (skybox gradients, FOV/Ortho, Camera Bookmarks, MSAA 4x, EDL, Cube Axes Grid, Transparent Export) |
| SpreadSheet View | Bottom Drawer **Spreadsheet** tab (with 3D click-to-highlight & CSV export) |
| Plot Over Line / Histogram View | Bottom Drawer **Chart / Histogram** tab (`XY Plot (ArcLength)` & `Scalar Histogram`) |
| Tools → Start Trace (`Python`) | Bottom Drawer **Rust Trace** tab (`pv.to_rust_trace()`) |
| File → Save State (`.pvsm`) | File → **Save State JSON...** (`.json`) |

### Viewport Mode Shortcuts
- **`1`** = Orbit mode (Left-drag orbit, Shift+Left / Middle-drag pan, Right-drag / Scroll zoom)
- **`2`** = Pick mode (3D raycast pick of point/cell ID, world position, and scalar value)
- **`3`** = Ruler mode (2-point 3D surface distance measurement)
- **`R`** / Double-click = Reset camera; **`F`** = Focus selected node
- **`W`** = Toggle 3D Filter Widgets; **`E`** = EDL; **`M`** = 4x MSAA; **`G`** = Axes Gizmo; **`B`** = Bounding Box; **`T`** = Turntable; **`Tab`** = Toggle Left Dock; **`Space`** = Play/Pause time steps; **`F1`** = Shortcuts

---

## 3. Current Scope & Features Not Yet in ParaViz

ParaViz covers core single-workstation visualization, filtering, volume rendering, and batch CLI rendering, but does **not** currently aim to replace every specialized subsystem of 25 years of VTK/ParaView:

- **Distributed MPI / Client-Server (`pvserver`)**: ParaViz runs on a single workstation/server (parallelized across all CPU cores with `rayon` and on the GPU via `wgpu`), without MPI domain decomposition or remote `pvserver` streaming.
- **CPU/GPU Path Tracing**: No OSPRay or OptiX photorealistic path tracing; rendering uses rasterization + screen-space EDL + raymarched volumes in WebGPU.
- **Specialized Formats**: Standalone VTKHDF / raw `.h5` (outside XDMF), CGNS, Exodus II, Silo, Plot3D, and STEP/IGES CAD B-Rep readers are not yet implemented (see [File Formats & I/O](file-formats.md) for the 16+ supported formats, including XDMF+HDF5).
- **In-GUI Python Programmable Filter**: The GUI does not embed a Python interpreter; custom scripted pipelines are written in Rust (`Source`/`Filter` traits) or via the `paraviz-py` PyO3 crate, while inline math expressions use `CalculatorFilter`.
- **Viewport Annotations**: Headless exports support a single active `Scalar Color Bar` overlay (`--scalar-bar` / `SceneSettings::show_scalar_bar`), rather than multiple independently draggable text boxes and multi-bar viewport layouts.
- **In-Situ Co-Processing**: No Catalyst / Ascent live simulation coupling yet.
