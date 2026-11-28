# ParaViz End-User Guide

ParaViz is a scientific visualization studio and headless renderer built in **Rust**, **WebGPU (`wgpu`)**, and **`egui`**, inspired by **Kitware ParaView**. Whether you explore multi-block CFD meshes interactively, raymarch 3D medical volumes, compare pipelines side-by-side in split views, or batch-render images on a headless server, this guide walks through every control, parameter, and workflow.

> **Parameter Documentation Index**
> - **Sources & Filters (every struct field, type, default & constraint)**: [Sources & Filters Parameter Reference](filters-and-sources.md)
> - **Display, Colormaps, Volumes, Cameras & Annotations**: [Rendering, Appearance & Scene Parameters](rendering-and-scene.md)
> - **GUI Menus, Shortcuts & Headless CLI Flags**: [Studio GUI & Headless CLI Reference](gui-and-cli.md)
> - **File Readers, Writers & Parser Bounds**: [File Formats & I/O Reference](file-formats.md)
> - **Rust & Python Scripting APIs**: [Rust API Guide (& Python Bindings)](rust-api.md)
> - **ParaView Migration & Scope**: [Differences from Classic ParaView](differences-from-paraview.md)

---

## 1. Getting Started & Launching ParaViz

### Desktop Studio Application
- **Linux**: Download `ParaViz-<tag>-x86_64.AppImage` (or `.deb`), mark executable (`chmod +x ParaViz-*.AppImage`), and launch it—or install the `.deb` package.
- **macOS**: Open `ParaViz-<tag>-macos-universal.dmg` and drag `ParaViz.app` to `Applications`.
- **Windows**: Extract `paraviz-<tag>-windows-x86_64.zip` and run `paraviz.exe`.
- **From Source**:
  ```bash
  # Launch empty Studio workspace
  cargo run --release -p paraviz-app

  # Open any supported dataset or saved JSON session state directly
  cargo run --release -p paraviz-app -- path/to/dataset.vtkhdf
  cargo run --release -p paraviz-app -- gui path/to/state.json
  ```

### Drag-and-Drop Opening
Drag and drop any supported file directly onto the ParaViz window:
- **VTK / Meshes**: `.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts`, `.pvd`, `.stl`, `.obj`, `.ply`, `.csv`, `.tsv`
- **HDF5 / CAE / CFD**: `.vtkhdf`, `.cgns`, `.exo`, `.e`, `.ex2`, `.xdmf`, `.xmf`, `.case` (EnSight Gold), `.foam` / `controlDict` (OpenFOAM)
- **Medical & Volumetric**: `.nii`, `.nii.gz`, `.nrrd`, `.nhdr`, `.mhd`, `.mha`, `.raw`, `.bin`, `.vol`
- **Saved Studio State**: `.json`

- **Built-In Auto-Updater** *(desktop non-WASM)*: Checks GitHub Releases asynchronously on startup (and via **Help → Check for Updates...**). When a newer SemVer release is available, click **`Update vX.Y.Z`** in the bottom status bar to download the platform binary (`tar.gz`, `.zip`, or `.AppImage`), verify its **SHA-256 checksum** against `SHA256SUMS.txt`, replace the executable atomically, and click **Restart Now**.

### Built-In Demo Gallery
Open **Demos** in the top menu (or **Help → Demo Gallery**) to generate one of five built-in scientific pipelines (Studio opens directly into **Swirling Vortex Streamlines + Glyphs**):
1. **Wavelet Isosurfaces + Slice** (`wavelet`) — `WaveletSource` outline + two `ContourFilter` isosurfaces + translucent `SliceFilter`
2. **Swirling Vortex Streamlines + Glyphs** (`vortex`) — `VectorFieldSource` + RK45 `StreamTracerFilter` + `TubeFilter` + `GlyphFilter` arrows
3. **Gyroid Minimal Surface** (`gyroid`) — `GyroidSource` contoured at iso-value `0.0`
4. **3D Wavelet Volume Raymarching** (`volume`) — Full `R32Float` GPU raymarching with custom `TransferFunction`
5. **Clipped Superquadric Torus** (`superquadric`) — `SuperquadricSource` clipped by an oblique plane

---

## 2. Studio Workspace Layout

### Top Menu Bar & Quick Toolbar
- **File**: Open via **File Browser...** (includes explicit `DatasetReader` dropdown), **Open File by Path...**, **Import Raw 3D Volume (.raw)...**, **Recent Files**, **Save/Load State JSON...**, **Export Active Dataset...** (`.vtp`, `.vtu`, `.vti`, `.vtk`, `.stl`, `.obj`, `.ply`, `.csv`, `.glb`, interactive `.html`), **Export A/B Compare HTML...** (one self-contained `.html`: two nodes, wipe slider, shared linked camera, common scalar legend), **Save Screenshot PNG...**, and **Export Animation (GIF)...**.
  - **Save State JSON...** keeps external dataset references. Datasets under the project folder use paths relative to the project, so you can move the folder with its data. When opening a project on desktop, select a replacement if a referenced dataset has moved. Save again to keep the new path.
  - **Save Portable Project...** includes the datasets and every advertised time step in one JSON file. Open it with **Load State JSON...** on desktop or in the browser. The file grows with the size of the dataset and the number of steps, so use referenced projects for large runs.
- **Sources**: Create any of the 14 procedural geometric & analytical sources (see [Sources Reference](filters-and-sources.md#1-procedural-sources-paraviz-sources)).
- **Filters**: Searchable **Filters** menu (`Search 39 filters...`) organized into **Common**, categorized submenus (**Extraction & Slicing**, **Flow, Vectors & Warp**, **Mesh & Topology**, **Fields, Probing & Resampling**, **Temporal**), and a scrollable **All Alphabetical (A-Z)** list — including **GPU Compute Contour (WebGPU)...** (`ImageData`) and **Append Visible Datasets**.
- **View**: Toggle **Inspector Dock (`Tab`)**, **Data Drawer**, **Zen Mode (Full Viewport)**, **Viewport Layout** (`1x1 Single`, `1x2 Split Horizontal`, `2x1 Split Vertical`, `2x2 Quad`), **Link Cameras**, **Scalar Color Bar**, **Time Annotation (`Time:`)**, **3D Filter Widgets (`W`)**, **Axes Gizmo (`G`)**, **Bounding Box (`B`)**, **Turntable Spin (`T`)**, **4× MSAA (`M`)**, **Eye-Dome Lighting (`E`)**, camera presets (`Isometric`, `±X`, `±Y`, `±Z`), **Reset Camera (`R`)**, **Focus Selected (`F`)**, and `DarkStudio` / `LightStudio` theme.
- **Toolbar Quick Controls**: Time-step transport (`Play/Pause` via `Space`, previous/next step, active `Time` selector), `1x1` / `1x2` / `2x1` / `2x2` layout buttons, **Link Cameras** toggle, **Scalar Color Bar** & **Time:** badge toggles, and interaction modes (`1` Orbit, `2` Pick, `3` Ruler).

---

## 3. Split Views & Linked Cameras

ParaViz supports up to **4 simultaneous panes** in the central workspace (`ViewportLayout`):
- **`1x1`** (`Single`): 1 full pane (`P1`).
- **`1x2`** (`SplitHorizontal`): 2 side-by-side panes (`P1` left, `P2` right).
- **`2x1`** (`SplitVertical`): 2 stacked panes (`P1` top, `P2` bottom).
- **`2x2`** (`Quad`): 4-quadrant grid (`P1`–`P4`).

### Per-Pane Controls
Each pane displays a compact header bar with:
1. **Pane Badge (`P1`..`P4`)**: Highlighted with the active accent border when selected; clicking anywhere inside a pane makes it active.
2. **Pane Mode Selector (`3D View` | `XY Chart` | `Spreadsheet`)**: Turn any pane into an independent WebGPU 3D viewport, live line/histogram XY Chart, or attribute Spreadsheet without occupying the bottom drawer.
3. **Per-Pane Node Visibility (`Vis: Global` vs Custom `Nodes (N)`)**: Toggle between following the global Pipeline Browser eye icons or selecting a custom subset of pipeline nodes per 3D pane—ideal for viewing raw mesh wireframe in `P1` next to contoured/sliced surfaces in `P2`.
4. **Independent Camera**: Each 3D pane stores its own `Camera` (`position`, `target`, `up`, `fov_y_deg`, `orthographic`, `ortho_scale`).
5. **Link Cameras**: Enable **Link Cameras** in the toolbar or **View** menu so orbiting, panning, zooming, camera presets, resets, or numeric edits in one 3D pane synchronize across all active 3D panes.

---

## 4. Background Asynchronous Loading & Cancellation

On desktop builds (`#[cfg(not(target_arch = "wasm32"))]`), heavy file parsing (`read_dataset_with_reader`, `read_raw_volume_with_type`, time-step inspection) and computationally heavy pipeline filter evaluations run on a **background worker thread**:
- The GUI stays responsive at 60+ FPS for camera navigation, layout changes, and UI interaction while large datasets load or compute.
- When a job is active, a floating progress pill appears displaying a spinner, job description, elapsed time in seconds, and a **Cancel** button.
- Clicking **Cancel** cooperatively aborts acceptance of the in-flight worker job and discards stale results safely if a new file open or parameter change supersedes it.

---

## 5. Left Dock: Pipeline Browser, MultiBlock Inspector & Parameter Tabs

### Pipeline Browser
- **Visibility Eyes**: Click the eye icon on any node to toggle global visibility; use **All**, **None**, or **Solo** for quick focus.
- **Filter Reuse (`Copy` / `Paste`)**: Press `Ctrl/Cmd+C` / `Ctrl/Cmd+V` (or right-click → **Copy Filter** / **Copy Filter Chain** / **Paste Filter / Chain**) to duplicate single filters or full downstream chains onto other branches.
- **Node Actions**: Rename inline, **Dup** (duplicate), **Delete** (`Delete`/`Backspace`).

### Properties Tab & MultiBlock Patches
- **Auto-Apply**: Enabled by default so every parameter tweak re-evaluates immediately. Disable **Auto-Apply** on multi-million-cell meshes to adjust multiple fields before clicking **Apply** (or **Discard**).
- **Filter Guides & Snap Helpers**: Filters such as `SliceFilter`, `ClipFilter`, `ElevationFilter`, and `ProbeOverLineFilter` include one-click plane/axis snap buttons (`X`, `Y`, `Z`, `Center`, `Fit Parent Bounds`).
- **MultiBlock Patches Inspector** *(automatically shown when selected node evaluates to `MultiBlockDataSet`, in both **Properties** and **Display**)*:
  - Expand **MultiBlock Patches (`N`)** to browse hierarchical patches (`group/leaf`).
  - Toggle individual patch visibility checkboxes (`style.hidden_blocks`).
  - Pick custom per-block solid colors (`style.block_colors`) or click **`×`** to clear the override back to global coloring.
  - Use **`All`**, **`None`**, **`Invert`**, and **`Solo`** (isolates the focused or last 3D-picked patch), or filter long lists via the **`Filter blocks...`** search box.
  - Bounds and `Reset Camera` automatically exclude hidden blocks (`ActorStyle::dataset_bounds`).

### Display Tab (Exact Parameter Controls)
For detailed API field behavior see [Rendering & Scene Parameters](rendering-and-scene.md):
- **Representation**: `Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`.
- **Color By & Component**: Choose `Solid Color` (with RGBA picker), any `Point: <name>` or `Cell: <name>` array, and for multi-component vectors select `Magnitude` or `X (0)` / `Y (1)` / `Z (2)` / `Comp c`.
- **Colormap & Range**: Choose from all 11 scientific presets (`ColorMapPreset::ALL`), toggle **Invert**, **Log Scale**, **Auto** range, **Fit Data**, **±Sym** (symmetric around `0`), or set exact numeric **Min** / **Max** values.
- **Material & Geometry Style**:
  - `Opacity`: `0.0..=1.0`
  - `Ambient`: `0.0..=1.0`
  - `Specular`: `0.0..=1.0`
  - `Shininess`: `1.0..=128.0`
  - `Two-Sided Lighting`: boolean
  - `Point Size`: `1.0..=24.0` px
  - `Line Width`: `0.5..=10.0` px
  - `Edge Width`: `0.25..=6.0` px (`SurfaceWithEdges`)
  - `Edge Color`: RGBA color picker
- **Volume Transfer Function** *(shown when `Representation == Volume`)*:
  - `Step Size`: `0.05..=2.0` voxels
  - `Gradient Shading`: central-difference Phong volumetric lighting
  - `3D Volume Clip Plane`: enables live GPU raymarch clipping with `Clip Origin` (`[x, y, z]`) and `Clip Normal` (`[nx, ny, nz]`)
  - **Opacity Presets**: `Linear`, `Soft`, `Iso-Shell`, `Peak`
  - **Histogram Opacity Curve Editor** + numeric **Control Points Table** (`t` in `0..=1`, `a` in `0..=1`, `+ Add Opacity Point`, `×` remove)
- **Scalar Color Bar** *(also accessible in Scene tab)*:
  - `Show Scalar Color Bar`, `Horizontal Orientation`, custom `Title` (`auto (scalar array)`), `Labels` count (`2..=7`), `Font Scale` (`0.6..=2.5`), `Reset Pos`, `Reset Size`.

### Information Tab
- **Last 3D Pick**: Displays picked `Actor`, world `(X, Y, Z)`, **`Block:`** slash-path (when picking into a `MultiBlockDataSet`), `Pt`, `Cell`, `Scalar`, plus **Reveal in Spreadsheet** and **Reveal Cell #** buttons.
- **Dataset Summary & Arrays**: Dataset variant, total points/cells, estimated RAM, spatial bounds (excluding hidden MultiBlock blocks), surface area/topology counts, and per-array type, tuple/component counts, range, and on-demand **Compute Mean / StdDev**.

### Scene Tab
- **Viewport Background**: 5 `ViewportBgPreset` themes + custom `Top` / `Bottom` RGB pickers.
- **Camera & Navigation**: `Orthographic Projection` toggle, `FOV (deg)` (`15..=110`), `Ortho Scale` (`0.01..=1000`), `Turntable` toggle + `Speed` (`0.05..=2.5`), numeric `Position` / `Target`, `Reset Camera (R)`, `Focus Selected (F)`, and **Bookmark View** create/apply/delete.
- **Render Quality & Overlays**: `4x MSAA Anti-Aliasing` (`M`), `Eye-Dome Lighting (EDL)` (`E`) + `EDL Strength` (`0.0..=4.0`), `3D Axes Gizmo` (`G`), `Scene Bounding Box` (`B`), `3D Cube Axes Grid`, `Transparent Background (Export)`, `3D Filter Widgets` (`W`), `Scalar Color Bar`, and **`Time Annotation (Time:)`** + optional custom `Time Label` (`hint: auto (Time: <t>)`).
- **Scalar Color Bar & Annotations**: Configure scalar bar layout and click **`+ Add Annotation`** to create custom 2D `ViewportAnnotation` badges with inline text edit, visibility checkbox, RGBA color picker, font size (`8..=48`), normalized `[X, Y]` (`0.0..=1.0`), or simply **drag any annotation pill or the Scalar Color Bar card/bottom-right corner directly in the 3D viewport**!

---

## 6. Bottom Drawer: Spreadsheet, Charts & Rust Trace

- **Spreadsheet**: Inspect `Points`, `Cells`, or `Field` attributes in pages, filter columns by name, click any row to place the 3D selection highlight marker in the viewport, and export to `.csv`.
- **Chart / Histogram**: Automatically switches to multi-series **XY Plot (`ArcLength`)** for `ProbeOverLineFilter` and `StreamTracerFilter` outputs (with `Markers`, `Log Y`, series checkboxes, and `.csv` export), or displays an interactive binned **Scalar Histogram** for any array/component.
- **Temporal Probe**: Plot a point-data component at a fixed position across the selected node’s available time steps. Select the node, open the drawer’s **Temporal Probe** tab, set the coordinates or use **Use picked position**, then select **Compute**. Hover **Missing** to see unavailable times. Export the full series with **Export CSV…**. Saving the session retains the last probe settings so you can recompute after reopening.
- **Rust Trace**: Live generated `paraviz` Rust API source reproducing your session (`ParaViz::to_rust_trace()`) with **Copy to Clipboard**.

---

## 7. Core Scientific Workflows

### Workflow A: Isosurfaces, Slices & Crisp Edges
1. Open a volumetric grid (e.g. `.vti`, `.vtkhdf`, `.nii.gz`) or add **Sources → Wavelet**.
2. Choose **Filters → Contour** (`ContourFilter`: set `field_name`, `iso_values`, `compute_normals`, `merge_vertices`) or **Filters → GPU Compute Contour (WebGPU)...** (`GpuContourEngine` directly on `ImageData`).
3. Select the parent source again and add **Filters → Slice** (`SliceFilter`: use `X`/`Y`/`Z`/`Center` snap buttons or toggle `crinkle` for uncut cells).
4. In **Display**, set **Representation** to `SurfaceWithEdges` and adjust `Edge Width` / `Edge Color` (zero Z-fighting via single-pass barycentric shader).

### Workflow B: 3D Volume Raymarching & Medical / Raw Imaging
1. Open `.nii` / `.nii.gz`, `.mha` / `.mhd`, `.nrrd`, or **File → Import Raw 3D Volume (.raw)...** (configure `dimensions`, `scalar_type`, `endian`, `origin`, `spacing`, `header_bytes`, `array_name`). For unstructured meshes, run **Filters → Resample To Image** first.
2. Switch **Representation** to `Volume`.
3. Tune **Volume Transfer Function**: pick a preset (`Soft`, `Iso-Shell`, `Peak`), drag curve control points over the scalar histogram, lower **Step Size** toward `0.1`–`0.3` for crisp sampling, toggle **Gradient Shading**, and enable **3D Volume Clip Plane** to inspect internal anatomy.

### Workflow C: Multi-Block CFD & Vortex Identification
1. Open an OpenFOAM case, EnSight `.case`, `.cgns`, `.exo`/`.ex2`, `.vtkhdf`, or `.vtm` assembly.
2. Click any patch in the 3D viewport (`1` Orbit / `2` Pick) to identify `Block: ...`, then use **MultiBlock Patches** (`Solo`, `Invert`, per-block color overrides) to isolate walls, inlets, or rotor blades.
3. Add **Filters → Stream Tracer** (works even on `CellData` vectors; toggle `W` for live 3D `Line`/`Sphere` seed widget preview), then chain **Filters → Tube** (`TubeFilter`: set `radius`, `sides`, `vary_radius_by_scalar`).
4. Add **Filters → Gradient** on `Velocity` to compute `Divergence`, `Vorticity`, `QCriterion`, and `Lambda2`, then contour `QCriterion` or `Lambda2`.

### Workflow D: Split-View Multi-Modal Comparison & Exports
1. Click **`1x2`** or **`2x2`** in the toolbar. Set `P1` to `3D View`, `P2` to `XY Chart` (showing a `ProbeOverLineFilter` profile), or keep two `3D View` panes with **Link Cameras** and per-pane `Nodes (N)` overrides.
2. Drag the **Scalar Color Bar** to the desired corner, resize its bottom-right grip, enable **Time:** if exploring a time series, and add a title via **Scene → Annotations**.
3. Export via **File → Save Screenshot PNG...** (up to 4K, optional transparent background + scalar bar + annotations), **Export Animation (GIF)...** (turntable or time-series GIF89a), **Export Active Dataset → `.glb`**, **`.html`** (zero-dependency interactive WebGL2 viewer), or **Export A/B Compare HTML...** — a single self-contained `.html` with side A (selected node) and side B (picked in the dialog) behind a wipe slider, one shared linked camera, and a common scalar color-bar legend.

---

## 8. Performance Tuning & Troubleshooting

- **Editing multi-million-cell pipelines**: Uncheck **Auto-Apply** in the Properties tab so parameter edits do not re-trigger background evaluations on every keystroke.
- **Mesh simplification**: Run **Filters → Decimate** (`target_reduction`, e.g., `0.75`) before exporting `.glb` or `.html` web viewers.
- **Volume rendering frame rate**: Keep `Step Size` near `0.5`–`1.0` while orbiting, then lower to `0.15`–`0.3` for final inspection or screenshots; `4x MSAA` can be toggled (`M`) independently.
- **Surface smoothing without shrinkage**: Use `SmoothFilter` with `SmoothingMethod::Taubin` rather than `Laplacian`.
- **Logarithmic colormap inactive**: `ColorMap::log_scale` requires strictly positive range & values (`min > 0`, `max > 0`); use **Fit Data** or adjust `Min` above `0`.
