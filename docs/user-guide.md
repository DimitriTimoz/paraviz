# ParaViz User Guide

ParaViz is a fast, modern scientific visualization studio built in Rust and WebGPU (`wgpu`), inspired by **Kitware ParaView**. Whether you explore 3D meshes in the interactive desktop app or automate batch renders on a headless server, this guide walks through every major feature and workflow.

> **Coming from ParaView?** See [**Differences from Classic ParaView**](differences-from-paraview.md) for a side-by-side comparison of architecture, shortcuts, improvements, and current scope.

---

## 1. Getting Started & Launching ParaViz

### Desktop App
- **Linux**: Download `ParaViz-<tag>-x86_64.AppImage` (or `.deb`), mark executable (`chmod +x ParaViz-*.AppImage`), and launch it—or install via `.deb`.
- **macOS**: Open `ParaViz-<tag>-macos-universal.dmg` and drag `ParaViz.app` to `Applications`.
- **Windows**: Extract `paraviz-<tag>-windows-x86_64.zip` and run `paraviz.exe`.
- **From Source**:
  ```bash
  cargo run --release -p paraviz-app
  # Or open a file / saved .json session directly:
  cargo run --release -p paraviz-app -- path/to/dataset.vtu
  ```

You can also **drag and drop** any supported dataset (`.vtk`, `.vtp`, `.vti`, `.vtu`, `.pvd`, `.stl`, `.obj`, `.ply`, `.csv`, `.nii.gz`, `.nrrd`, `.mha`, `.case`, `.xdmf`, `.foam`, `.raw`) or saved `.json` state directly onto the window.

### Built-In Demo Gallery
If you want to explore right away without external files, open **Help → Demo Gallery** (or **Demos** in the top menu) to launch one of five curated pipelines:
1. **Wavelet Isosurfaces + Slice** (`wavelet`)
2. **Swirling Vortex Streamlines + Glyphs** (`vortex`)
3. **Gyroid Minimal Surface** (`gyroid`)
4. **3D Wavelet Volume Raymarching** (`volume`)
5. **Clipped Superquadric Torus** (`superquadric`)

---

## 2. Studio Interface Tour

### Top Menu Bar
- **File**: Open via **File Browser...**, **Open File by Path...**, **Import Raw 3D Volume (.raw)...**, **Recent Files**, **Save/Load State JSON...**, **Export Active Dataset...** (`.vtp`, `.vtu`, `.vti`, `.vtk`, `.stl`, `.obj`, `.ply`, `.csv`, `.glb`, `.html`), **Save Screenshot PNG...**, and **Export Animation (GIF)...**.
- **Sources**: Generate 14 procedural geometric & volumetric datasets (`Sphere`, `Cube`, `Cone`, `Cylinder`, `Plane`, `Line`, `Arrow`, `Disk`, `Torus`, `Wavelet`, `VectorField`, `Superquadric`, `Gyroid`, `PointCloud`).
- **Filters**: Apply 37+ pipeline filters contextually enabled for the selected node, plus **GPU Compute Contour (WebGPU)...** for `ImageData` volumes and **Append Visible Datasets**.
- **Demos / View / Help**: Switch `Dark Studio` / `Light Studio` themes, snap camera views, open **Keyboard & Mouse Shortcuts (`F1`)**, or **Report an Issue / Bug...**.

### Left Dock: Pipeline Browser & Inspector
- **Pipeline Browser**:
  - Toggle visibility per node with the eye icon, or click **All** / **None** / **Solo** (show only the selected node).
  - Double-click a node name (or click **Rename**) to rename it; click **Dup** to duplicate a node.
  - **Copy / Paste Filters** (`Ctrl/Cmd+C` & `Ctrl/Cmd+V`, **Copy** / **Paste** buttons, or right-click menu): copy a single filter or an entire filter chain (**Copy Filter Chain**) and paste it onto any other dataset or branch.
- **Properties Tab**:
  - **Auto-Apply** is enabled by default so slider/dropdown edits immediately re-evaluate the pipeline. Uncheck **Auto-Apply** for multi-million-cell datasets to batch multiple edits and click **Apply** (or **Discard**) when ready.
- **Display Tab**:
  - **Representation**: `Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`.
  - **Color By**: `Solid Color` (with RGBA picker), any `Point` array, or any `Cell` array. Multi-component vectors offer `Magnitude` or individual `X`/`Y`/`Z`/component selectors.
  - **Colormap & Range**: 11 presets (`Cool to Warm`, `Fast`, `Viridis`, `Plasma`, `Inferno`, `Magma`, `Turbo`, `Rainbow`, `Grayscale`, `Black-Body Radiation`, `Cividis`), **Invert**, **Log Scale**, **Auto**, **Fit Data**, **±Sym** (symmetric around `0`), and manual **Min**/**Max**.
  - **Material & Geometry Style**: `Opacity`, `Ambient`, `Specular`, `Shininess`, `Two-Sided Lighting`, `Point Size`, `Line Width`, `Edge Width`, and `Edge Color`.
- **Info Tab**:
  - Shows dataset type, point/cell counts, in-memory footprint, spatial bounds (`X`/`Y`/`Z` + diagonal), surface area/topology counts (`PolyData`), and per-array type/components/range with on-demand **Compute Mean / StdDev**.
- **Scene Tab**:
  - **Viewport Background** skybox presets (`ViewportBgPreset`) and custom Top/Bottom RGB pickers.
  - **Camera** FOV / **Orthographic** toggle, numeric position/target editors, **Reset Camera (`R`)**, **Focus Selected (`F`)**, and **Bookmark View** buttons.
  - **Render Quality & Overlays**: `4x MSAA Anti-Aliasing` (`M`), `Eye-Dome Lighting (EDL)` (`E`) + `EDL Strength`, `3D Axes Gizmo` (`G`), `Scene Bounding Box` (`B`), `3D Cube Axes Grid`, `Transparent Background (Export)`, `3D Filter Widgets` (`W`), and `Scalar Color Bar`.

### Bottom Drawer
- **Spreadsheet**: Inspect `Points`, `Cells`, or `Field` rows in pages, filter columns by name, click any row to **highlight the element in 3D**, and click **Export CSV**.
- **Chart / Histogram**: Displays **XY Plot (ArcLength)** automatically for `ProbeOverLine` or streamline outputs (with `Markers`, `Log Y`, series toggles, and CSV export), or an interactive **Scalar Histogram** for any array/component.
- **Rust Trace**: Live generated `paraviz` Rust code reproducing your current pipeline, display styles, camera, and settings—click **Copy to Clipboard** to turn an interactive session into a reproducible script.

---

## 3. Common User Workflows

### Workflow A: Isosurfaces, Slices & Crisp Mesh Edges
1. Open a volumetric grid (or add **Sources → Wavelet**).
2. Choose **Filters → Contour** (CPU parallel Marching Cubes with vertex welding) or **Filters → GPU Compute Contour (WebGPU)...** (evaluates directly on the GPU for `ImageData`).
3. To inspect an interior cross-section, select the source again and add **Filters → Slice**. Use the snap buttons (`X`, `Y`, `Z` / center) in **Properties** or toggle **Crinkle** to show whole uncut cells intersecting the plane.
4. In **Display**, set **Representation** to `SurfaceWithEdges` and adjust `Edge Color` / `Edge Width`.

### Workflow B: 3D Volume Rendering & Medical Imaging (NIfTI / MetaImage / NRRD / Raw)
1. Load a `.nii` / `.nii.gz`, `.mha` / `.mhd`, `.nrrd`, or `.raw` volume (**File → Import Raw 3D Volume...** lets you specify X/Y/Z dimensions, scalar type `U8`–`F64`, endianness, spacing, origin, and header skip).
2. In **Display**, change **Representation** to `Volume`.
3. Under **Volume Transfer Function**:
   - Choose an opacity preset (**Linear**, **Soft**, **Iso-Shell**, or **Peak**) or drag points directly on the histogram curve editor (or edit exact `(t, alpha)` values in **Control Points Table**).
   - Lower **Step Size** (e.g., `0.2`–`0.4`) for finer raymarching detail, and enable **Gradient Shading** for 3D surface-like depth cues inside volumetric structures.
   - Enable **3D Volume Clip Plane** and drag `Clip Origin` / `Clip Normal` to interactively slice open the volume in real time on the GPU.

### Workflow C: CFD Flow & Vortex Visualization (OpenFOAM / EnSight / VTU)
1. Open an OpenFOAM case directory (or `controlDict`), EnSight `.case`, or `.vtu` mesh.
2. Add **Filters → Stream Tracer** (works even if your velocity vector field lives in `cell_data` rather than `point_data`). Toggle `W` to preview the 3D `Line` or `Sphere` seed widget in the viewport.
3. Select the streamline output and add **Filters → Tube** for shaded 3D streamtubes (computed with twist-free Bishop frames).
4. Optionally compute derived vortex metrics with **Filters → Gradient** (enables `Divergence`, `Vorticity`, `Q-Criterion`, and `Lambda2`).

### Workflow D: Quantitative Probing, Picking & Distance Measurement
- **3D Pick (`2` or click in Orbit)**: Click anywhere on a mesh to inspect the exact world `(X, Y, Z)`, nearest `Point #`, hit `Cell #`, and scalar value in the viewport HUD. Click **Reveal in Spreadsheet** to jump straight to that row, or **Focus** to center the camera there.
- **Distance Ruler (`3`)**: Click two 3D surface points to measure exact Euclidean distance `Δ` and per-axis deltas `ΔX, ΔY, ΔZ`.
- **Line Profile**: Add **Filters → Probe Over Line**, open the bottom drawer **Chart / Histogram** tab to view the multi-array XY plot along `ArcLength`, and export it to `.csv`.
- **Domain Integrals**: Add **Filters → Integrate Variables** to compute total arc length, surface area, or volume along with weighted field integrals and averages.

### Workflow E: Time Series, GIF Animations & Portable Web / 3D Exports
- **Time Series**: Open a `.pvd` collection—or apply **Filters → Add TimeSteps** to a static dataset—and press **`Space`** to play/pause timesteps.
- **Animated GIF**: Choose **File → Export Animation (GIF)...** to export either a **360° Turntable** or **Time Series** `.gif`, with optional **Transparent background** and **Scalar Color Bar**.
- **Interactive Web Sharing (`.html`)**: Choose **File → Export Active Dataset...** and click **`.html`** to write a single self-contained WebGL2 HTML page (with orbit/pan/zoom/wireframe controls and colormap vertex colors) that anyone can open in a browser without installing ParaViz.
- **Publication PNG**: Choose **File → Save Screenshot PNG...** up to 4K (`3840x2160`) with optional **Transparent background (alpha = 0)** and **Include Scalar Color Bar**.

---

## 4. Troubleshooting & Reporting Issues

- **Linux Wayland / Vulkan**: Make sure `libvulkan1` and your GPU's Vulkan driver (or `mesa-vulkan-drivers` for software `llvmpipe` rendering) are installed.
- **Heavy Datasets**: Uncheck **Auto-Apply** in **Properties**, disable `4x MSAA` during interaction, or simplify meshes with **Filters → Decimate (QEM)** before exporting.
- **Encountered a bug or crash?** Click **Help → Report an Issue / Bug...** (or **Report Issue** directly on the red error bar), or open an issue at **[github.com/DimitriTimoz/paraviz/issues](https://github.com/DimitriTimoz/paraviz/issues/new/choose)**. Including your **Rust Trace** output or saved `.json` state helps reproduce issues immediately.
