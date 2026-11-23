# Desktop Studio GUI & Headless CLI Reference

The `paraviz-app` crate compiles the `paraviz` binary (plus WASM WebGPU target support for the browser Studio), uniting an interactive `eframe`/`egui` + WebGPU desktop workstation with a scriptable headless CLI.

> **Related Documentation**
> - [End-User Manual](user-guide.md) — Walkthroughs, MultiBlock & split-view workflows
> - [Sources & Filters Parameter Reference](filters-and-sources.md) — Every filter & source parameter
> - [Rendering, Style & Scene Parameters](rendering-and-scene.md) — `ActorStyle`, `SceneSettings`, `ScalarBarConfig`, `ViewportAnnotation`

---

## 1. Launching the Studio GUI

```bash
# Empty Studio workspace
cargo run -p paraviz-app

# Open a dataset or saved JSON session directly
cargo run -p paraviz-app -- path/to/mesh.vtu
cargo run -p paraviz-app -- gui path/to/state.json
```

Drag and drop any supported scientific dataset (`.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts`, `.pvd`, `.vtkhdf`, `.cgns`, `.exo`, `.e`, `.ex2`, `.xdmf`, `.xmf`, `.case`, `.foam`, `.nii`, `.nii.gz`, `.nrrd`, `.nhdr`, `.mhd`, `.mha`, `.stl`, `.obj`, `.ply`, `.csv`, `.tsv`, `.raw`) or `.json` state onto the window to open it.

---

## 2. Studio Controls Reference

### Top Menu & Toolbar
- **File**: Open File Browser (with `DatasetReader` dropdown), Open by Path, Import Raw 3D Volume (`RawVolumeDescriptor`), Recent Files, Save/Load State `.json`, Export Dataset (`.vtp`, `.vtu`, `.vti`, `.vtk`, `.stl`, `.obj`, `.ply`, `.csv`, `.glb`, standalone WebGL2 `.html`), Save Screenshot `.png`, Export Animation `.gif` (turntable or time-series).
- **Sources**: All 14 geometric & analytical procedural sources.
- **Filters**: All 37 `FilterConfig` filters (contextually gated by input dataset type) + **GPU Compute Contour (WebGPU)...** + **Append Visible Datasets**.
- **Demos**: Launch any of the 5 built-in `DemoPreset` pipelines (Studio opens with **Swirling Vortex Streamlines + Glyphs**).
- **View & Quick Bar**:
  - Panels: `Inspector Dock (Tab)`, `Data Drawer`, `Zen Mode (Full Viewport)`.
  - Multi-pane layout: **Viewport Layout** (`1x1 Single`, `1x2 Split Horizontal`, `2x1 Split Vertical`, `2x2 Quad`) and **`Link Cameras`**.
  - Overlays & shaders: `Scalar Color Bar`, `Time Annotation (Time:)`, `3D Filter Widgets (W)`, `Axes Gizmo (G)`, `Bounding Box (B)`, `Turntable Spin (T)`, `4x MSAA (M)`, `Eye-Dome Lighting (E)` + `EDL Strength`.
  - Camera presets: `Isometric`, `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`, `Reset Camera (R)`, `Focus Selected (F)`, `DarkStudio` / `LightStudio`.
- **Async Background Worker Status Pill** *(desktop non-WASM)*: Displays spinner, active job label, elapsed timer, and **Cancel** button during background file I/O or heavy filter evaluation.
- **Help & Software Auto-Update**: `Keyboard & Mouse Shortcuts (F1)`, `Demo Gallery`, **`Check for Updates...`**, and `Report an Issue / Bug...`. On desktop native builds, ParaViz checks GitHub Releases asynchronously on launch, notifies in the status bar when a newer SemVer tag exists, verifies the downloaded archive/AppImage against `SHA256SUMS.txt`, and performs atomic self-replacement (`self-replace`).

### Split Viewport Panes (`P1`–`P4`)
- Each pane header lets you click `P1`..`P4` to focus, switch **`3D View` | `XY Chart` | `Spreadsheet`**, and toggle **Per-Pane Node Visibility** (`Vis: Global` vs custom `Nodes (N)` checklist).
- Active 3D pane supports **Orbit (`1`)**, **Pick (`2`)** (HUD shows `Block: <path>` on MultiBlock hits), **Ruler (`3`)**, interactive drag/bottom-right resize on **Scalar Color Bar**, and draggable **Annotation** badges.

### Left Dock Tabs
1. **Pipeline Browser**: Visibility toggles, `All` / `None` / `Solo`, `Copy` / `Paste` (`Ctrl/Cmd+C` & `V`), `Copy Filter Chain`, Rename, `Dup`, Delete.
2. **Properties**: `Auto-Apply` toggle (`Apply` / `Discard`), filter/source parameter editors with bounds & plane snap buttons, plus **MultiBlock Patches** hierarchy (`All`, `None`, `Invert`, `Solo`, search box, per-block visibility checkboxes, per-block RGBA color overrides + `"×"` reset).
3. **Display**:
   - `Representation` (`Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`)
   - `Color By` (`Solid Color`, `Point: ...`, `Cell: ...`, vector `Magnitude` or component `X/Y/Z`)
   - `Colormap` (11 presets, `Invert`, `Log Scale`, `Auto`, `Fit Data`, `±Sym`, numeric `Min`/`Max`)
   - `Material & Geometry Style` (`Opacity` `0..=1`, `Ambient` `0..=1`, `Specular` `0..=1`, `Shininess` `1..=128`, `Two-Sided Lighting`, `Point Size` `1..=24`, `Line Width` `0.5..=10`, `Edge Width` `0.25..=6`, `Edge Color`)
   - `Volume Transfer Function` (`Step Size` `0.05..=2.0`, `Gradient Shading`, `3D Volume Clip Plane`, presets `Linear` / `Soft` / `Iso-Shell` / `Peak`, interactive curve + `Control Points Table`)
   - `MultiBlock Patches` & `Scalar Color Bar` cards
4. **Information**: `Last 3D Pick` (`Actor`, `Pos`, `Block`, `Pt`, `Cell`, `Scalar`, `Reveal in Spreadsheet`, `Reveal Cell #`), Dataset type/counts/memory/bounds (respecting hidden MultiBlock patches), array table with `Compute Mean / StdDev`.
5. **Scene**: `Viewport Background` (`ViewportBgPreset` + Top/Bottom RGB), `Camera & Navigation` (`Orthographic Projection`, `FOV` `15..=110` deg, `Ortho Scale`, `Turntable` + `Speed` `0.05..=2.5`, Position/Target, Bookmarks), `Render Quality & Overlays` (MSAA, EDL, Gizmo, BBox, Cube Axes Grid, Transparent Background, 3D Filter Widgets, `Scalar Color Bar`, `Time Annotation (Time:)` + custom `Time Label`), `Scalar Color Bar` layout (`Horizontal Orientation`, `Title`, `Labels` `2..=7`, `Font Scale` `0.6..=2.5`, `Reset Pos`, `Reset Size`), and `Annotations` (`+ Add Annotation`, visible toggle, text, RGBA color, `Size` `8..=48`, normalized `X`/`Y` `0..=1`).

### Bottom Drawer Tabs
- **Spreadsheet**: Paged `Points` / `Cells` / `Field` table, column filter, click-to-highlight in 3D, `Export CSV`.
- **Chart / Histogram**: Multi-series `XY Plot (ArcLength)` for `ProbeOverLine` / streamlines (`Markers`, `Log Y`, CSV export) and binned `Scalar Histogram`.
- **Rust Trace**: Compilable `paraviz` Rust code with `Copy to Clipboard`.

---

## 3. Keyboard & Mouse Shortcuts (`F1` / `?`)

| Input | Action |
| --- | --- |
| `Left Drag` (Orbit mode) | Orbit camera around target |
| `Shift + Left Drag` / `Middle Drag` | Pan camera |
| `Right Drag` / `Scroll` | Zoom / Dolly camera |
| `Single Click` | 3D Raycast Pick (Pick/Orbit) or place Ruler point (Ruler mode) |
| `Double Click` | Reset active pane camera to fit scene |
| `R` | Reset active pane camera to all visible |
| `F` | Focus camera on selected node |
| `1` / `2` / `3` | Switch Orbit / Pick / Ruler mode |
| `W` | Toggle 3D Filter Widgets |
| `E` | Toggle Eye-Dome Lighting (EDL) |
| `M` | Toggle 4x MSAA |
| `G` / `B` | Toggle Axes Gizmo / Bounding Box |
| `T` | Toggle Turntable spin |
| `Tab` | Toggle Left Inspector Dock |
| `Space` | Play / Pause time steps |
| `Ctrl/Cmd + C` / `V` | Copy / Paste filter in Pipeline Browser |
| `Delete` / `Backspace` | Remove selected pipeline node |
| `F1` / `?` | Open Keyboard & Mouse Shortcuts modal |

---

## 4. Headless CLI Reference (`paraviz --help`)

Supported CLI input formats include Legacy VTK (`.vtk`), VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, `.pvtp`, `.pvtu`, `.pvti`), VTKHDF (`.vtkhdf`), CGNS (`.cgns`), Exodus II (`.exo`, `.e`, `.ex2`), XDMF (`.xdmf`, `.xmf`), EnSight Gold (`.case`), OpenFOAM (`.foam`), NIfTI (`.nii`, `.nii.gz`), NRRD (`.nrrd`, `.nhdr`), MetaImage (`.mhd`, `.mha`), STL/OBJ/PLY, and CSV/TSV (plus saved `.json` session state).

### `demo` — Render a built-in demo scene
```bash
paraviz demo <wavelet|vortex|gyroid|volume|superquadric> -o <out.png> [--width W] [--height H]
```
Defaults: `--width 1280`, `--height 720`.

### `render` — Render a dataset or `.json` state file headlessly
```bash
paraviz render <input> -o <out.png|out.gif> [OPTIONS]
```

| Option | Default | Accepted Values / Behavior |
| --- | --- | --- |
| `-o`, `--output <path>` | *(required)* | Output `.png` (or animated `.gif` when combined with `--turntable`) |
| `--width <px>`, `--height <px>` | `1280` × `720` | Positive integer pixel dimensions |
| `--repr <mode>` | dataset auto | `surface`, `edges`, `wireframe`, `points`, `gaussian`, `volume`, `outline` |
| `--cmap <preset>` | `cool_to_warm` | `cool_to_warm`, `viridis`, `plasma`, `inferno`, `magma`, `turbo`, `fast`, `rainbow`, `grayscale`, `cividis` |
| `--color-by <array>` | auto / solid | Point or cell array name, or `solid` |
| `--camera <preset>` | `iso` | `iso`, `+x`, `-x`, `+y`, `-y`, `+z`, `-z` |
| `--edl` | `false` | Enables Eye-Dome Lighting screen-space contrast pass |
| `--edl-strength <val>` | `1.0` | Non-negative finite float (`>= 0.0`); implies `--edl` |
| `--msaa <1\|4>` | `1` | `1` (off) or `4` (4× MSAA) |
| `--axes-grid` | `false` | Draws 3D Cube Axes Grid with numeric tick marks |
| `--transparent` | `false` | Sets background alpha = `0` for PNG/GIF compositing |
| `--scalar-bar` | `false` | Composites Scalar Color Bar (`ScalarBarConfig`) onto output |
| `--time-annotation` | `false` | Composites `Time: <t>` badge (`SceneSettings::show_time_annotation`) onto output |
| `--turntable <frames>` | `None` | Renders `<frames>` 360° camera rotation frames (`<stem>_000.png` or animated `.gif`) |
| `--gif-delay <ms>` | `60` | Frame duration in milliseconds for `.gif` output |
| `--filter <spec>` | `[]` | Repeatable filter chain stage (see below) |

**Supported `--filter <spec>` values** (parsed in order, repeatable):
- `contour:<v1,v2,...>` — `ContourFilter` at one or more scalar values
- `decimate:<ratio>` — `DecimateFilter` (`0.5` if ratio omitted)
- `smooth` — `SmoothFilter::default()` (`Taubin`, `20` iterations)
- `triangulate` — `TriangulateFilter::default()`
- `normals` — `GenerateNormalsFilter::default()`
- `edges` — `ExtractEdgesFilter`
- `surface` — `ExtractSurface::new()`
- `resample:<n>` — `ResampleToImageFilter` on `[n, n, n]` grid (`32` if omitted)
- `isovolume:<lo>,<hi>` — `IsoVolumeFilter` on scalar band `[lo, hi]`

Example:
```bash
cargo run -p paraviz-app -- render volume.vti -o iso.png \
  --filter contour:120,175 --filter smooth \
  --repr edges --cmap viridis --color-by RTData \
  --camera iso --edl --msaa 4 --axes-grid --scalar-bar --time-annotation
```

### `info` — Print dataset metadata & array ranges
```bash
cargo run -p paraviz-app -- info mesh.vtkhdf
```

### `convert` — Convert between scientific and 3D formats
Supported output extensions: `.vtk`, `.vtp`, `.vti`, `.vtu`, `.stl`, `.obj`, `.ply`, `.csv`, `.glb`, `.html`.
```bash
cargo run -p paraviz-app -- convert input.vtk output.glb
cargo run -p paraviz-app -- convert input.exo viewer.html
```
