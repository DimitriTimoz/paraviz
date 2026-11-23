# Desktop Studio GUI & Headless CLI

The `paraviz-app` crate builds the `paraviz` executable, providing both an interactive WebGPU desktop application (`eframe` / `egui` + `egui-wgpu`) and a headless batch CLI.

## Launching the Desktop GUI

```bash
# Empty workspace
cargo run -p paraviz-app

# Open a dataset or saved JSON state directly
cargo run -p paraviz-app -- path/to/mesh.vtu
cargo run -p paraviz-app -- gui path/to/state.json
```

You can also drag-and-drop any supported dataset or state file directly onto the application window.

---

## Studio Workspace Layout

### 1. Top Menu Bar & Quick Controls
- **File**: Open File / File Browser, Recent Files, Raw Volume Importer (`.raw`/`.bin`/`.vol`), Save/Load State (`.json`), Export Dataset (including `.glb` and interactive `.html` 3D web viewer with optional colormap vertex colors), Save Screenshot (`.png` with optional transparent background and Scalar Color Bar), and Export Animation (`.gif` turntable or time series).
- **Sources**: All 14 procedural geometric & analytical sources.
- **Filters**: All pipeline filters (enabled contextually for the selected node) plus **GPU Compute Contour (WebGPU)...** for `ImageData` volumes.
- **Demos**: Launch any of the 5 built-in `DemoPreset` scenes.
- **View**: Toggle Inspector Dock, Bottom Drawer, `DarkStudio` / `LightStudio` UI theme, camera presets (`Isometric`, `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`), `Reset Camera`, and `Focus Selected`.
- **Help**: Demo Gallery and Keyboard & Mouse Shortcuts (`F1`).

### 2. Left Dock: Pipeline Browser & Inspector
- **Pipeline Browser**: Tree view with visibility eye toggles, `All` / `None` / `Solo` actions, **Copy / Paste Filter** (`Ctrl/Cmd+C` & `Ctrl/Cmd+V`, plus right-click `Copy Filter Chain` / `Paste Filter Chain`), inline Rename, Duplicate, and Delete.
- **Properties Tab**: Auto-Apply toggle (with explicit Apply/Reset), dynamic array dropdowns, and bounds presets for sources and filters.
- **Display Tab**: Representation selector (`Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`), Color By array & component, Color Map preset + log-scale + custom range/rescale-to-data, Opacity, Volume Transfer Function editor, Point Size / Line Width / Edge Color & Width, and material lighting (`Ambient`, `Specular`, `Shininess`, `Two-Sided`).
- **Information Tab**: Dataset type, point/cell counts, memory usage, 3D bounding box, and per-array scalar ranges/types.
- **Scene Tab**: Background skybox gradient presets, FOV / Orthographic toggle, Camera Bookmarks, MSAA (`1x` / `4x`), Eye-Dome Lighting (`EDL` + strength), Axes Gizmo, Bounding Box, 3D Axes Grid, and Turntable spin.

### 3. Center WebGPU Viewport
- Supports three interaction modes (`1` Orbit, `2` Pick, `3` Ruler) with a floating HUD displaying picked point/cell IDs, world coordinates, scalar values, distance measurements, `Reveal in Spreadsheet`, `Focus`, and `Clear` actions.
- Renders optional 3D `ViewportWidget` guides (`Plane`, `Line`, `Box`, `Sphere`) synced with active filters (`W` to toggle).

### 4. Bottom Drawer
- **Spreadsheet**: Inspect Point, Cell, or Field attributes with column filtering, click-to-highlight in 3D, and **Export CSV**.
- **Chart / Histogram**: Interactive value histogram and `ProbeOverLine` XY plots.
- **Rust Trace**: Live generated Rust API code (`pv.to_rust_trace()`) reproducing the current interactive session.

---

## Keyboard & Mouse Controls (`F1` / `?`)

| Input | Action |
| --- | --- |
| `Left Drag` (Orbit mode) | Orbit camera around target |
| `Shift + Left Drag` / `Middle Drag` | Pan camera |
| `Right Drag` / `Scroll` | Zoom / Dolly camera |
| `Single Click` | 3D Raycast Pick (Pick/Orbit) or place Ruler point (Ruler mode) |
| `Double Click` | Reset camera to fit scene |
| `R` | Reset camera to all visible |
| `F` | Focus camera on selected node |
| `1` / `2` / `3` | Switch Orbit / Pick / Ruler mode |
| `W` | Toggle 3D Filter Widgets |
| `E` | Toggle Eye-Dome Lighting (EDL) |
| `M` | Toggle 4x MSAA |
| `G` / `B` | Toggle Axes Gizmo / Bounding Box |
| `T` | Toggle Turntable spin |
| `Tab` | Toggle Left Inspector Dock |
| `Space` | Play / Pause time steps (when a time series is loaded) |
| `Ctrl/Cmd + C` / `V` | Copy / Paste filter in Pipeline Browser |
| `Delete` / `Backspace` | Remove selected pipeline node |
| `F1` / `?` | Open Keyboard & Mouse Shortcuts modal |

---

## Headless CLI Reference

Run `paraviz --help` (or `cargo run -p paraviz-app -- --help`) for a summary of all commands:

### `demo` — Render a built-in demo scene
```bash
paraviz demo <wavelet|vortex|gyroid|volume|superquadric> -o <out.png> [--width W] [--height H]
```

### `render` — Render a dataset or `.json` state file headlessly
```bash
paraviz render <input> -o <out.png> [OPTIONS]
```

Options:
- `-o, --output <path>` — Output PNG path (required)
- `--width <px>`, `--height <px>` — Output dimensions (default: `1280` x `720`)
- `--repr <mode>` — `surface`, `edges`, `wireframe`, `points`, `gaussian`, `volume`, `outline`
- `--cmap <preset>` — `cool_to_warm`, `viridis`, `plasma`, `inferno`, `magma`, `turbo`, `fast`, `rainbow`, `grayscale`, `cividis`
- `--color-by <array>` — Point/cell array name, or `solid`
- `--camera <preset>` — `iso`, `+x`, `-x`, `+y`, `-y`, `+z`, `-z`
- `--edl` — Enable Eye-Dome Lighting
- `--edl-strength <val>` — EDL strength (`>= 0.0`, default: `1.0`)
- `--msaa <1|4>` — MSAA sample count (`1` or `4`, default: `1`)
- `--axes-grid` — Show 3D Cube Axes Grid with tick marks
- `--transparent` — Transparent background (`alpha = 0`) for PNG/GIF exports
- `--scalar-bar` — Composite a ParaView-style Scalar Color Bar with min/mid/max labels
- `--turntable <frames>` — Render a 360-degree turntable sequence (`<stem>_000.png`, ... or animated `.gif` when `-o` ends in `.gif`)
- `--gif-delay <ms>` — Frame delay in milliseconds for `.gif` exports (default: `60`)
- `--filter <spec>` — Chain one or more filters before rendering (repeatable):
  - `contour:<v1,v2,...>`
  - `decimate:<ratio>` (default `0.5`)
  - `smooth`
  - `triangulate`
  - `normals`
  - `edges`
  - `surface`
  - `resample:<n>` (uniform `n^3` `ResampleToImage`, default `32`)
  - `isovolume:<lo>,<hi>`

Example combining filters, EDL, MSAA, and turntable output:
```bash
cargo run -p paraviz-app -- render volume.vti -o iso.png \
  --filter contour:120,175 --filter smooth \
  --repr edges --cmap viridis --color-by RTData \
  --camera iso --edl --msaa 4 --axes-grid
```

### `info` — Print dataset metadata & array ranges
```bash
cargo run -p paraviz-app -- info mesh.vtu
```

### `convert` — Convert between scientific and 3D formats
```bash
cargo run -p paraviz-app -- convert input.vtk output.glb
cargo run -p paraviz-app -- convert input.vtu output.stl
```
