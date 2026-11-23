<img src="assets/icon.svg" width="64" height="64" alt="ParaViz logo">

# ParaViz

**A fast, modern scientific visualization studio and headless renderer inspired by Kitware ParaView — built with Rust and WebGPU.**

Explore 3D meshes, CFD flows, finite-element grids, and medical/volumetric scans in an interactive desktop app, automate batch renders from the command line, or embed pipelines directly in Rust and Python.

---

## Install & Launch

### Prebuilt Releases
Download ready-to-run packages from [**GitHub Releases**](https://github.com/DimitriTimoz/paraviz/releases):
- **Linux (`x86_64` & `aarch64`)**: `.AppImage` (portable, no install needed), `.deb`, or `.tar.gz`
- **macOS (Universal Apple Silicon + Intel)**: `.dmg` (`ParaViz.app`) or `.tar.gz`
- **Windows (`x86_64` & `aarch64`)**: `.zip` (`paraviz.exe`)

> On Linux, make the AppImage executable and run it directly:
> ```bash
> chmod +x ParaViz-*.AppImage
> ./ParaViz-*.AppImage path/to/dataset.vtu
> ```

### From Source
```bash
cargo run --release -p paraviz-app
# Or open a dataset / saved session directly:
cargo run --release -p paraviz-app -- path/to/mesh.vtu
```

---

## Key Capabilities

- **Interactive WebGPU Studio**: Real-time `Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, and `Outline` views with 4x MSAA, Eye-Dome Lighting (EDL), Dark/Light themes, and camera bookmarks.
- **Live Pipeline & Copy/Paste Filters**: 37+ parallel filters (isosurfaces, slices, clips, thresholds, streamlines & tubes, vector glyphs, QEM decimation, Delaunay 2D/3D, calculator, line probing, domain integration, time steps) with **Auto-Apply** and **Copy/Paste Filter Chains** (`Ctrl/Cmd+C`, `Ctrl/Cmd+V`).
- **3D Volume Rendering**: Hardware-agnostic `R32Float` raymarching with interactive histogram opacity curves, transfer-function presets, gradient shading, and a live **3D Volume Clip Plane**.
- **Inspection & Quantitative Plots**: 3D raycast picking (`Reveal in Spreadsheet`), 2-point 3D distance ruler, searchable & CSV-exportable attribute spreadsheets, `Plot Over Line` XY charts, and scalar histograms.
- **Broad Format Support**:
  - **VTK & 3D Meshes**: `.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd`, partitioned `.pvtp`/`.pvti`/`.pvtu`/`.pvtr`/`.pvts`, `.stl`, `.obj`, `.ply`, `.csv`/`.tsv`
  - **Medical & Volumetric**: NIfTI-1 (`.nii`, `.nii.gz`), NRRD (`.nrrd`, `.nhdr`), MetaImage (`.mhd`, `.mha`), Raw binary volumes (`.raw`)
  - **CFD / CAE**: EnSight Gold (`.case`), OpenFOAM (`polyMesh`), XDMF (`.xdmf`, `.xmf`)
- **Publication & Web Exports**:
  - High-res **PNG** screenshots (up to 4K, with optional **transparent background** & **Scalar Color Bar**)
  - Animated **GIF** exports (360° turntables & time series)
  - Binary **glTF 2.0 (`.glb`)** with baked colormap vertex colors
  - Self-contained interactive **WebGL2 `.html`** 3D viewers you can share and open in any browser

---

## How ParaViz Differs from Classic ParaView

- **Crisper Edges & Cleaner Volumes**: Single-pass barycentric `SurfaceWithEdges` eliminates polygon-offset Z-fighting, and ray-jittered `R32Float` volume raymarching avoids slicing banding.
- **Instant Live Editing**: `Auto-Apply` is enabled by default in Properties (with explicit Apply/Discard still available for heavy datasets), plus native Copy/Paste of filters and chains.
- **Built-In Portable Sharing**: Export animated `.gif` turntables, `.glb` meshes, and standalone WebGL2 `.html` viewers directly without plugins.
- **Lightweight Single Binary**: Fast startup with zero Qt/VTK installation overhead, plus live **Rust Trace** export alongside JSON session files.
- **Current Scope**: Focused on single-workstation visualization and batch CLI rendering (no distributed MPI `pvserver`, OSPRay path-tracing, or HDF5/CGNS/CAD B-Rep readers yet).

See the full side-by-side comparison in [**Differences from Classic ParaView**](docs/differences-from-paraview.md).

---

## Headless CLI Quick Start

Render, inspect, and convert datasets without opening a window:

```bash
# Render a built-in demo scene
cargo run -p paraviz-app -- demo wavelet -o wavelet.png --width 1280 --height 720

# Render a dataset with chained filters, EDL, 4x MSAA, transparent background & color bar
cargo run -p paraviz-app -- render volume.vti -o iso.png \
  --filter contour:140 --repr edges --cmap viridis --color-by RTData \
  --edl --msaa 4 --transparent --scalar-bar

# Export a 360-degree animated GIF turntable
cargo run -p paraviz-app -- render mesh.vtu -o spin.gif --turntable 36 --edl

# Inspect dataset metadata, bounds, and scalar ranges
cargo run -p paraviz-app -- info mesh.vtu

# Convert to binary glTF (.glb) or an interactive WebGL2 (.html) viewer
cargo run -p paraviz-app -- convert mesh.vtu viewer.html
```

---

## Rust API Quick Start

Use `paraviz` as a library in your own Rust project:

```rust
use paraviz::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pv = ParaViz::new();

    let wav = pv.add_source("Wavelet", WaveletSource::default());
    pv.hide(wav);

    let iso = pv.add_filter(
        "Contour",
        wav,
        ContourFilter::new("RTData", vec![110.0, 175.0]),
    )?;

    pv.show(iso)
        .representation(Representation::SurfaceWithEdges)
        .colormap(ColorMapPreset::Viridis)
        .color_by_point("RTData");

    pv.camera_preset(CameraPreset::Isometric)?;
    pv.render_png("wavelet.png", 1280, 720)?;
    Ok(())
}
```

Run the included examples (`examples/`):
```bash
cargo run --example wavelet_contour
cargo run --example streamlines_and_tubes
cargo run --example volume_raymarch
cargo run --release --example benchmark_pipeline
```

---

## Documentation

Full end-user and developer guides live in [`docs/`](docs/README.md):

- [**User Guide**](docs/user-guide.md) — Step-by-step Studio GUI workflows, volume rendering, CFD streamlines, picking, and exports
- [**Differences from Classic ParaView**](docs/differences-from-paraview.md) — UX improvements, shortcut mapping, and feature scope
- [**Desktop Studio GUI & Headless CLI**](docs/gui-and-cli.md) — Complete controls, keyboard/mouse shortcuts (`F1`), and CLI reference
- [**File Formats & I/O**](docs/file-formats.md) — Supported VTK, 3D, medical, and CFD formats & exports
- [**Sources & Filters Reference**](docs/filters-and-sources.md) — All 14 procedural sources, 37+ filters, and `CalculatorFilter` syntax
- [**Rust API Guide (& Python Bindings)**](docs/rust-api.md) — High-level `ParaViz` API, `DataSet` construction, and `paraviz-py`
- [**Architecture & WebGPU Rendering Pipeline**](docs/architecture.md) — Workspace crate breakdown, pipeline DAG, and WGSL shaders

---

## Feedback & Issues

Encountered a bug, rendering issue, file-loading error, or missing ParaView feature? Please **[open an issue on GitHub](https://github.com/DimitriTimoz/paraviz/issues/new/choose)**.

Including your OS/GPU backend, CLI command (or saved `.json` state / **Rust Trace** output), and steps to reproduce helps us investigate quickly.

---

## Development & Tests

```bash
# Run the full test suite
cargo test --workspace

# Generate local API documentation
cargo doc --workspace --exclude paraviz-app --no-deps --open

# Build a local Linux AppImage
cargo build --release -p paraviz-app
./packaging/linux/build-appimage.sh target/release/paraviz x86_64 ParaViz-x86_64.AppImage
```
