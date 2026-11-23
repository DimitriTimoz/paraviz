<img src="assets/icon.svg" width="64" height="64" alt="ParaViz logo">

# ParaViz

**Kitware ParaView rewritten in Rust using WebGPU (`wgpu`)**, with an ergonomic **Rust API**, a **parallel (`rayon`) scientific filter pipeline**, **VTK / mesh file I/O**, **WebGPU rendering**, an **interactive desktop GUI (`egui` + `egui-wgpu`)**, and a **headless CLI renderer**.

---

## Architecture

| Crate | ParaView / VTK Equivalent | Responsibilities |
| --- | --- | --- |
| [`paraviz-core`](crates/paraviz-core) | `vtkCommonDataModel`, `vtkCommonExecutionModel` | Typed multi-component `DataArray`, `FieldData`, `CellType`, `PolyData`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`, `MultiBlockDataSet`, `ColorMap` (11 presets + log scale + NaN color), `TransferFunction`, demand-driven `PipelineGraph` with dirty propagation & caching |
| [`paraviz-sources`](crates/paraviz-sources) | `vtkFiltersSources` | `SphereSource`, `CubeSource`, `ConeSource`, `CylinderSource`, `PlaneSource`, `LineSource`, `ArrowSource`, `DiskSource`, `TorusSource`, `WaveletSource`, `VectorFieldSource`, `SuperquadricSource`, `GyroidSource`, `PointCloudSource` |
| [`paraviz-filters`](crates/paraviz-filters) | `vtkFiltersCore` / `vtkFiltersGeneral` / `vtkFiltersFlowPaths` | Parallel (`rayon`) `ExtractSurface`, `ContourFilter` (256-case Marching Cubes + Marching Tetrahedra with edge-hashed vertex welding), `SliceFilter`, `ClipFilter`, `ThresholdFilter`, `IsoVolumeFilter`, `StreamTracerFilter` (RK4 / RK45, point & cell vectors) & `StreamTracerWithCustomSource`, `TubeFilter` (parallel-transport Bishop frames), `GlyphFilter`, `WarpByVectorFilter`, `WarpByScalarFilter`, `GradientFilter` (Gradient / Divergence / Vorticity / Q-Criterion / Lambda2), `CalculatorFilter`, `SmoothFilter` (Laplacian / Taubin), `DecimateFilter` (QEM), `Delaunay2DFilter`, `Delaunay3DFilter`, `ResampleToImageFilter`, `ResampleWithDatasetFilter`, `IntegrateVariablesFilter`, `ExtractBlockFilter`, `TemporalStatisticsFilter`, `AddTimeStepsFilter`, `TransformFilter`, `CellDataToPointDataFilter`, `PointDataToCellDataFilter`, `ProbeOverLineFilter`, `FeatureEdgesFilter`, `ConnectivityFilter`, `ShrinkFilter`, `CellCentersFilter`, `ReflectFilter`, `GenerateNormalsFilter`, `ElevationFilter`, `TriangulateFilter`, `ExtractEdgesFilter`, `CleanPolyDataFilter`, `append_datasets`, `compute_histogram`, `interpolate_datasets` |
| [`paraviz-io`](crates/paraviz-io) | `vtkIOXML`, `vtkIOLegacy`, `vtkIOGeometry` | Legacy `.vtk` (ASCII & big-endian binary), VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd` — ASCII, base64 binary, appended, and `vtkZLibDataCompressor` read/write), partitioned parallel XML readers (`.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts`), raw binary 3D volume reader (`read_raw_volume`), NIfTI-1 (`.nii`, `.nii.gz`), NRRD (`.nrrd`, `.nhdr`), MetaImage (`.mhd`, `.mha`), EnSight Gold (`.case`), XDMF (`.xdmf`, `.xmf`), OpenFOAM `polyMesh`, `.stl` (ASCII & binary), `.obj`, `.ply` (ASCII & binary LE/BE), `.csv`/`.tsv`, binary glTF 2.0 export (`.glb` with optional scalar `COLOR_0`), animated `.gif` export (`write_animated_gif`), self-contained interactive WebGL2 `.html` viewer export (`write_html_viewer`), plus `VtkFileSource` & `PvdSeriesSource` |
| [`paraviz-render`](crates/paraviz-render) | `vtkRenderingOpenGL2`, `vtkRenderingVolume` | WebGPU (`wgpu`) rendering engine: 3-point Light Kit, `Surface`, single-pass barycentric `SurfaceWithEdges`, `Wireframe`, sphere-impostor `Points`, `PointGaussian`, 3D `Volume` raymarching (`R32Float` + IGN ray jittering + central-difference gradient shading), `Outline`, 4x MSAA, Eye-Dome Lighting (`EDL`) depth-contrast pass, transparent background export, headless Scalar Color Bar compositor, 3D raycast picking (`pick_scene`), interactive 3D `ViewportWidget` overlays (`Plane`, `Line`, `Box`, `Sphere`), WebGPU Compute Marching Cubes (`GpuContourEngine`), 3D Orientation Axes Gizmo & Cube Axes Grid, cached GPU buffers, headless RGBA/PNG export |
| [`paraviz`](src/lib.rs) | `paraview.simple` | Ergonomic high-level Rust API (`ParaViz`, `DisplayHandle`, `SourceConfig`, `FilterConfig` with 37 filter variants, `DemoPreset`), `gpu_contour`, `pick`, `widgets`, `open_raw_volume`, `export_glb`, `export_html`, `render_turntable_gif`, `render_timeseries_gif`, JSON state save/load, Rust Trace export |
| [`paraviz-app`](crates/paraviz-app) | ParaView Desktop & `pvbatch` | Interactive WebGPU Studio GUI (`eframe`/`egui` + `egui-wgpu`) with `DarkStudio`/`LightStudio` themes, Pipeline Browser (`All`/`None`/`Solo`/Copy-Paste Filter & Chain/Rename/Duplicate), Auto-Apply Properties with array dropdowns & bounds presets, Display/Lighting & Color Map Editor, Scene tab (skybox gradients, FOV, turntable, Camera Bookmarks, Transparent Export), Raw Volume Importer, GPU Compute Contour Dialog, Export Animation (GIF), MSAA 4x & EDL Strength, 3D Picking reticle & Distance Ruler pins, 3D Viewport Widgets, Information & CSV-exportable Spreadsheet/XY Chart/Histogram tabs, In-App File Browser, Demo Gallery, Shortcuts (`F1`), Live Rust Trace, and headless CLI (`demo`, `render` with `--edl`/`--msaa`/`--filter`/`--axes-grid`/`--turntable`/`--transparent`/`--scalar-bar`, `info`, `convert` incl. `.glb`/`.html`) |

---

## Improvements Over Classic ParaView

1. **Zero-Z-Fighting Single-Pass `SurfaceWithEdges`**: Uses WGSL screen-space `fwidth()` on barycentric coordinates for crisp, anti-aliased wireframe overlays without polygon-offset artifacts.
2. **Banding-Free WebGPU Volume Raymarching**: Uses Interleaved Gradient Noise (IGN) ray jittering, full `R32Float` 3D precision (manual trilinear interpolation in WGSL requiring zero optional GPU features), central-difference gradient Phong shading, and early ray termination.
3. **Auto-Apply Toggle in Properties**: Eliminates the classic "forgot to click Apply" friction while preserving explicit Apply/Reset control for heavy pipelines.
4. **Watertight Parallel Marching Cubes**: Spatial edge-key vertex welding deduplicates shared vertices across cell edges in parallel, reducing triangle vertex memory by ~5-6× and yielding smooth normals.
5. **Twist-Free Streamline Tubes**: `TubeFilter` uses parallel-transport (Bishop) frames instead of Frenet frames to prevent twisting/pinching around inflection points.
6. **Live Rust Trace Export**: Generates reproducible Rust API code directly from an interactive pipeline session (`pv.to_rust_trace()`).

---

## Rust API Quick Start

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

Run the included examples:
```bash
cargo run --example wavelet_contour
cargo run --example streamlines_and_tubes
cargo run --example volume_raymarch
cargo run --release --example benchmark_pipeline
```

Sample release benchmark output (`benchmark_pipeline`, 128³ `WaveletSource`, Apple Metal WebGPU):
- `WaveletSource` 128³ (2,097,152 pts, 2,048,383 cells): `24.01 ms`
- CPU Parallel Rayon Marching Cubes (138,079 pts, 272,506 tris): `93.34 ms`
- WebGPU Compute Marching Cubes (`GpuContourEngine`, 138,079 pts, 272,506 tris): `96.69 ms`
- WebGPU Cached Render (1280x720, 4x MSAA + EDL + SurfaceWithEdges): `6.05 ms`

---

## Desktop App & CLI

Launch the interactive WebGPU ParaViz GUI:
```bash
cargo run -p paraviz-app
```

> **Prebuilt Multi-Platform Packages**: GitHub Actions (`.github/workflows/release.yml`) builds release packages for **Linux `x86_64` & `aarch64`** (`.AppImage`, `.deb`, `.tar.gz`), **macOS Universal** (`.dmg`, `.app`, `.tar.gz`), and **Windows `x86_64` & `aarch64`** (`.zip`).
> You can also build a local Linux AppImage via:
> ```bash
> cargo build --release -p paraviz-app
> ./packaging/linux/build-appimage.sh target/release/paraviz x86_64 ParaViz-x86_64.AppImage
> ```

Headless CLI commands:
```bash
# Render a built-in demo scene headlessly with WebGPU
cargo run -p paraviz-app -- demo wavelet -o wavelet.png --width 1280 --height 720

# Render any VTK / VTP / VTI / PVTP / PVTU / VTU / STL / OBJ / PLY / CSV / NIfTI / EnSight / XDMF / OpenFOAM dataset headlessly (with optional filter chains, axes grid, or turntable sequence)
cargo run -p paraviz-app -- render mesh.vtu -o out.png --repr edges --cmap viridis --camera iso --edl --msaa 4 --axes-grid

# Inspect dataset statistics, bounds, and field arrays
cargo run -p paraviz-app -- info mesh.vtu

# Convert between scientific and 3D formats (including binary glTF .glb)
cargo run -p paraviz-app -- convert input.vtk output.glb
```

---

## Documentation

Full guides and reference documentation live in [`docs/`](docs/README.md):

- [Architecture & WebGPU Rendering Pipeline](docs/architecture.md)
- [Rust API Guide](docs/rust-api.md)
- [Sources & Filters Reference](docs/filters-and-sources.md)
- [File Formats & I/O](docs/file-formats.md)
- [Desktop Studio GUI & Headless CLI](docs/gui-and-cli.md)

Generate local rustdoc HTML documentation:
```bash
cargo doc --workspace --no-deps --open
```

---

## Running Tests

```bash
cargo test --workspace
```

---

## Feedback & Issues

If you encounter a bug, crash, file-parsing error, rendering artifact, or missing ParaView feature, please **[open an issue on GitHub](https://github.com/DimitriTimoz/paraviz/issues/new/choose)**.

Including your OS/GPU backend, CLI command (or saved `.json` state / **Rust Trace** output), and steps to reproduce helps us resolve issues quickly.
