<img src="assets/icon.svg" width="64" height="64" alt="ParaViz logo">

# ParaViz

**Kitware ParaView rewritten in Rust using WebGPU (`wgpu`)**, with an ergonomic **Rust API**, a **parallel (`rayon`) scientific filter pipeline**, **VTK / mesh file I/O**, **WebGPU rendering**, an **interactive desktop GUI (`egui` + `egui-wgpu`)**, and a **headless CLI renderer**.

---

## Architecture

| Crate | ParaView / VTK Equivalent | Responsibilities |
| --- | --- | --- |
| [`paraviz-core`](crates/paraviz-core) | `vtkCommonDataModel`, `vtkCommonExecutionModel` | Typed multi-component `DataArray`, `FieldData`, `CellType`, `PolyData`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`, `MultiBlockDataSet`, `ColorMap` (11 presets + log scale + NaN color), `TransferFunction`, demand-driven `PipelineGraph` with dirty propagation & caching |
| [`paraviz-sources`](crates/paraviz-sources) | `vtkFiltersSources` | `SphereSource`, `CubeSource`, `ConeSource`, `CylinderSource`, `PlaneSource`, `LineSource`, `ArrowSource`, `DiskSource`, `TorusSource`, `WaveletSource`, `VectorFieldSource`, `SuperquadricSource`, `GyroidSource`, `PointCloudSource` |
| [`paraviz-filters`](crates/paraviz-filters) | `vtkFiltersCore` / `vtkFiltersGeneral` / `vtkFiltersFlowPaths` | Parallel (`rayon`) `ExtractSurface`, `ContourFilter` (256-case Marching Cubes + Marching Tetrahedra with edge-hashed vertex welding), `SliceFilter`, `ClipFilter`, `ThresholdFilter`, `StreamTracerFilter` (RK4 / RK45), `TubeFilter` (parallel-transport Bishop frames), `GlyphFilter`, `WarpByVectorFilter`, `WarpByScalarFilter`, `GradientFilter` (Gradient / Divergence / Vorticity / Q-Criterion), `CalculatorFilter`, `SmoothFilter` (Laplacian / Taubin), `TransformFilter`, `CellDataToPointDataFilter`, `PointDataToCellDataFilter`, `ProbeOverLineFilter` |
| [`paraviz-io`](crates/paraviz-io) | `vtkIOXML`, `vtkIOLegacy`, `vtkIOGeometry` | Legacy `.vtk` (ASCII & big-endian binary), VTK XML (`.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvd` — ASCII, base64 binary, appended, and `vtkZLibDataCompressor`), `.stl` (ASCII & binary), `.obj`, `.ply` (ASCII & binary LE/BE), `.csv`/`.tsv`, plus `VtkFileSource` & `PvdSeriesSource` |
| [`paraviz-render`](crates/paraviz-render) | `vtkRenderingOpenGL2`, `vtkRenderingVolume` | WebGPU (`wgpu`) rendering engine: `Surface`, single-pass barycentric `SurfaceWithEdges`, `Wireframe`, sphere-impostor `Points`, 3D `Volume` raymarching (`R32Float` + IGN ray jittering + central-difference gradient shading), `Outline`, 3D Orientation Axes Gizmo, cached GPU buffers, headless RGBA/PNG export |
| [`paraviz`](src/lib.rs) | `paraview.simple` | Ergonomic high-level Rust API (`ParaViz`, `DisplayHandle`, `SourceConfig`, `FilterConfig`, `DemoPreset`), JSON state save/load, Rust Trace export |
| [`paraviz-app`](crates/paraviz-app) | ParaView Desktop & `pvbatch` | Interactive WebGPU GUI (`eframe`/`egui` + `egui-wgpu`) with Pipeline Browser, Properties (with Auto-Apply!), Display/Color Map Editor, Information panel, SpreadSheet view, Live Rust Trace, and headless CLI (`demo`, `render`, `info`, `convert`) |

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
```

---

## Desktop App & CLI

Launch the interactive WebGPU ParaViz GUI:
```bash
cargo run -p paraviz-app
```

Headless CLI commands:
```bash
# Render a built-in demo scene headlessly with WebGPU
cargo run -p paraviz-app -- demo wavelet -o wavelet.png --width 1280 --height 720

# Render any VTK / VTP / VTI / VTU / STL / OBJ / PLY / CSV file headlessly
cargo run -p paraviz-app -- render mesh.vtu -o out.png --repr edges --cmap viridis --camera iso

# Inspect dataset statistics, bounds, and field arrays
cargo run -p paraviz-app -- info mesh.vtu

# Convert between scientific file formats
cargo run -p paraviz-app -- convert input.vtk output.vtp
```

---

## Running Tests

```bash
cargo test --workspace
```
