# ParaViz Documentation

Welcome to the **ParaViz** documentation — Kitware ParaView rewritten in Rust using WebGPU (`wgpu`).

## Guides & References

### For Users
- [**User Guide**](user-guide.md) — Complete walkthrough of the Studio GUI, step-by-step workflows (isosurfaces, volume raymarching & medical imaging, CFD streamlines, picking/probing/charts, animations, and web exports), and troubleshooting.
- [**Differences from Classic ParaView**](differences-from-paraview.md) — Technical improvements, workflow/shortcut mapping, and honest feature scope compared to Kitware ParaView.
- [**Desktop Studio GUI & Headless CLI**](gui-and-cli.md) — Detailed reference for every GUI tab, keyboard/mouse control (`F1`), and headless CLI option.
- [**File Formats & I/O**](file-formats.md) — Supported 3D, VTK, medical, and CFD formats, raw volume descriptors, `.pvd` time series, `.glb`, `.gif`, and `.html` exports.

### For Developers & Script Authors
- [**Architecture & Rendering Pipeline**](architecture.md) — Workspace crate structure, demand-driven `PipelineGraph`, WebGPU render passes, single-pass `SurfaceWithEdges`, `R32Float` IGN volume raymarching, Eye-Dome Lighting (EDL), and GPU Compute Marching Cubes.
- [**Rust API Guide (& Python Bindings)**](rust-api.md) — Using `ParaViz`, `DisplayHandle`, `PipelineGraph` & `DataSet` construction, direct `WebGpuRenderer` / `GpuContourEngine`, JSON state persistence, Rust Trace export, and `paraviz-py`.
- [**Sources & Filters Reference**](filters-and-sources.md) — Complete catalog of all 14 procedural sources, all 37 `FilterConfig` filters, multi-input filters, `CalculatorFilter` expression syntax, and analytical helpers.

## Building API Documentation (Rustdoc)

Generate and open the full crate API documentation locally:

```bash
cargo doc --workspace --exclude paraviz-app --no-deps --open
```

## Encountering an Issue?

If you run into a bug, unexpected error, rendering issue, or missing capability, please **[open an issue](https://github.com/DimitriTimoz/paraviz/issues/new/choose)** with your OS/GPU details and steps to reproduce (or a Rust trace / `.json` pipeline state).
