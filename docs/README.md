# ParaViz Documentation

Welcome to the **ParaViz** documentation — Kitware ParaView rewritten in Rust using WebGPU (`wgpu`).

## Guides & References

- [**Architecture & Rendering Pipeline**](architecture.md) — Workspace crate structure, demand-driven `PipelineGraph`, WebGPU render passes, single-pass `SurfaceWithEdges`, `R32Float` IGN volume raymarching, Eye-Dome Lighting (EDL), and GPU Compute Marching Cubes.
- [**Rust API Guide**](rust-api.md) — Using `ParaViz` (high-level session API), `DisplayHandle`, `PipelineGraph` & `DataSet` construction, direct `WebGpuRenderer` / `RenderScene`, JSON state serialization, and Rust Trace export.
- [**Sources & Filters Reference**](filters-and-sources.md) — Complete catalog of all 14 procedural sources, all 36 `FilterConfig` filters, multi-input filters, `CalculatorFilter` expression syntax, and analytical helpers.
- [**File Formats & I/O**](file-formats.md) — Readers, writers, compression modes, and pipeline sources for VTK Legacy/XML, `.pvd` time series, STL/OBJ/PLY/CSV, binary glTF (`.glb`), Raw Volumes, NIfTI-1, NRRD, MetaImage, EnSight Gold, XDMF, and OpenFOAM.
- [**Desktop Studio GUI & Headless CLI**](gui-and-cli.md) — Interactive `egui` + `egui-wgpu` Studio layout, Inspector & Drawer tabs, 3D Picking/Ruler/Widgets, keyboard & mouse shortcuts (`F1`), and headless CLI (`demo`, `render`, `info`, `convert`).

## Building API Documentation (Rustdoc)

Generate and open the full crate API documentation locally:

```bash
cargo doc --workspace --exclude paraviz-app --no-deps --open
```

## Encountering an Issue?

If you run into a bug, unexpected error, rendering issue, or missing capability, please **[open an issue](https://github.com/DimitriTimoz/paraviz/issues/new/choose)** with your OS/GPU details and steps to reproduce (or a Rust trace / `.json` pipeline state).
