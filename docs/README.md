# ParaViz Documentation

Welcome to the **ParaViz** documentation — a fast scientific visualization studio and headless renderer inspired by Kitware ParaView, built in **Rust** and **WebGPU (`wgpu`)**.

<p align="center">
  <img src="../assets/pipeline-architecture.svg" width="820" alt="ParaViz Demand-Driven Pipeline and WebGPU Architecture">
</p>

---

## Quick Start by Role

| Role | Where to Start | Key Topics |
| --- | --- | --- |
| **Desktop Analyst** | [**User Guide**](user-guide.md) | Studio workspace, volume raymarching, CFD streamlines, 3D picking, charts, and web exports |
| **ParaView User** | [**Differences from Classic ParaView**](differences-from-paraview.md) | UX improvements, shortcut mapping (`1`/`2`/`3`, `R`, `F`, `F1`), and current scope |
| **CLI Batch User** | [**Desktop Studio GUI & Headless CLI**](gui-and-cli.md#headless-cli-reference) | `demo`, `render`, `info`, `convert`, filter chaining, `--transparent`, `--scalar-bar`, and `.gif` turntables |
| **Rust API Developer** | [**Rust API Guide (& Python Bindings)**](rust-api.md) | `ParaViz` session API, `PipelineGraph`, `DataSet`, `WebGpuRenderer`, `GpuContourEngine`, and `to_rust_trace()` |
| **Python User** | [**Rust API Guide (& Python Bindings)**](rust-api.md#5-python-bindings-paraviz-py) | Standalone PyO3 `paraviz-py` bindings (`paraviz.Session` modeled after `paraview.simple`) |
| **Contributor** | [**Contributing Guide**](../CONTRIBUTING.md) & [**Architecture**](architecture.md) | Workspace crate breakdown, pipeline DAG, WGSL shaders, and test commands |

---

## Complete Reference Index

### User & Workflow Guides
- [**User Guide**](user-guide.md) — Complete walkthrough of the Studio GUI, step-by-step workflows (isosurfaces, volume raymarching & medical imaging, CFD streamlines, picking/probing/charts, animations, and web exports), and troubleshooting.
- [**Differences from Classic ParaView**](differences-from-paraview.md) — Technical improvements, workflow/shortcut mapping, and honest feature scope compared to Kitware ParaView.
- [**Desktop Studio GUI & Headless CLI**](gui-and-cli.md) — Detailed reference for every GUI tab, keyboard/mouse control (`F1`), and headless CLI option.
- [**File Formats & I/O**](file-formats.md) — Supported 3D, VTK, medical, and CFD formats, `RawVolumeDescriptor`, `.pvd` time series, `.glb`, `.gif`, and `.html` exports.

### Developer & Pipeline Reference
- [**Architecture & Rendering Pipeline**](architecture.md) — Workspace crate structure, demand-driven `PipelineGraph`, WebGPU render passes, single-pass `SurfaceWithEdges`, `R32Float` IGN volume raymarching, Eye-Dome Lighting (EDL), and GPU Compute Marching Cubes.
- [**Rust API Guide (& Python Bindings)**](rust-api.md) — Using `ParaViz`, `DisplayHandle`, `PipelineGraph` & `DataSet` construction, direct `WebGpuRenderer` / `GpuContourEngine`, JSON state persistence, Rust Trace export, and `paraviz-py`.
- [**Sources & Filters Reference**](filters-and-sources.md) — Complete catalog of all 14 procedural sources, all 37 `FilterConfig` filters, multi-input filters, `CalculatorFilter` expression syntax, and analytical helpers.

---

## Building API Documentation (Rustdoc)

Generate and open the full crate API documentation locally:

```bash
cargo doc --workspace --exclude paraviz-app --no-deps --open
```

---

## Encountering an Issue?

If you run into a bug, unexpected error, rendering glitch, or missing capability, please **[open an issue](https://github.com/DimitriTimoz/paraviz/issues/new/choose)** with your OS/GPU details and steps to reproduce (or a **Rust Trace** / `.json` pipeline state).
