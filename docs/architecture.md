# Architecture & Rendering Pipeline

ParaViz mirrors Kitware ParaView / VTK's modular architecture in safe, parallel Rust (`edition = "2024"`) paired with WebGPU (`wgpu 30.0`).

## Workspace Crates

| Crate | ParaView / VTK Equivalent | Core Responsibilities |
| --- | --- | --- |
| `paraviz-core` | `vtkCommonDataModel`, `vtkCommonExecutionModel` | Typed multi-component `DataArray` (`F32`, `F64`, `I8`–`U64`), `FieldData`, `CellType` (linear & quadratic VTK cells), `PolyData`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`, `MultiBlockDataSet`, `ColorMap` (11 presets + log scale + NaN color), `TransferFunction`, and demand-driven `PipelineGraph` |
| `paraviz-sources` | `vtkFiltersSources` | 14 procedural geometric & analytical `Source` generators |
| `paraviz-filters` | `vtkFiltersCore`, `vtkFiltersGeneral`, `vtkFiltersFlowPaths` | 36 parallel (`rayon`) geometric, volumetric, flow, resampling, decimation, triangulation, and temporal filters |
| `paraviz-io` | `vtkIOXML`, `vtkIOLegacy`, `vtkIOGeometry`, medical/CFD readers | Legacy `.vtk`, VTK XML (serial & partitioned), `.pvd` time series, `.stl`, `.obj`, `.ply`, `.csv`/`.tsv`, `.glb`, `.raw`, NIfTI-1, NRRD, MetaImage, EnSight Gold, XDMF, and OpenFOAM |
| `paraviz-render` | `vtkRenderingOpenGL2`, `vtkRenderingVolume` | `WebGpuRenderer` (WGSL shaders, 3-point Light Kit, `Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`, 4x MSAA, EDL, Axes Gizmo, 3D Widgets), `pick_scene`, and `GpuContourEngine` |
| `paraviz` (root) | `paraview.simple` | High-level `ParaViz` session API, `DisplayHandle`, `SourceConfig`, `FilterConfig`, `DemoPreset`, JSON state persistence, and `to_rust_trace()` |
| `paraviz-app` | ParaView Desktop & `pvbatch` | `eframe`/`egui` + `egui-wgpu` interactive GUI and headless CLI (`demo`, `render`, `info`, `convert`) |

---

## Demand-Driven Pipeline (`PipelineGraph`)

Equivalent to `vtkAlgorithm` and `vtkExecutive`, `paraviz_core::PipelineGraph` models a directed acyclic graph (DAG) of zero-input `Source` nodes, single-input `Filter` nodes, and `MultiInputFilter` nodes:

```text
  Source / Reader  ──▶  Filter A  ──▶  Filter B  ──▶  RenderScene Actor
                            │
                            └──▶  Filter C  ──▶  RenderScene Actor
```

1. **Metadata Pass (`request_information`)**: Sources advertise `PipelineInfo` (time steps, current time, optional bounds) without allocating heavy buffers.
2. **Dirty Propagation**: Calling `update_source` or `update_filter` (or changing `ParaViz::set_time`) invalidates the modified node and all downstream dependents.
3. **Pull Evaluation & Caching (`update`)**: Calling `graph.update(id, time)` or `pv.evaluate(id)` traverses upstream dependencies recursively, executes only dirty nodes, and caches `Arc<DataSet>` outputs so shared upstream nodes evaluate once per frame/timestep.
4. **Zero-Copy In-Memory Path**: `Source::execute_arc` avoids cloning large in-memory datasets such as `InMemorySource`.

---

## WebGPU Rendering Engine (`paraviz-render`)

`WebGpuRenderer` accepts a `RenderScene` (camera, `Vec<SceneActor>`, `SceneSettings`, `Vec<ViewportWidget>`, and optional `selection_highlight`) and renders either to a live `egui-wgpu` surface or headlessly to an `Rgba8Unorm` buffer / PNG.

### Supported Representations (`Representation`)

| Representation | GPU Technique |
| --- | --- |
| `Surface` | Triangle mesh with vertex/cell scalar LUT lookup, 3-point Light Kit (key, fill, back), configurable ambient/specular/shininess, and two-sided shading |
| `SurfaceWithEdges` | **Single-pass barycentric wireframe**: per-triangle barycentric coordinates + WGSL `fwidth()` screen-space derivatives draw crisp anti-aliased edges directly in the fragment shader with zero Z-fighting |
| `Wireframe` | Line-list rendering for cell boundaries and polylines (`LineWidth` configurable) |
| `Points` | GPU-instanced screen-aligned quads shaded as analytical **sphere impostors** with depth and normal lighting |
| `PointGaussian` | Soft Gaussian splat impostors for point clouds |
| `Volume` | 3D texture raymarching on `ImageData` using full `R32Float` precision, manual trilinear interpolation in WGSL (no optional `float32-filterable` GPU feature required), Interleaved Gradient Noise (IGN) ray jittering to eliminate wood-grain banding, central-difference gradient Phong shading, 1D RGBA `TransferFunction` lookup, early ray termination, and optional `volume_clip_plane` |
| `Outline` | Axis-aligned bounding box wireframe (`prepare_outline`) |

### Post-Processing, Overlays & GPU Compute

- **4x MSAA**: Configurable via `SceneSettings::msaa_samples` (`1` or `4`).
- **Eye-Dome Lighting (EDL)**: Depth-buffer neighborhood contrast post-process (`SceneSettings::edl_enabled`, `edl_strength`) that accentuates silhouettes, crevices, and dense meshes/point clouds.
- **Viewport Overlays**: Corner 3D Orientation Axes Gizmo (`show_axes_gizmo`), world bounding box (`show_bounding_box`), 3D Cube Axes Grid (`show_axes_grid`), interactive `ViewportWidget` overlays (`Plane`, `Line`, `Box`, `Sphere`), and selection highlight reticle (`selection_highlight`).
- **GPU Buffer Caching**: Actor geometry, scalar textures, and volume textures are cached across frames using `Arc` pointer/fingerprint checks and `fxhash` keys (`6 ms` cached 1280x720 4x MSAA + EDL frames observed in `benchmark_pipeline`).
- **WebGPU Compute Marching Cubes (`GpuContourEngine`)**: Extracts `ImageData` isosurfaces directly on the GPU via WGSL compute shaders (`contour_imagedata`), sharing the active `wgpu::Device`/`wgpu::Queue` or running headlessly.
- **3D Raycast Picking (`pick_scene`)**: Casts a ray from viewport pixel `(px, py)` against cached actor pick triangle meshes, returning `PickResult` with `actor_id`, `world_pos`, `distance`, nearest `point_index`, hit `cell_index`, and `scalar_value`.
