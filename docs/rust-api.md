# Rust API Guide

Add `paraviz` to your `Cargo.toml` (or depend on individual sub-crates such as `paraviz-core`, `paraviz-filters`, `paraviz-io`, `paraviz-render`, `paraviz-sources` if you don't need the full facade):

```toml
[dependencies]
paraviz = { path = "." } # or version/git dependency
```

Everything common is re-exported under `paraviz::prelude::*`.

---

## 1. High-Level `ParaViz` Session API

`ParaViz` is the Rust counterpart to `paraview.simple`. It manages a `PipelineGraph`, node display styles (`ActorStyle`), camera, global `SceneSettings`, time steps, state serialization, and lazy headless WebGPU rendering.

### Basic Pipeline & Headless PNG Render

```rust
use paraviz::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pv = ParaViz::new();

    // 1. Add a procedural source (or open a file with pv.open("mesh.vtu")?)
    let wav = pv.add_source("Wavelet", WaveletSource::default());
    pv.hide(wav);

    // 2. Attach a filter
    let iso = pv.add_filter(
        "Contour",
        wav,
        ContourFilter::new("RTData", vec![110.0, 175.0]),
    )?;

    // 3. Configure actor appearance via fluent DisplayHandle
    pv.show(iso)
        .representation(Representation::SurfaceWithEdges)
        .colormap(ColorMapPreset::Viridis)
        .color_by_point("RTData")
        .opacity(1.0);

    // 4. Configure scene settings & camera
    pv.settings.edl_enabled = true;
    pv.settings.msaa_samples = 4;
    pv.camera_preset(CameraPreset::Isometric)?;

    // 5. Render headlessly with WebGPU
    pv.render_png("wavelet.png", 1280, 720)?;
    Ok(())
}
```

### DisplayHandle Reference (`pv.show(node_id)`)

- `.visible(bool)` — Toggle visibility (`pv.hide(node_id)` sets `visible = false`)
- `.representation(Representation)` — `Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`
- `.colormap(ColorMapPreset)` / `.custom_colormap(ColorMap)` — Updates both `style.colormap` and `style.volume_tf.color_map`
- `.color_by_point(array)` / `.color_by_point_component(array, component)` / `.color_by_cell(array)` / `.solid_color([r, g, b, a])`
- `.scalar_range(min, max)` / `.auto_range(bool)` — Manual or auto-fitted scalar bounds (updates `volume_tf.scalar_range` too)
- `.opacity(f32)` *(clamped `0.0..=1.0`)* / `.point_size(f32)` *(min `1.0`)* / `.line_width(f32)` *(min `0.5`)*
- `.edge_color([r, g, b, a])` / `.edge_width(f32)` *(min `0.25`)*
- `.volume_tf(TransferFunction)` / `.volume_step_size(f32)` *(min `0.05`)*
- `.hide_block(path_or_name)` / `.show_block(path_or_name)` — Per-block visibility inside `MultiBlockDataSet`
- `.block_color(path_or_name, [r, g, b, a])` / `.clear_block_color(path_or_name)` — Per-block RGBA color override

For every underlying `ActorStyle`, `SceneSettings`, `Camera`, `ScalarBarConfig`, and `ViewportAnnotation` field, see [Rendering, Style & Scene Parameters](rendering-and-scene.md).

### Session Operations

| Method | Purpose |
| --- | --- |
| `ParaViz::new()` / `ParaViz::headless()` | Create session (lazy vs. eager headless WebGPU initialization) |
| `open(path)` / `open_with_reader(path, reader)` | Open any supported file using auto-detection or an explicit `DatasetReader` (`.pvd` creates `PvdSeriesSource`) |
| `open_preloaded(path, reader, ds, time_steps)` | Register a pre-decoded dataset (e.g., from a background worker) without re-reading disk on first evaluation |
| `open_raw_volume(label, path, &desc)` | Load a raw binary 3D volume using `RawVolumeDescriptor` |
| `add_source(label, src)` / `add_source_cfg(label, cfg)` | Add custom `Source` or serializable `SourceConfig` |
| `add_dataset(label, ds)` | Wrap an in-memory `DataSet` via `InMemorySource` |
| `add_filter(label, input, flt)` / `add_filter_cfg(label, input, cfg)` | Add single-input `Filter` or serializable `FilterConfig` |
| `add_multi_filter(label, inputs, flt)` | Register a `MultiInputFilter` |
| `resample_with_dataset(src, dst)` / `append_nodes(inputs)` | Convenience multi-input pipeline helpers |
| `stream_tracer_with_custom_source(field, seeds, cfg)` | Trace streamlines from a custom seed geometry node (`StreamTracerWithCustomSource`) |
| `update_source_cfg(id, cfg)` / `update_filter_cfg(id, cfg)` | Mutate a configurable node and invalidate downstream caches |
| `seed_evaluated(id, arc_ds, time)` | Populate an already-evaluated dataset into the pipeline cache (for background worker integration) |
| `gpu_contour(label, input, field_opt, iso)` | Run WebGPU Compute Marching Cubes (`GpuContourEngine`) on an `ImageData` node |
| `set_time(t)` / `all_time_steps()` | Switch active time step and query source time steps |
| `prepare_temporal_probe(id, request)?.run()` | Sample a point-data component at fixed world coordinates across the selected pipeline’s time steps. Missing samples are `None` and have an empty value in `to_csv()`. |
| `evaluate(id)` / `update_visible()` / `update_all()` | Execute demand-driven pipeline nodes |
| `pick(px, py, width, height)` | Cast a 3D raycast pick query against visible actors (`PickResult`, including `block_path`) |
| `render_rgba(w, h)` / `render_png(path, w, h)` | Render headless `Rgba8Unorm` pixels or PNG (supports `transparent_background`, `show_scalar_bar`, `ScalarBarConfig`, `show_time_annotation`, and `annotations`) |
| `render_turntable_gif(...)` / `render_timeseries_gif(...)` | Render 360-degree turntable or time-series animated GIF89a files |
| `export_dataset(id, path)` / `export_glb(id, path, use_cmap)` / `export_html(id, path, use_cmap)` | Export evaluated node to VTK/mesh/CSV/NIfTI/MHA, binary glTF 2.0, or interactive WebGL2 `.html` |
| `save_state_json()` / `load_state_json(json)` | Save and restore configured pipelines, in-memory data, styles, camera, and `SceneSettings` |
| `to_rust_trace()` | Emit reproducible Rust code for the current session |

State JSON preserves all input links for `append_nodes`, `resample_with_dataset`, and
`stream_tracer_with_custom_source`. Datasets added with `add_dataset` are present even
before evaluation. Custom Rust sources and filters with no serializable configuration
must be evaluated before saving and reload as static datasets. Floating-point arrays
preserve `NaN` and infinities in JSON. Older releases cannot read these non-finite
values or reconstruct the new multi-input state records.


---

## 2. Demo Presets (`DemoPreset`)

Five ready-to-run scientific scenes are built into `DemoPreset`:

- `DemoPreset::WaveletContour` (`"wavelet"`) — Wavelet outline + two `ContourFilter` isosurfaces + translucent `SliceFilter`
- `DemoPreset::VortexStreamlines` (`"vortex"`) — Swirling vortex vector field + RK45 `StreamTracerFilter` + `TubeFilter` + arrow `GlyphFilter`
- `DemoPreset::GyroidMinimalSurface` (`"gyroid"`) — Triply-periodic minimal surface extracted at `iso = 0.0`
- `DemoPreset::VolumeWavelet` (`"volume"`) — Direct 3D WebGPU volume raymarching with a custom `TransferFunction`
- `DemoPreset::SuperquadricClip` (`"superquadric"`) — Superquadric torus clipped by an oblique plane

```rust
let mut pv = DemoPreset::VortexStreamlines.build()?;
pv.render_png("vortex.png", 1024, 768)?;
println!("{}", pv.to_rust_trace());
```

---

## 3. Low-Level `DataSet`, `DataArray`, and `PipelineGraph`

For custom mesh generation or embedding without `ParaViz`:

```rust
use paraviz::core::{
    ArrayData, CellType, DataArray, DataSet, PipelineGraph, PolyData, UnstructuredGrid,
};
use paraviz::filters::{ContourFilter, ExtractSurface};
use paraviz::core::Filter;

// Build a PolyData triangle mesh
let mut poly = PolyData::from_triangles(
    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    vec![[0, 1, 2]],
);
poly.point_data.add_array(DataArray::new(
    "Temperature",
    1,
    ArrayData::F32(vec![300.0, 350.0, 400.0]),
));
poly.point_data.active_scalars = Some("Temperature".into());

// Build an UnstructuredGrid (e.g. a single tetrahedron)
let mut ug = UnstructuredGrid::new();
ug.points = vec![
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
ug.push_cell(CellType::Tetra, &[0, 1, 2, 3]);

// Apply any filter directly
let surface = ExtractSurface::new().apply(&DataSet::UnstructuredGrid(ug))?;
# Ok::<(), paraviz::core::PipelineError>(())
```

### Custom `Source` and `Filter` Implementations

Implement `paraviz_core::Source`, `paraviz_core::Filter`, or `paraviz_core::MultiInputFilter` (all require `Send + Sync`) to plug custom generators or algorithms directly into `PipelineGraph` or `ParaViz::add_source` / `ParaViz::add_filter`.

---

## 4. Direct `WebGpuRenderer` & `GpuContourEngine`

When rendering custom `RenderScene` objects without `ParaViz`:

```rust
use paraviz::prelude::*;
use std::sync::Arc;

let ds = WaveletSource::default().execute(0.0)?;
let img = ds.as_image_data().unwrap();

let mut renderer = WebGpuRenderer::new_headless()?;
let gpu_mc = GpuContourEngine::from_device(
    Arc::new(renderer.device().clone()),
    Arc::new(renderer.queue().clone()),
);
let iso_poly = gpu_mc.contour_imagedata(img, "RTData", 140.0)?;

let mut scene = RenderScene::new();
let mut actor = SceneActor::new(1, "GPU Iso", Arc::new(DataSet::PolyData(iso_poly)));
actor.style.representation = Representation::SurfaceWithEdges;
actor.style.colormap = ColorMap::from_preset(ColorMapPreset::Viridis);
actor.style.color_by = ColorBy::PointScalar {
    array: "RTData".into(),
    component: None,
};
scene.add_actor(actor);
scene.settings.msaa_samples = 4;
scene.settings.edl_enabled = true;
scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

let rgba = renderer.render_to_rgba(&scene, 1280, 720)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

See the runnable examples in [`examples/`](../examples):
- [`examples/wavelet_contour.rs`](../examples/wavelet_contour.rs)
- [`examples/streamlines_and_tubes.rs`](../examples/streamlines_and_tubes.rs)
- [`examples/volume_raymarch.rs`](../examples/volume_raymarch.rs)
- [`examples/benchmark_pipeline.rs`](../examples/benchmark_pipeline.rs)

---

## 5. Python Bindings (`paraviz-py`)

A standalone PyO3 extension crate lives in [`crates/paraviz-py/`](../crates/paraviz-py), exposing `paraviz.Session` with a workflow modeled after `paraview.simple`:

```python
import paraviz

s = paraviz.Session()
wav = s.wavelet("Wavelet")
iso = s.contour("Contour", wav, "RTData", [120.0, 175.0])
s.hide(wav)
s.show(iso, repr="edges", colormap="viridis", color_by="RTData")
s.reset_camera()
s.render_png("wavelet.png", 1280, 720)
print(s.to_rust_trace())
```

Build and install locally with `maturin` inside `crates/paraviz-py`:
```bash
cd crates/paraviz-py && maturin develop --release
```
