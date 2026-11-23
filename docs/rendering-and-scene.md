# Rendering, Appearance & Scene Parameters

Complete reference for every visual styling, colormap, volume transfer function, camera, lighting, viewport widget, scalar color bar, and annotation parameter across `paraviz-core` and `paraviz-render`.

> **Related Documentation**
> - [Sources & Filters Parameter Reference](filters-and-sources.md)
> - [Studio GUI & CLI Reference](gui-and-cli.md)
> - [End-User Manual](user-guide.md)

---

## 1. Per-Actor Display Style (`ActorStyle`)

Every visible or hidden pipeline node owns an `ActorStyle` (configurable in the **Display** inspector tab or via `pv.show(id)` `DisplayHandle` fluent methods in Rust).

| Parameter | Rust Type | Default | Fluent Setter (`DisplayHandle`) | Meaning / Range |
| --- | --- | --- | --- | --- |
| `visible` | `bool` | `true` *(when via `show`)* | `.visible(bool)` | Toggles visibility of the actor in the 3D scene |
| `representation` | `Representation` | `Surface` | `.representation(Representation)` | Draw mode (`Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, `PointGaussian`, `Volume`, `Outline`) |
| `color_by` | `ColorBy` | `Solid([0.82, 0.84, 0.88, 1.0])` | `.color_by_point` / `.color_by_cell` / `.solid_color` | Solid RGBA or array-mapped scalar coloring |
| `colormap` | `ColorMap` | `CoolToWarm` | `.colormap` / `.custom_colormap` | Lookup table, log-scale toggle, NaN color |
| `scalar_range` | `(f64, f64)` | `(0.0, 1.0)` | `.scalar_range(min, max)` | Normalization bounds `(min, max)` mapped to `[0.0, 1.0]` |
| `auto_range` | `bool` | `true` | `.auto_range(bool)` | Auto-fits `scalar_range` from data min/max when `true` |
| `opacity` | `f32` | `1.0` | `.opacity(f32)` | Global alpha multiplier (`0.0..=1.0`; `< 1.0` engages OIT/alpha-blended pass) |
| `point_size` | `f32` | `6.0` | `.point_size(f32)` | Screen/world impostor size in pixels for `Points` and `PointGaussian` |
| `line_width` | `f32` | `1.5` | `.line_width(f32)` | Line thickness in pixels for `Wireframe`, `Outline`, and polylines |
| `edge_color` | `[f32; 4]` | `[0.08, 0.08, 0.10, 1.0]` | `.edge_color([r, g, b, a])` | Overlaid mesh wireframe RGBA in `SurfaceWithEdges` |
| `edge_width` | `f32` | `1.25` | `.edge_width(f32)` | Screen-space barycentric edge thickness in `SurfaceWithEdges` |
| `specular` | `f32` | `0.25` | `node.style.specular` | Blinn-Phong specular reflection intensity (`0.0..=1.0`) |
| `shininess` | `f32` | `32.0` | `node.style.shininess` | Specular highlight exponent (`1.0..=256.0`) |
| `ambient` | `f32` | `0.18` | `node.style.ambient` | Ambient fill lighting contribution (`0.0..=1.0`) |
| `two_sided` | `bool` | `true` | `node.style.two_sided` | Automatically flips back-facing triangle normals toward camera |
| `volume_tf` | `TransferFunction` | `Default` | `.volume_tf(TransferFunction)` | RGB + piecewise-linear opacity transfer function for `Representation::Volume` |
| `volume_step_size` | `f32` | `0.5` | `.volume_step_size(f32)` | Raymarch step multiplier in voxel units (`min 0.05`; smaller = finer quality, higher GPU cost) |
| `volume_shading` | `bool` | `true` | `node.style.volume_shading` | Enables central-difference volumetric gradient Phong shading in raymarcher |
| `volume_clip_plane` | `Option<([f32; 3], [f32; 3])>` | `None` | `node.style.volume_clip_plane` | Live GPU raymarch clipping plane `(origin, normal)` (clips side where `(p - origin) · normal < 0`) |
| `hidden_blocks` | `BTreeSet<String>` | `{}` | `.hide_block` / `.show_block` | Hidden `MultiBlockDataSet` block names or slash-joined paths (e.g. `"wall"`, `"Base/per1"`) |
| `block_colors` | `BTreeMap<String, [f32; 4]>` | `{}` | `.block_color` / `.clear_block_color` | Per-block solid RGBA color overrides keyed by slash path or leaf name (path takes precedence) |

---

### `Representation` Variants

| Variant | Supported Inputs | Shader / GPU Technique |
| --- | --- | --- |
| `Surface` | Any (extracts surface on grids) | Triangle rasterization with Blinn-Phong shading and scalar LUT lookup |
| `SurfaceWithEdges` | Any | Single-pass WGSL barycentric `fwidth()` derivative edge antialiasing (zero Z-fighting) |
| `Wireframe` | Any | Topological cell edges rendered as thick quads/lines |
| `Points` | Any | Raycasted analytical sphere impostors per vertex |
| `PointGaussian` | Any | Radial Gaussian splat billboards with soft alpha falloff |
| `Volume` | `ImageData` | Full-precision `R32Float` 3D texture raymarching with IGN ray jittering, gradient Phong shading, and clip plane |
| `Outline` | Any | Axis-aligned bounding-box wireframe (`12` edges) |

---

### `ColorBy` Variants

| Variant | Fields | Behavior |
| --- | --- | --- |
| `Solid([f32; 4])` | `rgba` | Uniform constant RGBA color (unless overridden per-block via `ActorStyle::block_colors`) |
| `PointScalar { array, component }` | `array: String`, `component: Option<usize>` | Samples `point_data[array]`. `None` uses scalar value or vector L2 magnitude; `Some(c)` picks component index `c` (X=`0`, Y=`1`, Z=`2`) |
| `CellScalar { array, component }` | `array: String`, `component: Option<usize>` | Samples `cell_data[array]` per cell with identical `component` semantics |

---

## 2. Colormaps (`ColorMap` & `ColorMapPreset`)

### `ColorMap` Fields

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `String` | `"Cool to Warm"` | Preset or custom palette name |
| `stops` | `Vec<ColorStop>` | Preset stops | Ordered control points `ColorStop { t: f32, rgb: [f32; 3] }` on `[0.0, 1.0]` |
| `log_scale` | `bool` | `false` | Maps strictly positive scalars (`min > 0, max > 0, val > 0`) logarithmically |
| `nan_color` | `[f32; 4]` | `[0.5, 0.5, 0.0, 1.0]` | Color rendered for `NaN` or non-finite values |

### Built-in `ColorMapPreset` Values (`ColorMapPreset::ALL`)

| Enum Variant | Display Name | Visual Character |
| --- | --- | --- |
| `CoolToWarm` *(default)* | `Cool to Warm` | Diverging blue → neutral grey → red (Kenneth Moreland scientific standard) |
| `Fast` | `Fast (ParaView)` | High-contrast diverging ParaView default colormap |
| `Viridis` | `Viridis` | Perceptually uniform sequential indigo → teal → bright yellow |
| `Plasma` | `Plasma` | Perceptually uniform sequential deep blue → magenta → orange → yellow |
| `Inferno` | `Inferno` | Perceptually uniform black → crimson → amber → pale yellow |
| `Magma` | `Magma` | Perceptually uniform black → violet → coral → off-white |
| `Turbo` | `Turbo` | Smooth rainbow-like perceptually improved spectrum |
| `Rainbow` | `Rainbow (Jet)` | Classic blue → cyan → green → yellow → red |
| `Grayscale` | `Grayscale` | Linear near-black (`0.05`) to near-white (`0.98`) luminance ramp |
| `BlueRedYellow` | `Black-Body Radiation` | Black → red → yellow → white thermal emission ramp |
| `Cividis` | `Cividis` | Colorblind-safe blue → slate → yellow ramp |

---

## 3. Volume Transfer Function (`TransferFunction`)

Used when `ActorStyle::representation == Representation::Volume`. Uploaded to GPU as a combined RGBA8 1D lookup texture.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `color_map` | `ColorMap` | `CoolToWarm` | RGB color curve mapped across `scalar_range` |
| `opacity_points` | `Vec<OpacityPoint>` | `(0.0→0.0), (0.3→0.05), (0.65→0.45), (1.0→0.95)` | Piecewise-linear `(t, alpha)` stops in `[0.0, 1.0]` |
| `scalar_range` | `(f64, f64)` | `(0.0, 1.0)` | Scalar domain mapped to `[0.0, 1.0]` |
| `scalar_opacity_unit_distance` | `f32` | `1.0` | Beer-Lambert optical attenuation unit distance scale |

---

## 4. Camera & Projection (`Camera` & `CameraPreset`)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `Vec3` | `[0.0, 0.0, 4.0]` | World-space camera eye position |
| `target` | `Vec3` | `[0.0, 0.0, 0.0]` | World-space look-at focal center |
| `up` | `Vec3` | `[0.0, 1.0, 0.0]` | World-space up orientation vector |
| `fov_y_deg` | `f32` | `45.0` | Vertical field of view in degrees (perspective mode; clamped in zoom/projection) |
| `near` | `f32` | `0.01` | Near clip plane distance (updated dynamically by `fit_bounds`) |
| `far` | `f32` | `1000.0` | Far clip plane distance |
| `orthographic` | `bool` | `false` | Switches between Perspective (`false`) and Parallel Orthographic (`true`) projection |
| `ortho_scale` | `f32` | `1.5` | Half-height of the orthographic frustum in world units (adjusted by zoom) |

### `CameraPreset` Values
- `PosX`, `NegX`, `PosY`, `NegY`, `PosZ`, `NegZ` — Axis-aligned orthographic-style views around scene bounding-box center
- `Isometric` — Elevated 3D corner view `(1, 0.85, 1).normalize()` looking at scene center

---

## 5. Global Scene, Overlays & Color Bar (`SceneSettings`)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `bg_top` | `[f32; 3]` | `[0.13, 0.16, 0.22]` | Linear RGB color at top of viewport skybox gradient |
| `bg_bottom` | `[f32; 3]` | `[0.27, 0.32, 0.41]` | Linear RGB color at bottom of viewport skybox gradient |
| `show_axes_gizmo` | `bool` | `true` | Draws interactive 3D X/Y/Z orientation triad in corner |
| `show_bounding_box` | `bool` | `false` | Draws wireframe world-space bounding box of visible scene |
| `show_axes_grid` | `bool` | `false` | Draws ParaView-style 3D Cube Axes Grid with numeric tick marks (`X/Y/Z`) around visible scene bounds |
| `transparent_background` | `bool` | `false` | Clears background with `alpha = 0.0` (suppresses skybox) for transparent PNG & GIF exports |
| `show_scalar_bar` | `bool` | `false` | Enables the Scalar Color Bar legend in viewport and PNG/GIF exports |
| `scalar_bar` | `ScalarBarConfig` | `Default` | Controls Scalar Color Bar position, dimensions, orientation, title, and ticks (see below) |
| `show_time_annotation` | `bool` | `false` | Displays a `Time: <value>` badge in the viewport and headless PNG/GIF renders |
| `time_label` | `Option<String>` | `None` | Formatted time text (auto-populated by `ParaViz::build_scene()` from active `time` when `show_time_annotation` is enabled, or set manually) |
| `annotations` | `Vec<ViewportAnnotation>` | `[]` | Custom 2D text labels overlaid onto the viewport and headless exports |
| `edl_enabled` | `bool` | `false` | Enables screen-space Eye-Dome Lighting depth-contrast shading pass |
| `edl_strength` | `f32` | `1.0` | EDL edge-darkening intensity (`>= 0.0`) |
| `msaa_samples` | `u32` | `1` | Anti-aliasing sample count (`1` = off, `4` = 4× MSAA) |

---

### `ScalarBarConfig` Parameters

Controls the interactive (draggable/resizable in GUI) and software-composited (headless PNG/GIF) scalar color bar.

| Parameter | Type | Default | Range / Notes |
| --- | --- | --- | --- |
| `pos` | `Option<[f32; 2]>` | `None` | Normalized `[0.0..=1.0]` top-left viewport coordinates (`None` docks cleanly at bottom-right) |
| `size` | `[f32; 2]` | `[160.0, 220.0]` | Logical card dimensions `[width, height]` at 720p reference scale |
| `custom_title` | `Option<String>` | `None` | Custom title text; `None` displays the active scalar array name (plus `[comp]` if vector component selected) |
| `num_labels` | `usize` | `3` | Number of evenly spaced numeric tick marks along the ramp (**clamped `2..=7`**) |
| `font_scale` | `f32` | `1.0` | Font size multiplier for title and tick labels |
| `horizontal` | `bool` | `false` | `false` = vertical color ramp; `true` = horizontal color ramp |

---

### `ViewportAnnotation` Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `u64` | `0` | Unique identifier for UI editing/drag tracking |
| `text` | `String` | `""` | Label string rendered inside a subtle pill badge |
| `pos` | `[f32; 2]` | `[0.05, 0.05]` | Normalized `[0.0..=1.0]` top-left position inside the viewport |
| `font_size` | `f32` | `16.0` | Logical font pixel height |
| `color` | `[f32; 4]` | `[0.94, 0.96, 1.0, 1.0]` | Text RGBA color `[r, g, b, a]` in `0.0..=1.0` |
| `visible` | `bool` | `true` | Toggles rendering without deleting the annotation |

---

### Studio Background Presets (`ViewportBgPreset` in GUI)

- **`StudioSlate`**: top `[0.11, 0.14, 0.20]`, bottom `[0.22, 0.27, 0.36]`
- **`DeepObsidian`**: top `[0.05, 0.06, 0.09]`, bottom `[0.11, 0.13, 0.18]`
- **`ParaViewClassic`**: top `[0.20, 0.22, 0.30]`, bottom `[0.42, 0.45, 0.54]`
- **`MidnightBlue`**: top `[0.06, 0.10, 0.20]`, bottom `[0.15, 0.24, 0.40]`
- **`ArcticLight`**: top `[0.78, 0.83, 0.90]`, bottom `[0.94, 0.96, 0.99]`

---

## 6. 3D Guides & Picking (`ViewportWidget`, `PickResult`)

### `ViewportWidget` Variants
Rendered when 3D Filter Widgets are enabled (`W` in GUI) or populated in `RenderScene::widgets`:
- `ViewportWidget::Plane { origin: [f32; 3], normal: [f32; 3], size: f32 }`
- `ViewportWidget::Line { p1: [f32; 3], p2: [f32; 3] }`
- `ViewportWidget::Box { min: [f32; 3], max: [f32; 3] }`
- `ViewportWidget::Sphere { center: [f32; 3], radius: f32 }`

### `PickResult` Fields
Returned by `ParaViz::pick` / `pick_scene`:
- `actor_id: u64` — ID of the hit `SceneActor`
- `actor_name: String` — Pipeline node name
- `block_path: Option<String>` — Slash-joined child block path if picking inside a `MultiBlockDataSet` (e.g. `"Base/wall"`), else `None`
- `world_pos: [f32; 3]` — Exact 3D intersection coordinate
- `distance: f32` — Ray distance from camera eye
- `point_index: Option<usize>` — Nearest hit vertex index in the leaf dataset
- `cell_index: Option<usize>` — Hit cell index
- `scalar_value: Option<f32>` — Sampled active scalar value at hit location
