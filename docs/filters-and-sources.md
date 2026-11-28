# Sources & Filters — Complete Parameter Reference

Every procedural source (`paraviz-sources`) and visualization filter (`paraviz-filters`) in ParaViz exposes serializable configuration parameters (`SourceConfig` and `FilterConfig`). Every parameter below is documented with its exact field name, Rust type, default value, valid/clamped range, algorithmic behavior, and emitted dataset/arrays.

> **Related Documentation**
> - [End-User Manual](user-guide.md) — Interactive workflows, tuning advice, and shortcuts
> - [Rendering, Style & Scene Parameters](rendering-and-scene.md) — `ActorStyle`, colormaps, volume transfer functions, `Camera`, `SceneSettings`, `ScalarBarConfig`, and annotations
> - [Studio GUI & CLI Reference](gui-and-cli.md) — Dock panels, CLI flags, and `--filter` specifications

---

## Table of Contents

- [1. Procedural Sources (`paraviz-sources`)](#1-procedural-sources-paraviz-sources)
  - [Geometric Sources (`PolyData`)](#geometric-sources-polydata)
  - [Analytical & Volumetric Sources](#analytical--volumetric-sources)
- [2. Extraction, Isosurfaces & Slicing Filters](#2-extraction-isosurfaces--slicing-filters)
- [3. Flow, Vectors & Deformation Filters](#3-flow-vectors--deformation-filters)
- [4. Mesh Processing & Geometry Filters](#4-mesh-processing--geometry-filters)
- [5. Fields, Attributes, Probing & Resampling Filters](#5-fields-attributes-probing--resampling-filters)
- [6. Multi-Input Filters, GPU Compute Contour & Helpers](#6-multi-input-filters-gpu-compute-contour--helpers)
- [7. `CalculatorFilter` Expression Grammar & Built-Ins](#7-calculatorfilter-expression-grammar--built-ins)

---

## 1. Procedural Sources (`paraviz-sources`)

All 14 sources implement `paraviz_core::Source` and serialize cleanly in `SourceConfig`.

### Geometric Sources (`PolyData`)

#### `SphereSource` (`SourceConfig::Sphere`)
Generates a UV-parameterized 3D sphere (`PolyData`, poles on ±Z as single points, all cells triangles — matching `vtkSphereSource`) with analytical outward point `Normals`.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | World-space center `(x, y, z)` |
| `radius` | `f32` | `0.5` | Sphere radius in world units |
| `theta_resolution` | `usize` | `8` | Longitudinal slices around the polar axis (clamped `>= 3`) |
| `phi_resolution` | `usize` | `8` | Latitudinal rings from pole to pole (clamped `>= 3`) |

- **Output**: `PolyData` (`2 * theta * (phi - 2)` triangles, `2 + theta * (phi - 2)` points) + `Normals` (3-comp `point_data`).

---

#### `CubeSource` (`SourceConfig::Cube`)
Generates an axis-aligned hexahedral box (`PolyData` decomposed into 12 triangles, 24 duplicated vertices for crisp hard face normals).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Center `(x, y, z)` of the box |
| `lengths` | `[f32; 3]` | `[1.0, 1.0, 1.0]` | Full side lengths `[dx, dy, dz]` along X, Y, Z |

- **Output**: `PolyData` (12 triangles) + `Normals` (per-face normal) + `FaceId` (`0..=5` float scalar in `point_data`).

---

#### `ConeSource` (`SourceConfig::Cone`)
Generates a 3D cone oriented along `direction` around `center`.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Midpoint between base center and apex |
| `direction` | `[f32; 3]` | `[1.0, 0.0, 0.0]` | Axis vector pointing toward the apex (normalized; fallback `+X`) |
| `height` | `f32` | `1.0` | Total cone height from base to apex |
| `radius` | `f32` | `0.5` | Base circle radius |
| `resolution` | `usize` | `6` | Base polygon facet count (clamped `>= 3`) |
| `capping` | `bool` | `true` | Closes the circular base with a single n-gon when `true` |

- **Output**: `PolyData` (`resolution` side triangles + optional base n-gon; `resolution + 1` points, matching `vtkConeSource`), no data arrays.

---

#### `CylinderSource` (`SourceConfig::Cylinder`)
Generates a Y-axis-aligned cylinder centered at `center`.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Center `(x, y, z)` |
| `height` | `f32` | `1.0` | Cylinder height along the Y axis |
| `radius` | `f32` | `0.5` | Cylinder radius in X-Z plane |
| `resolution` | `usize` | `6` | Circumferential segments (clamped `>= 3`) |
| `capping` | `bool` | `true` | Closes top and bottom disks with single n-gons when `true` |

- **Output**: `PolyData` (`resolution` side quads + optional cap n-gons; `4 * resolution` points, matching `vtkCylinderSource`) + `Normals` (3-comp) + `TCoords` (2-comp `point_data`).

---

#### `PlaneSource` (`SourceConfig::Plane`)
Generates a subdivided parallelogram spanned by `point1 - origin` and `point2 - origin`.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `origin` | `[f32; 3]` | `[-0.5, -0.5, 0.0]` | Corner `(0, 0)` of the plane |
| `point1` | `[f32; 3]` | `[0.5, -0.5, 0.0]` | Corner `(1, 0)` defining the U axis |
| `point2` | `[f32; 3]` | `[-0.5, 0.5, 0.0]` | Corner `(0, 1)` defining the V axis |
| `x_resolution` | `usize` | `1` | Grid subdivisions along U (clamped `>= 1`) |
| `y_resolution` | `usize` | `1` | Grid subdivisions along V (clamped `>= 1`) |

- **Output**: `PolyData` (`nx * ny` quads, matching `vtkPlaneSource`) + constant point `Normals` + 2-component `TextureCoordinates` (`(u, v)` in `point_data`).

---

#### `LineSource` (`SourceConfig::Line`)
Generates a single subdivided polyline segment between `point1` and `point2`.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `point1` | `[f32; 3]` | `[-0.5, 0.0, 0.0]` | Segment start point |
| `point2` | `[f32; 3]` | `[0.5, 0.0, 0.0]` | Segment end point |
| `resolution` | `usize` | `6` | Number of segments (`resolution + 1` points; clamped `>= 1`) |

- **Output**: `PolyData` (`lines`) + `Texture Coordinates` (2-comp `(t, 0)` in `point_data`, matching `vtkLineSource`).

---

#### `ArrowSource` (`SourceConfig::Arrow`)
Generates a unit-length 3D arrow along `+X` from `(0, 0, 0)` to `(1, 0, 0)` (shaft + tip cone).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `tip_length` | `f32` | `0.35` | Fraction of unit length occupied by tip (`shaft_len = max(1.0 - tip_length, 0.05)`) |
| `tip_radius` | `f32` | `0.10` | Cone base radius of the arrowhead |
| `tip_resolution` | `usize` | `6` | Circumferential tip segments (clamped `>= 4`) |
| `shaft_radius` | `f32` | `0.03` | Cylindrical shaft radius |
| `shaft_resolution` | `usize` | `6` | Circumferential shaft segments (clamped `>= 4`) |

- **Output**: `PolyData` triangles, no data arrays (matching `vtkArrowSource`).

---

#### `DiskSource` (`SourceConfig::Disk`)
Generates an annular or filled disk in the XY plane (`z = 0`) centered at the origin.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `inner_radius` | `f32` | `0.5` | Inner hole radius (clamped `>= 0.0`) |
| `outer_radius` | `f32` | `1.0` | Outer disk radius (clamped `>= inner_radius + 1e-4`) |
| `radial_resolution` | `usize` | `8` | Concentric ring subdivisions (clamped `>= 1`) |
| `circum_resolution` | `usize` | `8` | Angular segments around the disk (clamped `>= 3`) |

- **Output**: `PolyData` (`radial * circum` quads; `(radial + 1) * circum` points, matching `vtkDiskSource`), no data arrays.

---

#### `TorusSource` (`SourceConfig::Torus`)
Generates a parametric torus centered at `center` lying in the XY plane around the Z axis.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Torus center `(x, y, z)` |
| `major_radius` | `f32` | `0.7` | Distance from torus center to tube centerline |
| `minor_radius` | `f32` | `0.25` | Tube cross-section radius |
| `major_resolution` | `usize` | `48` | Segments around the major ring (clamped `>= 6`) |
| `minor_resolution` | `usize` | `24` | Segments around the tube cross-section (clamped `>= 6`) |

- **Output**: `PolyData` + analytical `Normals` + `Harmonic` scalar (`0.5 * sin(2u) + 0.5 * sin(v)`).

---

### Analytical & Volumetric Sources

#### `WaveletSource` (`SourceConfig::Wavelet`)
Evaluates VTK's classic 3D `vtkRTAnalyticSource` formula:
`maximum * exp(-r^2 / (2 * std_dev^2)) + x_mag * sin(x_freq * dx + 2π * time) + y_mag * cos(y_freq * dy) + z_mag * sin(z_freq * dz)`.
Advertises 5 built-in time steps (`[0.0, 0.25, 0.5, 0.75, 1.0]`).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `whole_extent` | `[i32; 6]` | `[-10, 10, -10, 10, -10, 10]` | Inclusive index bounds `[xmin, xmax, ymin, ymax, zmin, zmax]` (`spacing = [1, 1, 1]`) |
| `center` | `[f64; 3]` | `[0.0, 0.0, 0.0]` | Gaussian peak center in grid coordinates |
| `maximum` | `f64` | `255.0` | Gaussian amplitude |
| `std_dev` | `f64` | `0.5` | Normalized Gaussian standard deviation |
| `x_freq`, `y_freq`, `z_freq` | `f64` | `60.0`, `30.0`, `40.0` | Angular frequencies along normalized axes `dx, dy, dz` |
| `x_mag`, `y_mag`, `z_mag` | `f64` | `10.0`, `18.0`, `5.0` | Sinusoidal perturbation amplitudes |

- **Output**: `ImageData` with point scalar array `"RTData"` (`f32`).

---

#### `VectorFieldSource` (`SourceConfig::VectorField`)
Generates a 3D analytical vector field (`"Velocity"`, 3-comp) and scalar magnitude (`"Speed"`, 1-comp) on a uniform `ImageData` grid, with optional time-phase modulation.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `dimensions` | `[usize; 3]` | `[25, 25, 25]` | Grid point counts `[nx, ny, nz]` |
| `origin` | `[f64; 3]` | `[-2.0, -2.0, -2.0]` | World-space lower corner |
| `spacing` | `[f64; 3]` | `[4/24, 4/24, 4/24]` | Cell spacing `[sx, sy, sz]` |
| `kind` | `VectorFieldKind` | `SwirlingVortex` | Variants: `SwirlingVortex`, `Dipole`, `HelicalTorus` |

- **Output**: `ImageData` with active vectors `"Velocity"` and active scalars `"Speed"`.

---

#### `SuperquadricSource` (`SourceConfig::Superquadric`)
Parametric superquadric toroid or ellipsoid (`vtkSuperquadricSource` equivalent).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `toroidal` | `bool` | `true` | `true` produces a super-toroid; `false` produces a super-ellipsoid |
| `thickness` | `f32` | `0.35` | Toroid hole thickness ratio (`clamped 0.05..=0.99`; smaller = thinner ring) |
| `phi_roundness` | `f32` | `0.7` | Exponent shaping latitude/tube profile (`1.0` = circular, `< 1` = boxy, `> 1` = pinched/star) |
| `theta_roundness` | `f32` | `0.7` | Exponent shaping longitude/ring profile |
| `size` | `f32` | `0.6` | Overall scale factor |
| `phi_resolution` | `usize` | `32` | Grid resolution in `phi` (clamped `>= 4`) |
| `theta_resolution` | `usize` | `32` | Grid resolution in `theta` (clamped `>= 4`) |

- **Output**: `PolyData` + `Normals` + `Radial` distance scalar (`point_data`).

---

#### `GyroidSource` (`SourceConfig::Gyroid`)
Evaluates the triply-periodic minimal surface implicit function `sin(fx + 2πt)*cos(fy) + sin(fy)*cos(fz) + sin(fz)*cos(fx + 2πt)` on a regular 3D `ImageData` grid.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `dimensions` | `[usize; 3]` | `[32, 32, 32]` | Grid point counts per axis (clamped `>= 2`) |
| `bounds_min` | `[f64; 3]` | `[-π, -π, -π]` | Minimum corner `(x, y, z)` |
| `bounds_max` | `[f64; 3]` | `[π, π, π]` | Maximum corner `(x, y, z)` |
| `frequency` | `f64` | `1.5` | Spatial wavenumber multiplier |

- **Output**: `ImageData` with scalar `"Gyroid"` (`f32`). Contour at `0.0` for the classic minimal surface.

---

#### `PointCloudSource` (`SourceConfig::PointCloud`)
Generates a Fibonacci-spiral point cloud with vertex cells.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `center` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Cloud center `(x, y, z)` |
| `radius` | `f32` | `1.0` | Outer sphere radius |
| `num_points` | `usize` | `1000` | Number of points (clamped `>= 1`) |
| `volumetric` | `bool` | `true` | `true` fills the ball volume (`r ∝ t^(1/3)`); `false` distributes points on the sphere shell |

- **Output**: `PolyData` (`verts`) + `Radius` scalar (`point_data`).

---

## 2. Extraction, Isosurfaces & Slicing Filters

#### `ExtractSurface` (`FilterConfig::ExtractSurface`)
Extracts outer boundary faces of volumetric grids (`ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`) using hidden internal face culling; passes `PolyData` through and flattens `MultiBlock`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `compute_normals` | `bool` | `true` | Generates smooth vertex `Normals` if the input lacks an active normals array |

---

#### `ContourFilter` (`FilterConfig::Contour`)
Parallel 256-case Marching Cubes (structured grids) and Marching Tetrahedra (`UnstructuredGrid`) extracting polygonal isosurfaces.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `field_name` | `String` | `""` | Point scalar array name; empty string falls back to `point_data.active_scalars` (or first array) |
| `iso_values` | `Vec<f64>` | `[0.0]` | List of iso-levels to extract and merge |
| `compute_normals` | `bool` | `true` | Computes smooth gradient/mesh normals |
| `merge_vertices` | `bool` | `true` | Canonical edge-key vertex welding (~5–6× lower memory and smooth shading) |

---

#### `SliceFilter` (`FilterConfig::Slice`)
Intersects any dataset (`ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`, `PolyData`, `MultiBlock`) with an infinite plane defined by `(origin, normal)`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `origin` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Point lying on the cut plane |
| `normal` | `[f32; 3]` | `[0.0, 0.0, 1.0]` | Plane normal vector (normalized internally; fallback `+Z`) |
| `crinkle` | `bool` | `false` | `false`: smooth interpolated cut surface; `true`: extracts whole uncut cells/faces straddling the plane |

---

#### `ClipFilter` (`FilterConfig::Clip`)
Clips geometry smoothly (or crinkle-clipped) against an analytical `ImplicitFunction` `f(p)`. Outputs `PolyData` for `PolyData` inputs and `UnstructuredGrid` for volumetric grids.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `function` | `ImplicitFunction` | `Plane { origin: [0,0,0], normal: [1,0,0] }` | Cutting geometry (`Plane`, `Sphere`, or `Box`) |
| `invert` | `bool` | `false` | `false` keeps `f(p) >= 0` (outside sphere/box, `+normal` side); `true` keeps `f(p) <= 0` |
| `crinkle` | `bool` | `false` | Keeps whole uncut cells intersecting the boundary instead of splitting cells |

**`ImplicitFunction` Variants**:
- `Plane { origin: [f32; 3], normal: [f32; 3] }` — Signed distance `(p - origin) · normal.normalize()`.
- `Sphere { center: [f32; 3], radius: f32 }` — Signed distance `|p - center| - radius`.
- `Box { min: [f32; 3], max: [f32; 3] }` — Axis-aligned box signed distance (`min`/`max` are component-wise ordered automatically).

---

#### `ThresholdFilter` (`FilterConfig::Threshold`)
Retains cells matching `[min, max]` on scalar value (or vector magnitude). Preserves `PolyData` vs `UnstructuredGrid`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `field_name` | `String` | `""` | Array name; empty falls back to active scalars of `association` |
| `association` | `AttributeAssociation` | `Point` | `Point`, `Cell`, or `Field` |
| `min` | `f64` | `0.0` | Lower inclusive scalar bound |
| `max` | `f64` | `1.0` | Upper inclusive scalar bound |
| `invert` | `bool` | `false` | Keeps cells **outside** `[min, max]` when `true` |
| `all_scalars` | `bool` | `true` | For `Point` arrays: `true` requires all cell vertices in range; `false` requires at least one vertex in range |

---

#### `IsoVolumeFilter` (`FilterConfig::IsoVolume`)
Smoothly clips a volumetric dataset to the scalar band `[min_value, max_value]` (equivalent to two scalar clips), producing smooth isosurface caps (`UnstructuredGrid`).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `field_name` | `String` | `""` | Point scalar array name (empty = active scalars / first array) |
| `min_value` | `f64` | `0.0` | Minimum scalar threshold |
| `max_value` | `f64` | `1.0` | Maximum scalar threshold (`new()` swaps automatically if `min > max`) |

---

#### `ExtractBlockFilter` (`FilterConfig::ExtractBlock`)
Selects subset blocks from a `MultiBlockDataSet`. Non-`MultiBlock` inputs pass through unchanged.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `block_indices` | `Vec<usize>` | `[]` | 0-based top-level block indices to retain (`[]` matches all indices) |
| `name_filter` | `String` | `""` | Case-insensitive substring filter on block names (`""` matches all names) |
| `unwrap_single` | `bool` | `false` *(constructors `by_index`/`by_name` set `true`)* | When exactly 1 block matches, unwraps it from `MultiBlockDataSet` directly |

---

## 3. Flow, Vectors & Deformation Filters

#### `StreamTracerFilter` (`FilterConfig::StreamTracer`)
Integrates 3D streamlines through point-data or cell-data vector fields (auto-converting cell vectors via `CellDataToPointDataFilter` if needed). Outputs `PolyData` polylines with `IntegrationTime`, `ArcLength`, `Speed`, and all interpolated arrays.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `vector_field` | `String` | `""` | Vector array name (empty = `active_vectors`) |
| `seeds` | `SeedSource` | `Line { p1: [-1,0,0], p2: [1,0,0], count: 10 }` | Seed geometry: `Line { p1, p2, count }`, `Sphere { center, radius, count }` (Fibonacci), or `Points(Vec<[f32; 3]>)` |
| `direction` | `IntegrationDirection` | `Both` | `Forward`, `Backward`, or `Both` |
| `integrator` | `Integrator` | `Rk4` | `Rk4` (fixed-step 4th order) or `Rk45` (adaptive Dormand-Prince 5(4)) |
| `step_size` | `f32` | `0.05` | Initial / fixed integration step size in world units |
| `max_steps` | `usize` | `500` | Maximum integration steps per direction |
| `max_length` | `f32` | `10.0` | Maximum arc length per streamline propagation |
| `terminal_speed` | `f32` | `1e-6` | Stops integration when `|v| < terminal_speed` |

---

#### `TubeFilter` (`FilterConfig::Tube`)
Extrudes `PolyData` polylines into cylindrical 3D tubes using twist-free parallel-transport (Bishop) frames.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `radius` | `f32` | `0.02` | Base tube radius (minimum radius when scalar modulation is active) |
| `vary_radius_by_scalar` | `Option<String>` | `None` | Point scalar array name modulating radius linearly from `radius` to `radius * radius_factor` |
| `radius_factor` | `f32` | `3.0` | Maximum multiplier when `vary_radius_by_scalar` is `Some` |
| `sides` | `usize` | `12` | Cross-section polygon sides (clamped `>= 3`) |
| `capping` | `bool` | `true` | Caps the start and end faces of each polyline tube |

---

#### `GlyphFilter` (`FilterConfig::Glyph`)
Places oriented and scaled 3D glyph meshes at uniformly strided dataset points.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `shape` | `GlyphShape` | `Arrow` | `Arrow`, `Cone`, `Sphere`, `Cube` |
| `vector_field` | `Option<String>` | `None` | Vector array orienting glyph `+X` axis (`None` = `active_vectors`) |
| `scalar_field` | `Option<String>` | `None` | Scalar array used when `scale_mode == ScaleMode::Scalar` (`None` = `active_scalars`) |
| `scale_mode` | `ScaleMode` | `VectorMagnitude` | `Scalar`, `VectorMagnitude`, or `Constant` |
| `scale_factor` | `f32` | `0.25` | Uniform multiplier applied after mode scaling |
| `max_points` | `usize` | `1000` | Maximum points sampled (`0` = all points) |

---

#### `WarpByVectorFilter` (`FilterConfig::WarpByVector`)
Displaces every point `p' = p + scale_factor * v`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `vector_field` | `String` | `""` | Point vector array name (empty = `active_vectors`) |
| `scale_factor` | `f32` | `1.0` | Displacement multiplier |

---

#### `WarpByScalarFilter` (`FilterConfig::WarpByScalar`)
Displaces every point `p' = p + scale_factor * scalar * n`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `scalar_field` | `String` | `""` | Point scalar array name (empty = `active_scalars`) |
| `normal` | `[f32; 3]` | `[0.0, 0.0, 1.0]` | Fallback displacement direction |
| `use_normal` | `bool` | `true` | Uses per-vertex `point_data` active normals when present instead of fixed `normal` |
| `scale_factor` | `f32` | `1.0` | Displacement multiplier |

---

## 4. Mesh Processing & Geometry Filters

#### `SmoothFilter` (`FilterConfig::Smooth`)
Smooths `PolyData` vertex positions while optionally pinning open boundaries.

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `method` | `SmoothingMethod` | `Taubin` | `Laplacian` (shrinks slightly) or `Taubin` (λ\|μ volume-preserving pass-band smoothing) |
| `iterations` | `usize` | `20` | Smoothing passes |
| `relaxation_factor` | `f32` | `0.5` | Step relaxation λ |
| `pass_band` | `f32` | `0.1` | Target pass-band cutoff for Taubin μ computation |
| `preserve_boundary` | `bool` | `true` | Keeps boundary edge vertices fixed |

---

#### `DecimateFilter` (`FilterConfig::Decimate`)
Garland-Heckbert Quadric Error Metric (QEM) triangle mesh simplification (auto-extracts surfaces on volumetric inputs).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `target_reduction` | `f32` | `0.5` | Fraction of triangles to remove (`0.0` = none, `0.5` = 50%, `0.9` = keep 10%; `new()` clamps `0.0..=0.99`) |
| `preserve_boundaries` | `bool` | `true` | Adds boundary constraint planes to prevent open edges collapsing |

---

#### `Delaunay2DFilter` (`FilterConfig::Delaunay2D`)
Bowyer-Watson 2D triangulation projected onto `(x, y)` while keeping original `z` and `point_data`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `alpha` | `f32` | `0.0` | Circumradius cutoff (`<= 0.0` keeps all convex-hull triangles; `> 0.0` prunes large exterior triangles) |

---

#### `Delaunay3DFilter` (`FilterConfig::Delaunay3D`)
Bowyer-Watson 3D tetrahedralization producing an `UnstructuredGrid` of `CellType::Tetra`. Requires at least 4 points.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `alpha` | `f32` | `0.0` | Circumsphere radius cutoff (`<= 0.0` keeps all tetrahedra) |

---

#### `TriangulateFilter` (`FilterConfig::Triangulate`)
Splits polygons (`quad`, `n-gon`) and `strips` into explicit triangles while remapping `cell_data`. Auto-extracts surface if input is volumetric.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `pass_verts_and_lines` | `bool` | `true` | Retains 0D `verts` and 1D `lines` alongside triangulated `polys` |

---

#### `CleanPolyDataFilter` (`FilterConfig::CleanPolyData`)
Spatial-hash point welding, degenerate cell removal, and vertex compaction on `PolyData`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `tolerance` | `f32` | `1e-5` | Spatial merge distance in world units (`<= 0.0` merges exact coincident vertices) |

---

#### `GenerateNormalsFilter` (`FilterConfig::GenerateNormals`)
Computes area-weighted vertex `Normals` on `PolyData`, splitting vertices across sharp dihedral edges.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `feature_angle_deg` | `f32` | `30.0` | Dihedral angle threshold in degrees above which edges split normals |
| `splitting` | `bool` | `true` | Duplicates vertices along sharp edges for crisp faceted creases |
| `flip_normals` | `bool` | `false` | Reverses polygon winding and negates resulting normals |
| `consistency` | `bool` | `true` | Enforces consistent winding across connected manifold components |

---

#### `FeatureEdgesFilter` (`FilterConfig::FeatureEdges`)
Extracts open boundary, non-manifold, and sharp dihedral edges as `PolyData` line segments.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `boundary_edges` | `bool` | `true` | Emit edges adjacent to only 1 face |
| `feature_edges` | `bool` | `true` | Emit edges whose adjacent faces exceed `feature_angle_deg` |
| `non_manifold_edges` | `bool` | `true` | Emit edges shared by 3 or more faces |
| `feature_angle_deg` | `f32` | `30.0` | Sharp dihedral threshold in degrees |

---

#### `ExtractEdgesFilter` (`FilterConfig::ExtractEdges`)
Extracts all unique topological edges of any dataset's cells as `PolyData` lines, preserving `point_data` and `field_data`. **No configurable parameters** (`ExtractEdgesFilter`).

---

#### `ConnectivityFilter` (`FilterConfig::Connectivity`)
Assigns a `RegionId` point & cell scalar array to every connected topological component (`PolyData` or `UnstructuredGrid`).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `extract_largest` | `bool` | `false` | Drops all components except the one with the largest cell count when `true` |

---

#### `ShrinkFilter` (`FilterConfig::Shrink`)
Scales each cell toward its geometric centroid (disconnecting shared vertices so individual cells are visible).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `shrink_factor` | `f32` | `0.5` | Clamped `0.0..=1.0` (`0.0` = point centroid, `1.0` = original size) |

---

#### `CellCentersFilter` (`FilterConfig::CellCenters`)
Computes every cell's geometric centroid as points in a `PolyData` cloud and transfers `cell_data` into `point_data`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `vertex_cells` | `bool` | `false` *(GUI defaults `true` / `Points` repr)* | Emits `PolyData::verts` indices for each centroid when `true` |

---

#### `ReflectFilter` (`FilterConfig::Reflect`)
Mirrors a dataset across a bounding-box plane, flipping face winding and reflecting `Normals` and `active_vectors`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `plane` | `ReflectPlane` | `XMin` | `XMin`, `XMax`, `YMin`, `YMax`, `ZMin`, `ZMax`, `CenterX`, `CenterY`, `CenterZ` |
| `copy_input` | `bool` | `true` | Appends original and mirrored geometry when `true`; outputs only reflection when `false` |

---

#### `TransformFilter` (`FilterConfig::Transform`)
Applies affine scaling, XYZ Euler rotation, and translation (`T * R_xyz * S`) to coordinates, `Normals`, and `active_vectors`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `translation` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | World-space translation `[dx, dy, dz]` |
| `rotation_deg` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | Euler angles in degrees `[rx, ry, rz]` (`EulerRot::XYZ`) |
| `scale` | `[f32; 3]` | `[1.0, 1.0, 1.0]` | Per-axis scale multipliers `[sx, sy, sz]` |

---

## 5. Fields, Attributes, Probing & Resampling Filters

#### `GradientFilter` (`FilterConfig::Gradient`)
Computes spatial derivatives on structured and unstructured datasets (finite-difference on structured grids, least-squares cell/node gradients on unstructured grids).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `field_name` | `String` | `""` | Point scalar or 3-comp vector array name (empty = active scalars/vectors) |
| `compute_gradient` | `bool` | `true` | Emits 3-comp gradient (for scalar) or 9-comp Jacobian (for vector) |
| `compute_divergence` | `bool` | `false` | Vector input only: scalar trace `∇ · v` |
| `compute_vorticity` | `bool` | `false` | Vector input only: 3-comp curl `∇ × v` |
| `compute_q_criterion` | `bool` | `false` | Vector input only: Hunt's `Q = 0.5 * (||Ω||^2 - ||S||^2)` vortex indicator |
| `compute_lambda2` | `bool` | `false` | Vector input only: Jeong-Hussain middle eigenvalue of `S^2 + Ω^2` |
| `gradient_name` | `String` | `"Gradient"` | Output array name for gradient |
| `divergence_name` | `String` | `"Divergence"` | Output array name for divergence |
| `vorticity_name` | `String` | `"Vorticity"` | Output array name for vorticity |
| `q_criterion_name` | `String` | `"QCriterion"` | Output array name for Q-criterion |
| `lambda2_name` | `String` | `"Lambda2"` | Output array name for Lambda2 |

---

#### `CalculatorFilter` (`FilterConfig::Calculator`)
Evaluates mathematical expressions in parallel with `rayon` over `point_data` or `cell_data` (see [Expression Grammar](#7-calculatorfilter-expression-grammar--built-ins)).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `expression` | `String` | `"coordsX"` | Mathematical expression string |
| `result_name` | `String` | `"Result"` | Name of the added output array |
| `association` | `AttributeAssociation` | `Point` | Target table: `Point` or `Cell` (for `Cell`, `coords*` uses cell centroids) |

---

#### `ElevationFilter` (`FilterConfig::Elevation`)
Projects every point linearly onto `low_point -> high_point`, clamped to `[0.0, 1.0]`. Errors if points are coincident.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `low_point` | `[f32; 3]` | `[0.0, 0.0, 0.0]` | World coordinate mapping to `0.0` |
| `high_point` | `[f32; 3]` | `[0.0, 0.0, 1.0]` | World coordinate mapping to `1.0` |
| `scalar_name` | `String` | `"Elevation"` | Output `point_data` array name |

---

#### `CellDataToPointDataFilter` (`FilterConfig::CellDataToPointData`)
Averages `cell_data` arrays from incident cells onto each vertex and writes them to `point_data`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `pass_cell_data` | `bool` | `false` | Preserves original `cell_data` arrays in output when `true`; clears them when `false` |

---

#### `PointDataToCellDataFilter` (`FilterConfig::PointDataToCellData`)
Averages vertex `point_data` arrays onto each cell and writes them to `cell_data`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `pass_point_data` | `bool` | `false` | Preserves original `point_data` arrays in output when `true`; clears them when `false` |

---

#### `ProbeOverLineFilter` (`FilterConfig::ProbeOverLine`)
Samples all dataset point fields along `[point1, point2]` into a `PolyData` polyline with `ArcLength` and `vtkValidPointMask` (triggers automatic **XY Plot (ArcLength)** in the Chart panel).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `point1` | `[f32; 3]` | `[-1.0, 0.0, 0.0]` | Segment start point |
| `point2` | `[f32; 3]` | `[1.0, 0.0, 0.0]` | Segment end point |
| `resolution` | `usize` | `100` | Number of line segments (`resolution + 1` sample points; clamped `>= 1`) |

---

#### `ResampleToImageFilter` (`FilterConfig::ResampleToImage`)
Samples any volumetric dataset onto a uniform `ImageData` grid (enabling GPU Volume Raymarching and GPU Compute Contour on unstructured meshes).

| Parameter | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `dimensions` | `[usize; 3]` | `[64, 64, 64]` | Sample grid dimensions `[nx, ny, nz]` (clamped `>= 2` per axis) |
| `custom_bounds` | `Option<[f64; 6]>` | `None` | Explicit `[xmin, xmax, ymin, ymax, zmin, zmax]`; `None` uses input bounds |

- **Output**: `ImageData` with all resampled `point_data` arrays plus `"vtkValidPointMask"` (`u8`: `1` inside domain, `0` outside).

---

#### `IntegrateVariablesFilter` (`FilterConfig::IntegrateVariables`)
Integrates geometry measure (`Length`, `Area`, or `Volume`) and all `point_data`/`cell_data` arrays over the dataset, emitting a single centroid vertex `PolyData`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `divide_cell_data_by_volume` | `bool` | `false` | Divides integrated field sums by total measure (`Length`/`Area`/`Volume`) to compute spatial averages |

---

#### `AppendDatasetsFilter` (`FilterConfig::AppendDatasets`)
Merges `PolyData` inputs into `PolyData`, or mixed/volumetric datasets into `UnstructuredGrid` (preserves common `point_data`/`cell_data`).

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `additional_inputs` | `Vec<DataSet>` | `[]` | Extra in-memory datasets appended alongside pipeline input(s) |

---

#### `TemporalStatisticsFilter` (`FilterConfig::TemporalStatistics`)
Aggregates vertex statistics across every block/time step of a `MultiBlockDataSet` into arrays suffixed `_average`, `_minimum`, `_maximum`, `_stddev`. Each component uses only finite values from steps with a matching array and shape. A component with no observations is NaN.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `compute_average` | `bool` | `true` | Emit `<name>_average` |
| `compute_minimum` | `bool` | `true` | Emit `<name>_minimum` |
| `compute_maximum` | `bool` | `true` | Emit `<name>_maximum` |
| `compute_stddev` | `bool` | `true` | Emit `<name>_stddev` (population standard deviation) |

---

#### `AddTimeStepsFilter` (`FilterConfig::AddTimeSteps`)
Wraps a static dataset into a `num_steps`-block `MultiBlockDataSet` (`step_0`, `step_1`, ...) stamped with `TimeValue` in `field_data`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `num_steps` | `usize` | `10` | Number of steps generated (clamped `>= 1`) |
| `start_time` | `f64` | `0.0` | Initial timestamp |
| `end_time` | `f64` | `1.0` | Final timestamp |

---

## 6. Multi-Input Filters, GPU Compute Contour & Helpers

- **`StreamTracerWithCustomSource`** (`pv.stream_tracer_with_custom_source(field_node, seed_node, config)`): Takes `base: StreamTracerFilter`; traces through `inputs[0]` (point or cell vectors) starting from every point of `inputs[1]`.
- **`ResampleWithDatasetFilter`** (`pv.resample_with_dataset(source_node, dest_node)`): No parameters (`ResampleWithDatasetFilter`); interpolates all `inputs[0]` fields at every vertex of `inputs[1]`, adding `"vtkValidPointMask"`.
- **`GpuContourEngine`** (`pv.gpu_contour(label, input, field_name_opt, iso_value)`): Executes WebGPU Marching Cubes directly on `ImageData` (`field_name` empty/None selects `active_scalars` or first scalar array; emits welded `PolyData` with gradient `Normals`).
- **`compute_histogram(dataset, array_name, association, component: Option<usize>, bins: usize)`**: Returns `HistogramResult { bin_edges, counts, min, max, mean, std_dev }`. Passing `component: None` evaluates Euclidean magnitude on vectors and scalar value on 1-component arrays; `bins == 0` returns `PipelineError::ExecutionFailed`.
- **`interpolate_datasets(a, b, t: f64)`**: Linearly blends matching point coordinates and `point_data`/`cell_data` arrays between two topologically identical datasets at `t in [0.0, 1.0]`.

---

## 7. `CalculatorFilter` Expression Grammar & Built-Ins

Parsed via a bounded recursive-descent parser and evaluated in parallel with `rayon`:

- **Binary Operators**: `+`, `-`, `*`, `/`, `^` (exponentiation; standard precedence `^` > `* /` > `+ -`)
- **Unary Operators**: `+`, `-`
- **Grouping**: `(` ... `)`
- **Built-in Variables**:
  - `coordsX`, `coordsY`, `coordsZ` — Individual vertex coordinates (or cell centroids for `AttributeAssociation::Cell`)
  - `coords` — 3-component position vector (valid inside `mag(coords)` or vector expressions)
- **Built-in Functions**:
  - 1-arg scalar/vector: `mag(v)`, `sin(x)`, `cos(x)`, `tan(x)`, `sqrt(x)`, `abs(x)`, `exp(x)`, `ln(x)`
  - 2-arg: `min(a, b)`, `max(a, b)`
  - 3-arg: `clamp(x, lo, hi)`
- **Array References**: Any scalar or 3-component vector array name present in the target `point_data` or `cell_data` (component-wise arithmetic supported when operating on vectors).
