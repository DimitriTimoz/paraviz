# Sources & Filters Reference

## Procedural Sources (`paraviz-sources`)

All 14 sources implement `paraviz_core::Source` and have matching variants in `paraviz::SourceConfig`.

### Geometric Sources (`PolyData`)

| Source | `SourceConfig` Variant | Description |
| --- | --- | --- |
| `SphereSource` | `SourceConfig::Sphere` | UV sphere (`center`, `radius`, `theta_resolution`, `phi_resolution`) with normals & `Elevation` array |
| `CubeSource` | `SourceConfig::Cube` | Axis-aligned box (`center`, `x_length`, `y_length`, `z_length`) |
| `ConeSource` | `SourceConfig::Cone` | Cone (`center`, `direction`, `radius`, `height`, `resolution`, `capping`) |
| `CylinderSource` | `SourceConfig::Cylinder` | Cylinder (`center`, `radius`, `height`, `resolution`, `capping`) |
| `PlaneSource` | `SourceConfig::Plane` | Subdivided quadrilateral (`origin`, `point1`, `point2`, `x_resolution`, `y_resolution`) |
| `LineSource` | `SourceConfig::Line` | Polyline segment (`point1`, `point2`, `resolution`) with normalized `ArcLength` |
| `ArrowSource` | `SourceConfig::Arrow` | 3D shaft + tip arrow mesh |
| `DiskSource` | `SourceConfig::Disk` | Annular or filled disk (`inner_radius`, `outer_radius`, `radial_resolution`, `circumferential_resolution`) |
| `TorusSource` | `SourceConfig::Torus` | Parametric torus (`major_radius`, `minor_radius`, `u_resolution`, `v_resolution`) |

### Analytical & Volumetric Sources

| Source | Output Type | Description |
| --- | --- | --- |
| `WaveletSource` | `ImageData` | VTK `vtkRTAnalyticSource` equivalent producing `RTData` on `whole_extent` (default `[-10..10]^3`) |
| `VectorFieldSource` | `ImageData` | 3D analytical vector fields (`VectorFieldKind`: `SwirlingVortex`, `Dipole`, `HelicalTorus`) emitting `Velocity` (3-comp) and `Speed` |
| `SuperquadricSource` | `PolyData` | Parametric superquadric surface / toroid (`toroidal`, `phi_roundness`, `theta_roundness`, `thickness`, `Radial` scalar) |
| `GyroidSource` | `ImageData` | Triply-periodic minimal surface implicit scalar field `Gyroid` (`sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)`) |
| `PointCloudSource` | `PolyData` | Procedural 3D point cloud with vertex cells and scalar attributes |

---

## Visualization Filters (`paraviz-filters`)

All single-input filters implement `paraviz_core::Filter` (with `rayon` parallelism where applicable) and serialize cleanly via `paraviz::FilterConfig`. Most filters automatically recurse over `MultiBlockDataSet` inputs.

### Extraction, Isosurfaces & Slicing

| Filter (`FilterConfig` Variant) | ParaView Equivalent | Behavior |
| --- | --- | --- |
| `ExtractSurface` (`ExtractSurface`) | Extract Surface | Extracts outer boundary faces of volumetric grids (`ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`) using hidden internal face culling |
| `ContourFilter` (`Contour`) | Contour | 256-case Marching Cubes (on `ImageData` / `RectilinearGrid` / `StructuredGrid`) and Marching Tetrahedra (on `UnstructuredGrid`) with parallel spatial edge-key vertex welding |
| `SliceFilter` (`Slice`) | Slice | Plane intersection (`origin`, `normal`) producing a watertight `PolyData` cross-section with interpolated point/cell attributes |
| `ClipFilter` (`Clip`) | Clip | Clips geometry against an `ImplicitFunction` (`Plane`, `Sphere`, `Box`) with optional `invert` and `crinkle` modes |
| `ThresholdFilter` (`Threshold`) | Threshold | Extracts cells whose scalar falls within `[lower, upper]` into an `UnstructuredGrid` |
| `IsoVolumeFilter` (`IsoVolume`) | Iso Volume | Smoothly clips volumetric cells to the scalar band `[min_value, max_value]` with smooth isosurface boundaries |
| `ExtractBlockFilter` (`ExtractBlock`) | Extract Block | Selects blocks from a `MultiBlockDataSet` by `block_indices` or `name_filter` substring (`unwrap_single`) |

### Flow, Vectors & Deformation

| Filter (`FilterConfig` Variant) | ParaView Equivalent | Behavior |
| --- | --- | --- |
| `StreamTracerFilter` (`StreamTracer`) | Stream Tracer | Traces streamlines using `Integrator::Rk4` or adaptive `Rk45` along `IntegrationDirection` (`Forward`, `Backward`, `Both`) from `SeedSource` (`Point`, `Line`, `Sphere`); supports both point- and cell-data vector fields |
| `StreamTracerWithCustomSource` *(multi-input)* | Stream Tracer With Custom Source | Traces streamlines through `inputs[0]` (point or cell vector field) using every point of `inputs[1]` as seed positions (`pv.stream_tracer_with_custom_source`) |
| `TubeFilter` (`Tube`) | Tube | Extrudes polylines into twist-free 3D tubes using parallel-transport (Bishop) frames (`radius`, `sides`) |
| `GlyphFilter` (`Glyph`) | Glyph | Places 3D `GlyphShape` (`Arrow`, `Sphere`, `Cone`, `Cube`) instances scaled by `ScaleMode` (`Off`, `Scalar`, `VectorMagnitude`) up to `max_points` |
| `WarpByVectorFilter` (`WarpByVector`) | Warp By Vector | Displaces points along a 3-component vector array scaled by `scale_factor` |
| `WarpByScalarFilter` (`WarpByScalar`) | Warp By Scalar | Displaces points along point normals (or fixed direction) by `scalar * scale_factor` |

### Mesh Processing & Geometry

| Filter (`FilterConfig` Variant) | ParaView Equivalent | Behavior |
| --- | --- | --- |
| `SmoothFilter` (`Smooth`) | Smooth | `SmoothingMethod::Laplacian` or `Taubin` (volume-preserving) mesh smoothing (`iterations`, `relaxation_factor`) |
| `DecimateFilter` (`Decimate`) | Decimate | Garland-Heckbert Quadric Error Metric (QEM) triangle mesh simplification (`target_reduction`, `preserve_boundaries`) |
| `Delaunay2DFilter` (`Delaunay2D`) | Delaunay 2D | Bowyer-Watson 2D triangulation in XY (preserving Z) with optional `alpha` circumradius cutoff |
| `Delaunay3DFilter` (`Delaunay3D`) | Delaunay 3D | Bowyer-Watson 3D tetrahedralization producing an `UnstructuredGrid` of `CellType::Tetra` |
| `TriangulateFilter` (`Triangulate`) | Triangulate | Splits polygons/quads into triangles |
| `CleanPolyDataFilter` (`CleanPolyData`) | Clean | Spatial tolerance vertex welding and degenerate cell removal |
| `GenerateNormalsFilter` (`GenerateNormals`) | Generate Surface Normals | Computes smooth/split vertex normals (`feature_angle`) |
| `FeatureEdgesFilter` (`FeatureEdges`) | Feature Edges | Extracts boundary, non-manifold, and sharp dihedral edges |
| `ExtractEdgesFilter` (`ExtractEdges`) | Extract Edges | Extracts all topological wireframe edges |
| `ConnectivityFilter` (`Connectivity`) | Connectivity | Labels connected mesh components (`RegionId`) |
| `ShrinkFilter` (`Shrink`) | Shrink | Scales each cell toward its centroid (`shrink_factor`) |
| `CellCentersFilter` (`CellCenters`) | Cell Centers | Converts cells into centroid vertex points preserving cell attributes |
| `ReflectFilter` (`Reflect`) | Reflect | Mirrors geometry across `ReflectPlane` (`XMin`, `XMax`, `YMin`, `YMax`, `ZMin`, `ZMax`, `Custom`) |
| `TransformFilter` (`Transform`) | Transform | Applies affine `translation`, `rotation_deg`, and `scale` |

### Fields, Attributes, Probing & Resampling

| Filter (`FilterConfig` Variant) | ParaView Equivalent | Behavior |
| --- | --- | --- |
| `GradientFilter` (`Gradient`) | Gradient | Finite-difference / least-squares spatial derivatives: gradient, optional `divergence`, `vorticity`, `q_criterion`, and `Lambda2` vortex criterion |
| `CalculatorFilter` (`Calculator`) | Calculator | Recursive-descent math expression evaluator over `Point` or `Cell` data |
| `ElevationFilter` (`Elevation`) | Elevation | Computes scalar projection along a line (`low_point` to `high_point`) |
| `CellDataToPointDataFilter` (`CellDataToPointData`) | Cell Data to Point Data | Averages incident cell arrays onto vertices |
| `PointDataToCellDataFilter` (`PointDataToCellData`) | Point Data to Cell Data | Averages cell vertex arrays onto cells |
| `ProbeOverLineFilter` (`ProbeOverLine`) | Plot Over Line | Samples dataset fields along a line segment (`point1`, `point2`, `resolution`) for XY charting |
| `ResampleToImageFilter` (`ResampleToImage`) | Resample To Image | Resamples any dataset onto a regular `ImageData` grid (`dimensions`, `custom_bounds`) for GPU Volume Raymarching or GPU Marching Cubes |
| `ResampleWithDatasetFilter` *(multi-input)* | Resample With Dataset | Samples `inputs[0]` fields at every point of `inputs[1]` (`pv.resample_with_dataset(src, dst)`) |
| `IntegrateVariablesFilter` (`IntegrateVariables`) | Integrate Variables | Computes total arc length / surface area / volume and geometry-weighted integrals/averages of all point and cell arrays |
| `AppendDatasetsFilter` (`AppendDatasets` / multi-input) | Append Datasets | Combines multiple datasets into a single mesh or grid (`append_datasets`, `pv.append_nodes`) |
| `TemporalStatisticsFilter` (`TemporalStatistics`) | Temporal Statistics | Computes point-wise `average`, `minimum`, `maximum`, and `stddev` across `MultiBlockDataSet` timesteps |
| `AddTimeStepsFilter` (`AddTimeSteps`) | Add TimeSteps | Generates `num_steps` synthetic time steps (`[start_time..end_time]`) over a static dataset |

### Standalone Analytical Helpers

- `compute_histogram(dataset, array_name, association, component, num_bins) -> Result<HistogramResult, PipelineError>`
- `interpolate_datasets(a, b, t) -> Result<DataSet, PipelineError>` — Temporal linear interpolation between two topologically matching datasets.

---

## `CalculatorFilter` Expression Syntax

`CalculatorFilter` parses mathematical expressions with a recursive-descent parser and executes in parallel with `rayon`:

- **Operators**: `+`, `-`, `*`, `/`, `^` (exponentiation), unary `-` / `+`, parentheses `(...)`
- **Built-in Coordinates**: `coordsX`, `coordsY`, `coordsZ`, and `coords` (3-component vector)
- **Built-in Functions**: `mag(v)`, `sin(x)`, `cos(x)`, `tan(x)`, `sqrt(x)`, `abs(x)`, `exp(x)`, `ln(x)`, `min(a, b)`, `max(a, b)`, `clamp(x, lo, hi)`
- **Array References**: Any point or cell array name present in the dataset (scalar or vector)

Example:
```rust
let calc = CalculatorFilter::new(
    "sqrt(coordsX^2 + coordsY^2) + mag(Velocity)",
    "CombinedMetric",
    AttributeAssociation::Point,
);
```
