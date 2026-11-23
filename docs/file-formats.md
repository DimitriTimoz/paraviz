# File Formats & I/O Reference (`paraviz-io`)

`paraviz_io::read_dataset(path)` auto-detects scientific formats by file extension and magic bytes (or explicitly via `DatasetReader`), and `paraviz_io::write_dataset(path, &dataset)` dispatches writers by extension. Every reader is written in 100% safe Rust (`#![forbid(unsafe_code)]`) with strict allocation and decompression bounds.

> **Related Documentation**
> - [End-User Manual](user-guide.md)
> - [Rendering, Style & Scene Parameters](rendering-and-scene.md)
> - [Rust API Guide](rust-api.md)

---

## 1. Supported Formats Summary

| Format | Extensions | Read | Write | Target `DataSet` & Notes |
| --- | --- | --- | --- | --- |
| **Legacy VTK** | `.vtk` | Yes | Yes | ASCII & Big-Endian Binary (`POLYDATA`, `STRUCTURED_POINTS`, `RECTILINEAR_GRID`, `STRUCTURED_GRID`, `UNSTRUCTURED_GRID`) via `VtkLegacyFormat` |
| **VTK XML** | `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm` | Yes | Yes | ASCII, Inline Base64 Binary, Appended (`raw` `_` & `base64`), and `vtkZLibDataCompressor` (`VtkXmlFormat`) |
| **Partitioned VTK XML** | `.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts` | Yes | — | Piece-based XML readers assembling sub-pieces safely relative to parent directory |
| **ParaView Time Series** | `.pvd` | Yes | Yes | `read_pvd_series`, `write_pvd_series`, and lazy on-demand `PvdSeriesSource` |
| **VTKHDF (HDF5)** | `.vtkhdf` | Yes | — | `read_vtkhdf`, `read_vtkhdf_at_time`, `inspect_vtkhdf_time_steps`: `ImageData`, `PolyData`, `UnstructuredGrid`, `PartitionedDataSet`, `MultiBlockDataSet`, temporal `Steps` |
| **CGNS (HDF5)** | `.cgns` | Yes | — | `read_cgns`: Multi-base/multi-zone `StructuredGrid`, `UnstructuredGrid`, or `MultiBlockDataSet` with `FlowSolution` arrays |
| **Exodus II** | `.exo`, `.e`, `.ex2` | Yes | — | `read_exodus`, `read_exodus_at_time`, `inspect_exodus_time_steps`: Both pure-Rust NetCDF Classic (`CDF\x01`/`CDF\x02`) and HDF5 NetCDF-4 element blocks + nodal/element variables + `time_whole` |
| **XDMF** | `.xdmf`, `.xmf` | Yes | — | `read_xdmf`, `read_xdmf_at_time`, `inspect_xdmf_time_steps`: XML, Binary, and pure-Rust HDF5 `Format="HDF"`, `Mixed` / Structured / Rectilinear / `ImageData`, `MultiBlock`, temporal collections |
| **EnSight Gold** | `.case` | Yes | — | `read_ensight_case`: ASCII geometry + node/element variables → `UnstructuredGrid` / `MultiBlockDataSet` |
| **OpenFOAM** | `.foam`, `controlDict`, case dir | Yes | — | `read_openfoam`: Parses `constant/polyMesh` + latest time fields → `PolyData` / `MultiBlockDataSet` |
| **STL** | `.stl` | Yes | Yes | ASCII & Binary triangle meshes (`PolyData`) with coincident vertex welding & facet normals |
| **Wavefront OBJ** | `.obj` | Yes | Yes | Vertices, normals, texture/polygon faces (`PolyData`) |
| **Stanford PLY** | `.ply` | Yes | Yes | ASCII, Binary Little-Endian & Binary Big-Endian (`PlyFormat`) |
| **Delimited Tables** | `.csv`, `.tsv` | Yes | Yes | Point clouds with numeric columns imported as `point_data` arrays (`PolyData`) |
| **NIfTI-1** | `.nii`, `.nii.gz` | Yes | Yes | `read_nifti`, `write_nifti`: Automatic `.gz` decompression/compression → `ImageData` |
| **NRRD** | `.nrrd`, `.nhdr` | Yes | — | `read_nrrd`: `raw` and `gzip` encodings (including detached `.nhdr` data files) → `ImageData` |
| **MetaImage** | `.mhd`, `.mha` | Yes | Yes (`.mha`) | `read_metaimage`, `write_metaimage`: Inline `.mha` and split `.mhd` + `.raw` → `ImageData` |
| **Raw Binary Volume** | `.raw`, `.bin`, `.vol` | Yes | — | `read_raw_volume`, `read_raw_volume_with_type` (`ImageData` configured via `RawVolumeDescriptor`) |
| **Binary glTF 2.0** | `.glb` | — | Yes | `write_glb`, `encode_glb_bytes`: Self-contained `POSITION`, `NORMAL`, `INDICES`, plus optional colormap-baked `COLOR_0` |
| **Animated GIF** | `.gif` | — | Yes | `write_animated_gif`, `encode_animated_gif_bytes`: Pure-Rust GIF89a with Netscape 2.0 loop, transparency, and LZW |
| **Interactive Web Viewer** | `.html`, `.htm` | — | Yes | `write_html_viewer`, `encode_html_viewer`: Zero-CDN standalone WebGL2 3D viewer with orbit/pan/zoom, wireframe, and vertex colors |

---

## 2. Explicit Reader Selection (`DatasetReader`)

Pass any `DatasetReader` variant to `ParaViz::open_with_reader(path, reader)` or `VtkFileSource::with_reader(path, reader)` (exposed in the GUI **Open File With Reader...** selector):

| Variant | Label | Target Formats |
| --- | --- | --- |
| `Auto` *(default)* | Auto-detect | Dispatches by extension and header magic bytes |
| `VtkHdf` | VTKHDF Reader (`.vtkhdf`) | HDF5-backed VTKHDF static & temporal datasets |
| `Cgns` | CGNS Reader (`.cgns`) | HDF5-backed CFD structured/unstructured zones |
| `Exodus` | Exodus II Reader (`.exo`, `.e`, `.ex2`) | NetCDF Classic (`CDF\x01`/`\x02`) & HDF5 NetCDF-4 |
| `Xdmf` | XDMF Reader (`.xdmf`, `.xmf`) | XML + Binary/HDF5 grids and temporal collections |
| `VtkXml` | VTK XML Reader (`.vtp`, `.vtu`, `.vti`, `.vtr`, `.vts`, `.vtm`, `.pvtp`, `.pvti`, `.pvtu`, `.pvd`) | XML VTK datasets |
| `VtkLegacy` | VTK Legacy Reader (`.vtk`) | ASCII & Big-Endian binary `.vtk` |
| `EnSight` | EnSight Gold Reader (`.case`) | CFD EnSight case files |
| `OpenFoam` | OpenFOAM Reader (`.foam`, `polyMesh`) | OpenFOAM mesh & field time directories |
| `Stl` | STL Mesh Reader (`.stl`) | ASCII & Binary STL |
| `Obj` | Wavefront OBJ Reader (`.obj`) | Wavefront `.obj` |
| `Ply` | PLY Mesh Reader (`.ply`) | ASCII & Binary LE/BE `.ply` |
| `Csv` | CSV / TSV Table Reader (`.csv`, `.tsv`) | Point tables |
| `Nifti` | NIfTI Volume Reader (`.nii`, `.nii.gz`) | 3D NIfTI volumes |
| `Nrrd` | NRRD Volume Reader (`.nrrd`, `.nhdr`) | 3D NRRD volumes |
| `MetaImage` | MetaImage Volume Reader (`.mhd`, `.mha`) | 3D MetaImage volumes |

---

## 3. Writer Format Enums

### `VtkXmlFormat`
Passed to `write_vtk_xml_with_format`:
- `Ascii` — Plain-text XML data arrays
- `Binary` — Inline Base64 binary arrays with `UInt32` length header
- `BinaryZlib` — Inline Base64 compressed with `vtkZLibDataCompressor`
- `BinaryCompressed` — Compressed inline VTK XML
- `Appended` — `_`-prefixed appended raw binary section at end of XML
- `AppendedCompressed` — `_`-prefixed appended zlib-compressed section

### `VtkLegacyFormat`
Passed to legacy `.vtk` writers:
- `Ascii` — Human-readable ASCII legacy VTK
- `Binary` — Big-Endian binary legacy VTK

### `PlyFormat`
Passed to PLY writers:
- `Ascii` — Text `format ascii 1.0`
- `BinaryLittleEndian` — `format binary_little_endian 1.0`
- `BinaryBigEndian` — `format binary_big_endian 1.0`

---

## 4. Raw Binary 3D Volume Descriptor (`RawVolumeDescriptor`)

Because raw `.raw` / `.bin` / `.vol` files lack headers, every dimension and layout property is configured explicitly via `RawVolumeDescriptor` (or the **Import Raw 3D Volume** modal in the GUI):

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dimensions` | `[usize; 3]` | `[1, 1, 1]` | Voxel counts `[nx, ny, nz]` (X fastest-varying) |
| `scalar_type` | `ScalarType` | `ScalarType::U8` | Element storage type: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`, `F32`, `F64` (raw helper also accepts `RawScalarType`) |
| `origin` | `[f64; 3]` | `[0.0, 0.0, 0.0]` | World-space lower corner `[ox, oy, oz]` |
| `spacing` | `[f64; 3]` | `[1.0, 1.0, 1.0]` | Voxel spacing `[sx, sy, sz]` |
| `endian` | `RawEndian` | `RawEndian::Little` | Byte order: `Little` or `Big` |
| `header_bytes` | `usize` | `0` | Leading header bytes to skip before reading voxel data |
| `array_name` | `String` | `"Scalars"` | Name assigned to the resulting `ImageData::point_data` scalar array |

```rust
use paraviz::prelude::*;

let desc = RawVolumeDescriptor {
    dimensions: [256, 256, 128],
    scalar_type: ScalarType::U16,
    endian: RawEndian::Little,
    origin: [0.0, 0.0, 0.0],
    spacing: [0.5, 0.5, 1.0],
    header_bytes: 0,
    array_name: "Density".into(),
};

let mut pv = ParaViz::new();
let vol = pv.open_raw_volume("CTScan", "scan_256x256x128_u16.raw", &desc)?;
pv.show(vol).representation(Representation::Volume);
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## 5. Export Function Parameters (`.glb`, `.gif`, `.html`)

- **Binary glTF (`write_glb(path, poly, color_map: Option<&ColorMap>, scalar_array: Option<&str>)`)**:
  - Encodes triangles (`MODE_TRIANGLES`), polylines (`MODE_LINES`), or vertices (`MODE_POINTS`) with `POSITION`, `NORMAL`, `INDICES`, and optional `COLOR_0` RGBA vertex attributes mapped via `color_map` over `scalar_array`.
- **Animated GIF (`write_animated_gif(path, frames_rgba, width, height, delay_cs: u16)`)**:
  - `delay_cs`: frame duration in centiseconds (`1/100` s; e.g., `6` = `60 ms`, `5` = 20 FPS). Supports transparent background frames when alpha is `0`.
  - Session helpers `ParaViz::render_turntable_gif(path, frames, width, height, delay_ms)` and `ParaViz::render_timeseries_gif(path, width, height, delay_ms)` accept millisecond delays clamped cleanly (`(delay_ms / 10).max(1)` centiseconds).
- **Interactive WebGL2 HTML (`write_html_viewer(path, poly, title, color_map: Option<&ColorMap>, scalar_array: Option<&str>)`)**:
  - Writes a single self-contained HTML5/WebGL2 page embedding geometry, normals, optional vertex colors, orbit/pan/zoom camera controls, and wireframe toggle.

---

## 6. Bounded Parsing & Security Limits (`limits.rs`)

All readers enforce strict bounds against malformed or adversarial files:

| Constant / Guard | Value | Purpose |
| --- | --- | --- |
| `MAX_DECOMPRESSED_BYTES` | `512 MiB` (`512 * 1024 * 1024`) | Hard cap on any single gzip, zlib, or LZMA/XZ decompressed payload |
| `MAX_ALLOC_BYTES` | `1 GiB` (`1024 * 1024 * 1024`) | Upper bound on any single parsed array or point/cell buffer allocation |
| `MAX_PREALLOC_ELEMS` | `1,000,000` | Maximum initial `Vec::with_capacity` hint trusted before verifying actual stream data |
| `MAX_XML_DEPTH` | `256` | Maximum XML tag nesting depth for VTK XML, `.pvd`, and XDMF parsers |
| `resolve_safe_path` | Path jail | Rejects NUL bytes, absolute paths, `..` traversal, and symlinks escaping the parent dataset directory |
