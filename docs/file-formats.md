# File Formats & I/O (`paraviz-io`)

`paraviz_io::read_dataset(path)` auto-detects file formats by extension and magic bytes, and `paraviz_io::write_dataset(path, &dataset)` dispatches writers by extension.

## Supported Formats Summary

| Format | Extensions | Read | Write | Target `DataSet` / Notes |
| --- | --- | --- | --- | --- |
| **Legacy VTK** | `.vtk` | Yes | Yes | ASCII & Big-Endian Binary (`POLYDATA`, `STRUCTURED_POINTS`, `RECTILINEAR_GRID`, `STRUCTURED_GRID`, `UNSTRUCTURED_GRID`) |
| **VTK XML** | `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm` | Yes | Yes | ASCII, Base64 Binary, Appended, and `vtkZLibDataCompressor` (`VtkXmlFormat`) |
| **Partitioned VTK XML** | `.pvtp`, `.pvti`, `.pvtu`, `.pvtr`, `.pvts` | Yes | — | Parallel/piece-based XML readers assembling pieces into a single dataset |
| **ParaView Time Series** | `.pvd` | Yes | Yes | `read_pvd_series`, `write_pvd_series`, and lazy `PvdSeriesSource` |
| **STL** | `.stl` | Yes | Yes | ASCII & Binary (`PolyData`) |
| **Wavefront OBJ** | `.obj` | Yes | Yes | Vertices, normals, polygons (`PolyData`) |
| **Stanford PLY** | `.ply` | Yes | Yes | ASCII, Binary Little-Endian & Binary Big-Endian (`PlyFormat`) |
| **Delimited Tables** | `.csv`, `.tsv` | Yes | Yes | Point clouds with numeric columns imported as point arrays (`PolyData`) |
| **Binary glTF 2.0** | `.glb` | — | Yes | Self-contained `POSITION`, `NORMAL`, `INDICES`, plus optional colormap-baked `COLOR_0` (`write_glb`, `encode_glb_bytes`) |
| **Animated GIF** | `.gif` | — | Yes | Pure-Rust GIF89a encoder with Netscape 2.0 loop & transparency (`write_animated_gif`, `render_turntable_gif`, `render_timeseries_gif`) |
| **Interactive Web Viewer** | `.html`, `.htm` | — | Yes | Self-contained WebGL2 3D viewer with orbit/pan/zoom, wireframe toggle & optional vertex colors (`write_html_viewer`, `export_html`) |
| **Raw Binary Volume** | `.raw`, `.bin`, `.vol` | Yes | — | `read_raw_volume(path, &RawVolumeDescriptor)` (`ImageData`) |
| **NIfTI-1** | `.nii`, `.nii.gz` | Yes | Yes | `read_nifti`, `write_nifti` (automatic gzip detection, `ImageData`) |
| **NRRD** | `.nrrd`, `.nhdr` | Yes | — | `read_nrrd` (`raw` and `gzip` encoding, `ImageData`) |
| **MetaImage** | `.mhd`, `.mha` | Yes | Yes (`.mha`) | `read_metaimage`, `write_metaimage` (`ImageData`) |
| **EnSight Gold** | `.case` | Yes | — | `read_ensight_case` (ASCII geometry + node/element variables -> `UnstructuredGrid` / `MultiBlockDataSet`) |
| **XDMF** | `.xdmf`, `.xmf` | Yes | — | `read_xdmf`, `read_xdmf_at_time` (XML, Binary, pure-Rust HDF5 `Format="HDF"`, `Mixed`/Structured/Rectilinear/`ImageData`, `MultiBlock`, temporal collections) |
| **OpenFOAM** | `.foam`, `controlDict`, case dir | Yes | — | `read_openfoam` (reads `constant/polyMesh` + latest time fields -> `PolyData`) |

---

## Raw Binary 3D Volume Descriptor

Raw volumetric files lack a self-describing header, so specify dimensions, scalar type, endianness, origin, and spacing via `RawVolumeDescriptor`:

```rust
use paraviz::prelude::*;

let desc = RawVolumeDescriptor {
    dimensions: [256, 256, 128],
    scalar_type: RawScalarType::U16,
    endian: RawEndian::Little,
    origin: [0.0, 0.0, 0.0],
    spacing: [1.0, 1.0, 1.0],
    header_bytes: 0,
    array_name: "Density".into(),
};

let mut pv = ParaViz::new();
let vol_id = pv.open_raw_volume("CTScan", "scan_256x256x128_u16.raw", &desc)?;
pv.show(vol_id).representation(Representation::Volume);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Supported `RawScalarType` variants: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `F32`, `F64`.

---

## Time Series (`PvdSeriesSource`)

`PvdSeriesSource` parses a `.pvd` XML manifest (`DataSet` elements with `timestep` and `file` attributes) without eagerly loading every timestep into RAM. When `pv.set_time(t)` and `pv.evaluate(id)` run, only the matching timestep file is loaded on demand.

```rust
let mut pv = ParaViz::new();
let series = pv.open("simulation.pvd")?;
for t in pv.all_time_steps() {
    pv.set_time(t);
    pv.render_png(format!("frame_{t:.2}.png"), 1280, 720)?;
}
```

---

## Binary glTF 2.0 Export (`.glb`)

Export any `PolyData` (or any surface-extractable dataset via `ParaViz::export_glb`) with optional vertex colors baked from a `ColorMap`:

```rust
pv.export_glb(iso_node, "isosurface.glb", true)?;
```
