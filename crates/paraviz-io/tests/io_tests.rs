use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use paraviz_core::{
    ArrayData, CellType, ColorMap, ColorMapPreset, DataArray, DataSet, ImageData,
    MultiBlockDataSet, PipelineGraph, PolyData, RectilinearGrid, ScalarType, Source,
    StructuredGrid, UnstructuredGrid,
};
use paraviz_io::csv::{read_csv, read_csv_str, write_csv};
use paraviz_io::gltf::write_glb;
use paraviz_io::mesh::{
    read_obj, read_obj_str, read_ply, read_stl, write_obj, write_ply_with_format, write_stl_ascii,
    write_stl_binary, PlyFormat,
};
use paraviz_io::raw_volume::{
    read_raw_volume, read_raw_volume_i16, read_raw_volume_u16, RawEndian, RawVolumeDescriptor,
};
use paraviz_io::vtk_legacy::{read_vtk_legacy, write_vtk_legacy_with_format, VtkLegacyFormat};
use paraviz_io::vtk_xml::{
    read_vtk_xml, read_vtk_xml_bytes, write_pvd_series, write_vtk_xml_with_format, VtkXmlFormat,
};
use paraviz_io::{read_dataset, read_pvd_series, write_dataset, PvdSeriesSource, VtkFileSource};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(name: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("paraviz_io_test_{name}_{pid}_{id}"));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sample_polydata() -> PolyData {
    let mut poly = PolyData::from_triangles(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.5],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    );
    poly.verts = vec![0, 3];
    poly.lines = vec![vec![0, 1, 2]];
    poly.strips = vec![vec![0, 1, 3, 2]];
    poly.point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![10.0, 20.0, 30.0, 40.0]));
    poly.point_data.set_vectors(DataArray::vectors_f32(
        "vel",
        vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    ));
    poly.compute_normals();
    let ncells = poly.num_cells();
    poly.cell_data.set_scalars(DataArray::new(
        "cell_id",
        1,
        ArrayData::I32((0..ncells as i32).collect()),
    ));
    poly.field_data
        .add_array(DataArray::scalars_f64("time_meta", vec![1.25]));
    poly
}

fn sample_imagedata() -> ImageData {
    let mut img = ImageData::new([3, 2, 2], [-1.0, 0.5, 2.0], [0.5, 0.25, 1.0]);
    let np = img.num_points();
    let nc = img.num_cells();
    img.point_data.set_scalars(DataArray::scalars_f32(
        "density",
        (0..np).map(|i| i as f32 * 0.5).collect(),
    ));
    img.cell_data.set_scalars(DataArray::scalars_f64(
        "cell_val",
        (0..nc).map(|i| i as f64 + 1.0).collect(),
    ));
    img
}

fn sample_rectilinear() -> RectilinearGrid {
    let mut rg = RectilinearGrid {
        x_coords: vec![0.0, 0.5, 2.0],
        y_coords: vec![-1.0, 1.0],
        z_coords: vec![0.0, 3.0],
        ..Default::default()
    };
    let np = rg.num_points();
    rg.point_data.set_scalars(DataArray::scalars_f32(
        "pressure",
        (0..np).map(|i| 100.0 + i as f32).collect(),
    ));
    rg
}

fn sample_structured() -> StructuredGrid {
    let dims = [2, 2, 2];
    let mut points = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                points.push([i as f32, j as f32 + 0.1 * k as f32, k as f32 * 2.0]);
            }
        }
    }
    let mut sg = StructuredGrid {
        dimensions: dims,
        points,
        ..Default::default()
    };
    sg.point_data.set_scalars(DataArray::scalars_f32(
        "phi",
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ));
    sg
}

fn sample_unstructured() -> UnstructuredGrid {
    let mut ug = UnstructuredGrid::new();
    ug.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    ug.push_cell(CellType::Tetra, &[0, 1, 2, 3]);
    ug.push_cell(CellType::Hexahedron, &[0, 1, 4, 2, 3, 5, 7, 6]);
    ug.push_cell(CellType::Triangle, &[0, 1, 2]);
    ug.point_data.set_scalars(DataArray::scalars_f32(
        "u",
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ));
    ug.cell_data
        .set_scalars(DataArray::new("mat", 1, ArrayData::I32(vec![1, 2, 3])));
    ug
}

#[test]
fn vtk_legacy_ascii_and_binary_roundtrip_all_datasets() {
    let dir = TestDir::new("vtk_legacy");

    let datasets: Vec<(&str, DataSet)> = vec![
        ("poly", DataSet::PolyData(sample_polydata())),
        ("img", DataSet::ImageData(sample_imagedata())),
        ("rect", DataSet::RectilinearGrid(sample_rectilinear())),
        ("struct", DataSet::StructuredGrid(sample_structured())),
        ("ugrid", DataSet::UnstructuredGrid(sample_unstructured())),
    ];

    for (name, ds) in &datasets {
        for (tag, fmt) in [
            ("ascii", VtkLegacyFormat::Ascii),
            ("bin", VtkLegacyFormat::Binary),
        ] {
            let p = dir.file(&format!("{name}_{tag}.vtk"));
            write_vtk_legacy_with_format(&p, ds, fmt).unwrap();
            let back = read_vtk_legacy(&p).unwrap();
            assert_eq!(back.num_points(), ds.num_points(), "{name} {tag} points");
            assert_eq!(back.num_cells(), ds.num_cells(), "{name} {tag} cells");
            assert!(
                (back.bounds().diagonal() - ds.bounds().diagonal()).abs() < 1e-4,
                "{name} {tag} bounds"
            );
            let orig_pd = ds.point_data().unwrap();
            let back_pd = back.point_data().unwrap();
            assert_eq!(back_pd.len(), orig_pd.len(), "{name} {tag} point arrays");
            for arr in orig_pd.arrays() {
                let got = back_pd.get(&arr.name).expect("missing point array");
                assert_eq!(got.num_components, arr.num_components);
                assert_eq!(got.num_tuples(), arr.num_tuples());
                for i in 0..arr.data.len() {
                    assert!(
                        (got.data.get_f64(i) - arr.data.get_f64(i)).abs() < 1e-4,
                        "{name} {tag} array {} idx {i}",
                        arr.name
                    );
                }
            }
        }
    }
}

#[test]
fn vtk_legacy_tensors_and_field_data() {
    let dir = TestDir::new("vtk_legacy_tensors");
    let vtk = "\
# vtk DataFile Version 4.2
Tensor and field test
ASCII
DATASET POLYDATA
POINTS 2 float
0 0 0
1 0 0
VERTICES 2 4
1 0
1 1
FIELD GlobalMeta 1
run_id 1 1 int
42
POINT_DATA 2
TENSORS stress float
1 0 0 0 2 0 0 0 3
4 0 0 0 5 0 0 0 6
";
    let path = dir.file("tensors.vtk");
    fs::write(&path, vtk).unwrap();
    let ds = read_dataset(&path).unwrap();
    let poly = ds.as_poly_data().unwrap();
    assert_eq!(poly.verts, vec![0, 1]);
    let stress = poly.point_data.get("stress").unwrap();
    assert_eq!(stress.num_components, 9);
    assert_eq!(stress.num_tuples(), 2);
    assert!((stress.data.get_f64(13) - 5.0).abs() < 1e-6);
    let run_id = poly.field_data.get("run_id").unwrap();
    assert_eq!(run_id.data.get_f64(0) as i32, 42);
}

#[test]
fn vtk_xml_all_formats_and_encodings_roundtrip() {
    let dir = TestDir::new("vtk_xml");

    let encodings = [
        ("ascii", VtkXmlFormat::Ascii),
        ("bin", VtkXmlFormat::Binary),
        ("zlib", VtkXmlFormat::BinaryCompressed),
        ("app", VtkXmlFormat::Appended),
        ("app_zlib", VtkXmlFormat::AppendedCompressed),
    ];

    let datasets: Vec<(&str, &str, DataSet)> = vec![
        ("poly", "vtp", DataSet::PolyData(sample_polydata())),
        ("img", "vti", DataSet::ImageData(sample_imagedata())),
        ("rect", "vtr", DataSet::RectilinearGrid(sample_rectilinear())),
        ("struct", "vts", DataSet::StructuredGrid(sample_structured())),
        ("ugrid", "vtu", DataSet::UnstructuredGrid(sample_unstructured())),
    ];

    for (name, ext, ds) in &datasets {
        for (enc_name, fmt) in encodings {
            let p = dir.file(&format!("{name}_{enc_name}.{ext}"));
            write_vtk_xml_with_format(&p, ds, fmt).unwrap();
            let back = read_vtk_xml(&p).unwrap();
            assert_eq!(back.num_points(), ds.num_points(), "{name} {enc_name}");
            assert_eq!(back.num_cells(), ds.num_cells(), "{name} {enc_name}");
            assert!(
                (back.bounds().diagonal() - ds.bounds().diagonal()).abs() < 1e-5,
                "{name} {enc_name}"
            );
            let orig_pd = ds.point_data().unwrap();
            let back_pd = back.point_data().unwrap();
            assert_eq!(back_pd.active_scalars, orig_pd.active_scalars);
            for arr in orig_pd.arrays() {
                let got = back_pd.get(&arr.name).unwrap();
                assert_eq!(got.num_components, arr.num_components);
                for i in 0..arr.data.len() {
                    assert!((got.data.get_f64(i) - arr.data.get_f64(i)).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn vtk_xml_uint32_big_endian_concatenated_zlib_and_appended_base64() {
    // Test VTK 0.1 style UInt32 header_type + BigEndian + concatenated base64 header/payload
    // with vtkZLibDataCompressor, and AppendedData encoding="base64".
    let pts: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut pts_be = Vec::new();
    for v in pts {
        pts_be.extend_from_slice(&v.to_be_bytes());
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&pts_be).unwrap();
    let pts_zlib = enc.finish().unwrap();

    // Header: num_blocks=1, block_size=36, last_block_size=36, comp_size=pts_zlib.len()
    let mut hdr_be = Vec::new();
    for w in [1u32, 36u32, 36u32, pts_zlib.len() as u32] {
        hdr_be.extend_from_slice(&w.to_be_bytes());
    }
    let pts_b64 = format!("{}{}", BASE64.encode(&hdr_be), BASE64.encode(&pts_zlib));

    let conn: [i32; 3] = [0, 1, 2];
    let offs: [i32; 1] = [3];
    let mut conn_be = Vec::new();
    for v in conn {
        conn_be.extend_from_slice(&v.to_be_bytes());
    }
    let mut offs_be = Vec::new();
    for v in offs {
        offs_be.extend_from_slice(&v.to_be_bytes());
    }

    let encode_uncomp_u32_be = |raw: &[u8]| -> String {
        let mut blk = Vec::new();
        blk.extend_from_slice(&(raw.len() as u32).to_be_bytes());
        blk.extend_from_slice(raw);
        BASE64.encode(&blk)
    };
    let conn_b64 = encode_uncomp_u32_be(&conn_be);
    let offs_b64 = encode_uncomp_u32_be(&offs_be);

    // Inline UInt32 BigEndian Zlib test
    let inline_xml = format!(
        r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="BigEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="binary">
          {pts_b64}
        </DataArray>
      </Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>"#
    );
    let ds = read_vtk_xml_bytes(inline_xml.as_bytes(), Path::new(".")).unwrap();
    let poly = ds.as_poly_data().unwrap();
    assert_eq!(poly.points, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert_eq!(poly.polys, vec![vec![0, 1, 2]]);

    // Appended base64 test (uncompressed UInt32 BigEndian)
    let appended_stream = format!("{conn_b64}{offs_b64}");
    let offs_offset = conn_b64.len();
    let appended_b64_xml = format!(
        r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="BigEndian" header_type="UInt32">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0</DataArray>
      </Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="appended" offset="0"/>
        <DataArray type="Int32" Name="offsets" format="appended" offset="{offs_offset}"/>
      </Polys>
    </Piece>
  </PolyData>
  <AppendedData encoding="base64">
   _{appended_stream}
  </AppendedData>
</VTKFile>"#
    );
    let ds2 = read_vtk_xml_bytes(appended_b64_xml.as_bytes(), Path::new(".")).unwrap();
    assert_eq!(ds2.as_poly_data().unwrap().polys, vec![vec![0, 1, 2]]);
}

#[test]
fn vtm_multiblock_and_pvd_series_roundtrip() {
    let dir = TestDir::new("vtm_pvd");

    let mut mb = MultiBlockDataSet::new();
    mb.add_block("surface", DataSet::PolyData(sample_polydata()));
    mb.add_block("volume", DataSet::ImageData(sample_imagedata()));
    mb.add_block("mesh", DataSet::UnstructuredGrid(sample_unstructured()));

    let vtm_path = dir.file("assembly.vtm");
    write_dataset(&vtm_path, &DataSet::MultiBlock(mb.clone())).unwrap();
    let loaded = read_dataset(&vtm_path).unwrap();
    let DataSet::MultiBlock(loaded_mb) = loaded else {
        panic!("Expected MultiBlock");
    };
    assert_eq!(loaded_mb.blocks.len(), 3);
    assert_eq!(loaded_mb.blocks[0].0, "surface");
    assert_eq!(loaded_mb.blocks[1].0, "volume");
    assert_eq!(loaded_mb.blocks[2].0, "mesh");
    assert_eq!(loaded_mb.num_points(), mb.num_points());

    // PVD time series
    let mut step0 = sample_polydata();
    step0
        .point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![0.0, 1.0, 2.0, 3.0]));
    let mut step1 = sample_polydata();
    step1
        .point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![10.0, 11.0, 12.0, 13.0]));

    let pvd_path = dir.file("sim.pvd");
    write_pvd_series(
        &pvd_path,
        &[
            (0.0, DataSet::PolyData(step0)),
            (0.5, DataSet::PolyData(step1)),
        ],
    )
    .unwrap();

    let series = read_pvd_series(&pvd_path).unwrap();
    assert_eq!(series.len(), 2);
    assert!((series[0].0 - 0.0).abs() < 1e-12);
    assert!((series[1].0 - 0.5).abs() < 1e-12);

    // Test PvdSeriesSource & VtkFileSource inside PipelineGraph
    let mut graph = PipelineGraph::new();
    let pvd_src = PvdSeriesSource::new(&pvd_path).unwrap();
    assert_eq!(pvd_src.request_information().time_steps, vec![0.0, 0.5]);
    let pvd_id = graph.add_source("pvd", Box::new(pvd_src));
    let out0 = graph.update(pvd_id, 0.0).unwrap();
    assert!(
        (out0.point_data().unwrap().get("temp").unwrap().get_scalar(0) - 0.0).abs() < 1e-6
    );
    let out1 = graph.update(pvd_id, 0.5).unwrap();
    assert!(
        (out1.point_data().unwrap().get("temp").unwrap().get_scalar(0) - 10.0).abs() < 1e-6
    );

    let file_id = graph.add_source("vtm", Box::new(VtkFileSource::new(&vtm_path)));
    let file_out = graph.update(file_id, 0.0).unwrap();
    assert_eq!(file_out.num_points(), mb.num_points());
}

#[test]
fn stl_ascii_and_binary_roundtrip() {
    let dir = TestDir::new("stl");
    let poly = PolyData::from_triangles(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    );

    let bin_path = dir.file("tet_bin.stl");
    write_stl_binary(&bin_path, &poly).unwrap();
    let back_bin = read_stl(&bin_path).unwrap();
    assert_eq!(back_bin.points.len(), 4);
    assert_eq!(back_bin.polys.len(), 4);
    assert!(back_bin.cell_data.active_normals_array().is_some());
    assert!(back_bin.point_data.active_normals_array().is_some());

    let asc_path = dir.file("tet_asc.stl");
    write_stl_ascii(&asc_path, &poly).unwrap();
    let back_asc = read_stl(&asc_path).unwrap();
    assert_eq!(back_asc.points.len(), 4);
    assert_eq!(back_asc.polys.len(), 4);

    // Zero normal in ASCII STL should compute geometric normal
    let zero_norm_stl = "\
solid tri
  facet normal 0 0 0
    outer loop
      vertex 0 0 0
      vertex 2 0 0
      vertex 0 2 0
    endloop
  endfacet
endsolid tri
";
    let zp = dir.file("zero_n.stl");
    fs::write(&zp, zero_norm_stl).unwrap();
    let ds = read_dataset(&zp).unwrap();
    let n = ds
        .as_poly_data()
        .unwrap()
        .cell_data
        .active_normals_array()
        .unwrap()
        .get_vec3(0);
    assert!((n[2] - 1.0).abs() < 1e-6);
}

#[test]
fn obj_roundtrip_and_variants() {
    let dir = TestDir::new("obj");

    // Test v/vt/vn, v//vn, negative indices, and n-gon (pentagon)
    let obj_src = "\
# Wavefront test
v 0 0 0
v 1 0 0
v 1.5 1 0
v 0.5 1.5 0
v -0.5 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0.5 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1 5/5/1
f -5/-5/-1 -4/-4/-1 -3/-3/-1
l 1 2
p 1 5
";
    let parsed = read_obj_str(obj_src).unwrap();
    assert_eq!(parsed.points.len(), 5);
    assert_eq!(parsed.polys.len(), 2);
    assert_eq!(parsed.polys[0].len(), 5); // n-gon preserved
    assert_eq!(parsed.polys[1], vec![0, 1, 2]); // negative indices resolved
    assert_eq!(parsed.lines, vec![vec![0, 1]]);
    assert_eq!(parsed.verts, vec![0, 4]);
    assert!(parsed.point_data.active_normals_array().is_some());
    assert!(parsed.point_data.get("TCoords").is_some());

    // Also test v//vn without vt
    let obj_vn_only = "\
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
";
    let vn_only = read_obj_str(obj_vn_only).unwrap();
    assert_eq!(vn_only.points.len(), 3);
    assert!(vn_only.point_data.active_normals_array().is_some());
    assert!(vn_only.point_data.get("TCoords").is_none());

    let out_path = dir.file("mesh.obj");
    write_obj(&out_path, &parsed).unwrap();
    let back = read_obj(&out_path).unwrap();
    assert_eq!(back.points.len(), parsed.points.len());
    assert_eq!(back.polys, parsed.polys);
    assert_eq!(back.lines, parsed.lines);
    assert_eq!(back.verts, parsed.verts);
    assert!(back.point_data.active_normals_array().is_some());
    assert!(back.point_data.get("TCoords").is_some());
}

#[test]
fn ply_ascii_and_binary_le_be_roundtrip() {
    let dir = TestDir::new("ply");
    let mut poly = PolyData::from_triangles(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    poly.compute_normals();
    poly.point_data.add_array(DataArray::colors_u8(
        "Colors",
        4,
        vec![255, 0, 0, 255, 0, 255, 0, 200, 0, 0, 255, 128],
    ));
    poly.point_data
        .add_array(DataArray::scalars_f32("confidence", vec![0.9, 0.8, 0.7]));
    poly.cell_data
        .add_array(DataArray::new("face_tag", 1, ArrayData::I32(vec![7])));

    for (tag, fmt) in [
        ("ascii", PlyFormat::Ascii),
        ("le", PlyFormat::BinaryLittleEndian),
        ("be", PlyFormat::BinaryBigEndian),
    ] {
        let p = dir.file(&format!("mesh_{tag}.ply"));
        write_ply_with_format(&p, &poly, fmt).unwrap();
        let back = read_ply(&p).unwrap();
        assert_eq!(back.points, poly.points, "{tag}");
        assert_eq!(back.polys, poly.polys, "{tag}");
        assert!(back.point_data.active_normals_array().is_some(), "{tag}");
        let colors = back.point_data.get("Colors").unwrap();
        assert_eq!(colors.num_components, 4);
        assert_eq!(colors.data.get_f64(0) as u8, 255);
        assert_eq!(colors.data.get_f64(7) as u8, 200);
        let conf = back.point_data.get("confidence").unwrap();
        assert!((conf.data.get_f64(2) - 0.7).abs() < 1e-6);
        let ftag = back.cell_data.get("face_tag").unwrap();
        assert_eq!(ftag.data.get_f64(0) as i32, 7);
    }
}

#[test]
fn csv_point_cloud_roundtrip_and_paraview_points_cols() {
    let dir = TestDir::new("csv");

    let pv_csv = "\
Points:0;Points:1;Points:2;Temperature;Pressure;Label
0.0;1.0;2.0;300.5;101325.0;nodeA
3.0;4.0;5.0;310.25;101200.5;nodeB
";
    let poly = read_csv_str(pv_csv).unwrap();
    assert_eq!(poly.points, vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    assert_eq!(poly.verts, vec![0, 1]);
    assert!(poly.point_data.get("Temperature").is_some());
    assert!(poly.point_data.get("Pressure").is_some());
    assert!(poly.point_data.get("Label").is_none()); // non-numeric skipped

    let csv_path = dir.file("points.csv");
    write_csv(&csv_path, &poly).unwrap();
    let back = read_csv(&csv_path).unwrap();
    assert_eq!(back.points, poly.points);
    assert_eq!(back.verts, poly.verts);
    assert!(
        (back.point_data.get("Temperature").unwrap().get_scalar(1) - 310.25).abs() < 1e-9
    );

    // Test magic-byte detection without extension
    let no_ext_vtk = dir.file("mystery_file");
    write_vtk_legacy_with_format(
        &no_ext_vtk,
        &DataSet::PolyData(sample_polydata()),
        VtkLegacyFormat::Binary,
    )
    .unwrap();
    let detected = read_dataset(&no_ext_vtk).unwrap();
    assert_eq!(detected.num_points(), 4);
}

#[test]
fn vtk_xml_binary_zlib_roundtrip_and_multiblock() {
    let dir = TestDir::new("vtk_xml_zlib");

    let datasets: Vec<(&str, &str, DataSet)> = vec![
        ("poly", "vtp", DataSet::PolyData(sample_polydata())),
        ("img", "vti", DataSet::ImageData(sample_imagedata())),
        ("rect", "vtr", DataSet::RectilinearGrid(sample_rectilinear())),
        ("struct", "vts", DataSet::StructuredGrid(sample_structured())),
        ("ugrid", "vtu", DataSet::UnstructuredGrid(sample_unstructured())),
    ];

    for (name, ext, ds) in &datasets {
        let p = dir.file(&format!("{name}_zlib.{ext}"));
        write_vtk_xml_with_format(&p, ds, VtkXmlFormat::BinaryZlib).unwrap();
        let xml_text = fs::read_to_string(&p).unwrap();
        assert!(
            xml_text.contains("compressor=\"vtkZLibDataCompressor\""),
            "{name} missing compressor attr"
        );
        assert!(
            xml_text.contains("format=\"binary\""),
            "{name} missing format=binary"
        );

        let back = read_vtk_xml(&p).unwrap();
        assert_eq!(back.num_points(), ds.num_points(), "{name}");
        assert_eq!(back.num_cells(), ds.num_cells(), "{name}");
        assert!(
            (back.bounds().diagonal() - ds.bounds().diagonal()).abs() < 1e-5,
            "{name}"
        );
        let orig_pd = ds.point_data().unwrap();
        let back_pd = back.point_data().unwrap();
        assert_eq!(back_pd.active_scalars, orig_pd.active_scalars);
        for arr in orig_pd.arrays() {
            let got = back_pd.get(&arr.name).unwrap();
            assert_eq!(got.num_components, arr.num_components);
            for i in 0..arr.data.len() {
                assert!((got.data.get_f64(i) - arr.data.get_f64(i)).abs() < 1e-5);
            }
        }
    }

    // Large PolyData (> 32 KiB raw points) to exercise multi-block vtkZLibDataCompressor headers
    let n_pts = 4096usize;
    let pts: Vec<[f32; 3]> = (0..n_pts)
        .map(|i| [i as f32 * 0.1, (i as f32).sin(), (i as f32).cos()])
        .collect();
    let tris: Vec<[u32; 3]> = (0..(n_pts as u32 - 2))
        .map(|i| [i, i + 1, i + 2])
        .collect();
    let mut big_poly = PolyData::from_triangles(pts, tris);
    big_poly.point_data.set_scalars(DataArray::scalars_f32(
        "wave",
        (0..n_pts).map(|i| (i as f32 * 0.01).sin()).collect(),
    ));
    let big_path = dir.file("big_zlib.vtp");
    write_vtk_xml_with_format(
        &big_path,
        &DataSet::PolyData(big_poly.clone()),
        VtkXmlFormat::BinaryZlib,
    )
    .unwrap();
    let big_back = read_vtk_xml(&big_path).unwrap();
    let big_poly_back = big_back.as_poly_data().unwrap();
    assert_eq!(big_poly_back.points.len(), n_pts);
    assert_eq!(big_poly_back.polys.len(), big_poly.polys.len());
    assert_eq!(big_poly_back.points[4095], big_poly.points[4095]);
    let wave_back = big_poly_back.point_data.get("wave").unwrap();
    assert!(
        (wave_back.get_scalar(3500) - big_poly.point_data.get("wave").unwrap().get_scalar(3500))
            .abs()
            < 1e-6
    );
}

#[test]
fn partitioned_vtk_xml_pvtp_pvtu_pvti() {
    let dir = TestDir::new("pvtk");
    fs::create_dir_all(dir.file("pieces")).unwrap();

    // 1. .pvtp (PPolyData)
    let mut p0 = PolyData::from_triangles(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    p0.point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![1.0, 2.0, 3.0]));
    p0.cell_data
        .set_scalars(DataArray::new("part", 1, ArrayData::I32(vec![0])));

    let mut p1 = PolyData::from_triangles(
        vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    p1.point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![4.0, 5.0, 6.0]));
    p1.cell_data
        .set_scalars(DataArray::new("part", 1, ArrayData::I32(vec![1])));

    write_vtk_xml_with_format(
        dir.file("pieces/poly_0.vtp"),
        &DataSet::PolyData(p0),
        VtkXmlFormat::BinaryZlib,
    )
    .unwrap();
    write_vtk_xml_with_format(
        dir.file("pieces/poly_1.vtp"),
        &DataSet::PolyData(p1),
        VtkXmlFormat::Ascii,
    )
    .unwrap();

    let pvtp_xml = r#"<?xml version="1.0"?>
<VTKFile type="PPolyData" version="0.1" byte_order="LittleEndian">
  <PPolyData GhostLevel="0">
    <PPointData Scalars="temp">
      <PDataArray type="Float32" Name="temp"/>
    </PPointData>
    <PCellData Scalars="part">
      <PDataArray type="Int32" Name="part"/>
    </PCellData>
    <PPoints>
      <PDataArray type="Float32" NumberOfComponents="3"/>
    </PPoints>
    <Piece Source="pieces/poly_0.vtp"/>
    <Piece Source="pieces/poly_1.vtp"></Piece>
  </PPolyData>
</VTKFile>"#;
    let pvtp_path = dir.file("mesh.pvtp");
    fs::write(&pvtp_path, pvtp_xml).unwrap();
    let merged_poly_ds = read_dataset(&pvtp_path).unwrap();
    let merged_poly = merged_poly_ds.as_poly_data().unwrap();
    assert_eq!(merged_poly.points.len(), 6);
    assert_eq!(merged_poly.polys, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(
        merged_poly.point_data.active_scalars.as_deref(),
        Some("temp")
    );
    let temps = merged_poly.point_data.get("temp").unwrap();
    assert_eq!(temps.num_tuples(), 6);
    assert!((temps.get_scalar(5) - 6.0).abs() < 1e-6);
    let parts = merged_poly.cell_data.get("part").unwrap();
    assert_eq!(parts.get_scalar(0) as i32, 0);
    assert_eq!(parts.get_scalar(1) as i32, 1);

    // 2. .pvtu (PUnstructuredGrid)
    let mut u0 = UnstructuredGrid::new();
    u0.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    u0.push_cell(CellType::Tetra, &[0, 1, 2, 3]);
    u0.point_data
        .set_scalars(DataArray::scalars_f32("u", vec![10.0, 20.0, 30.0, 40.0]));
    u0.cell_data
        .set_scalars(DataArray::new("rank", 1, ArrayData::I32(vec![0])));

    let mut u1 = UnstructuredGrid::new();
    u1.points = vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
    u1.push_cell(CellType::Triangle, &[0, 1, 2]);
    u1.point_data
        .set_scalars(DataArray::scalars_f32("u", vec![50.0, 60.0, 70.0]));
    u1.cell_data
        .set_scalars(DataArray::new("rank", 1, ArrayData::I32(vec![1])));

    write_vtk_xml_with_format(
        dir.file("pieces/ug_0.vtu"),
        &DataSet::UnstructuredGrid(u0),
        VtkXmlFormat::BinaryZlib,
    )
    .unwrap();
    write_vtk_xml_with_format(
        dir.file("pieces/ug_1.vtu"),
        &DataSet::UnstructuredGrid(u1),
        VtkXmlFormat::Binary,
    )
    .unwrap();

    let pvtu_xml = r#"<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <PUnstructuredGrid GhostLevel="0">
    <PPointData Scalars="u">
      <PDataArray type="Float32" Name="u"/>
    </PPointData>
    <PCellData Scalars="rank">
      <PDataArray type="Int32" Name="rank"/>
    </PCellData>
    <Piece Source="pieces/ug_0.vtu"/>
    <Piece Source="pieces/ug_1.vtu"/>
  </PUnstructuredGrid>
</VTKFile>"#;
    let pvtu_path = dir.file("grid.pvtu");
    fs::write(&pvtu_path, pvtu_xml).unwrap();
    let merged_ug_ds = read_dataset(&pvtu_path).unwrap();
    let merged_ug = merged_ug_ds.as_unstructured_grid().unwrap();
    assert_eq!(merged_ug.num_points(), 7);
    assert_eq!(merged_ug.num_cells(), 2);
    assert_eq!(merged_ug.cell_types, vec![CellType::Tetra, CellType::Triangle]);
    assert_eq!(merged_ug.cell_points(0), &[0, 1, 2, 3]);
    assert_eq!(merged_ug.cell_points(1), &[4, 5, 6]);
    assert!((merged_ug.point_data.get("u").unwrap().get_scalar(6) - 70.0).abs() < 1e-6);
    assert_eq!(
        merged_ug.cell_data.get("rank").unwrap().get_scalar(1) as i32,
        1
    );

    // 3. .pvti (PImageData)
    let mut img0 = ImageData::new([2, 3, 2], [0.0, 0.0, 0.0], [0.5, 1.0, 2.0]);
    let mut vals0 = Vec::new();
    for k in 0..2 {
        for j in 0..3 {
            for i in 0..2 {
                vals0.push((i + 10 * j + 100 * k) as f32);
            }
        }
    }
    img0.point_data
        .set_scalars(DataArray::scalars_f32("density", vals0));

    let mut img1 = ImageData::new([2, 3, 2], [1.0, 0.0, 0.0], [0.5, 1.0, 2.0]);
    let mut vals1 = Vec::new();
    for k in 0..2 {
        for j in 0..3 {
            for i in 2..4 {
                vals1.push((i + 10 * j + 100 * k) as f32);
            }
        }
    }
    img1.point_data
        .set_scalars(DataArray::scalars_f32("density", vals1));

    write_vtk_xml_with_format(
        dir.file("pieces/img_0.vti"),
        &DataSet::ImageData(img0),
        VtkXmlFormat::BinaryZlib,
    )
    .unwrap();
    write_vtk_xml_with_format(
        dir.file("pieces/img_1.vti"),
        &DataSet::ImageData(img1),
        VtkXmlFormat::BinaryZlib,
    )
    .unwrap();

    let pvti_xml = r#"<?xml version="1.0"?>
<VTKFile type="PImageData" version="0.1" byte_order="LittleEndian">
  <PImageData WholeExtent="0 3 0 2 0 1" Origin="0 0 0" Spacing="0.5 1 2">
    <PPointData Scalars="density">
      <PDataArray type="Float32" Name="density"/>
    </PPointData>
    <Piece Extent="0 1 0 2 0 1" Source="pieces/img_0.vti"/>
    <Piece Extent="2 3 0 2 0 1" Source="pieces/img_1.vti"/>
  </PImageData>
</VTKFile>"#;
    let pvti_path = dir.file("vol.pvti");
    fs::write(&pvti_path, pvti_xml).unwrap();
    let merged_img_ds = read_dataset(&pvti_path).unwrap();
    let merged_img = merged_img_ds.as_image_data().unwrap();
    assert_eq!(merged_img.dimensions, [4, 3, 2]);
    assert_eq!(merged_img.origin, [0.0, 0.0, 0.0]);
    assert_eq!(merged_img.spacing, [0.5, 1.0, 2.0]);
    let dens = merged_img.point_data.get("density").unwrap();
    for k in 0..2 {
        for j in 0..3 {
            for i in 0..4 {
                let idx = merged_img.point_index(i, j, k);
                let expected = (i + 10 * j + 100 * k) as f64;
                assert!((dens.get_scalar(idx) - expected).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn raw_volume_reader_all_types_and_endianness() {
    let dir = TestDir::new("raw_vol");
    let dims = [2, 2, 2]; // 8 voxels
    let header = vec![0xAAu8; 16];

    // U8
    let u8_path = dir.file("vol_u8.raw");
    let mut u8_bytes = header.clone();
    u8_bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    fs::write(&u8_path, &u8_bytes).unwrap();
    let u8_img = read_raw_volume(
        &u8_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::U8,
            origin: [-1.0, 0.0, 1.0],
            spacing: [0.5, 0.5, 2.0],
            endian: RawEndian::Little,
            header_bytes: 16,
            array_name: "ct".into(),
        },
    )
    .unwrap();
    assert_eq!(u8_img.dimensions, dims);
    assert_eq!(u8_img.origin, [-1.0, 0.0, 1.0]);
    assert_eq!(u8_img.spacing, [0.5, 0.5, 2.0]);
    assert_eq!(u8_img.point_data.get("ct").unwrap().get_scalar(7) as u8, 8);

    // I32 BigEndian
    let i32_path = dir.file("vol_i32_be.raw");
    let i32_vals: [i32; 8] = [-10, -5, 0, 5, 10, 20, 30, 40];
    let mut i32_bytes = Vec::new();
    for v in i32_vals {
        i32_bytes.extend_from_slice(&v.to_be_bytes());
    }
    fs::write(&i32_path, &i32_bytes).unwrap();
    let i32_img = read_raw_volume(
        &i32_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::I32,
            endian: RawEndian::Big,
            array_name: "signed".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        i32_img.point_data.get("signed").unwrap().get_scalar(0) as i32,
        -10
    );
    assert_eq!(
        i32_img.point_data.get("signed").unwrap().get_scalar(7) as i32,
        40
    );

    // U32 LittleEndian
    let u32_path = dir.file("vol_u32_le.raw");
    let mut u32_bytes = Vec::new();
    for v in 100u32..108u32 {
        u32_bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&u32_path, &u32_bytes).unwrap();
    let u32_img = read_raw_volume(
        &u32_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::U32,
            endian: RawEndian::Little,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        u32_img.point_data.active_scalars_array().unwrap().get_scalar(5) as u32,
        105
    );

    // F32 BigEndian + F64 LittleEndian
    let f32_path = dir.file("vol_f32_be.raw");
    let mut f32_bytes = Vec::new();
    for i in 0..8 {
        f32_bytes.extend_from_slice(&(i as f32 * 0.25).to_be_bytes());
    }
    fs::write(&f32_path, &f32_bytes).unwrap();
    let f32_img = read_raw_volume(
        &f32_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::F32,
            endian: RawEndian::Big,
            array_name: "f32_arr".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((f32_img.point_data.get("f32_arr").unwrap().get_scalar(3) - 0.75).abs() < 1e-6);

    let f64_path = dir.file("vol_f64_le.raw");
    let mut f64_bytes = Vec::new();
    for i in 0..8 {
        f64_bytes.extend_from_slice(&(i as f64 * 1.125).to_le_bytes());
    }
    fs::write(&f64_path, &f64_bytes).unwrap();
    let f64_img = read_raw_volume(
        &f64_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::F64,
            endian: RawEndian::Little,
            array_name: "f64_arr".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((f64_img.point_data.get("f64_arr").unwrap().get_scalar(4) - 4.5).abs() < 1e-12);

    // I16 & U16 helpers
    let i16_path = dir.file("vol_i16_be.raw");
    let mut i16_bytes = Vec::new();
    for i in -4i16..4i16 {
        i16_bytes.extend_from_slice(&i.to_be_bytes());
    }
    fs::write(&i16_path, &i16_bytes).unwrap();
    let i16_img = read_raw_volume_i16(
        &i16_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            endian: RawEndian::Big,
            array_name: "hu".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(i16_img.point_data.get("hu").unwrap().get_scalar(0) as i32, -4);

    let u16_path = dir.file("vol_u16_le.raw");
    let mut u16_bytes = Vec::new();
    for i in 1000u16..1008u16 {
        u16_bytes.extend_from_slice(&i.to_le_bytes());
    }
    fs::write(&u16_path, &u16_bytes).unwrap();
    let u16_img = read_raw_volume_u16(
        &u16_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            endian: RawEndian::Little,
            array_name: "u16".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        u16_img.point_data.get("u16").unwrap().get_scalar(7) as u32,
        1007
    );

    // Truncated file returns error
    let trunc_path = dir.file("truncated.raw");
    fs::write(&trunc_path, [0u8; 4]).unwrap();
    assert!(read_raw_volume(
        &trunc_path,
        &RawVolumeDescriptor {
            dimensions: dims,
            scalar_type: ScalarType::U8,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn glb_export_header_chunks_alignment_and_json_schema() {
    let dir = TestDir::new("glb");
    let mut poly = sample_polydata();
    // sample_polydata has 4 points, 2 triangles, normals, and "temp" scalars
    poly.point_data.add_array(DataArray::scalars_f32(
        "pressure",
        vec![0.0, 0.25, 0.75, 1.0],
    ));

    // 1. Export without color map (via write_dataset dispatch)
    let plain_glb = dir.file("plain.glb");
    write_dataset(&plain_glb, &DataSet::PolyData(poly.clone())).unwrap();
    let plain_bytes = fs::read(&plain_glb).unwrap();
    let (plain_json, plain_bin) = parse_and_validate_glb(&plain_bytes);
    let plain_attrs = &plain_json["meshes"][0]["primitives"][0]["attributes"];
    assert!(plain_attrs.get("POSITION").is_some());
    assert!(plain_attrs.get("NORMAL").is_some());
    assert!(plain_attrs.get("COLOR_0").is_none());
    assert!(plain_json["meshes"][0]["primitives"][0]["indices"].is_u64());
    assert!(!plain_bin.is_empty());

    // 2. Export with ColorMap + explicit scalar array
    let cmap = ColorMap::from_preset(ColorMapPreset::Viridis);
    let colored_glb = dir.file("colored.glb");
    write_glb(&colored_glb, &poly, Some(&cmap), Some("pressure")).unwrap();
    let colored_bytes = fs::read(&colored_glb).unwrap();
    let (json, bin) = parse_and_validate_glb(&colored_bytes);

    let prim = &json["meshes"][0]["primitives"][0];
    assert_eq!(prim["mode"], 4);
    let attrs = &prim["attributes"];
    let pos_acc_idx = attrs["POSITION"].as_u64().unwrap() as usize;
    let norm_acc_idx = attrs["NORMAL"].as_u64().unwrap() as usize;
    let col_acc_idx = attrs["COLOR_0"].as_u64().unwrap() as usize;
    let idx_acc_idx = prim["indices"].as_u64().unwrap() as usize;

    let accessors = json["accessors"].as_array().unwrap();
    let buffer_views = json["bufferViews"].as_array().unwrap();

    // Validate POSITION min/max
    let pos_acc = &accessors[pos_acc_idx];
    assert_eq!(pos_acc["type"], "VEC3");
    assert_eq!(pos_acc["componentType"], 5126);
    assert_eq!(pos_acc["count"], 4);
    assert_eq!(pos_acc["min"], serde_json::json!([0.0, 0.0, 0.0]));
    assert_eq!(pos_acc["max"], serde_json::json!([1.0, 1.0, 0.5]));

    // Validate NORMAL
    let norm_acc = &accessors[norm_acc_idx];
    assert_eq!(norm_acc["type"], "VEC3");
    assert_eq!(norm_acc["componentType"], 5126);
    assert_eq!(norm_acc["count"], 4);

    // Validate COLOR_0 values match ColorMap::map_value
    let col_acc = &accessors[col_acc_idx];
    assert_eq!(col_acc["type"], "VEC4");
    assert_eq!(col_acc["componentType"], 5126);
    assert_eq!(col_acc["count"], 4);
    let col_bv = &buffer_views[col_acc["bufferView"].as_u64().unwrap() as usize];
    let col_off = col_bv["byteOffset"].as_u64().unwrap() as usize;
    let first_rgba = [
        f32::from_le_bytes(bin[col_off..col_off + 4].try_into().unwrap()),
        f32::from_le_bytes(bin[col_off + 4..col_off + 8].try_into().unwrap()),
        f32::from_le_bytes(bin[col_off + 8..col_off + 12].try_into().unwrap()),
        f32::from_le_bytes(bin[col_off + 12..col_off + 16].try_into().unwrap()),
    ];
    let expected_rgba = cmap.map_value(0.0, (0.0, 1.0));
    for c in 0..4 {
        assert!((first_rgba[c] - expected_rgba[c]).abs() < 1e-6);
    }

    // Validate INDICES
    let idx_acc = &accessors[idx_acc_idx];
    assert_eq!(idx_acc["type"], "SCALAR");
    assert_eq!(idx_acc["componentType"], 5125);
    assert_eq!(idx_acc["count"], 12); // 2 polys (6) + 1 strip of 4 verts -> 2 tris (6)
}

fn parse_and_validate_glb(bytes: &[u8]) -> (serde_json::Value, Vec<u8>) {
    assert!(bytes.len() >= 28, "GLB too short");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let total_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert_eq!(magic, 0x4654_6C67);
    assert_eq!(version, 2);
    assert_eq!(total_len, bytes.len());
    assert_eq!(total_len % 4, 0);

    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_type = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    assert_eq!(json_type, 0x4E4F_534A);
    assert_eq!(json_len % 4, 0);
    let json_slice = &bytes[20..20 + json_len];
    let json: serde_json::Value = serde_json::from_slice(json_slice).unwrap();

    let bin_hdr = 20 + json_len;
    let bin_len = u32::from_le_bytes(bytes[bin_hdr..bin_hdr + 4].try_into().unwrap()) as usize;
    let bin_type = u32::from_le_bytes(bytes[bin_hdr + 4..bin_hdr + 8].try_into().unwrap());
    assert_eq!(bin_type, 0x004E_4942);
    assert_eq!(bin_len % 4, 0);
    assert_eq!(bin_hdr + 8 + bin_len, bytes.len());
    let bin = bytes[bin_hdr + 8..].to_vec();

    assert_eq!(json["asset"]["version"], "2.0");
    assert_eq!(json["buffers"][0]["byteLength"].as_u64().unwrap() as usize, bin_len);

    for bv in json["bufferViews"].as_array().unwrap() {
        let off = bv["byteOffset"].as_u64().unwrap() as usize;
        let len = bv["byteLength"].as_u64().unwrap() as usize;
        assert_eq!(off % 4, 0, "bufferView byteOffset not 4-byte aligned");
        assert!(off + len <= bin_len, "bufferView out of bounds");
    }

    (json, bin)
}

#[test]
fn xdmf_variants_and_reader_selection() {
    use paraviz_io::{detect_reader, read_dataset_with_reader, DatasetReader};

    let dir = TestDir::new("xdmf_variants");

    // 1) X_Y_Z + Mixed topology + CDATA + Vector attribute + Reference
    let xdmf_mixed = r#"<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="MixedMesh" GridType="Uniform">
      <Topology TopologyType="Mixed" NumberOfElements="2">
        <DataItem Format="XML" Dimensions="9">
          4 0 1 2  5 1 2 3 4
        </DataItem>
      </Topology>
      <Geometry GeometryType="X_Y_Z">
        <DataItem Name="X" Format="XML" Dimensions="5"><![CDATA[0 1 0 1 2]]></DataItem>
        <DataItem Name="Y" Format="XML" Dimensions="5">0 0 1 1 0.5</DataItem>
        <DataItem Name="Z" Format="XML" Dimensions="5">0 0 0 0 0</DataItem>
      </Geometry>
      <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
        <DataItem Format="XML" Dimensions="5 3">
          1 0 0  2 0 0  3 0 0  4 0 0  5 0 0
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>"#;
    let mixed_path = dir.file("mixed.xmf");
    fs::write(&mixed_path, xdmf_mixed).unwrap();
    assert_eq!(detect_reader(&mixed_path).unwrap(), DatasetReader::Xdmf);
    let (ds, reader) = read_dataset_with_reader(&mixed_path, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Xdmf);
    let ug = ds.as_unstructured_grid().unwrap();
    assert_eq!(ug.num_points(), 5);
    assert_eq!(ug.num_cells(), 2);
    assert_eq!(ug.cell_types, vec![CellType::Triangle, CellType::Quad]);
    let vel = ug.point_data.get("Velocity").unwrap();
    assert_eq!(vel.num_components, 3);
    assert_eq!(vel.num_tuples(), 5);

    // 2) 3DCoRectMesh + ORIGIN_DXDYDZ -> ImageData
    let xdmf_img = r#"<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="Uniform" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="2 3 4"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">1.0 2.0 3.0</DataItem>
        <DataItem Format="XML" Dimensions="3">0.5 0.25 0.1</DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>"#;
    let img_path = dir.file("uniform.xdmf");
    fs::write(&img_path, xdmf_img).unwrap();
    let img_ds = read_dataset(&img_path).unwrap();
    let img = img_ds.as_image_data().unwrap();
    assert_eq!(img.dimensions, [4, 3, 2]);
    assert_eq!(img.origin, [3.0, 2.0, 1.0]);
    assert_eq!(img.spacing, [0.1, 0.25, 0.5]);

    // 3) Binary f64 BigEndian + Seek + Reference + .xml extension auto-detected as XDMF
    let bin_path = dir.file("pts_be.bin");
    let mut bin_bytes = vec![0xAAu8; 8]; // 8-byte Seek header
    for val in [0.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin_bytes.extend_from_slice(&val.to_be_bytes());
    }
    fs::write(&bin_path, &bin_bytes).unwrap();

    let xml_xdmf = r#"<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <DataItem Name="SharedPts" Format="Binary" NumberType="Float" Precision="8" Endian="Big" Seek="8" Dimensions="3 3">pts_be.bin</DataItem>
    <Grid Name="Tri" GridType="Uniform">
      <Topology TopologyType="Triangle" NumberOfElements="1">
        <DataItem Format="XML" Dimensions="1 3">0 1 2</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name='SharedPts']</DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>"#;
    let xml_path = dir.file("custom_ext.xml");
    fs::write(&xml_path, xml_xdmf).unwrap();
    assert_eq!(detect_reader(&xml_path).unwrap(), DatasetReader::Xdmf);
    let (forced_ds, forced_reader) =
        read_dataset_with_reader(&xml_path, DatasetReader::Xdmf).unwrap();
    assert_eq!(forced_reader, DatasetReader::Xdmf);
    let tri_ug = forced_ds.as_unstructured_grid().unwrap();
    assert_eq!(tri_ug.num_points(), 3);
    assert_eq!(tri_ug.num_cells(), 1);
    assert_eq!(tri_ug.points[1], [1.0, 0.0, 0.0]);
}

#[test]
fn truncated_and_incomplete_files_return_error_without_panicking() {
    let dir = TestDir::new("truncated_files");

    // 1) Truncated XDMF XML mid-tag / unclosed
    let trunc_xmf = dir.file("trunc.xmf");
    fs::write(
        &trunc_xmf,
        r#"<?xml version="1.0" ?><Xdmf Version="2.0"><Domain><Grid Name="G"><Topology TopologyType="Triangle"><DataItem>"#,
    )
    .unwrap();
    assert!(read_dataset(&trunc_xmf).is_err());

    // 2) Truncated / garbage HDF5 referenced by XDMF
    let bad_h5 = dir.file("half_written.h5");
    fs::write(&bad_h5, b"\x89HDF\r\n\x1a\n\x00\x00\x00\x00truncated").unwrap();
    let xmf_h5 = dir.file("half.xmf");
    fs::write(
        &xmf_h5,
        r#"<?xml version="1.0" ?>
<Xdmf Version="2.0"><Domain><Grid Name="G" GridType="Uniform">
  <Topology TopologyType="Triangle" NumberOfElements="1">
    <DataItem Dimensions="3" NumberType="Int" Format="HDF">half_written.h5:/topo</DataItem>
  </Topology>
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="9" NumberType="Float" Format="HDF">half_written.h5:/xyz</DataItem>
  </Geometry>
</Grid></Domain></Xdmf>"#,
    )
    .unwrap();
    assert!(read_dataset(&xmf_h5).is_err());

    // 3) Out-of-range cell vertex index (half-written points vs topology)
    let oob_xmf = dir.file("oob.xmf");
    fs::write(
        &oob_xmf,
        r#"<?xml version="1.0" ?>
<Xdmf Version="2.0"><Domain><Grid Name="G" GridType="Uniform">
  <Topology TopologyType="Triangle" NumberOfElements="1">
    <DataItem Dimensions="3" Format="XML">0 1 999</DataItem>
  </Topology>
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="6" Format="XML">0 0 0  1 0 0</DataItem>
  </Geometry>
</Grid></Domain></Xdmf>"#,
    )
    .unwrap();
    assert!(read_dataset(&oob_xmf).is_err());
}

#[test]
fn xdmf_temporal_collection_time_steps_and_source() {
    use paraviz_core::Source;
    use paraviz_io::{inspect_xdmf_time_steps, read_xdmf_at_time};

    let dir = TestDir::new("xdmf_temporal");
    let xml = r#"<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Grid Name="Step0" GridType="Uniform">
        <Time Value="0.5"/>
        <Topology TopologyType="Triangle" NumberOfElements="1">
          <DataItem Format="XML" Dimensions="3">0 1 2</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="XML" Dimensions="9">0 0 0  1 0 0  0 1 0</DataItem>
        </Geometry>
        <Attribute Name="Temp" Center="Node">
          <DataItem Format="XML" Dimensions="3">10.0 10.0 10.0</DataItem>
        </Attribute>
      </Grid>
      <Grid Name="Step1" GridType="Uniform">
        <Time Value="1.25"/>
        <Topology TopologyType="Triangle" NumberOfElements="1">
          <DataItem Format="XML" Dimensions="3">0 1 2</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="XML" Dimensions="9">0 0 0  1 0 0  0 1 0</DataItem>
        </Geometry>
        <Attribute Name="Temp" Center="Node">
          <DataItem Format="XML" Dimensions="3">42.0 42.0 42.0</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>"#;
    let path = dir.file("series.xmf");
    fs::write(&path, xml).unwrap();

    let steps = inspect_xdmf_time_steps(&path).unwrap();
    assert_eq!(steps, vec![0.5, 1.25]);

    let ds0 = read_xdmf_at_time(&path, 0.5).unwrap();
    assert!(
        (ds0.as_unstructured_grid()
            .unwrap()
            .point_data
            .get("Temp")
            .unwrap()
            .get_scalar(0)
            - 10.0)
            .abs()
            < 1e-6
    );

    let ds1 = read_xdmf_at_time(&path, 1.25).unwrap();
    assert!(
        (ds1.as_unstructured_grid()
            .unwrap()
            .point_data
            .get("Temp")
            .unwrap()
            .get_scalar(0)
            - 42.0)
            .abs()
            < 1e-6
    );

    let src = VtkFileSource::new(&path);
    let info = src.request_information();
    assert_eq!(info.time_steps, vec![0.5, 1.25]);
    let s1 = src.execute(1.25).unwrap();
    assert!(
        (s1.as_unstructured_grid()
            .unwrap()
            .point_data
            .get("Temp")
            .unwrap()
            .get_scalar(0)
            - 42.0)
            .abs()
            < 1e-6
    );
}

// ============================================================================
// Pure-Rust HDF5 v2 & NetCDF Classic / CDF-2 test fixture builders
// ============================================================================

struct Hdf5Builder {
    buf: Vec<u8>,
}

impl Hdf5Builder {
    fn new() -> Self {
        // Reserve 48 bytes for v2 superblock
        Self { buf: vec![0u8; 48] }
    }

    fn finish(mut self, root_addr: u64) -> Vec<u8> {
        let total_len = self.buf.len() as u64;
        let mut sb = Vec::with_capacity(48);
        sb.extend_from_slice(b"\x89HDF\r\n\x1a\n");
        sb.push(2); // version
        sb.push(8); // offset_size
        sb.push(8); // length_size
        sb.push(0); // consistency_flags
        sb.extend_from_slice(&0u64.to_le_bytes()); // base_address
        sb.extend_from_slice(&u64::MAX.to_le_bytes()); // extension_address
        sb.extend_from_slice(&total_len.to_le_bytes()); // eof_address
        sb.extend_from_slice(&root_addr.to_le_bytes());
        let csum = hdf5_reader::checksum::jenkins_lookup3(&sb[..44]);
        sb.extend_from_slice(&csum.to_le_bytes());
        self.buf[..48].copy_from_slice(&sb);
        self.buf
    }

    fn write_blob(&mut self, data: &[u8]) -> u64 {
        let addr = self.buf.len() as u64;
        self.buf.extend_from_slice(data);
        addr
    }

    fn write_ohdr_v2(&mut self, msgs: &[(u8, Vec<u8>)]) -> u64 {
        let addr = self.buf.len() as u64;
        let mut payload = Vec::new();
        for (ty, data) in msgs {
            payload.push(*ty);
            payload.extend_from_slice(&(data.len() as u16).to_le_bytes());
            payload.push(0); // msg_flags
            payload.extend_from_slice(data);
        }
        let mut ohdr = Vec::new();
        ohdr.extend_from_slice(b"OHDR");
        ohdr.push(2); // version
        ohdr.push(0x02); // flags: 4-byte chunk0_size, no timestamps, no attr phase change, no creation order
        ohdr.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        ohdr.extend_from_slice(&payload);
        let csum = hdf5_reader::checksum::jenkins_lookup3(&ohdr);
        ohdr.extend_from_slice(&csum.to_le_bytes());
        self.buf.extend_from_slice(&ohdr);
        addr
    }

    fn link_msg(name: &str, target_addr: u64) -> (u8, Vec<u8>) {
        let mut m = Vec::new();
        m.push(1); // version 1
        m.push(0x00); // flags: 1-byte name len, hard link
        m.push(name.len() as u8);
        m.extend_from_slice(name.as_bytes());
        m.extend_from_slice(&target_addr.to_le_bytes());
        (0x06, m)
    }

    fn dtype_f32() -> Vec<u8> {
        let mut m = Vec::new();
        // class=1 (float), version=1, flags=0 (LE)
        m.extend_from_slice(&((1u32) | (1u32 << 4)).to_le_bytes());
        m.extend_from_slice(&4u32.to_le_bytes()); // size=4
        // properties: bit_offset(0), bit_precision(32), exp_loc(23), exp_size(8), mant_loc(0), mant_size(23), exp_bias(127)
        m.extend_from_slice(&0u16.to_le_bytes());
        m.extend_from_slice(&32u16.to_le_bytes());
        m.push(23);
        m.push(8);
        m.push(0);
        m.push(23);
        m.extend_from_slice(&127u32.to_le_bytes());
        m
    }

    fn dtype_f64() -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&((1u32) | (1u32 << 4)).to_le_bytes());
        m.extend_from_slice(&8u32.to_le_bytes());
        m.extend_from_slice(&0u16.to_le_bytes());
        m.extend_from_slice(&64u16.to_le_bytes());
        m.push(52);
        m.push(11);
        m.push(0);
        m.push(52);
        m.extend_from_slice(&1023u32.to_le_bytes());
        m
    }

    fn dtype_i32() -> Vec<u8> {
        let mut m = Vec::new();
        // class=0, version=1, flags=0x08 (signed, LE) => class_and_flags = (1<<4) | (0x08<<8)
        m.extend_from_slice(&((1u32 << 4) | (0x08u32 << 8)).to_le_bytes());
        m.extend_from_slice(&4u32.to_le_bytes());
        m.extend_from_slice(&0u16.to_le_bytes());
        m.extend_from_slice(&32u16.to_le_bytes());
        m
    }

    fn dtype_i64() -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&((1u32 << 4) | (0x08u32 << 8)).to_le_bytes());
        m.extend_from_slice(&8u32.to_le_bytes());
        m.extend_from_slice(&0u16.to_le_bytes());
        m.extend_from_slice(&64u16.to_le_bytes());
        m
    }

    fn dtype_u8() -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&(1u32 << 4).to_le_bytes());
        m.extend_from_slice(&1u32.to_le_bytes());
        m.extend_from_slice(&0u16.to_le_bytes());
        m.extend_from_slice(&8u16.to_le_bytes());
        m
    }

    fn dtype_fixed_str(len: usize) -> Vec<u8> {
        let mut m = Vec::new();
        // class=3 (string), version=1, padding=0 (NullTerminate), charset=0 (ASCII)
        m.extend_from_slice(&(3u32 | (1u32 << 4)).to_le_bytes());
        m.extend_from_slice(&(len as u32).to_le_bytes());
        m
    }

    fn dataspace_v2(dims: &[u64]) -> Vec<u8> {
        let mut m = Vec::new();
        m.push(2); // version 2
        m.push(dims.len() as u8); // rank
        m.push(0); // flags
        m.push(if dims.is_empty() { 0 } else { 1 }); // Scalar=0, Simple=1
        for &d in dims {
            m.extend_from_slice(&d.to_le_bytes());
        }
        m
    }

    fn attr_v2(name: &str, dt: &[u8], dims: &[u64], raw: &[u8]) -> (u8, Vec<u8>) {
        let ds = Self::dataspace_v2(dims);
        let name_bytes = format!("{name}\0");
        let mut m = Vec::new();
        m.push(2); // version 2
        m.push(0); // flags
        m.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        m.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        m.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        m.extend_from_slice(name_bytes.as_bytes());
        m.extend_from_slice(dt);
        m.extend_from_slice(&ds);
        m.extend_from_slice(raw);
        (0x0C, m)
    }

    fn attr_str(name: &str, val: &str) -> (u8, Vec<u8>) {
        let s = format!("{val}\0");
        let dt = Self::dtype_fixed_str(s.len());
        Self::attr_v2(name, &dt, &[], s.as_bytes())
    }

    fn attr_i32_1d(name: &str, vals: &[i32]) -> (u8, Vec<u8>) {
        let dt = Self::dtype_i32();
        let mut raw = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        Self::attr_v2(name, &dt, &[vals.len() as u64], &raw)
    }

    fn attr_f64_1d(name: &str, vals: &[f64]) -> (u8, Vec<u8>) {
        let dt = Self::dtype_f64();
        let mut raw = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        Self::attr_v2(name, &dt, &[vals.len() as u64], &raw)
    }

    fn add_dataset(&mut self, dt: &[u8], dims: &[u64], raw: &[u8], attrs: Vec<(u8, Vec<u8>)>) -> u64 {
        let data_addr = self.write_blob(raw);
        let ds = Self::dataspace_v2(dims);
        let mut layout = Vec::new();
        layout.push(3); // version 3
        layout.push(1); // contiguous
        layout.extend_from_slice(&data_addr.to_le_bytes());
        layout.extend_from_slice(&(raw.len() as u64).to_le_bytes());

        let mut msgs = vec![
            (0x01, ds),
            (0x03, dt.to_vec()),
            (0x08, layout),
        ];
        msgs.extend(attrs);
        self.write_ohdr_v2(&msgs)
    }

    fn add_group(&mut self, children: &[(&str, u64)], attrs: Vec<(u8, Vec<u8>)>) -> u64 {
        let mut msgs = Vec::new();
        for &(name, addr) in children {
            msgs.push(Self::link_msg(name, addr));
        }
        msgs.extend(attrs);
        self.write_ohdr_v2(&msgs)
    }
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn f64_bytes(vals: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn i64_bytes(vals: &[i64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn i32_bytes(vals: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

struct NcBuilder {
    is_cdf2: bool,
    numrecs: u32,
    dims: Vec<(String, u32)>,
    vars: Vec<NcVarSpec>,
}

struct NcVarSpec {
    name: String,
    dimids: Vec<u32>,
    attrs: Vec<(String, u32, Vec<u8>)>, // name, nc_type, raw_be_bytes
    nc_type: u32,
    is_record: bool,
    data_be: Vec<u8>, // flat BE bytes (for record var, numrecs concatenated records)
}

impl NcBuilder {
    fn new(is_cdf2: bool, numrecs: u32) -> Self {
        Self {
            is_cdf2,
            numrecs,
            dims: Vec::new(),
            vars: Vec::new(),
        }
    }

    fn add_dim(&mut self, name: &str, len: u32) -> u32 {
        let id = self.dims.len() as u32;
        self.dims.push((name.to_string(), len));
        id
    }

    fn add_var(
        &mut self,
        name: &str,
        dimids: Vec<u32>,
        nc_type: u32,
        attrs: Vec<(String, u32, Vec<u8>)>,
        data_be: Vec<u8>,
    ) {
        let is_record = !dimids.is_empty() && self.dims[dimids[0] as usize].1 == 0;
        self.vars.push(NcVarSpec {
            name: name.to_string(),
            dimids,
            attrs,
            nc_type,
            is_record,
            data_be,
        });
    }

    fn write_nc_string(out: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_be_bytes());
        out.extend_from_slice(b);
        let pad = (4 - (b.len() % 4)) % 4;
        out.extend(std::iter::repeat_n(0u8, pad));
    }

    fn write_attrs(out: &mut Vec<u8>, attrs: &[(String, u32, Vec<u8>)]) {
        if attrs.is_empty() {
            out.extend_from_slice(&0u32.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            return;
        }
        out.extend_from_slice(&12u32.to_be_bytes());
        out.extend_from_slice(&(attrs.len() as u32).to_be_bytes());
        for (name, nc_type, raw) in attrs {
            Self::write_nc_string(out, name);
            out.extend_from_slice(&nc_type.to_be_bytes());
            let elem_sz = match nc_type {
                1 | 2 => 1,
                3 => 2,
                4 | 5 => 4,
                6 => 8,
                _ => 1,
            };
            out.extend_from_slice(&((raw.len() / elem_sz) as u32).to_be_bytes());
            out.extend_from_slice(raw);
            let pad = (4 - (raw.len() % 4)) % 4;
            out.extend(std::iter::repeat_n(0u8, pad));
        }
    }

    fn build(self) -> Vec<u8> {
        let mut hdr = Vec::new();
        hdr.extend_from_slice(if self.is_cdf2 { b"CDF\x02" } else { b"CDF\x01" });
        hdr.extend_from_slice(&self.numrecs.to_be_bytes());

        // dim_list
        if self.dims.is_empty() {
            hdr.extend_from_slice(&0u32.to_be_bytes());
            hdr.extend_from_slice(&0u32.to_be_bytes());
        } else {
            hdr.extend_from_slice(&10u32.to_be_bytes());
            hdr.extend_from_slice(&(self.dims.len() as u32).to_be_bytes());
            for (name, len) in &self.dims {
                Self::write_nc_string(&mut hdr, name);
                hdr.extend_from_slice(&len.to_be_bytes());
            }
        }

        // global attrs (empty)
        Self::write_attrs(&mut hdr, &[]);

        // var_list with placeholder begin offsets
        if self.vars.is_empty() {
            hdr.extend_from_slice(&0u32.to_be_bytes());
            hdr.extend_from_slice(&0u32.to_be_bytes());
        } else {
            hdr.extend_from_slice(&11u32.to_be_bytes());
            hdr.extend_from_slice(&(self.vars.len() as u32).to_be_bytes());
        }

        let mut begin_positions = Vec::with_capacity(self.vars.len());
        let mut var_vsizes = Vec::with_capacity(self.vars.len());

        for v in &self.vars {
            Self::write_nc_string(&mut hdr, &v.name);
            hdr.extend_from_slice(&(v.dimids.len() as u32).to_be_bytes());
            for &did in &v.dimids {
                hdr.extend_from_slice(&did.to_be_bytes());
            }
            Self::write_attrs(&mut hdr, &v.attrs);
            hdr.extend_from_slice(&v.nc_type.to_be_bytes());

            let per_rec_bytes = if v.is_record {
                let nr = self.numrecs.max(1) as usize;
                v.data_be.len() / nr
            } else {
                v.data_be.len()
            };
            let padded_vsize = ((per_rec_bytes + 3) & !3) as u32;
            var_vsizes.push((per_rec_bytes, padded_vsize));
            hdr.extend_from_slice(&padded_vsize.to_be_bytes());
            begin_positions.push(hdr.len());
            if self.is_cdf2 {
                hdr.extend_from_slice(&0u64.to_be_bytes());
            } else {
                hdr.extend_from_slice(&0u32.to_be_bytes());
            }
        }

        let mut file = hdr;
        // Write fixed variables first, then interleaved record variables
        for (i, v) in self.vars.iter().enumerate() {
            if v.is_record {
                continue;
            }
            let start = file.len() as u64;
            let pos = begin_positions[i];
            if self.is_cdf2 {
                file[pos..pos + 8].copy_from_slice(&start.to_be_bytes());
            } else {
                file[pos..pos + 4].copy_from_slice(&(start as u32).to_be_bytes());
            }
            file.extend_from_slice(&v.data_be);
            let pad = (4 - (v.data_be.len() % 4)) % 4;
            file.extend(std::iter::repeat_n(0u8, pad));
        }

        let mut rec_offset = file.len() as u64;
        for (i, v) in self.vars.iter().enumerate() {
            if !v.is_record {
                continue;
            }
            let pos = begin_positions[i];
            if self.is_cdf2 {
                file[pos..pos + 8].copy_from_slice(&rec_offset.to_be_bytes());
            } else {
                file[pos..pos + 4].copy_from_slice(&(rec_offset as u32).to_be_bytes());
            }
            rec_offset += var_vsizes[i].1 as u64;
        }

        for r in 0..(self.numrecs as usize) {
            for (i, v) in self.vars.iter().enumerate() {
                if !v.is_record {
                    continue;
                }
                let (per_rec_bytes, padded_vsize) = var_vsizes[i];
                let s = r * per_rec_bytes;
                file.extend_from_slice(&v.data_be[s..s + per_rec_bytes]);
                let pad = (padded_vsize as usize) - per_rec_bytes;
                file.extend(std::iter::repeat_n(0u8, pad));
            }
        }

        file
    }
}

#[test]
fn vtkhdf_imagedata_polydata_ugrid_composite_and_temporal() {
    use paraviz_core::Source;
    use paraviz_io::{
        detect_reader, inspect_vtkhdf_time_steps, read_dataset_with_reader, read_vtkhdf,
        read_vtkhdf_at_time, DatasetReader,
    };

    let dir = TestDir::new("vtkhdf");

    // 1) ImageData (static)
    {
        let mut b = Hdf5Builder::new();
        let dens = b.add_dataset(
            &Hdf5Builder::dtype_f32(),
            &[2, 2, 3],
            &f32_bytes(&(0..12).map(|i| i as f32).collect::<Vec<_>>()),
            vec![],
        );
        let pd = b.add_group(&[("density", dens)], vec![]);
        let cval = b.add_dataset(
            &Hdf5Builder::dtype_f64(),
            &[1, 1, 2],
            &f64_bytes(&[10.0, 20.0]),
            vec![],
        );
        let cd = b.add_group(&[("cell_val", cval)], vec![]);
        let vtkhdf_grp = b.add_group(
            &[("PointData", pd), ("CellData", cd)],
            vec![
                Hdf5Builder::attr_str("Type", "ImageData"),
                Hdf5Builder::attr_i32_1d("WholeExtent", &[0, 2, 0, 1, 0, 1]),
                Hdf5Builder::attr_f64_1d("Origin", &[-1.0, 0.5, 2.0]),
                Hdf5Builder::attr_f64_1d("Spacing", &[0.5, 0.25, 1.0]),
            ],
        );
        let root = b.add_group(&[("VTKHDF", vtkhdf_grp)], vec![]);
        let bytes = b.finish(root);

        let p = dir.file("img.vtkhdf");
        fs::write(&p, &bytes).unwrap();
        assert_eq!(detect_reader(&p).unwrap(), DatasetReader::VtkHdf);
        let ds = read_vtkhdf(&p).unwrap();
        let img = ds.as_image_data().unwrap();
        assert_eq!(img.dimensions, [3, 2, 2]);
        assert_eq!(img.origin, [-1.0, 0.5, 2.0]);
        assert_eq!(img.spacing, [0.5, 0.25, 1.0]);
        assert_eq!(img.point_data.get("density").unwrap().num_tuples(), 12);
        assert_eq!(img.cell_data.get("cell_val").unwrap().num_tuples(), 2);

        // Also verify .h5 auto-detection for VTKHDF
        let h5_p = dir.file("img_auto.h5");
        fs::write(&h5_p, &bytes).unwrap();
        assert_eq!(detect_reader(&h5_p).unwrap(), DatasetReader::VtkHdf);
    }

    // 2) Temporal PolyData with 2 time steps
    {
        let mut b = Hdf5Builder::new();
        let npts = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[3, 3]), vec![]);
        let pts = b.add_dataset(
            &Hdf5Builder::dtype_f32(),
            &[6, 3],
            &f32_bytes(&[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // step 0
                0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, // step 1
            ]),
            vec![],
        );
        let p_ncells = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[1, 1]), vec![]);
        let p_nconn = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[3, 3]), vec![]);
        let p_offs = b.add_dataset(&Hdf5Builder::dtype_i64(), &[4], &i64_bytes(&[0, 3, 0, 3]), vec![]);
        let p_conn = b.add_dataset(&Hdf5Builder::dtype_i64(), &[6], &i64_bytes(&[0, 1, 2, 0, 1, 2]), vec![]);
        let polygons = b.add_group(
            &[
                ("NumberOfCells", p_ncells),
                ("NumberOfConnectivityIds", p_nconn),
                ("Offsets", p_offs),
                ("Connectivity", p_conn),
            ],
            vec![],
        );
        let temp_ds = b.add_dataset(
            &Hdf5Builder::dtype_f64(),
            &[6],
            &f64_bytes(&[10.0, 10.0, 10.0, 99.0, 99.0, 99.0]),
            vec![],
        );
        let pd = b.add_group(&[("Temp", temp_ds)], vec![]);

        let s_vals = b.add_dataset(&Hdf5Builder::dtype_f64(), &[2], &f64_bytes(&[0.25, 1.5]), vec![]);
        let s_poff = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[0, 3]), vec![]);
        let s_partoff = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[0, 1]), vec![]);
        let s_nparts = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[1, 1]), vec![]);
        let s_coff = b.add_dataset(
            &Hdf5Builder::dtype_i64(),
            &[2, 4],
            &i64_bytes(&[0, 0, 0, 0, 0, 0, 1, 0]),
            vec![],
        );
        let s_cid = b.add_dataset(
            &Hdf5Builder::dtype_i64(),
            &[2, 4],
            &i64_bytes(&[0, 0, 0, 0, 0, 0, 3, 0]),
            vec![],
        );
        let s_pdo = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[0, 3]), vec![]);
        let pd_offs = b.add_group(&[("Temp", s_pdo)], vec![]);
        let steps = b.add_group(
            &[
                ("Values", s_vals),
                ("PointOffsets", s_poff),
                ("PartOffsets", s_partoff),
                ("NumberOfParts", s_nparts),
                ("CellOffsets", s_coff),
                ("ConnectivityIdOffsets", s_cid),
                ("PointDataOffsets", pd_offs),
            ],
            vec![],
        );
        let vtkhdf_grp = b.add_group(
            &[
                ("NumberOfPoints", npts),
                ("Points", pts),
                ("Polygons", polygons),
                ("PointData", pd),
                ("Steps", steps),
            ],
            vec![Hdf5Builder::attr_str("Type", "PolyData")],
        );
        let root = b.add_group(&[("VTKHDF", vtkhdf_grp)], vec![]);
        let p = dir.file("poly_steps.vtkhdf");
        fs::write(&p, b.finish(root)).unwrap();

        assert_eq!(inspect_vtkhdf_time_steps(&p).unwrap(), vec![0.25, 1.5]);
        let ds1 = read_vtkhdf_at_time(&p, 1.5).unwrap();
        let poly1 = ds1.as_poly_data().unwrap();
        assert_eq!(poly1.num_points(), 3);
        assert_eq!(poly1.polys, vec![vec![0, 1, 2]]);
        assert_eq!(poly1.points[1], [2.0, 0.0, 1.0]);
        assert!((poly1.point_data.get("Temp").unwrap().get_scalar(0) - 99.0).abs() < 1e-6);

        let src = VtkFileSource::new(&p);
        assert_eq!(src.request_information().time_steps, vec![0.25, 1.5]);
        let s0 = src.execute(0.25).unwrap();
        assert!((s0.as_poly_data().unwrap().point_data.get("Temp").unwrap().get_scalar(0) - 10.0).abs() < 1e-6);
    }

    // 3) UnstructuredGrid inside MultiBlockDataSet
    {
        let mut b = Hdf5Builder::new();
        let npts = b.add_dataset(&Hdf5Builder::dtype_i64(), &[1], &i64_bytes(&[4]), vec![]);
        let pts = b.add_dataset(
            &Hdf5Builder::dtype_f32(),
            &[4, 3],
            &f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            vec![],
        );
        let ncells = b.add_dataset(&Hdf5Builder::dtype_i64(), &[1], &i64_bytes(&[1]), vec![]);
        let types = b.add_dataset(&Hdf5Builder::dtype_u8(), &[1], &[10u8], vec![]); // VTK_TETRA=10
        let offs = b.add_dataset(&Hdf5Builder::dtype_i64(), &[2], &i64_bytes(&[0, 4]), vec![]);
        let conn = b.add_dataset(&Hdf5Builder::dtype_i64(), &[4], &i64_bytes(&[0, 1, 2, 3]), vec![]);
        let mat = b.add_dataset(&Hdf5Builder::dtype_i32(), &[1], &i32_bytes(&[7]), vec![]);
        let cd = b.add_group(&[("mat", mat)], vec![]);
        let block0 = b.add_group(
            &[
                ("NumberOfPoints", npts),
                ("Points", pts),
                ("NumberOfCells", ncells),
                ("Types", types),
                ("Offsets", offs),
                ("Connectivity", conn),
                ("CellData", cd),
            ],
            vec![Hdf5Builder::attr_str("Type", "UnstructuredGrid")],
        );
        let mb = b.add_group(
            &[("TetBlock", block0)],
            vec![Hdf5Builder::attr_str("Type", "MultiBlockDataSet")],
        );
        let root = b.add_group(&[("VTKHDF", mb)], vec![]);
        let p = dir.file("composite.vtkhdf");
        fs::write(&p, b.finish(root)).unwrap();

        let (ds, r) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
        assert_eq!(r, DatasetReader::VtkHdf);
        let DataSet::MultiBlock(mb_ds) = ds else {
            panic!("expected MultiBlock");
        };
        assert_eq!(mb_ds.blocks.len(), 1);
        assert_eq!(mb_ds.blocks[0].0, "TetBlock");
        let ug = mb_ds.blocks[0].1.as_unstructured_grid().unwrap();
        assert_eq!(ug.cell_types, vec![CellType::Tetra]);
        assert_eq!(ug.cell_data.get("mat").unwrap().get_scalar(0) as i32, 7);
    }
}

#[test]
fn cgns_structured_and_unstructured_zones() {
    use paraviz_io::{detect_reader, read_cgns, DatasetReader};

    let dir = TestDir::new("cgns");

    // Structured zone + Unstructured zone (MIXED + TRI_3) in one CGNS file
    let mut b = Hdf5Builder::new();
    let lib_ver = b.add_dataset(&Hdf5Builder::dtype_f32(), &[1], &f32_bytes(&[4.2]), vec![]);

    // Zone 1: Structured 2x2x2
    let z1_data = b.add_dataset(&Hdf5Builder::dtype_i32(), &[3, 3], &i32_bytes(&[2, 2, 2, 1, 1, 1, 0, 0, 0]), vec![]);
    let z1_zt_data = b.add_dataset(&Hdf5Builder::dtype_u8(), &[10], b"Structured", vec![]);
    let z1_zt = b.add_group(&[(" data", z1_zt_data)], vec![Hdf5Builder::attr_str("label", "ZoneType_t")]);
    let cx = b.add_dataset(&Hdf5Builder::dtype_f32(), &[2, 2, 2], &f32_bytes(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), vec![]);
    let cy = b.add_dataset(&Hdf5Builder::dtype_f32(), &[2, 2, 2], &f32_bytes(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]), vec![]);
    let cz = b.add_dataset(&Hdf5Builder::dtype_f32(), &[2, 2, 2], &f32_bytes(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), vec![]);
    let z1_gc = b.add_group(
        &[("CoordinateX", cx), ("CoordinateY", cy), ("CoordinateZ", cz)],
        vec![Hdf5Builder::attr_str("label", "GridCoordinates_t")],
    );
    let pres_d = b.add_dataset(&Hdf5Builder::dtype_f64(), &[2, 2, 2], &f64_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), vec![]);
    let pres_g = b.add_group(&[(" data", pres_d)], vec![Hdf5Builder::attr_str("label", "DataArray_t")]);
    let gl_v_d = b.add_dataset(&Hdf5Builder::dtype_u8(), &[6], b"Vertex", vec![]);
    let gl_v = b.add_group(&[(" data", gl_v_d)], vec![Hdf5Builder::attr_str("label", "GridLocation_t")]);
    let z1_fs = b.add_group(
        &[("GridLocation", gl_v), ("Pressure", pres_g)],
        vec![Hdf5Builder::attr_str("label", "FlowSolution_t")],
    );
    let zone1 = b.add_group(
        &[(" data", z1_data), ("ZoneType", z1_zt), ("GridCoordinates", z1_gc), ("FlowSolution", z1_fs)],
        vec![Hdf5Builder::attr_str("label", "Zone_t")],
    );

    // Zone 2: Unstructured with MIXED (1 TRI_3 + 1 QUAD_4) and CellCenter solution
    let z2_data = b.add_dataset(&Hdf5Builder::dtype_i32(), &[3], &i32_bytes(&[4, 2, 0]), vec![]);
    let z2_zt_data = b.add_dataset(&Hdf5Builder::dtype_u8(), &[12], b"Unstructured", vec![]);
    let z2_zt = b.add_group(&[(" data", z2_zt_data)], vec![Hdf5Builder::attr_str("label", "ZoneType_t")]);
    let ucx = b.add_dataset(&Hdf5Builder::dtype_f32(), &[4], &f32_bytes(&[0.0, 1.0, 1.0, 0.0]), vec![]);
    let ucy = b.add_dataset(&Hdf5Builder::dtype_f32(), &[4], &f32_bytes(&[0.0, 0.0, 1.0, 1.0]), vec![]);
    let z2_gc = b.add_group(
        &[("CoordinateX", ucx), ("CoordinateY", ucy)],
        vec![Hdf5Builder::attr_str("label", "GridCoordinates_t")],
    );
    // Elements_t: MIXED (20), Range=[1, 2], Connectivity = [5, 1, 2, 3,  7, 1, 2, 3, 4] (1-based!)
    let el_meta = b.add_dataset(&Hdf5Builder::dtype_i32(), &[2], &i32_bytes(&[20, 0]), vec![]);
    let el_range = b.add_dataset(&Hdf5Builder::dtype_i32(), &[2], &i32_bytes(&[1, 2]), vec![]);
    let el_conn = b.add_dataset(&Hdf5Builder::dtype_i32(), &[9], &i32_bytes(&[5, 1, 2, 3, 7, 1, 2, 3, 4]), vec![]);
    let elems = b.add_group(
        &[(" data", el_meta), ("ElementRange", el_range), ("ElementConnectivity", el_conn)],
        vec![Hdf5Builder::attr_str("label", "Elements_t")],
    );
    let gl_c_d = b.add_dataset(&Hdf5Builder::dtype_u8(), &[10], b"CellCenter", vec![]);
    let gl_c = b.add_group(&[(" data", gl_c_d)], vec![Hdf5Builder::attr_str("label", "GridLocation_t")]);
    let mach_d = b.add_dataset(&Hdf5Builder::dtype_f64(), &[2], &f64_bytes(&[0.8, 1.2]), vec![]);
    let z2_fs = b.add_group(
        &[("GridLocation", gl_c), ("Mach", mach_d)],
        vec![Hdf5Builder::attr_str("label", "FlowSolution_t")],
    );
    let zone2 = b.add_group(
        &[(" data", z2_data), ("ZoneType", z2_zt), ("GridCoordinates", z2_gc), ("MixedElems", elems), ("FlowSolutionCells", z2_fs)],
        vec![Hdf5Builder::attr_str("label", "Zone_t")],
    );

    let base = b.add_group(
        &[("StructuredZone", zone1), ("UnstructuredZone", zone2)],
        vec![Hdf5Builder::attr_str("label", "CGNSBase_t")],
    );
    let root = b.add_group(&[("CGNSLibraryVersion", lib_ver), ("Base", base)], vec![]);
    let bytes = b.finish(root);

    let p = dir.file("flow.cgns");
    fs::write(&p, &bytes).unwrap();
    assert_eq!(detect_reader(&p).unwrap(), DatasetReader::Cgns);

    let h5_p = dir.file("flow_auto.h5");
    fs::write(&h5_p, &bytes).unwrap();
    assert_eq!(detect_reader(&h5_p).unwrap(), DatasetReader::Cgns);

    let ds = read_cgns(&p).unwrap();
    let DataSet::MultiBlock(mb) = ds else {
        panic!("expected MultiBlock");
    };
    assert_eq!(mb.blocks.len(), 2);
    let DataSet::StructuredGrid(sg) = &mb.blocks[0].1 else {
        panic!("expected StructuredGrid");
    };
    assert_eq!(sg.dimensions, [2, 2, 2]);
    assert_eq!(sg.point_data.get("Pressure").unwrap().num_tuples(), 8);

    let ug = mb.blocks[1].1.as_unstructured_grid().unwrap();
    assert_eq!(ug.num_points(), 4);
    assert_eq!(ug.cell_types, vec![CellType::Triangle, CellType::Quad]);
    assert!((ug.cell_data.get("Mach").unwrap().get_scalar(1) - 1.2).abs() < 1e-6);
}

#[test]
fn exodus_classic_cdf2_and_hdf5_multiblock_temporal() {
    use paraviz_core::Source;
    use paraviz_io::{
        detect_reader, inspect_exodus_time_steps, read_dataset_with_reader, read_exodus_at_time,
        DatasetReader,
    };

    let dir = TestDir::new("exodus");

    // 1) NetCDF Classic (CDF\x01) multi-block with nodal/elem vars & time_whole
    let build_exodus_nc = |is_cdf2: bool| -> Vec<u8> {
        let mut nc = NcBuilder::new(is_cdf2, 2);
        let d_time = nc.add_dim("time_step", 0);
        let d_nodes = nc.add_dim("num_nodes", 5);
        let d_el1 = nc.add_dim("num_el_in_blk1", 1);
        let d_npel1 = nc.add_dim("num_nod_per_el1", 3);
        let d_el2 = nc.add_dim("num_el_in_blk2", 1);
        let d_npel2 = nc.add_dim("num_nod_per_el2", 4);
        let d_blk = nc.add_dim("num_el_blk", 2);
        let d_nvar = nc.add_dim("num_nod_var", 1);
        let d_evar = nc.add_dim("num_elem_var", 1);
        let d_str = nc.add_dim("len_string", 8);

        let f64_be = |vals: &[f64]| -> Vec<u8> {
            vals.iter().flat_map(|v| v.to_be_bytes()).collect()
        };
        let i32_be = |vals: &[i32]| -> Vec<u8> {
            vals.iter().flat_map(|v| v.to_be_bytes()).collect()
        };

        nc.add_var("coordx", vec![d_nodes], 6, vec![], f64_be(&[0.0, 1.0, 0.0, 1.0, 2.0]));
        nc.add_var("coordy", vec![d_nodes], 6, vec![], f64_be(&[0.0, 0.0, 1.0, 1.0, 0.5]));
        nc.add_var(
            "connect1",
            vec![d_el1, d_npel1],
            4,
            vec![("elem_type".into(), 2, b"TRI3\0".to_vec())],
            i32_be(&[1, 2, 3]), // 1-based!
        );
        nc.add_var(
            "connect2",
            vec![d_el2, d_npel2],
            4,
            vec![("elem_type".into(), 2, b"QUAD4\0".to_vec())],
            i32_be(&[2, 5, 4, 3]),
        );
        nc.add_var(
            "eb_names",
            vec![d_blk, d_str],
            2,
            vec![],
            b"TriPart\0QuadPrt\0".to_vec(),
        );
        nc.add_var("time_whole", vec![d_time], 6, vec![], f64_be(&[0.1, 2.0]));
        nc.add_var("name_nod_var", vec![d_nvar, d_str], 2, vec![], b"disp_x\0\0".to_vec());
        nc.add_var("name_elem_var", vec![d_evar, d_str], 2, vec![], b"stress\0\0".to_vec());
        nc.add_var(
            "vals_nod_var1",
            vec![d_time, d_nodes],
            6,
            vec![],
            f64_be(&[
                0.0, 0.1, 0.2, 0.3, 0.4, // step 0
                1.0, 1.1, 1.2, 1.3, 1.4, // step 1
            ]),
        );
        nc.add_var(
            "vals_elem_var1eb1",
            vec![d_time, d_el1],
            6,
            vec![],
            f64_be(&[100.0, 250.0]),
        );
        nc.add_var(
            "vals_elem_var1eb2",
            vec![d_time, d_el2],
            6,
            vec![],
            f64_be(&[300.0, 450.0]),
        );
        nc.build()
    };

    for (tag, is_cdf2, ext_file) in [
        ("cdf1", false, "mesh.exo"),
        ("cdf2", true, "mesh_64.ex2"),
    ] {
        let bytes = build_exodus_nc(is_cdf2);
        let p = dir.file(ext_file);
        fs::write(&p, &bytes).unwrap();
        assert_eq!(detect_reader(&p).unwrap(), DatasetReader::Exodus, "{tag}");

        // Extensionless magic detection
        let magic_p = dir.file(&format!("noext_{tag}"));
        fs::write(&magic_p, &bytes).unwrap();
        assert_eq!(detect_reader(&magic_p).unwrap(), DatasetReader::Exodus);

        let steps = inspect_exodus_time_steps(&p).unwrap();
        assert_eq!(steps, vec![0.1, 2.0]);

        let ds1 = read_exodus_at_time(&p, 2.0).unwrap();
        let DataSet::MultiBlock(mb) = ds1 else {
            panic!("expected MultiBlock");
        };
        assert_eq!(mb.blocks.len(), 2);
        assert_eq!(mb.blocks[0].0, "TriPart");
        assert_eq!(mb.blocks[1].0, "QuadPrt");

        let b0 = mb.blocks[0].1.as_unstructured_grid().unwrap();
        assert_eq!(b0.cell_types, vec![CellType::Triangle]);
        assert!((b0.point_data.get("disp_x").unwrap().get_scalar(0) - 1.0).abs() < 1e-6);
        assert!((b0.cell_data.get("stress").unwrap().get_scalar(0) - 250.0).abs() < 1e-6);

        let b1 = mb.blocks[1].1.as_unstructured_grid().unwrap();
        assert_eq!(b1.cell_types, vec![CellType::Quad]);
        assert!((b1.cell_data.get("stress").unwrap().get_scalar(0) - 450.0).abs() < 1e-6);

        let src = VtkFileSource::new(&p);
        assert_eq!(src.request_information().time_steps, vec![0.1, 2.0]);
        let ds0 = src.execute(0.1).unwrap();
        let DataSet::MultiBlock(mb0) = ds0 else {
            panic!("expected MultiBlock");
        };
        let b0_s0 = mb0.blocks[0].1.as_unstructured_grid().unwrap();
        assert!((b0_s0.cell_data.get("stress").unwrap().get_scalar(0) - 100.0).abs() < 1e-6);
    }

    // 2) HDF5-backed NetCDF-4 Exodus II single block (.e)
    {
        let mut b = Hdf5Builder::new();
        let cx = b.add_dataset(&Hdf5Builder::dtype_f64(), &[4], &f64_bytes(&[0.0, 1.0, 0.0, 0.0]), vec![]);
        let cy = b.add_dataset(&Hdf5Builder::dtype_f64(), &[4], &f64_bytes(&[0.0, 0.0, 1.0, 0.0]), vec![]);
        let cz = b.add_dataset(&Hdf5Builder::dtype_f64(), &[4], &f64_bytes(&[0.0, 0.0, 0.0, 1.0]), vec![]);
        let conn1 = b.add_dataset(
            &Hdf5Builder::dtype_i32(),
            &[1, 4],
            &i32_bytes(&[1, 2, 3, 4]),
            vec![Hdf5Builder::attr_str("elem_type", "TETRA")],
        );
        let nod_names = b.add_dataset(&Hdf5Builder::dtype_u8(), &[1, 4], b"phi\0", vec![]);
        let vals_nod1 = b.add_dataset(&Hdf5Builder::dtype_f32(), &[1, 4], &f32_bytes(&[5.0, 6.0, 7.0, 8.0]), vec![]);
        let root = b.add_group(
            &[
                ("coordx", cx),
                ("coordy", cy),
                ("coordz", cz),
                ("connect1", conn1),
                ("name_nod_var", nod_names),
                ("vals_nod_var1", vals_nod1),
            ],
            vec![],
        );
        let bytes = b.finish(root);

        let p = dir.file("single_tet.e");
        fs::write(&p, &bytes).unwrap();
        let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
        assert_eq!(reader, DatasetReader::Exodus);
        let ug = ds.as_unstructured_grid().unwrap();
        assert_eq!(ug.cell_types, vec![CellType::Tetra]);
        assert!((ug.point_data.get("phi").unwrap().get_scalar(3) - 8.0).abs() < 1e-6);
    }
}

#[test]
fn off_roundtrip_and_coff() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("off");
    let coff_src = "\
COFF
# Simple colored triangle
3 1 0
0.0 0.0 0.0 255 0 0 255
1.0 0.0 0.0 0 255 0 255
0.0 1.0 0.0 0 0 255 255
3 0 1 2
";
    let p = dir.file("tri.off");
    fs::write(&p, coff_src).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Off);
    let poly = ds.as_poly_data().unwrap();
    assert_eq!(poly.points.len(), 3);
    assert_eq!(poly.polys, vec![vec![0, 1, 2]]);
    assert!(poly.point_data.get("Colors").is_some());

    let out_p = dir.file("roundtrip.off");
    write_dataset(&out_p, &ds).unwrap();
    let ds2 = read_dataset(&out_p).unwrap();
    let poly2 = ds2.as_poly_data().unwrap();
    assert_eq!(poly2.points.len(), 3);
    assert!(poly2.point_data.get("Colors").is_some());
}

#[test]
fn gltf_and_glb_roundtrip_and_transforms() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("gltf_read");
    let mut poly = sample_polydata();
    poly.point_data.add_array(DataArray::scalars_f32(
        "pressure",
        vec![0.0, 0.25, 0.75, 1.0],
    ));
    let cmap = ColorMap::from_preset(ColorMapPreset::Viridis);
    let glb_path = dir.file("mesh.glb");
    write_glb(&glb_path, &poly, Some(&cmap), Some("pressure")).unwrap();

    let (ds, reader) = read_dataset_with_reader(&glb_path, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Gltf);
    let read_poly = ds.as_poly_data().unwrap();
    assert_eq!(read_poly.points.len(), 4);
    assert!(!read_poly.polys.is_empty());
    assert!(read_poly.point_data.get("Normals").is_some());
    assert!(read_poly.point_data.get("Colors").is_some());
}

#[test]
fn paraview_series_temporal_source() {
    use paraviz_io::{detect_reader, DatasetReader};
    let dir = TestDir::new("series");
    let mut p0 = sample_polydata();
    p0.point_data
        .add_array(DataArray::scalars_f32("t", vec![0.0; 4]));
    let mut p1 = sample_polydata();
    p1.point_data
        .add_array(DataArray::scalars_f32("t", vec![2.5; 4]));

    write_dataset(&dir.file("step_0.vtp"), &DataSet::PolyData(p0)).unwrap();
    write_dataset(&dir.file("step_1.vtp"), &DataSet::PolyData(p1)).unwrap();

    let series_json = r#"{
  "file-series-version": "1.0",
  "files": [
    { "name": "step_0.vtp", "time": 0.0 },
    { "name": "step_1.vtp", "time": 2.5 }
  ]
}"#;
    let series_path = dir.file("sim.vtp.series");
    fs::write(&series_path, series_json).unwrap();

    assert_eq!(detect_reader(&series_path).unwrap(), DatasetReader::Series);
    let src = VtkFileSource::new(&series_path);
    let info = src.request_information();
    assert_eq!(info.time_steps, vec![0.0, 2.5]);
    let ds1 = src.execute(2.5).unwrap();
    let poly1 = ds1.as_poly_data().unwrap();
    assert!((poly1.point_data.get("t").unwrap().get_scalar(0) - 2.5).abs() < 1e-6);
}

#[test]
fn molecular_xyz_and_pdb_temporal() {
    use paraviz_io::{detect_reader, DatasetReader};
    let dir = TestDir::new("mol");
    let xyz_txt = "\
2
frame 0
H 0.0 0.0 0.0
O 1.0 0.0 0.0
2
frame 1
H 0.5 0.0 0.0
O 1.5 0.0 0.0
";
    let xyz_path = dir.file("water.xyz");
    fs::write(&xyz_path, xyz_txt).unwrap();
    assert_eq!(detect_reader(&xyz_path).unwrap(), DatasetReader::Xyz);
    let src = VtkFileSource::new(&xyz_path);
    assert_eq!(src.request_information().time_steps, vec![0.0, 1.0]);
    let ds1 = src.execute(1.0).unwrap();
    let p1 = ds1.as_poly_data().unwrap();
    assert_eq!(p1.points[0][0], 0.5);
    assert_eq!(p1.point_data.get("AtomicNumber").unwrap().get_scalar(1) as i32, 8);

    let pdb_txt = "\
MODEL        1
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 20.00           C
CONECT    1    2
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1       4.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       5.000   2.000   3.000  1.00 20.00           C
ENDMDL
";
    let pdb_path = dir.file("mol.pdb");
    fs::write(&pdb_path, pdb_txt).unwrap();
    assert_eq!(detect_reader(&pdb_path).unwrap(), DatasetReader::Pdb);
    let pdb_src = VtkFileSource::new(&pdb_path);
    assert_eq!(pdb_src.request_information().time_steps, vec![0.0, 1.0]);
    let m0 = pdb_src.execute(0.0).unwrap();
    let mp0 = m0.as_poly_data().unwrap();
    assert_eq!(mp0.points.len(), 2);
    assert_eq!(mp0.lines, vec![vec![0, 1]]);
}

#[test]
fn x3d_indexed_face_set_and_transform() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("x3d");
    let x3d_txt = r#"<?xml version="1.0" encoding="UTF-8"?>
<X3D profile="Interchange" version="3.3">
  <Scene>
    <Transform translation="10 0 0">
      <Shape>
        <IndexedFaceSet coordIndex="0 1 2 -1">
          <Coordinate point="0 0 0 1 0 0 0 1 0"/>
          <Color color="1 0 0 0 1 0 0 0 1"/>
        </IndexedFaceSet>
      </Shape>
    </Transform>
  </Scene>
</X3D>"#;
    let p = dir.file("tri.x3d");
    fs::write(&p, x3d_txt).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::X3d);
    let poly = ds.as_poly_data().unwrap();
    assert_eq!(poly.points[0], [10.0, 0.0, 0.0]);
    assert_eq!(poly.polys, vec![vec![0, 1, 2]]);
    assert!(poly.point_data.get("Colors").is_some());
}

#[test]
fn gmsh_v2_and_v4_ascii() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("gmsh");
    let msh2 = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
1
1 4 2 99 1 1 2 3 4
$EndElements
$NodeData
1
\"temp\"
1
0.0
3
0
1
4
1 10.0
2 20.0
3 30.0
4 40.0
$EndNodeData
";
    let p2 = dir.file("tet_v2.msh");
    fs::write(&p2, msh2).unwrap();
    let (ds2, reader) = read_dataset_with_reader(&p2, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Gmsh);
    let ug2 = ds2.as_unstructured_grid().unwrap();
    assert_eq!(ug2.points.len(), 4);
    assert_eq!(ug2.cell_types, vec![CellType::Tetra]);
    assert!((ug2.point_data.get("temp").unwrap().get_scalar(3) - 40.0).abs() < 1e-6);

    let msh4 = "\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 3 1 3
2 1 0 3
1
2
3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
1 1 1 1
2 1 2 1
1 1 2 3
$EndElements
";
    let p4 = dir.file("tri_v4.msh");
    fs::write(&p4, msh4).unwrap();
    let ds4 = read_dataset(&p4).unwrap();
    let ug4 = ds4.as_unstructured_grid().unwrap();
    assert_eq!(ug4.points.len(), 3);
    assert_eq!(ug4.cell_types, vec![CellType::Triangle]);
}

#[test]
fn plot3d_ascii_and_binary_with_q() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("plot3d");
    // Binary Fortran unformatted 2x2x1 single-grid .xyz + .q
    fn f_rec(payload: &[u8]) -> Vec<u8> {
        let len = (payload.len() as u32).to_le_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&len);
        out.extend_from_slice(payload);
        out.extend_from_slice(&len);
        out
    }

    let mut hdr = Vec::new();
    for v in [2i32, 2, 1] {
        hdr.extend_from_slice(&v.to_le_bytes());
    }
    let xs = [0.0f32, 1.0, 0.0, 1.0];
    let ys = [0.0f32, 0.0, 1.0, 1.0];
    let zs = [0.0f32, 0.0, 0.0, 0.0];
    let mut coords = Vec::new();
    for v in xs.iter().chain(ys.iter()).chain(zs.iter()) {
        coords.extend_from_slice(&v.to_le_bytes());
    }
    let mut xyz_bytes = Vec::new();
    xyz_bytes.extend_from_slice(&f_rec(&hdr));
    xyz_bytes.extend_from_slice(&f_rec(&coords));

    let xyz_path = dir.file("grid.xyz");
    fs::write(&xyz_path, &xyz_bytes).unwrap();

    // Companion .q file
    let cond = [0.0f32; 4];
    let mut cond_b = Vec::new();
    for v in cond {
        cond_b.extend_from_slice(&v.to_le_bytes());
    }
    let mut q_vals = Vec::new();
    for _ in 0..4 { q_vals.extend_from_slice(&2.0f32.to_le_bytes()); } // rho = 2
    for _ in 0..4 { q_vals.extend_from_slice(&4.0f32.to_le_bytes()); } // mx = 4 -> vx = 2
    for _ in 0..4 { q_vals.extend_from_slice(&0.0f32.to_le_bytes()); } // my
    for _ in 0..4 { q_vals.extend_from_slice(&0.0f32.to_le_bytes()); } // mz
    for _ in 0..4 { q_vals.extend_from_slice(&10.0f32.to_le_bytes()); } // energy
    let mut q_bytes = Vec::new();
    q_bytes.extend_from_slice(&f_rec(&hdr));
    q_bytes.extend_from_slice(&f_rec(&cond_b));
    q_bytes.extend_from_slice(&f_rec(&q_vals));
    fs::write(dir.file("grid.q"), &q_bytes).unwrap();

    let (ds, reader) = read_dataset_with_reader(&xyz_path, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Plot3d);
    let DataSet::StructuredGrid(sg) = ds else { panic!("expected StructuredGrid") };
    assert_eq!(sg.dimensions, [2, 2, 1]);
    assert!((sg.point_data.get("Density").unwrap().get_scalar(0) - 2.0).abs() < 1e-6);
    assert!((sg.point_data.get("Velocity").unwrap().get_magnitude(0) - 2.0).abs() < 1e-6);
}

#[test]
fn tecplot_ordered_and_fe_zones() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("tecplot");
    let tec = "\
TITLE = \"Test\"
VARIABLES = \"X\", \"Y\", \"Z\", \"P\"
ZONE T=\"TriZone\", NODES=3, ELEMENTS=1, DATAPACKING=POINT, ZONETYPE=FETRIANGLE
0.0 0.0 0.0 1.0
1.0 0.0 0.0 2.0
0.0 1.0 0.0 3.0
1 2 3
";
    let p = dir.file("flow.dat");
    fs::write(&p, tec).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Tecplot);
    let ug = ds.as_unstructured_grid().unwrap();
    assert_eq!(ug.points.len(), 3);
    assert_eq!(ug.cell_types, vec![CellType::Triangle]);
    assert!((ug.point_data.get("P").unwrap().get_scalar(2) - 3.0).abs() < 1e-6);
}

#[test]
fn abaqus_inp_mesh_and_elsets() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("abaqus");
    let inp = "\
*HEADING
Simple Abaqus Tet
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=EALL
10, 1, 2, 3, 4
*ELSET, ELSET=CORE
10
";
    let p = dir.file("part.inp");
    fs::write(&p, inp).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Abaqus);
    let ug = ds.as_unstructured_grid().unwrap();
    assert_eq!(ug.points.len(), 4);
    assert_eq!(ug.cell_types, vec![CellType::Tetra]);
    assert_eq!(ug.cell_data.get("ElsetId").unwrap().get_scalar(0) as i32, 1);
}

#[test]
fn netcdf3_cf_rectilinear_and_temporal() {
    use paraviz_io::{detect_reader, DatasetReader};
    let dir = TestDir::new("netcdf_cf");
    fn pad4(buf: &mut Vec<u8>) {
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
    }
    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
        buf.extend_from_slice(s.as_bytes());
        pad4(buf);
    }

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"CDF\x01");
    hdr.extend_from_slice(&0u32.to_be_bytes()); // numrecs
    // NC_DIMENSION = 10, count = 3
    hdr.extend_from_slice(&10u32.to_be_bytes());
    hdr.extend_from_slice(&3u32.to_be_bytes());
    push_str(&mut hdr, "time"); hdr.extend_from_slice(&2u32.to_be_bytes());
    push_str(&mut hdr, "y"); hdr.extend_from_slice(&2u32.to_be_bytes());
    push_str(&mut hdr, "x"); hdr.extend_from_slice(&3u32.to_be_bytes());
    // Global atts = ABSENT (0)
    hdr.extend_from_slice(&0u32.to_be_bytes());
    // NC_VARIABLE = 11, count = 4 (time, x, y, temp)
    hdr.extend_from_slice(&11u32.to_be_bytes());
    hdr.extend_from_slice(&4u32.to_be_bytes());

    let mut var_offset_positions = Vec::new();
    push_str(&mut hdr, "time");
    hdr.extend_from_slice(&1u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&6u32.to_be_bytes());
    hdr.extend_from_slice(&16u32.to_be_bytes());
    var_offset_positions.push(hdr.len());
    hdr.extend_from_slice(&0u32.to_be_bytes());

    push_str(&mut hdr, "x");
    hdr.extend_from_slice(&1u32.to_be_bytes());
    hdr.extend_from_slice(&2u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&5u32.to_be_bytes());
    hdr.extend_from_slice(&12u32.to_be_bytes());
    var_offset_positions.push(hdr.len());
    hdr.extend_from_slice(&0u32.to_be_bytes());

    push_str(&mut hdr, "y");
    hdr.extend_from_slice(&1u32.to_be_bytes());
    hdr.extend_from_slice(&1u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&5u32.to_be_bytes());
    hdr.extend_from_slice(&8u32.to_be_bytes());
    var_offset_positions.push(hdr.len());
    hdr.extend_from_slice(&0u32.to_be_bytes());

    push_str(&mut hdr, "temp");
    hdr.extend_from_slice(&3u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&1u32.to_be_bytes());
    hdr.extend_from_slice(&2u32.to_be_bytes());
    hdr.extend_from_slice(&0u32.to_be_bytes());
    hdr.extend_from_slice(&5u32.to_be_bytes());
    hdr.extend_from_slice(&48u32.to_be_bytes());
    var_offset_positions.push(hdr.len());
    hdr.extend_from_slice(&0u32.to_be_bytes());

    let mut data_off = hdr.len() as u32;
    hdr[var_offset_positions[0]..var_offset_positions[0] + 4].copy_from_slice(&data_off.to_be_bytes());
    for t in [0.0f64, 5.0] {
        hdr.extend_from_slice(&t.to_be_bytes());
    }
    data_off = hdr.len() as u32;
    hdr[var_offset_positions[1]..var_offset_positions[1] + 4].copy_from_slice(&data_off.to_be_bytes());
    for x in [0.0f32, 1.0, 2.0] {
        hdr.extend_from_slice(&x.to_be_bytes());
    }
    data_off = hdr.len() as u32;
    hdr[var_offset_positions[2]..var_offset_positions[2] + 4].copy_from_slice(&data_off.to_be_bytes());
    for y in [0.0f32, 10.0] {
        hdr.extend_from_slice(&y.to_be_bytes());
    }
    data_off = hdr.len() as u32;
    hdr[var_offset_positions[3]..var_offset_positions[3] + 4].copy_from_slice(&data_off.to_be_bytes());
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0] {
        hdr.extend_from_slice(&v.to_be_bytes());
    }

    let nc_path = dir.file("ocean.nc");
    fs::write(&nc_path, &hdr).unwrap();
    assert_eq!(detect_reader(&nc_path).unwrap(), DatasetReader::NetCdf);
    let src = VtkFileSource::new(&nc_path);
    assert_eq!(src.request_information().time_steps, vec![0.0, 5.0]);
    let ds1 = src.execute(5.0).unwrap();
    let DataSet::RectilinearGrid(rg) = ds1 else { panic!("expected RectilinearGrid") };
    assert_eq!(rg.dimensions(), [3, 2, 1]);
    assert!((rg.point_data.get("temp").unwrap().get_scalar(0) - 11.0).abs() < 1e-6);
}

#[test]
fn yales2_hdf5_patch_layout() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("yales2");
    let mut b = Hdf5Builder::new();
    let xyz = b.add_dataset(
        &Hdf5Builder::dtype_f32(),
        &[3, 3],
        &f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        vec![],
    );
    let conn = b.add_dataset(
        &Hdf5Builder::dtype_i32(),
        &[4],
        &i32_bytes(&[4, 0, 1, 2]),
        vec![],
    );
    let u = b.add_dataset(
        &Hdf5Builder::dtype_f32(),
        &[3],
        &f32_bytes(&[7.0, 8.0, 9.0]),
        vec![],
    );
    let patch = b.add_group(&[("XYZ", xyz), ("ELEM2NODE", conn), ("P", u)], vec![]);
    let root = b.add_group(&[("Patch_001-wall", patch)], vec![]);
    let bytes = b.finish(root);

    let p = dir.file("pipe.solutbound.h5");
    fs::write(&p, &bytes).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Yales2Hdf5);
    let ug = ds.as_unstructured_grid().unwrap();
    assert_eq!(ug.points.len(), 3);
    assert_eq!(ug.cell_types, vec![CellType::Triangle]);
    assert!((ug.point_data.get("P").unwrap().get_scalar(2) - 9.0).abs() < 1e-6);
}

#[test]
fn las_lidar_point_cloud_and_colors() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("las");
    let mut buf = vec![0u8; 227 + 26];
    buf[0..4].copy_from_slice(b"LASF");
    buf[24] = 1;
    buf[25] = 2;
    buf[96..100].copy_from_slice(&227u32.to_le_bytes());
    buf[104] = 2;
    buf[105..107].copy_from_slice(&26u16.to_le_bytes());
    buf[107..111].copy_from_slice(&1u32.to_le_bytes());
    buf[131..139].copy_from_slice(&0.01f64.to_le_bytes());
    buf[139..147].copy_from_slice(&0.01f64.to_le_bytes());
    buf[147..155].copy_from_slice(&0.01f64.to_le_bytes());
    buf[227..231].copy_from_slice(&100i32.to_le_bytes());
    buf[231..235].copy_from_slice(&200i32.to_le_bytes());
    buf[235..239].copy_from_slice(&300i32.to_le_bytes());
    buf[239..241].copy_from_slice(&500u16.to_le_bytes());
    buf[242] = 2;
    buf[247..249].copy_from_slice(&(200u16 << 8).to_le_bytes());

    let p = dir.file("scan.las");
    fs::write(&p, &buf).unwrap();
    let (ds, reader) = read_dataset_with_reader(&p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Las);
    let poly = ds.as_poly_data().unwrap();
    assert_eq!(poly.points.len(), 1);
    assert!((poly.points[0][0] - 1.0).abs() < 1e-5);
    assert_eq!(poly.point_data.get("Intensity").unwrap().get_scalar(0) as u16, 500);
    assert_eq!(poly.point_data.get("Classification").unwrap().get_scalar(0) as u8, 2);
}

#[test]
fn image2d_pnm_and_dicom() {
    use paraviz_io::{read_dataset_with_reader, DatasetReader};
    let dir = TestDir::new("img2d");
    let ppm = b"P3\n2 1\n255\n255 0 0  0 255 0\n";
    let ppm_p = dir.file("img.ppm");
    fs::write(&ppm_p, ppm).unwrap();
    let (ds, reader) = read_dataset_with_reader(&ppm_p, DatasetReader::Auto).unwrap();
    assert_eq!(reader, DatasetReader::Image2d);
    let img = ds.as_image_data().unwrap();
    assert_eq!(img.dimensions, [2, 1, 1]);
    assert!(img.point_data.get("Colors").is_some());

    let mut dcm = vec![0u8; 132];
    dcm[128..132].copy_from_slice(b"DICM");
    fn add_tag_u16(buf: &mut Vec<u8>, g: u16, e: u16, val: u16) {
        buf.extend_from_slice(&g.to_le_bytes());
        buf.extend_from_slice(&e.to_le_bytes());
        buf.extend_from_slice(b"US");
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
    add_tag_u16(&mut dcm, 0x0028, 0x0010, 2);
    add_tag_u16(&mut dcm, 0x0028, 0x0011, 2);
    add_tag_u16(&mut dcm, 0x0028, 0x0100, 16);
    dcm.extend_from_slice(&0x7FE0u16.to_le_bytes());
    dcm.extend_from_slice(&0x0010u16.to_le_bytes());
    dcm.extend_from_slice(b"OW");
    dcm.extend_from_slice(&0u16.to_le_bytes());
    dcm.extend_from_slice(&8u32.to_le_bytes());
    for px in [10u16, 20, 30, 40] {
        dcm.extend_from_slice(&px.to_le_bytes());
    }
    let dcm_p = dir.file("slice.dcm");
    fs::write(&dcm_p, &dcm).unwrap();
    let (dds, dreader) = read_dataset_with_reader(&dcm_p, DatasetReader::Auto).unwrap();
    assert_eq!(dreader, DatasetReader::Dicom);
    let dimg = dds.as_image_data().unwrap();
    assert_eq!(dimg.dimensions, [2, 2, 1]);
    assert_eq!(dimg.point_data.get("Scalars").unwrap().get_scalar(3) as u16, 40);
}

