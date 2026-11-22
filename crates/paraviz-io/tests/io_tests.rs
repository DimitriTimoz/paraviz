use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, ImageData, MultiBlockDataSet, PipelineGraph,
    PolyData, RectilinearGrid, Source, StructuredGrid, UnstructuredGrid,
};
use paraviz_io::csv::{read_csv, read_csv_str, write_csv};
use paraviz_io::mesh::{
    read_obj, read_obj_str, read_ply, read_stl, write_obj, write_ply_with_format, write_stl_ascii,
    write_stl_binary, PlyFormat,
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
