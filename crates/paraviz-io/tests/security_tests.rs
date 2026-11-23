use paraviz_core::PolyData;
use paraviz_io::html_viewer::encode_html_viewer;
use std::fs;

#[test]
fn html_viewer_escapes_script_breakout_in_title() {
    let mut poly = PolyData::new();
    poly.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    poly.polys = vec![vec![0, 1, 2]];

    let evil_title = "</script><script>alert('xss')</script>";
    let html = encode_html_viewer(&poly, evil_title, None, None);

    // Must not contain raw </script> inside the JSON payload
    let json_start = html
        .find(r#"<script id="mesh-data" type="application/json">"#)
        .expect("mesh-data script tag missing");
    let json_end = html[json_start..]
        .find("</script>")
        .expect("closing script tag missing");
    let embedded = &html[json_start..json_start + json_end];
    assert!(
        !embedded.contains("</script>"),
        "embedded JSON must not contain raw </script>"
    );
    assert!(
        embedded.contains(r"\u003c/script\u003e"),
        "embedded JSON should escape '<' and '>'"
    );
    assert!(html.contains("&lt;/script&gt;"));
}

#[test]
fn vtk_legacy_rejects_huge_point_count_without_oom() {
    let payload = b"# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\nPOINTS 400000000000 float\n";
    let res = paraviz_io::vtk_legacy::read_vtk_legacy_bytes(payload);
    assert!(res.is_err(), "huge POINTS header must fail cleanly");
}

#[test]
fn ply_rejects_huge_vertex_count_without_oom() {
    let payload = b"ply\nformat ascii 1.0\nelement vertex 400000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    let res = paraviz_io::mesh::read_ply_bytes(payload);
    assert!(res.is_err(), "huge PLY vertex count must fail cleanly");
}

#[test]
fn vtk_xml_rejects_lz4_header_bomb_without_oom() {
    // Craft a VTK XML PolyData with LZ4 compressor and a header declaring 2 GiB uncompressed block
    let mut hdr = Vec::new();
    // num_blocks = 1, block_size = 2_000_000_000, last_block_size = 2_000_000_000, c_len = 4
    hdr.extend_from_slice(&1u32.to_le_bytes());
    hdr.extend_from_slice(&2_000_000_000u32.to_le_bytes());
    hdr.extend_from_slice(&2_000_000_000u32.to_le_bytes());
    hdr.extend_from_slice(&4u32.to_le_bytes());
    hdr.extend_from_slice(&[0u8; 4]);

    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&hdr);
    let xml = format!(
        r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkLZ4DataCompressor">
  <PolyData>
    <Piece NumberOfPoints="1" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="binary">{b64}</DataArray>
      </Points>
    </Piece>
  </PolyData>
</VTKFile>"#
    );
    let res = paraviz_io::vtk_xml::read_vtk_xml_bytes(xml.as_bytes(), std::path::Path::new("."));
    assert!(res.is_err(), "LZ4 header bomb must be rejected");
}

#[test]
fn vtk_xml_rejects_uint64_header_overflow() {
    let mut hdr = Vec::new();
    hdr.extend_from_slice(&u64::MAX.to_le_bytes());
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&hdr);
    let xml = format!(
        r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt64">
  <PolyData>
    <Piece NumberOfPoints="1" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="binary">{b64}</DataArray>
      </Points>
    </Piece>
  </PolyData>
</VTKFile>"#
    );
    let res = paraviz_io::vtk_xml::read_vtk_xml_bytes(xml.as_bytes(), std::path::Path::new("."));
    assert!(res.is_err(), "UInt64::MAX binary header must fail cleanly");
}

#[test]
fn path_traversal_blocked_across_readers() {
    let dir = std::env::temp_dir().join(format!(
        "paraviz_sec_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // 1. PVD traversal
    let pvd_path = dir.join("evil.pvd");
    fs::write(
        &pvd_path,
        r#"<VTKFile type="Collection"><Collection><DataSet timestep="0" file="../../etc/passwd"/></Collection></VTKFile>"#,
    )
    .unwrap();
    assert!(paraviz_io::read_pvd_series(&pvd_path).is_err());

    // 2. XDMF HDF5 absolute path & traversal
    let xdmf_abs = dir.join("abs.xdmf");
    fs::write(
        &xdmf_abs,
        r#"<Xdmf Version="3.0"><Domain><Grid Name="G"><Topology TopologyType="Triangle" NumberOfElements="1"/><Geometry GeometryType="XYZ"><DataItem Dimensions="3 3" Format="HDF">/etc/passwd:/pts</DataItem></Geometry></Grid></Domain></Xdmf>"#,
    )
    .unwrap();
    assert!(paraviz_io::read_xdmf(&xdmf_abs).is_err());

    // 3. MetaImage traversal
    let mhd = dir.join("evil.mhd");
    fs::write(
        &mhd,
        "ObjectType = Image\nNDims = 3\nDimSize = 2 2 2\nElementType = MET_UCHAR\nElementDataFile = ../../etc/passwd\n",
    )
    .unwrap();
    assert!(paraviz_io::read_metaimage(&mhd).is_err());

    // 4. NRRD traversal
    let nrrd = dir.join("evil.nhdr");
    fs::write(
        &nrrd,
        "NRRD0004\ntype: uint8\ndimension: 3\nsizes: 2 2 2\nencoding: raw\ndata file: /etc/passwd\n",
    )
    .unwrap();
    assert!(paraviz_io::read_nrrd(&nrrd).is_err());

    // 5. EnSight traversal
    let case = dir.join("evil.case");
    fs::write(&case, "FORMAT\ntype: ensight gold\nGEOMETRY\nmodel: ../../etc/passwd\n").unwrap();
    assert!(paraviz_io::read_ensight_case(&case).is_err());

    let _ = fs::remove_dir_all(&dir);
}
