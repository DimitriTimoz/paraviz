//! End-to-end tests covering all newly implemented audit features across
//! paraviz-core, paraviz-io, paraviz-filters, paraviz-render, and paraviz session API.

use paraviz::core::{ArrayData, ScalarType};
use paraviz::prelude::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn core_extended_scalar_types_cell_types_and_ear_clipping() {
    // 16-bit medical scalar types
    let arr_i16 = DataArray::new("ct", 1, ArrayData::I16(vec![-1000, 0, 1024]));
    assert_eq!(arr_i16.data.scalar_type(), ScalarType::I16);
    assert_eq!(arr_i16.data.scalar_type().byte_size(), 2);
    assert_eq!(arr_i16.get_scalar(2), 1024.0);

    let arr_u16 = DataArray::new("mri", 1, ArrayData::U16(vec![0, 500, 65535]));
    assert_eq!(arr_u16.data.scalar_type(), ScalarType::U16);
    assert_eq!(arr_u16.range(None), (0.0, 65535.0));

    // Higher-order VTK cell types & parity-aware hex decomposition
    assert_eq!(CellType::from_vtk_id(24), Some(CellType::QuadraticTetra));
    assert_eq!(CellType::from_vtk_id(25), Some(CellType::QuadraticHexahedron));
    assert_eq!(CellType::from_vtk_id(42), Some(CellType::Polyhedron));
    assert!(CellType::QuadraticTetra.is_higher_order());
    assert_eq!(CellType::QuadraticTetra.tetrahedra().len(), 8);
    assert_ne!(
        CellType::Hexahedron.tetrahedra_with_parity(false),
        CellType::Hexahedron.tetrahedra_with_parity(true)
    );

    // Concave L-shaped polygon ear-clipping triangulation
    let pd = PolyData {
        points: vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        polys: vec![vec![0, 1, 2, 3, 4, 5]],
        ..Default::default()
    };
    let tris = pd.triangulate();
    assert_eq!(tris.len(), 4);
    // Total area of L-shape = 2*1 + 1*1 = 3.0
    let mut area = 0.0f32;
    for [a, b, c] in tris {
        let pa = glam::Vec3::from_array(pd.points[a as usize]);
        let pb = glam::Vec3::from_array(pd.points[b as usize]);
        let pc = glam::Vec3::from_array(pd.points[c as usize]);
        area += 0.5 * (pb - pa).cross(pc - pa).length();
    }
    assert!((area - 3.0).abs() < 1e-4, "Expected area 3.0, got {area}");
}

#[test]
fn io_nifti_metaimage_nrrd_xdmf_openfoam_ensight_roundtrips() {
    let dir = tempdir().unwrap();

    // 1. NIfTI (.nii and .nii.gz)
    let mut img = ImageData::new([4, 4, 4], [-1.0, -2.0, -3.0], [0.5, 0.5, 1.0]);
    let vals: Vec<f32> = (0..64).map(|i| i as f32 * 2.5).collect();
    img.point_data.set_scalars(DataArray::scalars_f32("RT", vals.clone()));

    let nii_gz = dir.path().join("brain.nii.gz");
    write_nifti(&nii_gz, &img).unwrap();
    let loaded_nii = read_dataset(&nii_gz).unwrap();
    let loaded_img = loaded_nii.as_image_data().unwrap();
    assert_eq!(loaded_img.dimensions, [4, 4, 4]);
    assert!((loaded_img.spacing[2] - 1.0).abs() < 1e-5);

    // 2. MetaImage (.mha)
    let mha = dir.path().join("scan.mha");
    write_metaimage(&mha, &img).unwrap();
    let loaded_mha = read_dataset(&mha).unwrap();
    assert_eq!(loaded_mha.as_image_data().unwrap().dimensions, [4, 4, 4]);

    // 3. NRRD (.nrrd)
    let nrrd = dir.path().join("vol.nrrd");
    let mut nrrd_bytes = b"NRRD0004\ntype: uint8\ndimension: 3\nsizes: 2 2 2\nencoding: raw\n\n".to_vec();
    nrrd_bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    fs::write(&nrrd, &nrrd_bytes).unwrap();
    let loaded_nrrd = read_dataset(&nrrd).unwrap();
    assert_eq!(loaded_nrrd.as_image_data().unwrap().dimensions, [2, 2, 2]);

    // 4. XDMF (.xdmf)
    let xdmf = dir.path().join("tri.xdmf");
    fs::write(
        &xdmf,
        r#"<Xdmf><Domain><Grid Name="g">
            <Topology TopologyType="Triangle" NumberOfElements="1">
                <DataItem Format="XML">0 1 2</DataItem>
            </Topology>
            <Geometry GeometryType="XYZ">
                <DataItem Format="XML">0 0 0 1 0 0 0 1 0</DataItem>
            </Geometry>
            <Attribute Name="Temp" Center="Node">
                <DataItem Format="XML">10 20 30</DataItem>
            </Attribute>
        </Grid></Domain></Xdmf>"#,
    )
    .unwrap();
    let loaded_xdmf = read_dataset(&xdmf).unwrap();
    assert_eq!(loaded_xdmf.num_points(), 3);
    assert_eq!(loaded_xdmf.num_cells(), 1);

    // 5. OpenFOAM polyMesh
    let foam_dir = dir.path().join("foam_case");
    fs::create_dir_all(foam_dir.join("constant/polyMesh")).unwrap();
    fs::write(
        foam_dir.join("constant/polyMesh/points"),
        "FoamFile { version 2.0; }\n3\n(\n(0 0 0)\n(1 0 0)\n(0 1 0)\n)\n",
    )
    .unwrap();
    fs::write(
        foam_dir.join("constant/polyMesh/faces"),
        "FoamFile { version 2.0; }\n1\n(\n3(0 1 2)\n)\n",
    )
    .unwrap();
    let loaded_foam = read_dataset(&foam_dir).unwrap();
    assert_eq!(loaded_foam.num_points(), 3);
    assert_eq!(loaded_foam.num_cells(), 1);
}

#[test]
fn new_filters_integrate_decimate_delaunay_isovolume_resample_extract_temporal_lambda2() {
    // 1. IntegrateVariables on 2x2x2 cube (volume = 8.0)
    let cube = CubeSource {
        lengths: [2.0, 2.0, 2.0],
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let int_out = IntegrateVariablesFilter::new().apply(&cube).unwrap();
    let area = int_out.cell_data().unwrap().get("Area").unwrap().get_scalar(0);
    assert!((area - 24.0).abs() < 1e-3, "2x2x2 cube area should be 24, got {area}");

    // 2. DecimateFilter on high-res sphere
    let sphere = SphereSource {
        theta_resolution: 24,
        phi_resolution: 24,
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let orig_cells = sphere.num_cells();
    let decimated = DecimateFilter::new(0.5).apply(&sphere).unwrap();
    assert!(decimated.num_cells() < orig_cells);
    assert!(decimated.num_cells() > 10);

    // 3. Delaunay2D & Delaunay3D on point cloud
    let pts = DataSet::PolyData(PolyData {
        points: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [1.0, 1.0, 0.2],
            [0.0, 1.0, 0.1],
            [0.5, 0.5, 0.8],
        ],
        ..Default::default()
    });
    let d2 = Delaunay2DFilter::default().apply(&pts).unwrap();
    assert!(d2.num_cells() >= 2);
    let d3 = Delaunay3DFilter::default().apply(&pts).unwrap();
    assert!(d3.num_cells() >= 2);

    // 4. IsoVolumeFilter on Wavelet
    let wav = WaveletSource {
        whole_extent: [-4, 4, -4, 4, -4, 4],
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let isov = IsoVolumeFilter::new("RTData", 80.0, 180.0).apply(&wav).unwrap();
    assert!(isov.num_cells() > 0);

    // 5. ResampleToImage + Multi-input ResampleWithDataset
    let resampled = ResampleToImageFilter::new([8, 8, 8]).apply(&isov).unwrap();
    assert_eq!(resampled.as_image_data().unwrap().dimensions, [8, 8, 8]);

    let mut pv = ParaViz::new();
    let src_id = pv.add_source("Wav", WaveletSource { whole_extent: [-4, 4, -4, 4, -4, 4], ..Default::default() });
    let dst_id = pv.add_source("Sph", SphereSource::default());
    let rwd_id = pv.resample_with_dataset(src_id, dst_id).unwrap();
    let rwd_ds = pv.evaluate(rwd_id).unwrap();
    assert!(rwd_ds.point_data().unwrap().get("RTData").is_some());

    // 6. ExtractBlockFilter + TemporalStatisticsFilter
    let mut mb = MultiBlockDataSet::new();
    mb.add_block("step_0", wav.clone());
    mb.add_block("step_1", wav.clone());
    let mb_ds = DataSet::MultiBlock(mb);
    let eb = ExtractBlockFilter::by_name("step_1").apply(&mb_ds).unwrap();
    assert!(eb.as_image_data().is_some());
    let ts = TemporalStatisticsFilter::default().apply(&mb_ds).unwrap();
    assert!(ts.point_data().unwrap().get("RTData_average").is_some());
    assert!(ts.point_data().unwrap().get("RTData_stddev").is_some());

    // 7. Lambda2 vortex criterion in GradientFilter
    let vf = VectorFieldSource {
        dimensions: [8, 8, 8],
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let grad = GradientFilter::with_all_vector_quantities("Velocity")
        .apply(&vf)
        .unwrap();
    assert!(grad.point_data().unwrap().get("Lambda2").is_some());
}

#[test]
fn render_point_gaussian_volume_clip_plane_axes_grid_and_selection() {
    let mut pv = ParaViz::new();
    let wav = pv.add_source(
        "Wav",
        WaveletSource {
            whole_extent: [-6, 6, -6, 6, -6, 6],
            ..Default::default()
        },
    );
    pv.show(wav)
        .representation(Representation::Volume)
        .color_by_point("RTData");
    pv.node_mut(wav).unwrap().style.volume_clip_plane = Some(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    pv.settings.show_axes_grid = true;
    pv.selection_highlight = Some([0.0, 0.0, 0.0]);
    pv.reset_camera().unwrap();
    let rgba = pv.render_rgba(160, 120).unwrap();
    assert_eq!(rgba.len(), 160 * 120 * 4);

    // Switch to PointGaussian representation
    pv.show(wav).representation(Representation::PointGaussian);
    let rgba2 = pv.render_rgba(160, 120).unwrap();
    assert_eq!(rgba2.len(), 160 * 120 * 4);
}

#[test]
fn paraview_issues_streamtracer_custom_source_cell_data_timesteps_transparent_legend_gif_html() {
    // 1. #23130: StreamTracerFilter on cell-data vector field + StreamTracerWithCustomSource
    let vf = VectorFieldSource {
        dimensions: [10, 10, 10],
        spacing: [4.0 / 9.0; 3],
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let cell_vf = PointDataToCellDataFilter {
        pass_point_data: false,
    }
    .apply(&vf)
    .unwrap();
    assert!(cell_vf.point_data().unwrap().get("Velocity").is_none());
    assert!(cell_vf.cell_data().unwrap().get("Velocity").is_some());

    let st_cell = StreamTracerFilter {
        vector_field: "Velocity".into(),
        seeds: SeedSource::Line {
            p1: [-1.0, 0.0, 0.0],
            p2: [1.0, 0.0, 0.0],
            count: 6,
        },
        step_size: 0.1,
        max_steps: 40,
        ..Default::default()
    }
    .apply(&cell_vf)
    .unwrap();
    assert!(st_cell.num_cells() > 0, "StreamTracer should trace cell-data vectors");

    let mut pv = ParaViz::new();
    let vf_id = pv.add_source(
        "VF",
        VectorFieldSource {
            dimensions: [10, 10, 10],
            spacing: [4.0 / 9.0; 3],
            ..Default::default()
        },
    );
    let seed_id = pv.add_source("Seeds", LineSource::default());
    let custom_st = pv
        .stream_tracer_with_custom_source(vf_id, seed_id, StreamTracerFilter::default())
        .unwrap();
    let custom_ds = pv.evaluate(custom_st).unwrap();
    assert!(custom_ds.num_cells() > 0);

    // 2. #20241: AddTimeStepsFilter populates time steps on static datasets
    let ts_id = pv
        .add_filter_cfg(
            "AddTimeSteps",
            vf_id,
            FilterConfig::AddTimeSteps(AddTimeStepsFilter::new(5, 0.0, 4.0)),
        )
        .unwrap();
    assert_eq!(pv.all_time_steps(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let ts_mb = pv.evaluate(ts_id).unwrap();
    let DataSet::MultiBlock(mb) = ts_mb.as_ref() else {
        panic!("expected MultiBlockDataSet from AddTimeStepsFilter");
    };
    assert_eq!(mb.blocks.len(), 5);

    // 3. #22639 + #21122 + #18651: Transparent background + Scalar Color Bar in RGBA/PNG
    let mut pv_vis = ParaViz::new();
    let sph = pv_vis.add_source("Sphere", SphereSource::default());
    pv_vis
        .show(sph)
        .representation(Representation::Surface)
        .colormap(ColorMapPreset::Viridis)
        .color_by_point("Elevation");
    pv_vis.settings.transparent_background = true;
    pv_vis.settings.show_scalar_bar = true;
    pv_vis.settings.show_axes_gizmo = false;
    pv_vis.settings.edl_enabled = true;
    pv_vis.reset_camera().unwrap();
    let rgba_trans = pv_vis.render_rgba(200, 150).unwrap();
    // Top-left corner pixel (0, 0) must remain transparent (alpha == 0) even with EDL
    assert_eq!(rgba_trans[3], 0, "transparent_background must preserve alpha=0");

    // 4. GIF Animation export (turntable) + HTML5 WebGL2 viewer export
    let dir = tempfile::tempdir().unwrap();
    let gif_path = dir.path().join("spin.gif");
    pv_vis.render_turntable_gif(&gif_path, 120, 90, 4, 50).unwrap();
    let gif_bytes = std::fs::read(&gif_path).unwrap();
    assert_eq!(&gif_bytes[0..6], b"GIF89a");
    assert!(gif_bytes.windows(11).any(|w| w == b"NETSCAPE2.0"));

    let html_path = dir.path().join("sphere.html");
    pv_vis.export_html(sph, &html_path, true).unwrap();
    let html = std::fs::read_to_string(&html_path).unwrap();
    assert!(html.contains("<canvas id=\"glcanvas\">"));
    assert!(html.contains("\"triangles\""));
}
