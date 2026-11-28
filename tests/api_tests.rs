use paraviz::prelude::*;
use std::fs;

#[test]
fn fluent_pipeline_render_and_state_roundtrip() {
    let mut pv = ParaViz::new();
    let wav = pv
        .add_source_cfg("Wavelet", SourceConfig::Wavelet(WaveletSource::with_extent(8)))
        .unwrap();
    pv.hide(wav);

    let iso = pv
        .add_filter_cfg(
            "Iso",
            wav,
            FilterConfig::Contour(ContourFilter::new("RTData", vec![120.0, 180.0])),
        )
        .unwrap();
    pv.show(iso)
        .representation(Representation::SurfaceWithEdges)
        .colormap(ColorMapPreset::Viridis)
        .color_by_point("RTData");

    pv.camera_preset(CameraPreset::Isometric).unwrap();

    let rgba = pv.render_rgba(256, 192).unwrap();
    assert_eq!(rgba.len(), 256 * 192 * 4);
    // Ensure rendered image is not uniform solid color
    let first = &rgba[0..4];
    let differs = rgba.as_chunks::<4>().0.iter().any(|px| px != first);
    assert!(differs, "Rendered image should contain shaded geometry");

    // Save/load JSON state
    let json = pv.save_state_json().unwrap();
    let mut restored = ParaViz::load_state_json(&json).unwrap();
    assert_eq!(restored.nodes().len(), 2);
    let rgba2 = restored.render_rgba(128, 96).unwrap();
    assert_eq!(rgba2.len(), 128 * 96 * 4);

    // Rust Trace generation
    let trace = pv.to_rust_trace();
    assert!(trace.contains("add_source_cfg"));
    assert!(trace.contains("add_filter_cfg"));
    assert!(trace.contains("SourceConfig::from_json"));
    assert!(trace.contains("FilterConfig::from_json"));
    assert!(trace.contains("ColorMapPreset::Viridis"));
    assert!(trace.contains("color_by_point(\"RTData\")"));
}

#[test]
fn all_demo_presets_render_cleanly() {
    for preset in DemoPreset::ALL {
        let mut pv = preset.build().expect("demo preset build");
        let tmp = std::env::temp_dir().join(format!("paraviz_test_{}.png", preset.slug()));
        pv.render_png(&tmp, 160, 120).expect("demo preset render_png");
        let meta = fs::metadata(&tmp).unwrap();
        assert!(meta.len() > 200, "PNG for {} too small", preset.slug());
        let _ = fs::remove_file(&tmp);
    }
}

#[test]
fn time_stepping_and_io_export() {
    let mut pv = ParaViz::new();
    let wav = pv
        .add_source_cfg("Wavelet", SourceConfig::Wavelet(WaveletSource::with_extent(4)))
        .unwrap();
    let steps = pv.all_time_steps();
    assert!(!steps.is_empty());

    let ds0 = pv.evaluate(wav).unwrap();
    pv.set_time(0.5);
    let ds1 = pv.evaluate(wav).unwrap();
    let s0 = ds0.point_data().unwrap().get("RTData").unwrap().get_scalar(0);
    let s1 = ds1.point_data().unwrap().get("RTData").unwrap().get_scalar(0);
    // vtkRTAnalyticSource is time-invariant (verified against ParaView 6.0.1)
    assert_eq!(s0, s1);

    let out_vti = std::env::temp_dir().join("paraviz_api_export.vti");
    pv.export_dataset(wav, &out_vti).unwrap();
    let reloaded = ParaViz::read_file(&out_vti).unwrap();
    assert_eq!(reloaded.num_points(), ds1.num_points());
    let _ = fs::remove_file(&out_vti);
}

#[test]
fn gpu_compute_contour_picking_and_glb_export() {
    let mut pv = ParaViz::new();
    let wav = pv
        .add_source_cfg("Wavelet", SourceConfig::Wavelet(WaveletSource::with_extent(8)))
        .unwrap();
    pv.hide(wav);

    let gpu_iso = pv
        .gpu_contour("GPU Iso", wav, Some("RTData"), 140.0)
        .unwrap();
    pv.show(gpu_iso)
        .representation(Representation::Surface)
        .colormap(ColorMapPreset::Viridis)
        .color_by_point("RTData");
    pv.settings.msaa_samples = 4;
    pv.settings.edl_enabled = true;
    pv.camera_preset(CameraPreset::Isometric).unwrap();

    let hit = pv.pick(160.0, 120.0, 320, 240).unwrap();
    assert!(hit.is_some(), "Center pick should hit GPU-contoured isosurface");

    let out_glb = std::env::temp_dir().join("paraviz_gpu_iso.glb");
    pv.export_glb(gpu_iso, &out_glb, true).unwrap();
    assert!(fs::metadata(&out_glb).unwrap().len() > 100);
    let _ = fs::remove_file(&out_glb);

    // State JSON roundtrip should also preserve custom/in-memory nodes (like gpu_iso)
    let state_json = pv.save_state_json().unwrap();
    let restored = ParaViz::load_state_json(&state_json).unwrap();
    assert_eq!(restored.nodes().len(), 2);
}

#[test]
fn multiblock_block_controls_time_annotation_and_preloaded_seed() {
    let mut mb = MultiBlockDataSet::new();
    let sphere = SphereSource {
        radius: 1.0,
        theta_resolution: 12,
        phi_resolution: 12,
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    let cube = CubeSource {
        center: [3.0, 0.0, 0.0],
        ..Default::default()
    }
    .execute(0.0)
    .unwrap();
    mb.add_block("inlet", sphere);
    mb.add_block("wall", cube);

    let mut pv = ParaViz::new();
    let id = pv.add_dataset("Assembly", DataSet::MultiBlock(mb));
    pv.show(id)
        .hide_block("wall")
        .block_color("inlet", [0.9, 0.2, 0.2, 1.0]);
    pv.set_time(2.5);
    pv.settings.show_time_annotation = true;
    pv.settings.show_scalar_bar = true;
    pv.settings.scalar_bar.horizontal = true;
    pv.settings.annotations.push(ViewportAnnotation {
        id: 1,
        text: "CFD Case".into(),
        pos: [0.05, 0.08],
        font_size: 16.0,
        color: [1.0, 1.0, 1.0, 1.0],
        visible: true,
    });
    pv.camera_preset(CameraPreset::Isometric).unwrap();

    let scene = pv.build_scene().unwrap();
    assert_eq!(scene.settings.time_label.as_deref(), Some("Time: 2.5000"));
    assert!(scene.actors[0].style.hidden_blocks.contains("wall"));

    let rgba = pv.render_rgba(320, 240).unwrap();
    assert_eq!(rgba.len(), 320 * 240 * 4);
}

#[test]
fn state_roundtrip_preserves_live_inputs_and_unevaluated_datasets() {
    use paraviz::core::ArrayData;

    let mut session = ParaViz::new();
    let mut mesh = PolyData::new();
    mesh.points.push([0.0, 0.0, 0.0]);
    mesh.point_data.add_array(DataArray::new(
        "temperature", 1, ArrayData::F64(vec![f64::NAN]),
    ));
    session.add_dataset("Unmeasured temperature", DataSet::PolyData(mesh));
    let left = session.add_source_cfg("Left", SourceConfig::Cube(CubeSource::default())).unwrap();
    let right = session.add_source_cfg("Right", SourceConfig::Cube(CubeSource::default())).unwrap();
    session.append_nodes(vec![left, right]).unwrap();

    let json = session.save_state_json().unwrap();
    for json in [json.clone(), json.replace("\"NaN\"", "null")] {
        let mut restored = ParaViz::load_state_json(&json).unwrap();
        assert_eq!(restored.nodes().len(), 4);
        let data_id = restored.nodes()[0].id;
        let data = restored.evaluate(data_id).unwrap();
        assert!(data.point_data().unwrap().get("temperature").unwrap().get_scalar(0).is_nan());

        let right = restored.nodes()[2].id;
        let append = restored.nodes()[3].id;
        assert!(restored.evaluate(append).unwrap().bounds().max.x < 10.0);
        restored.update_source_cfg(right, SourceConfig::Cube(CubeSource {
            center: [10.0, 0.0, 0.0],
            ..Default::default()
        })).unwrap();
        assert!(restored.node(append).unwrap().dataset().is_none());
        assert!(restored.evaluate(append).unwrap().bounds().max.x > 10.0);
    }
}

#[test]
fn temporal_probe_samples_the_selected_pipeline_and_keeps_gaps() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("series.pvd");
    let frames = [(0.0, Some(-10.0)), (1.0, None), (2.0, Some(30.0))].map(|(time, value)| {
        let mut grid = ImageData::new([2, 2, 2], [0.0; 3], [1.0; 3]);
        if let Some(value) = value {
            grid.point_data.set_scalars(DataArray::scalars_f32("Pressure", vec![value; 8]));
        }
        (time, DataSet::ImageData(grid))
    });
    write_pvd_series(&path, &frames).unwrap();

    let mut session = ParaViz::new();
    let node = session.open(path).unwrap();
    session.set_time(2.0);
    let before = session.evaluate(node).unwrap();
    let isolated = session.add_source_cfg("Other times", SourceConfig::Wavelet(WaveletSource::default())).unwrap();
    assert!(!session.node(isolated).unwrap().time_steps.is_empty());

    let result = session.prepare_temporal_probe(node, TemporalProbeRequest {
        position: [0.5; 3],
        field: "Pressure".into(),
        component: 0,
    }).unwrap().run().unwrap();
    assert_eq!(result.samples.iter().map(|step| step.value).collect::<Vec<_>>(), vec![Some(-10.0), None, Some(30.0)]);
    assert!(result.to_csv().contains("1,,missing,\"Pressure\",0,0.5,0.5,0.5"));
    session.temporal_probe = Some((node, result.request.clone()));
    let restored = ParaViz::load_state_json(&session.save_state_json().unwrap()).unwrap();
    let (restored_node, request) = restored.temporal_probe.clone().unwrap();
    assert_eq!(restored.prepare_temporal_probe(restored_node, request).unwrap().run().unwrap(), result);
    assert_eq!(session.time, 2.0);
    assert!(std::sync::Arc::ptr_eq(&before, &session.evaluate(node).unwrap()));

    let steps: Vec<_> = frames.iter().map(|(_, dataset)| dataset).collect();
    let statistics = TemporalStatisticsFilter::default().compute_over_steps(&steps).unwrap();
    let fields = statistics.point_data().unwrap();
    assert_eq!(fields.get("Pressure_average").unwrap().get_scalar(0), 10.0);
    assert_eq!(fields.get("Pressure_stddev").unwrap().get_scalar(0), 20.0);
}

#[test]
fn projects_survive_moves_relocation_and_offline_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let original = directory.path().join("original");
    fs::create_dir_all(original.join("data")).unwrap();
    let series = original.join("data").join("run.pvd");
    let frames = [0.0, 1.0].map(|time| {
        let mut grid = ImageData::new([2, 2, 2], [0.0; 3], [1.0; 3]);
        grid.point_data.set_scalars(DataArray::scalars_f32("Pressure", vec![time as f32; 8]));
        (time, DataSet::ImageData(grid))
    });
    write_pvd_series(&series, &frames).unwrap();

    let mut session = ParaViz::new();
    let source = session.open(&series).unwrap();
    let join = session.append_nodes(vec![source]).unwrap();
    let request = TemporalProbeRequest {
        position: [0.5; 3],
        field: "Pressure".into(),
        component: 0,
    };
    session.temporal_probe = Some((source, request));
    session.set_time(1.0);
    session.show(join);

    let referenced = original.join("project.json");
    let portable = directory.path().join("portable.json");
    session.save_project(&referenced, ProjectSaveMode::Referenced).unwrap();
    session.save_project(&portable, ProjectSaveMode::Embedded).unwrap();
    let before = fs::read_to_string(&referenced).unwrap();
    assert!(before.contains("\"data/run.pvd\""));
    assert!(ParaViz::load_state_json(&before).is_err());
    let portable_contents = fs::read(&portable).unwrap();

    let moved = directory.path().join("moved");
    fs::rename(&original, &moved).unwrap();
    let moved_project = moved.join("project.json");
    let mut restored = ParaViz::load_project(&moved_project).unwrap();
    assert_eq!(restored.time, 1.0);
    assert_eq!(restored.all_time_steps(), vec![0.0, 1.0]);
    assert!(restored.update_visible().is_ok());

    let new_data_dir = directory.path().join("relocated_data");
    fs::rename(moved.join("data"), &new_data_dir).unwrap();
    assert!(ParaViz::load_project(&moved_project).is_err());
    let restored = ParaViz::load_project_with_relocation(&moved_project, |missing| {
        assert_eq!(missing, moved.join("data/run.pvd"));
        Some(new_data_dir.join("run.pvd"))
    }).unwrap();
    assert_eq!(fs::read_to_string(&moved_project).unwrap(), before);
    let (node, request) = restored.temporal_probe.clone().unwrap();
    let expected = restored.prepare_temporal_probe(node, request).unwrap().run().unwrap();

    fs::remove_dir_all(&new_data_dir).unwrap();
    let offline = ParaViz::load_project(&portable).unwrap();
    let (node, request) = offline.temporal_probe.clone().unwrap();
    assert_eq!(offline.prepare_temporal_probe(node, request).unwrap().run().unwrap(), expected);
    assert!(ParaViz::load_state_json(std::str::from_utf8(&portable_contents).unwrap()).is_ok());

    assert!(session.save_project(&portable, ProjectSaveMode::Embedded).is_err());
    assert_eq!(fs::read(&portable).unwrap(), portable_contents);
}

#[test]
fn export_html_ab_single_file_with_slider_and_common_legend() {
    let mut pv = ParaViz::new();
    let wav = pv
        .add_source_cfg("Wavelet", SourceConfig::Wavelet(WaveletSource::with_extent(6)))
        .unwrap();
    let iso_a = pv
        .add_filter_cfg(
            "IsoA",
            wav,
            FilterConfig::Contour(ContourFilter::new("RTData", vec![120.0])),
        )
        .unwrap();
    let iso_b = pv
        .add_filter_cfg(
            "IsoB",
            wav,
            FilterConfig::Contour(ContourFilter::new("RTData", vec![200.0])),
        )
        .unwrap();
    pv.show(iso_a).color_by_point("RTData");
    pv.show(iso_b).color_by_point("RTData");

    let out = std::env::temp_dir().join("paraviz_ab_test.html");
    pv.export_html_ab(iso_a, iso_b, &out, true).unwrap();
    let html = fs::read_to_string(&out).unwrap();
    let _ = fs::remove_file(&out);

    assert_eq!(html.matches("<canvas").count(), 1, "single canvas/file");
    assert!(html.contains(r#"id="abSlider""#), "A/B wipe slider missing");
    assert!(html.contains(r#"id="legendBar""#), "common legend missing");
    assert!(html.contains(r#""labelA":"IsoA""#));
    assert!(html.contains(r#""labelB":"IsoB""#));
    // common legend over the union of both RTData ranges
    assert!(html.contains(r#""array":"RTData""#));
    assert!(html.contains(r#""a":{"#), "side A payload missing");
    assert!(html.contains(r#""b":{"#), "side B payload missing");
}
