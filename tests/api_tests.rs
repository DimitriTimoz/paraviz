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
    assert_ne!(s0, s1);

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
}
