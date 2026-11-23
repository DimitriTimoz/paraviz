use paraviz::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 128^3 = 2,097,152 voxels (~127^3 = 2,048,383 hexahedral cells)
    let wavelet = WaveletSource {
        whole_extent: [-63, 64, -63, 64, -63, 64],
        ..Default::default()
    };

    let t0 = Instant::now();
    let ds = wavelet.execute(0.0)?;
    let t_src = t0.elapsed();
    let img = ds.as_image_data().unwrap();
    println!(
        "1. WaveletSource 128^3 ({} pts, {} cells): {:.2} ms",
        img.num_points(),
        img.num_cells(),
        t_src.as_secs_f64() * 1000.0
    );

    let t1 = Instant::now();
    let cpu_iso = ContourFilter::new("RTData", vec![140.0]).apply(&ds)?;
    let t_cpu_mc = t1.elapsed();
    println!(
        "2. CPU Parallel Rayon Marching Cubes: {} pts, {} tris in {:.2} ms",
        cpu_iso.num_points(),
        cpu_iso.num_cells(),
        t_cpu_mc.as_secs_f64() * 1000.0
    );

    let mut renderer = WebGpuRenderer::new_headless()?;
    let gpu_mc = GpuContourEngine::from_device(
        std::sync::Arc::new(renderer.device().clone()),
        std::sync::Arc::new(renderer.queue().clone()),
    );
    let t2 = Instant::now();
    let gpu_iso = gpu_mc.contour_imagedata(img, "RTData", 140.0)?;
    let t_gpu_mc = t2.elapsed();
    println!(
        "3. WebGPU Compute Marching Cubes: {} pts, {} tris in {:.2} ms",
        gpu_iso.num_points(),
        gpu_iso.num_cells(),
        t_gpu_mc.as_secs_f64() * 1000.0
    );

    let mut scene = RenderScene::new();
    let mut actor = SceneActor::new(1, "GPU Iso", std::sync::Arc::new(DataSet::PolyData(gpu_iso)));
    actor.style.representation = Representation::SurfaceWithEdges;
    actor.style.colormap = ColorMap::from_preset(ColorMapPreset::Viridis);
    actor.style.color_by = ColorBy::PointScalar {
        array: "RTData".into(),
        component: None,
    };
    scene.add_actor(actor);
    scene.settings.msaa_samples = 4;
    scene.settings.edl_enabled = true;
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    // Warmup GPU upload + pipeline compilation
    let _ = renderer.render_to_rgba(&scene, 1280, 720)?;

    let t3 = Instant::now();
    let _ = renderer.render_to_rgba(&scene, 1280, 720)?;
    let t_render = t3.elapsed();
    println!(
        "4. WebGPU Cached Render (1280x720, 4x MSAA + EDL + SurfaceWithEdges): {:.2} ms",
        t_render.as_secs_f64() * 1000.0
    );

    Ok(())
}
