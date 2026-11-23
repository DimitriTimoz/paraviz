use glam::Vec3;
use paraviz_core::{
    BoundingBox, ColorMap, ColorMapPreset, DataArray, DataSet, ImageData, OpacityPoint, PolyData,
    TransferFunction,
};
use paraviz_render::{
    pick_scene, ActorStyle, Camera, CameraPreset, ColorBy, GpuContourEngine, RenderScene,
    Representation, SceneActor, ViewportWidget, WebGpuRenderer,
};
use std::collections::HashMap;
use std::sync::Arc;

fn make_box_polydata() -> DataSet {
    let points = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let quads = vec![
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    let scalars: Vec<f32> = points.iter().map(|p| p[0] + p[1] + p[2]).collect();
    let mut poly = PolyData {
        points,
        polys: quads,
        ..Default::default()
    };
    poly.point_data
        .set_scalars(DataArray::scalars_f32("elevation", scalars));
    DataSet::PolyData(poly)
}

fn make_sphere_imagedata(dim: usize) -> DataSet {
    let mut img = ImageData::new(
        [dim, dim, dim],
        [-1.0, -1.0, -1.0],
        [2.0 / (dim - 1) as f64; 3],
    );
    let mut vals = Vec::with_capacity(dim * dim * dim);
    for k in 0..dim {
        for j in 0..dim {
            for i in 0..dim {
                let p = Vec3::from_array(img.point_coords(i, j, k));
                let r = p.length();
                let v = (1.0 - r).max(0.0);
                vals.push(v);
            }
        }
    }
    img.point_data
        .set_scalars(DataArray::scalars_f32("density", vals));
    DataSet::ImageData(img)
}

fn count_distinct_colors(rgba: &[u8]) -> usize {
    let mut set = std::collections::HashSet::new();
    for px in rgba.as_chunks::<4>().0 {
        set.insert([px[0], px[1], px[2]]);
    }
    set.len()
}

fn center_pixel(rgba: &[u8], width: u32, height: u32) -> [u8; 4] {
    let x = (width / 2) as usize;
    let y = (height / 2) as usize;
    let idx = (y * width as usize + x) * 4;
    [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
}

fn top_right_pixel(rgba: &[u8], width: u32) -> [u8; 4] {
    let x = (width - 2) as usize;
    let y = 1usize;
    let idx = (y * width as usize + x) * 4;
    [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
}

#[test]
fn camera_math_orbit_pan_zoom_fit_and_presets() {
    let bounds = BoundingBox::new(Vec3::new(-2.0, -1.0, -3.0), Vec3::new(4.0, 5.0, 3.0));
    let mut cam = Camera::default();

    cam.apply_preset(CameraPreset::Isometric, &bounds);
    let center = bounds.center();
    assert!((cam.target - center).length() < 1e-4);

    let dist0 = cam.distance();
    cam.orbit(0.4, 0.2);
    assert!((cam.distance() - dist0).abs() < 1e-3);
    assert!(cam.view_direction().dot(cam.up).abs() < 1e-4);

    cam.zoom(2.0);
    assert!((cam.distance() - dist0 * 0.5).abs() < 1e-3);

    let prev_pos = cam.position;
    let prev_target = cam.target;
    cam.pan(0.1, -0.05);
    let delta_pos = cam.position - prev_pos;
    let delta_target = cam.target - prev_target;
    assert!((delta_pos - delta_target).length() < 1e-5);
    assert!(delta_pos.length() > 1e-3);

    // Test all presets produce finite view-projection matrices
    let presets = [
        CameraPreset::PosX,
        CameraPreset::NegX,
        CameraPreset::PosY,
        CameraPreset::NegY,
        CameraPreset::PosZ,
        CameraPreset::NegZ,
        CameraPreset::Isometric,
    ];
    for p in presets {
        cam.apply_preset(p, &bounds);
        let vp = cam.view_proj_matrix(16.0 / 9.0);
        assert!(vp.is_finite(), "Preset {p:?} produced non-finite matrix");
        let clip_center = vp.project_point3(center);
        assert!(clip_center.z >= 0.0 && clip_center.z <= 1.0);
    }

    // Orthographic projection check
    cam.orthographic = true;
    let ortho_vp = cam.view_proj_matrix(1.5);
    assert!(ortho_vp.is_finite());
    let projected = ortho_vp.project_point3(center);
    assert!(projected.x.abs() < 1e-3 && projected.y.abs() < 1e-3);
}

#[test]
fn render_surface_rgba_and_png() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let dataset = Arc::new(make_box_polydata());

    let mut scene = RenderScene::new();
    let mut actor = SceneActor::new(1, "box", dataset);
    actor.style = ActorStyle {
        representation: Representation::Surface,
        color_by: ColorBy::PointScalar {
            array: "elevation".into(),
            component: None,
        },
        colormap: ColorMap::from_preset(ColorMapPreset::Viridis),
        ..Default::default()
    };
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    let rgba = renderer.render_to_rgba(&scene, 128, 128).expect("render rgba");
    assert_eq!(rgba.len(), 128 * 128 * 4);
    assert!(count_distinct_colors(&rgba) > 10);

    // Center pixel hits the box and differs from top-right background pixel
    assert_ne!(center_pixel(&rgba, 128, 128), top_right_pixel(&rgba, 128));

    // Write PNG and verify signature + dimensions
    let tmp = std::env::temp_dir().join("paraviz_render_surface_test.png");
    renderer
        .render_to_png(&scene, 128, 128, &tmp)
        .expect("render png");
    let bytes = std::fs::read(&tmp).expect("read png");
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn render_surface_with_edges_wireframe_points_and_outline() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let dataset = Arc::new(make_box_polydata());

    let mut scene = RenderScene::new();
    scene.settings.show_axes_gizmo = false;
    let mut actor = SceneActor::new(1, "box", dataset);
    actor.style.color_by = ColorBy::Solid([0.85, 0.35, 0.25, 1.0]);
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    // Surface vs SurfaceWithEdges should differ because of edges
    scene.actors[0].style.representation = Representation::Surface;
    let surf_rgba = renderer.render_to_rgba(&scene, 128, 128).unwrap();

    scene.actors[0].style.representation = Representation::SurfaceWithEdges;
    scene.actors[0].style.edge_width = 2.0;
    let edges_rgba = renderer.render_to_rgba(&scene, 128, 128).unwrap();
    assert_ne!(surf_rgba, edges_rgba);

    // Wireframe
    scene.actors[0].style.representation = Representation::Wireframe;
    let wire_rgba = renderer.render_to_rgba(&scene, 128, 128).unwrap();
    assert!(count_distinct_colors(&wire_rgba) >= 2);

    // Points (sphere impostors)
    scene.actors[0].style.representation = Representation::Points;
    scene.actors[0].style.point_size = 14.0;
    scene.actors[0].style.color_by = ColorBy::PointScalar {
        array: "elevation".into(),
        component: None,
    };
    let pts_rgba = renderer.render_to_rgba(&scene, 128, 128).unwrap();
    assert!(count_distinct_colors(&pts_rgba) > 8);

    // Outline + camera orbit caching check
    scene.actors[0].style.representation = Representation::Outline;
    let out1 = renderer.render_to_rgba(&scene, 128, 128).unwrap();
    scene.camera.orbit(0.5, 0.2);
    let out2 = renderer.render_to_rgba(&scene, 128, 128).unwrap();
    assert_ne!(out1, out2);
}

#[test]
fn render_volume_raymarching_sphere() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let vol_ds = Arc::new(make_sphere_imagedata(24));

    let mut tf = TransferFunction::with_preset(ColorMapPreset::Plasma, (0.0, 1.0));
    tf.opacity_points = vec![
        OpacityPoint { t: 0.0, alpha: 0.0 },
        OpacityPoint { t: 0.15, alpha: 0.0 },
        OpacityPoint { t: 0.5, alpha: 0.65 },
        OpacityPoint { t: 1.0, alpha: 0.98 },
    ];
    tf.scalar_opacity_unit_distance = 0.5;

    let mut scene = RenderScene::new();
    scene.settings.show_axes_gizmo = false;
    scene.settings.bg_top = [0.05, 0.05, 0.08];
    scene.settings.bg_bottom = [0.05, 0.05, 0.08];

    let mut actor = SceneActor::new(7, "sphere_vol", vol_ds);
    actor.style = ActorStyle {
        representation: Representation::Volume,
        color_by: ColorBy::PointScalar {
            array: "density".into(),
            component: None,
        },
        volume_tf: tf,
        volume_step_size: 0.4,
        volume_shading: true,
        ..Default::default()
    };
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    let rgba = renderer.render_to_rgba(&scene, 160, 160).expect("volume render");
    assert_eq!(rgba.len(), 160 * 160 * 4);

    let center = center_pixel(&rgba, 160, 160);
    let bg = top_right_pixel(&rgba, 160);
    assert_ne!(center, bg, "volume sphere center should differ from background");
    // Center of density=1.0 mapped with Plasma should be bright warm yellow-orange
    assert!(center[0] > 120, "expected warm bright center pixel, got {center:?}");
}

fn make_uv_sphere_polydata(radius: f32, stacks: usize, slices: usize) -> PolyData {
    let mut points = Vec::new();
    let mut scalars = Vec::new();
    for i in 0..=stacks {
        let phi = std::f32::consts::PI * (i as f32 / stacks as f32);
        let y = radius * phi.cos();
        let r_ring = radius * phi.sin();
        for j in 0..slices {
            let theta = std::f32::consts::TAU * (j as f32 / slices as f32);
            let x = r_ring * theta.cos();
            let z = r_ring * theta.sin();
            points.push([x, y, z]);
            scalars.push(y);
        }
    }
    let mut tris = Vec::new();
    for i in 0..stacks {
        for j in 0..slices {
            let j_next = (j + 1) % slices;
            let a = (i * slices + j) as u32;
            let b = (i * slices + j_next) as u32;
            let c = ((i + 1) * slices + j) as u32;
            let d = ((i + 1) * slices + j_next) as u32;
            if i != 0 {
                tris.push([a, b, d]);
            }
            if i != stacks - 1 {
                tris.push([a, d, c]);
            }
        }
    }
    let mut poly = PolyData::from_triangles(points, tris);
    poly.point_data
        .set_scalars(DataArray::scalars_f32("height", scalars));
    poly.compute_normals();
    poly
}

#[test]
fn edl_changes_shading_on_sphere() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let sphere = Arc::new(DataSet::PolyData(make_uv_sphere_polydata(1.0, 20, 32)));

    let mut scene = RenderScene::new();
    scene.settings.show_axes_gizmo = false;
    let mut actor = SceneActor::new(10, "sphere", sphere);
    actor.style.color_by = ColorBy::Solid([0.75, 0.78, 0.85, 1.0]);
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    scene.settings.edl_enabled = false;
    let rgba_off = renderer.render_to_rgba(&scene, 160, 160).expect("render no-edl");

    scene.settings.edl_enabled = true;
    scene.settings.edl_strength = 2.0;
    let rgba_on = renderer.render_to_rgba(&scene, 160, 160).expect("render edl");

    assert_ne!(rgba_off, rgba_on, "EDL should alter pixel shading on sphere");
    // Background corner should remain unaffected by EDL
    assert_eq!(top_right_pixel(&rgba_off, 160), top_right_pixel(&rgba_on, 160));
}

#[test]
fn msaa_4x_renders_cleanly() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let dataset = Arc::new(make_box_polydata());

    let mut scene = RenderScene::new();
    scene.settings.show_axes_gizmo = true;
    let mut actor = SceneActor::new(20, "box", dataset);
    actor.style.color_by = ColorBy::PointScalar {
        array: "elevation".into(),
        component: None,
    };
    actor.style.colormap = ColorMap::from_preset(ColorMapPreset::Viridis);
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::Isometric, &scene.bounds());

    scene.settings.msaa_samples = 1;
    let rgba_1x = renderer.render_to_rgba(&scene, 128, 128).expect("1x render");

    scene.settings.msaa_samples = 4;
    let rgba_4x = renderer.render_to_rgba(&scene, 128, 128).expect("4x MSAA render");
    assert_eq!(rgba_4x.len(), 128 * 128 * 4);
    assert_ne!(center_pixel(&rgba_4x, 128, 128), top_right_pixel(&rgba_4x, 128));
    assert_ne!(rgba_1x, rgba_4x, "4x MSAA should anti-alias silhouette edges differently from 1x");

    // Non-standard sample counts should clamp cleanly without panicking
    scene.settings.msaa_samples = 8;
    let rgba_clamped = renderer.render_to_rgba(&scene, 128, 128).expect("clamped MSAA render");
    assert_eq!(rgba_clamped, rgba_4x);
}

#[test]
fn pick_scene_hits_sphere_and_misses_background() {
    let sphere = Arc::new(DataSet::PolyData(make_uv_sphere_polydata(1.0, 24, 36)));
    let mut scene = RenderScene::new();
    let actor = SceneActor::new(42, "pick_sphere", sphere);
    scene.add_actor(actor);
    scene.camera.apply_preset(CameraPreset::PosZ, &scene.bounds());

    let (w, h) = (200u32, 200u32);
    let (ro, rd) = scene.camera.screen_ray(w as f32 * 0.5, h as f32 * 0.5, w, h);
    assert!((rd.length() - 1.0).abs() < 1e-4);
    assert!((ro.cross(rd)).length() < 1e-4 || rd.dot(-ro.normalize()) > 0.99);

    let hit = pick_scene(&scene, w as f32 * 0.5, h as f32 * 0.5, w, h)
        .expect("center ray should hit sphere");
    assert_eq!(hit.actor_id, 42);
    assert_eq!(hit.actor_name, "pick_sphere");
    assert!(hit.point_index.is_some());
    assert!(hit.cell_index.is_some());
    assert!(hit.scalar_value.is_some());

    let wp = Vec3::from_array(hit.world_pos);
    assert!((wp.length() - 1.0).abs() < 0.05, "hit {wp:?} not on unit sphere");
    assert!(wp.z > 0.9, "expected front-facing +Z hemisphere hit, got {wp:?}");

    // Background corner should miss
    assert!(pick_scene(&scene, 2.0, 2.0, w, h).is_none());
}

#[test]
fn viewport_widgets_render_cleanly() {
    let mut renderer = WebGpuRenderer::new_headless().expect("headless WebGPU init");
    let mut scene = RenderScene::new();
    scene.settings.show_axes_gizmo = false;
    scene.camera.position = Vec3::new(3.0, 2.5, 4.0);
    scene.camera.target = Vec3::ZERO;

    let base_rgba = renderer.render_to_rgba(&scene, 160, 160).expect("render base");

    scene.widgets = vec![
        ViewportWidget::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            size: 2.0,
        },
        ViewportWidget::Line {
            p1: [-1.2, -0.5, 0.0],
            p2: [1.2, 0.8, 0.2],
        },
        ViewportWidget::Box {
            min: [-0.8, -0.8, -0.8],
            max: [0.8, 0.8, 0.8],
        },
        ViewportWidget::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
    ];

    let widget_rgba = renderer.render_to_rgba(&scene, 160, 160).expect("render widgets");
    assert_ne!(base_rgba, widget_rgba, "widgets should render visibly over background");

    // Also verify widgets work cleanly under 4x MSAA
    scene.settings.msaa_samples = 4;
    let widget_msaa = renderer
        .render_to_rgba(&scene, 160, 160)
        .expect("render widgets 4x MSAA");
    assert_eq!(widget_msaa.len(), 160 * 160 * 4);
}

#[test]
fn gpu_compute_marching_cubes_watertight_sphere() {
    let engine = GpuContourEngine::new_headless().expect("headless compute init");
    let dim = 24usize;
    let spacing = 3.0 / (dim - 1) as f64;
    let mut img = ImageData::new([dim, dim, dim], [-1.5, -1.5, -1.5], [spacing; 3]);
    let mut sdf = Vec::with_capacity(dim * dim * dim);
    for k in 0..dim {
        for j in 0..dim {
            for i in 0..dim {
                let p = Vec3::from_array(img.point_coords(i, j, k));
                sdf.push(p.length() - 1.0);
            }
        }
    }
    img.point_data.set_scalars(DataArray::scalars_f32("sdf", sdf));

    let poly = engine
        .contour_imagedata(&img, "sdf", 0.0)
        .expect("GPU marching cubes");
    assert!(!poly.points.is_empty(), "expected non-empty points");
    assert!(!poly.polys.is_empty(), "expected non-empty triangles");

    let normals = poly
        .point_data
        .active_normals_array()
        .expect("expected GPU-computed smooth normals");
    assert_eq!(normals.num_tuples(), poly.points.len());

    let scalars = poly
        .point_data
        .get("sdf")
        .expect("expected interpolated sdf scalars");
    assert_eq!(scalars.num_tuples(), poly.points.len());

    for (i, &pt) in poly.points.iter().enumerate() {
        let p = Vec3::from_array(pt);
        assert!(
            (p.length() - 1.0).abs() < 0.08,
            "point {p:?} radius {} too far from 1.0",
            p.length()
        );
        assert!(
            scalars.get_scalar(i).abs() < 0.05,
            "interpolated scalar {} not near iso=0",
            scalars.get_scalar(i)
        );
        let n = Vec3::from_array(normals.get_vec3(i));
        assert!((n.length() - 1.0).abs() < 1e-3, "non-unit normal {n:?}");
        assert!(
            n.dot(p.normalize()) > 0.9,
            "normal {n:?} should point radially outward at {p:?}"
        );
    }

    // Verify watertight closed 2-manifold (every undirected edge shared by exactly 2 triangles)
    let mut edge_counts: HashMap<(u32, u32), usize> = HashMap::new();
    for tri in &poly.polys {
        assert_eq!(tri.len(), 3);
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }
    assert!(
        edge_counts.values().all(|&c| c == 2),
        "expected watertight closed 2-manifold with every edge having degree 2"
    );
}
