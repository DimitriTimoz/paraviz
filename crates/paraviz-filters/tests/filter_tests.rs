use glam::Vec3;
use paraviz_core::{
    AttributeAssociation, CellType, DataArray, DataSet, Filter, ImageData, MultiBlockDataSet,
    PipelineGraph, PolyData, RectilinearGrid, StructuredGrid, UnstructuredGrid,
};
use paraviz_filters::{
    extract_surface, CalculatorFilter, CellDataToPointDataFilter, ClipFilter, ContourFilter,
    ExtractSurface, GlyphFilter, GlyphShape, GradientFilter, ImplicitFunction,
    IntegrationDirection, Integrator, PointDataToCellDataFilter, ProbeOverLineFilter, ScaleMode,
    SeedSource, SliceFilter, SmoothFilter, SmoothingMethod, StreamTracerFilter, ThresholdFilter,
    TransformFilter, TubeFilter, WarpByScalarFilter, WarpByVectorFilter,
};
use paraviz_sources::{LineSource, PlaneSource, SphereSource, VectorFieldSource, WaveletSource};
use std::collections::HashMap;

fn make_sphere_image(dims: [usize; 3], half: f64) -> ImageData {
    let [nx, ny, nz] = dims;
    let origin = [-half, -half, -half];
    let spacing = [
        (2.0 * half) / (nx - 1) as f64,
        (2.0 * half) / (ny - 1) as f64,
        (2.0 * half) / (nz - 1) as f64,
    ];
    let mut img = ImageData::new(dims, origin, spacing);
    let mut r2 = Vec::with_capacity(img.num_points());
    let mut lin = Vec::with_capacity(img.num_points());
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = img.point_coords(i, j, k);
                r2.push(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                lin.push(2.0 * p[0] + 3.0 * p[1] - 4.0 * p[2]);
            }
        }
    }
    img.point_data.set_scalars(DataArray::scalars_f32("r2", r2));
    img.point_data
        .add_array(DataArray::scalars_f32("linear", lin));
    img
}

fn make_rectilinear_grid() -> RectilinearGrid {
    let coords = vec![-2.0f32, -0.5, 0.5, 2.0];
    let mut rg = RectilinearGrid {
        x_coords: coords.clone(),
        y_coords: coords.clone(),
        z_coords: coords,
        ..Default::default()
    };
    let mut lin = Vec::with_capacity(rg.num_points());
    let mut r2 = Vec::with_capacity(rg.num_points());
    let mut vel = Vec::with_capacity(rg.num_points());
    for p in rg.all_points() {
        lin.push(2.0 * p[0] + 3.0 * p[1] - 4.0 * p[2]);
        r2.push(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        vel.push([1.0, 0.0, 0.0]);
    }
    rg.point_data.set_scalars(DataArray::scalars_f32("r2", r2));
    rg.point_data.add_array(DataArray::scalars_f32("linear", lin));
    rg.point_data.set_vectors(DataArray::vectors_f32("vel", vel));
    rg
}

fn make_structured_grid(dims: [usize; 3]) -> StructuredGrid {
    let img = make_sphere_image(dims, 2.0);
    let points = img.all_points();
    let mut vel = Vec::with_capacity(points.len());
    for p in &points {
        vel.push([-p[1], p[0], 0.0]);
    }
    let mut sg = StructuredGrid {
        dimensions: dims,
        points,
        point_data: img.point_data,
        ..Default::default()
    };
    sg.point_data.set_vectors(DataArray::vectors_f32("vel", vel));
    sg
}

fn image_to_unstructured_hex(img: &ImageData) -> UnstructuredGrid {
    let [nx, ny, nz] = img.dimensions;
    let mut ug = UnstructuredGrid::new();
    ug.points = img.all_points();
    ug.point_data = img.point_data.clone();
    for k in 0..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let idx = |ii, jj, kk| img.point_index(ii, jj, kk) as u32;
                ug.push_cell(
                    CellType::Hexahedron,
                    &[
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    ],
                );
            }
        }
    }
    ug
}

#[test]
fn test_extract_surface_all_dataset_types() {
    // 3x3x3 grid has 2x2x2 = 8 cells, 6 * 4 = 24 boundary faces (no interior faces).
    let img = make_sphere_image([3, 3, 3], 1.0);
    let surf_img = extract_surface(&DataSet::ImageData(img.clone())).unwrap();
    assert_eq!(surf_img.polys.len(), 24);
    assert_eq!(surf_img.num_points(), 26); // 27 - 1 center interior point compacted away!
    assert!(surf_img.point_data.active_normals_array().is_some());

    // Rectilinear
    let rg = make_rectilinear_grid(); // 4x4x4 -> 3x3x3 cells -> 6 * 9 = 54 boundary faces
    let surf_rg = extract_surface(&DataSet::RectilinearGrid(rg)).unwrap();
    assert_eq!(surf_rg.polys.len(), 54);
    assert_eq!(surf_rg.num_points(), 64 - 8); // 8 interior vertices compacted

    // Structured
    let sg = make_structured_grid([3, 3, 3]);
    let surf_sg = extract_surface(&DataSet::StructuredGrid(sg)).unwrap();
    assert_eq!(surf_sg.polys.len(), 24);
    assert_eq!(surf_sg.num_points(), 26);

    // UnstructuredGrid with canonical face hashing: 2x2x2 hexes -> 24 boundary faces
    let ug = image_to_unstructured_hex(&img);
    let surf_ug = ExtractSurface::new().apply(&DataSet::UnstructuredGrid(ug)).unwrap();
    let surf_ug = surf_ug.as_poly_data().unwrap();
    assert_eq!(surf_ug.polys.len(), 24);
    assert_eq!(surf_ug.num_points(), 26);

    // PolyData without normals gets normals computed
    let mut pd = PolyData::from_triangles(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    assert!(pd.point_data.active_normals_array().is_none());
    pd = extract_surface(&DataSet::PolyData(pd)).unwrap();
    let n = pd.point_data.active_normals_array().unwrap().get_vec3(0);
    assert!((n[2] - 1.0).abs() < 1e-5);

    // MultiBlock
    let mut mb = MultiBlockDataSet::new();
    mb.add_block("a", DataSet::ImageData(img));
    mb.add_block("b", DataSet::PolyData(SphereSource::default().build()));
    let surf_mb = extract_surface(&DataSet::MultiBlock(mb)).unwrap();
    assert!(surf_mb.polys.len() > 24);
}

#[test]
fn test_contour_marching_cubes_and_tetrahedra() {
    let img = make_sphere_image([17, 17, 17], 3.0);
    let filter = ContourFilter {
        field_name: "r2".into(),
        iso_values: vec![4.0],
        compute_normals: true,
        merge_vertices: true,
    };
    let res = filter.apply(&DataSet::ImageData(img.clone())).unwrap();
    let pd = res.as_poly_data().unwrap();
    assert!(!pd.polys.is_empty());
    assert!(pd.point_data.active_normals_array().is_some());

    // Every vertex should lie near sphere radius sqrt(4) = 2.0 and have interpolated r2 ~= 4.0.
    let r2_arr = pd.point_data.get("r2").unwrap();
    for (i, &p) in pd.points.iter().enumerate() {
        let r = Vec3::from_array(p).length();
        assert!((r - 2.0).abs() < 0.15, "point radius {r} too far from 2.0");
        assert!((r2_arr.get_scalar(i) - 4.0).abs() < 0.05);
    }

    // Verify closed 2-manifold (every edge shared by exactly 2 triangles) when merge_vertices = true!
    let mut edge_counts: HashMap<(u32, u32), usize> = HashMap::new();
    for tri in &pd.polys {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }
    assert!(
        edge_counts.values().all(|&c| c == 2),
        "expected watertight closed manifold with edge degree 2"
    );

    // Unmerged should have 3 * num_tris vertices
    let unmerged = ContourFilter {
        merge_vertices: false,
        ..filter.clone()
    }
    .apply(&DataSet::ImageData(img))
    .unwrap();
    let pd_unmerged = unmerged.as_poly_data().unwrap();
    assert_eq!(pd_unmerged.num_points(), pd_unmerged.polys.len() * 3);
    assert!(pd.num_points() < pd_unmerged.num_points());

    // Multi-iso + RectilinearGrid + StructuredGrid
    let rg = make_rectilinear_grid();
    let rg_iso = ContourFilter::new("r2", vec![1.0, 2.5])
        .apply(&DataSet::RectilinearGrid(rg))
        .unwrap();
    assert!(!rg_iso.as_poly_data().unwrap().polys.is_empty());

    let sg = make_structured_grid([9, 9, 9]);
    let sg_iso = ContourFilter::new("r2", vec![2.25])
        .apply(&DataSet::StructuredGrid(sg))
        .unwrap();
    assert!(!sg_iso.as_poly_data().unwrap().polys.is_empty());

    // Marching Tetrahedra on UnstructuredGrid
    let small_img = make_sphere_image([9, 9, 9], 3.0);
    let ug = image_to_unstructured_hex(&small_img);
    let ug_iso = ContourFilter::new("r2", vec![4.0])
        .apply(&DataSet::UnstructuredGrid(ug))
        .unwrap();
    let ug_pd = ug_iso.as_poly_data().unwrap();
    assert!(!ug_pd.polys.is_empty());
    for &p in &ug_pd.points {
        let r = Vec3::from_array(p).length();
        assert!((r - 2.0).abs() < 0.35);
    }
}

#[test]
fn test_slice_filter_smooth_and_crinkle() {
    let img = make_sphere_image([9, 9, 9], 2.0);
    let slice = SliceFilter {
        origin: [0.25, 0.0, 0.0],
        normal: [1.0, 1.0, 0.0],
        crinkle: false,
    };
    let out = slice.apply(&DataSet::ImageData(img.clone())).unwrap();
    let pd = out.as_poly_data().unwrap();
    assert!(!pd.polys.is_empty());

    let n = Vec3::new(1.0, 1.0, 0.0).normalize();
    let orig = Vec3::new(0.25, 0.0, 0.0);
    let lin_arr = pd.point_data.get("linear").unwrap();
    for (i, &p) in pd.points.iter().enumerate() {
        let d = (Vec3::from_array(p) - orig).dot(n);
        assert!(d.abs() < 1e-4, "slice point off plane by {d}");
        let expected_lin = 2.0 * p[0] + 3.0 * p[1] - 4.0 * p[2];
        assert!((lin_arr.get_scalar(i) as f32 - expected_lin).abs() < 1e-3);
    }

    // Crinkle slice
    let crinkle_out = SliceFilter {
        crinkle: true,
        ..slice.clone()
    }
    .apply(&DataSet::ImageData(img.clone()))
    .unwrap();
    assert!(!crinkle_out.as_poly_data().unwrap().polys.is_empty());

    // UnstructuredGrid slice
    let ug = image_to_unstructured_hex(&img);
    let ug_slice = slice.apply(&DataSet::UnstructuredGrid(ug)).unwrap();
    assert!(!ug_slice.as_poly_data().unwrap().polys.is_empty());

    // PolyData slice -> line segments
    let sphere = SphereSource::default().build();
    let pd_slice = SliceFilter::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        .apply(&DataSet::PolyData(sphere))
        .unwrap();
    assert!(!pd_slice.as_poly_data().unwrap().lines.is_empty());
}

#[test]
fn test_clip_filter_plane_sphere_box() {
    let img = make_sphere_image([9, 9, 9], 2.0);
    // Clip with Sphere (invert = true keeps inside sphere r <= 1.2)
    let clip_in = ClipFilter {
        function: ImplicitFunction::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.2,
        },
        invert: true,
        crinkle: false,
    };
    let res = clip_in.apply(&DataSet::ImageData(img.clone())).unwrap();
    let ug = res.as_unstructured_grid().unwrap();
    assert!(ug.num_cells() > 0);
    for &p in &ug.points {
        assert!(Vec3::from_array(p).length() <= 1.2 + 1e-4);
    }

    // Clip with Box (invert = false keeps outside box)
    let clip_box = ClipFilter {
        function: ImplicitFunction::Box {
            min: [-0.5, -0.5, -0.5],
            max: [0.5, 0.5, 0.5],
        },
        invert: false,
        crinkle: false,
    };
    let box_res = clip_box.apply(&DataSet::ImageData(img)).unwrap();
    assert!(box_res.as_unstructured_grid().unwrap().num_cells() > 0);

    // Clip PolyData with Plane
    let sphere = SphereSource::default().build();
    let clip_pd = ClipFilter::new(ImplicitFunction::Plane {
        origin: [0.0, 0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    })
    .apply(&DataSet::PolyData(sphere))
    .unwrap();
    let pd = clip_pd.as_poly_data().unwrap();
    assert!(!pd.polys.is_empty());
    for &p in &pd.points {
        assert!(p[1] >= -1e-5);
    }
}

#[test]
fn test_threshold_filter() {
    let mut img = make_sphere_image([5, 5, 5], 2.0);
    let n_cells = img.num_cells();
    let cell_ids: Vec<f32> = (0..n_cells).map(|i| i as f32).collect();
    img.cell_data
        .set_scalars(DataArray::scalars_f32("cell_id", cell_ids));

    // Point threshold + invert partitions cells cleanly
    let t_keep = ThresholdFilter {
        field_name: "r2".into(),
        association: AttributeAssociation::Point,
        min: 0.0,
        max: 4.0,
        invert: false,
        all_scalars: true,
    };
    let t_inv = ThresholdFilter {
        invert: true,
        ..t_keep.clone()
    };
    let kept = t_keep.apply(&DataSet::ImageData(img.clone())).unwrap();
    let inverted = t_inv.apply(&DataSet::ImageData(img.clone())).unwrap();
    assert_eq!(kept.num_cells() + inverted.num_cells(), n_cells);
    assert!(kept.num_points() < img.num_points());

    // Cell threshold
    let t_cell = ThresholdFilter {
        field_name: "cell_id".into(),
        association: AttributeAssociation::Cell,
        min: 10.0,
        max: 19.0,
        invert: false,
        all_scalars: true,
    };
    let cell_res = t_cell.apply(&DataSet::ImageData(img)).unwrap();
    assert_eq!(cell_res.num_cells(), 10);

    // PolyData threshold preserves PolyData output
    let sphere = SphereSource::default().build();
    let pd_res = ThresholdFilter::new("Elevation", 0.0, 1.0)
        .apply(&DataSet::PolyData(sphere))
        .unwrap();
    assert!(pd_res.as_poly_data().is_some());
    assert!(pd_res.num_cells() > 0);
}

#[test]
fn test_streamlines_rk4_and_rk45() {
    // 1. Uniform flow on ImageData: verify exact IntegrationTime, ArcLength, Speed
    let mut uniform = ImageData::new([9, 9, 9], [-2.0, -2.0, -2.0], [0.5, 0.5, 0.5]);
    let vels = vec![[2.0f32, 0.0, 0.0]; uniform.num_points()];
    let temps: Vec<f32> = uniform.all_points().iter().map(|p| p[1] * 10.0).collect();
    uniform.point_data.set_vectors(DataArray::vectors_f32("Velocity", vels));
    uniform.point_data.set_scalars(DataArray::scalars_f32("Temp", temps));

    let tracer = StreamTracerFilter {
        vector_field: "Velocity".into(),
        seeds: SeedSource::Points(vec![[-1.0, 0.5, 0.0]]),
        direction: IntegrationDirection::Forward,
        integrator: Integrator::Rk4,
        step_size: 0.25,
        max_steps: 20,
        max_length: 2.0,
        terminal_speed: 1e-6,
    };
    let out = tracer.apply(&DataSet::ImageData(uniform)).unwrap();
    let pd = out.as_poly_data().unwrap();
    assert_eq!(pd.lines.len(), 1);
    let last = pd.points.last().unwrap();
    assert!((last[0] - 1.0).abs() < 1e-4);
    assert!((last[1] - 0.5).abs() < 1e-4);

    let arc = pd.point_data.get("ArcLength").unwrap();
    let time = pd.point_data.get("IntegrationTime").unwrap();
    let speed = pd.point_data.get("Speed").unwrap();
    let temp = pd.point_data.get("Temp").unwrap();
    let n = pd.num_points();
    assert!((arc.get_scalar(n - 1) - 2.0).abs() < 1e-4);
    assert!((time.get_scalar(n - 1) - 1.0).abs() < 1e-4);
    assert!((speed.get_scalar(n - 1) - 2.0).abs() < 1e-4);
    assert!((temp.get_scalar(n - 1) - 5.0).abs() < 1e-4);

    // 2. SwirlingVortex with RK45 + Both directions + Line/Sphere seeds
    let vortex = VectorFieldSource::default().build(0.0);
    let tracer_rk45 = StreamTracerFilter {
        vector_field: "Velocity".into(),
        seeds: SeedSource::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 0.8,
            count: 8,
        },
        direction: IntegrationDirection::Both,
        integrator: Integrator::Rk45,
        step_size: 0.1,
        max_steps: 100,
        max_length: 4.0,
        terminal_speed: 1e-5,
    };
    let res_rk45 = tracer_rk45.apply(&DataSet::ImageData(vortex)).unwrap();
    assert_eq!(res_rk45.as_poly_data().unwrap().lines.len(), 8);

    // 3. RectilinearGrid, StructuredGrid, UnstructuredGrid
    let rg = make_rectilinear_grid();
    let rg_lines = StreamTracerFilter {
        vector_field: "vel".into(),
        seeds: SeedSource::Line {
            p1: [-1.0, -0.5, 0.0],
            p2: [-1.0, 0.5, 0.0],
            count: 3,
        },
        direction: IntegrationDirection::Forward,
        integrator: Integrator::Rk4,
        step_size: 0.2,
        max_steps: 15,
        max_length: 2.0,
        terminal_speed: 1e-6,
    }
    .apply(&DataSet::RectilinearGrid(rg))
    .unwrap();
    assert_eq!(rg_lines.as_poly_data().unwrap().lines.len(), 3);

    let sg = make_structured_grid([7, 7, 7]);
    let sg_lines = StreamTracerFilter {
        vector_field: "vel".into(),
        seeds: SeedSource::Points(vec![[0.8, 0.0, 0.0]]),
        direction: IntegrationDirection::Forward,
        integrator: Integrator::Rk4,
        step_size: 0.1,
        max_steps: 20,
        max_length: 1.5,
        terminal_speed: 1e-6,
    }
    .apply(&DataSet::StructuredGrid(sg))
    .unwrap();
    assert_eq!(sg_lines.as_poly_data().unwrap().lines.len(), 1);
}

#[test]
fn test_tube_filter_bishop_frames_and_varying_radius() {
    // 90-degree bent polyline + scalar varying from 0 to 1
    let mut line_pd = LineSource {
        point1: [0.0, 0.0, 0.0],
        point2: [2.0, 0.0, 0.0],
        resolution: 8,
    }
    .build();
    // Bend the second half upward along +Y to test Bishop parallel transport
    for p in line_pd.points.iter_mut().skip(5) {
        p[1] = p[0] - 1.0;
        p[0] = 1.0;
    }

    let tube = TubeFilter {
        radius: 0.1,
        vary_radius_by_scalar: Some("ArcLength".into()),
        radius_factor: 2.0,
        sides: 8,
        capping: true,
    };
    let out = tube.apply(&DataSet::PolyData(line_pd)).unwrap();
    let pd = out.as_poly_data().unwrap();
    // 9 rings * 8 sides + 2 caps * (1 + 8) = 72 + 18 = 90 points
    assert_eq!(pd.num_points(), 90);
    // 8 segments * 8 * 2 tris + 2 caps * 8 tris = 128 + 16 = 144 triangles
    assert_eq!(pd.polys.len(), 144);
    assert!(pd.point_data.active_normals_array().is_some());
    assert!(pd.point_data.get("ArcLength").is_some());

    // First ring radius should be ~0.1, last ring radius should be ~0.2
    let p_start = Vec3::from_array(pd.points[0]);
    assert!((p_start.length() - 0.1).abs() < 1e-4);
    let last_ring_p = Vec3::from_array(pd.points[8 * 8]);
    let last_center = Vec3::new(1.0, 1.0, 0.0);
    assert!(((last_ring_p - last_center).length() - 0.2).abs() < 1e-4);
}

#[test]
fn test_glyph_filter_shapes_and_scaling() {
    let mut pts_pd = PolyData::new();
    pts_pd.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    pts_pd.verts = vec![0, 1, 2];
    pts_pd.point_data.set_vectors(DataArray::vectors_f32(
        "vel",
        vec![[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [-1.0, 0.0, 0.0]],
    ));
    pts_pd
        .point_data
        .set_scalars(DataArray::scalars_f32("temp", vec![10.0, 20.0, 30.0]));

    for shape in [
        GlyphShape::Arrow,
        GlyphShape::Cone,
        GlyphShape::Sphere,
        GlyphShape::Cube,
    ] {
        let g = GlyphFilter {
            shape,
            vector_field: Some("vel".into()),
            scalar_field: Some("temp".into()),
            scale_mode: ScaleMode::VectorMagnitude,
            scale_factor: 0.5,
            max_points: 2,
        };
        let out = g.apply(&DataSet::PolyData(pts_pd.clone())).unwrap();
        let out_pd = out.as_poly_data().unwrap();
        assert!(!out_pd.polys.is_empty());
        assert!(out_pd.point_data.get("temp").is_some());
        assert!(out_pd.point_data.active_normals_array().is_some());
    }
}

#[test]
fn test_warp_filters() {
    let plane = PlaneSource {
        x_resolution: 2,
        y_resolution: 2,
        ..Default::default()
    }
    .build();
    let warped_s = WarpByScalarFilter {
        scalar_field: "RadialDist".into(),
        normal: [0.0, 0.0, 1.0],
        use_normal: true,
        scale_factor: 2.0,
    }
    .apply(&DataSet::PolyData(plane.clone()))
    .unwrap();
    let w_pd = warped_s.as_poly_data().unwrap();
    let r_arr = plane.point_data.get("RadialDist").unwrap();
    for (i, &p) in w_pd.points.iter().enumerate() {
        assert!((p[2] - 2.0 * r_arr.get_scalar(i) as f32).abs() < 1e-5);
    }

    // WarpByVector on ImageData -> StructuredGrid
    let mut img = ImageData::new([3, 3, 3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    img.point_data.set_vectors(DataArray::vectors_f32(
        "disp",
        vec![[1.0, 2.0, 3.0]; img.num_points()],
    ));
    let warped_v = WarpByVectorFilter::new("disp", 0.5)
        .apply(&DataSet::ImageData(img))
        .unwrap();
    assert_eq!(warped_v.type_name(), "Structured (Curvilinear) Grid");
    assert!((warped_v.bounds().min.x - 0.5).abs() < 1e-5);
    assert!((warped_v.bounds().min.y - 1.0).abs() < 1e-5);
    assert!((warped_v.bounds().min.z - 1.5).abs() < 1e-5);
}

#[test]
fn test_gradient_filter_scalar_and_vector_quantities() {
    // 1. Linear scalar field f(x,y,z) = 2x + 3y - 4z -> exact gradient [2, 3, -4]
    let img = make_sphere_image([5, 5, 5], 2.0);
    let g_img = GradientFilter::new("linear")
        .apply(&DataSet::ImageData(img.clone()))
        .unwrap();
    let grad = g_img
        .point_data()
        .unwrap()
        .get("Gradient")
        .unwrap();
    for i in 0..g_img.num_points() {
        let v = grad.get_vec3(i);
        assert!((v[0] - 2.0).abs() < 1e-4);
        assert!((v[1] - 3.0).abs() < 1e-4);
        assert!((v[2] + 4.0).abs() < 1e-4);
    }

    // Same on UnstructuredGrid (weighted least squares)
    let ug = image_to_unstructured_hex(&img);
    let g_ug = GradientFilter::new("linear")
        .apply(&DataSet::UnstructuredGrid(ug))
        .unwrap();
    let grad_ug = g_ug.point_data().unwrap().get("Gradient").unwrap();
    for i in 0..g_ug.num_points() {
        let v = grad_ug.get_vec3(i);
        assert!((v[0] - 2.0).abs() < 1e-3);
        assert!((v[1] - 3.0).abs() < 1e-3);
        assert!((v[2] + 4.0).abs() < 1e-3);
    }

    // 2. Rigid body rotation u = (-y, x, 0) -> Div = 0, Vorticity = (0, 0, 2), Q = 1
    let mut rot_img = ImageData::new([5, 5, 5], [-1.0, -1.0, -1.0], [0.5, 0.5, 0.5]);
    let rot_vel: Vec<[f32; 3]> = rot_img
        .all_points()
        .iter()
        .map(|p| [-p[1], p[0], 0.0])
        .collect();
    rot_img
        .point_data
        .set_vectors(DataArray::vectors_f32("u", rot_vel));
    let res = GradientFilter::with_all_vector_quantities("u")
        .apply(&DataSet::ImageData(rot_img))
        .unwrap();
    let pdata = res.point_data().unwrap();
    let div = pdata.get("Divergence").unwrap();
    let vort = pdata.get("Vorticity").unwrap();
    let q = pdata.get("QCriterion").unwrap();
    for i in 0..res.num_points() {
        assert!(div.get_scalar(i).abs() < 1e-4);
        let w = vort.get_vec3(i);
        assert!(w[0].abs() < 1e-4 && w[1].abs() < 1e-4 && (w[2] - 2.0).abs() < 1e-4);
        assert!((q.get_scalar(i) - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_calculator_filter() {
    let mut img = ImageData::new([4, 4, 4], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]);
    let vels = vec![[3.0f32, 4.0, 0.0]; img.num_points()];
    img.point_data
        .set_vectors(DataArray::vectors_f32("vel", vels));

    let calc = CalculatorFilter::new(
        "coordsX * 2 + (-coordsY) + mag(vel) + sin(0) + cos(0) + clamp(sqrt(abs(-9)), 0, 2) + min(5, max(1, exp(0) + ln(1))) + 2^3",
        "expr_out",
        AttributeAssociation::Point,
    );
    let out = calc.apply(&DataSet::ImageData(img)).unwrap();
    let arr = out.point_data().unwrap().get("expr_out").unwrap();
    // At index 0: coord = (1, 2, 3) => 2 - 2 + 5 + 0 + 1 + 2 + 1 + 8 = 17
    assert!((arr.get_scalar(0) - 17.0).abs() < 1e-4);

    // Invalid expression produces error
    let bad = CalculatorFilter::new("unknown_var + 1", "x", AttributeAssociation::Point);
    assert!(bad.apply(&out).is_err());
}

#[test]
fn test_smooth_filter_laplacian_and_taubin() {
    let mut sphere = SphereSource {
        radius: 1.0,
        theta_resolution: 24,
        phi_resolution: 16,
        ..Default::default()
    }
    .build();
    // Add deterministic radial perturbation
    for (i, p) in sphere.points.iter_mut().enumerate() {
        let v = Vec3::from_array(*p);
        let noise = if i % 2 == 0 { 0.08 } else { -0.08 };
        *p = (v * (1.0 + noise)).to_array();
    }

    let lap = SmoothFilter::new(SmoothingMethod::Laplacian, 20)
        .apply(&DataSet::PolyData(sphere.clone()))
        .unwrap();
    let tau = SmoothFilter::new(SmoothingMethod::Taubin, 20)
        .apply(&DataSet::PolyData(sphere))
        .unwrap();

    let mean_radius = |pd: &PolyData| -> f32 {
        pd.points
            .iter()
            .map(|&p| Vec3::from_array(p).length())
            .sum::<f32>()
            / pd.num_points() as f32
    };
    let lap_r = mean_radius(lap.as_poly_data().unwrap());
    let tau_r = mean_radius(tau.as_poly_data().unwrap());
    // Taubin should preserve radius ~1.0 much closer than Laplacian shrinkage!
    assert!((tau_r - 1.0).abs() < (lap_r - 1.0).abs());
    assert!((tau_r - 1.0).abs() < 0.05);
}

#[test]
fn test_transform_filter() {
    let pd = PolyData::from_triangles(
        vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        vec![[0, 1, 2]],
    );
    let mut ds_pd = pd;
    ds_pd
        .point_data
        .set_normals(DataArray::vectors_f32("Normals", vec![[0.0, 0.0, 1.0]; 3]));
    ds_pd
        .point_data
        .set_vectors(DataArray::vectors_f32("Vel", vec![[1.0, 0.0, 0.0]; 3]));

    let tf = TransformFilter::new([10.0, 0.0, 0.0], [0.0, 0.0, 90.0], [2.0, 2.0, 2.0]);
    let out = tf.apply(&DataSet::PolyData(ds_pd)).unwrap();
    let out_pd = out.as_poly_data().unwrap();
    // Point (1,0,0) scaled by 2 -> (2,0,0), rotated +90deg around Z -> (0,2,0), translated -> (10,2,0)
    let p0 = out_pd.points[0];
    assert!((p0[0] - 10.0).abs() < 1e-4);
    assert!((p0[1] - 2.0).abs() < 1e-4);
    assert!(p0[2].abs() < 1e-4);

    // Vector (1,0,0) scaled by 2 and rotated 90deg Z -> (0, 2, 0)
    let v0 = out_pd.point_data.get("Vel").unwrap().get_vec3(0);
    assert!(v0[0].abs() < 1e-4 && (v0[1] - 2.0).abs() < 1e-4);
}

#[test]
fn test_cell_point_conversion_and_probe_over_line() {
    let mut img = make_sphere_image([5, 5, 5], 2.0);
    // PointDataToCellData
    let c_ds = PointDataToCellDataFilter::new()
        .apply(&DataSet::ImageData(img.clone()))
        .unwrap();
    let cell_lin = c_ds.cell_data().unwrap().get("linear").unwrap();
    assert_eq!(cell_lin.num_tuples(), img.num_cells());

    // CellDataToPointData
    img.cell_data
        .set_scalars(DataArray::scalars_f32("const_c", vec![7.0; img.num_cells()]));
    let p_ds = CellDataToPointDataFilter::new()
        .apply(&DataSet::ImageData(img.clone()))
        .unwrap();
    let pt_const = p_ds.point_data().unwrap().get("const_c").unwrap();
    assert_eq!(pt_const.num_tuples(), img.num_points());
    assert!((pt_const.get_scalar(0) - 7.0).abs() < 1e-5);

    // ProbeOverLine across domain + outside
    let probe = ProbeOverLineFilter::new([-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], 60);
    let probed = probe.apply(&DataSet::ImageData(img)).unwrap();
    let pd = probed.as_poly_data().unwrap();
    assert_eq!(pd.num_points(), 61);
    assert_eq!(pd.lines.len(), 1);
    let mask = pd.point_data.get("vtkValidPointMask").unwrap();
    let lin = pd.point_data.get("linear").unwrap();
    let arc = pd.point_data.get("ArcLength").unwrap();
    assert_eq!(mask.get_scalar(0) as u8, 0); // x = -3 is outside [-2, 2]
    assert_eq!(mask.get_scalar(30) as u8, 1); // x = 0 is inside
    assert!((arc.get_scalar(60) - 6.0).abs() < 1e-4);
    // At index 40: x = -3 + 40 * 0.1 = 1.0, y=0, z=0 -> linear = 2.0
    assert!((lin.get_scalar(40) - 2.0).abs() < 1e-3);
}

#[test]
fn test_pipeline_graph_integration() {
    let mut graph = PipelineGraph::new();
    let src = graph.add_source("Wavelet", Box::new(WaveletSource::with_extent(8)));
    let iso = graph
        .add_filter(
            "Contour",
            src,
            Box::new(ContourFilter::new("RTData", vec![150.0])),
        )
        .unwrap();
    let clip = graph
        .add_filter(
            "Clip",
            iso,
            Box::new(ClipFilter::new(ImplicitFunction::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            })),
        )
        .unwrap();
    let smooth = graph
        .add_filter(
            "Smooth",
            clip,
            Box::new(SmoothFilter::new(SmoothingMethod::Taubin, 5)),
        )
        .unwrap();
    let xform = graph
        .add_filter(
            "Transform",
            smooth,
            Box::new(TransformFilter::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])),
        )
        .unwrap();

    let out = graph.update(xform, 0.0).unwrap();
    let pd = out.as_poly_data().unwrap();
    assert!(!pd.polys.is_empty());
    assert!(pd.bounds().min.x >= 1.0 - 1e-3);
}
