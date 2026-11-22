use paraviz_core::Source;
use paraviz_sources::*;

#[test]
fn all_geometric_sources_produce_valid_polydata() {
    let sources: Vec<Box<dyn Source>> = vec![
        Box::new(SphereSource::default()),
        Box::new(CubeSource::default()),
        Box::new(ConeSource::default()),
        Box::new(CylinderSource::default()),
        Box::new(PlaneSource::default()),
        Box::new(LineSource::default()),
        Box::new(ArrowSource::default()),
        Box::new(DiskSource::default()),
        Box::new(TorusSource::default()),
        Box::new(PointCloudSource::default()),
        Box::new(SuperquadricSource::default()),
    ];

    for src in sources {
        let ds = src.execute(0.0).expect("source execute");
        assert!(ds.num_points() > 0, "{} produced 0 points", src.name());
        assert!(ds.num_cells() > 0, "{} produced 0 cells", src.name());
        assert!(ds.bounds().is_valid(), "{} invalid bounds", src.name());
    }
}

#[test]
fn wavelet_and_vector_field_sources() {
    let wav = WaveletSource::default();
    let ds0 = wav.execute(0.0).unwrap();
    let ds1 = wav.execute(0.25).unwrap();
    let img0 = ds0.as_image_data().unwrap();
    let img1 = ds1.as_image_data().unwrap();
    assert_eq!(img0.dimensions, [21, 21, 21]);
    let arr0 = img0.point_data.get("RTData").unwrap();
    let arr1 = img1.point_data.get("RTData").unwrap();
    let (mn, mx) = arr0.range(Some(0));
    assert!(mx > mn + 50.0);
    assert_ne!(arr0.get_scalar(0), arr1.get_scalar(0));

    for kind in [
        VectorFieldKind::SwirlingVortex,
        VectorFieldKind::Dipole,
        VectorFieldKind::HelicalTorus,
    ] {
        let vf = VectorFieldSource {
            kind,
            ..Default::default()
        };
        let ds = vf.execute(0.0).unwrap();
        let img = ds.as_image_data().unwrap();
        assert!(img.point_data.get("Velocity").is_some());
        assert!(img.point_data.get("Speed").is_some());
    }

    let gyroid = GyroidSource::default();
    let gds = gyroid.execute(0.0).unwrap();
    let gimg = gds.as_image_data().unwrap();
    assert_eq!(gimg.dimensions, [32, 32, 32]);
    let garr = gimg.point_data.get("Gyroid").unwrap();
    let (gmin, gmax) = garr.range(Some(0));
    assert!(gmin < -1.0 && gmax > 1.0);
}
