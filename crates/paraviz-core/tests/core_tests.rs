use paraviz_core::*;

struct ConstWaveSource {
    value: f32,
}

impl Source for ConstWaveSource {
    fn name(&self) -> &str {
        "ConstWave"
    }

    fn execute(&self, time: f64) -> std::result::Result<DataSet, PipelineError> {
        let mut img = ImageData::new([4, 4, 4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let n = img.num_points();
        let vals = vec![self.value + time as f32; n];
        img.point_data.set_scalars(DataArray::scalars_f32("S", vals));
        Ok(DataSet::ImageData(img))
    }
}

struct ScaleFilter {
    factor: f32,
}

impl Filter for ScaleFilter {
    fn name(&self) -> &str {
        "Scale"
    }

    fn apply(&self, input: &DataSet) -> std::result::Result<DataSet, PipelineError> {
        let mut out = input.clone();
        if let Some(pd) = out.point_data_mut()
            && let Some(arr) = pd.get("S").cloned()
        {
            let scaled: Vec<f32> = arr
                .to_scalar_f32(Some(0))
                .into_iter()
                .map(|v| v * self.factor)
                .collect();
            pd.set_scalars(DataArray::scalars_f32("S", scaled));
        }
        Ok(out)
    }
}

#[test]
fn pipeline_dirty_propagation_and_caching() {
    let mut graph = PipelineGraph::new();
    let src = graph.add_source("Src", Box::new(ConstWaveSource { value: 2.0 }));
    let flt = graph
        .add_filter("Scale", src, Box::new(ScaleFilter { factor: 3.0 }))
        .unwrap();

    let out1 = graph.update(flt, 0.0).unwrap();
    let s1 = out1.point_data().unwrap().get("S").unwrap().get_scalar(0);
    assert!((s1 - 6.0).abs() < 1e-6);

    // Cached call returns identical Arc
    let out2 = graph.update(flt, 0.0).unwrap();
    assert!(std::sync::Arc::ptr_eq(&out1, &out2));

    // Update source invalidates downstream
    graph
        .update_source(src, Box::new(ConstWaveSource { value: 5.0 }))
        .unwrap();
    let out3 = graph.update(flt, 1.0).unwrap();
    let s3 = out3.point_data().unwrap().get("S").unwrap().get_scalar(0);
    assert!((s3 - 18.0).abs() < 1e-6);
}

#[test]
fn image_trilinear_sampling_and_colormap() {
    let mut img = ImageData::new([3, 3, 3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let mut vals = Vec::new();
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                vals.push((i + j + k) as f32);
            }
        }
    }
    img.point_data.set_scalars(DataArray::scalars_f32("Linear", vals));
    let sample = img
        .sample_point_array("Linear", glam::Vec3::new(0.5, 0.5, 0.5))
        .unwrap();
    assert!((sample[0] - 1.5).abs() < 1e-6);
    assert!(img
        .sample_point_array("Linear", glam::Vec3::new(-0.1, 0.0, 0.0))
        .is_none());

    for preset in ColorMapPreset::ALL {
        let cm = ColorMap::from_preset(*preset);
        let lut = cm.to_rgba8_lut(256);
        assert_eq!(lut.len(), 256 * 4);
    }

    let tf = TransferFunction::with_preset(ColorMapPreset::Viridis, (0.0, 10.0));
    let tf_lut = tf.to_rgba8_lut(128);
    assert_eq!(tf_lut.len(), 128 * 4);
}

#[test]
fn unstructured_and_multiblock_stats() {
    let mut ug = UnstructuredGrid::new();
    ug.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    ug.push_cell(CellType::Tetra, &[0, 1, 2, 3]);
    assert_eq!(ug.num_points(), 4);
    assert_eq!(ug.num_cells(), 1);
    assert_eq!(CellType::Hexahedron.tetrahedra().len(), 5);

    let mut mb = MultiBlockDataSet::new();
    mb.add_block("tet", DataSet::UnstructuredGrid(ug));
    assert_eq!(mb.num_points(), 4);
    assert_eq!(mb.num_cells(), 1);
}
