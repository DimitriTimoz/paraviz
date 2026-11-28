use paraviz::ParaViz;
use paraviz_core::{AttributeAssociation, DataArray, DataSet, Filter, PolyData};
use paraviz_filters::CalculatorFilter;

#[test]
fn polydata_out_of_bounds_indices_do_not_panic() {
    let mut poly = PolyData::new();
    poly.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // Index 99 is out of bounds
    poly.polys = vec![vec![0, 1, 99], vec![0, 1, 2, 99]];
    poly.strips = vec![vec![0, 1, 99]];

    poly.compute_normals();
    let tris = poly.triangulate();
    assert!(tris.is_empty());
}

#[test]
fn data_array_deserialize_enforces_invariants() {
    let bad_zero = r#"{"name":"x","num_components":0,"data":{"F32":[1.0]}}"#;
    assert!(serde_json::from_str::<DataArray>(bad_zero).is_err());

    let bad_div = r#"{"name":"x","num_components":3,"data":{"F32":[1.0,2.0]}}"#;
    assert!(serde_json::from_str::<DataArray>(bad_div).is_err());
}

#[test]
fn calculator_rejects_deep_recursion_without_stack_overflow() {
    let mut poly = PolyData::new();
    poly.points = vec![[1.0, 2.0, 3.0]];
    let ds = DataSet::PolyData(poly);

    let depth = 500;
    let expr = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
    let calc = CalculatorFilter::new("out", expr, AttributeAssociation::Point);
    let res = calc.apply(&ds);
    assert!(res.is_err(), "deeply nested expression must fail cleanly");
}

#[test]
fn session_load_rejects_excessive_pipeline_depth() {
    let mut nodes = Vec::new();
    nodes.push(serde_json::json!({
        "id": 1,
        "name": "src",
        "parent": null,
        "source": { "Sphere": { "center": [0.0, 0.0, 0.0], "radius": 1.0, "theta_resolution": 8, "phi_resolution": 8 } },
        "filter": null,
        "custom_dataset": null,
        "style": {}
    }));
    for i in 2..=300u64 {
        nodes.push(serde_json::json!({
            "id": i,
            "name": format!("f{i}"),
            "parent": i - 1,
            "source": null,
            "filter": { "Transform": { "translate": [0.0, 0.0, 0.0], "rotate_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0] } },
            "custom_dataset": null,
            "style": {}
        }));
    }
    let json = serde_json::json!({
        "nodes": nodes,
        "camera": { "position": [0.0, 0.0, 5.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "fov_y_deg": 45.0, "near": 0.1, "far": 100.0, "parallel_projection": false, "parallel_scale": 1.0 },
        "settings": {},
        "time": 0.0
    });
    let res = ParaViz::load_state_json(json.to_string());
    assert!(res.is_err(), "excessively deep session pipeline must fail cleanly");
}
