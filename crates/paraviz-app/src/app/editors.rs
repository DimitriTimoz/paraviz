use super::*;

pub(super) fn drag_vec3(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;
    ui.label(label);
    ui.horizontal(|ui| {
        changed |= ui
            .add(egui::DragValue::new(&mut v[0]).speed(speed).prefix("X:"))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut v[1]).speed(speed).prefix("Y:"))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut v[2]).speed(speed).prefix("Z:"))
            .changed();
    });
    changed
}

pub(super) fn drag_vec3_f64(ui: &mut egui::Ui, label: &str, v: &mut [f64; 3], speed: f64) -> bool {
    let mut changed = false;
    ui.label(label);
    ui.horizontal(|ui| {
        changed |= ui
            .add(egui::DragValue::new(&mut v[0]).speed(speed).prefix("X:"))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut v[1]).speed(speed).prefix("Y:"))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut v[2]).speed(speed).prefix("Z:"))
            .changed();
    });
    changed
}

pub(super) fn drag_dims3(ui: &mut egui::Ui, label: &str, dims: &mut [usize; 3], min_val: usize) -> bool {
    let mut changed = false;
    ui.label(label);
    ui.horizontal(|ui| {
        for (i, prefix) in [(0, "X:"), (1, "Y:"), (2, "Z:")] {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut dims[i])
                        .speed(1)
                        .range(min_val..=256)
                        .prefix(prefix),
                )
                .changed();
        }
    });
    changed
}

pub(super) fn array_picker_combo(
    ui: &mut egui::Ui,
    id_salt: &str,
    label: &str,
    value: &mut String,
    parent_ds: Option<&DataSet>,
    association: AttributeAssociation,
    vectors_only: bool,
) -> bool {
    let mut changed = false;
    let names: Vec<(String, (f64, f64))> = parent_ds
        .and_then(|ds| match association {
            AttributeAssociation::Point => ds.point_data(),
            AttributeAssociation::Cell => ds.cell_data(),
            AttributeAssociation::Field => ds.field_data(),
        })
        .map(|fd| {
            fd.arrays()
                .iter()
                .filter(|a| !vectors_only || a.num_components >= 3)
                .map(|a| (a.name.clone(), a.range(None)))
                .collect()
        })
        .unwrap_or_default();

    ui.horizontal(|ui| {
        ui.label(label);
        if !names.is_empty() {
            egui::ComboBox::from_id_salt(id_salt)
                .selected_text(if value.is_empty() { "<select>" } else { value.as_str() })
                .show_ui(ui, |ui| {
                    for (name, _) in &names {
                        if ui.selectable_label(value == name, name).clicked() {
                            *value = name.clone();
                            changed = true;
                        }
                    }
                });
        } else {
            changed |= ui.text_edit_singleline(value).changed();
        }
    });
    changed
}

pub(super) fn edit_source_config(ui: &mut egui::Ui, cfg: &mut SourceConfig) -> bool {
    let mut changed = false;
    match cfg {
        SourceConfig::Sphere(s) => {
            changed |= drag_vec3(ui, "Center", &mut s.center, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut s.radius)
                        .speed(0.02)
                        .range(0.01..=1000.0)
                        .prefix("Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut s.theta_resolution, 3..=128).text("Theta Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut s.phi_resolution, 3..=128).text("Phi Res"))
                .changed();
        }
        SourceConfig::Cube(c) => {
            changed |= drag_vec3(ui, "Center", &mut c.center, 0.05);
            changed |= drag_vec3(ui, "Lengths", &mut c.lengths, 0.05);
        }
        SourceConfig::Cone(c) => {
            changed |= drag_vec3(ui, "Center", &mut c.center, 0.05);
            changed |= drag_vec3(ui, "Direction", &mut c.direction, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut c.height)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .prefix("Height: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut c.radius)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .prefix("Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut c.resolution, 3..=128).text("Resolution"))
                .changed();
            changed |= ui.checkbox(&mut c.capping, "Capping").changed();
        }
        SourceConfig::Cylinder(c) => {
            changed |= drag_vec3(ui, "Center", &mut c.center, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut c.height)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .prefix("Height: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut c.radius)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .prefix("Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut c.resolution, 3..=128).text("Resolution"))
                .changed();
            changed |= ui.checkbox(&mut c.capping, "Capping").changed();
        }
        SourceConfig::Plane(p) => {
            changed |= drag_vec3(ui, "Origin", &mut p.origin, 0.05);
            changed |= drag_vec3(ui, "Point 1", &mut p.point1, 0.05);
            changed |= drag_vec3(ui, "Point 2", &mut p.point2, 0.05);
            changed |= ui
                .add(egui::Slider::new(&mut p.x_resolution, 1..=128).text("X Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut p.y_resolution, 1..=128).text("Y Res"))
                .changed();
        }
        SourceConfig::Line(l) => {
            changed |= drag_vec3(ui, "Point 1", &mut l.point1, 0.05);
            changed |= drag_vec3(ui, "Point 2", &mut l.point2, 0.05);
            changed |= ui
                .add(egui::Slider::new(&mut l.resolution, 1..=512).text("Resolution"))
                .changed();
        }
        SourceConfig::Arrow(a) => {
            changed |= ui
                .add(egui::Slider::new(&mut a.tip_length, 0.05..=0.9).text("Tip Length"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut a.tip_radius, 0.01..=0.5).text("Tip Radius"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut a.tip_resolution, 4..=64).text("Tip Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut a.shaft_radius, 0.005..=0.3).text("Shaft Radius"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut a.shaft_resolution, 4..=64).text("Shaft Res"))
                .changed();
        }
        SourceConfig::Disk(d) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut d.inner_radius)
                        .speed(0.02)
                        .range(0.0..=500.0)
                        .prefix("Inner Radius: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut d.outer_radius)
                        .speed(0.02)
                        .range(0.01..=500.0)
                        .prefix("Outer Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut d.radial_resolution, 1..=64).text("Radial Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut d.circum_resolution, 3..=128).text("Circum Res"))
                .changed();
        }
        SourceConfig::Torus(t) => {
            changed |= drag_vec3(ui, "Center", &mut t.center, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut t.major_radius)
                        .speed(0.02)
                        .range(0.05..=500.0)
                        .prefix("Major Radius: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut t.minor_radius)
                        .speed(0.02)
                        .range(0.01..=200.0)
                        .prefix("Minor Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut t.major_resolution, 6..=128).text("Major Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut t.minor_resolution, 6..=128).text("Minor Res"))
                .changed();
        }
        SourceConfig::Wavelet(w) => {
            ui.label("Whole Extent:");
            ui.horizontal(|ui| {
                for i in 0..3 {
                    changed |= ui
                        .add(egui::DragValue::new(&mut w.whole_extent[i * 2]).speed(1))
                        .changed();
                    changed |= ui
                        .add(egui::DragValue::new(&mut w.whole_extent[i * 2 + 1]).speed(1))
                        .changed();
                }
            });
            changed |= drag_vec3_f64(ui, "Center", &mut w.center, 0.1);
            changed |= ui
                .add(egui::DragValue::new(&mut w.maximum).speed(1.0).prefix("Maximum: "))
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut w.std_dev)
                        .speed(0.02)
                        .range(0.05..=10.0)
                        .prefix("StdDev: "),
                )
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.x_freq).speed(1.0).prefix("X Freq: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.y_freq).speed(1.0).prefix("Y Freq: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.z_freq).speed(1.0).prefix("Z Freq: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.x_mag).speed(0.5).prefix("X Mag: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.y_mag).speed(0.5).prefix("Y Mag: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut w.z_mag).speed(0.5).prefix("Z Mag: "))
                .changed();
        }
        SourceConfig::VectorField(vf) => {
            egui::ComboBox::from_label("Field Kind")
                .selected_text(format!("{:?}", vf.kind))
                .show_ui(ui, |ui| {
                    for k in [
                        VectorFieldKind::SwirlingVortex,
                        VectorFieldKind::Dipole,
                        VectorFieldKind::HelicalTorus,
                    ] {
                        if ui
                            .selectable_value(&mut vf.kind, k, format!("{k:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            changed |= drag_dims3(ui, "Dimensions", &mut vf.dimensions, 4);
            changed |= drag_vec3_f64(ui, "Origin", &mut vf.origin, 0.05);
            changed |= drag_vec3_f64(ui, "Spacing", &mut vf.spacing, 0.01);
        }
        SourceConfig::Superquadric(sq) => {
            changed |= ui.checkbox(&mut sq.toroidal, "Toroidal").changed();
            changed |= ui
                .add(egui::Slider::new(&mut sq.thickness, 0.05..=0.99).text("Thickness"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut sq.phi_roundness, 0.05..=3.0).text("Phi Roundness"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut sq.theta_roundness, 0.05..=3.0).text("Theta Roundness"))
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut sq.size)
                        .speed(0.02)
                        .range(0.05..=100.0)
                        .prefix("Size: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut sq.phi_resolution, 4..=128).text("Phi Res"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut sq.theta_resolution, 4..=128).text("Theta Res"))
                .changed();
        }
        SourceConfig::Gyroid(g) => {
            changed |= drag_dims3(ui, "Dimensions", &mut g.dimensions, 4);
            changed |= drag_vec3_f64(ui, "Bounds Min", &mut g.bounds_min, 0.1);
            changed |= drag_vec3_f64(ui, "Bounds Max", &mut g.bounds_max, 0.1);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut g.frequency)
                        .speed(0.05)
                        .range(0.1..=20.0)
                        .prefix("Frequency: "),
                )
                .changed();
        }
        SourceConfig::PointCloud(pc) => {
            changed |= drag_vec3(ui, "Center", &mut pc.center, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut pc.radius)
                        .speed(0.02)
                        .range(0.01..=1000.0)
                        .prefix("Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut pc.num_points, 10..=50_000).text("Points"))
                .changed();
            changed |= ui.checkbox(&mut pc.volumetric, "Volumetric").changed();
        }
        SourceConfig::File(p) => {
            ui.label(format!("Path: {}", p.display()));
            let detected = paraviz::io::detect_reader(&*p)
                .map(|r| r.short_name())
                .unwrap_or("Unknown");
            ui.label(format!("Active Reader: {detected} (Auto)"));
            let mut selected = paraviz::io::DatasetReader::Auto;
            ui.horizontal(|ui| {
                ui.label("Force Reader:");
                egui::ComboBox::from_id_salt("node_force_reader_combo")
                    .selected_text(selected.label())
                    .show_ui(ui, |ui| {
                        for &r in paraviz::io::DatasetReader::ALL {
                            ui.selectable_value(&mut selected, r, r.label());
                        }
                    });
            });
            if selected != paraviz::io::DatasetReader::Auto {
                *cfg = SourceConfig::FileWithReader {
                    path: p.clone(),
                    reader: selected,
                };
                changed = true;
            }
        }
        SourceConfig::FileWithReader { path: p, reader } => {
            ui.label(format!("Path: {}", p.display()));
            let active_label = if *reader == paraviz::io::DatasetReader::Auto {
                paraviz::io::detect_reader(p)
                    .map(|r| format!("{} (Auto)", r.short_name()))
                    .unwrap_or_else(|_| "Auto".into())
            } else {
                format!("{} (Forced)", reader.short_name())
            };
            ui.label(format!("Active Reader: {active_label}"));
            let prev = *reader;
            ui.horizontal(|ui| {
                ui.label("Reader:");
                egui::ComboBox::from_id_salt("node_reader_combo")
                    .selected_text(reader.label())
                    .show_ui(ui, |ui| {
                        for &r in paraviz::io::DatasetReader::ALL {
                            ui.selectable_value(reader, r, r.label());
                        }
                    });
            });
            if *reader != prev {
                changed = true;
            }
        }
        SourceConfig::PvdSeries(p) => {
            ui.label(format!("Path: {}", p.display()));
            ui.label("Active Reader: PVD Time-Series Reader");
        }
    }
    changed
}

pub(super) fn edit_plane_snap_buttons(
    ui: &mut egui::Ui,
    origin: &mut [f32; 3],
    normal: &mut [f32; 3],
    parent_ds: Option<&DataSet>,
) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        if ui.small_button("X Normal").clicked() {
            *normal = [1.0, 0.0, 0.0];
            changed = true;
        }
        if ui.small_button("Y Normal").clicked() {
            *normal = [0.0, 1.0, 0.0];
            changed = true;
        }
        if ui.small_button("Z Normal").clicked() {
            *normal = [0.0, 0.0, 1.0];
            changed = true;
        }
        if ui.small_button("Flip").clicked() {
            *normal = [-normal[0], -normal[1], -normal[2]];
            changed = true;
        }
        if let Some(ds) = parent_ds {
            let bb = ds.bounds();
            if bb.is_valid() && ui.small_button("Center").clicked() {
                *origin = bb.center().to_array();
                changed = true;
            }
        }
    });
    changed
}

pub(super) fn edit_filter_config(
    ui: &mut egui::Ui,
    cfg: &mut FilterConfig,
    parent_ds: Option<&DataSet>,
) -> bool {
    let mut changed = false;
    match cfg {
        FilterConfig::ExtractSurface(f) => {
            changed |= ui
                .checkbox(&mut f.compute_normals, "Compute Normals")
                .changed();
        }
        FilterConfig::Contour(f) => {
            changed |= array_picker_combo(
                ui,
                "contour_field",
                "Scalar Field:",
                &mut f.field_name,
                parent_ds,
                AttributeAssociation::Point,
                false,
            );
            changed |= ui
                .checkbox(&mut f.compute_normals, "Compute Normals")
                .changed();
            changed |= ui
                .checkbox(&mut f.merge_vertices, "Merge Vertices")
                .changed();
            ui.label("Iso Values:");
            let mut remove_idx: Option<usize> = None;
            for (idx, val) in f.iso_values.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(egui::DragValue::new(val).speed(0.5))
                        .changed();
                    if ui.small_button("×").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(idx) = remove_idx
                && f.iso_values.len() > 1
            {
                f.iso_values.remove(idx);
                changed = true;
            }
            ui.horizontal(|ui| {
                if ui.button("+ Add Iso Value").clicked() {
                    let next = f.iso_values.last().copied().unwrap_or(0.0) + 10.0;
                    f.iso_values.push(next);
                    changed = true;
                }
                if let Some((mn, mx)) = parent_ds
                    .and_then(|d| d.point_data())
                    .and_then(|p| p.get(&f.field_name))
                    .map(|a| a.range(None))
                    && ui.small_button("3-Level").clicked()
                {
                    f.iso_values = vec![
                        mn + (mx - mn) * 0.25,
                        mn + (mx - mn) * 0.50,
                        mn + (mx - mn) * 0.75,
                    ];
                    changed = true;
                }
            });
        }
        FilterConfig::Slice(f) => {
            changed |= drag_vec3(ui, "Origin", &mut f.origin, 0.05);
            changed |= drag_vec3(ui, "Normal", &mut f.normal, 0.05);
            changed |= edit_plane_snap_buttons(ui, &mut f.origin, &mut f.normal, parent_ds);
            changed |= ui.checkbox(&mut f.crinkle, "Crinkle Cut").changed();
        }
        FilterConfig::Clip(f) => {
            let kind_label = match &f.function {
                ImplicitFunction::Plane { .. } => "Plane",
                ImplicitFunction::Sphere { .. } => "Sphere",
                ImplicitFunction::Box { .. } => "Box",
            };
            egui::ComboBox::from_label("Clip Type")
                .selected_text(kind_label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(f.function, ImplicitFunction::Plane { .. }),
                            "Plane",
                        )
                        .clicked()
                    {
                        f.function = ImplicitFunction::Plane {
                            origin: [0.0, 0.0, 0.0],
                            normal: [1.0, 0.0, 0.0],
                        };
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(f.function, ImplicitFunction::Sphere { .. }),
                            "Sphere",
                        )
                        .clicked()
                    {
                        f.function = ImplicitFunction::Sphere {
                            center: [0.0, 0.0, 0.0],
                            radius: 1.0,
                        };
                        changed = true;
                    }
                    if ui
                        .selectable_label(matches!(f.function, ImplicitFunction::Box { .. }), "Box")
                        .clicked()
                    {
                        f.function = ImplicitFunction::Box {
                            min: [-1.0, -1.0, -1.0],
                            max: [1.0, 1.0, 1.0],
                        };
                        changed = true;
                    }
                });
            match &mut f.function {
                ImplicitFunction::Plane { origin, normal } => {
                    changed |= drag_vec3(ui, "Origin", origin, 0.05);
                    changed |= drag_vec3(ui, "Normal", normal, 0.05);
                    changed |= edit_plane_snap_buttons(ui, origin, normal, parent_ds);
                }
                ImplicitFunction::Sphere { center, radius } => {
                    changed |= drag_vec3(ui, "Center", center, 0.05);
                    changed |= ui
                        .add(
                            egui::DragValue::new(radius)
                                .speed(0.05)
                                .range(0.01..=1000.0)
                                .prefix("Radius: "),
                        )
                        .changed();
                }
                ImplicitFunction::Box { min, max } => {
                    changed |= drag_vec3(ui, "Min", min, 0.05);
                    changed |= drag_vec3(ui, "Max", max, 0.05);
                    if let Some(ds) = parent_ds {
                        let bb = ds.bounds();
                        if bb.is_valid() && ui.small_button("Fit Parent Bounds").clicked() {
                            *min = bb.min.to_array();
                            *max = bb.max.to_array();
                            changed = true;
                        }
                    }
                }
            }
            changed |= ui.checkbox(&mut f.invert, "Invert").changed();
            changed |= ui.checkbox(&mut f.crinkle, "Crinkle").changed();
        }
        FilterConfig::Threshold(f) => {
            changed |= array_picker_combo(
                ui,
                "threshold_field",
                "Field:",
                &mut f.field_name,
                parent_ds,
                f.association,
                false,
            );
            egui::ComboBox::from_label("Association")
                .selected_text(format!("{:?}", f.association))
                .show_ui(ui, |ui| {
                    for assoc in [AttributeAssociation::Point, AttributeAssociation::Cell] {
                        if ui
                            .selectable_value(&mut f.association, assoc, format!("{assoc:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            changed |= ui
                .add(egui::DragValue::new(&mut f.min).speed(0.5).prefix("Min: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut f.max).speed(0.5).prefix("Max: "))
                .changed();
            if let Some((mn, mx)) = parent_ds
                .and_then(|d| match f.association {
                    AttributeAssociation::Point => d.point_data(),
                    AttributeAssociation::Cell => d.cell_data(),
                    AttributeAssociation::Field => d.field_data(),
                })
                .and_then(|fd| fd.get(&f.field_name))
                .map(|a| a.range(None))
                && ui.small_button("Fit Data Range").clicked()
            {
                f.min = mn;
                f.max = mx;
                changed = true;
            }
            changed |= ui.checkbox(&mut f.invert, "Invert").changed();
            changed |= ui.checkbox(&mut f.all_scalars, "All Scalars").changed();
        }
        FilterConfig::StreamTracer(f) => {
            changed |= array_picker_combo(
                ui,
                "stream_vec_field",
                "Vector Field:",
                &mut f.vector_field,
                parent_ds,
                AttributeAssociation::Point,
                true,
            );
            egui::ComboBox::from_label("Integrator")
                .selected_text(format!("{:?}", f.integrator))
                .show_ui(ui, |ui| {
                    for intg in [Integrator::Rk4, Integrator::Rk45] {
                        if ui
                            .selectable_value(&mut f.integrator, intg, format!("{intg:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            egui::ComboBox::from_label("Direction")
                .selected_text(format!("{:?}", f.direction))
                .show_ui(ui, |ui| {
                    for dir in [
                        IntegrationDirection::Forward,
                        IntegrationDirection::Backward,
                        IntegrationDirection::Both,
                    ] {
                        if ui
                            .selectable_value(&mut f.direction, dir, format!("{dir:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            let seed_kind = match &f.seeds {
                SeedSource::Line { .. } => "Line",
                SeedSource::Sphere { .. } => "Sphere",
                SeedSource::Points(_) => "Points",
            };
            egui::ComboBox::from_label("Seed Type")
                .selected_text(seed_kind)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(matches!(f.seeds, SeedSource::Line { .. }), "Line")
                        .clicked()
                    {
                        f.seeds = SeedSource::default();
                        changed = true;
                    }
                    if ui
                        .selectable_label(matches!(f.seeds, SeedSource::Sphere { .. }), "Sphere")
                        .clicked()
                    {
                        f.seeds = SeedSource::Sphere {
                            center: [0.0, 0.0, 0.0],
                            radius: 1.0,
                            count: 16,
                        };
                        changed = true;
                    }
                });
            match &mut f.seeds {
                SeedSource::Line { p1, p2, count } => {
                    changed |= drag_vec3(ui, "Seed P1", p1, 0.05);
                    changed |= drag_vec3(ui, "Seed P2", p2, 0.05);
                    changed |= ui
                        .add(egui::Slider::new(count, 1..=200).text("Seed Count"))
                        .changed();
                }
                SeedSource::Sphere {
                    center,
                    radius,
                    count,
                } => {
                    changed |= drag_vec3(ui, "Seed Center", center, 0.05);
                    changed |= ui
                        .add(
                            egui::DragValue::new(radius)
                                .speed(0.05)
                                .range(0.01..=500.0)
                                .prefix("Seed Radius: "),
                        )
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(count, 1..=200).text("Seed Count"))
                        .changed();
                }
                SeedSource::Points(pts) => {
                    ui.label(format!("Explicit seeds: {}", pts.len()));
                }
            }
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.step_size)
                        .speed(0.005)
                        .range(0.001..=10.0)
                        .prefix("Step Size: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut f.max_steps, 10..=5000).text("Max Steps"))
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.max_length)
                        .speed(0.2)
                        .range(0.1..=1000.0)
                        .prefix("Max Length: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.terminal_speed)
                        .speed(1e-5)
                        .range(0.0..=1.0)
                        .prefix("Terminal Speed: "),
                )
                .changed();
        }
        FilterConfig::Tube(f) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.radius)
                        .speed(0.005)
                        .range(0.001..=10.0)
                        .prefix("Radius: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut f.sides, 3..=48).text("Sides"))
                .changed();
            changed |= ui.checkbox(&mut f.capping, "Capping").changed();
            let mut vary = f.vary_radius_by_scalar.clone().unwrap_or_default();
            if array_picker_combo(
                ui,
                "tube_vary_scalar",
                "Vary By Scalar:",
                &mut vary,
                parent_ds,
                AttributeAssociation::Point,
                false,
            ) {
                f.vary_radius_by_scalar = if vary.trim().is_empty() {
                    None
                } else {
                    Some(vary)
                };
                changed = true;
            }
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.radius_factor)
                        .speed(0.1)
                        .range(0.1..=20.0)
                        .prefix("Radius Factor: "),
                )
                .changed();
        }
        FilterConfig::Glyph(f) => {
            egui::ComboBox::from_label("Shape")
                .selected_text(format!("{:?}", f.shape))
                .show_ui(ui, |ui| {
                    for s in [
                        GlyphShape::Arrow,
                        GlyphShape::Cone,
                        GlyphShape::Sphere,
                        GlyphShape::Cube,
                    ] {
                        if ui
                            .selectable_value(&mut f.shape, s, format!("{s:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            egui::ComboBox::from_label("Scale Mode")
                .selected_text(format!("{:?}", f.scale_mode))
                .show_ui(ui, |ui| {
                    for m in [
                        ScaleMode::VectorMagnitude,
                        ScaleMode::Scalar,
                        ScaleMode::Constant,
                    ] {
                        if ui
                            .selectable_value(&mut f.scale_mode, m, format!("{m:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            let mut vf = f.vector_field.clone().unwrap_or_default();
            if array_picker_combo(
                ui,
                "glyph_vec",
                "Vector Field:",
                &mut vf,
                parent_ds,
                AttributeAssociation::Point,
                true,
            ) {
                f.vector_field = if vf.trim().is_empty() { None } else { Some(vf) };
                changed = true;
            }
            let mut sf = f.scalar_field.clone().unwrap_or_default();
            if array_picker_combo(
                ui,
                "glyph_scalar",
                "Scalar Field:",
                &mut sf,
                parent_ds,
                AttributeAssociation::Point,
                false,
            ) {
                f.scalar_field = if sf.trim().is_empty() { None } else { Some(sf) };
                changed = true;
            }
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.scale_factor)
                        .speed(0.02)
                        .range(0.001..=100.0)
                        .prefix("Scale Factor: "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut f.max_points, 10..=10_000).text("Max Points"))
                .changed();
        }
        FilterConfig::WarpByVector(f) => {
            changed |= array_picker_combo(
                ui,
                "warp_vec",
                "Vector Field:",
                &mut f.vector_field,
                parent_ds,
                AttributeAssociation::Point,
                true,
            );
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.scale_factor)
                        .speed(0.05)
                        .prefix("Scale Factor: "),
                )
                .changed();
        }
        FilterConfig::WarpByScalar(f) => {
            changed |= array_picker_combo(
                ui,
                "warp_scalar",
                "Scalar Field:",
                &mut f.scalar_field,
                parent_ds,
                AttributeAssociation::Point,
                false,
            );
            changed |= ui.checkbox(&mut f.use_normal, "Use Normal").changed();
            changed |= drag_vec3(ui, "Normal", &mut f.normal, 0.05);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.scale_factor)
                        .speed(0.05)
                        .prefix("Scale Factor: "),
                )
                .changed();
        }
        FilterConfig::Gradient(f) => {
            changed |= array_picker_combo(
                ui,
                "grad_field",
                "Field:",
                &mut f.field_name,
                parent_ds,
                AttributeAssociation::Point,
                false,
            );
            changed |= ui
                .checkbox(&mut f.compute_gradient, "Compute Gradient")
                .changed();
            changed |= ui
                .checkbox(&mut f.compute_divergence, "Compute Divergence")
                .changed();
            changed |= ui
                .checkbox(&mut f.compute_vorticity, "Compute Vorticity")
                .changed();
            changed |= ui
                .checkbox(&mut f.compute_q_criterion, "Compute Q-Criterion")
                .changed();
            changed |= ui
                .checkbox(&mut f.compute_lambda2, "Compute Lambda2-Criterion")
                .changed();
        }
        FilterConfig::Calculator(f) => {
            ui.label("Expression:");
            changed |= ui.text_edit_singleline(&mut f.expression).changed();
            if let Some(ds) = parent_ds
                && let Some(fd) = match f.association {
                    AttributeAssociation::Point => ds.point_data(),
                    AttributeAssociation::Cell => ds.cell_data(),
                    AttributeAssociation::Field => ds.field_data(),
                }
                && !fd.is_empty()
            {
                ui.horizontal_wrapped(|ui| {
                    ui.weak("Insert:");
                    for arr in fd.arrays() {
                        if ui.small_button(&arr.name).clicked() {
                            if !f.expression.is_empty() && !f.expression.ends_with(' ') {
                                f.expression.push(' ');
                            }
                            f.expression.push_str(&arr.name);
                            changed = true;
                        }
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Result Name:");
                changed |= ui.text_edit_singleline(&mut f.result_name).changed();
            });
            egui::ComboBox::from_label("Association")
                .selected_text(format!("{:?}", f.association))
                .show_ui(ui, |ui| {
                    for assoc in [AttributeAssociation::Point, AttributeAssociation::Cell] {
                        if ui
                            .selectable_value(&mut f.association, assoc, format!("{assoc:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        }
        FilterConfig::Smooth(f) => {
            egui::ComboBox::from_label("Method")
                .selected_text(format!("{:?}", f.method))
                .show_ui(ui, |ui| {
                    for m in [SmoothingMethod::Taubin, SmoothingMethod::Laplacian] {
                        if ui
                            .selectable_value(&mut f.method, m, format!("{m:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            changed |= ui
                .add(egui::Slider::new(&mut f.iterations, 1..=200).text("Iterations"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut f.relaxation_factor, 0.01..=1.0).text("Relaxation"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut f.pass_band, 0.01..=2.0).text("Pass Band"))
                .changed();
            changed |= ui
                .checkbox(&mut f.preserve_boundary, "Preserve Boundary")
                .changed();
        }
        FilterConfig::Transform(f) => {
            changed |= drag_vec3(ui, "Translation", &mut f.translation, 0.05);
            changed |= drag_vec3(ui, "Rotation (deg)", &mut f.rotation_deg, 1.0);
            changed |= drag_vec3(ui, "Scale", &mut f.scale, 0.05);
        }
        FilterConfig::CellDataToPointData(f) => {
            changed |= ui
                .checkbox(&mut f.pass_cell_data, "Pass Cell Data")
                .changed();
        }
        FilterConfig::PointDataToCellData(f) => {
            changed |= ui
                .checkbox(&mut f.pass_point_data, "Pass Point Data")
                .changed();
        }
        FilterConfig::ProbeOverLine(f) => {
            changed |= drag_vec3(ui, "Point 1", &mut f.point1, 0.05);
            changed |= drag_vec3(ui, "Point 2", &mut f.point2, 0.05);
            if let Some(ds) = parent_ds {
                let bb = ds.bounds();
                if bb.is_valid() {
                    let c = bb.center().to_array();
                    ui.horizontal(|ui| {
                        if ui.small_button("Along X").clicked() {
                            f.point1 = [bb.min.x, c[1], c[2]];
                            f.point2 = [bb.max.x, c[1], c[2]];
                            changed = true;
                        }
                        if ui.small_button("Along Y").clicked() {
                            f.point1 = [c[0], bb.min.y, c[2]];
                            f.point2 = [c[0], bb.max.y, c[2]];
                            changed = true;
                        }
                        if ui.small_button("Along Z").clicked() {
                            f.point1 = [c[0], c[1], bb.min.z];
                            f.point2 = [c[0], c[1], bb.max.z];
                            changed = true;
                        }
                        if ui.small_button("Diag").clicked() {
                            f.point1 = bb.min.to_array();
                            f.point2 = bb.max.to_array();
                            changed = true;
                        }
                    });
                }
            }
            changed |= ui
                .add(egui::Slider::new(&mut f.resolution, 2..=1000).text("Resolution"))
                .changed();
        }
        FilterConfig::FeatureEdges(f) => {
            changed |= ui
                .checkbox(&mut f.boundary_edges, "Boundary Edges")
                .changed();
            changed |= ui
                .checkbox(&mut f.feature_edges, "Feature Edges")
                .changed();
            changed |= ui
                .checkbox(&mut f.non_manifold_edges, "Non-Manifold Edges")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut f.feature_angle_deg, 0.0..=180.0)
                        .text("Feature Angle (deg)"),
                )
                .changed();
        }
        FilterConfig::ExtractEdges(_) => {
            ui.weak("Extracts all unique topological cell edges as wireframe lines.");
        }
        FilterConfig::Connectivity(f) => {
            changed |= ui
                .checkbox(&mut f.extract_largest, "Extract Largest Region")
                .changed();
        }
        FilterConfig::Shrink(f) => {
            changed |= ui
                .add(egui::Slider::new(&mut f.shrink_factor, 0.0..=1.0).text("Shrink Factor"))
                .changed();
        }
        FilterConfig::CellCenters(f) => {
            changed |= ui
                .checkbox(&mut f.vertex_cells, "Generate Vertex Cells")
                .changed();
        }
        FilterConfig::Reflect(f) => {
            egui::ComboBox::from_label("Reflect Plane")
                .selected_text(format!("{:?}", f.plane))
                .show_ui(ui, |ui| {
                    for plane in [
                        ReflectPlane::XMin,
                        ReflectPlane::XMax,
                        ReflectPlane::YMin,
                        ReflectPlane::YMax,
                        ReflectPlane::ZMin,
                        ReflectPlane::ZMax,
                        ReflectPlane::CenterX,
                        ReflectPlane::CenterY,
                        ReflectPlane::CenterZ,
                    ] {
                        if ui
                            .selectable_value(&mut f.plane, plane, format!("{plane:?}"))
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            changed |= ui.checkbox(&mut f.copy_input, "Copy Input").changed();
        }
        FilterConfig::GenerateNormals(f) => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut f.feature_angle_deg, 0.0..=180.0)
                        .text("Feature Angle (deg)"),
                )
                .changed();
            changed |= ui
                .checkbox(&mut f.splitting, "Split Sharp Edges")
                .changed();
            changed |= ui.checkbox(&mut f.flip_normals, "Flip Normals").changed();
            changed |= ui
                .checkbox(&mut f.consistency, "Enforce Consistency")
                .changed();
        }
        FilterConfig::Triangulate(f) => {
            changed |= ui
                .checkbox(&mut f.pass_verts_and_lines, "Pass Verts & Lines")
                .changed();
        }
        FilterConfig::CleanPolyData(f) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.tolerance)
                        .speed(1e-5)
                        .range(0.0..=10.0)
                        .max_decimals(6)
                        .prefix("Merge Tolerance: "),
                )
                .changed();
        }
        FilterConfig::Elevation(f) => {
            changed |= drag_vec3(ui, "Low Point", &mut f.low_point, 0.05);
            changed |= drag_vec3(ui, "High Point", &mut f.high_point, 0.05);
            ui.horizontal(|ui| {
                ui.label("Scalar Name:");
                changed |= ui.text_edit_singleline(&mut f.scalar_name).changed();
            });
        }
        FilterConfig::Decimate(f) => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut f.target_reduction, 0.0..=0.95)
                        .text("Target Reduction"),
                )
                .changed();
            changed |= ui
                .checkbox(&mut f.preserve_boundaries, "Preserve Boundaries")
                .changed();
        }
        FilterConfig::Delaunay2D(f) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.alpha)
                        .speed(0.05)
                        .range(0.0..=1e6)
                        .prefix("Alpha (0 = off): "),
                )
                .changed();
        }
        FilterConfig::Delaunay3D(f) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.alpha)
                        .speed(0.05)
                        .range(0.0..=1e6)
                        .prefix("Alpha (0 = off): "),
                )
                .changed();
        }
        FilterConfig::IsoVolume(f) => {
            changed |= array_picker_combo(
                ui,
                "isovol_field",
                "Scalar Field:",
                &mut f.field_name,
                parent_ds,
                AttributeAssociation::Point,
                false,
            );
            changed |= ui
                .add(egui::DragValue::new(&mut f.min_value).speed(0.1).prefix("Min: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut f.max_value).speed(0.1).prefix("Max: "))
                .changed();
        }
        FilterConfig::ResampleToImage(f) => {
            ui.horizontal(|ui| {
                ui.label("Dimensions:");
                for d in &mut f.dimensions {
                    changed |= ui
                        .add(egui::DragValue::new(d).range(2..=256))
                        .changed();
                }
            });
        }
        FilterConfig::IntegrateVariables(f) => {
            changed |= ui
                .checkbox(
                    &mut f.divide_cell_data_by_volume,
                    "Divide Cell Data By Volume/Area",
                )
                .changed();
        }
        FilterConfig::ExtractBlock(f) => {
            ui.horizontal(|ui| {
                ui.label("Name Filter:");
                changed |= ui.text_edit_singleline(&mut f.name_filter).changed();
            });
            changed |= ui
                .checkbox(&mut f.unwrap_single, "Unwrap Single Block")
                .changed();
        }
        FilterConfig::TemporalStatistics(f) => {
            changed |= ui.checkbox(&mut f.compute_average, "Average").changed();
            changed |= ui.checkbox(&mut f.compute_minimum, "Minimum").changed();
            changed |= ui.checkbox(&mut f.compute_maximum, "Maximum").changed();
            changed |= ui.checkbox(&mut f.compute_stddev, "Std. Deviation").changed();
        }
        FilterConfig::AddTimeSteps(f) => {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.num_steps)
                        .range(1..=200)
                        .prefix("Steps: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.start_time)
                        .speed(0.1)
                        .prefix("Start: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.end_time)
                        .speed(0.1)
                        .prefix("End: "),
                )
                .changed();
        }
        FilterConfig::AppendDatasets(_) => {
            ui.weak("Appends all blocks of a MultiBlock input into a single mesh.");
        }
    }
    changed
}

