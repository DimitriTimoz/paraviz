use super::drawer::polydata_surface_area;
use super::editors::*;
use super::*;

impl ParaVizApp {
    pub(super) fn draw_properties_tab(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        let Some(sel_id) = self.selected_node else {
            section_card(ui, self.theme, "Properties", |ui| {
                ui.weak("Select a node in the Pipeline Browser to configure its parameters.");
            });
            return;
        };

        let (parent_id, src_cfg, flt_cfg) = match self.session.node(sel_id) {
            Ok(n) => (n.parent, n.source_config.clone(), n.filter_config.clone()),
            Err(_) => return,
        };
        let parent_ds = parent_id
            .and_then(|pid| self.session.node(pid).ok())
            .and_then(|pn| pn.dataset().cloned());

        let mut apply_clicked = false;
        let mut discard_clicked = false;
        section_card(ui, self.theme, "", |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.auto_apply, "Auto-Apply");
                if ui.button("Apply").clicked() {
                    apply_clicked = true;
                }
                if (self.pending_source.is_some() || self.pending_filter.is_some())
                    && ui.small_button("Discard").clicked()
                {
                    discard_clicked = true;
                }
            });
        });
        if discard_clicked {
            self.pending_source = None;
            self.pending_filter = None;
            self.invalidate_viewport();
        }

        if let Some(orig_src) = src_cfg {
            let mut working = match &self.pending_source {
                Some((id, cfg)) if *id == sel_id => cfg.clone(),
                _ => orig_src,
            };
            let mut changed = false;
            section_card(ui, self.theme, "Source Parameters", |ui| {
                changed = edit_source_config(ui, &mut working);
            });
            if changed {
                if self.auto_apply {
                    if let Err(e) = self.session.update_source_cfg(sel_id, working) {
                        self.error_message = Some(format!("Update source error: {e}"));
                    } else {
                        self.pending_source = None;
                        self.error_message = None;
                        self.clear_stats_caches_for_node(sel_id);
                        self.invalidate_viewport();
                    }
                } else {
                    self.pending_source = Some((sel_id, working));
                }
            } else if apply_clicked
                && let Some((id, pending)) = self.pending_source.take()
                && id == sel_id
            {
                if let Err(e) = self.session.update_source_cfg(sel_id, pending) {
                    self.error_message = Some(format!("Update source error: {e}"));
                } else {
                    self.error_message = None;
                    self.clear_stats_caches_for_node(sel_id);
                    self.invalidate_viewport();
                }
            }
        } else if let Some(orig_flt) = flt_cfg {
            let mut working = match &self.pending_filter {
                Some((id, cfg)) if *id == sel_id => cfg.clone(),
                _ => orig_flt,
            };
            let mut changed = false;
            section_card(ui, self.theme, "Filter Parameters", |ui| {
                changed = edit_filter_config(ui, &mut working, parent_ds.as_deref());
            });
            if changed {
                if self.auto_apply {
                    self.apply_filter_update(sel_id, working);
                } else {
                    self.pending_filter = Some((sel_id, working));
                    self.invalidate_viewport();
                }
            } else if apply_clicked
                && let Some((id, pending)) = self.pending_filter.take()
                && id == sel_id
            {
                self.apply_filter_update(id, pending);
            }
        } else {
            section_card(ui, self.theme, "In-Memory Dataset", |ui| {
                ui.label(
                    egui::RichText::new("Computed/merged node (no editable filter parameters).")
                        .color(pal.muted),
                );
            });
        }

        self.draw_multiblock_patches_section(ui, sel_id, "props");
    }

    pub(super) fn draw_display_tab(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            section_card(ui, self.theme, "Display", |ui| {
                ui.weak("Select a node to configure display styling.");
            });
            return;
        };

        let _ = self.session.evaluate(sel_id);

        // Pre-compute/fetch histogram for the interactive 2D volume transfer function curve editor.
        let volume_hist: Option<HistogramResult> =
            if let Ok(node_ro) = self.session.node(sel_id)
                && node_ro.style.representation == Representation::Volume
                && let Some(ds) = node_ro.dataset().cloned()
            {
                let field_key = match &node_ro.style.color_by {
                    ColorBy::PointScalar { array, component } => {
                        Some((array.clone(), AttributeAssociation::Point, *component))
                    }
                    ColorBy::CellScalar { array, component } => {
                        Some((array.clone(), AttributeAssociation::Cell, *component))
                    }
                    ColorBy::Solid(_) => ds
                        .point_data()
                        .and_then(|p| p.active_scalars_array())
                        .map(|a| (a.name.clone(), AttributeAssociation::Point, None)),
                };
                if let Some((array_name, assoc, comp)) = field_key {
                    let cache_key = (sel_id, array_name.clone(), assoc, comp);
                    if let Some(cached) = self.volume_tf_hist_cache.get(&cache_key) {
                        Some(cached.clone())
                    } else if let Ok(computed) =
                        compute_histogram(&ds, &array_name, assoc, comp, 48)
                    {
                        self.volume_tf_hist_cache
                            .insert(cache_key, computed.clone());
                        Some(computed)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let Ok(node) = self.session.node_mut(sel_id) else {
            return;
        };

        let ds_opt = node.cached_dataset.clone();
        let prev_style = node.style.clone();
        let style = &mut node.style;

        section_card(ui, self.theme, "Representation & Coloring", |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut style.visible, "Visible");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Reset Style").clicked() {
                        let vis = style.visible;
                        *style = ActorStyle {
                            visible: vis,
                            ..Default::default()
                        };
                    }
                });
            });

            egui::ComboBox::from_label("Representation")
                .selected_text(format!("{:?}", style.representation))
                .show_ui(ui, |ui| {
                    for r in [
                        Representation::Surface,
                        Representation::SurfaceWithEdges,
                        Representation::Wireframe,
                        Representation::Points,
                        Representation::PointGaussian,
                        Representation::Volume,
                        Representation::Outline,
                    ] {
                        ui.selectable_value(&mut style.representation, r, format!("{r:?}"));
                    }
                });

            let mut color_label = match &style.color_by {
                ColorBy::Solid(_) => "Solid Color".to_string(),
                ColorBy::PointScalar { array, .. } => format!("Point: {array}"),
                ColorBy::CellScalar { array, .. } => format!("Cell: {array}"),
            };
            let mut chosen_array_range: Option<(f64, f64)> = None;
            let mut chosen_components: usize = 1;

            if let Some(ds) = &ds_opt {
                match &style.color_by {
                    ColorBy::PointScalar { array, component } => {
                        if let Some(a) = ds.point_data().and_then(|p| p.get(array)) {
                            chosen_array_range = Some(a.range(*component));
                            chosen_components = a.num_components;
                        }
                    }
                    ColorBy::CellScalar { array, component } => {
                        if let Some(a) = ds.cell_data().and_then(|c| c.get(array)) {
                            chosen_array_range = Some(a.range(*component));
                            chosen_components = a.num_components;
                        }
                    }
                    ColorBy::Solid(_) => {}
                }
            }

            egui::ComboBox::from_label("Color By")
                .selected_text(&color_label)
                .show_ui(ui, |ui| {
                    let is_solid = matches!(style.color_by, ColorBy::Solid(_));
                    if ui.selectable_label(is_solid, "Solid Color").clicked() {
                        style.color_by = ColorBy::Solid([0.82, 0.84, 0.88, 1.0]);
                        color_label = "Solid Color".into();
                    }
                    if let Some(ds) = &ds_opt {
                        if let Some(pd) = ds.point_data() {
                            for arr in pd.arrays() {
                                let active = matches!(
                                    &style.color_by,
                                    ColorBy::PointScalar { array, .. } if array == &arr.name
                                );
                                if ui
                                    .selectable_label(active, format!("Point: {}", arr.name))
                                    .clicked()
                                {
                                    let rng = arr.range(None);
                                    style.color_by = ColorBy::PointScalar {
                                        array: arr.name.clone(),
                                        component: None,
                                    };
                                    if style.auto_range {
                                        style.scalar_range = rng;
                                        style.volume_tf.scalar_range = rng;
                                    }
                                }
                            }
                        }
                        if let Some(cd) = ds.cell_data() {
                            for arr in cd.arrays() {
                                let active = matches!(
                                    &style.color_by,
                                    ColorBy::CellScalar { array, .. } if array == &arr.name
                                );
                                if ui
                                    .selectable_label(active, format!("Cell: {}", arr.name))
                                    .clicked()
                                {
                                    let rng = arr.range(None);
                                    style.color_by = ColorBy::CellScalar {
                                        array: arr.name.clone(),
                                        component: None,
                                    };
                                    if style.auto_range {
                                        style.scalar_range = rng;
                                        style.volume_tf.scalar_range = rng;
                                    }
                                }
                            }
                        }
                    }
                });

            match &mut style.color_by {
                ColorBy::Solid(rgba) => {
                    ui.horizontal(|ui| {
                        ui.label("Solid RGBA:");
                        ui.color_edit_button_rgba_unmultiplied(rgba);
                    });
                }
                ColorBy::PointScalar { component, .. } | ColorBy::CellScalar { component, .. } => {
                    if chosen_components > 1 {
                        let comp_text = match *component {
                            None => "Magnitude".to_string(),
                            Some(c) => format!("Component {c}"),
                        };
                        egui::ComboBox::from_label("Component")
                            .selected_text(comp_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(component, None, "Magnitude");
                                for c in 0..chosen_components {
                                    let label = match c {
                                        0 => "X (0)".to_string(),
                                        1 => "Y (1)".to_string(),
                                        2 => "Z (2)".to_string(),
                                        _ => format!("Comp {c}"),
                                    };
                                    ui.selectable_value(component, Some(c), label);
                                }
                            });
                    }
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("colormap_preset_combo")
                    .selected_text(&style.colormap.name)
                    .show_ui(ui, |ui| {
                        for &preset in ColorMapPreset::ALL {
                            let selected = style.colormap.name == preset.name();
                            if ui.selectable_label(selected, preset.name()).clicked() {
                                let log = style.colormap.log_scale;
                                let mut cm = ColorMap::from_preset(preset);
                                cm.log_scale = log;
                                style.colormap = cm.clone();
                                style.volume_tf.color_map = cm;
                            }
                        }
                    });
                if ui
                    .small_button("Invert")
                    .on_hover_text("Invert colormap stop order")
                    .clicked()
                {
                    for stop in &mut style.colormap.stops {
                        stop.t = 1.0 - stop.t;
                    }
                    style
                        .colormap
                        .stops
                        .sort_by(|a, b| a.t.total_cmp(&b.t));
                    style.volume_tf.color_map = style.colormap.clone();
                }
            });

            paint_colormap_preview(ui, &style.colormap, 16.0);

            if ui
                .checkbox(&mut style.colormap.log_scale, "Log Scale")
                .changed()
            {
                style.volume_tf.color_map.log_scale = style.colormap.log_scale;
            }

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut style.auto_range, "Auto");
                if ui.small_button("Fit Data").clicked()
                    && let Some((mn, mx)) = chosen_array_range
                {
                    style.scalar_range = (mn, mx);
                    style.volume_tf.scalar_range = (mn, mx);
                }
                if ui
                    .small_button("±Sym")
                    .on_hover_text("Symmetric range around 0")
                    .clicked()
                {
                    let (mn, mx) = chosen_array_range.unwrap_or(style.scalar_range);
                    let m = mn.abs().max(mx.abs()).max(1e-6);
                    style.auto_range = false;
                    style.scalar_range = (-m, m);
                    style.volume_tf.scalar_range = (-m, m);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Min:");
                let mut mn = style.scalar_range.0;
                if ui
                    .add(egui::DragValue::new(&mut mn).speed(0.1))
                    .changed()
                {
                    style.auto_range = false;
                    style.scalar_range.0 = mn;
                    style.volume_tf.scalar_range.0 = mn;
                }
                ui.label("Max:");
                let mut mx = style.scalar_range.1;
                if ui
                    .add(egui::DragValue::new(&mut mx).speed(0.1))
                    .changed()
                {
                    style.auto_range = false;
                    style.scalar_range.1 = mx;
                    style.volume_tf.scalar_range.1 = mx;
                }
            });
        });

        section_card(ui, self.theme, "Material & Geometry Style", |ui| {
            ui.add(egui::Slider::new(&mut style.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut style.ambient, 0.0..=1.0).text("Ambient"));
            ui.add(egui::Slider::new(&mut style.specular, 0.0..=1.0).text("Specular"));
            ui.add(egui::Slider::new(&mut style.shininess, 1.0..=128.0).text("Shininess"));
            ui.checkbox(&mut style.two_sided, "Two-Sided Lighting");
            ui.separator();
            ui.add(egui::Slider::new(&mut style.point_size, 1.0..=24.0).text("Point Size"));
            ui.add(egui::Slider::new(&mut style.line_width, 0.5..=10.0).text("Line Width"));
            ui.add(egui::Slider::new(&mut style.edge_width, 0.25..=6.0).text("Edge Width"));
            ui.horizontal(|ui| {
                ui.label("Edge Color:");
                ui.color_edit_button_rgba_unmultiplied(&mut style.edge_color);
            });
        });

        if style.representation == Representation::Volume {
            section_card(ui, self.theme, "Volume Transfer Function", |ui| {
                ui.add(
                    egui::Slider::new(&mut style.volume_step_size, 0.05..=2.0).text("Step Size"),
                );
                ui.checkbox(&mut style.volume_shading, "Gradient Shading");
                let mut vol_clip_on = style.volume_clip_plane.is_some();
                if ui.checkbox(&mut vol_clip_on, "3D Volume Clip Plane").changed() {
                    if vol_clip_on {
                        let center = ds_opt
                            .as_ref()
                            .map(|d| d.bounds().center().to_array())
                            .unwrap_or([0.0, 0.0, 0.0]);
                        style.volume_clip_plane = Some((center, [1.0, 0.0, 0.0]));
                    } else {
                        style.volume_clip_plane = None;
                    }
                }
                if let Some((orig, norm)) = &mut style.volume_clip_plane {
                    drag_vec3(ui, "Clip Origin", orig, 0.1);
                    drag_vec3(ui, "Clip Normal", norm, 0.05);
                }

                ui.horizontal_wrapped(|ui| {
                    ui.label("Presets:");
                    if ui.small_button("Linear").clicked() {
                        style.volume_tf.opacity_points = vec![
                            OpacityPoint { t: 0.0, alpha: 0.0 },
                            OpacityPoint { t: 1.0, alpha: 0.85 },
                        ];
                    }
                    if ui.small_button("Soft").clicked() {
                        style.volume_tf.opacity_points = vec![
                            OpacityPoint { t: 0.0, alpha: 0.0 },
                            OpacityPoint { t: 0.25, alpha: 0.02 },
                            OpacityPoint { t: 0.55, alpha: 0.35 },
                            OpacityPoint { t: 0.85, alpha: 0.85 },
                            OpacityPoint { t: 1.0, alpha: 0.98 },
                        ];
                    }
                    if ui.small_button("Iso-Shell").clicked() {
                        style.volume_tf.opacity_points = vec![
                            OpacityPoint { t: 0.0, alpha: 0.0 },
                            OpacityPoint { t: 0.42, alpha: 0.0 },
                            OpacityPoint { t: 0.50, alpha: 0.85 },
                            OpacityPoint { t: 0.58, alpha: 0.0 },
                            OpacityPoint { t: 1.0, alpha: 0.0 },
                        ];
                    }
                    if ui.small_button("Peak").clicked() {
                        style.volume_tf.opacity_points = vec![
                            OpacityPoint { t: 0.0, alpha: 0.0 },
                            OpacityPoint { t: 0.70, alpha: 0.0 },
                            OpacityPoint { t: 1.0, alpha: 0.98 },
                        ];
                    }
                });

                draw_opacity_tf_editor(ui, &mut style.volume_tf, volume_hist.as_ref(), self.theme);

                ui.collapsing("Control Points Table", |ui| {
                    let mut remove_idx: Option<usize> = None;
                    for (idx, pt) in style.volume_tf.opacity_points.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("#{idx}"));
                            ui.add(
                                egui::DragValue::new(&mut pt.t)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("t="),
                            );
                            ui.add(
                                egui::DragValue::new(&mut pt.alpha)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("a="),
                            );
                            if ui.small_button("×").clicked() {
                                remove_idx = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx
                        && style.volume_tf.opacity_points.len() > 2
                    {
                        style.volume_tf.opacity_points.remove(idx);
                    }
                    if ui.button("+ Add Opacity Point").clicked() {
                        style
                            .volume_tf
                            .opacity_points
                            .push(OpacityPoint { t: 0.5, alpha: 0.35 });
                        style
                            .volume_tf
                            .opacity_points
                            .sort_by(|a, b| a.t.total_cmp(&b.t));
                    }
                });
            });
        }

        let style_changed = *style != prev_style;
        if style_changed {
            self.invalidate_viewport();
        }

        self.draw_multiblock_patches_section(ui, sel_id, "display");

        let prev_sb = self.session.settings.scalar_bar.clone();
        let prev_show_sb = self.show_scalar_bar;
        self.draw_scalar_bar_card(ui, "display");
        if self.session.settings.scalar_bar != prev_sb || self.show_scalar_bar != prev_show_sb {
            self.invalidate_viewport();
        }
    }

    pub(super) fn draw_information_tab(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        let Some(sel_id) = self.selected_node else {
            section_card(ui, self.theme, "Information", |ui| {
                ui.weak("Select a node to inspect dataset metadata.");
            });
            return;
        };
        let _ = self.session.evaluate(sel_id);
        let Ok(node) = self.session.node(sel_id).cloned() else {
            return;
        };
        let Some(ds) = node.dataset().cloned() else {
            section_card(ui, self.theme, "Information", |ui| {
                ui.weak("Dataset not evaluated yet.");
            });
            return;
        };

        if let Some(pick) = self.last_pick.clone() {
            section_card(ui, self.theme, "Last 3D Pick", |ui| {
                ui.monospace(format!(
                    "Actor: {}  Pos: ({:.3}, {:.3}, {:.3})",
                    pick.actor_name, pick.world_pos[0], pick.world_pos[1], pick.world_pos[2]
                ));
                if let Some(bp) = &pick.block_path {
                    ui.monospace(format!("Block: {bp}"));
                }
                let pt_str = pick.point_index.map_or_else(|| "-".into(), |i| i.to_string());
                let cell_str = pick.cell_index.map_or_else(|| "-".into(), |i| i.to_string());
                let scalar_str = pick
                    .scalar_value
                    .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
                ui.monospace(format!("Pt: {pt_str}  Cell: {cell_str}  Scalar: {scalar_str}"));
                if let Some((domain, idx)) = pick
                    .point_index
                    .map(|i| (PickDomain::Point, i))
                    .or_else(|| pick.cell_index.map(|i| (PickDomain::Cell, i)))
                {
                    ui.horizontal(|ui| {
                        if ui.small_button("Reveal in Spreadsheet").clicked() {
                            self.reveal_in_spreadsheet(NodeId(pick.actor_id), domain, idx);
                        }
                        if pick.point_index.is_some()
                            && let Some(ci) = pick.cell_index
                            && ui.small_button(format!("Reveal Cell #{ci}")).clicked()
                        {
                            self.reveal_in_spreadsheet(NodeId(pick.actor_id), PickDomain::Cell, ci);
                        }
                    });
                }
            });
        }

        section_card(ui, self.theme, "Dataset Summary", |ui| {
            egui::Grid::new("info_summary_grid")
                .num_columns(2)
                .spacing([12.0, 4.0])
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Name").color(pal.muted));
                    ui.strong(&node.name);
                    ui.end_row();
                    ui.label(egui::RichText::new("Type").color(pal.muted));
                    ui.label(ds.type_name());
                    ui.end_row();
                    ui.label(egui::RichText::new("Points").color(pal.muted));
                    ui.monospace(format!("{}", ds.num_points()));
                    ui.end_row();
                    ui.label(egui::RichText::new("Cells").color(pal.muted));
                    ui.monospace(format!("{}", ds.num_cells()));
                    ui.end_row();
                    ui.label(egui::RichText::new("Memory").color(pal.muted));
                    ui.monospace(format_memory_bytes(ds.memory_bytes()));
                    ui.end_row();
                });
        });

        let bb = node.style.dataset_bounds(&ds);
        if bb.is_valid() {
            let ext = bb.max - bb.min;
            section_card(ui, self.theme, "Spatial Bounds", |ui| {
                ui.monospace(format!("X: [{:.4}, {:.4}]  (Δ {:.4})", bb.min.x, bb.max.x, ext.x));
                ui.monospace(format!("Y: [{:.4}, {:.4}]  (Δ {:.4})", bb.min.y, bb.max.y, ext.y));
                ui.monospace(format!("Z: [{:.4}, {:.4}]  (Δ {:.4})", bb.min.z, bb.max.z, ext.z));
                ui.monospace(format!("Diagonal: {:.4}", bb.diagonal()));
            });
        }

        if let Some(pd) = ds.as_poly_data() {
            let area = polydata_surface_area(pd);
            section_card(ui, self.theme, "Mesh Topology & Area", |ui| {
                egui::Grid::new("poly_topo_grid")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Surface Area").color(pal.muted));
                        ui.monospace(format!("{area:.5}"));
                        ui.end_row();
                        ui.label(egui::RichText::new("Polygons / Strips").color(pal.muted));
                        ui.monospace(format!("{} / {}", pd.polys.len(), pd.strips.len()));
                        ui.end_row();
                        ui.label(egui::RichText::new("Lines / Verts").color(pal.muted));
                        ui.monospace(format!("{} / {}", pd.lines.len(), pd.verts.len()));
                        ui.end_row();
                    });
            });
        }

        for (title, assoc, fd_opt) in [
            ("Point Arrays", AttributeAssociation::Point, ds.point_data()),
            ("Cell Arrays", AttributeAssociation::Cell, ds.cell_data()),
        ] {
            if let Some(fd) = fd_opt
                && !fd.is_empty()
            {
                section_card(ui, self.theme, &format!("{title} ({})", fd.len()), |ui| {
                    for arr in fd.arrays() {
                        let (mn, mx) = arr.range(None);
                        ui.horizontal(|ui| {
                            ui.strong(&arr.name);
                            ui.label(
                                egui::RichText::new(format!(
                                    "({}, {}c)",
                                    arr.data.scalar_type().vtk_name(),
                                    arr.num_components
                                ))
                                .size(11.0)
                                .color(pal.muted),
                            );
                        });
                        ui.monospace(format!("  Range: [{mn:.4}, {mx:.4}]"));
                        let key = (sel_id, arr.name.clone(), assoc);
                        if let Some(stats) = self.info_stats_cache.get(&key) {
                            ui.monospace(format!(
                                "  Mean: {:.4} | StdDev: {:.4}",
                                stats.mean, stats.std_dev
                            ));
                        } else if ui.small_button("Compute Mean / StdDev").clicked()
                            && let Ok(res) = compute_histogram(&ds, &arr.name, assoc, None, 16)
                        {
                            self.info_stats_cache.insert(key, res);
                        }
                        ui.add_space(2.0);
                    }
                });
            }
        }
    }

    pub(super) fn draw_scene_tab(&mut self, ui: &mut egui::Ui) {
        let prev_camera = self.session.camera.clone();
        let prev_settings = self.session.settings.clone();
        let prev_widgets = self.show_widgets;

        section_card(ui, self.theme, "Viewport Background", |ui| {
            ui.horizontal_wrapped(|ui| {
                for &preset in ViewportBgPreset::ALL {
                    let (top, bot) = preset.colors();
                    let active = (self.session.settings.bg_top[0] - top[0]).abs() < 0.01
                        && (self.session.settings.bg_bottom[0] - bot[0]).abs() < 0.01;
                    if ui.selectable_label(active, preset.name()).clicked() {
                        self.session.settings.bg_top = top;
                        self.session.settings.bg_bottom = bot;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Top:");
                ui.color_edit_button_rgb(&mut self.session.settings.bg_top);
                ui.label("Bottom:");
                ui.color_edit_button_rgb(&mut self.session.settings.bg_bottom);
            });
        });

        section_card(ui, self.theme, "Camera & Navigation", |ui| {
            ui.checkbox(&mut self.session.camera.orthographic, "Orthographic Projection");
            if !self.session.camera.orthographic {
                ui.add(
                    egui::Slider::new(&mut self.session.camera.fov_y_deg, 15.0..=110.0)
                        .text("FOV (deg)"),
                );
            } else {
                ui.add(
                    egui::DragValue::new(&mut self.session.camera.ortho_scale)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .prefix("Ortho Scale: "),
                );
            }
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.turntable_enabled, "Turntable");
                ui.add_enabled(
                    self.turntable_enabled,
                    egui::Slider::new(&mut self.turntable_speed, 0.05..=2.5).text("Speed"),
                );
            });
            ui.label("Position:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.session.camera.position.x).speed(0.1).prefix("X:"));
                ui.add(egui::DragValue::new(&mut self.session.camera.position.y).speed(0.1).prefix("Y:"));
                ui.add(egui::DragValue::new(&mut self.session.camera.position.z).speed(0.1).prefix("Z:"));
            });
            ui.label("Target:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.session.camera.target.x).speed(0.1).prefix("X:"));
                ui.add(egui::DragValue::new(&mut self.session.camera.target.y).speed(0.1).prefix("Y:"));
                ui.add(egui::DragValue::new(&mut self.session.camera.target.z).speed(0.1).prefix("Z:"));
            });
            ui.horizontal(|ui| {
                if ui.button("Reset Camera (R)").clicked() {
                    self.reset_active_pane_camera();
                }
                if ui.button("Focus Selected (F)").clicked() {
                    self.focus_selected_node();
                }
                if ui.button("Bookmark View").clicked() {
                    let idx = self.camera_bookmarks.len() + 1;
                    self.camera_bookmarks.push(CameraBookmark {
                        name: format!("View {idx}"),
                        camera: self.session.camera.clone(),
                    });
                }
            });
            if !self.camera_bookmarks.is_empty() {
                ui.add_space(3.0);
                let mut remove_bm: Option<usize> = None;
                ui.horizontal_wrapped(|ui| {
                    for (idx, bm) in self.camera_bookmarks.iter().enumerate() {
                        if ui.small_button(&bm.name).clicked() {
                            self.session.camera = bm.camera.clone();
                        }
                        if ui.small_button("×").on_hover_text("Delete bookmark").clicked() {
                            remove_bm = Some(idx);
                        }
                    }
                });
                if let Some(idx) = remove_bm {
                    self.camera_bookmarks.remove(idx);
                }
            }
        });

        section_card(ui, self.theme, "Render Quality & Overlays", |ui| {
            let mut msaa_4x = self.session.settings.msaa_samples >= 4;
            if ui.checkbox(&mut msaa_4x, "4x MSAA Anti-Aliasing").changed() {
                self.session.settings.msaa_samples = if msaa_4x { 4 } else { 1 };
            }
            ui.checkbox(&mut self.session.settings.edl_enabled, "Eye-Dome Lighting (EDL)");
            if self.session.settings.edl_enabled {
                ui.add(
                    egui::Slider::new(&mut self.session.settings.edl_strength, 0.0..=4.0)
                        .text("EDL Strength"),
                );
            }
            ui.checkbox(&mut self.session.settings.show_axes_gizmo, "3D Axes Gizmo");
            ui.checkbox(&mut self.session.settings.show_bounding_box, "Scene Bounding Box");
            ui.checkbox(&mut self.session.settings.show_axes_grid, "3D Cube Axes Grid");
            ui.checkbox(
                &mut self.session.settings.transparent_background,
                "Transparent Background (Export)",
            );
            ui.checkbox(&mut self.show_widgets, "3D Filter Widgets");
            ui.checkbox(&mut self.show_scalar_bar, "Scalar Color Bar");
            ui.checkbox(
                &mut self.session.settings.show_time_annotation,
                "Time Annotation (Time:)",
            );
            if self.session.settings.show_time_annotation {
                let mut label_buf = self
                    .session
                    .settings
                    .time_label
                    .clone()
                    .unwrap_or_default();
                ui.horizontal(|ui| {
                    ui.label("Time Label:");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut label_buf)
                                .hint_text("auto (Time: <t>)")
                                .desired_width(150.0),
                        )
                        .changed()
                    {
                        self.session.settings.time_label = if label_buf.trim().is_empty() {
                            None
                        } else {
                            Some(label_buf)
                        };
                    }
                });
            }
        });

        self.draw_scalar_bar_card(ui, "scene");
        self.draw_annotations_card(ui);

        if self.session.camera != prev_camera {
            self.sync_active_camera_to_panes();
            self.invalidate_viewport();
        } else if self.session.settings != prev_settings || self.show_widgets != prev_widgets {
            self.invalidate_viewport();
        }
    }

    fn draw_scalar_bar_card(&mut self, ui: &mut egui::Ui, id_salt: &str) {
        section_card(ui, self.theme, "Scalar Color Bar", |ui| {
            ui.push_id(("scalar_bar_cfg", id_salt), |ui| {
                ui.checkbox(&mut self.show_scalar_bar, "Show Scalar Color Bar");
                let sb = &mut self.session.settings.scalar_bar;
                ui.checkbox(&mut sb.horizontal, "Horizontal Orientation");
                let mut title_buf = sb.custom_title.clone().unwrap_or_default();
                ui.horizontal(|ui| {
                    ui.label("Title:");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut title_buf)
                                .hint_text("auto (scalar array)")
                                .desired_width(160.0),
                        )
                        .changed()
                    {
                        sb.custom_title = if title_buf.trim().is_empty() {
                            None
                        } else {
                            Some(title_buf)
                        };
                    }
                });
                let mut labels_u32 = sb.num_labels.clamp(2, 7) as u32;
                if ui
                    .add(egui::Slider::new(&mut labels_u32, 2..=7).text("Labels"))
                    .changed()
                {
                    sb.num_labels = labels_u32 as usize;
                }
                ui.add(
                    egui::Slider::new(&mut sb.font_scale, 0.6..=2.5)
                        .max_decimals(2)
                        .text("Font Scale"),
                );
                ui.horizontal(|ui| {
                    if ui.button("Reset Pos").clicked() {
                        sb.pos = None;
                    }
                    if ui.button("Reset Size").clicked() {
                        sb.size = ScalarBarConfig::default().size;
                    }
                });
            });
        });
    }

    fn draw_annotations_card(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        section_card(ui, self.theme, "Annotations", |ui| {
            ui.horizontal(|ui| {
                if ui.button("+ Add Annotation").clicked() {
                    let id = self.next_annotation_id;
                    self.next_annotation_id += 1;
                    let idx = self.session.settings.annotations.len() + 1;
                    let offset_y = (0.14 + (idx.saturating_sub(1) as f32) * 0.06).min(0.85);
                    self.session.settings.annotations.push(ViewportAnnotation {
                        id,
                        text: format!("Note {idx}"),
                        pos: [0.05, offset_y],
                        font_size: 16.0,
                        color: [0.94, 0.96, 1.0, 1.0],
                        visible: true,
                    });
                }
                if !self.session.settings.annotations.is_empty() {
                    ui.label(
                        egui::RichText::new("Drag pills in 3D view")
                            .size(11.0)
                            .color(pal.muted),
                    );
                }
            });

            if self.session.settings.annotations.is_empty() {
                ui.weak("No viewport annotations yet.");
                return;
            }

            let mut remove_idx: Option<usize> = None;
            for (idx, ann) in self.session.settings.annotations.iter_mut().enumerate() {
                ui.push_id(("ann_row", ann.id, idx), |ui| {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut ann.visible, "");
                        ui.add(
                            egui::TextEdit::singleline(&mut ann.text)
                                .desired_width(140.0)
                                .hint_text("Annotation text..."),
                        );
                        ui.color_edit_button_rgba_unmultiplied(&mut ann.color);
                        if ui
                            .small_button("×")
                            .on_hover_text("Delete annotation")
                            .clicked()
                        {
                            remove_idx = Some(idx);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut ann.font_size)
                                .speed(0.25)
                                .range(8.0..=48.0)
                                .prefix("Size: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut ann.pos[0])
                                .speed(0.005)
                                .range(0.0..=1.0)
                                .max_decimals(3)
                                .prefix("X: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut ann.pos[1])
                                .speed(0.005)
                                .range(0.0..=1.0)
                                .max_decimals(3)
                                .prefix("Y: "),
                        );
                    });
                });
            }
            if let Some(idx) = remove_idx {
                self.session.settings.annotations.remove(idx);
            }
        });
    }

    fn draw_multiblock_patches_section(&mut self, ui: &mut egui::Ui, sel_id: NodeId, id_salt: &str) {
        let _ = self.session.evaluate(sel_id);
        let ds_opt = self
            .session
            .node(sel_id)
            .ok()
            .and_then(|n| n.dataset().cloned());
        let Some(DataSet::MultiBlock(mb)) = ds_opt.as_deref() else {
            return;
        };

        let entries = collect_multiblock_entries(mb);
        if entries.is_empty() {
            return;
        }
        let leaf_paths: Vec<String> = entries
            .iter()
            .filter(|e| !e.is_group)
            .map(|e| e.key().to_string())
            .collect();
        let all_keys: Vec<String> = entries.iter().map(|e| e.key().to_string()).collect();

        let pal = palette(self.theme);
        let picked_fallback = self
            .last_pick
            .as_ref()
            .filter(|p| NodeId(p.actor_id) == sel_id)
            .and_then(|p| p.block_path.clone());
        let focused_path = self
            .selected_multiblock_patch
            .as_ref()
            .filter(|(nid, _)| *nid == sel_id)
            .map(|(_, p)| p.clone())
            .or(picked_fallback);

        let Ok(node) = self.session.node_mut(sel_id) else {
            return;
        };
        let prev_hidden = node.style.hidden_blocks.clone();
        let prev_colors = node.style.block_colors.clone();
        let style = &mut node.style;
        let mut new_focused: Option<String> = None;

        section_card(ui, self.theme, "", |ui| {
            egui::CollapsingHeader::new(format!("MultiBlock Patches ({})", entries.len()))
                .id_salt(("multiblock_patches_hdr", id_salt, sel_id.0))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui.small_button("All").clicked() {
                            style.hidden_blocks.clear();
                        }
                        if ui.small_button("None").clicked() {
                            let targets = if leaf_paths.is_empty() {
                                &all_keys
                            } else {
                                &leaf_paths
                            };
                            for k in targets {
                                style.hidden_blocks.insert(k.clone());
                            }
                        }
                        if ui.small_button("Invert").clicked() {
                            let targets: Vec<&MultiBlockEntry> = if leaf_paths.is_empty() {
                                entries.iter().collect()
                            } else {
                                entries.iter().filter(|e| !e.is_group).collect()
                            };
                            for e in targets {
                                let was_vis = style.is_block_visible(&e.path, &e.name);
                                if was_vis {
                                    style.hidden_blocks.insert(e.key().to_string());
                                } else {
                                    unhide_block_key(&mut style.hidden_blocks, &e.path, &e.name);
                                }
                            }
                        }
                        let can_solo = focused_path.is_some();
                        if ui
                            .add_enabled(can_solo, egui::Button::new("Solo").small())
                            .on_hover_text("Show only the focused/picked block")
                            .clicked()
                            && let Some(ref solo_target) = focused_path
                        {
                            style.hidden_blocks.clear();
                            let targets = if leaf_paths.is_empty() {
                                &all_keys
                            } else {
                                &leaf_paths
                            };
                            let prefix = format!("{solo_target}/");
                            for k in targets {
                                if k != solo_target && !k.starts_with(&prefix) {
                                    style.hidden_blocks.insert(k.clone());
                                }
                            }
                        }
                    });

                    if entries.len() > 1 {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.multiblock_filter)
                                .hint_text("Filter blocks...")
                                .desired_width(f32::INFINITY),
                        );
                    }

                    let needle = self.multiblock_filter.trim().to_ascii_lowercase();
                    egui::ScrollArea::vertical()
                        .id_salt(("multiblock_patches_scroll", id_salt, sel_id.0))
                        .max_height(210.0)
                        .show(ui, |ui| {
                            for (idx, entry) in entries.iter().enumerate() {
                                if !needle.is_empty()
                                    && !entry.path.to_ascii_lowercase().contains(&needle)
                                    && !entry.name.to_ascii_lowercase().contains(&needle)
                                {
                                    continue;
                                }
                                let key = entry.key().to_string();
                                let mut visible = style.is_block_visible(&entry.path, &entry.name);
                                let is_focused = focused_path.as_deref() == Some(key.as_str());
                                let override_col =
                                    style.block_color_override(&entry.path, &entry.name);

                                ui.push_id(("mb_row", idx, &entry.path), |ui| {
                                    ui.horizontal(|ui| {
                                        if entry.depth > 0 {
                                            ui.add_space(entry.depth as f32 * 12.0);
                                        }
                                        if ui.checkbox(&mut visible, "").changed() {
                                            if visible {
                                                unhide_block_key(
                                                    &mut style.hidden_blocks,
                                                    &entry.path,
                                                    &entry.name,
                                                );
                                            } else {
                                                style.hidden_blocks.insert(key.clone());
                                            }
                                        }
                                        let mut col =
                                            override_col.unwrap_or([0.82, 0.84, 0.88, 1.0]);
                                        if ui
                                            .color_edit_button_rgba_unmultiplied(&mut col)
                                            .changed()
                                        {
                                            style.block_colors.insert(key.clone(), col);
                                        }
                                        if override_col.is_some()
                                            && ui
                                                .small_button("×")
                                                .on_hover_text("Clear block color override")
                                                .clicked()
                                        {
                                            style.block_colors.remove(&entry.path);
                                            style.block_colors.remove(&entry.name);
                                        }
                                        let display_name = if entry.name.is_empty() {
                                            "(unnamed)"
                                        } else {
                                            entry.name.as_str()
                                        };
                                        let label_resp =
                                            ui.selectable_label(is_focused, display_name);
                                        if label_resp.clicked() {
                                            new_focused = Some(key.clone());
                                        }
                                        label_resp.on_hover_text(format!(
                                            "{} ({} pts, {} cells)",
                                            entry.path, entry.num_points, entry.num_cells
                                        ));
                                        if entry.is_group {
                                            ui.label(
                                                egui::RichText::new("[Group]")
                                                    .size(10.0)
                                                    .color(pal.muted),
                                            );
                                        }
                                    });
                                });
                            }
                        });
                });
        });

        let style_changed =
            node.style.hidden_blocks != prev_hidden || node.style.block_colors != prev_colors;
        if let Some(p) = new_focused {
            self.selected_multiblock_patch = Some((sel_id, p));
        }
        if style_changed {
            self.clear_stats_caches_for_node(sel_id);
            self.invalidate_viewport();
        }
    }
}

struct MultiBlockEntry {
    depth: usize,
    name: String,
    path: String,
    is_group: bool,
    num_points: usize,
    num_cells: usize,
}

impl MultiBlockEntry {
    fn key(&self) -> &str {
        if self.path.is_empty() {
            &self.name
        } else {
            &self.path
        }
    }
}

fn collect_multiblock_entries(mb: &MultiBlockDataSet) -> Vec<MultiBlockEntry> {
    fn visit(
        mb: &MultiBlockDataSet,
        parent_path: &str,
        depth: usize,
        out: &mut Vec<MultiBlockEntry>,
    ) {
        for (name, child) in &mb.blocks {
            let child_path = if parent_path.is_empty() {
                name.clone()
            } else if name.is_empty() {
                parent_path.to_string()
            } else {
                format!("{parent_path}/{name}")
            };
            let is_group = matches!(child, DataSet::MultiBlock(_));
            out.push(MultiBlockEntry {
                depth,
                name: name.clone(),
                path: child_path.clone(),
                is_group,
                num_points: child.num_points(),
                num_cells: child.num_cells(),
            });
            if let DataSet::MultiBlock(sub) = child {
                visit(sub, &child_path, depth + 1, out);
            }
        }
    }
    let mut out = Vec::new();
    visit(mb, "", 0, &mut out);
    out
}

fn unhide_block_key(hidden: &mut BTreeSet<String>, path: &str, name: &str) {
    if !path.is_empty() {
        hidden.remove(path);
        // Also clear any hidden ancestor group path so unhiding a leaf makes it visible.
        let mut cur = path;
        while let Some((parent, _)) = cur.rsplit_once('/') {
            hidden.remove(parent);
            cur = parent;
        }
    }
    if !name.is_empty() {
        hidden.remove(name);
    }
}

pub(super) fn paint_colormap_preview(ui: &mut egui::Ui, cmap: &ColorMap, height: f32) {
    let width = ui.available_width().max(60.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let steps = 48;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let rgb = cmap.sample_rgb((t0 + t1) * 0.5);
        let c = egui::Color32::from_rgb(
            (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
        );
        let x0 = rect.left() + t0 * rect.width();
        let x1 = rect.left() + t1 * rect.width();
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
            0.0,
            c,
        );
    }
    painter.rect_stroke(
        rect,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(110)),
        egui::StrokeKind::Inside,
    );
}

