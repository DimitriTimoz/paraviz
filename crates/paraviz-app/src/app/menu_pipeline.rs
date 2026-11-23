use super::*;

impl ParaVizApp {
    pub(super) fn draw_menu_bar(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        egui::MenuBar::new().ui(ui, |ui| {
            paint_logo_badge(ui, self.theme);
            ui.label(egui::RichText::new("ParaViz").strong().size(13.5));
            pill_badge(
                ui,
                "WebGPU",
                pal.accent_dim,
                pal.accent,
            );
            ui.separator();

            ui.menu_button("File", |ui| {
                if ui.button("Open File...").clicked() {
                    ui.close();
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(p) = native_pick_scientific_file(self.last_export_dir.as_deref()) {
                        self.open_picked_scientific_file(p);
                    }
                    #[cfg(target_arch = "wasm32")]
                    spawn_wasm_open_dialog(self.pending_file_loads.clone(), false);
                }
                if ui.button("Open File by Path / With Reader...").clicked() {
                    self.modal = Some(AppModal::OpenFile {
                        path: String::new(),
                        reader: paraviz::io::DatasetReader::Auto,
                    });
                    ui.close();
                }
                if ui.button("Import Raw 3D Volume (.raw)...").clicked() {
                    self.modal = Some(AppModal::RawVolume(RawVolumeModalState::with_path("")));
                    ui.close();
                }
                if !self.recent_files.is_empty() {
                    ui.menu_button("Recent Files", |ui| {
                        let recents = self.recent_files.clone();
                        for p in recents {
                            let label = p
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("File");
                            if ui
                                .button(label)
                                .on_hover_text(p.display().to_string())
                                .clicked()
                            {
                                self.load_path(&p);
                                ui.close();
                            }
                        }
                    });
                }
                ui.separator();
                if ui.button("Save State JSON...").clicked() {
                    let _path = self.default_export_path("paraviz_state.json");
                    ui.close();
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(p) = native_save_file(
                        "Save Pipeline State",
                        &_path,
                        &[("ParaViz State JSON", &["json"])],
                    ) {
                        self.execute_modal_action(ModalAction::SaveState(p));
                    }
                    #[cfg(target_arch = "wasm32")]
                    match self.session.save_state_json() {
                        Ok(json) => {
                            wasm_download_bytes("paraviz_state.json", json.as_bytes(), "application/json");
                            self.status_message = "Downloaded paraviz_state.json".into();
                        }
                        Err(e) => self.error_message = Some(format!("Serialize failed: {e}")),
                    }
                }
                if ui.button("Load State JSON...").clicked() {
                    ui.close();
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(p) = native_pick_state_file(self.last_export_dir.as_deref()) {
                        self.load_path(&p);
                    }
                    #[cfg(target_arch = "wasm32")]
                    spawn_wasm_open_dialog(self.pending_file_loads.clone(), true);
                }
                ui.separator();
                let has_sel = self.selected_node.is_some();
                if ui
                    .add_enabled(has_sel, egui::Button::new("Export Active Dataset..."))
                    .clicked()
                {
                    let path = self.default_export_path("export.vtp");
                    ui.close();
                    if let Some(p) = native_save_file(
                        "Export Active Dataset",
                        &path,
                        &[
                            ("VTK PolyData", &["vtp"]),
                            ("VTK Unstructured", &["vtu"]),
                            ("VTK ImageData", &["vti"]),
                            ("Legacy VTK", &["vtk"]),
                            ("STL Mesh", &["stl"]),
                            ("Wavefront OBJ", &["obj"]),
                            ("PLY Mesh", &["ply"]),
                            ("CSV Table", &["csv"]),
                            ("glTF Binary", &["glb"]),
                            ("Interactive HTML", &["html"]),
                        ],
                    ) {
                        self.execute_modal_action(ModalAction::ExportDataset(p));
                    }
                }
                if ui.button("Save Screenshot PNG...").clicked() {
                    let (w, h) = self.active_viewport_tex_size();
                    let path = self.default_export_path("screenshot.png");
                    self.modal = Some(AppModal::SaveScreenshot(ScreenshotModalState {
                        path,
                        width: w.max(1280),
                        height: h.max(720),
                        use_viewport_size: true,
                        transparent: self.session.settings.transparent_background,
                        include_scalar_bar: self.show_scalar_bar,
                    }));
                    ui.close();
                }
                if ui.button("Export Animation (GIF)...").clicked() {
                    let (w, h) = self.active_viewport_tex_size();
                    let has_ts = !self.session.all_time_steps().is_empty();
                    let path = self.default_export_path("paraviz_animation.gif");
                    self.modal = Some(AppModal::ExportGif(GifModalState {
                        path,
                        width: w.clamp(320, 1280),
                        height: h.clamp(240, 720),
                        frames: 36,
                        delay_ms: if has_ts { 120 } else { 60 },
                        is_turntable: !has_ts,
                        transparent: self.session.settings.transparent_background,
                        include_scalar_bar: self.show_scalar_bar,
                    }));
                    ui.close();
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    ui.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("Sources", |ui| {
                let entries: Vec<(&'static str, SourceConfig)> = vec![
                    ("Sphere", SourceConfig::Sphere(SphereSource::default())),
                    ("Cube", SourceConfig::Cube(CubeSource::default())),
                    ("Cone", SourceConfig::Cone(ConeSource::default())),
                    ("Cylinder", SourceConfig::Cylinder(CylinderSource::default())),
                    ("Plane", SourceConfig::Plane(PlaneSource::default())),
                    ("Line", SourceConfig::Line(LineSource::default())),
                    ("Arrow", SourceConfig::Arrow(ArrowSource::default())),
                    ("Disk", SourceConfig::Disk(DiskSource::default())),
                    ("Torus", SourceConfig::Torus(TorusSource::default())),
                    ("Wavelet", SourceConfig::Wavelet(WaveletSource::default())),
                    (
                        "Vector Field",
                        SourceConfig::VectorField(VectorFieldSource::default()),
                    ),
                    (
                        "Superquadric",
                        SourceConfig::Superquadric(SuperquadricSource::default()),
                    ),
                    ("Gyroid", SourceConfig::Gyroid(GyroidSource::default())),
                    (
                        "Point Cloud",
                        SourceConfig::PointCloud(PointCloudSource::default()),
                    ),
                ];
                for (name, cfg) in entries {
                    if ui.button(name).clicked() {
                        self.add_source(cfg);
                        ui.close();
                    }
                }
            });

            egui::containers::menu::MenuButton::new("Filters")
                .config(
                    egui::containers::menu::MenuConfig::new()
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                )
                .ui(ui, |ui| {
                let enabled = self.selected_node.is_some();
                let is_imagedata = self
                    .selected_node
                    .and_then(|id| self.session.node(id).ok())
                    .and_then(|n| n.dataset())
                    .is_some_and(|ds| ds.as_image_data().is_some());
                let has_visible = self.session.nodes().iter().any(|n| n.style.visible);

                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.filter_menu_search)
                            .hint_text("Search 39 filters...")
                            .desired_width(170.0),
                    );
                    if !self.filter_menu_search.is_empty() && ui.small_button("×").clicked() {
                        self.filter_menu_search.clear();
                    }
                });
                ui.separator();

                let needle = self.filter_menu_search.trim().to_ascii_lowercase();
                if !needle.is_empty() {
                    let matches: Vec<FilterMenuEntry> = filter_menu_catalog()
                        .into_iter()
                        .filter(|e| {
                            e.label.to_ascii_lowercase().contains(&needle)
                                || e.category.to_ascii_lowercase().contains(&needle)
                        })
                        .collect();
                    if matches.is_empty() {
                        ui.weak("No matching filters.");
                    } else if let Some(act) = draw_filter_entries_scroll(
                        ui,
                        "filters_search_scroll",
                        340.0,
                        matches,
                        enabled,
                        is_imagedata,
                        has_visible,
                        true,
                    ) {
                        self.dispatch_filter_menu_action(act);
                        self.filter_menu_search.clear();
                        ui.close();
                    }
                    return;
                }

                let mut chosen: Option<FilterMenuAction> = None;

                ui.menu_button(
                    if self.workspace_profile == WorkspaceProfile::Full {
                        "Common".to_string()
                    } else {
                        format!("Common ({})", self.workspace_profile.label())
                    },
                    |ui| {
                        let common_entries: Vec<FilterMenuEntry> = filter_menu_catalog()
                            .into_iter()
                            .filter(|e| {
                                (e.common || self.favorite_filters.contains(e.label))
                                    && self.workspace_profile.matches_filter(e.label)
                            })
                            .collect();
                        if let Some(act) = draw_filter_entries_scroll(
                            ui,
                            "filters_common_scroll",
                            380.0,
                            common_entries,
                            enabled,
                            is_imagedata,
                            has_visible,
                            false,
                        ) {
                            chosen = Some(act);
                        }
                    },
                );
                ui.separator();

                for &cat in FILTER_CATEGORIES {
                    ui.menu_button(cat, |ui| {
                        let cat_entries: Vec<FilterMenuEntry> = filter_menu_catalog()
                            .into_iter()
                            .filter(|e| e.category == cat)
                            .collect();
                        if let Some(act) = draw_filter_entries_scroll(
                            ui,
                            ("filter_cat_scroll", cat),
                            380.0,
                            cat_entries,
                            enabled,
                            is_imagedata,
                            has_visible,
                            false,
                        ) {
                            chosen = Some(act);
                        }
                    });
                }

                ui.separator();
                ui.menu_button("All Alphabetical (A-Z)", |ui| {
                    let mut all = filter_menu_catalog();
                    all.sort_by(|a, b| a.label.cmp(b.label));
                    if let Some(act) = draw_filter_entries_scroll(
                        ui,
                        "filters_az_scroll",
                        380.0,
                        all,
                        enabled,
                        is_imagedata,
                        has_visible,
                        true,
                    ) {
                        chosen = Some(act);
                    }
                });

                if let Some(act) = chosen {
                    self.dispatch_filter_menu_action(act);
                    ui.close();
                }
            });

            ui.menu_button("Demos", |ui| {
                if ui.button("Open Demo Gallery...").clicked() {
                    self.modal = Some(AppModal::DemoGallery);
                    ui.close();
                }
                ui.separator();
                for &preset in DemoPreset::ALL {
                    if ui.button(preset.name()).clicked() {
                        self.load_demo(preset);
                        ui.close();
                    }
                }
            });

            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.show_left_dock, "Inspector Dock (Tab)");
                ui.checkbox(
                    &mut self.show_drawer,
                    "Data Drawer (Spreadsheet / Chart / Trace)",
                );
                ui.checkbox(&mut self.zen_mode, "Zen Mode (Full Viewport)");
                ui.separator();
                ui.menu_button("Viewport Layout", |ui| {
                    for &layout in ViewportLayout::ALL {
                        if ui
                            .selectable_label(self.viewport_layout == layout, layout.menu_label())
                            .clicked()
                        {
                            set_viewport_layout(
                                &mut self.viewport_layout,
                                &mut self.active_pane,
                                layout,
                            );
                            self.invalidate_viewport();
                            ui.close();
                        }
                    }
                });
                if ui
                    .checkbox(&mut self.link_cameras, "Link Cameras")
                    .changed()
                    && self.link_cameras
                {
                    self.sync_camera_from_pane(self.active_pane);
                    self.invalidate_viewport();
                }
                ui.separator();
                ui.checkbox(&mut self.show_scalar_bar, "Scalar Color Bar");
                if ui
                    .checkbox(
                        &mut self.session.settings.show_time_annotation,
                        "Time Annotation (Time:)",
                    )
                    .changed()
                {
                    self.invalidate_viewport();
                }
                if ui
                    .checkbox(&mut self.show_widgets, "3D Filter Widgets (W)")
                    .changed()
                {
                    self.invalidate_viewport();
                }
                if ui
                    .checkbox(&mut self.session.settings.show_axes_gizmo, "Axes Gizmo (G)")
                    .changed()
                {
                    self.invalidate_viewport();
                }
                if ui
                    .checkbox(
                        &mut self.session.settings.show_bounding_box,
                        "Bounding Box (B)",
                    )
                    .changed()
                {
                    self.invalidate_viewport();
                }
                ui.checkbox(&mut self.turntable_enabled, "Turntable Spin (T)");
                ui.separator();
                let mut msaa_4x = self.session.settings.msaa_samples >= 4;
                if ui.checkbox(&mut msaa_4x, "4x MSAA (M)").changed() {
                    self.session.settings.msaa_samples = if msaa_4x { 4 } else { 1 };
                    self.invalidate_viewport();
                }
                if ui
                    .checkbox(
                        &mut self.session.settings.edl_enabled,
                        "Eye-Dome Lighting (E)",
                    )
                    .changed()
                {
                    self.invalidate_viewport();
                }
                if self.session.settings.edl_enabled
                    && ui
                        .add(
                            egui::Slider::new(&mut self.session.settings.edl_strength, 0.0..=4.0)
                                .text("EDL Strength"),
                        )
                        .changed()
                {
                    self.invalidate_viewport();
                }
                ui.separator();
                ui.menu_button("UI Theme", |ui| {
                    if ui
                        .selectable_label(self.theme == UiTheme::DarkStudio, "Dark Studio")
                        .clicked()
                    {
                        self.theme = UiTheme::DarkStudio;
                        self.theme_applied = false;
                        ui.close();
                    }
                    if ui
                        .selectable_label(self.theme == UiTheme::LightStudio, "Light Studio")
                        .clicked()
                    {
                        self.theme = UiTheme::LightStudio;
                        self.theme_applied = false;
                        ui.close();
                    }
                });
                ui.separator();
                if ui.button("Rebuild UI Font Atlas").clicked() {
                    ui.ctx().set_fonts(egui::FontDefinitions::default());
                    self.theme_applied = false;
                    ui.close();
                }
            });

            ui.menu_button("Help", |ui| {
                if ui.button("Keyboard & Mouse Shortcuts (F1)").clicked() {
                    self.modal = Some(AppModal::Shortcuts);
                    ui.close();
                }
                if ui.button("Demo Gallery").clicked() {
                    self.modal = Some(AppModal::DemoGallery);
                    ui.close();
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.separator();
                    if ui.button("Check for Updates...").clicked() {
                        self.trigger_update_check(true);
                        self.show_update_modal = true;
                        ui.close();
                    }
                }
                ui.separator();
                if ui
                    .button("Report an Issue / Bug...")
                    .on_hover_text("Open GitHub Issues to report a bug or request a feature")
                    .clicked()
                {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(
                        "https://github.com/DimitriTimoz/paraviz/issues/new/choose",
                    ));
                    ui.close();
                }
            });

            // Right-aligned quick utility pills
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("?").on_hover_text("Shortcuts (F1)").clicked() {
                    self.modal = Some(AppModal::Shortcuts);
                }
                let theme_lbl = match self.theme {
                    UiTheme::DarkStudio => "Light UI",
                    UiTheme::LightStudio => "Dark UI",
                };
                if ui.small_button(theme_lbl).clicked() {
                    self.theme = match self.theme {
                        UiTheme::DarkStudio => UiTheme::LightStudio,
                        UiTheme::LightStudio => UiTheme::DarkStudio,
                    };
                    self.theme_applied = false;
                }
                ui.toggle_value(&mut self.zen_mode, "Zen")
                    .on_hover_text("Hide side & bottom panels");
                if ui.small_button("Open...").on_hover_text("Open File (Native Explorer)").clicked() {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(p) = native_pick_scientific_file(self.last_export_dir.as_deref()) {
                        self.open_picked_scientific_file(p);
                    }
                    #[cfg(target_arch = "wasm32")]
                    spawn_wasm_open_dialog(self.pending_file_loads.clone(), false);
                }
            });
        });
    }

    pub(super) fn draw_toolbar(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        ui.horizontal_wrapped(|ui| {
            // Undo / Redo + Command Palette + Workspace Profile
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("Undo").small())
                            .on_hover_text("Undo (Ctrl/Cmd+Z)")
                            .clicked()
                        {
                            self.undo();
                        }
                        if ui
                            .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("Redo").small())
                            .on_hover_text("Redo (Ctrl/Cmd+Shift+Z)")
                            .clicked()
                        {
                            self.redo();
                        }
                        ui.separator();
                        if ui
                            .small_button("⌘K Search")
                            .on_hover_text("Quick Command & Filter Palette (Ctrl/Cmd+K or Ctrl+Space)")
                            .clicked()
                        {
                            self.command_palette_open = true;
                            self.command_palette_query.clear();
                            self.command_palette_index = 0;
                            self.command_palette_focus_pending = true;
                        }
                        ui.separator();
                        egui::ComboBox::from_id_salt("workspace_profile_combo")
                            .selected_text(format!("Profile: {}", self.workspace_profile.label()))
                            .width(96.0)
                            .show_ui(ui, |ui| {
                                for &prof in WorkspaceProfile::ALL {
                                    if ui
                                        .selectable_value(
                                            &mut self.workspace_profile,
                                            prof,
                                            prof.label(),
                                        )
                                        .clicked()
                                    {
                                        self.status_message =
                                            format!("Workspace profile: {}", prof.label());
                                    }
                                }
                            });
                    });
                });

            // Camera group
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.small_button("Reset").on_hover_text("Reset camera (R)").clicked() {
                            self.reset_active_pane_camera();
                        }
                        if ui.small_button("Fit").on_hover_text("Focus selected node (F)").clicked() {
                            self.focus_selected_node();
                        }
                        for (label, preset) in [
                            ("+X", CameraPreset::PosX),
                            ("-X", CameraPreset::NegX),
                            ("+Y", CameraPreset::PosY),
                            ("-Y", CameraPreset::NegY),
                            ("+Z", CameraPreset::PosZ),
                            ("-Z", CameraPreset::NegZ),
                            ("Iso", CameraPreset::Isometric),
                        ] {
                            if ui.small_button(label).clicked() {
                                self.camera_preset_active(preset);
                            }
                        }
                        if ui
                            .toggle_value(
                                &mut self.panes[self.active_pane].camera.orthographic,
                                "Ortho",
                            )
                            .changed()
                        {
                            self.sync_camera_from_pane(self.active_pane);
                            self.invalidate_viewport();
                        }
                        ui.toggle_value(&mut self.turntable_enabled, "Spin")
                            .on_hover_text("Turntable orbit spin (T)");
                    });
                });

            // Interaction mode group
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.viewport_mode, ViewportMode::Orbit, "Orbit")
                            .on_hover_text("Orbit / Pan / Zoom (1)");
                        ui.selectable_value(&mut self.viewport_mode, ViewportMode::Pick, "Pick")
                            .on_hover_text("3D Raycast Pick (2)");
                        ui.selectable_value(&mut self.viewport_mode, ViewportMode::Ruler, "Ruler")
                            .on_hover_text("2-Point Distance Ruler (3)");
                        if !self.ruler_points.is_empty() && ui.small_button("Clear").clicked() {
                            self.ruler_points.clear();
                            self.invalidate_viewport();
                        }
                    });
                });

            // Render & overlay toggles
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .toggle_value(&mut self.session.settings.show_axes_gizmo, "Axes")
                            .changed()
                        {
                            self.invalidate_viewport();
                        }
                        if ui
                            .toggle_value(&mut self.session.settings.show_bounding_box, "Bounds")
                            .changed()
                        {
                            self.invalidate_viewport();
                        }
                        if ui.toggle_value(&mut self.show_widgets, "Widgets").changed() {
                            self.invalidate_viewport();
                        }
                        ui.toggle_value(&mut self.show_scalar_bar, "ColorBar");
                        if ui
                            .toggle_value(
                                &mut self.session.settings.show_time_annotation,
                                "Time:",
                            )
                            .on_hover_text("Toggle Time: annotation badge")
                            .changed()
                        {
                            self.invalidate_viewport();
                        }
                        let mut msaa_4x = self.session.settings.msaa_samples >= 4;
                        if ui.toggle_value(&mut msaa_4x, "MSAA 4x").changed() {
                            self.session.settings.msaa_samples = if msaa_4x { 4 } else { 1 };
                            self.invalidate_viewport();
                        }
                        if ui
                            .toggle_value(&mut self.session.settings.edl_enabled, "EDL")
                            .changed()
                        {
                            self.invalidate_viewport();
                        }
                        if self.session.settings.edl_enabled
                            && ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.session.settings.edl_strength,
                                        0.0..=4.0,
                                    )
                                    .max_decimals(2)
                                    .show_value(true),
                                )
                                .changed()
                        {
                            self.invalidate_viewport();
                        }
                    });
                });

            // Panel toggles
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.toggle_value(&mut self.show_left_dock, "Inspector");
                        ui.toggle_value(&mut self.show_drawer, "Data / Chart");
                    });
                });

            // Split layout & linked camera controls
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for &layout in ViewportLayout::ALL {
                            if ui
                                .selectable_label(self.viewport_layout == layout, layout.label())
                                .on_hover_text(layout.menu_label())
                                .clicked()
                            {
                                set_viewport_layout(
                                    &mut self.viewport_layout,
                                    &mut self.active_pane,
                                    layout,
                                );
                                self.invalidate_viewport();
                            }
                        }
                        ui.separator();
                        if ui
                            .toggle_value(&mut self.link_cameras, "Link Cameras")
                            .on_hover_text("Sync camera navigation across all 3D panes")
                            .changed()
                            && self.link_cameras
                        {
                            self.sync_camera_from_pane(self.active_pane);
                            self.invalidate_viewport();
                        }
                    });
                });

            // Time steps transport
            let steps = self.session.all_time_steps();
            if !steps.is_empty() {
                egui::Frame::new()
                    .fill(pal.card_bg)
                    .stroke(egui::Stroke::new(1.0, pal.card_border))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::symmetric(6, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Time:");
                            if ui.small_button("|<").clicked() {
                                self.session.set_time(steps[0]);
                                self.clear_stats_caches();
                                self.invalidate_viewport();
                            }
                            if ui.small_button("<").clicked() {
                                self.step_time(-1);
                            }
                            let play_label = if self.playing { "Pause" } else { "Play" };
                            if ui.small_button(play_label).clicked() {
                                self.playing = !self.playing;
                            }
                            if ui.small_button(">").clicked() {
                                self.step_time(1);
                            }
                            if ui.small_button(">|").clicked() {
                                self.session.set_time(*steps.last().unwrap());
                                self.clear_stats_caches();
                                self.invalidate_viewport();
                            }
                            let min_t = steps[0];
                            let max_t = *steps.last().unwrap();
                            let mut t = self.session.time;
                            if (max_t - min_t).abs() > 1e-12 {
                                if ui
                                    .add(
                                        egui::Slider::new(&mut t, min_t..=max_t)
                                            .max_decimals(3)
                                            .show_value(true),
                                    )
                                    .changed()
                                {
                                    self.session.set_time(t);
                                    self.clear_stats_caches();
                                    self.invalidate_viewport();
                                }
                            } else {
                                ui.label(format!("{t:.3}"));
                            }
                        });
                    });
            }
        });
    }

    pub(super) fn draw_left_dock(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);

        // Pipeline Browser Header + Quick Actions
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Pipeline").size(13.5).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let has_sel = self.selected_node.is_some();
                if ui
                    .add_enabled(has_sel, egui::Button::new("Del").small())
                    .on_hover_text("Delete selected node (Del)")
                    .clicked()
                {
                    self.delete_selected_node();
                }
                if ui
                    .add_enabled(has_sel, egui::Button::new("Dup").small())
                    .on_hover_text("Duplicate selected node")
                    .clicked()
                {
                    self.duplicate_selected_node();
                }
                let can_copy = self
                    .selected_node
                    .and_then(|id| self.session.node(id).ok())
                    .is_some_and(|n| n.filter_config.is_some());
                if ui
                    .add_enabled(can_copy, egui::Button::new("Copy").small())
                    .on_hover_text("Copy selected filter (Ctrl/Cmd+C)")
                    .clicked()
                    && let Some(sel) = self.selected_node
                {
                    self.copy_filter_from_node(sel);
                }
                if ui
                    .add_enabled(
                        has_sel && self.copied_filter.is_some(),
                        egui::Button::new("Paste").small(),
                    )
                    .on_hover_text("Paste copied filter onto selected node (Ctrl/Cmd+V)")
                    .clicked()
                    && let Some(sel) = self.selected_node
                {
                    self.paste_filter_to_node(sel);
                }
                if ui
                    .add_enabled(has_sel, egui::Button::new("Rename").small())
                    .on_hover_text("Rename selected node")
                    .clicked()
                    && let Some(sel) = self.selected_node
                    && let Ok(n) = self.session.node(sel)
                {
                    self.renaming_node = Some((sel, n.name.clone()));
                }
                if ui
                    .add_enabled(has_sel, egui::Button::new("Solo").small())
                    .on_hover_text("Show only selected node")
                    .clicked()
                {
                    self.solo_selected_node();
                }
                if ui
                    .small_button("None")
                    .on_hover_text("Hide all nodes")
                    .clicked()
                {
                    self.set_all_visibility(false);
                }
                if ui
                    .small_button("All")
                    .on_hover_text("Show all nodes")
                    .clicked()
                {
                    self.set_all_visibility(true);
                }
            });
        });
        ui.add_space(3.0);

        let ordered = self.pipeline_tree_order();
        let mut toggle_vis: Option<(NodeId, bool)> = None;
        let mut to_select: Option<NodeId> = None;
        let mut to_delete: Option<NodeId> = None;
        let mut to_copy: Option<(NodeId, bool)> = None;
        let mut to_paste: Option<(NodeId, bool)> = None;
        let mut commit_rename: Option<(NodeId, String)> = None;
        let mut cancel_rename = false;

        egui::Frame::new()
            .fill(pal.card_bg)
            .stroke(egui::Stroke::new(1.0, pal.card_border))
            .corner_radius(egui::CornerRadius::same(7))
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("pipeline_browser_scroll")
                    .max_height(185.0)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        if ordered.is_empty() {
                            ui.weak("Pipeline empty. Load a Demo or Source.");
                        }
                        for row in ordered {
                            let id = row.id;
                            let depth = row.depth;
                            let name = row.name;
                            let visible = row.visible;
                            let badge = row.badge;
                            let selected = self.selected_node == Some(id);
                            let row_bg = if selected {
                                pal.accent_dim
                            } else {
                                egui::Color32::TRANSPARENT
                            };
                            egui::Frame::new()
                                .fill(row_bg)
                                .corner_radius(egui::CornerRadius::same(5))
                                .inner_margin(egui::Margin::symmetric(4, 2))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        if depth > 0 {
                                            ui.add_space((depth - 1) as f32 * 12.0);
                                            ui.label(
                                                egui::RichText::new("└")
                                                    .color(pal.muted)
                                                    .monospace(),
                                            );
                                        }
                                        if visibility_eye_button(ui, visible, self.theme)
                                            .on_hover_text("Toggle visibility")
                                            .clicked()
                                        {
                                            toggle_vis = Some((id, !visible));
                                        }
                                        pill_badge(ui, badge.0, badge.1, badge.2);

                                        if let Some((rid, ref mut buf)) = self.renaming_node
                                            && rid == id
                                        {
                                            let r = ui.add(
                                                egui::TextEdit::singleline(buf).desired_width(120.0),
                                            );
                                            if r.lost_focus()
                                                || ui.input(|i| i.key_pressed(egui::Key::Enter))
                                            {
                                                commit_rename = Some((id, buf.clone()));
                                            } else if ui.input(|i| i.key_pressed(egui::Key::Escape))
                                            {
                                                cancel_rename = true;
                                            }
                                        } else {
                                            let lbl = ui.selectable_label(selected, &name);
                                            if lbl.clicked() {
                                                to_select = Some(id);
                                            }
                                            if lbl.double_clicked() {
                                                self.renaming_node = Some((id, name.clone()));
                                            }
                                            let is_filter = self
                                                .session
                                                .node(id)
                                                .is_ok_and(|n| n.filter_config.is_some());
                                            let has_copied = self.copied_filter.is_some();
                                            let has_chain = !self.copied_filter_chain.is_empty();
                                            lbl.context_menu(|ui| {
                                                if ui
                                                    .add_enabled(is_filter, egui::Button::new("Copy Filter"))
                                                    .clicked()
                                                {
                                                    to_copy = Some((id, false));
                                                    ui.close();
                                                }
                                                if ui
                                                    .add_enabled(is_filter, egui::Button::new("Copy Filter Chain"))
                                                    .clicked()
                                                {
                                                    to_copy = Some((id, true));
                                                    ui.close();
                                                }
                                                if ui
                                                    .add_enabled(has_copied, egui::Button::new("Paste Filter Here"))
                                                    .clicked()
                                                {
                                                    to_paste = Some((id, false));
                                                    ui.close();
                                                }
                                                if ui
                                                    .add_enabled(has_chain, egui::Button::new("Paste Filter Chain Here"))
                                                    .clicked()
                                                {
                                                    to_paste = Some((id, true));
                                                    ui.close();
                                                }
                                                ui.separator();
                                                if ui.button("Rename").clicked() {
                                                    self.renaming_node = Some((id, name.clone()));
                                                    ui.close();
                                                }
                                                if ui.button("Delete").clicked() {
                                                    to_delete = Some(id);
                                                    ui.close();
                                                }
                                            });
                                        }

                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui
                                                    .small_button("×")
                                                    .on_hover_text("Delete node")
                                                    .clicked()
                                                {
                                                    to_delete = Some(id);
                                                }
                                            },
                                        );
                                    });
                                });
                        }
                    });
            });

        if let Some((id, new_name)) = commit_rename {
            let trimmed = new_name.trim();
            if !trimmed.is_empty()
                && let Ok(n) = self.session.node_mut(id)
            {
                n.name = trimmed.to_string();
            }
            self.renaming_node = None;
        } else if cancel_rename {
            self.renaming_node = None;
        }
        if let Some((id, vis)) = toggle_vis {
            if vis {
                self.session.show(id);
            } else {
                self.session.hide(id);
            }
            self.invalidate_viewport();
        }
        if let Some(id) = to_select {
            self.select_node(id);
        }
        if let Some((id, is_chain)) = to_copy {
            if is_chain {
                self.copy_filter_chain_from_node(id);
            } else {
                self.copy_filter_from_node(id);
            }
        }
        if let Some((id, is_chain)) = to_paste {
            if is_chain {
                self.paste_filter_chain_to_node(id);
            } else {
                self.paste_filter_to_node(id);
            }
        }
        if let Some(id) = to_delete {
            self.remove_node_and_cleanup(id);
        }

        ui.add_space(6.0);

        // Segmented Inspector Tab Bar
        let has_pending = self.pending_source.is_some() || self.pending_filter.is_some();
        egui::Frame::new()
            .fill(pal.card_bg)
            .stroke(egui::Stroke::new(1.0, pal.card_border))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(3))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let prop_label = if has_pending { "Properties *" } else { "Properties" };
                    ui.selectable_value(&mut self.inspector_tab, InspectorTab::Properties, prop_label);
                    ui.selectable_value(&mut self.inspector_tab, InspectorTab::Display, "Display");
                    ui.selectable_value(
                        &mut self.inspector_tab,
                        InspectorTab::Information,
                        "Info",
                    );
                    ui.selectable_value(&mut self.inspector_tab, InspectorTab::Scene, "Scene");
                });
            });
        ui.add_space(5.0);

        egui::ScrollArea::vertical()
            .id_salt("inspector_scroll")
            .show(ui, |ui| match self.inspector_tab {
                InspectorTab::Properties => self.draw_properties_tab(ui),
                InspectorTab::Display => self.draw_display_tab(ui),
                InspectorTab::Information => self.draw_information_tab(ui),
                InspectorTab::Scene => self.draw_scene_tab(ui),
            });
    }

    fn pipeline_tree_order(&self) -> Vec<PipelineTreeRow> {
        let mut out = Vec::new();
        fn visit(
            nodes: &[paraviz::SessionNode],
            parent: Option<NodeId>,
            depth: usize,
            out: &mut Vec<PipelineTreeRow>,
        ) {
            for n in nodes {
                if n.parent == parent {
                    let badge =
                        node_badge_style(n.dataset().map(|d| d.as_ref()), n.filter_config.is_some());
                    out.push(PipelineTreeRow {
                        id: n.id,
                        depth,
                        name: n.name.clone(),
                        visible: n.style.visible,
                        badge,
                    });
                    visit(nodes, Some(n.id), depth + 1, out);
                }
            }
        }
        visit(self.session.nodes(), None, 0, &mut out);
        out
    }

    fn dispatch_filter_menu_action(&mut self, action: FilterMenuAction) {
        match action {
            FilterMenuAction::Config(cfg) => self.add_filter_to_selected(cfg),
            FilterMenuAction::GpuContour => self.open_gpu_contour_modal(),
            FilterMenuAction::AppendVisible => self.append_visible_datasets(),
        }
    }

    pub(super) fn draw_command_palette(&mut self, ctx: &egui::Context) {
        if !self.command_palette_open {
            return;
        }
        let pal = palette(self.theme);

        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.command_palette_open = false;
            return;
        }

        enum PaletteItemKind {
            Filter(FilterMenuAction),
            Action(&'static str),
        }

        struct PaletteItem {
            label: &'static str,
            category: &'static str,
            score: i32,
            is_fav: bool,
            is_filter: bool,
            enabled: bool,
            kind: PaletteItemKind,
        }

        let has_sel = self.selected_node.is_some();
        let is_imagedata = self
            .selected_node
            .and_then(|id| self.session.node(id).ok())
            .and_then(|n| n.dataset())
            .is_some_and(|ds| ds.as_image_data().is_some());
        let has_visible = self.session.nodes().iter().any(|n| n.style.visible);

        let query = self.command_palette_query.trim().to_ascii_lowercase();
        let mut items: Vec<PaletteItem> = Vec::new();

        for entry in filter_menu_catalog() {
            let label_l = entry.label.to_ascii_lowercase();
            let cat_l = entry.category.to_ascii_lowercase();
            let match_score = if query.is_empty() {
                10
            } else if label_l.starts_with(&query) {
                100
            } else if label_l.contains(&query) {
                60
            } else if cat_l.contains(&query) {
                30
            } else {
                continue;
            };
            let is_fav = self.favorite_filters.contains(entry.label);
            let usage = self.filter_usage_counts.get(entry.label).copied().unwrap_or(0) as i32;
            let profile_bonus = if self.workspace_profile != WorkspaceProfile::Full
                && self.workspace_profile.matches_filter(entry.label)
            {
                25
            } else {
                0
            };
            let enabled = match entry.action {
                FilterMenuAction::Config(_) => has_sel,
                FilterMenuAction::GpuContour => is_imagedata,
                FilterMenuAction::AppendVisible => has_visible,
            };
            items.push(PaletteItem {
                label: entry.label,
                category: entry.category,
                score: match_score + (if is_fav { 50 } else { 0 }) + usage * 5 + profile_bonus,
                is_fav,
                is_filter: true,
                enabled,
                kind: PaletteItemKind::Filter(entry.action),
            });
        }

        let actions: &[(&str, &str)] = &[
            ("Reset Camera", "View"),
            ("Focus Selected Node", "View"),
            ("Toggle Dark / Light Theme", "View"),
            ("Toggle Axes Gizmo", "View"),
            ("Toggle Bounding Box", "View"),
            ("Toggle Eye-Dome Lighting (EDL)", "View"),
            ("Toggle 4x MSAA", "View"),
            ("Toggle Zen Mode", "View"),
            ("Undo Pipeline Action", "Edit"),
            ("Redo Pipeline Action", "Edit"),
            ("Workspace Profile: Full", "Profile"),
            ("Workspace Profile: CFD", "Profile"),
            ("Workspace Profile: Medical", "Profile"),
            ("Workspace Profile: Mesh", "Profile"),
        ];
        for &(label, cat) in actions {
            let label_l = label.to_ascii_lowercase();
            let match_score = if query.is_empty() {
                5
            } else if label_l.starts_with(&query) {
                95
            } else if label_l.contains(&query) {
                55
            } else {
                continue;
            };
            items.push(PaletteItem {
                label,
                category: cat,
                score: match_score,
                is_fav: false,
                is_filter: false,
                enabled: true,
                kind: PaletteItemKind::Action(label),
            });
        }

        items.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.label.cmp(b.label)));
        if !items.is_empty() {
            self.command_palette_index = self.command_palette_index.min(items.len() - 1);
        } else {
            self.command_palette_index = 0;
        }

        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) && !items.is_empty() {
            self.command_palette_index = (self.command_palette_index + 1) % items.len();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !items.is_empty() {
            self.command_palette_index = self
                .command_palette_index
                .checked_sub(1)
                .unwrap_or(items.len() - 1);
        }

        let enter_pressed = ctx.input(|i| i.key_pressed(egui::Key::Enter));
        let mut trigger_idx: Option<usize> = if enter_pressed && !items.is_empty() {
            Some(self.command_palette_index)
        } else {
            None
        };
        let mut toggle_fav: Option<&'static str> = None;

        egui::Window::new("Command Palette")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_width(480.0)
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 72.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("⌘K").color(pal.accent).strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.command_palette_query)
                            .hint_text("Type a filter or action (Up/Down, Enter, Esc)...")
                            .desired_width(400.0),
                    );
                    if self.command_palette_focus_pending {
                        resp.request_focus();
                        self.command_palette_focus_pending = false;
                    }
                    if resp.changed() {
                        self.command_palette_index = 0;
                    }
                    if ui.small_button("×").clicked() {
                        self.command_palette_open = false;
                    }
                });
                ui.separator();

                if items.is_empty() {
                    ui.weak("No matching filters or actions.");
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for (idx, item) in items.iter().enumerate() {
                                let is_sel = idx == self.command_palette_index;
                                ui.horizontal(|ui| {
                                    if item.is_filter {
                                        let star = if item.is_fav { "★" } else { "☆" };
                                        if ui
                                            .small_button(star)
                                            .on_hover_text("Pin/unpin favorite filter")
                                            .clicked()
                                        {
                                            toggle_fav = Some(item.label);
                                        }
                                    } else {
                                        ui.label("  ");
                                    }
                                    let text = egui::RichText::new(item.label).color(if item.enabled {
                                        ui.visuals().text_color()
                                    } else {
                                        pal.muted
                                    });
                                    let btn = ui.selectable_label(is_sel, text);
                                    if btn.clicked() && item.enabled {
                                        trigger_idx = Some(idx);
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(item.category)
                                                    .small()
                                                    .color(pal.muted),
                                            );
                                        },
                                    );
                                });
                            }
                        });
                }
            });

        if let Some(fav) = toggle_fav {
            let key = fav.to_string();
            if !self.favorite_filters.remove(&key) {
                self.favorite_filters.insert(key);
            }
        }

        if let Some(idx) = trigger_idx {
            let item = items.swap_remove(idx);
            if item.enabled {
                self.command_palette_open = false;
                match item.kind {
                    PaletteItemKind::Filter(act) => {
                        *self.filter_usage_counts.entry(item.label.to_string()).or_insert(0) += 1;
                        self.dispatch_filter_menu_action(act);
                    }
                    PaletteItemKind::Action(name) => match name {
                        "Reset Camera" => self.reset_active_pane_camera(),
                        "Focus Selected Node" => self.focus_selected_node(),
                        "Toggle Dark / Light Theme" => {
                            self.theme = match self.theme {
                                UiTheme::DarkStudio => UiTheme::LightStudio,
                                UiTheme::LightStudio => UiTheme::DarkStudio,
                            };
                            self.theme_applied = false;
                        }
                        "Toggle Axes Gizmo" => {
                            self.session.settings.show_axes_gizmo =
                                !self.session.settings.show_axes_gizmo;
                            self.invalidate_viewport();
                        }
                        "Toggle Bounding Box" => {
                            self.session.settings.show_bounding_box =
                                !self.session.settings.show_bounding_box;
                            self.invalidate_viewport();
                        }
                        "Toggle Eye-Dome Lighting (EDL)" => {
                            self.session.settings.edl_enabled = !self.session.settings.edl_enabled;
                            self.invalidate_viewport();
                        }
                        "Toggle 4x MSAA" => {
                            self.session.settings.msaa_samples =
                                if self.session.settings.msaa_samples >= 4 {
                                    1
                                } else {
                                    4
                                };
                            self.invalidate_viewport();
                        }
                        "Toggle Zen Mode" => {
                            self.zen_mode = !self.zen_mode;
                        }
                        "Undo Pipeline Action" => self.undo(),
                        "Redo Pipeline Action" => self.redo(),
                        "Workspace Profile: Full" => self.workspace_profile = WorkspaceProfile::Full,
                        "Workspace Profile: CFD" => self.workspace_profile = WorkspaceProfile::Cfd,
                        "Workspace Profile: Medical" => {
                            self.workspace_profile = WorkspaceProfile::Medical
                        }
                        "Workspace Profile: Mesh" => self.workspace_profile = WorkspaceProfile::Mesh,
                        _ => {}
                    },
                }
            }
        }
    }
}

const FILTER_CATEGORIES: &[&str] = &[
    "Extraction & Slicing",
    "Flow, Vectors & Warp",
    "Mesh & Topology",
    "Fields, Probing & Resampling",
    "Temporal",
];

enum FilterMenuAction {
    Config(FilterConfig),
    GpuContour,
    AppendVisible,
}

struct FilterMenuEntry {
    label: &'static str,
    category: &'static str,
    common: bool,
    action: FilterMenuAction,
}

#[allow(clippy::too_many_arguments)]
fn draw_filter_entries_scroll(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    max_height: f32,
    entries: Vec<FilterMenuEntry>,
    enabled: bool,
    is_imagedata: bool,
    has_visible: bool,
    show_category_hint: bool,
) -> Option<FilterMenuAction> {
    let mut chosen = None;
    egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .max_height(max_height)
        .show(ui, |ui| {
            for entry in entries {
                let is_enabled = match entry.action {
                    FilterMenuAction::Config(_) => enabled,
                    FilterMenuAction::GpuContour => is_imagedata,
                    FilterMenuAction::AppendVisible => has_visible,
                };
                let mut resp = ui.add_enabled(is_enabled, egui::Button::new(entry.label));
                if show_category_hint {
                    resp = resp.on_hover_text(entry.category);
                }
                if resp.clicked() {
                    chosen = Some(entry.action);
                }
            }
        });
    chosen
}

fn filter_menu_catalog() -> Vec<FilterMenuEntry> {
    let f = |label, category, common, cfg: FilterConfig| FilterMenuEntry {
        label,
        category,
        common,
        action: FilterMenuAction::Config(cfg),
    };
    vec![
        // Extraction & Slicing
        f(
            "Extract Surface",
            "Extraction & Slicing",
            true,
            FilterConfig::ExtractSurface(ExtractSurface::new()),
        ),
        f(
            "Contour",
            "Extraction & Slicing",
            true,
            FilterConfig::Contour(ContourFilter::default()),
        ),
        FilterMenuEntry {
            label: "GPU Compute Contour (WebGPU)...",
            category: "Extraction & Slicing",
            common: true,
            action: FilterMenuAction::GpuContour,
        },
        f(
            "Slice",
            "Extraction & Slicing",
            true,
            FilterConfig::Slice(SliceFilter::default()),
        ),
        f(
            "Clip",
            "Extraction & Slicing",
            true,
            FilterConfig::Clip(ClipFilter::default()),
        ),
        f(
            "Threshold",
            "Extraction & Slicing",
            true,
            FilterConfig::Threshold(ThresholdFilter::default()),
        ),
        f(
            "IsoVolume",
            "Extraction & Slicing",
            false,
            FilterConfig::IsoVolume(IsoVolumeFilter::default()),
        ),
        f(
            "Extract Block",
            "Extraction & Slicing",
            false,
            FilterConfig::ExtractBlock(ExtractBlockFilter::default()),
        ),
        // Flow, Vectors & Warp
        f(
            "Stream Tracer",
            "Flow, Vectors & Warp",
            true,
            FilterConfig::StreamTracer(StreamTracerFilter::default()),
        ),
        f(
            "Tube",
            "Flow, Vectors & Warp",
            true,
            FilterConfig::Tube(TubeFilter::default()),
        ),
        f(
            "Glyph",
            "Flow, Vectors & Warp",
            true,
            FilterConfig::Glyph(GlyphFilter::default()),
        ),
        f(
            "Warp By Vector",
            "Flow, Vectors & Warp",
            false,
            FilterConfig::WarpByVector(WarpByVectorFilter::default()),
        ),
        f(
            "Warp By Scalar",
            "Flow, Vectors & Warp",
            false,
            FilterConfig::WarpByScalar(WarpByScalarFilter::default()),
        ),
        f(
            "Gradient / Vorticity",
            "Flow, Vectors & Warp",
            true,
            FilterConfig::Gradient(GradientFilter::default()),
        ),
        // Mesh & Topology
        f(
            "Smooth",
            "Mesh & Topology",
            false,
            FilterConfig::Smooth(SmoothFilter::default()),
        ),
        f(
            "Decimate (QEM)",
            "Mesh & Topology",
            false,
            FilterConfig::Decimate(DecimateFilter::default()),
        ),
        f(
            "Delaunay 2D",
            "Mesh & Topology",
            false,
            FilterConfig::Delaunay2D(Delaunay2DFilter::default()),
        ),
        f(
            "Delaunay 3D",
            "Mesh & Topology",
            false,
            FilterConfig::Delaunay3D(Delaunay3DFilter::default()),
        ),
        f(
            "Triangulate",
            "Mesh & Topology",
            false,
            FilterConfig::Triangulate(TriangulateFilter::default()),
        ),
        f(
            "Clean PolyData",
            "Mesh & Topology",
            false,
            FilterConfig::CleanPolyData(CleanPolyDataFilter::default()),
        ),
        f(
            "Generate Normals",
            "Mesh & Topology",
            false,
            FilterConfig::GenerateNormals(GenerateNormalsFilter::default()),
        ),
        f(
            "Feature Edges",
            "Mesh & Topology",
            false,
            FilterConfig::FeatureEdges(FeatureEdgesFilter::default()),
        ),
        f(
            "Extract All Edges",
            "Mesh & Topology",
            false,
            FilterConfig::ExtractEdges(ExtractEdgesFilter),
        ),
        f(
            "Connectivity",
            "Mesh & Topology",
            false,
            FilterConfig::Connectivity(ConnectivityFilter::default()),
        ),
        f(
            "Shrink",
            "Mesh & Topology",
            false,
            FilterConfig::Shrink(ShrinkFilter::default()),
        ),
        f(
            "Cell Centers",
            "Mesh & Topology",
            false,
            FilterConfig::CellCenters(CellCentersFilter::default()),
        ),
        f(
            "Reflect",
            "Mesh & Topology",
            false,
            FilterConfig::Reflect(ReflectFilter::default()),
        ),
        f(
            "Transform",
            "Mesh & Topology",
            false,
            FilterConfig::Transform(TransformFilter::default()),
        ),
        // Fields, Probing & Resampling
        f(
            "Calculator",
            "Fields, Probing & Resampling",
            true,
            FilterConfig::Calculator(CalculatorFilter::default()),
        ),
        f(
            "Elevation",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::Elevation(ElevationFilter::default()),
        ),
        f(
            "Cell Data to Point Data",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::CellDataToPointData(CellDataToPointDataFilter::default()),
        ),
        f(
            "Point Data to Cell Data",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::PointDataToCellData(PointDataToCellDataFilter::default()),
        ),
        f(
            "Probe Over Line",
            "Fields, Probing & Resampling",
            true,
            FilterConfig::ProbeOverLine(ProbeOverLineFilter::default()),
        ),
        f(
            "Resample To Image",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::ResampleToImage(ResampleToImageFilter::default()),
        ),
        f(
            "Integrate Variables",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::IntegrateVariables(IntegrateVariablesFilter::default()),
        ),
        f(
            "Append Datasets",
            "Fields, Probing & Resampling",
            false,
            FilterConfig::AppendDatasets(AppendDatasetsFilter::default()),
        ),
        FilterMenuEntry {
            label: "Append Visible Datasets",
            category: "Fields, Probing & Resampling",
            common: false,
            action: FilterMenuAction::AppendVisible,
        },
        // Temporal
        f(
            "Temporal Statistics",
            "Temporal",
            false,
            FilterConfig::TemporalStatistics(TemporalStatisticsFilter::default()),
        ),
        f(
            "Add TimeSteps",
            "Temporal",
            false,
            FilterConfig::AddTimeSteps(AddTimeStepsFilter::default()),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn filter_menu_catalog_covers_all_filters_with_valid_categories() {
        let entries = filter_menu_catalog();
        assert_eq!(entries.len(), 39);
        let mut seen = HashSet::new();
        for e in &entries {
            assert!(
                FILTER_CATEGORIES.contains(&e.category),
                "Unknown category '{}' for '{}'",
                e.category,
                e.label
            );
            assert!(seen.insert(e.label), "Duplicate filter label '{}'", e.label);
        }
    }
}
