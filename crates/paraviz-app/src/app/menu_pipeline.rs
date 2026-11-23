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
                    let (w, h) = self.viewport_tex_size;
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
                    let (w, h) = self.viewport_tex_size;
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

            ui.menu_button("Filters", |ui| {
                let enabled = self.selected_node.is_some();
                let is_imagedata = self
                    .selected_node
                    .and_then(|id| self.session.node(id).ok())
                    .and_then(|n| n.dataset())
                    .is_some_and(|ds| ds.as_image_data().is_some());
                let has_visible = self.session.nodes().iter().any(|n| n.style.visible);

                let entries: Vec<(&'static str, FilterConfig)> = vec![
                    (
                        "Extract Surface",
                        FilterConfig::ExtractSurface(ExtractSurface::new()),
                    ),
                    ("Contour", FilterConfig::Contour(ContourFilter::default())),
                    ("Slice", FilterConfig::Slice(SliceFilter::default())),
                    ("Clip", FilterConfig::Clip(ClipFilter::default())),
                    (
                        "Threshold",
                        FilterConfig::Threshold(ThresholdFilter::default()),
                    ),
                    (
                        "Stream Tracer",
                        FilterConfig::StreamTracer(StreamTracerFilter::default()),
                    ),
                    ("Tube", FilterConfig::Tube(TubeFilter::default())),
                    ("Glyph", FilterConfig::Glyph(GlyphFilter::default())),
                    (
                        "Warp By Vector",
                        FilterConfig::WarpByVector(WarpByVectorFilter::default()),
                    ),
                    (
                        "Warp By Scalar",
                        FilterConfig::WarpByScalar(WarpByScalarFilter::default()),
                    ),
                    (
                        "Gradient / Vorticity",
                        FilterConfig::Gradient(GradientFilter::default()),
                    ),
                    (
                        "Calculator",
                        FilterConfig::Calculator(CalculatorFilter::default()),
                    ),
                    ("Smooth", FilterConfig::Smooth(SmoothFilter::default())),
                    (
                        "Transform",
                        FilterConfig::Transform(TransformFilter::default()),
                    ),
                    (
                        "Cell Data to Point Data",
                        FilterConfig::CellDataToPointData(CellDataToPointDataFilter::default()),
                    ),
                    (
                        "Point Data to Cell Data",
                        FilterConfig::PointDataToCellData(PointDataToCellDataFilter::default()),
                    ),
                    (
                        "Probe Over Line",
                        FilterConfig::ProbeOverLine(ProbeOverLineFilter::default()),
                    ),
                    (
                        "Feature Edges",
                        FilterConfig::FeatureEdges(FeatureEdgesFilter::default()),
                    ),
                    (
                        "Extract All Edges",
                        FilterConfig::ExtractEdges(ExtractEdgesFilter),
                    ),
                    (
                        "Connectivity",
                        FilterConfig::Connectivity(ConnectivityFilter::default()),
                    ),
                    ("Shrink", FilterConfig::Shrink(ShrinkFilter::default())),
                    (
                        "Cell Centers",
                        FilterConfig::CellCenters(CellCentersFilter::default()),
                    ),
                    ("Reflect", FilterConfig::Reflect(ReflectFilter::default())),
                    (
                        "Generate Normals",
                        FilterConfig::GenerateNormals(GenerateNormalsFilter::default()),
                    ),
                    (
                        "Triangulate",
                        FilterConfig::Triangulate(TriangulateFilter::default()),
                    ),
                    (
                        "Clean PolyData",
                        FilterConfig::CleanPolyData(CleanPolyDataFilter::default()),
                    ),
                    (
                        "Elevation",
                        FilterConfig::Elevation(ElevationFilter::default()),
                    ),
                    (
                        "Decimate (QEM)",
                        FilterConfig::Decimate(DecimateFilter::default()),
                    ),
                    (
                        "Delaunay 2D",
                        FilterConfig::Delaunay2D(Delaunay2DFilter::default()),
                    ),
                    (
                        "Delaunay 3D",
                        FilterConfig::Delaunay3D(Delaunay3DFilter::default()),
                    ),
                    (
                        "IsoVolume",
                        FilterConfig::IsoVolume(IsoVolumeFilter::default()),
                    ),
                    (
                        "Resample To Image",
                        FilterConfig::ResampleToImage(ResampleToImageFilter::default()),
                    ),
                    (
                        "Integrate Variables",
                        FilterConfig::IntegrateVariables(IntegrateVariablesFilter::default()),
                    ),
                    (
                        "Extract Block",
                        FilterConfig::ExtractBlock(ExtractBlockFilter::default()),
                    ),
                    (
                        "Temporal Statistics",
                        FilterConfig::TemporalStatistics(TemporalStatisticsFilter::default()),
                    ),
                    (
                        "Add TimeSteps",
                        FilterConfig::AddTimeSteps(AddTimeStepsFilter::default()),
                    ),
                    (
                        "Append Datasets",
                        FilterConfig::AppendDatasets(AppendDatasetsFilter::default()),
                    ),
                ];
                for (name, cfg) in entries {
                    if ui.add_enabled(enabled, egui::Button::new(name)).clicked() {
                        self.add_filter_to_selected(cfg);
                        ui.close();
                    }
                }
                ui.separator();
                if ui
                    .add_enabled(
                        is_imagedata,
                        egui::Button::new("GPU Compute Contour (WebGPU)..."),
                    )
                    .clicked()
                {
                    self.open_gpu_contour_modal();
                    ui.close();
                }
                if ui
                    .add_enabled(has_visible, egui::Button::new("Append Visible Datasets"))
                    .clicked()
                {
                    self.append_visible_datasets();
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
                ui.checkbox(&mut self.show_scalar_bar, "Scalar Color Bar");
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
            // Camera group
            egui::Frame::new()
                .fill(pal.card_bg)
                .stroke(egui::Stroke::new(1.0, pal.card_border))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(6, 3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.small_button("Reset").on_hover_text("Reset camera (R)").clicked() {
                            let _ = self.session.reset_camera();
                            self.invalidate_viewport();
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
                                let _ = self.session.camera_preset(preset);
                                self.invalidate_viewport();
                            }
                        }
                        if ui
                            .toggle_value(&mut self.session.camera.orthographic, "Ortho")
                            .changed()
                        {
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
            let removed = self.session.remove_node(id);
            self.info_stats_cache.retain(|(nid, _, _), _| !removed.contains(nid));
            self.volume_tf_hist_cache
                .retain(|(nid, _, _, _), _| !removed.contains(nid));
            if self
                .picked_highlight
                .as_ref()
                .is_some_and(|h| removed.contains(&h.node_id))
            {
                self.picked_highlight = None;
            }
            if self.selected_node.is_some_and(|s| removed.contains(&s)) {
                self.selected_node = self.session.nodes().last().map(|n| n.id);
                self.pending_source = None;
                self.pending_filter = None;
            }
            self.invalidate_viewport();
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
}
