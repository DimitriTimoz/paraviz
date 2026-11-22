use crate::cli::format_memory_bytes;
use paraviz::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Properties,
    Display,
    Information,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawerTab {
    Spreadsheet,
    RustTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpreadsheetTarget {
    Points,
    Cells,
}

#[derive(Debug, Clone)]
enum PathModal {
    OpenFile(String),
    SaveState(String),
    LoadState(String),
    ExportDataset(String),
    SaveScreenshot(String),
}

pub struct ParaVizApp {
    session: ParaViz,
    renderer: WebGpuRenderer,
    device: wgpu::Device,
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    viewport_tex_id: Option<egui::TextureId>,
    viewport_tex_size: (u32, u32),

    selected_node: Option<NodeId>,
    auto_apply: bool,
    pending_source: Option<(NodeId, SourceConfig)>,
    pending_filter: Option<(NodeId, FilterConfig)>,

    inspector_tab: InspectorTab,
    show_drawer: bool,
    drawer_tab: DrawerTab,
    spreadsheet_target: SpreadsheetTarget,
    show_scalar_bar: bool,

    playing: bool,
    time_accumulator: f64,
    play_fps: f64,

    last_frame_time: Instant,
    fps_smoothed: f32,
    last_render_ms: f32,

    modal: Option<PathModal>,
    status_message: String,
    error_message: Option<String>,
}

impl ParaVizApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be configured with the WGPU backend");
        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let egui_renderer = rs.renderer.clone();

        let renderer = WebGpuRenderer::from_device(
            Arc::new(device.clone()),
            Arc::new(queue),
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let mut app = Self {
            session: ParaViz::new(),
            renderer,
            device,
            egui_renderer,
            viewport_tex_id: None,
            viewport_tex_size: (0, 0),
            selected_node: None,
            auto_apply: true,
            pending_source: None,
            pending_filter: None,
            inspector_tab: InspectorTab::Properties,
            show_drawer: false,
            drawer_tab: DrawerTab::Spreadsheet,
            spreadsheet_target: SpreadsheetTarget::Points,
            show_scalar_bar: true,
            playing: false,
            time_accumulator: 0.0,
            play_fps: 4.0,
            last_frame_time: Instant::now(),
            fps_smoothed: 60.0,
            last_render_ms: 0.0,
            modal: None,
            status_message: "Ready".into(),
            error_message: None,
        };

        if let Some(path) = initial_path {
            app.load_path(&path);
        } else {
            app.load_demo(DemoPreset::WaveletContour);
        }

        app
    }

    fn load_demo(&mut self, preset: DemoPreset) {
        match preset.build() {
            Ok(mut pv) => {
                let _ = pv.update_visible();
                self.selected_node = pv.nodes().last().map(|n| n.id);
                self.session = pv;
                self.pending_source = None;
                self.pending_filter = None;
                self.playing = false;
                self.error_message = None;
                self.status_message = format!("Loaded demo: {}", preset.name());
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load demo: {e}"));
            }
        }
    }

    fn load_path(&mut self, path: &Path) {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "json" {
            match std::fs::read_to_string(path).and_then(|s| {
                ParaViz::load_state_json(&s).map_err(|e| std::io::Error::other(e.to_string()))
            }) {
                Ok(mut pv) => {
                    let _ = pv.update_visible();
                    self.selected_node = pv.nodes().last().map(|n| n.id);
                    self.session = pv;
                    self.pending_source = None;
                    self.pending_filter = None;
                    self.playing = false;
                    self.error_message = None;
                    self.status_message = format!("Loaded state: {}", path.display());
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to load state: {e}"));
                }
            }
            return;
        }

        let is_first = self.session.nodes().is_empty();
        match self.session.open(path) {
            Ok(id) => {
                let _ = self.session.evaluate(id);
                self.session.show(id);
                if is_first {
                    let _ = self.session.camera_preset(CameraPreset::Isometric);
                } else {
                    let _ = self.session.reset_camera();
                }
                self.selected_node = Some(id);
                self.pending_source = None;
                self.pending_filter = None;
                self.error_message = None;
                self.status_message = format!("Opened '{}'", path.display());
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to open '{}': {e}", path.display()));
            }
        }
    }

    fn add_source(&mut self, cfg: SourceConfig) {
        let is_first = self.session.nodes().is_empty();
        let label = self.unique_label(&cfg.default_label());
        match self.session.add_source_cfg(label.clone(), cfg) {
            Ok(id) => {
                let _ = self.session.evaluate(id);
                self.session.show(id);
                if is_first {
                    let _ = self.session.camera_preset(CameraPreset::Isometric);
                } else {
                    let _ = self.session.reset_camera();
                }
                self.selected_node = Some(id);
                self.pending_source = None;
                self.pending_filter = None;
                self.error_message = None;
                self.status_message = format!("Added source '{label}'");
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to add source: {e}"));
            }
        }
    }

    fn add_filter_to_selected(&mut self, mut cfg: FilterConfig) {
        let Some(parent_id) = self.selected_node else {
            self.error_message = Some("Select a pipeline node first to apply a filter.".into());
            return;
        };

        // Populate smart defaults from the parent's dataset.
        if let Ok(ds) = self.session.evaluate(parent_id) {
            let bb = ds.bounds();
            let center = bb.center().to_array();
            let active_scalar = ds
                .point_data()
                .and_then(|p| p.active_scalars_array())
                .map(|a| (a.name.clone(), a.range(None)));
            let active_vector = ds
                .point_data()
                .and_then(|p| p.active_vectors_array())
                .map(|a| a.name.clone());

            match &mut cfg {
                FilterConfig::Contour(f) => {
                    if let Some((name, (mn, mx))) = active_scalar {
                        f.field_name = name;
                        f.iso_values = vec![(mn + mx) * 0.5];
                    }
                }
                FilterConfig::Slice(f) => {
                    f.origin = center;
                }
                FilterConfig::Clip(f) => {
                    f.function = ImplicitFunction::Plane {
                        origin: center,
                        normal: [1.0, 0.0, 0.0],
                    };
                }
                FilterConfig::Threshold(f) => {
                    if let Some((name, (mn, mx))) = active_scalar {
                        f.field_name = name;
                        f.min = mn + (mx - mn) * 0.25;
                        f.max = mx;
                    }
                }
                FilterConfig::StreamTracer(f) => {
                    if let Some(vname) = active_vector {
                        f.vector_field = vname;
                    }
                    let r = (bb.diagonal() * 0.25).max(0.2);
                    f.seeds = SeedSource::Sphere {
                        center,
                        radius: r,
                        count: 16,
                    };
                }
                FilterConfig::Glyph(f) => {
                    f.vector_field = active_vector;
                    f.scalar_field = active_scalar.map(|(n, _)| n);
                }
                FilterConfig::WarpByVector(f) => {
                    if let Some(vname) = active_vector {
                        f.vector_field = vname;
                    }
                }
                FilterConfig::WarpByScalar(f) => {
                    if let Some((name, _)) = active_scalar {
                        f.scalar_field = name;
                    }
                }
                FilterConfig::Gradient(f) => {
                    if let Some(vname) = active_vector {
                        f.field_name = vname;
                    } else if let Some((name, _)) = active_scalar {
                        f.field_name = name;
                    }
                }
                FilterConfig::Calculator(f) => {
                    if let Some((name, _)) = active_scalar {
                        f.expression = format!("{name} * 2.0");
                    }
                }
                FilterConfig::ProbeOverLine(f)
                    if bb.is_valid() => {
                        f.point1 = [bb.min.x, center[1], center[2]];
                        f.point2 = [bb.max.x, center[1], center[2]];
                    }
                _ => {}
            }
        }

        let label = self.unique_label(cfg.default_label());
        match self.session.add_filter_cfg(label.clone(), parent_id, cfg) {
            Ok(id) => {
                let eval_res = self.session.evaluate(id);
                self.session.show(id);
                if let Err(e) = eval_res {
                    self.error_message = Some(format!("Filter '{label}' failed: {e}"));
                } else {
                    self.error_message = None;
                    self.status_message = format!("Applied filter '{label}'");
                }
                self.selected_node = Some(id);
                self.pending_source = None;
                self.pending_filter = None;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to add filter: {e}"));
            }
        }
    }

    fn unique_label(&self, base: &str) -> String {
        if !self.session.nodes().iter().any(|n| n.name == base) {
            return base.to_string();
        }
        let mut idx = 2;
        loop {
            let candidate = format!("{base}_{idx}");
            if !self.session.nodes().iter().any(|n| n.name == candidate) {
                return candidate;
            }
            idx += 1;
        }
    }

    fn select_node(&mut self, id: NodeId) {
        if self.selected_node != Some(id) {
            self.selected_node = Some(id);
            self.pending_source = None;
            self.pending_filter = None;
        }
    }

    fn step_time(&mut self, delta_steps: isize) {
        let steps = self.session.all_time_steps();
        if steps.is_empty() {
            return;
        }
        let cur_idx = steps
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - self.session.time)
                    .abs()
                    .total_cmp(&(*b - self.session.time).abs())
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        let next_idx =
            (cur_idx as isize + delta_steps).rem_euclid(steps.len() as isize) as usize;
        self.session.set_time(steps[next_idx]);
    }

    fn handle_shortcuts_and_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        for path in dropped {
            self.load_path(&path);
        }

        if !ctx.egui_wants_keyboard_input() && self.modal.is_none() {
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                let _ = self.session.reset_camera();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space))
                && !self.session.all_time_steps().is_empty()
            {
                self.playing = !self.playing;
            }
        }
    }

    fn draw_menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open File...").clicked() {
                    self.modal = Some(PathModal::OpenFile(String::new()));
                    ui.close();
                }
                ui.menu_button("Demo Scenes", |ui| {
                    for &preset in DemoPreset::ALL {
                        if ui.button(preset.name()).clicked() {
                            self.load_demo(preset);
                            ui.close();
                        }
                    }
                });
                ui.separator();
                if ui.button("Save State JSON...").clicked() {
                    self.modal = Some(PathModal::SaveState("paraviz_state.json".into()));
                    ui.close();
                }
                if ui.button("Load State JSON...").clicked() {
                    self.modal = Some(PathModal::LoadState("paraviz_state.json".into()));
                    ui.close();
                }
                ui.separator();
                let has_sel = self.selected_node.is_some();
                if ui
                    .add_enabled(has_sel, egui::Button::new("Export Active Dataset..."))
                    .clicked()
                {
                    self.modal = Some(PathModal::ExportDataset("export.vtp".into()));
                    ui.close();
                }
                if ui.button("Save Screenshot PNG...").clicked() {
                    self.modal = Some(PathModal::SaveScreenshot("screenshot.png".into()));
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
                ];
                for (name, cfg) in entries {
                    if ui.add_enabled(enabled, egui::Button::new(name)).clicked() {
                        self.add_filter_to_selected(cfg);
                        ui.close();
                    }
                }
            });

            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.show_drawer, "Inspector Drawer (Spreadsheet / Trace)");
                ui.checkbox(&mut self.show_scalar_bar, "Scalar Color Bar");
                ui.checkbox(&mut self.session.settings.show_axes_gizmo, "Axes Gizmo");
                ui.checkbox(
                    &mut self.session.settings.show_bounding_box,
                    "Bounding Box",
                );
                ui.checkbox(&mut self.session.settings.edl_enabled, "Eye-Dome Lighting");
            });
        });
    }

    fn draw_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Reset (R)").clicked() {
                let _ = self.session.reset_camera();
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
                if ui.button(label).clicked() {
                    let _ = self.session.camera_preset(preset);
                }
            }
            ui.separator();
            ui.toggle_value(&mut self.session.camera.orthographic, "Ortho");
            ui.toggle_value(&mut self.session.settings.show_axes_gizmo, "Axes");
            ui.toggle_value(&mut self.session.settings.show_bounding_box, "Bounds");
            ui.toggle_value(&mut self.session.settings.edl_enabled, "EDL");
            ui.toggle_value(&mut self.show_scalar_bar, "ScalarBar");
            ui.separator();
            ui.toggle_value(&mut self.show_drawer, "Spreadsheet / Trace");

            let steps = self.session.all_time_steps();
            if !steps.is_empty() {
                ui.separator();
                ui.label("Time:");
                if ui.button("|<").clicked() {
                    self.session.set_time(steps[0]);
                }
                if ui.button("<").clicked() {
                    self.step_time(-1);
                }
                let play_label = if self.playing { "Pause" } else { "Play" };
                if ui.button(play_label).clicked() {
                    self.playing = !self.playing;
                }
                if ui.button(">").clicked() {
                    self.step_time(1);
                }
                if ui.button(">|").clicked() {
                    self.session.set_time(*steps.last().unwrap());
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
                    }
                } else {
                    ui.label(format!("{t:.3}"));
                }
            }
        });
    }

    fn draw_left_dock(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pipeline Browser");
        ui.separator();

        // Build ordered tree list (roots then recursive children).
        let ordered = self.pipeline_tree_order();
        let mut toggle_vis: Option<(NodeId, bool)> = None;
        let mut to_select: Option<NodeId> = None;
        let mut to_delete: Option<NodeId> = None;

        egui::ScrollArea::vertical()
            .id_salt("pipeline_browser_scroll")
            .max_height(190.0)
            .show(ui, |ui| {
                if ordered.is_empty() {
                    ui.weak("Pipeline is empty. Use Sources or File -> Open.");
                }
                for (id, depth, name, visible) in ordered {
                    ui.horizontal(|ui| {
                        if depth > 0 {
                            ui.add_space(depth as f32 * 14.0);
                            ui.weak("└");
                        }
                        let eye = if visible { "👁" } else { "○" };
                        if ui.small_button(eye).on_hover_text("Toggle visibility").clicked() {
                            toggle_vis = Some((id, !visible));
                        }
                        let selected = self.selected_node == Some(id);
                        if ui.selectable_label(selected, &name).clicked() {
                            to_select = Some(id);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✕").on_hover_text("Delete node").clicked() {
                                to_delete = Some(id);
                            }
                        });
                    });
                }
            });

        if let Some((id, vis)) = toggle_vis {
            if vis {
                self.session.show(id);
            } else {
                self.session.hide(id);
            }
        }
        if let Some(id) = to_select {
            self.select_node(id);
        }
        if let Some(id) = to_delete {
            let removed = self.session.remove_node(id);
            if self.selected_node.is_some_and(|s| removed.contains(&s)) {
                self.selected_node = self.session.nodes().last().map(|n| n.id);
                self.pending_source = None;
                self.pending_filter = None;
            }
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.inspector_tab,
                InspectorTab::Properties,
                "Properties",
            );
            ui.selectable_value(&mut self.inspector_tab, InspectorTab::Display, "Display");
            ui.selectable_value(
                &mut self.inspector_tab,
                InspectorTab::Information,
                "Information",
            );
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .id_salt("inspector_scroll")
            .show(ui, |ui| match self.inspector_tab {
                InspectorTab::Properties => self.draw_properties_tab(ui),
                InspectorTab::Display => self.draw_display_tab(ui),
                InspectorTab::Information => self.draw_information_tab(ui),
            });
    }

    fn pipeline_tree_order(&self) -> Vec<(NodeId, usize, String, bool)> {
        let mut out = Vec::new();
        fn visit(
            nodes: &[paraviz::SessionNode],
            parent: Option<NodeId>,
            depth: usize,
            out: &mut Vec<(NodeId, usize, String, bool)>,
        ) {
            for n in nodes {
                if n.parent == parent {
                    out.push((n.id, depth, n.name.clone(), n.style.visible));
                    visit(nodes, Some(n.id), depth + 1, out);
                }
            }
        }
        visit(self.session.nodes(), None, 0, &mut out);
        out
    }

    fn draw_properties_tab(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to edit its properties.");
            return;
        };

        let (src_cfg, flt_cfg) = match self.session.node(sel_id) {
            Ok(n) => (n.source_config.clone(), n.filter_config.clone()),
            Err(_) => return,
        };

        let mut apply_clicked = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_apply, "Auto-Apply");
            if ui.button("Apply").clicked() {
                apply_clicked = true;
            }
        });
        ui.separator();

        if let Some(orig_src) = src_cfg {
            let mut working = match &self.pending_source {
                Some((id, cfg)) if *id == sel_id => cfg.clone(),
                _ => orig_src,
            };
            let changed = edit_source_config(ui, &mut working);
            if changed {
                if self.auto_apply {
                    if let Err(e) = self.session.update_source_cfg(sel_id, working) {
                        self.error_message = Some(format!("Update source error: {e}"));
                    } else {
                        self.pending_source = None;
                        self.error_message = None;
                    }
                } else {
                    self.pending_source = Some((sel_id, working));
                }
            } else if apply_clicked
                && let Some((id, pending)) = self.pending_source.take()
                    && id == sel_id {
                        if let Err(e) = self.session.update_source_cfg(sel_id, pending) {
                            self.error_message = Some(format!("Update source error: {e}"));
                        } else {
                            self.error_message = None;
                        }
                    }
        } else if let Some(orig_flt) = flt_cfg {
            let mut working = match &self.pending_filter {
                Some((id, cfg)) if *id == sel_id => cfg.clone(),
                _ => orig_flt,
            };
            let changed = edit_filter_config(ui, &mut working);
            if changed {
                if self.auto_apply {
                    if let Err(e) = self.session.update_filter_cfg(sel_id, working) {
                        self.error_message = Some(format!("Update filter error: {e}"));
                    } else {
                        self.pending_filter = None;
                        self.error_message = None;
                    }
                } else {
                    self.pending_filter = Some((sel_id, working));
                }
            } else if apply_clicked
                && let Some((id, pending)) = self.pending_filter.take()
                    && id == sel_id {
                        if let Err(e) = self.session.update_filter_cfg(sel_id, pending) {
                            self.error_message = Some(format!("Update filter error: {e}"));
                        } else {
                            self.error_message = None;
                        }
                    }
        } else {
            ui.weak("Custom in-memory node (no editable config).");
        }
    }

    fn draw_display_tab(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to configure display styling.");
            return;
        };

        let _ = self.session.evaluate(sel_id);
        let Ok(node) = self.session.node_mut(sel_id) else {
            return;
        };

        let ds_opt = node.cached_dataset.clone();
        let style = &mut node.style;

        ui.checkbox(&mut style.visible, "Visible");

        // Representation
        egui::ComboBox::from_label("Representation")
            .selected_text(format!("{:?}", style.representation))
            .show_ui(ui, |ui| {
                for r in [
                    Representation::Surface,
                    Representation::SurfaceWithEdges,
                    Representation::Wireframe,
                    Representation::Points,
                    Representation::Volume,
                    Representation::Outline,
                ] {
                    ui.selectable_value(&mut style.representation, r, format!("{r:?}"));
                }
            });

        // Color By
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
        ui.label("Colormap");
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

        // Live colormap gradient preview strip
        paint_colormap_preview(ui, &style.colormap, 18.0);

        if ui
            .checkbox(&mut style.colormap.log_scale, "Log Scale")
            .changed()
        {
            style.volume_tf.color_map.log_scale = style.colormap.log_scale;
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut style.auto_range, "Auto-Range");
            if ui.button("Rescale to Data").clicked()
                && let Some((mn, mx)) = chosen_array_range {
                    style.scalar_range = (mn, mx);
                    style.volume_tf.scalar_range = (mn, mx);
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

        ui.separator();
        ui.add(egui::Slider::new(&mut style.opacity, 0.0..=1.0).text("Opacity"));
        ui.add(egui::Slider::new(&mut style.point_size, 1.0..=24.0).text("Point Size"));
        ui.add(egui::Slider::new(&mut style.line_width, 0.5..=10.0).text("Line Width"));
        ui.add(egui::Slider::new(&mut style.edge_width, 0.25..=6.0).text("Edge Width"));
        ui.horizontal(|ui| {
            ui.label("Edge Color:");
            ui.color_edit_button_rgba_unmultiplied(&mut style.edge_color);
        });

        if style.representation == Representation::Volume {
            ui.separator();
            ui.label("Volume Transfer Function");
            ui.add(
                egui::Slider::new(&mut style.volume_step_size, 0.05..=2.0).text("Step Size"),
            );
            ui.checkbox(&mut style.volume_shading, "Volume Shading");
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
                    if ui.small_button("✕").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(idx) = remove_idx
                && style.volume_tf.opacity_points.len() > 2 {
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
        }
    }

    fn draw_information_tab(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to inspect dataset metadata.");
            return;
        };
        let _ = self.session.evaluate(sel_id);
        let Ok(node) = self.session.node(sel_id) else {
            return;
        };
        let Some(ds) = node.dataset() else {
            ui.weak("Dataset not evaluated yet.");
            return;
        };

        ui.label(format!("Name: {}", node.name));
        ui.label(format!("Type: {}", ds.type_name()));
        ui.label(format!("Points: {}", ds.num_points()));
        ui.label(format!("Cells: {}", ds.num_cells()));
        ui.label(format!(
            "Memory: {}",
            format_memory_bytes(ds.memory_bytes())
        ));
        let bb = ds.bounds();
        if bb.is_valid() {
            ui.separator();
            ui.label("Bounds:");
            ui.monospace(format!("X: [{:.4}, {:.4}]", bb.min.x, bb.max.x));
            ui.monospace(format!("Y: [{:.4}, {:.4}]", bb.min.y, bb.max.y));
            ui.monospace(format!("Z: [{:.4}, {:.4}]", bb.min.z, bb.max.z));
        }

        if let Some(pd) = ds.point_data() {
            ui.separator();
            ui.strong(format!("Point Arrays ({})", pd.len()));
            for arr in pd.arrays() {
                let (mn, mx) = arr.range(None);
                ui.monospace(format!(
                    "{} ({}, {}c) [{:.4}, {:.4}]",
                    arr.name,
                    arr.data.scalar_type().vtk_name(),
                    arr.num_components,
                    mn,
                    mx
                ));
            }
        }
        if let Some(cd) = ds.cell_data() {
            ui.separator();
            ui.strong(format!("Cell Arrays ({})", cd.len()));
            for arr in cd.arrays() {
                let (mn, mx) = arr.range(None);
                ui.monospace(format!(
                    "{} ({}, {}c) [{:.4}, {:.4}]",
                    arr.name,
                    arr.data.scalar_type().vtk_name(),
                    arr.num_components,
                    mn,
                    mx
                ));
            }
        }
    }

    fn draw_drawer(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::Spreadsheet, "Spreadsheet");
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::RustTrace, "Rust Trace");
            if self.drawer_tab == DrawerTab::Spreadsheet {
                ui.separator();
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Points,
                    "Point Data",
                );
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Cells,
                    "Cell Data",
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Close").clicked() {
                    self.show_drawer = false;
                }
            });
        });
        ui.separator();

        match self.drawer_tab {
            DrawerTab::Spreadsheet => self.draw_spreadsheet(ui),
            DrawerTab::RustTrace => {
                let mut trace = self.session.to_rust_trace();
                ui.horizontal(|ui| {
                    if ui.button("Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(trace.clone());
                        self.status_message = "Copied Rust trace to clipboard".into();
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("rust_trace_scroll")
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut trace)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY),
                        );
                    });
            }
        }
    }

    fn draw_spreadsheet(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to inspect data values.");
            return;
        };
        let Ok(node) = self.session.node(sel_id) else {
            return;
        };
        let Some(ds) = node.dataset() else {
            ui.weak("Dataset not evaluated.");
            return;
        };

        let pts = extract_dataset_points(ds);
        let max_rows = 200usize;

        egui::ScrollArea::both()
            .id_salt("spreadsheet_scroll")
            .show(ui, |ui| {
                match self.spreadsheet_target {
                    SpreadsheetTarget::Points => {
                        let arrays = ds.point_data().map(|p| p.arrays()).unwrap_or(&[]);
                        let count = ds.num_points().min(max_rows);
                        egui::Grid::new("spreadsheet_points_grid")
                            .striped(true)
                            .min_col_width(80.0)
                            .show(ui, |ui| {
                                ui.strong("Point ID");
                                ui.strong("X");
                                ui.strong("Y");
                                ui.strong("Z");
                                for arr in arrays {
                                    ui.strong(&arr.name);
                                }
                                ui.end_row();

                                for row in 0..count {
                                    ui.monospace(format!("{row}"));
                                    if let Some(p) = pts.get(row) {
                                        ui.monospace(format!("{:.4}", p[0]));
                                        ui.monospace(format!("{:.4}", p[1]));
                                        ui.monospace(format!("{:.4}", p[2]));
                                    } else {
                                        ui.monospace("-");
                                        ui.monospace("-");
                                        ui.monospace("-");
                                    }
                                    for arr in arrays {
                                        ui.monospace(format_tuple(arr, row));
                                    }
                                    ui.end_row();
                                }
                            });
                    }
                    SpreadsheetTarget::Cells => {
                        let arrays = ds.cell_data().map(|c| c.arrays()).unwrap_or(&[]);
                        let count = ds.num_cells().min(max_rows);
                        if arrays.is_empty() {
                            ui.weak("Selected dataset has no CellData arrays.");
                            return;
                        }
                        egui::Grid::new("spreadsheet_cells_grid")
                            .striped(true)
                            .min_col_width(80.0)
                            .show(ui, |ui| {
                                ui.strong("Cell ID");
                                for arr in arrays {
                                    ui.strong(&arr.name);
                                }
                                ui.end_row();
                                for row in 0..count {
                                    ui.monospace(format!("{row}"));
                                    for arr in arrays {
                                        ui.monospace(format_tuple(arr, row));
                                    }
                                    ui.end_row();
                                }
                            });
                    }
                }
            });
    }

    fn draw_viewport(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let ppp = ui.ctx().pixels_per_point();
        let w = (avail.x * ppp).round().max(1.0) as u32;
        let h = (avail.y * ppp).round().max(1.0) as u32;

        let t0 = Instant::now();
        let scene_res = self.session.build_scene();
        let mut total_points = 0usize;
        let mut total_cells = 0usize;
        if let Ok(scene) = &scene_res {
            for a in &scene.actors {
                total_points += a.dataset.num_points();
                total_cells += a.dataset.num_cells();
            }
            match self.renderer.render_to_texture(scene, w, h) {
                Ok(view) => {
                    let mut egui_r = self.egui_renderer.write();
                    match self.viewport_tex_id {
                        None => {
                            let id = egui_r.register_native_texture(
                                &self.device,
                                view,
                                wgpu::FilterMode::Linear,
                            );
                            self.viewport_tex_id = Some(id);
                            self.viewport_tex_size = (w, h);
                        }
                        Some(id) if self.viewport_tex_size != (w, h) => {
                            egui_r.update_egui_texture_from_wgpu_texture(
                                &self.device,
                                view,
                                wgpu::FilterMode::Linear,
                                id,
                            );
                            self.viewport_tex_size = (w, h);
                        }
                        Some(_) => {}
                    }
                }
                Err(e) => {
                    self.error_message = Some(format!("Render error: {e}"));
                }
            }
        } else if let Err(e) = &scene_res {
            self.error_message = Some(format!("Pipeline error: {e}"));
        }
        self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let rect = ui.available_rect_before_wrap();
        if let Some(tex_id) = self.viewport_tex_id {
            let img = egui::Image::new(egui::load::SizedTexture::new(tex_id, avail))
                .sense(egui::Sense::click_and_drag());
            let resp = ui.add(img);

            let shift = ui.input(|i| i.modifiers.shift);
            if resp.dragged_by(egui::PointerButton::Primary) {
                let d = resp.drag_delta();
                if shift {
                    self.session
                        .camera
                        .pan(d.x / avail.y.max(1.0), d.y / avail.y.max(1.0));
                } else {
                    self.session.camera.orbit(-d.x * 0.008, -d.y * 0.008);
                }
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.session
                    .camera
                    .pan(d.x / avail.y.max(1.0), d.y / avail.y.max(1.0));
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Secondary) {
                let d = resp.drag_delta();
                self.session.camera.dolly(-d.y * 0.008);
                ui.ctx().request_repaint();
            }

            if resp.hovered() {
                let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_y.abs() > 0.01 {
                    self.session.camera.dolly(scroll_y * 0.0025);
                    ui.ctx().request_repaint();
                }
            }

            if resp.double_clicked() {
                let _ = self.session.reset_camera();
                ui.ctx().request_repaint();
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Initializing WebGPU viewport...");
            });
        }

        // HUD Overlay (top-left of viewport)
        let proj_mode = if self.session.camera.orthographic {
            "Ortho"
        } else {
            "Persp"
        };
        let hud_text = format!(
            "WebGPU | {:.1} FPS ({:.2} ms) | {} pts | {} cells | {}",
            self.fps_smoothed, self.last_render_ms, total_points, total_cells, proj_mode
        );
        let painter = ui.painter_at(rect);
        let hud_pos = rect.left_top() + egui::vec2(12.0, 10.0);
        let galley = painter.layout_no_wrap(
            hud_text,
            egui::FontId::monospace(11.5),
            egui::Color32::from_rgb(230, 236, 245),
        );
        let hud_bg = egui::Rect::from_min_size(
            hud_pos - egui::vec2(6.0, 4.0),
            galley.size() + egui::vec2(12.0, 8.0),
        );
        painter.rect_filled(hud_bg, 4.0, egui::Color32::from_rgba_unmultiplied(15, 18, 26, 175));
        painter.galley(hud_pos, galley, egui::Color32::WHITE);

        // Scalar Color Bar Overlay (bottom-right of viewport)
        if self.show_scalar_bar
            && let Some((title, cmap, range)) = self.active_scalar_bar_info() {
                paint_scalar_bar_overlay(&painter, rect, &title, &cmap, range);
            }

        // Drop target indicator
        if ui.input(|i| !i.raw.hovered_files.is_empty()) {
            painter.rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(30, 110, 220, 45),
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop scientific file or .json state to open",
                egui::FontId::proportional(20.0),
                egui::Color32::WHITE,
            );
        }
    }

    fn active_scalar_bar_info(&self) -> Option<(String, ColorMap, (f64, f64))> {
        let pick_node = self
            .selected_node
            .and_then(|id| self.session.node(id).ok())
            .filter(|n| n.style.visible && !matches!(n.style.color_by, ColorBy::Solid(_)))
            .or_else(|| {
                self.session
                    .nodes()
                    .iter()
                    .rev()
                    .find(|n| n.style.visible && !matches!(n.style.color_by, ColorBy::Solid(_)))
            })?;

        let (title, range) = match &pick_node.style.color_by {
            ColorBy::PointScalar { array, component }
            | ColorBy::CellScalar { array, component } => {
                let comp_suffix = component.map(|c| format!("[{c}]")).unwrap_or_default();
                let title = format!("{array}{comp_suffix}");
                let range = if pick_node.style.auto_range {
                    pick_node
                        .dataset()
                        .and_then(|ds| {
                            ds.point_data()
                                .and_then(|p| p.get(array))
                                .or_else(|| ds.cell_data().and_then(|c| c.get(array)))
                        })
                        .map(|a| a.range(*component))
                        .unwrap_or(pick_node.style.scalar_range)
                } else {
                    pick_node.style.scalar_range
                };
                (title, range)
            }
            ColorBy::Solid(_) => return None,
        };
        Some((title, pick_node.style.colormap.clone(), range))
    }

    fn draw_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.modal.take() else {
            return;
        };
        let title = match &modal {
            PathModal::OpenFile(_) => "Open Scientific File",
            PathModal::SaveState(_) => "Save Pipeline State (JSON)",
            PathModal::LoadState(_) => "Load Pipeline State (JSON)",
            PathModal::ExportDataset(_) => "Export Active Dataset",
            PathModal::SaveScreenshot(_) => "Save Screenshot (PNG)",
        };
        let mut keep_open = true;
        let mut confirm = false;

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                let path_str = match &mut modal {
                    PathModal::OpenFile(s)
                    | PathModal::SaveState(s)
                    | PathModal::LoadState(s)
                    | PathModal::ExportDataset(s)
                    | PathModal::SaveScreenshot(s) => s,
                };
                ui.label("File path:");
                let resp = ui.add(
                    egui::TextEdit::singleline(path_str)
                        .desired_width(360.0)
                        .hint_text("path/to/file"),
                );
                if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    confirm = true;
                }
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        confirm = true;
                    }
                    if ui.button("Cancel").clicked() {
                        keep_open = false;
                    }
                });
            });

        if confirm {
            match modal {
                PathModal::OpenFile(p) => {
                    let trimmed = p.trim();
                    if !trimmed.is_empty() {
                        self.load_path(Path::new(trimmed));
                    }
                }
                PathModal::LoadState(p) => {
                    let trimmed = p.trim();
                    if !trimmed.is_empty() {
                        self.load_path(Path::new(trimmed));
                    }
                }
                PathModal::SaveState(p) => match self.session.save_state_json() {
                    Ok(json) => match std::fs::write(p.trim(), json) {
                        Ok(()) => {
                            self.error_message = None;
                            self.status_message = format!("Saved state -> {}", p.trim());
                        }
                        Err(e) => self.error_message = Some(format!("Write failed: {e}")),
                    },
                    Err(e) => self.error_message = Some(format!("Serialize failed: {e}")),
                },
                PathModal::ExportDataset(p) => {
                    if let Some(id) = self.selected_node {
                        match self.session.export_dataset(id, p.trim()) {
                            Ok(()) => {
                                self.error_message = None;
                                self.status_message = format!("Exported dataset -> {}", p.trim());
                            }
                            Err(e) => self.error_message = Some(format!("Export failed: {e}")),
                        }
                    }
                }
                PathModal::SaveScreenshot(p) => {
                    let (w, h) = self.viewport_tex_size;
                    let w = w.max(640);
                    let h = h.max(480);
                    match self.session.build_scene().and_then(|scene| {
                        self.renderer
                            .render_to_png(&scene, w, h, p.trim())
                            .map_err(SessionError::from)
                    }) {
                        Ok(()) => {
                            self.error_message = None;
                            self.status_message =
                                format!("Saved screenshot ({w}x{h}) -> {}", p.trim());
                        }
                        Err(e) => self.error_message = Some(format!("Screenshot failed: {e}")),
                    }
                }
            }
        } else if keep_open {
            self.modal = Some(modal);
        }
    }
}

impl eframe::App for ParaVizApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;
        if dt > 1e-5 {
            let inst_fps = (1.0 / dt) as f32;
            self.fps_smoothed = self.fps_smoothed * 0.9 + inst_fps * 0.1;
        }

        if self.playing {
            let steps = self.session.all_time_steps();
            if steps.is_empty() {
                self.playing = false;
            } else {
                self.time_accumulator += dt * self.play_fps;
                if self.time_accumulator >= 1.0 {
                    let advance = self.time_accumulator.floor() as isize;
                    self.time_accumulator -= advance as f64;
                    self.step_time(advance);
                }
                ui.ctx().request_repaint();
            }
        }

        self.handle_shortcuts_and_drops(ui.ctx());

        egui::Panel::top("paraviz_top_bar").show(ui, |ui| {
            self.draw_menu_bar(ui);
            ui.separator();
            self.draw_toolbar(ui);
        });

        egui::Panel::left("paraviz_left_dock")
            .default_size(310.0)
            .min_size(240.0)
            .show(ui, |ui| {
                self.draw_left_dock(ui);
            });

        egui::Panel::bottom("paraviz_status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(err) = &self.error_message {
                    ui.colored_label(egui::Color32::from_rgb(255, 110, 110), err);
                    if ui.small_button("Dismiss").clicked() {
                        self.error_message = None;
                    }
                } else {
                    ui.label(&self.status_message);
                }
            });
        });

        if self.show_drawer {
            egui::Panel::bottom("paraviz_inspector_drawer")
                .resizable(true)
                .default_size(220.0)
                .min_size(120.0)
                .show(ui, |ui| {
                    self.draw_drawer(ui);
                });
        }

        egui::CentralPanel::no_frame().show(ui, |ui| {
            self.draw_viewport(ui);
        });

        self.draw_modal(ui.ctx());
    }
}

fn drag_vec3(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f32) -> bool {
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

fn drag_vec3_f64(ui: &mut egui::Ui, label: &str, v: &mut [f64; 3], speed: f64) -> bool {
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

fn drag_dims3(ui: &mut egui::Ui, label: &str, dims: &mut [usize; 3], min_val: usize) -> bool {
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

fn edit_source_config(ui: &mut egui::Ui, cfg: &mut SourceConfig) -> bool {
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
        SourceConfig::File(p) | SourceConfig::PvdSeries(p) => {
            ui.label(format!("Path: {}", p.display()));
        }
    }
    changed
}

fn edit_filter_config(ui: &mut egui::Ui, cfg: &mut FilterConfig) -> bool {
    let mut changed = false;
    match cfg {
        FilterConfig::ExtractSurface(f) => {
            changed |= ui
                .checkbox(&mut f.compute_normals, "Compute Normals")
                .changed();
        }
        FilterConfig::Contour(f) => {
            ui.horizontal(|ui| {
                ui.label("Scalar Field:");
                changed |= ui.text_edit_singleline(&mut f.field_name).changed();
            });
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
                    if ui.small_button("✕").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(idx) = remove_idx
                && f.iso_values.len() > 1 {
                    f.iso_values.remove(idx);
                    changed = true;
                }
            if ui.button("+ Add Iso Value").clicked() {
                let next = f.iso_values.last().copied().unwrap_or(0.0) + 10.0;
                f.iso_values.push(next);
                changed = true;
            }
        }
        FilterConfig::Slice(f) => {
            changed |= drag_vec3(ui, "Origin", &mut f.origin, 0.05);
            changed |= drag_vec3(ui, "Normal", &mut f.normal, 0.05);
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
                }
            }
            changed |= ui.checkbox(&mut f.invert, "Invert").changed();
            changed |= ui.checkbox(&mut f.crinkle, "Crinkle").changed();
        }
        FilterConfig::Threshold(f) => {
            ui.horizontal(|ui| {
                ui.label("Field:");
                changed |= ui.text_edit_singleline(&mut f.field_name).changed();
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
            changed |= ui
                .add(egui::DragValue::new(&mut f.min).speed(0.5).prefix("Min: "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut f.max).speed(0.5).prefix("Max: "))
                .changed();
            changed |= ui.checkbox(&mut f.invert, "Invert").changed();
            changed |= ui.checkbox(&mut f.all_scalars, "All Scalars").changed();
        }
        FilterConfig::StreamTracer(f) => {
            ui.horizontal(|ui| {
                ui.label("Vector Field:");
                changed |= ui.text_edit_singleline(&mut f.vector_field).changed();
            });
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
            ui.horizontal(|ui| {
                ui.label("Vary By Scalar:");
                if ui.text_edit_singleline(&mut vary).changed() {
                    f.vary_radius_by_scalar = if vary.trim().is_empty() {
                        None
                    } else {
                        Some(vary)
                    };
                    changed = true;
                }
            });
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
            ui.horizontal(|ui| {
                ui.label("Vector Field:");
                if ui.text_edit_singleline(&mut vf).changed() {
                    f.vector_field = if vf.trim().is_empty() { None } else { Some(vf) };
                    changed = true;
                }
            });
            let mut sf = f.scalar_field.clone().unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("Scalar Field:");
                if ui.text_edit_singleline(&mut sf).changed() {
                    f.scalar_field = if sf.trim().is_empty() { None } else { Some(sf) };
                    changed = true;
                }
            });
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
            ui.horizontal(|ui| {
                ui.label("Vector Field:");
                changed |= ui.text_edit_singleline(&mut f.vector_field).changed();
            });
            changed |= ui
                .add(
                    egui::DragValue::new(&mut f.scale_factor)
                        .speed(0.05)
                        .prefix("Scale Factor: "),
                )
                .changed();
        }
        FilterConfig::WarpByScalar(f) => {
            ui.horizontal(|ui| {
                ui.label("Scalar Field:");
                changed |= ui.text_edit_singleline(&mut f.scalar_field).changed();
            });
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
            ui.horizontal(|ui| {
                ui.label("Field:");
                changed |= ui.text_edit_singleline(&mut f.field_name).changed();
            });
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
        }
        FilterConfig::Calculator(f) => {
            ui.label("Expression:");
            changed |= ui.text_edit_singleline(&mut f.expression).changed();
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
            changed |= ui
                .add(egui::Slider::new(&mut f.resolution, 2..=1000).text("Resolution"))
                .changed();
        }
    }
    changed
}

fn paint_colormap_preview(ui: &mut egui::Ui, cmap: &ColorMap, height: f32) {
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
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(110)),
        egui::StrokeKind::Inside,
    );
}

fn paint_scalar_bar_overlay(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    title: &str,
    cmap: &ColorMap,
    range: (f64, f64),
) {
    let bar_w = 18.0;
    let bar_h = (viewport_rect.height() * 0.34).clamp(110.0, 220.0);
    let pad = 14.0;
    let box_w = 132.0;
    let box_h = bar_h + 42.0;

    if viewport_rect.width() < box_w + 40.0 || viewport_rect.height() < box_h + 40.0 {
        return;
    }

    let box_rect = egui::Rect::from_min_size(
        egui::pos2(
            viewport_rect.right() - box_w - pad,
            viewport_rect.bottom() - box_h - pad,
        ),
        egui::vec2(box_w, box_h),
    );
    painter.rect_filled(
        box_rect,
        6.0,
        egui::Color32::from_rgba_unmultiplied(15, 18, 26, 185),
    );
    painter.rect_stroke(
        box_rect,
        6.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(140, 155, 180, 90)),
        egui::StrokeKind::Inside,
    );

    painter.text(
        egui::pos2(box_rect.left() + 10.0, box_rect.top() + 8.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(12.0),
        egui::Color32::from_rgb(235, 240, 248),
    );

    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(box_rect.left() + 12.0, box_rect.top() + 30.0),
        egui::vec2(bar_w, bar_h),
    );

    let segs = 48;
    for i in 0..segs {
        let frac0 = i as f32 / segs as f32;
        let frac1 = (i + 1) as f32 / segs as f32;
        let t = 1.0 - (frac0 + frac1) * 0.5;
        let rgb = cmap.sample_rgb(t);
        let c = egui::Color32::from_rgb(
            (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
        );
        let y0 = bar_rect.top() + frac0 * bar_rect.height();
        let y1 = bar_rect.top() + frac1 * bar_rect.height();
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(bar_rect.left(), y0),
                egui::pos2(bar_rect.right(), y1),
            ),
            0.0,
            c,
        );
    }
    painter.rect_stroke(
        bar_rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 215, 225)),
        egui::StrokeKind::Inside,
    );

    let mid = 0.5 * (range.0 + range.1);
    let label_x = bar_rect.right() + 8.0;
    let font = egui::FontId::monospace(10.5);
    let text_color = egui::Color32::from_rgb(225, 230, 240);
    painter.text(
        egui::pos2(label_x, bar_rect.top()),
        egui::Align2::LEFT_TOP,
        format!("{:.4}", range.1),
        font.clone(),
        text_color,
    );
    painter.text(
        egui::pos2(label_x, bar_rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{mid:.4}"),
        font.clone(),
        text_color,
    );
    painter.text(
        egui::pos2(label_x, bar_rect.bottom()),
        egui::Align2::LEFT_BOTTOM,
        format!("{:.4}", range.0),
        font,
        text_color,
    );
}

fn extract_dataset_points(ds: &DataSet) -> Vec<[f32; 3]> {
    match ds {
        DataSet::PolyData(p) => p.points.clone(),
        DataSet::ImageData(i) => {
            let max_pts = i.num_points().min(200);
            let [nx, ny, _] = i.dimensions;
            (0..max_pts)
                .map(|idx| {
                    let k = idx / (nx * ny).max(1);
                    let rem = idx % (nx * ny).max(1);
                    let j = rem / nx.max(1);
                    let ii = rem % nx.max(1);
                    i.point_coords(ii, j, k)
                })
                .collect()
        }
        DataSet::RectilinearGrid(r) => {
            let max_pts = r.num_points().min(200);
            let [nx, ny, _] = r.dimensions();
            (0..max_pts)
                .map(|idx| {
                    let k = idx / (nx * ny).max(1);
                    let rem = idx % (nx * ny).max(1);
                    let j = rem / nx.max(1);
                    let ii = rem % nx.max(1);
                    r.point_coords(ii, j, k)
                })
                .collect()
        }
        DataSet::StructuredGrid(s) => s.points.clone(),
        DataSet::UnstructuredGrid(u) => u.points.clone(),
        DataSet::MultiBlock(mb) => mb
            .blocks
            .first()
            .map(|(_, b)| extract_dataset_points(b))
            .unwrap_or_default(),
    }
}

fn format_tuple(arr: &DataArray, row: usize) -> String {
    if row >= arr.num_tuples() {
        return "-".into();
    }
    let vals = arr.get_tuple(row);
    if vals.len() == 1 {
        format!("{:.4}", vals[0])
    } else {
        let parts: Vec<String> = vals.iter().map(|v| format!("{v:.3}")).collect();
        format!("({})", parts.join(", "))
    }
}
