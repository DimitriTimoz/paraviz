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
    Chart,
    RustTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChartMode {
    XYPlot,
    Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewportMode {
    Orbit,
    Pick,
    Ruler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpreadsheetTarget {
    Points,
    Cells,
}

#[derive(Debug, Clone)]
struct FileBrowserState {
    current_dir: PathBuf,
    filter_text: String,
    path_input: String,
}

impl FileBrowserState {
    fn new() -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let path_input = current_dir.display().to_string();
        Self {
            current_dir,
            filter_text: String::new(),
            path_input,
        }
    }
}

#[derive(Debug, Clone)]
enum PathModal {
    OpenFile(String),
    FileBrowser(FileBrowserState),
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
    chart_mode: ChartMode,
    hist_array_name: String,
    hist_association: AttributeAssociation,
    hist_component: Option<usize>,
    hist_bins: usize,
    spreadsheet_target: SpreadsheetTarget,
    show_scalar_bar: bool,
    show_widgets: bool,

    viewport_mode: ViewportMode,
    last_pick: Option<PickResult>,
    ruler_points: Vec<[f32; 3]>,
    export_glb_colors: bool,

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
            chart_mode: ChartMode::Histogram,
            hist_array_name: String::new(),
            hist_association: AttributeAssociation::Point,
            hist_component: None,
            hist_bins: 20,
            spreadsheet_target: SpreadsheetTarget::Points,
            show_scalar_bar: true,
            show_widgets: true,
            viewport_mode: ViewportMode::Orbit,
            last_pick: None,
            ruler_points: Vec::new(),
            export_glb_colors: true,
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
                FilterConfig::Elevation(f) if bb.is_valid() => {
                    let dz = (bb.max.z - bb.min.z).abs();
                    let dy = (bb.max.y - bb.min.y).abs();
                    if dz > 1e-4 {
                        f.low_point = [center[0], center[1], bb.min.z];
                        f.high_point = [center[0], center[1], bb.max.z];
                    } else if dy > 1e-4 {
                        f.low_point = [center[0], bb.min.y, center[2]];
                        f.high_point = [center[0], bb.max.y, center[2]];
                    } else {
                        f.low_point = [bb.min.x, center[1], center[2]];
                        f.high_point = [bb.max.x, center[1], center[2]];
                    }
                }
                _ => {}
            }
        }

        let is_cell_centers = matches!(cfg, FilterConfig::CellCenters(_));
        let is_feature_edges = matches!(cfg, FilterConfig::FeatureEdges(_));
        let is_probe = matches!(cfg, FilterConfig::ProbeOverLine(_));
        let label = self.unique_label(cfg.default_label());
        match self.session.add_filter_cfg(label.clone(), parent_id, cfg) {
            Ok(id) => {
                let eval_res = self.session.evaluate(id);
                let handle = self.session.show(id);
                if is_cell_centers {
                    handle.representation(Representation::Points);
                } else if is_feature_edges {
                    handle.representation(Representation::Wireframe);
                }
                if let Err(e) = eval_res {
                    self.error_message = Some(format!("Filter '{label}' failed: {e}"));
                } else {
                    self.error_message = None;
                    self.status_message = format!("Applied filter '{label}'");
                }
                if is_probe {
                    self.chart_mode = ChartMode::XYPlot;
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

    fn apply_gpu_contour_to_selected(&mut self) {
        let Some(sel_id) = self.selected_node else {
            self.error_message = Some("Select an ImageData node first.".into());
            return;
        };
        let Ok(ds) = self.session.evaluate(sel_id) else {
            self.error_message = Some("Failed to evaluate selected node.".into());
            return;
        };
        let Some(img) = ds.as_image_data() else {
            self.error_message =
                Some("GPU Compute Contour requires an ImageData dataset.".into());
            return;
        };
        let scalar_info = img
            .point_data
            .active_scalars_array()
            .or_else(|| img.point_data.arrays().first())
            .map(|a| (a.name.clone(), a.range(None)));
        let (field_name, iso_value) = match scalar_info {
            Some((name, (mn, mx))) => (Some(name), (mn + mx) * 0.5),
            None => (None, 0.0),
        };
        let label = self.unique_label("GpuContour");
        match self
            .session
            .gpu_contour(label.clone(), sel_id, field_name.as_deref(), iso_value)
        {
            Ok(new_id) => {
                let _ = self.session.evaluate(new_id);
                self.session.show(new_id);
                self.selected_node = Some(new_id);
                self.pending_source = None;
                self.pending_filter = None;
                self.error_message = None;
                self.status_message =
                    format!("Applied GPU Compute Contour '{label}' (iso={iso_value:.3})");
            }
            Err(e) => {
                self.error_message = Some(format!("GPU Compute Contour failed: {e}"));
            }
        }
    }

    fn append_visible_datasets(&mut self) {
        if let Err(e) = self.session.update_visible() {
            self.error_message = Some(format!("Append failed: {e}"));
            return;
        }
        let merged_res = {
            let refs: Vec<&DataSet> = self
                .session
                .nodes()
                .iter()
                .filter(|n| n.style.visible)
                .filter_map(|n| n.dataset().map(|arc| arc.as_ref()))
                .collect();
            if refs.is_empty() {
                self.error_message = Some("No visible datasets to append.".into());
                return;
            }
            append_datasets(&refs)
        };
        match merged_res {
            Ok(ds) => {
                let label = self.unique_label("AppendDatasets");
                let id = self.session.add_dataset(label.clone(), ds);
                let _ = self.session.evaluate(id);
                self.session.show(id);
                self.selected_node = Some(id);
                self.pending_source = None;
                self.pending_filter = None;
                self.error_message = None;
                self.status_message = format!("Appended visible datasets -> '{label}'");
            }
            Err(e) => {
                self.error_message = Some(format!("Append Datasets failed: {e}"));
            }
        }
    }

    fn sync_viewport_widgets(&mut self) {
        self.session.widgets.clear();
        if self.show_widgets
            && let Some(sel_id) = self.selected_node
            && let Ok(node) = self.session.node(sel_id)
        {
            let active_cfg = self
                .pending_filter
                .as_ref()
                .filter(|(id, _)| *id == sel_id)
                .map(|(_, cfg)| cfg)
                .or(node.filter_config.as_ref());

            if let Some(flt) = active_cfg {
                let ref_bounds = node
                    .parent
                    .and_then(|pid| self.session.node(pid).ok())
                    .and_then(|pn| pn.dataset())
                    .map(|ds| ds.bounds())
                    .filter(|b| b.is_valid())
                    .or_else(|| node.dataset().map(|ds| ds.bounds()).filter(|b| b.is_valid()));
                let plane_size = ref_bounds
                    .map(|b| (b.diagonal() * 0.85).max(0.5))
                    .unwrap_or(4.0);

                match flt {
                    FilterConfig::Slice(f) => {
                        self.session.widgets.push(ViewportWidget::Plane {
                            origin: f.origin,
                            normal: f.normal,
                            size: plane_size,
                        });
                    }
                    FilterConfig::Clip(f) => match &f.function {
                        ImplicitFunction::Plane { origin, normal } => {
                            self.session.widgets.push(ViewportWidget::Plane {
                                origin: *origin,
                                normal: *normal,
                                size: plane_size,
                            });
                        }
                        ImplicitFunction::Sphere { center, radius } => {
                            self.session.widgets.push(ViewportWidget::Sphere {
                                center: *center,
                                radius: *radius,
                            });
                        }
                        ImplicitFunction::Box { min, max } => {
                            self.session.widgets.push(ViewportWidget::Box {
                                min: *min,
                                max: *max,
                            });
                        }
                    },
                    FilterConfig::ProbeOverLine(f) => {
                        self.session.widgets.push(ViewportWidget::Line {
                            p1: f.point1,
                            p2: f.point2,
                        });
                    }
                    FilterConfig::StreamTracer(f) => match &f.seeds {
                        SeedSource::Line { p1, p2, .. } => {
                            self.session
                                .widgets
                                .push(ViewportWidget::Line { p1: *p1, p2: *p2 });
                        }
                        SeedSource::Sphere { center, radius, .. } => {
                            self.session.widgets.push(ViewportWidget::Sphere {
                                center: *center,
                                radius: *radius,
                            });
                        }
                        SeedSource::Points(_) => {}
                    },
                    FilterConfig::Elevation(f) => {
                        self.session.widgets.push(ViewportWidget::Line {
                            p1: f.low_point,
                            p2: f.high_point,
                        });
                    }
                    _ => {}
                }
            }
        }

        if self.ruler_points.len() == 2
            && (self.show_widgets || self.viewport_mode == ViewportMode::Ruler)
        {
            self.session.widgets.push(ViewportWidget::Line {
                p1: self.ruler_points[0],
                p2: self.ruler_points[1],
            });
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
            let has_arc = self
                .session
                .node(id)
                .ok()
                .and_then(|n| n.dataset())
                .and_then(|ds| ds.as_poly_data())
                .is_some_and(|pd| pd.point_data.get("ArcLength").is_some());
            self.chart_mode = if has_arc {
                ChartMode::XYPlot
            } else {
                ChartMode::Histogram
            };
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
                if ui.button("File Browser...").clicked() {
                    self.modal = Some(PathModal::FileBrowser(FileBrowserState::new()));
                    ui.close();
                }
                if ui.button("Open File by Path...").clicked() {
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
                        "Elevation",
                        FilterConfig::Elevation(ElevationFilter::default()),
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
                        egui::Button::new("GPU Compute Contour (WebGPU)"),
                    )
                    .clicked()
                {
                    self.apply_gpu_contour_to_selected();
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

            ui.menu_button("View", |ui| {
                ui.checkbox(
                    &mut self.show_drawer,
                    "Inspector Drawer (Spreadsheet / Chart / Trace)",
                );
                ui.checkbox(&mut self.show_scalar_bar, "Scalar Color Bar");
                ui.checkbox(&mut self.show_widgets, "3D Filter Widgets");
                ui.checkbox(&mut self.session.settings.show_axes_gizmo, "Axes Gizmo");
                ui.checkbox(
                    &mut self.session.settings.show_bounding_box,
                    "Bounding Box",
                );
                let mut msaa_4x = self.session.settings.msaa_samples >= 4;
                if ui.checkbox(&mut msaa_4x, "4x MSAA").changed() {
                    self.session.settings.msaa_samples = if msaa_4x { 4 } else { 1 };
                }
                ui.checkbox(&mut self.session.settings.edl_enabled, "Eye-Dome Lighting");
                if self.session.settings.edl_enabled {
                    ui.add(
                        egui::Slider::new(&mut self.session.settings.edl_strength, 0.0..=4.0)
                            .text("EDL Strength"),
                    );
                }
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
            ui.label("Mode:");
            ui.selectable_value(&mut self.viewport_mode, ViewportMode::Orbit, "Orbit");
            ui.selectable_value(&mut self.viewport_mode, ViewportMode::Pick, "Pick");
            ui.selectable_value(&mut self.viewport_mode, ViewportMode::Ruler, "Ruler");
            if !self.ruler_points.is_empty() && ui.button("Clear Ruler").clicked() {
                self.ruler_points.clear();
            }

            ui.separator();
            ui.toggle_value(&mut self.session.camera.orthographic, "Ortho");
            ui.toggle_value(&mut self.session.settings.show_axes_gizmo, "Axes");
            ui.toggle_value(&mut self.session.settings.show_bounding_box, "Bounds");
            ui.toggle_value(&mut self.show_widgets, "Widgets");
            let mut msaa_4x = self.session.settings.msaa_samples >= 4;
            if ui.toggle_value(&mut msaa_4x, "MSAA 4x").changed() {
                self.session.settings.msaa_samples = if msaa_4x { 4 } else { 1 };
            }
            ui.toggle_value(&mut self.session.settings.edl_enabled, "EDL");
            if self.session.settings.edl_enabled {
                ui.add(
                    egui::Slider::new(&mut self.session.settings.edl_strength, 0.0..=4.0)
                        .text("Strength")
                        .max_decimals(2),
                );
            }
            ui.toggle_value(&mut self.show_scalar_bar, "ScalarBar");
            ui.separator();
            ui.toggle_value(&mut self.show_drawer, "Spreadsheet / Chart / Trace");

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
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::Chart, "Chart / Histogram");
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
            DrawerTab::Chart => self.draw_chart_tab(ui),
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

    fn draw_chart_tab(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to view XY plots or scalar histograms.");
            return;
        };
        let _ = self.session.evaluate(sel_id);
        let Ok(node) = self.session.node(sel_id) else {
            return;
        };
        let Some(ds) = node.dataset().cloned() else {
            ui.weak("Dataset not evaluated.");
            return;
        };

        let has_arc_length = ds
            .as_poly_data()
            .and_then(|pd| pd.point_data.get("ArcLength"))
            .is_some();

        if !has_arc_length && self.chart_mode == ChartMode::XYPlot {
            self.chart_mode = ChartMode::Histogram;
        }

        ui.horizontal(|ui| {
            if has_arc_length {
                ui.selectable_value(
                    &mut self.chart_mode,
                    ChartMode::XYPlot,
                    "XY Plot (ArcLength)",
                );
                ui.selectable_value(
                    &mut self.chart_mode,
                    ChartMode::Histogram,
                    "Scalar Histogram",
                );
                ui.separator();
            }

            if self.chart_mode == ChartMode::Histogram {
                // Collect available point and cell arrays
                let mut candidates: Vec<(AttributeAssociation, String, usize)> = Vec::new();
                if let Some(pd) = ds.point_data() {
                    for arr in pd.arrays() {
                        candidates.push((
                            AttributeAssociation::Point,
                            arr.name.clone(),
                            arr.num_components,
                        ));
                    }
                }
                if let Some(cd) = ds.cell_data() {
                    for arr in cd.arrays() {
                        candidates.push((
                            AttributeAssociation::Cell,
                            arr.name.clone(),
                            arr.num_components,
                        ));
                    }
                }

                if candidates.is_empty() {
                    ui.weak("Dataset has no point or cell scalar arrays.");
                    return;
                }

                let current_valid = candidates.iter().any(|(a, n, _)| {
                    *a == self.hist_association && *n == self.hist_array_name
                });
                if !current_valid {
                    if let Some(active) = ds
                        .point_data()
                        .and_then(|p| p.active_scalars_array())
                    {
                        self.hist_association = AttributeAssociation::Point;
                        self.hist_array_name = active.name.clone();
                    } else if let Some(active) = ds
                        .cell_data()
                        .and_then(|c| c.active_scalars_array())
                    {
                        self.hist_association = AttributeAssociation::Cell;
                        self.hist_array_name = active.name.clone();
                    } else {
                        self.hist_association = candidates[0].0;
                        self.hist_array_name = candidates[0].1.clone();
                    }
                    self.hist_component = None;
                }

                let selected_comps = candidates
                    .iter()
                    .find(|(a, n, _)| {
                        *a == self.hist_association && *n == self.hist_array_name
                    })
                    .map(|(_, _, c)| *c)
                    .unwrap_or(1);

                let sel_label = format!(
                    "{}: {}",
                    match self.hist_association {
                        AttributeAssociation::Point => "Point",
                        AttributeAssociation::Cell => "Cell",
                        AttributeAssociation::Field => "Field",
                    },
                    self.hist_array_name
                );
                egui::ComboBox::from_id_salt("hist_array_combo")
                    .selected_text(sel_label)
                    .show_ui(ui, |ui| {
                        for (assoc, name, _) in &candidates {
                            let prefix = match assoc {
                                AttributeAssociation::Point => "Point",
                                AttributeAssociation::Cell => "Cell",
                                AttributeAssociation::Field => "Field",
                            };
                            let active = *assoc == self.hist_association
                                && *name == self.hist_array_name;
                            if ui
                                .selectable_label(active, format!("{prefix}: {name}"))
                                .clicked()
                            {
                                self.hist_association = *assoc;
                                self.hist_array_name = name.clone();
                                self.hist_component = None;
                            }
                        }
                    });

                if selected_comps > 1 {
                    let comp_text = match self.hist_component {
                        None => "Magnitude".to_string(),
                        Some(c) => format!("Comp {c}"),
                    };
                    egui::ComboBox::from_id_salt("hist_comp_combo")
                        .selected_text(comp_text)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.hist_component, None, "Magnitude");
                            for c in 0..selected_comps {
                                ui.selectable_value(
                                    &mut self.hist_component,
                                    Some(c),
                                    format!("Comp {c}"),
                                );
                            }
                        });
                }

                ui.add(egui::Slider::new(&mut self.hist_bins, 4..=64).text("Bins"));
            }
        });

        match self.chart_mode {
            ChartMode::XYPlot if has_arc_length => {
                if let Some(pd) = ds.as_poly_data() {
                    paint_xy_arclength_plot(ui, pd);
                }
            }
            _ => {
                if self.hist_array_name.is_empty() {
                    ui.weak("No scalar array selected for histogram.");
                    return;
                }
                match compute_histogram(
                    &ds,
                    &self.hist_array_name,
                    self.hist_association,
                    self.hist_component,
                    self.hist_bins,
                ) {
                    Ok(hist) => {
                        ui.horizontal(|ui| {
                            let total: usize = hist.counts.iter().sum();
                            ui.monospace(format!(
                                "min={:.4}  max={:.4}  mean={:.4}  std_dev={:.4}  N={}",
                                hist.min, hist.max, hist.mean, hist.std_dev, total
                            ));
                        });
                        paint_histogram_chart(ui, &hist);
                    }
                    Err(e) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 130, 130),
                            format!("Histogram unavailable: {e}"),
                        );
                    }
                }
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

        self.sync_viewport_widgets();

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
            if self.viewport_mode == ViewportMode::Orbit
                && resp.dragged_by(egui::PointerButton::Primary)
            {
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

            if resp.clicked_by(egui::PointerButton::Primary)
                && let Some(pos) = resp.interact_pointer_pos()
                && let Ok(scene) = &scene_res
            {
                let local = pos - resp.rect.min;
                let px = local.x * ppp;
                let py = local.y * ppp;
                let hit = pick_scene(scene, px, py, w, h);
                match self.viewport_mode {
                    ViewportMode::Orbit | ViewportMode::Pick => {
                        self.last_pick = hit;
                    }
                    ViewportMode::Ruler => {
                        if let Some(h_res) = hit {
                            if self.ruler_points.len() >= 2 {
                                self.ruler_points.clear();
                            }
                            self.ruler_points.push(h_res.world_pos);
                            self.last_pick = Some(h_res);
                            self.sync_viewport_widgets();
                            ui.ctx().request_repaint();
                        }
                    }
                }
            }

            if resp.double_clicked() && self.viewport_mode == ViewportMode::Orbit {
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
        let msaa_label = if self.session.settings.edl_enabled {
            "EDL".to_string()
        } else if self.session.settings.msaa_samples >= 4 {
            "MSAA 4x".to_string()
        } else {
            "1x".to_string()
        };
        let mode_label = match self.viewport_mode {
            ViewportMode::Orbit => "Orbit",
            ViewportMode::Pick => "Pick",
            ViewportMode::Ruler => "Ruler",
        };
        let mut hud_lines = vec![format!(
            "WebGPU | {:.1} FPS ({:.2} ms) | {} pts | {} cells | {} | {} | Mode: {}",
            self.fps_smoothed,
            self.last_render_ms,
            total_points,
            total_cells,
            proj_mode,
            msaa_label,
            mode_label
        )];
        if let Some(pick) = &self.last_pick {
            let pt_str = pick.point_index.map_or_else(|| "-".into(), |i| i.to_string());
            let cell_str = pick.cell_index.map_or_else(|| "-".into(), |i| i.to_string());
            let scalar_str = pick
                .scalar_value
                .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            hud_lines.push(format!(
                "Pick: Actor {} | Pos ({:.3}, {:.3}, {:.3}) | Pt {} | Cell {} | Scalar {}",
                pick.actor_name,
                pick.world_pos[0],
                pick.world_pos[1],
                pick.world_pos[2],
                pt_str,
                cell_str,
                scalar_str
            ));
        }
        if self.ruler_points.len() == 1 {
            let p1 = self.ruler_points[0];
            hud_lines.push(format!(
                "Ruler: P1 ({:.3}, {:.3}, {:.3}) — click second point",
                p1[0], p1[1], p1[2]
            ));
        } else if self.ruler_points.len() == 2 {
            let p1 = glam::Vec3::from_array(self.ruler_points[0]);
            let p2 = glam::Vec3::from_array(self.ruler_points[1]);
            let dist = (p2 - p1).length();
            hud_lines.push(format!(
                "Ruler: Distance = {:.4} | ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3})",
                dist, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z
            ));
        }
        let hud_text = hud_lines.join("\n");

        let painter = ui.painter_at(rect);
        let hud_pos = rect.left_top() + egui::vec2(12.0, 10.0);
        let galley = painter.layout(
            hud_text,
            egui::FontId::monospace(11.5),
            egui::Color32::from_rgb(230, 236, 245),
            f32::INFINITY,
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
            PathModal::FileBrowser(_) => "File Browser",
            PathModal::SaveState(_) => "Save Pipeline State (JSON)",
            PathModal::LoadState(_) => "Load Pipeline State (JSON)",
            PathModal::ExportDataset(_) => "Export Active Dataset",
            PathModal::SaveScreenshot(_) => "Save Screenshot (PNG)",
        };
        let mut keep_open = true;
        let mut confirm = false;
        let mut switch_to_browser = false;
        let mut picked_file: Option<PathBuf> = None;

        if let PathModal::FileBrowser(ref mut fb) = modal {
            egui::Window::new(title)
                .collapsible(false)
                .resizable(true)
                .default_size([560.0, 420.0])
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button(".. Up").clicked()
                            && let Some(parent) = fb.current_dir.parent()
                        {
                            fb.current_dir = parent.to_path_buf();
                            fb.path_input = fb.current_dir.display().to_string();
                        }
                        if ui.button("CWD").clicked() {
                            fb.current_dir =
                                std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                            fb.path_input = fb.current_dir.display().to_string();
                        }
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut fb.path_input)
                                .desired_width(340.0),
                        );
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() {
                                picked_file = Some(p);
                            }
                        }
                        if ui.button("Go").clicked() {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() {
                                picked_file = Some(p);
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Filter:");
                        ui.add(
                            egui::TextEdit::singleline(&mut fb.filter_text)
                                .desired_width(220.0)
                                .hint_text("substring filter"),
                        );
                        ui.weak("Supported: .vtk .vtp .vti .vtu .pvtp .pvtu .stl .obj .ply .csv .json");
                    });
                    ui.separator();

                    let mut dirs: Vec<(String, PathBuf)> = Vec::new();
                    let mut files: Vec<(String, PathBuf)> = Vec::new();
                    let mut read_err: Option<String> = None;

                    match std::fs::read_dir(&fb.current_dir) {
                        Ok(entries) => {
                            let needle = fb.filter_text.trim().to_ascii_lowercase();
                            for entry in entries.flatten() {
                                let path = entry.path();
                                let name = entry.file_name().to_string_lossy().to_string();
                                if !needle.is_empty()
                                    && !name.to_ascii_lowercase().contains(&needle)
                                {
                                    continue;
                                }
                                if path.is_dir() {
                                    dirs.push((name, path));
                                } else if is_supported_open_file(&path) {
                                    files.push((name, path));
                                }
                            }
                            dirs.sort_by(|a, b| {
                                a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase())
                            });
                            files.sort_by(|a, b| {
                                a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase())
                            });
                        }
                        Err(e) => {
                            read_err = Some(format!("Cannot read directory: {e}"));
                        }
                    }

                    egui::ScrollArea::vertical()
                        .id_salt("file_browser_scroll")
                        .max_height(280.0)
                        .show(ui, |ui| {
                            if let Some(err) = read_err {
                                ui.colored_label(egui::Color32::from_rgb(255, 130, 130), err);
                            } else if dirs.is_empty() && files.is_empty() {
                                ui.weak("No subdirectories or supported scientific files here.");
                            }
                            for (name, path) in dirs {
                                if ui.selectable_label(false, format!("[DIR] {name}")).clicked() {
                                    fb.current_dir = path;
                                    fb.path_input = fb.current_dir.display().to_string();
                                }
                            }
                            for (name, path) in files {
                                if ui.selectable_label(false, format!("      {name}")).clicked() {
                                    picked_file = Some(path);
                                }
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if let Some(path) = picked_file {
                self.load_path(&path);
            } else if keep_open {
                self.modal = Some(modal);
            }
            return;
        }

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                let is_open = matches!(modal, PathModal::OpenFile(_));
                let is_export = matches!(modal, PathModal::ExportDataset(_));
                let path_str = match &mut modal {
                    PathModal::OpenFile(s)
                    | PathModal::SaveState(s)
                    | PathModal::LoadState(s)
                    | PathModal::ExportDataset(s)
                    | PathModal::SaveScreenshot(s) => s,
                    PathModal::FileBrowser(_) => unreachable!(),
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
                if is_export {
                    ui.horizontal(|ui| {
                        ui.label("Format:");
                        for ext in ["vtp", "vtu", "vti", "vtk", "stl", "obj", "ply", "csv", "glb"]
                        {
                            if ui.small_button(format!(".{ext}")).clicked() {
                                let p = Path::new(path_str.trim());
                                let stem = p
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or("export");
                                let next = match p.parent() {
                                    Some(par) if !par.as_os_str().is_empty() => {
                                        par.join(format!("{stem}.{ext}"))
                                    }
                                    _ => PathBuf::from(format!("{stem}.{ext}")),
                                };
                                *path_str = next.display().to_string();
                            }
                        }
                    });
                    if path_str.trim().to_ascii_lowercase().ends_with(".glb") {
                        ui.checkbox(
                            &mut self.export_glb_colors,
                            "Include scalar vertex colors (COLOR_0)",
                        );
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        confirm = true;
                    }
                    if is_open && ui.button("Browse...").clicked() {
                        switch_to_browser = true;
                    }
                    if ui.button("Cancel").clicked() {
                        keep_open = false;
                    }
                });
            });

        if switch_to_browser {
            self.modal = Some(PathModal::FileBrowser(FileBrowserState::new()));
            return;
        }

        if confirm {
            match modal {
                PathModal::OpenFile(p) => {
                    let trimmed = p.trim();
                    if !trimmed.is_empty() {
                        self.load_path(Path::new(trimmed));
                    }
                }
                PathModal::FileBrowser(_) => {}
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
                        let trimmed = p.trim();
                        let res = if trimmed.to_ascii_lowercase().ends_with(".glb") {
                            self.session
                                .export_glb(id, trimmed, self.export_glb_colors)
                        } else {
                            self.session.export_dataset(id, trimmed)
                        };
                        match res {
                            Ok(()) => {
                                self.error_message = None;
                                self.status_message = format!("Exported dataset -> {trimmed}");
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
        FilterConfig::Elevation(f) => {
            changed |= drag_vec3(ui, "Low Point", &mut f.low_point, 0.05);
            changed |= drag_vec3(ui, "High Point", &mut f.high_point, 0.05);
            ui.horizontal(|ui| {
                ui.label("Scalar Name:");
                changed |= ui.text_edit_singleline(&mut f.scalar_name).changed();
            });
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

fn is_supported_open_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "vtk"
            | "vtp"
            | "vti"
            | "vtu"
            | "vtr"
            | "vts"
            | "vtm"
            | "pvd"
            | "pvtp"
            | "pvti"
            | "pvtu"
            | "stl"
            | "obj"
            | "ply"
            | "csv"
            | "tsv"
            | "json"
    )
}

const CHART_PALETTE: [egui::Color32; 6] = [
    egui::Color32::from_rgb(78, 195, 255),
    egui::Color32::from_rgb(255, 166, 77),
    egui::Color32::from_rgb(122, 232, 138),
    egui::Color32::from_rgb(245, 114, 152),
    egui::Color32::from_rgb(198, 148, 255),
    egui::Color32::from_rgb(255, 224, 92),
];

fn paint_xy_arclength_plot(ui: &mut egui::Ui, pd: &PolyData) {
    let Some(arc_arr) = pd.point_data.get("ArcLength") else {
        ui.weak("No ArcLength array found.");
        return;
    };
    let n = arc_arr.num_tuples();
    if n < 2 {
        ui.weak("At least 2 points are required for XY line plotting.");
        return;
    }

    let x_vals: Vec<f64> = (0..n).map(|i| arc_arr.get_magnitude(i)).collect();
    let series: Vec<(&str, Vec<f64>, egui::Color32)> = pd
        .point_data
        .arrays()
        .iter()
        .filter(|a| {
            a.name != "ArcLength"
                && a.name != "vtkValidPointMask"
                && a.num_tuples() == n
        })
        .take(CHART_PALETTE.len())
        .enumerate()
        .map(|(idx, arr)| {
            let ys: Vec<f64> = (0..n).map(|i| arr.get_magnitude(i)).collect();
            (arr.name.as_str(), ys, CHART_PALETTE[idx % CHART_PALETTE.len()])
        })
        .collect();

    if series.is_empty() {
        ui.weak("No scalar arrays to plot against ArcLength.");
        return;
    }

    let x_min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (x_min, x_max) = if (x_max - x_min).abs() < 1e-12 {
        (x_min - 0.5, x_max + 0.5)
    } else {
        (x_min, x_max)
    };

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, ys, _) in &series {
        for &y in ys {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        ui.weak("Plotted arrays contain no finite values.");
        return;
    }
    if (y_max - y_min).abs() < 1e-12 {
        y_min -= 0.5;
        y_max += 0.5;
    } else {
        let pad = (y_max - y_min) * 0.05;
        y_min -= pad;
        y_max += pad;
    }

    let avail = ui.available_size();
    let plot_h = avail.y.max(110.0);
    let plot_w = avail.x.max(200.0);
    let (outer_rect, resp) =
        ui.allocate_exact_size(egui::vec2(plot_w, plot_h), egui::Sense::hover());
    let painter = ui.painter_at(outer_rect);

    let left_pad = 64.0;
    let right_pad = if outer_rect.width() > 360.0 { 148.0 } else { 14.0 };
    let top_pad = 10.0;
    let bottom_pad = 24.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(outer_rect.left() + left_pad, outer_rect.top() + top_pad),
        egui::pos2(
            (outer_rect.right() - right_pad).max(outer_rect.left() + left_pad + 40.0),
            (outer_rect.bottom() - bottom_pad).max(outer_rect.top() + top_pad + 40.0),
        ),
    );

    painter.rect_filled(plot_rect, 2.0, egui::Color32::from_rgb(18, 22, 30));

    // Gridlines
    let grid_color = egui::Color32::from_rgba_unmultiplied(120, 135, 160, 45);
    for g in 0..=4 {
        let frac = g as f32 / 4.0;
        let gx = egui::lerp(plot_rect.left()..=plot_rect.right(), frac);
        let gy = egui::lerp(plot_rect.bottom()..=plot_rect.top(), frac);
        painter.line_segment(
            [
                egui::pos2(gx, plot_rect.top()),
                egui::pos2(gx, plot_rect.bottom()),
            ],
            egui::Stroke::new(1.0, grid_color),
        );
        painter.line_segment(
            [
                egui::pos2(plot_rect.left(), gy),
                egui::pos2(plot_rect.right(), gy),
            ],
            egui::Stroke::new(1.0, grid_color),
        );
    }

    painter.rect_stroke(
        plot_rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 95, 120)),
        egui::StrokeKind::Inside,
    );

    // Axis labels
    let label_font = egui::FontId::monospace(10.0);
    let label_col = egui::Color32::from_rgb(200, 210, 225);
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.top()),
        egui::Align2::RIGHT_TOP,
        format!("{y_max:.3}"),
        label_font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.center().y),
        egui::Align2::RIGHT_CENTER,
        format!("{:.3}", 0.5 * (y_min + y_max)),
        label_font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("{y_min:.3}"),
        label_font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("{x_min:.3}"),
        label_font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.center().x, plot_rect.bottom() + 4.0),
        egui::Align2::CENTER_TOP,
        "ArcLength",
        label_font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.bottom() + 4.0),
        egui::Align2::RIGHT_TOP,
        format!("{x_max:.3}"),
        label_font.clone(),
        label_col,
    );

    let to_screen = |x: f64, y: f64| -> egui::Pos2 {
        let tx = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0) as f32;
        let ty = ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            egui::lerp(plot_rect.left()..=plot_rect.right(), tx),
            egui::lerp(plot_rect.bottom()..=plot_rect.top(), ty),
        )
    };

    for (_, ys, color) in &series {
        let pts: Vec<egui::Pos2> = x_vals
            .iter()
            .zip(ys.iter())
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| to_screen(x, y))
            .collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.8, *color)));
        }
    }

    // Color legend
    if outer_rect.width() > 360.0 {
        let lx = plot_rect.right() + 10.0;
        let mut ly = plot_rect.top() + 4.0;
        for (name, _, color) in &series {
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(lx, ly + 3.0), egui::vec2(10.0, 8.0)),
                2.0,
                *color,
            );
            painter.text(
                egui::pos2(lx + 15.0, ly),
                egui::Align2::LEFT_TOP,
                *name,
                egui::FontId::proportional(11.5),
                egui::Color32::from_rgb(225, 232, 242),
            );
            ly += 17.0;
        }
    }

    // Interactive hover crosshair
    if resp.hovered()
        && let Some(pointer) = resp.hover_pos()
        && plot_rect.contains(pointer)
    {
        let frac = ((pointer.x - plot_rect.left()) / plot_rect.width().max(1.0)).clamp(0.0, 1.0);
        let target_x = x_min + (x_max - x_min) * frac as f64;
        let nearest = x_vals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - target_x).abs().total_cmp(&(*b - target_x).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let sx = to_screen(x_vals[nearest], y_min).x;
        painter.line_segment(
            [
                egui::pos2(sx, plot_rect.top()),
                egui::pos2(sx, plot_rect.bottom()),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(235, 240, 250, 150)),
        );

        let mut tip_lines = vec![format!("ArcLength: {:.4}", x_vals[nearest])];
        for (name, ys, color) in &series {
            let yv = ys[nearest];
            if yv.is_finite() {
                let pt = to_screen(x_vals[nearest], yv);
                painter.circle_filled(pt, 3.5, *color);
                tip_lines.push(format!("{name}: {yv:.4}"));
            }
        }

        let tip_galley = painter.layout(
            tip_lines.join("\n"),
            label_font,
            egui::Color32::WHITE,
            f32::INFINITY,
        );
        let tip_pos = egui::pos2(
            (sx + 10.0).min(plot_rect.right() - tip_galley.size().x - 10.0),
            (pointer.y - tip_galley.size().y * 0.5)
                .clamp(plot_rect.top() + 4.0, (plot_rect.bottom() - tip_galley.size().y - 8.0).max(plot_rect.top() + 4.0)),
        );
        let tip_bg = egui::Rect::from_min_size(
            tip_pos - egui::vec2(5.0, 3.0),
            tip_galley.size() + egui::vec2(10.0, 6.0),
        );
        painter.rect_filled(
            tip_bg,
            3.0,
            egui::Color32::from_rgba_unmultiplied(12, 16, 24, 215),
        );
        painter.galley(tip_pos, tip_galley, egui::Color32::WHITE);
    }
}

fn paint_histogram_chart(ui: &mut egui::Ui, hist: &HistogramResult) {
    let bins = hist.counts.len();
    if bins == 0 || hist.bin_edges.len() < 2 {
        ui.weak("Empty histogram.");
        return;
    }

    let max_count = hist.counts.iter().copied().max().unwrap_or(1).max(1);
    let avail = ui.available_size();
    let plot_h = avail.y.max(100.0);
    let plot_w = avail.x.max(200.0);
    let (outer_rect, resp) =
        ui.allocate_exact_size(egui::vec2(plot_w, plot_h), egui::Sense::hover());
    let painter = ui.painter_at(outer_rect);

    let left_pad = 54.0;
    let right_pad = 14.0;
    let top_pad = 10.0;
    let bottom_pad = 24.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(outer_rect.left() + left_pad, outer_rect.top() + top_pad),
        egui::pos2(
            (outer_rect.right() - right_pad).max(outer_rect.left() + left_pad + 40.0),
            (outer_rect.bottom() - bottom_pad).max(outer_rect.top() + top_pad + 40.0),
        ),
    );

    painter.rect_filled(plot_rect, 2.0, egui::Color32::from_rgb(18, 22, 30));

    let grid_color = egui::Color32::from_rgba_unmultiplied(120, 135, 160, 45);
    for g in 0..=4 {
        let frac = g as f32 / 4.0;
        let gy = egui::lerp(plot_rect.bottom()..=plot_rect.top(), frac);
        painter.line_segment(
            [
                egui::pos2(plot_rect.left(), gy),
                egui::pos2(plot_rect.right(), gy),
            ],
            egui::Stroke::new(1.0, grid_color),
        );
    }

    let hovered_bin = resp
        .hover_pos()
        .filter(|p| plot_rect.contains(*p))
        .map(|p| {
            let frac = ((p.x - plot_rect.left()) / plot_rect.width().max(1.0)).clamp(0.0, 0.9999);
            ((frac * bins as f32).floor() as usize).min(bins - 1)
        });

    let bin_w = plot_rect.width() / bins as f32;
    for (i, &cnt) in hist.counts.iter().enumerate() {
        let x0 = plot_rect.left() + i as f32 * bin_w + 1.0;
        let x1 = (plot_rect.left() + (i + 1) as f32 * bin_w - 1.0).max(x0 + 1.0);
        let h_frac = cnt as f32 / max_count as f32;
        let y0 = egui::lerp(plot_rect.bottom()..=plot_rect.top(), h_frac);
        let bar_rect =
            egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, plot_rect.bottom()));
        let col = if hovered_bin == Some(i) {
            egui::Color32::from_rgb(255, 195, 77)
        } else {
            egui::Color32::from_rgb(76, 175, 240)
        };
        painter.rect_filled(bar_rect, 1.0, col);
    }

    painter.rect_stroke(
        plot_rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 95, 120)),
        egui::StrokeKind::Inside,
    );

    let font = egui::FontId::monospace(10.0);
    let label_col = egui::Color32::from_rgb(200, 210, 225);
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.top()),
        egui::Align2::RIGHT_TOP,
        format!("{max_count}"),
        font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        "0",
        font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("{:.3}", hist.bin_edges[0]),
        font.clone(),
        label_col,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.bottom() + 4.0),
        egui::Align2::RIGHT_TOP,
        format!("{:.3}", hist.bin_edges[bins]),
        font.clone(),
        label_col,
    );

    if let Some(bi) = hovered_bin
        && let Some(pointer) = resp.hover_pos()
    {
        let lo = hist.bin_edges[bi];
        let hi = hist.bin_edges[bi + 1];
        let cnt = hist.counts[bi];
        let tip = format!("[{lo:.4}, {hi:.4}]: {cnt}");
        let galley = painter.layout_no_wrap(tip, font, egui::Color32::WHITE);
        let tip_pos = egui::pos2(
            (pointer.x + 10.0).min(plot_rect.right() - galley.size().x - 10.0),
            (plot_rect.top() + 6.0).max(pointer.y - 24.0),
        );
        let tip_bg = egui::Rect::from_min_size(
            tip_pos - egui::vec2(5.0, 3.0),
            galley.size() + egui::vec2(10.0, 6.0),
        );
        painter.rect_filled(
            tip_bg,
            3.0,
            egui::Color32::from_rgba_unmultiplied(12, 16, 24, 215),
        );
        painter.galley(tip_pos, galley, egui::Color32::WHITE);
    }
}
