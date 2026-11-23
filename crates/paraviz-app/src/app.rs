use crate::charts::{draw_chart_tab, draw_opacity_tf_editor, ChartMode, ChartState};
use crate::cli::format_memory_bytes;
use crate::modals::{
    draw_modal, is_raw_volume_ext, AppModal, ExportSpreadsheetModalState, FileBrowserState,
    GifModalState, GpuContourModalState, ModalAction, RawVolumeModalState, ScreenshotModalState,
};
use crate::theme::{
    apply_theme, node_badge_style, paint_logo_badge, palette, pill_badge, section_card,
    visibility_eye_button, UiTheme, ViewportBgPreset,
};
use paraviz::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Properties,
    Display,
    Information,
    Scene,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawerTab {
    Spreadsheet,
    Chart,
    RustTrace,
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
    Field,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickDomain {
    Point,
    Cell,
}

#[derive(Debug, Clone, PartialEq)]
struct PickedHighlight {
    node_id: NodeId,
    domain: PickDomain,
    element_index: usize,
    world_pos: Option<[f32; 3]>,
    scalar_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct ViewportCacheKey {
    scene_revision: u64,
    width: u32,
    height: u32,
    camera: Camera,
    settings: SceneSettings,
    widgets: Vec<ViewportWidget>,
    picked_highlight: Option<PickedHighlight>,
}

struct PipelineTreeRow {
    id: NodeId,
    depth: usize,
    name: String,
    visible: bool,
    badge: (&'static str, egui::Color32, egui::Color32),
}

#[derive(Debug, Clone)]
struct CameraBookmark {
    name: String,
    camera: Camera,
}

pub struct ParaVizApp {
    session: ParaViz,
    renderer: WebGpuRenderer,
    device: wgpu::Device,
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    viewport_tex_id: Option<egui::TextureId>,
    viewport_tex_size: (u32, u32),
    scene_revision: u64,
    viewport_cache_key: Option<ViewportCacheKey>,
    cached_scene: Option<RenderScene>,
    cached_total_points: usize,
    cached_total_cells: usize,

    theme: UiTheme,
    theme_applied: bool,
    selected_node: Option<NodeId>,
    renaming_node: Option<(NodeId, String)>,
    copied_filter: Option<(String, FilterConfig)>,
    copied_filter_chain: Vec<(String, FilterConfig)>,
    auto_apply: bool,
    pending_source: Option<(NodeId, SourceConfig)>,
    pending_filter: Option<(NodeId, FilterConfig)>,

    inspector_tab: InspectorTab,
    show_left_dock: bool,
    zen_mode: bool,
    show_drawer: bool,
    drawer_tab: DrawerTab,
    chart_state: ChartState,
    spreadsheet_target: SpreadsheetTarget,
    spreadsheet_offset: usize,
    spreadsheet_page_size: usize,
    spreadsheet_filter: String,
    spreadsheet_scroll_to_top: bool,
    show_scalar_bar: bool,
    show_widgets: bool,

    viewport_mode: ViewportMode,
    last_pick: Option<PickResult>,
    picked_highlight: Option<PickedHighlight>,
    highlight_flash_until: Option<Instant>,
    ruler_points: Vec<[f32; 3]>,
    export_glb_colors: bool,

    turntable_enabled: bool,
    turntable_speed: f32,

    playing: bool,
    time_accumulator: f64,
    play_fps: f64,

    last_frame_time: Instant,
    fps_smoothed: f32,
    last_render_ms: f32,

    recent_files: Vec<PathBuf>,
    camera_bookmarks: Vec<CameraBookmark>,
    info_stats_cache: HashMap<(NodeId, String, AttributeAssociation), HistogramResult>,
    volume_tf_hist_cache: HashMap<(NodeId, String, AttributeAssociation, Option<usize>), HistogramResult>,

    modal: Option<AppModal>,
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

        let theme = UiTheme::DarkStudio;
        apply_theme(&cc.egui_ctx, theme);

        let mut app = Self {
            session: ParaViz::new(),
            renderer,
            device,
            egui_renderer,
            viewport_tex_id: None,
            viewport_tex_size: (0, 0),
            scene_revision: 1,
            viewport_cache_key: None,
            cached_scene: None,
            cached_total_points: 0,
            cached_total_cells: 0,
            theme,
            theme_applied: true,
            selected_node: None,
            renaming_node: None,
            copied_filter: None,
            copied_filter_chain: Vec::new(),
            auto_apply: true,
            pending_source: None,
            pending_filter: None,
            inspector_tab: InspectorTab::Properties,
            show_left_dock: true,
            zen_mode: false,
            show_drawer: false,
            drawer_tab: DrawerTab::Spreadsheet,
            chart_state: ChartState::default(),
            spreadsheet_target: SpreadsheetTarget::Points,
            spreadsheet_offset: 0,
            spreadsheet_page_size: 200,
            spreadsheet_filter: String::new(),
            spreadsheet_scroll_to_top: false,
            show_scalar_bar: true,
            show_widgets: true,
            viewport_mode: ViewportMode::Orbit,
            last_pick: None,
            picked_highlight: None,
            highlight_flash_until: None,
            ruler_points: Vec::new(),
            export_glb_colors: true,
            turntable_enabled: false,
            turntable_speed: 0.45,
            playing: false,
            time_accumulator: 0.0,
            play_fps: 4.0,
            last_frame_time: Instant::now(),
            fps_smoothed: 60.0,
            last_render_ms: 0.0,
            recent_files: Vec::new(),
            camera_bookmarks: Vec::new(),
            info_stats_cache: HashMap::new(),
            volume_tf_hist_cache: HashMap::new(),
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

    fn invalidate_viewport(&mut self) {
        self.scene_revision = self.scene_revision.wrapping_add(1);
    }

    fn clear_stats_caches(&mut self) {
        self.info_stats_cache.clear();
        self.volume_tf_hist_cache.clear();
    }

    fn clear_stats_caches_for_node(&mut self, node_id: NodeId) {
        self.info_stats_cache.retain(|(nid, _, _), _| *nid != node_id);
        self.volume_tf_hist_cache
            .retain(|(nid, _, _, _), _| *nid != node_id);
    }

    fn focus_on_world_pos(&mut self, pos: [f32; 3]) {
        let p = glam::Vec3::from_array(pos);
        let offset = self.session.camera.position - self.session.camera.target;
        let dist = offset.length().max(1e-4);
        let dir = offset.normalize_or(glam::Vec3::Z);
        let target_dist = self
            .selected_node
            .and_then(|id| self.session.node(id).ok())
            .and_then(|n| n.dataset())
            .map(|ds| ds.bounds().diagonal() * 0.35)
            .filter(|d| *d > 1e-4)
            .map(|d| dist.min(d))
            .unwrap_or(dist);
        self.session.camera.target = p;
        self.session.camera.position = p + dir * target_dist;
        if self.session.camera.orthographic {
            self.session.camera.ortho_scale =
                (self.session.camera.ortho_scale * (target_dist / dist)).max(1e-5);
        }
        self.invalidate_viewport();
    }

    fn reveal_in_spreadsheet(&mut self, node_id: NodeId, domain: PickDomain, element_index: usize) {
        self.select_node(node_id);
        self.show_drawer = true;
        self.drawer_tab = DrawerTab::Spreadsheet;
        self.spreadsheet_target = match domain {
            PickDomain::Point => SpreadsheetTarget::Points,
            PickDomain::Cell => SpreadsheetTarget::Cells,
        };
        self.spreadsheet_offset = element_index;
        self.spreadsheet_scroll_to_top = true;
        self.highlight_flash_until = Some(Instant::now() + Duration::from_secs(2));
        self.status_message = match domain {
            PickDomain::Point => format!("Revealed Point #{element_index} in Spreadsheet"),
            PickDomain::Cell => format!("Revealed Cell #{element_index} in Spreadsheet"),
        };
    }

    fn highlight_from_spreadsheet_row(
        &mut self,
        node_id: NodeId,
        ds: &DataSet,
        domain: PickDomain,
        row: usize,
    ) {
        let world_pos = match domain {
            PickDomain::Point => dataset_point_at(ds, row),
            PickDomain::Cell => dataset_cell_center(ds, row),
        };
        let scalar_value = self
            .session
            .node(node_id)
            .ok()
            .and_then(|n| dataset_element_scalar(ds, &n.style.color_by, domain, row));
        self.picked_highlight = Some(PickedHighlight {
            node_id,
            domain,
            element_index: row,
            world_pos,
            scalar_value,
        });
        self.highlight_flash_until = Some(Instant::now() + Duration::from_secs(2));
        self.status_message = match domain {
            PickDomain::Point => format!("Highlighted Point #{row}"),
            PickDomain::Cell => format!("Highlighted Cell #{row}"),
        };
        self.invalidate_viewport();
    }

    fn push_recent_file(&mut self, path: &Path) {
        let p = path.to_path_buf();
        self.recent_files.retain(|existing| existing != &p);
        self.recent_files.insert(0, p);
        self.recent_files.truncate(8);
    }

    fn load_demo(&mut self, preset: DemoPreset) {
        match preset.build() {
            Ok(mut pv) => {
                let _ = pv.update_visible();
                self.selected_node = pv.nodes().last().map(|n| n.id);
                self.session = pv;
                self.pending_source = None;
                self.pending_filter = None;
                self.renaming_node = None;
                self.last_pick = None;
                self.picked_highlight = None;
                self.highlight_flash_until = None;
                self.ruler_points.clear();
                self.clear_stats_caches();
                self.playing = false;
                self.error_message = None;
                self.invalidate_viewport();
                self.status_message = format!("Loaded demo: {}", preset.name());
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load demo: {e}"));
            }
        }
    }

    fn load_path(&mut self, path: &Path) {
        if is_raw_volume_ext(path) {
            self.modal = Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                path.display().to_string(),
            )));
            return;
        }

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
                    self.renaming_node = None;
                    self.last_pick = None;
                    self.picked_highlight = None;
                    self.highlight_flash_until = None;
                    self.ruler_points.clear();
                    self.clear_stats_caches();
                    self.playing = false;
                    self.error_message = None;
                    self.invalidate_viewport();
                    self.push_recent_file(path);
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
                self.select_node(id);
                self.clear_stats_caches();
                self.error_message = None;
                self.invalidate_viewport();
                self.push_recent_file(path);
                self.status_message = format!("Opened '{}'", path.display());
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to open '{}': {e}", path.display()));
            }
        }
    }

    fn import_raw_volume(&mut self, rv: RawVolumeModalState) {
        let desc = RawVolumeDescriptor {
            dimensions: rv.dimensions,
            scalar_type: paraviz_core::ScalarType::F32, // overridden by read_raw_volume_with_type
            origin: rv.origin,
            spacing: rv.spacing,
            endian: rv.endian,
            header_bytes: rv.header_bytes,
            array_name: if rv.array_name.trim().is_empty() {
                "Scalars".into()
            } else {
                rv.array_name.trim().to_string()
            },
        };
        let path = PathBuf::from(rv.path.trim());
        match paraviz::io::read_raw_volume_with_type(&path, &desc, rv.scalar_type) {
            Ok(img) => {
                let is_first = self.session.nodes().is_empty();
                let base_name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("RawVolume");
                let label = self.unique_label(base_name);
                let id = self.session.add_dataset(label.clone(), DataSet::ImageData(img));
                let _ = self.session.evaluate(id);
                self.session.show(id).representation(Representation::Volume);
                if is_first {
                    let _ = self.session.camera_preset(CameraPreset::Isometric);
                } else {
                    let _ = self.session.reset_camera();
                }
                self.select_node(id);
                self.error_message = None;
                self.invalidate_viewport();
                self.push_recent_file(&path);
                self.status_message = format!(
                    "Imported raw volume '{label}' ({}×{}×{})",
                    rv.dimensions[0], rv.dimensions[1], rv.dimensions[2]
                );
            }
            Err(e) => {
                self.error_message = Some(format!("Raw volume import failed: {e}"));
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
                self.select_node(id);
                self.error_message = None;
                self.invalidate_viewport();
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
                FilterConfig::ProbeOverLine(f) if bb.is_valid() => {
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
        let is_wire_edges = matches!(
            cfg,
            FilterConfig::FeatureEdges(_) | FilterConfig::ExtractEdges(_)
        );
        let is_probe = matches!(cfg, FilterConfig::ProbeOverLine(_));
        let label = self.unique_label(cfg.default_label());
        match self.session.add_filter_cfg(label.clone(), parent_id, cfg) {
            Ok(id) => {
                let eval_res = self.session.evaluate(id);
                let handle = self.session.show(id);
                if is_cell_centers {
                    handle.representation(Representation::Points);
                } else if is_wire_edges {
                    handle.representation(Representation::Wireframe);
                }
                if let Err(e) = eval_res {
                    self.error_message = Some(format!("Filter '{label}' failed: {e}"));
                } else {
                    self.error_message = None;
                    self.status_message = format!("Applied filter '{label}'");
                }
                self.select_node(id);
                self.invalidate_viewport();
                if is_probe {
                    self.chart_state.mode = ChartMode::XYPlot;
                    self.drawer_tab = DrawerTab::Chart;
                    self.show_drawer = true;
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to add filter: {e}"));
            }
        }
    }

    fn open_gpu_contour_modal(&mut self) {
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
        let arrays: Vec<(String, (f64, f64))> = img
            .point_data
            .arrays()
            .iter()
            .map(|a| (a.name.clone(), a.range(None)))
            .collect();
        let (field_name, range) = img
            .point_data
            .active_scalars_array()
            .map(|a| (a.name.clone(), a.range(None)))
            .or_else(|| arrays.first().cloned())
            .unwrap_or_else(|| (String::new(), (0.0, 1.0)));
        let iso_value = 0.5 * (range.0 + range.1);
        self.modal = Some(AppModal::GpuContour(GpuContourModalState {
            node_id: sel_id,
            available_arrays: arrays,
            field_name,
            iso_value,
            range,
        }));
    }

    fn run_gpu_contour(&mut self, node_id: NodeId, field_name: &str, iso_value: f64) {
        let label = self.unique_label("GpuContour");
        let field_opt = if field_name.is_empty() {
            None
        } else {
            Some(field_name)
        };
        match self
            .session
            .gpu_contour(label.clone(), node_id, field_opt, iso_value)
        {
            Ok(new_id) => {
                let _ = self.session.evaluate(new_id);
                self.session.show(new_id);
                self.select_node(new_id);
                self.error_message = None;
                self.invalidate_viewport();
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
                self.select_node(id);
                self.error_message = None;
                self.invalidate_viewport();
                self.status_message = format!("Appended visible datasets -> '{label}'");
            }
            Err(e) => {
                self.error_message = Some(format!("Append Datasets failed: {e}"));
            }
        }
    }

    fn focus_selected_node(&mut self) {
        let Some(sel_id) = self.selected_node else {
            let _ = self.session.reset_camera();
            self.invalidate_viewport();
            return;
        };
        let _ = self.session.evaluate(sel_id);
        if let Ok(node) = self.session.node(sel_id)
            && let Some(ds) = node.dataset()
        {
            let bb = ds.bounds();
            if bb.is_valid() {
                self.session.camera.fit_bounds(&bb);
                self.invalidate_viewport();
                return;
            }
        }
        let _ = self.session.reset_camera();
        self.invalidate_viewport();
    }

    fn duplicate_selected_node(&mut self) {
        let Some(sel_id) = self.selected_node else {
            return;
        };
        let Ok(node) = self.session.node(sel_id).cloned() else {
            return;
        };
        let label = self.unique_label(&format!("{}_copy", node.name));
        let new_id_res = if let Some(src) = node.source_config {
            self.session.add_source_cfg(label.clone(), src)
        } else if let (Some(parent), Some(flt)) = (node.parent, node.filter_config) {
            self.session.add_filter_cfg(label.clone(), parent, flt)
        } else if let Some(ds) = node.cached_dataset {
            Ok(self.session.add_dataset(label.clone(), (*ds).clone()))
        } else {
            return;
        };
        match new_id_res {
            Ok(new_id) => {
                if let Ok(n) = self.session.node_mut(new_id) {
                    n.style = node.style;
                }
                let _ = self.session.evaluate(new_id);
                self.select_node(new_id);
                self.invalidate_viewport();
                self.status_message = format!("Duplicated node -> '{label}'");
            }
            Err(e) => {
                self.error_message = Some(format!("Duplicate failed: {e}"));
            }
        }
    }

    fn copy_filter_from_node(&mut self, id: NodeId) {
        let Ok(node) = self.session.node(id) else {
            return;
        };
        if let Some(cfg) = &node.filter_config {
            self.copied_filter = Some((node.name.clone(), cfg.clone()));
            self.status_message = format!("Copied filter '{}'", node.name);
        }
    }

    fn copy_filter_chain_from_node(&mut self, id: NodeId) {
        let mut chain = Vec::new();
        let mut cur = Some(id);
        while let Some(nid) = cur {
            let Ok(node) = self.session.node(nid) else {
                break;
            };
            if let Some(cfg) = &node.filter_config {
                chain.push((node.name.clone(), cfg.clone()));
                cur = node.parent;
            } else {
                break;
            }
        }
        chain.reverse();
        if !chain.is_empty() {
            self.copied_filter = chain.last().cloned();
            let count = chain.len();
            self.copied_filter_chain = chain;
            self.status_message = format!("Copied {count}-filter chain");
        }
    }

    fn paste_filter_to_node(&mut self, target_id: NodeId) {
        let Some((src_name, cfg)) = self.copied_filter.clone() else {
            return;
        };
        let label = self.unique_label(&src_name);
        match self.session.add_filter_cfg(label.clone(), target_id, cfg) {
            Ok(new_id) => {
                let eval_res = self.session.evaluate(new_id);
                self.session.show(new_id);
                if let Err(e) = eval_res {
                    self.error_message = Some(format!("Paste filter failed: {e}"));
                } else {
                    self.error_message = None;
                    self.status_message = format!("Pasted filter '{label}'");
                }
                self.select_node(new_id);
                self.invalidate_viewport();
            }
            Err(e) => {
                self.error_message = Some(format!("Paste filter failed: {e}"));
            }
        }
    }

    fn paste_filter_chain_to_node(&mut self, mut target_id: NodeId) {
        if self.copied_filter_chain.is_empty() {
            self.paste_filter_to_node(target_id);
            return;
        }
        let chain = self.copied_filter_chain.clone();
        let count = chain.len();
        for (idx, (src_name, cfg)) in chain.into_iter().enumerate() {
            let label = self.unique_label(&src_name);
            match self.session.add_filter_cfg(label, target_id, cfg) {
                Ok(new_id) => {
                    if idx + 1 < count {
                        self.session.hide(new_id);
                    } else {
                        self.session.show(new_id);
                    }
                    let _ = self.session.evaluate(new_id);
                    target_id = new_id;
                }
                Err(e) => {
                    self.error_message = Some(format!("Paste filter chain failed: {e}"));
                    return;
                }
            }
        }
        self.select_node(target_id);
        self.invalidate_viewport();
        self.status_message = format!("Pasted {count}-filter chain");
    }

    fn solo_selected_node(&mut self) {
        let Some(sel_id) = self.selected_node else {
            return;
        };
        let ids: Vec<NodeId> = self.session.nodes().iter().map(|n| n.id).collect();
        for id in ids {
            if id == sel_id {
                self.session.show(id);
            } else {
                self.session.hide(id);
            }
        }
        self.invalidate_viewport();
    }

    fn set_all_visibility(&mut self, visible: bool) {
        let ids: Vec<NodeId> = self.session.nodes().iter().map(|n| n.id).collect();
        for id in ids {
            if visible {
                self.session.show(id);
            } else {
                self.session.hide(id);
            }
        }
        self.invalidate_viewport();
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
            self.renaming_node = None;
            self.spreadsheet_offset = 0;
            let has_arc = self
                .session
                .node(id)
                .ok()
                .and_then(|n| n.dataset())
                .and_then(|ds| ds.as_poly_data())
                .is_some_and(|pd| pd.point_data.get("ArcLength").is_some());
            self.chart_state.mode = if has_arc {
                ChartMode::XYPlot
            } else {
                ChartMode::Histogram
            };
            self.invalidate_viewport();
        }
    }

    fn delete_selected_node(&mut self) {
        if let Some(id) = self.selected_node {
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
        self.clear_stats_caches();
        self.invalidate_viewport();
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

        if !ctx.egui_wants_keyboard_input() && self.modal.is_none() && self.renaming_node.is_none() {
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                let _ = self.session.reset_camera();
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::F)) {
                self.focus_selected_node();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                self.viewport_mode = ViewportMode::Orbit;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                self.viewport_mode = ViewportMode::Pick;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                self.viewport_mode = ViewportMode::Ruler;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::W)) {
                self.show_widgets = !self.show_widgets;
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::E)) {
                self.session.settings.edl_enabled = !self.session.settings.edl_enabled;
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::M)) {
                self.session.settings.msaa_samples = if self.session.settings.msaa_samples >= 4 {
                    1
                } else {
                    4
                };
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::G)) {
                self.session.settings.show_axes_gizmo = !self.session.settings.show_axes_gizmo;
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::B)) {
                self.session.settings.show_bounding_box =
                    !self.session.settings.show_bounding_box;
                self.invalidate_viewport();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::T)) {
                self.turntable_enabled = !self.turntable_enabled;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
                self.show_left_dock = !self.show_left_dock;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::F1))
                || ctx.input(|i| i.key_pressed(egui::Key::Questionmark))
            {
                self.modal = Some(AppModal::Shortcuts);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
            {
                self.delete_selected_node();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space))
                && !self.session.all_time_steps().is_empty()
            {
                self.playing = !self.playing;
            }
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::C))
                && let Some(sel) = self.selected_node
            {
                self.copy_filter_from_node(sel);
            }
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V))
                && let Some(sel) = self.selected_node
            {
                self.paste_filter_to_node(sel);
            }
        }
    }

    fn draw_menu_bar(&mut self, ui: &mut egui::Ui) {
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
                if ui.button("File Browser...").clicked() {
                    self.modal = Some(AppModal::FileBrowser(FileBrowserState::new()));
                    ui.close();
                }
                if ui.button("Open File by Path...").clicked() {
                    self.modal = Some(AppModal::OpenFile(String::new()));
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
                    self.modal = Some(AppModal::SaveState("paraviz_state.json".into()));
                    ui.close();
                }
                if ui.button("Load State JSON...").clicked() {
                    self.modal = Some(AppModal::LoadState("paraviz_state.json".into()));
                    ui.close();
                }
                ui.separator();
                let has_sel = self.selected_node.is_some();
                if ui
                    .add_enabled(has_sel, egui::Button::new("Export Active Dataset..."))
                    .clicked()
                {
                    self.modal = Some(AppModal::ExportDataset("export.vtp".into()));
                    ui.close();
                }
                if ui.button("Save Screenshot PNG...").clicked() {
                    let (w, h) = self.viewport_tex_size;
                    self.modal = Some(AppModal::SaveScreenshot(ScreenshotModalState {
                        path: "screenshot.png".into(),
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
                    self.modal = Some(AppModal::ExportGif(GifModalState {
                        path: "paraviz_animation.gif".into(),
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
                if ui.small_button("Browse").on_hover_text("Open File Browser").clicked() {
                    self.modal = Some(AppModal::FileBrowser(FileBrowserState::new()));
                }
            });
        });
    }

    fn draw_toolbar(&mut self, ui: &mut egui::Ui) {
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

    fn draw_left_dock(&mut self, ui: &mut egui::Ui) {
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

    fn draw_properties_tab(&mut self, ui: &mut egui::Ui) {
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
                    if let Err(e) = self.session.update_filter_cfg(sel_id, working) {
                        self.error_message = Some(format!("Update filter error: {e}"));
                    } else {
                        self.pending_filter = None;
                        self.error_message = None;
                        self.clear_stats_caches_for_node(sel_id);
                        self.invalidate_viewport();
                    }
                } else {
                    self.pending_filter = Some((sel_id, working));
                    self.invalidate_viewport();
                }
            } else if apply_clicked
                && let Some((id, pending)) = self.pending_filter.take()
                && id == sel_id
            {
                if let Err(e) = self.session.update_filter_cfg(sel_id, pending) {
                    self.error_message = Some(format!("Update filter error: {e}"));
                } else {
                    self.error_message = None;
                    self.clear_stats_caches_for_node(sel_id);
                    self.invalidate_viewport();
                }
            }
        } else {
            section_card(ui, self.theme, "In-Memory Dataset", |ui| {
                ui.label(
                    egui::RichText::new("Computed/merged node (no editable filter parameters).")
                        .color(pal.muted),
                );
            });
        }
    }

    fn draw_display_tab(&mut self, ui: &mut egui::Ui) {
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
    }

    fn draw_information_tab(&mut self, ui: &mut egui::Ui) {
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

        let bb = ds.bounds();
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

    fn draw_scene_tab(&mut self, ui: &mut egui::Ui) {
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
                    let _ = self.session.reset_camera();
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
        });

        if self.session.camera != prev_camera
            || self.session.settings != prev_settings
            || self.show_widgets != prev_widgets
        {
            self.invalidate_viewport();
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
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Field,
                    "Field Data",
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
            DrawerTab::Chart => {
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
                if let Some(res) = draw_chart_tab(ui, &ds, &mut self.chart_state, self.theme) {
                    match res {
                        Ok(msg) => {
                            self.error_message = None;
                            self.status_message = msg;
                        }
                        Err(err) => self.error_message = Some(err),
                    }
                }
            }
            DrawerTab::RustTrace => {
                let mut trace = self.session.to_rust_trace();
                ui.horizontal(|ui| {
                    if ui.button("Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(trace.clone());
                        self.status_message = "Copied Rust trace to clipboard".into();
                    }
                    if ui.button("Save trace.rs").clicked() {
                        match std::fs::write("paraviz_trace.rs", &trace) {
                            Ok(()) => {
                                self.error_message = None;
                                self.status_message = "Saved Rust trace -> paraviz_trace.rs".into();
                            }
                            Err(e) => {
                                self.error_message = Some(format!("Failed to write trace: {e}"));
                            }
                        }
                    }
                    ui.weak(format!("{} lines", trace.lines().count()));
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
        let node_name = node.name.clone();
        let Some(ds) = node.dataset().cloned() else {
            ui.weak("Dataset not evaluated.");
            return;
        };

        let total_rows = match self.spreadsheet_target {
            SpreadsheetTarget::Points => ds.num_points(),
            SpreadsheetTarget::Cells => ds.num_cells(),
            SpreadsheetTarget::Field => ds
                .field_data()
                .map(|f| f.arrays().iter().map(|a| a.num_tuples()).max().unwrap_or(0))
                .unwrap_or(0),
        };
        let page = self.spreadsheet_page_size.max(1);
        let max_offset = total_rows.saturating_sub(1);
        if self.spreadsheet_offset > max_offset {
            self.spreadsheet_offset = 0;
        }
        let end_row = (self.spreadsheet_offset + page).min(total_rows);

        let mut pending_highlight: Option<(PickDomain, usize, bool)> = None;
        let mut focus_world_pos: Option<[f32; 3]> = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Rows {}..{} of {}",
                if total_rows == 0 { 0 } else { self.spreadsheet_offset },
                end_row,
                total_rows
            ));
            if ui
                .add_enabled(self.spreadsheet_offset > 0, egui::Button::new("|<").small())
                .clicked()
            {
                self.spreadsheet_offset = 0;
            }
            if ui
                .add_enabled(self.spreadsheet_offset > 0, egui::Button::new("Prev").small())
                .clicked()
            {
                self.spreadsheet_offset = self.spreadsheet_offset.saturating_sub(page);
            }
            if ui
                .add_enabled(end_row < total_rows, egui::Button::new("Next").small())
                .clicked()
            {
                self.spreadsheet_offset = end_row;
            }
            egui::ComboBox::from_id_salt("spreadsheet_page_size")
                .selected_text(format!("{page}/page"))
                .show_ui(ui, |ui| {
                    for sz in [100usize, 200, 500] {
                        ui.selectable_value(&mut self.spreadsheet_page_size, sz, format!("{sz}/page"));
                    }
                });
            ui.separator();
            ui.label("Col Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.spreadsheet_filter)
                    .desired_width(120.0)
                    .hint_text("array name..."),
            );
            if ui.small_button("Export CSV").clicked() {
                let (domain_slug, domain_label) = match self.spreadsheet_target {
                    SpreadsheetTarget::Points => ("points", "Point Data"),
                    SpreadsheetTarget::Cells => ("cells", "Cell Data"),
                    SpreadsheetTarget::Field => ("field", "Field Data"),
                };
                let stem: String = node_name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let stem = stem.trim_matches('_');
                let default_path = if stem.is_empty() {
                    format!("spreadsheet_{domain_slug}.csv")
                } else {
                    format!("{stem}_{domain_slug}.csv")
                };
                self.modal = Some(AppModal::ExportSpreadsheetCsv(ExportSpreadsheetModalState {
                    path: default_path,
                    domain_label: domain_label.to_string(),
                    col_filter: self.spreadsheet_filter.clone(),
                    row_count: total_rows,
                }));
            }
            if let Some(hl) = &self.picked_highlight
                && hl.node_id == sel_id
            {
                ui.separator();
                let tag = match hl.domain {
                    PickDomain::Point => format!("Highlighted Point #{}", hl.element_index),
                    PickDomain::Cell => format!("Highlighted Cell #{}", hl.element_index),
                };
                ui.label(egui::RichText::new(tag).strong().color(palette(self.theme).warning));
                if let Some(pos) = hl.world_pos
                    && ui.small_button("Focus").clicked()
                {
                    focus_world_pos = Some(pos);
                }
            }
        });
        ui.separator();

        let col_needle = self.spreadsheet_filter.trim().to_ascii_lowercase();
        let matches_col = |name: &str| -> bool {
            col_needle.is_empty() || name.to_ascii_lowercase().contains(&col_needle)
        };

        let mut scroll = egui::ScrollArea::both().id_salt("spreadsheet_scroll");
        if self.spreadsheet_scroll_to_top {
            scroll = scroll.vertical_scroll_offset(0.0);
            self.spreadsheet_scroll_to_top = false;
        }

        let active_hl = self
            .picked_highlight
            .as_ref()
            .filter(|hl| hl.node_id == sel_id)
            .map(|hl| (hl.domain, hl.element_index));

        scroll.show(ui, |ui| match self.spreadsheet_target {
            SpreadsheetTarget::Points => {
                let arrays: Vec<&DataArray> = ds
                    .point_data()
                    .map(|p| {
                        p.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                egui::Grid::new("spreadsheet_points_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Point ID");
                        ui.strong("X");
                        ui.strong("Y");
                        ui.strong("Z");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();

                        for row in self.spreadsheet_offset..end_row {
                            let selected = active_hl == Some((PickDomain::Point, row));
                            let id_resp = ui
                                .selectable_label(selected, format!("{row}"))
                                .on_hover_text("Click to highlight in 3D • Double-click to focus");
                            if id_resp.double_clicked() {
                                pending_highlight = Some((PickDomain::Point, row, true));
                            } else if id_resp.clicked() {
                                pending_highlight = Some((PickDomain::Point, row, false));
                            }
                            if let Some(p) = dataset_point_at(&ds, row) {
                                ui.monospace(format!("{:.4}", p[0]));
                                ui.monospace(format!("{:.4}", p[1]));
                                ui.monospace(format!("{:.4}", p[2]));
                            } else {
                                ui.monospace("-");
                                ui.monospace("-");
                                ui.monospace("-");
                            }
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
            SpreadsheetTarget::Cells => {
                let arrays: Vec<&DataArray> = ds
                    .cell_data()
                    .map(|c| {
                        c.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                if total_rows == 0 {
                    ui.weak("Selected dataset has no cells.");
                    return;
                }
                if arrays.is_empty() && !col_needle.is_empty() {
                    ui.weak("No CellData arrays match the column filter.");
                    return;
                }
                egui::Grid::new("spreadsheet_cells_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Cell ID");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();
                        for row in self.spreadsheet_offset..end_row {
                            let selected = active_hl == Some((PickDomain::Cell, row));
                            let id_resp = ui
                                .selectable_label(selected, format!("{row}"))
                                .on_hover_text("Click to highlight in 3D • Double-click to focus");
                            if id_resp.double_clicked() {
                                pending_highlight = Some((PickDomain::Cell, row, true));
                            } else if id_resp.clicked() {
                                pending_highlight = Some((PickDomain::Cell, row, false));
                            }
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
            SpreadsheetTarget::Field => {
                let arrays: Vec<&DataArray> = ds
                    .field_data()
                    .map(|f| {
                        f.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                if arrays.is_empty() {
                    ui.weak("Selected dataset has no matching FieldData arrays.");
                    return;
                }
                egui::Grid::new("spreadsheet_field_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Tuple");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();
                        for row in self.spreadsheet_offset..end_row {
                            ui.monospace(format!("{row}"));
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
        });

        if let Some((domain, row, also_focus)) = pending_highlight {
            self.highlight_from_spreadsheet_row(sel_id, &ds, domain, row);
            if also_focus
                && let Some(pos) = self.picked_highlight.as_ref().and_then(|h| h.world_pos)
            {
                self.focus_on_world_pos(pos);
            }
        } else if let Some(pos) = focus_world_pos {
            self.focus_on_world_pos(pos);
        }
    }

    fn draw_viewport(&mut self, ui: &mut egui::Ui) {
        let pal = palette(self.theme);
        let avail = ui.available_size();
        let ppp = ui.ctx().pixels_per_point();
        let w = (avail.x * ppp).round().max(1.0) as u32;
        let h = (avail.y * ppp).round().max(1.0) as u32;

        self.sync_viewport_widgets();

        let cache_key = ViewportCacheKey {
            scene_revision: self.scene_revision,
            width: w,
            height: h,
            camera: self.session.camera.clone(),
            settings: self.session.settings.clone(),
            widgets: self.session.widgets.clone(),
            picked_highlight: self.picked_highlight.clone(),
        };
        let needs_render = self.viewport_tex_id.is_none()
            || self.viewport_cache_key.as_ref() != Some(&cache_key);

        if needs_render {
            let t0 = Instant::now();
            let scene_res = self.session.build_scene();
            if let Ok(scene) = scene_res {
                let mut total_points = 0usize;
                let mut total_cells = 0usize;
                for a in &scene.actors {
                    total_points += a.dataset.num_points();
                    total_cells += a.dataset.num_cells();
                }
                match self.renderer.render_to_texture(&scene, w, h) {
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
                        self.cached_scene = Some(scene);
                        self.cached_total_points = total_points;
                        self.cached_total_cells = total_cells;
                        self.viewport_cache_key = Some(cache_key);
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Render error: {e}"));
                    }
                }
            } else if let Err(e) = scene_res {
                self.error_message = Some(format!("Pipeline error: {e}"));
                self.cached_scene = None;
                self.cached_total_points = 0;
                self.cached_total_cells = 0;
                self.viewport_cache_key = Some(cache_key);
            }
            self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }

        let total_points = self.cached_total_points;
        let total_cells = self.cached_total_cells;

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
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.session
                    .camera
                    .pan(d.x / avail.y.max(1.0), d.y / avail.y.max(1.0));
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Secondary) {
                let d = resp.drag_delta();
                self.session.camera.dolly(-d.y * 0.008);
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            }

            if resp.hovered() {
                let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_y.abs() > 0.01 {
                    self.session.camera.dolly(scroll_y * 0.0025);
                    self.invalidate_viewport();
                    ui.ctx().request_repaint();
                }
            }

            let prev_hud_rect = ui.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("viewport_hud_rect")));
            if resp.clicked_by(egui::PointerButton::Primary)
                && let Some(pos) = resp.interact_pointer_pos()
                && prev_hud_rect.is_none_or(|hr| !hr.contains(pos))
                && let Some(scene) = &self.cached_scene
            {
                let local = pos - resp.rect.min;
                let px = local.x * ppp;
                let py = local.y * ppp;
                let hit = pick_scene(scene, px, py, w, h);
                match self.viewport_mode {
                    ViewportMode::Orbit | ViewportMode::Pick => {
                        self.picked_highlight = hit.as_ref().and_then(|h_res| {
                            let (domain, element_index) = h_res
                                .point_index
                                .map(|i| (PickDomain::Point, i))
                                .or_else(|| h_res.cell_index.map(|i| (PickDomain::Cell, i)))?;
                            Some(PickedHighlight {
                                node_id: NodeId(h_res.actor_id),
                                domain,
                                element_index,
                                world_pos: Some(h_res.world_pos),
                                scalar_value: h_res.scalar_value,
                            })
                        });
                        if self.picked_highlight.is_some() {
                            self.highlight_flash_until =
                                Some(Instant::now() + Duration::from_secs(2));
                        }
                        self.last_pick = hit;
                        self.invalidate_viewport();
                    }
                    ViewportMode::Ruler => {
                        if let Some(h_res) = hit {
                            if self.ruler_points.len() >= 2 {
                                self.ruler_points.clear();
                            }
                            self.ruler_points.push(h_res.world_pos);
                            self.picked_highlight = h_res
                                .point_index
                                .map(|i| (PickDomain::Point, i))
                                .or_else(|| h_res.cell_index.map(|i| (PickDomain::Cell, i)))
                                .map(|(domain, element_index)| PickedHighlight {
                                    node_id: NodeId(h_res.actor_id),
                                    domain,
                                    element_index,
                                    world_pos: Some(h_res.world_pos),
                                    scalar_value: h_res.scalar_value,
                                });
                            self.last_pick = Some(h_res);
                            self.sync_viewport_widgets();
                            self.invalidate_viewport();
                            ui.ctx().request_repaint();
                        }
                    }
                }
            }

            if resp.double_clicked() && self.viewport_mode == ViewportMode::Orbit {
                let _ = self.session.reset_camera();
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Initializing WebGPU viewport...");
            });
        }

        let painter = ui.painter_at(rect);

        // Screen-space 3D projected Pick reticle, Spreadsheet Highlight pin & Ruler endpoint pins
        let aspect = rect.width().max(1.0) / rect.height().max(1.0);
        let vp = self.session.camera.view_proj_matrix(aspect);
        if let Some(pick) = &self.last_pick
            && let Some(sp) = project_world_to_screen(vp, rect, pick.world_pos)
        {
            painter.circle_stroke(sp, 8.0, egui::Stroke::new(1.8, pal.accent));
            painter.circle_filled(sp, 2.5, egui::Color32::WHITE);
        }
        let now = Instant::now();
        let is_flashing = self
            .highlight_flash_until
            .is_some_and(|until| now < until);
        if is_flashing {
            ui.ctx().request_repaint_after(Duration::from_millis(80));
        } else {
            self.highlight_flash_until = None;
        }
        if let Some(hl) = &self.picked_highlight
            && let Some(wpos) = hl.world_pos
            && let Some(sp) = project_world_to_screen(vp, rect, wpos)
        {
            let ring_r = if is_flashing {
                let phase = ((now.elapsed().as_secs_f32() * 6.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
                10.0 + phase * 4.0
            } else {
                10.5
            };
            painter.circle_stroke(sp, ring_r, egui::Stroke::new(2.0, pal.warning));
            painter.circle_filled(sp, 3.0, pal.warning);
            let badge = match hl.domain {
                PickDomain::Point => format!("Pt #{}", hl.element_index),
                PickDomain::Cell => format!("Cell #{}", hl.element_index),
            };
            painter.text(
                sp + egui::vec2(12.0, -10.0),
                egui::Align2::LEFT_BOTTOM,
                badge,
                egui::FontId::monospace(11.0),
                pal.warning,
            );
        }
        for (idx, pt) in self.ruler_points.iter().enumerate() {
            if let Some(sp) = project_world_to_screen(vp, rect, *pt) {
                painter.circle_filled(sp, 5.0, pal.warning);
                painter.circle_stroke(sp, 7.5, egui::Stroke::new(1.5, egui::Color32::WHITE));
                painter.text(
                    sp + egui::vec2(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("P{}", idx + 1),
                    egui::FontId::monospace(11.0),
                    pal.warning,
                );
            }
        }
        if self.ruler_points.len() == 2 {
            let p1 = glam::Vec3::from_array(self.ruler_points[0]);
            let p2 = glam::Vec3::from_array(self.ruler_points[1]);
            let mid = ((p1 + p2) * 0.5).to_array();
            let dist = (p2 - p1).length();
            if let Some(sp) = project_world_to_screen(vp, rect, mid) {
                let badge = format!("{dist:.4}");
                let g = painter.layout_no_wrap(
                    badge,
                    egui::FontId::monospace(11.5),
                    pal.warning,
                );
                let bg = egui::Rect::from_center_size(sp, g.size() + egui::vec2(10.0, 6.0));
                painter.rect_filled(bg, 4.0, pal.hud_bg);
                painter.rect_stroke(
                    bg,
                    4.0,
                    egui::Stroke::new(1.0, pal.warning),
                    egui::StrokeKind::Inside,
                );
                painter.galley(bg.left_top() + egui::vec2(5.0, 3.0), g, pal.warning);
            }
        }

        // Top-left frosted HUD pill
        let proj_mode = if self.session.camera.orthographic {
            "Ortho"
        } else {
            "Persp"
        };
        let quality_label = if self.session.settings.edl_enabled {
            format!("EDL {:.1}", self.session.settings.edl_strength)
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
            "WebGPU  {:.0} FPS ({:.1} ms)  |  {} pts  {} cells  |  {}  |  {}  |  Mode: {}",
            self.fps_smoothed,
            self.last_render_ms,
            total_points,
            total_cells,
            proj_mode,
            quality_label,
            mode_label
        )];
        if let Some(pick) = &self.last_pick {
            let pt_str = pick.point_index.map_or_else(|| "-".into(), |i| i.to_string());
            let cell_str = pick.cell_index.map_or_else(|| "-".into(), |i| i.to_string());
            let scalar_str = pick
                .scalar_value
                .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            hud_lines.push(format!(
                "Pick: {}  Pos ({:.3}, {:.3}, {:.3})  Pt {}  Cell {}  Scalar {}",
                pick.actor_name,
                pick.world_pos[0],
                pick.world_pos[1],
                pick.world_pos[2],
                pt_str,
                cell_str,
                scalar_str
            ));
        }
        if let Some(hl) = &self.picked_highlight {
            let prefix = match hl.domain {
                PickDomain::Point => format!("Highlighted Point #{}", hl.element_index),
                PickDomain::Cell => format!("Highlighted Cell #{}", hl.element_index),
            };
            let pos_str = hl
                .world_pos
                .map_or_else(String::new, |p| format!("  Pos ({:.3}, {:.3}, {:.3})", p[0], p[1], p[2]));
            let scalar_str = hl
                .scalar_value
                .map_or_else(String::new, |v| format!("  Scalar {v:.4}"));
            hud_lines.push(format!("{prefix}{pos_str}{scalar_str}"));
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
                "Ruler: Distance = {:.4}  |  ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3})",
                dist, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z
            ));
        }

        let hud_pos = rect.left_top() + egui::vec2(12.0, 12.0);
        let galley = painter.layout(
            hud_lines.join("\n"),
            egui::FontId::monospace(11.5),
            egui::Color32::from_rgb(232, 238, 248),
            f32::INFINITY,
        );
        let has_pick_actions = self.last_pick.is_some() || self.picked_highlight.is_some();
        let action_bar_h = if has_pick_actions { 24.0 } else { 0.0 };
        let hud_bg = egui::Rect::from_min_size(
            hud_pos - egui::vec2(8.0, 6.0),
            galley.size() + egui::vec2(16.0, 12.0 + action_bar_h),
        );
        let border_col = if is_flashing {
            pal.warning
        } else {
            pal.hud_border
        };
        painter.rect_filled(hud_bg, 6.0, pal.hud_bg);
        painter.rect_stroke(
            hud_bg,
            6.0,
            egui::Stroke::new(if is_flashing { 1.6 } else { 1.0 }, border_col),
            egui::StrokeKind::Inside,
        );
        painter.galley(hud_pos, galley.clone(), egui::Color32::WHITE);

        if has_pick_actions {
            ui.data_mut(|d| d.insert_temp(egui::Id::NULL.with("viewport_hud_rect"), hud_bg));
            let btn_rect = egui::Rect::from_min_max(
                egui::pos2(hud_pos.x, hud_pos.y + galley.size().y + 4.0),
                egui::pos2(hud_bg.right() - 8.0, hud_bg.bottom() - 4.0),
            );
            let mut reveal_target: Option<(NodeId, PickDomain, usize)> = None;
            let mut focus_target: Option<[f32; 3]> = None;
            let mut clear_pick = false;
            let pick_clone = self.last_pick.clone();
            let hl_clone = self.picked_highlight.clone();
            let mut child = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(btn_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            if let Some(pick) = &pick_clone {
                if let Some((domain, idx)) = pick
                    .point_index
                    .map(|i| (PickDomain::Point, i))
                    .or_else(|| pick.cell_index.map(|i| (PickDomain::Cell, i)))
                    && child.small_button("Reveal in Spreadsheet").clicked()
                {
                    reveal_target = Some((NodeId(pick.actor_id), domain, idx));
                }
                if pick.point_index.is_some()
                    && let Some(ci) = pick.cell_index
                    && child.small_button(format!("Cell #{ci}")).clicked()
                {
                    reveal_target = Some((NodeId(pick.actor_id), PickDomain::Cell, ci));
                }
            } else if let Some(hl) = &hl_clone
                && child.small_button("Reveal in Spreadsheet").clicked()
            {
                reveal_target = Some((hl.node_id, hl.domain, hl.element_index));
            }
            if let Some(wpos) = hl_clone
                .as_ref()
                .and_then(|h| h.world_pos)
                .or_else(|| pick_clone.as_ref().map(|p| p.world_pos))
                && child.small_button("Focus").clicked()
            {
                focus_target = Some(wpos);
            }
            if child.small_button("Clear").clicked() {
                clear_pick = true;
            }
            if let Some((nid, dom, idx)) = reveal_target {
                self.reveal_in_spreadsheet(nid, dom, idx);
            }
            if let Some(pos) = focus_target {
                self.focus_on_world_pos(pos);
            }
            if clear_pick {
                self.last_pick = None;
                self.picked_highlight = None;
                self.highlight_flash_until = None;
                self.invalidate_viewport();
            }
        } else {
            ui.data_mut(|d| d.remove::<egui::Rect>(egui::Id::NULL.with("viewport_hud_rect")));
        }

        // Empty-state welcome launcher
        if self.session.nodes().is_empty() && self.modal.is_none() {
            let card_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(380.0, 160.0));
            if rect.contains(card_rect.min) && rect.contains(card_rect.max) {
                let mut child = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(card_rect)
                        .layout(egui::Layout::top_down(egui::Align::Center)),
                );
                egui::Frame::new()
                    .fill(pal.hud_bg)
                    .stroke(egui::Stroke::new(1.0, pal.hud_border))
                    .corner_radius(egui::CornerRadius::same(10))
                    .inner_margin(egui::Margin::same(16))
                    .show(&mut child, |ui| {
                        ui.heading("Welcome to ParaViz");
                        ui.label(
                            egui::RichText::new("Load a scientific dataset or launch a 3D demo")
                                .color(pal.muted),
                        );
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            if ui.button("Open Demo Gallery").clicked() {
                                self.modal = Some(AppModal::DemoGallery);
                            }
                            if ui.button("Browse Files").clicked() {
                                self.modal = Some(AppModal::FileBrowser(FileBrowserState::new()));
                            }
                        });
                    });
            }
        }

        // Scalar Color Bar Overlay (bottom-right of viewport)
        if self.show_scalar_bar
            && let Some((title, cmap, range)) = self.active_scalar_bar_info()
        {
            paint_scalar_bar_overlay(&painter, rect, &title, &cmap, range, self.theme);
        }

        // Drop target indicator
        if ui.input(|i| !i.raw.hovered_files.is_empty()) {
            painter.rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(56, 189, 248, 40),
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop scientific file, .raw volume, or .json state to open",
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
                let log_suffix = if pick_node.style.colormap.log_scale {
                    " (log)"
                } else {
                    ""
                };
                let title = format!("{array}{comp_suffix}{log_suffix}");
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

    fn handle_modals(&mut self, ctx: &egui::Context) {
        let action = draw_modal(
            ctx,
            &mut self.modal,
            &mut self.export_glb_colors,
            self.viewport_tex_size,
            self.theme,
        );
        match action {
            ModalAction::None => {}
            ModalAction::OpenFile(p) | ModalAction::LoadState(p) => self.load_path(&p),
            ModalAction::ImportRawVolume(rv) => self.import_raw_volume(rv),
            ModalAction::RunGpuContour {
                node_id,
                field_name,
                iso_value,
            } => self.run_gpu_contour(node_id, &field_name, iso_value),
            ModalAction::LoadDemo(preset) => self.load_demo(preset),
            ModalAction::SaveState(p) => match self.session.save_state_json() {
                Ok(json) => match std::fs::write(&p, json) {
                    Ok(()) => {
                        self.error_message = None;
                        self.status_message = format!("Saved state -> {}", p.display());
                    }
                    Err(e) => self.error_message = Some(format!("Write failed: {e}")),
                },
                Err(e) => self.error_message = Some(format!("Serialize failed: {e}")),
            },
            ModalAction::ExportDataset(p) => {
                if let Some(id) = self.selected_node {
                    let ext = p
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|e| e.to_ascii_lowercase())
                        .unwrap_or_default();
                    let res = if ext == "glb" {
                        self.session.export_glb(id, &p, self.export_glb_colors)
                    } else if ext == "html" || ext == "htm" {
                        self.session.export_html(id, &p, self.export_glb_colors)
                    } else {
                        self.session.export_dataset(id, &p)
                    };
                    match res {
                        Ok(()) => {
                            self.error_message = None;
                            self.status_message = format!("Exported dataset -> {}", p.display());
                        }
                        Err(e) => self.error_message = Some(format!("Export failed: {e}")),
                    }
                }
            }
            ModalAction::ExportSpreadsheetCsv(p) => {
                if let Some(sel_id) = self.selected_node
                    && let Ok(node) = self.session.node(sel_id)
                    && let Some(ds) = node.dataset().cloned()
                {
                    match export_spreadsheet_csv(
                        &ds,
                        self.spreadsheet_target,
                        &self.spreadsheet_filter,
                        &p,
                    ) {
                        Ok(msg) => {
                            self.error_message = None;
                            self.status_message = msg;
                        }
                        Err(err) => self.error_message = Some(err),
                    }
                } else {
                    self.error_message =
                        Some("No evaluated dataset selected for CSV export.".into());
                }
            }
            ModalAction::SaveScreenshot {
                path,
                width,
                height,
                transparent,
                include_scalar_bar,
            } => {
                let res = self.session.build_scene().and_then(|mut scene| {
                    scene.settings.transparent_background = transparent;
                    scene.settings.show_scalar_bar = include_scalar_bar;
                    self.renderer
                        .render_to_png(&scene, width, height, &path)
                        .map_err(SessionError::from)
                });
                match res {
                    Ok(()) => {
                        self.error_message = None;
                        self.status_message = format!(
                            "Saved screenshot ({width}x{height}) -> {}",
                            path.display()
                        );
                    }
                    Err(e) => self.error_message = Some(format!("Screenshot failed: {e}")),
                }
            }
            ModalAction::ExportGif {
                path,
                width,
                height,
                frames,
                delay_ms,
                is_turntable,
                transparent,
                include_scalar_bar,
            } => {
                let prev_trans = self.session.settings.transparent_background;
                let prev_bar = self.session.settings.show_scalar_bar;
                self.session.settings.transparent_background = transparent;
                self.session.settings.show_scalar_bar = include_scalar_bar;
                let res = if is_turntable {
                    self.session
                        .render_turntable_gif(&path, width, height, frames, delay_ms)
                } else {
                    self.session
                        .render_timeseries_gif(&path, width, height, delay_ms)
                };
                self.session.settings.transparent_background = prev_trans;
                self.session.settings.show_scalar_bar = prev_bar;
                match res {
                    Ok(()) => {
                        self.error_message = None;
                        self.status_message = format!(
                            "Exported animated GIF ({width}x{height}) -> {}",
                            path.display()
                        );
                    }
                    Err(e) => self.error_message = Some(format!("GIF export failed: {e}")),
                }
            }
        }
    }
}

impl eframe::App for ParaVizApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            apply_theme(ui.ctx(), self.theme);
            self.theme_applied = true;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;
        if dt > 1e-5 {
            let inst_fps = (1.0 / dt) as f32;
            self.fps_smoothed = self.fps_smoothed * 0.9 + inst_fps * 0.1;
        }

        if self.turntable_enabled {
            self.session
                .camera
                .orbit((dt as f32) * self.turntable_speed, 0.0);
            self.invalidate_viewport();
            ui.ctx().request_repaint();
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
            if !self.zen_mode {
                ui.add_space(2.0);
                self.draw_toolbar(ui);
            }
        });

        if !self.zen_mode && self.show_left_dock {
            egui::Panel::left("paraviz_left_dock")
                .default_size(325.0)
                .min_size(250.0)
                .show(ui, |ui| {
                    self.draw_left_dock(ui);
                });
        }

        let pal = palette(self.theme);
        egui::Panel::bottom("paraviz_status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(err) = &self.error_message {
                    pill_badge(
                        ui,
                        "ERR",
                        pal.danger.gamma_multiply(0.25),
                        pal.danger,
                    );
                    ui.colored_label(pal.danger, err);
                    if ui
                        .small_button("Report Issue")
                        .on_hover_text(
                            "Encountered a bug? Open an issue at github.com/DimitriTimoz/paraviz/issues",
                        )
                        .clicked()
                    {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(
                            "https://github.com/DimitriTimoz/paraviz/issues/new/choose",
                        ));
                    }
                    if ui.small_button("Dismiss").clicked() {
                        self.error_message = None;
                    }
                } else {
                    pill_badge(
                        ui,
                        "OK",
                        pal.success.gamma_multiply(0.22),
                        pal.success,
                    );
                    ui.label(&self.status_message);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let total_mem: usize = self
                        .session
                        .nodes()
                        .iter()
                        .filter_map(|n| n.dataset().map(|d| d.memory_bytes()))
                        .sum();
                    ui.label(
                        egui::RichText::new(format!(
                            "Nodes: {}  |  Mem: {}  |  Dist: {:.2}  |  F1: Help",
                            self.session.nodes().len(),
                            format_memory_bytes(total_mem),
                            self.session.camera.distance()
                        ))
                        .size(11.0)
                        .monospace()
                        .color(pal.muted),
                    );
                });
            });
        });

        if !self.zen_mode && self.show_drawer {
            egui::Panel::bottom("paraviz_inspector_drawer")
                .resizable(true)
                .default_size(240.0)
                .min_size(130.0)
                .show(ui, |ui| {
                    self.draw_drawer(ui);
                });
        }

        egui::CentralPanel::no_frame().show(ui, |ui| {
            self.draw_viewport(ui);
        });

        self.handle_modals(ui.ctx());
    }
}

fn project_world_to_screen(
    vp: glam::Mat4,
    rect: egui::Rect,
    world: [f32; 3],
) -> Option<egui::Pos2> {
    let clip = vp * glam::Vec4::new(world[0], world[1], world[2], 1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !(0.0..=1.0).contains(&ndc.z) {
        return None;
    }
    let sx = rect.left() + (ndc.x * 0.5 + 0.5) * rect.width();
    let sy = rect.top() + (0.5 - ndc.y * 0.5) * rect.height();
    let p = egui::pos2(sx, sy);
    rect.contains(p).then_some(p)
}

fn polydata_surface_area(pd: &PolyData) -> f64 {
    let mut area = 0.0f64;
    let pts = &pd.points;
    for poly in &pd.polys {
        if poly.len() < 3 {
            continue;
        }
        let Some(&p0) = pts.get(poly[0] as usize) else {
            continue;
        };
        let v0 = glam::Vec3::from_array(p0);
        for k in 1..(poly.len() - 1) {
            if let (Some(&p1), Some(&p2)) =
                (pts.get(poly[k] as usize), pts.get(poly[k + 1] as usize))
            {
                let e1 = glam::Vec3::from_array(p1) - v0;
                let e2 = glam::Vec3::from_array(p2) - v0;
                area += 0.5 * e1.cross(e2).length() as f64;
            }
        }
    }
    for strip in &pd.strips {
        for k in 0..strip.len().saturating_sub(2) {
            if let (Some(&p0), Some(&p1), Some(&p2)) = (
                pts.get(strip[k] as usize),
                pts.get(strip[k + 1] as usize),
                pts.get(strip[k + 2] as usize),
            ) {
                let v0 = glam::Vec3::from_array(p0);
                let e1 = glam::Vec3::from_array(p1) - v0;
                let e2 = glam::Vec3::from_array(p2) - v0;
                area += 0.5 * e1.cross(e2).length() as f64;
            }
        }
    }
    area
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn append_array_headers(csv: &mut String, arrays: &[&DataArray]) {
    for arr in arrays {
        if arr.num_components == 1 {
            csv.push(',');
            csv.push_str(&csv_escape(&arr.name));
        } else {
            for c in 0..arr.num_components {
                csv.push(',');
                csv.push_str(&csv_escape(&format!("{}_{}", arr.name, c)));
            }
        }
    }
    csv.push('\n');
}

fn append_array_row(csv: &mut String, arrays: &[&DataArray], row: usize) {
    for arr in arrays {
        if row < arr.num_tuples() {
            for v in arr.get_tuple(row) {
                csv.push_str(&format!(",{v}"));
            }
        } else {
            for _ in 0..arr.num_components {
                csv.push_str(",0");
            }
        }
    }
    csv.push('\n');
}

fn export_spreadsheet_csv(
    ds: &DataSet,
    target: SpreadsheetTarget,
    col_filter: &str,
    path: &Path,
) -> Result<String, String> {
    let col_needle = col_filter.trim().to_ascii_lowercase();
    let matches_col = |name: &str| -> bool {
        col_needle.is_empty() || name.to_ascii_lowercase().contains(&col_needle)
    };

    let mut csv = String::new();
    let (domain_label, n) = match target {
        SpreadsheetTarget::Points => {
            let arrays: Vec<&DataArray> = ds
                .point_data()
                .map(|p| p.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            let n = ds.num_points();
            csv.push_str("PointID,X,Y,Z");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                if let Some(p) = dataset_point_at(ds, row) {
                    csv.push_str(&format!("{row},{},{},{}", p[0], p[1], p[2]));
                } else {
                    csv.push_str(&format!("{row},0,0,0"));
                }
                append_array_row(&mut csv, &arrays, row);
            }
            ("Point", n)
        }
        SpreadsheetTarget::Cells => {
            let arrays: Vec<&DataArray> = ds
                .cell_data()
                .map(|c| c.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            let n = ds.num_cells();
            if n == 0 {
                return Err("Selected dataset has no cells to export.".into());
            }
            if !col_needle.is_empty() && arrays.is_empty() {
                return Err(format!(
                    "No CellData columns match filter '{}'.",
                    col_filter.trim()
                ));
            }
            csv.push_str("CellID");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                csv.push_str(&format!("{row}"));
                append_array_row(&mut csv, &arrays, row);
            }
            ("Cell", n)
        }
        SpreadsheetTarget::Field => {
            let arrays: Vec<&DataArray> = ds
                .field_data()
                .map(|f| f.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            if arrays.is_empty() {
                return Err(if col_needle.is_empty() {
                    "Selected dataset has no FieldData arrays to export.".into()
                } else {
                    format!(
                        "No FieldData columns match filter '{}'.",
                        col_filter.trim()
                    )
                });
            }
            let n = arrays.iter().map(|a| a.num_tuples()).max().unwrap_or(0);
            csv.push_str("Tuple");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                csv.push_str(&format!("{row}"));
                append_array_row(&mut csv, &arrays, row);
            }
            ("Field", n)
        }
    };

    std::fs::write(path, csv).map_err(|e| format!("Failed to export CSV: {e}"))?;
    Ok(format!("Exported {n} {domain_label} rows -> {}", path.display()))
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

fn array_picker_combo(
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

fn edit_plane_snap_buttons(
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

fn edit_filter_config(
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
        3.0,
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
    theme: UiTheme,
) {
    let pal = palette(theme);
    let bar_w = 18.0;
    let bar_h = (viewport_rect.height() * 0.34).clamp(115.0, 225.0);
    let pad = 14.0;
    let box_w = 142.0;
    let box_h = bar_h + 44.0;

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
    painter.rect_filled(box_rect, 7.0, pal.hud_bg);
    painter.rect_stroke(
        box_rect,
        7.0,
        egui::Stroke::new(1.0, pal.hud_border),
        egui::StrokeKind::Inside,
    );

    let title_col = match theme {
        UiTheme::DarkStudio => egui::Color32::from_rgb(235, 240, 248),
        UiTheme::LightStudio => egui::Color32::from_rgb(25, 32, 48),
    };
    painter.text(
        egui::pos2(box_rect.left() + 10.0, box_rect.top() + 8.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(12.0),
        title_col,
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
        1.0,
        egui::Stroke::new(1.0, pal.hud_border),
        egui::StrokeKind::Inside,
    );

    let mid = 0.5 * (range.0 + range.1);
    let label_x = bar_rect.right() + 9.0;
    let font = egui::FontId::monospace(10.5);
    painter.text(
        egui::pos2(label_x, bar_rect.top()),
        egui::Align2::LEFT_TOP,
        format!("{:.4}", range.1),
        font.clone(),
        title_col,
    );
    painter.text(
        egui::pos2(label_x, bar_rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{mid:.4}"),
        font.clone(),
        title_col,
    );
    painter.text(
        egui::pos2(label_x, bar_rect.bottom()),
        egui::Align2::LEFT_BOTTOM,
        format!("{:.4}", range.0),
        font,
        title_col,
    );
}

fn dataset_point_at(ds: &DataSet, row: usize) -> Option<[f32; 3]> {
    match ds {
        DataSet::PolyData(p) => p.points.get(row).copied(),
        DataSet::ImageData(i) => {
            if row >= i.num_points() {
                return None;
            }
            let [nx, ny, _] = i.dimensions;
            let k = row / (nx * ny).max(1);
            let rem = row % (nx * ny).max(1);
            let j = rem / nx.max(1);
            let ii = rem % nx.max(1);
            Some(i.point_coords(ii, j, k))
        }
        DataSet::RectilinearGrid(r) => {
            if row >= r.num_points() {
                return None;
            }
            let [nx, ny, _] = r.dimensions();
            let k = row / (nx * ny).max(1);
            let rem = row % (nx * ny).max(1);
            let j = rem / nx.max(1);
            let ii = rem % nx.max(1);
            Some(r.point_coords(ii, j, k))
        }
        DataSet::StructuredGrid(s) => s.points.get(row).copied(),
        DataSet::UnstructuredGrid(u) => u.points.get(row).copied(),
        DataSet::MultiBlock(mb) => mb
            .blocks
            .first()
            .and_then(|(_, b)| dataset_point_at(b, row)),
    }
}

fn centroid_from_indices(points: &[[f32; 3]], indices: &[u32]) -> Option<[f32; 3]> {
    if indices.is_empty() {
        return None;
    }
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0usize;
    for &idx in indices {
        if let Some(&p) = points.get(idx as usize) {
            sum += glam::Vec3::from_array(p);
            count += 1;
        }
    }
    (count > 0).then(|| (sum / count as f32).to_array())
}

fn structured_cell_center(
    dims: [usize; 3],
    row: usize,
    coords_fn: impl Fn(usize, usize, usize) -> [f32; 3],
) -> Option<[f32; 3]> {
    let cx = dims[0].saturating_sub(1).max(1);
    let cy = dims[1].saturating_sub(1).max(1);
    let cz = dims[2].saturating_sub(1).max(1);
    if dims.contains(&0) || row >= cx * cy * cz {
        return None;
    }
    let k = row / (cx * cy);
    let rem = row % (cx * cy);
    let j = rem / cx;
    let i = rem % cx;
    let i1 = (i + 1).min(dims[0] - 1);
    let j1 = (j + 1).min(dims[1] - 1);
    let k1 = (k + 1).min(dims[2] - 1);
    let p0 = glam::Vec3::from_array(coords_fn(i, j, k));
    let p1 = glam::Vec3::from_array(coords_fn(i1, j1, k1));
    Some(((p0 + p1) * 0.5).to_array())
}

fn dataset_cell_center(ds: &DataSet, row: usize) -> Option<[f32; 3]> {
    match ds {
        DataSet::PolyData(p) => {
            if let Some(v) = p.verts.get(row) {
                return centroid_from_indices(&p.points, std::slice::from_ref(v));
            }
            let r1 = row.saturating_sub(p.verts.len());
            if row >= p.verts.len() && r1 < p.lines.len() {
                return centroid_from_indices(&p.points, &p.lines[r1]);
            }
            let r2 = r1.saturating_sub(p.lines.len());
            if row >= p.verts.len() + p.lines.len() && r2 < p.polys.len() {
                return centroid_from_indices(&p.points, &p.polys[r2]);
            }
            let r3 = r2.saturating_sub(p.polys.len());
            if row >= p.verts.len() + p.lines.len() + p.polys.len() && r3 < p.strips.len() {
                return centroid_from_indices(&p.points, &p.strips[r3]);
            }
            None
        }
        DataSet::ImageData(i) => structured_cell_center(i.dimensions, row, |ii, jj, kk| {
            i.point_coords(ii, jj, kk)
        }),
        DataSet::RectilinearGrid(r) => {
            structured_cell_center(r.dimensions(), row, |ii, jj, kk| r.point_coords(ii, jj, kk))
        }
        DataSet::StructuredGrid(s) => {
            structured_cell_center(s.dimensions, row, |ii, jj, kk| s.point_coords(ii, jj, kk))
        }
        DataSet::UnstructuredGrid(u) => {
            if row >= u.num_cells() {
                return None;
            }
            centroid_from_indices(&u.points, u.cell_points(row))
        }
        DataSet::MultiBlock(mb) => mb
            .blocks
            .first()
            .and_then(|(_, b)| dataset_cell_center(b, row)),
    }
}

fn dataset_element_scalar(
    ds: &DataSet,
    color_by: &ColorBy,
    domain: PickDomain,
    row: usize,
) -> Option<f64> {
    let fd = match domain {
        PickDomain::Point => ds.point_data()?,
        PickDomain::Cell => ds.cell_data()?,
    };
    let (arr, comp) = match (domain, color_by) {
        (PickDomain::Point, ColorBy::PointScalar { array, component })
        | (PickDomain::Cell, ColorBy::CellScalar { array, component }) => {
            let a = if array.is_empty() {
                fd.active_scalars_array()?
            } else {
                fd.get(array)?
            };
            (a, *component)
        }
        _ => (fd.active_scalars_array()?, None),
    };
    if row >= arr.num_tuples() {
        return None;
    }
    Some(match comp {
        Some(c) if c < arr.num_components => arr.data.get_f64(row * arr.num_components + c),
        _ => arr.get_magnitude(row),
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset() -> DataSet {
        let mut pd = PolyData::from_triangles(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            vec![[0, 1, 2]],
        );
        pd.point_data.add_array(DataArray::scalars_f64("Temp", vec![10.0, 20.0, 30.0]));
        pd.point_data.add_array(DataArray::scalars_f64("Pressure", vec![1.0, 2.0, 3.0]));
        pd.cell_data.add_array(DataArray::scalars_f64("CellArea", vec![2.0]));
        pd.cell_data.add_array(DataArray::scalars_f64("ZoneID", vec![7.0]));
        pd.field_data.add_array(DataArray::scalars_f64("TimeValue", vec![42.5]));
        DataSet::PolyData(pd)
    }

    #[test]
    fn export_spreadsheet_csv_respects_active_domain_and_filter() {
        let ds = sample_dataset();
        let dir = std::env::temp_dir();

        // 1. Points domain with column filter
        let pts_path = dir.join(format!("paraviz_test_pts_{}.csv", std::process::id()));
        let msg = export_spreadsheet_csv(&ds, SpreadsheetTarget::Points, "press", &pts_path)
            .expect("points export");
        assert!(msg.contains("3 Point rows"));
        let pts_csv = std::fs::read_to_string(&pts_path).unwrap();
        assert!(pts_csv.starts_with("PointID,X,Y,Z,Pressure\n"));
        assert!(!pts_csv.contains("Temp"));
        let _ = std::fs::remove_file(&pts_path);

        // 2. Cells domain
        let cells_path = dir.join(format!("paraviz_test_cells_{}.csv", std::process::id()));
        let msg = export_spreadsheet_csv(&ds, SpreadsheetTarget::Cells, "Zone", &cells_path)
            .expect("cells export");
        assert!(msg.contains("1 Cell rows"));
        let cells_csv = std::fs::read_to_string(&cells_path).unwrap();
        assert_eq!(cells_csv, "CellID,ZoneID\n0,7\n");
        let _ = std::fs::remove_file(&cells_path);

        // 3. Field domain
        let field_path = dir.join(format!("paraviz_test_field_{}.csv", std::process::id()));
        let msg = export_spreadsheet_csv(&ds, SpreadsheetTarget::Field, "", &field_path)
            .expect("field export");
        assert!(msg.contains("1 Field rows"));
        let field_csv = std::fs::read_to_string(&field_path).unwrap();
        assert_eq!(field_csv, "Tuple,TimeValue\n0,42.5\n");
        let _ = std::fs::remove_file(&field_path);
    }

    #[test]
    fn viewport_cache_key_detects_mutations_and_stays_equal_when_idle() {
        let base = ViewportCacheKey {
            scene_revision: 1,
            width: 1280,
            height: 720,
            camera: Camera::default(),
            settings: SceneSettings::default(),
            widgets: Vec::new(),
            picked_highlight: None,
        };
        assert_eq!(base, base.clone());

        let mut resized = base.clone();
        resized.width = 1281;
        assert_ne!(base, resized);

        let mut cam_moved = base.clone();
        cam_moved.camera.fov_y_deg = 55.0;
        assert_ne!(base, cam_moved);

        let mut hl_changed = base.clone();
        hl_changed.picked_highlight = Some(PickedHighlight {
            node_id: NodeId(1),
            domain: PickDomain::Point,
            element_index: 4,
            world_pos: Some([1.0, 2.0, 3.0]),
            scalar_value: Some(0.5),
        });
        assert_ne!(base, hl_changed);
    }

    #[test]
    fn dataset_cell_center_computes_triangle_centroid() {
        let ds = sample_dataset();
        let c = dataset_cell_center(&ds, 0).expect("triangle centroid");
        assert!((c[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((c[1] - 2.0 / 3.0).abs() < 1e-5);
        assert!(c[2].abs() < 1e-5);
    }
}
