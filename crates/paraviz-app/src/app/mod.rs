pub(crate) use crate::charts::{draw_chart_tab, draw_opacity_tf_editor, ChartMode, ChartState};
pub(crate) use crate::cli::format_memory_bytes;
pub(crate) use crate::modals::{
    draw_modal, is_raw_volume_ext, native_save_file, AppModal, ExportAbModalState, GifModalState,
    GpuContourModalState, ModalAction, RawVolumeModalState, ScreenshotModalState,
};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use crate::modals::{native_pick_scientific_file, native_pick_state_file};
#[cfg(target_arch = "wasm32")]
pub(crate) use crate::modals::{spawn_wasm_open_dialog, wasm_download_bytes};
pub(crate) use crate::theme::{
    apply_theme, node_badge_style, paint_logo_badge, palette, pill_badge, section_card,
    visibility_eye_button, UiTheme, ViewportBgPreset,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::updater::{
    current_version, spawn_check_updates, spawn_install_update, InstallKind, ReleaseUpdateInfo,
    UpdateEvent, UpdateStatus, RELEASES_PAGE_URL,
};
use paraviz::PipelineState;
use paraviz::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::Arc;
use web_time::{Duration, Instant};

mod drawer;
mod editors;
mod inspector;
mod menu_pipeline;
#[cfg(test)]
mod showcase_video;
mod viewport;
mod temporal_probe;

use drawer::{dataset_cell_center, dataset_element_scalar, dataset_point_at, export_spreadsheet_csv};

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
    TemporalProbe,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewportLayout {
    Single,
    SplitHorizontal,
    SplitVertical,
    Quad,
}

impl ViewportLayout {
    pub(crate) const ALL: &'static [Self] = &[
        Self::Single,
        Self::SplitHorizontal,
        Self::SplitVertical,
        Self::Quad,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Single => "1x1",
            Self::SplitHorizontal => "1x2",
            Self::SplitVertical => "2x1",
            Self::Quad => "2x2",
        }
    }

    pub(crate) fn menu_label(self) -> &'static str {
        match self {
            Self::Single => "1x1 Single",
            Self::SplitHorizontal => "1x2 Split Horizontal",
            Self::SplitVertical => "2x1 Split Vertical",
            Self::Quad => "2x2 Quad",
        }
    }

    pub(crate) fn pane_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::SplitHorizontal | Self::SplitVertical => 2,
            Self::Quad => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WorkspaceProfile {
    #[default]
    Full,
    Cfd,
    Medical,
    Mesh,
}

impl WorkspaceProfile {
    pub(crate) const ALL: &'static [Self] = &[Self::Full, Self::Cfd, Self::Medical, Self::Mesh];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Cfd => "CFD",
            Self::Medical => "Medical",
            Self::Mesh => "Mesh",
        }
    }

    pub(crate) fn matches_filter(self, filter_label: &str) -> bool {
        match self {
            Self::Full => true,
            Self::Cfd => matches!(
                filter_label,
                "Slice"
                    | "Clip"
                    | "Threshold"
                    | "Contour (Iso)"
                    | "Stream Tracer"
                    | "Warp by Vector"
                    | "Glyph (Arrows / Spheres)"
                    | "Calculator"
                    | "Gradient"
                    | "Vorticity / Q-Criterion"
                    | "Plot Over Line"
                    | "Probe Location"
                    | "Integrate Variables"
                    | "Cell Data to Point Data"
                    | "Point Data to Cell Data"
            ),
            Self::Medical => matches!(
                filter_label,
                "Slice"
                    | "Clip"
                    | "Threshold"
                    | "Contour (Iso)"
                    | "GPU Marching Cubes (ImageData)"
                    | "Extract Subset (VOI)"
                    | "Resample to Image"
                    | "Histogram"
            ),
            Self::Mesh => matches!(
                filter_label,
                "Smooth Mesh"
                    | "Decimate"
                    | "Subdivide"
                    | "Compute Normals"
                    | "Extract Surface"
                    | "Transform"
                    | "Triangulate"
                    | "Connectivity"
                    | "Clip"
                    | "Slice"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PaneKind {
    #[default]
    View3D,
    Chart,
    Spreadsheet,
}

impl PaneKind {
    pub(crate) const ALL: &'static [Self] = &[Self::View3D, Self::Chart, Self::Spreadsheet];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::View3D => "3D View",
            Self::Chart => "XY Chart",
            Self::Spreadsheet => "Spreadsheet",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ViewportCacheKey {
    scene_revision: u64,
    width: u32,
    height: u32,
    camera: Camera,
    settings: SceneSettings,
    widgets: Vec<ViewportWidget>,
    picked_highlight: Option<PickedHighlight>,
    visible_nodes_override: Option<BTreeSet<NodeId>>,
}

#[derive(Clone, Default)]
pub(crate) struct ViewportPane {
    pub(crate) kind: PaneKind,
    pub(crate) camera: Camera,
    pub(crate) tex_id: Option<egui::TextureId>,
    pub(crate) tex_size: (u32, u32),
    pub(crate) cache_key: Option<ViewportCacheKey>,
    pub(crate) cached_scene: Option<RenderScene>,
    pub(crate) cached_total_points: usize,
    pub(crate) cached_total_cells: usize,
    pub(crate) visible_nodes_override: Option<BTreeSet<NodeId>>,
}

pub(crate) fn set_viewport_layout(
    layout: &mut ViewportLayout,
    active_pane: &mut usize,
    next: ViewportLayout,
) {
    *layout = next;
    let max_idx = next.pane_count().saturating_sub(1);
    if *active_pane > max_idx {
        *active_pane = max_idx;
    }
}

pub(crate) fn sync_linked_pane_cameras(
    panes: &mut [ViewportPane],
    source_idx: usize,
    link_cameras: bool,
) {
    if !link_cameras {
        return;
    }
    let Some(cam) = panes.get(source_idx).map(|p| p.camera.clone()) else {
        return;
    };
    for (idx, pane) in panes.iter_mut().enumerate() {
        if idx != source_idx && pane.kind == PaneKind::View3D {
            pane.camera = cam.clone();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ActiveWorkerJob {
    pub(crate) id: u64,
    pub(crate) label: String,
    pub(crate) started_at: Instant,
    pub(crate) cancel_flag: Arc<AtomicBool>,
    pub(crate) cancel_node_to_remove: Option<NodeId>,
    pub(crate) restore_visibility: Vec<(NodeId, bool)>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum WorkerPayload {
    OpenFile {
        path: PathBuf,
        reader: paraviz::io::DatasetReader,
        used_reader: Option<paraviz::io::DatasetReader>,
        result: Result<(DataSet, Vec<f64>), String>,
    },
    RawVolume {
        path: PathBuf,
        dimensions: [usize; 3],
        result: Result<ImageData, String>,
    },
    EvaluateNode {
        node_id: NodeId,
        label: String,
        time: f64,
        show_on_finish: bool,
        repr_override: Option<Representation>,
        open_probe_chart: bool,
        result: Result<DataSet, String>,
    },
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct WorkerCompletion {
    pub(crate) job_id: u64,
    pub(crate) cancel_flag: Arc<AtomicBool>,
    pub(crate) payload: WorkerPayload,
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

type PendingFileLoads = Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>;

pub struct ParaVizApp {
    session: ParaViz,
    renderer: WebGpuRenderer,
    device: wgpu::Device,
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    viewport_layout: ViewportLayout,
    panes: [ViewportPane; 4],
    active_pane: usize,
    link_cameras: bool,
    scene_revision: u64,

    theme: UiTheme,
    theme_applied: bool,
    selected_node: Option<NodeId>,
    renaming_node: Option<(NodeId, String)>,
    copied_filter: Option<(String, FilterConfig)>,
    copied_filter_chain: Vec<(String, FilterConfig)>,
    undo_stack: Vec<PipelineState>,
    redo_stack: Vec<PipelineState>,
    last_undo_snapshot_at: Option<Instant>,
    workspace_profile: WorkspaceProfile,
    command_palette_open: bool,
    command_palette_query: String,
    command_palette_index: usize,
    command_palette_focus_pending: bool,
    filter_usage_counts: HashMap<String, usize>,
    favorite_filters: BTreeSet<String>,
    auto_apply: bool,
    pending_source: Option<(NodeId, SourceConfig)>,
    pending_filter: Option<(NodeId, FilterConfig)>,
    filter_menu_search: String,
    multiblock_filter: String,
    selected_multiblock_patch: Option<(NodeId, String)>,
    next_annotation_id: u64,

    inspector_tab: InspectorTab,
    show_left_dock: bool,
    zen_mode: bool,
    show_drawer: bool,
    drawer_tab: DrawerTab,
    chart_state: ChartState,
    temporal_probe: temporal_probe::TemporalProbePanel,
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
    last_export_dir: Option<PathBuf>,
    pending_file_loads: PendingFileLoads,
    camera_bookmarks: Vec<CameraBookmark>,
    info_stats_cache: HashMap<(NodeId, String, AttributeAssociation), HistogramResult>,
    volume_tf_hist_cache: HashMap<(NodeId, String, AttributeAssociation, Option<usize>), HistogramResult>,

    modal: Option<AppModal>,
    status_message: String,
    error_message: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    next_job_id: u64,
    #[cfg(not(target_arch = "wasm32"))]
    active_job: Option<ActiveWorkerJob>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_tx: mpsc::Sender<WorkerCompletion>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_rx: mpsc::Receiver<WorkerCompletion>,
    #[cfg(not(target_arch = "wasm32"))]
    update_status: UpdateStatus,
    #[cfg(not(target_arch = "wasm32"))]
    update_check_manual: bool,
    #[cfg(not(target_arch = "wasm32"))]
    show_update_modal: bool,
    #[cfg(not(target_arch = "wasm32"))]
    update_tx: mpsc::Sender<UpdateEvent>,
    #[cfg(not(target_arch = "wasm32"))]
    update_rx: mpsc::Receiver<UpdateEvent>,
}

impl ParaVizApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be configured with the WGPU backend");
        let mut app = Self::build_app_state(
            rs.device.clone(),
            rs.queue.clone(),
            rs.renderer.clone(),
            &cc.egui_ctx,
        );

        #[cfg(not(target_arch = "wasm32"))]
        app.trigger_update_check(false);

        if let Some(path) = initial_path {
            app.load_path(&path);
        } else {
            app.load_demo(DemoPreset::VortexStreamlines);
        }

        app
    }

    #[cfg(test)]
    pub(crate) fn new_headless(
        device: wgpu::Device,
        queue: wgpu::Queue,
        egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
        egui_ctx: &egui::Context,
    ) -> Self {
        Self::build_app_state(device, queue, egui_renderer, egui_ctx)
    }

    #[cfg(test)]
    pub(crate) fn wait_for_idle_jobs(&mut self) {
        for _ in 0..600 {
            self.poll_worker_jobs();
            if self.active_job.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn build_app_state(
        device: wgpu::Device,
        queue: wgpu::Queue,
        egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
        egui_ctx: &egui::Context,
    ) -> Self {
        let renderer = WebGpuRenderer::from_device(
            Arc::new(device.clone()),
            Arc::new(queue),
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let theme = UiTheme::DarkStudio;
        apply_theme(egui_ctx, theme);

        #[cfg(not(target_arch = "wasm32"))]
        let (worker_tx, worker_rx) = mpsc::channel();
        #[cfg(not(target_arch = "wasm32"))]
        let (update_tx, update_rx) = mpsc::channel();

        Self {
            session: ParaViz::new(),
            renderer,
            device,
            egui_renderer,
            viewport_layout: ViewportLayout::Single,
            panes: std::array::from_fn(|_| ViewportPane::default()),
            active_pane: 0,
            link_cameras: false,
            scene_revision: 1,
            theme,
            theme_applied: false,
            selected_node: None,
            renaming_node: None,
            copied_filter: None,
            copied_filter_chain: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            last_undo_snapshot_at: None,
            workspace_profile: WorkspaceProfile::Full,
            command_palette_open: false,
            command_palette_query: String::new(),
            command_palette_index: 0,
            command_palette_focus_pending: false,
            filter_usage_counts: HashMap::new(),
            favorite_filters: BTreeSet::new(),
            auto_apply: true,
            pending_source: None,
            pending_filter: None,
            filter_menu_search: String::new(),
            multiblock_filter: String::new(),
            selected_multiblock_patch: None,
            next_annotation_id: 1,
            inspector_tab: InspectorTab::Properties,
            show_left_dock: true,
            zen_mode: false,
            show_drawer: false,
            drawer_tab: DrawerTab::Spreadsheet,
            chart_state: ChartState::default(),
            temporal_probe: Default::default(),
            spreadsheet_target: SpreadsheetTarget::Points,
            spreadsheet_offset: 0,
            spreadsheet_page_size: 200,
            spreadsheet_filter: String::new(),
            spreadsheet_scroll_to_top: false,
            show_scalar_bar: true,
            show_widgets: false,
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
            last_export_dir: None,
            pending_file_loads: Arc::new(std::sync::Mutex::new(Vec::new())),
            camera_bookmarks: Vec::new(),
            info_stats_cache: HashMap::new(),
            volume_tf_hist_cache: HashMap::new(),
            modal: None,
            status_message: "Ready".into(),
            error_message: None,
            #[cfg(not(target_arch = "wasm32"))]
            next_job_id: 1,
            #[cfg(not(target_arch = "wasm32"))]
            active_job: None,
            #[cfg(not(target_arch = "wasm32"))]
            worker_tx,
            #[cfg(not(target_arch = "wasm32"))]
            worker_rx,
            #[cfg(not(target_arch = "wasm32"))]
            update_status: UpdateStatus::Idle,
            #[cfg(not(target_arch = "wasm32"))]
            update_check_manual: false,
            #[cfg(not(target_arch = "wasm32"))]
            show_update_modal: false,
            #[cfg(not(target_arch = "wasm32"))]
            update_tx,
            #[cfg(not(target_arch = "wasm32"))]
            update_rx,
        }
    }

    fn invalidate_viewport(&mut self) {
        self.scene_revision = self.scene_revision.wrapping_add(1);
    }

    pub(super) fn active_viewport_tex_size(&self) -> (u32, u32) {
        let size = self.panes[self.active_pane].tex_size;
        if size.0 > 0 && size.1 > 0 {
            size
        } else {
            self.panes
                .iter()
                .find(|p| p.tex_size.0 > 0 && p.tex_size.1 > 0)
                .map(|p| p.tex_size)
                .unwrap_or((1280, 720))
        }
    }

    pub(super) fn sync_active_camera_to_panes(&mut self) {
        let cam = self.session.camera.clone();
        self.panes[self.active_pane].camera = cam.clone();
        if self.link_cameras {
            sync_linked_pane_cameras(&mut self.panes, self.active_pane, true);
        }
    }

    pub(super) fn sync_all_panes_to_session_camera(&mut self) {
        let cam = self.session.camera.clone();
        for pane in &mut self.panes {
            pane.camera = cam.clone();
        }
    }

    pub(super) fn sync_camera_from_pane(&mut self, source_pane: usize) {
        sync_linked_pane_cameras(&mut self.panes, source_pane, self.link_cameras);
        if source_pane == self.active_pane || self.link_cameras {
            self.session.camera = self.panes[source_pane].camera.clone();
        }
    }

    pub(super) fn reset_active_pane_camera(&mut self) {
        let idx = self.active_pane;
        if let Ok(scene) = self.build_pane_scene(idx) {
            let bounds = scene.bounds();
            self.panes[idx].camera.fit_bounds(&bounds);
            self.sync_camera_from_pane(idx);
            self.invalidate_viewport();
        } else {
            let _ = self.session.reset_camera();
            self.sync_active_camera_to_panes();
            self.invalidate_viewport();
        }
    }

    pub(super) fn camera_preset_active(&mut self, preset: CameraPreset) {
        let idx = self.active_pane;
        if self.panes[idx].visible_nodes_override.is_some()
            && let Ok(scene) = self.build_pane_scene(idx)
        {
            let bounds = scene.bounds();
            self.panes[idx].camera.apply_preset(preset, &bounds);
            self.sync_camera_from_pane(idx);
            self.invalidate_viewport();
            return;
        }
        let _ = self.session.camera_preset(preset);
        self.sync_active_camera_to_panes();
        self.invalidate_viewport();
    }

    pub(super) fn build_pane_scene(&mut self, pane_idx: usize) -> Result<RenderScene, SessionError> {
        let pane_cam = self.panes[pane_idx].camera.clone();
        let override_ids = self.panes[pane_idx].visible_nodes_override.clone();
        if let Some(allowed) = override_ids {
            let eval_ids: Vec<NodeId> = self
                .session
                .nodes()
                .iter()
                .filter(|n| allowed.contains(&n.id))
                .map(|n| n.id)
                .collect();
            for id in eval_ids {
                let _ = self.session.evaluate(id)?;
            }
            let mut settings = self.session.settings.clone();
            if settings.show_time_annotation && settings.time_label.is_none() {
                settings.time_label = Some(format!("Time: {:.4}", self.session.time));
            }
            let mut scene = RenderScene {
                camera: pane_cam,
                actors: Vec::new(),
                settings,
                widgets: self.session.widgets.clone(),
                selection_highlight: self.session.selection_highlight,
            };
            for node in self.session.nodes() {
                if !allowed.contains(&node.id) {
                    continue;
                }
                if let Some(ds) = &node.cached_dataset {
                    let mut style = node.style.clone();
                    style.visible = true;
                    scene.add_actor(SceneActor {
                        id: node.id.0,
                        name: node.name.clone(),
                        dataset: ds.clone(),
                        style,
                    });
                }
            }
            Ok(scene)
        } else {
            let mut scene = self.session.build_scene()?;
            scene.camera = pane_cam;
            Ok(scene)
        }
    }

    pub(super) fn remove_node_and_cleanup(&mut self, id: NodeId) {
        let removed = self.session.remove_node(id);
        self.info_stats_cache
            .retain(|(nid, _, _), _| !removed.contains(nid));
        self.volume_tf_hist_cache
            .retain(|(nid, _, _, _), _| !removed.contains(nid));
        for pane in &mut self.panes {
            if let Some(set) = &mut pane.visible_nodes_override {
                for r in &removed {
                    set.remove(r);
                }
            }
        }
        if self
            .selected_multiblock_patch
            .as_ref()
            .is_some_and(|(nid, _)| removed.contains(nid))
        {
            self.selected_multiblock_patch = None;
        }
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
        let cam = &mut self.panes[self.active_pane].camera;
        let offset = cam.position - cam.target;
        let dist = offset.length().max(1e-4);
        let dir = offset.normalize_or(glam::Vec3::Z);
        let target_dist = self
            .selected_node
            .and_then(|id| self.session.node(id).ok())
            .and_then(|n| n.dataset().map(|ds| n.style.dataset_bounds(ds)))
            .map(|bb| bb.diagonal() * 0.35)
            .filter(|d| *d > 1e-4)
            .map(|d| dist.min(d))
            .unwrap_or(dist);
        cam.target = p;
        cam.position = p + dir * target_dist;
        if cam.orthographic {
            cam.ortho_scale = (cam.ortho_scale * (target_dist / dist)).max(1e-5);
        }
        self.sync_camera_from_pane(self.active_pane);
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

    pub(super) fn default_export_path(&self, filename: &str) -> String {
        if let Some(dir) = &self.last_export_dir {
            dir.join(filename).display().to_string()
        } else {
            filename.to_string()
        }
    }

    pub(super) fn remember_export_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            self.last_export_dir = Some(parent.to_path_buf());
        }
    }

    fn push_recent_file(&mut self, path: &Path) {
        let p = path.to_path_buf();
        self.recent_files.retain(|existing| existing != &p);
        self.recent_files.insert(0, p);
        self.recent_files.truncate(8);
    }

    fn reset_pane_overrides_on_session_replace(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.last_undo_snapshot_at = None;
        self.reset_pane_overrides();
    }

    fn reset_pane_overrides(&mut self) {
        self.temporal_probe = Default::default();
        self.selected_multiblock_patch = None;
        for pane in &mut self.panes {
            pane.visible_nodes_override = None;
        }
        self.next_annotation_id = self
            .session
            .settings
            .annotations
            .iter()
            .map(|a| a.id)
            .max()
            .unwrap_or(0)
            + 1;
        self.sync_all_panes_to_session_camera();
    }

    fn load_demo(&mut self, preset: DemoPreset) {
        #[cfg(not(target_arch = "wasm32"))]
        self.cancel_active_job(false);
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
                self.reset_pane_overrides_on_session_replace();
                self.invalidate_viewport();
                self.status_message = format!("Loaded demo: {}", preset.name());
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load demo: {e}"));
            }
        }
    }

    fn install_loaded_state(&mut self, mut session: ParaViz, status: String) {
        self.error_message = session.update_visible().err().map(|error| error.to_string());
        self.selected_node = session.temporal_probe.as_ref().map(|(id, _)| *id)
            .or_else(|| session.nodes().last().map(|node| node.id));
        self.session = session;
        self.pending_source = None;
        self.pending_filter = None;
        self.renaming_node = None;
        self.last_pick = None;
        self.picked_highlight = None;
        self.highlight_flash_until = None;
        self.ruler_points.clear();
        self.clear_stats_caches();
        self.playing = false;
        self.reset_pane_overrides_on_session_replace();
        self.invalidate_viewport();
        self.status_message = status;
    }

    pub(super) fn load_bytes_from_web(&mut self, name: &str, bytes: &[u8]) {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".json") {
            match std::str::from_utf8(bytes)
                .map_err(|e| e.to_string())
                .and_then(|s| ParaViz::load_state_json(s).map_err(|e| e.to_string()))
            {
                Ok(session) => self.install_loaded_state(session, format!("Loaded state: {name}")),
                Err(e) => {
                    self.error_message = Some(format!("Failed to load state {name}: {e}"));
                }
            }
        } else {
            self.error_message = Some(format!(
                "Browser build currently loads .json state files and built-in Demos/Sources; use the native desktop app for '{name}'."
            ));
        }
    }

    fn load_path(&mut self, path: &Path) {
        self.load_path_with_reader(path, paraviz::io::DatasetReader::Auto);
    }

    fn load_path_with_reader(&mut self, path: &Path, reader: paraviz::io::DatasetReader) {
        if is_raw_volume_ext(path) && reader == paraviz::io::DatasetReader::Auto {
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
            #[cfg(not(target_arch = "wasm32"))]
            {
                let project_dir = path.parent();
                let loaded = ParaViz::load_project_with_relocation(path, |missing| {
                    let mut dialog = rfd::FileDialog::new().set_title(format!(
                        "Locate missing dataset {}", missing.display()
                    ));
                    if let Some(directory) = missing.parent().filter(|dir| dir.is_dir())
                        .or(project_dir)
                    {
                        dialog = dialog.set_directory(directory);
                    }
                    dialog.pick_file()
                });
                match loaded {
                    Ok(session) => {
                        self.cancel_active_job(false);
                        self.install_loaded_state(session, format!("Loaded state: {}", path.display()));
                        self.push_recent_file(path);
                    }
                    Err(error) => self.error_message = Some(format!("Failed to load state: {error}")),
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                match std::fs::read(path) {
                    Ok(bytes) => self.load_bytes_from_web(&path.display().to_string(), &bytes),
                    Err(error) => self.error_message = Some(format!("Failed to load state: {error}")),
                }
            }
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.spawn_open_file_job(path.to_path_buf(), reader);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let is_first = self.session.nodes().is_empty();
            let used_reader = match reader {
                paraviz::io::DatasetReader::Auto => paraviz::io::detect_reader(path).ok(),
                explicit => Some(explicit),
            };
            match self.session.open_with_reader(path, reader) {
                Ok(id) => {
                    if is_first
                        && let Some(first_t) = self.session.all_time_steps().first().copied()
                    {
                        self.session.time = first_t;
                    }
                    if let Err(e) = self.session.evaluate(id) {
                        let _ = self.session.remove_node(id);
                        let reader_desc = used_reader
                            .map(|r| format!(" ({})", r.short_name()))
                            .unwrap_or_default();
                        self.error_message = Some(format!(
                            "Failed to read '{}'{}: {e}",
                            path.display(),
                            reader_desc
                        ));
                        return;
                    }
                    self.session.show(id);
                    if is_first {
                        let _ = self.session.camera_preset(CameraPreset::Isometric);
                    } else {
                        let _ = self.session.reset_camera();
                    }
                    self.sync_all_panes_to_session_camera();
                    self.select_node(id);
                    self.clear_stats_caches();
                    self.error_message = None;
                    self.invalidate_viewport();
                    self.push_recent_file(path);
                    let reader_desc = used_reader
                        .map(|r| r.short_name())
                        .unwrap_or("Auto");
                    let counts_desc = self
                        .session
                        .node(id)
                        .ok()
                        .and_then(|n| n.dataset())
                        .map(|ds| format!(" ({} pts, {} cells)", ds.num_points(), ds.num_cells()))
                        .unwrap_or_default();
                    self.status_message = format!(
                        "Opened '{}' via {}{}",
                        path.display(),
                        reader_desc,
                        counts_desc
                    );
                }
                Err(e) => {
                    let reader_desc = used_reader
                        .map(|r| format!(" ({})", r.short_name()))
                        .unwrap_or_default();
                    self.error_message = Some(format!(
                        "Failed to open '{}'{}: {e}",
                        path.display(),
                        reader_desc
                    ));
                }
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.spawn_raw_volume_job(path, desc, rv.scalar_type);
        }
        #[cfg(target_arch = "wasm32")]
        match paraviz::io::read_raw_volume_with_type(&path, &desc, rv.scalar_type) {
            Ok(img) => {
                self.finish_raw_volume_import(&path, rv.dimensions, img);
            }
            Err(e) => {
                self.error_message = Some(format!("Raw volume import failed: {e}"));
            }
        }
    }

    fn finish_raw_volume_import(
        &mut self,
        path: &Path,
        dimensions: [usize; 3],
        img: ImageData,
    ) {
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
        self.sync_all_panes_to_session_camera();
        self.select_node(id);
        self.error_message = None;
        self.invalidate_viewport();
        self.push_recent_file(path);
        self.status_message = format!(
            "Imported raw volume '{label}' ({}×{}×{})",
            dimensions[0], dimensions[1], dimensions[2]
        );
    }

    fn add_source(&mut self, cfg: SourceConfig) {
        self.push_undo_snapshot();
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
                self.sync_all_panes_to_session_camera();
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
        self.push_undo_snapshot();

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
        let repr_override = if is_cell_centers {
            Some(Representation::Points)
        } else if is_wire_edges {
            Some(Representation::Wireframe)
        } else {
            None
        };
        let is_probe = matches!(cfg, FilterConfig::ProbeOverLine(_));
        let label = self.unique_label(cfg.default_label());
        #[cfg(not(target_arch = "wasm32"))]
        let parent_ds = self
            .session
            .node(parent_id)
            .ok()
            .and_then(|n| n.dataset().cloned());
        match self.session.add_filter_cfg(label.clone(), parent_id, cfg.clone()) {
            Ok(id) => {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(input_ds) = parent_ds {
                    if let Some(repr) = repr_override
                        && let Ok(n) = self.session.node_mut(id)
                    {
                        n.style.representation = repr;
                    }
                    self.session.hide(id);
                    self.select_node(id);
                    self.spawn_filter_eval_job(
                        id,
                        label,
                        cfg,
                        input_ds,
                        true,
                        repr_override,
                        is_probe,
                        Some(id),
                        Vec::new(),
                    );
                    return;
                }

                let eval_res = self.session.evaluate(id);
                let handle = self.session.show(id);
                if let Some(repr) = repr_override {
                    handle.representation(repr);
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

    pub(super) fn apply_filter_update(&mut self, id: NodeId, cfg: FilterConfig) {
        self.push_undo_snapshot_throttled();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cancel_active_job(false);
            let parent_ds = self
                .session
                .node(id)
                .ok()
                .and_then(|n| n.parent)
                .and_then(|pid| self.session.node(pid).ok())
                .and_then(|pn| pn.dataset().cloned());
            if let Some(input_ds) = parent_ds {
                let mut restore_visibility = Vec::new();
                let mut stack = vec![id];
                while let Some(cur) = stack.pop() {
                    if let Ok(n) = self.session.node(cur) {
                        restore_visibility.push((cur, n.style.visible));
                    }
                    for child in self.session.nodes() {
                        if child.parent == Some(cur) {
                            stack.push(child.id);
                        }
                    }
                }
                let label = self
                    .session
                    .node(id)
                    .map(|n| n.name.clone())
                    .unwrap_or_else(|_| cfg.default_label().to_string());
                if let Err(e) = self.session.update_filter_cfg(id, cfg.clone()) {
                    self.error_message = Some(format!("Update filter error: {e}"));
                    return;
                }
                for &(nid, _) in &restore_visibility {
                    self.session.hide(nid);
                }
                self.pending_filter = None;
                self.error_message = None;
                self.clear_stats_caches_for_node(id);
                self.spawn_filter_eval_job(
                    id,
                    label,
                    cfg,
                    input_ds,
                    false,
                    None,
                    false,
                    None,
                    restore_visibility,
                );
                return;
            }
        }

        if let Err(e) = self.session.update_filter_cfg(id, cfg) {
            self.error_message = Some(format!("Update filter error: {e}"));
        } else {
            self.pending_filter = None;
            self.error_message = None;
            self.clear_stats_caches_for_node(id);
            self.invalidate_viewport();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn next_worker_job_id(&mut self) -> (u64, Arc<AtomicBool>) {
        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        (self.next_job_id, Arc::new(AtomicBool::new(false)))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn cancel_active_job(&mut self, update_status: bool) {
        if let Some(job) = self.active_job.take() {
            job.cancel_flag.store(true, Ordering::Relaxed);
            for (nid, vis) in job.restore_visibility {
                if vis {
                    if self.session.node(nid).is_ok() {
                        self.session.show(nid);
                    }
                } else {
                    self.session.hide(nid);
                }
            }
            if let Some(rm_id) = job.cancel_node_to_remove
                && self.session.node(rm_id).is_ok()
            {
                self.remove_node_and_cleanup(rm_id);
            }
            if update_status {
                self.status_message = format!("Cancelled: {}", job.label);
            }
            self.invalidate_viewport();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_open_file_job(&mut self, path: PathBuf, reader: paraviz::io::DatasetReader) {
        self.cancel_active_job(false);
        let (job_id, cancel_flag) = self.next_worker_job_id();
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("file")
            .to_string();
        let label = format!("Loading {fname}...");
        let used_reader = match reader {
            paraviz::io::DatasetReader::Auto => paraviz::io::detect_reader(&path).ok(),
            explicit => Some(explicit),
        };
        self.active_job = Some(ActiveWorkerJob {
            id: job_id,
            label: label.clone(),
            started_at: Instant::now(),
            cancel_flag: cancel_flag.clone(),
            cancel_node_to_remove: None,
            restore_visibility: Vec::new(),
        });
        self.status_message = label;
        self.error_message = None;

        let tx = self.worker_tx.clone();
        std::thread::spawn(move || {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let result = if ext == "pvd" && reader == paraviz::io::DatasetReader::Auto {
                paraviz::io::PvdSeriesSource::new(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|pvd| {
                        let ts = pvd.time_steps();
                        let first_t = ts.first().copied().unwrap_or(0.0);
                        pvd.execute(first_t)
                            .map(|ds| (ds, ts))
                            .map_err(|e| e.to_string())
                    })
            } else {
                paraviz::io::read_dataset_with_reader(&path, reader)
                    .map(|(ds, resolved)| {
                        let ts = inspect_file_time_steps(&path, resolved);
                        (ds, ts)
                    })
                    .map_err(|e| e.to_string())
            };
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let _ = tx.send(WorkerCompletion {
                job_id,
                cancel_flag,
                payload: WorkerPayload::OpenFile {
                    path,
                    reader,
                    used_reader,
                    result,
                },
            });
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_raw_volume_job(
        &mut self,
        path: PathBuf,
        desc: RawVolumeDescriptor,
        scalar_type: RawScalarType,
    ) {
        self.cancel_active_job(false);
        let (job_id, cancel_flag) = self.next_worker_job_id();
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("volume")
            .to_string();
        let label = format!("Importing {fname}...");
        let dimensions = desc.dimensions;
        self.active_job = Some(ActiveWorkerJob {
            id: job_id,
            label: label.clone(),
            started_at: Instant::now(),
            cancel_flag: cancel_flag.clone(),
            cancel_node_to_remove: None,
            restore_visibility: Vec::new(),
        });
        self.status_message = label;
        self.error_message = None;

        let tx = self.worker_tx.clone();
        std::thread::spawn(move || {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let result = paraviz::io::read_raw_volume_with_type(&path, &desc, scalar_type)
                .map_err(|e| e.to_string());
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let _ = tx.send(WorkerCompletion {
                job_id,
                cancel_flag,
                payload: WorkerPayload::RawVolume {
                    path,
                    dimensions,
                    result,
                },
            });
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    fn spawn_filter_eval_job(
        &mut self,
        node_id: NodeId,
        label: String,
        cfg: FilterConfig,
        input_ds: Arc<DataSet>,
        show_on_finish: bool,
        repr_override: Option<Representation>,
        open_probe_chart: bool,
        cancel_node_to_remove: Option<NodeId>,
        restore_visibility: Vec<(NodeId, bool)>,
    ) {
        self.cancel_active_job(false);
        let (job_id, cancel_flag) = self.next_worker_job_id();
        let time = self.session.time;
        let job_label = format!("Evaluating {label}...");
        self.active_job = Some(ActiveWorkerJob {
            id: job_id,
            label: job_label.clone(),
            started_at: Instant::now(),
            cancel_flag: cancel_flag.clone(),
            cancel_node_to_remove,
            restore_visibility,
        });
        self.status_message = job_label;

        let tx = self.worker_tx.clone();
        std::thread::spawn(move || {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let filter = cfg.build_filter();
            let result = filter.apply(&input_ds).map_err(|e| e.to_string());
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            let _ = tx.send(WorkerCompletion {
                job_id,
                cancel_flag,
                payload: WorkerPayload::EvaluateNode {
                    node_id,
                    label,
                    time,
                    show_on_finish,
                    repr_override,
                    open_probe_chart,
                    result,
                },
            });
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_worker_jobs(&mut self) {
        let completions: Vec<WorkerCompletion> = self.worker_rx.try_iter().collect();
        for comp in completions {
            if comp.cancel_flag.load(Ordering::Relaxed) {
                continue;
            }
            let Some(active) = self.active_job.as_ref() else {
                continue;
            };
            if active.id != comp.job_id {
                continue;
            }
            let finished_job = self.active_job.take().unwrap();
            match comp.payload {
                WorkerPayload::OpenFile {
                    path,
                    reader,
                    used_reader,
                    result,
                } => match result {
                    Ok((dataset, time_steps)) => {
                        let is_first = self.session.nodes().is_empty();
                        if is_first
                            && let Some(first_t) = time_steps.first().copied()
                        {
                            self.session.time = first_t;
                        }
                        match self
                            .session
                            .open_preloaded(&path, reader, dataset, time_steps)
                        {
                            Ok(id) => {
                                self.session.show(id);
                                if is_first {
                                    let _ = self.session.camera_preset(CameraPreset::Isometric);
                                } else {
                                    let _ = self.session.reset_camera();
                                }
                                self.sync_all_panes_to_session_camera();
                                self.select_node(id);
                                self.clear_stats_caches();
                                self.error_message = None;
                                self.invalidate_viewport();
                                self.push_recent_file(&path);
                                let reader_desc =
                                    used_reader.map(|r| r.short_name()).unwrap_or("Auto");
                                let counts_desc = self
                                    .session
                                    .node(id)
                                    .ok()
                                    .and_then(|n| n.dataset())
                                    .map(|ds| {
                                        format!(" ({} pts, {} cells)", ds.num_points(), ds.num_cells())
                                    })
                                    .unwrap_or_default();
                                self.status_message = format!(
                                    "Opened '{}' via {}{}",
                                    path.display(),
                                    reader_desc,
                                    counts_desc
                                );
                            }
                            Err(e) => {
                                let reader_desc = used_reader
                                    .map(|r| format!(" ({})", r.short_name()))
                                    .unwrap_or_default();
                                self.error_message = Some(format!(
                                    "Failed to open '{}'{}: {e}",
                                    path.display(),
                                    reader_desc
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        let reader_desc = used_reader
                            .map(|r| format!(" ({})", r.short_name()))
                            .unwrap_or_default();
                        self.error_message = Some(format!(
                            "Failed to open '{}'{}: {e}",
                            path.display(),
                            reader_desc
                        ));
                    }
                },
                WorkerPayload::RawVolume {
                    path,
                    dimensions,
                    result,
                } => match result {
                    Ok(img) => self.finish_raw_volume_import(&path, dimensions, img),
                    Err(e) => {
                        self.error_message = Some(format!("Raw volume import failed: {e}"));
                    }
                },
                WorkerPayload::EvaluateNode {
                    node_id,
                    label,
                    time,
                    show_on_finish,
                    repr_override,
                    open_probe_chart,
                    result,
                } => {
                    for (nid, vis) in finished_job.restore_visibility {
                        if self.session.node(nid).is_ok() {
                            if vis {
                                self.session.show(nid);
                            } else {
                                self.session.hide(nid);
                            }
                        }
                    }
                    match result {
                        Ok(ds) => {
                            if let Err(e) = self.session.seed_evaluated(node_id, Arc::new(ds), time)
                            {
                                self.error_message = Some(format!("Filter '{label}' failed: {e}"));
                            } else {
                                if show_on_finish && self.session.node(node_id).is_ok() {
                                    let handle = self.session.show(node_id);
                                    if let Some(repr) = repr_override {
                                        handle.representation(repr);
                                    }
                                }
                                self.error_message = None;
                                self.status_message = format!("Applied filter '{label}'");
                                self.select_node(node_id);
                                self.clear_stats_caches_for_node(node_id);
                                self.invalidate_viewport();
                                if open_probe_chart {
                                    self.chart_state.mode = ChartMode::XYPlot;
                                    self.drawer_tab = DrawerTab::Chart;
                                    self.show_drawer = true;
                                }
                            }
                        }
                        Err(e) => {
                            if finished_job.cancel_node_to_remove == Some(node_id)
                                && self.session.node(node_id).is_ok()
                            {
                                self.remove_node_and_cleanup(node_id);
                            }
                            self.error_message = Some(format!("Filter '{label}' failed: {e}"));
                        }
                    }
                }
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
            self.reset_active_pane_camera();
            return;
        };
        let _ = self.session.evaluate(sel_id);
        if let Ok(node) = self.session.node(sel_id)
            && let Some(ds) = node.dataset()
        {
            let bb = node.style.dataset_bounds(ds);
            if bb.is_valid() {
                self.panes[self.active_pane].camera.fit_bounds(&bb);
                self.sync_camera_from_pane(self.active_pane);
                self.invalidate_viewport();
                return;
            }
        }
        self.reset_active_pane_camera();
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
            self.remove_node_and_cleanup(id);
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
        let pending: Vec<(String, Vec<u8>)> = self
            .pending_file_loads
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default();
        for (name, bytes) in pending {
            self.load_bytes_from_web(&name, &bytes);
        }

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

        if ctx.input(|i| {
            (i.modifiers.command && i.key_pressed(egui::Key::K))
                || (i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(egui::Key::Space))
        }) && self.modal.is_none()
            && self.renaming_node.is_none()
        {
            self.command_palette_open = !self.command_palette_open;
            if self.command_palette_open {
                self.command_palette_query.clear();
                self.command_palette_index = 0;
                self.command_palette_focus_pending = true;
            }
        }

        if !ctx.egui_wants_keyboard_input()
            && self.modal.is_none()
            && self.renaming_node.is_none()
            && !self.command_palette_open
        {
            if ctx.input(|i| i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z))
                || ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Y))
            {
                self.redo();
            } else if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z)) {
                self.undo();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                self.reset_active_pane_camera();
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


    // Filter jobs temporarily hide actors while they evaluate. Save the intended visibility.
    fn history_snapshot(&mut self) -> Result<PipelineState, SessionError> {
        #[cfg(not(target_arch = "wasm32"))]
        let original_visibility: Vec<_> = self.active_job.iter()
            .flat_map(|job| {
                job.restore_visibility.iter().copied()
                    .chain(job.cancel_node_to_remove.map(|id| (id, true)))
            })
            .filter_map(|(id, visible)| {
                let node = self.session.node_mut(id).ok()?;
                Some((id, std::mem::replace(&mut node.style.visible, visible)))
            })
            .collect();
        let result = self.session.state_snapshot();
        #[cfg(not(target_arch = "wasm32"))]
        for (id, visible) in original_visibility.into_iter().rev() {
            if let Ok(node) = self.session.node_mut(id) {
                node.style.visible = visible;
            }
        }
        result
    }

    pub(super) fn push_undo_snapshot(&mut self) {
        if let Ok(snapshot) = self.history_snapshot() {
            self.last_undo_snapshot_at = Some(Instant::now());
            self.redo_stack.clear();
            if self.undo_stack.last().is_some_and(|last| last == &snapshot) {
                return;
            }
            self.undo_stack.push(snapshot);
            if self.undo_stack.len() > 40 {
                self.undo_stack.remove(0);
            }
        }
    }

    pub(super) fn push_undo_snapshot_throttled(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_undo_snapshot_at
            && now.duration_since(prev) < Duration::from_millis(450) {
                return;
            }
        self.push_undo_snapshot();
    }

    fn restore_history(&mut self, snapshot: &PipelineState) -> Result<PipelineState, SessionError> {
        let restored = ParaViz::from_state_snapshot(snapshot.clone())?;
        let current = self.history_snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(job) = self.active_job.take() {
            // The whole session is being replaced. Its temporary actor state goes with it.
            job.cancel_flag.store(true, Ordering::Relaxed);
        }
        self.session = restored;
        self.pending_source = None;
        self.pending_filter = None;
        self.renaming_node = None;
        self.picked_highlight = None;
        self.last_pick = None;
        self.last_undo_snapshot_at = None;
        self.selected_node = self.selected_node
            .filter(|id| self.session.node(*id).is_ok())
            .or_else(|| self.session.nodes().last().map(|node| node.id));
        self.clear_stats_caches();
        self.reset_pane_overrides();
        self.invalidate_viewport();
        self.error_message = None;
        Ok(current)
    }

    pub(super) fn undo(&mut self) {
        let Some(snapshot) = self.undo_stack.pop() else {
            self.status_message = "Nothing to undo".into();
            return;
        };
        match self.restore_history(&snapshot) {
            Ok(current) => {
                self.redo_stack.push(current);
                self.status_message = "Undid last pipeline action".into();
            }
            Err(err) => {
                self.undo_stack.push(snapshot);
                self.error_message = Some(format!("Undo failed: {err}"));
            }
        }
    }

    pub(super) fn redo(&mut self) {
        let Some(snapshot) = self.redo_stack.pop() else {
            self.status_message = "Nothing to redo".into();
            return;
        };
        match self.restore_history(&snapshot) {
            Ok(current) => {
                self.undo_stack.push(current);
                self.status_message = "Redid pipeline action".into();
            }
            Err(err) => {
                self.redo_stack.push(snapshot);
                self.error_message = Some(format!("Redo failed: {err}"));
            }
        }
    }

    fn handle_modals(&mut self, ctx: &egui::Context) {
        let vp_size = self.active_viewport_tex_size();
        let node_labels: Vec<(NodeId, String)> = self
            .session
            .nodes()
            .iter()
            .map(|n| (n.id, n.name.clone()))
            .collect();
        let action = draw_modal(
            ctx,
            &mut self.modal,
            &mut self.export_glb_colors,
            vp_size,
            self.theme,
            &node_labels,
        );
        self.execute_modal_action(action);
    }

    pub(super) fn open_picked_scientific_file(&mut self, p: PathBuf) {
        if is_raw_volume_ext(&p) {
            self.modal = Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                p.display().to_string(),
            )));
        } else {
            self.load_path(&p);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn save_project_file(&mut self, path: &Path, mode: ProjectSaveMode) {
        match self.session.save_project(path, mode) {
            Ok(()) => {
                self.remember_export_dir(path);
                self.error_message = None;
                self.status_message = format!("Saved project to {}", path.display());
            }
            Err(error) => self.error_message = Some(format!("Project save failed: {error}")),
        }
    }

    pub(super) fn execute_modal_action(&mut self, action: ModalAction) {
        match action {
            ModalAction::None => {}
            ModalAction::OpenFile(p, reader) => self.load_path_with_reader(&p, reader),
            ModalAction::LoadState(p) => self.load_path(&p),
            ModalAction::ImportRawVolume(rv) => self.import_raw_volume(rv),
            ModalAction::RunGpuContour {
                node_id,
                field_name,
                iso_value,
            } => self.run_gpu_contour(node_id, &field_name, iso_value),
            ModalAction::LoadDemo(preset) => self.load_demo(preset),
            ModalAction::SaveState(p) => {
                #[cfg(not(target_arch = "wasm32"))]
                self.save_project_file(&p, ProjectSaveMode::Referenced);
                #[cfg(target_arch = "wasm32")]
                let _ = p;
            }
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
                            self.remember_export_dir(&p);
                            self.error_message = None;
                            self.status_message = format!("Exported dataset -> {}", p.display());
                        }
                        Err(e) => self.error_message = Some(format!("Export failed: {e}")),
                    }
                }
            }
            ModalAction::ExportAbHtml { path: p, node_b } => {
                if let Some(id_a) = self.selected_node {
                    match self
                        .session
                        .export_html_ab(id_a, node_b, &p, self.export_glb_colors)
                    {
                        Ok(()) => {
                            self.remember_export_dir(&p);
                            self.error_message = None;
                            self.status_message =
                                format!("Exported A/B compare viewer -> {}", p.display());
                        }
                        Err(e) => self.error_message = Some(format!("Export failed: {e}")),
                    }
                } else {
                    self.error_message =
                        Some("Select a pipeline node first (side A of the A/B compare)".into());
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
                            self.remember_export_dir(&p);
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
                        self.remember_export_dir(&path);
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
                        self.remember_export_dir(&path);
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


impl ParaVizApp {
    pub(crate) fn draw_studio_ui(&mut self, ui: &mut egui::Ui) {
        if !self.theme_applied {
            apply_theme(ui.ctx(), self.theme);
            self.theme_applied = true;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.poll_worker_jobs();
            self.poll_update_events();
            if self.active_job.is_some()
                || matches!(
                    self.update_status,
                    UpdateStatus::Checking | UpdateStatus::Downloading { .. }
                )
            {
                ui.ctx().request_repaint();
            }
        }

        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;
        if dt > 1e-5 {
            let inst_fps = (1.0 / dt) as f32;
            self.fps_smoothed = self.fps_smoothed * 0.9 + inst_fps * 0.1;
        }

        if self.turntable_enabled {
            self.panes[self.active_pane]
                .camera
                .orbit((dt as f32) * self.turntable_speed, 0.0);
            self.sync_camera_from_pane(self.active_pane);
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
                #[cfg(not(target_arch = "wasm32"))]
                let mut cancel_clicked = false;
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(job) = &self.active_job {
                    ui.spinner();
                    let elapsed = job.started_at.elapsed().as_secs_f32();
                    ui.label(
                        egui::RichText::new(format!("{} ({elapsed:.1}s)", job.label))
                            .color(pal.accent)
                            .strong(),
                    );
                    if ui.small_button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                    ui.separator();
                } else if let Some(err) = &self.error_message {
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
                #[cfg(not(target_arch = "wasm32"))]
                if cancel_clicked {
                    self.cancel_active_job(true);
                }

                #[cfg(target_arch = "wasm32")]
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
                    #[cfg(not(target_arch = "wasm32"))]
                    match &self.update_status {
                        UpdateStatus::Available(info) => {
                            if ui
                                .small_button(
                                    egui::RichText::new(format!("Update v{}", info.version))
                                        .color(pal.accent)
                                        .strong(),
                                )
                                .on_hover_text("Click to review and install the new ParaViz release")
                                .clicked()
                            {
                                self.show_update_modal = true;
                            }
                            ui.separator();
                        }
                        UpdateStatus::Downloading { version, .. } => {
                            ui.label(
                                egui::RichText::new(format!("Updating to v{version}..."))
                                    .size(11.0)
                                    .color(pal.accent),
                            );
                            ui.spinner();
                            ui.separator();
                        }
                        UpdateStatus::Installed { version } => {
                            if ui
                                .small_button(
                                    egui::RichText::new(format!("Restart for v{version}"))
                                        .color(pal.success)
                                        .strong(),
                                )
                                .clicked()
                            {
                                self.show_update_modal = true;
                            }
                            ui.separator();
                        }
                        _ => {}
                    }
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
        self.draw_command_palette(ui.ctx());

        #[cfg(not(target_arch = "wasm32"))]
        self.draw_update_modal(ui.ctx());
    }
}

impl eframe::App for ParaVizApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw_studio_ui(ui);
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ParaVizApp {
    pub(super) fn trigger_update_check(&mut self, manual: bool) {
        if matches!(
            self.update_status,
            UpdateStatus::Checking | UpdateStatus::Downloading { .. }
        ) {
            return;
        }
        self.update_check_manual = manual;
        self.update_status = UpdateStatus::Checking;
        spawn_check_updates(self.update_tx.clone());
    }

    fn poll_update_events(&mut self) {
        while let Ok(ev) = self.update_rx.try_recv() {
            match ev {
                UpdateEvent::Checked(Ok(Some(info))) => {
                    self.status_message = format!("ParaViz v{} release available", info.version);
                    self.update_status = UpdateStatus::Available(info);
                }
                UpdateEvent::Checked(Ok(None)) => {
                    let current = current_version().to_string();
                    if self.update_check_manual {
                        self.status_message = format!("ParaViz v{current} is up to date");
                    }
                    self.update_status = UpdateStatus::UpToDate { current };
                }
                UpdateEvent::Checked(Err(err)) => {
                    if self.update_check_manual {
                        self.update_status = UpdateStatus::Error(err);
                    } else {
                        // Stay unobtrusive when offline during background startup check.
                        self.update_status = UpdateStatus::Idle;
                    }
                }
                UpdateEvent::Installed(Ok(version)) => {
                    self.status_message =
                        format!("Installed ParaViz v{version}. Restart to finish updating.");
                    self.update_status = UpdateStatus::Installed { version };
                }
                UpdateEvent::Installed(Err(err)) => {
                    self.error_message = Some(format!("Auto-update failed: {err}"));
                    self.update_status = UpdateStatus::Error(err);
                }
            }
        }
    }

    fn start_install_update(&mut self, info: ReleaseUpdateInfo) {
        let asset_name = info
            .asset
            .as_ref()
            .map(|a| a.name.clone())
            .unwrap_or_default();
        self.update_status = UpdateStatus::Downloading {
            version: info.version.clone(),
            asset_name,
        };
        spawn_install_update(info, self.update_tx.clone());
    }

    fn restart_application(&self, ctx: &egui::Context) {
        if let Some(appimage) = std::env::var_os("APPIMAGE") {
            let _ = std::process::Command::new(appimage).spawn();
        } else if let Ok(exe) = std::env::current_exe() {
            let _ = std::process::Command::new(exe).spawn();
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn draw_update_modal(&mut self, ctx: &egui::Context) {
        if !self.show_update_modal {
            return;
        }
        let pal = palette(self.theme);
        let mut open = self.show_update_modal;
        let mut action_recheck = false;
        let mut action_install: Option<ReleaseUpdateInfo> = None;
        let mut action_restart = false;
        let mut close_clicked = false;

        egui::Window::new("ParaViz Software Update")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .min_width(420.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Installed version:").color(pal.muted));
                    ui.monospace(format!("v{}", current_version()));
                });
                ui.separator();

                match &self.update_status {
                    UpdateStatus::Idle => {
                        ui.label("Click below to check GitHub Releases for a newer version.");
                        ui.add_space(6.0);
                        if ui.button("Check Now").clicked() {
                            action_recheck = true;
                        }
                    }
                    UpdateStatus::Checking => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Checking GitHub Releases...");
                        });
                    }
                    UpdateStatus::UpToDate { current } => {
                        ui.colored_label(
                            pal.success,
                            format!("You are running the latest version (v{current})."),
                        );
                        ui.add_space(6.0);
                        if ui.button("Check Again").clicked() {
                            action_recheck = true;
                        }
                    }
                    UpdateStatus::Available(info) => {
                        ui.label(
                            egui::RichText::new(format!(
                                "New release available: {}",
                                info.tag_name
                            ))
                            .strong()
                            .size(15.0)
                            .color(pal.accent),
                        );
                        if let Some(asset) = &info.asset {
                            let mb = asset.size as f64 / (1024.0 * 1024.0);
                            ui.label(
                                egui::RichText::new(format!("Package: {} ({mb:.1} MiB)", asset.name))
                                    .size(12.0)
                                    .color(pal.muted),
                            );
                        }
                        if info.checksums_url.is_some() {
                            ui.label(
                                egui::RichText::new("SHA-256 checksum verification: enabled (SHA256SUMS.txt)")
                                    .size(11.5)
                                    .color(pal.success),
                            );
                        }
                        if !info.notes.trim().is_empty() {
                            ui.add_space(4.0);
                            egui::ScrollArea::vertical()
                                .max_height(150.0)
                                .show(ui, |ui| {
                                    ui.monospace(info.notes.trim());
                                });
                        }
                        ui.add_space(6.0);
                        ui.horizontal_wrapped(|ui| {
                            let can_self_replace = info.asset.is_some()
                                && info.checksums_url.is_some()
                                && info.install_kind != InstallKind::CargoOrManual;
                            if can_self_replace {
                                if ui.button("Download & Install Update").clicked() {
                                    action_install = Some(info.clone());
                                }
                            } else if info.install_kind == InstallKind::CargoOrManual {
                                ui.label("Cargo install detected: run `cargo install paraviz-app --force` or download binary.");
                            }
                            if ui.button("Open Release Page").clicked() {
                                ui.ctx().open_url(egui::OpenUrl::new_tab(&info.html_url));
                            }
                        });
                    }
                    UpdateStatus::Downloading {
                        version,
                        asset_name,
                    } => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!(
                                "Downloading & verifying v{version} ({asset_name})..."
                            ));
                        });
                    }
                    UpdateStatus::Installed { version } => {
                        ui.colored_label(
                            pal.success,
                            format!(
                                "ParaViz v{version} installed and verified via SHA-256!"
                            ),
                        );
                        ui.label("Restart ParaViz now to launch the updated version.");
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Restart Now").clicked() {
                                action_restart = true;
                            }
                            if ui.button("Later").clicked() {
                                close_clicked = true;
                            }
                        });
                    }
                    UpdateStatus::Error(err) => {
                        ui.colored_label(pal.danger, format!("Update check failed: {err}"));
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Retry").clicked() {
                                action_recheck = true;
                            }
                            if ui.button("Open Releases").clicked() {
                                ui.ctx()
                                    .open_url(egui::OpenUrl::new_tab(RELEASES_PAGE_URL));
                            }
                        });
                    }
                }
            });

        self.show_update_modal = open && !close_clicked;
        if action_recheck {
            self.trigger_update_check(true);
        } else if let Some(info) = action_install {
            self.start_install_update(info);
        } else if action_restart {
            self.restart_application(ctx);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn inspect_file_time_steps(path: &Path, reader: paraviz::io::DatasetReader) -> Vec<f64> {
    paraviz::io::VtkFileSource::with_reader(path, reader)
        .request_information()
        .time_steps
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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn undo_discards_a_pending_filter_result_and_redo_restores_visibility() {
        let gpu = WebGpuRenderer::new_headless().expect("headless GPU");
        let device = gpu.device().clone();
        let renderer = egui_wgpu::Renderer::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            Default::default(),
        );
        let mut app = ParaVizApp::new_headless(
            device,
            gpu.queue().clone(),
            Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
            &egui::Context::default(),
        );
        let source = app.session.add_dataset("Mesh", sample_dataset());
        let original = FilterConfig::Transform(TransformFilter::default());
        let filter = app.session.add_filter_cfg("Transform", source, original.clone()).unwrap();
        app.session.show(filter);
        app.push_undo_snapshot();

        let changed = FilterConfig::Transform(TransformFilter {
            translation: [1.0, 0.0, 0.0],
            ..Default::default()
        });
        app.session.update_filter_cfg(filter, changed.clone()).unwrap();
        app.session.hide(filter);
        let (job_id, cancel_flag) = app.next_worker_job_id();
        app.active_job = Some(ActiveWorkerJob {
            id: job_id,
            label: "Evaluating Transform".into(),
            started_at: Instant::now(),
            cancel_flag: cancel_flag.clone(),
            cancel_node_to_remove: None,
            restore_visibility: vec![(filter, true)],
        });
        app.worker_tx.send(WorkerCompletion {
            job_id,
            cancel_flag: cancel_flag.clone(),
            payload: WorkerPayload::EvaluateNode {
                node_id: filter,
                label: "Transform".into(),
                time: 0.0,
                show_on_finish: false,
                repr_override: None,
                open_probe_chart: false,
                result: Ok(sample_dataset()),
            },
        }).unwrap();

        app.undo();
        assert!(cancel_flag.load(Ordering::Relaxed));
        app.poll_worker_jobs();
        let node = app.session.node(filter).unwrap();
        assert_eq!(node.filter_config.as_ref(), Some(&original));
        assert!(node.dataset().is_none());
        app.redo();
        let node = app.session.node(filter).unwrap();
        assert_eq!(node.filter_config.as_ref(), Some(&changed));
        assert!(node.style.visible);
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
            visible_nodes_override: None,
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

    #[test]
    fn viewport_layout_pane_counts_and_active_clamping() {
        assert_eq!(ViewportLayout::Single.pane_count(), 1);
        assert_eq!(ViewportLayout::SplitHorizontal.pane_count(), 2);
        assert_eq!(ViewportLayout::SplitVertical.pane_count(), 2);
        assert_eq!(ViewportLayout::Quad.pane_count(), 4);

        let mut layout = ViewportLayout::Quad;
        let mut active = 3;
        set_viewport_layout(&mut layout, &mut active, ViewportLayout::SplitHorizontal);
        assert_eq!(layout, ViewportLayout::SplitHorizontal);
        assert_eq!(active, 1);

        set_viewport_layout(&mut layout, &mut active, ViewportLayout::Single);
        assert_eq!(active, 0);
    }

    #[test]
    fn linked_pane_cameras_sync_only_3d_panes_when_enabled() {
        let mut panes: [ViewportPane; 4] = std::array::from_fn(|_| ViewportPane::default());
        panes[2].kind = PaneKind::Chart;
        panes[0].camera.fov_y_deg = 62.0;
        panes[0].camera.orbit(0.4, 0.2);

        sync_linked_pane_cameras(&mut panes, 0, false);
        assert_ne!(panes[1].camera, panes[0].camera);

        sync_linked_pane_cameras(&mut panes, 0, true);
        assert_eq!(panes[1].camera, panes[0].camera);
        assert_eq!(panes[3].camera, panes[0].camera);
        // Chart pane camera stays untouched
        assert_eq!(panes[2].camera, Camera::default());
    }
}
