use crate::theme::{palette, section_card, UiTheme};
use paraviz::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct FileBrowserState {
    pub current_dir: PathBuf,
    pub filter_text: String,
    pub path_input: String,
}

impl FileBrowserState {
    pub fn new() -> Self {
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
pub struct RawVolumeModalState {
    pub path: String,
    pub dimensions: [usize; 3],
    pub scalar_type: RawScalarType,
    pub endian: RawEndian,
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub header_bytes: usize,
    pub array_name: String,
}

impl RawVolumeModalState {
    pub fn with_path(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            dimensions: [64, 64, 64],
            scalar_type: RawScalarType::U8,
            endian: RawEndian::Little,
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            header_bytes: 0,
            array_name: "Scalars".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuContourModalState {
    pub node_id: NodeId,
    pub available_arrays: Vec<(String, (f64, f64))>,
    pub field_name: String,
    pub iso_value: f64,
    pub range: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct ScreenshotModalState {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub use_viewport_size: bool,
}

#[derive(Debug, Clone)]
pub enum AppModal {
    OpenFile(String),
    FileBrowser(FileBrowserState),
    RawVolume(RawVolumeModalState),
    GpuContour(GpuContourModalState),
    SaveState(String),
    LoadState(String),
    ExportDataset(String),
    SaveScreenshot(ScreenshotModalState),
    DemoGallery,
    Shortcuts,
}

pub enum ModalAction {
    None,
    OpenFile(PathBuf),
    LoadState(PathBuf),
    SaveState(PathBuf),
    ExportDataset(PathBuf),
    SaveScreenshot { path: PathBuf, width: u32, height: u32 },
    ImportRawVolume(RawVolumeModalState),
    RunGpuContour {
        node_id: NodeId,
        field_name: String,
        iso_value: f64,
    },
    LoadDemo(DemoPreset),
}

pub fn is_supported_open_file(path: &Path) -> bool {
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
            | "raw"
            | "bin"
            | "vol"
    )
}

pub fn is_raw_volume_ext(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(ext.to_ascii_lowercase().as_str(), "raw" | "bin" | "vol")
}

pub fn draw_modal(
    ctx: &egui::Context,
    modal_slot: &mut Option<AppModal>,
    export_glb_colors: &mut bool,
    viewport_size: (u32, u32),
    theme: UiTheme,
) -> ModalAction {
    let Some(mut modal) = modal_slot.take() else {
        return ModalAction::None;
    };

    let pal = palette(theme);
    let mut keep_open = true;
    let mut action = ModalAction::None;

    match &mut modal {
        AppModal::FileBrowser(fb) => {
            let mut picked: Option<PathBuf> = None;
            egui::Window::new("File Browser")
                .collapsible(false)
                .resizable(true)
                .default_size([600.0, 450.0])
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
                        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
                            && ui.button("Home").clicked()
                        {
                            fb.current_dir = home;
                            fb.path_input = fb.current_dir.display().to_string();
                        }
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut fb.path_input).desired_width(300.0),
                        );
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() {
                                picked = Some(p);
                            }
                        }
                        if ui.button("Go").clicked() {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() {
                                picked = Some(p);
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Filter:");
                        ui.add(
                            egui::TextEdit::singleline(&mut fb.filter_text)
                                .desired_width(210.0)
                                .hint_text("search filename..."),
                        );
                        ui.label(
                            egui::RichText::new(
                                ".vtk .vtp .vti .vtu .pvtp .pvtu .stl .obj .ply .csv .raw .json",
                            )
                            .size(10.5)
                            .color(pal.muted),
                        );
                    });
                    ui.separator();

                    let mut dirs: Vec<(String, PathBuf)> = Vec::new();
                    let mut files: Vec<(String, PathBuf, u64)> = Vec::new();
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
                                    let sz = entry.metadata().map(|m| m.len()).unwrap_or(0);
                                    files.push((name, path, sz));
                                }
                            }
                            dirs.sort_by(|a, b| {
                                a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase())
                            });
                            files.sort_by(|a, b| {
                                a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase())
                            });
                        }
                        Err(e) => read_err = Some(format!("Cannot read directory: {e}")),
                    }

                    egui::ScrollArea::vertical()
                        .id_salt("file_browser_scroll")
                        .max_height(300.0)
                        .show(ui, |ui| {
                            if let Some(err) = read_err {
                                ui.colored_label(pal.danger, err);
                            } else if dirs.is_empty() && files.is_empty() {
                                ui.weak("No subdirectories or supported scientific files here.");
                            }
                            for (name, path) in dirs {
                                if ui
                                    .selectable_label(
                                        false,
                                        egui::RichText::new(format!("[DIR]  {name}"))
                                            .color(pal.accent),
                                    )
                                    .clicked()
                                {
                                    fb.current_dir = path;
                                    fb.path_input = fb.current_dir.display().to_string();
                                }
                            }
                            for (name, path, sz) in files {
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(false, format!("       {name}")).clicked()
                                    {
                                        picked = Some(path);
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(format_short_size(sz))
                                                    .size(10.5)
                                                    .monospace()
                                                    .color(pal.muted),
                                            );
                                        },
                                    );
                                });
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if let Some(path) = picked {
                if is_raw_volume_ext(&path) {
                    *modal_slot = Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                        path.display().to_string(),
                    )));
                    return ModalAction::None;
                }
                return ModalAction::OpenFile(path);
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::RawVolume(rv) => {
            let mut import_now = false;
            egui::Window::new("Import Raw 3D Volume (.raw / .bin / .vol)")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label("File path:");
                    ui.add(
                        egui::TextEdit::singleline(&mut rv.path)
                            .desired_width(380.0)
                            .hint_text("path/to/volume.raw"),
                    );
                    ui.add_space(4.0);
                    ui.label("Dimensions (X, Y, Z):");
                    ui.horizontal(|ui| {
                        for (i, prefix) in [(0, "X:"), (1, "Y:"), (2, "Z:")] {
                            ui.add(
                                egui::DragValue::new(&mut rv.dimensions[i])
                                    .speed(1)
                                    .range(1..=4096)
                                    .prefix(prefix),
                            );
                        }
                    });
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_label("Scalar Type")
                            .selected_text(format!("{:?}", rv.scalar_type))
                            .show_ui(ui, |ui| {
                                for st in [
                                    RawScalarType::U8,
                                    RawScalarType::I16,
                                    RawScalarType::U16,
                                    RawScalarType::I32,
                                    RawScalarType::U32,
                                    RawScalarType::I64,
                                    RawScalarType::F32,
                                    RawScalarType::F64,
                                ] {
                                    ui.selectable_value(&mut rv.scalar_type, st, format!("{st:?}"));
                                }
                            });
                        egui::ComboBox::from_label("Endian")
                            .selected_text(format!("{:?}", rv.endian))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut rv.endian,
                                    RawEndian::Little,
                                    "Little",
                                );
                                ui.selectable_value(&mut rv.endian, RawEndian::Big, "Big");
                            });
                    });
                    ui.label("Spacing (X, Y, Z):");
                    ui.horizontal(|ui| {
                        for (i, prefix) in [(0, "X:"), (1, "Y:"), (2, "Z:")] {
                            ui.add(
                                egui::DragValue::new(&mut rv.spacing[i])
                                    .speed(0.05)
                                    .range(1e-5..=1000.0)
                                    .prefix(prefix),
                            );
                        }
                    });
                    ui.label("Origin (X, Y, Z):");
                    ui.horizontal(|ui| {
                        for (i, prefix) in [(0, "X:"), (1, "Y:"), (2, "Z:")] {
                            ui.add(
                                egui::DragValue::new(&mut rv.origin[i])
                                    .speed(0.1)
                                    .prefix(prefix),
                            );
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Header Bytes:");
                        ui.add(egui::DragValue::new(&mut rv.header_bytes).speed(1));
                        ui.label("Array Name:");
                        ui.add(
                            egui::TextEdit::singleline(&mut rv.array_name).desired_width(120.0),
                        );
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Import Volume").clicked() {
                            import_now = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if import_now {
                if !rv.path.trim().is_empty() {
                    return ModalAction::ImportRawVolume(rv.clone());
                }
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::GpuContour(gc) => {
            let mut run_now = false;
            egui::Window::new("GPU Compute Contour (WebGPU Marching Cubes)")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    if !gc.available_arrays.is_empty() {
                        egui::ComboBox::from_label("Scalar Array")
                            .selected_text(&gc.field_name)
                            .show_ui(ui, |ui| {
                                for (name, rng) in &gc.available_arrays {
                                    if ui.selectable_label(gc.field_name == *name, name).clicked() {
                                        gc.field_name = name.clone();
                                        gc.range = *rng;
                                        gc.iso_value = 0.5 * (rng.0 + rng.1);
                                    }
                                }
                            });
                    }
                    let (mn, mx) = gc.range;
                    let span_ok = (mx - mn).abs() > 1e-12;
                    ui.label(format!("Data Range: [{mn:.4}, {mx:.4}]"));
                    if span_ok {
                        ui.add(
                            egui::Slider::new(&mut gc.iso_value, mn..=mx)
                                .text("Iso Value")
                                .max_decimals(4),
                        );
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut gc.iso_value)
                                .speed(0.5)
                                .prefix("Iso Value: "),
                        );
                    }
                    ui.horizontal(|ui| {
                        for (lbl, frac) in [("25%", 0.25), ("50%", 0.5), ("75%", 0.75)] {
                            if ui.small_button(lbl).clicked() {
                                gc.iso_value = mn + (mx - mn) * frac;
                            }
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Run GPU Contour").clicked() {
                            run_now = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if run_now {
                return ModalAction::RunGpuContour {
                    node_id: gc.node_id,
                    field_name: gc.field_name.clone(),
                    iso_value: gc.iso_value,
                };
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::SaveScreenshot(ss) => {
            let mut confirm = false;
            egui::Window::new("Save Screenshot (PNG)")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label("Output PNG path:");
                    ui.add(egui::TextEdit::singleline(&mut ss.path).desired_width(360.0));
                    ui.checkbox(&mut ss.use_viewport_size, "Match current viewport size");
                    if ss.use_viewport_size {
                        ss.width = viewport_size.0.max(640);
                        ss.height = viewport_size.1.max(480);
                    }
                    ui.horizontal(|ui| {
                        ui.add_enabled(
                            !ss.use_viewport_size,
                            egui::DragValue::new(&mut ss.width)
                                .range(64..=7680)
                                .prefix("W: "),
                        );
                        ui.add_enabled(
                            !ss.use_viewport_size,
                            egui::DragValue::new(&mut ss.height)
                                .range(64..=4320)
                                .prefix("H: "),
                        );
                    });
                    ui.horizontal(|ui| {
                        for (lbl, w, h) in [
                            ("720p", 1280, 720),
                            ("1080p", 1920, 1080),
                            ("1440p", 2560, 1440),
                            ("4K", 3840, 2160),
                        ] {
                            if ui.small_button(lbl).clicked() {
                                ss.use_viewport_size = false;
                                ss.width = w;
                                ss.height = h;
                            }
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Save PNG").clicked() {
                            confirm = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if confirm {
                let p = ss.path.trim();
                if !p.is_empty() {
                    return ModalAction::SaveScreenshot {
                        path: PathBuf::from(p),
                        width: ss.width.max(64),
                        height: ss.height.max(64),
                    };
                }
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::DemoGallery => {
            let mut chosen: Option<DemoPreset> = None;
            egui::Window::new("ParaViz Demo Gallery")
                .collapsible(false)
                .resizable(false)
                .default_width(520.0)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    let demos = [
                        (
                            DemoPreset::WaveletContour,
                            "3D Wavelet Volume + Parallel Marching Cubes Isosurfaces + Z-Slice",
                        ),
                        (
                            DemoPreset::VortexStreamlines,
                            "RK45 Swirling Vortex Streamlines + Bishop Tubes + 3D Velocity Arrow Glyphs",
                        ),
                        (
                            DemoPreset::GyroidMinimalSurface,
                            "Triply Periodic Minimal Surface (TPMS) Isosurface Extraction",
                        ),
                        (
                            DemoPreset::VolumeWavelet,
                            "Hardware-Free R32Float 3D Volume Raymarching with Gradient Shading",
                        ),
                        (
                            DemoPreset::SuperquadricClip,
                            "Superquadric Torus with Implicit Plane Clip & Barycentric Edges",
                        ),
                    ];
                    for (preset, desc) in demos {
                        section_card(ui, theme, preset.name(), |ui| {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(desc).size(11.5).color(pal.muted));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.button("Load").clicked() {
                                            chosen = Some(preset);
                                        }
                                    },
                                );
                            });
                        });
                    }
                    if ui.button("Close").clicked() {
                        keep_open = false;
                    }
                });

            if let Some(p) = chosen {
                return ModalAction::LoadDemo(p);
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::Shortcuts => {
            egui::Window::new("Keyboard & Mouse Controls")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    egui::Grid::new("shortcuts_grid")
                        .striped(true)
                        .min_col_width(140.0)
                        .show(ui, |ui| {
                            for (key, desc) in [
                                ("Left Drag (Orbit mode)", "Orbit camera around target"),
                                ("Shift + Left / Middle Drag", "Pan camera"),
                                ("Right Drag / Scroll", "Zoom / Dolly camera"),
                                ("Single Click", "3D Raycast Pick / Ruler point"),
                                ("Double Click", "Reset camera to fit scene"),
                                ("R", "Reset camera to all visible"),
                                ("F", "Focus camera on selected node"),
                                ("1 / 2 / 3", "Switch Orbit / Pick / Ruler mode"),
                                ("W", "Toggle 3D Filter Widgets"),
                                ("E", "Toggle Eye-Dome Lighting (EDL)"),
                                ("M", "Toggle 4x MSAA"),
                                ("G / B", "Toggle Axes Gizmo / Bounding Box"),
                                ("T", "Toggle Turntable spin"),
                                ("Tab", "Toggle Left Inspector Dock"),
                                ("Space", "Play / Pause time steps"),
                                ("Delete / Backspace", "Remove selected pipeline node"),
                                ("F1 / ?", "Open this shortcuts guide"),
                            ] {
                                ui.monospace(key);
                                ui.label(desc);
                                ui.end_row();
                            }
                        });
                    ui.add_space(6.0);
                    if ui.button("Close").clicked() {
                        keep_open = false;
                    }
                });
            if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::OpenFile(_)
        | AppModal::SaveState(_)
        | AppModal::LoadState(_)
        | AppModal::ExportDataset(_) => {
            let title = match &modal {
                AppModal::OpenFile(_) => "Open Scientific File",
                AppModal::SaveState(_) => "Save Pipeline State (JSON)",
                AppModal::LoadState(_) => "Load Pipeline State (JSON)",
                AppModal::ExportDataset(_) => "Export Active Dataset",
                _ => "",
            };
            let mut confirm = false;
            let mut switch_to_browser = false;

            egui::Window::new(title)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    let is_open = matches!(modal, AppModal::OpenFile(_));
                    let is_export = matches!(modal, AppModal::ExportDataset(_));
                    let path_str = match &mut modal {
                        AppModal::OpenFile(s)
                        | AppModal::SaveState(s)
                        | AppModal::LoadState(s)
                        | AppModal::ExportDataset(s) => s,
                        _ => unreachable!(),
                    };
                    ui.label("File path:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(path_str)
                            .desired_width(380.0)
                            .hint_text("path/to/file"),
                    );
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        confirm = true;
                    }
                    if is_export {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Format:");
                            for ext in
                                ["vtp", "vtu", "vti", "vtk", "stl", "obj", "ply", "csv", "glb"]
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
                                export_glb_colors,
                                "Include scalar vertex colors (COLOR_0)",
                            );
                        }
                    }
                    ui.separator();
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
                *modal_slot = Some(AppModal::FileBrowser(FileBrowserState::new()));
                return ModalAction::None;
            }

            if confirm {
                let path_str = match &modal {
                    AppModal::OpenFile(s)
                    | AppModal::SaveState(s)
                    | AppModal::LoadState(s)
                    | AppModal::ExportDataset(s) => s.trim(),
                    _ => "",
                };
                if !path_str.is_empty() {
                    let p = PathBuf::from(path_str);
                    action = match &modal {
                        AppModal::OpenFile(_) => {
                            if is_raw_volume_ext(&p) {
                                *modal_slot =
                                    Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                                        p.display().to_string(),
                                    )));
                                return ModalAction::None;
                            }
                            ModalAction::OpenFile(p)
                        }
                        AppModal::LoadState(_) => ModalAction::LoadState(p),
                        AppModal::SaveState(_) => ModalAction::SaveState(p),
                        AppModal::ExportDataset(_) => ModalAction::ExportDataset(p),
                        _ => ModalAction::None,
                    };
                }
            } else if keep_open {
                *modal_slot = Some(modal);
            }
        }
    }

    action
}

fn format_short_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
