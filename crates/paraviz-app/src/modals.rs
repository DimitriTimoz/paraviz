use crate::theme::{palette, section_card, UiTheme};
use paraviz::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct FileBrowserState {
    pub current_dir: PathBuf,
    pub filter_text: String,
    pub path_input: String,
    pub reader: paraviz::io::DatasetReader,
    pub return_to_modal: Option<Box<AppModal>>,
}

impl FileBrowserState {
    pub fn new() -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let path_input = current_dir.display().to_string();
        Self {
            current_dir,
            filter_text: String::new(),
            path_input,
            reader: paraviz::io::DatasetReader::Auto,
            return_to_modal: None,
        }
    }

    pub fn with_reader(mut self, reader: paraviz::io::DatasetReader) -> Self {
        self.reader = reader;
        self
    }

    pub fn for_save_modal(current_path: &str, return_modal: AppModal) -> Self {
        let p = Path::new(current_path.trim());
        let dir = p
            .parent()
            .filter(|d| !d.as_os_str().is_empty() && d.is_dir())
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let path_input = dir.display().to_string();
        Self {
            current_dir: dir,
            filter_text: String::new(),
            path_input,
            reader: paraviz::io::DatasetReader::Auto,
            return_to_modal: Some(Box::new(return_modal)),
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
    pub transparent: bool,
    pub include_scalar_bar: bool,
}

#[derive(Debug, Clone)]
pub struct GifModalState {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub delay_ms: u16,
    pub is_turntable: bool,
    pub transparent: bool,
    pub include_scalar_bar: bool,
}

#[derive(Debug, Clone)]
pub struct ExportSpreadsheetModalState {
    pub path: String,
    pub domain_label: String,
    pub col_filter: String,
    pub row_count: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AppModal {
    OpenFile {
        path: String,
        reader: paraviz::io::DatasetReader,
    },
    FileBrowser(FileBrowserState),
    RawVolume(RawVolumeModalState),
    GpuContour(GpuContourModalState),
    SaveState(String),
    LoadState(String),
    ExportDataset(String),
    ExportSpreadsheetCsv(ExportSpreadsheetModalState),
    SaveScreenshot(ScreenshotModalState),
    ExportGif(GifModalState),
    DemoGallery,
    Shortcuts,
}

pub enum ModalAction {
    None,
    OpenFile(PathBuf, paraviz::io::DatasetReader),
    LoadState(PathBuf),
    SaveState(PathBuf),
    ExportDataset(PathBuf),
    ExportSpreadsheetCsv(PathBuf),
    SaveScreenshot {
        path: PathBuf,
        width: u32,
        height: u32,
        transparent: bool,
        include_scalar_bar: bool,
    },
    ExportGif {
        path: PathBuf,
        width: u32,
        height: u32,
        frames: u32,
        delay_ms: u16,
        is_turntable: bool,
        transparent: bool,
        include_scalar_bar: bool,
    },
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
            | "pvtr"
            | "pvts"
            | "xdmf"
            | "xmf"
            | "vtkhdf"
            | "cgns"
            | "exo"
            | "e"
            | "ex2"
            | "h5"
            | "hdf"
            | "hdf5"
            | "case"
            | "nrrd"
            | "nhdr"
            | "mhd"
            | "mha"
            | "nii"
            | "gz"
            | "foam"
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
            let mut pick_folder_for_save = false;
            let mut cancel_to_return = false;
            let is_save_picker = fb.return_to_modal.is_some();
            let win_title = if is_save_picker {
                "Choose Export Destination Folder"
            } else {
                "File Browser"
            };
            egui::Window::new(win_title)
                .collapsible(false)
                .resizable(true)
                .default_size([620.0, 460.0])
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui.button(".. Up").clicked()
                            && let Some(parent) = fb.current_dir.parent()
                        {
                            fb.current_dir = parent.to_path_buf();
                            fb.path_input = fb.current_dir.display().to_string();
                        }
                        for (label, dir_opt) in quick_location_dirs() {
                            if let Some(dir) = dir_opt
                                && ui.button(label).clicked()
                            {
                                fb.current_dir = dir;
                                fb.path_input = fb.current_dir.display().to_string();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut fb.path_input).desired_width(480.0),
                        );
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() && !is_save_picker {
                                picked = Some(p);
                            }
                        }
                        if ui.button("Go").clicked() {
                            let p = PathBuf::from(fb.path_input.trim());
                            if p.is_dir() {
                                fb.current_dir = p;
                            } else if p.is_file() && !is_save_picker {
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
                        if !is_save_picker {
                            ui.label(
                                egui::RichText::new(
                                    ".vtk .vtp .vti .vtu .pvtp .pvtu .vtkhdf .cgns .exo .e .ex2 .stl .obj .ply .csv .raw .json",
                                )
                                .size(10.5)
                                .color(pal.muted),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("Navigate into a folder, then click \"Use This Folder\"")
                                    .size(11.0)
                                    .color(pal.muted),
                            );
                        }
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
                        if is_save_picker {
                            if ui.button("Use This Folder").clicked() {
                                pick_folder_for_save = true;
                            }
                            if ui.button("Cancel").clicked() {
                                cancel_to_return = true;
                            }
                        } else {
                            ui.label("Reader:");
                            egui::ComboBox::from_id_salt("file_browser_reader_combo")
                                .selected_text(fb.reader.label())
                                .width(260.0)
                                .show_ui(ui, |ui| {
                                    for &r in paraviz::io::DatasetReader::ALL {
                                        ui.selectable_value(&mut fb.reader, r, r.label());
                                    }
                                });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("Close").clicked() {
                                        keep_open = false;
                                    }
                                },
                            );
                        }
                    });
                });

            if pick_folder_for_save
                && let Some(mut ret) = fb.return_to_modal.take()
            {
                apply_dir_to_modal(&mut ret, &fb.current_dir);
                *modal_slot = Some(*ret);
                return ModalAction::None;
            }
            if cancel_to_return
                && let Some(ret) = fb.return_to_modal.take()
            {
                *modal_slot = Some(*ret);
                return ModalAction::None;
            }
            if let Some(path) = picked {
                if let Some(mut ret) = fb.return_to_modal.take() {
                    if let Some(parent) = path.parent() {
                        apply_dir_to_modal(&mut ret, parent);
                    }
                    *modal_slot = Some(*ret);
                    return ModalAction::None;
                }
                if is_raw_volume_ext(&path) && fb.reader == paraviz::io::DatasetReader::Auto {
                    *modal_slot = Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                        path.display().to_string(),
                    )));
                    return ModalAction::None;
                }
                return ModalAction::OpenFile(path, fb.reader);
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
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut rv.path)
                                .desired_width(305.0)
                                .hint_text("path/to/volume.raw"),
                        );
                        #[cfg(not(target_arch = "wasm32"))]
                        if ui.button("Browse...").clicked()
                            && let Some(p) = rfd::FileDialog::new()
                                .set_title("Select Raw 3D Volume")
                                .add_filter("Raw Volumes", &["raw", "bin", "vol"])
                                .add_filter("All Files", &["*"])
                                .pick_file()
                        {
                            rv.path = p.display().to_string();
                        }
                    });
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
                    draw_save_destination_picker(
                        ui,
                        "Output PNG path:",
                        &mut ss.path,
                        "Save Screenshot (PNG)",
                        &[("PNG Image", &["png"])],
                    );
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
                    ui.checkbox(&mut ss.transparent, "Transparent background (alpha = 0)");
                    ui.checkbox(&mut ss.include_scalar_bar, "Include Scalar Color Bar");
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
                        transparent: ss.transparent,
                        include_scalar_bar: ss.include_scalar_bar,
                    };
                }
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::ExportGif(gm) => {
            let mut confirm = false;
            egui::Window::new("Export Animation (GIF)")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    draw_save_destination_picker(
                        ui,
                        "Output GIF path:",
                        &mut gm.path,
                        "Export Animation (GIF)",
                        &[("GIF Animation", &["gif"])],
                    );
                    ui.horizontal(|ui| {
                        ui.label("Mode:");
                        ui.radio_value(&mut gm.is_turntable, true, "360° Turntable");
                        ui.radio_value(&mut gm.is_turntable, false, "Time Series");
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut gm.width)
                                .range(64..=1920)
                                .prefix("W: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut gm.height)
                                .range(64..=1080)
                                .prefix("H: "),
                        );
                        if gm.is_turntable {
                            ui.add(
                                egui::DragValue::new(&mut gm.frames)
                                    .range(4..=180)
                                    .prefix("Frames: "),
                            );
                        }
                        ui.add(
                            egui::DragValue::new(&mut gm.delay_ms)
                                .range(20..=2000)
                                .prefix("Delay ms: "),
                        );
                    });
                    ui.checkbox(&mut gm.transparent, "Transparent background (alpha = 0)");
                    ui.checkbox(&mut gm.include_scalar_bar, "Include Scalar Color Bar");
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Export GIF").clicked() {
                            confirm = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if confirm {
                let p = gm.path.trim();
                if !p.is_empty() {
                    return ModalAction::ExportGif {
                        path: PathBuf::from(p),
                        width: gm.width.max(64),
                        height: gm.height.max(64),
                        frames: gm.frames.max(4),
                        delay_ms: gm.delay_ms.max(20),
                        is_turntable: gm.is_turntable,
                        transparent: gm.transparent,
                        include_scalar_bar: gm.include_scalar_bar,
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
                                ("Ctrl/Cmd + C / V", "Copy / Paste filter in Pipeline"),
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

        AppModal::ExportSpreadsheetCsv(es) => {
            let mut confirm = false;
            egui::Window::new("Export Spreadsheet CSV")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Domain:").color(pal.muted));
                        ui.strong(&es.domain_label);
                        ui.label(
                            egui::RichText::new(format!("({} rows)", es.row_count)).color(pal.muted),
                        );
                    });
                    if !es.col_filter.trim().is_empty() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Column filter:").color(pal.muted));
                            ui.monospace(es.col_filter.trim());
                        });
                    }
                    draw_save_destination_picker(
                        ui,
                        "Output CSV path:",
                        &mut es.path,
                        "Export Spreadsheet CSV",
                        &[("CSV Table", &["csv"])],
                    );
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Export CSV").clicked() {
                            confirm = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });

            if confirm {
                let p = es.path.trim();
                if !p.is_empty() {
                    return ModalAction::ExportSpreadsheetCsv(PathBuf::from(p));
                }
            } else if keep_open {
                *modal_slot = Some(modal);
            }
            return ModalAction::None;
        }

        AppModal::OpenFile { .. }
        | AppModal::SaveState(_)
        | AppModal::LoadState(_)
        | AppModal::ExportDataset(_) => {
            let title = match &modal {
                AppModal::OpenFile { .. } => "Open Scientific File",
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
                    let is_open = matches!(modal, AppModal::OpenFile { .. });
                    let is_export = matches!(modal, AppModal::ExportDataset(_));
                    let is_save = matches!(modal, AppModal::SaveState(_) | AppModal::ExportDataset(_));
                    let path_str = match &mut modal {
                        AppModal::OpenFile { path, .. }
                        | AppModal::SaveState(path)
                        | AppModal::LoadState(path)
                        | AppModal::ExportDataset(path) => path,
                        _ => unreachable!(),
                    };
                    if is_save {
                        let (dlg_title, filters): (&str, &[(&str, &[&str])]) = if is_export {
                            (
                                "Export Active Dataset",
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
                            )
                        } else {
                            ("Save Pipeline State", &[("ParaViz State JSON", &["json"])])
                        };
                        draw_save_destination_picker(ui, "Save to:", path_str, dlg_title, filters);
                    } else {
                        ui.label("File path:");
                        let resp = ui.add(
                            egui::TextEdit::singleline(path_str)
                                .desired_width(380.0)
                                .hint_text("path/to/file"),
                        );
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            confirm = true;
                        }
                    }
                    if is_export {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Format:");
                            for ext in
                                ["vtp", "vtu", "vti", "vtk", "stl", "obj", "ply", "csv", "glb", "html"]
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
                        let lower = path_str.trim().to_ascii_lowercase();
                        if lower.ends_with(".glb") || lower.ends_with(".html") || lower.ends_with(".htm") {
                            ui.checkbox(
                                export_glb_colors,
                                "Include scalar vertex colors (colormap)",
                            );
                        }
                    }
                    if let AppModal::OpenFile { path, reader } = &mut modal {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Reader:");
                            egui::ComboBox::from_id_salt("open_file_reader_combo")
                                .selected_text(reader.label())
                                .width(280.0)
                                .show_ui(ui, |ui| {
                                    for &r in paraviz::io::DatasetReader::ALL {
                                        ui.selectable_value(reader, r, r.label());
                                    }
                                });
                        });
                        let trimmed = path.trim();
                        if !trimmed.is_empty() && *reader == paraviz::io::DatasetReader::Auto {
                            if let Ok(detected) = paraviz::io::detect_reader(trimmed) {
                                ui.label(
                                    egui::RichText::new(format!("Detected: {}", detected.short_name()))
                                        .small()
                                        .weak(),
                                );
                            }
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
                let fb = match modal {
                    AppModal::SaveState(ref s) | AppModal::ExportDataset(ref s) => {
                        let p = s.clone();
                        FileBrowserState::for_save_modal(&p, modal)
                    }
                    AppModal::OpenFile { reader, .. } => FileBrowserState::new().with_reader(reader),
                    _ => FileBrowserState::new(),
                };
                *modal_slot = Some(AppModal::FileBrowser(fb));
                return ModalAction::None;
            }

            if confirm {
                let path_str = match &modal {
                    AppModal::OpenFile { path, .. }
                    | AppModal::SaveState(path)
                    | AppModal::LoadState(path)
                    | AppModal::ExportDataset(path) => path.trim(),
                    _ => "",
                };
                if !path_str.is_empty() {
                    let p = PathBuf::from(path_str);
                    action = match &modal {
                        AppModal::OpenFile { reader, .. } => {
                            if is_raw_volume_ext(&p) {
                                *modal_slot =
                                    Some(AppModal::RawVolume(RawVolumeModalState::with_path(
                                        p.display().to_string(),
                                    )));
                                return ModalAction::None;
                            }
                            ModalAction::OpenFile(p, *reader)
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

fn quick_location_dirs() -> [(&'static str, Option<PathBuf>); 5] {
    let cwd = std::env::current_dir().ok();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let desktop = home.as_ref().map(|h| h.join("Desktop")).filter(|p| p.is_dir());
    let downloads = home
        .as_ref()
        .map(|h| h.join("Downloads"))
        .filter(|p| p.is_dir());
    let documents = home
        .as_ref()
        .map(|h| h.join("Documents"))
        .filter(|p| p.is_dir());
    [
        ("Desktop", desktop),
        ("Downloads", downloads),
        ("Documents", documents),
        ("Home", home),
        ("CWD", cwd),
    ]
}

fn replace_parent_dir(path_str: &mut String, dir: &Path) {
    let cur = Path::new(path_str.trim());
    let fname = cur
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("export");
    *path_str = dir.join(fname).display().to_string();
}

fn apply_dir_to_modal(modal: &mut AppModal, dir: &Path) {
    match modal {
        AppModal::SaveScreenshot(ss) => replace_parent_dir(&mut ss.path, dir),
        AppModal::ExportGif(gm) => replace_parent_dir(&mut gm.path, dir),
        AppModal::ExportSpreadsheetCsv(es) => replace_parent_dir(&mut es.path, dir),
        AppModal::SaveState(s) | AppModal::ExportDataset(s) => replace_parent_dir(s, dir),
        _ => {}
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_pick_scientific_file(start_dir: Option<&Path>) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new()
        .set_title("Open Scientific Dataset")
        .add_filter(
            "Scientific Datasets",
            &[
                "vtp", "vtu", "vti", "vtr", "vts", "vtk", "pvtp", "pvtu", "pvti", "pvtr", "pvts",
                "pvd", "vtkhdf", "cgns", "exo", "e", "ex2", "h5", "hdf", "hdf5", "stl", "obj",
                "ply", "csv", "nii", "gz", "mhd", "mha", "nrrd", "nhdr", "case", "xdmf", "xmf",
                "raw", "bin", "vol", "foam", "json",
            ],
        )
        .add_filter("All Files", &["*"]);
    if let Some(dir) = start_dir.filter(|d| d.is_dir()) {
        dlg = dlg.set_directory(dir);
    }
    dlg.pick_file()
}

#[cfg(target_arch = "wasm32")]
pub fn native_pick_scientific_file(_start_dir: Option<&Path>) -> Option<PathBuf> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_pick_state_file(start_dir: Option<&Path>) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new()
        .set_title("Load Pipeline State")
        .add_filter("ParaViz State JSON", &["json"])
        .add_filter("All Files", &["*"]);
    if let Some(dir) = start_dir.filter(|d| d.is_dir()) {
        dlg = dlg.set_directory(dir);
    }
    dlg.pick_file()
}

#[cfg(target_arch = "wasm32")]
pub fn native_pick_state_file(_start_dir: Option<&Path>) -> Option<PathBuf> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_save_file(
    title: &str,
    current_path: &str,
    filters: &[(&str, &[&str])],
) -> Option<PathBuf> {
    let mut dlg = rfd::FileDialog::new().set_title(title);
    let p = Path::new(current_path.trim());
    if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty() && d.is_dir()) {
        dlg = dlg.set_directory(dir);
    }
    if let Some(fname) = p.file_name().and_then(|s| s.to_str()).filter(|s| !s.is_empty()) {
        dlg = dlg.set_file_name(fname);
    }
    for (name, exts) in filters {
        dlg = dlg.add_filter(*name, exts);
    }
    dlg = dlg.add_filter("All Files", &["*"]);
    dlg.save_file()
}

#[cfg(target_arch = "wasm32")]
pub fn native_save_file(
    _title: &str,
    _current_path: &str,
    _filters: &[(&str, &[&str])],
) -> Option<PathBuf> {
    None
}

#[cfg(target_arch = "wasm32")]
pub fn spawn_wasm_open_dialog(
    queue: std::sync::Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>,
    json_only: bool,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let mut dlg = rfd::AsyncFileDialog::new();
        if json_only {
            dlg = dlg
                .set_title("Load Pipeline State")
                .add_filter("ParaViz State JSON", &["json"]);
        } else {
            dlg = dlg.set_title("Open File").add_filter(
                "ParaViz Files",
                &[
                    "json", "vtp", "vtu", "vti", "vtk", "stl", "obj", "ply", "csv",
                ],
            );
        }
        if let Some(file) = dlg.pick_file().await {
            let name = file.file_name();
            let bytes = file.read().await;
            if let Ok(mut q) = queue.lock() {
                q.push((name, bytes));
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
pub fn wasm_download_bytes(filename: &str, bytes: &[u8], mime: &str) {
    use wasm_bindgen::JsCast as _;
    let Ok(uint8) = js_sys::Uint8Array::from(bytes).dyn_into::<wasm_bindgen::JsValue>() else {
        return;
    };
    let parts = js_sys::Array::of1(&uint8);
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &bag) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };
    if let Some(doc) = web_sys::window().and_then(|w| w.document())
        && let Ok(el) = doc.create_element("a")
        && let Ok(anchor) = el.dyn_into::<web_sys::HtmlAnchorElement>()
    {
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}

fn draw_save_destination_picker(
    ui: &mut egui::Ui,
    label: &str,
    path_str: &mut String,
    dialog_title: &str,
    filters: &[(&str, &[&str])],
) {
    ui.label(label);
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(path_str).desired_width(305.0));
        if ui
            .button("Browse...")
            .on_hover_text("Open native file explorer")
            .clicked()
            && let Some(picked) = native_save_file(dialog_title, path_str, filters)
        {
            *path_str = picked.display().to_string();
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Folder:").size(11.0));
        for (name, dir_opt) in quick_location_dirs() {
            if let Some(dir) = dir_opt
                && ui.small_button(name).clicked()
            {
                replace_parent_dir(path_str, &dir);
            }
        }
    });
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
