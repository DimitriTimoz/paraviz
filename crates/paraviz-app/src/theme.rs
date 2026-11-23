use paraviz::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiTheme {
    DarkStudio,
    LightStudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportBgPreset {
    StudioSlate,
    DeepObsidian,
    ParaViewClassic,
    MidnightBlue,
    ArcticLight,
}

impl ViewportBgPreset {
    pub const ALL: &'static [Self] = &[
        Self::StudioSlate,
        Self::DeepObsidian,
        Self::ParaViewClassic,
        Self::MidnightBlue,
        Self::ArcticLight,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::StudioSlate => "Studio Slate",
            Self::DeepObsidian => "Deep Obsidian",
            Self::ParaViewClassic => "ParaView Classic",
            Self::MidnightBlue => "Midnight Blue",
            Self::ArcticLight => "Arctic Light",
        }
    }

    pub fn colors(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::StudioSlate => ([0.11, 0.14, 0.20], [0.22, 0.27, 0.36]),
            Self::DeepObsidian => ([0.05, 0.06, 0.09], [0.11, 0.13, 0.18]),
            Self::ParaViewClassic => ([0.20, 0.22, 0.30], [0.42, 0.45, 0.54]),
            Self::MidnightBlue => ([0.06, 0.10, 0.20], [0.15, 0.24, 0.40]),
            Self::ArcticLight => ([0.78, 0.83, 0.90], [0.94, 0.96, 0.99]),
        }
    }
}

pub struct ThemePalette {
    pub accent: egui::Color32,
    pub accent_dim: egui::Color32,
    pub card_bg: egui::Color32,
    pub card_border: egui::Color32,
    pub hud_bg: egui::Color32,
    pub hud_border: egui::Color32,
    pub success: egui::Color32,
    pub warning: egui::Color32,
    pub danger: egui::Color32,
    pub muted: egui::Color32,
    pub plot_bg: egui::Color32,
    pub plot_grid: egui::Color32,
    pub plot_axis: egui::Color32,
}

pub fn palette(theme: UiTheme) -> ThemePalette {
    match theme {
        UiTheme::DarkStudio => ThemePalette {
            accent: egui::Color32::from_rgb(56, 189, 248),
            accent_dim: egui::Color32::from_rgba_unmultiplied(56, 189, 248, 38),
            card_bg: egui::Color32::from_rgb(24, 29, 40),
            card_border: egui::Color32::from_rgb(44, 53, 72),
            hud_bg: egui::Color32::from_rgba_unmultiplied(13, 17, 25, 205),
            hud_border: egui::Color32::from_rgba_unmultiplied(92, 118, 160, 100),
            success: egui::Color32::from_rgb(74, 222, 128),
            warning: egui::Color32::from_rgb(251, 191, 36),
            danger: egui::Color32::from_rgb(248, 113, 113),
            muted: egui::Color32::from_rgb(142, 155, 178),
            plot_bg: egui::Color32::from_rgb(16, 20, 28),
            plot_grid: egui::Color32::from_rgba_unmultiplied(120, 138, 168, 42),
            plot_axis: egui::Color32::from_rgb(198, 210, 228),
        },
        UiTheme::LightStudio => ThemePalette {
            accent: egui::Color32::from_rgb(14, 116, 196),
            accent_dim: egui::Color32::from_rgba_unmultiplied(14, 116, 196, 30),
            card_bg: egui::Color32::from_rgb(244, 246, 250),
            card_border: egui::Color32::from_rgb(208, 215, 228),
            hud_bg: egui::Color32::from_rgba_unmultiplied(245, 248, 252, 220),
            hud_border: egui::Color32::from_rgba_unmultiplied(120, 140, 170, 120),
            success: egui::Color32::from_rgb(22, 163, 74),
            warning: egui::Color32::from_rgb(217, 119, 6),
            danger: egui::Color32::from_rgb(220, 38, 38),
            muted: egui::Color32::from_rgb(96, 108, 128),
            plot_bg: egui::Color32::from_rgb(252, 253, 255),
            plot_grid: egui::Color32::from_rgba_unmultiplied(100, 116, 140, 36),
            plot_axis: egui::Color32::from_rgb(50, 62, 82),
        },
    }
}

/// Applies the ParaViz Studio visual theme and spacing to `ctx`.
pub fn apply_theme(ctx: &egui::Context, theme: UiTheme) {
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(7.0, 5.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(12);
    style.spacing.menu_margin = egui::Margin::same(8);
    style.spacing.indent = 16.0;
    style.spacing.slider_width = 135.0;

    let mut vis = match theme {
        UiTheme::DarkStudio => egui::Visuals::dark(),
        UiTheme::LightStudio => egui::Visuals::light(),
    };

    let pal = palette(theme);
    vis.window_corner_radius = egui::CornerRadius::same(8);
    vis.menu_corner_radius = egui::CornerRadius::same(6);
    vis.slider_trailing_fill = true;
    vis.striped = true;
    vis.collapsing_header_frame = true;

    match theme {
        UiTheme::DarkStudio => {
            vis.panel_fill = egui::Color32::from_rgb(17, 21, 30);
            vis.window_fill = egui::Color32::from_rgb(21, 26, 36);
            vis.extreme_bg_color = egui::Color32::from_rgb(12, 15, 22);
            vis.faint_bg_color = egui::Color32::from_rgb(24, 30, 42);
            vis.code_bg_color = egui::Color32::from_rgb(14, 18, 26);
            vis.window_stroke = egui::Stroke::new(1.0, pal.card_border);
            vis.hyperlink_color = pal.accent;
            vis.selection = egui::style::Selection {
                bg_fill: egui::Color32::from_rgba_unmultiplied(56, 189, 248, 55),
                stroke: egui::Stroke::new(1.0, pal.accent),
            };

            let r = egui::CornerRadius::same(5);
            vis.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(24, 29, 40);
            vis.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(20, 25, 35);
            vis.widgets.noninteractive.bg_stroke =
                egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 48, 66));
            vis.widgets.noninteractive.fg_stroke =
                egui::Stroke::new(1.0, egui::Color32::from_rgb(218, 226, 238));
            vis.widgets.noninteractive.corner_radius = r;

            vis.widgets.inactive.bg_fill = egui::Color32::from_rgb(32, 39, 54);
            vis.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(28, 34, 47);
            vis.widgets.inactive.bg_stroke =
                egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 58, 78));
            vis.widgets.inactive.fg_stroke =
                egui::Stroke::new(1.0, egui::Color32::from_rgb(225, 232, 244));
            vis.widgets.inactive.corner_radius = r;

            vis.widgets.hovered.bg_fill = egui::Color32::from_rgb(44, 54, 74);
            vis.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(39, 48, 66);
            vis.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, pal.accent);
            vis.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
            vis.widgets.hovered.corner_radius = r;

            vis.widgets.active.bg_fill = egui::Color32::from_rgb(28, 92, 132);
            vis.widgets.active.weak_bg_fill = egui::Color32::from_rgb(28, 92, 132);
            vis.widgets.active.bg_stroke = egui::Stroke::new(1.0, pal.accent);
            vis.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
            vis.widgets.active.corner_radius = r;

            vis.widgets.open.bg_fill = egui::Color32::from_rgb(36, 45, 62);
            vis.widgets.open.weak_bg_fill = egui::Color32::from_rgb(36, 45, 62);
            vis.widgets.open.bg_stroke = egui::Stroke::new(1.0, pal.accent);
            vis.widgets.open.corner_radius = r;
        }
        UiTheme::LightStudio => {
            vis.panel_fill = egui::Color32::from_rgb(236, 240, 246);
            vis.window_fill = egui::Color32::from_rgb(248, 250, 253);
            vis.extreme_bg_color = egui::Color32::from_rgb(255, 255, 255);
            vis.faint_bg_color = egui::Color32::from_rgb(229, 234, 243);
            vis.window_stroke = egui::Stroke::new(1.0, pal.card_border);
            vis.hyperlink_color = pal.accent;
            vis.selection = egui::style::Selection {
                bg_fill: egui::Color32::from_rgba_unmultiplied(14, 116, 196, 40),
                stroke: egui::Stroke::new(1.0, pal.accent),
            };

            let r = egui::CornerRadius::same(5);
            vis.widgets.inactive.corner_radius = r;
            vis.widgets.hovered.corner_radius = r;
            vis.widgets.active.corner_radius = r;
            vis.widgets.noninteractive.corner_radius = r;
            vis.widgets.open.corner_radius = r;
        }
    }

    style.visuals = vis;
    ctx.set_global_style(style);
}

/// Renders a styled card container with a subtle border and section heading.
pub fn section_card(
    ui: &mut egui::Ui,
    theme: UiTheme,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let pal = palette(theme);
    egui::Frame::new()
        .fill(pal.card_bg)
        .stroke(egui::Stroke::new(1.0, pal.card_border))
        .corner_radius(egui::CornerRadius::same(7))
        .inner_margin(egui::Margin::same(9))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            if !title.is_empty() {
                ui.label(
                    egui::RichText::new(title)
                        .size(12.5)
                        .strong()
                        .color(pal.accent),
                );
                ui.add_space(4.0);
            }
            add_contents(ui);
        });
    ui.add_space(5.0);
}

/// Renders a compact pill badge.
pub fn pill_badge(ui: &mut egui::Ui, text: &str, bg: egui::Color32, fg: egui::Color32) {
    egui::Frame::new()
        .fill(bg)
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(10.0)
                    .monospace()
                    .strong()
                    .color(fg),
            );
        });
}

/// Returns `(short_tag, bg_color, text_color)` for a pipeline node's dataset/kind.
pub fn node_badge_style(ds: Option<&DataSet>, is_filter: bool) -> (&'static str, egui::Color32, egui::Color32) {
    match ds {
        Some(DataSet::ImageData(_)) => (
            "IMG",
            egui::Color32::from_rgba_unmultiplied(168, 85, 247, 50),
            egui::Color32::from_rgb(216, 180, 254),
        ),
        Some(DataSet::PolyData(_)) => (
            "POLY",
            egui::Color32::from_rgba_unmultiplied(56, 189, 248, 45),
            egui::Color32::from_rgb(125, 211, 252),
        ),
        Some(DataSet::UnstructuredGrid(_)) => (
            "UG",
            egui::Color32::from_rgba_unmultiplied(245, 158, 11, 45),
            egui::Color32::from_rgb(252, 211, 77),
        ),
        Some(DataSet::RectilinearGrid(_)) | Some(DataSet::StructuredGrid(_)) => (
            "GRID",
            egui::Color32::from_rgba_unmultiplied(52, 211, 153, 45),
            egui::Color32::from_rgb(110, 231, 183),
        ),
        Some(DataSet::MultiBlock(_)) => (
            "BLK",
            egui::Color32::from_rgba_unmultiplied(244, 114, 182, 45),
            egui::Color32::from_rgb(249, 168, 212),
        ),
        None => {
            if is_filter {
                (
                    "FLT",
                    egui::Color32::from_rgba_unmultiplied(148, 163, 184, 40),
                    egui::Color32::from_rgb(203, 213, 225),
                )
            } else {
                (
                    "SRC",
                    egui::Color32::from_rgba_unmultiplied(148, 163, 184, 40),
                    egui::Color32::from_rgb(203, 213, 225),
                )
            }
        }
    }
}

/// Paints a crisp vector eye toggle icon inside `ui` and returns its `Response`.
pub fn visibility_eye_button(ui: &mut egui::Ui, visible: bool, theme: UiTheme) -> egui::Response {
    let size = egui::vec2(20.0, 18.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let pal = palette(theme);
    let painter = ui.painter_at(rect);
    let c = rect.center();

    let stroke_col = if visible {
        if resp.hovered() {
            egui::Color32::WHITE
        } else {
            pal.accent
        }
    } else if resp.hovered() {
        pal.muted
    } else {
        pal.muted.gamma_multiply(0.55)
    };

    if resp.hovered() {
        painter.rect_filled(rect, 4.0, pal.accent_dim);
    }

    // Almond eye outline
    let w = 6.5f32;
    let h = 4.2f32;
    let top_pts = vec![
        egui::pos2(c.x - w, c.y),
        egui::pos2(c.x - w * 0.45, c.y - h),
        egui::pos2(c.x + w * 0.45, c.y - h),
        egui::pos2(c.x + w, c.y),
    ];
    let bot_pts = vec![
        egui::pos2(c.x - w, c.y),
        egui::pos2(c.x - w * 0.45, c.y + h),
        egui::pos2(c.x + w * 0.45, c.y + h),
        egui::pos2(c.x + w, c.y),
    ];
    painter.add(egui::Shape::line(top_pts, egui::Stroke::new(1.3, stroke_col)));
    painter.add(egui::Shape::line(bot_pts, egui::Stroke::new(1.3, stroke_col)));

    if visible {
        painter.circle_filled(c, 2.3, stroke_col);
    } else {
        painter.line_segment(
            [egui::pos2(c.x - 5.5, c.y + 4.5), egui::pos2(c.x + 5.5, c.y - 4.5)],
            egui::Stroke::new(1.3, stroke_col),
        );
    }

    resp
}

/// Paints a small geometric cube/prism logo mark in the menu bar.
pub fn paint_logo_badge(ui: &mut egui::Ui, theme: UiTheme) {
    let pal = palette(theme);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 5.0, egui::Color32::from_rgb(79, 70, 229));
    let c = rect.center();
    let r = 5.5;
    let pts = [
        egui::pos2(c.x, c.y - r),
        egui::pos2(c.x + r * 0.88, c.y - r * 0.35),
        egui::pos2(c.x + r * 0.88, c.y + r * 0.55),
        egui::pos2(c.x, c.y + r),
        egui::pos2(c.x - r * 0.88, c.y + r * 0.55),
        egui::pos2(c.x - r * 0.88, c.y - r * 0.35),
        egui::pos2(c.x, c.y - r),
    ];
    p.add(egui::Shape::line(pts.to_vec(), egui::Stroke::new(1.3, egui::Color32::WHITE)));
    p.line_segment([c, egui::pos2(c.x, c.y + r)], egui::Stroke::new(1.1, pal.accent));
    p.line_segment(
        [c, egui::pos2(c.x + r * 0.88, c.y - r * 0.35)],
        egui::Stroke::new(1.1, pal.accent),
    );
    p.line_segment(
        [c, egui::pos2(c.x - r * 0.88, c.y - r * 0.35)],
        egui::Stroke::new(1.1, pal.accent),
    );
}
