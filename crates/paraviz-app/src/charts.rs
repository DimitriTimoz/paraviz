use crate::theme::{palette, UiTheme};
use paraviz::prelude::*;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartMode {
    XYPlot,
    Histogram,
}

pub struct ChartState {
    pub mode: ChartMode,
    pub hidden_series: HashSet<String>,
    pub show_markers: bool,
    pub log_y: bool,
    pub hist_array_name: String,
    pub hist_association: AttributeAssociation,
    pub hist_component: Option<usize>,
    pub hist_bins: usize,
    pub csv_export_path: String,
}

impl Default for ChartState {
    fn default() -> Self {
        Self {
            mode: ChartMode::Histogram,
            hidden_series: HashSet::new(),
            show_markers: false,
            log_y: false,
            hist_array_name: String::new(),
            hist_association: AttributeAssociation::Point,
            hist_component: None,
            hist_bins: 24,
            csv_export_path: "chart_export.csv".into(),
        }
    }
}

const CHART_PALETTE: [egui::Color32; 6] = [
    egui::Color32::from_rgb(56, 189, 248),
    egui::Color32::from_rgb(251, 146, 60),
    egui::Color32::from_rgb(74, 222, 128),
    egui::Color32::from_rgb(244, 114, 182),
    egui::Color32::from_rgb(192, 132, 252),
    egui::Color32::from_rgb(250, 204, 21),
];

/// Renders the Chart / Histogram drawer tab. Returns `Some(Ok(msg))` or `Some(Err(msg))`
/// when a CSV export action runs.
pub fn draw_chart_tab(
    ui: &mut egui::Ui,
    ds: &DataSet,
    state: &mut ChartState,
    theme: UiTheme,
) -> Option<Result<String, String>> {
    let mut feedback: Option<Result<String, String>> = None;

    let has_arc_length = ds
        .as_poly_data()
        .and_then(|pd| pd.point_data.get("ArcLength"))
        .is_some();

    if !has_arc_length && state.mode == ChartMode::XYPlot {
        state.mode = ChartMode::Histogram;
    }

    ui.horizontal_wrapped(|ui| {
        if has_arc_length {
            ui.selectable_value(&mut state.mode, ChartMode::XYPlot, "XY Plot (ArcLength)");
            ui.selectable_value(&mut state.mode, ChartMode::Histogram, "Scalar Histogram");
            ui.separator();
        }

        if state.mode == ChartMode::XYPlot && has_arc_length {
            ui.checkbox(&mut state.show_markers, "Markers");
            ui.checkbox(&mut state.log_y, "Log Y");
            ui.separator();
            ui.add(
                egui::TextEdit::singleline(&mut state.csv_export_path)
                    .desired_width(140.0)
                    .hint_text("plot.csv"),
            );
            if ui.button("Export CSV").clicked()
                && let Some(pd) = ds.as_poly_data()
            {
                feedback = Some(export_xy_csv(pd, Path::new(state.csv_export_path.trim())));
            }
        } else {
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

            let current_valid = candidates
                .iter()
                .any(|(a, n, _)| *a == state.hist_association && *n == state.hist_array_name);
            if !current_valid {
                if let Some(active) = ds.point_data().and_then(|p| p.active_scalars_array()) {
                    state.hist_association = AttributeAssociation::Point;
                    state.hist_array_name = active.name.clone();
                } else if let Some(active) = ds.cell_data().and_then(|c| c.active_scalars_array()) {
                    state.hist_association = AttributeAssociation::Cell;
                    state.hist_array_name = active.name.clone();
                } else {
                    state.hist_association = candidates[0].0;
                    state.hist_array_name = candidates[0].1.clone();
                }
                state.hist_component = None;
            }

            let selected_comps = candidates
                .iter()
                .find(|(a, n, _)| *a == state.hist_association && *n == state.hist_array_name)
                .map(|(_, _, c)| *c)
                .unwrap_or(1);

            let sel_label = format!(
                "{}: {}",
                match state.hist_association {
                    AttributeAssociation::Point => "Point",
                    AttributeAssociation::Cell => "Cell",
                    AttributeAssociation::Field => "Field",
                },
                state.hist_array_name
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
                        let active =
                            *assoc == state.hist_association && *name == state.hist_array_name;
                        if ui
                            .selectable_label(active, format!("{prefix}: {name}"))
                            .clicked()
                        {
                            state.hist_association = *assoc;
                            state.hist_array_name = name.clone();
                            state.hist_component = None;
                        }
                    }
                });

            if selected_comps > 1 {
                let comp_text = match state.hist_component {
                    None => "Magnitude".to_string(),
                    Some(c) => format!("Comp {c}"),
                };
                egui::ComboBox::from_id_salt("hist_comp_combo")
                    .selected_text(comp_text)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.hist_component, None, "Magnitude");
                        for c in 0..selected_comps {
                            ui.selectable_value(
                                &mut state.hist_component,
                                Some(c),
                                format!("Comp {c}"),
                            );
                        }
                    });
            }

            ui.add(egui::Slider::new(&mut state.hist_bins, 4..=64).text("Bins"));
        }
    });

    match state.mode {
        ChartMode::XYPlot if has_arc_length => {
            if let Some(pd) = ds.as_poly_data() {
                paint_xy_arclength_plot(ui, pd, state, theme);
            }
        }
        _ => {
            if state.hist_array_name.is_empty() {
                ui.weak("No scalar array selected for histogram.");
                return feedback;
            }
            match compute_histogram(
                ds,
                &state.hist_array_name,
                state.hist_association,
                state.hist_component,
                state.hist_bins,
            ) {
                Ok(hist) => {
                    let pal = palette(theme);
                    let total: usize = hist.counts.iter().sum();
                    ui.horizontal_wrapped(|ui| {
                        for (lbl, val) in [
                            ("Min", format!("{:.4}", hist.min)),
                            ("Max", format!("{:.4}", hist.max)),
                            ("Mean", format!("{:.4}", hist.mean)),
                            ("StdDev", format!("{:.4}", hist.std_dev)),
                            ("N", format!("{total}")),
                        ] {
                            egui::Frame::new()
                                .fill(pal.card_bg)
                                .stroke(egui::Stroke::new(1.0, pal.card_border))
                                .corner_radius(egui::CornerRadius::same(4))
                                .inner_margin(egui::Margin::symmetric(6, 2))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{lbl}:"))
                                                .size(11.0)
                                                .color(pal.muted),
                                        );
                                        ui.label(
                                            egui::RichText::new(val)
                                                .size(11.0)
                                                .monospace()
                                                .strong(),
                                        );
                                    });
                                });
                        }
                    });
                    ui.add_space(3.0);
                    paint_histogram_chart(ui, &hist, theme);
                }
                Err(e) => {
                    ui.colored_label(
                        palette(theme).danger,
                        format!("Histogram unavailable: {e}"),
                    );
                }
            }
        }
    }

    feedback
}

fn export_xy_csv(pd: &PolyData, path: &Path) -> Result<String, String> {
    let arc = pd
        .point_data
        .get("ArcLength")
        .ok_or_else(|| "Dataset has no ArcLength array".to_string())?;
    let n = arc.num_tuples();
    let cols: Vec<&DataArray> = pd
        .point_data
        .arrays()
        .iter()
        .filter(|a| a.name != "ArcLength" && a.num_tuples() == n)
        .collect();

    let mut csv = String::from("ArcLength");
    for col in &cols {
        csv.push(',');
        csv.push_str(&col.name);
    }
    csv.push('\n');

    for i in 0..n {
        csv.push_str(&format!("{}", arc.get_magnitude(i)));
        for col in &cols {
            csv.push(',');
            csv.push_str(&format!("{}", col.get_magnitude(i)));
        }
        csv.push('\n');
    }

    std::fs::write(path, csv).map_err(|e| format!("Failed to write CSV: {e}"))?;
    Ok(format!("Exported XY plot CSV -> {}", path.display()))
}

fn paint_xy_arclength_plot(
    ui: &mut egui::Ui,
    pd: &PolyData,
    state: &mut ChartState,
    theme: UiTheme,
) {
    let pal = palette(theme);
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
    let all_series: Vec<(String, Vec<f64>, egui::Color32)> = pd
        .point_data
        .arrays()
        .iter()
        .filter(|a| {
            a.name != "ArcLength" && a.name != "vtkValidPointMask" && a.num_tuples() == n
        })
        .take(CHART_PALETTE.len())
        .enumerate()
        .map(|(idx, arr)| {
            let ys: Vec<f64> = (0..n)
                .map(|i| {
                    let v = arr.get_magnitude(i);
                    if state.log_y {
                        if v > 0.0 { v.log10() } else { f64::NAN }
                    } else {
                        v
                    }
                })
                .collect();
            (
                arr.name.clone(),
                ys,
                CHART_PALETTE[idx % CHART_PALETTE.len()],
            )
        })
        .collect();

    if all_series.is_empty() {
        ui.weak("No scalar arrays to plot against ArcLength.");
        return;
    }

    // Series visibility chips
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Series:").size(11.0).color(pal.muted));
        for (name, _, color) in &all_series {
            let visible = !state.hidden_series.contains(name);
            let label = egui::RichText::new(format!("■ {name}"))
                .size(11.5)
                .color(if visible { *color } else { pal.muted });
            if ui.selectable_label(visible, label).clicked() {
                if visible {
                    state.hidden_series.insert(name.clone());
                } else {
                    state.hidden_series.remove(name);
                }
            }
        }
    });

    let active_series: Vec<(&str, &Vec<f64>, egui::Color32)> = all_series
        .iter()
        .filter(|(name, _, _)| !state.hidden_series.contains(name))
        .map(|(name, ys, col)| (name.as_str(), ys, *col))
        .collect();

    if active_series.is_empty() {
        ui.weak("Enable at least one series above to plot.");
        return;
    }

    paint_xy_series(ui, &x_vals, &active_series, PlotOptions {
        x_label: "ArcLength",
        show_markers: state.show_markers,
        log_y: state.log_y,
        show_gaps: false,
    }, theme);
}

pub fn paint_temporal_probe(ui: &mut egui::Ui, result: &TemporalProbeResult, theme: UiTheme) {
    let times: Vec<_> = result.samples.iter().map(|sample| sample.time).collect();
    let values: Vec<_> = result.samples.iter().map(|sample| sample.value.unwrap_or(f64::NAN)).collect();
    let active = [(result.request.field.as_str(), &values, CHART_PALETTE[0])];
    paint_xy_series(ui, &times, &active, PlotOptions {
        x_label: "Time",
        show_markers: true,
        log_y: false,
        show_gaps: true,
    }, theme);
}

struct PlotOptions<'a> {
    x_label: &'a str,
    show_markers: bool,
    log_y: bool,
    show_gaps: bool,
}

fn paint_xy_series(
    ui: &mut egui::Ui,
    x_vals: &[f64],
    active_series: &[(&str, &Vec<f64>, egui::Color32)],
    options: PlotOptions<'_>,
    theme: UiTheme,
) {
    let pal = palette(theme);
    let x_min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (x_min, x_max) = if (x_max - x_min).abs() < 1e-12 {
        (x_min - 0.5, x_max + 0.5)
    } else {
        (x_min, x_max)
    };

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, ys, _) in active_series {
        for &y in ys.iter() {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        ui.weak("Active series contain no finite values for current scale.");
        return;
    }
    if (y_max - y_min).abs() < 1e-12 {
        y_min -= 0.5;
        y_max += 0.5;
    } else {
        let pad = (y_max - y_min) * 0.06;
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
    let right_pad = 16.0;
    let top_pad = 8.0;
    let bottom_pad = 24.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(outer_rect.left() + left_pad, outer_rect.top() + top_pad),
        egui::pos2(
            (outer_rect.right() - right_pad).max(outer_rect.left() + left_pad + 40.0),
            (outer_rect.bottom() - bottom_pad).max(outer_rect.top() + top_pad + 40.0),
        ),
    );

    painter.rect_filled(plot_rect, 4.0, pal.plot_bg);

    for g in 0..=4 {
        let frac = g as f32 / 4.0;
        let gx = egui::lerp(plot_rect.left()..=plot_rect.right(), frac);
        let gy = egui::lerp(plot_rect.bottom()..=plot_rect.top(), frac);
        painter.line_segment(
            [
                egui::pos2(gx, plot_rect.top()),
                egui::pos2(gx, plot_rect.bottom()),
            ],
            egui::Stroke::new(1.0, pal.plot_grid),
        );
        painter.line_segment(
            [
                egui::pos2(plot_rect.left(), gy),
                egui::pos2(plot_rect.right(), gy),
            ],
            egui::Stroke::new(1.0, pal.plot_grid),
        );
    }

    painter.rect_stroke(
        plot_rect,
        4.0,
        egui::Stroke::new(1.0, pal.card_border),
        egui::StrokeKind::Inside,
    );

    let label_font = egui::FontId::monospace(10.0);
    let y_fmt = |v: f64| -> String {
        if options.log_y {
            format!("1e{v:.2}")
        } else {
            format!("{v:.3}")
        }
    };
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.top()),
        egui::Align2::RIGHT_TOP,
        y_fmt(y_max),
        label_font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.center().y),
        egui::Align2::RIGHT_CENTER,
        y_fmt(0.5 * (y_min + y_max)),
        label_font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        y_fmt(y_min),
        label_font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("{x_min:.3}"),
        label_font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.center().x, plot_rect.bottom() + 4.0),
        egui::Align2::CENTER_TOP,
        options.x_label,
        label_font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.bottom() + 4.0),
        egui::Align2::RIGHT_TOP,
        format!("{x_max:.3}"),
        label_font.clone(),
        pal.plot_axis,
    );

    let to_screen = |x: f64, y: f64| -> egui::Pos2 {
        let tx = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0) as f32;
        let ty = ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            egui::lerp(plot_rect.left()..=plot_rect.right(), tx),
            egui::lerp(plot_rect.bottom()..=plot_rect.top(), ty),
        )
    };

    for (_, ys, color) in active_series {
        let mut previous = None;
        for (&x, &y) in x_vals.iter().zip(ys.iter()) {
            if x.is_finite() && y.is_finite() {
                let point = to_screen(x, y);
                if let Some(previous) = previous {
                    painter.line_segment([previous, point], egui::Stroke::new(2.0, *color));
                }
                if options.show_markers {
                    painter.circle_filled(point, 2.5, *color);
                }
                previous = Some(point);
            } else if options.show_gaps {
                previous = None;
                let bottom = to_screen(x, y_min);
                painter.line_segment(
                    [bottom, bottom - egui::vec2(0.0, 8.0)],
                    egui::Stroke::new(2.0, pal.warning),
                );
            }
        }
    }

    // Hover crosshair
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
            egui::Stroke::new(1.0, pal.accent.gamma_multiply(0.7)),
        );

        let mut tip_lines = vec![format!("{}: {:.4}", options.x_label, x_vals[nearest])];
        for (name, ys, color) in active_series {
            let yv = ys[nearest];
            if yv.is_finite() {
                let pt = to_screen(x_vals[nearest], yv);
                painter.circle_filled(pt, 4.0, *color);
                let disp = if options.log_y { 10f64.powf(yv) } else { yv };
                tip_lines.push(format!("{name}: {disp:.4}"));
            } else if options.show_gaps {
                tip_lines.push(format!("{name}: missing"));
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
            (pointer.y - tip_galley.size().y * 0.5).clamp(
                plot_rect.top() + 4.0,
                (plot_rect.bottom() - tip_galley.size().y - 8.0).max(plot_rect.top() + 4.0),
            ),
        );
        let tip_bg = egui::Rect::from_min_size(
            tip_pos - egui::vec2(6.0, 4.0),
            tip_galley.size() + egui::vec2(12.0, 8.0),
        );
        painter.rect_filled(tip_bg, 4.0, pal.hud_bg);
        painter.rect_stroke(
            tip_bg,
            4.0,
            egui::Stroke::new(1.0, pal.hud_border),
            egui::StrokeKind::Inside,
        );
        painter.galley(tip_pos, tip_galley, egui::Color32::WHITE);
    }
}

fn paint_histogram_chart(ui: &mut egui::Ui, hist: &HistogramResult, theme: UiTheme) {
    let pal = palette(theme);
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
    let right_pad = 16.0;
    let top_pad = 8.0;
    let bottom_pad = 24.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(outer_rect.left() + left_pad, outer_rect.top() + top_pad),
        egui::pos2(
            (outer_rect.right() - right_pad).max(outer_rect.left() + left_pad + 40.0),
            (outer_rect.bottom() - bottom_pad).max(outer_rect.top() + top_pad + 40.0),
        ),
    );

    painter.rect_filled(plot_rect, 4.0, pal.plot_bg);

    for g in 0..=4 {
        let frac = g as f32 / 4.0;
        let gy = egui::lerp(plot_rect.bottom()..=plot_rect.top(), frac);
        painter.line_segment(
            [
                egui::pos2(plot_rect.left(), gy),
                egui::pos2(plot_rect.right(), gy),
            ],
            egui::Stroke::new(1.0, pal.plot_grid),
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
            pal.warning
        } else {
            pal.accent
        };
        painter.rect_filled(bar_rect, 2.0, col);
    }

    painter.rect_stroke(
        plot_rect,
        4.0,
        egui::Stroke::new(1.0, pal.card_border),
        egui::StrokeKind::Inside,
    );

    let font = egui::FontId::monospace(10.0);
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.top()),
        egui::Align2::RIGHT_TOP,
        format!("{max_count}"),
        font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.left() - 6.0, plot_rect.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        "0",
        font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("{:.3}", hist.bin_edges[0]),
        font.clone(),
        pal.plot_axis,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.bottom() + 4.0),
        egui::Align2::RIGHT_TOP,
        format!("{:.3}", hist.bin_edges[bins]),
        font.clone(),
        pal.plot_axis,
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
            tip_pos - egui::vec2(6.0, 4.0),
            galley.size() + egui::vec2(12.0, 8.0),
        );
        painter.rect_filled(tip_bg, 4.0, pal.hud_bg);
        painter.rect_stroke(
            tip_bg,
            4.0,
            egui::Stroke::new(1.0, pal.hud_border),
            egui::StrokeKind::Inside,
        );
        painter.galley(tip_pos, galley, egui::Color32::WHITE);
    }
}

fn tf_point_to_screen(t: f32, alpha: f32, rect: egui::Rect) -> egui::Pos2 {
    let x = egui::lerp(rect.left()..=rect.right(), t.clamp(0.0, 1.0));
    let y = egui::lerp(rect.bottom()..=rect.top(), alpha.clamp(0.0, 1.0));
    egui::pos2(x, y)
}

fn screen_to_tf_point(pos: egui::Pos2, rect: egui::Rect) -> (f32, f32) {
    let t = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
    let alpha = ((rect.bottom() - pos.y) / rect.height().max(1.0)).clamp(0.0, 1.0);
    (t, alpha)
}

fn nearest_tf_point(
    pos: egui::Pos2,
    points: &[OpacityPoint],
    rect: egui::Rect,
    max_dist_px: f32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, pt) in points.iter().enumerate() {
        let sp = tf_point_to_screen(pt.t, pt.alpha, rect);
        let d = sp.distance(pos);
        if d <= max_dist_px && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((idx, d));
        }
    }
    best.map(|(i, _)| i)
}

pub(crate) fn insert_opacity_point_sorted(points: &mut Vec<OpacityPoint>, t: f32, alpha: f32) {
    points.push(OpacityPoint {
        t: t.clamp(0.0, 1.0),
        alpha: alpha.clamp(0.0, 1.0),
    });
    points.sort_by(|a, b| a.t.total_cmp(&b.t));
}

pub(crate) fn remove_interior_opacity_point(points: &mut Vec<OpacityPoint>, idx: usize) -> bool {
    if points.len() > 2 && idx > 0 && idx + 1 < points.len() {
        points.remove(idx);
        true
    } else {
        false
    }
}

pub(crate) fn move_opacity_point_clamped(
    points: &mut [OpacityPoint],
    idx: usize,
    raw_t: f32,
    raw_alpha: f32,
) -> bool {
    if idx >= points.len() {
        return false;
    }
    let min_t = if idx == 0 {
        0.0
    } else {
        (points[idx - 1].t + 1e-4).min(1.0)
    };
    let max_t = if idx + 1 >= points.len() {
        1.0
    } else {
        (points[idx + 1].t - 1e-4).max(0.0)
    };
    let new_t = raw_t.clamp(min_t, max_t.max(min_t));
    let new_alpha = raw_alpha.clamp(0.0, 1.0);
    let moved = (points[idx].t - new_t).abs() > 1e-5 || (points[idx].alpha - new_alpha).abs() > 1e-5;
    points[idx].t = new_t;
    points[idx].alpha = new_alpha;
    moved
}

/// Interactive 2D piecewise-linear opacity transfer function curve editor overlaid on
/// the active scalar field's histogram bars. Returns `true` if `tf.opacity_points` changed.
pub fn draw_opacity_tf_editor(
    ui: &mut egui::Ui,
    tf: &mut TransferFunction,
    hist: Option<&HistogramResult>,
    theme: UiTheme,
) -> bool {
    let pal = palette(theme);
    let mut changed = false;

    if tf.opacity_points.is_empty() {
        tf.opacity_points = vec![
            OpacityPoint { t: 0.0, alpha: 0.0 },
            OpacityPoint { t: 1.0, alpha: 0.85 },
        ];
        changed = true;
    } else if tf.opacity_points.len() == 1 {
        let existing = tf.opacity_points[0];
        if existing.t < 0.5 {
            tf.opacity_points.push(OpacityPoint { t: 1.0, alpha: 0.85 });
        } else {
            tf.opacity_points.insert(0, OpacityPoint { t: 0.0, alpha: 0.0 });
        }
        changed = true;
    }
    tf.opacity_points.sort_by(|a, b| a.t.total_cmp(&b.t));

    let width = ui.available_width().max(180.0);
    let height = 136.0;
    let (outer_rect, resp) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    let plot_rect = outer_rect.shrink2(egui::vec2(7.0, 7.0));
    let painter = ui.painter_at(outer_rect);

    painter.rect_filled(outer_rect, 4.0, pal.plot_bg);

    for frac in [0.25f32, 0.5, 0.75] {
        let gx = egui::lerp(plot_rect.left()..=plot_rect.right(), frac);
        let gy = egui::lerp(plot_rect.bottom()..=plot_rect.top(), frac);
        painter.line_segment(
            [egui::pos2(gx, plot_rect.top()), egui::pos2(gx, plot_rect.bottom())],
            egui::Stroke::new(1.0, pal.plot_grid),
        );
        painter.line_segment(
            [egui::pos2(plot_rect.left(), gy), egui::pos2(plot_rect.right(), gy)],
            egui::Stroke::new(1.0, pal.plot_grid),
        );
    }

    // Background histogram bars mapped into the active TF scalar range
    if let Some(h) = hist
        && !h.counts.is_empty()
        && h.bin_edges.len() >= 2
    {
        let max_count = h.counts.iter().copied().max().unwrap_or(1).max(1) as f32;
        let (r_min, r_max) = tf.scalar_range;
        let span = r_max - r_min;
        let n_bins = h.counts.len();
        for (i, &cnt) in h.counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let (t0, t1) = if span.abs() > 1e-12 {
                (
                    ((h.bin_edges[i] - r_min) / span) as f32,
                    ((h.bin_edges[i + 1] - r_min) / span) as f32,
                )
            } else {
                (i as f32 / n_bins as f32, (i + 1) as f32 / n_bins as f32)
            };
            let t0 = t0.clamp(0.0, 1.0);
            let t1 = t1.clamp(0.0, 1.0);
            if t1 <= t0 {
                continue;
            }
            let x0 = egui::lerp(plot_rect.left()..=plot_rect.right(), t0);
            let x1 = egui::lerp(plot_rect.left()..=plot_rect.right(), t1).max(x0 + 1.0);
            let h_frac = (cnt as f32 / max_count).clamp(0.02, 1.0);
            let y0 = egui::lerp(plot_rect.bottom()..=plot_rect.top(), h_frac);
            let rgb = tf.color_map.sample_rgb((t0 + t1) * 0.5);
            let bar_col = egui::Color32::from_rgba_unmultiplied(
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
                72,
            );
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, plot_rect.bottom())),
                1.0,
                bar_col,
            );
        }
    }

    // Handle pointer interactions (drag, insert on empty click, delete interior on right/double click)
    let drag_id = resp.id.with("tf_drag_idx");
    let mut dragging_idx: Option<usize> = ui
        .data(|d| d.get_temp::<Option<usize>>(drag_id))
        .flatten()
        .filter(|&i| i < tf.opacity_points.len());

    if resp.drag_started_by(egui::PointerButton::Primary)
        && let Some(pos) = resp.interact_pointer_pos()
    {
        dragging_idx = nearest_tf_point(pos, &tf.opacity_points, plot_rect, 11.0);
        ui.data_mut(|d| d.insert_temp(drag_id, dragging_idx));
    }

    if resp.dragged_by(egui::PointerButton::Primary)
        && let Some(idx) = dragging_idx
        && let Some(pos) = resp.interact_pointer_pos()
    {
        let (raw_t, raw_a) = screen_to_tf_point(pos, plot_rect);
        if move_opacity_point_clamped(&mut tf.opacity_points, idx, raw_t, raw_a) {
            changed = true;
        }
    }

    if resp.drag_stopped() {
        dragging_idx = None;
        ui.data_mut(|d| d.remove::<Option<usize>>(drag_id));
    }

    let remove_trigger = resp.clicked_by(egui::PointerButton::Secondary)
        || resp.double_clicked_by(egui::PointerButton::Primary);
    if remove_trigger
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some(idx) = nearest_tf_point(pos, &tf.opacity_points, plot_rect, 11.0)
    {
        if remove_interior_opacity_point(&mut tf.opacity_points, idx) {
            changed = true;
            if dragging_idx == Some(idx) {
                dragging_idx = None;
                ui.data_mut(|d| d.remove::<Option<usize>>(drag_id));
            }
        }
    } else if resp.clicked_by(egui::PointerButton::Primary)
        && !resp.double_clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && nearest_tf_point(pos, &tf.opacity_points, plot_rect, 11.0).is_none()
    {
        let (t, alpha) = screen_to_tf_point(pos, plot_rect);
        insert_opacity_point_sorted(&mut tf.opacity_points, t, alpha);
        changed = true;
    }

    // Translucent area fill under the piecewise-linear curve
    let pts = &tf.opacity_points;
    let draw_seg_fill = |p0: OpacityPoint, p1: OpacityPoint| {
        let s0 = tf_point_to_screen(p0.t, p0.alpha, plot_rect);
        let s1 = tf_point_to_screen(p1.t, p1.alpha, plot_rect);
        let b0 = egui::pos2(s0.x, plot_rect.bottom());
        let b1 = egui::pos2(s1.x, plot_rect.bottom());
        let rgb = tf.color_map.sample_rgb((p0.t + p1.t) * 0.5);
        let fill = egui::Color32::from_rgba_unmultiplied(
            (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            52,
        );
        painter.add(egui::Shape::convex_polygon(
            vec![b0, s0, s1, b1],
            fill,
            egui::Stroke::NONE,
        ));
    };

    if pts[0].t > 0.0 {
        let p0 = OpacityPoint {
            t: 0.0,
            alpha: pts[0].alpha,
        };
        draw_seg_fill(p0, pts[0]);
        painter.line_segment(
            [
                tf_point_to_screen(p0.t, p0.alpha, plot_rect),
                tf_point_to_screen(pts[0].t, pts[0].alpha, plot_rect),
            ],
            egui::Stroke::new(2.0, pal.accent),
        );
    }
    for win in pts.windows(2) {
        draw_seg_fill(win[0], win[1]);
        painter.line_segment(
            [
                tf_point_to_screen(win[0].t, win[0].alpha, plot_rect),
                tf_point_to_screen(win[1].t, win[1].alpha, plot_rect),
            ],
            egui::Stroke::new(2.0, pal.accent),
        );
    }
    if let Some(&last) = pts.last()
        && last.t < 1.0
    {
        let p1 = OpacityPoint {
            t: 1.0,
            alpha: last.alpha,
        };
        draw_seg_fill(last, p1);
        painter.line_segment(
            [
                tf_point_to_screen(last.t, last.alpha, plot_rect),
                tf_point_to_screen(p1.t, p1.alpha, plot_rect),
            ],
            egui::Stroke::new(2.0, pal.accent),
        );
    }

    let hovered_idx = resp
        .hover_pos()
        .and_then(|pos| nearest_tf_point(pos, pts, plot_rect, 11.0));
    let active_idx = dragging_idx.or(hovered_idx);

    for (idx, pt) in pts.iter().enumerate() {
        let center = tf_point_to_screen(pt.t, pt.alpha, plot_rect);
        let active = active_idx == Some(idx);
        let radius = if active { 6.5 } else { 5.0 };
        let rgb = tf.color_map.sample_rgb(pt.t);
        let fill = egui::Color32::from_rgb(
            (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
        );
        let stroke_col = if active {
            pal.warning
        } else {
            egui::Color32::WHITE
        };
        painter.circle_filled(center, radius, fill);
        painter.circle_stroke(center, radius, egui::Stroke::new(1.8, stroke_col));
    }

    painter.rect_stroke(
        outer_rect,
        4.0,
        egui::Stroke::new(1.0, pal.card_border),
        egui::StrokeKind::Inside,
    );

    if let Some(idx) = active_idx
        && let Some(pt) = pts.get(idx)
    {
        let (r_min, r_max) = tf.scalar_range;
        let scalar = r_min + (pt.t as f64) * (r_max - r_min);
        let label = format!("#{idx} t={:.2} α={:.2} ({:.3})", pt.t, pt.alpha, scalar);
        let galley =
            painter.layout_no_wrap(label, egui::FontId::monospace(10.0), egui::Color32::WHITE);
        let tip_pos = egui::pos2(
            (plot_rect.left() + 4.0).min(outer_rect.right() - galley.size().x - 8.0),
            plot_rect.top() + 2.0,
        );
        let tip_bg = egui::Rect::from_min_size(
            tip_pos - egui::vec2(4.0, 2.0),
            galley.size() + egui::vec2(8.0, 4.0),
        );
        painter.rect_filled(tip_bg, 3.0, pal.hud_bg);
        painter.galley(tip_pos, galley, egui::Color32::WHITE);
    }

    ui.label(
        egui::RichText::new("Click: add • Drag: move • Right/Double-click: remove")
            .size(10.0)
            .color(pal.muted),
    );

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opacity_curve_insert_move_and_remove_keeps_endpoints() {
        let mut pts = vec![
            OpacityPoint { t: 0.0, alpha: 0.0 },
            OpacityPoint { t: 1.0, alpha: 0.9 },
        ];
        insert_opacity_point_sorted(&mut pts, 0.45, 0.6);
        assert_eq!(pts.len(), 3);
        assert!((pts[1].t - 0.45).abs() < 1e-6);
        assert!((pts[1].alpha - 0.6).abs() < 1e-6);

        // Moving interior point clamps within neighbor t values and [0, 1] alpha
        assert!(move_opacity_point_clamped(&mut pts, 1, 1.5, 1.2));
        assert!(pts[1].t > pts[0].t && pts[1].t < pts[2].t);
        assert!((pts[1].alpha - 1.0).abs() < 1e-6);

        // Endpoints cannot be removed; interior point can
        assert!(!remove_interior_opacity_point(&mut pts, 0));
        assert!(!remove_interior_opacity_point(&mut pts, 2));
        assert!(remove_interior_opacity_point(&mut pts, 1));
        assert_eq!(pts.len(), 2);
        assert!(!remove_interior_opacity_point(&mut pts, 1));
    }
}
