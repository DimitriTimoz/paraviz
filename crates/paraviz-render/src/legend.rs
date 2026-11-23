//! Software RGBA compositor for ParaView-style Scalar Color Bar legends, time badges,
//! and custom viewport text annotations in headless screenshots and animations.

use crate::scene::{ActorStyle, ColorBy, RenderScene, ScalarBarConfig, ViewportAnnotation};
use paraviz_core::{ColorMap, DataSet};

/// Composites all enabled 2D viewport overlays onto `rgba` (`width * height * 4`):
/// - Movable / resizable Scalar Color Bar when `scene.settings.show_scalar_bar` is true
/// - `Time: ...` badge when `scene.settings.show_time_annotation` is true and `time_label` is set
/// - Visible custom [`ViewportAnnotation`] text badges from `scene.settings.annotations`
pub fn composite_viewport_overlays(scene: &RenderScene, rgba: &mut [u8], width: u32, height: u32) {
    if rgba.len() < (width as usize * height as usize * 4) || width == 0 || height == 0 {
        return;
    }
    if scene.settings.show_scalar_bar {
        composite_scalar_bar(scene, rgba, width, height);
    }
    if scene.settings.show_time_annotation
        && let Some(label) = scene.settings.time_label.as_deref()
        && !label.trim().is_empty()
    {
        draw_time_annotation(rgba, width, height, label);
    }
    for ann in &scene.settings.annotations {
        if ann.visible && !ann.text.is_empty() {
            draw_viewport_annotation(rgba, width, height, ann);
        }
    }
}

/// Finds the first visible scalar-colored actor in `scene` and composites a Scalar Color Bar
/// using [`ScalarBarConfig`] layout settings from `scene.settings.scalar_bar`.
pub fn composite_scalar_bar(scene: &RenderScene, rgba: &mut [u8], width: u32, height: u32) {
    let Some((default_title, cmap, range)) = active_scalar_bar(scene) else {
        return;
    };
    let cfg = &scene.settings.scalar_bar;
    let title = cfg
        .custom_title
        .as_deref()
        .filter(|t| !t.trim().is_empty())
        .unwrap_or(&default_title);
    draw_scalar_bar_rgba(rgba, width, height, title, &cmap, range, cfg);
}

fn active_scalar_bar(scene: &RenderScene) -> Option<(String, ColorMap, (f64, f64))> {
    for actor in &scene.actors {
        if !actor.style.visible {
            continue;
        }
        let (arr_name, comp, is_point) = match &actor.style.color_by {
            ColorBy::PointScalar { array, component } => (array, *component, true),
            ColorBy::CellScalar { array, component } => (array, *component, false),
            ColorBy::Solid(_) => continue,
        };
        let range = if actor.style.auto_range {
            dataset_scalar_range(&actor.dataset, &actor.style, "", None, arr_name, comp, is_point)
                .unwrap_or(actor.style.scalar_range)
        } else {
            actor.style.scalar_range
        };
        let title = match comp {
            Some(c) => format!("{arr_name}[{c}]"),
            None => arr_name.clone(),
        };
        return Some((title, actor.style.colormap.clone(), range));
    }
    None
}

fn dataset_scalar_range(
    dataset: &DataSet,
    style: &ActorStyle,
    block_path: &str,
    inherited_override: Option<[f32; 4]>,
    arr_name: &str,
    comp: Option<usize>,
    is_point: bool,
) -> Option<(f64, f64)> {
    if let DataSet::MultiBlock(mb) = dataset {
        let mut acc: Option<(f64, f64)> = None;
        for (name, child) in &mb.blocks {
            let child_path = if block_path.is_empty() {
                name.clone()
            } else if name.is_empty() {
                block_path.to_string()
            } else {
                format!("{block_path}/{name}")
            };
            if !style.is_block_visible(&child_path, name) {
                continue;
            }
            let child_override = style
                .block_color_override(&child_path, name)
                .or(inherited_override);
            if let Some((mn, mx)) =
                dataset_scalar_range(child, style, &child_path, child_override, arr_name, comp, is_point)
            {
                acc = Some(match acc {
                    Some((amn, amx)) => (amn.min(mn), amx.max(mx)),
                    None => (mn, mx),
                });
            }
        }
        return acc;
    }
    if inherited_override.is_some() {
        return None;
    }
    let fd = if is_point {
        dataset.point_data()
    } else {
        dataset.cell_data()
    }?;
    let arr = if arr_name.is_empty() {
        fd.active_scalars_array()
    } else {
        fd.get(arr_name)
    }?;
    Some(arr.range(comp))
}

fn draw_scalar_bar_rgba(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    title: &str,
    cmap: &ColorMap,
    range: (f64, f64),
    cfg: &ScalarBarConfig,
) {
    let w = width as i32;
    let h = height as i32;
    if w < 72 || h < 56 {
        return;
    }

    let font_scale = cfg.font_scale.clamp(0.5, 4.0).round().max(1.0) as i32;
    let ref_scale = (height as f32 / 720.0).clamp(0.55, 2.5);
    let default_size = ScalarBarConfig::default().size;
    let logical_size = if cfg.horizontal && cfg.size == default_size {
        [240.0, 80.0]
    } else {
        cfg.size
    };

    let (box_w, box_h) = if cfg.horizontal {
        let bw = (logical_size[0].clamp(120.0, 960.0) * ref_scale).round() as i32;
        let bh = (logical_size[1].clamp(58.0, 480.0) * ref_scale).round() as i32;
        (bw.min(w - 16).max(96), bh.min(h - 16).max(56))
    } else {
        let bw = (logical_size[0].clamp(100.0, 640.0) * ref_scale).round() as i32;
        let bh = (logical_size[1].clamp(96.0, 960.0) * ref_scale).round() as i32;
        (bw.min(w - 16).max(88), bh.min(h - 16).max(72))
    };

    let pad = 12i32.min((w - box_w) / 2).min((h - box_h) / 2).max(0);
    let (x0, y0) = match cfg.pos {
        Some([nx, ny]) => {
            let px = (nx.clamp(0.0, 1.0) * w as f32).round() as i32;
            let py = (ny.clamp(0.0, 1.0) * h as f32).round() as i32;
            (
                px.clamp(0, (w - box_w).max(0)),
                py.clamp(0, (h - box_h).max(0)),
            )
        }
        None => (w - box_w - pad, h - box_h - pad),
    };

    let mut canvas = RgbaCanvas {
        rgba,
        w: width,
        h: height,
    };

    // Translucent dark card with subtle border
    canvas.blend_rect(x0, y0, box_w, box_h, [18, 22, 32, 215]);
    canvas.stroke_rect(x0, y0, box_w, box_h, [75, 88, 114, 240]);

    let title_h = 7 * font_scale;
    let max_title_chars = ((box_w - 20) / (6 * font_scale)).max(1) as usize;
    let short_title: String = title.chars().take(max_title_chars).collect();
    canvas.draw_text_5x7(
        x0 + 10,
        y0 + 8,
        &short_title,
        [235, 240, 248, 255],
        font_scale,
    );

    let (lo, hi) = range;
    let n_labels = cfg.num_labels.clamp(2, 7);
    let label_col = [220, 226, 238, 255];
    let mid_col = [175, 185, 205, 255];

    if cfg.horizontal {
        let bx = x0 + 12;
        let by = y0 + 14 + title_h;
        let bar_w = (box_w - 24).max(16);
        let bar_h = (box_h - (by - y0) - 18 - 7 * font_scale).clamp(10, 26);

        for dx in 0..bar_w {
            let t = (dx as f32 + 0.5) / (bar_w as f32);
            let rgb = cmap.sample_rgb(t);
            let col = [
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
            canvas.fill_rect(bx + dx, by, 1, bar_h, col);
        }
        canvas.stroke_rect(bx, by, bar_w, bar_h, [110, 125, 150, 255]);

        let label_y = by + bar_h + 6;
        let mut last_end_x = i32::MIN;
        for i in 0..n_labels {
            let frac = i as f64 / (n_labels - 1) as f64;
            let tx = bx + (frac * (bar_w - 1) as f64).round() as i32;
            canvas.fill_rect(tx, by + bar_h, 1, 4, label_col);

            let val = tick_value(lo, hi, frac, cmap.log_scale);
            let lbl = fmt_val(val);
            let text_w = text_pixel_width(&lbl, font_scale);
            let lx = if i == 0 {
                bx
            } else if i == n_labels - 1 {
                (bx + bar_w - text_w).max(bx)
            } else {
                (tx - text_w / 2).clamp(bx, (bx + bar_w - text_w).max(bx))
            };
            if i == 0 || i == n_labels - 1 || lx > last_end_x + 4 {
                let col = if i == 0 || i == n_labels - 1 {
                    label_col
                } else {
                    mid_col
                };
                canvas.draw_text_5x7(lx, label_y, &lbl, col, font_scale);
                last_end_x = lx + text_w;
            }
        }
    } else {
        let bar_w = (16 * font_scale).clamp(14, 28).min((box_w / 4).max(12));
        let bx = x0 + 12;
        let by = y0 + 16 + title_h;
        let bar_h = (box_h - (by - y0) - 12).max(24);

        for dy in 0..bar_h {
            let t = 1.0 - (dy as f32 + 0.5) / (bar_h as f32);
            let rgb = cmap.sample_rgb(t);
            let col = [
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
            canvas.fill_rect(bx, by + dy, bar_w, 1, col);
        }
        canvas.stroke_rect(bx, by, bar_w, bar_h, [110, 125, 150, 255]);

        let label_x = bx + bar_w + 8;
        let max_lbl_chars = ((x0 + box_w - 6 - label_x) / (6 * font_scale)).max(1) as usize;
        // Draw top (max) first, then bottom (min), then intermediate ticks if spacing allows.
        let mut order: Vec<usize> = Vec::with_capacity(n_labels);
        order.push(n_labels - 1);
        if n_labels > 1 {
            order.push(0);
        }
        for i in (1..n_labels.saturating_sub(1)).rev() {
            order.push(i);
        }
        let max_y = by + bar_h - 7 * font_scale;

        // First compute all tick positions; draw min & max always, intermediates if non-overlapping.
        let mut occupied: Vec<(i32, i32)> = Vec::with_capacity(n_labels);
        for &i in &order {
            let frac = i as f64 / (n_labels - 1) as f64;
            let ty = by + ((1.0 - frac) * (bar_h - 1) as f64).round() as i32;
            canvas.fill_rect(bx + bar_w, ty, 4, 1, label_col);

            let ly = (ty - (7 * font_scale) / 2).clamp(by, max_y.max(by));
            let lh = 7 * font_scale;
            let is_endpoint = i == 0 || i == n_labels - 1;
            let overlaps = !is_endpoint
                && occupied
                    .iter()
                    .any(|&(oy, oh)| ly < oy + oh + 2 && ly + lh + 2 > oy);
            if overlaps {
                continue;
            }
            occupied.push((ly, lh));

            let val = tick_value(lo, hi, frac, cmap.log_scale);
            let lbl: String = fmt_val(val).chars().take(max_lbl_chars).collect();
            let col = if is_endpoint { label_col } else { mid_col };
            canvas.draw_text_5x7(label_x, ly, &lbl, col, font_scale);
        }
    }
}

fn draw_time_annotation(rgba: &mut [u8], width: u32, height: u32, label: &str) {
    let trimmed = label.trim();
    let text = if trimmed.starts_with("Time:") {
        trimmed.to_string()
    } else {
        format!("Time: {trimmed}")
    };
    let scale = if height >= 900 { 2 } else { 1 };
    let text_w = text_pixel_width(&text, scale);
    let text_h = 7 * scale;
    let pad_x = 8 * scale;
    let pad_y = 5 * scale;
    let box_w = text_w + pad_x * 2;
    let box_h = text_h + pad_y * 2;
    let margin = 12i32;
    let w = width as i32;
    let h = height as i32;
    if w < box_w + 4 || h < box_h + 4 {
        return;
    }
    let x0 = (w - box_w - margin).max(0);
    let y0 = margin.min((h - box_h).max(0));

    let mut canvas = RgbaCanvas {
        rgba,
        w: width,
        h: height,
    };
    canvas.blend_rect(x0, y0, box_w, box_h, [18, 22, 32, 215]);
    canvas.stroke_rect(x0, y0, box_w, box_h, [86, 108, 146, 240]);
    canvas.draw_text_5x7(
        x0 + pad_x,
        y0 + pad_y,
        &text,
        [238, 244, 255, 255],
        scale,
    );
}

fn draw_viewport_annotation(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    ann: &ViewportAnnotation,
) {
    let scale = (ann.font_size / 7.0).round().clamp(1.0, 6.0) as i32;
    let lines: Vec<&str> = ann.text.lines().collect();
    if lines.is_empty() {
        return;
    }
    let max_line_w = lines
        .iter()
        .map(|l| text_pixel_width(l, scale))
        .max()
        .unwrap_or(0);
    let line_step = 9 * scale;
    let total_text_h = (lines.len() as i32 - 1) * line_step + 7 * scale;
    let pad_x = 6 + 2 * scale;
    let pad_y = 4 + 2 * scale;
    let box_w = (max_line_w + pad_x * 2).max(12);
    let box_h = (total_text_h + pad_y * 2).max(12);
    let w = width as i32;
    let h = height as i32;
    if w < 16 || h < 16 {
        return;
    }

    let raw_x = (ann.pos[0].clamp(0.0, 1.0) * w as f32).round() as i32;
    let raw_y = (ann.pos[1].clamp(0.0, 1.0) * h as f32).round() as i32;
    let x0 = raw_x.clamp(0, (w - box_w).max(0));
    let y0 = raw_y.clamp(0, (h - box_h).max(0));

    let mut canvas = RgbaCanvas {
        rgba,
        w: width,
        h: height,
    };
    canvas.blend_rect(x0, y0, box_w, box_h, [16, 20, 30, 205]);
    canvas.stroke_rect(x0, y0, box_w, box_h, [75, 88, 114, 225]);

    let col = [
        (ann.color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (ann.color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (ann.color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (ann.color[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ];
    for (idx, line) in lines.iter().enumerate() {
        canvas.draw_text_5x7(
            x0 + pad_x,
            y0 + pad_y + idx as i32 * line_step,
            line,
            col,
            scale,
        );
    }
}

fn tick_value(lo: f64, hi: f64, frac: f64, log_scale: bool) -> f64 {
    if log_scale && lo > 0.0 && hi > lo {
        lo * (hi / lo).powf(frac)
    } else {
        lo + (hi - lo) * frac
    }
}

fn text_pixel_width(text: &str, scale: i32) -> i32 {
    let n = text.chars().count() as i32;
    if n <= 0 {
        0
    } else {
        (n * 6 - 1) * scale
    }
}

fn fmt_val(v: f64) -> String {
    let av = v.abs();
    if av >= 1e4 || (av > 0.0 && av < 1e-2) {
        format!("{v:.2e}")
    } else if av >= 100.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

struct RgbaCanvas<'a> {
    rgba: &'a mut [u8],
    w: u32,
    h: u32,
}

impl RgbaCanvas<'_> {
    fn blend_rect(&mut self, x: i32, y: i32, w: i32, h: i32, col: [u8; 4]) {
        let a = col[3] as f32 / 255.0;
        let inv = 1.0 - a;
        for py in y.max(0)..(y + h).min(self.h as i32) {
            for px in x.max(0)..(x + w).min(self.w as i32) {
                let idx = ((py as u32 * self.w + px as u32) * 4) as usize;
                self.rgba[idx] = (col[0] as f32 * a + self.rgba[idx] as f32 * inv) as u8;
                self.rgba[idx + 1] = (col[1] as f32 * a + self.rgba[idx + 1] as f32 * inv) as u8;
                self.rgba[idx + 2] = (col[2] as f32 * a + self.rgba[idx + 2] as f32 * inv) as u8;
                self.rgba[idx + 3] = self.rgba[idx + 3].max(col[3]);
            }
        }
    }

    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, col: [u8; 4]) {
        for py in y.max(0)..(y + h).min(self.h as i32) {
            for px in x.max(0)..(x + w).min(self.w as i32) {
                let idx = ((py as u32 * self.w + px as u32) * 4) as usize;
                self.rgba[idx..idx + 4].copy_from_slice(&col);
            }
        }
    }

    fn stroke_rect(&mut self, x: i32, y: i32, w: i32, h: i32, col: [u8; 4]) {
        self.fill_rect(x, y, w, 1, col);
        self.fill_rect(x, y + h - 1, w, 1, col);
        self.fill_rect(x, y, 1, h, col);
        self.fill_rect(x + w - 1, y, 1, h, col);
    }

    fn draw_text_5x7(&mut self, x: i32, y: i32, text: &str, col: [u8; 4], scale: i32) {
        let mut cx = x;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col_bit in 0..5 {
                    if (bits >> (4 - col_bit)) & 1 == 1 {
                        self.fill_rect(
                            cx + col_bit * scale,
                            y + row as i32 * scale,
                            scale,
                            scale,
                            col,
                        );
                    }
                }
            }
            cx += 6 * scale;
        }
    }
}

fn glyph_5x7(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' | 'a' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' | 'b' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' | 'c' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' | 'd' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' | 'e' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' | 'f' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' | 'g' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' | 'h' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' | 'i' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' | 'j' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' | 'k' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' | 'l' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' | 'm' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' | 'n' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' | 'o' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' | 'p' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' | 'q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' | 'r' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' | 's' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' | 't' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' | 'u' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' | 'v' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' | 'w' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' | 'x' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' | 'y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' | 'z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '!' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
        '?' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
        ';' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08],
        '*' => [0x00, 0x15, 0x0E, 0x1F, 0x0E, 0x15, 0x00],
        '\'' => [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00],
        '"' => [0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00],
        '@' => [0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0F],
        '#' => [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
        '°' => [0x06, 0x09, 0x06, 0x00, 0x00, 0x00, 0x00],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        _ => [0x00; 7],
    }
}
