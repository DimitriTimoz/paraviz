//! Software RGBA compositor for ParaView-style Scalar Color Bar legends in headless
//! screenshots and animations (addressing ParaView GitLab `#18651` and `#21122`).

use crate::scene::{ColorBy, RenderScene};
use paraviz_core::ColorMap;

/// Finds the first visible scalar-colored actor in `scene` and composites a Scalar Color Bar
/// with variable title and min / mid / max tick labels in the bottom-right corner of `rgba`.
pub fn composite_scalar_bar(scene: &RenderScene, rgba: &mut [u8], width: u32, height: u32) {
    let Some((title, cmap, range)) = active_scalar_bar(scene) else {
        return;
    };
    draw_scalar_bar_rgba(rgba, width, height, &title, &cmap, range);
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
            let fd = if is_point {
                actor.dataset.point_data()
            } else {
                actor.dataset.cell_data()
            };
            fd.and_then(|d| d.get(arr_name))
                .map(|a| a.range(comp))
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

fn draw_scalar_bar_rgba(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    title: &str,
    cmap: &ColorMap,
    range: (f64, f64),
) {
    let w = width as i32;
    let h = height as i32;
    let bar_w = 20i32;
    let bar_h = ((height as f32 * 0.34).clamp(96.0, 240.0)) as i32;
    let box_w = 160i32;
    let box_h = bar_h + 48;
    let pad = 16i32;
    if w < box_w + 32 || h < box_h + 32 {
        return;
    }

    let x0 = w - box_w - pad;
    let y0 = h - box_h - pad;

    // Translucent dark card with subtle border
    blend_rect(rgba, width, height, x0, y0, box_w, box_h, [18, 22, 32, 215]);
    stroke_rect(rgba, width, height, x0, y0, box_w, box_h, [75, 88, 114, 240]);

    // Variable title (clipped to fit)
    let short_title: String = title.chars().take(18).collect();
    draw_text_5x7(
        rgba,
        width,
        height,
        x0 + 10,
        y0 + 10,
        &short_title,
        [235, 240, 248, 255],
        1,
    );

    // Gradient bar
    let bx = x0 + 12;
    let by = y0 + 30;
    for dy in 0..bar_h {
        let t = 1.0 - (dy as f32 + 0.5) / (bar_h as f32);
        let rgb = cmap.sample_rgb(t);
        let col = [
            (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ];
        fill_rect(rgba, width, height, bx, by + dy, bar_w, 1, col);
    }
    stroke_rect(rgba, width, height, bx, by, bar_w, bar_h, [110, 125, 150, 255]);

    // Min / Mid / Max tick marks and formatted labels
    let (lo, hi) = range;
    let mid = if cmap.log_scale && lo > 0.0 && hi > lo {
        (lo * hi).sqrt()
    } else {
        0.5 * (lo + hi)
    };
    let label_x = bx + bar_w + 8;
    let label_col = [220, 226, 238, 255];

    let max_lbl = fmt_val(hi);
    let mid_lbl = fmt_val(mid);
    let min_lbl = fmt_val(lo);

    // Tick lines
    fill_rect(rgba, width, height, bx + bar_w, by, 4, 1, label_col);
    fill_rect(rgba, width, height, bx + bar_w, by + bar_h / 2, 4, 1, label_col);
    fill_rect(rgba, width, height, bx + bar_w, by + bar_h - 1, 4, 1, label_col);

    draw_text_5x7(rgba, width, height, label_x, by, &max_lbl, label_col, 1);
    draw_text_5x7(
        rgba,
        width,
        height,
        label_x,
        by + bar_h / 2 - 3,
        &mid_lbl,
        [175, 185, 205, 255],
        1,
    );
    draw_text_5x7(
        rgba,
        width,
        height,
        label_x,
        by + bar_h - 7,
        &min_lbl,
        label_col,
        1,
    );
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

fn blend_rect(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    col: [u8; 4],
) {
    let a = col[3] as f32 / 255.0;
    let inv = 1.0 - a;
    for py in y.max(0)..(y + h).min(img_h as i32) {
        for px in x.max(0)..(x + w).min(img_w as i32) {
            let idx = ((py as u32 * img_w + px as u32) * 4) as usize;
            rgba[idx] = (col[0] as f32 * a + rgba[idx] as f32 * inv) as u8;
            rgba[idx + 1] = (col[1] as f32 * a + rgba[idx + 1] as f32 * inv) as u8;
            rgba[idx + 2] = (col[2] as f32 * a + rgba[idx + 2] as f32 * inv) as u8;
            rgba[idx + 3] = rgba[idx + 3].max(col[3]);
        }
    }
}

fn fill_rect(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    col: [u8; 4],
) {
    for py in y.max(0)..(y + h).min(img_h as i32) {
        for px in x.max(0)..(x + w).min(img_w as i32) {
            let idx = ((py as u32 * img_w + px as u32) * 4) as usize;
            rgba[idx..idx + 4].copy_from_slice(&col);
        }
    }
}

fn stroke_rect(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    col: [u8; 4],
) {
    fill_rect(rgba, img_w, img_h, x, y, w, 1, col);
    fill_rect(rgba, img_w, img_h, x, y + h - 1, w, 1, col);
    fill_rect(rgba, img_w, img_h, x, y, 1, h, col);
    fill_rect(rgba, img_w, img_h, x + w - 1, y, 1, h, col);
}

fn draw_text_5x7(
    rgba: &mut [u8],
    img_w: u32,
    img_h: u32,
    x: i32,
    y: i32,
    text: &str,
    col: [u8; 4],
    scale: i32,
) {
    let mut cx = x;
    for ch in text.chars() {
        let glyph = glyph_5x7(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col_bit in 0..5 {
                if (bits >> (4 - col_bit)) & 1 == 1 {
                    fill_rect(
                        rgba,
                        img_w,
                        img_h,
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
        _ => [0x00; 7],
    }
}
