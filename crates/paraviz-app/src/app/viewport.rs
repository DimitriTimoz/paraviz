use super::*;

impl ParaVizApp {
    pub(super) fn draw_viewport(&mut self, ui: &mut egui::Ui) {
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
                            if ui.button("Open File...").clicked()
                                && let Some(p) =
                                    native_pick_scientific_file(self.last_export_dir.as_deref())
                            {
                                self.open_picked_scientific_file(p);
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

}

pub(super) fn project_world_to_screen(
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


pub(super) fn paint_scalar_bar_overlay(
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

