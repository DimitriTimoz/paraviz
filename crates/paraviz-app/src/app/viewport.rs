use super::*;

impl ParaVizApp {
    pub(super) fn draw_viewport(&mut self, ui: &mut egui::Ui) {
        let full_rect = ui.available_rect_before_wrap();
        let pane_rects = split_viewport_rects(full_rect, self.viewport_layout);

        for (idx, rect) in pane_rects.into_iter().enumerate() {
            if idx >= self.panes.len() {
                break;
            }
            self.draw_pane(ui, idx, rect);
        }

        // Floating progress overlay in viewport when a background worker job is active
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(job) = &self.active_job {
            let pal = palette(self.theme);
            let elapsed = job.started_at.elapsed().as_secs_f32();
            let pill_w = 320.0f32.min((full_rect.width() - 24.0).max(160.0));
            let pill_rect = egui::Rect::from_center_size(
                egui::pos2(full_rect.center().x, full_rect.bottom() - 26.0),
                egui::vec2(pill_w, 30.0),
            );
            let mut cancel_clicked = false;
            if full_rect.width() >= 200.0 && full_rect.height() >= 80.0 {
                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(pill_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                    |ui| {
                        egui::Frame::new()
                            .fill(pal.hud_bg)
                            .stroke(egui::Stroke::new(1.0, pal.accent))
                            .corner_radius(egui::CornerRadius::same(14))
                            .inner_margin(egui::Margin::symmetric(10, 4))
                            .show(ui, |ui| {
                                ui.spinner();
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} ({elapsed:.1}s)",
                                        job.label
                                    ))
                                    .size(11.5)
                                    .color(egui::Color32::WHITE),
                                );
                                if ui.small_button("Cancel").clicked() {
                                    cancel_clicked = true;
                                }
                            });
                    },
                );
            }
            if cancel_clicked {
                self.cancel_active_job(true);
            }
        }
    }

    fn draw_pane(&mut self, ui: &mut egui::Ui, pane_idx: usize, outer_rect: egui::Rect) {
        let pal = palette(self.theme);
        let is_split = self.viewport_layout != ViewportLayout::Single;
        let is_active = self.active_pane == pane_idx;

        if ui.input(|i| i.pointer.primary_pressed())
            && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
            && outer_rect.contains(pos)
            && self.active_pane != pane_idx
        {
            self.active_pane = pane_idx;
            if self.panes[pane_idx].kind == PaneKind::View3D {
                self.session.camera = self.panes[pane_idx].camera.clone();
            }
        }

        let header_h = 25.0;
        let header_rect = egui::Rect::from_min_max(
            outer_rect.min,
            egui::pos2(outer_rect.right(), (outer_rect.top() + header_h).min(outer_rect.bottom())),
        );
        let content_rect = egui::Rect::from_min_max(
            egui::pos2(outer_rect.left() + 1.0, header_rect.bottom()),
            egui::pos2(
                (outer_rect.right() - 1.0).max(outer_rect.left() + 2.0),
                (outer_rect.bottom() - 1.0).max(header_rect.bottom() + 1.0),
            ),
        );

        let painter = ui.painter_at(outer_rect);
        painter.rect_filled(header_rect, 0.0, pal.card_bg);
        painter.line_segment(
            [
                egui::pos2(header_rect.left(), header_rect.bottom()),
                egui::pos2(header_rect.right(), header_rect.bottom()),
            ],
            egui::Stroke::new(1.0, pal.card_border),
        );

        // Compact pane header
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(header_rect.shrink2(egui::vec2(6.0, 2.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
            |ui| {
                ui.push_id(("pane_hdr", pane_idx), |ui| {
                    let badge_bg = if is_active {
                        pal.accent_dim
                    } else {
                        egui::Color32::TRANSPARENT
                    };
                    let badge_fg = if is_active { pal.accent } else { pal.muted };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(format!("P{}", pane_idx + 1))
                                    .size(10.5)
                                    .monospace()
                                    .strong()
                                    .color(badge_fg),
                            )
                            .fill(badge_bg)
                            .corner_radius(egui::CornerRadius::same(4)),
                        )
                        .clicked()
                    {
                        self.active_pane = pane_idx;
                        if self.panes[pane_idx].kind == PaneKind::View3D {
                            self.session.camera = self.panes[pane_idx].camera.clone();
                        }
                    }

                    let prev_kind = self.panes[pane_idx].kind;
                    for &kind in PaneKind::ALL {
                        if ui
                            .selectable_label(self.panes[pane_idx].kind == kind, kind.label())
                            .clicked()
                        {
                            self.panes[pane_idx].kind = kind;
                            self.active_pane = pane_idx;
                        }
                    }
                    if self.panes[pane_idx].kind != prev_kind {
                        if self.panes[pane_idx].kind == PaneKind::View3D {
                            self.session.camera = self.panes[pane_idx].camera.clone();
                        }
                        self.invalidate_viewport();
                    }

                    if self.panes[pane_idx].kind == PaneKind::View3D {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let is_custom = self.panes[pane_idx].visible_nodes_override.is_some();
                            if is_custom {
                                let node_list: Vec<(NodeId, String)> = self
                                    .session
                                    .nodes()
                                    .iter()
                                    .map(|n| (n.id, n.name.clone()))
                                    .collect();
                                let count = self.panes[pane_idx]
                                    .visible_nodes_override
                                    .as_ref()
                                    .map_or(0, BTreeSet::len);
                                ui.menu_button(format!("Nodes ({count})"), |ui| {
                                    let Some(set) =
                                        self.panes[pane_idx].visible_nodes_override.as_mut()
                                    else {
                                        return;
                                    };
                                    let mut vis_changed = false;
                                    for (nid, name) in &node_list {
                                        let mut checked = set.contains(nid);
                                        if ui.checkbox(&mut checked, name).changed() {
                                            if checked {
                                                set.insert(*nid);
                                            } else {
                                                set.remove(nid);
                                            }
                                            vis_changed = true;
                                        }
                                    }
                                    if vis_changed {
                                        self.invalidate_viewport();
                                    }
                                });
                            }
                            let toggle_lbl = if is_custom {
                                "Vis: Custom"
                            } else {
                                "Vis: Global"
                            };
                            if ui
                                .small_button(toggle_lbl)
                                .on_hover_text(
                                    "Toggle between global pipeline visibility and per-pane node visibility",
                                )
                                .clicked()
                            {
                                if is_custom {
                                    self.panes[pane_idx].visible_nodes_override = None;
                                } else {
                                    let vis_set: BTreeSet<NodeId> = self
                                        .session
                                        .nodes()
                                        .iter()
                                        .filter(|n| n.style.visible)
                                        .map(|n| n.id)
                                        .collect();
                                    self.panes[pane_idx].visible_nodes_override = Some(vis_set);
                                }
                                self.invalidate_viewport();
                            }
                        });
                    }
                });
            },
        );

        let kind = self.panes[pane_idx].kind;
        ui.scope_builder(egui::UiBuilder::new().max_rect(content_rect), |ui| {
            ui.push_id(("pane_body", pane_idx), |ui| match kind {
                PaneKind::View3D => self.draw_3d_pane(ui, pane_idx, content_rect),
                PaneKind::Chart => {
                    egui::Frame::new()
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            self.draw_chart_view(ui);
                        });
                }
                PaneKind::Spreadsheet => {
                    egui::Frame::new()
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            self.draw_spreadsheet_view(ui, true);
                        });
                }
            });
        });

        if is_split {
            let stroke = if is_active {
                egui::Stroke::new(2.0, pal.accent)
            } else {
                egui::Stroke::new(1.0, pal.card_border)
            };
            painter.rect_stroke(outer_rect, 2.0, stroke, egui::StrokeKind::Inside);
        }
    }

    fn draw_3d_pane(&mut self, ui: &mut egui::Ui, pane_idx: usize, rect: egui::Rect) {
        let pal = palette(self.theme);
        let avail = rect.size();
        let ppp = ui.ctx().pixels_per_point();
        let w = (avail.x * ppp).round().max(1.0) as u32;
        let h = (avail.y * ppp).round().max(1.0) as u32;

        self.sync_viewport_widgets();

        let cache_key = ViewportCacheKey {
            scene_revision: self.scene_revision,
            width: w,
            height: h,
            camera: self.panes[pane_idx].camera.clone(),
            settings: self.session.settings.clone(),
            widgets: self.session.widgets.clone(),
            picked_highlight: self.picked_highlight.clone(),
            visible_nodes_override: self.panes[pane_idx].visible_nodes_override.clone(),
        };
        let needs_render = self.panes[pane_idx].tex_id.is_none()
            || self.panes[pane_idx].cache_key.as_ref() != Some(&cache_key);

        if needs_render {
            let t0 = Instant::now();
            let scene_res = self.build_pane_scene(pane_idx);
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
                        match self.panes[pane_idx].tex_id {
                            None => {
                                let id = egui_r.register_native_texture(
                                    &self.device,
                                    view,
                                    wgpu::FilterMode::Linear,
                                );
                                self.panes[pane_idx].tex_id = Some(id);
                                self.panes[pane_idx].tex_size = (w, h);
                            }
                            Some(id) if self.panes[pane_idx].tex_size != (w, h) => {
                                egui_r.update_egui_texture_from_wgpu_texture(
                                    &self.device,
                                    view,
                                    wgpu::FilterMode::Linear,
                                    id,
                                );
                                self.panes[pane_idx].tex_size = (w, h);
                            }
                            Some(_) => {}
                        }
                        self.panes[pane_idx].cached_scene = Some(scene);
                        self.panes[pane_idx].cached_total_points = total_points;
                        self.panes[pane_idx].cached_total_cells = total_cells;
                        self.panes[pane_idx].cache_key = Some(cache_key);
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Render error: {e}"));
                    }
                }
            } else if let Err(e) = scene_res {
                self.error_message = Some(format!("Pipeline error: {e}"));
                self.panes[pane_idx].cached_scene = None;
                self.panes[pane_idx].cached_total_points = 0;
                self.panes[pane_idx].cached_total_cells = 0;
                self.panes[pane_idx].cache_key = Some(cache_key);
            }
            self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }

        let total_points = self.panes[pane_idx].cached_total_points;
        let total_cells = self.panes[pane_idx].cached_total_cells;

        let hud_rect_key = egui::Id::NULL.with(("viewport_hud_rect", pane_idx));
        let sb_rect_key = egui::Id::NULL.with(("viewport_sb_rect", pane_idx));
        let ann_rects_key = egui::Id::NULL.with(("viewport_ann_rects", pane_idx));
        let prev_hud_rect = ui.data(|d| d.get_temp::<egui::Rect>(hud_rect_key));
        let prev_sb_rect = ui.data(|d| d.get_temp::<egui::Rect>(sb_rect_key));
        let prev_ann_rects = ui
            .data(|d| d.get_temp::<Vec<egui::Rect>>(ann_rects_key))
            .unwrap_or_default();
        let overlay_contains = |pos: egui::Pos2| -> bool {
            prev_hud_rect.is_some_and(|r| r.contains(pos))
                || prev_sb_rect.is_some_and(|r| r.contains(pos))
                || prev_ann_rects.iter().any(|r| r.contains(pos))
        };

        if let Some(tex_id) = self.panes[pane_idx].tex_id {
            let img = egui::Image::new(egui::load::SizedTexture::new(tex_id, avail))
                .sense(egui::Sense::click_and_drag());
            let resp = ui.add(img);

            let shift = ui.input(|i| i.modifiers.shift);
            if self.viewport_mode == ViewportMode::Orbit
                && resp.dragged_by(egui::PointerButton::Primary)
            {
                let d = resp.drag_delta();
                if shift {
                    self.panes[pane_idx]
                        .camera
                        .pan(d.x / avail.y.max(1.0), d.y / avail.y.max(1.0));
                } else {
                    self.panes[pane_idx]
                        .camera
                        .orbit(-d.x * 0.008, -d.y * 0.008);
                }
                self.sync_camera_from_pane(pane_idx);
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.panes[pane_idx]
                    .camera
                    .pan(d.x / avail.y.max(1.0), d.y / avail.y.max(1.0));
                self.sync_camera_from_pane(pane_idx);
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Secondary) {
                let d = resp.drag_delta();
                self.panes[pane_idx].camera.dolly(-d.y * 0.008);
                self.sync_camera_from_pane(pane_idx);
                self.invalidate_viewport();
                ui.ctx().request_repaint();
            }

            if resp.hovered() {
                let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_y.abs() > 0.01 {
                    self.panes[pane_idx].camera.dolly(scroll_y * 0.0025);
                    self.sync_camera_from_pane(pane_idx);
                    self.invalidate_viewport();
                    ui.ctx().request_repaint();
                }
            }

            if resp.clicked_by(egui::PointerButton::Primary)
                && let Some(pos) = resp.interact_pointer_pos()
                && !overlay_contains(pos)
                && let Some(scene) = &self.panes[pane_idx].cached_scene
            {
                let local = pos - resp.rect.min;
                let px = local.x * ppp;
                let py = local.y * ppp;
                let hit = pick_scene(scene, px, py, w, h);
                match self.viewport_mode {
                    ViewportMode::Orbit | ViewportMode::Pick => {
                        if let Some(h_res) = &hit
                            && let Some(bp) = &h_res.block_path
                        {
                            self.selected_multiblock_patch =
                                Some((NodeId(h_res.actor_id), bp.clone()));
                        }
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

            if resp.double_clicked()
                && self.viewport_mode == ViewportMode::Orbit
                && resp.interact_pointer_pos().is_none_or(|p| !overlay_contains(p))
            {
                self.active_pane = pane_idx;
                self.reset_active_pane_camera();
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
        let vp = self.panes[pane_idx].camera.view_proj_matrix(aspect);
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
                let phase =
                    ((now.elapsed().as_secs_f32() * 6.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
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
        let proj_mode = if self.panes[pane_idx].camera.orthographic {
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
            let block_str = pick
                .block_path
                .as_deref()
                .map(|b| format!("  Block: {b}"))
                .unwrap_or_default();
            hud_lines.push(format!(
                "Pick: {}{}  Pos ({:.3}, {:.3}, {:.3})  Pt {}  Cell {}  Scalar {}",
                pick.actor_name,
                block_str,
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
            let pos_str = hl.world_pos.map_or_else(String::new, |p| {
                format!("  Pos ({:.3}, {:.3}, {:.3})", p[0], p[1], p[2])
            });
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
            (rect.width() - 40.0).max(140.0),
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
            ui.data_mut(|d| d.insert_temp(hud_rect_key, hud_bg));
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
                self.active_pane = pane_idx;
                self.focus_on_world_pos(pos);
            }
            if clear_pick {
                self.last_pick = None;
                self.picked_highlight = None;
                self.highlight_flash_until = None;
                self.invalidate_viewport();
            }
        } else {
            ui.data_mut(|d| d.remove::<egui::Rect>(hud_rect_key));
        }

        // Empty-state welcome launcher
        if pane_idx == 0 && self.session.nodes().is_empty() && self.modal.is_none() {
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
                            if ui.button("Open File...").clicked() {
                                #[cfg(not(target_arch = "wasm32"))]
                                if let Some(p) =
                                    native_pick_scientific_file(self.last_export_dir.as_deref())
                                {
                                    self.open_picked_scientific_file(p);
                                }
                                #[cfg(target_arch = "wasm32")]
                                spawn_wasm_open_dialog(self.pending_file_loads.clone(), false);
                            }
                        });
                    });
            }
        }

        // Top-right Time: annotation badge
        if self.session.settings.show_time_annotation {
            let text = format_time_annotation_text(
                self.session.settings.time_label.as_deref(),
                self.session.time,
            );
            paint_time_annotation_badge(&painter, rect, &text, self.theme);
        }

        // Draggable ViewportAnnotation pills
        let mut new_ann_rects = Vec::new();
        let mut ann_changed = false;
        let title_col = match self.theme {
            UiTheme::DarkStudio => egui::Color32::from_rgb(235, 240, 248),
            UiTheme::LightStudio => egui::Color32::from_rgb(25, 32, 48),
        };
        for ann in &mut self.session.settings.annotations {
            if !ann.visible || ann.text.trim().is_empty() {
                continue;
            }
            let font_size = ann.font_size.clamp(8.0, 48.0);
            let col = egui::Color32::from_rgba_unmultiplied(
                (ann.color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (ann.color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (ann.color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                (ann.color[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            );
            let galley = painter.layout(
                ann.text.clone(),
                egui::FontId::proportional(font_size),
                col,
                (rect.width() - 24.0).max(80.0),
            );
            let pill_size = galley.size() + egui::vec2(16.0, 10.0);
            let top_left = compute_annotation_pos(rect, ann.pos, pill_size);
            let pill_rect = egui::Rect::from_min_size(top_left, pill_size);
            let ann_resp = ui.interact(
                pill_rect,
                ui.scope_id().with(("vp_ann_pill", pane_idx, ann.id)),
                egui::Sense::click_and_drag(),
            );
            if ann_resp.dragged() {
                drag_annotation_pos(&mut ann.pos, pill_rect, rect, ann_resp.drag_delta());
                ann_changed = true;
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                ui.ctx().request_repaint();
            } else if ann_resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
            let stroke_col = if ann_resp.hovered() || ann_resp.dragged() {
                pal.accent
            } else {
                pal.hud_border
            };
            painter.rect_filled(pill_rect, 6.0, pal.hud_bg);
            painter.rect_stroke(
                pill_rect,
                6.0,
                egui::Stroke::new(1.0, stroke_col),
                egui::StrokeKind::Inside,
            );
            painter.galley(pill_rect.min + egui::vec2(8.0, 5.0), galley, col);
            new_ann_rects.push(pill_rect);
        }
        if ann_changed {
            self.invalidate_viewport();
        }
        ui.data_mut(|d| d.insert_temp(ann_rects_key, new_ann_rects));

        // Interactive Movable / Resizable Scalar Color Bar Overlay
        let mut next_sb_rect: Option<egui::Rect> = None;
        if self.show_scalar_bar
            && let Some((default_title, cmap, range)) = self.active_scalar_bar_info(pane_idx)
        {
            let title = self
                .session
                .settings
                .scalar_bar
                .custom_title
                .as_deref()
                .filter(|t| !t.trim().is_empty())
                .map(str::to_string)
                .unwrap_or(default_title);
            let mut sb_changed = false;
            if let Some(box_rect) =
                compute_scalar_bar_rect(rect, &self.session.settings.scalar_bar)
            {
                let resize_size = 14.0;
                let resize_rect = egui::Rect::from_min_max(
                    egui::pos2(box_rect.right() - resize_size, box_rect.bottom() - resize_size),
                    box_rect.max,
                );
                let resize_resp = ui.interact(
                    resize_rect,
                    ui.scope_id().with(("scalar_bar_resize", pane_idx)),
                    egui::Sense::click_and_drag(),
                );
                let body_resp = ui.interact(
                    box_rect,
                    ui.scope_id().with(("scalar_bar_drag", pane_idx)),
                    egui::Sense::click_and_drag(),
                );
                if resize_resp.dragged() {
                    resize_scalar_bar(
                        &mut self.session.settings.scalar_bar,
                        box_rect,
                        rect,
                        resize_resp.drag_delta(),
                    );
                    sb_changed = true;
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeNwSe);
                    ui.ctx().request_repaint();
                } else if body_resp.dragged() {
                    drag_scalar_bar(
                        &mut self.session.settings.scalar_bar,
                        box_rect,
                        rect,
                        body_resp.drag_delta(),
                    );
                    sb_changed = true;
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                    ui.ctx().request_repaint();
                } else if resize_resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeNwSe);
                } else if body_resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }

                let active_border = resize_resp.hovered()
                    || resize_resp.dragged()
                    || body_resp.hovered()
                    || body_resp.dragged();
                paint_scalar_bar_card(
                    &painter,
                    box_rect,
                    &title,
                    &cmap,
                    range,
                    &self.session.settings.scalar_bar,
                    self.theme,
                    title_col,
                    active_border,
                );
                next_sb_rect = Some(box_rect);
            }
            if sb_changed {
                self.invalidate_viewport();
            }
        }
        if let Some(sb_rect) = next_sb_rect {
            ui.data_mut(|d| d.insert_temp(sb_rect_key, sb_rect));
        } else {
            ui.data_mut(|d| d.remove::<egui::Rect>(sb_rect_key));
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

    fn active_scalar_bar_info(&self, pane_idx: usize) -> Option<(String, ColorMap, (f64, f64))> {
        let override_ids = self.panes[pane_idx].visible_nodes_override.as_ref();
        let is_pane_visible = |n: &paraviz::SessionNode| -> bool {
            match override_ids {
                Some(set) => set.contains(&n.id),
                None => n.style.visible,
            }
        };
        let pick_node = self
            .selected_node
            .and_then(|id| self.session.node(id).ok())
            .filter(|n| is_pane_visible(n) && !matches!(n.style.color_by, ColorBy::Solid(_)))
            .or_else(|| {
                self.session
                    .nodes()
                    .iter()
                    .rev()
                    .find(|n| is_pane_visible(n) && !matches!(n.style.color_by, ColorBy::Solid(_)))
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

pub(crate) fn split_viewport_rects(
    full_rect: egui::Rect,
    layout: ViewportLayout,
) -> Vec<egui::Rect> {
    let gap = 4.0;
    match layout {
        ViewportLayout::Single => vec![full_rect],
        ViewportLayout::SplitHorizontal => {
            let half_w = ((full_rect.width() - gap) * 0.5).max(1.0);
            let left = egui::Rect::from_min_size(full_rect.min, egui::vec2(half_w, full_rect.height()));
            let right = egui::Rect::from_min_max(
                egui::pos2(left.right() + gap, full_rect.top()),
                full_rect.max,
            );
            vec![left, right]
        }
        ViewportLayout::SplitVertical => {
            let half_h = ((full_rect.height() - gap) * 0.5).max(1.0);
            let top = egui::Rect::from_min_size(full_rect.min, egui::vec2(full_rect.width(), half_h));
            let bottom = egui::Rect::from_min_max(
                egui::pos2(full_rect.left(), top.bottom() + gap),
                full_rect.max,
            );
            vec![top, bottom]
        }
        ViewportLayout::Quad => {
            let half_w = ((full_rect.width() - gap) * 0.5).max(1.0);
            let half_h = ((full_rect.height() - gap) * 0.5).max(1.0);
            let x_mid = full_rect.left() + half_w;
            let y_mid = full_rect.top() + half_h;
            vec![
                egui::Rect::from_min_max(full_rect.min, egui::pos2(x_mid, y_mid)),
                egui::Rect::from_min_max(
                    egui::pos2(x_mid + gap, full_rect.top()),
                    egui::pos2(full_rect.right(), y_mid),
                ),
                egui::Rect::from_min_max(
                    egui::pos2(full_rect.left(), y_mid + gap),
                    egui::pos2(x_mid, full_rect.bottom()),
                ),
                egui::Rect::from_min_max(egui::pos2(x_mid + gap, y_mid + gap), full_rect.max),
            ]
        }
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

pub(crate) fn format_time_annotation_text(time_label: Option<&str>, session_time: f64) -> String {
    match time_label.map(str::trim).filter(|s| !s.is_empty()) {
        Some(t) if t.starts_with("Time:") => t.to_string(),
        Some(t) => format!("Time: {t}"),
        None => format!("Time: {session_time:.4}"),
    }
}

fn paint_time_annotation_badge(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    text: &str,
    theme: UiTheme,
) {
    let pal = palette(theme);
    let text_col = match theme {
        UiTheme::DarkStudio => egui::Color32::from_rgb(238, 244, 255),
        UiTheme::LightStudio => egui::Color32::from_rgb(25, 32, 48),
    };
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::monospace(12.5), text_col);
    let box_size = galley.size() + egui::vec2(16.0, 10.0);
    if viewport_rect.width() < box_size.x + 24.0 || viewport_rect.height() < box_size.y + 24.0 {
        return;
    }
    let box_rect = egui::Rect::from_min_size(
        egui::pos2(viewport_rect.right() - box_size.x - 12.0, viewport_rect.top() + 12.0),
        box_size,
    );
    painter.rect_filled(box_rect, 6.0, pal.hud_bg);
    painter.rect_stroke(
        box_rect,
        6.0,
        egui::Stroke::new(1.0, pal.hud_border),
        egui::StrokeKind::Inside,
    );
    painter.galley(box_rect.min + egui::vec2(8.0, 5.0), galley, text_col);
}

pub(crate) fn compute_annotation_pos(
    viewport_rect: egui::Rect,
    pos: [f32; 2],
    pill_size: egui::Vec2,
) -> egui::Pos2 {
    let max_x = (viewport_rect.right() - pill_size.x).max(viewport_rect.left());
    let max_y = (viewport_rect.bottom() - pill_size.y).max(viewport_rect.top());
    let raw_x = viewport_rect.left() + pos[0].clamp(0.0, 1.0) * viewport_rect.width();
    let raw_y = viewport_rect.top() + pos[1].clamp(0.0, 1.0) * viewport_rect.height();
    egui::pos2(
        raw_x.clamp(viewport_rect.left(), max_x),
        raw_y.clamp(viewport_rect.top(), max_y),
    )
}

pub(crate) fn drag_annotation_pos(
    pos: &mut [f32; 2],
    current_rect: egui::Rect,
    viewport_rect: egui::Rect,
    delta: egui::Vec2,
) {
    let max_x = (viewport_rect.right() - current_rect.width()).max(viewport_rect.left());
    let max_y = (viewport_rect.bottom() - current_rect.height()).max(viewport_rect.top());
    let nx = (current_rect.left() + delta.x).clamp(viewport_rect.left(), max_x);
    let ny = (current_rect.top() + delta.y).clamp(viewport_rect.top(), max_y);
    pos[0] = ((nx - viewport_rect.left()) / viewport_rect.width().max(1.0)).clamp(0.0, 1.0);
    pos[1] = ((ny - viewport_rect.top()) / viewport_rect.height().max(1.0)).clamp(0.0, 1.0);
}

fn scalar_bar_ref_scale(viewport_rect: egui::Rect) -> f32 {
    (viewport_rect.height() / 720.0).clamp(0.65, 2.0)
}

pub(crate) fn compute_scalar_bar_rect(
    viewport_rect: egui::Rect,
    cfg: &ScalarBarConfig,
) -> Option<egui::Rect> {
    if viewport_rect.width() < 120.0 || viewport_rect.height() < 80.0 {
        return None;
    }
    let ref_scale = scalar_bar_ref_scale(viewport_rect);
    let default_size = ScalarBarConfig::default().size;
    let logical_size = if cfg.horizontal && cfg.size == default_size {
        [240.0, 80.0]
    } else {
        cfg.size
    };
    let (box_w, box_h) = if cfg.horizontal {
        let bw = (logical_size[0].clamp(120.0, 960.0) * ref_scale)
            .min((viewport_rect.width() - 16.0).max(96.0))
            .max(96.0);
        let bh = (logical_size[1].clamp(58.0, 480.0) * ref_scale)
            .min((viewport_rect.height() - 16.0).max(56.0))
            .max(56.0);
        (bw, bh)
    } else {
        let bw = (logical_size[0].clamp(100.0, 640.0) * ref_scale)
            .min((viewport_rect.width() - 16.0).max(88.0))
            .max(88.0);
        let bh = (logical_size[1].clamp(96.0, 960.0) * ref_scale)
            .min((viewport_rect.height() - 16.0).max(72.0))
            .max(72.0);
        (bw, bh)
    };

    let pad = 14.0;
    let (x0, y0) = match cfg.pos {
        Some([nx, ny]) => {
            let max_x = (viewport_rect.right() - box_w).max(viewport_rect.left());
            let max_y = (viewport_rect.bottom() - box_h).max(viewport_rect.top());
            let px = viewport_rect.left() + nx.clamp(0.0, 1.0) * viewport_rect.width();
            let py = viewport_rect.top() + ny.clamp(0.0, 1.0) * viewport_rect.height();
            (
                px.clamp(viewport_rect.left(), max_x),
                py.clamp(viewport_rect.top(), max_y),
            )
        }
        None => (
            (viewport_rect.right() - box_w - pad).max(viewport_rect.left()),
            (viewport_rect.bottom() - box_h - pad).max(viewport_rect.top()),
        ),
    };
    Some(egui::Rect::from_min_size(
        egui::pos2(x0, y0),
        egui::vec2(box_w, box_h),
    ))
}

pub(crate) fn drag_scalar_bar(
    cfg: &mut ScalarBarConfig,
    current_box: egui::Rect,
    viewport_rect: egui::Rect,
    delta: egui::Vec2,
) {
    let max_x = (viewport_rect.right() - current_box.width()).max(viewport_rect.left());
    let max_y = (viewport_rect.bottom() - current_box.height()).max(viewport_rect.top());
    let new_x = (current_box.left() + delta.x).clamp(viewport_rect.left(), max_x);
    let new_y = (current_box.top() + delta.y).clamp(viewport_rect.top(), max_y);
    cfg.pos = Some([
        ((new_x - viewport_rect.left()) / viewport_rect.width().max(1.0)).clamp(0.0, 1.0),
        ((new_y - viewport_rect.top()) / viewport_rect.height().max(1.0)).clamp(0.0, 1.0),
    ]);
}

pub(crate) fn resize_scalar_bar(
    cfg: &mut ScalarBarConfig,
    current_box: egui::Rect,
    viewport_rect: egui::Rect,
    delta: egui::Vec2,
) {
    // Anchor top-left when resizing from bottom-right corner if previously auto-anchored
    if cfg.pos.is_none() {
        cfg.pos = Some([
            ((current_box.left() - viewport_rect.left()) / viewport_rect.width().max(1.0))
                .clamp(0.0, 1.0),
            ((current_box.top() - viewport_rect.top()) / viewport_rect.height().max(1.0))
                .clamp(0.0, 1.0),
        ]);
    }
    let ref_scale = scalar_bar_ref_scale(viewport_rect);
    let (min_w, max_w, min_h, max_h) = if cfg.horizontal {
        (120.0, 960.0, 58.0, 480.0)
    } else {
        (100.0, 640.0, 96.0, 960.0)
    };
    let max_px_w = (viewport_rect.right() - current_box.left()).max(min_w * ref_scale);
    let max_px_h = (viewport_rect.bottom() - current_box.top()).max(min_h * ref_scale);
    let new_px_w = (current_box.width() + delta.x).clamp(min_w * ref_scale, max_px_w);
    let new_px_h = (current_box.height() + delta.y).clamp(min_h * ref_scale, max_px_h);
    cfg.size = [
        (new_px_w / ref_scale).clamp(min_w, max_w),
        (new_px_h / ref_scale).clamp(min_h, max_h),
    ];
}

#[allow(clippy::too_many_arguments)]
fn paint_scalar_bar_card(
    painter: &egui::Painter,
    box_rect: egui::Rect,
    title: &str,
    cmap: &ColorMap,
    range: (f64, f64),
    cfg: &ScalarBarConfig,
    theme: UiTheme,
    title_col: egui::Color32,
    highlighted: bool,
) {
    let pal = palette(theme);
    let font_scale = cfg.font_scale.clamp(0.5, 4.0);
    let border_col = if highlighted { pal.accent } else { pal.hud_border };

    painter.rect_filled(box_rect, 7.0, pal.hud_bg);
    painter.rect_stroke(
        box_rect,
        7.0,
        egui::Stroke::new(if highlighted { 1.4 } else { 1.0 }, border_col),
        egui::StrokeKind::Inside,
    );

    let title_font = egui::FontId::proportional(11.5 * font_scale);
    let title_galley = painter.layout_no_wrap(title.to_string(), title_font, title_col);
    let title_h = title_galley.size().y;
    painter.galley(
        egui::pos2(box_rect.left() + 10.0, box_rect.top() + 7.0),
        title_galley,
        title_col,
    );

    let n_labels = cfg.num_labels.clamp(2, 7);
    let label_font = egui::FontId::monospace(10.0 * font_scale);
    let segs = 48;
    let (lo, hi) = range;
    let tick_val = |frac: f64| -> f64 {
        if cmap.log_scale && lo > 0.0 && hi > lo {
            lo * (hi / lo).powf(frac)
        } else {
            lo + (hi - lo) * frac
        }
    };
    let fmt_tick = |v: f64| -> String {
        let av = v.abs();
        if av >= 1e4 || (av > 0.0 && av < 1e-2) {
            format!("{v:.2e}")
        } else if av >= 100.0 {
            format!("{v:.1}")
        } else {
            format!("{v:.3}")
        }
    };

    if cfg.horizontal {
        let bx = box_rect.left() + 12.0;
        let by = box_rect.top() + 12.0 + title_h;
        let bar_w = (box_rect.width() - 24.0).max(20.0);
        let bar_h = (box_rect.bottom() - by - (14.0 * font_scale + 12.0)).clamp(10.0, 26.0);
        let bar_rect = egui::Rect::from_min_size(egui::pos2(bx, by), egui::vec2(bar_w, bar_h));

        for i in 0..segs {
            let f0 = i as f32 / segs as f32;
            let f1 = (i + 1) as f32 / segs as f32;
            let rgb = cmap.sample_rgb((f0 + f1) * 0.5);
            let c = egui::Color32::from_rgb(
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            );
            let x0 = bar_rect.left() + f0 * bar_rect.width();
            let x1 = bar_rect.left() + f1 * bar_rect.width();
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0, bar_rect.top()),
                    egui::pos2(x1, bar_rect.bottom()),
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

        let label_y = bar_rect.bottom() + 4.0;
        let mut last_right = f32::NEG_INFINITY;
        for i in 0..n_labels {
            let frac = i as f64 / (n_labels - 1) as f64;
            let tx = bar_rect.left() + frac as f32 * bar_rect.width();
            let align = if i == 0 {
                egui::Align2::LEFT_TOP
            } else if i + 1 == n_labels {
                egui::Align2::RIGHT_TOP
            } else {
                egui::Align2::CENTER_TOP
            };
            let text = fmt_tick(tick_val(frac));
            let g = painter.layout_no_wrap(text.clone(), label_font.clone(), title_col);
            let left_x = match align {
                egui::Align2::LEFT_TOP => tx,
                egui::Align2::RIGHT_TOP => tx - g.size().x,
                _ => tx - g.size().x * 0.5,
            };
            if i == 0 || i + 1 == n_labels || left_x > last_right + 4.0 {
                painter.text(
                    egui::pos2(tx, label_y),
                    align,
                    text,
                    label_font.clone(),
                    title_col,
                );
                last_right = left_x + g.size().x;
            }
        }
    } else {
        let bar_w = (16.0 * font_scale).clamp(12.0, 26.0);
        let bx = box_rect.left() + 12.0;
        let by = box_rect.top() + 14.0 + title_h;
        let bar_h = (box_rect.bottom() - by - 12.0).max(24.0);
        let bar_rect = egui::Rect::from_min_size(egui::pos2(bx, by), egui::vec2(bar_w, bar_h));

        for i in 0..segs {
            let f0 = i as f32 / segs as f32;
            let f1 = (i + 1) as f32 / segs as f32;
            let t = 1.0 - (f0 + f1) * 0.5;
            let rgb = cmap.sample_rgb(t);
            let c = egui::Color32::from_rgb(
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            );
            let y0 = bar_rect.top() + f0 * bar_rect.height();
            let y1 = bar_rect.top() + f1 * bar_rect.height();
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

        let label_x = bar_rect.right() + 8.0;
        let label_h = 12.0 * font_scale;
        let mut occupied: Vec<(f32, f32)> = Vec::new();
        let mut order: Vec<usize> = Vec::with_capacity(n_labels);
        order.push(n_labels - 1);
        order.push(0);
        for i in (1..n_labels.saturating_sub(1)).rev() {
            order.push(i);
        }
        for i in order {
            let frac = i as f64 / (n_labels - 1) as f64;
            let ty = bar_rect.top() + (1.0 - frac as f32) * bar_rect.height();
            let (align, ly) = if i + 1 == n_labels {
                (egui::Align2::LEFT_TOP, bar_rect.top())
            } else if i == 0 {
                (egui::Align2::LEFT_BOTTOM, bar_rect.bottom() - label_h)
            } else {
                (egui::Align2::LEFT_CENTER, ty - label_h * 0.5)
            };
            let is_endpoint = i == 0 || i + 1 == n_labels;
            let overlaps = !is_endpoint
                && occupied
                    .iter()
                    .any(|&(oy, oh)| ly < oy + oh + 2.0 && ly + label_h + 2.0 > oy);
            if overlaps {
                continue;
            }
            occupied.push((ly, label_h));
            let anchor_y = if i + 1 == n_labels {
                bar_rect.top()
            } else if i == 0 {
                bar_rect.bottom()
            } else {
                ty
            };
            painter.text(
                egui::pos2(label_x, anchor_y),
                align,
                fmt_tick(tick_val(frac)),
                label_font.clone(),
                title_col,
            );
        }
    }

    // Subtle bottom-right resize grip indicator
    let grip_col = if highlighted { pal.accent } else { pal.muted };
    for offset in [4.0, 8.0, 11.5] {
        painter.line_segment(
            [
                egui::pos2(box_rect.right() - offset, box_rect.bottom() - 2.0),
                egui::pos2(box_rect.right() - 2.0, box_rect.bottom() - offset),
            ],
            egui::Stroke::new(1.1, grip_col),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_bar_rect_drag_and_resize_clamped() {
        let vp = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 720.0));
        let mut cfg = ScalarBarConfig::default();

        // Default position places card at bottom-right
        let r0 = compute_scalar_bar_rect(vp, &cfg).expect("scalar bar rect");
        assert!(r0.right() <= vp.right());
        assert!(r0.bottom() <= vp.bottom());
        assert!(r0.left() > vp.center().x);

        // Drag clamps inside viewport
        drag_scalar_bar(&mut cfg, r0, vp, egui::vec2(-2000.0, -2000.0));
        let pos = cfg.pos.expect("pos set after drag");
        assert_eq!(pos, [0.0, 0.0]);

        // Resize clamps to valid bounds and pins pos if previously None
        let mut cfg2 = ScalarBarConfig::default();
        let r1 = compute_scalar_bar_rect(vp, &cfg2).unwrap();
        resize_scalar_bar(&mut cfg2, r1, vp, egui::vec2(40.0, 30.0));
        assert!(cfg2.pos.is_some());
        assert!(cfg2.size[0] >= 100.0 && cfg2.size[0] <= 640.0);
        assert!(cfg2.size[1] >= 96.0 && cfg2.size[1] <= 960.0);
    }

    #[test]
    fn annotation_pos_and_time_badge_formatting() {
        let vp = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(800.0, 600.0));
        let mut pos = [0.1, 0.2];
        let pill = egui::vec2(100.0, 28.0);
        let p = compute_annotation_pos(vp, pos, pill);
        let r = egui::Rect::from_min_size(p, pill);
        drag_annotation_pos(&mut pos, r, vp, egui::vec2(2000.0, -500.0));
        assert!((0.0..=1.0).contains(&pos[0]));
        assert_eq!(pos[1], 0.0);

        assert_eq!(format_time_annotation_text(None, 1.25), "Time: 1.2500");
        assert_eq!(
            format_time_annotation_text(Some("0.5 s"), 0.0),
            "Time: 0.5 s"
        );
        assert_eq!(
            format_time_annotation_text(Some("Time: 42"), 0.0),
            "Time: 42"
        );
    }

    #[test]
    fn split_viewport_rects_produce_expected_counts() {
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 600.0));
        assert_eq!(split_viewport_rects(full, ViewportLayout::Single).len(), 1);
        let lr = split_viewport_rects(full, ViewportLayout::SplitHorizontal);
        assert_eq!(lr.len(), 2);
        assert!(lr[0].right() < lr[1].left());

        let tb = split_viewport_rects(full, ViewportLayout::SplitVertical);
        assert_eq!(tb.len(), 2);
        assert!(tb[0].bottom() < tb[1].top());

        let quad = split_viewport_rects(full, ViewportLayout::Quad);
        assert_eq!(quad.len(), 4);
    }
}
