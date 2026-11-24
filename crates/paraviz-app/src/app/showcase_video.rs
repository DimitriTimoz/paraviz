use super::*;
use paraviz::core::{CellType, DataArray, DataSet, ImageData, UnstructuredGrid};
use std::io::Write;
use std::process::{Child, Command, Stdio};

const VID_W: u32 = 1920;
const VID_H: u32 = 1080;
const PPP: f32 = 1.25;
const LOGICAL_W: f32 = VID_W as f32 / PPP; // 1536.0
const LOGICAL_H: f32 = VID_H as f32 / PPP; // 864.0
const FPS: usize = 30;

const TOP_MENU_RECT: egui::Rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1536.0, 32.0));
const TOOLBAR_RECT: egui::Rect = egui::Rect::from_min_max(egui::pos2(0.0, 24.0), egui::pos2(1536.0, 130.0));
const LEFT_DOCK_RECT: egui::Rect = egui::Rect::from_min_max(egui::pos2(0.0, 80.0), egui::pos2(360.0, 840.0));
const INSPECTOR_TABS_RECT: egui::Rect =
    egui::Rect::from_min_max(egui::pos2(0.0, 120.0), egui::pos2(360.0, 380.0));

#[derive(Clone)]
struct ClickRipple {
    pos: egui::Pos2,
    start_time: f64,
}

#[derive(Clone)]
struct CalloutState {
    badge: String,
    title: String,
    subtitle: String,
    above_drawer: bool,
}

#[derive(Clone)]
struct KeyBadge {
    text: String,
    expires_at: f64,
}

#[derive(Clone)]
struct RecordedMark {
    kind: &'static str,
    name: String,
    time_sec: f64,
}

#[derive(Default)]
struct FrameEventBatch {
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
}

struct HeadlessStudio {
    device: wgpu::Device,
    queue: wgpu::Queue,
    egui_ctx: egui::Context,
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    app: ParaVizApp,
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    ffmpeg_child: Option<Child>,
    frame_idx: usize,
    sim_time: f64,
    cursor_pos: egui::Pos2,
    ripples: Vec<ClickRipple>,
    callout: Option<CalloutState>,
    key_badge: Option<KeyBadge>,
    zoom: f32,
    target_zoom: f32,
    focus_uv: [f32; 2],
    target_focus_uv: [f32; 2],
    intro_outro_overlay: Option<(String, String, String, f32)>,
    visible_labels: Vec<(String, egui::Rect)>,
    marks: Vec<RecordedMark>,
    rgba_buf: Vec<u8>,
    zoomed_buf: Vec<u8>,
}

impl HeadlessStudio {
    fn new(out_mp4: &Path) -> Self {
        let probe = WebGpuRenderer::new_headless().expect("headless wgpu adapter/device");
        let device = (*probe.device_arc()).clone();
        let queue = (*probe.queue_arc()).clone();

        let egui_ctx = egui::Context::default();

        let egui_renderer = Arc::new(egui::epaint::mutex::RwLock::new(
            egui_wgpu::Renderer::new(
                &device,
                wgpu::TextureFormat::Rgba8Unorm,
                egui_wgpu::RendererOptions {
                    dithering: false,
                    ..Default::default()
                },
            ),
        ));

        let app = ParaVizApp::new_headless(
            device.clone(),
            queue.clone(),
            egui_renderer.clone(),
            &egui_ctx,
        );

        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("showcase-studio-target"),
            size: wgpu::Extent3d {
                width: VID_W,
                height: VID_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("showcase-studio-staging"),
            size: (VID_W * VID_H * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ffmpeg_child = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                &format!("{VID_W}x{VID_H}"),
                "-r",
                &FPS.to_string(),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                out_mp4.to_str().expect("utf8 output mp4 path"),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn ffmpeg encoder");

        Self {
            device,
            queue,
            egui_ctx,
            egui_renderer,
            app,
            target_texture,
            target_view,
            staging_buffer,
            ffmpeg_child: Some(ffmpeg_child),
            frame_idx: 0,
            sim_time: 0.0,
            cursor_pos: egui::pos2(LOGICAL_W * 0.56, LOGICAL_H * 0.48),
            ripples: Vec::new(),
            callout: None,
            key_badge: None,
            zoom: 1.0,
            target_zoom: 1.0,
            focus_uv: [0.5, 0.5],
            target_focus_uv: [0.5, 0.5],
            intro_outro_overlay: None,
            visible_labels: Vec::new(),
            marks: Vec::new(),
            rgba_buf: vec![0u8; (VID_W * VID_H * 4) as usize],
            zoomed_buf: vec![0u8; (VID_W * VID_H * 4) as usize],
        }
    }

    fn current_sec(&self) -> f64 {
        self.frame_idx as f64 / FPS as f64
    }

    fn mark_act(&mut self, name: &str) {
        self.marks.push(RecordedMark {
            kind: "act",
            name: name.to_string(),
            time_sec: self.current_sec(),
        });
    }

    fn mark_click(&mut self, name: &str) {
        self.marks.push(RecordedMark {
            kind: "click",
            name: name.to_string(),
            time_sec: self.current_sec(),
        });
    }

    fn set_callout(&mut self, badge: &str, title: &str, subtitle: &str, above_drawer: bool) {
        self.callout = Some(CalloutState {
            badge: badge.to_string(),
            title: title.to_string(),
            subtitle: subtitle.to_string(),
            above_drawer,
        });
    }

    fn set_focus_zoom(&mut self, zoom: f32, uv: [f32; 2]) {
        self.target_zoom = zoom.clamp(1.0, 1.35);
        self.target_focus_uv = uv;
    }

    fn show_key_badge(&mut self, text: &str, duration_sec: f64) {
        self.key_badge = Some(KeyBadge {
            text: text.to_string(),
            expires_at: self.sim_time + duration_sec,
        });
    }

    fn step_frame(&mut self, batch: FrameEventBatch, record: bool) {
        self.app.fps_smoothed = 60.0;
        self.app.last_frame_time = Instant::now();
        self.app.wait_for_idle_jobs();

        self.zoom += (self.target_zoom - self.zoom) * 0.14;
        self.focus_uv[0] += (self.target_focus_uv[0] - self.focus_uv[0]) * 0.14;
        self.focus_uv[1] += (self.target_focus_uv[1] - self.focus_uv[1]) * 0.14;

        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(LOGICAL_W, LOGICAL_H),
            )),
            time: Some(self.sim_time),
            predicted_dt: 1.0 / FPS as f32,
            ..Default::default()
        };
        raw_input
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(PPP);
        raw_input
            .events
            .push(egui::Event::ModifiersChanged(batch.modifiers));
        raw_input.events.push(egui::Event::PointerMoved(self.cursor_pos));
        raw_input.events.extend(batch.events);

        let cursor_pos = self.cursor_pos;
        let sim_time = self.sim_time;
        let ripples = self.ripples.clone();
        let callout = self.callout.clone();
        let key_badge = self.key_badge.clone();
        let intro_outro = self.intro_outro_overlay.clone();
        let theme = self.app.theme;

        let full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            self.app.draw_studio_ui(ui);
            paint_demo_overlays(
                ui.ctx(),
                theme,
                cursor_pos,
                sim_time,
                &ripples,
                callout.as_ref(),
                key_badge.as_ref(),
                intro_outro.as_ref(),
            );
        });

        self.app.wait_for_idle_jobs();
        self.ripples.retain(|r| (self.sim_time - r.start_time) < 0.45);
        if self
            .key_badge
            .as_ref()
            .is_some_and(|kb| self.sim_time >= kb.expires_at)
        {
            self.key_badge = None;
        }

        self.visible_labels.clear();
        for clipped in &full_output.shapes {
            collect_labels(&clipped.shape, clipped.clip_rect, &mut self.visible_labels);
        }

        if record {
            self.render_and_write_frame(&full_output);
            self.frame_idx += 1;
        } else {
            for (id, deltas) in &full_output.textures_delta.set {
                for delta in deltas {
                    self.egui_renderer
                        .write()
                        .update_texture(&self.device, &self.queue, *id, delta);
                }
            }
            for id in &full_output.textures_delta.free {
                self.egui_renderer.write().free_texture(id);
            }
        }
        self.sim_time += 1.0 / FPS as f64;
    }

    fn render_and_write_frame(&mut self, full_output: &egui::FullOutput) {
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .write()
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes.clone(), full_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [VID_W, VID_H],
            pixels_per_point: full_output.pixels_per_point,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("showcase-studio-encoder"),
            });

        let user_cmd_bufs = self.egui_renderer.write().update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("showcase-studio-rpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut rpass = rpass.forget_lifetime();
            let guard = self.egui_renderer.read();
            guard.render(&mut rpass, &paint_jobs, &screen_desc);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(VID_W * 4),
                    rows_per_image: Some(VID_H),
                },
            },
            wgpu::Extent3d {
                width: VID_W,
                height: VID_H,
                depth_or_array_layers: 1,
            },
        );

        let mut cmd_bufs = user_cmd_bufs;
        cmd_bufs.push(encoder.finish());
        self.queue.submit(cmd_bufs);

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll headless studio frame");
        rx.recv().expect("map recv").expect("map ok");
        {
            let mapped = slice.get_mapped_range().expect("mapped staging view");
            self.rgba_buf.copy_from_slice(&mapped);
        }
        self.staging_buffer.unmap();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.write().free_texture(id);
        }

        let out_slice = if (self.zoom - 1.0).abs() > 0.002 {
            apply_smooth_zoom(
                &self.rgba_buf,
                &mut self.zoomed_buf,
                VID_W as usize,
                VID_H as usize,
                self.zoom,
                self.focus_uv,
            );
            &self.zoomed_buf
        } else {
            &self.rgba_buf
        };

        if let Some(child) = &mut self.ffmpeg_child
            && let Some(stdin) = child.stdin.as_mut()
        {
            stdin.write_all(out_slice).expect("write rgba frame to ffmpeg");
        }
    }

    fn warmup(&mut self) {
        for _ in 0..3 {
            self.step_frame(FrameEventBatch::default(), false);
        }
    }

    fn idle_frames(&mut self, count: usize) {
        for _ in 0..count {
            self.step_frame(FrameEventBatch::default(), true);
        }
    }

    fn find_label(&self, target: &str, region: Option<egui::Rect>) -> Option<egui::Pos2> {
        let mut best_contains: Option<egui::Pos2> = None;
        for (text, rect) in &self.visible_labels {
            let center = rect.center();
            if let Some(reg) = region
                && !reg.contains(center)
            {
                continue;
            }
            if text == target {
                return Some(center);
            }
            if best_contains.is_none() && text.contains(target) {
                best_contains = Some(center);
            }
        }
        best_contains
    }

    fn move_cursor_smooth(&mut self, target: egui::Pos2, frames: usize) {
        let start = self.cursor_pos;
        let dx = target.x - start.x;
        let dy = target.y - start.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let arc_perp = if dist > 30.0 {
            let nx = -dy / dist;
            let ny = dx / dist;
            let amp = (dist * 0.05).min(24.0);
            egui::vec2(nx * amp, ny * amp)
        } else {
            egui::Vec2::ZERO
        };

        for i in 1..=frames.max(1) {
            let t = i as f32 / frames.max(1) as f32;
            let ease = if t < 0.5 {
                4.0 * t * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
            };
            let arc = (t * std::f32::consts::PI).sin();
            self.cursor_pos = egui::pos2(
                start.x + dx * ease + arc_perp.x * arc,
                start.y + dy * ease + arc_perp.y * arc,
            );
            self.step_frame(FrameEventBatch::default(), true);
        }
    }

    fn click_pos(&mut self, pos: egui::Pos2, move_frames: usize, tag: &str) {
        self.move_cursor_smooth(pos, move_frames);
        self.idle_frames(4);
        self.mark_click(tag);
        self.ripples.push(ClickRipple {
            pos,
            start_time: self.sim_time,
        });
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            true,
        );
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            true,
        );
        self.app.wait_for_idle_jobs();
    }

    fn click_label(
        &mut self,
        label: &str,
        region: Option<egui::Rect>,
        fallback: egui::Pos2,
        move_frames: usize,
    ) {
        self.step_frame(FrameEventBatch::default(), false);
        let pos = self.find_label(label, region).unwrap_or(fallback);
        self.click_pos(pos, move_frames, label);
    }

    fn drag_smooth(&mut self, start: egui::Pos2, delta: egui::Vec2, frames: usize) {
        self.move_cursor_smooth(start, 12);
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            true,
        );
        for i in 1..=frames.max(1) {
            let t = i as f32 / frames.max(1) as f32;
            let ease = t * t * (3.0 - 2.0 * t);
            self.cursor_pos = start + delta * ease;
            self.step_frame(FrameEventBatch::default(), true);
        }
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::PointerButton {
                    pos: self.cursor_pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            true,
        );
    }

    fn press_key(&mut self, key: egui::Key, modifiers: egui::Modifiers) {
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers,
                }],
                modifiers,
            },
            true,
        );
        self.step_frame(
            FrameEventBatch {
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                modifiers: egui::Modifiers::NONE,
            },
            true,
        );
    }

    fn type_text_animated(&mut self, text: &str, frames_per_char: usize) {
        for ch in text.chars() {
            let mut s = String::new();
            s.push(ch);
            self.step_frame(
                FrameEventBatch {
                    events: vec![egui::Event::Text(s)],
                    ..Default::default()
                },
                true,
            );
            if frames_per_char > 1 {
                self.idle_frames(frames_per_char - 1);
            }
        }
    }

    fn scroll_smooth(&mut self, pos: egui::Pos2, total_delta_y: f32, frames: usize) {
        self.move_cursor_smooth(pos, 10);
        let step_y = total_delta_y / frames.max(1) as f32;
        for _ in 0..frames.max(1) {
            self.step_frame(
                FrameEventBatch {
                    events: vec![egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(0.0, step_y),
                        phase: egui::TouchPhase::Move,
                        source: egui::MouseWheelSource::Unknown,
                        modifiers: egui::Modifiers::NONE,
                    }],
                    ..Default::default()
                },
                true,
            );
        }
    }

    fn reset_scene_for_act(&mut self, path: &Path) {
        self.app.playing = false;
        self.app.turntable_enabled = false;
        self.app.show_drawer = false;
        self.app.show_widgets = false;
        self.app.inspector_tab = InspectorTab::Properties;
        self.app.viewport_layout = ViewportLayout::Single;
        self.app.active_pane = 0;
        for pane in &mut self.app.panes {
            pane.kind = PaneKind::View3D;
            pane.visible_nodes_override = None;
        }
        self.app.last_pick = None;
        self.app.picked_highlight = None;
        self.app.ruler_points.clear();
        self.app.undo_stack.clear();
        self.app.redo_stack.clear();
        self.app.error_message = None;
        self.app.session = ParaViz::new();
        self.app.selected_node = None;
        self.app.clear_stats_caches();
        self.app.load_path(path);
        self.app.wait_for_idle_jobs();
        self.app.invalidate_viewport();
        self.warmup();
    }

    fn finish(mut self) -> Vec<RecordedMark> {
        if let Some(mut child) = self.ffmpeg_child.take() {
            drop(child.stdin.take());
            let status = child.wait().expect("wait ffmpeg");
            assert!(status.success(), "ffmpeg exited with {status}");
        }
        self.marks
    }
}

fn collect_labels(
    shape: &egui::epaint::Shape,
    clip_rect: egui::Rect,
    out: &mut Vec<(String, egui::Rect)>,
) {
    match shape {
        egui::epaint::Shape::Vec(children) => {
            for c in children {
                collect_labels(c, clip_rect, out);
            }
        }
        egui::epaint::Shape::Text(ts) => {
            let txt = ts.galley.job.text.trim();
            if !txt.is_empty() {
                let rect = egui::Rect::from_min_size(ts.pos, ts.galley.size());
                if clip_rect.contains(rect.center()) {
                    out.push((txt.to_string(), rect));
                }
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_demo_overlays(
    ctx: &egui::Context,
    theme: UiTheme,
    cursor_pos: egui::Pos2,
    sim_time: f64,
    ripples: &[ClickRipple],
    callout: Option<&CalloutState>,
    key_badge: Option<&KeyBadge>,
    intro_outro: Option<&(String, String, String, f32)>,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::unique("showcase_demo_overlay"),
    ));
    let pal = palette(theme);

    // 1. Intro / Outro glass overlay
    if let Some((badge, title, subtitle, alpha_f)) = intro_outro {
        let a = (alpha_f.clamp(0.0, 1.0) * 255.0) as u8;
        let full = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(LOGICAL_W, LOGICAL_H));
        painter.rect_filled(
            full,
            0.0,
            egui::Color32::from_rgba_unmultiplied(10, 14, 22, (alpha_f * 185.0) as u8),
        );
        let card = egui::Rect::from_center_size(
            egui::pos2(LOGICAL_W * 0.5, LOGICAL_H * 0.5),
            egui::vec2(760.0, 220.0),
        );
        painter.rect_filled(
            card,
            14.0,
            egui::Color32::from_rgba_unmultiplied(18, 22, 32, (alpha_f * 242.0) as u8),
        );
        painter.rect_stroke(
            card,
            14.0,
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(56, 189, 248, (alpha_f * 200.0) as u8),
            ),
            egui::StrokeKind::Inside,
        );
        painter.text(
            card.left_top() + egui::vec2(36.0, 32.0),
            egui::Align2::LEFT_TOP,
            badge,
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgba_unmultiplied(56, 189, 248, a),
        );
        painter.text(
            card.left_top() + egui::vec2(36.0, 62.0),
            egui::Align2::LEFT_TOP,
            title,
            egui::FontId::proportional(34.0),
            egui::Color32::from_rgba_unmultiplied(244, 247, 252, a),
        );
        painter.text(
            card.left_top() + egui::vec2(36.0, 122.0),
            egui::Align2::LEFT_TOP,
            subtitle,
            egui::FontId::proportional(17.0),
            egui::Color32::from_rgba_unmultiplied(180, 195, 218, a),
        );
    }

    // 2. Chapter Callout Card (bottom-right above status bar, or top-center when drawer open)
    if let Some(co) = callout {
        let card_w = 505.0;
        let card_h = 74.0;
        let card = if co.above_drawer {
            egui::Rect::from_min_size(
                egui::pos2(LOGICAL_W - card_w - 18.0, 148.0),
                egui::vec2(card_w, card_h),
            )
        } else {
            egui::Rect::from_min_size(
                egui::pos2(LOGICAL_W - card_w - 18.0, LOGICAL_H - card_h - 34.0),
                egui::vec2(card_w, card_h),
            )
        };
        painter.rect_filled(
            card,
            10.0,
            egui::Color32::from_rgba_unmultiplied(15, 19, 28, 228),
        );
        painter.rect_stroke(
            card,
            10.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(56, 189, 248, 165)),
            egui::StrokeKind::Inside,
        );
        let accent_bar = egui::Rect::from_min_size(
            card.left_top() + egui::vec2(12.0, 13.0),
            egui::vec2(4.0, card_h - 26.0),
        );
        painter.rect_filled(accent_bar, 2.0, pal.accent);
        painter.text(
            card.left_top() + egui::vec2(26.0, 11.0),
            egui::Align2::LEFT_TOP,
            &co.badge,
            egui::FontId::proportional(11.0),
            pal.accent,
        );
        painter.text(
            card.left_top() + egui::vec2(26.0, 27.0),
            egui::Align2::LEFT_TOP,
            &co.title,
            egui::FontId::proportional(16.0),
            egui::Color32::from_rgb(242, 246, 252),
        );
        painter.text(
            card.left_top() + egui::vec2(26.0, 49.0),
            egui::Align2::LEFT_TOP,
            &co.subtitle,
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgb(174, 188, 210),
        );
    }

    // 3. Shortcut Keycap Badge
    if let Some(kb) = key_badge {
        let rem = (kb.expires_at - sim_time).clamp(0.0, 0.35) / 0.35;
        let alpha = (rem * 240.0) as u8;
        let pill = egui::Rect::from_center_size(
            egui::pos2(LOGICAL_W * 0.5, LOGICAL_H - 64.0),
            egui::vec2((kb.text.len() as f32 * 9.5 + 36.0).max(120.0), 34.0),
        );
        painter.rect_filled(
            pill,
            8.0,
            egui::Color32::from_rgba_unmultiplied(15, 20, 30, alpha),
        );
        painter.rect_stroke(
            pill,
            8.0,
            egui::Stroke::new(1.2, egui::Color32::from_rgba_unmultiplied(56, 189, 248, alpha)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            pill.center(),
            egui::Align2::CENTER_CENTER,
            &kb.text,
            egui::FontId::monospace(14.0),
            egui::Color32::from_rgba_unmultiplied(240, 246, 255, alpha),
        );
    }

    // 4. Concentric Click Ripples
    for r in ripples {
        let age = ((sim_time - r.start_time) / 0.45) as f32;
        if (0.0..1.0).contains(&age) {
            let alpha = ((1.0 - age) * 220.0) as u8;
            let radius = 6.0 + age * 28.0;
            painter.circle_stroke(
                r.pos,
                radius,
                egui::Stroke::new(2.2, egui::Color32::from_rgba_unmultiplied(56, 189, 248, alpha)),
            );
            painter.circle_stroke(
                r.pos,
                radius * 0.55,
                egui::Stroke::new(
                    1.5,
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha / 2),
                ),
            );
        }
    }

    // 5. Crisp Vector Pointer Cursor
    let p = cursor_pos;
    let shadow_off = egui::vec2(1.5, 2.0);
    let arrow = [
        p,
        p + egui::vec2(0.0, 18.0),
        p + egui::vec2(4.6, 13.8),
        p + egui::vec2(8.0, 20.5),
        p + egui::vec2(10.6, 19.2),
        p + egui::vec2(7.2, 12.6),
        p + egui::vec2(13.0, 12.2),
    ];
    let shadow_pts: Vec<egui::Pos2> = arrow.iter().map(|q| *q + shadow_off).collect();
    painter.add(egui::Shape::convex_polygon(
        vec![shadow_pts[0], shadow_pts[1], shadow_pts[2], shadow_pts[6]],
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 110),
        egui::Stroke::NONE,
    ));
    painter.add(egui::Shape::closed_line(
        arrow.to_vec(),
        egui::Stroke::new(2.6, egui::Color32::from_rgb(12, 16, 24)),
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![arrow[0], arrow[1], arrow[2], arrow[6]],
        egui::Color32::WHITE,
        egui::Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![arrow[2], arrow[3], arrow[4], arrow[5]],
        egui::Color32::WHITE,
        egui::Stroke::NONE,
    ));
}

fn apply_smooth_zoom(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    zoom: f32,
    focus_uv: [f32; 2],
) {
    let inv_z = 1.0 / zoom.max(1.0);
    let view_w = w as f32 * inv_z;
    let view_h = h as f32 * inv_z;
    let cx = (focus_uv[0] * w as f32).clamp(view_w * 0.5, w as f32 - view_w * 0.5);
    let cy = (focus_uv[1] * h as f32).clamp(view_h * 0.5, h as f32 - view_h * 0.5);
    let x0 = cx - view_w * 0.5;
    let y0 = cy - view_h * 0.5;

    for y in 0..h {
        let sy = (y0 + (y as f32 + 0.5) * inv_z - 0.5).clamp(0.0, (h - 2) as f32);
        let iy = sy as usize;
        let fy = sy - iy as f32;
        let row0 = iy * w * 4;
        let row1 = (iy + 1) * w * 4;
        let dst_row = y * w * 4;

        for x in 0..w {
            let sx = (x0 + (x as f32 + 0.5) * inv_z - 0.5).clamp(0.0, (w - 2) as f32);
            let ix = sx as usize;
            let fx = sx - ix as f32;

            let idx00 = row0 + ix * 4;
            let idx10 = idx00 + 4;
            let idx01 = row1 + ix * 4;
            let idx11 = idx01 + 4;

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let out = dst_row + x * 4;
            for c in 0..4 {
                let v = src[idx00 + c] as f32 * w00
                    + src[idx10 + c] as f32 * w10
                    + src[idx01 + c] as f32 * w01
                    + src[idx11 + c] as f32 * w11;
                dst[out + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Procedural Scientific Datasets for Acts 1 - 5
// ---------------------------------------------------------------------------

fn generate_cfd_mixing_duct_vti(path: &Path) {
    let [nx, ny, nz] = [42usize, 34, 34];
    let mut img = ImageData::new(
        [nx, ny, nz],
        [-2.0, -1.0, -1.0],
        [4.0 / (nx - 1) as f64, 2.0 / (ny - 1) as f64, 2.0 / (nz - 1) as f64],
    );
    let n = nx * ny * nz;
    let mut vel = Vec::with_capacity(n);
    let mut pres = Vec::with_capacity(n);
    let mut vort = Vec::with_capacity(n);

    for k in 0..nz {
        let z = -1.0 + 2.0 * k as f32 / (nz - 1) as f32;
        for j in 0..ny {
            let y = -1.0 + 2.0 * j as f32 / (ny - 1) as f32;
            for i in 0..nx {
                let x = -2.0 + 4.0 * i as f32 / (nx - 1) as f32;
                let dy1 = y - 0.36 * (1.6 * x).sin();
                let dz1 = z - 0.36 * (1.6 * x).cos();
                let r1 = dy1 * dy1 + dz1 * dz1 + 0.06;

                let dy2 = y + 0.36 * (1.6 * x).sin();
                let dz2 = z + 0.36 * (1.6 * x).cos();
                let r2 = dy2 * dy2 + dz2 * dz2 + 0.06;

                let vx = 1.65 - 0.35 * (y * y + z * z) + 0.30 * (2.0 * x).cos();
                let vy = -dz1 / r1 + dz2 / r2;
                let vz = dy1 / r1 - dy2 / r2;
                vel.push([vx, vy, vz]);
                pres.push(1.2 - 0.35 * (1.0 / r1 + 1.0 / r2).min(4.5) + 0.15 * x);
                vort.push((1.0 / r1 + 1.0 / r2).min(12.0));
            }
        }
    }

    img.point_data.add_array(DataArray::vectors_f32("Velocity", vel));
    img.point_data.add_array(DataArray::scalars_f32("Pressure", pres));
    img.point_data
        .add_array(DataArray::scalars_f32("VorticityMag", vort));
    paraviz::io::write_dataset(path, &DataSet::ImageData(img)).expect("write cfd vti");
}

fn generate_fea_l_bracket_vtu(path: &Path) {
    let [nx, ny, nz] = [42usize, 32, 12];
    let mut ug = UnstructuredGrid::new();
    let mut stress = Vec::with_capacity(nx * ny * nz);
    let mut disp = Vec::with_capacity(nx * ny * nz);

    for k in 0..nz {
        let z = -0.35 + 0.70 * k as f32 / (nz - 1) as f32;
        for j in 0..ny {
            let y = 1.80 * j as f32 / (ny - 1) as f32;
            for i in 0..nx {
                let x = 2.40 * i as f32 / (nx - 1) as f32;
                // Bending + fillet + bolt-hole Kirsch stress concentration
                let fillet_d = ((x - 0.85).powi(2) + (y - 0.65).powi(2)).sqrt() + 0.12;
                let hole_d = ((x - 1.85).powi(2) + z.powi(2)).sqrt() + 0.08;
                let vm = 35.0
                    + 185.0 * (-fillet_d * 2.4).exp()
                    + 110.0 * (-((hole_d - 0.24) / 0.12).powi(2)).exp()
                    + 28.0 * (x / 2.4);
                let dx = 0.04 * x * y * 0.4;
                let dy = -0.14 * (x / 2.4).powi(2);
                let dz = 0.02 * z * (x / 2.4);
                ug.points.push([x, y, z]);
                stress.push(vm);
                disp.push([dx, dy, dz]);
            }
        }
    }

    let idx = |i: usize, j: usize, k: usize| -> u32 { (i + nx * (j + ny * k)) as u32 };
    for k in 0..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let cx = 2.40 * (i as f32 + 0.5) / (nx - 1) as f32;
                let cy = 1.80 * (j as f32 + 0.5) / (ny - 1) as f32;
                let cz = -0.35 + 0.70 * (k as f32 + 0.5) / (nz - 1) as f32;
                // Carve L-bracket cutout (x > 0.85 && y > 0.65) and bolt hole at (1.85, z=0)
                if cx > 0.88 && cy > 0.68 {
                    continue;
                }
                if ((cx - 1.85).powi(2) + cz.powi(2)).sqrt() < 0.21 {
                    continue;
                }
                if ((cy - 1.35).powi(2) + cz.powi(2)).sqrt() < 0.18 && cx < 0.88 {
                    continue;
                }
                ug.push_cell(
                    CellType::Hexahedron,
                    &[
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    ],
                );
            }
        }
    }

    ug.point_data
        .add_array(DataArray::scalars_f32("VonMises_MPa", stress));
    ug.point_data
        .add_array(DataArray::vectors_f32("Displacement", disp));
    paraviz::io::write_dataset(path, &DataSet::UnstructuredGrid(ug)).expect("write fea vtu");
}

fn generate_medical_ct_scan_vti(path: &Path) {
    let [nx, ny, nz] = [56usize, 56, 56];
    let mut img = ImageData::new(
        [nx, ny, nz],
        [-1.0, -1.0, -1.0],
        [2.0 / (nx - 1) as f64, 2.0 / (ny - 1) as f64, 2.0 / (nz - 1) as f64],
    );
    let mut hu = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz {
        let z = -1.0 + 2.0 * k as f32 / (nz - 1) as f32;
        for j in 0..ny {
            let y = -1.0 + 2.0 * j as f32 / (ny - 1) as f32;
            for i in 0..nx {
                let x = -1.0 + 2.0 * i as f32 / (nx - 1) as f32;
                let body = (x / 0.82).powi(2) + (y / 0.68).powi(2) + (z / 0.78).powi(2);
                let lung_l = ((x + 0.30) / 0.24).powi(2) + (y / 0.28).powi(2) + (z / 0.42).powi(2);
                let lung_r = ((x - 0.30) / 0.24).powi(2) + (y / 0.28).powi(2) + (z / 0.42).powi(2);
                let spine = ((x) / 0.14).powi(2) + ((y + 0.38) / 0.14).powi(2);
                let aorta = ((x - 0.05) / 0.11).powi(2) + ((y + 0.08) / 0.11).powi(2);
                let rib_ring = (body.sqrt() - 0.78).abs();

                let mut val = 0.0f32;
                if body < 1.0 {
                    val = 120.0;
                    if rib_ring < 0.055 && ((z * 16.0).sin() > 0.15) {
                        val = 740.0;
                    }
                    if spine < 1.0 && z.abs() < 0.72 {
                        val = 800.0;
                    }
                    if lung_l < 1.0 || lung_r < 1.0 {
                        val = 25.0;
                    }
                    if aorta < 1.0 && z.abs() < 0.65 {
                        val = 980.0;
                    }
                }
                hu.push(val);
            }
        }
    }
    img.point_data.add_array(DataArray::scalars_f32("HU_Density", hu));
    paraviz::io::write_dataset(path, &DataSet::ImageData(img)).expect("write medical vti");
}

fn generate_karman_xdmf_series(path: &Path) {
    let [nx, ny, nz] = [30usize, 18, 2];
    let npts = nx * ny * nz;
    let mut coords = String::new();
    for k in 0..nz {
        let z = -0.05 + 0.10 * k as f32;
        for j in 0..ny {
            let y = -1.0 + 2.0 * j as f32 / (ny - 1) as f32;
            for i in 0..nx {
                let x = 4.0 * i as f32 / (nx - 1) as f32;
                coords.push_str(&format!("{x:.3} {y:.3} {z:.3} "));
            }
        }
    }

    let mut conn = String::new();
    let mut ncells = 0usize;
    let pid = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    for k in 0..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                conn.push_str(&format!(
                    "{} {} {} {} {} {} {} {} ",
                    pid(i, j, k),
                    pid(i + 1, j, k),
                    pid(i + 1, j + 1, k),
                    pid(i, j + 1, k),
                    pid(i, j, k + 1),
                    pid(i + 1, j, k + 1),
                    pid(i + 1, j + 1, k + 1),
                    pid(i, j + 1, k + 1),
                ));
                ncells += 1;
            }
        }
    }

    let mut grids_xml = String::new();
    let nsteps = 16;
    for step in 0..nsteps {
        let t = step as f32 * 0.25;
        let mut scalars = String::new();
        for _k in 0..nz {
            for j in 0..ny {
                let y = -1.0 + 2.0 * j as f32 / (ny - 1) as f32;
                for i in 0..nx {
                    let x = 4.0 * i as f32 / (nx - 1) as f32;
                    let phase = 2.6 * x - 2.2 * t;
                    let vortex = phase.sin() * (-1.9 * y * y).exp() * (1.0 - (-1.4 * x).exp());
                    let u_mag = 1.0 + 0.85 * vortex;
                    scalars.push_str(&format!("{u_mag:.3} "));
                }
            }
        }
        let geom_block = if step == 0 {
            format!(
                r#"<Topology Name="Topo" TopologyType="Hexahedron" NumberOfElements="{ncells}">
          <DataItem Name="Conn" Format="XML" Dimensions="{ncells} 8">{conn}</DataItem>
        </Topology>
        <Geometry Name="Geom" GeometryType="XYZ">
          <DataItem Name="Pts" Format="XML" Dimensions="{npts} 3">{coords}</DataItem>
        </Geometry>"#
            )
        } else {
            r#"<Topology Reference="/Xdmf/Domain/Grid/Grid[1]/Topology"/>
        <Geometry Reference="/Xdmf/Domain/Grid/Grid[1]/Geometry"/>"#
                .to_string()
        };
        grids_xml.push_str(&format!(
            r#"      <Grid Name="Step_{step}" GridType="Uniform">
        <Time Value="{t:.2}"/>
        {geom_block}
        <Attribute Name="WakeVelocity" AttributeType="Scalar" Center="Node">
          <DataItem Format="XML" Dimensions="{npts}">{scalars}</DataItem>
        </Attribute>
      </Grid>
"#
        ));
    }

    let xml = format!(
        r#"<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="KarmanWake" GridType="Collection" CollectionType="Temporal">
{grids_xml}    </Grid>
  </Domain>
</Xdmf>
"#
    );
    std::fs::write(path, xml).expect("write xdmf series");
}

fn generate_protein_ligand_xyz(path: &Path) {
    let mut lines = Vec::new();
    let n_helix = 180;
    let mut coords = Vec::new();
    for i in 0..n_helix {
        let t = i as f32 * 0.32;
        let z = (i as f32 - n_helix as f32 * 0.5) * 0.11;
        coords.push(("P", 2.1 * t.cos(), 2.1 * t.sin(), z));
        coords.push(("O", 1.65 * (t + 0.22).cos(), 1.65 * (t + 0.22).sin(), z));
        coords.push((
            "N",
            2.1 * (t + std::f32::consts::PI).cos(),
            2.1 * (t + std::f32::consts::PI).sin(),
            z,
        ));
        coords.push((
            "C",
            1.65 * (t + std::f32::consts::PI + 0.22).cos(),
            1.65 * (t + std::f32::consts::PI + 0.22).sin(),
            z,
        ));
        if i % 3 == 0 {
            for r in -3i32..=3 {
                let frac = r as f32 / 3.5;
                let sym = if r == 0 {
                    "S"
                } else if r.abs() % 2 == 1 {
                    "C"
                } else {
                    "H"
                };
                coords.push((sym, frac * 1.55 * t.cos(), frac * 1.55 * t.sin(), z));
            }
        }
    }
    lines.push(coords.len().to_string());
    lines.push("ParaViz Protein Helix & Ligand".to_string());
    for (s, x, y, z) in coords {
        lines.push(format!("{s} {x:.4} {y:.4} {z:.4}"));
    }
    std::fs::write(path, lines.join("\n")).expect("write xyz");
}

#[test]
#[ignore = "Generates full 1080p headless Studio interface demo video via ffmpeg"]
fn generate_showcase_video() {
    let out_dir = PathBuf::from(format!("/tmp/pv_demo_{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("create showcase temp dir");

    let cfd_path = out_dir.join("cfd_mixing_duct.vti");
    let fea_path = out_dir.join("fea_l_bracket.vtu");
    let med_path = out_dir.join("medical_ct_scan.vti");
    let xdmf_path = out_dir.join("karman_wake_series.xmf");
    let xyz_path = out_dir.join("protein_ligand.xyz");

    generate_cfd_mixing_duct_vti(&cfd_path);
    generate_fea_l_bracket_vtu(&fea_path);
    generate_medical_ct_scan_vti(&med_path);
    generate_karman_xdmf_series(&xdmf_path);
    generate_protein_ligand_xyz(&xyz_path);

    let out_mp4 = std::env::var("PARAVIZ_SHOWCASE_MP4")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.join("paraviz_studio_silent.mp4"));
    let meta_json = std::env::var("PARAVIZ_SHOWCASE_META")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.join("paraviz_showcase_meta.json"));

    let mut studio = HeadlessStudio::new(&out_mp4);
    studio.app.load_path(&cfd_path);
    studio.app.wait_for_idle_jobs();
    studio.warmup();

    // =========================================================================
    // INTRO: Live Studio reveal (2.4s)
    // =========================================================================
    studio.mark_act("00_intro");
    studio.intro_outro_overlay = Some((
        "PARAVIZ  v0.2.10  •  RUST + WEBGPU".into(),
        "Interactive 3D Scientific Visualization Studio".into(),
        "Live Interface Walkthrough: CFD, Structural FEA, Medical Volumes & Time Series".into(),
        1.0,
    ));
    studio.idle_frames(52);
    for step in 1..=18 {
        let alpha = 1.0 - step as f32 / 18.0;
        studio.intro_outro_overlay = Some((
            "PARAVIZ  v0.2.10  •  RUST + WEBGPU".into(),
            "Interactive 3D Scientific Visualization Studio".into(),
            "Live Interface Walkthrough: CFD, Structural FEA, Medical Volumes & Time Series".into(),
            alpha,
        ));
        studio.step_frame(FrameEventBatch::default(), true);
    }
    studio.intro_outro_overlay = None;

    // =========================================================================
    // ACT 1: CFD Mixing Duct + Workspace Profile + Command Palette (Stream Tracer & Live Slice)
    // =========================================================================
    studio.mark_act("01_cfd");
    studio.set_callout(
        "01 / 05  •  CFD FLOW & COMMAND PALETTE",
        "Twin-Vortex Mixing Duct",
        "CFD profile, live 3D Stream Tracer & interactive Slice",
        false,
    );

    // Switch Workspace Profile to CFD via Toolbar ComboBox
    studio.click_label("Profile: Full", Some(TOOLBAR_RECT), egui::pos2(248.0, 44.0), 14);
    studio.idle_frames(6);
    studio.click_label("CFD", None, egui::pos2(248.0, 88.0), 10);
    studio.idle_frames(8);

    // Enable Bounds and EDL in toolbar
    studio.click_label("Bounds", Some(TOOLBAR_RECT), egui::pos2(725.0, 44.0), 12);
    studio.click_label("EDL", Some(TOOLBAR_RECT), egui::pos2(890.0, 44.0), 10);

    // Orbit 3D Viewport smoothly with real mouse drag
    studio.drag_smooth(egui::pos2(780.0, 440.0), egui::vec2(140.0, -55.0), 28);

    // Open Command Palette via toolbar "⌘K Search" button
    studio.show_key_badge("Cmd + K  —  Command Palette", 2.0);
    studio.set_focus_zoom(1.18, [0.5, 0.25]);
    studio.click_label("⌘K Search", Some(TOOLBAR_RECT), egui::pos2(190.0, 44.0), 12);
    studio.idle_frames(8);
    studio.type_text_animated("Stream Tracer", 2);
    studio.idle_frames(12);
    studio.press_key(egui::Key::Enter, egui::Modifiers::NONE);
    studio.app.wait_for_idle_jobs();
    studio.set_focus_zoom(1.0, [0.5, 0.5]);
    studio.idle_frames(22);

    // Orbit around the new 3D streamlines
    studio.drag_smooth(egui::pos2(880.0, 430.0), egui::vec2(-165.0, 40.0), 30);

    // Select source mixing duct in Pipeline Browser, open Command Palette with Cmd+K -> Slice
    studio.click_label("cfd_mixing_duct.vti", Some(LEFT_DOCK_RECT), egui::pos2(115.0, 128.0), 10);
    studio.show_key_badge("Cmd + K  —  Slice Filter", 1.8);
    studio.press_key(egui::Key::K, egui::Modifiers::COMMAND);
    studio.idle_frames(6);
    studio.type_text_animated("Slice", 2);
    studio.idle_frames(8);
    studio.press_key(egui::Key::Enter, egui::Modifiers::NONE);
    studio.app.wait_for_idle_jobs();

    // Ensure Properties tab is active and drag Slice Origin X interactively
    studio.click_label("Properties", Some(INSPECTOR_TABS_RECT), egui::pos2(45.0, 210.0), 10);
    studio.set_focus_zoom(1.16, [0.24, 0.45]);
    if let Some(orig_pos) = studio.find_label("Origin", Some(LEFT_DOCK_RECT)) {
        let drag_x = egui::pos2(orig_pos.x + 12.0, orig_pos.y + 20.0);
        studio.drag_smooth(drag_x, egui::vec2(55.0, 0.0), 30);
        studio.drag_smooth(egui::pos2(drag_x.x + 55.0, drag_x.y), egui::vec2(-85.0, 0.0), 30);
    } else {
        studio.idle_frames(40);
    }
    studio.set_focus_zoom(1.0, [0.5, 0.5]);
    studio.idle_frames(14);

    // =========================================================================
    // ACT 2: Structural FEA L-Bracket — Filters Menu, 3D Picking & Spreadsheet
    // =========================================================================
    studio.mark_act("02_fea");
    studio.set_callout(
        "02 / 05  •  STRUCTURAL FEA & 3D PICKING",
        "Perforated L-Bracket Stress Analysis",
        "Von Mises stress mesh, 3D raycast picking & Spreadsheet",
        true,
    );

    studio.reset_scene_for_act(&fea_path);

    // Showcase Filters menu browsing briefly
    studio.click_label("Filters", Some(TOP_MENU_RECT), egui::pos2(198.0, 14.0), 12);
    studio.idle_frames(12);
    studio.move_cursor_smooth(egui::pos2(215.0, 145.0), 10);
    studio.idle_frames(8);
    studio.press_key(egui::Key::Escape, egui::Modifiers::NONE);

    // Extract PolyData boundary surface & Solo it so 3D pick returns exact Pt/Cell IDs
    studio.press_key(egui::Key::K, egui::Modifiers::COMMAND);
    studio.idle_frames(5);
    studio.type_text_animated("Extract Surface", 2);
    studio.idle_frames(8);
    studio.press_key(egui::Key::Enter, egui::Modifiers::NONE);
    studio.app.wait_for_idle_jobs();
    studio.click_label("Solo", Some(LEFT_DOCK_RECT), egui::pos2(76.0, 96.0), 10);

    // Switch to Display tab and choose SurfaceWithEdges
    studio.click_label("Display", Some(INSPECTOR_TABS_RECT), egui::pos2(105.0, 195.0), 12);
    studio.idle_frames(8);
    studio.click_label("Surface", Some(LEFT_DOCK_RECT), egui::pos2(140.0, 285.0), 10);
    studio.idle_frames(6);
    studio.click_label("SurfaceWithEdges", None, egui::pos2(140.0, 315.0), 8);
    studio.idle_frames(8);

    // Orbit bracket to expose inner fillet
    studio.drag_smooth(egui::pos2(860.0, 450.0), egui::vec2(-115.0, -45.0), 28);

    // Switch to Pick mode and pick a guaranteed mesh point via pick_scene candidates
    studio.show_key_badge("Mode: 3D Raycast Pick", 2.0);
    studio.click_label("Pick", Some(TOOLBAR_RECT), egui::pos2(600.0, 44.0), 12);

    let (tw, th) = studio.app.panes[0].tex_size;
    let pane_x0 = (LOGICAL_W - tw as f32 / PPP).max(0.0);
    let pane_y0 = (LOGICAL_H - 22.0 - th as f32 / PPP).max(0.0);
    let candidates = [
        (995.0_f32, 505.0_f32),
        (975.0, 480.0),
        (1010.0, 525.0),
        (965.0, 600.0),
        (1020.0, 615.0),
        (955.0, 420.0),
    ];
    let pick_pt = studio.app.panes[0]
        .cached_scene
        .as_ref()
        .and_then(|scene| {
            candidates.iter().copied().find(|&(cx, cy)| {
                let px = (cx - pane_x0) * PPP;
                let py = (cy - pane_y0) * PPP;
                paraviz::render::pick_scene(scene, px, py, tw, th)
                    .is_some_and(|hit| hit.point_index.is_some() || hit.cell_index.is_some())
            })
        })
        .map(|(x, y)| egui::pos2(x, y))
        .unwrap_or_else(|| egui::pos2(995.0, 505.0));
    studio.click_pos(pick_pt, 18, "viewport_pick");
    if studio.app.picked_highlight.is_none() {
        for &(cx, cy) in &candidates {
            let pt = egui::pos2(cx, cy);
            if pt != pick_pt {
                studio.click_pos(pt, 6, "viewport_pick_retry");
                if studio.app.picked_highlight.is_some() {
                    break;
                }
            }
        }
    }
    studio.idle_frames(14);

    // Click "Reveal in Spreadsheet" button on HUD
    studio.set_focus_zoom(1.14, [0.52, 0.56]);
    studio.click_label(
        "Reveal in Spreadsheet",
        None,
        egui::pos2(410.0, 175.0),
        14,
    );
    studio.idle_frames(28);

    // =========================================================================
    // ACT 3: Split View (1x2) + Plot Over Line XY Chart + Rust Trace + Undo/Redo
    // =========================================================================
    studio.mark_act("03_split");
    studio.set_callout(
        "03 / 05  •  SPLIT VIEWPORT, CHARTS & RUST TRACE",
        "Probe Over Line, XY Plots & Code Trace",
        "Side-by-side 3D + XY Chart, Rust Trace & Undo / Redo",
        true,
    );
    studio.set_focus_zoom(1.0, [0.5, 0.5]);

    // Select bracket node, switch to Orbit, and add Probe Over Line via Cmd+K
    studio.click_label("fea_l_bracket.vtu", Some(LEFT_DOCK_RECT), egui::pos2(115.0, 128.0), 8);
    studio.click_label("Orbit", Some(TOOLBAR_RECT), egui::pos2(555.0, 44.0), 10);
    studio.press_key(egui::Key::K, egui::Modifiers::COMMAND);
    studio.idle_frames(5);
    studio.type_text_animated("Probe Over Line", 2);
    studio.idle_frames(8);
    studio.press_key(egui::Key::Enter, egui::Modifiers::NONE);
    studio.app.wait_for_idle_jobs();

    // Turn on Widgets so the 3D probe line widget appears
    studio.click_label("Widgets", Some(TOOLBAR_RECT), egui::pos2(775.0, 44.0), 10);
    studio.idle_frames(10);

    // Split View 1x2 -> click "XY Chart" directly in Pane 2 header
    studio.show_key_badge("1×2 Split Viewport", 2.0);
    studio.click_label("1x2", Some(TOOLBAR_RECT), egui::pos2(1070.0, 44.0), 12);
    studio.idle_frames(10);
    let pane2_region = egui::Rect::from_min_max(egui::pos2(900.0, 50.0), egui::pos2(1536.0, 155.0));
    studio.click_label("XY Chart", Some(pane2_region), egui::pos2(1080.0, 105.0), 12);
    studio.idle_frames(18);

    // Orbit Pane 1 while XY Chart renders live in Pane 2
    studio.drag_smooth(egui::pos2(620.0, 320.0), egui::vec2(120.0, 35.0), 26);

    // Switch bottom drawer to Rust Trace tab and zoom gently on live Rust code
    let drawer_tabs_region = egui::Rect::from_min_max(egui::pos2(300.0, 540.0), egui::pos2(950.0, 660.0));
    studio.click_label("Rust Trace", Some(drawer_tabs_region), egui::pos2(540.0, 600.0), 14);
    studio.set_focus_zoom(1.18, [0.48, 0.78]);
    studio.idle_frames(34);
    studio.set_focus_zoom(1.0, [0.5, 0.5]);

    // Collapse drawer, return to 1x1, demonstrate Undo -> Redo
    studio.click_label("Close", Some(egui::Rect::from_min_max(egui::pos2(1200.0, 540.0), egui::pos2(1536.0, 660.0))), egui::pos2(1490.0, 600.0), 10);
    studio.click_label("1x1", Some(TOOLBAR_RECT), egui::pos2(1040.0, 44.0), 10);
    studio.show_key_badge("Cmd + Z  —  Full Undo / Redo", 1.8);
    studio.click_label("Undo", Some(TOOLBAR_RECT), egui::pos2(24.0, 44.0), 10);
    studio.idle_frames(12);
    studio.click_label("Redo", Some(TOOLBAR_RECT), egui::pos2(68.0, 44.0), 10);
    studio.idle_frames(10);

    // =========================================================================
    // ACT 4: 3D Medical CT Volume Raymarching + Opacity TF Curve + Clip Plane
    // =========================================================================
    studio.mark_act("04_volume");
    studio.set_callout(
        "04 / 05  •  MEDICAL 3D VOLUME RAYMARCHING",
        "CT Scan Opacity Curve & Live Clip Plane",
        "R32Float raymarching, opacity TF curve & 3D clip plane",
        false,
    );

    studio.reset_scene_for_act(&med_path);

    // Switch Workspace Profile to Medical
    studio.click_label("Profile: CFD", Some(TOOLBAR_RECT), egui::pos2(248.0, 44.0), 12);
    studio.idle_frames(5);
    studio.click_label("Medical", None, egui::pos2(248.0, 108.0), 8);

    // Switch to Display tab and pick Representation -> Volume (ImageData opens as Outline)
    studio.click_label("Display", Some(INSPECTOR_TABS_RECT), egui::pos2(105.0, 195.0), 12);
    studio.step_frame(FrameEventBatch::default(), false);
    let init_vol_repr = if studio.find_label("Outline", Some(LEFT_DOCK_RECT)).is_some() {
        "Outline"
    } else {
        "Surface"
    };
    studio.click_label(init_vol_repr, Some(LEFT_DOCK_RECT), egui::pos2(140.0, 285.0), 10);
    studio.idle_frames(5);
    studio.click_label("Volume", None, egui::pos2(140.0, 405.0), 8);
    studio.idle_frames(18);

    // Scroll Left Dock down to reveal Volume Transfer Function controls cleanly
    studio.scroll_smooth(egui::pos2(175.0, 520.0), -330.0, 14);
    studio.idle_frames(4);

    // Focus zoom on Opacity Transfer Function presets & interactive histogram canvas
    studio.set_focus_zoom(1.22, [0.22, 0.56]);
    studio.click_label("Peak", Some(LEFT_DOCK_RECT), egui::pos2(220.0, 520.0), 12);
    studio.idle_frames(18);
    studio.click_label("Soft", Some(LEFT_DOCK_RECT), egui::pos2(130.0, 520.0), 10);
    studio.idle_frames(18);

    // Drag directly inside the Opacity TF Canvas (below "Peak") to demonstrate interactive curve editing
    if let Some(peak_pos) = studio.find_label("Peak", Some(LEFT_DOCK_RECT)) {
        let tf_pt = egui::pos2(peak_pos.x - 45.0, peak_pos.y + 68.0);
        studio.drag_smooth(tf_pt, egui::vec2(14.0, -24.0), 22);
    }

    // Enable 3D Volume Clip Plane and orbit volume
    studio.click_label("3D Volume Clip Plane", Some(LEFT_DOCK_RECT), egui::pos2(115.0, 465.0), 12);
    studio.set_focus_zoom(1.0, [0.5, 0.5]);
    studio.idle_frames(12);
    studio.drag_smooth(egui::pos2(860.0, 440.0), egui::vec2(175.0, -45.0), 42);

    // =========================================================================
    // ACT 5: XDMF Temporal Timeline Play + Protein .xyz + Theme Toggle + Outro
    // =========================================================================
    studio.mark_act("05_timeline");
    studio.set_callout(
        "05 / 05  •  TEMPORAL XDMF & MOLECULAR .XYZ",
        "16-Timestep Vortex Wake Playback",
        "On-demand 16-timestep XDMF playback & time badge",
        false,
    );

    studio.reset_scene_for_act(&xdmf_path);

    // Switch to Display -> SurfaceWithEdges so the Karman wake mesh shows crisp cell shading
    studio.click_label("Display", Some(INSPECTOR_TABS_RECT), egui::pos2(105.0, 195.0), 10);
    studio.idle_frames(5);
    studio.click_label("Surface", Some(LEFT_DOCK_RECT), egui::pos2(140.0, 285.0), 10);
    studio.idle_frames(5);
    studio.click_label("SurfaceWithEdges", None, egui::pos2(140.0, 315.0), 8);
    studio.idle_frames(6);

    // Enable Time annotation and click Play on the Time Transport bar
    studio.click_label("Time:", Some(TOOLBAR_RECT), egui::pos2(815.0, 44.0), 12);
    studio.click_label("Play", Some(TOOLBAR_RECT), egui::pos2(225.0, 74.0), 12);

    // Advance time deterministically across timesteps while keeping the flat wake broadside
    let steps = studio.app.session.all_time_steps();
    let total_play_frames = 84;
    for f in 0..total_play_frames {
        if !steps.is_empty() {
            let idx = (f / 5) % steps.len();
            studio.app.session.set_time(steps[idx]);
            studio.app.invalidate_viewport();
        }
        let dyaw = 0.0045 * (f as f32 * 0.075).cos();
        studio.app.panes[0].camera.orbit(dyaw, 0.0012);
        studio.step_frame(FrameEventBatch::default(), true);
    }
    studio.click_label("Pause", Some(TOOLBAR_RECT), egui::pos2(225.0, 74.0), 10);

    // Load Protein Helix + Ligand .xyz and switch to PointGaussian + Light UI toggle
    studio.reset_scene_for_act(&xyz_path);

    studio.set_callout(
        "05 / 05  •  MOLECULAR & THEME STUDIO",
        "Protein Helix (.xyz) & Instant Themes",
        "PointGaussian atomic splats with Light & Dark themes",
        false,
    );
    studio.click_label("Display", Some(INSPECTOR_TABS_RECT), egui::pos2(105.0, 195.0), 10);
    studio.step_frame(FrameEventBatch::default(), false);
    let repr_label = if studio.find_label("Points", Some(LEFT_DOCK_RECT)).is_some() {
        "Points"
    } else {
        "Surface"
    };
    studio.click_label(repr_label, Some(LEFT_DOCK_RECT), egui::pos2(140.0, 285.0), 10);
    studio.idle_frames(5);
    studio.click_label("PointGaussian", None, egui::pos2(140.0, 360.0), 8);
    if let Some(id) = studio.app.selected_node {
        if let Ok(node) = studio.app.session.node_mut(id) {
            node.style.point_size = 9.0;
        }
        studio.app.invalidate_viewport();
    }
    studio.idle_frames(10);

    studio.drag_smooth(egui::pos2(860.0, 440.0), egui::vec2(140.0, 30.0), 28);
    studio.click_label("Light UI", Some(TOP_MENU_RECT), egui::pos2(1490.0, 14.0), 14);
    studio.idle_frames(22);
    studio.click_label("Dark UI", Some(TOP_MENU_RECT), egui::pos2(1490.0, 14.0), 10);
    studio.idle_frames(18);

    // =========================================================================
    // OUTRO Overlay over live Studio
    // =========================================================================
    studio.mark_act("06_outro");
    studio.callout = None;
    for step in 1..=16 {
        let alpha = step as f32 / 16.0;
        studio.intro_outro_overlay = Some((
            "OPEN SOURCE  •  MIT / APACHE-2.0".into(),
            "Try ParaViz Live: dimitritimoz.github.io/paraviz".into(),
            "Native Desktop (Linux / macOS / Windows)  •  WebGPU Browser App  •  Rust & Python API".into(),
            alpha,
        ));
        studio.app.panes[0].camera.orbit(0.015, 0.0);
        studio.step_frame(FrameEventBatch::default(), true);
    }
    for _ in 0..64 {
        studio.app.panes[0].camera.orbit(0.015, 0.0);
        studio.step_frame(FrameEventBatch::default(), true);
    }

    let marks = studio.finish();
    let json_items: Vec<String> = marks
        .iter()
        .map(|m| {
            format!(
                r#"  {{"kind": "{}", "name": "{}", "time_sec": {:.3}}}"#,
                m.kind,
                m.name.replace('"', "\\\""),
                m.time_sec
            )
        })
        .collect();
    std::fs::write(&meta_json, format!("[\n{}\n]\n", json_items.join(",\n")))
        .expect("write showcase meta json");
    println!(
        "Generated studio showcase MP4: {} ({} marks -> {})",
        out_mp4.display(),
        marks.len(),
        meta_json.display()
    );
    if std::env::var_os("PARAVIZ_SHOWCASE_MP4").is_some() {
        let _ = std::fs::remove_dir_all(&out_dir);
    }
}
