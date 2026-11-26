use super::*;

#[derive(Default)]
pub(super) struct TemporalProbePanel {
    node: Option<NodeId>,
    position: [f32; 3],
    field: String,
    component: usize,
    result: Option<TemporalProbeResult>,
    #[cfg(not(target_arch = "wasm32"))]
    job: Option<mpsc::Receiver<Result<TemporalProbeResult, SessionError>>>,
}

impl ParaVizApp {
    pub(super) fn draw_temporal_probe(&mut self, ui: &mut egui::Ui) {
        let Some(node) = self.selected_node else {
            ui.weak("Select a pipeline node with time steps.");
            return;
        };
        let panel = &mut self.temporal_probe;
        if panel.node != Some(node) {
            *panel = TemporalProbePanel {
                node: Some(node),
                ..Default::default()
            };
            if let Some((_, request)) = self
                .session
                .temporal_probe
                .as_ref()
                .filter(|(id, _)| *id == node)
            {
                panel.position = request.position;
                panel.field.clone_from(&request.field);
                panel.component = request.component;
            } else if let Ok(entry) = self.session.node(node)
                && let Some(dataset) = entry.dataset()
            {
                let bounds = dataset.bounds();
                if bounds.is_valid() {
                    panel.position = bounds.center().to_array();
                }
                if let Some(fields) = dataset.point_data()
                    && let Some(array) = fields
                        .active_scalars_array()
                        .or_else(|| fields.arrays().first())
                {
                    panel.field.clone_from(&array.name);
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let busy = if let Some(job) = &panel.job {
            match job.try_recv() {
                Ok(result) => {
                    panel.job = None;
                    match result {
                        Ok(result) => panel.result = Some(result),
                        Err(error) => self.error_message = Some(error.to_string()),
                    }
                    false
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ui.ctx().request_repaint_after(Duration::from_millis(80));
                    true
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    panel.job = None;
                    self.error_message = Some("Temporal probe stopped unexpectedly".into());
                    false
                }
            }
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let busy = false;

        let pick = self
            .last_pick
            .as_ref()
            .filter(|pick| pick.actor_id == node.0)
            .map(|pick| pick.world_pos);
        let mut compute = false;
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (axis, value) in ["X", "Y", "Z"].into_iter().zip(&mut panel.position) {
                    ui.add(
                        egui::DragValue::new(value)
                            .speed(0.01)
                            .prefix(format!("{axis} ")),
                    );
                }
                if ui
                    .add_enabled(pick.is_some(), egui::Button::new("Use picked position"))
                    .clicked()
                    && let Some(position) = pick
                {
                    panel.position = position;
                }
                ui.label("Point field");
                ui.add(egui::TextEdit::singleline(&mut panel.field).desired_width(105.0));
                ui.add(egui::DragValue::new(&mut panel.component).prefix("Component "))
                    .on_hover_text("Zero-based component index");
                compute = ui
                    .add_enabled(!panel.field.trim().is_empty(), egui::Button::new("Compute"))
                    .clicked();
            });
        });
        if compute {
            let request = TemporalProbeRequest {
                position: panel.position,
                field: panel.field.trim().to_owned(),
                component: panel.component,
            };
            match self.session.prepare_temporal_probe(node, request.clone()) {
                Ok(probe) => {
                    self.session.temporal_probe = Some((node, request));
                    self.error_message = None;
                    panel.result = None;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let (tx, rx) = mpsc::channel();
                        panel.job = Some(rx);
                        let ctx = ui.ctx().clone();
                        std::thread::spawn(move || {
                            let _ = tx.send(probe.run());
                            ctx.request_repaint();
                        });
                        ui.ctx().request_repaint();
                    }
                    #[cfg(target_arch = "wasm32")]
                    match probe.run() {
                        Ok(result) => panel.result = Some(result),
                        Err(error) => self.error_message = Some(error.to_string()),
                    }
                }
                Err(error) => self.error_message = Some(error.to_string()),
            }
        }
        if busy {
            ui.label("Sampling time steps…");
        }
        if let Some(result) = &panel.result {
            let request = &result.request;
            let [x, y, z] = request.position;
            let mut missing = result
                .samples
                .iter()
                .filter(|sample| sample.value.is_none());
            let count = missing.clone().count();
            let mut export = false;
            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "{} [{}] at ({x:.4}, {y:.4}, {z:.4})",
                    request.field, request.component
                ));
                if count > 0 {
                    let mut times = missing
                        .by_ref()
                        .take(20)
                        .map(|sample| sample.time.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    if count > 20 {
                        times.push_str(", …");
                    }
                    ui.colored_label(palette(self.theme).warning, format!("{count} missing"))
                        .on_hover_text(format!("Missing samples at {times}"));
                }
                export = ui.button("Export CSV…").clicked();
            });
            crate::charts::paint_temporal_probe(ui, result, self.theme);
            if export {
                let csv = result.to_csv();
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = native_save_file(
                    "Export Temporal Probe",
                    &self.default_export_path("temporal_probe.csv"),
                    &[("CSV", &["csv"])],
                ) {
                    match std::fs::write(&path, csv) {
                        Ok(()) => {
                            self.remember_export_dir(&path);
                            self.error_message = None;
                            self.status_message =
                                format!("Exported temporal probe to {}", path.display());
                        }
                        Err(error) => {
                            self.error_message = Some(format!("CSV export failed: {error}"))
                        }
                    }
                }
                #[cfg(target_arch = "wasm32")]
                wasm_download_bytes("temporal_probe.csv", csv.as_bytes(), "text/csv");
            }
        }
    }
}
