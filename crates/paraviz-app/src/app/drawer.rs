use super::*;

impl ParaVizApp {
    pub(super) fn draw_drawer(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::Spreadsheet, "Spreadsheet");
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::Chart, "Chart / Histogram");
            ui.selectable_value(&mut self.drawer_tab, DrawerTab::RustTrace, "Rust Trace");
            if self.drawer_tab == DrawerTab::Spreadsheet {
                ui.separator();
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Points,
                    "Point Data",
                );
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Cells,
                    "Cell Data",
                );
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Field,
                    "Field Data",
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Close").clicked() {
                    self.show_drawer = false;
                }
            });
        });
        ui.separator();

        match self.drawer_tab {
            DrawerTab::Spreadsheet => {
                ui.push_id("drawer_spreadsheet", |ui| {
                    self.draw_spreadsheet_view(ui, false);
                });
            }
            DrawerTab::Chart => {
                ui.push_id("drawer_chart", |ui| {
                    self.draw_chart_view(ui);
                });
            }
            DrawerTab::RustTrace => {
                let mut trace = self.session.to_rust_trace();
                ui.horizontal(|ui| {
                    if ui.button("Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(trace.clone());
                        self.status_message = "Copied Rust trace to clipboard".into();
                    }
                    if ui.button("Save trace.rs").clicked() {
                        match std::fs::write("paraviz_trace.rs", &trace) {
                            Ok(()) => {
                                self.error_message = None;
                                self.status_message = "Saved Rust trace -> paraviz_trace.rs".into();
                            }
                            Err(e) => {
                                self.error_message = Some(format!("Failed to write trace: {e}"));
                            }
                        }
                    }
                    ui.weak(format!("{} lines", trace.lines().count()));
                });
                egui::ScrollArea::vertical()
                    .id_salt("rust_trace_scroll")
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut trace)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY),
                        );
                    });
            }
        }
    }

    pub(super) fn draw_chart_view(&mut self, ui: &mut egui::Ui) {
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to view XY plots or scalar histograms.");
            return;
        };
        let _ = self.session.evaluate(sel_id);
        let Ok(node) = self.session.node(sel_id) else {
            return;
        };
        let Some(ds) = node.dataset().cloned() else {
            ui.weak("Dataset not evaluated.");
            return;
        };
        if let Some(res) = draw_chart_tab(ui, &ds, &mut self.chart_state, self.theme) {
            match res {
                Ok(msg) => {
                    self.error_message = None;
                    self.status_message = msg;
                }
                Err(err) => self.error_message = Some(err),
            }
        }
    }

    pub(super) fn draw_spreadsheet_view(&mut self, ui: &mut egui::Ui, show_target_tabs: bool) {
        if show_target_tabs {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Points,
                    "Point Data",
                );
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Cells,
                    "Cell Data",
                );
                ui.selectable_value(
                    &mut self.spreadsheet_target,
                    SpreadsheetTarget::Field,
                    "Field Data",
                );
            });
            ui.separator();
        }
        let Some(sel_id) = self.selected_node else {
            ui.weak("Select a node to inspect data values.");
            return;
        };
        let Ok(node) = self.session.node(sel_id) else {
            return;
        };
        let node_name = node.name.clone();
        let Some(ds) = node.dataset().cloned() else {
            ui.weak("Dataset not evaluated.");
            return;
        };

        let total_rows = match self.spreadsheet_target {
            SpreadsheetTarget::Points => ds.num_points(),
            SpreadsheetTarget::Cells => ds.num_cells(),
            SpreadsheetTarget::Field => ds
                .field_data()
                .map(|f| f.arrays().iter().map(|a| a.num_tuples()).max().unwrap_or(0))
                .unwrap_or(0),
        };
        let page = self.spreadsheet_page_size.max(1);
        let max_offset = total_rows.saturating_sub(1);
        if self.spreadsheet_offset > max_offset {
            self.spreadsheet_offset = 0;
        }
        let end_row = (self.spreadsheet_offset + page).min(total_rows);

        let mut pending_highlight: Option<(PickDomain, usize, bool)> = None;
        let mut focus_world_pos: Option<[f32; 3]> = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Rows {}..{} of {}",
                if total_rows == 0 { 0 } else { self.spreadsheet_offset },
                end_row,
                total_rows
            ));
            if ui
                .add_enabled(self.spreadsheet_offset > 0, egui::Button::new("|<").small())
                .clicked()
            {
                self.spreadsheet_offset = 0;
            }
            if ui
                .add_enabled(self.spreadsheet_offset > 0, egui::Button::new("Prev").small())
                .clicked()
            {
                self.spreadsheet_offset = self.spreadsheet_offset.saturating_sub(page);
            }
            if ui
                .add_enabled(end_row < total_rows, egui::Button::new("Next").small())
                .clicked()
            {
                self.spreadsheet_offset = end_row;
            }
            egui::ComboBox::from_id_salt("spreadsheet_page_size")
                .selected_text(format!("{page}/page"))
                .show_ui(ui, |ui| {
                    for sz in [100usize, 200, 500] {
                        ui.selectable_value(&mut self.spreadsheet_page_size, sz, format!("{sz}/page"));
                    }
                });
            ui.separator();
            ui.label("Col Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.spreadsheet_filter)
                    .desired_width(120.0)
                    .hint_text("array name..."),
            );
            if ui.small_button("Export CSV").clicked() {
                let domain_slug = match self.spreadsheet_target {
                    SpreadsheetTarget::Points => "points",
                    SpreadsheetTarget::Cells => "cells",
                    SpreadsheetTarget::Field => "field",
                };
                let stem: String = node_name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let stem = stem.trim_matches('_');
                let fname = if stem.is_empty() {
                    format!("spreadsheet_{domain_slug}.csv")
                } else {
                    format!("{stem}_{domain_slug}.csv")
                };
                let default_path = self.default_export_path(&fname);
                if let Some(p) = native_save_file(
                    "Export Spreadsheet CSV",
                    &default_path,
                    &[("CSV Table", &["csv"])],
                ) {
                    self.execute_modal_action(ModalAction::ExportSpreadsheetCsv(p));
                }
            }
            if let Some(hl) = &self.picked_highlight
                && hl.node_id == sel_id
            {
                ui.separator();
                let tag = match hl.domain {
                    PickDomain::Point => format!("Highlighted Point #{}", hl.element_index),
                    PickDomain::Cell => format!("Highlighted Cell #{}", hl.element_index),
                };
                ui.label(egui::RichText::new(tag).strong().color(palette(self.theme).warning));
                if let Some(pos) = hl.world_pos
                    && ui.small_button("Focus").clicked()
                {
                    focus_world_pos = Some(pos);
                }
            }
        });
        ui.separator();

        let col_needle = self.spreadsheet_filter.trim().to_ascii_lowercase();
        let matches_col = |name: &str| -> bool {
            col_needle.is_empty() || name.to_ascii_lowercase().contains(&col_needle)
        };

        let mut scroll = egui::ScrollArea::both().id_salt("spreadsheet_scroll");
        if self.spreadsheet_scroll_to_top {
            scroll = scroll.vertical_scroll_offset(0.0);
            self.spreadsheet_scroll_to_top = false;
        }

        let active_hl = self
            .picked_highlight
            .as_ref()
            .filter(|hl| hl.node_id == sel_id)
            .map(|hl| (hl.domain, hl.element_index));

        scroll.show(ui, |ui| match self.spreadsheet_target {
            SpreadsheetTarget::Points => {
                let arrays: Vec<&DataArray> = ds
                    .point_data()
                    .map(|p| {
                        p.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                egui::Grid::new("spreadsheet_points_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Point ID");
                        ui.strong("X");
                        ui.strong("Y");
                        ui.strong("Z");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();

                        for row in self.spreadsheet_offset..end_row {
                            let selected = active_hl == Some((PickDomain::Point, row));
                            let id_resp = ui
                                .selectable_label(selected, format!("{row}"))
                                .on_hover_text("Click to highlight in 3D • Double-click to focus");
                            if id_resp.double_clicked() {
                                pending_highlight = Some((PickDomain::Point, row, true));
                            } else if id_resp.clicked() {
                                pending_highlight = Some((PickDomain::Point, row, false));
                            }
                            if let Some(p) = dataset_point_at(&ds, row) {
                                ui.monospace(format!("{:.4}", p[0]));
                                ui.monospace(format!("{:.4}", p[1]));
                                ui.monospace(format!("{:.4}", p[2]));
                            } else {
                                ui.monospace("-");
                                ui.monospace("-");
                                ui.monospace("-");
                            }
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
            SpreadsheetTarget::Cells => {
                let arrays: Vec<&DataArray> = ds
                    .cell_data()
                    .map(|c| {
                        c.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                if total_rows == 0 {
                    ui.weak("Selected dataset has no cells.");
                    return;
                }
                if arrays.is_empty() && !col_needle.is_empty() {
                    ui.weak("No CellData arrays match the column filter.");
                    return;
                }
                egui::Grid::new("spreadsheet_cells_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Cell ID");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();
                        for row in self.spreadsheet_offset..end_row {
                            let selected = active_hl == Some((PickDomain::Cell, row));
                            let id_resp = ui
                                .selectable_label(selected, format!("{row}"))
                                .on_hover_text("Click to highlight in 3D • Double-click to focus");
                            if id_resp.double_clicked() {
                                pending_highlight = Some((PickDomain::Cell, row, true));
                            } else if id_resp.clicked() {
                                pending_highlight = Some((PickDomain::Cell, row, false));
                            }
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
            SpreadsheetTarget::Field => {
                let arrays: Vec<&DataArray> = ds
                    .field_data()
                    .map(|f| {
                        f.arrays()
                            .iter()
                            .filter(|a| matches_col(&a.name))
                            .collect()
                    })
                    .unwrap_or_default();
                if arrays.is_empty() {
                    ui.weak("Selected dataset has no matching FieldData arrays.");
                    return;
                }
                egui::Grid::new("spreadsheet_field_grid")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.strong("Tuple");
                        for arr in &arrays {
                            ui.strong(&arr.name);
                        }
                        ui.end_row();
                        for row in self.spreadsheet_offset..end_row {
                            ui.monospace(format!("{row}"));
                            for arr in &arrays {
                                ui.monospace(format_tuple(arr, row));
                            }
                            ui.end_row();
                        }
                    });
            }
        });

        if let Some((domain, row, also_focus)) = pending_highlight {
            self.highlight_from_spreadsheet_row(sel_id, &ds, domain, row);
            if also_focus
                && let Some(pos) = self.picked_highlight.as_ref().and_then(|h| h.world_pos)
            {
                self.focus_on_world_pos(pos);
            }
        } else if let Some(pos) = focus_world_pos {
            self.focus_on_world_pos(pos);
        }
    }

}

pub(super) fn polydata_surface_area(pd: &PolyData) -> f64 {
    let mut area = 0.0f64;
    let pts = &pd.points;
    for poly in &pd.polys {
        if poly.len() < 3 {
            continue;
        }
        let Some(&p0) = pts.get(poly[0] as usize) else {
            continue;
        };
        let v0 = glam::Vec3::from_array(p0);
        for k in 1..(poly.len() - 1) {
            if let (Some(&p1), Some(&p2)) =
                (pts.get(poly[k] as usize), pts.get(poly[k + 1] as usize))
            {
                let e1 = glam::Vec3::from_array(p1) - v0;
                let e2 = glam::Vec3::from_array(p2) - v0;
                area += 0.5 * e1.cross(e2).length() as f64;
            }
        }
    }
    for strip in &pd.strips {
        for k in 0..strip.len().saturating_sub(2) {
            if let (Some(&p0), Some(&p1), Some(&p2)) = (
                pts.get(strip[k] as usize),
                pts.get(strip[k + 1] as usize),
                pts.get(strip[k + 2] as usize),
            ) {
                let v0 = glam::Vec3::from_array(p0);
                let e1 = glam::Vec3::from_array(p1) - v0;
                let e2 = glam::Vec3::from_array(p2) - v0;
                area += 0.5 * e1.cross(e2).length() as f64;
            }
        }
    }
    area
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn append_array_headers(csv: &mut String, arrays: &[&DataArray]) {
    for arr in arrays {
        if arr.num_components == 1 {
            csv.push(',');
            csv.push_str(&csv_escape(&arr.name));
        } else {
            for c in 0..arr.num_components {
                csv.push(',');
                csv.push_str(&csv_escape(&format!("{}_{}", arr.name, c)));
            }
        }
    }
    csv.push('\n');
}

fn append_array_row(csv: &mut String, arrays: &[&DataArray], row: usize) {
    for arr in arrays {
        if row < arr.num_tuples() {
            for v in arr.get_tuple(row) {
                csv.push_str(&format!(",{v}"));
            }
        } else {
            for _ in 0..arr.num_components {
                csv.push_str(",0");
            }
        }
    }
    csv.push('\n');
}

pub(super) fn export_spreadsheet_csv(
    ds: &DataSet,
    target: SpreadsheetTarget,
    col_filter: &str,
    path: &Path,
) -> Result<String, String> {
    let col_needle = col_filter.trim().to_ascii_lowercase();
    let matches_col = |name: &str| -> bool {
        col_needle.is_empty() || name.to_ascii_lowercase().contains(&col_needle)
    };

    let mut csv = String::new();
    let (domain_label, n) = match target {
        SpreadsheetTarget::Points => {
            let arrays: Vec<&DataArray> = ds
                .point_data()
                .map(|p| p.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            let n = ds.num_points();
            csv.push_str("PointID,X,Y,Z");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                if let Some(p) = dataset_point_at(ds, row) {
                    csv.push_str(&format!("{row},{},{},{}", p[0], p[1], p[2]));
                } else {
                    csv.push_str(&format!("{row},0,0,0"));
                }
                append_array_row(&mut csv, &arrays, row);
            }
            ("Point", n)
        }
        SpreadsheetTarget::Cells => {
            let arrays: Vec<&DataArray> = ds
                .cell_data()
                .map(|c| c.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            let n = ds.num_cells();
            if n == 0 {
                return Err("Selected dataset has no cells to export.".into());
            }
            if !col_needle.is_empty() && arrays.is_empty() {
                return Err(format!(
                    "No CellData columns match filter '{}'.",
                    col_filter.trim()
                ));
            }
            csv.push_str("CellID");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                csv.push_str(&format!("{row}"));
                append_array_row(&mut csv, &arrays, row);
            }
            ("Cell", n)
        }
        SpreadsheetTarget::Field => {
            let arrays: Vec<&DataArray> = ds
                .field_data()
                .map(|f| f.arrays().iter().filter(|a| matches_col(&a.name)).collect())
                .unwrap_or_default();
            if arrays.is_empty() {
                return Err(if col_needle.is_empty() {
                    "Selected dataset has no FieldData arrays to export.".into()
                } else {
                    format!(
                        "No FieldData columns match filter '{}'.",
                        col_filter.trim()
                    )
                });
            }
            let n = arrays.iter().map(|a| a.num_tuples()).max().unwrap_or(0);
            csv.push_str("Tuple");
            append_array_headers(&mut csv, &arrays);
            for row in 0..n {
                csv.push_str(&format!("{row}"));
                append_array_row(&mut csv, &arrays, row);
            }
            ("Field", n)
        }
    };

    std::fs::write(path, csv).map_err(|e| format!("Failed to export CSV: {e}"))?;
    Ok(format!("Exported {n} {domain_label} rows -> {}", path.display()))
}


pub(super) fn dataset_point_at(ds: &DataSet, row: usize) -> Option<[f32; 3]> {
    match ds {
        DataSet::PolyData(p) => p.points.get(row).copied(),
        DataSet::ImageData(i) => {
            if row >= i.num_points() {
                return None;
            }
            let [nx, ny, _] = i.dimensions;
            let k = row / (nx * ny).max(1);
            let rem = row % (nx * ny).max(1);
            let j = rem / nx.max(1);
            let ii = rem % nx.max(1);
            Some(i.point_coords(ii, j, k))
        }
        DataSet::RectilinearGrid(r) => {
            if row >= r.num_points() {
                return None;
            }
            let [nx, ny, _] = r.dimensions();
            let k = row / (nx * ny).max(1);
            let rem = row % (nx * ny).max(1);
            let j = rem / nx.max(1);
            let ii = rem % nx.max(1);
            Some(r.point_coords(ii, j, k))
        }
        DataSet::StructuredGrid(s) => s.points.get(row).copied(),
        DataSet::UnstructuredGrid(u) => u.points.get(row).copied(),
        DataSet::MultiBlock(mb) => mb
            .blocks
            .first()
            .and_then(|(_, b)| dataset_point_at(b, row)),
    }
}

fn centroid_from_indices(points: &[[f32; 3]], indices: &[u32]) -> Option<[f32; 3]> {
    if indices.is_empty() {
        return None;
    }
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0usize;
    for &idx in indices {
        if let Some(&p) = points.get(idx as usize) {
            sum += glam::Vec3::from_array(p);
            count += 1;
        }
    }
    (count > 0).then(|| (sum / count as f32).to_array())
}

fn structured_cell_center(
    dims: [usize; 3],
    row: usize,
    coords_fn: impl Fn(usize, usize, usize) -> [f32; 3],
) -> Option<[f32; 3]> {
    let cx = dims[0].saturating_sub(1).max(1);
    let cy = dims[1].saturating_sub(1).max(1);
    let cz = dims[2].saturating_sub(1).max(1);
    if dims.contains(&0) || row >= cx * cy * cz {
        return None;
    }
    let k = row / (cx * cy);
    let rem = row % (cx * cy);
    let j = rem / cx;
    let i = rem % cx;
    let i1 = (i + 1).min(dims[0] - 1);
    let j1 = (j + 1).min(dims[1] - 1);
    let k1 = (k + 1).min(dims[2] - 1);
    let p0 = glam::Vec3::from_array(coords_fn(i, j, k));
    let p1 = glam::Vec3::from_array(coords_fn(i1, j1, k1));
    Some(((p0 + p1) * 0.5).to_array())
}

pub(super) fn dataset_cell_center(ds: &DataSet, row: usize) -> Option<[f32; 3]> {
    match ds {
        DataSet::PolyData(p) => {
            if let Some(v) = p.verts.get(row) {
                return centroid_from_indices(&p.points, std::slice::from_ref(v));
            }
            let r1 = row.saturating_sub(p.verts.len());
            if row >= p.verts.len() && r1 < p.lines.len() {
                return centroid_from_indices(&p.points, &p.lines[r1]);
            }
            let r2 = r1.saturating_sub(p.lines.len());
            if row >= p.verts.len() + p.lines.len() && r2 < p.polys.len() {
                return centroid_from_indices(&p.points, &p.polys[r2]);
            }
            let r3 = r2.saturating_sub(p.polys.len());
            if row >= p.verts.len() + p.lines.len() + p.polys.len() && r3 < p.strips.len() {
                return centroid_from_indices(&p.points, &p.strips[r3]);
            }
            None
        }
        DataSet::ImageData(i) => structured_cell_center(i.dimensions, row, |ii, jj, kk| {
            i.point_coords(ii, jj, kk)
        }),
        DataSet::RectilinearGrid(r) => {
            structured_cell_center(r.dimensions(), row, |ii, jj, kk| r.point_coords(ii, jj, kk))
        }
        DataSet::StructuredGrid(s) => {
            structured_cell_center(s.dimensions, row, |ii, jj, kk| s.point_coords(ii, jj, kk))
        }
        DataSet::UnstructuredGrid(u) => {
            if row >= u.num_cells() {
                return None;
            }
            centroid_from_indices(&u.points, u.cell_points(row))
        }
        DataSet::MultiBlock(mb) => mb
            .blocks
            .first()
            .and_then(|(_, b)| dataset_cell_center(b, row)),
    }
}

pub(super) fn dataset_element_scalar(
    ds: &DataSet,
    color_by: &ColorBy,
    domain: PickDomain,
    row: usize,
) -> Option<f64> {
    let fd = match domain {
        PickDomain::Point => ds.point_data()?,
        PickDomain::Cell => ds.cell_data()?,
    };
    let (arr, comp) = match (domain, color_by) {
        (PickDomain::Point, ColorBy::PointScalar { array, component })
        | (PickDomain::Cell, ColorBy::CellScalar { array, component }) => {
            let a = if array.is_empty() {
                fd.active_scalars_array()?
            } else {
                fd.get(array)?
            };
            (a, *component)
        }
        _ => (fd.active_scalars_array()?, None),
    };
    if row >= arr.num_tuples() {
        return None;
    }
    Some(match comp {
        Some(c) if c < arr.num_components => arr.data.get_f64(row * arr.num_components + c),
        _ => arr.get_magnitude(row),
    })
}

pub(super) fn format_tuple(arr: &DataArray, row: usize) -> String {
    if row >= arr.num_tuples() {
        return "-".into();
    }
    let vals = arr.get_tuple(row);
    if vals.len() == 1 {
        format!("{:.4}", vals[0])
    } else {
        let parts: Vec<String> = vals.iter().map(|v| format!("{v:.3}")).collect();
        format!("({})", parts.join(", "))
    }
}
