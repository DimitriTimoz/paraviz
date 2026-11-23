mod app;
mod charts;
mod cli;
mod modals;
mod theme;

use cli::{parse_args, run_headless_command, CliCommand};
use std::sync::Arc;

fn build_app_icon() -> egui::IconData {
    let size = 64u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 + 0.5) / size as f32;
            let ny = (y as f32 + 0.5) / size as f32;
            let dx = (nx - 0.5).abs() - 0.36;
            let dy = (ny - 0.5).abs() - 0.36;
            let dist = dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0) - 0.12;
            let alpha = ((-dist * 32.0).clamp(0.0, 1.0) * 255.0) as u8;

            // Indigo -> Cyan studio gradient
            let t = (nx * 0.65 + ny * 0.35).clamp(0.0, 1.0);
            let mut r = (79.0 * (1.0 - t) + 14.0 * t) as u8;
            let mut g = (70.0 * (1.0 - t) + 165.0 * t) as u8;
            let mut b = (229.0 * (1.0 - t) + 233.0 * t) as u8;

            // Crisp geometric diamond/prism highlight in center
            let cx = (nx - 0.5).abs();
            let cy = (ny - 0.5).abs();
            let diamond = cx + cy;
            if (0.20..=0.25).contains(&diamond) || (cx < 0.025 && ny > 0.5 && ny < 0.74) {
                r = 248;
                g = 251;
                b = 255;
            }

            let idx = ((y * size + x) * 4) as usize;
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = alpha;
        }
    }
    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}

fn main() {
    env_logger::init();

    let cmd = match parse_args(std::env::args()) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Error: {err}\n");
            eprintln!("{}", cli::help_text());
            std::process::exit(2);
        }
    };

    match cmd {
        CliCommand::Gui(initial_path) => {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("ParaViz — Scientific Visualization Studio (WebGPU)")
                    .with_inner_size([1480.0, 920.0])
                    .with_min_inner_size([960.0, 620.0])
                    .with_icon(Arc::new(build_app_icon()))
                    .with_drag_and_drop(true),
                renderer: eframe::Renderer::Wgpu,
                ..Default::default()
            };
            if let Err(err) = eframe::run_native(
                "ParaViz — Scientific Visualization Studio (WebGPU)",
                options,
                Box::new(move |cc| Ok(Box::new(app::ParaVizApp::new(cc, initial_path)))),
            ) {
                eprintln!("GUI error: {err}");
                eprintln!(
                    "Encountered an issue? Please report it at https://github.com/DimitriTimoz/paraviz/issues"
                );
                std::process::exit(1);
            }
        }
        other => {
            if let Err(err) = run_headless_command(other) {
                eprintln!("Error: {err}");
                eprintln!(
                    "Encountered an unexpected issue? Please open an issue at https://github.com/DimitriTimoz/paraviz/issues"
                );
                std::process::exit(1);
            }
        }
    }
}
