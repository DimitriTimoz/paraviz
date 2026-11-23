mod app;
mod charts;
mod cli;
mod modals;
mod theme;

use cli::{parse_args, run_headless_command, CliCommand};
use std::sync::Arc;

fn build_app_icon() -> egui::IconData {
    let icon_bytes: &[u8] = include_bytes!("../../../assets/icon.png");
    let decoder = png::Decoder::new(std::io::Cursor::new(icon_bytes));
    let mut reader = decoder.read_info().expect("valid bundled icon.png");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(256 * 256 * 4)];
    let info = reader.next_frame(&mut buf).expect("valid icon.png frame");
    buf.truncate(info.buffer_size());
    egui::IconData {
        rgba: buf,
        width: info.width,
        height: info.height,
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
