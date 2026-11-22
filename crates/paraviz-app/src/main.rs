mod app;
mod cli;

use cli::{parse_args, run_headless_command, CliCommand};

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
                    .with_title("ParaViz — Scientific Visualization (WebGPU)")
                    .with_inner_size([1440.0, 900.0])
                    .with_drag_and_drop(true),
                renderer: eframe::Renderer::Wgpu,
                ..Default::default()
            };
            if let Err(err) = eframe::run_native(
                "ParaViz — Scientific Visualization (WebGPU)",
                options,
                Box::new(move |cc| Ok(Box::new(app::ParaVizApp::new(cc, initial_path)))),
            ) {
                eprintln!("GUI error: {err}");
                std::process::exit(1);
            }
        }
        other => {
            if let Err(err) = run_headless_command(other) {
                eprintln!("Error: {err}");
                std::process::exit(1);
            }
        }
    }
}
