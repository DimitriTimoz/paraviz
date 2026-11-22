use paraviz::io::{read_dataset, write_dataset};
use paraviz::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum CliCommand {
    Gui(Option<PathBuf>),
    Demo {
        preset: DemoPreset,
        output: PathBuf,
        width: u32,
        height: u32,
    },
    Render {
        input: PathBuf,
        output: PathBuf,
        width: u32,
        height: u32,
        repr: Option<Representation>,
        cmap: Option<ColorMapPreset>,
        color_by: Option<String>,
        camera: Option<CameraPreset>,
    },
    Info {
        input: PathBuf,
    },
    Convert {
        input: PathBuf,
        output: PathBuf,
    },
    Help,
    Version,
}

pub fn help_text() -> &'static str {
    "\
ParaViz — Scientific Visualization (Rust + WebGPU)

USAGE:
    paraviz [file_or_state]
    paraviz gui [file_or_state]
    paraviz demo <wavelet|vortex|gyroid|volume|superquadric> -o <out.png> [--width W] [--height H]
    paraviz render <input> -o <out.png> [OPTIONS]
    paraviz info <input>
    paraviz convert <input> <output>
    paraviz --help | -h
    paraviz --version | -V

RENDER OPTIONS:
    -o, --output <path>       Output PNG path (required for demo/render)
    --width <px>              Output image width in pixels (default: 1280)
    --height <px>             Output image height in pixels (default: 720)
    --repr <mode>             surface | edges | wireframe | points | volume | outline
    --cmap <preset>           cool_to_warm | viridis | plasma | inferno | magma |
                              turbo | fast | rainbow | grayscale | cividis
    --color-by <array>        Point/cell array name (or 'solid')
    --camera <preset>         iso | +x | -x | +y | -y | +z | -z
"
}

pub fn parse_args<I, S>(args: I) -> Result<CliCommand, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let raw: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
    // Skip argv[0]
    let rest = if raw.len() > 1 { &raw[1..] } else { &[] };
    if rest.is_empty() {
        return Ok(CliCommand::Gui(None));
    }

    match rest[0].as_str() {
        "-h" | "--help" | "help" => Ok(CliCommand::Help),
        "-V" | "--version" | "version" => Ok(CliCommand::Version),
        "gui" => {
            if rest.len() > 2 {
                return Err(format!("Unexpected extra argument for gui: {}", rest[2]));
            }
            let path = rest.get(1).map(PathBuf::from);
            Ok(CliCommand::Gui(path))
        }
        "demo" => parse_demo_args(&rest[1..]),
        "render" => parse_render_args(&rest[1..]),
        "info" => {
            if rest.len() != 2 {
                return Err("Usage: paraviz info <input>".into());
            }
            Ok(CliCommand::Info {
                input: PathBuf::from(&rest[1]),
            })
        }
        "convert" => {
            if rest.len() != 3 {
                return Err("Usage: paraviz convert <input> <output>".into());
            }
            Ok(CliCommand::Convert {
                input: PathBuf::from(&rest[1]),
                output: PathBuf::from(&rest[2]),
            })
        }
        arg if arg.starts_with('-') => Err(format!("Unknown flag '{arg}'. Run with --help.")),
        file_arg => {
            if rest.len() > 1 {
                return Err(format!(
                    "Unexpected argument '{}'. Run with --help.",
                    rest[1]
                ));
            }
            Ok(CliCommand::Gui(Some(PathBuf::from(file_arg))))
        }
    }
}

fn parse_demo_args(args: &[String]) -> Result<CliCommand, String> {
    if args.is_empty() {
        return Err(
            "Missing demo preset. Expected one of: wavelet, vortex, gyroid, volume, superquadric"
                .into(),
        );
    }
    let preset = DemoPreset::from_slug(&args[0]).ok_or_else(|| {
        format!(
            "Unknown demo preset '{}'. Expected one of: wavelet, vortex, gyroid, volume, superquadric",
            args[0]
        )
    })?;

    let mut output: Option<PathBuf> = None;
    let mut width: u32 = 1280;
    let mut height: u32 = 720;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let val = args.get(i).ok_or("Missing path after -o/--output")?;
                output = Some(PathBuf::from(val));
            }
            "--width" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --width")?;
                width = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid width '{val}'"))?;
            }
            "--height" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --height")?;
                height = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid height '{val}'"))?;
            }
            other => return Err(format!("Unknown option '{other}' for demo subcommand")),
        }
        i += 1;
    }

    let output = output.ok_or("Missing required -o <out.png> for demo subcommand")?;
    if width == 0 || height == 0 {
        return Err("Width and height must be positive".into());
    }
    Ok(CliCommand::Demo {
        preset,
        output,
        width,
        height,
    })
}

fn parse_render_args(args: &[String]) -> Result<CliCommand, String> {
    if args.is_empty() {
        return Err("Usage: paraviz render <input> -o <out.png> [OPTIONS]".into());
    }
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut width: u32 = 1280;
    let mut height: u32 = 720;
    let mut repr: Option<Representation> = None;
    let mut cmap: Option<ColorMapPreset> = None;
    let mut color_by: Option<String> = None;
    let mut camera: Option<CameraPreset> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let val = args.get(i).ok_or("Missing path after -o/--output")?;
                output = Some(PathBuf::from(val));
            }
            "--width" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --width")?;
                width = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid width '{val}'"))?;
            }
            "--height" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --height")?;
                height = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid height '{val}'"))?;
            }
            "--repr" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --repr")?;
                repr = Some(parse_representation(val)?);
            }
            "--cmap" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --cmap")?;
                cmap = Some(parse_colormap(val)?);
            }
            "--color-by" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --color-by")?;
                color_by = Some(val.clone());
            }
            "--camera" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --camera")?;
                camera = Some(parse_camera_preset(val)?);
            }
            flag if flag.starts_with('-') => {
                return Err(format!("Unknown option '{flag}' for render subcommand"));
            }
            positional => {
                if input.is_none() {
                    input = Some(PathBuf::from(positional));
                } else {
                    return Err(format!("Unexpected positional argument '{positional}'"));
                }
            }
        }
        i += 1;
    }

    let input = input.ok_or("Missing <input> path for render subcommand")?;
    let output = output.ok_or("Missing required -o <out.png> for render subcommand")?;
    if width == 0 || height == 0 {
        return Err("Width and height must be positive".into());
    }

    Ok(CliCommand::Render {
        input,
        output,
        width,
        height,
        repr,
        cmap,
        color_by,
        camera,
    })
}

pub fn parse_representation(s: &str) -> Result<Representation, String> {
    match s.to_ascii_lowercase().as_str() {
        "surface" => Ok(Representation::Surface),
        "edges" | "surface_with_edges" | "surfacewithedges" => {
            Ok(Representation::SurfaceWithEdges)
        }
        "wireframe" | "wire" => Ok(Representation::Wireframe),
        "points" | "point" => Ok(Representation::Points),
        "volume" | "vol" => Ok(Representation::Volume),
        "outline" => Ok(Representation::Outline),
        _ => Err(format!(
            "Unknown representation '{s}'. Expected: surface, edges, wireframe, points, volume, outline"
        )),
    }
}

pub fn parse_colormap(s: &str) -> Result<ColorMapPreset, String> {
    match s.to_ascii_lowercase().as_str() {
        "cool_to_warm" | "cooltowarm" => Ok(ColorMapPreset::CoolToWarm),
        "viridis" => Ok(ColorMapPreset::Viridis),
        "plasma" => Ok(ColorMapPreset::Plasma),
        "inferno" => Ok(ColorMapPreset::Inferno),
        "magma" => Ok(ColorMapPreset::Magma),
        "turbo" => Ok(ColorMapPreset::Turbo),
        "fast" => Ok(ColorMapPreset::Fast),
        "rainbow" | "jet" => Ok(ColorMapPreset::Rainbow),
        "grayscale" | "gray" => Ok(ColorMapPreset::Grayscale),
        "black_body" | "blue_red_yellow" => Ok(ColorMapPreset::BlueRedYellow),
        "cividis" => Ok(ColorMapPreset::Cividis),
        _ => Err(format!(
            "Unknown colormap '{s}'. Expected: cool_to_warm, viridis, plasma, inferno, magma, turbo, fast, rainbow, grayscale, cividis"
        )),
    }
}

pub fn parse_camera_preset(s: &str) -> Result<CameraPreset, String> {
    match s.to_ascii_lowercase().as_str() {
        "iso" | "isometric" => Ok(CameraPreset::Isometric),
        "+x" | "posx" => Ok(CameraPreset::PosX),
        "-x" | "negx" => Ok(CameraPreset::NegX),
        "+y" | "posy" => Ok(CameraPreset::PosY),
        "-y" | "negy" => Ok(CameraPreset::NegY),
        "+z" | "posz" => Ok(CameraPreset::PosZ),
        "-z" | "negz" => Ok(CameraPreset::NegZ),
        _ => Err(format!(
            "Unknown camera preset '{s}'. Expected: iso, +x, -x, +y, -y, +z, -z"
        )),
    }
}

pub fn format_memory_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub fn format_dataset_info(ds: &DataSet, path: &Path) -> String {
    let mut out = String::new();
    let bb = ds.bounds();
    out.push_str(&format!("File: {}\n", path.display()));
    out.push_str(&format!("Type: {}\n", ds.type_name()));
    out.push_str(&format!("Points: {}\n", ds.num_points()));
    out.push_str(&format!("Cells: {}\n", ds.num_cells()));
    out.push_str(&format!(
        "Memory: {} ({} bytes)\n",
        format_memory_bytes(ds.memory_bytes()),
        ds.memory_bytes()
    ));
    if bb.is_valid() {
        out.push_str(&format!(
            "Bounds: X=[{:.6}, {:.6}] Y=[{:.6}, {:.6}] Z=[{:.6}, {:.6}]\n",
            bb.min.x, bb.max.x, bb.min.y, bb.max.y, bb.min.z, bb.max.z
        ));
    } else {
        out.push_str("Bounds: <empty>\n");
    }

    if let Some(pd) = ds.point_data() {
        out.push_str(&format!("Point Arrays ({}):\n", pd.len()));
        for arr in pd.arrays() {
            let (mn, mx) = arr.range(None);
            out.push_str(&format!(
                "  - {} [type={}, components={}, tuples={}, range=[{:.6}, {:.6}]]\n",
                arr.name,
                arr.data.scalar_type().vtk_name(),
                arr.num_components,
                arr.num_tuples(),
                mn,
                mx
            ));
        }
    }

    if let Some(cd) = ds.cell_data() {
        out.push_str(&format!("Cell Arrays ({}):\n", cd.len()));
        for arr in cd.arrays() {
            let (mn, mx) = arr.range(None);
            out.push_str(&format!(
                "  - {} [type={}, components={}, tuples={}, range=[{:.6}, {:.6}]]\n",
                arr.name,
                arr.data.scalar_type().vtk_name(),
                arr.num_components,
                arr.num_tuples(),
                mn,
                mx
            ));
        }
    }
    out
}

pub fn run_headless_command(cmd: CliCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        CliCommand::Help => {
            print!("{}", help_text());
            Ok(())
        }
        CliCommand::Version => {
            println!("paraviz {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        CliCommand::Demo {
            preset,
            output,
            width,
            height,
        } => {
            let mut pv = preset.build()?;
            pv.render_png(&output, width, height)?;
            println!(
                "Rendered demo '{}' ({}x{}) -> {}",
                preset.name(),
                width,
                height,
                output.display()
            );
            Ok(())
        }
        CliCommand::Render {
            input,
            output,
            width,
            height,
            repr,
            cmap,
            color_by,
            camera,
        } => {
            let ext = input
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();

            let mut pv = if ext == "json" {
                let json = std::fs::read_to_string(&input)?;
                ParaViz::load_state_json(&json)?
            } else {
                let mut session = ParaViz::new();
                let id = session.open(&input)?;
                let ds = session.evaluate(id)?;
                let handle = session.show(id);
                let handle = if let Some(r) = repr {
                    handle.representation(r)
                } else {
                    handle
                };
                let handle = if let Some(c) = cmap {
                    handle.colormap(c)
                } else {
                    handle
                };
                if let Some(arr) = color_by {
                    if arr.eq_ignore_ascii_case("solid") {
                        handle.solid_color([0.82, 0.84, 0.88, 1.0]);
                    } else if ds.point_data().and_then(|p| p.get(&arr)).is_some() {
                        handle.color_by_point(arr);
                    } else if ds.cell_data().and_then(|c| c.get(&arr)).is_some() {
                        handle.color_by_cell(arr);
                    } else {
                        handle.color_by_point(arr);
                    }
                }
                session.camera_preset(camera.unwrap_or(CameraPreset::Isometric))?;
                session
            };

            if ext == "json"
                && let Some(cam) = camera {
                    pv.camera_preset(cam)?;
                }

            pv.render_png(&output, width, height)?;
            println!(
                "Rendered '{}' ({}x{}) -> {}",
                input.display(),
                width,
                height,
                output.display()
            );
            Ok(())
        }
        CliCommand::Info { input } => {
            let ds = read_dataset(&input)?;
            print!("{}", format_dataset_info(&ds, &input));
            Ok(())
        }
        CliCommand::Convert { input, output } => {
            let ds = read_dataset(&input)?;
            write_dataset(&output, &ds)?;
            println!(
                "Converted '{}' ({}, {} points, {} cells) -> '{}'",
                input.display(),
                ds.type_name(),
                ds.num_points(),
                ds.num_cells(),
                output.display()
            );
            Ok(())
        }
        CliCommand::Gui(_) => Err("Gui command must be launched via eframe".into()),
    }
}
