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
        edl: bool,
        edl_strength: Option<f32>,
        msaa: Option<u32>,
        axes_grid: bool,
        turntable: Option<u32>,
        filters: Vec<String>,
        transparent: bool,
        scalar_bar: bool,
        time_annotation: bool,
        gif_delay_ms: Option<u16>,
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
    paraviz convert <input> <output>   (supports .vtk, .vtp, .vti, .vtu, .stl, .obj, .ply, .csv, .glb, .html)
    paraviz --help | -h
    paraviz --version | -V

SUPPORTED INPUT FORMATS:
    VTK Legacy (.vtk), VTK XML (.vtp, .vti, .vtu, .vtr, .vts, .vtm, .pvd, .pvtp, .pvtu, .pvti),
    ParaView File Series (.series), VTKHDF (.vtkhdf), CGNS (.cgns), Exodus II (.exo, .e, .ex2),
    YALES2 HDF5 (.h5), XDMF (.xdmf, .xmf), EnSight Gold (.case), OpenFOAM (.foam), Gmsh (.msh),
    PLOT3D (.xyz, .p3d), Tecplot ASCII (.dat, .tec, .plt), Abaqus (.inp), NetCDF CF (.nc, .cdf),
    NIfTI (.nii, .nii.gz), NRRD (.nrrd, .nhdr), MetaImage (.mhd, .mha), DICOM (.dcm),
    Raw Volume (.raw), 2D Images (.png, .bmp, .ppm, .pgm, .pbm),
    Mesh (.stl, .obj, .ply, .off, .gltf, .glb, .x3d), LiDAR (.las),
    Molecular (.xyz, .pdb, .ent), CSV/TSV (.csv, .tsv)

RENDER OPTIONS:
    -o, --output <path>       Output PNG or animated GIF path (required for demo/render)
    --width <px>              Output image width in pixels (default: 1280)
    --height <px>             Output image height in pixels (default: 720)
    --repr <mode>             surface | edges | wireframe | points | gaussian | volume | outline
    --cmap <preset>           cool_to_warm | viridis | plasma | inferno | magma |
                              turbo | fast | rainbow | grayscale | cividis
    --color-by <array>        Point/cell array name (or 'solid')
    --camera <preset>         iso | +x | -x | +y | -y | +z | -z
    --edl                     Enable Eye-Dome Lighting post-process
    --edl-strength <val>      Eye-Dome Lighting strength (default: 1.0)
    --msaa <1|4>              Multisample anti-aliasing sample count (default: 1)
    --axes-grid               Show 3D Cube Axes Grid with tick marks
    --transparent             Transparent background (alpha=0) for PNG/GIF export
    --scalar-bar              Composite Scalar Color Bar with min/mid/max labels
    --time-annotation         Composite Time: annotation badge onto PNG/GIF output
    --turntable <frames>      Render a 360-degree turntable sequence (or animated .gif if -o ends in .gif)
    --gif-delay <ms>          Frame delay in milliseconds when exporting animated .gif (default: 60)
    --filter <spec>           Chain filter(s): contour:<v> | decimate:<r> | smooth |
                              triangulate | normals | edges | resample:<n> | isovolume:<lo>,<hi>

ISSUES & FEEDBACK:
    Encountering a bug or missing feature? Please open an issue at:
    https://github.com/DimitriTimoz/paraviz/issues
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
    let mut edl = false;
    let mut edl_strength: Option<f32> = None;
    let mut msaa: Option<u32> = None;
    let mut axes_grid = false;
    let mut turntable: Option<u32> = None;
    let mut filters: Vec<String> = Vec::new();
    let mut transparent = false;
    let mut scalar_bar = false;
    let mut time_annotation = false;
    let mut gif_delay_ms: Option<u16> = None;

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
            "--edl" => {
                edl = true;
            }
            "--edl-strength" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or("Missing value after --edl-strength")?;
                let s = val
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid EDL strength '{val}'"))?;
                if !s.is_finite() || s < 0.0 {
                    return Err(format!("EDL strength must be >= 0.0, got '{val}'"));
                }
                edl_strength = Some(s);
                edl = true;
            }
            "--msaa" => {
                i += 1;
                let val = args.get(i).ok_or("Missing value after --msaa")?;
                let samples = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid MSAA sample count '{val}'"))?;
                if samples != 1 && samples != 4 {
                    return Err(format!(
                        "Unsupported MSAA sample count '{val}'. Expected 1 or 4"
                    ));
                }
                msaa = Some(samples);
            }
            "--axes-grid" => {
                axes_grid = true;
            }
            "--transparent" => {
                transparent = true;
            }
            "--scalar-bar" => {
                scalar_bar = true;
            }
            "--time-annotation" => {
                time_annotation = true;
            }
            "--gif-delay" => {
                i += 1;
                let val = args.get(i).ok_or("Missing milliseconds after --gif-delay")?;
                let ms = val
                    .parse::<u16>()
                    .map_err(|_| format!("Invalid --gif-delay '{val}'"))?;
                gif_delay_ms = Some(ms.max(10));
            }
            "--turntable" => {
                i += 1;
                let val = args.get(i).ok_or("Missing frame count after --turntable")?;
                let frames = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid turntable frame count '{val}'"))?;
                if frames == 0 {
                    return Err("Turntable frame count must be >= 1".into());
                }
                turntable = Some(frames);
            }
            "--filter" => {
                i += 1;
                let val = args.get(i).ok_or("Missing filter spec after --filter")?;
                filters.push(val.clone());
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
        edl,
        edl_strength,
        msaa,
        axes_grid,
        turntable,
        filters,
        transparent,
        scalar_bar,
        time_annotation,
        gif_delay_ms,
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
        "gaussian" | "pointgaussian" | "point_gaussian" => Ok(Representation::PointGaussian),
        "volume" | "vol" => Ok(Representation::Volume),
        "outline" => Ok(Representation::Outline),
        _ => Err(format!(
            "Unknown representation '{s}'. Expected: surface, edges, wireframe, points, gaussian, volume, outline"
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
            edl,
            edl_strength,
            msaa,
            axes_grid,
            turntable,
            filters,
            transparent,
            scalar_bar,
            time_annotation,
            gif_delay_ms,
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
                let mut active_id = session.open(&input)?;
                for spec in &filters {
                    let cfg = parse_cli_filter(spec)?;
                    session.hide(active_id);
                    active_id = session.add_filter_cfg(cfg.default_label(), active_id, cfg)?;
                }
                let ds = session.evaluate(active_id)?;
                let handle = session.show(active_id);
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
                && let Some(cam) = camera
            {
                pv.camera_preset(cam)?;
            }

            if edl {
                pv.settings.edl_enabled = true;
            }
            if let Some(strength) = edl_strength {
                pv.settings.edl_strength = strength;
            }
            if let Some(samples) = msaa {
                pv.settings.msaa_samples = samples;
            }
            if axes_grid {
                pv.settings.show_axes_grid = true;
            }
            if transparent {
                pv.settings.transparent_background = true;
            }
            if scalar_bar {
                pv.settings.show_scalar_bar = true;
            }
            if time_annotation {
                pv.settings.show_time_annotation = true;
            }

            let is_gif = output
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gif"));

            if is_gif {
                if let Some(frames) = turntable {
                    let delay = gif_delay_ms.unwrap_or(60);
                    pv.render_turntable_gif(&output, width, height, frames, delay)?;
                    println!(
                        "Rendered {frames}-frame turntable GIF ({width}x{height}) -> {}",
                        output.display()
                    );
                } else {
                    let delay = gif_delay_ms.unwrap_or(100);
                    pv.render_timeseries_gif(&output, width, height, delay)?;
                    println!(
                        "Rendered animated GIF ({width}x{height}) -> {}",
                        output.display()
                    );
                }
            } else if let Some(frames) = turntable {
                let stem = output
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("frame");
                let parent = output.parent().unwrap_or_else(|| Path::new("."));
                let step_rad = std::f32::consts::TAU / (frames as f32);
                for f in 0..frames {
                    let frame_path = parent.join(format!("{stem}_{f:03}.png"));
                    pv.render_png(&frame_path, width, height)?;
                    pv.camera.orbit(step_rad, 0.0);
                }
                println!(
                    "Rendered {frames} turntable frames ({width}x{height}) -> {}/{stem}_*.png",
                    parent.display()
                );
            } else {
                pv.render_png(&output, width, height)?;
                println!(
                    "Rendered '{}' ({}x{}) -> {}",
                    input.display(),
                    width,
                    height,
                    output.display()
                );
            }
            Ok(())
        }
        CliCommand::Info { input } => {
            let ds = read_dataset(&input)?;
            print!("{}", format_dataset_info(&ds, &input));
            Ok(())
        }
        CliCommand::Convert { input, output } => {
            let ds = read_dataset(&input)?;
            let out_ext = output
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if out_ext == "glb" {
                let poly = match &ds {
                    DataSet::PolyData(p) => p.clone(),
                    other => paraviz::filters::extract_surface(other)?,
                };
                let scalar_name = poly
                    .point_data
                    .active_scalars_array()
                    .map(|a| a.name.clone());
                let cmap = scalar_name.as_ref().map(|_| ColorMap::default());
                paraviz::io::write_glb(&output, &poly, cmap.as_ref(), scalar_name.as_deref())?;
            } else if out_ext == "html" || out_ext == "htm" {
                let poly = match &ds {
                    DataSet::PolyData(p) => p.clone(),
                    other => paraviz::filters::extract_surface(other)?,
                };
                let scalar_name = poly
                    .point_data
                    .active_scalars_array()
                    .map(|a| a.name.clone());
                let cmap = scalar_name.as_ref().map(|_| ColorMap::default());
                let title = input
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ParaViz");
                paraviz::io::write_html_viewer(
                    &output,
                    &poly,
                    title,
                    cmap.as_ref(),
                    scalar_name.as_deref(),
                )?;
            } else {
                write_dataset(&output, &ds)?;
            }
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

fn parse_cli_filter(spec: &str) -> Result<FilterConfig, String> {
    let (kind, arg) = spec.split_once(':').unwrap_or((spec, ""));
    match kind.to_ascii_lowercase().as_str() {
        "contour" => {
            let vals: Vec<f64> = arg
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<f64>().map_err(|_| format!("Invalid isovalue '{s}'")))
                .collect::<Result<Vec<_>, _>>()?;
            if vals.is_empty() {
                return Err("contour filter requires at least one value, e.g. --filter contour:120".into());
            }
            Ok(FilterConfig::Contour(ContourFilter {
                iso_values: vals,
                ..Default::default()
            }))
        }
        "decimate" => {
            let r: f32 = if arg.is_empty() {
                0.5
            } else {
                arg.parse().map_err(|_| format!("Invalid decimate ratio '{arg}'"))?
            };
            Ok(FilterConfig::Decimate(DecimateFilter::new(r)))
        }
        "smooth" => Ok(FilterConfig::Smooth(SmoothFilter::default())),
        "triangulate" => Ok(FilterConfig::Triangulate(TriangulateFilter::default())),
        "normals" => Ok(FilterConfig::GenerateNormals(GenerateNormalsFilter::default())),
        "edges" => Ok(FilterConfig::ExtractEdges(ExtractEdgesFilter)),
        "surface" => Ok(FilterConfig::ExtractSurface(ExtractSurface::new())),
        "resample" => {
            let n: usize = if arg.is_empty() {
                32
            } else {
                arg.parse().map_err(|_| format!("Invalid resample dim '{arg}'"))?
            };
            Ok(FilterConfig::ResampleToImage(ResampleToImageFilter::new([n, n, n])))
        }
        "isovolume" => {
            let parts: Vec<f64> = arg
                .split(',')
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();
            if parts.len() != 2 {
                return Err("isovolume requires min,max e.g. --filter isovolume:50,150".into());
            }
            Ok(FilterConfig::IsoVolume(IsoVolumeFilter::new("", parts[0], parts[1])))
        }
        other => Err(format!("Unknown CLI filter '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_render_time_annotation_flag() {
        let cmd = parse_args([
            "paraviz",
            "render",
            "input.vtp",
            "-o",
            "out.png",
            "--time-annotation",
        ])
        .expect("parse render with --time-annotation");
        match cmd {
            CliCommand::Render {
                time_annotation,
                scalar_bar,
                ..
            } => {
                assert!(time_annotation);
                assert!(!scalar_bar);
            }
            other => panic!("unexpected command: {other:?}"),
        }

        let cmd_default = parse_args(["paraviz", "render", "input.vtp", "-o", "out.png"])
            .expect("parse render without --time-annotation");
        match cmd_default {
            CliCommand::Render {
                time_annotation, ..
            } => assert!(!time_annotation),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn help_text_lists_time_annotation_and_formats() {
        let txt = help_text();
        assert!(txt.contains("--time-annotation"));
        assert!(txt.contains(".vtkhdf"));
        assert!(txt.contains(".cgns"));
        assert!(txt.contains(".exo"));
    }
}
