use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_file(ext: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("paraviz_cli_test_{}_{}.{}", std::process::id(), id, ext))
}

fn paraviz_bin() -> &'static str {
    env!("CARGO_BIN_EXE_paraviz")
}

#[test]
fn cli_help_and_version() {
    let help = Command::new(paraviz_bin())
        .arg("--help")
        .output()
        .expect("failed to execute paraviz --help");
    assert!(help.status.success(), "stderr: {}", String::from_utf8_lossy(&help.stderr));
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(stdout.contains("USAGE:"));
    assert!(stdout.contains("paraviz demo"));
    assert!(stdout.contains("paraviz render"));
    assert!(stdout.contains("paraviz info"));
    assert!(stdout.contains("paraviz convert"));

    let ver = Command::new(paraviz_bin())
        .arg("--version")
        .output()
        .expect("failed to execute paraviz --version");
    assert!(ver.status.success());
    assert!(String::from_utf8_lossy(&ver.stdout).contains("paraviz"));
}

#[test]
fn cli_demo_wavelet_renders_png() {
    let out_png = temp_file("png");
    let output = Command::new(paraviz_bin())
        .args([
            "demo",
            "wavelet",
            "-o",
            out_png.to_str().unwrap(),
            "--width",
            "320",
            "--height",
            "240",
        ])
        .output()
        .expect("failed to execute paraviz demo wavelet");
    assert!(
        output.status.success(),
        "demo failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out_png).expect("demo png should exist");
    assert!(bytes.len() > 100, "png file unexpectedly small");
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n", "valid PNG signature");
    let _ = std::fs::remove_file(&out_png);
}

#[test]
fn cli_info_convert_and_render_roundtrip() {
    let vtp_path = temp_file("vtp");
    let stl_path = temp_file("stl");
    let png_path = temp_file("png");

    // Generate a sample PolyData via paraviz-sources + paraviz-io
    let sphere = paraviz::sources::SphereSource::default().build();
    let ds = paraviz::core::DataSet::PolyData(sphere);
    paraviz::io::write_dataset(&vtp_path, &ds).expect("write vtp");

    // 1. paraviz info
    let info = Command::new(paraviz_bin())
        .args(["info", vtp_path.to_str().unwrap()])
        .output()
        .expect("failed to run paraviz info");
    assert!(
        info.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&info.stderr)
    );
    let info_out = String::from_utf8_lossy(&info.stdout);
    assert!(info_out.contains("Type: Polygonal Mesh (PolyData)"));
    assert!(info_out.contains("Points:"));
    assert!(info_out.contains("Cells:"));
    assert!(info_out.contains("Elevation"));

    // 2. paraviz convert
    let convert = Command::new(paraviz_bin())
        .args([
            "convert",
            vtp_path.to_str().unwrap(),
            stl_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run paraviz convert");
    assert!(
        convert.status.success(),
        "convert failed: {}",
        String::from_utf8_lossy(&convert.stderr)
    );
    assert!(stl_path.exists());
    let roundtrip = paraviz::io::read_dataset(&stl_path).expect("read converted stl");
    assert!(roundtrip.num_cells() > 0);

    // 3. paraviz render
    let render = Command::new(paraviz_bin())
        .args([
            "render",
            vtp_path.to_str().unwrap(),
            "-o",
            png_path.to_str().unwrap(),
            "--width",
            "320",
            "--height",
            "240",
            "--repr",
            "edges",
            "--cmap",
            "viridis",
            "--color-by",
            "Elevation",
            "--camera",
            "+z",
        ])
        .output()
        .expect("failed to run paraviz render");
    assert!(
        render.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );
    let png_bytes = std::fs::read(&png_path).expect("render png should exist");
    assert_eq!(&png_bytes[0..8], b"\x89PNG\r\n\x1a\n");

    let _ = std::fs::remove_file(&vtp_path);
    let _ = std::fs::remove_file(&stl_path);
    let _ = std::fs::remove_file(&png_path);
}

#[test]
fn cli_demo_volume_and_error_handling() {
    let out_png = temp_file("png");
    let output = Command::new(paraviz_bin())
        .args([
            "demo",
            "volume",
            "-o",
            out_png.to_str().unwrap(),
            "--width",
            "256",
            "--height",
            "256",
        ])
        .output()
        .expect("failed to execute paraviz demo volume");
    assert!(
        output.status.success(),
        "demo volume failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out_png).expect("volume demo png should exist");
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");
    let _ = std::fs::remove_file(&out_png);

    let bad = Command::new(paraviz_bin())
        .args(["demo", "unknown_preset", "-o", "out.png"])
        .output()
        .expect("failed to run invalid demo");
    assert!(!bad.status.success(), "unknown demo preset should fail");
}
