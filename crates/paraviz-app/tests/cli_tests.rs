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

#[test]
fn cli_convert_to_glb_and_render_edl_msaa() {
    let vtp_path = temp_file("vtp");
    let vti_path = temp_file("vti");
    let glb_from_vtp = temp_file("glb");
    let glb_from_vti = temp_file("glb");
    let edl_png = temp_file("png");
    let msaa_png = temp_file("png");

    // 1. PolyData (.vtp) -> .glb conversion
    let sphere = paraviz::sources::SphereSource::default().build();
    paraviz::io::write_dataset(&vtp_path, &paraviz::core::DataSet::PolyData(sphere))
        .expect("write vtp");

    let conv_poly = Command::new(paraviz_bin())
        .args([
            "convert",
            vtp_path.to_str().unwrap(),
            glb_from_vtp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run paraviz convert to glb");
    assert!(
        conv_poly.status.success(),
        "convert vtp->glb failed: {}",
        String::from_utf8_lossy(&conv_poly.stderr)
    );
    let glb_bytes = std::fs::read(&glb_from_vtp).expect("glb file should exist");
    assert!(glb_bytes.len() > 20, "glb unexpectedly small");
    assert_eq!(&glb_bytes[0..4], b"glTF", "valid GLB magic header");

    // 2. ImageData (.vti) -> .glb conversion (auto surface extraction)
    let wavelet = paraviz::sources::WaveletSource {
        whole_extent: [-4, 4, -4, 4, -4, 4],
        ..Default::default()
    }
    .build(0.0);
    paraviz::io::write_dataset(&vti_path, &paraviz::core::DataSet::ImageData(wavelet))
        .expect("write vti");

    let conv_img = Command::new(paraviz_bin())
        .args([
            "convert",
            vti_path.to_str().unwrap(),
            glb_from_vti.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run paraviz convert vti->glb");
    assert!(
        conv_img.status.success(),
        "convert vti->glb failed: {}",
        String::from_utf8_lossy(&conv_img.stderr)
    );
    let glb_img_bytes = std::fs::read(&glb_from_vti).expect("vti glb file should exist");
    assert_eq!(&glb_img_bytes[0..4], b"glTF", "valid GLB magic header");

    // 3. paraviz render --edl --msaa 4
    let render_edl = Command::new(paraviz_bin())
        .args([
            "render",
            vtp_path.to_str().unwrap(),
            "-o",
            edl_png.to_str().unwrap(),
            "--width",
            "320",
            "--height",
            "240",
            "--edl",
            "--msaa",
            "4",
        ])
        .output()
        .expect("failed to run paraviz render --edl --msaa 4");
    assert!(
        render_edl.status.success(),
        "render --edl --msaa 4 failed: {}",
        String::from_utf8_lossy(&render_edl.stderr)
    );
    let edl_bytes = std::fs::read(&edl_png).expect("edl png should exist");
    assert_eq!(&edl_bytes[0..8], b"\x89PNG\r\n\x1a\n");

    // 4. paraviz render --msaa 4 (without EDL so 4x MSAA path is exercised directly)
    let render_msaa = Command::new(paraviz_bin())
        .args([
            "render",
            vtp_path.to_str().unwrap(),
            "-o",
            msaa_png.to_str().unwrap(),
            "--width",
            "320",
            "--height",
            "240",
            "--msaa",
            "4",
        ])
        .output()
        .expect("failed to run paraviz render --msaa 4");
    assert!(
        render_msaa.status.success(),
        "render --msaa 4 failed: {}",
        String::from_utf8_lossy(&render_msaa.stderr)
    );
    let msaa_bytes = std::fs::read(&msaa_png).expect("msaa png should exist");
    assert_eq!(&msaa_bytes[0..8], b"\x89PNG\r\n\x1a\n");

    // 5. Invalid --msaa value fails cleanly
    let bad_msaa = Command::new(paraviz_bin())
        .args([
            "render",
            vtp_path.to_str().unwrap(),
            "-o",
            msaa_png.to_str().unwrap(),
            "--msaa",
            "3",
        ])
        .output()
        .expect("failed to run invalid msaa");
    assert!(!bad_msaa.status.success(), "--msaa 3 should fail");

    let _ = std::fs::remove_file(&vtp_path);
    let _ = std::fs::remove_file(&vti_path);
    let _ = std::fs::remove_file(&glb_from_vtp);
    let _ = std::fs::remove_file(&glb_from_vti);
    let _ = std::fs::remove_file(&edl_png);
    let _ = std::fs::remove_file(&msaa_png);
}
