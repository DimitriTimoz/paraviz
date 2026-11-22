use paraviz::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pv = DemoPreset::VolumeWavelet.build()?;
    let out_path = std::env::temp_dir().join("paraviz_volume_example.png");
    pv.render_png(&out_path, 800, 600)?;
    println!("Rendered 3D Wavelet volume raymarching demo to {}", out_path.display());
    Ok(())
}
