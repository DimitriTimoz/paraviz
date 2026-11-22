use paraviz::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pv = DemoPreset::WaveletContour.build()?;
    let out_path = std::env::temp_dir().join("paraviz_wavelet_example.png");
    pv.render_png(&out_path, 800, 600)?;
    println!("Rendered Wavelet contour demo to {}", out_path.display());
    Ok(())
}
