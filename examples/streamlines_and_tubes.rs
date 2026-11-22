use paraviz::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pv = DemoPreset::VortexStreamlines.build()?;
    let out_path = std::env::temp_dir().join("paraviz_streamlines_example.png");
    pv.render_png(&out_path, 1024, 768)?;
    println!("Rendered Vortex streamlines + tubes + glyphs demo to {}", out_path.display());
    println!("\nGenerated Rust trace:\n{}", pv.to_rust_trace());
    Ok(())
}
