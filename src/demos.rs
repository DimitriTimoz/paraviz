//! Built-in scientific visualization demo pipelines showcasing ParaViz capabilities.

use crate::session::{FilterConfig, ParaViz, SessionError, SourceConfig};
use paraviz_core::{ColorMapPreset, OpacityPoint, TransferFunction};
use paraviz_filters::{
    ClipFilter, ContourFilter, GlyphFilter, GlyphShape, ImplicitFunction, IntegrationDirection,
    Integrator, ScaleMode, SeedSource, SliceFilter, StreamTracerFilter, TubeFilter,
};
use paraviz_render::{CameraPreset, Representation};
use paraviz_sources::{
    GyroidSource, SuperquadricSource, VectorFieldKind, VectorFieldSource, WaveletSource,
};
use serde::{Deserialize, Serialize};

/// Built-in demonstration scenes accessible via Rust API, CLI, and GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DemoPreset {
    WaveletContour,
    VortexStreamlines,
    GyroidMinimalSurface,
    VolumeWavelet,
    SuperquadricClip,
}

impl DemoPreset {
    pub const ALL: &'static [Self] = &[
        Self::WaveletContour,
        Self::VortexStreamlines,
        Self::GyroidMinimalSurface,
        Self::VolumeWavelet,
        Self::SuperquadricClip,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::WaveletContour => "Wavelet Isosurfaces + Slice",
            Self::VortexStreamlines => "Swirling Vortex Streamlines + Glyphs",
            Self::GyroidMinimalSurface => "Gyroid Minimal Surface",
            Self::VolumeWavelet => "3D Wavelet Volume Raymarching",
            Self::SuperquadricClip => "Clipped Superquadric Torus",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::WaveletContour => "wavelet",
            Self::VortexStreamlines => "vortex",
            Self::GyroidMinimalSurface => "gyroid",
            Self::VolumeWavelet => "volume",
            Self::SuperquadricClip => "superquadric",
        }
    }

    pub fn from_slug(s: impl AsRef<str>) -> Option<Self> {
        match s.as_ref().to_ascii_lowercase().as_str() {
            "wavelet" | "wavelet_contour" => Some(Self::WaveletContour),
            "vortex" | "streamlines" => Some(Self::VortexStreamlines),
            "gyroid" => Some(Self::GyroidMinimalSurface),
            "volume" | "volume_wavelet" => Some(Self::VolumeWavelet),
            "superquadric" | "clip" => Some(Self::SuperquadricClip),
            _ => None,
        }
    }

    /// Constructs the demo pipeline, fits the camera, and returns the ready-to-render [`ParaViz`] session.
    pub fn build(self) -> Result<ParaViz, SessionError> {
        let mut pv = ParaViz::new();
        match self {
            Self::WaveletContour => {
                let wav = pv.add_source_cfg(
                    "Wavelet",
                    SourceConfig::Wavelet(WaveletSource {
                        whole_extent: [-20, 20, -20, 20, -20, 20],
                        x_freq: 20.0,
                        y_freq: 14.0,
                        z_freq: 16.0,
                        std_dev: 0.48,
                        ..Default::default()
                    }),
                )?;
                pv.show(wav)
                    .representation(Representation::Outline)
                    .solid_color([0.7, 0.75, 0.85, 1.0]);

                let iso = pv.add_filter_cfg(
                    "Contour (148, 210)",
                    wav,
                    FilterConfig::Contour(ContourFilter::new("RTData", vec![148.0, 210.0])),
                )?;
                pv.show(iso)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::Turbo)
                    .color_by_point("RTData");

                let slice = pv.add_filter_cfg(
                    "Slice Z=0",
                    wav,
                    FilterConfig::Slice(SliceFilter::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])),
                )?;
                pv.show(slice)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::CoolToWarm)
                    .color_by_point("RTData")
                    .opacity(0.62);

                pv.settings.msaa_samples = 4;
                pv.settings.edl_enabled = true;
                pv.settings.edl_strength = 0.65;
            }
            Self::VortexStreamlines => {
                let vf = pv.add_source_cfg(
                    "SwirlingVortex",
                    SourceConfig::VectorField(VectorFieldSource {
                        dimensions: [32, 32, 32],
                        kind: VectorFieldKind::SwirlingVortex,
                        ..Default::default()
                    }),
                )?;
                pv.show(vf)
                    .representation(Representation::Outline)
                    .solid_color([0.65, 0.7, 0.8, 1.0]);

                let streams = pv.add_filter_cfg(
                    "StreamTracer (RK45)",
                    vf,
                    FilterConfig::StreamTracer(StreamTracerFilter {
                        vector_field: "Velocity".into(),
                        seeds: SeedSource::Sphere {
                            center: [0.0, 0.0, 0.0],
                            radius: 1.25,
                            count: 40,
                        },
                        direction: IntegrationDirection::Both,
                        integrator: Integrator::Rk45,
                        step_size: 0.045,
                        max_steps: 320,
                        max_length: 11.0,
                        terminal_speed: 1e-4,
                    }),
                )?;
                pv.hide(streams);

                let tubes = pv.add_filter_cfg(
                    "StreamTubes",
                    streams,
                    FilterConfig::Tube(TubeFilter::new(0.038, 16)),
                )?;
                pv.show(tubes)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::Plasma)
                    .color_by_point("Speed");

                let mid_slice = pv.add_filter_cfg(
                    "Midplane Z=0",
                    vf,
                    FilterConfig::Slice(SliceFilter::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])),
                )?;
                pv.show(mid_slice)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::Viridis)
                    .color_by_point("Speed")
                    .opacity(0.38);

                let glyphs = pv.add_filter_cfg(
                    "VelocityGlyphs",
                    mid_slice,
                    FilterConfig::Glyph(GlyphFilter {
                        shape: GlyphShape::Arrow,
                        vector_field: Some("Velocity".into()),
                        scalar_field: Some("Speed".into()),
                        scale_mode: ScaleMode::VectorMagnitude,
                        scale_factor: 0.24,
                        max_points: 160,
                    }),
                )?;
                pv.show(glyphs)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::Turbo)
                    .color_by_point("Speed");

                pv.settings.msaa_samples = 4;
                pv.settings.edl_enabled = true;
                pv.settings.edl_strength = 0.6;
            }
            Self::GyroidMinimalSurface => {
                let gyroid = pv.add_source_cfg(
                    "Gyroid",
                    SourceConfig::Gyroid(GyroidSource {
                        dimensions: [44, 44, 44],
                        ..Default::default()
                    }),
                )?;
                pv.hide(gyroid);

                let iso = pv.add_filter_cfg(
                    "Gyroid (iso=0)",
                    gyroid,
                    FilterConfig::Contour(ContourFilter::new("Gyroid", vec![0.0])),
                )?;
                pv.show(iso)
                    .representation(Representation::Surface)
                    .colormap(ColorMapPreset::CoolToWarm)
                    .color_by_point("Gyroid");

                pv.settings.msaa_samples = 4;
                pv.settings.edl_enabled = true;
                pv.settings.edl_strength = 0.6;
            }
            Self::VolumeWavelet => {
                let wav = pv.add_source_cfg("Wavelet", SourceConfig::Wavelet(WaveletSource::default()))?;
                let mut tf = TransferFunction::with_preset(ColorMapPreset::Inferno, (30.0, 265.0));
                tf.opacity_points = vec![
                    OpacityPoint { t: 0.0, alpha: 0.0 },
                    OpacityPoint { t: 0.25, alpha: 0.02 },
                    OpacityPoint { t: 0.55, alpha: 0.35 },
                    OpacityPoint { t: 0.85, alpha: 0.85 },
                    OpacityPoint { t: 1.0, alpha: 0.98 },
                ];
                pv.show(wav)
                    .representation(Representation::Volume)
                    .colormap(ColorMapPreset::Inferno)
                    .color_by_point("RTData")
                    .scalar_range(30.0, 265.0)
                    .volume_tf(tf)
                    .volume_step_size(0.35);
            }
            Self::SuperquadricClip => {
                let sq = pv.add_source_cfg(
                    "Superquadric",
                    SourceConfig::Superquadric(SuperquadricSource::default()),
                )?;
                pv.hide(sq);

                let clipped = pv.add_filter_cfg(
                    "Clip (Plane)",
                    sq,
                    FilterConfig::Clip(ClipFilter {
                        function: ImplicitFunction::Plane {
                            origin: [0.0, 0.0, 0.0],
                            normal: [1.0, 0.5, 0.2],
                        },
                        invert: false,
                        crinkle: false,
                    }),
                )?;
                pv.show(clipped)
                    .representation(Representation::SurfaceWithEdges)
                    .colormap(ColorMapPreset::Fast)
                    .color_by_point("Radial");
            }
        }
        pv.camera_preset(CameraPreset::Isometric)?;
        Ok(pv)
    }
}
