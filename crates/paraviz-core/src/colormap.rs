//! Scientific colormaps (lookup tables) and opacity transfer functions for surface and volume rendering.

use serde::{Deserialize, Serialize};

/// Built-in ParaView / Matplotlib scientific colormap presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ColorMapPreset {
    #[default]
    CoolToWarm,
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Turbo,
    Fast,
    Rainbow,
    Grayscale,
    BlueRedYellow,
    Cividis,
}

impl ColorMapPreset {
    /// All available built-in presets.
    pub const ALL: &'static [Self] = &[
        Self::CoolToWarm,
        Self::Fast,
        Self::Viridis,
        Self::Plasma,
        Self::Inferno,
        Self::Magma,
        Self::Turbo,
        Self::Rainbow,
        Self::Grayscale,
        Self::BlueRedYellow,
        Self::Cividis,
    ];

    /// Display name matching ParaView's Color Map Editor.
    pub fn name(self) -> &'static str {
        match self {
            Self::CoolToWarm => "Cool to Warm",
            Self::Viridis => "Viridis",
            Self::Plasma => "Plasma",
            Self::Inferno => "Inferno",
            Self::Magma => "Magma",
            Self::Turbo => "Turbo",
            Self::Fast => "Fast (ParaView)",
            Self::Rainbow => "Rainbow (Jet)",
            Self::Grayscale => "Grayscale",
            Self::BlueRedYellow => "Black-Body Radiation",
            Self::Cividis => "Cividis",
        }
    }
}

/// An RGB control point on `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorStop {
    pub t: f32,
    pub rgb: [f32; 3],
}

/// An opacity control point on `[0.0, 1.0]` for volume rendering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpacityPoint {
    pub t: f32,
    pub alpha: f32,
}

/// Continuous RGB colormap with optional logarithmic mapping and out-of-range clamping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorMap {
    pub name: String,
    pub stops: Vec<ColorStop>,
    pub log_scale: bool,
    pub nan_color: [f32; 4],
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::from_preset(ColorMapPreset::CoolToWarm)
    }
}

impl ColorMap {
    /// Builds a [`ColorMap`] from a built-in [`ColorMapPreset`].
    pub fn from_preset(preset: ColorMapPreset) -> Self {
        let stops = match preset {
            ColorMapPreset::CoolToWarm => vec![
                ColorStop { t: 0.0, rgb: [0.230, 0.299, 0.754] },
                ColorStop { t: 0.25, rgb: [0.553, 0.690, 0.996] },
                ColorStop { t: 0.5, rgb: [0.865, 0.865, 0.865] },
                ColorStop { t: 0.75, rgb: [0.957, 0.604, 0.486] },
                ColorStop { t: 1.0, rgb: [0.706, 0.016, 0.150] },
            ],
            ColorMapPreset::Viridis => vec![
                ColorStop { t: 0.0, rgb: [0.267, 0.005, 0.329] },
                ColorStop { t: 0.25, rgb: [0.230, 0.322, 0.546] },
                ColorStop { t: 0.5, rgb: [0.128, 0.567, 0.551] },
                ColorStop { t: 0.75, rgb: [0.369, 0.789, 0.383] },
                ColorStop { t: 1.0, rgb: [0.993, 0.906, 0.144] },
            ],
            ColorMapPreset::Plasma => vec![
                ColorStop { t: 0.0, rgb: [0.050, 0.030, 0.528] },
                ColorStop { t: 0.25, rgb: [0.494, 0.012, 0.658] },
                ColorStop { t: 0.5, rgb: [0.798, 0.280, 0.470] },
                ColorStop { t: 0.75, rgb: [0.973, 0.581, 0.254] },
                ColorStop { t: 1.0, rgb: [0.940, 0.975, 0.131] },
            ],
            ColorMapPreset::Inferno => vec![
                ColorStop { t: 0.0, rgb: [0.001, 0.000, 0.014] },
                ColorStop { t: 0.25, rgb: [0.342, 0.062, 0.429] },
                ColorStop { t: 0.5, rgb: [0.735, 0.216, 0.330] },
                ColorStop { t: 0.75, rgb: [0.978, 0.558, 0.035] },
                ColorStop { t: 1.0, rgb: [0.988, 0.998, 0.645] },
            ],
            ColorMapPreset::Magma => vec![
                ColorStop { t: 0.0, rgb: [0.001, 0.000, 0.014] },
                ColorStop { t: 0.25, rgb: [0.316, 0.072, 0.485] },
                ColorStop { t: 0.5, rgb: [0.716, 0.215, 0.475] },
                ColorStop { t: 0.75, rgb: [0.987, 0.536, 0.382] },
                ColorStop { t: 1.0, rgb: [0.987, 0.991, 0.749] },
            ],
            ColorMapPreset::Turbo => vec![
                ColorStop { t: 0.0, rgb: [0.190, 0.072, 0.232] },
                ColorStop { t: 0.2, rgb: [0.276, 0.443, 0.918] },
                ColorStop { t: 0.4, rgb: [0.106, 0.812, 0.722] },
                ColorStop { t: 0.6, rgb: [0.639, 0.969, 0.247] },
                ColorStop { t: 0.8, rgb: [0.988, 0.588, 0.133] },
                ColorStop { t: 1.0, rgb: [0.478, 0.016, 0.012] },
            ],
            ColorMapPreset::Fast => vec![
                ColorStop { t: 0.0, rgb: [0.056, 0.056, 0.470] },
                ColorStop { t: 0.25, rgb: [0.180, 0.480, 0.750] },
                ColorStop { t: 0.5, rgb: [0.890, 0.910, 0.820] },
                ColorStop { t: 0.75, rgb: [0.890, 0.470, 0.200] },
                ColorStop { t: 1.0, rgb: [0.580, 0.060, 0.080] },
            ],
            ColorMapPreset::Rainbow => vec![
                ColorStop { t: 0.0, rgb: [0.0, 0.0, 0.56] },
                ColorStop { t: 0.2, rgb: [0.0, 0.35, 1.0] },
                ColorStop { t: 0.4, rgb: [0.0, 0.92, 0.85] },
                ColorStop { t: 0.6, rgb: [0.55, 1.0, 0.25] },
                ColorStop { t: 0.8, rgb: [1.0, 0.55, 0.0] },
                ColorStop { t: 1.0, rgb: [0.60, 0.0, 0.0] },
            ],
            ColorMapPreset::Grayscale => vec![
                ColorStop { t: 0.0, rgb: [0.05, 0.05, 0.05] },
                ColorStop { t: 1.0, rgb: [0.98, 0.98, 0.98] },
            ],
            ColorMapPreset::BlueRedYellow => vec![
                ColorStop { t: 0.0, rgb: [0.0, 0.0, 0.0] },
                ColorStop { t: 0.33, rgb: [0.90, 0.0, 0.0] },
                ColorStop { t: 0.67, rgb: [1.0, 0.85, 0.0] },
                ColorStop { t: 1.0, rgb: [1.0, 1.0, 1.0] },
            ],
            ColorMapPreset::Cividis => vec![
                ColorStop { t: 0.0, rgb: [0.0, 0.135, 0.305] },
                ColorStop { t: 0.25, rgb: [0.259, 0.306, 0.424] },
                ColorStop { t: 0.5, rgb: [0.490, 0.486, 0.471] },
                ColorStop { t: 0.75, rgb: [0.741, 0.686, 0.435] },
                ColorStop { t: 1.0, rgb: [0.996, 0.914, 0.220] },
            ],
        };
        Self {
            name: preset.name().to_string(),
            stops,
            log_scale: false,
            nan_color: [0.5, 0.5, 0.0, 1.0],
        }
    }

    /// Evaluates the colormap at normalized position `t` in `[0.0, 1.0]`.
    pub fn sample_rgb(&self, t: f32) -> [f32; 3] {
        if self.stops.is_empty() {
            return [1.0, 1.0, 1.0];
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.stops[0].t {
            return self.stops[0].rgb;
        }
        for win in self.stops.windows(2) {
            let a = win[0];
            let b = win[1];
            if t <= b.t {
                let denom = (b.t - a.t).max(1e-7);
                let f = (t - a.t) / denom;
                return [
                    a.rgb[0] + (b.rgb[0] - a.rgb[0]) * f,
                    a.rgb[1] + (b.rgb[1] - a.rgb[1]) * f,
                    a.rgb[2] + (b.rgb[2] - a.rgb[2]) * f,
                ];
            }
        }
        self.stops.last().unwrap().rgb
    }

    /// Maps a scalar value given `(min, max)` range to RGBA `[f32; 4]`.
    pub fn map_value(&self, value: f64, range: (f64, f64)) -> [f32; 4] {
        if !value.is_finite() {
            return self.nan_color;
        }
        let (min_v, max_v) = range;
        let t = if (max_v - min_v).abs() < 1e-15 {
            0.5
        } else if self.log_scale && min_v > 0.0 && max_v > 0.0 && value > 0.0 {
            ((value.ln() - min_v.ln()) / (max_v.ln() - min_v.ln())) as f32
        } else {
            ((value - min_v) / (max_v - min_v)) as f32
        };
        let rgb = self.sample_rgb(t);
        [rgb[0], rgb[1], rgb[2], 1.0]
    }

    /// Generates a 1D RGBA8 LUT texture (256 texels by default).
    pub fn to_rgba8_lut(&self, width: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(width * 4);
        for i in 0..width {
            let t = i as f32 / (width.saturating_sub(1).max(1)) as f32;
            let rgb = self.sample_rgb(t);
            bytes.push((rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push(255);
        }
        bytes
    }
}

/// Combined color + piecewise opacity transfer function for volume rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransferFunction {
    pub color_map: ColorMap,
    pub opacity_points: Vec<OpacityPoint>,
    pub scalar_range: (f64, f64),
    pub scalar_opacity_unit_distance: f32,
}

impl Default for TransferFunction {
    fn default() -> Self {
        Self {
            color_map: ColorMap::from_preset(ColorMapPreset::CoolToWarm),
            opacity_points: vec![
                OpacityPoint { t: 0.0, alpha: 0.0 },
                OpacityPoint { t: 0.3, alpha: 0.05 },
                OpacityPoint { t: 0.65, alpha: 0.45 },
                OpacityPoint { t: 1.0, alpha: 0.95 },
            ],
            scalar_range: (0.0, 1.0),
            scalar_opacity_unit_distance: 1.0,
        }
    }
}

impl TransferFunction {
    pub fn with_preset(preset: ColorMapPreset, scalar_range: (f64, f64)) -> Self {
        Self {
            color_map: ColorMap::from_preset(preset),
            scalar_range,
            ..Default::default()
        }
    }

    /// Evaluates piecewise-linear opacity at normalized `t` in `[0.0, 1.0]`.
    pub fn sample_opacity(&self, t: f32) -> f32 {
        if self.opacity_points.is_empty() {
            return 1.0;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.opacity_points[0].t {
            return self.opacity_points[0].alpha;
        }
        for win in self.opacity_points.windows(2) {
            let a = win[0];
            let b = win[1];
            if t <= b.t {
                let denom = (b.t - a.t).max(1e-7);
                let f = (t - a.t) / denom;
                return a.alpha + (b.alpha - a.alpha) * f;
            }
        }
        self.opacity_points.last().unwrap().alpha
    }

    /// Generates a 1D RGBA8 LUT texture combining RGB color and piecewise opacity.
    pub fn to_rgba8_lut(&self, width: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(width * 4);
        for i in 0..width {
            let t = i as f32 / (width.saturating_sub(1).max(1)) as f32;
            let rgb = self.color_map.sample_rgb(t);
            let a = self.sample_opacity(t);
            bytes.push((rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        bytes
    }
}
