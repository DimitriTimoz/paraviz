//! Scene graph, actor representations, scalar color-mapping styles, and viewport settings.

use crate::camera::Camera;
use paraviz_core::{BoundingBox, ColorMap, DataSet, TransferFunction};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Geometric / volumetric visual representation of an actor (matching ParaView's Representation dropdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Representation {
    #[default]
    Surface,
    SurfaceWithEdges,
    Wireframe,
    Points,
    PointGaussian,
    Volume,
    Outline,
}

/// How an actor's geometry is colored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColorBy {
    Solid([f32; 4]),
    PointScalar {
        array: String,
        component: Option<usize>,
    },
    CellScalar {
        array: String,
        component: Option<usize>,
    },
}

impl Default for ColorBy {
    fn default() -> Self {
        Self::Solid([0.82, 0.84, 0.88, 1.0])
    }
}

/// Visual appearance and shading properties for a [`SceneActor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorStyle {
    pub visible: bool,
    pub representation: Representation,
    pub color_by: ColorBy,
    pub colormap: ColorMap,
    pub scalar_range: (f64, f64),
    pub auto_range: bool,
    pub opacity: f32,
    pub point_size: f32,
    pub line_width: f32,
    pub edge_color: [f32; 4],
    pub edge_width: f32,
    pub specular: f32,
    pub shininess: f32,
    pub ambient: f32,
    pub two_sided: bool,
    pub volume_tf: TransferFunction,
    pub volume_step_size: f32,
    pub volume_shading: bool,
    /// Optional live 3D clipping plane `(origin, normal)` applied during volume raymarching.
    #[serde(default)]
    pub volume_clip_plane: Option<([f32; 3], [f32; 3])>,
    /// Hidden MultiBlock block names or slash-joined paths (e.g. `"wall"` or `"Base/per1"`).
    #[serde(default)]
    pub hidden_blocks: BTreeSet<String>,
    /// Per-block solid RGBA overrides keyed by block name or slash-joined path.
    #[serde(default)]
    pub block_colors: BTreeMap<String, [f32; 4]>,
}

impl Default for ActorStyle {
    fn default() -> Self {
        Self {
            visible: true,
            representation: Representation::Surface,
            color_by: ColorBy::default(),
            colormap: ColorMap::default(),
            scalar_range: (0.0, 1.0),
            auto_range: true,
            opacity: 1.0,
            point_size: 6.0,
            line_width: 1.5,
            edge_color: [0.08, 0.08, 0.10, 1.0],
            edge_width: 1.25,
            specular: 0.25,
            shininess: 32.0,
            ambient: 0.18,
            two_sided: true,
            volume_tf: TransferFunction::default(),
            volume_step_size: 0.5,
            volume_shading: true,
            volume_clip_plane: None,
            hidden_blocks: BTreeSet::new(),
            block_colors: BTreeMap::new(),
        }
    }
}

impl ActorStyle {
    /// Returns `true` if a MultiBlock child with `path` (e.g. `"Base/per1"`) and `name` (`"per1"`)
    /// is visible according to [`Self::hidden_blocks`].
    pub fn is_block_visible(&self, path: &str, name: &str) -> bool {
        if self.hidden_blocks.is_empty() {
            return true;
        }
        if !path.is_empty() && self.hidden_blocks.contains(path) {
            return false;
        }
        if !name.is_empty() && self.hidden_blocks.contains(name) {
            return false;
        }
        true
    }

    /// Returns an explicit per-block RGBA color override matching `path` or `name` (path wins).
    pub fn block_color_override(&self, path: &str, name: &str) -> Option<[f32; 4]> {
        if self.block_colors.is_empty() {
            return None;
        }
        if !path.is_empty()
            && let Some(&rgba) = self.block_colors.get(path)
        {
            return Some(rgba);
        }
        if !name.is_empty()
            && let Some(&rgba) = self.block_colors.get(name)
        {
            return Some(rgba);
        }
        None
    }

    /// Computes the world-space bounding box of `dataset`, excluding blocks listed in [`Self::hidden_blocks`].
    pub fn dataset_bounds(&self, dataset: &DataSet) -> BoundingBox {
        if self.hidden_blocks.is_empty() {
            return dataset.bounds();
        }
        self.dataset_bounds_inner(dataset, "")
    }

    fn dataset_bounds_inner(&self, dataset: &DataSet, parent_path: &str) -> BoundingBox {
        match dataset {
            DataSet::MultiBlock(mb) => {
                let mut bb = BoundingBox::EMPTY;
                for (name, block) in &mb.blocks {
                    let child_path = if parent_path.is_empty() {
                        name.clone()
                    } else if name.is_empty() {
                        parent_path.to_string()
                    } else {
                        format!("{parent_path}/{name}")
                    };
                    if !self.is_block_visible(&child_path, name) {
                        continue;
                    }
                    bb.union(self.dataset_bounds_inner(block, &child_path));
                }
                bb
            }
            _ => dataset.bounds(),
        }
    }
}

/// A renderable dataset instance paired with its display style.
#[derive(Debug, Clone)]
pub struct SceneActor {
    pub id: u64,
    pub name: String,
    pub dataset: Arc<DataSet>,
    pub style: ActorStyle,
}

impl SceneActor {
    /// Creates a new visible [`SceneActor`] with default styling.
    pub fn new(id: u64, name: impl Into<String>, dataset: Arc<DataSet>) -> Self {
        Self {
            id,
            name: name.into(),
            dataset,
            style: ActorStyle::default(),
        }
    }
}

/// Configuration for the movable, resizable viewport Scalar Color Bar legend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScalarBarConfig {
    /// Normalized top-left position `[0.0..=1.0]` in the viewport (`None` defaults to bottom-right).
    pub pos: Option<[f32; 2]>,
    /// Logical card size `[width, height]` at 720p reference resolution.
    pub size: [f32; 2],
    /// Optional custom title overriding the scalar array name.
    pub custom_title: Option<String>,
    /// Number of tick labels along the bar (clamped to `2..=7`).
    pub num_labels: usize,
    /// Font scale multiplier (default `1.0`).
    pub font_scale: f32,
    /// Horizontal orientation instead of vertical (default `false`).
    pub horizontal: bool,
}

impl Default for ScalarBarConfig {
    fn default() -> Self {
        Self {
            pos: None,
            size: [160.0, 220.0],
            custom_title: None,
            num_labels: 3,
            font_scale: 1.0,
            horizontal: false,
        }
    }
}

/// Custom 2D text annotation badge positioned in normalized viewport coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewportAnnotation {
    pub id: u64,
    pub text: String,
    /// Normalized top-left position `[0.0..=1.0]` in the viewport.
    pub pos: [f32; 2],
    pub font_size: f32,
    pub color: [f32; 4],
    pub visible: bool,
}

impl Default for ViewportAnnotation {
    fn default() -> Self {
        Self {
            id: 0,
            text: String::new(),
            pos: [0.05, 0.05],
            font_size: 16.0,
            color: [0.94, 0.96, 1.0, 1.0],
            visible: true,
        }
    }
}

/// Global viewport rendering settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneSettings {
    pub bg_top: [f32; 3],
    pub bg_bottom: [f32; 3],
    pub show_axes_gizmo: bool,
    pub show_bounding_box: bool,
    #[serde(default)]
    pub show_axes_grid: bool,
    /// Clear background to alpha=0 (no skybox) for transparent PNG/GIF exports.
    #[serde(default)]
    pub transparent_background: bool,
    /// Composite a ParaView-style Scalar Color Bar with min/mid/max labels onto headless/screenshot outputs.
    #[serde(default)]
    pub show_scalar_bar: bool,
    /// Layout and appearance settings for the Scalar Color Bar.
    #[serde(default)]
    pub scalar_bar: ScalarBarConfig,
    /// Composite a `Time: ...` pill badge when [`Self::time_label`] is set.
    #[serde(default)]
    pub show_time_annotation: bool,
    /// Formatted time value displayed in the time annotation badge.
    #[serde(default)]
    pub time_label: Option<String>,
    /// User-defined 2D viewport text annotations.
    #[serde(default)]
    pub annotations: Vec<ViewportAnnotation>,
    pub edl_enabled: bool,
    pub edl_strength: f32,
    pub msaa_samples: u32,
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            bg_top: [0.13, 0.16, 0.22],
            bg_bottom: [0.27, 0.32, 0.41],
            show_axes_gizmo: true,
            show_bounding_box: false,
            show_axes_grid: false,
            transparent_background: false,
            show_scalar_bar: false,
            scalar_bar: ScalarBarConfig::default(),
            show_time_annotation: false,
            time_label: None,
            annotations: Vec::new(),
            edl_enabled: false,
            edl_strength: 1.0,
            msaa_samples: 1,
        }
    }
}

/// Interactive 3D viewport widget overlays rendered as crisp wireframe/translucent guides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ViewportWidget {
    Plane {
        origin: [f32; 3],
        normal: [f32; 3],
        size: f32,
    },
    Line {
        p1: [f32; 3],
        p2: [f32; 3],
    },
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
}

/// Complete scene state passed to [`crate::WebGpuRenderer`].
#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub camera: Camera,
    pub actors: Vec<SceneActor>,
    pub settings: SceneSettings,
    pub widgets: Vec<ViewportWidget>,
    /// Optional 3D world-space selection highlight marker (from Spreadsheet or 3D pick).
    pub selection_highlight: Option<[f32; 3]>,
}

impl RenderScene {
    /// Creates an empty [`RenderScene`] with default camera and settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an actor to the scene.
    pub fn add_actor(&mut self, actor: SceneActor) {
        self.actors.push(actor);
    }

    /// Combined world-space bounding box of all visible actors (or all actors if none are visible).
    pub fn bounds(&self) -> BoundingBox {
        let mut bb = BoundingBox::EMPTY;
        for actor in self.actors.iter().filter(|a| a.style.visible) {
            bb.union(actor.style.dataset_bounds(&actor.dataset));
        }
        if !bb.is_valid() {
            for actor in &self.actors {
                bb.union(actor.style.dataset_bounds(&actor.dataset));
            }
        }
        bb
    }

    /// Fits the scene camera to the combined bounds of the scene.
    pub fn fit_camera(&mut self) {
        let bb = self.bounds();
        self.camera.fit_bounds(&bb);
    }
}
