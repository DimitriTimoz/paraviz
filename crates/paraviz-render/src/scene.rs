//! Scene graph, actor representations, scalar color-mapping styles, and viewport settings.

use crate::camera::Camera;
use paraviz_core::{BoundingBox, ColorMap, DataSet, TransferFunction};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Geometric / volumetric visual representation of an actor (matching ParaView's Representation dropdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Representation {
    #[default]
    Surface,
    SurfaceWithEdges,
    Wireframe,
    Points,
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

/// Global viewport rendering settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneSettings {
    pub bg_top: [f32; 3],
    pub bg_bottom: [f32; 3],
    pub show_axes_gizmo: bool,
    pub show_bounding_box: bool,
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
            edl_enabled: false,
            edl_strength: 1.0,
            msaa_samples: 1,
        }
    }
}

/// Complete scene state passed to [`crate::WebGpuRenderer`].
#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub camera: Camera,
    pub actors: Vec<SceneActor>,
    pub settings: SceneSettings,
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
            bb.union(actor.dataset.bounds());
        }
        if !bb.is_valid() {
            for actor in &self.actors {
                bb.union(actor.dataset.bounds());
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
