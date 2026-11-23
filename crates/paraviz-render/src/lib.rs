//! ParaViz WebGPU (`wgpu`) scientific visualization rendering engine.
//!
//! Provides camera math, scene/style definitions, WGSL shaders, mesh/volume preparation,
//! 3D raycast picking, GPU compute Marching Cubes contouring, and a cached headless/live
//! [`WebGpuRenderer`] supporting Surface, SurfaceWithEdges, Wireframe, Points (sphere impostors),
//! Volume raymarching, Outline, 3D Viewport Widgets, Eye-Dome Lighting, 4x MSAA, and 3D Axes Gizmo.

pub mod camera;
pub mod compute;
pub mod mesh_prep;
pub mod pick;
pub mod renderer;
pub mod scene;
pub mod shaders;

pub use camera::{Camera, CameraPreset};
pub use compute::GpuContourEngine;
pub use pick::{pick_scene, PickResult};
pub use renderer::{RenderError, WebGpuRenderer};
pub use scene::{
    ActorStyle, ColorBy, RenderScene, Representation, SceneActor, SceneSettings, ViewportWidget,
};
