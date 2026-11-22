//! ParaViz WebGPU (`wgpu`) scientific visualization rendering engine.
//!
//! Provides camera math, scene/style definitions, WGSL shaders, mesh/volume preparation,
//! and a cached headless/live [`WebGpuRenderer`] supporting Surface, SurfaceWithEdges,
//! Wireframe, Points (sphere impostors), Volume raymarching, Outline, and 3D Axes Gizmo.

pub mod camera;
pub mod mesh_prep;
pub mod renderer;
pub mod scene;
pub mod shaders;

pub use camera::{Camera, CameraPreset};
pub use renderer::{RenderError, WebGpuRenderer};
pub use scene::{ActorStyle, ColorBy, RenderScene, Representation, SceneActor, SceneSettings};
