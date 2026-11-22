//! 3D perspective and orthographic camera with ParaView-compatible presets and interaction.

use glam::{Mat4, Quat, Vec3};
use paraviz_core::BoundingBox;
use serde::{Deserialize, Serialize};

/// Standard axis-aligned and isometric camera presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CameraPreset {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
    Isometric,
}

/// 3D camera supporting perspective and orthographic projections, orbit, pan, zoom/dolly, and bounds fitting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub orthographic: bool,
    pub ortho_scale: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 4.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_deg: 45.0,
            near: 0.01,
            far: 1000.0,
            orthographic: false,
            ortho_scale: 1.5,
        }
    }
}

impl Camera {
    /// Unit vector from `position` toward `target`.
    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.position).normalize_or(Vec3::NEG_Z)
    }

    /// Unit screen-right vector.
    pub fn right(&self) -> Vec3 {
        let dir = self.view_direction();
        let up = self.up.normalize_or(Vec3::Y);
        dir.cross(up).normalize_or(Vec3::X)
    }

    /// Distance between `position` and `target`.
    pub fn distance(&self) -> f32 {
        (self.position - self.target).length().max(1e-5)
    }

    /// Orbits the camera around `target` by `delta_yaw_rad` (around `up`) and `delta_pitch_rad` (around `right`).
    pub fn orbit(&mut self, delta_yaw_rad: f32, delta_pitch_rad: f32) {
        let offset = self.position - self.target;
        let dist = offset.length().max(1e-5);
        let up = self.up.normalize_or(Vec3::Y);
        let view_dir = (-offset / dist).normalize_or(Vec3::NEG_Z);
        let right = view_dir.cross(up).normalize_or(Vec3::X);

        let q_yaw = Quat::from_axis_angle(up, delta_yaw_rad);
        let q_pitch = Quat::from_axis_angle(right, delta_pitch_rad);
        let q = (q_yaw * q_pitch).normalize();

        let new_offset = (q * offset).normalize_or(Vec3::Z) * dist;
        let new_view_dir = (-new_offset / dist).normalize_or(Vec3::NEG_Z);
        let new_right = (q * right).normalize_or(Vec3::X);
        let new_up = new_right.cross(new_view_dir).normalize_or(Vec3::Y);

        self.position = self.target + new_offset;
        self.up = new_up;
    }

    /// Pans `position` and `target` parallel to the image plane.
    /// `delta_x` and `delta_y` are normalized viewport fractions (e.g., `dx / viewport_height`).
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let view_dir = self.view_direction();
        let right = self.right();
        let cam_up = right.cross(view_dir).normalize_or(Vec3::Y);

        let span = if self.orthographic {
            self.ortho_scale.max(1e-5) * 2.0
        } else {
            let half_fov = (self.fov_y_deg.to_radians() * 0.5).clamp(0.01, 1.5);
            2.0 * self.distance() * half_fov.tan()
        };

        let shift = -right * (delta_x * span) + cam_up * (delta_y * span);
        self.position += shift;
        self.target += shift;
    }

    /// Zooms by `factor` (`factor > 1.0` zooms in, decreasing distance and `ortho_scale`).
    pub fn zoom(&mut self, factor: f32) {
        let f = factor.max(1e-4);
        let offset = self.position - self.target;
        let new_dist = (offset.length() / f).max(1e-4);
        let dir = offset.normalize_or(Vec3::Z);
        self.position = self.target + dir * new_dist;
        self.ortho_scale = (self.ortho_scale / f).max(1e-5);
    }

    /// Dollies along the viewing ray (`delta > 0` moves closer toward `target`).
    pub fn dolly(&mut self, delta: f32) {
        let factor = delta.clamp(-10.0, 10.0).exp();
        self.zoom(factor);
    }

    /// Repositions the camera along its current viewing direction to frame `bounds`.
    pub fn fit_bounds(&mut self, bounds: &BoundingBox) {
        let (center, radius) = if bounds.is_valid() {
            let r = (bounds.diagonal() * 0.5).max(1e-3);
            (bounds.center(), r)
        } else {
            (Vec3::ZERO, 1.0)
        };

        let dir = (self.position - self.target).normalize_or(Vec3::Z);
        let half_fov = (self.fov_y_deg.to_radians() * 0.5).clamp(0.05, 1.45);
        let dist = (radius / half_fov.sin()) * 1.15;

        self.target = center;
        self.position = center + dir * dist;
        self.ortho_scale = radius * 1.15;
        self.near = (dist - radius * 4.0).max(radius * 0.002).max(1e-4);
        self.far = (dist + radius * 12.0).max(self.near + 10.0);

        // Keep `up` orthogonal to viewing direction.
        let view_dir = -dir;
        let up = self.up.normalize_or(Vec3::Y);
        let right = view_dir.cross(up);
        if right.length_squared() < 1e-6 {
            let fallback = if view_dir.y.abs() > 0.9 { Vec3::NEG_Z } else { Vec3::Y };
            let r = view_dir.cross(fallback).normalize_or(Vec3::X);
            self.up = r.cross(view_dir).normalize_or(Vec3::Y);
        } else {
            let r = right.normalize();
            self.up = r.cross(view_dir).normalize_or(Vec3::Y);
        }
    }

    /// Applies a standard [`CameraPreset`] and fits `bounds`.
    pub fn apply_preset(&mut self, preset: CameraPreset, bounds: &BoundingBox) {
        let (dir, up) = match preset {
            CameraPreset::PosX => (Vec3::X, Vec3::Y),
            CameraPreset::NegX => (Vec3::NEG_X, Vec3::Y),
            CameraPreset::PosY => (Vec3::Y, Vec3::NEG_Z),
            CameraPreset::NegY => (Vec3::NEG_Y, Vec3::Z),
            CameraPreset::PosZ => (Vec3::Z, Vec3::Y),
            CameraPreset::NegZ => (Vec3::NEG_Z, Vec3::Y),
            CameraPreset::Isometric => (Vec3::new(1.0, 1.0, 1.0).normalize(), Vec3::Y),
        };
        let center = if bounds.is_valid() {
            bounds.center()
        } else {
            Vec3::ZERO
        };
        self.target = center;
        self.position = center + dir;
        self.up = up;
        self.fit_bounds(bounds);
    }

    /// Right-handed world-to-view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let up = self.up.normalize_or(Vec3::Y);
        glam::camera::rh::view::look_at_mat4(self.position, self.target, up)
    }

    /// WebGPU `[0, 1]` depth-range projection matrix for viewport `aspect = width / height`.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let aspect = aspect.max(1e-4);
        let near = self.near.max(1e-5);
        let far = self.far.max(near + 1e-3);
        if self.orthographic {
            let half_h = self.ortho_scale.max(1e-5);
            let half_w = half_h * aspect;
            glam::camera::rh::proj::directx::orthographic(
                -half_w, half_w, -half_h, half_h, near, far,
            )
        } else {
            let fov = self.fov_y_deg.to_radians().clamp(0.05, 3.0);
            glam::camera::rh::proj::directx::perspective(fov, aspect, near, far)
        }
    }

    /// Combined `projection * view` matrix.
    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}
