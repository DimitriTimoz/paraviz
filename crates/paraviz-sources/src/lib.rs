//! Procedural geometric and analytical scientific dataset sources for ParaViz
//! (equivalent to ParaView's Sources menu / `vtkSphereSource`, `vtkRTAnalyticSource`, etc.).

pub mod analytical;
pub mod geometric;

pub use analytical::{
    GyroidSource, PointCloudSource, SuperquadricSource, VectorFieldKind, VectorFieldSource,
    WaveletSource,
};
pub use geometric::{
    ArrowSource, ConeSource, CubeSource, CylinderSource, DiskSource, LineSource, PlaneSource,
    SphereSource, TorusSource,
};
