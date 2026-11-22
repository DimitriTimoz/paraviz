//! VTK cell types and topological utilities.

use serde::{Deserialize, Serialize};

/// VTK-compatible topological cell types (`vtkCellType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CellType {
    Vertex = 1,
    PolyVertex = 2,
    Line = 3,
    PolyLine = 4,
    Triangle = 5,
    TriangleStrip = 6,
    Polygon = 7,
    Pixel = 8,
    Quad = 9,
    Tetra = 10,
    Voxel = 11,
    Hexahedron = 12,
    Wedge = 13,
    Pyramid = 14,
}

impl CellType {
    /// Parses a VTK numeric cell type ID.
    pub fn from_vtk_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(Self::Vertex),
            2 => Some(Self::PolyVertex),
            3 => Some(Self::Line),
            4 => Some(Self::PolyLine),
            5 => Some(Self::Triangle),
            6 => Some(Self::TriangleStrip),
            7 => Some(Self::Polygon),
            8 => Some(Self::Pixel),
            9 => Some(Self::Quad),
            10 => Some(Self::Tetra),
            11 => Some(Self::Voxel),
            12 => Some(Self::Hexahedron),
            13 => Some(Self::Wedge),
            14 => Some(Self::Pyramid),
            _ => None,
        }
    }

    /// Returns the VTK numeric cell type ID.
    pub fn vtk_id(self) -> u8 {
        self as u8
    }

    /// Returns the topological dimension (0 = vertex, 1 = line, 2 = surface, 3 = volumetric).
    pub fn dimension(self) -> u8 {
        match self {
            Self::Vertex | Self::PolyVertex => 0,
            Self::Line | Self::PolyLine => 1,
            Self::Triangle | Self::TriangleStrip | Self::Polygon | Self::Pixel | Self::Quad => 2,
            Self::Tetra | Self::Voxel | Self::Hexahedron | Self::Wedge | Self::Pyramid => 3,
        }
    }

    /// Returns the polygonal boundary faces for 3D cells as local vertex index lists.
    /// For 2D cells, returns a single face covering all vertices (or reordered for Pixel).
    pub fn faces(self, num_pts: usize) -> Vec<Vec<usize>> {
        match self {
            Self::Triangle | Self::Polygon => vec![(0..num_pts).collect()],
            Self::Quad => vec![vec![0, 1, 2, 3]],
            Self::Pixel => vec![vec![0, 1, 3, 2]],
            Self::Tetra => vec![
                vec![0, 2, 1],
                vec![0, 1, 3],
                vec![1, 2, 3],
                vec![2, 0, 3],
            ],
            Self::Hexahedron => vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
            Self::Voxel => vec![
                vec![0, 2, 3, 1],
                vec![4, 5, 7, 6],
                vec![0, 1, 5, 4],
                vec![1, 3, 7, 5],
                vec![3, 2, 6, 7],
                vec![2, 0, 4, 6],
            ],
            Self::Wedge => vec![
                vec![0, 1, 2],
                vec![3, 5, 4],
                vec![0, 3, 4, 1],
                vec![1, 4, 5, 2],
                vec![2, 5, 3, 0],
            ],
            Self::Pyramid => vec![
                vec![0, 3, 2, 1],
                vec![0, 1, 4],
                vec![1, 2, 4],
                vec![2, 3, 4],
                vec![3, 0, 4],
            ],
            _ => Vec::new(),
        }
    }

    /// Decomposes a 3D cell into tetrahedra (local vertex index quadruples) for algorithms like
    /// Marching Tetrahedra or plane slicing of arbitrary unstructured grids.
    pub fn tetrahedra(self) -> Vec<[usize; 4]> {
        match self {
            Self::Tetra => vec![[0, 1, 2, 3]],
            Self::Pyramid => vec![[0, 1, 2, 4], [0, 2, 3, 4]],
            Self::Wedge => vec![[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            Self::Hexahedron => vec![
                [0, 1, 2, 5],
                [0, 2, 3, 7],
                [0, 4, 5, 7],
                [2, 5, 6, 7],
                [0, 2, 5, 7],
            ],
            Self::Voxel => vec![
                [0, 1, 3, 5],
                [0, 3, 2, 6],
                [0, 4, 5, 6],
                [3, 5, 7, 6],
                [0, 3, 5, 6],
            ],
            _ => Vec::new(),
        }
    }
}
