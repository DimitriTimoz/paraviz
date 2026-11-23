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
    PentagonalPrism = 15,
    HexagonalPrism = 16,
    QuadraticEdge = 21,
    QuadraticTriangle = 22,
    QuadraticQuad = 23,
    QuadraticTetra = 24,
    QuadraticHexahedron = 25,
    QuadraticWedge = 26,
    QuadraticPyramid = 27,
    BiQuadraticQuad = 28,
    TriQuadraticHexahedron = 29,
    Polyhedron = 42,
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
            15 => Some(Self::PentagonalPrism),
            16 => Some(Self::HexagonalPrism),
            21 => Some(Self::QuadraticEdge),
            22 => Some(Self::QuadraticTriangle),
            23 => Some(Self::QuadraticQuad),
            24 => Some(Self::QuadraticTetra),
            25 => Some(Self::QuadraticHexahedron),
            26 => Some(Self::QuadraticWedge),
            27 => Some(Self::QuadraticPyramid),
            28 => Some(Self::BiQuadraticQuad),
            29 => Some(Self::TriQuadraticHexahedron),
            42 => Some(Self::Polyhedron),
            _ => None,
        }
    }

    /// Returns the VTK numeric cell type ID.
    pub fn vtk_id(self) -> u8 {
        self as u8
    }

    /// Returns `true` if this is a higher-order (quadratic/bi-quadratic/tri-quadratic) cell.
    pub fn is_higher_order(self) -> bool {
        matches!(
            self,
            Self::QuadraticEdge
                | Self::QuadraticTriangle
                | Self::QuadraticQuad
                | Self::QuadraticTetra
                | Self::QuadraticHexahedron
                | Self::QuadraticWedge
                | Self::QuadraticPyramid
                | Self::BiQuadraticQuad
                | Self::TriQuadraticHexahedron
        )
    }

    /// Returns the topological dimension (0 = vertex, 1 = line, 2 = surface, 3 = volumetric).
    pub fn dimension(self) -> u8 {
        match self {
            Self::Vertex | Self::PolyVertex => 0,
            Self::Line | Self::PolyLine | Self::QuadraticEdge => 1,
            Self::Triangle
            | Self::TriangleStrip
            | Self::Polygon
            | Self::Pixel
            | Self::Quad
            | Self::QuadraticTriangle
            | Self::QuadraticQuad
            | Self::BiQuadraticQuad => 2,
            Self::Tetra
            | Self::Voxel
            | Self::Hexahedron
            | Self::Wedge
            | Self::Pyramid
            | Self::PentagonalPrism
            | Self::HexagonalPrism
            | Self::QuadraticTetra
            | Self::QuadraticHexahedron
            | Self::QuadraticWedge
            | Self::QuadraticPyramid
            | Self::TriQuadraticHexahedron
            | Self::Polyhedron => 3,
        }
    }

    /// Returns the polygonal boundary faces for 3D cells as local vertex index lists.
    /// For 2D cells, returns subdivided linear faces (for higher-order) or a single face.
    pub fn faces(self, num_pts: usize) -> Vec<Vec<usize>> {
        match self {
            Self::Triangle | Self::Polygon => vec![(0..num_pts).collect()],
            Self::Quad => vec![vec![0, 1, 2, 3]],
            Self::Pixel => vec![vec![0, 1, 3, 2]],
            Self::QuadraticTriangle if num_pts >= 6 => vec![
                vec![0, 3, 5],
                vec![3, 1, 4],
                vec![5, 4, 2],
                vec![3, 4, 5],
            ],
            Self::QuadraticTriangle => vec![vec![0, 1, 2]],
            Self::QuadraticQuad if num_pts >= 8 => vec![
                vec![0, 4, 7],
                vec![4, 1, 5],
                vec![5, 2, 6],
                vec![6, 3, 7],
                vec![4, 5, 6, 7],
            ],
            Self::BiQuadraticQuad if num_pts >= 9 => vec![
                vec![0, 4, 8, 7],
                vec![4, 1, 5, 8],
                vec![8, 5, 2, 6],
                vec![7, 8, 6, 3],
            ],
            Self::QuadraticQuad | Self::BiQuadraticQuad => vec![vec![0, 1, 2, 3]],
            Self::Tetra | Self::QuadraticTetra => vec![
                vec![0, 2, 1],
                vec![0, 1, 3],
                vec![1, 2, 3],
                vec![2, 0, 3],
            ],
            Self::Hexahedron | Self::QuadraticHexahedron | Self::TriQuadraticHexahedron => vec![
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
            Self::Wedge | Self::QuadraticWedge => vec![
                vec![0, 1, 2],
                vec![3, 5, 4],
                vec![0, 3, 4, 1],
                vec![1, 4, 5, 2],
                vec![2, 5, 3, 0],
            ],
            Self::Pyramid | Self::QuadraticPyramid => vec![
                vec![0, 3, 2, 1],
                vec![0, 1, 4],
                vec![1, 2, 4],
                vec![2, 3, 4],
                vec![3, 0, 4],
            ],
            Self::PentagonalPrism => vec![
                vec![0, 4, 3, 2, 1],
                vec![5, 6, 7, 8, 9],
                vec![0, 1, 6, 5],
                vec![1, 2, 7, 6],
                vec![2, 3, 8, 7],
                vec![3, 4, 9, 8],
                vec![4, 0, 5, 9],
            ],
            Self::HexagonalPrism => vec![
                vec![0, 5, 4, 3, 2, 1],
                vec![6, 7, 8, 9, 10, 11],
                vec![0, 1, 7, 6],
                vec![1, 2, 8, 7],
                vec![2, 3, 9, 8],
                vec![3, 4, 10, 9],
                vec![4, 5, 11, 10],
                vec![5, 0, 6, 11],
            ],
            _ => Vec::new(),
        }
    }

    /// Decomposes a 3D cell into tetrahedra (local vertex index quadruples) for algorithms like
    /// Marching Tetrahedra or plane slicing of arbitrary unstructured grids.
    pub fn tetrahedra(self) -> Vec<[usize; 4]> {
        self.tetrahedra_with_parity(false)
    }

    /// Parity-aware tetrahedral decomposition. For `Hexahedron` and `Voxel`, alternating `parity`
    /// (`(i + j + k) % 2 == 1` on structured grids) flips the 5-tet diagonal orientation so adjacent
    /// cells share matching face diagonals with zero topological cracks.
    pub fn tetrahedra_with_parity(self, parity: bool) -> Vec<[usize; 4]> {
        match self {
            Self::Tetra => vec![[0, 1, 2, 3]],
            Self::QuadraticTetra => vec![
                // Subdivide 10-node quadratic tetra into 8 linear sub-tetrahedra
                [0, 4, 6, 7],
                [4, 1, 5, 8],
                [6, 5, 2, 9],
                [7, 8, 9, 3],
                [4, 5, 6, 8],
                [4, 6, 7, 8],
                [6, 5, 8, 9],
                [6, 7, 8, 9],
            ],
            Self::Pyramid | Self::QuadraticPyramid => vec![[0, 1, 2, 4], [0, 2, 3, 4]],
            Self::Wedge | Self::QuadraticWedge => vec![[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            Self::Hexahedron | Self::QuadraticHexahedron | Self::TriQuadraticHexahedron => {
                if !parity {
                    vec![
                        [0, 1, 2, 5],
                        [0, 2, 3, 7],
                        [0, 4, 5, 7],
                        [2, 5, 6, 7],
                        [0, 2, 5, 7],
                    ]
                } else {
                    vec![
                        [1, 2, 3, 6],
                        [0, 1, 3, 4],
                        [1, 4, 5, 6],
                        [3, 4, 6, 7],
                        [1, 3, 4, 6],
                    ]
                }
            }
            Self::Voxel => {
                if !parity {
                    vec![
                        [0, 1, 3, 5],
                        [0, 3, 2, 6],
                        [0, 4, 5, 6],
                        [3, 5, 7, 6],
                        [0, 3, 5, 6],
                    ]
                } else {
                    vec![
                        [1, 3, 2, 7],
                        [0, 1, 2, 4],
                        [1, 4, 5, 7],
                        [2, 4, 7, 6],
                        [1, 2, 4, 7],
                    ]
                }
            }
            Self::PentagonalPrism => vec![
                // Split pentagon base (0..5) into 3 wedges around vertex 0
                [0, 1, 2, 5], [1, 2, 5, 6], [2, 5, 6, 7],
                [0, 2, 3, 5], [2, 3, 5, 7], [3, 5, 7, 8],
                [0, 3, 4, 5], [3, 4, 5, 8], [4, 5, 8, 9],
            ],
            Self::HexagonalPrism => vec![
                // Split hexagon base (0..6) into 4 wedges around vertex 0
                [0, 1, 2, 6], [1, 2, 6, 7], [2, 6, 7, 8],
                [0, 2, 3, 6], [2, 3, 6, 8], [3, 6, 8, 9],
                [0, 3, 4, 6], [3, 4, 6, 9], [4, 6, 9, 10],
                [0, 4, 5, 6], [4, 5, 6, 10], [5, 6, 10, 11],
            ],
            _ => Vec::new(),
        }
    }
}
