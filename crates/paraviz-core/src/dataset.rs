//! Scientific dataset types (equivalent to `vtkDataSet` hierarchy):
//! `PolyData`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `UnstructuredGrid`, and `MultiBlockDataSet`.

use crate::array::DataArray;
use crate::cell::CellType;
use crate::field::FieldData;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Axis-aligned 3D bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl BoundingBox {
    /// Empty inverted bounding box ready for expansion.
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Creates a bounding box from min/max points.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Computes a tight bounding box from a slice of `[f32; 3]` points.
    pub fn from_points(points: &[[f32; 3]]) -> Self {
        if points.is_empty() {
            return Self::new(Vec3::ZERO, Vec3::ZERO);
        }
        let mut bb = Self::EMPTY;
        for &p in points {
            bb.expand_point(Vec3::from_array(p));
        }
        bb
    }

    /// Expands the box to contain `p`.
    #[inline]
    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Merges another bounding box into `self`.
    #[inline]
    pub fn union(&mut self, other: Self) {
        if other.is_valid() {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Returns `true` if the bounding box has finite min <= max.
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x
            && self.min.y <= self.max.y
            && self.min.z <= self.max.z
            && self.min.is_finite()
            && self.max.is_finite()
    }

    /// Center of the bounding box.
    pub fn center(&self) -> Vec3 {
        if !self.is_valid() {
            Vec3::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    /// Diagonal size vector.
    pub fn size(&self) -> Vec3 {
        if !self.is_valid() {
            Vec3::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Diagonal length.
    pub fn diagonal(&self) -> f32 {
        self.size().length()
    }
}

/// Polygonal mesh dataset (equivalent to `vtkPolyData`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PolyData {
    pub points: Vec<[f32; 3]>,
    /// Standalone vertex indices.
    pub verts: Vec<u32>,
    /// Polylines: each polyline is a list of point indices.
    pub lines: Vec<Vec<u32>>,
    /// Polygons (triangles, quads, n-gons): each polygon is a list of point indices.
    pub polys: Vec<Vec<u32>>,
    /// Triangle strips: each strip is a list of point indices.
    pub strips: Vec<Vec<u32>>,
    pub point_data: FieldData,
    pub cell_data: FieldData,
    pub field_data: FieldData,
}

impl PolyData {
    /// Creates an empty `PolyData`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `PolyData` from points and triangle index triples.
    pub fn from_triangles(points: Vec<[f32; 3]>, triangles: Vec<[u32; 3]>) -> Self {
        let polys = triangles.into_iter().map(|t| t.to_vec()).collect();
        Self {
            points,
            polys,
            ..Default::default()
        }
    }

    /// Number of points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Total number of cells (verts + lines + polys + strips).
    pub fn num_cells(&self) -> usize {
        self.verts.len() + self.lines.len() + self.polys.len() + self.strips.len()
    }

    /// Axis-aligned bounding box.
    pub fn bounds(&self) -> BoundingBox {
        BoundingBox::from_points(&self.points)
    }

    /// Computes smooth vertex normals if `polys` are present and stores them in `point_data`.
    pub fn compute_normals(&mut self) {
        let mut normals = vec![Vec3::ZERO; self.points.len()];
        for poly in &self.polys {
            if poly.len() < 3 {
                continue;
            }
            let p0 = Vec3::from_array(self.points[poly[0] as usize]);
            let p1 = Vec3::from_array(self.points[poly[1] as usize]);
            let p2 = Vec3::from_array(self.points[poly[2] as usize]);
            let n = (p1 - p0).cross(p2 - p0);
            for &idx in poly {
                if let Some(acc) = normals.get_mut(idx as usize) {
                    *acc += n;
                }
            }
        }
        let flat: Vec<[f32; 3]> = normals
            .into_iter()
            .map(|n| {
                let len = n.length();
                if len > 1e-12 {
                    (n / len).to_array()
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect();
        self.point_data
            .set_normals(DataArray::vectors_f32("Normals", flat));
    }

    /// Triangulates polygons and triangle strips into a flat list of `[u32; 3]` triangles.
    pub fn triangulate(&self) -> Vec<[u32; 3]> {
        let mut tris = Vec::with_capacity(self.polys.len() * 2);
        for poly in &self.polys {
            if poly.len() < 3 {
                continue;
            }
            let v0 = poly[0];
            for k in 1..(poly.len() - 1) {
                tris.push([v0, poly[k], poly[k + 1]]);
            }
        }
        for strip in &self.strips {
            if strip.len() < 3 {
                continue;
            }
            for k in 0..(strip.len() - 2) {
                if k % 2 == 0 {
                    tris.push([strip[k], strip[k + 1], strip[k + 2]]);
                } else {
                    tris.push([strip[k + 1], strip[k], strip[k + 2]]);
                }
            }
        }
        tris
    }
}

/// Uniform 3D structured grid / voxel volume (equivalent to `vtkImageData`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageData {
    /// Number of points along `[nx, ny, nz]`.
    pub dimensions: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub point_data: FieldData,
    pub cell_data: FieldData,
    pub field_data: FieldData,
}

impl Default for ImageData {
    fn default() -> Self {
        Self {
            dimensions: [1, 1, 1],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: FieldData::new(),
            cell_data: FieldData::new(),
            field_data: FieldData::new(),
        }
    }
}

impl ImageData {
    /// Creates a new `ImageData` with given point dimensions, origin, and voxel spacing.
    pub fn new(dimensions: [usize; 3], origin: [f64; 3], spacing: [f64; 3]) -> Self {
        Self {
            dimensions,
            origin,
            spacing,
            ..Default::default()
        }
    }

    /// Total number of grid points (`nx * ny * nz`).
    #[inline]
    pub fn num_points(&self) -> usize {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Total number of grid cells (`(nx-1).max(1) * ...` when 2D/3D).
    pub fn num_cells(&self) -> usize {
        let cx = if self.dimensions[0] > 1 {
            self.dimensions[0] - 1
        } else if self.dimensions[0] == 1 {
            1
        } else {
            0
        };
        let cy = if self.dimensions[1] > 1 {
            self.dimensions[1] - 1
        } else if self.dimensions[1] == 1 {
            1
        } else {
            0
        };
        let cz = if self.dimensions[2] > 1 {
            self.dimensions[2] - 1
        } else if self.dimensions[2] == 1 {
            1
        } else {
            0
        };
        if self.dimensions.contains(&0) {
            0
        } else {
            cx * cy * cz
        }
    }

    /// Linear point index from `(i, j, k)` using VTK Fortran-style ordering (`i + nx*(j + ny*k)`).
    #[inline]
    pub fn point_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dimensions[0] * (j + self.dimensions[1] * k)
    }

    /// World-space coordinates of grid point `(i, j, k)`.
    #[inline]
    pub fn point_coords(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        [
            (self.origin[0] + i as f64 * self.spacing[0]) as f32,
            (self.origin[1] + j as f64 * self.spacing[1]) as f32,
            (self.origin[2] + k as f64 * self.spacing[2]) as f32,
        ]
    }

    /// Axis-aligned bounding box of the grid points.
    pub fn bounds(&self) -> BoundingBox {
        if self.num_points() == 0 {
            return BoundingBox::new(Vec3::ZERO, Vec3::ZERO);
        }
        let p0 = Vec3::new(
            self.origin[0] as f32,
            self.origin[1] as f32,
            self.origin[2] as f32,
        );
        let p1 = Vec3::new(
            (self.origin[0] + (self.dimensions[0].saturating_sub(1)) as f64 * self.spacing[0])
                as f32,
            (self.origin[1] + (self.dimensions[1].saturating_sub(1)) as f64 * self.spacing[1])
                as f32,
            (self.origin[2] + (self.dimensions[2].saturating_sub(1)) as f64 * self.spacing[2])
                as f32,
        );
        BoundingBox::new(p0.min(p1), p0.max(p1))
    }

    /// Materializes all point positions into a `Vec<[f32; 3]>`.
    pub fn all_points(&self) -> Vec<[f32; 3]> {
        let [nx, ny, nz] = self.dimensions;
        let mut pts = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    pts.push(self.point_coords(i, j, k));
                }
            }
        }
        pts
    }

    /// Trilinearly samples a point-data `DataArray` at a world-space position `pos`.
    /// Returns `None` if `pos` is outside the grid bounds or the array is missing.
    pub fn sample_point_array(&self, array_name: &str, pos: Vec3) -> Option<Vec<f64>> {
        let arr = self.point_data.get(array_name)?;
        let [nx, ny, nz] = self.dimensions;
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        let fx = (pos.x as f64 - self.origin[0]) / self.spacing[0];
        let fy = (pos.y as f64 - self.origin[1]) / self.spacing[1];
        let fz = (pos.z as f64 - self.origin[2]) / self.spacing[2];

        if fx < 0.0
            || fy < 0.0
            || fz < 0.0
            || fx > (nx - 1) as f64
            || fy > (ny - 1) as f64
            || fz > (nz - 1) as f64
        {
            return None;
        }

        let i0 = (fx.floor() as usize).min(nx.saturating_sub(2));
        let j0 = (fy.floor() as usize).min(ny.saturating_sub(2));
        let k0 = (fz.floor() as usize).min(nz.saturating_sub(2));
        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let k1 = (k0 + 1).min(nz - 1);

        let tx = if nx > 1 { fx - i0 as f64 } else { 0.0 };
        let ty = if ny > 1 { fy - j0 as f64 } else { 0.0 };
        let tz = if nz > 1 { fz - k0 as f64 } else { 0.0 };

        let nc = arr.num_components;
        let idx = [
            self.point_index(i0, j0, k0),
            self.point_index(i1, j0, k0),
            self.point_index(i0, j1, k0),
            self.point_index(i1, j1, k0),
            self.point_index(i0, j0, k1),
            self.point_index(i1, j0, k1),
            self.point_index(i0, j1, k1),
            self.point_index(i1, j1, k1),
        ];

        let mut out = vec![0.0; nc];
        for c in 0..nc {
            let v000 = arr.data.get_f64(idx[0] * nc + c);
            let v100 = arr.data.get_f64(idx[1] * nc + c);
            let v010 = arr.data.get_f64(idx[2] * nc + c);
            let v110 = arr.data.get_f64(idx[3] * nc + c);
            let v001 = arr.data.get_f64(idx[4] * nc + c);
            let v101 = arr.data.get_f64(idx[5] * nc + c);
            let v011 = arr.data.get_f64(idx[6] * nc + c);
            let v111 = arr.data.get_f64(idx[7] * nc + c);

            let c00 = v000 * (1.0 - tx) + v100 * tx;
            let c10 = v010 * (1.0 - tx) + v110 * tx;
            let c01 = v001 * (1.0 - tx) + v101 * tx;
            let c11 = v011 * (1.0 - tx) + v111 * tx;

            let c0 = c00 * (1.0 - ty) + c10 * ty;
            let c1 = c01 * (1.0 - ty) + c11 * ty;

            out[c] = c0 * (1.0 - tz) + c1 * tz;
        }
        Some(out)
    }
}

/// Non-uniform axis-aligned grid (equivalent to `vtkRectilinearGrid`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RectilinearGrid {
    pub x_coords: Vec<f32>,
    pub y_coords: Vec<f32>,
    pub z_coords: Vec<f32>,
    pub point_data: FieldData,
    pub cell_data: FieldData,
    pub field_data: FieldData,
}

impl RectilinearGrid {
    pub fn dimensions(&self) -> [usize; 3] {
        [self.x_coords.len(), self.y_coords.len(), self.z_coords.len()]
    }

    pub fn num_points(&self) -> usize {
        self.x_coords.len() * self.y_coords.len() * self.z_coords.len()
    }

    pub fn num_cells(&self) -> usize {
        self.x_coords.len().saturating_sub(1).max(1)
            * self.y_coords.len().saturating_sub(1).max(1)
            * self.z_coords.len().saturating_sub(1).max(1)
    }

    #[inline]
    pub fn point_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.x_coords.len() * (j + self.y_coords.len() * k)
    }

    #[inline]
    pub fn point_coords(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        [self.x_coords[i], self.y_coords[j], self.z_coords[k]]
    }

    pub fn bounds(&self) -> BoundingBox {
        if self.x_coords.is_empty() || self.y_coords.is_empty() || self.z_coords.is_empty() {
            return BoundingBox::new(Vec3::ZERO, Vec3::ZERO);
        }
        let (xmin, xmax) = min_max_slice(&self.x_coords);
        let (ymin, ymax) = min_max_slice(&self.y_coords);
        let (zmin, zmax) = min_max_slice(&self.z_coords);
        BoundingBox::new(Vec3::new(xmin, ymin, zmin), Vec3::new(xmax, ymax, zmax))
    }

    pub fn all_points(&self) -> Vec<[f32; 3]> {
        let mut pts = Vec::with_capacity(self.num_points());
        for &z in &self.z_coords {
            for &y in &self.y_coords {
                for &x in &self.x_coords {
                    pts.push([x, y, z]);
                }
            }
        }
        pts
    }
}

/// Curvilinear structured grid with explicit 3D point coordinates (equivalent to `vtkStructuredGrid`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuredGrid {
    pub dimensions: [usize; 3],
    pub points: Vec<[f32; 3]>,
    pub point_data: FieldData,
    pub cell_data: FieldData,
    pub field_data: FieldData,
}

impl StructuredGrid {
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_cells(&self) -> usize {
        if self.dimensions.contains(&0) {
            return 0;
        }
        self.dimensions[0].saturating_sub(1).max(1)
            * self.dimensions[1].saturating_sub(1).max(1)
            * self.dimensions[2].saturating_sub(1).max(1)
    }

    #[inline]
    pub fn point_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dimensions[0] * (j + self.dimensions[1] * k)
    }

    #[inline]
    pub fn point_coords(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        self.points[self.point_index(i, j, k)]
    }

    pub fn bounds(&self) -> BoundingBox {
        BoundingBox::from_points(&self.points)
    }
}

/// Arbitrary topological mesh of mixed 0D/1D/2D/3D cells (equivalent to `vtkUnstructuredGrid`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UnstructuredGrid {
    pub points: Vec<[f32; 3]>,
    /// Flat vertex connectivity buffer.
    pub connectivity: Vec<u32>,
    /// Offsets array of length `num_cells + 1` into `connectivity`.
    pub offsets: Vec<u32>,
    /// Per-cell topological type.
    pub cell_types: Vec<CellType>,
    pub point_data: FieldData,
    pub cell_data: FieldData,
    pub field_data: FieldData,
}

impl UnstructuredGrid {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            ..Default::default()
        }
    }

    /// Appends a single cell to the grid.
    pub fn push_cell(&mut self, cell_type: CellType, vertex_indices: &[u32]) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.connectivity.extend_from_slice(vertex_indices);
        self.offsets.push(self.connectivity.len() as u32);
        self.cell_types.push(cell_type);
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_cells(&self) -> usize {
        self.cell_types.len()
    }

    /// Returns the point indices slice for `cell_idx`.
    #[inline]
    pub fn cell_points(&self, cell_idx: usize) -> &[u32] {
        let start = self.offsets[cell_idx] as usize;
        let end = self.offsets[cell_idx + 1] as usize;
        &self.connectivity[start..end]
    }

    pub fn bounds(&self) -> BoundingBox {
        BoundingBox::from_points(&self.points)
    }
}

/// Composite hierarchical dataset containing named child datasets (equivalent to `vtkMultiBlockDataSet`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiBlockDataSet {
    pub blocks: Vec<(String, DataSet)>,
}

impl MultiBlockDataSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_block(&mut self, name: impl Into<String>, dataset: DataSet) {
        self.blocks.push((name.into(), dataset));
    }

    pub fn num_points(&self) -> usize {
        self.blocks.iter().map(|(_, d)| d.num_points()).sum()
    }

    pub fn num_cells(&self) -> usize {
        self.blocks.iter().map(|(_, d)| d.num_cells()).sum()
    }

    pub fn bounds(&self) -> BoundingBox {
        let mut bb = BoundingBox::EMPTY;
        for (_, d) in &self.blocks {
            bb.union(d.bounds());
        }
        if bb.is_valid() {
            bb
        } else {
            BoundingBox::new(Vec3::ZERO, Vec3::ZERO)
        }
    }
}

/// Unified scientific dataset enum (equivalent to `vtkDataObject` / `vtkDataSet`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSet {
    PolyData(PolyData),
    ImageData(ImageData),
    RectilinearGrid(RectilinearGrid),
    StructuredGrid(StructuredGrid),
    UnstructuredGrid(UnstructuredGrid),
    MultiBlock(MultiBlockDataSet),
}

impl DataSet {
    /// Human-readable dataset type name matching ParaView Information panel.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::PolyData(_) => "Polygonal Mesh (PolyData)",
            Self::ImageData(_) => "Image (Uniform Rectilinear Grid)",
            Self::RectilinearGrid(_) => "Rectilinear Grid",
            Self::StructuredGrid(_) => "Structured (Curvilinear) Grid",
            Self::UnstructuredGrid(_) => "Unstructured Grid",
            Self::MultiBlock(_) => "Multi-block Dataset",
        }
    }

    pub fn num_points(&self) -> usize {
        match self {
            Self::PolyData(d) => d.num_points(),
            Self::ImageData(d) => d.num_points(),
            Self::RectilinearGrid(d) => d.num_points(),
            Self::StructuredGrid(d) => d.num_points(),
            Self::UnstructuredGrid(d) => d.num_points(),
            Self::MultiBlock(d) => d.num_points(),
        }
    }

    pub fn num_cells(&self) -> usize {
        match self {
            Self::PolyData(d) => d.num_cells(),
            Self::ImageData(d) => d.num_cells(),
            Self::RectilinearGrid(d) => d.num_cells(),
            Self::StructuredGrid(d) => d.num_cells(),
            Self::UnstructuredGrid(d) => d.num_cells(),
            Self::MultiBlock(d) => d.num_cells(),
        }
    }

    pub fn bounds(&self) -> BoundingBox {
        match self {
            Self::PolyData(d) => d.bounds(),
            Self::ImageData(d) => d.bounds(),
            Self::RectilinearGrid(d) => d.bounds(),
            Self::StructuredGrid(d) => d.bounds(),
            Self::UnstructuredGrid(d) => d.bounds(),
            Self::MultiBlock(d) => d.bounds(),
        }
    }

    pub fn point_data(&self) -> Option<&FieldData> {
        match self {
            Self::PolyData(d) => Some(&d.point_data),
            Self::ImageData(d) => Some(&d.point_data),
            Self::RectilinearGrid(d) => Some(&d.point_data),
            Self::StructuredGrid(d) => Some(&d.point_data),
            Self::UnstructuredGrid(d) => Some(&d.point_data),
            Self::MultiBlock(_) => None,
        }
    }

    pub fn point_data_mut(&mut self) -> Option<&mut FieldData> {
        match self {
            Self::PolyData(d) => Some(&mut d.point_data),
            Self::ImageData(d) => Some(&mut d.point_data),
            Self::RectilinearGrid(d) => Some(&mut d.point_data),
            Self::StructuredGrid(d) => Some(&mut d.point_data),
            Self::UnstructuredGrid(d) => Some(&mut d.point_data),
            Self::MultiBlock(_) => None,
        }
    }

    pub fn cell_data(&self) -> Option<&FieldData> {
        match self {
            Self::PolyData(d) => Some(&d.cell_data),
            Self::ImageData(d) => Some(&d.cell_data),
            Self::RectilinearGrid(d) => Some(&d.cell_data),
            Self::StructuredGrid(d) => Some(&d.cell_data),
            Self::UnstructuredGrid(d) => Some(&d.cell_data),
            Self::MultiBlock(_) => None,
        }
    }

    pub fn cell_data_mut(&mut self) -> Option<&mut FieldData> {
        match self {
            Self::PolyData(d) => Some(&mut d.cell_data),
            Self::ImageData(d) => Some(&mut d.cell_data),
            Self::RectilinearGrid(d) => Some(&mut d.cell_data),
            Self::StructuredGrid(d) => Some(&mut d.cell_data),
            Self::UnstructuredGrid(d) => Some(&mut d.cell_data),
            Self::MultiBlock(_) => None,
        }
    }

    pub fn field_data(&self) -> Option<&FieldData> {
        match self {
            Self::PolyData(d) => Some(&d.field_data),
            Self::ImageData(d) => Some(&d.field_data),
            Self::RectilinearGrid(d) => Some(&d.field_data),
            Self::StructuredGrid(d) => Some(&d.field_data),
            Self::UnstructuredGrid(d) => Some(&d.field_data),
            Self::MultiBlock(_) => None,
        }
    }

    /// Approximate in-memory footprint in bytes for ParaView Information tab.
    pub fn memory_bytes(&self) -> usize {
        let field_bytes = |f: &FieldData| -> usize {
            f.arrays()
                .iter()
                .map(|a| {
                    let elem = match a.data.scalar_type() {
                        crate::array::ScalarType::U8 => 1,
                        crate::array::ScalarType::F32
                        | crate::array::ScalarType::I32
                        | crate::array::ScalarType::U32 => 4,
                        crate::array::ScalarType::F64 | crate::array::ScalarType::I64 => 8,
                    };
                    a.data.len() * elem
                })
                .sum()
        };
        match self {
            Self::PolyData(d) => {
                d.points.len() * 12
                    + d.verts.len() * 4
                    + d.lines.iter().map(|l| l.len() * 4).sum::<usize>()
                    + d.polys.iter().map(|p| p.len() * 4).sum::<usize>()
                    + d.strips.iter().map(|s| s.len() * 4).sum::<usize>()
                    + field_bytes(&d.point_data)
                    + field_bytes(&d.cell_data)
                    + field_bytes(&d.field_data)
            }
            Self::ImageData(d) => {
                field_bytes(&d.point_data) + field_bytes(&d.cell_data) + field_bytes(&d.field_data)
            }
            Self::RectilinearGrid(d) => {
                (d.x_coords.len() + d.y_coords.len() + d.z_coords.len()) * 4
                    + field_bytes(&d.point_data)
                    + field_bytes(&d.cell_data)
                    + field_bytes(&d.field_data)
            }
            Self::StructuredGrid(d) => {
                d.points.len() * 12
                    + field_bytes(&d.point_data)
                    + field_bytes(&d.cell_data)
                    + field_bytes(&d.field_data)
            }
            Self::UnstructuredGrid(d) => {
                d.points.len() * 12
                    + d.connectivity.len() * 4
                    + d.offsets.len() * 4
                    + d.cell_types.len()
                    + field_bytes(&d.point_data)
                    + field_bytes(&d.cell_data)
                    + field_bytes(&d.field_data)
            }
            Self::MultiBlock(mb) => mb.blocks.iter().map(|(_, b)| b.memory_bytes()).sum(),
        }
    }

    pub fn as_poly_data(&self) -> Option<&PolyData> {
        if let Self::PolyData(p) = self {
            Some(p)
        } else {
            None
        }
    }

    pub fn as_image_data(&self) -> Option<&ImageData> {
        if let Self::ImageData(i) = self {
            Some(i)
        } else {
            None
        }
    }

    pub fn as_unstructured_grid(&self) -> Option<&UnstructuredGrid> {
        if let Self::UnstructuredGrid(u) = self {
            Some(u)
        } else {
            None
        }
    }
}

impl From<PolyData> for DataSet {
    fn from(value: PolyData) -> Self {
        Self::PolyData(value)
    }
}

impl From<ImageData> for DataSet {
    fn from(value: ImageData) -> Self {
        Self::ImageData(value)
    }
}

impl From<RectilinearGrid> for DataSet {
    fn from(value: RectilinearGrid) -> Self {
        Self::RectilinearGrid(value)
    }
}

impl From<StructuredGrid> for DataSet {
    fn from(value: StructuredGrid) -> Self {
        Self::StructuredGrid(value)
    }
}

impl From<UnstructuredGrid> for DataSet {
    fn from(value: UnstructuredGrid) -> Self {
        Self::UnstructuredGrid(value)
    }
}

impl From<MultiBlockDataSet> for DataSet {
    fn from(value: MultiBlockDataSet) -> Self {
        Self::MultiBlock(value)
    }
}

fn min_max_slice(vals: &[f32]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in vals {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    (mn, mx)
}
