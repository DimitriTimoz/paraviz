//! Topological edge extraction filter (`ExtractEdgesFilter`, equivalent to `vtkExtractEdges`).

use crate::util::{canonical_edge_key, structured_cells, FxHashSet, GridPoints};
use paraviz_core::{
    CellType, DataSet, Filter, MultiBlockDataSet, PipelineError, PolyData,
};
use serde::{Deserialize, Serialize};

/// Extracts all unique topological edges of a dataset's cells as `PolyData` line segments,
/// preserving `point_data` and `field_data`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractEdgesFilter;

impl ExtractEdgesFilter {
    /// Creates an `ExtractEdgesFilter`.
    pub fn new() -> Self {
        Self
    }
}

impl Filter for ExtractEdgesFilter {
    fn name(&self) -> &str {
        "ExtractEdgesFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let grid_pts = GridPoints::from_dataset(input)?;
        let n_pts = grid_pts.len();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        let mut edges: Vec<(u32, u32)> = Vec::new();

        let mut push_edge = |a: u32, b: u32| {
            if a == b || (a as usize) >= n_pts || (b as usize) >= n_pts {
                return;
            }
            let key = canonical_edge_key(a, b);
            if seen.insert(key) {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edges.push((lo, hi));
            }
        };

        match input {
            DataSet::PolyData(pd) => {
                for line in &pd.lines {
                    for w in line.windows(2) {
                        push_edge(w[0], w[1]);
                    }
                }
                for poly in &pd.polys {
                    let m = poly.len();
                    if m >= 2 {
                        for k in 0..m {
                            push_edge(poly[k], poly[(k + 1) % m]);
                        }
                    }
                }
                for strip in &pd.strips {
                    for k in 0..strip.len().saturating_sub(2) {
                        let a = strip[k];
                        let b = strip[k + 1];
                        let c = strip[k + 2];
                        push_edge(a, b);
                        push_edge(b, c);
                        push_edge(c, a);
                    }
                }
            }
            DataSet::UnstructuredGrid(ug) => {
                for c in 0..ug.num_cells() {
                    emit_cell_edges(ug.cell_types[c], ug.cell_points(c), &mut push_edge);
                }
            }
            DataSet::ImageData(img) => {
                for (ct, pts) in structured_cells(img.dimensions, true) {
                    emit_cell_edges(ct, &pts, &mut push_edge);
                }
            }
            DataSet::RectilinearGrid(rg) => {
                for (ct, pts) in structured_cells(rg.dimensions(), true) {
                    emit_cell_edges(ct, &pts, &mut push_edge);
                }
            }
            DataSet::StructuredGrid(sg) => {
                for (ct, pts) in structured_cells(sg.dimensions, false) {
                    emit_cell_edges(ct, &pts, &mut push_edge);
                }
            }
            DataSet::MultiBlock(_) => unreachable!(),
        }

        edges.sort_unstable();
        let lines: Vec<Vec<u32>> = edges.into_iter().map(|(a, b)| vec![a, b]).collect();

        Ok(DataSet::PolyData(PolyData {
            points: grid_pts.to_vec_par(),
            lines,
            point_data: input.point_data().cloned().unwrap_or_default(),
            field_data: input.field_data().cloned().unwrap_or_default(),
            ..Default::default()
        }))
    }
}

fn emit_cell_edges(ct: CellType, pts: &[u32], push: &mut impl FnMut(u32, u32)) {
    match ct {
        CellType::Line | CellType::PolyLine => {
            for w in pts.windows(2) {
                push(w[0], w[1]);
            }
        }
        CellType::Triangle | CellType::Quad | CellType::Polygon => {
            let m = pts.len();
            for k in 0..m {
                push(pts[k], pts[(k + 1) % m]);
            }
        }
        CellType::Pixel if pts.len() >= 4 => {
            push(pts[0], pts[1]);
            push(pts[1], pts[3]);
            push(pts[3], pts[2]);
            push(pts[2], pts[0]);
        }
        CellType::Tetra if pts.len() >= 4 => {
            for &(i, j) in &[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)] {
                push(pts[i], pts[j]);
            }
        }
        CellType::Hexahedron if pts.len() >= 8 => {
            for &(i, j) in &[
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ] {
                push(pts[i], pts[j]);
            }
        }
        CellType::Voxel if pts.len() >= 8 => {
            for &(i, j) in &[
                (0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ] {
                push(pts[i], pts[j]);
            }
        }
        CellType::Wedge if pts.len() >= 6 => {
            for &(i, j) in &[
                (0, 1), (1, 2), (2, 0),
                (3, 4), (4, 5), (5, 3),
                (0, 3), (1, 4), (2, 5),
            ] {
                push(pts[i], pts[j]);
            }
        }
        CellType::Pyramid if pts.len() >= 5 => {
            for &(i, j) in &[
                (0, 1), (1, 2), (2, 3), (3, 0),
                (0, 4), (1, 4), (2, 4), (3, 4),
            ] {
                push(pts[i], pts[j]);
            }
        }
        _ => {}
    }
}
