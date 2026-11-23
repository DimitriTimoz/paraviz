//! Connected component labeling and extraction filter (`ConnectivityFilter`,
//! equivalent to `vtkConnectivityFilter`).

use crate::util::{compact_polydata_points, compact_unstructured_points};
use paraviz_core::{
    ArrayData, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError, PolyData,
    UnstructuredGrid,
};
use serde::{Deserialize, Serialize};

/// Assigns a `RegionId` point and cell scalar array to each connected topological
/// component of a `PolyData` or `UnstructuredGrid`, and optionally extracts only the largest region.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConnectivityFilter {
    /// When `true`, keeps only the topological component with the most cells.
    pub extract_largest: bool,
}

impl ConnectivityFilter {
    /// Creates a `ConnectivityFilter`.
    pub fn new(extract_largest: bool) -> Self {
        Self { extract_largest }
    }
}

impl Filter for ConnectivityFilter {
    fn name(&self) -> &str {
        "ConnectivityFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.apply_polydata(pd))),
            DataSet::UnstructuredGrid(ug) => Ok(DataSet::UnstructuredGrid(self.apply_unstructured(ug))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => Err(PipelineError::ExecutionFailed(
                "ConnectivityFilter requires PolyData or UnstructuredGrid input".into(),
            )),
        }
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

fn assign_point_regions<'a>(
    n_pts: usize,
    cell_vertices: impl Iterator<Item = &'a [u32]>,
) -> (Vec<i32>, usize) {
    let mut dsu = DisjointSet::new(n_pts);
    for verts in cell_vertices {
        let mut first_valid: Option<usize> = None;
        for &vi in verts {
            let idx = vi as usize;
            if idx < n_pts {
                if let Some(root) = first_valid {
                    dsu.union(root, idx);
                } else {
                    first_valid = Some(idx);
                }
            }
        }
    }

    let mut root_to_id = vec![-1i32; n_pts];
    let mut point_regions = vec![0i32; n_pts];
    let mut next_id = 0i32;
    for (i, slot) in point_regions.iter_mut().enumerate() {
        let r = dsu.find(i);
        if root_to_id[r] < 0 {
            root_to_id[r] = next_id;
            next_id += 1;
        }
        *slot = root_to_id[r];
    }
    (point_regions, next_id as usize)
}

fn find_largest_region(
    num_regions: usize,
    point_regions: &[i32],
    cell_regions: &[i32],
) -> i32 {
    if num_regions == 0 {
        return 0;
    }
    if !cell_regions.is_empty() {
        let mut cell_counts = vec![0usize; num_regions];
        for &r in cell_regions {
            if r >= 0 {
                cell_counts[r as usize] += 1;
            }
        }
        let mut best_id = 0usize;
        for (id, &cnt) in cell_counts.iter().enumerate() {
            if cnt > cell_counts[best_id] {
                best_id = id;
            }
        }
        return best_id as i32;
    }
    let mut pt_counts = vec![0usize; num_regions];
    for &r in point_regions {
        if r >= 0 {
            pt_counts[r as usize] += 1;
        }
    }
    let mut best_id = 0usize;
    for (id, &cnt) in pt_counts.iter().enumerate() {
        if cnt > pt_counts[best_id] {
            best_id = id;
        }
    }
    best_id as i32
}

impl ConnectivityFilter {
    fn apply_polydata(&self, pd: &PolyData) -> PolyData {
        let n_pts = pd.points.len();
        let all_cells = pd
            .verts
            .iter()
            .map(std::slice::from_ref)
            .chain(pd.lines.iter().map(Vec::as_slice))
            .chain(pd.polys.iter().map(Vec::as_slice))
            .chain(pd.strips.iter().map(Vec::as_slice));

        let (point_regions, num_regions) = assign_point_regions(n_pts, all_cells);

        let region_of_cell = |verts: &[u32]| -> i32 {
            verts
                .iter()
                .find_map(|&vi| point_regions.get(vi as usize).copied())
                .unwrap_or(0)
        };

        let mut cell_regions = Vec::with_capacity(pd.num_cells());
        for &v in &pd.verts {
            cell_regions.push(region_of_cell(&[v]));
        }
        for line in &pd.lines {
            cell_regions.push(region_of_cell(line));
        }
        for poly in &pd.polys {
            cell_regions.push(region_of_cell(poly));
        }
        for strip in &pd.strips {
            cell_regions.push(region_of_cell(strip));
        }

        if !self.extract_largest {
            let mut out = pd.clone();
            out.point_data
                .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(point_regions)));
            out.cell_data
                .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(cell_regions)));
            return out;
        }

        let target = find_largest_region(num_regions, &point_regions, &cell_regions);
        let mut verts = Vec::new();
        let mut lines = Vec::new();
        let mut polys = Vec::new();
        let mut strips = Vec::new();
        let mut kept_cells = Vec::new();
        let mut cidx = 0usize;

        for &v in &pd.verts {
            if cell_regions[cidx] == target {
                verts.push(v);
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for line in &pd.lines {
            if cell_regions[cidx] == target {
                lines.push(line.clone());
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for poly in &pd.polys {
            if cell_regions[cidx] == target {
                polys.push(poly.clone());
                kept_cells.push(cidx);
            }
            cidx += 1;
        }
        for strip in &pd.strips {
            if cell_regions[cidx] == target {
                strips.push(strip.clone());
                kept_cells.push(cidx);
            }
            cidx += 1;
        }

        let mut cell_data = if pd.cell_data.is_empty() {
            FieldData::new()
        } else {
            pd.cell_data.gather(&kept_cells)
        };
        cell_data.set_scalars(DataArray::new(
            "RegionId",
            1,
            ArrayData::I32(vec![0; kept_cells.len()]),
        ));

        let mut out = compact_polydata_points(PolyData {
            points: pd.points.clone(),
            verts,
            lines,
            polys,
            strips,
            point_data: pd.point_data.clone(),
            cell_data,
            field_data: pd.field_data.clone(),
        });
        let out_pts = out.num_points();
        out.point_data
            .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(vec![0; out_pts])));
        out
    }

    fn apply_unstructured(&self, ug: &UnstructuredGrid) -> UnstructuredGrid {
        let n_pts = ug.points.len();
        let n_cells = ug.num_cells();
        let (point_regions, num_regions) =
            assign_point_regions(n_pts, (0..n_cells).map(|c| ug.cell_points(c)));

        let cell_regions: Vec<i32> = (0..n_cells)
            .map(|c| {
                ug.cell_points(c)
                    .iter()
                    .find_map(|&vi| point_regions.get(vi as usize).copied())
                    .unwrap_or(0)
            })
            .collect();

        if !self.extract_largest {
            let mut out = ug.clone();
            out.point_data
                .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(point_regions)));
            out.cell_data
                .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(cell_regions)));
            return out;
        }

        let target = find_largest_region(num_regions, &point_regions, &cell_regions);
        let mut out = UnstructuredGrid::new();
        out.points = ug.points.clone();
        out.point_data = ug.point_data.clone();
        out.field_data = ug.field_data.clone();
        let mut kept_cells = Vec::new();
        for (c, &region) in cell_regions.iter().enumerate() {
            if region == target {
                out.push_cell(ug.cell_types[c], ug.cell_points(c));
                kept_cells.push(c);
            }
        }
        if !ug.cell_data.is_empty() {
            out.cell_data = ug.cell_data.gather(&kept_cells);
        }
        out.cell_data.set_scalars(DataArray::new(
            "RegionId",
            1,
            ArrayData::I32(vec![0; kept_cells.len()]),
        ));
        let mut out = compact_unstructured_points(out);
        let out_pts = out.num_points();
        out.point_data
            .set_scalars(DataArray::new("RegionId", 1, ArrayData::I32(vec![0; out_pts])));
        out
    }
}
