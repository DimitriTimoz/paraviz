//! Polygon and triangle-strip triangulation filter (`TriangulateFilter`,
//! equivalent to `vtkTriangleFilter`).

use crate::surface::extract_surface;
use paraviz_core::{
    DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError, PolyData,
};
use serde::{Deserialize, Serialize};

/// Converts polygonal faces (quads, n-gons) and triangle strips into explicit triangles,
/// preserving `verts`, `lines`, `point_data`, and remapping `cell_data`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriangulateFilter {
    /// Keep vertex and polyline cells in the output alongside triangles.
    pub pass_verts_and_lines: bool,
}

impl Default for TriangulateFilter {
    fn default() -> Self {
        Self {
            pass_verts_and_lines: true,
        }
    }
}

impl TriangulateFilter {
    /// Creates a `TriangulateFilter` with default settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Filter for TriangulateFilter {
    fn name(&self) -> &str {
        "TriangulateFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        match input {
            DataSet::PolyData(pd) => Ok(DataSet::PolyData(self.triangulate_polydata(pd))),
            DataSet::MultiBlock(mb) => {
                let mut out = MultiBlockDataSet::new();
                for (name, block) in &mb.blocks {
                    out.add_block(name.clone(), self.apply(block)?);
                }
                Ok(DataSet::MultiBlock(out))
            }
            _ => {
                let surf = extract_surface(input)?;
                Ok(DataSet::PolyData(self.triangulate_polydata(&surf)))
            }
        }
    }
}

impl TriangulateFilter {
    fn triangulate_polydata(&self, pd: &PolyData) -> PolyData {
        let mut verts = Vec::new();
        let mut lines = Vec::new();
        let mut polys = Vec::with_capacity(pd.polys.len());
        let mut cell_map: Vec<usize> = Vec::new();
        let mut cidx = 0usize;

        for &v in &pd.verts {
            if self.pass_verts_and_lines {
                verts.push(v);
                cell_map.push(cidx);
            }
            cidx += 1;
        }
        for line in &pd.lines {
            if self.pass_verts_and_lines {
                lines.push(line.clone());
                cell_map.push(cidx);
            }
            cidx += 1;
        }
        for poly in &pd.polys {
            if poly.len() >= 3 {
                for k in 1..(poly.len() - 1) {
                    polys.push(vec![poly[0], poly[k], poly[k + 1]]);
                    cell_map.push(cidx);
                }
            }
            cidx += 1;
        }
        for strip in &pd.strips {
            if strip.len() >= 3 {
                for k in 0..(strip.len() - 2) {
                    if k % 2 == 0 {
                        polys.push(vec![strip[k], strip[k + 1], strip[k + 2]]);
                    } else {
                        polys.push(vec![strip[k + 1], strip[k], strip[k + 2]]);
                    }
                    cell_map.push(cidx);
                }
            }
            cidx += 1;
        }

        let cell_data = if pd.cell_data.is_empty() {
            FieldData::new()
        } else {
            pd.cell_data.gather(&cell_map)
        };

        PolyData {
            points: pd.points.clone(),
            verts,
            lines,
            polys,
            strips: Vec::new(),
            point_data: pd.point_data.clone(),
            cell_data,
            field_data: pd.field_data.clone(),
        }
    }
}
