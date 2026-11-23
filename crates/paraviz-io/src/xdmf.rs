//! XDMF (`.xdmf`, `.xmf`) XML scientific dataset reader supporting XDMF v1/v2/v3,
//! inline XML / CDATA / Binary (with Precision 1/2/4/8, Big/Little Endian, Seek, ZLib),
//! XPath `Reference` resolution, `HyperSlab`, structured/rectilinear/uniform/unstructured/mixed
//! topologies, and spatial/temporal grid collections.

use crate::error::{IoError, Result};
use flate2::read::ZlibDecoder;
use paraviz_core::{
    ArrayData, CellType, DataArray, DataSet, FieldData, ImageData, MultiBlockDataSet,
    RectilinearGrid, StructuredGrid, UnstructuredGrid,
};
use quick_xml::events::Event;
use quick_xml::{Reader, XmlVersion};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Reads an XDMF (`.xdmf` / `.xmf`) file into a [`DataSet`].
pub fn read_xdmf(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let xml = fs::read_to_string(path)?;
    read_xdmf_str(&xml, base_dir)
}

/// Reads an XDMF XML string using `base_dir` for external binary / XInclude references.
pub fn read_xdmf_str(xml: &str, base_dir: &Path) -> Result<DataSet> {
    let root = parse_xml_tree(xml, base_dir, 0)?;
    let mut named_items: HashMap<String, XmlNode> = HashMap::new();
    collect_named_data_items(&root, &mut named_items);

    let ctx = XdmfContext {
        base_dir,
        root: &root,
        named_items: &named_items,
        hdf5_cache: RefCell::new(HashMap::new()),
    };

    // Find all top-level <Grid> elements under <Domain> (or directly under <Xdmf>)
    let mut grids = Vec::new();
    find_top_level_grids(&root, &mut grids);

    if grids.is_empty() {
        return Err(IoError::Parse(
            "XDMF file contains no <Grid> elements".into(),
        ));
    }

    if grids.len() == 1 {
        parse_grid_node(grids[0], &ctx)
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (idx, g) in grids.iter().enumerate() {
            let name = g
                .attr("Name")
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("Grid_{idx}"));
            let ds = parse_grid_node(g, &ctx)?;
            mb.add_block(name, ds);
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

#[derive(Debug, Clone)]
struct XmlNode {
    name: String,
    attrs: Vec<(String, String)>,
    text: String,
    children: Vec<XmlNode>,
}

impl XmlNode {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    fn local_name(&self) -> &str {
        self.name
            .rsplit_once(':')
            .map(|(_, local)| local)
            .unwrap_or(&self.name)
    }

    fn child(&self, name: &str) -> Option<&XmlNode> {
        self.children
            .iter()
            .find(|c| c.local_name().eq_ignore_ascii_case(name))
    }

    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlNode> {
        self.children
            .iter()
            .filter(move |c| c.local_name().eq_ignore_ascii_case(name))
    }

    fn is_data_item(&self) -> bool {
        let ln = self.local_name();
        ln.eq_ignore_ascii_case("DataItem") || ln.eq_ignore_ascii_case("DataStructure")
    }

    fn data_items(&self) -> impl Iterator<Item = &XmlNode> {
        self.children.iter().filter(|c| c.is_data_item())
    }
}

fn parse_xml_tree(xml: &str, base_dir: &Path, depth: usize) -> Result<XmlNode> {
    if depth > 16 {
        return Err(IoError::Parse("XDMF XInclude recursion depth exceeded".into()));
    }
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut stack: Vec<XmlNode> = vec![XmlNode {
        name: "__root__".into(),
        attrs: Vec::new(),
        text: String::new(),
        children: Vec::new(),
    }];

    loop {
        match reader.read_event()? {
            Event::Start(e) => {
                let name = e.name().as_ref().to_string();
                let mut attrs = Vec::new();
                for a in e.attributes().flatten() {
                    let k = a.key.as_ref().to_string();
                    let v = a.normalized_value(XmlVersion::default())?.into_owned();
                    attrs.push((k, v));
                }
                stack.push(XmlNode {
                    name,
                    attrs,
                    text: String::new(),
                    children: Vec::new(),
                });
            }
            Event::Empty(e) => {
                let name = e.name().as_ref().to_string();
                let mut attrs = Vec::new();
                for a in e.attributes().flatten() {
                    let k = a.key.as_ref().to_string();
                    let v = a.normalized_value(XmlVersion::default())?.into_owned();
                    attrs.push((k, v));
                }
                let node = XmlNode {
                    name,
                    attrs,
                    text: String::new(),
                    children: Vec::new(),
                };
                if node.local_name().eq_ignore_ascii_case("include")
                    && let Some(href) = node.attr("href")
                {
                    let inc_path = base_dir.join(href);
                    if let Ok(inc_xml) = fs::read_to_string(&inc_path) {
                        let inc_dir = inc_path.parent().unwrap_or(base_dir);
                        let inc_root = parse_xml_tree(&inc_xml, inc_dir, depth + 1)?;
                        if let Some(top) = stack.last_mut() {
                            top.children.extend(inc_root.children);
                        }
                        continue;
                    }
                }
                if let Some(top) = stack.last_mut() {
                    top.children.push(node);
                }
            }
            Event::Text(t) => {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&t);
                }
            }
            Event::CData(t) => {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(t.as_ref());
                }
            }
            Event::End(_) => {
                if stack.len() > 1 {
                    let finished = stack.pop().unwrap();
                    if finished.local_name().eq_ignore_ascii_case("include")
                        && let Some(href) = finished.attr("href")
                    {
                        let inc_path = base_dir.join(href);
                        if let Ok(inc_xml) = fs::read_to_string(&inc_path) {
                            let inc_dir = inc_path.parent().unwrap_or(base_dir);
                            let inc_root = parse_xml_tree(&inc_xml, inc_dir, depth + 1)?;
                            if let Some(top) = stack.last_mut() {
                                top.children.extend(inc_root.children);
                            }
                            continue;
                        }
                    }
                    if let Some(top) = stack.last_mut() {
                        top.children.push(finished);
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }

    Ok(stack.pop().unwrap())
}

fn collect_named_data_items(node: &XmlNode, map: &mut HashMap<String, XmlNode>) {
    if node.is_data_item()
        && let Some(name) = node.attr("Name")
    {
        map.insert(name.to_string(), node.clone());
    }
    for child in &node.children {
        collect_named_data_items(child, map);
    }
}

fn find_top_level_grids<'a>(node: &'a XmlNode, out: &mut Vec<&'a XmlNode>) {
    for child in &node.children {
        if child.local_name().eq_ignore_ascii_case("Grid") {
            out.push(child);
        } else if child.local_name().eq_ignore_ascii_case("Xdmf")
            || child.local_name().eq_ignore_ascii_case("Domain")
        {
            find_top_level_grids(child, out);
        }
    }
}

struct XdmfContext<'a> {
    base_dir: &'a Path,
    root: &'a XmlNode,
    named_items: &'a HashMap<String, XmlNode>,
    hdf5_cache: RefCell<HashMap<PathBuf, hdf5_reader::Hdf5File>>,
}

impl<'a> XdmfContext<'a> {
    fn resolve_reference(&self, node: &'a XmlNode) -> Option<&'a XmlNode> {
        let ref_attr = node.attr("Reference")?;
        let xpath = if ref_attr.eq_ignore_ascii_case("XML") {
            node.text.trim()
        } else {
            ref_attr.trim()
        };
        if xpath.is_empty() {
            return None;
        }
        // Extract [@Name='foo'] or [@Name="foo"] if present
        if let Some(pos) = xpath.rfind("@Name=") {
            let after = &xpath[pos + 6..];
            let trimmed = after.trim_start_matches(['\'', '"']);
            if let Some(end) = trimmed.find(['\'', '"']) {
                let name = &trimmed[..end];
                if let Some(found) = self.named_items.get(name) {
                    return Some(found);
                }
            }
        }
        // Fallback: simple slash path traversal
        let parts: Vec<&str> = xpath.trim_start_matches('/').split('/').collect();
        let mut cur = self.root;
        for part in parts {
            let tag = part.split('[').next().unwrap_or(part);
            if tag.is_empty() {
                continue;
            }
            cur = cur.children.iter().find(|c| c.local_name().eq_ignore_ascii_case(tag))?;
        }
        Some(cur)
    }
}

fn collect_child_grids<'a>(node: &'a XmlNode, out: &mut Vec<&'a XmlNode>) {
    for child in &node.children {
        if child.local_name().eq_ignore_ascii_case("Grid") {
            out.push(child);
        } else if child.local_name().eq_ignore_ascii_case("Xdmf")
            || child.local_name().eq_ignore_ascii_case("Domain")
        {
            collect_child_grids(child, out);
        }
    }
}

fn parse_grid_node(grid_node: &XmlNode, ctx: &XdmfContext<'_>) -> Result<DataSet> {
    let grid_type = grid_node
        .attr("GridType")
        .or_else(|| grid_node.attr("Type"))
        .unwrap_or("Uniform");

    if grid_type.eq_ignore_ascii_case("Collection") || grid_type.eq_ignore_ascii_case("Tree") {
        let coll_type = grid_node.attr("CollectionType").unwrap_or("Spatial");
        let mut child_grids: Vec<&XmlNode> = Vec::new();
        collect_child_grids(grid_node, &mut child_grids);
        if child_grids.is_empty() {
            return Ok(DataSet::UnstructuredGrid(UnstructuredGrid::new()));
        }
        if coll_type.eq_ignore_ascii_case("Temporal") {
            // Return first valid time step for single-dataset read_xdmf
            return parse_grid_node(child_grids[0], ctx);
        }
        if child_grids.len() == 1 {
            return parse_grid_node(child_grids[0], ctx);
        }
        let mut mb = MultiBlockDataSet::new();
        for (i, cg) in child_grids.iter().enumerate() {
            let name = cg
                .attr("Name")
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("Block_{i}"));
            mb.add_block(name, parse_grid_node(cg, ctx)?);
        }
        return Ok(DataSet::MultiBlock(mb));
    }

    let topo_node = grid_node
        .child("Topology")
        .map(|t| ctx.resolve_reference(t).unwrap_or(t));
    let geom_node = grid_node
        .child("Geometry")
        .map(|g| ctx.resolve_reference(g).unwrap_or(g));

    let topo_type = topo_node
        .and_then(|t| t.attr("TopologyType").or_else(|| t.attr("Type")))
        .unwrap_or("");
    let topo_dims = topo_node
        .and_then(|t| t.attr("Dimensions"))
        .map(parse_dims)
        .unwrap_or_default();
    let nodes_per_elem = topo_node
        .and_then(|t| t.attr("NodesPerElement"))
        .and_then(|s| s.trim().parse::<usize>().ok());

    let geom_type = geom_node
        .and_then(|g| g.attr("GeometryType").or_else(|| g.attr("Type")))
        .unwrap_or("XYZ");

    let mut point_data = FieldData::new();
    let mut cell_data = FieldData::new();
    let mut field_data = FieldData::new();

    for attr_node in grid_node.children_named("Attribute") {
        let name = attr_node
            .attr("Name")
            .filter(|s| !s.is_empty())
            .unwrap_or("Attribute")
            .to_string();
        let center = attr_node.attr("Center").unwrap_or("Node");
        let attr_type = attr_node
            .attr("AttributeType")
            .or_else(|| attr_node.attr("Type"))
            .unwrap_or("Scalar");

        let Some(di) = attr_node.data_items().next() else {
            continue;
        };
        let (vals, di_dims) = read_data_item_f64(di, ctx)?;
        let num_components = infer_num_components(attr_type, &di_dims, vals.len());
        let arr = make_data_array(name, num_components, vals);

        if center.eq_ignore_ascii_case("Cell") {
            cell_data.add_array(arr);
        } else if center.eq_ignore_ascii_case("Grid") {
            field_data.add_array(arr);
        } else {
            point_data.add_array(arr);
        }
    }

    // Handle structured/uniform/rectilinear grids
    if is_corect_topo(topo_type) || geom_type.eq_ignore_ascii_case("ORIGIN_DXDYDZ") || geom_type.eq_ignore_ascii_case("ORIGIN_DXDY") {
        let dims = xdmf_dims_to_vtk_3d(&topo_dims);
        let (origin, spacing) = parse_origin_spacing(geom_node, ctx)?;
        let mut img = ImageData::new(dims, origin, spacing);
        img.point_data = point_data;
        img.cell_data = cell_data;
        img.field_data = field_data;
        return Ok(DataSet::ImageData(img));
    }

    if is_rect_topo(topo_type) || geom_type.eq_ignore_ascii_case("VXVYVZ") || geom_type.eq_ignore_ascii_case("VXVY") {
        let (x_coords, y_coords, z_coords) = parse_rectilinear_coords(geom_node, ctx)?;
        let rect = RectilinearGrid {
            x_coords,
            y_coords,
            z_coords,
            point_data,
            cell_data,
            field_data,
        };
        return Ok(DataSet::RectilinearGrid(rect));
    }

    let points = parse_geometry_points(geom_node, geom_type, ctx)?;

    if is_smesh_topo(topo_type) {
        let dims = xdmf_dims_to_vtk_3d(&topo_dims);
        let sg = StructuredGrid {
            dimensions: dims,
            points,
            point_data,
            cell_data,
            field_data,
        };
        return Ok(DataSet::StructuredGrid(sg));
    }

    // UnstructuredGrid
    let mut grid = UnstructuredGrid::new();
    grid.points = points;
    grid.point_data = point_data;
    grid.cell_data = cell_data;
    grid.field_data = field_data;

    if let Some(tn) = topo_node {
        if let Some(di) = tn.data_items().next() {
            let (raw_topo, _) = read_data_item_u32(di, ctx)?;
            if topo_type.eq_ignore_ascii_case("Mixed") {
                push_mixed_cells(&mut grid, &raw_topo);
            } else if topo_type.eq_ignore_ascii_case("Polyvertex") {
                for &idx in &raw_topo {
                    grid.push_cell(CellType::Vertex, &[idx]);
                }
            } else if let Some((ct, stride)) = map_topology_type(topo_type, nodes_per_elem) {
                if stride > 0 {
                    for chunk in raw_topo.chunks_exact(stride) {
                        grid.push_cell(ct, chunk);
                    }
                }
            } else {
                // Unknown topology name: if raw_topo is present, try Triangle (3)
                for chunk in raw_topo.chunks_exact(3) {
                    grid.push_cell(CellType::Triangle, chunk);
                }
            }
        } else if topo_type.eq_ignore_ascii_case("Polyvertex") {
            for i in 0..grid.points.len() as u32 {
                grid.push_cell(CellType::Vertex, &[i]);
            }
        }
    }

    Ok(DataSet::UnstructuredGrid(grid))
}

fn is_corect_topo(t: &str) -> bool {
    t.eq_ignore_ascii_case("3DCoRectMesh")
        || t.eq_ignore_ascii_case("3DCoRectilinearMesh")
        || t.eq_ignore_ascii_case("2DCoRectMesh")
        || t.eq_ignore_ascii_case("2DCoRectilinearMesh")
}

fn is_rect_topo(t: &str) -> bool {
    t.eq_ignore_ascii_case("3DRectMesh")
        || t.eq_ignore_ascii_case("3DRectilinearMesh")
        || t.eq_ignore_ascii_case("2DRectMesh")
        || t.eq_ignore_ascii_case("2DRectilinearMesh")
}

fn is_smesh_topo(t: &str) -> bool {
    t.eq_ignore_ascii_case("3DSMesh") || t.eq_ignore_ascii_case("2DSMesh")
}

fn parse_dims(s: &str) -> Vec<usize> {
    s.split_whitespace()
        .filter_map(|p| p.parse::<usize>().ok())
        .collect()
}

/// XDMF Dimensions are in C-order `[nz, ny, nx]` (or `[ny, nx]`). VTK `ImageData` / `StructuredGrid`
/// expects `[nx, ny, nz]`.
fn xdmf_dims_to_vtk_3d(dims: &[usize]) -> [usize; 3] {
    match dims.len() {
        0 => [1, 1, 1],
        1 => [dims[0].max(1), 1, 1],
        2 => [dims[1].max(1), dims[0].max(1), 1],
        _ => [dims[2].max(1), dims[1].max(1), dims[0].max(1)],
    }
}

fn parse_origin_spacing(
    geom_node: Option<&XmlNode>,
    ctx: &XdmfContext<'_>,
) -> Result<([f64; 3], [f64; 3])> {
    let Some(gn) = geom_node else {
        return Ok(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    };
    let items: Vec<&XmlNode> = gn.data_items().collect();
    let origin = if let Some(di) = items.first() {
        let (v, _) = read_data_item_f64(di, ctx)?;
        match v.len() {
            0 | 1 => [0.0, 0.0, 0.0],
            2 => [v[1], v[0], 0.0],
            _ => [v[2], v[1], v[0]],
        }
    } else {
        [0.0, 0.0, 0.0]
    };
    let spacing = if let Some(di) = items.get(1) {
        let (v, _) = read_data_item_f64(di, ctx)?;
        match v.len() {
            0 | 1 => [1.0, 1.0, 1.0],
            2 => [v[1], v[0], 1.0],
            _ => [v[2], v[1], v[0]],
        }
    } else {
        [1.0, 1.0, 1.0]
    };
    Ok((origin, spacing))
}

fn parse_rectilinear_coords(
    geom_node: Option<&XmlNode>,
    ctx: &XdmfContext<'_>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let Some(gn) = geom_node else {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    };
    let items: Vec<&XmlNode> = gn.data_items().collect();
    let x = if let Some(di) = items.first() {
        read_data_item_f64(di, ctx)?.0.into_iter().map(|v| v as f32).collect()
    } else {
        vec![0.0]
    };
    let y = if let Some(di) = items.get(1) {
        read_data_item_f64(di, ctx)?.0.into_iter().map(|v| v as f32).collect()
    } else {
        vec![0.0]
    };
    let z = if let Some(di) = items.get(2) {
        read_data_item_f64(di, ctx)?.0.into_iter().map(|v| v as f32).collect()
    } else {
        vec![0.0]
    };
    Ok((x, y, z))
}

fn parse_geometry_points(
    geom_node: Option<&XmlNode>,
    geom_type: &str,
    ctx: &XdmfContext<'_>,
) -> Result<Vec<[f32; 3]>> {
    let Some(gn) = geom_node else {
        return Ok(Vec::new());
    };
    let items: Vec<&XmlNode> = gn.data_items().collect();
    if items.is_empty() {
        return Ok(Vec::new());
    }

    if geom_type.eq_ignore_ascii_case("X_Y_Z") {
        let x = items.first().map(|d| read_data_item_f64(d, ctx)).transpose()?.map(|r| r.0).unwrap_or_default();
        let y = items.get(1).map(|d| read_data_item_f64(d, ctx)).transpose()?.map(|r| r.0).unwrap_or_default();
        let z = items.get(2).map(|d| read_data_item_f64(d, ctx)).transpose()?.map(|r| r.0).unwrap_or_default();
        let n = x.len().min(y.len()).min(z.len());
        return Ok((0..n).map(|i| [x[i] as f32, y[i] as f32, z[i] as f32]).collect());
    }

    if geom_type.eq_ignore_ascii_case("X_Y") {
        let x = items.first().map(|d| read_data_item_f64(d, ctx)).transpose()?.map(|r| r.0).unwrap_or_default();
        let y = items.get(1).map(|d| read_data_item_f64(d, ctx)).transpose()?.map(|r| r.0).unwrap_or_default();
        let n = x.len().min(y.len());
        return Ok((0..n).map(|i| [x[i] as f32, y[i] as f32, 0.0]).collect());
    }

    if geom_type.eq_ignore_ascii_case("XY") {
        let (vals, _) = read_data_item_f64(items[0], ctx)?;
        return Ok(vals.chunks_exact(2).map(|c| [c[0] as f32, c[1] as f32, 0.0]).collect());
    }

    // Default: XYZ (or X_Y_Z if 3 DataItems were provided even without GeometryType="X_Y_Z")
    if items.len() == 3 {
        let x = read_data_item_f64(items[0], ctx)?.0;
        let y = read_data_item_f64(items[1], ctx)?.0;
        let z = read_data_item_f64(items[2], ctx)?.0;
        let n = x.len().min(y.len()).min(z.len());
        return Ok((0..n).map(|i| [x[i] as f32, y[i] as f32, z[i] as f32]).collect());
    }

    let (vals, _) = read_data_item_f64(items[0], ctx)?;
    Ok(vals
        .chunks_exact(3)
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect())
}

fn map_topology_type(topo_type: &str, nodes_per_elem: Option<usize>) -> Option<(CellType, usize)> {
    match topo_type.to_ascii_lowercase().as_str() {
        "triangle" => Some((CellType::Triangle, 3)),
        "quadrilateral" => Some((CellType::Quad, 4)),
        "tetrahedron" => Some((CellType::Tetra, 4)),
        "pyramid" => Some((CellType::Pyramid, 5)),
        "wedge" => Some((CellType::Wedge, 6)),
        "hexahedron" => Some((CellType::Hexahedron, 8)),
        "polyline" => {
            let n = nodes_per_elem.unwrap_or(2);
            if n == 2 {
                Some((CellType::Line, 2))
            } else {
                Some((CellType::PolyLine, n))
            }
        }
        "polygon" => Some((CellType::Polygon, nodes_per_elem.unwrap_or(0))),
        "edge_3" => Some((CellType::QuadraticEdge, 3)),
        "tri_6" => Some((CellType::QuadraticTriangle, 6)),
        "quad_8" => Some((CellType::QuadraticQuad, 8)),
        "quad_9" => Some((CellType::BiQuadraticQuad, 9)),
        "tet_10" => Some((CellType::QuadraticTetra, 10)),
        "pyramid_13" => Some((CellType::QuadraticPyramid, 13)),
        "wedge_15" => Some((CellType::QuadraticWedge, 15)),
        "hex_20" => Some((CellType::QuadraticHexahedron, 20)),
        "hex_27" => Some((CellType::TriQuadraticHexahedron, 27)),
        _ => None,
    }
}

fn push_mixed_cells(grid: &mut UnstructuredGrid, raw: &[u32]) {
    let mut i = 0;
    while i < raw.len() {
        let code = raw[i];
        i += 1;
        match code {
            // 0x1: Polyvertex (followed by count)
            1 => {
                if i >= raw.len() { break; }
                let n = raw[i] as usize;
                i += 1;
                if i + n > raw.len() { break; }
                grid.push_cell(CellType::PolyVertex, &raw[i..i + n]);
                i += n;
            }
            // 0x2: Polyline (followed by count)
            2 => {
                if i >= raw.len() { break; }
                let n = raw[i] as usize;
                i += 1;
                if i + n > raw.len() { break; }
                let ct = if n == 2 { CellType::Line } else { CellType::PolyLine };
                grid.push_cell(ct, &raw[i..i + n]);
                i += n;
            }
            // 0x3: Polygon (followed by count)
            3 => {
                if i >= raw.len() { break; }
                let n = raw[i] as usize;
                i += 1;
                if i + n > raw.len() { break; }
                grid.push_cell(CellType::Polygon, &raw[i..i + n]);
                i += n;
            }
            4 => {
                if i + 3 > raw.len() { break; }
                grid.push_cell(CellType::Triangle, &raw[i..i + 3]);
                i += 3;
            }
            5 => {
                if i + 4 > raw.len() { break; }
                grid.push_cell(CellType::Quad, &raw[i..i + 4]);
                i += 4;
            }
            6 => {
                if i + 4 > raw.len() { break; }
                grid.push_cell(CellType::Tetra, &raw[i..i + 4]);
                i += 4;
            }
            7 => {
                if i + 5 > raw.len() { break; }
                grid.push_cell(CellType::Pyramid, &raw[i..i + 5]);
                i += 5;
            }
            8 => {
                if i + 6 > raw.len() { break; }
                grid.push_cell(CellType::Wedge, &raw[i..i + 6]);
                i += 6;
            }
            9 => {
                if i + 8 > raw.len() { break; }
                grid.push_cell(CellType::Hexahedron, &raw[i..i + 8]);
                i += 8;
            }
            0x22 => {
                if i + 3 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticEdge, &raw[i..i + 3]);
                i += 3;
            }
            0x23 => {
                if i + 9 > raw.len() { break; }
                grid.push_cell(CellType::BiQuadraticQuad, &raw[i..i + 9]);
                i += 9;
            }
            0x24 => {
                if i + 6 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticTriangle, &raw[i..i + 6]);
                i += 6;
            }
            0x25 => {
                if i + 8 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticQuad, &raw[i..i + 8]);
                i += 8;
            }
            0x26 => {
                if i + 10 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticTetra, &raw[i..i + 10]);
                i += 10;
            }
            0x27 => {
                if i + 13 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticPyramid, &raw[i..i + 13]);
                i += 13;
            }
            0x28 => {
                if i + 15 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticWedge, &raw[i..i + 15]);
                i += 15;
            }
            0x30 => {
                if i + 20 > raw.len() { break; }
                grid.push_cell(CellType::QuadraticHexahedron, &raw[i..i + 20]);
                i += 20;
            }
            0x32 => {
                if i + 27 > raw.len() { break; }
                grid.push_cell(CellType::TriQuadraticHexahedron, &raw[i..i + 27]);
                i += 27;
            }
            _ => break,
        }
    }
}

fn infer_num_components(attr_type: &str, di_dims: &[usize], total_len: usize) -> usize {
    let nc = match attr_type.to_ascii_lowercase().as_str() {
        "vector" => 3,
        "tensor6" => 6,
        "tensor" | "matrix" => 9,
        _ => {
            if di_dims.len() >= 2 {
                let last = *di_dims.last().unwrap_or(&1);
                if (last == 3 || last == 6 || last == 9) && total_len.is_multiple_of(last) {
                    last
                } else {
                    1
                }
            } else {
                1
            }
        }
    };
    if nc > 1 && total_len.is_multiple_of(nc) {
        nc
    } else {
        1
    }
}

fn make_data_array(name: String, num_components: usize, vals: Vec<f64>) -> DataArray {
    let f32s: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
    DataArray::new(name, num_components, ArrayData::F32(f32s))
}

#[derive(Clone, Copy)]
enum NumKind {
    Float,
    Int,
    UInt,
}

fn parse_num_spec(node: &XmlNode) -> (NumKind, usize, bool) {
    let num_type = node
        .attr("NumberType")
        .or_else(|| node.attr("DataType"))
        .unwrap_or("Float");
    let kind = if num_type.eq_ignore_ascii_case("Int") || num_type.eq_ignore_ascii_case("Char") {
        NumKind::Int
    } else if num_type.eq_ignore_ascii_case("UInt") || num_type.eq_ignore_ascii_case("UChar") {
        NumKind::UInt
    } else {
        NumKind::Float
    };
    let prec = node
        .attr("Precision")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(if num_type.eq_ignore_ascii_case("Char") || num_type.eq_ignore_ascii_case("UChar") {
            1
        } else {
            4
        });
    let big_endian = node
        .attr("Endian")
        .map(|s| s.eq_ignore_ascii_case("Big"))
        .unwrap_or(false);
    (kind, prec, big_endian)
}

fn read_binary_payload(node: &XmlNode, base_dir: &Path) -> Result<Vec<u8>> {
    let rel = node.text.trim();
    let mut bytes = fs::read(base_dir.join(rel))?;
    let seek = node
        .attr("Seek")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);
    if seek > 0 && seek <= bytes.len() {
        bytes = bytes[seek..].to_vec();
    }
    if node
        .attr("Compression")
        .is_some_and(|c| c.eq_ignore_ascii_case("ZLib"))
    {
        let mut decoder = ZlibDecoder::new(bytes.as_slice());
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| IoError::Parse(format!("XDMF ZLib decompression failed: {e}")))?;
        bytes = out;
    }
    Ok(bytes)
}

fn read_data_item_f64(node: &XmlNode, ctx: &XdmfContext<'_>) -> Result<(Vec<f64>, Vec<usize>)> {
    if let Some(target) = ctx.resolve_reference(node) {
        return read_data_item_f64(target, ctx);
    }

    let dims = node
        .attr("Dimensions")
        .map(parse_dims)
        .unwrap_or_default();
    let item_type = node.attr("ItemType").unwrap_or("Uniform");

    if item_type.eq_ignore_ascii_case("HyperSlab") {
        let children: Vec<&XmlNode> = node.data_items().collect();
        if children.len() >= 2 {
            let (slab, _) = read_data_item_u32(children[0], ctx)?;
            let (src, src_dims) = read_data_item_f64(children[1], ctx)?;
            return Ok((apply_hyperslab(&src, &src_dims, &slab), dims));
        }
    }

    if item_type.eq_ignore_ascii_case("Function") {
        let children: Vec<&XmlNode> = node.data_items().collect();
        let func = node.attr("Function").unwrap_or("");
        if func.to_ascii_uppercase().starts_with("JOIN") && !children.is_empty() {
            let mut inputs = Vec::with_capacity(children.len());
            for ch in &children {
                inputs.push(read_data_item_f64(ch, ctx)?.0);
            }
            let min_len = inputs.iter().map(|v| v.len()).min().unwrap_or(0);
            let mut joined = Vec::with_capacity(min_len * inputs.len());
            for i in 0..min_len {
                for inp in &inputs {
                    joined.push(inp[i]);
                }
            }
            return Ok((joined, dims));
        } else if let Some(first) = children.first() {
            return read_data_item_f64(first, ctx);
        }
    }

    let format = node.attr("Format").unwrap_or("XML");
    if format.eq_ignore_ascii_case("HDF") {
        let (vals, shape) = read_hdf5_f64(node.text.trim(), ctx)?;
        let out_dims = if dims.is_empty() { shape } else { dims };
        return Ok((vals, out_dims));
    }

    if format.eq_ignore_ascii_case("Binary") {
        let bytes = read_binary_payload(node, ctx.base_dir)?;
        let (kind, prec, be) = parse_num_spec(node);
        let vals = decode_binary_f64(&bytes, kind, prec, be);
        return Ok((vals, dims));
    }

    let vals: Vec<f64> = node
        .text
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    Ok((vals, dims))
}

fn read_data_item_u32(node: &XmlNode, ctx: &XdmfContext<'_>) -> Result<(Vec<u32>, Vec<usize>)> {
    if let Some(target) = ctx.resolve_reference(node) {
        return read_data_item_u32(target, ctx);
    }

    let dims = node
        .attr("Dimensions")
        .map(parse_dims)
        .unwrap_or_default();
    let format = node.attr("Format").unwrap_or("XML");
    if format.eq_ignore_ascii_case("HDF") {
        let (vals, shape) = read_hdf5_u32(node.text.trim(), ctx)?;
        let out_dims = if dims.is_empty() { shape } else { dims };
        return Ok((vals, out_dims));
    }

    if format.eq_ignore_ascii_case("Binary") {
        let bytes = read_binary_payload(node, ctx.base_dir)?;
        let (kind, prec, be) = parse_num_spec(node);
        let vals = decode_binary_u32(&bytes, kind, prec, be);
        return Ok((vals, dims));
    }

    let vals: Vec<u32> = node
        .text
        .split_whitespace()
        .filter_map(|t| t.parse::<u32>().ok())
        .collect();
    Ok((vals, dims))
}

fn decode_binary_f64(bytes: &[u8], kind: NumKind, prec: usize, be: bool) -> Vec<f64> {
    match (kind, prec) {
        (NumKind::Float, 8) => bytes
            .chunks_exact(8)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                if be { f64::from_be_bytes(a) } else { f64::from_le_bytes(a) }
            })
            .collect(),
        (NumKind::Float, _) => bytes
            .chunks_exact(4)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3]];
                (if be { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::Int, 8) => bytes
            .chunks_exact(8)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                (if be { i64::from_be_bytes(a) } else { i64::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::Int, 2) => bytes
            .chunks_exact(2)
            .map(|c| {
                let a = [c[0], c[1]];
                (if be { i16::from_be_bytes(a) } else { i16::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::Int, 1) => bytes.iter().map(|&b| (b as i8) as f64).collect(),
        (NumKind::Int, _) => bytes
            .chunks_exact(4)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3]];
                (if be { i32::from_be_bytes(a) } else { i32::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::UInt, 8) => bytes
            .chunks_exact(8)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                (if be { u64::from_be_bytes(a) } else { u64::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::UInt, 2) => bytes
            .chunks_exact(2)
            .map(|c| {
                let a = [c[0], c[1]];
                (if be { u16::from_be_bytes(a) } else { u16::from_le_bytes(a) }) as f64
            })
            .collect(),
        (NumKind::UInt, 1) => bytes.iter().map(|&b| b as f64).collect(),
        (NumKind::UInt, _) => bytes
            .chunks_exact(4)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3]];
                (if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) }) as f64
            })
            .collect(),
    }
}

fn decode_binary_u32(bytes: &[u8], kind: NumKind, prec: usize, be: bool) -> Vec<u32> {
    match (kind, prec) {
        (NumKind::Int, 8) | (NumKind::UInt, 8) => bytes
            .chunks_exact(8)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                (if be { u64::from_be_bytes(a) } else { u64::from_le_bytes(a) }) as u32
            })
            .collect(),
        (NumKind::Int, 2) | (NumKind::UInt, 2) => bytes
            .chunks_exact(2)
            .map(|c| {
                let a = [c[0], c[1]];
                (if be { u16::from_be_bytes(a) } else { u16::from_le_bytes(a) }) as u32
            })
            .collect(),
        (NumKind::Int, 1) | (NumKind::UInt, 1) => bytes.iter().map(|&b| b as u32).collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|c| {
                let a = [c[0], c[1], c[2], c[3]];
                if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) }
            })
            .collect(),
    }
}

fn apply_hyperslab(src: &[f64], src_dims: &[usize], slab: &[u32]) -> Vec<f64> {
    // 1D or flat fallback if dims don't match
    if slab.len() < 3 {
        return src.to_vec();
    }
    let ndim = slab.len() / 3;
    if ndim == 1 || src_dims.len() != ndim {
        let start = slab[0] as usize;
        let stride = (slab[1] as usize).max(1);
        let count = slab[2] as usize;
        return (0..count)
            .filter_map(|i| src.get(start + i * stride).copied())
            .collect();
    }
    // General N-D C-order hyperslab
    let starts = &slab[0..ndim];
    let strides = &slab[ndim..2 * ndim];
    let counts = &slab[2 * ndim..3 * ndim];

    let mut dim_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        dim_strides[d] = dim_strides[d + 1] * src_dims[d + 1];
    }

    let total: usize = counts.iter().map(|&c| c as usize).product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; ndim];
    for _ in 0..total {
        let mut flat = 0usize;
        for d in 0..ndim {
            let coord = starts[d] as usize + idx[d] * (strides[d] as usize).max(1);
            flat += coord * dim_strides[d];
        }
        if let Some(&v) = src.get(flat) {
            out.push(v);
        }
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < counts[d] as usize {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn parse_hdf5_spec(spec: &str) -> Result<(PathBuf, String)> {
    let trimmed = spec.trim();
    let (file_part, ds_part) = trimmed.rsplit_once(':').ok_or_else(|| {
        IoError::Parse(format!(
            "Invalid XDMF HDF5 DataItem '{trimmed}': expected '<file.h5>:/dataset/path'"
        ))
    })?;
    Ok((PathBuf::from(file_part.trim()), ds_part.trim().to_string()))
}

fn with_hdf5_dataset<T>(
    spec: &str,
    ctx: &XdmfContext<'_>,
    f: impl FnOnce(&hdf5_reader::Dataset) -> Result<T>,
) -> Result<T> {
    let (rel_path, ds_path) = parse_hdf5_spec(spec)?;
    let full_path = if rel_path.is_absolute() {
        rel_path
    } else {
        ctx.base_dir.join(rel_path)
    };
    let mut cache = ctx.hdf5_cache.borrow_mut();
    if !cache.contains_key(&full_path) {
        let file = hdf5_reader::Hdf5File::open(&full_path).map_err(|e| {
            IoError::Parse(format!(
                "Failed to open HDF5 file '{}': {e}",
                full_path.display()
            ))
        })?;
        cache.insert(full_path.clone(), file);
    }
    let file = &cache[&full_path];
    let ds = file.dataset(&ds_path).map_err(|e| {
        IoError::Parse(format!(
            "Failed to open HDF5 dataset '{ds_path}' in '{}': {e}",
            full_path.display()
        ))
    })?;
    f(&ds)
}

fn read_hdf5_f64(spec: &str, ctx: &XdmfContext<'_>) -> Result<(Vec<f64>, Vec<usize>)> {
    use hdf5_reader::Datatype;
    with_hdf5_dataset(spec, ctx, |ds| {
        let shape: Vec<usize> = ds.shape().iter().map(|&d| d as usize).collect();
        let vals: Vec<f64> = match ds.dtype() {
            Datatype::FloatingPoint { size: 8, .. } => ds
                .read_array::<f64>()
                .map_err(|e| IoError::Parse(format!("HDF5 read f64 failed: {e}")))?
                .into_raw_vec_and_offset()
                .0,
            Datatype::FloatingPoint { size: 4, .. } => ds
                .read_array::<f32>()
                .map_err(|e| IoError::Parse(format!("HDF5 read f32 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 8, signed: true, .. } => ds
                .read_array::<i64>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i64 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 8, signed: false, .. } => ds
                .read_array::<u64>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u64 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 4, signed: true, .. } => ds
                .read_array::<i32>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i32 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 4, signed: false, .. } => ds
                .read_array::<u32>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u32 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 2, signed: true, .. } => ds
                .read_array::<i16>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i16 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 2, signed: false, .. } => ds
                .read_array::<u16>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u16 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 1, signed: true, .. } => ds
                .read_array::<i8>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i8 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            Datatype::FixedPoint { size: 1, signed: false, .. } => ds
                .read_array::<u8>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u8 failed: {e}")))?
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            other => {
                return Err(IoError::UnsupportedFormat(format!(
                    "Unsupported HDF5 numeric datatype: {other:?}"
                )));
            }
        };
        Ok((vals, shape))
    })
}

fn read_hdf5_u32(spec: &str, ctx: &XdmfContext<'_>) -> Result<(Vec<u32>, Vec<usize>)> {
    use hdf5_reader::Datatype;
    with_hdf5_dataset(spec, ctx, |ds| {
        let shape: Vec<usize> = ds.shape().iter().map(|&d| d as usize).collect();
        let vals: Vec<u32> = match ds.dtype() {
            Datatype::FixedPoint { size: 4, signed: false, .. } => ds
                .read_array::<u32>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u32 failed: {e}")))?
                .into_raw_vec_and_offset()
                .0,
            Datatype::FixedPoint { size: 4, signed: true, .. } => ds
                .read_array::<i32>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i32 failed: {e}")))?
                .into_iter()
                .map(|v| v.max(0) as u32)
                .collect(),
            Datatype::FixedPoint { size: 8, signed: false, .. } => ds
                .read_array::<u64>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u64 failed: {e}")))?
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            Datatype::FixedPoint { size: 8, signed: true, .. } => ds
                .read_array::<i64>()
                .map_err(|e| IoError::Parse(format!("HDF5 read i64 failed: {e}")))?
                .into_iter()
                .map(|v| v.max(0) as u32)
                .collect(),
            Datatype::FixedPoint { size: 2, signed: false, .. } => ds
                .read_array::<u16>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u16 failed: {e}")))?
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            Datatype::FixedPoint { size: 1, signed: false, .. } => ds
                .read_array::<u8>()
                .map_err(|e| IoError::Parse(format!("HDF5 read u8 failed: {e}")))?
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            other => {
                return Err(IoError::UnsupportedFormat(format!(
                    "Unsupported HDF5 integer datatype: {other:?}"
                )));
            }
        };
        Ok((vals, shape))
    })
}
