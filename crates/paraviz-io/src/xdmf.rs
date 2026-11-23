//! XDMF (`.xdmf`, `.xmf`) XML scientific dataset reader supporting inline XML and raw binary DataItems.

use crate::error::{IoError, Result};
use paraviz_core::{CellType, DataArray, DataSet, UnstructuredGrid};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs;
use std::path::Path;

/// Reads an XDMF (`.xdmf` / `.xmf`) file with XML or Binary `DataItem` payloads into a [`DataSet`].
pub fn read_xdmf(path: impl AsRef<Path>) -> Result<DataSet> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let xml = fs::read_to_string(path)?;
    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(true);

    let mut topo_type = String::new();
    let mut in_topo = false;
    let mut in_geom = false;
    let mut attr_name = String::new();
    let mut attr_center = String::from("Node");
    let mut in_attr = false;
    let mut cur_format = String::from("XML");
    let mut cur_text = String::new();
    let mut in_data_item = false;

    let mut grid = UnstructuredGrid::new();
    let mut raw_topo: Vec<u32> = Vec::new();

    loop {
        match reader.read_event()? {
            Event::Start(e) => match e.name().as_ref() {
                "Topology" => {
                    in_topo = true;
                    for a in e.attributes().flatten() {
                        if a.key.as_ref() == "TopologyType" {
                            topo_type = a.value.into_owned();
                        }
                    }
                }
                "Geometry" => in_geom = true,
                "Attribute" => {
                    in_attr = true;
                    attr_name.clear();
                    attr_center = "Node".into();
                    for a in e.attributes().flatten() {
                        match a.key.as_ref() {
                            "Name" => attr_name = a.value.into_owned(),
                            "Center" => attr_center = a.value.into_owned(),
                            _ => {}
                        }
                    }
                }
                "DataItem" => {
                    in_data_item = true;
                    cur_text.clear();
                    cur_format = "XML".into();
                    for a in e.attributes().flatten() {
                        if a.key.as_ref() == "Format" {
                            cur_format = a.value.into_owned();
                        }
                    }
                }
                _ => {}
            },
            Event::Text(t) => {
                if in_data_item {
                    cur_text.push_str(&t);
                }
            }
            Event::End(e) => match e.name().as_ref() {
                "DataItem" => {
                    in_data_item = false;
                    if in_topo {
                        raw_topo = parse_u32_item(&cur_text, &cur_format, base_dir)?;
                    } else if in_geom {
                        let vals = parse_f32_item(&cur_text, &cur_format, base_dir)?;
                        grid.points = vals
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                    } else if in_attr {
                        let vals = parse_f32_item(&cur_text, &cur_format, base_dir)?;
                        let arr = DataArray::scalars_f32(
                            if attr_name.is_empty() {
                                "Attribute".to_string()
                            } else {
                                attr_name.clone()
                            },
                            vals,
                        );
                        if attr_center.eq_ignore_ascii_case("Cell") {
                            grid.cell_data.add_array(arr);
                        } else {
                            grid.point_data.add_array(arr);
                        }
                    }
                }
                "Topology" => in_topo = false,
                "Geometry" => in_geom = false,
                "Attribute" => in_attr = false,
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }

    let (ct, stride) = match topo_type.as_str() {
        "Triangle" => (CellType::Triangle, 3),
        "Quadrilateral" => (CellType::Quad, 4),
        "Tetrahedron" => (CellType::Tetra, 4),
        "Pyramid" => (CellType::Pyramid, 5),
        "Wedge" => (CellType::Wedge, 6),
        "Hexahedron" => (CellType::Hexahedron, 8),
        "Polyline" => (CellType::Line, 2),
        _ => (CellType::Triangle, 3),
    };
    for chunk in raw_topo.chunks_exact(stride) {
        grid.push_cell(ct, chunk);
    }

    Ok(DataSet::UnstructuredGrid(grid))
}

fn parse_u32_item(text: &str, format: &str, base_dir: &Path) -> Result<Vec<u32>> {
    if format.eq_ignore_ascii_case("Binary") {
        let bytes = fs::read(base_dir.join(text.trim()))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else if format.eq_ignore_ascii_case("HDF") {
        Err(IoError::UnsupportedFormat(
            "XDMF HDF5 external storage requires HDF5; use XML or Binary DataItems".into(),
        ))
    } else {
        Ok(text
            .split_whitespace()
            .filter_map(|t| t.parse::<u32>().ok())
            .collect())
    }
}

fn parse_f32_item(text: &str, format: &str, base_dir: &Path) -> Result<Vec<f32>> {
    if format.eq_ignore_ascii_case("Binary") {
        let bytes = fs::read(base_dir.join(text.trim()))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else if format.eq_ignore_ascii_case("HDF") {
        Err(IoError::UnsupportedFormat(
            "XDMF HDF5 external storage requires HDF5; use XML or Binary DataItems".into(),
        ))
    } else {
        Ok(text
            .split_whitespace()
            .filter_map(|t| t.parse::<f32>().ok())
            .collect())
    }
}
