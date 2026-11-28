//! Extensible 3D (`.x3d`) XML scene/mesh reader.
//!
//! Parses `<IndexedFaceSet>`, `<IndexedLineSet>`, `<PointSet>`, `<Coordinate>`,
//! `<Normal>`, `<Color>`, and nested `<Transform>` nodes into [`PolyData`] / [`MultiBlockDataSet`].

use crate::error::{IoError, Result};
use crate::limits::MAX_XML_DEPTH;
use glam::{Mat4, Quat, Vec3};
use paraviz_core::{ArrayData, DataArray, DataSet, MultiBlockDataSet, PolyData};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs;
use std::path::Path;

/// Reads an X3D (`.x3d`) XML file into a [`DataSet`].
pub fn read_x3d(path: impl AsRef<Path>) -> Result<DataSet> {
    let xml = fs::read_to_string(path)?;
    read_x3d_str(&xml)
}

/// Parses an X3D XML string into a [`DataSet`].
pub fn read_x3d_str(xml: impl AsRef<str>) -> Result<DataSet> {
    let xml = xml.as_ref();
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut transform_stack: Vec<Mat4> = vec![Mat4::IDENTITY];
    let mut blocks: Vec<(String, PolyData)> = Vec::new();
    let mut current_geom: Option<X3dGeomBuilder> = None;
    let mut depth: usize = 0;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                depth += 1;
                if depth > MAX_XML_DEPTH {
                    return Err(IoError::Parse("X3D XML nesting depth exceeded".into()));
                }
                let name = e.name().as_ref().to_string();
                let attrs = collect_attrs(&e);
                match name.as_str() {
                    "Transform" => {
                        let mat = parse_transform(&attrs);
                        let top = *transform_stack.last().unwrap_or(&Mat4::IDENTITY);
                        transform_stack.push(top * mat);
                    }
                    "IndexedFaceSet" | "IndexedLineSet" | "PointSet" => {
                        current_geom = Some(X3dGeomBuilder::new(&name, &attrs));
                    }
                    "Coordinate" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(pts) = attrs.iter().find(|(k, _)| k == "point").map(|(_, v)| v)
                            {
                                geom.points = parse_vec3_list(pts)?;
                            }
                    }
                    "Normal" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(ns) = attrs.iter().find(|(k, _)| k == "vector").map(|(_, v)| v)
                            {
                                geom.normals = parse_vec3_list(ns)?;
                            }
                    }
                    "Color" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(cs) = attrs.iter().find(|(k, _)| k == "color").map(|(_, v)| v)
                            {
                                geom.colors = parse_vec3_list(cs)?;
                            }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let name = e.name().as_ref().to_string();
                let attrs = collect_attrs(&e);
                match name.as_str() {
                    "Coordinate" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(pts) = attrs.iter().find(|(k, _)| k == "point").map(|(_, v)| v)
                            {
                                geom.points = parse_vec3_list(pts)?;
                            }
                    }
                    "Normal" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(ns) = attrs.iter().find(|(k, _)| k == "vector").map(|(_, v)| v)
                            {
                                geom.normals = parse_vec3_list(ns)?;
                            }
                    }
                    "Color" => {
                        if let Some(geom) = current_geom.as_mut()
                            && let Some(cs) = attrs.iter().find(|(k, _)| k == "color").map(|(_, v)| v)
                            {
                                geom.colors = parse_vec3_list(cs)?;
                            }
                    }
                    "IndexedFaceSet" | "IndexedLineSet" | "PointSet" => {
                        let mat = *transform_stack.last().unwrap_or(&Mat4::IDENTITY);
                        let builder = X3dGeomBuilder::new(&name, &attrs);
                        let poly = builder.build(mat)?;
                        blocks.push((format!("Shape_{}", blocks.len()), poly));
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                depth = depth.saturating_sub(1);
                let name = e.name().as_ref().to_string();
                match name.as_str() {
                    "Transform" => {
                        if transform_stack.len() > 1 {
                            transform_stack.pop();
                        }
                    }
                    "IndexedFaceSet" | "IndexedLineSet" | "PointSet" => {
                        if let Some(builder) = current_geom.take() {
                            let mat = *transform_stack.last().unwrap_or(&Mat4::IDENTITY);
                            let poly = builder.build(mat)?;
                            blocks.push((format!("Shape_{}", blocks.len()), poly));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(IoError::Parse(format!("X3D XML parse error: {e}"))),
            _ => {}
        }
    }

    if blocks.is_empty() {
        return Err(IoError::Parse("X3D file contains no geometry nodes".into()));
    }
    if blocks.len() == 1 {
        Ok(DataSet::PolyData(blocks.into_iter().next().unwrap().1))
    } else {
        let mut mb = MultiBlockDataSet::new();
        for (name, poly) in blocks {
            mb.add_block(name, DataSet::PolyData(poly));
        }
        Ok(DataSet::MultiBlock(mb))
    }
}

struct X3dGeomBuilder {
    kind: String,
    coord_index: Vec<i32>,
    points: Vec<Vec3>,
    normals: Vec<Vec3>,
    colors: Vec<Vec3>,
}

impl X3dGeomBuilder {
    fn new(kind: &str, attrs: &[(String, String)]) -> Self {
        let coord_index = attrs
            .iter()
            .find(|(k, _)| k == "coordIndex")
            .map(|(_, v)| {
                v.split(|c: char| c.is_whitespace() || c == ',')
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<i32>().ok())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            kind: kind.to_string(),
            coord_index,
            points: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
        }
    }

    fn build(self, transform: Mat4) -> Result<PolyData> {
        let mut poly = PolyData::new();
        poly.points = self
            .points
            .into_iter()
            .map(|p| transform.transform_point3(p).to_array())
            .collect();

        let n_pts = poly.points.len();
        if self.kind == "PointSet" {
            poly.verts = (0..n_pts as u32).collect();
        } else {
            let mut current = Vec::new();
            for idx in self.coord_index {
                if idx < 0 {
                    if !current.is_empty() {
                        if self.kind == "IndexedLineSet" && current.len() >= 2 {
                            poly.lines.push(std::mem::take(&mut current));
                        } else if current.len() >= 3 {
                            poly.polys.push(std::mem::take(&mut current));
                        } else {
                            current.clear();
                        }
                    }
                } else {
                    let u = idx as u32;
                    if (u as usize) >= n_pts {
                        return Err(IoError::InvalidData(format!(
                            "X3D coordIndex {u} out of bounds ({n_pts})"
                        )));
                    }
                    current.push(u);
                }
            }
            if !current.is_empty() {
                if self.kind == "IndexedLineSet" && current.len() >= 2 {
                    poly.lines.push(current);
                } else if current.len() >= 3 {
                    poly.polys.push(current);
                }
            }
        }

        if self.normals.len() == n_pts && n_pts > 0 {
            let flat: Vec<f32> = self
                .normals
                .into_iter()
                .flat_map(|n| {
                    let tn = transform.transform_vector3(n).normalize_or_zero();
                    [tn.x, tn.y, tn.z]
                })
                .collect();
            poly.point_data
                .add_array(DataArray::new("Normals", 3, ArrayData::F32(flat)));
        }

        if self.colors.len() == n_pts && n_pts > 0 {
            let flat: Vec<u8> = self
                .colors
                .into_iter()
                .flat_map(|c| {
                    [
                        (c.x.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (c.y.clamp(0.0, 1.0) * 255.0).round() as u8,
                        (c.z.clamp(0.0, 1.0) * 255.0).round() as u8,
                        255u8,
                    ]
                })
                .collect();
            poly.point_data
                .add_array(DataArray::new("Colors", 4, ArrayData::U8(flat)));
        }

        Ok(poly)
    }
}

fn collect_attrs(e: &quick_xml::events::BytesStart<'_>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for attr in e.attributes().flatten() {
        let k = attr.key.as_ref().to_string();
        if let Ok(v) = attr.normalized_value(quick_xml::XmlVersion::default()) {
            out.push((k, v.into_owned()));
        }
    }
    out
}

fn parse_vec3_list(s: &str) -> Result<Vec<Vec3>> {
    let nums: Vec<f32> = s
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|tok| !tok.is_empty())
        .map(|tok| {
            tok.parse::<f32>()
                .map_err(|_| IoError::Parse(format!("Invalid X3D float '{tok}'")))
        })
        .collect::<Result<_>>()?;
    if !nums.len().is_multiple_of(3) {
        return Err(IoError::Parse(format!(
            "X3D vector list length {} is not divisible by 3",
            nums.len()
        )));
    }
    Ok(nums
        .as_chunks::<3>().0.iter()
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect())
}

fn parse_transform(attrs: &[(String, String)]) -> Mat4 {
    let mut translation = Vec3::ZERO;
    let mut rotation = Quat::IDENTITY;
    let mut scale = Vec3::ONE;

    for (k, v) in attrs {
        let nums: Vec<f32> = v
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();
        match k.as_str() {
            "translation" if nums.len() == 3 => {
                translation = Vec3::new(nums[0], nums[1], nums[2]);
            }
            "scale" if nums.len() == 3 => {
                scale = Vec3::new(nums[0], nums[1], nums[2]);
            }
            "rotation" if nums.len() == 4 => {
                let axis = Vec3::new(nums[0], nums[1], nums[2]).normalize_or_zero();
                if axis.length_squared() > 0.0 {
                    rotation = Quat::from_axis_angle(axis, nums[3]);
                }
            }
            _ => {}
        }
    }
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}
