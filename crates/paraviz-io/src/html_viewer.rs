//! Self-contained interactive WebGL2 HTML5 3D viewer exporter (`.html`) for [`PolyData`]
//! (addressing ParaView Discourse requests for lightweight portable web sharing).

use crate::error::Result;
use glam::Vec3;
use paraviz_core::{ColorMap, PolyData};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Writes a [`PolyData`] surface mesh as a standalone interactive WebGL2 `.html` viewer.
///
/// Zero external CDN dependencies: embeds vertex positions, normals, optional colormap-mapped
/// vertex colors, triangle indices, and interactive orbit/pan/zoom/wireframe controls.
pub fn write_html_viewer(
    path: impl AsRef<Path>,
    poly: &PolyData,
    title: impl AsRef<str>,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Result<()> {
    let html = encode_html_viewer(poly, title, color_map, scalar_array);
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(html.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Writes an **A/B comparison** viewer: two meshes in one self-contained `.html` file,
/// one shared (linked) camera, a wipe slider to compare them, and a common scalar
/// color-bar legend when a colormap + scalar array are provided.
#[allow(clippy::too_many_arguments)]
pub fn write_html_viewer_ab(
    path: impl AsRef<Path>,
    poly_a: &PolyData,
    label_a: impl AsRef<str>,
    poly_b: &PolyData,
    label_b: impl AsRef<str>,
    title: impl AsRef<str>,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Result<()> {
    let html = encode_html_viewer_ab(
        poly_a, label_a, poly_b, label_b, title, color_map, scalar_array,
    );
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(html.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Encodes a [`PolyData`] mesh into a self-contained interactive WebGL2 HTML document.
pub fn encode_html_viewer(
    poly: &PolyData,
    title: impl AsRef<str>,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> String {
    let title = title.as_ref();
    let mut positions: Vec<f32> = Vec::with_capacity(poly.points.len() * 3);
    for p in &poly.points {
        positions.extend_from_slice(p);
    }

    let indices = collect_triangle_indices(poly);
    let normals = compute_vertex_normals(poly, &indices);
    let colors = compute_vertex_colors(poly, color_map, scalar_array);

    let bb = poly.bounds();
    let center = bb.center();
    let diag = bb.diagonal().max(1e-3);

    let payload = serde_json::json!({
        "title": title,
        "points": poly.num_points(),
        "triangles": indices.len() / 3,
        "center": [center.x, center.y, center.z],
        "radius": diag * 0.55,
        "positions": positions,
        "normals": normals,
        "colors": colors,
        "indices": indices,
    });

    let raw_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into());
    // Escape '<', '>', '&' and JS line terminators so embedded JSON cannot break out of <script>.
    let json_str = raw_json
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    let escaped_title = title
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ParaViz — {escaped_title}</title>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #141824; color: #e6ebf5; font-family: system-ui, -apple-system, sans-serif; }}
  canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  .hud {{ position: fixed; top: 14px; left: 14px; background: rgba(18, 22, 34, 0.85); border: 1px solid #2e384f; border-radius: 8px; padding: 10px 14px; font-size: 12px; line-height: 1.5; backdrop-filter: blur(6px); }}
  .hud strong {{ font-size: 13px; color: #7cc4ff; }}
  .controls {{ position: fixed; bottom: 14px; left: 14px; display: flex; gap: 8px; }}
  button {{ background: #1f273a; color: #e6ebf5; border: 1px solid #394663; border-radius: 6px; padding: 6px 11px; font-size: 12px; cursor: pointer; }}
  button:hover {{ background: #2c3854; }}
</style>
</head>
<body>
<canvas id="glcanvas"></canvas>
<div class="hud">
  <div><strong>{escaped_title}</strong></div>
  <div id="stats"></div>
  <div style="opacity:0.75;margin-top:4px;">Drag: Orbit &middot; Shift+Drag: Pan &middot; Wheel: Zoom</div>
</div>
<div class="controls">
  <button id="resetBtn">Reset Camera</button>
  <button id="wireBtn">Toggle Wireframe</button>
</div>
<script id="mesh-data" type="application/json">{json_str}</script>
<script>
const data = JSON.parse(document.getElementById('mesh-data').textContent);
document.getElementById('stats').textContent = data.points.toLocaleString() + ' pts | ' + data.triangles.toLocaleString() + ' tris';

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', {{ antialias: true }});
if (!gl) {{ alert('WebGL2 not supported in this browser.'); }}

const vs = `#version 300 es
in vec3 aPos;
in vec3 aNorm;
in vec3 aCol;
uniform mat4 uVP;
out vec3 vNorm;
out vec3 vCol;
void main() {{
  gl_Position = uVP * vec4(aPos, 1.0);
  vNorm = aNorm;
  vCol = aCol;
}}`;

const fs = `#version 300 es
precision highp float;
in vec3 vNorm;
in vec3 vCol;
uniform int uWire;
out vec4 outCol;
void main() {{
  if (uWire == 1) {{ outCol = vec4(0.9, 0.95, 1.0, 0.9); return; }}
  vec3 n = normalize(vNorm);
  vec3 l1 = normalize(vec3(0.4, 0.7, 0.8));
  vec3 l2 = normalize(vec3(-0.6, -0.3, 0.4));
  float diff = max(abs(dot(n, l1)), 0.0) * 0.75 + max(abs(dot(n, l2)), 0.0) * 0.25;
  vec3 rgb = vCol * (0.22 + 0.78 * diff);
  outCol = vec4(rgb, 1.0);
}}`;

function compile(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog);
gl.useProgram(prog);

const posArr = new Float32Array(data.positions);
const normArr = new Float32Array(data.normals);
const colArr = new Float32Array(data.colors);
const idxArr = new Uint32Array(data.indices);

function bindBuf(arr, loc, size) {{
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
}}
bindBuf(posArr, gl.getAttribLocation(prog, 'aPos'), 3);
bindBuf(normArr, gl.getAttribLocation(prog, 'aNorm'), 3);
bindBuf(colArr, gl.getAttribLocation(prog, 'aCol'), 3);

const ibo = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxArr, gl.STATIC_DRAW);

let yaw = 0.65, pitch = 0.45, dist = data.radius * 3.2;
let target = [...data.center], wireframe = false;
function resetCam() {{ yaw = 0.65; pitch = 0.45; dist = data.radius * 3.2; target = [...data.center]; }}
document.getElementById('resetBtn').onclick = resetCam;
document.getElementById('wireBtn').onclick = () => {{ wireframe = !wireframe; }};

let dragging = false, lx = 0, ly = 0;
canvas.addEventListener('mousedown', e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lx, dy = e.clientY - ly;
  lx = e.clientX; ly = e.clientY;
  if (e.shiftKey) {{
    const s = dist * 0.0018;
    target[0] -= (Math.cos(yaw) * dx) * s;
    target[1] += dy * s;
    target[2] += (Math.sin(yaw) * dx) * s;
  }} else {{
    yaw -= dx * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  }}
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.001);
}}, {{ passive: false }});

function mat4Mul(a, b) {{
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return o;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}}
function lookAt(eye, ctr, up) {{
  let zx = eye[0]-ctr[0], zy = eye[1]-ctr[1], zz = eye[2]-ctr[2];
  let zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
  let xx = up[1]*zz - up[2]*zy, xy = up[2]*zx - up[0]*zz, xz = up[0]*zy - up[1]*zx;
  let xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
  let yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
  return new Float32Array([
    xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
    -(xx*eye[0]+xy*eye[1]+xz*eye[2]), -(yx*eye[0]+yy*eye[1]+yz*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1
  ]);
}}

const uVP = gl.getUniformLocation(prog, 'uVP');
const uWire = gl.getUniformLocation(prog, 'uWire');

function frame() {{
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; }}
  gl.viewport(0, 0, w, h);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.08, 0.10, 0.15, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const eye = [
    target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
    target[1] + dist * Math.sin(pitch),
    target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
  ];
  const vp = mat4Mul(perspective(Math.PI / 4, w / Math.max(h, 1), dist * 0.01, dist * 100), lookAt(eye, target, [0, 1, 0]));
  gl.uniformMatrix4fv(uVP, false, vp);
  gl.uniform1i(uWire, wireframe ? 1 : 0);
  gl.drawElements(wireframe ? gl.LINES : gl.TRIANGLES, idxArr.length, gl.UNSIGNED_INT, 0);
  requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
</script>
</body>
</html>
"#
    )
}

fn collect_triangle_indices(poly: &PolyData) -> Vec<u32> {
    let mut indices = Vec::new();
    for p in &poly.polys {
        if p.len() >= 3 {
            for k in 1..(p.len() - 1) {
                indices.push(p[0]);
                indices.push(p[k]);
                indices.push(p[k + 1]);
            }
        }
    }
    for s in &poly.strips {
        for k in 0..s.len().saturating_sub(2) {
            if k % 2 == 0 {
                indices.push(s[k]);
                indices.push(s[k + 1]);
                indices.push(s[k + 2]);
            } else {
                indices.push(s[k + 1]);
                indices.push(s[k]);
                indices.push(s[k + 2]);
            }
        }
    }
    indices
}

fn compute_vertex_normals(poly: &PolyData, indices: &[u32]) -> Vec<f32> {
    if let Some(norm_arr) = poly.point_data.active_normals_array()
        && norm_arr.num_tuples() == poly.num_points()
        && norm_arr.num_components >= 3
    {
        let mut out = Vec::with_capacity(poly.num_points() * 3);
        for i in 0..poly.num_points() {
            let v = norm_arr.get_vec3(i);
            out.extend_from_slice(&v);
        }
        return out;
    }

    let mut acc = vec![Vec3::ZERO; poly.num_points()];
    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if a < poly.points.len() && b < poly.points.len() && c < poly.points.len() {
            let pa = Vec3::from_array(poly.points[a]);
            let pb = Vec3::from_array(poly.points[b]);
            let pc = Vec3::from_array(poly.points[c]);
            let n = (pb - pa).cross(pc - pa);
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
    }
    let mut out = Vec::with_capacity(poly.num_points() * 3);
    for n in acc {
        let u = n.normalize_or(Vec3::Y);
        out.extend_from_slice(&[u.x, u.y, u.z]);
    }
    out
}

fn compute_vertex_colors(
    poly: &PolyData,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> Vec<f32> {
    let n = poly.num_points();
    if let (Some(cmap), Some(arr_name)) = (color_map, scalar_array)
        && let Some(arr) = poly.point_data.get(arr_name)
        && arr.num_tuples() == n
    {
        let range = arr.range(None);
        return compute_vertex_colors_range(poly, Some(cmap), arr_name, range);
    }
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        out.extend_from_slice(&[0.78, 0.82, 0.88]);
    }
    out
}

/// Vertex colors mapped through `cmap` over an explicit shared `(lo, hi)` range, so that
/// the A/B viewer's common legend is meaningful for both sides.
fn compute_vertex_colors_range(
    poly: &PolyData,
    color_map: Option<&ColorMap>,
    scalar_array: &str,
    range: (f64, f64),
) -> Vec<f32> {
    let n = poly.num_points();
    if let (Some(cmap), Some(arr)) = (color_map, poly.point_data.get(scalar_array))
        && arr.num_tuples() == n
    {
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let val = arr.get_magnitude(i);
            let rgba = cmap.map_value(val, range);
            out.extend_from_slice(&[rgba[0], rgba[1], rgba[2]]);
        }
        return out;
    }
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        out.extend_from_slice(&[0.78, 0.82, 0.88]);
    }
    out
}

/// Union of the scalar array ranges of both sides (the common legend domain).
fn scalar_range_union(
    poly_a: &PolyData,
    poly_b: &PolyData,
    scalar_array: &str,
) -> Option<(f64, f64)> {
    let ra = poly_a.point_data.get(scalar_array).map(|a| a.range(None));
    let rb = poly_b.point_data.get(scalar_array).map(|a| a.range(None));
    match (ra, rb) {
        (Some((a0, a1)), Some((b0, b1))) => Some((a0.min(b0), a1.max(b1))),
        (Some(r), None) | (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

fn mesh_payload(poly: &PolyData, colors: Vec<f32>, indices: &[u32]) -> serde_json::Value {
    let mut positions: Vec<f32> = Vec::with_capacity(poly.points.len() * 3);
    for p in &poly.points {
        positions.extend_from_slice(p);
    }
    let normals = compute_vertex_normals(poly, indices);
    serde_json::json!({
        "points": poly.num_points(),
        "triangles": indices.len() / 3,
        "positions": positions,
        "normals": normals,
        "colors": colors,
        "indices": indices,
    })
}

/// Encodes an **A/B comparison** into a self-contained interactive WebGL2 HTML document:
/// one canvas, one shared camera, a wipe slider (`A` left / `B` right) and a common
/// scalar color-bar legend. Both sides are colored over the union of their scalar ranges.
pub fn encode_html_viewer_ab(
    poly_a: &PolyData,
    label_a: impl AsRef<str>,
    poly_b: &PolyData,
    label_b: impl AsRef<str>,
    title: impl AsRef<str>,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> String {
    let label_a = label_a.as_ref();
    let label_b = label_b.as_ref();
    let title = title.as_ref();
    let indices_a = collect_triangle_indices(poly_a);
    let indices_b = collect_triangle_indices(poly_b);

    let range = scalar_array.and_then(|name| scalar_range_union(poly_a, poly_b, name));
    let colors_a = match (color_map, scalar_array, range) {
        (Some(cmap), Some(name), Some(r)) => {
            compute_vertex_colors_range(poly_a, Some(cmap), name, r)
        }
        _ => compute_vertex_colors(poly_a, None, None),
    };
    let colors_b = match (color_map, scalar_array, range) {
        (Some(cmap), Some(name), Some(r)) => {
            compute_vertex_colors_range(poly_b, Some(cmap), name, r)
        }
        _ => compute_vertex_colors(poly_b, None, None),
    };

    // union bounds -> one shared camera fit
    let bb_a = poly_a.bounds();
    let bb_b = poly_b.bounds();
    let lo = bb_a.min.min(bb_b.min);
    let hi = bb_a.max.max(bb_b.max);
    let center = (lo + hi) * 0.5;
    let diag = (hi - lo).length().max(1e-3);

    // common legend: sample the colormap into CSS gradient stops + range labels
    let legend = match (color_map, scalar_array, range) {
        (Some(cmap), Some(name), Some((lo, hi))) => {
            let stops: Vec<[u8; 3]> = (0..=24)
                .map(|i| {
                    let t = i as f64 / 24.0;
                    let rgba = cmap.map_value(lo + t * (hi - lo), (lo, hi));
                    [
                        (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]
                })
                .collect();
            serde_json::json!({
                "array": name,
                "min": lo,
                "max": hi,
                "stops": stops,
            })
        }
        _ => serde_json::Value::Null,
    };

    let payload = serde_json::json!({
        "title": title,
        "labelA": label_a,
        "labelB": label_b,
        "center": [center.x, center.y, center.z],
        "radius": diag * 0.55,
        "legend": legend,
        "a": mesh_payload(poly_a, colors_a, &indices_a),
        "b": mesh_payload(poly_b, colors_b, &indices_b),
    });

    let raw_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into());
    // Escape '<', '>', '&' and JS line terminators so embedded JSON cannot break out of <script>.
    let json_str = raw_json
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    let escaped_title = escape_html(title);
    let escaped_a = escape_html(label_a);
    let escaped_b = escape_html(label_b);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ParaViz — {escaped_title}</title>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #141824; color: #e6ebf5; font-family: system-ui, -apple-system, sans-serif; }}
  canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  .hud {{ position: fixed; top: 14px; left: 14px; background: rgba(18, 22, 34, 0.85); border: 1px solid #2e384f; border-radius: 8px; padding: 10px 14px; font-size: 12px; line-height: 1.5; backdrop-filter: blur(6px); }}
  .hud strong {{ font-size: 13px; color: #7cc4ff; }}
  .tagA, .tagB {{ display: inline-block; border-radius: 4px; padding: 0 6px; font-weight: 600; }}
  .tagA {{ background: #274a6d; color: #9fd0ff; }}
  .tagB {{ background: #5d3a1e; color: #ffc48f; }}
  .controls {{ position: fixed; bottom: 14px; left: 14px; display: flex; gap: 8px; }}
  button {{ background: #1f273a; color: #e6ebf5; border: 1px solid #394663; border-radius: 6px; padding: 6px 11px; font-size: 12px; cursor: pointer; }}
  button:hover {{ background: #2c3854; }}
  .abbar {{ position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); width: min(520px, 60vw); background: rgba(18, 22, 34, 0.88); border: 1px solid #2e384f; border-radius: 10px; padding: 10px 14px 12px; text-align: center; backdrop-filter: blur(6px); }}
  .abbar .labels {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }}
  .abbar input[type=range] {{ width: 100%; accent-color: #7cc4ff; cursor: ew-resize; }}
  .legend {{ position: fixed; top: 14px; right: 14px; background: rgba(18, 22, 34, 0.85); border: 1px solid #2e384f; border-radius: 8px; padding: 10px 12px; font-size: 11px; backdrop-filter: blur(6px); }}
  .legend .bar {{ width: 180px; height: 12px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.25); }}
  .legend .ticks {{ display: flex; justify-content: space-between; width: 180px; margin-top: 3px; opacity: 0.85; }}
  .legend .name {{ margin-bottom: 5px; color: #7cc4ff; font-weight: 600; }}
</style>
</head>
<body>
<canvas id="glcanvas"></canvas>
<div class="hud">
  <div><strong>{escaped_title}</strong></div>
  <div><span class="tagA">A</span> {escaped_a} &nbsp; <span class="tagB">B</span> {escaped_b}</div>
  <div id="stats"></div>
  <div style="opacity:0.75;margin-top:4px;">Drag: Orbit &middot; Shift+Drag: Pan &middot; Wheel: Zoom &middot; Slider: A/B wipe</div>
</div>
<div class="legend" id="legend" style="display:none;">
  <div class="name" id="legendName"></div>
  <div class="bar" id="legendBar"></div>
  <div class="ticks"><span id="legendMin"></span><span id="legendMax"></span></div>
</div>
<div class="controls">
  <button id="resetBtn">Reset Camera</button>
  <button id="wireBtn">Toggle Wireframe</button>
</div>
<div class="abbar">
  <div class="labels"><span class="tagA">A — {escaped_a}</span><span class="tagB">B — {escaped_b}</span></div>
  <input id="abSlider" type="range" min="0" max="1000" value="500" />
</div>
<script id="mesh-data" type="application/json">{json_str}</script>
<script>
const data = JSON.parse(document.getElementById('mesh-data').textContent);
document.getElementById('stats').textContent =
  'A: ' + data.a.points.toLocaleString() + ' pts / ' + data.a.triangles.toLocaleString() + ' tris' +
  '  ·  B: ' + data.b.points.toLocaleString() + ' pts / ' + data.b.triangles.toLocaleString() + ' tris';

if (data.legend) {{
  const lg = data.legend;
  const stops = lg.stops.map(s => 'rgb(' + s[0] + ',' + s[1] + ',' + s[2] + ')').join(',');
  document.getElementById('legend').style.display = 'block';
  document.getElementById('legendName').textContent = lg.array;
  document.getElementById('legendBar').style.background = 'linear-gradient(to right,' + stops + ')';
  document.getElementById('legendMin').textContent = Number(lg.min).toPrecision(4);
  document.getElementById('legendMax').textContent = Number(lg.max).toPrecision(4);
}}

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', {{ antialias: true }});
if (!gl) {{ alert('WebGL2 not supported in this browser.'); }}

const vs = `#version 300 es
in vec3 aPos;
in vec3 aNorm;
in vec3 aCol;
uniform mat4 uVP;
out vec3 vNorm;
out vec3 vCol;
out vec2 vNdc;
void main() {{
  vec4 clip = uVP * vec4(aPos, 1.0);
  gl_Position = clip;
  vNdc = clip.xy / clip.w;
  vNorm = aNorm;
  vCol = aCol;
}}`;

const fs = `#version 300 es
precision highp float;
in vec3 vNorm;
in vec3 vCol;
in vec2 vNdc;
uniform int uWire;
uniform float uSplit;   // wipe position in NDC x: <= split shows A, > split shows B
uniform int uSide;     // 0 = draw A (discard right of split), 1 = draw B (discard left)
out vec4 outCol;
void main() {{
  if (uSide == 0 && vNdc.x > uSplit) discard;
  if (uSide == 1 && vNdc.x <= uSplit) discard;
  if (uWire == 1) {{ outCol = vec4(0.9, 0.95, 1.0, 0.9); return; }}
  vec3 n = normalize(vNorm);
  vec3 l1 = normalize(vec3(0.4, 0.7, 0.8));
  vec3 l2 = normalize(vec3(-0.6, -0.3, 0.4));
  float diff = max(abs(dot(n, l1)), 0.0) * 0.75 + max(abs(dot(n, l2)), 0.0) * 0.25;
  vec3 rgb = vCol * (0.22 + 0.78 * diff);
  outCol = vec4(rgb, 1.0);
}}`;

function compile(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog);
gl.useProgram(prog);

function makeMesh(m) {{
  const pos = new Float32Array(m.positions);
  const norm = new Float32Array(m.normals);
  const col = new Float32Array(m.colors);
  const idx = new Uint32Array(m.indices);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  function bindBuf(arr, loc, size) {{
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }}
  bindBuf(pos, gl.getAttribLocation(prog, 'aPos'), 3);
  bindBuf(norm, gl.getAttribLocation(prog, 'aNorm'), 3);
  bindBuf(col, gl.getAttribLocation(prog, 'aCol'), 3);
  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return {{ vao, count: idx.length }};
}}
const meshA = makeMesh(data.a);
const meshB = makeMesh(data.b);

let yaw = 0.65, pitch = 0.45, dist = data.radius * 3.2;
let target = [...data.center], wireframe = false;
function resetCam() {{ yaw = 0.65; pitch = 0.45; dist = data.radius * 3.2; target = [...data.center]; }}
document.getElementById('resetBtn').onclick = resetCam;
document.getElementById('wireBtn').onclick = () => {{ wireframe = !wireframe; }};

let dragging = false, lx = 0, ly = 0;
canvas.addEventListener('mousedown', e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lx, dy = e.clientY - ly;
  lx = e.clientX; ly = e.clientY;
  if (e.shiftKey) {{
    const s = dist * 0.0018;
    target[0] -= (Math.cos(yaw) * dx) * s;
    target[1] += dy * s;
    target[2] += (Math.sin(yaw) * dx) * s;
  }} else {{
    yaw -= dx * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  }}
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.001);
}}, {{ passive: false }});

// A/B wipe slider: 0 = full A, 1 = full B
const slider = document.getElementById('abSlider');
function splitNdc() {{ return -1.4 + (slider.value / 1000) * 2.8; }}

function mat4Mul(a, b) {{
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return o;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}}
function lookAt(eye, ctr, up) {{
  let zx = eye[0]-ctr[0], zy = eye[1]-ctr[1], zz = eye[2]-ctr[2];
  let zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
  let xx = up[1]*zz - up[2]*zy, xy = up[2]*zx - up[0]*zz, xz = up[0]*zy - up[1]*zx;
  let xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
  let yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
  return new Float32Array([
    xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
    -(xx*eye[0]+xy*eye[1]+xz*eye[2]), -(yx*eye[0]+yy*eye[1]+yz*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1
  ]);
}}

const uVP = gl.getUniformLocation(prog, 'uVP');
const uWire = gl.getUniformLocation(prog, 'uWire');
const uSplit = gl.getUniformLocation(prog, 'uSplit');
const uSide = gl.getUniformLocation(prog, 'uSide');

function frame() {{
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; }}
  gl.viewport(0, 0, w, h);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.08, 0.10, 0.15, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // one shared camera for both sides (linked views)
  const eye = [
    target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
    target[1] + dist * Math.sin(pitch),
    target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
  ];
  const vp = mat4Mul(perspective(Math.PI / 4, w / Math.max(h, 1), dist * 0.01, dist * 100), lookAt(eye, target, [0, 1, 0]));
  gl.uniformMatrix4fv(uVP, false, vp);
  gl.uniform1i(uWire, wireframe ? 1 : 0);
  gl.uniform1f(uSplit, splitNdc());
  gl.uniform1i(uSide, 0);
  gl.bindVertexArray(meshA.vao);
  gl.drawElements(wireframe ? gl.LINES : gl.TRIANGLES, meshA.count, gl.UNSIGNED_INT, 0);
  gl.uniform1i(uSide, 1);
  gl.bindVertexArray(meshB.vao);
  gl.drawElements(wireframe ? gl.LINES : gl.TRIANGLES, meshB.count, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
  requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
</script>
</body>
</html>
"#
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
