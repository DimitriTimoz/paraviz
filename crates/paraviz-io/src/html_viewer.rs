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
    title: &str,
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

/// Encodes a [`PolyData`] mesh into a self-contained interactive WebGL2 HTML document.
pub fn encode_html_viewer(
    poly: &PolyData,
    title: &str,
    color_map: Option<&ColorMap>,
    scalar_array: Option<&str>,
) -> String {
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
        let (lo, hi) = arr.range(None);
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let val = arr.get_magnitude(i);
            let rgba = cmap.map_value(val, (lo, hi));
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
