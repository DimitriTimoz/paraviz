<p align="center">
  <img src="assets/hero-banner.svg" width="840" alt="ParaViz — Modern 3D Scientific Visualization Studio">
</p>

<p align="center">
  <a href="https://dimitritimoz.github.io/paraviz/"><img src="https://img.shields.io/badge/Try_in_Browser-WebGPU-0ea5e9.svg?style=for-the-badge" alt="Try Live Web App"></a>
  <a href="https://github.com/DimitriTimoz/paraviz/releases"><img src="https://img.shields.io/github/v/release/DimitriTimoz/paraviz?include_prereleases&style=for-the-badge&color=38bdf8" alt="Download Latest Release"></a>
</p>

<p align="center">
  <a href="https://github.com/DimitriTimoz/paraviz/actions/workflows/ci.yml"><img src="https://github.com/DimitriTimoz/paraviz/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Rust-2024_%2B_WebGPU-6366f1.svg" alt="Rust + WebGPU">
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://dimitritimoz.github.io/paraviz/"><strong>Live Web App</strong></a> ·
  <a href="#get-started">Download</a> ·
  <a href="#built-for-your-workflow">Workflows</a> ·
  <a href="#supported-formats">Formats</a> ·
  <a href="#why-paraviz">Why ParaViz</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="https://github.com/DimitriTimoz/paraviz/issues/new/choose">Feedback</a>
</p>

<h3 align="center">
  Fast, zero-bloat 3D scientific visualization — in a single native app or directly in your browser.
</h3>

Explore CFD simulations, finite-element meshes, medical scans, and 3D point clouds without waiting on heavy C++/Qt installs. **ParaViz** brings GPU-accelerated rendering, instant feedback, and 1-click shareable exports to engineers, researchers, and scientists.

<p align="center">
  <img src="assets/showcase.png" width="820" alt="ParaViz Studio Showcase">
</p>

---

## Get Started

### 1. Try Instantly in Your Browser (No Install)
Open [**dimitritimoz.github.io/paraviz**](https://dimitritimoz.github.io/paraviz/) and drag-and-drop your dataset. Works on any modern browser with WebGPU or WebGL2.

### 2. Download the Desktop App
Grab a standalone binary from [**GitHub Releases**](https://github.com/DimitriTimoz/paraviz/releases) — no external runtime or VTK dependencies required:

| Platform | Architectures | Download |
| --- | --- | --- |
| **macOS** | Apple Silicon + Intel (Universal) | `.dmg` (`ParaViz.app`) or `.tar.gz` |
| **Windows** | `x86_64`, `aarch64` | `.zip` (`paraviz.exe`) |
| **Linux** | `x86_64`, `aarch64` | `.AppImage` (portable), `.deb`, or `.tar.gz` |

---

## Built for Your Workflow

- **CFD & Aerodynamics**: Inspect OpenFOAM, CGNS, EnSight, PLOT3D, Tecplot, XDMF, and YALES2 simulations with interactive slices, clips (`Min/Max` or `Center/Size`), streamlines, vorticity/Q-criterion, and line probes.
- **FEA & Structural Analysis**: Load Exodus II, Abaqus (`.inp`), Gmsh (`.msh`), and VTK meshes with crisp edge rendering, displacement warping, and multi-block patch inspection.
- **Medical & Volumetric Imaging**: Explore NIfTI, NRRD, MetaImage, DICOM, and raw 3D volumes with real-time 3D raymarching, interactive opacity curves, and live volume clip planes.
- **3D Meshes, LiDAR & Molecules**: Open STL, OBJ, PLY, OFF, glTF/GLB, X3D, LAS point clouds, and multi-frame XYZ/PDB molecular trajectories.
- **Share Results in One Click**:
  - **Interactive HTML (`viewer.html`)** — send a self-contained 3D view that anyone can orbit in their browser
  - **Animated GIF Turntables** — 360° loops and time-series animations for slides & papers
  - **Publication PNGs** — high-res renders with optional transparent background, color bar, and time badge
  - **Binary glTF (`.glb`)** — export colored meshes straight into Blender, web viewers, or presentations

---

## Why ParaViz

ParaViz rethinks the everyday scientific visualization experience from the ground up:

- **No More Menu Hunting (`Ctrl/Cmd + K`)**: Jump straight to any filter, action, or favorite (`★`) from a keyboard-first Command Palette ranked by what you actually use.
- **Focused Workspace Profiles**: Switch between **Full**, **CFD**, **Medical**, and **Mesh** profiles to declutter toolbars and surface only the tools you need.
- **Instant Feedback + Full Undo**: **Auto-Apply** updates the 3D view live as you drag sliders, backed by **Undo/Redo (`Ctrl/Cmd + Z`)**, non-blocking background workers, and **Copy/Paste Filter Chains**.
- **Clean Visuals Out of the Box**: Anti-aliased mesh edges with zero Z-fighting, smooth volume raymarching, Eye-Dome Lighting (EDL), 4x MSAA, split-screen `3D + XY Chart + Spreadsheet` views, and Dark/Light studio themes.

| Everyday Workflow | Classic ParaView | ParaViz |
| --- | --- | --- |
| **Install & Footprint** | Multi-hundred-MB C++/Qt/Python install | Single native binary or instant Web App |
| **Quick Search** | `Ctrl+Space` (mouse-heavy, unranked) | `Ctrl/Cmd+K` / `Ctrl+Space` with favorites & frequency ranking |
| **Editing Properties** | Manual *Apply* click on every change | **Auto-Apply** + **Undo/Redo** (`Ctrl/Cmd+Z`) |
| **Domain Clutter** | Hundreds of mixed filters | **Workspace Profiles** (`CFD`, `Medical`, `Mesh`, `Full`) |
| **Reusing Filters** | Manual recreation | **Copy / Paste Filter & Filter Chain** (`Ctrl/Cmd+C` / `V`) |
| **Sharing 3D Views** | Requires external web tooling | Built-in **`.html` 3D viewer**, **`.gif`**, **`.glb`**, and **PNG** |

> **Current Scope (`v0.2.x`)**: Built for fast single-workstation visualization, batch CLI rendering, and Rust/Python automation. Distributed MPI (`pvserver`), OSPRay/OptiX path tracing, and proprietary binary ANSYS Fluent `.cas`/`.dat` or STEP/IGES CAD kernels are out of scope. See [**Differences from Classic ParaView**](docs/differences-from-paraview.md).

---

## Supported Formats

- **VTK & Time Series**: `.vtk`, `.vtp`, `.vti`, `.vtu`, `.vtr`, `.vts`, `.vtm`, `.pvtp`/`.pvti`/`.pvtu`/`.pvtr`/`.pvts`, `.pvd`, `.series`, `.vtkhdf`
- **CFD / FEA / Climate**: `.cgns`, `.exo`/`.e`/`.ex2` (Exodus II), `.h5` (YALES2 / VTKHDF / CGNS / Exodus), `.xdmf`/`.xmf`, `.case` (EnSight), `.foam` / `polyMesh` (OpenFOAM), `.msh` (Gmsh), `.xyz`/`.p3d`/`.q` (PLOT3D), `.dat`/`.tec`/`.plt` (Tecplot), `.inp` (Abaqus), `.nc`/`.cdf` (NetCDF CF)
- **3D Meshes & Point Clouds**: `.stl`, `.obj`, `.ply`, `.off`, `.gltf`/`.glb`, `.x3d`, `.las`, `.csv`/`.tsv`
- **Medical, Images & Molecular**: `.nii`/`.nii.gz` (NIfTI), `.nrrd`/`.nhdr`, `.mhd`/`.mha`, `.dcm` (DICOM), `.raw`/`.bin`/`.vol`, `.png`, `.bmp`, `.ppm`/`.pgm`/`.pbm`, `.xyz`/`.pdb`/`.ent`

See the full [**File Formats & I/O Reference**](docs/file-formats.md).

---

## Automate & Embed (CLI, Rust & Python)

Batch-render screenshots, convert formats, or embed visualization pipelines in your own tools:

```bash
# Render a dataset with contour, color bar, and Eye-Dome Lighting
paraviz render volume.vti -o iso.png --filter contour:140 --cmap viridis --edl --scalar-bar

# Export an interactive 3D HTML viewer or 360° GIF turntable
paraviz convert mesh.vtu viewer.html
paraviz render mesh.vtu -o spin.gif --turntable 36
```

- [**Headless CLI & GUI Guide**](docs/gui-and-cli.md)
- [**Rust API & Python Bindings (`paraviz-py`)**](docs/rust-api.md)
- [**Sources & Filters Reference**](docs/filters-and-sources.md)
- [**Architecture & WebGPU Pipeline**](docs/architecture.md)

<details>
<summary><strong>Build from Source & Run Tests (Developers)</strong></summary>

```bash
# Launch the desktop studio
cargo run --release -p paraviz-app

# Run the workspace test suite
cargo test --workspace
```

</details>

---

## Feedback & Contributing

Have a feature request, dataset that won't load, or bug report? [**Open an issue on GitHub**](https://github.com/DimitriTimoz/paraviz/issues/new/choose).

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
