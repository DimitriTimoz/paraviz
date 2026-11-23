# Agent Guide — ParaViz

## Project Context

**ParaViz** is a scientific visualization studio written in Rust and powered by WebGPU (`wgpu`), reimagining Kitware ParaView with a desktop GUI, WebAssembly web app, headless CLI, and Python bindings.

### Workspace Architecture

| Crate / Path | Responsibility |
|---|---|
| `.` (`paraviz`) | Top-level fluent API, pipeline graph, and JSON session persistence ([`src/session.rs`](src/session.rs)). |
| `crates/paraviz-core` | Core scientific data model (`DataSet`, `PolyData`, `UnstructuredGrid`, `ImageData`, `RectilinearGrid`, `StructuredGrid`, `MultiBlockDataSet`), typed `DataArray`, `ColorMap`, and `Source` / `Filter` traits. |
| `crates/paraviz-sources` | Procedural data generators (`WaveletSource`, `SphereSource`, `PlaneSource`, `CylinderSource`, vector fields, etc.). |
| `crates/paraviz-filters` | Data processing filters (`ContourFilter`, `ClipFilter`, `SliceFilter`, `ThresholdFilter`, `CalculatorFilter`, `StreamTracerFilter`, `DecimateFilter`, `DelaunayFilter`, `ExtractSurface`, etc.). |
| `crates/paraviz-io` | Scientific file format readers/writers (VTK Legacy & XML, XDMF/HDF5, NIfTI, MetaImage, NRRD, EnSight, OpenFOAM, STL/OBJ/PLY, CSV, GLB, GIF, WebGL2 HTML viewer) and shared security bounds (`limits.rs`). |
| `crates/paraviz-render` | WebGPU (`wgpu`) rendering engine, WGSL shaders, EDL, MSAA, volume raymarching, and picking. |
| `crates/paraviz-app` | `eframe`/`egui` desktop & WASM GUI and headless CLI (`paraviz`). |
| `crates/paraviz-py` | Standalone PyO3 Python bindings (`import paraviz`) with its own `[workspace]` and `Cargo.lock`. |
| `packaging/` & `.github/workflows/` | Linux AppImage / `.deb`, macOS `.app`/`.dmg`, Windows `.zip` packaging, CI, and GitHub Pages deployment. |

### Engineering & Security Invariants

- **Zero `unsafe` code** across workspace crates.
- **Untrusted file parsing**: All readers in `paraviz-io` must use bounded decompression (`limits::MAX_DECOMPRESSED_BYTES`), checked integer arithmetic and bounded pre-allocations (`limits::safe_prealloc_capacity`), strict relative path resolution (`limits::resolve_safe_path`), and XML/expression recursion limits.
- **Release overflow checks**: `[profile.release]` enables `overflow-checks = true` in `Cargo.toml`.
- **HTML export**: Embedded JSON in `<script>` tags must escape `<`, `>`, `&`, and JS line terminators.
- **CI/CD**: Workflows default to `permissions: contents: read`, never interpolate `${{ ... }}` directly inside `run:` scripts (always pass via `env:` and validate), and verify downloaded release tooling with pinned versions and SHA-256 checksums.
- **Always update documentation**: Every change to features, GUI controls, CLI flags, supported file formats, filters/sources, rendering/scene options, public Rust/Python APIs, or architecture must update the relevant documentation (`README.md`, `docs/` guides, rustdoc comments, and CLI `--help` text) before finishing or committing.
- **Always commit completed work**: Never leave verified changes uncommitted at the end of a task; create clean, atomic Conventional Commits once checks pass.

### Pre-Commit Verification

Run these checks before committing:

```bash
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets
cargo test --workspace
```

If `crates/paraviz-py` or public `paraviz` APIs changed, also verify:

```bash
cargo check --manifest-path crates/paraviz-py/Cargo.toml
```

---

## Commit Guidelines

Follow **Conventional Commits** with small, atomic, single-purpose commits.

### Format

```text
<type>(<scope>): <imperative summary>

[optional body explaining WHY, wrapped at 72 characters]
```

- **Types**:
  - `feat`: new feature or reader/filter capability
  - `fix`: bug fix
  - `security`: vulnerability or hardening fix
  - `refactor`: internal restructuring without behavior change
  - `test`: adding or updating tests
  - `docs`: documentation (`README.md`, `AGENT.md`, `docs/`, etc.)
  - `ci`: GitHub Actions workflows
  - `build` / `packaging`: `Cargo.toml`, lockfiles, packaging scripts
  - `perf`: performance improvement
- **Scopes**: use crate or subsystem names (`core`, `io`, `filters`, `render`, `sources`, `app`, `py`, `session`, `ci`, `packaging`), or comma-separated scopes when tightly coupled (e.g. `io,app`).
- **Subject rules**:
  - Use the imperative mood ("harden", "fix", "add", not "hardened" or "adds").
  - Keep it concise (ideally <= 72 characters) and do not end with a period.

### Best Practices

1. **One logical change per commit**: Separate security hardening in `paraviz-io`, `paraviz-core`/`paraviz-filters`, CI/packaging, `paraviz-py`, and documentation into distinct commits so each can be reviewed, cherry-picked, or reverted independently.
2. **Include tests with the fix**: Add regression tests alongside the code they verify (or in the dedicated test commit for that layer).
3. **Always keep docs in sync**: Update `README.md`, relevant files in `docs/` (`architecture.md`, `differences-from-paraview.md`, `file-formats.md`, `filters-and-sources.md`, `gui-and-cli.md`, `rendering-and-scene.md`, `rust-api.md`, `user-guide.md`), and `--help`/docstrings whenever behavior, UI, CLI, or APIs change.
4. **Never commit untracked junk or secrets**: Check `git status` and `git diff --cached` before committing; never commit `.DS_Store`, `target/`, or credentials.
5. **Never bypass hooks or checks**: Do not use `--no-verify` to skip failing checks.
6. **Always commit before finishing**: After passing verification, commit all intended changes in logical Conventional Commits and confirm `git status` is clean.
