# Contributing to ParaViz

Thank you for your interest in contributing to **ParaViz**!

## Reporting Issues & Suggesting Features
- Use the [GitHub Issue Templates](https://github.com/DimitriTimoz/paraviz/issues/new/choose) to report bugs or request ParaView filters, file formats, or rendering features.
- For bugs, please include your OS, GPU/WebGPU backend, CLI command (or **Rust Trace** / `.json` state), and steps to reproduce.

## Development Setup
1. Install a recent stable Rust toolchain (`rustup toolchain install stable`).
2. On Linux, install the X11/Wayland and Vulkan development packages listed in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
3. Run checks and tests before opening a pull request:
   ```bash
   cargo check --workspace --all-targets
   cargo clippy --workspace --all-targets
   cargo doc --workspace --exclude paraviz-app --no-deps
   cargo test --workspace
   ```

## Architecture & Style
- See [`docs/architecture.md`](docs/architecture.md) for the workspace crate layout (`paraviz-core`, `paraviz-sources`, `paraviz-filters`, `paraviz-io`, `paraviz-render`, `paraviz`, `paraviz-app`, and `paraviz-py`).
- Keep modules focused and well-organized, add tests for new filters, readers/writers, or rendering features, and update [`docs/`](docs/README.md) when adding user-facing capabilities.

## License
By contributing, you agree that your contributions will be dual-licensed under `MIT OR Apache-2.0`.
