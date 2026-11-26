//! Native project file I/O, relative dataset paths, and self-contained exports.

use crate::embedded_source::{EmbeddedFrame, EmbeddedSource};
use crate::session::{ParaViz, PipelineState, ProjectPathBase, SessionError, SourceConfig};
use paraviz_core::PipelineError;
use paraviz_io::{IoError, resolve_safe_path};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Controls whether a saved project links to datasets or includes their evaluated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectSaveMode {
    /// Preserve original readers and use relative paths for files under the project directory.
    Referenced,
    /// Include every advertised time step of each file source in the project JSON.
    Embedded,
}

impl ParaViz {
    /// Writes a native project atomically so an unsuccessful save leaves any existing file intact.
    pub fn save_project(
        &self,
        path: impl AsRef<Path>,
        mode: ProjectSaveMode,
    ) -> Result<(), SessionError> {
        let path = project_location(path.as_ref())?;
        let directory = project_directory(&path)?;
        let mut state = self.state_snapshot()?;
        for node in &mut state.nodes {
            let Some(config) = &mut node.source else {
                continue;
            };
            let Some(source_path) = file_path(config) else {
                continue;
            };
            match mode {
                ProjectSaveMode::Referenced => {
                    let absolute = std::path::absolute(&*source_path).map_err(IoError::from)?;
                    let relative = absolute
                        .strip_prefix(directory)
                        .ok()
                        .and_then(portable_relative_path)
                        .filter(|relative| resolve_safe_path(directory, relative).is_ok());
                    *source_path = match relative {
                        Some(relative) => {
                            state.project_path_base = Some(ProjectPathBase::ProjectDirectory);
                            PathBuf::from(relative)
                        }
                        None => absolute,
                    };
                }
                ProjectSaveMode::Embedded => {
                    let source = config.build_source()?;
                    let mut times = source.request_information().time_steps;
                    times.retain(|time| time.is_finite());
                    times.sort_by(f64::total_cmp);
                    times.dedup();
                    let frame_times: Vec<_> = if times.is_empty() {
                        vec![None]
                    } else {
                        times.into_iter().map(Some).collect()
                    };
                    let mut frames = Vec::with_capacity(frame_times.len());
                    for time in frame_times {
                        let at = time.unwrap_or(self.time);
                        let dataset = source.execute_arc(at).map_err(|error| {
                            PipelineError::ExecutionFailed(format!(
                                "Cannot embed '{}' at time {at}: {error}",
                                node.name
                            ))
                        })?;
                        frames.push(EmbeddedFrame { time, dataset });
                    }
                    *config = SourceConfig::Embedded(EmbeddedSource::new(frames));
                }
            }
        }

        let mut output = tempfile::NamedTempFile::new_in(directory).map_err(IoError::from)?;
        {
            let mut writer = BufWriter::new(output.as_file_mut());
            match mode {
                ProjectSaveMode::Referenced => serde_json::to_writer_pretty(&mut writer, &state)?,
                ProjectSaveMode::Embedded => serde_json::to_writer(&mut writer, &state)?,
            }
            writer.flush().map_err(IoError::from)?;
        }
        output
            .persist(&path)
            .map_err(|error| IoError::from(error.error))?;
        Ok(())
    }

    /// Opens a native project, resolving project-relative sources from the location of the JSON.
    pub fn load_project(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        Self::load_project_with_relocation(path, |_| None)
    }

    /// Opens a project, asking for a replacement when a referenced source has moved.
    /// The callback receives the resolved missing path once per distinct source.
    /// The project's JSON is unchanged until it is explicitly saved again.
    pub fn load_project_with_relocation(
        path: impl AsRef<Path>,
        mut relocate: impl FnMut(&Path) -> Option<PathBuf>,
    ) -> Result<Self, SessionError> {
        let path = project_location(path.as_ref())?;
        let directory = project_directory(&path)?;
        let file = File::open(&path).map_err(IoError::from)?;
        let mut state: PipelineState = serde_json::from_reader(BufReader::new(file))?;
        state.validate_size()?;
        let project_relative = state.project_path_base.take().is_some();
        let mut relocated = HashMap::new();
        for node in &mut state.nodes {
            let Some(source_path) = node.source.as_mut().and_then(file_path) else {
                continue;
            };
            let candidate = if project_relative && source_path.is_relative() {
                let relative = source_path.to_str().ok_or_else(|| {
                    IoError::InvalidData("Invalid text in project-relative dataset path".into())
                })?;
                resolve_safe_path(directory, relative)?;
                directory.join(relative)
            } else {
                std::path::absolute(&*source_path).map_err(IoError::from)?
            };
            *source_path = if candidate.try_exists().map_err(IoError::from)? {
                candidate
            } else if let Some(replacement) = relocated.get(&candidate) {
                PathBuf::from(replacement)
            } else {
                let replacement =
                    relocate(&candidate).ok_or_else(|| IoError::MissingFile(candidate.clone()))?;
                let replacement = std::path::absolute(replacement).map_err(IoError::from)?;
                if !replacement.try_exists().map_err(IoError::from)? {
                    return Err(IoError::MissingFile(replacement).into());
                }
                relocated.insert(candidate, replacement.clone());
                replacement
            };
        }
        Self::from_state_snapshot(state)
    }
}

fn file_path(config: &mut SourceConfig) -> Option<&mut PathBuf> {
    match config {
        SourceConfig::File(path)
        | SourceConfig::FileWithReader { path, .. }
        | SourceConfig::PvdSeries(path) => Some(path),
        _ => None,
    }
}

fn project_location(path: &Path) -> Result<PathBuf, SessionError> {
    Ok(std::path::absolute(path).map_err(IoError::from)?)
}

fn project_directory(path: &Path) -> Result<&Path, SessionError> {
    path.parent()
        .ok_or_else(|| IoError::InvalidData("Project path must name a file".into()).into())
}

fn portable_relative_path(relative: &Path) -> Option<String> {
    let parts: Vec<_> = relative
        .components()
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<_>>()?;
    let path = parts.join("/");
    (!path.is_empty() && path.trim() == path).then_some(path)
}
