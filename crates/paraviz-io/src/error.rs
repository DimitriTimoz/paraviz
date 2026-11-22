//! Error types for `paraviz-io`.

use std::path::PathBuf;
use thiserror::Error;

/// Errors returned by scientific file format readers and writers.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("XML parse error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("XML attribute error: {0}")]
    XmlAttr(#[from] quick_xml::events::attributes::AttrError),

    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Missing referenced file: {0}")]
    MissingFile(PathBuf),

    #[error("Core error: {0}")]
    Core(#[from] paraviz_core::CoreError),
}

pub type Result<T> = std::result::Result<T, IoError>;
