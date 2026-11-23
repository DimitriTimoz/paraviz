//! Background release checker and SHA-256 verified self-updater for native desktop builds.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

const GITHUB_LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/DimitriTimoz/paraviz/releases/latest";
const MAX_RELEASE_JSON_BYTES: u64 = 2 * 1024 * 1024;
const MAX_SUMS_BYTES: u64 = 1024 * 1024;
const MAX_ASSET_BYTES: u64 = 256 * 1024 * 1024;
const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

pub const RELEASES_PAGE_URL: &str = "https://github.com/DimitriTimoz/paraviz/releases";

#[derive(Debug, Clone)]
pub struct UpdateAsset {
    pub name: String,
    pub download_url: String,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct ReleaseUpdateInfo {
    pub version: String,
    pub tag_name: String,
    pub html_url: String,
    pub notes: String,
    pub asset: Option<UpdateAsset>,
    pub checksums_url: Option<String>,
    pub install_kind: InstallKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallKind {
    AppImage,
    ArchiveBinary,
    CargoOrManual,
}

#[derive(Debug, Clone)]
pub enum UpdateStatus {
    Idle,
    Checking,
    UpToDate { current: String },
    Available(ReleaseUpdateInfo),
    Downloading { version: String, asset_name: String },
    Installed { version: String },
    Error(String),
}

pub enum UpdateEvent {
    Checked(Result<Option<ReleaseUpdateInfo>, String>),
    Installed(Result<String, String>),
}

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn detect_install_kind() -> InstallKind {
    if std::env::var_os("APPIMAGE").is_some() {
        return InstallKind::AppImage;
    }
    if let Ok(exe) = std::env::current_exe() {
        let s = exe.to_string_lossy();
        if s.contains("/.cargo/bin/") || s.contains("\\.cargo\\bin\\") {
            return InstallKind::CargoOrManual;
        }
    }
    if matches!(std::env::consts::OS, "linux" | "macos" | "windows") {
        InstallKind::ArchiveBinary
    } else {
        InstallKind::CargoOrManual
    }
}

pub fn is_newer_version(current: &str, candidate_tag: &str) -> Result<bool, String> {
    let cur = semver::Version::parse(current.trim().trim_start_matches('v'))
        .map_err(|e| format!("Invalid current semver '{current}': {e}"))?;
    let next = semver::Version::parse(candidate_tag.trim().trim_start_matches('v'))
        .map_err(|e| format!("Invalid release semver '{candidate_tag}': {e}"))?;
    Ok(next > cur)
}

pub fn spawn_check_updates(tx: mpsc::Sender<UpdateEvent>) {
    std::thread::spawn(move || {
        let res = check_latest_release();
        let _ = tx.send(UpdateEvent::Checked(res));
    });
}

pub fn spawn_install_update(info: ReleaseUpdateInfo, tx: mpsc::Sender<UpdateEvent>) {
    std::thread::spawn(move || {
        let res = download_and_install_update(&info);
        let _ = tx.send(UpdateEvent::Installed(res));
    });
}

pub fn check_latest_release() -> Result<Option<ReleaseUpdateInfo>, String> {
    let agent = http_agent();
    let mut resp = agent
        .get(GITHUB_LATEST_RELEASE_URL)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", &format!("paraviz/{}", current_version()))
        .call()
        .map_err(|e| format!("Failed to reach GitHub Releases API: {e}"))?;

    let bytes = read_body_limited(resp.body_mut(), MAX_RELEASE_JSON_BYTES)?;
    let release: GitHubRelease = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Failed to parse GitHub release metadata: {e}"))?;

    if release.draft || release.prerelease {
        return Ok(None);
    }

    if !is_newer_version(current_version(), &release.tag_name)? {
        return Ok(None);
    }

    let install_kind = detect_install_kind();
    let asset = select_platform_asset(
        &release.assets,
        std::env::consts::OS,
        std::env::consts::ARCH,
        install_kind == InstallKind::AppImage,
    );
    let checksums_url = release
        .assets
        .iter()
        .find(|a| a.name == "SHA256SUMS.txt")
        .map(|a| a.browser_download_url.clone());

    let clean_ver = release.tag_name.trim_start_matches('v').to_string();
    Ok(Some(ReleaseUpdateInfo {
        version: clean_ver,
        tag_name: release.tag_name,
        html_url: release.html_url,
        notes: release.body,
        asset,
        checksums_url,
        install_kind,
    }))
}

fn select_platform_asset(
    assets: &[GitHubAsset],
    os: &str,
    arch: &str,
    is_appimage: bool,
) -> Option<UpdateAsset> {
    let find_matching = |pred: &dyn Fn(&str) -> bool| -> Option<UpdateAsset> {
        assets.iter().find(|a| pred(&a.name)).map(|a| UpdateAsset {
            name: a.name.clone(),
            download_url: a.browser_download_url.clone(),
            size: a.size,
        })
    };

    match os {
        "linux" => {
            if is_appimage
                && let Some(appimg) =
                    find_matching(&|n| n.ends_with(".AppImage") && n.contains(arch))
            {
                return Some(appimg);
            }
            find_matching(&|n| n.contains(&format!("linux-{arch}")) && n.ends_with(".tar.gz"))
        }
        "macos" => find_matching(&|n| n.contains("macos-universal") && n.ends_with(".tar.gz")),
        "windows" => {
            find_matching(&|n| n.contains(&format!("windows-{arch}")) && n.ends_with(".zip"))
        }
        _ => None,
    }
}

fn download_and_install_update(info: &ReleaseUpdateInfo) -> Result<String, String> {
    let asset = info
        .asset
        .as_ref()
        .ok_or_else(|| "No compatible binary package found for this OS/architecture.".to_string())?;
    let sums_url = info
        .checksums_url
        .as_ref()
        .ok_or_else(|| "Release is missing SHA256SUMS.txt verification manifest.".to_string())?;

    let agent = http_agent();
    let user_agent = format!("paraviz/{}", current_version());

    let mut sums_resp = agent
        .get(sums_url)
        .header("User-Agent", &user_agent)
        .call()
        .map_err(|e| format!("Failed to download SHA256SUMS.txt: {e}"))?;
    let sums_bytes = read_body_limited(sums_resp.body_mut(), MAX_SUMS_BYTES)?;
    let sums_text = std::str::from_utf8(&sums_bytes)
        .map_err(|e| format!("SHA256SUMS.txt is not valid UTF-8: {e}"))?;
    let expected_sha256 = parse_sha256sums_entry(sums_text, &asset.name).ok_or_else(|| {
        format!(
            "SHA256SUMS.txt does not contain a checksum for '{}'",
            asset.name
        )
    })?;

    let mut pkg_resp = agent
        .get(&asset.download_url)
        .header("User-Agent", &user_agent)
        .call()
        .map_err(|e| format!("Failed to download '{}': {e}", asset.name))?;
    let pkg_bytes = read_body_limited(pkg_resp.body_mut(), MAX_ASSET_BYTES)?;

    verify_sha256(&pkg_bytes, &expected_sha256)?;

    let staging_dir = tempfile_dir()?;
    let bin_name = if cfg!(windows) { "paraviz.exe" } else { "paraviz" };
    let staged_exe = staging_dir.join(bin_name);

    if asset.name.ends_with(".tar.gz") {
        extract_binary_from_targz(&pkg_bytes, bin_name, &staged_exe)?;
    } else if asset.name.ends_with(".zip") {
        extract_binary_from_zip(&pkg_bytes, bin_name, &staged_exe)?;
    } else if asset.name.ends_with(".AppImage") {
        fs::write(&staged_exe, &pkg_bytes)
            .map_err(|e| format!("Failed to write staged AppImage: {e}"))?;
    } else {
        return Err(format!("Unsupported release archive '{}'", asset.name));
    }

    set_executable_permissions(&staged_exe)?;

    if info.install_kind == InstallKind::AppImage
        && let Some(appimage_path) = std::env::var_os("APPIMAGE")
    {
        let target = PathBuf::from(appimage_path);
        fs::rename(&staged_exe, &target)
            .or_else(|_| fs::copy(&staged_exe, &target).map(|_| ()))
            .map_err(|e| {
                format!(
                    "Failed to replace AppImage at '{}': {e}",
                    target.display()
                )
            })?;
    } else {
        self_replace::self_replace(&staged_exe).map_err(|e| {
            format!("Failed to replace executable cleanly (permissions?): {e}")
        })?;
    }

    let _ = fs::remove_dir_all(&staging_dir);
    Ok(info.version.clone())
}

pub fn parse_sha256sums_entry(contents: &str, filename: &str) -> Option<String> {
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let hash = parts.next()?.trim();
        let raw_name = parts.next()?.trim().trim_start_matches('*');
        let base_name = Path::new(raw_name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(raw_name);
        if (raw_name == filename || base_name == filename)
            && hash.len() == 64
            && hash.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Some(hash.to_ascii_lowercase());
        }
    }
    None
}

pub fn verify_sha256(bytes: &[u8], expected_hex: &str) -> Result<(), String> {
    let digest = Sha256::digest(bytes);
    let mut actual_hex = String::with_capacity(64);
    for b in digest {
        let _ = write!(&mut actual_hex, "{b:02x}");
    }
    if actual_hex.eq_ignore_ascii_case(expected_hex.trim()) {
        Ok(())
    } else {
        Err(format!(
            "SHA-256 verification failed (expected {expected_hex}, got {actual_hex})"
        ))
    }
}

fn extract_binary_from_targz(
    archive_bytes: &[u8],
    bin_name: &str,
    dest: &Path,
) -> Result<(), String> {
    let gz = flate2::read::GzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = tar::Archive::new(gz);
    let entries = archive
        .entries()
        .map_err(|e| format!("Failed to read tar.gz entries: {e}"))?;
    for entry_res in entries {
        let mut entry = entry_res.map_err(|e| format!("Corrupt tar entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("Invalid tar entry path: {e}"))?;
        let file_name = path.file_name().and_then(|s| s.to_str());
        if file_name == Some(bin_name) && entry.header().entry_type().is_file() {
            let mut out = File::create(dest)
                .map_err(|e| format!("Failed to create staged binary '{}': {e}", dest.display()))?;
            let mut buf = Vec::new();
            (&mut entry)
                .take(MAX_ASSET_BYTES + 1)
                .read_to_end(&mut buf)
                .map_err(|e| format!("Failed to extract binary from tar.gz: {e}"))?;
            if (buf.len() as u64) > MAX_ASSET_BYTES {
                return Err("Extracted binary exceeds maximum allowed size.".into());
            }
            std::io::Write::write_all(&mut out, &buf)
                .map_err(|e| format!("Failed to write staged binary: {e}"))?;
            return Ok(());
        }
    }
    Err(format!("Archive did not contain '{bin_name}'"))
}

fn extract_binary_from_zip(
    archive_bytes: &[u8],
    bin_name: &str,
    dest: &Path,
) -> Result<(), String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(archive_bytes))
        .map_err(|e| format!("Failed to open zip archive: {e}"))?;
    for idx in 0..zip.len() {
        let mut entry = zip
            .by_index(idx)
            .map_err(|e| format!("Corrupt zip entry: {e}"))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry
            .name()
            .map_err(|e| format!("Invalid zip entry name: {e}"))?
            .into_owned();
        let file_name = Path::new(&name).file_name().and_then(|s| s.to_str());
        if file_name == Some(bin_name) {
            let mut buf = Vec::new();
            (&mut entry)
                .take(MAX_ASSET_BYTES + 1)
                .read_to_end(&mut buf)
                .map_err(|e| format!("Failed to read '{bin_name}' from zip: {e}"))?;
            if (buf.len() as u64) > MAX_ASSET_BYTES {
                return Err("Extracted zip binary exceeds maximum allowed size.".into());
            }
            fs::write(dest, &buf)
                .map_err(|e| format!("Failed to write staged binary '{}': {e}", dest.display()))?;
            return Ok(());
        }
    }
    Err(format!("Zip archive did not contain '{bin_name}'"))
}

#[cfg(unix)]
fn set_executable_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)
        .map_err(|e| format!("Failed to read file metadata: {e}"))?
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)
        .map_err(|e| format!("Failed to set executable permissions: {e}"))
}

#[cfg(not(unix))]
fn set_executable_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn tempfile_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("paraviz-update-{}", std::process::id()));
    if dir.exists() {
        let _ = fs::remove_dir_all(&dir);
    }
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create staging dir: {e}"))?;
    Ok(dir)
}

fn http_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(HTTP_TIMEOUT))
        .build();
    config.into()
}

fn read_body_limited(body: &mut ureq::Body, max_bytes: u64) -> Result<Vec<u8>, String> {
    let mut reader = body.as_reader().take(max_bytes + 1);
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("Failed reading HTTP response: {e}"))?;
    if (buf.len() as u64) > max_bytes {
        return Err(format!("HTTP payload exceeds {max_bytes} byte limit"));
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn semver_comparison_handles_v_prefix() {
        assert!(is_newer_version("0.2.5", "v0.2.6").unwrap());
        assert!(is_newer_version("0.2.5", "v0.3.0").unwrap());
        assert!(!is_newer_version("0.2.5", "v0.2.5").unwrap());
        assert!(!is_newer_version("0.2.6", "v0.2.5").unwrap());
    }

    #[test]
    fn parse_sha256sums_and_verify_digest() {
        let data = b"paraviz-binary-payload";
        let digest = Sha256::digest(data);
        let mut hex = String::new();
        for b in digest {
            let _ = write!(&mut hex, "{b:02x}");
        }
        let manifest = format!(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  other.tar.gz\n\
             {hex}  ./paraviz-v0.2.6-linux-x86_64.tar.gz\n"
        );
        let parsed =
            parse_sha256sums_entry(&manifest, "paraviz-v0.2.6-linux-x86_64.tar.gz").unwrap();
        assert_eq!(parsed, hex);
        assert!(verify_sha256(data, &parsed).is_ok());
        assert!(verify_sha256(b"tampered", &parsed).is_err());
    }

    #[test]
    fn platform_asset_selection_prefers_appimage_when_applicable() {
        let assets = vec![
            GitHubAsset {
                name: "paraviz-v0.2.6-linux-x86_64.tar.gz".into(),
                browser_download_url: "https://example.invalid/linux.tar.gz".into(),
                size: 100,
            },
            GitHubAsset {
                name: "ParaViz-v0.2.6-x86_64.AppImage".into(),
                browser_download_url: "https://example.invalid/linux.AppImage".into(),
                size: 200,
            },
            GitHubAsset {
                name: "paraviz-v0.2.6-macos-universal.tar.gz".into(),
                browser_download_url: "https://example.invalid/mac.tar.gz".into(),
                size: 300,
            },
            GitHubAsset {
                name: "paraviz-v0.2.6-windows-x86_64.zip".into(),
                browser_download_url: "https://example.invalid/win.zip".into(),
                size: 400,
            },
        ];

        assert_eq!(
            select_platform_asset(&assets, "linux", "x86_64", true)
                .unwrap()
                .name,
            "ParaViz-v0.2.6-x86_64.AppImage"
        );
        assert_eq!(
            select_platform_asset(&assets, "linux", "x86_64", false)
                .unwrap()
                .name,
            "paraviz-v0.2.6-linux-x86_64.tar.gz"
        );
        assert_eq!(
            select_platform_asset(&assets, "macos", "aarch64", false)
                .unwrap()
                .name,
            "paraviz-v0.2.6-macos-universal.tar.gz"
        );
        assert_eq!(
            select_platform_asset(&assets, "windows", "x86_64", false)
                .unwrap()
                .name,
            "paraviz-v0.2.6-windows-x86_64.zip"
        );
    }

    #[test]
    fn extracts_binary_from_targz_and_zip() {
        let tmp = tempfile_dir().unwrap();
        let payload = b"#!/bin/sh\necho paraviz\n";

        // Test tar.gz extraction
        let mut tgz_buf = Vec::new();
        {
            let gz = flate2::write::GzEncoder::new(&mut tgz_buf, flate2::Compression::default());
            let mut builder = tar::Builder::new(gz);
            let mut hdr = tar::Header::new_gnu();
            hdr.set_size(payload.len() as u64);
            hdr.set_mode(0o755);
            hdr.set_cksum();
            builder.append_data(&mut hdr, "paraviz", &payload[..]).unwrap();
            let gz = builder.into_inner().unwrap();
            gz.finish().unwrap();
        }
        let out_tgz = tmp.join("paraviz_from_tgz");
        extract_binary_from_targz(&tgz_buf, "paraviz", &out_tgz).unwrap();
        assert_eq!(fs::read(&out_tgz).unwrap(), payload);

        // Test zip extraction
        let mut zip_buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut zip_buf));
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("paraviz.exe", opts).unwrap();
            zw.write_all(payload).unwrap();
            zw.finish().unwrap();
        }
        let out_zip = tmp.join("paraviz_from_zip.exe");
        extract_binary_from_zip(&zip_buf, "paraviz.exe", &out_zip).unwrap();
        assert_eq!(fs::read(&out_zip).unwrap(), payload);

        let _ = fs::remove_dir_all(&tmp);
    }
}
