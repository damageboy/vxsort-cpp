use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use uica_data::{DATAPACK_MANIFEST_SCHEMA_VERSION, DataPackManifest};

const DEFAULT_UICA_DATA_BASE_URL: &str = "https://uica.houmus.org/data";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchUicaDataConfig {
    pub target_cpus: Vec<String>,
    pub data_dir: PathBuf,
    pub base_url: String,
}

impl Default for FetchUicaDataConfig {
    fn default() -> Self {
        Self {
            target_cpus: Vec::new(),
            data_dir: PathBuf::from("uica-data"),
            base_url: DEFAULT_UICA_DATA_BASE_URL.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchUicaDataReport {
    pub manifest_path: PathBuf,
    pub arches: Vec<String>,
}

pub fn default_uica_data_base_url() -> &'static str {
    DEFAULT_UICA_DATA_BASE_URL
}

pub fn fetch_uica_data(config: &FetchUicaDataConfig) -> Result<FetchUicaDataReport, String> {
    if config.target_cpus.is_empty() {
        return Err("--target-cpu is required for fetch-uica-data".to_owned());
    }

    let manifest_bytes = fetch_bytes(&join_url(&config.base_url, "manifest.json"))?;
    let manifest: DataPackManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("failed to parse uiCA manifest: {error}"))?;
    if manifest.schema_version != DATAPACK_MANIFEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported uiCA manifest schema: {}",
            manifest.schema_version
        ));
    }

    fs::create_dir_all(&config.data_dir)
        .map_err(|error| format!("failed to create {}: {error}", config.data_dir.display()))?;
    let manifest_path = config.data_dir.join("manifest.json");
    fs::write(&manifest_path, &manifest_bytes)
        .map_err(|error| format!("failed to write {}: {error}", manifest_path.display()))?;

    let mut arches = Vec::new();
    for requested in &config.target_cpus {
        let (arch, entry) = manifest
            .architectures
            .iter()
            .find(|(arch, _)| arch.eq_ignore_ascii_case(requested))
            .ok_or_else(|| format!("architecture '{requested}' not found in uiCA manifest"))?;
        let bytes = fetch_bytes(&join_url(&config.base_url, &entry.path))?;
        let destination = config.data_dir.join(&entry.path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(&destination, bytes)
            .map_err(|error| format!("failed to write {}: {error}", destination.display()))?;
        arches.push(arch.clone());
    }

    Ok(FetchUicaDataReport {
        manifest_path,
        arches,
    })
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    if let Some(path) = url.strip_prefix("file://") {
        return fs::read(Path::new(path)).map_err(|error| format!("failed to read {url}: {error}"));
    }

    let response = ureq::get(url)
        .call()
        .map_err(|error| format!("failed to download {url}: {error}"))?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read response from {url}: {error}"))?;
    Ok(bytes)
}
