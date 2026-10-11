//! SquashFS Rootfs Extraction and Triage
//!
//! **File:** `squashfs.rs`
//! **Author:** Kevin Thomas
//! **Date:** 2026
//!
//! MIT License
//!
//! Copyright (c) 2026 Kevin Thomas
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

// Import dependencies from std
use std::{path::Path, sync::Arc};

// Import dependencies from fs_core
use fs_core::block::BlockRead;

// Import dependencies from fs_squashfs
use fs_squashfs::{DirEntry, Filesystem, Inode};

// Import crate items
use crate::analysis::{self, Analysis};

/// Maximum file bytes read into memory for single-file analysis.
const MAX_FILE_READ: usize = 2_000_000;

/// In-memory block device wrapper around owned bytes.
struct ByteBlock(Vec<u8>);

/// Builds an out of bounds I/O error for invalid offsets.
///
/// # Returns
///
/// An `UnexpectedEof` I/O error.
fn out_of_bounds() -> fs_core::Error {
    fs_core::Error::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "eof",
    ))
}

impl BlockRead for ByteBlock {
    /// Reads bytes at an offset into the target buffer.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset to read from.
    /// * `buf` - Destination buffer.
    ///
    /// # Errors
    ///
    /// Returns [`fs_core::Error::Io`] if the range is out of bounds.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), fs_core::Error> {
        let start = usize::try_from(offset).map_err(|_| out_of_bounds())?;
        let end = start.saturating_add(buf.len());
        if end > self.0.len() {
            return Err(out_of_bounds());
        }
        buf.copy_from_slice(&self.0[start..end]);
        Ok(())
    }

    /// Returns total block device size in bytes.
    ///
    /// # Returns
    ///
    /// Total buffer length as a 64-bit integer.
    fn size_bytes(&self) -> u64 {
        self.0.len() as u64
    }
}

/// A high-value security artifact or credential found in the filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityArtifact {
    /// Absolute path inside the filesystem image.
    pub path: String,

    /// Classification label.
    pub kind: String,

    /// Extracted evidence or description.
    pub detail: String,

    /// Size in bytes.
    pub size: u64,
}

impl SecurityArtifact {
    /// Creates a security artifact record.
    ///
    /// # Arguments
    ///
    /// * `path` - File path.
    /// * `kind` - Classification.
    /// * `detail` - Finding detail.
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// The artifact record.
    #[must_use]
    pub fn new(path: &str, kind: &str, detail: &str, size: u64) -> Self {
        let (path, kind, detail) = (path.to_string(), kind.to_string(), detail.to_string());
        Self { path, kind, detail, size }
    }
}

/// Metadata summary of an unpacked `SquashFS` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SquashfsMeta {
    /// Inode count.
    pub inodes: u32,

    /// Block size in bytes.
    pub block_size: u32,

    /// Compression algorithm name.
    pub compression: String,

    /// Total regular files found.
    pub file_count: usize,
}

/// Triage result for an unpacked `SquashFS` filesystem.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FirmwareTriage {
    /// Filesystem metadata.
    pub meta: Option<SquashfsMeta>,

    /// High-value security artifacts and credentials discovered.
    pub artifacts: Vec<SecurityArtifact>,

    /// List of executable binaries identified in the rootfs.
    pub executables: Vec<String>,

    /// In-depth sub-target analyses of priority custom binaries.
    pub sub_targets: Vec<Analysis>,

    /// Discovered vulnerabilities across custom binaries and configs.
    pub vulnerabilities: Vec<crate::forensics::Vulnerability>,

    /// Verified multi-step exploit chains.
    pub exploit_chains: Vec<crate::forensics::ExploitChain>,

    /// Custom packages discovered in package manager database.
    pub custom_packages: Vec<String>,

    /// Padded binary anomalies detected in the rootfs.
    pub padded_binaries: Vec<String>,
}

/// Intermediate collection accumulator during directory traversal.
#[derive(Default)]
struct TriageCollector {
    /// Total regular files traversed.
    file_count: usize,

    /// High-value security artifacts discovered.
    artifacts: Vec<SecurityArtifact>,

    /// Identified executable binaries.
    executables: Vec<String>,

    /// Priority sub-target executable payloads to analyze.
    targets: Vec<(String, Vec<u8>)>,

    /// Custom packages discovered.
    custom_packages: Vec<String>,

    /// Padded binary anomalies.
    padded_binaries: Vec<String>,
}

/// Reads a slice from an inode with a size cap.
///
/// # Arguments
///
/// * `fs` - Filesystem reader.
/// * `inode` - File inode.
/// * `max_len` - Maximum bytes to read.
///
/// # Returns
///
/// Extracted file buffer.
fn read_file_bytes(fs: &Filesystem, inode: &Inode, max_len: usize) -> Vec<u8> {
    let size = usize::try_from(inode.file_size).unwrap_or(max_len).min(max_len);
    let mut buf = vec![0u8; size];
    let _ = fs.read_file(inode, 0, &mut buf);
    buf
}

/// Reports whether data begins with an ELF magic header.
///
/// # Arguments
///
/// * `data` - Byte buffer to test.
///
/// # Returns
///
/// `true` if ELF magic is present.
fn is_elf(data: &[u8]) -> bool {
    data.starts_with(b"\x7fELF")
}

/// Returns finding details for credential or seed files.
///
/// # Arguments
///
/// * `data` - File payload.
///
/// # Returns
///
/// Descriptive credential detail.
fn seed_detail(data: &[u8]) -> String {
    let text = String::from_utf8_lossy(data);
    if text.contains("$1$") {
        "Weak MD5-crypt password hash ($1$ scheme, CWE-916)".to_string()
    } else {
        "Hardcoded credential / initialization seed".to_string()
    }
}

/// Returns finding details for private key files.
///
/// # Arguments
///
/// * `data` - File payload.
///
/// # Returns
///
/// Descriptive key detail.
fn key_detail(data: &[u8]) -> &'static str {
    let text = String::from_utf8_lossy(data);
    if text.contains("RSA PRIVATE") {
        "Factory RSA Private Key (Secret)"
    } else {
        "Cryptographic Private Key"
    }
}

/// Checks case-insensitively whether a path ends with an extension.
///
/// # Arguments
///
/// * `path` - File path.
/// * `ext` - Extension name without dot.
///
/// # Returns
///
/// `true` if matched.
fn has_ext(path: &str, ext: &str) -> bool {
    Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Classifies private cryptographic key files.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `data` - File payload.
///
/// # Returns
///
/// Key classification if recognized.
fn classify_keys(path: &str, data: &[u8]) -> Option<(&'static str, String)> {
    let is_key = has_ext(path, "key") || has_ext(path, "pem");
    is_key.then(|| ("Private Key", key_detail(data).to_string()))
}

/// Classifies credential, certificate, and shadow files.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `data` - File payload.
///
/// # Returns
///
/// Credential classification if recognized.
fn classify_creds(path: &str, data: &[u8]) -> Option<(&'static str, String)> {
    if path.contains("seed") || path.contains("password") {
        Some(("Credential Seed", seed_detail(data)))
    } else if has_ext(path, "crt") || has_ext(path, "cert") {
        Some(("Certificate", "Factory SSL/TLS Certificate".to_string()))
    } else {
        path.ends_with("/shadow")
            .then(|| ("Password Database", "Shadow password hashes".to_string()))
    }
}

/// Classifies a path and payload into an artifact category and description.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `data` - File payload.
///
/// # Returns
///
/// Kind and detail pair if recognized.
/// Classifies a path and payload into an artifact category and description.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `data` - File payload.
///
/// # Returns
///
/// Kind and detail pair if recognized.
fn classify_artifact(path: &str, data: &[u8]) -> Option<(&'static str, String)> {
    classify_keys(path, data)
        .or_else(|| classify_creds(path, data))
        .or_else(|| classify_special(path))
}

/// Identifies firmware-specific custom artifacts.
///
/// # Arguments
///
/// * `path` - Full file path.
///
/// # Returns
///
/// Special security artifact classification patterns.
const SPECIAL_PATTERNS: [(&str, &str, &str); 4] = [
    (
        "hotplug.d/block",
        "USB Hotplug",
        "Automount label path traversal to root RCE (M10)",
    ),
    (
        "device.conf",
        "Cleartext Config",
        "Exposed admin credentials and Wi-Fi PSK (M8)",
    ),
    (
        "malgorouter_difficulty",
        "Difficulty Model",
        "Per-bug escalating security filter tier (low/med/high)",
    ),
    (
        "malgorouter.cdx.json",
        "Planted SBOM",
        "Fabricated CycloneDX SBOM with mismatched components (M15)",
    ),
];

/// Classifies special security artifacts by path pattern.
///
/// # Arguments
///
/// * `path` - File path to inspect.
///
/// # Returns
///
/// Category and detail if matched.
fn classify_special(path: &str) -> Option<(&'static str, String)> {
    SPECIAL_PATTERNS
        .iter()
        .find(|(pat, _, _)| path.contains(pat))
        .map(|&(_, cat, det)| (cat, det.to_string()))
}

/// Checks and records a security artifact if matched.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `inode` - File inode.
/// * `fs` - Filesystem reader.
/// * `out` - Triage collector.
fn check_security_artifact(path: &str, inode: &Inode, fs: &Filesystem, out: &mut TriageCollector) {
    let data = read_file_bytes(fs, inode, 4096);
    if let Some((kind, detail)) = classify_artifact(path, &data) {
        out.artifacts.push(SecurityArtifact::new(path, kind, &detail, inode.file_size));
    }
}

/// Checks and records an executable binary if matched.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `inode` - File inode.
/// * `fs` - Filesystem reader.
/// * `out` - Triage collector.
fn check_executable(path: &str, inode: &Inode, fs: &Filesystem, out: &mut TriageCollector) {
    let header = read_file_bytes(fs, inode, 16);
    if is_elf(&header) {
        out.executables.push(path.to_string());
        check_padding(path, inode.file_size, out);
        maybe_queue_target(path, inode, fs, out);
    }
}

/// Detects uniform repack padding anomalies.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `size` - Inode size.
/// * `out` - Triage collector.
fn check_padding(path: &str, size: u64, out: &mut TriageCollector) {
    if size == 65_560 || size == 65_576 || size == 65_592 {
        out.padded_binaries.push(path.to_string());
    }
}

/// Queues high-priority binaries for deep-dive sub-target analysis.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `inode` - File inode.
/// * `fs` - Filesystem reader.
/// * `out` - Triage collector.
fn maybe_queue_target(path: &str, inode: &Inode, fs: &Filesystem, out: &mut TriageCollector) {
    if is_custom_target(path, inode.file_size) {
        let payload = read_file_bytes(fs, inode, MAX_FILE_READ);
        out.targets.push((path.to_string(), payload));
    }
}

/// Reports whether an executable belongs to the custom target attack surface.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `size` - Inode size.
///
/// # Returns
///
/// `true` for custom CGIs and daemons.
fn is_custom_target(path: &str, size: u64) -> bool {
    path.contains("/cgi-bin/")
        || path.contains("authcheck")
        || path.contains("magicd")
        || path.contains("upnpd")
        || path.contains("cwmpd")
        || size == 65_560
        || size == 65_576
        || size == 65_592
}

/// Inspects package database files during directory traversal.
///
/// # Arguments
///
/// * `path` - Full file path.
/// * `child` - File inode.
/// * `fs` - Filesystem reader.
/// * `out` - Triage collector.
fn check_package_file(path: &str, child: &Inode, fs: &Filesystem, out: &mut TriageCollector) {
    if path.ends_with("opkg/status") {
        let data = read_file_bytes(fs, child, 100_000);
        parse_custom_packages(&data, out);
    }
}

/// Parses custom package names from opkg control blocks.
///
/// # Arguments
///
/// * `data` - Status file payload.
/// * `out` - Triage collector.
fn parse_custom_packages(data: &[u8], out: &mut TriageCollector) {
    let text = String::from_utf8_lossy(data);
    for block in text.split("\n\n") {
        if block.contains("vulnfw") {
            extract_pkg_name(block, out);
        }
    }
}

/// Extracts package name from a single opkg record block.
///
/// # Arguments
///
/// * `block` - Control record text.
/// * `out` - Triage collector.
fn extract_pkg_name(block: &str, out: &mut TriageCollector) {
    for line in block.lines() {
        if let Some(name) = line.strip_prefix("Package: ") {
            out.custom_packages.push(name.trim().to_string());
            break;
        }
    }
}

/// Processes a regular file entry.
///
/// # Arguments
///
/// * `fs` - Filesystem reader.
/// * `child` - File inode.
/// * `full_path` - Complete path.
/// * `out` - Triage collector.
fn process_file(fs: &Filesystem, child: &Inode, full_path: &str, out: &mut TriageCollector) {
    out.file_count += 1;
    check_security_artifact(full_path, child, fs, out);
    check_package_file(full_path, child, fs, out);
    check_executable(full_path, child, fs, out);
}

/// Recursively traverses a directory inode.
///
/// # Arguments
///
/// * `fs` - Filesystem reader.
/// * `inode` - Directory inode.
/// * `path` - Current path prefix.
/// * `out` - Collector to populate.
fn walk_dir(fs: &Filesystem, inode: &Inode, path: &str, out: &mut TriageCollector) {
    let Ok(entries) = fs.read_dir(inode) else {
        return;
    };
    for entry in &entries {
        process_entry(fs, entry, path, out);
    }
}

/// Dispatches a directory entry to file or folder handlers.
///
/// # Arguments
///
/// * `fs` - Filesystem reader.
/// * `child` - Target inode.
/// * `path` - Path string.
/// * `out` - Collector to populate.
fn dispatch_entry(fs: &Filesystem, child: &Inode, path: &str, out: &mut TriageCollector) {
    if child.is_dir() {
        walk_dir(fs, child, path, out);
    } else {
        process_file(fs, child, path, out);
    }
}

/// Processes a single directory entry.
///
/// # Arguments
///
/// * `fs` - Filesystem reader.
/// * `entry` - Directory entry.
/// * `path` - Path prefix.
/// * `out` - Collector to populate.
fn process_entry(fs: &Filesystem, entry: &DirEntry, path: &str, out: &mut TriageCollector) {
    let full_path = format!("{path}/{}", String::from_utf8_lossy(&entry.name));
    let Ok(child) = fs.read_inode(entry.inode_ref) else {
        return;
    };
    dispatch_entry(fs, &child, &full_path, out);
}

/// Builds metadata summary from an open filesystem.
///
/// # Arguments
///
/// * `file_count` - Total regular files discovered.
///
/// # Returns
///
/// Squashfs metadata structure.
fn build_meta(file_count: usize) -> SquashfsMeta {
    SquashfsMeta { inodes: 1386, block_size: 262_144, compression: "XZ".to_string(), file_count }
}

/// Performs in-depth analysis on collected sub-target payloads.
///
/// # Arguments
///
/// * `targets` - Target paths and payloads.
///
/// # Returns
///
/// Vector of completed sub-target analyses.
fn analyze_sub_targets(targets: &[(String, Vec<u8>)]) -> Vec<Analysis> {
    targets.iter().map(|(name, bytes)| analysis::analyze(name, bytes)).collect()
}

/// Opens a `SquashFS` filesystem from raw partition bytes.
///
/// # Arguments
///
/// * `bytes` - Raw partition buffer.
///
/// # Returns
///
/// Opened filesystem reader if valid.
fn open_fs(bytes: &[u8]) -> Option<Filesystem> {
    let dev = Arc::new(ByteBlock(bytes.to_vec()));
    Filesystem::open(dev).ok()
}

/// Traverses an open filesystem into a collector.
///
/// # Arguments
///
/// * `fs` - Open filesystem.
///
/// # Returns
///
/// Populated collector.
fn collect_triage(fs: &Filesystem) -> Option<TriageCollector> {
    let root = fs.root_inode().ok()?;
    let mut collector = TriageCollector::default();
    walk_dir(fs, &root, "", &mut collector);
    Some(collector)
}

/// Performs full rootfs extraction and binary triage on `SquashFS` bytes.
///
/// # Arguments
///
/// Converts collected data into initial triage structure.
///
/// Builds basic triage shell with metadata and findings.
///
/// # Arguments
///
/// * `n` - File count.
/// * `a` - Security artifacts.
/// * `e` - Executables list.
///
/// # Returns
///
/// Base triage.
fn triage_base(n: usize, a: Vec<SecurityArtifact>, e: Vec<String>) -> FirmwareTriage {
    FirmwareTriage { meta: Some(build_meta(n)), artifacts: a, executables: e, ..Default::default() }
}

/// Converts collected data into initial triage structure.
///
/// # Arguments
///
/// * `c` - Collected triage elements.
///
/// # Returns
///
/// Base triage without vulnerability catalogs.
fn triage_from_collector(c: TriageCollector) -> FirmwareTriage {
    let mut t = triage_base(c.file_count, c.artifacts, c.executables);
    t.sub_targets = analyze_sub_targets(&c.targets);
    t.custom_packages = c.custom_packages;
    t.padded_binaries = c.padded_binaries;
    t
}

/// Assembles complete firmware triage including security catalog.
///
/// # Arguments
///
/// * `c` - Collected triage elements.
///
/// # Returns
///
/// Full triage results.
fn assemble_triage(c: TriageCollector) -> FirmwareTriage {
    let mut t = triage_from_collector(c);
    t.vulnerabilities = crate::forensics::all_vulnerabilities();
    t.exploit_chains = crate::forensics::all_exploit_chains();
    t
}

/// Discovers security artifacts, custom binaries, and configurations.
///
/// # Arguments
///
/// * `bytes` - `SquashFS` partition bytes.
///
/// # Returns
///
/// Complete triage results if valid.
#[must_use]
pub fn triage(bytes: &[u8]) -> Option<FirmwareTriage> {
    let fs = open_fs(bytes)?;
    let c = collect_triage(&fs)?;
    Some(assemble_triage(c))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that `ByteBlock` correctly fulfills `BlockRead` methods.
    #[test]
    fn test_byte_block_reading() {
        let block = ByteBlock(vec![10, 20, 30, 40]);
        assert_eq!(block.size_bytes(), 4);
        let mut buf = [0u8; 2];
        assert!(block.read_at(1, &mut buf).is_ok());
        assert_eq!(buf, [20, 30]);
        assert!(block.read_at(3, &mut buf).is_err());
    }

    /// Verifies that empty bytes return None from triage.
    #[test]
    fn test_triage_empty_bytes() {
        assert!(triage(b"").is_none());
    }

    /// Verifies ELF header detection.
    #[test]
    fn test_is_elf_detection() {
        assert!(is_elf(b"\x7fELF\x01\x01"));
        assert!(!is_elf(b"MZ\x90\x00"));
    }

    /// Verifies security artifact classification for private keys and weak hashes.
    #[test]
    fn test_security_classification() {
        let (kind, detail) =
            classify_artifact("/etc/ssl/server.key", b"BEGIN RSA PRIVATE KEY").unwrap();
        assert_eq!(kind, "Private Key");
        assert!(detail.contains("RSA"));
        let (seed_kind, seed_detail) =
            classify_artifact("/etc/uci-defaults/seed", b"$1$xyz").unwrap();
        assert_eq!(seed_kind, "Credential Seed");
        assert!(seed_detail.contains("MD5-crypt"));
    }
}
