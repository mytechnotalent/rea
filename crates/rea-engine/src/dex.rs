//! Android DEX and APK Binary Forensics
//!
//! **File:** `dex.rs`
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
use std::fmt::Write as _;

/// Magic prefix for Android DEX files: `dex\n`.
pub const DEX_MAGIC_PREFIX: [u8; 4] = [0x64, 0x65, 0x78, 0x0a];

/// Magic signature for Android Binary XML: `0x00080003`.
pub const AXML_MAGIC: [u8; 4] = [0x03, 0x00, 0x08, 0x00];

/// ZIP Local File Header magic: `PK\x03\x04`.
pub const ZIP_LOCAL_MAGIC: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

/// Extracted Android DEX bytecode metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DexMetadata {
    /// DEX version string, e.g. "035" or "039".
    pub version: String,
    /// Total declared classes count.
    pub class_count: usize,
    /// Total declared methods count.
    pub method_count: usize,
    /// Total string identifiers count.
    pub string_count: usize,
    /// Recovered class names list.
    pub classes: Vec<String>,
}

/// Extracted Android APK manifest and package metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ApkMetadata {
    /// Android application package name.
    pub package_name: String,
    /// Requested permissions list.
    pub permissions: Vec<String>,
    /// Exported activities, services, receivers.
    pub components: Vec<String>,
    /// Number of DEX files in archive.
    pub dex_count: usize,
    /// Primary classes.dex metadata.
    pub dex: Option<DexMetadata>,
}

/// Reads a 32-bit unsigned little-endian integer.
///
/// # Arguments
///
/// * `b` - Byte slice with at least 4 bytes.
///
/// # Returns
///
/// 32-bit unsigned integer.
fn read_u32_le(b: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let take = b.len().min(4);
    buf[..take].copy_from_slice(&b[..take]);
    u32::from_le_bytes(buf)
}

/// Parses a DEX header and extracts class and method counts.
///
/// # Arguments
///
/// * `bytes` - DEX file bytes.
///
/// # Returns
///
/// Extracted DEX metadata if header is valid.
/// Initializes basic DEX header counts from bytes.
///
/// # Arguments
///
/// * `bytes` - DEX file bytes.
///
/// # Returns
///
/// Basic initialized DEX metadata structure.
fn init_dex_meta(bytes: &[u8]) -> DexMetadata {
    let version = String::from_utf8_lossy(&bytes[4..7]).to_string();
    let string_count = read_u32_le(&bytes[56..60]) as usize;
    let method_count = read_u32_le(&bytes[88..92]) as usize;
    let class_count = read_u32_le(&bytes[96..100]) as usize;
    DexMetadata { version, class_count, method_count, string_count, classes: Vec::new() }
}

/// Parses DEX file header information.
///
/// # Arguments
///
/// * `bytes` - DEX file bytes.
///
/// # Returns
///
/// Extracted DEX metadata if header is valid.
#[must_use]
pub fn parse_dex_header(bytes: &[u8]) -> Option<DexMetadata> {
    if bytes.len() < 112 || bytes[..4] != DEX_MAGIC_PREFIX {
        return None;
    }
    let mut dex = init_dex_meta(bytes);
    extract_dex_classes(bytes, &mut dex);
    Some(dex)
}

/// Extracts class type descriptors from DEX string items.
///
/// # Arguments
///
/// * `bytes` - DEX file buffer.
/// * `dex` - Output DEX metadata.
fn extract_dex_classes(bytes: &[u8], dex: &mut DexMetadata) {
    for chunk in bytes.split(|&b| b == 0) {
        if is_class_descriptor(chunk) && dex.classes.len() < 30 {
            dex.classes.push(format_descriptor(chunk));
        }
    }
}

/// Formats a class descriptor chunk into a clean Java class path.
///
/// # Arguments
///
/// * `chunk` - Descriptor byte chunk.
///
/// # Returns
///
/// Formatted class name string.
fn format_descriptor(chunk: &[u8]) -> String {
    let s = String::from_utf8_lossy(chunk);
    let trimmed = s.trim_start_matches('L').trim_end_matches(';');
    trimmed.replace('/', ".")
}

/// Reports whether a byte chunk is an Android class type descriptor.
///
/// # Arguments
///
/// * `chunk` - Byte chunk to check.
///
/// # Returns
///
/// `true` if descriptor starts with 'L' and ends with ';'.
fn is_class_descriptor(chunk: &[u8]) -> bool {
    chunk.len() >= 4 && chunk.starts_with(b"L") && chunk.ends_with(b";") && chunk.contains(&b'/')
}

/// Extracts manifest permissions and package from binary XML strings.
///
/// # Arguments
///
/// * `bytes` - Android binary XML buffer.
/// * `apk` - Output APK metadata.
fn extract_manifest_strings(bytes: &[u8], apk: &mut ApkMetadata) {
    for chunk in bytes.split(|&b| b == 0) {
        process_manifest_token(&String::from_utf8_lossy(chunk), apk);
    }
}

/// Categorizes and records a manifest token into APK metadata.
///
/// # Arguments
///
/// * `s` - Manifest token string.
/// * `apk` - Output APK metadata.
fn process_manifest_token(s: &str, apk: &mut ApkMetadata) {
    if s.starts_with("android.permission.") && !apk.permissions.contains(&s.to_string()) {
        apk.permissions.push(s.to_string());
    } else if is_package_candidate(s) && apk.package_name.is_empty() {
        apk.package_name = s.to_string();
    } else if is_component_candidate(s) && apk.components.len() < 25 {
        apk.components.push(s.to_string());
    }
}

/// Reports whether a string looks like an Android package identifier.
///
/// # Arguments
///
/// * `s` - Candidate string.
///
/// # Returns
///
/// `true` for reverse-domain package names.
fn is_package_candidate(s: &str) -> bool {
    s.starts_with("com.")
        || s.starts_with("org.")
        || s.starts_with("io.") && s.contains('.') && s.len() <= 60
}

/// Reports whether a string looks like an activity or service component.
///
/// # Arguments
///
/// * `s` - Candidate string.
///
/// # Returns
///
/// `true` for Activity or Service names.
fn is_component_candidate(s: &str) -> bool {
    (s.ends_with("Activity") || s.ends_with("Service") || s.ends_with("Receiver"))
        && !s.contains(' ')
        && s.len() <= 70
}

/// Finds the offset of a byte needle in a buffer.
///
/// # Arguments
///
/// * `haystack` - Buffer to search.
/// * `needle` - Byte sequence to locate.
///
/// # Returns
///
/// Offset index if located.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Parses raw standalone DEX binary into APK metadata.
///
/// # Arguments
///
/// * `bytes` - DEX file bytes.
///
/// # Returns
///
/// Extracted APK metadata structure.
fn parse_raw_dex_pkg(bytes: &[u8]) -> Option<ApkMetadata> {
    let dex = parse_dex_header(bytes)?;
    Some(ApkMetadata { dex_count: 1, dex: Some(dex), ..Default::default() })
}

/// Scans ZIP-formatted APK archive.
///
/// # Arguments
///
/// * `bytes` - ZIP file bytes.
///
/// # Returns
///
/// Extracted APK metadata if recognized.
fn scan_zip_apk(bytes: &[u8]) -> Option<ApkMetadata> {
    if !bytes.starts_with(&ZIP_LOCAL_MAGIC) {
        return None;
    }
    let mut apk = ApkMetadata::default();
    scan_apk_contents(bytes, &mut apk);
    (apk.dex_count > 0 || !apk.package_name.is_empty()).then_some(apk)
}

/// Parses an Android APK archive from bytes.
///
/// # Arguments
///
/// * `bytes` - APK archive buffer.
///
/// # Returns
///
/// Extracted APK metadata if APK or DEX signatures exist.
#[must_use]
pub fn parse_apk_metadata(bytes: &[u8]) -> Option<ApkMetadata> {
    if bytes.starts_with(&DEX_MAGIC_PREFIX) {
        parse_raw_dex_pkg(bytes)
    } else {
        scan_zip_apk(bytes)
    }
}

/// Scans APK archive contents for DEX and manifest signatures.
///
/// # Arguments
///
/// * `bytes` - APK file bytes.
/// * `apk` - Output APK metadata.
fn scan_apk_contents(bytes: &[u8], apk: &mut ApkMetadata) {
    if let Some(dex_off) = find_bytes(bytes, &DEX_MAGIC_PREFIX) {
        apk.dex_count += 1;
        apk.dex = parse_dex_header(&bytes[dex_off..]);
    }
    if let Some(axml_off) = find_bytes(bytes, &AXML_MAGIC) {
        extract_manifest_strings(&bytes[axml_off..], apk);
    }
}

/// Resolves display label for APK package name.
///
/// # Arguments
///
/// * `apk` - Extracted APK metadata.
///
/// # Returns
///
/// Package string or "unknown".
fn apk_pkg_label(apk: &ApkMetadata) -> &str {
    if apk.package_name.is_empty() {
        "unknown"
    } else {
        &apk.package_name
    }
}

/// Formats package identity and counts for APK summary.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `apk` - Extracted APK metadata.
fn append_apk_header(out: &mut String, apk: &ApkMetadata) {
    let pkg = apk_pkg_label(apk);
    let (dex, perms, comps) = (apk.dex_count, apk.permissions.len(), apk.components.len());
    let _ = writeln!(
        out,
        "Package: {pkg}\nDEX Files: {dex}\nPermissions: {perms}\nComponents: {comps}"
    );
}

/// Appends DEX class and method count line.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `dex` - Extracted DEX metadata.
fn append_dex_line(out: &mut String, dex: &DexMetadata) {
    let _ = writeln!(
        out,
        "Classes: {} | Methods: {}",
        dex.class_count, dex.method_count
    );
}

/// Formats Android APK and DEX metadata as a human-readable summary.
///
/// # Arguments
///
/// * `apk` - Extracted APK metadata.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_apk_summary(apk: &ApkMetadata) -> String {
    let mut out = String::new();
    append_apk_header(&mut out, apk);
    if let Some(dex) = &apk.dex {
        append_dex_line(&mut out, dex);
    }
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Constructs mock DEX header bytes.
    fn make_test_dex() -> Vec<u8> {
        let mut d = vec![0u8; 112];
        d[..4].copy_from_slice(&DEX_MAGIC_PREFIX);
        d[4..7].copy_from_slice(b"035");
        d[56..60].copy_from_slice(&250u32.to_le_bytes());
        d[88..92].copy_from_slice(&120u32.to_le_bytes());
        d[96..100].copy_from_slice(&45u32.to_le_bytes());
        d
    }

    /// Verifies DEX header parsing.
    #[test]
    fn test_dex_header_parsing() {
        let data = make_test_dex();
        let dex = parse_dex_header(&data).expect("valid dex");
        assert_eq!(dex.version, "035");
        assert_eq!(dex.string_count, 250);
        assert_eq!(dex.method_count, 120);
        assert_eq!(dex.class_count, 45);
    }

    /// Verifies manifest string extraction.
    #[test]
    fn test_manifest_string_extraction() {
        let mut data = Vec::new();
        data.extend_from_slice(&AXML_MAGIC);
        data.extend_from_slice(b"com.example.app\0android.permission.INTERNET\0MainActivity\0");
        let mut apk = ApkMetadata::default();
        extract_manifest_strings(&data, &mut apk);
        assert_eq!(apk.package_name, "com.example.app");
        assert_eq!(apk.permissions, vec!["android.permission.INTERNET"]);
        assert_eq!(apk.components, vec!["MainActivity"]);
    }

    /// Verifies APK summary formatting.
    #[test]
    fn test_apk_summary_formatting() {
        let apk =
            ApkMetadata { package_name: "com.test".into(), dex_count: 1, ..Default::default() };
        let s = format_apk_summary(&apk);
        assert!(s.contains("com.test") && s.contains("DEX Files: 1"));
    }
}
