//! Flutter Application and APK Native Library Analysis
//!
//! **File:** `flutter.rs`
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

/// Zip local file header signature.
const LOCAL_MAGIC: &[u8; 4] = b"PK\x03\x04";

/// Zip central directory entry signature.
const CENTRAL_MAGIC: &[u8; 4] = b"PK\x01\x02";

/// Zip end-of-central-directory signature.
const EOCD_MAGIC: &[u8; 4] = b"PK\x05\x06";

/// A parsed zip central directory entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ZipEntry {
    /// Entry path.
    pub name: String,
    /// Compression method.
    pub method: u16,
    /// CRC-32 of the uncompressed data.
    pub crc: u32,
    /// Compressed size in bytes.
    pub comp_size: u32,
    /// Uncompressed size in bytes.
    pub uncomp_size: u32,
}

/// Flutter native libraries for one ABI.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FlutterAbi {
    /// ABI directory name.
    pub abi: String,
    /// The Dart AOT snapshot library.
    pub libapp: Option<ZipEntry>,
    /// The Flutter engine library.
    pub libflutter: Option<ZipEntry>,
}

/// Recovered Flutter application inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FlutterApk {
    /// Native libraries grouped by ABI.
    pub abis: Vec<FlutterAbi>,
    /// Bundled Flutter asset paths.
    pub assets: Vec<String>,
    /// Whether Flutter markers were present.
    pub is_flutter: bool,
}

/// Reports whether a buffer is a zip archive.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the local header magic matches.
#[must_use]
pub fn is_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(LOCAL_MAGIC)
}

/// Reads a little-endian 16-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, or `None`.
fn read_u16_le(bytes: &[u8], offset: usize) -> Option<u16> {
    let b = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

/// Reads a little-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, or `None`.
fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Locates the end-of-central-directory record.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The record offset, or `None`.
fn find_eocd(bytes: &[u8]) -> Option<usize> {
    let start = bytes.len().saturating_sub(65_557);
    let tail = bytes.get(start..)?;
    let at = tail.windows(4).rposition(|w| w == EOCD_MAGIC)?;
    Some(start + at)
}

/// Reads the name of a directory entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Name offset.
/// * `len` - Name length.
///
/// # Returns
///
/// The decoded name.
fn entry_name(bytes: &[u8], pos: usize, len: usize) -> String {
    let raw = bytes.get(pos..pos + len).unwrap_or_default();
    String::from_utf8_lossy(raw).to_string()
}

/// Builds a directory entry from its central header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Central header offset.
/// * `name_len` - Name length.
///
/// # Returns
///
/// The parsed entry.
fn build_entry(bytes: &[u8], pos: usize, name_len: usize) -> ZipEntry {
    ZipEntry {
        name: entry_name(bytes, pos + 46, name_len),
        method: read_u16_le(bytes, pos + 10).unwrap_or(0),
        crc: read_u32_le(bytes, pos + 16).unwrap_or(0),
        comp_size: read_u32_le(bytes, pos + 20).unwrap_or(0),
        uncomp_size: read_u32_le(bytes, pos + 24).unwrap_or(0),
    }
}

/// Reads one central directory entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Entry offset.
///
/// # Returns
///
/// The entry and the next offset, or `None`.
fn read_entry(bytes: &[u8], pos: usize) -> Option<(ZipEntry, usize)> {
    if bytes.get(pos..pos + 4)? != CENTRAL_MAGIC {
        return None;
    }
    let name_len = usize::from(read_u16_le(bytes, pos + 28).unwrap_or(0));
    let extra_len = usize::from(read_u16_le(bytes, pos + 30).unwrap_or(0));
    let comment_len = usize::from(read_u16_le(bytes, pos + 32).unwrap_or(0));
    let entry = build_entry(bytes, pos, name_len);
    Some((entry, pos + 46 + name_len + extra_len + comment_len))
}

/// Collects central directory entries.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - Central directory offset.
/// * `count` - Declared entry count.
///
/// # Returns
///
/// Every parsed entry.
fn collect_entries(bytes: &[u8], start: usize, count: usize) -> Vec<ZipEntry> {
    let mut entries = Vec::new();
    let mut pos = start;
    while entries.len() < count && step_entry(bytes, &mut pos, &mut entries) {}
    entries
}

/// Reads and stores one directory entry at a cursor.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Cursor advanced past the entry.
/// * `entries` - Accumulated entries.
///
/// # Returns
///
/// `true` when an entry was read.
fn step_entry(bytes: &[u8], pos: &mut usize, entries: &mut Vec<ZipEntry>) -> bool {
    if let Some((entry, next)) = read_entry(bytes, *pos) {
        entries.push(entry);
        *pos = next;
        return true;
    }
    false
}

/// Parses the central directory of a zip archive.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Every entry in the archive.
fn parse_central(bytes: &[u8]) -> Vec<ZipEntry> {
    let Some(eocd) = find_eocd(bytes) else {
        return Vec::new();
    };
    let count = usize::from(read_u16_le(bytes, eocd + 10).unwrap_or(0));
    let start = usize::try_from(read_u32_le(bytes, eocd + 16).unwrap_or(0)).unwrap_or(0);
    collect_entries(bytes, start, count)
}

/// Resolves the ABI directory of a native library path.
///
/// # Arguments
///
/// * `name` - Entry path.
///
/// # Returns
///
/// The ABI name, or `unknown`.
fn abi_of(name: &str) -> String {
    let parts: Vec<&str> = name.split('/').collect();
    parts.get(1).copied().unwrap_or("unknown").to_string()
}

/// Ensures an ABI group exists.
///
/// # Arguments
///
/// * `abis` - ABI groups.
/// * `abi` - ABI name.
fn ensure_abi(abis: &mut Vec<FlutterAbi>, abi: &str) {
    if !abis.iter().any(|a| a.abi == abi) {
        abis.push(FlutterAbi { abi: abi.to_string(), ..Default::default() });
    }
}

/// Stores a native library in its ABI group.
///
/// # Arguments
///
/// * `target` - ABI group.
/// * `entry` - Library entry.
fn set_library(target: &mut FlutterAbi, entry: &ZipEntry) {
    if entry.name.ends_with("libapp.so") {
        target.libapp = Some(entry.clone());
    } else {
        target.libflutter = Some(entry.clone());
    }
}

/// Adds a native library to the inventory.
///
/// # Arguments
///
/// * `entry` - Library entry.
/// * `apk` - Output inventory.
fn add_library(entry: &ZipEntry, apk: &mut FlutterApk) {
    let abi = abi_of(&entry.name);
    ensure_abi(&mut apk.abis, &abi);
    if let Some(target) = apk.abis.iter_mut().find(|a| a.abi == abi) {
        set_library(target, entry);
    }
}

/// Classifies one archive entry.
///
/// # Arguments
///
/// * `entry` - Directory entry.
/// * `apk` - Output inventory.
fn classify_entry(entry: &ZipEntry, apk: &mut FlutterApk) {
    if entry.name.starts_with("assets/flutter_assets/") {
        apk.assets.push(entry.name.clone());
    } else if entry.name.ends_with("libapp.so") || entry.name.ends_with("libflutter.so") {
        add_library(entry, apk);
    }
}

/// Parses a Flutter APK or zip archive.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered inventory, or `None` when not a zip.
#[must_use]
pub fn parse_flutter(bytes: &[u8]) -> Option<FlutterApk> {
    if !is_zip(bytes) {
        return None;
    }
    Some(build_flutter(&parse_central(bytes)))
}

/// Builds a Flutter inventory from parsed entries.
///
/// # Arguments
///
/// * `entries` - Archive directory entries.
///
/// # Returns
///
/// The recovered inventory.
fn build_flutter(entries: &[ZipEntry]) -> FlutterApk {
    let mut apk = FlutterApk::default();
    for entry in entries {
        classify_entry(entry, &mut apk);
    }
    apk.is_flutter = !apk.abis.is_empty();
    apk
}

/// Formats a Flutter inventory as a human-readable text block.
///
/// # Arguments
///
/// * `apk` - Recovered inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_flutter_summary(apk: &FlutterApk) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Flutter application: {}", apk.is_flutter);
    let _ = writeln!(out, "ABIs: {}", apk.abis.len());
    let _ = writeln!(out, "Assets: {}", apk.assets.len());
    for abi in &apk.abis {
        let _ = writeln!(out, "  {} {}", abi.abi, describe_libs(abi));
    }
    out
}

/// Describes the libraries of one ABI group.
///
/// # Arguments
///
/// * `abi` - ABI group.
///
/// # Returns
///
/// A description of the app and engine libraries.
fn describe_libs(abi: &FlutterAbi) -> String {
    let app = abi
        .libapp
        .as_ref()
        .map_or("none".to_string(), |e| format!("libapp crc {:08x}", e.crc));
    let engine = abi.libflutter.as_ref().map_or("none".to_string(), |e| {
        format!("libflutter crc {:08x}", e.crc)
    });
    format!("{app}; {engine}")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a central directory entry.
    ///
    /// # Arguments
    ///
    /// * `name` - Entry path.
    /// * `crc` - CRC-32 value.
    ///
    /// # Returns
    ///
    /// The encoded central header.
    fn central(name: &str, crc: u32) -> Vec<u8> {
        let mut data = vec![0u8; 46];
        data[..4].copy_from_slice(CENTRAL_MAGIC);
        data[16..20].copy_from_slice(&crc.to_le_bytes());
        data[28..30].copy_from_slice(&u16::try_from(name.len()).unwrap().to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data
    }

    /// Builds a zip with the given central entries.
    ///
    /// # Arguments
    ///
    /// * `entries` - Encoded central entries.
    ///
    /// # Returns
    ///
    /// The complete archive bytes.
    fn zip(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut data = LOCAL_MAGIC.to_vec();
        let offset = u32::try_from(data.len()).unwrap();
        let central_dir: Vec<u8> = entries.iter().flatten().copied().collect();
        data.extend_from_slice(&central_dir);
        let count = u16::try_from(entries.len()).unwrap();
        append_eocd(&mut data, &central_dir, offset, count);
        data
    }

    /// Appends an end-of-central-directory record.
    ///
    /// # Arguments
    ///
    /// * `data` - Archive buffer.
    /// * `central` - Central directory bytes.
    /// * `offset` - Central directory offset.
    /// * `count` - Entry count.
    fn append_eocd(data: &mut Vec<u8>, central: &[u8], offset: u32, count: u16) {
        data.extend_from_slice(EOCD_MAGIC);
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&count.to_le_bytes());
        data.extend_from_slice(&count.to_le_bytes());
        data.extend_from_slice(&u32::try_from(central.len()).unwrap().to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&[0, 0]);
    }

    /// Verifies Flutter library detection across ABIs.
    #[test]
    fn test_flutter_abis() {
        let apk = parse_flutter(&flutter_zip()).unwrap();
        assert!(apk.is_flutter);
        assert_eq!(apk.abis.len(), 1);
        assert_eq!(apk.abis[0].abi, "arm64-v8a");
        assert_eq!(apk.abis[0].libapp.as_ref().unwrap().crc, 0x1122_3344);
        assert_eq!(apk.assets.len(), 1);
    }

    /// Builds a Flutter archive with one ABI.
    ///
    /// # Returns
    ///
    /// The archive bytes.
    fn flutter_zip() -> Vec<u8> {
        zip(&[
            central("lib/arm64-v8a/libapp.so", 0x1122_3344),
            central("lib/arm64-v8a/libflutter.so", 0x5566_7788),
            central("assets/flutter_assets/AssetManifest.json", 1),
        ])
    }

    /// Verifies ABI path parsing.
    #[test]
    fn test_flutter_abi_of() {
        assert_eq!(abi_of("lib/x86_64/libapp.so"), "x86_64");
        assert_eq!(abi_of("classes.dex"), "unknown");
    }

    /// Verifies zip detection.
    #[test]
    fn test_flutter_zip_detection() {
        assert!(is_zip(LOCAL_MAGIC));
        assert!(!is_zip(b"MZ"));
    }
}
