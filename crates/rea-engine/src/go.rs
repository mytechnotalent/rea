//! Go Runtime and Build Metadata Recovery
//!
//! **File:** `go.rs`
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

/// Magic pattern identifying Go buildinfo header: `\xff Go buildinf:`.
pub const GO_BUILDINF_MAGIC: &[u8; 14] = b"\xff Go buildinf:";

/// Magic pattern identifying Go 1.18+ pclntab headers.
pub const PCLNTAB_MAGIC_118: [u8; 4] = [0xf0, 0xff, 0xff, 0xff];

/// Magic pattern identifying Go 1.16 pclntab headers.
pub const PCLNTAB_MAGIC_116: [u8; 4] = [0xfa, 0xff, 0xff, 0xff];

/// Magic pattern identifying Go 1.2 pclntab headers.
pub const PCLNTAB_MAGIC_120: [u8; 4] = [0xfb, 0xff, 0xff, 0xff];

/// Recovered Go dependency module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GoModule {
    /// Module import path.
    pub path: String,
    /// Semantic version.
    pub version: String,
    /// Hash or checksum.
    pub checksum: String,
}

/// Extracted Go runtime, build settings, and function inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GoMetadata {
    /// Toolchain version string.
    pub version: String,
    /// Main module package path.
    pub module_path: String,
    /// Dependencies list.
    pub dependencies: Vec<GoModule>,
    /// Key-value compiler and runtime build settings.
    pub build_settings: Vec<(String, String)>,
    /// Recovered function symbol names.
    pub functions: Vec<String>,
    /// Total recovered function count.
    pub func_count: usize,
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

/// Extracts a null-terminated ASCII string from a slice.
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
///
/// # Returns
///
/// Validated UTF-8 string up to null terminator.
fn read_cstring(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

/// Parses module lines from Go modinfo text.
///
/// # Arguments
///
/// * `meta` - Output Go metadata to populate.
/// * `line` - Single modinfo line.
fn parse_modinfo_line(meta: &mut GoMetadata, line: &str) {
    let parts: Vec<&str> = line.split('\t').collect();
    match parts.first().copied() {
        Some("path") if parts.len() > 1 => meta.module_path = parts[1].to_string(),
        Some("mod" | "dep") if parts.len() >= 3 => append_mod(meta, &parts),
        Some("build") if parts.len() > 1 => append_build(meta, parts[1]),
        _ => {}
    }
}

/// Appends a module dependency to Go metadata.
///
/// # Arguments
///
/// * `meta` - Target metadata structure.
/// * `parts` - Split modinfo line fields.
fn append_mod(meta: &mut GoMetadata, parts: &[&str]) {
    let path = (*parts.get(1).unwrap_or(&"")).to_string();
    let version = (*parts.get(2).unwrap_or(&"")).to_string();
    let checksum = (*parts.get(3).unwrap_or(&"")).to_string();
    meta.dependencies.push(GoModule { path, version, checksum });
}

/// Appends a build setting to Go metadata.
///
/// # Arguments
///
/// * `meta` - Target metadata structure.
/// * `entry` - Key=value string.
fn append_build(meta: &mut GoMetadata, entry: &str) {
    if let Some((k, v)) = entry.split_once('=') {
        meta.build_settings.push((k.to_string(), v.to_string()));
    }
}

/// Decodes the Go modinfo text payload.
///
/// # Arguments
///
/// * `meta` - Target metadata structure.
/// * `payload` - Text content slice.
fn parse_modinfo_text(meta: &mut GoMetadata, payload: &str) {
    for line in payload.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            parse_modinfo_line(meta, trimmed);
        }
    }
}

/// Extracts module info text following buildinfo header.
///
/// # Arguments
///
/// * `bytes` - Binary byte slice.
/// * `start` - Offset after header.
/// * `meta` - Output Go metadata.
fn extract_modinfo_slice(bytes: &[u8], start: usize, meta: &mut GoMetadata) {
    if let Some(mod_off) = find_bytes(&bytes[start..], b"path\t") {
        let text_slice = &bytes[start + mod_off..];
        let end = text_slice.iter().position(|&b| b == 0).unwrap_or(text_slice.len());
        parse_modinfo_text(meta, &String::from_utf8_lossy(&text_slice[..end]));
    }
}

/// Extracts Go build info and toolchain version from byte buffer.
///
/// # Arguments
///
/// * `bytes` - Binary byte slice.
/// * `meta` - Output Go metadata.
fn extract_buildinfo(bytes: &[u8], meta: &mut GoMetadata) {
    let Some(off) = find_bytes(bytes, GO_BUILDINF_MAGIC) else {
        return;
    };
    let start = off + GO_BUILDINF_MAGIC.len() + 2;
    if start < bytes.len() {
        extract_modinfo_slice(bytes, start, meta);
    }
}

/// Inspects pclntab header and recovers function table count.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `meta` - Output Go metadata.
fn extract_pclntab(bytes: &[u8], meta: &mut GoMetadata) {
    let magics = [PCLNTAB_MAGIC_118, PCLNTAB_MAGIC_116, PCLNTAB_MAGIC_120];
    for m in magics {
        if let Some(off) = find_bytes(bytes, &m) {
            parse_pclntab_at(bytes, off, meta);
            break;
        }
    }
}

/// Reads pclntab function count field.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `off` - Offset of pclntab header.
/// * `ptr_sz` - Pointer size in bytes.
///
/// # Returns
///
/// Recovered function count clamped to 500,000.
fn read_pclntab_nfunc(bytes: &[u8], off: usize, ptr_sz: usize) -> usize {
    let nfunc_bytes = &bytes[off + 8..off + 8 + ptr_sz.min(8)];
    read_uint_le(nfunc_bytes).min(500_000) as usize
}

/// Parses pclntab header fields at a located offset.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `off` - Offset of pclntab header.
/// * `meta` - Output Go metadata.
fn parse_pclntab_at(bytes: &[u8], off: usize, meta: &mut GoMetadata) {
    if off + 16 > bytes.len() {
        return;
    }
    let ptr_sz = bytes[off + 7] as usize;
    if ptr_sz == 4 || ptr_sz == 8 {
        meta.func_count = read_pclntab_nfunc(bytes, off, ptr_sz);
        extract_known_func_names(bytes, off, meta);
    }
}

/// Reads a little-endian unsigned integer from bytes.
///
/// # Arguments
///
/// * `b` - Byte slice up to 8 bytes.
///
/// # Returns
///
/// Unsigned 64-bit integer.
fn read_uint_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let take = b.len().min(8);
    buf[..take].copy_from_slice(&b[..take]);
    u64::from_le_bytes(buf)
}

/// Scans function name candidates within pclntab string pool.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `off` - Offset of pclntab.
/// * `meta` - Output Go metadata.
fn extract_known_func_names(bytes: &[u8], off: usize, meta: &mut GoMetadata) {
    let pool_start = (off + 64).min(bytes.len());
    let pool_end = (off + 65536).min(bytes.len());
    let slice = &bytes[pool_start..pool_end];
    for chunk in slice.split(|&b| b == 0) {
        if is_go_func_name(chunk) && meta.functions.len() < 50 {
            meta.functions.push(String::from_utf8_lossy(chunk).to_string());
        }
    }
}

/// Reports whether a byte chunk matches a Go function symbol pattern.
///
/// # Arguments
///
/// * `chunk` - Byte chunk to check.
///
/// # Returns
///
/// `true` if chunk appears to be a package-qualified function name.
fn is_go_func_name(chunk: &[u8]) -> bool {
    if chunk.len() < 4 || chunk.len() > 120 {
        return false;
    }
    let text = String::from_utf8_lossy(chunk);
    text.contains('.') && !text.contains(' ') && text.chars().all(|c| c.is_ascii_graphic())
}

/// Detects toolchain version substring in binary text.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
///
/// # Returns
///
/// Recovered toolchain version string if found.
fn find_go_version(bytes: &[u8]) -> String {
    let needle = b"go1.";
    let Some(pos) = find_bytes(bytes, needle) else {
        return String::new();
    };
    read_cstring(&bytes[pos..pos + 32.min(bytes.len() - pos)])
}

/// Parses Go metadata from a binary image buffer.
///
/// Reports whether buffer contains Go headers.
///
/// # Arguments
///
/// * `bytes` - Binary file buffer.
///
/// # Returns
///
/// `true` if buildinfo or pclntab header is detected.
fn has_go_signatures(bytes: &[u8]) -> bool {
    let has_build = find_bytes(bytes, GO_BUILDINF_MAGIC).is_some();
    let has_pcln = find_bytes(bytes, &PCLNTAB_MAGIC_118).is_some()
        || find_bytes(bytes, &PCLNTAB_MAGIC_116).is_some()
        || find_bytes(bytes, &PCLNTAB_MAGIC_120).is_some();
    has_build || has_pcln
}

/// Parses Go metadata from a binary image buffer.
///
/// # Arguments
///
/// * `bytes` - Binary file buffer.
///
/// # Returns
///
/// Extracted Go metadata if Go signatures are present.
#[must_use]
pub fn parse_go_metadata(bytes: &[u8]) -> Option<GoMetadata> {
    if !has_go_signatures(bytes) {
        return None;
    }
    let mut meta = GoMetadata { version: find_go_version(bytes), ..Default::default() };
    extract_buildinfo(bytes, &mut meta);
    extract_pclntab(bytes, &mut meta);
    Some(meta)
}

/// Resolves display label for Go runtime version.
///
/// # Arguments
///
/// * `m` - Extracted Go metadata.
///
/// # Returns
///
/// Version string or "unknown".
fn go_runtime_label(m: &GoMetadata) -> &str {
    if m.version.is_empty() {
        "unknown"
    } else {
        &m.version
    }
}

/// Appends Go version and module lines.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `m` - Extracted Go metadata.
fn append_go_id(out: &mut String, m: &GoMetadata) {
    let ver = go_runtime_label(m);
    let path = if m.module_path.is_empty() {
        "none"
    } else {
        &m.module_path
    };
    let _ = writeln!(out, "Go Runtime: {ver}\nModule Path: {path}");
}

/// Formats Go metadata summary as a human-readable text block.
///
/// # Arguments
///
/// * `m` - Extracted Go metadata.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_go_summary(m: &GoMetadata) -> String {
    let mut out = String::new();
    append_go_id(&mut out, m);
    let (funcs, deps) = (m.func_count, m.dependencies.len());
    let _ = writeln!(
        out,
        "Functions: {funcs} recovered\nDependencies: {deps} modules"
    );
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Constructs mock Go buildinfo test bytes.
    fn make_test_buildinfo() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(GO_BUILDINF_MAGIC);
        d.extend_from_slice(&[8, 0]);
        d.extend_from_slice(b"path\tgithub.com/example/app\nmod\tgithub.com/gin-gonic/gin\tv1.9.1\thash\nbuild\tCGO_ENABLED=0\n\0");
        d
    }

    /// Verifies Go buildinfo and modinfo extraction.
    #[test]
    fn test_go_buildinfo_parsing() {
        let data = make_test_buildinfo();
        let mut meta = GoMetadata::default();
        extract_buildinfo(&data, &mut meta);
        assert_eq!(meta.module_path, "github.com/example/app");
        assert_eq!(meta.dependencies.len(), 1);
        assert_eq!(meta.dependencies[0].version, "v1.9.1");
        assert_eq!(meta.build_settings.len(), 1);
    }

    /// Verifies pclntab function count parsing.
    #[test]
    fn test_pclntab_parsing() {
        let mut data = Vec::new();
        data.extend_from_slice(&PCLNTAB_MAGIC_118);
        data.extend_from_slice(&[0, 0, 1, 8]);
        data.extend_from_slice(&142u64.to_le_bytes());
        let mut meta = GoMetadata::default();
        parse_pclntab_at(&data, 0, &mut meta);
        assert_eq!(meta.func_count, 142);
    }

    /// Verifies Go summary formatting.
    #[test]
    fn test_go_summary_format() {
        let m = GoMetadata {
            version: "go1.22.4".into(),
            module_path: "app".into(),
            ..Default::default()
        };
        let s = format_go_summary(&m);
        assert!(s.contains("go1.22.4") && s.contains("app"));
    }
}
