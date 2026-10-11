//! Electron ASAR Archive Parsing and JavaScript Triage
//!
//! **File:** `asar.rs`
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

// Import dependencies from serde_json
use serde_json::Value;

/// An Electron security finding or dangerous setting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsarWarning {
    /// Severity level.
    pub severity: &'static str,
    /// Vulnerability title.
    pub title: &'static str,
    /// Location file path.
    pub location: String,
    /// Evidence snippet or detail.
    pub detail: String,
}

/// An identified Electron IPC communication channel.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsarIpcChannel {
    /// Direction or role.
    pub direction: &'static str,
    /// Channel name.
    pub name: String,
    /// Source file.
    pub file: String,
}

/// A hardcoded secret or credential discovered in JavaScript.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsarSecret {
    /// Secret type category.
    pub kind: &'static str,
    /// Originating file.
    pub file: String,
    /// Extracted token or snippet.
    pub snippet: String,
}

/// Results of Electron ASAR archive triage.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AsarTriage {
    /// Total file count in archive.
    pub file_count: usize,
    /// Total unpacked size.
    pub unpacked_size: u64,
    /// File paths in archive.
    pub files: Vec<String>,
    /// Application name from `package.json`.
    pub app_name: Option<String>,
    /// Application version.
    pub app_version: Option<String>,
    /// Main script entry point.
    pub main_script: Option<String>,
    /// Hardcoded secrets found.
    pub secrets: Vec<AsarSecret>,
    /// Identified IPC channels.
    pub ipc_channels: Vec<AsarIpcChannel>,
    /// Electron security warnings.
    pub warnings: Vec<AsarWarning>,
}

/// Discovered file entry metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct FileMeta {
    /// File path.
    path: String,
    /// File size in bytes.
    size: u64,
    /// File offset within payload.
    offset: u64,
}

/// Reads a 32-bit little-endian integer from a slice.
///
/// # Arguments
///
/// * `b` - Source byte slice.
/// * `idx` - Offset index.
///
/// # Returns
///
/// The 32-bit integer, or `None` if out of bounds.
fn read_u32(b: &[u8], idx: usize) -> Option<u32> {
    let s = b.get(idx..idx + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Parses the ASAR header bounds.
///
/// # Arguments
///
/// * `bytes` - ASAR file bytes.
///
/// # Returns
///
/// A pair containing the JSON header length and payload base offset.
fn header_bounds(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() < 16 || read_u32(bytes, 0)? != 4 {
        return None;
    }
    let json_len = read_u32(bytes, 12)? as usize;
    let base = 16 + ((json_len + 3) & !3);
    Some((json_len, base))
}

/// Extracts the JSON header string from bytes.
///
/// # Arguments
///
/// * `bytes` - ASAR file bytes.
/// * `len` - Length of JSON header.
///
/// # Returns
///
/// UTF-8 header string slice, or `None` if invalid.
fn extract_header_json(bytes: &[u8], len: usize) -> Option<&str> {
    let slice = bytes.get(16..16 + len)?;
    std::str::from_utf8(slice).ok()
}

/// Checks if an entry is a directory and returns its sub-files value.
///
/// # Arguments
///
/// * `obj` - JSON directory entry value.
///
/// # Returns
///
/// Inner files map if the value is a directory.
fn dir_files(obj: &Value) -> Option<&serde_json::Map<String, Value>> {
    obj.get("files")?.as_object()
}

/// Extracts size and offset from a leaf file value.
///
/// # Arguments
///
/// * `path` - Virtual archive file path.
/// * `obj` - JSON file node value.
///
/// # Returns
///
/// Parsed file metadata, or `None` if missing attributes.
fn parse_leaf_meta(path: &str, obj: &Value) -> Option<FileMeta> {
    let size = obj.get("size")?.as_u64()?;
    let offset_str = obj.get("offset")?.as_str()?;
    let offset = offset_str.parse::<u64>().ok()?;
    Some(FileMeta { path: path.to_string(), size, offset })
}

/// Generates archive path for an entry.
///
/// # Arguments
///
/// * `prefix` - Parent directory path.
/// * `name` - File or directory name.
///
/// # Returns
///
/// Joined path string.
fn entry_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

/// Traverses a single directory entry.
///
/// # Arguments
///
/// * `prefix` - Parent directory path.
/// * `name` - Entry name.
/// * `val` - JSON node value.
/// * `out` - Output metadata accumulator.
fn walk_entry(prefix: &str, name: &str, val: &Value, out: &mut Vec<FileMeta>) {
    let path = entry_path(prefix, name);
    if let Some(sub) = dir_files(val) {
        walk_dir(&path, sub, out);
    } else if let Some(meta) = parse_leaf_meta(&path, val) {
        out.push(meta);
    }
}

/// Recursively traverses directory entries.
///
/// # Arguments
///
/// * `prefix` - Current directory prefix.
/// * `dir` - JSON directory map.
/// * `out` - Output metadata accumulator.
fn walk_dir(prefix: &str, dir: &serde_json::Map<String, Value>, out: &mut Vec<FileMeta>) {
    for (name, val) in dir {
        walk_entry(prefix, name, val, out);
    }
}

/// Collects file records from parsed JSON header.
///
/// # Arguments
///
/// * `root` - Root JSON header value.
///
/// # Returns
///
/// Flat list of file metadata records.
fn collect_files(root: &Value) -> Vec<FileMeta> {
    let mut files = Vec::new();
    if let Some(top) = root.get("files").and_then(Value::as_object) {
        walk_dir("", top, &mut files);
    }
    files
}

/// Reads file bytes from the archive.
///
/// # Arguments
///
/// * `archive` - Full archive bytes.
/// * `base` - Payload base offset.
/// * `meta` - File metadata record.
///
/// # Returns
///
/// Slice of file payload, or `None` if out of bounds.
fn file_bytes<'a>(archive: &'a [u8], base: usize, meta: &FileMeta) -> Option<&'a [u8]> {
    let start = base.checked_add(usize::try_from(meta.offset).ok()?)?;
    let len = usize::try_from(meta.size).ok()?;
    archive.get(start..start.checked_add(len)?)
}

/// Inspects `package.json` for application metadata.
///
/// # Arguments
///
/// * `content` - Raw `package.json` bytes.
/// * `t` - Triage report accumulator.
fn check_package(content: &[u8], t: &mut AsarTriage) {
    let Ok(v) = serde_json::from_slice::<Value>(content) else {
        return;
    };
    t.app_name = v.get("name").and_then(Value::as_str).map(ToString::to_string);
    t.app_version = v.get("version").and_then(Value::as_str).map(ToString::to_string);
    t.main_script = v.get("main").and_then(Value::as_str).map(ToString::to_string);
}

/// Security flag definition for string inspection.
struct SecurityRule {
    /// Needle to search for.
    needle: &'static str,
    /// Rule title.
    title: &'static str,
    /// Severity.
    severity: &'static str,
}

/// Table of common Electron security misconfigurations.
const RULES: [SecurityRule; 5] = [
    SecurityRule {
        needle: "nodeIntegration: true",
        title: "Node Integration Enabled in Renderer",
        severity: "High",
    },
    SecurityRule {
        needle: "contextIsolation: false",
        title: "Context Isolation Disabled",
        severity: "High",
    },
    SecurityRule { needle: "webSecurity: false", title: "Web Security Disabled", severity: "High" },
    SecurityRule {
        needle: "allowRunningInsecureContent: true",
        title: "Mixed Content Allowed",
        severity: "Medium",
    },
    SecurityRule {
        needle: "shell.openExternal(",
        title: "Untrusted Protocol Launch via openExternal",
        severity: "Medium",
    },
];

/// Evaluates one security misconfiguration rule against text.
///
/// # Arguments
///
/// * `path` - File path under inspection.
/// * `text` - Script source code text.
/// * `r` - Security rule definition.
/// * `out` - Output warning list.
fn check_rule(path: &str, text: &str, r: &SecurityRule, out: &mut Vec<AsarWarning>) {
    if text.contains(r.needle) {
        let (s, t, d) = (r.severity, r.title, r.needle.to_string());
        out.push(AsarWarning { severity: s, title: t, location: path.to_string(), detail: d });
    }
}

/// Scans text for security misconfigurations.
///
/// # Arguments
///
/// * `path` - Script file path.
/// * `text` - Script text content.
/// * `out` - Output warning list.
fn scan_rules(path: &str, text: &str, out: &mut Vec<AsarWarning>) {
    for r in &RULES {
        check_rule(path, text, r, out);
    }
}

/// Common secret prefixes checked in scripts.
const SECRET_PREFIXES: [(&str, &str); 5] = [
    ("AKIA", "AWS Access Key"),
    ("AIza", "Google API Key"),
    ("ghp_", "GitHub Personal Access Token"),
    ("sk-proj-", "OpenAI API Key"),
    ("-----BEGIN PRIVATE KEY-----", "RSA/ECDSA Private Key"),
];

/// Checks for a specific secret prefix in text.
///
/// # Arguments
///
/// * `path` - Source file path.
/// * `text` - Script text content.
/// * `pfx` - Secret token prefix needle.
/// * `kind` - Human-readable secret label.
/// * `out` - Output secret list.
fn check_secret(path: &str, text: &str, pfx: &str, kind: &'static str, out: &mut Vec<AsarSecret>) {
    if let Some(pos) = text.find(pfx) {
        let end = text[pos..]
            .find(['"', '\'', ' ', '\n', '\r', ';'])
            .map_or(text.len(), |e| pos + e);
        let snippet = text[pos..end.min(pos + 60)].to_string();
        out.push(AsarSecret { kind, file: path.to_string(), snippet });
    }
}

/// Scans text for hardcoded secret patterns.
///
/// # Arguments
///
/// * `path` - Source file path.
/// * `text` - Script text content.
/// * `out` - Output secret list.
fn scan_secrets(path: &str, text: &str, out: &mut Vec<AsarSecret>) {
    for (prefix, kind) in &SECRET_PREFIXES {
        check_secret(path, text, prefix, kind, out);
    }
}

/// Identifies IPC method call prefixes.
const IPC_PREFIXES: [(&str, &str); 4] = [
    ("ipcRenderer.send(\"", "Renderer->Main"),
    ("ipcRenderer.invoke(\"", "Renderer->Main"),
    ("ipcMain.handle(\"", "Main Handler"),
    ("ipcMain.on(\"", "Main Listener"),
];

/// Extracts channel name following prefix.
///
/// # Arguments
///
/// * `text` - Script text content.
/// * `pos` - Byte offset of call prefix.
/// * `pfx_len` - Length of prefix needle.
///
/// # Returns
///
/// Quoted channel name, or `None` if malformed.
fn extract_channel(text: &str, pos: usize, pfx_len: usize) -> Option<String> {
    let after = text.get(pos + pfx_len..)?;
    let quote = after.find('"')?;
    Some(after[..quote].to_string())
}

/// Scans for occurrences of a single IPC call prefix.
///
/// # Arguments
///
/// * `path` - Source file path.
/// * `text` - Script text content.
/// * `pfx` - IPC call prefix needle.
/// * `dir` - Communication direction label.
/// * `out` - Output IPC channel list.
fn check_ipc(path: &str, text: &str, pfx: &str, dir: &'static str, out: &mut Vec<AsarIpcChannel>) {
    let mut idx = 0;
    while let Some(rel) = text[idx..].find(pfx) {
        let pos = idx + rel;
        if let Some(ch) = extract_channel(text, pos, pfx.len()) {
            out.push(AsarIpcChannel { direction: dir, name: ch, file: path.to_string() });
        }
        idx = pos + pfx.len();
    }
}

/// Scans text for Electron IPC channel invocations.
///
/// # Arguments
///
/// * `path` - Source file path.
/// * `text` - Script text content.
/// * `out` - Output IPC channel list.
fn scan_ipc(path: &str, text: &str, out: &mut Vec<AsarIpcChannel>) {
    for (pfx, dir) in &IPC_PREFIXES {
        check_ipc(path, text, pfx, dir, out);
    }
}

/// Checks whether a path corresponds to a JavaScript script.
///
/// # Arguments
///
/// * `path` - File path in archive.
///
/// # Returns
///
/// `true` for .js, .cjs, or .mjs extensions.
fn is_js_script(path: &str) -> bool {
    let p = std::path::Path::new(path);
    p.extension().is_some_and(|e| {
        e.eq_ignore_ascii_case("js")
            || e.eq_ignore_ascii_case("cjs")
            || e.eq_ignore_ascii_case("mjs")
    })
}

/// Inspects script contents for security warnings, secrets, and IPC.
///
/// # Arguments
///
/// * `path` - Script file path.
/// * `text` - Script text content.
/// * `t` - Triage report accumulator.
fn inspect_script(path: &str, text: &str, t: &mut AsarTriage) {
    scan_rules(path, text, &mut t.warnings);
    scan_secrets(path, text, &mut t.secrets);
    scan_ipc(path, text, &mut t.ipc_channels);
}

/// Inspects a JavaScript or JSON file.
///
/// # Arguments
///
/// * `path` - File path in archive.
/// * `bytes` - File content bytes.
/// * `t` - Triage report accumulator.
fn inspect_file(path: &str, bytes: &[u8], t: &mut AsarTriage) {
    if path == "package.json" || path.ends_with("/package.json") {
        check_package(bytes, t);
    }
    if is_js_script(path) {
        inspect_script(path, &String::from_utf8_lossy(bytes), t);
    }
}

/// Aggregates file statistics and inspects individual files.
///
/// # Arguments
///
/// * `archive` - Full archive bytes.
/// * `base` - Payload base offset.
/// * `list` - Discovered file metadata list.
/// * `t` - Triage report accumulator.
fn process_entries(archive: &[u8], base: usize, list: &[FileMeta], t: &mut AsarTriage) {
    for m in list {
        t.files.push(m.path.clone());
        t.unpacked_size = t.unpacked_size.saturating_add(m.size);
        if let Some(content) = file_bytes(archive, base, m) {
            inspect_file(&m.path, content, t);
        }
    }
}

/// Parses an Electron ASAR archive from raw bytes.
///
/// # Arguments
///
/// * `bytes` - ASAR archive file buffer.
///
/// # Returns
///
/// Discovered triage details, or `None` if invalid.
#[must_use]
pub fn parse(bytes: &[u8]) -> Option<AsarTriage> {
    let (json_len, base) = header_bounds(bytes)?;
    let json_str = extract_header_json(bytes, json_len)?;
    let root: Value = serde_json::from_str(json_str).ok()?;
    let files = collect_files(&root);
    let mut triage = AsarTriage { file_count: files.len(), ..Default::default() };
    process_entries(bytes, base, &files, &mut triage);
    Some(triage)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Tests invalid buffer rejection.
    #[test]
    fn test_parse_invalid() {
        assert!(parse(b"not an asar").is_none());
        assert!(parse(&[]).is_none());
    }

    /// Builds synthetic archive header bytes.
    ///
    /// # Arguments
    ///
    /// * `jb` - JSON header bytes.
    /// * `pad` - Padding byte count.
    ///
    /// # Returns
    ///
    /// Header buffer with length words.
    fn build_test_header(jb: &[u8], pad: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&u32::try_from(jb.len() + pad + 8).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&u32::try_from(jb.len() + pad + 4).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&u32::try_from(jb.len()).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(jb);
        buf.extend(std::iter::repeat_n(0, pad));
        buf
    }

    /// Builds synthetic test ASAR archive bytes.
    ///
    /// # Returns
    ///
    /// Complete mock archive containing metadata and payload.
    fn build_test_asar() -> Vec<u8> {
        let json = r#"{"files":{"package.json":{"size":27,"offset":"0"},"main.js":{"size":35,"offset":"27"}}}"#;
        let pad = (4 - (json.len() % 4)) % 4;
        let mut buf = build_test_header(json.as_bytes(), pad);
        buf.extend_from_slice(br#"{"name":"test-app","v":"1"}"#);
        buf.extend_from_slice(b"const x = \"nodeIntegration: true\";\n");
        buf
    }

    /// Tests synthetic ASAR archive header parsing.
    #[test]
    fn test_parse_synthetic_asar() {
        let triage = parse(&build_test_asar()).expect("parsed");
        assert_eq!(triage.file_count, 2);
        assert_eq!(triage.app_name.as_deref(), Some("test-app"));
        assert!(!triage.warnings.is_empty());
    }
}
