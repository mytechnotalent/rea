//! WebAssembly Module Parsing and Validation
//!
//! **File:** `wasm.rs`
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

/// WebAssembly binary magic prefix.
pub const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d];

/// WebAssembly binary format version understood by this parser.
pub const WASM_VERSION: u32 = 1;

/// Canonical section names indexed by section id.
const SECTION_NAMES: [&str; 13] = [
    "custom",
    "type",
    "import",
    "function",
    "table",
    "memory",
    "global",
    "export",
    "start",
    "element",
    "code",
    "data",
    "data count",
];

/// Import and export descriptor kind labels indexed by tag byte.
const KIND_LABELS: [&str; 4] = ["func", "table", "memory", "global"];

/// A parsed section header.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasmSection {
    /// Section id.
    pub id: u8,
    /// Section name, the custom name for id zero.
    pub name: String,
    /// Offset of the section body within the module.
    pub offset: usize,
    /// Body length in bytes.
    pub size: usize,
}

/// An imported item and its source module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasmImport {
    /// Import module name.
    pub module: String,
    /// Imported field name.
    pub name: String,
    /// Descriptor kind: `func`, `table`, `memory`, or `global`.
    pub kind: String,
}

/// An exported item.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasmExport {
    /// Export name.
    pub name: String,
    /// Descriptor kind.
    pub kind: String,
}

/// Recovered WebAssembly module inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasmModule {
    /// Binary format version.
    pub version: u32,
    /// Sections in file order.
    pub sections: Vec<WasmSection>,
    /// Imported items.
    pub imports: Vec<WasmImport>,
    /// Exported items.
    pub exports: Vec<WasmExport>,
    /// Declared function count.
    pub functions: usize,
    /// Memory limit descriptions.
    pub memories: Vec<String>,
    /// Encoded size of each function body.
    pub code_sizes: Vec<usize>,
    /// Validation problems discovered while parsing.
    pub warnings: Vec<String>,
}

/// Reports whether a buffer begins with the WebAssembly magic.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the magic prefix matches.
#[must_use]
pub fn is_wasm(bytes: &[u8]) -> bool {
    bytes.starts_with(&WASM_MAGIC)
}

/// Reads an unsigned LEB128 integer at a cursor.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `pos` - Cursor advanced past the integer.
///
/// # Returns
///
/// The decoded value, or `None` on truncation.
fn read_uleb(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let slice = bytes.get(*pos..)?;
    let (value, used) = decode_uleb(slice)?;
    *pos += used;
    Some(value)
}

/// Decodes an unsigned LEB128 integer and its encoded length.
///
/// # Arguments
///
/// * `bytes` - Buffer beginning at the integer.
///
/// # Returns
///
/// The value and the number of bytes consumed.
fn decode_uleb(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut result = 0u32;
    for (i, &byte) in bytes.iter().enumerate().take(5) {
        result |= u32::from(byte & 0x7f) << u32::try_from(i * 7).ok()?;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
    }
    None
}

/// Reads a length-prefixed UTF-8 name at a cursor.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `pos` - Cursor advanced past the name.
///
/// # Returns
///
/// The decoded name, or `None` on truncation.
fn read_name(bytes: &[u8], pos: &mut usize) -> Option<String> {
    let len = to_usize(read_uleb(bytes, pos)?)?;
    let end = pos.checked_add(len)?;
    let raw = bytes.get(*pos..end)?;
    *pos = end;
    Some(String::from_utf8_lossy(raw).to_string())
}

/// Converts a 32-bit count into a host index.
///
/// # Arguments
///
/// * `value` - Value to convert.
///
/// # Returns
///
/// The value as `usize`, or `None` when it does not fit.
fn to_usize(value: u32) -> Option<usize> {
    usize::try_from(value).ok()
}

/// Resolves a descriptor kind label from its tag byte.
///
/// # Arguments
///
/// * `byte` - Descriptor tag byte.
///
/// # Returns
///
/// A canonical kind label or `unknown`.
fn kind_label(byte: u8) -> String {
    let label = KIND_LABELS.get(usize::from(byte)).copied();
    label.unwrap_or("unknown").to_string()
}

/// Resolves a canonical section name for a section id.
///
/// # Arguments
///
/// * `id` - Section id byte.
///
/// # Returns
///
/// A canonical section name or `unknown`.
fn canonical_name(id: u8) -> &'static str {
    let name = SECTION_NAMES.get(usize::from(id)).copied();
    name.unwrap_or("unknown")
}

/// Resolves the display name of a section.
///
/// # Arguments
///
/// * `id` - Section id byte.
/// * `body` - Section body.
///
/// # Returns
///
/// The custom name for id zero, otherwise the canonical name.
fn section_name(id: u8, body: &[u8]) -> String {
    if id == 0 {
        let mut pos = 0;
        read_name(body, &mut pos).unwrap_or_default()
    } else {
        canonical_name(id).to_string()
    }
}

/// Parses a WebAssembly module from a buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Recovered module inventory, or `None` when the magic is absent.
#[must_use]
pub fn parse_wasm(bytes: &[u8]) -> Option<WasmModule> {
    if !is_wasm(bytes) {
        return None;
    }
    let mut module = WasmModule { version: read_version(bytes), ..Default::default() };
    parse_sections(bytes, &mut module);
    record_warnings(&mut module);
    Some(module)
}

/// Reads the little-endian version word of a module.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The version field, or zero when truncated.
fn read_version(bytes: &[u8]) -> u32 {
    let word = bytes.get(4..8).unwrap_or_default();
    let mut buf = [0u8; 4];
    buf[..word.len()].copy_from_slice(word);
    u32::from_le_bytes(buf)
}

/// Iterates every section starting after the header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `module` - Output module inventory.
fn parse_sections(bytes: &[u8], module: &mut WasmModule) {
    let mut pos = 8;
    while pos < bytes.len() {
        if parse_section(bytes, &mut pos, module).is_none() {
            break;
        }
    }
}

/// Parses one section header, body, and dispatch.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Cursor advanced past the section.
/// * `module` - Output module inventory.
///
/// # Returns
///
/// `Some` when a section was parsed, otherwise `None`.
fn parse_section(bytes: &[u8], pos: &mut usize, module: &mut WasmModule) -> Option<()> {
    let id = *bytes.get(*pos)?;
    *pos += 1;
    let size = to_usize(read_uleb(bytes, pos)?)?;
    let end = pos.saturating_add(size).min(bytes.len());
    let body = &bytes[*pos..end];
    handle_section(id, *pos, size, body, module);
    *pos = end;
    Some(())
}

/// Records and dispatches one section.
///
/// # Arguments
///
/// * `id` - Section id byte.
/// * `offset` - Body offset.
/// * `size` - Body length.
/// * `body` - Section body.
/// * `module` - Output module inventory.
fn handle_section(id: u8, offset: usize, size: usize, body: &[u8], module: &mut WasmModule) {
    record_section(id, offset, size, body, module);
    dispatch_section(id, body, module);
}

/// Records a section header into the module inventory.
///
/// # Arguments
///
/// * `id` - Section id byte.
/// * `offset` - Body offset.
/// * `size` - Body length.
/// * `body` - Section body.
/// * `module` - Output module inventory.
fn record_section(id: u8, offset: usize, size: usize, body: &[u8], module: &mut WasmModule) {
    let name = section_name(id, body);
    module.sections.push(WasmSection { id, name, offset, size });
}

/// Routes a section body to the matching parser.
///
/// # Arguments
///
/// * `id` - Section id byte.
/// * `body` - Section body.
/// * `module` - Output module inventory.
fn dispatch_section(id: u8, body: &[u8], module: &mut WasmModule) {
    match id {
        2 => parse_imports(body, module),
        3 => module.functions = count_entries(body),
        5 => parse_memories(body, module),
        7 => parse_exports(body, module),
        10 => parse_code(body, module),
        _ => {}
    }
}

/// Parses the code section function bodies.
///
/// # Arguments
///
/// * `body` - Code section body.
/// * `module` - Output module inventory.
fn parse_code(body: &[u8], module: &mut WasmModule) {
    let mut pos = 0;
    let count = read_uleb(body, &mut pos).and_then(to_usize).unwrap_or(0);
    while module.code_sizes.len() < count && push_code(body, &mut pos, module) {}
}

/// Records one code-section function body size.
///
/// # Arguments
///
/// * `body` - Code section body.
/// * `pos` - Cursor advanced past the body.
/// * `module` - Output module inventory.
///
/// # Returns
///
/// `true` while the cursor remains within the section.
fn push_code(body: &[u8], pos: &mut usize, module: &mut WasmModule) -> bool {
    let Some(size) = read_uleb(body, pos).and_then(to_usize) else {
        return false;
    };
    module.code_sizes.push(size);
    *pos = pos.saturating_add(size);
    *pos <= body.len()
}

/// Reads the leading vector count of a section body.
///
/// # Arguments
///
/// * `body` - Section body.
///
/// # Returns
///
/// The declared entry count, or zero when truncated.
fn count_entries(body: &[u8]) -> usize {
    let mut pos = 0;
    read_uleb(body, &mut pos).and_then(to_usize).unwrap_or(0)
}

/// Parses the import section into the module inventory.
///
/// # Arguments
///
/// * `body` - Import section body.
/// * `module` - Output module inventory.
fn parse_imports(body: &[u8], module: &mut WasmModule) {
    let mut pos = 0;
    let count = read_uleb(body, &mut pos).and_then(to_usize).unwrap_or(0);
    for _ in 0..count {
        if read_import(body, &mut pos, module).is_none() {
            break;
        }
    }
}

/// Reads one import descriptor into the module inventory.
///
/// # Arguments
///
/// * `body` - Import section body.
/// * `pos` - Cursor advanced past the descriptor.
/// * `module` - Output module inventory.
///
/// # Returns
///
/// `Some` when a descriptor was read.
fn read_import(body: &[u8], pos: &mut usize, module: &mut WasmModule) -> Option<()> {
    let name = read_name(body, pos)?;
    let field = read_name(body, pos)?;
    let kind = *body.get(*pos)?;
    *pos += 1;
    let import = WasmImport { module: name, name: field, kind: kind_label(kind) };
    module.imports.push(import);
    Some(())
}

/// Parses the export section into the module inventory.
///
/// # Arguments
///
/// * `body` - Export section body.
/// * `module` - Output module inventory.
fn parse_exports(body: &[u8], module: &mut WasmModule) {
    let mut pos = 0;
    let count = read_uleb(body, &mut pos).and_then(to_usize).unwrap_or(0);
    for _ in 0..count {
        if read_export(body, &mut pos, module).is_none() {
            break;
        }
    }
}

/// Reads one export descriptor into the module inventory.
///
/// # Arguments
///
/// * `body` - Export section body.
/// * `pos` - Cursor advanced past the descriptor.
/// * `module` - Output module inventory.
///
/// # Returns
///
/// `Some` when a descriptor was read.
fn read_export(body: &[u8], pos: &mut usize, module: &mut WasmModule) -> Option<()> {
    let name = read_name(body, pos)?;
    let kind = *body.get(*pos)?;
    *pos += 1;
    let export = WasmExport { name, kind: kind_label(kind) };
    module.exports.push(export);
    Some(())
}

/// Parses the memory section into the module inventory.
///
/// # Arguments
///
/// * `body` - Memory section body.
/// * `module` - Output module inventory.
fn parse_memories(body: &[u8], module: &mut WasmModule) {
    let mut pos = 0;
    let count = read_uleb(body, &mut pos).and_then(to_usize).unwrap_or(0);
    for _ in 0..count {
        if let Some(text) = read_memory(body, &mut pos) {
            module.memories.push(text);
        }
    }
}

/// Reads one memory limit descriptor.
///
/// # Arguments
///
/// * `body` - Memory section body.
/// * `pos` - Cursor advanced past the descriptor.
///
/// # Returns
///
/// A human-readable limit description.
fn read_memory(body: &[u8], pos: &mut usize) -> Option<String> {
    let flags = read_uleb(body, pos)?;
    let min = read_uleb(body, pos)?;
    let max = (flags & 1 == 1).then(|| read_uleb(body, pos)).flatten();
    Some(format_memory(min, max))
}

/// Formats a memory limit description.
///
/// # Arguments
///
/// * `min` - Minimum page count.
/// * `max` - Optional maximum page count.
///
/// # Returns
///
/// A human-readable limit description.
fn format_memory(min: u32, max: Option<u32>) -> String {
    match max {
        Some(limit) => format!("min {min} pages, max {limit} pages"),
        None => format!("min {min} pages"),
    }
}

/// Records validation warnings on the module inventory.
///
/// # Arguments
///
/// * `module` - Output module inventory.
fn record_warnings(module: &mut WasmModule) {
    if module.version != WASM_VERSION {
        module.warnings.push(format!("unexpected version {}", module.version));
    }
}

/// Formats a WebAssembly inventory as a human-readable text block.
///
/// # Arguments
///
/// * `m` - Recovered module inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_wasm_summary(m: &WasmModule) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "WebAssembly version {}", m.version);
    let _ = writeln!(out, "Sections: {}", m.sections.len());
    let _ = writeln!(out, "Functions: {}", m.functions);
    append_wasm_counts(&mut out, m);
    out
}

/// Appends import and export counts.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `m` - Recovered module inventory.
fn append_wasm_counts(out: &mut String, m: &WasmModule) {
    let _ = writeln!(out, "Imports: {}", m.imports.len());
    let _ = writeln!(out, "Exports: {}", m.exports.len());
    let _ = writeln!(out, "Code bodies: {}", m.code_sizes.len());
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a module header with the given trailing section bytes.
    ///
    /// # Arguments
    ///
    /// * `tail` - Section bytes following the eight-byte header.
    ///
    /// # Returns
    ///
    /// A complete module buffer.
    fn module_with(tail: &[u8]) -> Vec<u8> {
        let mut data = WASM_MAGIC.to_vec();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(tail);
        data
    }

    /// Verifies WebAssembly magic detection.
    #[test]
    fn test_wasm_detection() {
        assert!(is_wasm(&WASM_MAGIC));
        assert!(!is_wasm(b"\x7fELF"));
    }

    /// Verifies header-only parsing and version recovery.
    #[test]
    fn test_wasm_header() {
        let module = parse_wasm(&module_with(&[])).unwrap();
        assert_eq!(module.version, 1);
        assert_eq!(module.sections.len(), 0);
    }

    /// Verifies import section parsing.
    #[test]
    fn test_wasm_imports() {
        let body = [0x01, 0x03, b'e', b'n', b'v', 0x03, b'f', b'o', b'o', 0x00];
        let mut tail = vec![0x02, u8::try_from(body.len()).unwrap()];
        tail.extend_from_slice(&body);
        let module = parse_wasm(&module_with(&tail)).unwrap();
        assert_eq!(module.imports.len(), 1);
        assert_eq!(module.imports[0].module, "env");
        assert_eq!(module.imports[0].name, "foo");
        assert_eq!(module.imports[0].kind, "func");
    }

    /// Verifies export section parsing and function counting.
    #[test]
    fn test_wasm_exports() {
        let body = [0x01, 0x04, b'm', b'a', b'i', b'n', 0x00, 0x00];
        let mut tail = vec![0x07, u8::try_from(body.len()).unwrap()];
        tail.extend_from_slice(&body);
        let module = parse_wasm(&module_with(&tail)).unwrap();
        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].name, "main");
    }

    /// Verifies that a custom section exposes its name.
    #[test]
    fn test_wasm_custom_section_name() {
        let body = [0x04, b'n', b'a', b'm', b'e', 0x01, 0x02];
        let mut tail = vec![0x00, u8::try_from(body.len()).unwrap()];
        tail.extend_from_slice(&body);
        let module = parse_wasm(&module_with(&tail)).unwrap();
        assert_eq!(module.sections[0].name, "name");
    }

    /// Verifies summary formatting.
    #[test]
    fn test_wasm_summary() {
        let module = WasmModule { version: 1, functions: 3, ..Default::default() };
        let text = format_wasm_summary(&module);
        assert!(text.contains("Functions: 3"));
    }

    /// Verifies code-section function body parsing.
    #[test]
    fn test_wasm_code_section() {
        let body = [0x01, 0x04, 0xaa, 0xbb, 0xcc, 0xdd];
        let mut tail = vec![0x0a, u8::try_from(body.len()).unwrap()];
        tail.extend_from_slice(&body);
        let module = parse_wasm(&module_with(&tail)).unwrap();
        assert_eq!(module.code_sizes, vec![4]);
    }
}
