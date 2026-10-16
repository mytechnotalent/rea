//! .NET ECMA-335 CLI Managed Assembly Metadata Recovery
//!
//! **File:** `dotnet.rs`
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

/// Magic signature identifying ECMA-335 CLI metadata header: `"BSJB"`.
pub const BSJB_MAGIC: [u8; 4] = [0x42, 0x53, 0x4a, 0x42];

/// External assembly reference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DotNetAssemblyRef {
    /// Assembly name.
    pub name: String,
    /// Assembly version.
    pub version: String,
}

/// Extracted .NET CLR metadata and type inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DotNetMetadata {
    /// Target CLR runtime version string.
    pub runtime_version: String,
    /// Root assembly definition name.
    pub assembly_name: String,
    /// Count of declared types (`TypeDef`).
    pub type_count: usize,
    /// Count of declared methods (`MethodDef`).
    pub method_count: usize,
    /// Recovered declared type names.
    pub types: Vec<String>,
    /// Recovered declared method names.
    pub methods: Vec<String>,
    /// Stream names present in the assembly.
    pub streams: Vec<String>,

    /// Referenced assemblies from the `AssemblyRef` table.
    pub references: Vec<DotNetAssemblyRef>,
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

/// Reads a 32-bit little endian integer from a slice.
///
/// # Arguments
///
/// * `b` - Byte slice of at least 4 bytes.
///
/// # Returns
///
/// Unsigned 32-bit value.
fn read_u32_le(b: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let take = b.len().min(4);
    buf[..take].copy_from_slice(&b[..take]);
    u32::from_le_bytes(buf)
}

/// Reads a null-terminated UTF-8 string from a slice up to maximum bytes.
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `max_len` - Maximum scan length.
///
/// # Returns
///
/// String content.
fn read_cstring_bounded(bytes: &[u8], max_len: usize) -> String {
    let slice = &bytes[..max_len.min(bytes.len())];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).trim().to_string()
}

/// Parses the BSJB metadata root header fields.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `off` - Offset of BSJB magic.
/// * `meta` - Output .NET metadata.
fn parse_bsjb_root(bytes: &[u8], off: usize, meta: &mut DotNetMetadata) {
    if off + 16 > bytes.len() {
        return;
    }
    let vlen = read_u32_le(&bytes[off + 12..off + 16]) as usize;
    if vlen > 0 && off + 16 + vlen <= bytes.len() {
        meta.runtime_version = read_cstring_bounded(&bytes[off + 16..], vlen);
        parse_streams(bytes, off, off + 16 + vlen, meta);
    }
}

/// Reads stream count from metadata header.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `start` - Offset of streams count.
///
/// # Returns
///
/// Stream count value.
fn stream_header_count(bytes: &[u8], start: usize) -> usize {
    if start + 4 > bytes.len() {
        0
    } else {
        u16::from_le_bytes([bytes[start + 2], bytes[start + 3]]) as usize
    }
}

/// Parses metadata stream headers following the version string.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `bsjb_off` - Root BSJB offset.
/// * `start` - Offset of streams count.
/// * `meta` - Output .NET metadata.
fn parse_streams(bytes: &[u8], bsjb_off: usize, start: usize, meta: &mut DotNetMetadata) {
    let mut cur = start + 4;
    for _ in 0..stream_header_count(bytes, start).min(10) {
        let Some(adv) = parse_stream_entry(bytes, bsjb_off, cur, meta) else {
            break;
        };
        cur += adv;
    }
}

/// Records a parsed stream into metadata and inspects payload.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `loc` - Stream location tuple (`bsjb`, `s_off`, `s_sz`).
/// * `name` - Stream identifier name.
/// * `meta` - Output .NET metadata.
///
/// # Returns
///
/// Byte size consumed by stream entry.
fn record_stream_meta(
    bytes: &[u8],
    loc: (usize, usize, usize),
    name: &str,
    meta: &mut DotNetMetadata,
) -> usize {
    meta.streams.push(name.to_string());
    inspect_stream(bytes, (loc.0 + loc.1, loc.2), name, meta);
    8 + ((name.len() + 4) & !3)
}

/// Reads the stream offset and size from header slice.
fn read_stream_offset_size(bytes: &[u8], cur: usize) -> Option<(usize, usize)> {
    let s = bytes.get(cur..cur + 8)?;
    Some((
        read_u32_le(&s[..4]) as usize,
        read_u32_le(&s[4..8]) as usize,
    ))
}

/// Parses a single stream header entry.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `bsjb` - Root BSJB offset.
/// * `cur` - Current stream header offset.
/// * `meta` - Output .NET metadata.
///
/// # Returns
///
/// Number of bytes consumed by entry if parsed.
fn parse_stream_entry(
    bytes: &[u8],
    bsjb: usize,
    cur: usize,
    meta: &mut DotNetMetadata,
) -> Option<usize> {
    let (s_off, s_sz) = read_stream_offset_size(bytes, cur)?;
    let name = read_cstring_bounded(bytes.get(cur + 8..)?, 32);
    (!name.is_empty()).then(|| record_stream_meta(bytes, (bsjb, s_off, s_sz), &name, meta))
}

/// Inspects specific stream payloads like `#Strings` or `#~`.
///
/// # Arguments
///
/// * `bytes` - Binary buffer.
/// * `loc` - Stream start offset and size tuple.
/// * `name` - Stream identifier name.
/// * `meta` - Output .NET metadata.
fn inspect_stream(bytes: &[u8], loc: (usize, usize), name: &str, meta: &mut DotNetMetadata) {
    if let Some(slice) = bytes.get(loc.0..loc.0.saturating_add(loc.1)) {
        dispatch_stream(slice, name, meta);
    }
}

/// Dispatches stream slice to appropriate parser based on stream name.
fn dispatch_stream(slice: &[u8], name: &str, meta: &mut DotNetMetadata) {
    match name {
        "#~" | "#-" => parse_tables_stream(slice, meta),
        "#Strings" => parse_strings_stream(slice, meta),
        _ => {}
    }
}

/// Parses the `#~` metadata tables stream header for row counts.
///
/// # Arguments
///
/// * `slice` - Tables stream bytes.
/// * `meta` - Output .NET metadata.
fn parse_tables_stream(slice: &[u8], meta: &mut DotNetMetadata) {
    if slice.len() < 24 {
        return;
    }
    let valid = u64::from_le_bytes([
        slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15],
    ]);
    scan_table_rows(slice, valid, meta);
}

/// Scans row counts for each present table in `#~`.
///
/// # Arguments
///
/// * `slice` - Tables stream byte slice.
/// * `valid` - Bitmask of present tables.
/// * `meta` - Output .NET metadata.
fn scan_table_rows(slice: &[u8], valid: u64, meta: &mut DotNetMetadata) {
    let mut cur = 24;
    for table_idx in 0..64 {
        if (valid & (1 << table_idx)) != 0 && !scan_one_table(slice, table_idx, &mut cur, meta) {
            break;
        }
    }
}

/// Reads one table row count and updates cursor offset.
///
/// # Arguments
///
/// * `slice` - Tables stream bytes.
/// * `idx` - Table index.
/// * `cur` - Current stream offset.
/// * `meta` - Output .NET metadata.
///
/// # Returns
///
/// `true` if read was within bounds.
fn scan_one_table(slice: &[u8], idx: usize, cur: &mut usize, meta: &mut DotNetMetadata) -> bool {
    if *cur + 4 > slice.len() {
        return false;
    }
    let rows = read_u32_le(&slice[*cur..*cur + 4]) as usize;
    assign_table_rows(idx, rows, meta);
    *cur += 4;
    true
}

/// Assigns parsed row count to corresponding .NET metadata field.
///
/// # Arguments
///
/// * `idx` - Metadata table index.
/// * `rows` - Number of rows in table.
/// * `meta` - Output .NET metadata.
fn assign_table_rows(idx: usize, rows: usize, meta: &mut DotNetMetadata) {
    match idx {
        0x02 => meta.type_count = rows,
        0x06 => meta.method_count = rows,
        _ => {}
    }
}

/// Collects declared type and method names from `#Strings` pool.
///
/// # Arguments
///
/// * `slice` - Strings stream byte slice.
/// * `meta` - Output .NET metadata.
fn parse_strings_stream(slice: &[u8], meta: &mut DotNetMetadata) {
    for chunk in slice.split(|&b| b == 0) {
        if is_valid_ident(chunk) {
            record_ident(chunk, meta);
        }
    }
}

/// Reports whether a file name is a .NET assembly artifact.
///
/// # Arguments
///
/// * `s` - File name string.
///
/// # Returns
///
/// `true` for .dll or .exe extension.
fn is_assembly_name(s: &str) -> bool {
    let p = std::path::Path::new(s);
    p.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("dll") || e.eq_ignore_ascii_case("exe"))
}

/// Records a valid string identifier into types or assembly name.
///
/// # Arguments
///
/// * `chunk` - Identifier byte chunk.
/// * `meta` - Output .NET metadata.
fn record_ident(chunk: &[u8], meta: &mut DotNetMetadata) {
    let s = String::from_utf8_lossy(chunk).to_string();
    if meta.assembly_name.is_empty() && is_assembly_name(&s) {
        meta.assembly_name.clone_from(&s);
    }
    if meta.types.len() < 40 && is_type_name(&s) {
        meta.types.push(s);
    }
}

/// Reports whether a string looks like a declared .NET type name.
///
/// # Arguments
///
/// * `s` - Identifier string.
///
/// # Returns
///
/// `true` for plausible .NET type names.
fn is_type_name(s: &str) -> bool {
    let first = s.chars().next().unwrap_or(' ');
    first.is_ascii_uppercase() && !s.contains('<') && s.len() >= 3 && s.len() <= 60
}

/// Reports whether a byte chunk is an ASCII identifier.
///
/// # Arguments
///
/// * `chunk` - Byte chunk to inspect.
///
/// # Returns
///
/// `true` for printable non-space ASCII strings.
fn is_valid_ident(chunk: &[u8]) -> bool {
    chunk.len() >= 2 && chunk.len() <= 80 && chunk.iter().all(|&b| b.is_ascii_graphic())
}

/// Parses .NET CLI metadata from binary file buffer.
///
/// # Arguments
///
/// * `bytes` - Binary file bytes.
///
/// # Returns
///
/// Extracted .NET metadata if BSJB signature exists.
#[must_use]
pub fn parse_dotnet_metadata(bytes: &[u8]) -> Option<DotNetMetadata> {
    let off = find_bytes(bytes, &BSJB_MAGIC)?;
    let mut meta = DotNetMetadata::default();
    parse_bsjb_root(bytes, off, &mut meta);
    crate::dotnet_tables::recover_references(bytes, off, &mut meta);
    Some(meta)
}

/// Resolves display label for .NET runtime version.
///
/// # Arguments
///
/// * `m` - Extracted .NET metadata.
///
/// # Returns
///
/// Runtime version string or "unknown".
fn dotnet_runtime_ver(m: &DotNetMetadata) -> &str {
    if m.runtime_version.is_empty() {
        "unknown"
    } else {
        &m.runtime_version
    }
}

/// Formats .NET CLI metadata summary as a human-readable text block.
///
/// # Arguments
///
/// * `m` - Extracted .NET metadata.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_dotnet_summary(m: &DotNetMetadata) -> String {
    let ver = dotnet_runtime_ver(m);
    let (t, meth) = (m.type_count, m.method_count);
    let streams = m.streams.join(", ");
    let refs = m.references.len();
    format!(
        ".NET Runtime: {ver}\nDeclared Types: {t}\nDeclared Methods: {meth}\nStreams: {streams}\nReferenced Assemblies: {refs}\n"
    )
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Constructs mock BSJB header bytes.
    fn make_test_bsjb() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&BSJB_MAGIC);
        d.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
        d.extend_from_slice(&12u32.to_le_bytes());
        d.extend_from_slice(b"v4.0.30319\0\0");
        d.extend_from_slice(&[0, 0, 0, 0]);
        d
    }

    /// Verifies BSJB header and version parsing.
    #[test]
    fn test_dotnet_bsjb_header() {
        let data = make_test_bsjb();
        let mut meta = DotNetMetadata::default();
        parse_bsjb_root(&data, 0, &mut meta);
        assert_eq!(meta.runtime_version, "v4.0.30319");
    }

    /// Verifies table row count assignment.
    #[test]
    fn test_assign_table_rows() {
        let mut m = DotNetMetadata::default();
        assign_table_rows(0x02, 38, &mut m);
        assign_table_rows(0x06, 154, &mut m);
        assert_eq!(m.type_count, 38);
        assert_eq!(m.method_count, 154);
    }

    /// Verifies .NET summary formatting.
    #[test]
    fn test_dotnet_summary_format() {
        let m = DotNetMetadata {
            runtime_version: "v4.0.30319".into(),
            type_count: 10,
            ..Default::default()
        };
        let s = format_dotnet_summary(&m);
        assert!(s.contains("v4.0.30319") && s.contains("Declared Types: 10"));
    }
}
