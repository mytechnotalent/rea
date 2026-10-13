//! Android Resource Table (resources.arsc) Parsing
//!
//! **File:** `arsc.rs`
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

/// Resource table header chunk type.
const TABLE_TYPE: u16 = 0x0002;

/// Global string pool chunk type.
const STRING_POOL_TYPE: u16 = 0x0001;

/// Package chunk type.
const PACKAGE_TYPE: u16 = 0x0200;

/// Configuration chunk type.
const CONFIG_TYPE: u16 = 0x0201;

/// Flag marking a UTF-8 string pool.
const UTF8_FLAG: u32 = 0x0000_0100;

/// Most strings retained from the global pool.
const MAX_STRINGS: usize = 4096;

/// Largest reasonable table header size.
const MAX_HEADER: u16 = 64;

/// Recovered Android resource table inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ArscTable {
    /// Declared package count.
    pub package_count: u32,
    /// Global string-pool values.
    pub strings: Vec<String>,
    /// Package names.
    pub packages: Vec<String>,
    /// Configuration locales.
    pub locales: Vec<String>,
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

/// Reports whether a buffer is an Android resource table.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the header chunk matches.
#[must_use]
pub fn is_arsc(bytes: &[u8]) -> bool {
    read_u16_le(bytes, 0) == Some(TABLE_TYPE)
        && read_u16_le(bytes, 2).is_some_and(|h| (12..=MAX_HEADER).contains(&h))
}

/// Reads a chunk header size.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Chunk offset.
///
/// # Returns
///
/// The declared header size.
fn header_size(bytes: &[u8], pos: usize) -> usize {
    usize::from(read_u16_le(bytes, pos + 2).unwrap_or(0))
}

/// Parses a resource table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered table, or `None`.
#[must_use]
pub fn parse_arsc(bytes: &[u8]) -> Option<ArscTable> {
    if !is_arsc(bytes) {
        return None;
    }
    let mut table =
        ArscTable { package_count: read_u32_le(bytes, 8).unwrap_or(0), ..Default::default() };
    parse_chunks(bytes, header_size(bytes, 0), &mut table);
    Some(table)
}

/// Walks a contiguous run of chunks.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - First chunk offset.
/// * `table` - Output table.
fn parse_chunks(bytes: &[u8], start: usize, table: &mut ArscTable) {
    let mut pos = start;
    while pos + 8 <= bytes.len() {
        let Some(next) = parse_chunk(bytes, pos, table) else {
            break;
        };
        pos = next;
    }
}

/// Parses one chunk and returns the next offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Chunk offset.
/// * `table` - Output table.
///
/// # Returns
///
/// The next chunk offset, or `None`.
fn parse_chunk(bytes: &[u8], pos: usize, table: &mut ArscTable) -> Option<usize> {
    let kind = read_u16_le(bytes, pos)?;
    let size = usize::try_from(read_u32_le(bytes, pos + 4)?).ok()?;
    if size < 8 {
        return None;
    }
    dispatch_chunk(bytes, pos, kind, size, table);
    Some(pos.saturating_add(size))
}

/// Routes a chunk to the matching parser.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Chunk offset.
/// * `kind` - Chunk type.
/// * `size` - Chunk size.
/// * `table` - Output table.
fn dispatch_chunk(bytes: &[u8], pos: usize, kind: u16, size: usize, table: &mut ArscTable) {
    match kind {
        STRING_POOL_TYPE => table.strings = parse_string_pool(bytes, pos).unwrap_or_default(),
        PACKAGE_TYPE => parse_package(bytes, pos, size, table),
        _ => {}
    }
}

/// Parses a string pool chunk.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - Pool chunk offset.
///
/// # Returns
///
/// The decoded strings, or `None`.
fn parse_string_pool(bytes: &[u8], start: usize) -> Option<Vec<String>> {
    let count = usize::try_from(read_u32_le(bytes, start + 8)?).ok()?;
    let flags = read_u32_le(bytes, start + 16)?;
    let base = start.checked_add(usize::try_from(read_u32_le(bytes, start + 20)?).ok()?)?;
    let strings = (0..count.min(MAX_STRINGS))
        .filter_map(|i| pool_string(bytes, start + 28 + i * 4, base, flags));
    Some(strings.collect())
}

/// Reads one string pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `offset_pos` - Offset of the string's offset word.
/// * `base` - Start of the string data.
/// * `flags` - Pool flags.
///
/// # Returns
///
/// The decoded string, or `None`.
fn pool_string(bytes: &[u8], offset_pos: usize, base: usize, flags: u32) -> Option<String> {
    let offset = usize::try_from(read_u32_le(bytes, offset_pos)?).ok()?;
    let at = base.checked_add(offset)?;
    if flags & UTF8_FLAG != 0 {
        read_utf8_string(bytes, at)
    } else {
        read_utf16_string(bytes, at)
    }
}

/// Reads a length-prefixed UTF-8 pool string.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - String offset.
///
/// # Returns
///
/// The decoded string, or `None`.
fn read_utf8_string(bytes: &[u8], at: usize) -> Option<String> {
    let first = *bytes.get(at)?;
    let (len, start) = utf8_len(bytes, at, first);
    let raw = bytes.get(start..start + len)?;
    Some(String::from_utf8_lossy(raw).to_string())
}

/// Decodes a UTF-8 string pool length prefix.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - String offset.
/// * `first` - First length byte.
///
/// # Returns
///
/// The length and the data offset.
fn utf8_len(bytes: &[u8], at: usize, first: u8) -> (usize, usize) {
    if first & 0x80 != 0 {
        let second = bytes.get(at + 1).copied().unwrap_or(0);
        let length = (u16::from(first & 0x7f) << 8) | u16::from(second);
        (usize::from(length), at + 2)
    } else {
        (usize::from(first), at + 1)
    }
}

/// Reads a length-prefixed UTF-16 pool string.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - String offset.
///
/// # Returns
///
/// The decoded string, or `None`.
fn read_utf16_string(bytes: &[u8], at: usize) -> Option<String> {
    let (len, start) = utf16_len(bytes, at);
    let raw = bytes.get(start..start + len * 2)?;
    let units: Vec<u16> =
        (0..len).map(|i| u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]])).collect();
    Some(String::from_utf16_lossy(&units))
}

/// Decodes a UTF-16 string pool length prefix.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - String offset.
///
/// # Returns
///
/// The length and the data offset.
fn utf16_len(bytes: &[u8], at: usize) -> (usize, usize) {
    let first = read_u16_le(bytes, at).unwrap_or(0);
    if first & 0x8000 != 0 {
        return (utf16_len_wide(bytes, at, first), at + 4);
    }
    (usize::from(first), at + 2)
}

/// Decodes a two-unit UTF-16 string pool length.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - String offset.
/// * `first` - First length unit.
///
/// # Returns
///
/// The decoded length.
fn utf16_len_wide(bytes: &[u8], at: usize, first: u16) -> usize {
    let second = read_u16_le(bytes, at + 2).unwrap_or(0);
    let combined = (u32::from(first & 0x7fff) << 16) | u32::from(second);
    usize::try_from(combined).unwrap_or(0)
}

/// Parses a package chunk.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - Package chunk offset.
/// * `size` - Package chunk size.
/// * `table` - Output table.
fn parse_package(bytes: &[u8], start: usize, size: usize, table: &mut ArscTable) {
    table.packages.push(read_utf16_name(bytes, start + 12));
    let begin = start.saturating_add(header_size(bytes, start));
    collect_locales(bytes, begin, start.saturating_add(size), table);
}

/// Walks the configuration chunks of a package.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - First child offset.
/// * `end` - Package end offset.
/// * `table` - Output table.
fn collect_locales(bytes: &[u8], start: usize, end: usize, table: &mut ArscTable) {
    let mut pos = start;
    while pos + 8 <= end {
        let Some(next) = locale_chunk(bytes, pos, table) else {
            break;
        };
        pos = next;
    }
}

/// Parses one package child chunk.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Chunk offset.
/// * `table` - Output table.
///
/// # Returns
///
/// The next offset, or `None`.
fn locale_chunk(bytes: &[u8], pos: usize, table: &mut ArscTable) -> Option<usize> {
    let kind = read_u16_le(bytes, pos)?;
    let size = usize::try_from(read_u32_le(bytes, pos + 4)?).ok()?;
    if size < 8 {
        return None;
    }
    (kind == CONFIG_TYPE).then(|| record_locale(bytes, pos, table));
    Some(pos.saturating_add(size))
}

/// Records a configuration locale.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Configuration chunk offset.
/// * `table` - Output table.
fn record_locale(bytes: &[u8], pos: usize, table: &mut ArscTable) {
    let text = locale_text(bytes, pos + 8);
    if !text.is_empty() && !table.locales.iter().any(|l| l == &text) {
        table.locales.push(text);
    }
}

/// Formats a configuration locale.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - Language field offset.
///
/// # Returns
///
/// The `language-country` locale text.
fn locale_text(bytes: &[u8], at: usize) -> String {
    let language = ascii_field(bytes, at);
    let country = ascii_field(bytes, at + 2);
    format!("{language}-{country}")
}

/// Reads a two-byte ASCII field, stopping at a null.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - Field offset.
///
/// # Returns
///
/// The field text.
fn ascii_field(bytes: &[u8], at: usize) -> String {
    let raw = bytes.get(at..at + 2).unwrap_or_default();
    raw.iter().take_while(|&&b| b != 0).map(|&b| char::from(b)).collect()
}

/// Reads a 128-character UTF-16LE package name.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `at` - Name offset.
///
/// # Returns
///
/// The decoded name.
fn read_utf16_name(bytes: &[u8], at: usize) -> String {
    let raw = bytes.get(at..at + 256).unwrap_or(&[]);
    let units: Vec<u16> = (0..raw.len() / 2).map(|i| utf16_unit(raw, i)).collect();
    let name: Vec<u16> = units.into_iter().take_while(|&u| u != 0).collect();
    String::from_utf16_lossy(&name)
}

/// Reads one little-endian UTF-16 unit.
///
/// # Arguments
///
/// * `raw` - Source bytes.
/// * `i` - Unit index.
///
/// # Returns
///
/// The decoded unit.
fn utf16_unit(raw: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]])
}

/// Formats a resource table as a human-readable text block.
///
/// # Arguments
///
/// * `table` - Recovered table.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_arsc_summary(table: &ArscTable) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Android resource table");
    let _ = writeln!(out, "Packages: {}", table.package_count);
    let _ = writeln!(out, "Strings: {}", table.strings.len());
    let _ = writeln!(out, "Locales: {}", table.locales.len());
    append_packages(&mut out, table);
    out
}

/// Appends package name lines.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `table` - Recovered table.
fn append_packages(out: &mut String, table: &ArscTable) {
    for name in &table.packages {
        let _ = writeln!(out, "  package {name}");
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Encoded resource table header used by tests.
    const SAMPLE_HEADER: [u8; 12] = [
        0x02, 0x00, 0x0c, 0x00, 0x30, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    ];

    /// Encoded UTF-8 string pool used by tests.
    const SAMPLE_POOL: [u8; 36] = [
        0x01, 0x00, 0x1c, 0x00, 0x24, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x03, b'a', b'b', b'c',
    ];

    /// Builds a table header with a UTF-8 string pool.
    ///
    /// # Returns
    ///
    /// The encoded table bytes.
    fn sample_arsc() -> Vec<u8> {
        let mut data = SAMPLE_HEADER.to_vec();
        data.extend_from_slice(&SAMPLE_POOL);
        data.push(0);
        data
    }

    /// Verifies resource table detection.
    #[test]
    fn test_arsc_detection() {
        assert!(is_arsc(&sample_arsc()));
        assert!(!is_arsc(b"not a resource table"));
    }

    /// Verifies string pool extraction.
    #[test]
    fn test_arsc_strings() {
        let table = parse_arsc(&sample_arsc()).unwrap();
        assert_eq!(table.package_count, 1);
        assert_eq!(table.strings, vec!["abc".to_string()]);
    }

    /// Verifies summary formatting.
    #[test]
    fn test_arsc_summary() {
        let table = parse_arsc(&sample_arsc()).unwrap();
        assert!(format_arsc_summary(&table).contains("resource table"));
    }
}
