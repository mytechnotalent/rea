//! JavaScript Source Map Resolution
//!
//! **File:** `sourcemap.rs`
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

// Import dependencies from serde_json
use serde_json::Value;

/// Recovered source map inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceMap {
    /// Source map format version.
    pub version: i64,
    /// Generated file name.
    pub file: String,
    /// Original source paths.
    pub sources: Vec<String>,
    /// Original symbol names.
    pub names: Vec<String>,
    /// Encoded `VLQ` mappings string.
    pub mappings: String,
    /// Number of generated lines.
    pub generated_lines: usize,
    /// Number of mapping segments.
    pub segment_count: usize,
    /// Whether embedded source content is present.
    pub has_content: bool,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// A decoded mapping segment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceSegment {
    /// Zero-based generated line.
    pub generated_line: usize,
    /// Zero-based generated column.
    pub generated_column: usize,
    /// Resolved source path, when present.
    pub source: Option<String>,
    /// Zero-based original line.
    pub original_line: usize,
    /// Zero-based original column.
    pub original_column: usize,
    /// Resolved name, when present.
    pub name: Option<String>,
}

/// Most decoded mapping segments retained.
const MAX_SEGMENTS: usize = 8192;

/// Base64 alphabet used by `VLQ` mappings.
const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Cumulative `VLQ` decode state.
#[derive(Debug, Clone, Copy, Default)]
struct SegmentState {
    /// Current generated column.
    generated_column: i64,
    /// Current source index.
    source: i64,
    /// Current original line.
    original_line: i64,
    /// Current original column.
    original_column: i64,
    /// Current name index.
    name: i64,
}

/// Incremental base64-`VLQ` reader.
#[derive(Default)]
struct VlqReader {
    /// Decoded values.
    values: Vec<i64>,
    /// Bit shift of the current value.
    shift: u32,
    /// Accumulator of the current value.
    acc: i64,
}

impl VlqReader {
    /// Feeds one character and returns a completed value.
    ///
    /// # Arguments
    ///
    /// * `c` - Base64 digit.
    ///
    /// # Returns
    ///
    /// The decoded value when the digit terminates one.
    fn push(&mut self, c: u8) -> Option<i64> {
        let digit = b64_value(c)?;
        self.acc += (digit & 0x1f) << self.shift;
        self.shift += 5;
        if digit & 0x20 == 0 {
            return Some(self.take());
        }
        None
    }

    /// Completes the current value and resets the reader.
    ///
    /// # Returns
    ///
    /// The signed value.
    fn take(&mut self) -> i64 {
        let value = unsign(self.acc);
        self.acc = 0;
        self.shift = 0;
        value
    }
}

/// Reports whether a buffer is a JSON source map.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when both `mappings` and `sources` keys are present.
#[must_use]
pub fn is_source_map(bytes: &[u8]) -> bool {
    let parsed = serde_json::from_slice::<Value>(bytes);
    parsed.is_ok_and(|v| v.get("mappings").is_some() && v.get("sources").is_some())
}

/// Parses a source map from a buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered source map, or `None`.
#[must_use]
pub fn parse_sourcemap(bytes: &[u8]) -> Option<SourceMap> {
    if !is_source_map(bytes) {
        return None;
    }
    let value: Value = serde_json::from_slice(bytes).ok()?;
    Some(build_sourcemap(&value))
}

/// Builds a source map inventory from parsed JSON.
///
/// # Arguments
///
/// * `value` - Parsed JSON value.
///
/// # Returns
///
/// The recovered source map.
fn build_sourcemap(value: &Value) -> SourceMap {
    let mappings = value.get("mappings").and_then(Value::as_str).unwrap_or("");
    let (generated_lines, segment_count) = mapping_stats(mappings);
    let mut map = base_map(value, generated_lines, segment_count);
    map.file = string_of(value, "file");
    map.sources = string_list(value, "sources");
    map.names = string_list(value, "names");
    map.mappings = mappings.to_string();
    map
}

/// Builds the scalar fields of a source map.
///
/// # Arguments
///
/// * `value` - Parsed JSON value.
/// * `generated_lines` - Generated line count.
/// * `segment_count` - Segment count.
///
/// # Returns
///
/// A partially populated source map.
fn base_map(value: &Value, generated_lines: usize, segment_count: usize) -> SourceMap {
    SourceMap {
        version: value.get("version").and_then(Value::as_i64).unwrap_or(0),
        generated_lines,
        segment_count,
        has_content: value.get("sourcesContent").is_some(),
        ..Default::default()
    }
}

/// Reads a string field from parsed JSON.
///
/// # Arguments
///
/// * `value` - Parsed JSON value.
/// * `key` - Object key.
///
/// # Returns
///
/// The string value, or an empty string.
fn string_of(value: &Value, key: &str) -> String {
    let found = value.get(key).and_then(Value::as_str).unwrap_or_default();
    found.to_string()
}

/// Reads a string array field from parsed JSON.
///
/// # Arguments
///
/// * `value` - Parsed JSON value.
/// * `key` - Object key.
///
/// # Returns
///
/// The string items.
fn string_list(value: &Value, key: &str) -> Vec<String> {
    let items = value.get(key).and_then(Value::as_array);
    let mapped = items.map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect());
    mapped.unwrap_or_default()
}

/// Counts generated lines and mapping segments.
///
/// # Arguments
///
/// * `mappings` - Encoded `VLQ` mappings string.
///
/// # Returns
///
/// The line count and the non-empty segment count.
fn mapping_stats(mappings: &str) -> (usize, usize) {
    let lines = mappings.split(';').count();
    let segments = mappings.split([';', ',']).filter(|s| !s.is_empty()).count();
    (lines, segments)
}

/// Finds a `sourceMappingURL` directive in JavaScript text.
///
/// # Arguments
///
/// * `text` - JavaScript source bytes.
///
/// # Returns
///
/// The referenced URL, or `None`.
#[must_use]
pub fn find_source_map_url(text: &[u8]) -> Option<String> {
    let marker = b"sourceMappingURL=";
    let at = find_bytes(text, marker)? + marker.len();
    let tail = text.get(at..)?;
    let end = tail.iter().position(|&b| is_url_end(b)).unwrap_or(tail.len());
    Some(String::from_utf8_lossy(&tail[..end]).trim().to_string())
}

/// Reports whether a byte terminates a `sourceMappingURL` value.
///
/// # Arguments
///
/// * `byte` - Byte to classify.
///
/// # Returns
///
/// `true` for line breaks, spaces, and comment closers.
fn is_url_end(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r' | b' ' | b'*' | b'\t')
}

/// Finds a byte needle in a buffer.
///
/// # Arguments
///
/// * `haystack` - Buffer to search.
/// * `needle` - Byte sequence to locate.
///
/// # Returns
///
/// The offset, or `None`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Formats a source map as a human-readable text block.
///
/// # Arguments
///
/// * `map` - Recovered source map.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_sourcemap_summary(map: &SourceMap) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Source map version {}", map.version);
    let _ = writeln!(out, "Generated file: {}", map.file);
    append_sourcemap_counts(&mut out, map);
    out
}

/// Appends source and mapping counts.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `map` - Recovered source map.
fn append_sourcemap_counts(out: &mut String, map: &SourceMap) {
    let _ = writeln!(out, "Sources: {}", map.sources.len());
    let _ = writeln!(out, "Names: {}", map.names.len());
    let _ = writeln!(out, "Generated lines: {}", map.generated_lines);
    let _ = writeln!(out, "Segments: {}", map.segment_count);
    let _ = writeln!(out, "Embedded content: {}", map.has_content);
}

/// Decodes a base64 `VLQ` segment into its field values.
///
/// # Arguments
///
/// * `segment` - One comma-delimited segment.
///
/// # Returns
///
/// The decoded signed fields.
fn decode_segment(segment: &str) -> Vec<i64> {
    let mut reader = VlqReader::default();
    for &c in segment.as_bytes() {
        if let Some(value) = reader.push(c) {
            reader.values.push(value);
        }
    }
    reader.values
}

/// Resolves a base64 digit to its six-bit value.
///
/// # Arguments
///
/// * `c` - ASCII character.
///
/// # Returns
///
/// The value, or `None`.
fn b64_value(c: u8) -> Option<i64> {
    let position = BASE64.iter().position(|&b| b == c)?;
    i64::try_from(position).ok()
}

/// Converts a `VLQ` sign bit into a signed value.
///
/// # Arguments
///
/// * `value` - Unsigned accumulator.
///
/// # Returns
///
/// The signed value.
fn unsign(value: i64) -> i64 {
    let magnitude = value >> 1;
    if value & 1 == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// Converts a decoded index into a host index.
///
/// # Arguments
///
/// * `value` - Signed value.
///
/// # Returns
///
/// The value as `usize`, or zero.
fn to_usize(value: i64) -> usize {
    usize::try_from(value).unwrap_or(0)
}

/// Resolves a source path by index.
///
/// # Arguments
///
/// * `map` - Recovered source map.
/// * `index` - Source index.
///
/// # Returns
///
/// The path, or `None`.
fn source_at(map: &SourceMap, index: i64) -> Option<String> {
    let at = usize::try_from(index).ok()?;
    map.sources.get(at).cloned()
}

/// Resolves a symbol name by index.
///
/// # Arguments
///
/// * `map` - Recovered source map.
/// * `index` - Name index.
///
/// # Returns
///
/// The name, or `None`.
fn name_at(map: &SourceMap, index: i64) -> Option<String> {
    let at = usize::try_from(index).ok()?;
    map.names.get(at).cloned()
}

/// Streaming decoder for `VLQ` mappings.
struct Decoder<'a> {
    /// Cumulative decode state.
    state: SegmentState,
    /// Source map providing the tables.
    map: &'a SourceMap,
    /// Accumulated segments.
    out: Vec<SourceSegment>,
}

impl Decoder<'_> {
    /// Decodes every line of the mappings string.
    fn run(&mut self) {
        for (line, group) in self.map.mappings.split(';').enumerate() {
            self.state.generated_column = 0;
            self.line(group, line);
        }
    }

    /// Decodes one generated line.
    ///
    /// # Arguments
    ///
    /// * `group` - Semicolon-delimited line group.
    /// * `line` - Zero-based generated line.
    fn line(&mut self, group: &str, line: usize) {
        for segment in group.split(',') {
            if self.out.len() >= MAX_SEGMENTS {
                return;
            }
            if let Some(seg) = self.segment(segment, line) {
                self.out.push(seg);
            }
        }
    }

    /// Decodes one segment.
    ///
    /// # Arguments
    ///
    /// * `segment` - Comma-delimited segment.
    /// * `line` - Zero-based generated line.
    ///
    /// # Returns
    ///
    /// The decoded segment, or `None`.
    fn segment(&mut self, segment: &str, line: usize) -> Option<SourceSegment> {
        if segment.is_empty() {
            return None;
        }
        let fields = decode_segment(segment);
        Some(self.apply(&fields, line))
    }

    /// Applies a segment's field deltas.
    ///
    /// # Arguments
    ///
    /// * `fields` - Decoded fields.
    /// * `line` - Zero-based generated line.
    ///
    /// # Returns
    ///
    /// The decoded segment.
    fn apply(&mut self, fields: &[i64], line: usize) -> SourceSegment {
        self.state.generated_column += fields[0];
        if fields.len() >= 4 {
            self.original(fields);
        }
        if fields.len() >= 5 {
            self.state.name += fields[4];
        }
        self.finish(fields.len(), line)
    }

    /// Applies the original-position deltas.
    ///
    /// # Arguments
    ///
    /// * `fields` - Decoded fields.
    fn original(&mut self, fields: &[i64]) {
        self.state.source += fields[1];
        self.state.original_line += fields[2];
        self.state.original_column += fields[3];
    }

    /// Builds a segment from the current state.
    ///
    /// # Arguments
    ///
    /// * `field_count` - Number of decoded fields.
    /// * `line` - Zero-based generated line.
    ///
    /// # Returns
    ///
    /// The decoded segment.
    fn finish(&self, field_count: usize, line: usize) -> SourceSegment {
        SourceSegment {
            generated_line: line,
            generated_column: to_usize(self.state.generated_column),
            source: source_at(self.map, self.state.source),
            original_line: to_usize(self.state.original_line),
            original_column: to_usize(self.state.original_column),
            name: self.name(field_count),
        }
    }

    /// Resolves the optional name for a segment.
    ///
    /// # Arguments
    ///
    /// * `field_count` - Number of decoded fields.
    ///
    /// # Returns
    ///
    /// The name, or `None`.
    fn name(&self, field_count: usize) -> Option<String> {
        if field_count >= 5 {
            return name_at(self.map, self.state.name);
        }
        None
    }
}

/// Decodes a source map's `VLQ` mappings into positioned segments.
///
/// # Arguments
///
/// * `map` - Recovered source map.
///
/// # Returns
///
/// The decoded segments, capped at [`MAX_SEGMENTS`].
#[must_use]
pub fn decode_mappings(map: &SourceMap) -> Vec<SourceSegment> {
    let mut decoder = Decoder { state: SegmentState::default(), map, out: Vec::new() };
    decoder.run();
    decoder.out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal source map JSON buffer.
    ///
    /// # Returns
    ///
    /// The encoded JSON bytes.
    fn sample_map() -> Vec<u8> {
        let json = r#"{"version":3,"file":"app.js","sources":["a.ts","b.ts"],"names":["foo"],"mappings":"AAAA,CAAC;AACA"}"#;
        json.as_bytes().to_vec()
    }

    /// Verifies source map detection.
    #[test]
    fn test_sourcemap_detection() {
        assert!(is_source_map(&sample_map()));
        assert!(!is_source_map(b"{\"version\":3}"));
    }

    /// Verifies source and name extraction.
    #[test]
    fn test_sourcemap_parse() {
        let map = parse_sourcemap(&sample_map()).unwrap();
        assert_eq!(map.version, 3);
        assert_eq!(map.sources, vec!["a.ts".to_string(), "b.ts".to_string()]);
        assert_eq!(map.names, vec!["foo".to_string()]);
        assert_eq!(map.generated_lines, 2);
    }

    /// Verifies mapping statistics.
    #[test]
    fn test_sourcemap_stats() {
        assert_eq!(mapping_stats("AAAA,CAAC;AACA"), (2, 3));
    }

    /// Verifies the source map URL directive.
    #[test]
    fn test_sourcemap_url() {
        let js = b"var x=1;\n//# sourceMappingURL=app.js.map\n";
        assert_eq!(find_source_map_url(js), Some("app.js.map".to_string()));
    }

    /// Verifies summary formatting.
    #[test]
    fn test_sourcemap_summary() {
        let map = parse_sourcemap(&sample_map()).unwrap();
        assert!(format_sourcemap_summary(&map).contains("app.js"));
    }

    /// Verifies `VLQ` mapping decoding.
    #[test]
    fn test_sourcemap_decode() {
        let json = r#"{"version":3,"sources":["a.ts"],"names":[],"mappings":"AAAA"}"#;
        let map = parse_sourcemap(json.as_bytes()).unwrap();
        let segments = decode_mappings(&map);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].source, Some("a.ts".to_string()));
        assert_eq!(segments[0].generated_line, 0);
    }
}
