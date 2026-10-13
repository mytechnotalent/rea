//! Apple Property List and Application Bundle Analysis
//!
//! **File:** `plist.rs`
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

/// Binary property list magic.
pub const BPLIST_MAGIC: &[u8; 8] = b"bplist00";

/// A decoded property list value.
#[derive(Debug, Clone, PartialEq)]
enum PlistValue {
    /// A dictionary of key-value pairs.
    Dict(Vec<(String, PlistValue)>),
    /// An array of values.
    Array(Vec<PlistValue>),
    /// A string value.
    Str(String),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Real(f64),
    /// A boolean value.
    Bool(bool),
    /// An opaque data value with its length.
    Data(usize),
    /// A null value.
    Null,
}

/// A dictionary key and its decoded value.
type DictItem = (String, PlistValue);

/// A decoded dictionary.
type Dict = Vec<DictItem>;

/// Binary plist trailer fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Bplist {
    /// Offset table entry width.
    int_size: usize,
    /// Object reference width.
    ref_size: usize,
    /// Number of objects.
    count: usize,
    /// Index of the top object.
    top: usize,
    /// Offset of the offset table.
    table_off: usize,
}

/// A decoded property list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plist {
    /// Source format label.
    pub format: String,
    /// Flattened top-level key-value entries.
    pub entries: Vec<(String, String)>,
}

/// Recovered application bundle metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BundleInfo {
    /// Bundle identifier.
    pub identifier: String,
    /// Bundle display name.
    pub name: String,
    /// Bundle short version.
    pub version: String,
    /// Executable file name.
    pub executable: String,
    /// Minimum operating system version.
    pub minimum_os: String,
    /// Target platform name.
    pub platform: String,
}

/// Reports whether a buffer is a binary property list.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the magic matches.
#[must_use]
pub fn is_binary_plist(bytes: &[u8]) -> bool {
    bytes.starts_with(BPLIST_MAGIC)
}

/// Reports whether a buffer is an XML property list.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when a `<plist` element is present near the start.
#[must_use]
pub fn is_xml_plist(bytes: &[u8]) -> bool {
    let head = bytes.get(..bytes.len().min(512)).unwrap_or(bytes);
    find_bytes(head, b"<plist").is_some()
}

/// Reports whether a buffer is a property list of either encoding.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` for binary or XML property lists.
#[must_use]
pub fn is_plist(bytes: &[u8]) -> bool {
    is_binary_plist(bytes) || is_xml_plist(bytes)
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

/// Reads a big-endian unsigned integer.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `pos` - Offset of the integer.
/// * `size` - Width in bytes.
///
/// # Returns
///
/// The decoded value.
fn read_uint(bytes: &[u8], pos: usize, size: usize) -> u64 {
    let mut value = 0u64;
    for i in 0..size.min(8) {
        let byte = bytes.get(pos + i).copied().unwrap_or(0);
        value = (value << 8) | u64::from(byte);
    }
    value
}

/// Parses a property list from a buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The decoded list, or `None` when unrecognized.
#[must_use]
pub fn parse_plist(bytes: &[u8]) -> Option<Plist> {
    if is_binary_plist(bytes) {
        return parse_binary(bytes);
    }
    if is_xml_plist(bytes) {
        return Some(parse_xml(bytes));
    }
    None
}

/// Parses a binary property list.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The decoded list, or `None`.
fn parse_binary(bytes: &[u8]) -> Option<Plist> {
    let trailer = bytes.get(bytes.len().checked_sub(32)?..)?;
    let table = read_trailer(trailer)?;
    let value = object_at(bytes, &table, table.top)?;
    Some(Plist { format: "binary".to_string(), entries: flatten_dict(&value) })
}

/// Reads the binary plist trailer.
///
/// # Arguments
///
/// * `trailer` - Trailing 32 bytes.
///
/// # Returns
///
/// The trailer fields, or `None`.
fn read_trailer(trailer: &[u8]) -> Option<Bplist> {
    let int_size = usize::from(*trailer.get(6)?);
    let ref_size = usize::from(*trailer.get(7)?);
    let count = usize::try_from(read_uint(trailer, 8, 8)).ok()?;
    let top = usize::try_from(read_uint(trailer, 16, 8)).ok()?;
    let table_off = usize::try_from(read_uint(trailer, 24, 8)).ok()?;
    Some(Bplist { int_size, ref_size, count, top, table_off })
}

/// Resolves the offset of an object.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `index` - Object index.
///
/// # Returns
///
/// The object offset, or `None`.
fn offset_of(bytes: &[u8], table: &Bplist, index: usize) -> Option<usize> {
    let at = table.table_off.checked_add(index.checked_mul(table.int_size)?)?;
    usize::try_from(read_uint(bytes, at, table.int_size)).ok()
}

/// Decodes an object by index.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `index` - Object index.
///
/// # Returns
///
/// The decoded value, or `None`.
fn object_at(bytes: &[u8], table: &Bplist, index: usize) -> Option<PlistValue> {
    let offset = offset_of(bytes, table, index)?;
    let marker = *bytes.get(offset)?;
    decode_object(bytes, table, marker, offset)
}

/// Decodes an object at an offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The decoded value, or `None`.
fn decode_object(bytes: &[u8], table: &Bplist, marker: u8, offset: usize) -> Option<PlistValue> {
    match marker >> 4 {
        0xd => container_value(bytes, table, marker, offset, true),
        0xa | 0xc => container_value(bytes, table, marker, offset, false),
        _ => decode_scalar(bytes, marker, offset),
    }
}

/// Decodes a dictionary or array object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
/// * `dict` - Whether the object is a dictionary.
///
/// # Returns
///
/// The decoded value, or `None`.
fn container_value(
    bytes: &[u8],
    table: &Bplist,
    marker: u8,
    offset: usize,
    dict: bool,
) -> Option<PlistValue> {
    if dict {
        let pairs = dict_value(bytes, table, marker, offset)?;
        return Some(PlistValue::Dict(pairs));
    }
    let items = array_value(bytes, table, marker, offset)?;
    Some(PlistValue::Array(items))
}

/// Decodes a scalar object.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The decoded value, or `None`.
fn decode_scalar(bytes: &[u8], marker: u8, offset: usize) -> Option<PlistValue> {
    match marker >> 4 {
        0x0 => Some(PlistValue::Bool(marker == 0x09)),
        0x1 => Some(PlistValue::Int(int_value(bytes, marker, offset))),
        0x2 => Some(PlistValue::Real(real_value(bytes, marker, offset))),
        0x5 | 0x6 => Some(PlistValue::Str(string_value(bytes, marker, offset)?)),
        0x4 => Some(PlistValue::Data(data_len(bytes, marker, offset))),
        _ => Some(PlistValue::Null),
    }
}

/// Reads an integer object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The integer value.
fn int_value(bytes: &[u8], marker: u8, offset: usize) -> i64 {
    let size = 1usize << (marker & 0x0f);
    let raw = read_uint(bytes, offset + 1, size);
    i64::from_ne_bytes(raw.to_ne_bytes())
}

/// Reads a floating-point object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The floating-point value.
fn real_value(bytes: &[u8], marker: u8, offset: usize) -> f64 {
    let size = 1usize << (marker & 0x0f);
    if size == 4 {
        let bits = u32::try_from(read_uint(bytes, offset + 1, 4)).unwrap_or(0);
        f64::from(f32::from_bits(bits))
    } else {
        f64::from_bits(read_uint(bytes, offset + 1, 8))
    }
}

/// Reads the length prefix of a sized object.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `offset` - Object offset.
/// * `marker` - Object marker byte.
///
/// # Returns
///
/// The element count and the data offset.
fn read_size(bytes: &[u8], offset: usize, marker: u8) -> (usize, usize) {
    let low = marker & 0x0f;
    if low != 0x0f {
        return (usize::from(low), offset + 1);
    }
    let m = bytes.get(offset + 1).copied().unwrap_or(0);
    let size = 1usize << (m & 0x0f);
    let count = usize::try_from(read_uint(bytes, offset + 2, size)).unwrap_or(0);
    (count, offset + 2 + size)
}

/// Resolves the byte width of one character of a string object.
///
/// # Arguments
///
/// * `marker` - Object marker byte.
///
/// # Returns
///
/// Two for UTF-16 strings, otherwise one.
fn char_width(marker: u8) -> usize {
    if marker >> 4 == 0x6 {
        2
    } else {
        1
    }
}

/// Reads a string object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The decoded string, or `None`.
fn string_value(bytes: &[u8], marker: u8, offset: usize) -> Option<String> {
    let (size, pos) = read_size(bytes, offset, marker);
    let raw = bytes.get(pos..pos + size * char_width(marker))?;
    if marker >> 4 == 0x6 {
        Some(utf16_string(raw))
    } else {
        Some(String::from_utf8_lossy(raw).to_string())
    }
}

/// Decodes a big-endian UTF-16 string.
///
/// # Arguments
///
/// * `raw` - Encoded bytes.
///
/// # Returns
///
/// The decoded string.
fn utf16_string(raw: &[u8]) -> String {
    let pairs = (0..raw.len() / 2).map(|i| u16::from_be_bytes([raw[i * 2], raw[i * 2 + 1]]));
    String::from_utf16_lossy(&pairs.collect::<Vec<u16>>())
}

/// Reads a data object length.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The data length in bytes.
fn data_len(bytes: &[u8], marker: u8, offset: usize) -> usize {
    let (size, _pos) = read_size(bytes, offset, marker);
    size
}

/// Reads an object reference.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `pos` - Reference offset.
///
/// # Returns
///
/// The referenced value, or `None`.
fn ref_at(bytes: &[u8], table: &Bplist, pos: usize) -> Option<PlistValue> {
    let index = usize::try_from(read_uint(bytes, pos, table.ref_size)).ok()?;
    object_at(bytes, table, index)
}

/// Reads an array object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The decoded elements, or `None`.
fn array_value(bytes: &[u8], table: &Bplist, marker: u8, offset: usize) -> Option<Vec<PlistValue>> {
    let (size, pos) = read_size(bytes, offset, marker);
    (0..size).map(|i| ref_at(bytes, table, pos + i * table.ref_size)).collect()
}

/// Reads a dictionary object value.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `marker` - Object marker byte.
/// * `offset` - Object offset.
///
/// # Returns
///
/// The decoded pairs, or `None`.
fn dict_value(bytes: &[u8], table: &Bplist, marker: u8, offset: usize) -> Option<Dict> {
    let (size, pos) = read_size(bytes, offset, marker);
    let values = pos + size * table.ref_size;
    (0..size).map(|i| entry_at(bytes, table, pos, values, i)).collect()
}

/// Reads one dictionary entry by index.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `keys` - Key reference base offset.
/// * `values` - Value reference base offset.
/// * `i` - Entry index.
///
/// # Returns
///
/// The key and value, or `None`.
fn entry_at(
    bytes: &[u8],
    table: &Bplist,
    keys: usize,
    values: usize,
    i: usize,
) -> Option<DictItem> {
    let key_pos = keys + i * table.ref_size;
    let val_pos = values + i * table.ref_size;
    dict_entry(bytes, table, key_pos, val_pos)
}

/// Reads one dictionary entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `table` - Trailer fields.
/// * `key_pos` - Key reference offset.
/// * `val_pos` - Value reference offset.
///
/// # Returns
///
/// The key and value, or `None`.
fn dict_entry(bytes: &[u8], table: &Bplist, key_pos: usize, val_pos: usize) -> Option<DictItem> {
    let key = ref_at(bytes, table, key_pos)?;
    let value = ref_at(bytes, table, val_pos)?;
    Some((render(&key), value))
}

/// Renders a value as display text.
///
/// # Arguments
///
/// * `value` - Value to render.
///
/// # Returns
///
/// The rendered text.
fn render(value: &PlistValue) -> String {
    match value {
        PlistValue::Str(s) => s.clone(),
        PlistValue::Int(i) => i.to_string(),
        PlistValue::Real(r) => r.to_string(),
        PlistValue::Bool(b) => b.to_string(),
        _ => render_container(value),
    }
}

/// Renders a container value as display text.
///
/// # Arguments
///
/// * `value` - Value to render.
///
/// # Returns
///
/// The rendered text.
fn render_container(value: &PlistValue) -> String {
    match value {
        PlistValue::Data(n) => format!("<{n} bytes>"),
        PlistValue::Array(a) => format!("[{} items]", a.len()),
        PlistValue::Dict(d) => format!("{{{} keys}}", d.len()),
        _ => "null".to_string(),
    }
}

/// Flattens a top-level dictionary into string pairs.
///
/// # Arguments
///
/// * `value` - Decoded top-level value.
///
/// # Returns
///
/// The flattened key-value entries.
fn flatten_dict(value: &PlistValue) -> Vec<(String, String)> {
    match value {
        PlistValue::Dict(pairs) => pairs.iter().map(|(k, v)| (k.clone(), render(v))).collect(),
        _ => Vec::new(),
    }
}

/// Parses an XML property list.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The decoded list.
fn parse_xml(bytes: &[u8]) -> Plist {
    let text = String::from_utf8_lossy(bytes);
    Plist { format: "xml".to_string(), entries: xml_entries(&text) }
}

/// Extracts key-value entries from XML plist text.
///
/// # Arguments
///
/// * `text` - XML text.
///
/// # Returns
///
/// The extracted entries.
fn xml_entries(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some((key, after)) = next_key(rest) {
        let (value, tail) = next_value(after);
        out.push((key, value));
        rest = tail;
    }
    out
}

/// Reads the next `<key>` element.
///
/// # Arguments
///
/// * `text` - Remaining XML text.
///
/// # Returns
///
/// The key and the remaining text, or `None`.
fn next_key(text: &str) -> Option<(String, &str)> {
    let start = text.find("<key>")? + 5;
    let end = text[start..].find("</key>")? + start;
    Some((text[start..end].to_string(), &text[end + 6..]))
}

/// Reads the value element following a key.
///
/// # Arguments
///
/// * `text` - Remaining XML text.
///
/// # Returns
///
/// The rendered value and the remaining text.
fn next_value(text: &str) -> (String, &str) {
    let tags = ["string", "integer", "real", "date", "data"];
    for tag in tags {
        if let Some(found) = tagged_value(text, tag) {
            return found;
        }
    }
    (String::new(), text)
}

/// Reads a tagged XML value element.
///
/// # Arguments
///
/// * `text` - Remaining XML text.
/// * `tag` - Element tag name.
///
/// # Returns
///
/// The value and the remaining text, or `None`.
fn tagged_value<'a>(text: &'a str, tag: &str) -> Option<(String, &'a str)> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)? + start;
    Some((text[start..end].to_string(), &text[end + close.len()..]))
}

/// Looks up a key in a decoded list.
///
/// # Arguments
///
/// * `plist` - Decoded list.
/// * `key` - Key to find.
///
/// # Returns
///
/// The value text, or an empty string.
fn lookup(plist: &Plist, key: &str) -> String {
    plist
        .entries
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

/// Extracts application bundle metadata from a property list.
///
/// # Arguments
///
/// * `plist` - Decoded list.
///
/// # Returns
///
/// The recovered bundle metadata.
#[must_use]
pub fn analyze_bundle(plist: &Plist) -> BundleInfo {
    BundleInfo {
        identifier: lookup(plist, "CFBundleIdentifier"),
        name: lookup(plist, "CFBundleName"),
        version: lookup(plist, "CFBundleShortVersionString"),
        executable: lookup(plist, "CFBundleExecutable"),
        minimum_os: lookup(plist, "MinimumOSVersion"),
        platform: lookup(plist, "DTPlatformName"),
    }
}

/// Formats a property list as a human-readable text block.
///
/// # Arguments
///
/// * `plist` - Decoded list.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_plist_summary(plist: &Plist) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Property list: {}", plist.format);
    let _ = writeln!(out, "Keys: {}", plist.entries.len());
    for (key, value) in plist.entries.iter().take(40) {
        let _ = writeln!(out, "  {key} = {value}");
    }
    out
}

/// Formats bundle metadata as a human-readable text block.
///
/// # Arguments
///
/// * `bundle` - Recovered bundle metadata.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_bundle_summary(bundle: &BundleInfo) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Identifier: {}", bundle.identifier);
    let _ = writeln!(out, "Name: {}", bundle.name);
    let _ = writeln!(out, "Version: {}", bundle.version);
    let _ = writeln!(out, "Executable: {}", bundle.executable);
    let _ = writeln!(out, "Minimum OS: {}", bundle.minimum_os);
    let _ = writeln!(out, "Platform: {}", bundle.platform);
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies property list detection.
    #[test]
    fn test_plist_detection() {
        let xml = b"<?xml version=\"1.0\"?><plist version=\"1.0\">";
        assert!(is_binary_plist(b"bplist00"));
        assert!(is_xml_plist(xml));
        assert!(is_plist(b"bplist00"));
        assert!(!is_plist(b"MZ"));
    }

    /// Verifies XML plist extraction.
    #[test]
    fn test_plist_xml() {
        let xml = "<plist><dict><key>CFBundleName</key><string>Notes</string></dict></plist>";
        let plist = parse_plist(xml.as_bytes()).unwrap();
        let first = ("CFBundleName".to_string(), "Notes".to_string());
        assert_eq!(plist.format, "xml");
        assert_eq!(plist.entries[0], first);
    }

    /// Verifies bundle metadata extraction.
    #[test]
    fn test_plist_bundle() {
        let plist = Plist { format: "xml".to_string(), entries: bundle_entries() };
        let bundle = analyze_bundle(&plist);
        assert_eq!(bundle.identifier, "com.example.app");
        assert_eq!(bundle.minimum_os, "14.0");
    }

    /// Builds bundle metadata entries.
    ///
    /// # Returns
    ///
    /// Two identifier and OS entries.
    fn bundle_entries() -> Vec<(String, String)> {
        let identifier = (
            "CFBundleIdentifier".to_string(),
            "com.example.app".to_string(),
        );
        let minimum = ("MinimumOSVersion".to_string(), "14.0".to_string());
        vec![identifier, minimum]
    }

    /// Verifies that a binary plist round-trips a dictionary.
    #[test]
    fn test_plist_binary() {
        let data = binary_plist();
        let plist = parse_plist(&data).unwrap();
        assert_eq!(plist.entries[0], ("k".to_string(), "v".to_string()));
    }

    /// Builds a two-object binary plist.
    ///
    /// # Returns
    ///
    /// The encoded module bytes.
    fn binary_plist() -> Vec<u8> {
        let mut data = BPLIST_MAGIC.to_vec();
        data.extend_from_slice(&[0xd1, 0x01, 0x02, 0x51, 0x6b, 0x51, 0x76]);
        data.extend_from_slice(&[0x08, 0x0b, 0x0d]);
        data.extend_from_slice(&trailer_bytes());
        data
    }

    /// Builds a binary plist trailer.
    ///
    /// # Returns
    ///
    /// The 32-byte trailer.
    fn trailer_bytes() -> Vec<u8> {
        let mut trailer = vec![0u8; 32];
        trailer[6] = 1;
        trailer[7] = 1;
        trailer[15] = 3;
        trailer[31] = 15;
        trailer
    }

    /// Verifies summary formatting.
    #[test]
    fn test_plist_summary() {
        let plist = Plist { format: "xml".into(), entries: vec![("a".into(), "b".into())] };
        assert!(format_plist_summary(&plist).contains("a = b"));
    }
}
