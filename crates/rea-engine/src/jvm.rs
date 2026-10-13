//! Java Class File and JAR Parsing
//!
//! **File:** `jvm.rs`
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

/// Java class file magic.
pub const CLASS_MAGIC: [u8; 4] = [0xca, 0xfe, 0xba, 0xbe];

/// A single constant-pool entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum CpEntry {
    /// A UTF-8 string constant.
    Utf8(String),
    /// A class reference naming another pool entry.
    ClassRef(u16),
    /// A name-and-type descriptor pair.
    NameAndType(u16, u16),
    /// Any entry this parser does not model.
    #[default]
    Other,
}

/// A declared field or method.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JavaMember {
    /// Member name.
    pub name: String,
    /// JVM descriptor.
    pub descriptor: String,
    /// Access flags.
    pub access: u16,
}

/// Recovered Java class inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JavaClass {
    /// Class file minor version.
    pub minor: u16,
    /// Class file major version.
    pub major: u16,
    /// Class access flags.
    pub access: u16,
    /// Fully qualified this-class name.
    pub this_class: String,
    /// Fully qualified super-class name.
    pub super_class: String,
    /// Implemented interface names.
    pub interfaces: Vec<String>,
    /// Declared fields.
    pub fields: Vec<JavaMember>,
    /// Declared methods.
    pub methods: Vec<JavaMember>,
    /// Every UTF-8 constant in the pool.
    pub strings: Vec<String>,
}

/// Reports whether a buffer is a Java class file.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the magic matches.
#[must_use]
pub fn is_class(bytes: &[u8]) -> bool {
    bytes.starts_with(&CLASS_MAGIC)
}

/// Reads a big-endian 16-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, or `None`.
fn be_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let b = bytes.get(offset..offset + 2)?;
    Some(u16::from_be_bytes([b[0], b[1]]))
}

/// Reads a big-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, or `None`.
fn be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset + 4)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Reads the size of a non-modeled constant-pool entry.
///
/// # Arguments
///
/// * `tag` - Constant-pool tag.
///
/// # Returns
///
/// The encoded size in bytes, excluding the tag.
fn skip_size(tag: u8) -> usize {
    match tag {
        3 | 4 | 9 | 10 | 11 | 17 | 18 => 5,
        8 | 16 | 19 | 20 => 3,
        15 => 4,
        _ => 1,
    }
}

/// Reads a UTF-8 constant-pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the length prefix.
///
/// # Returns
///
/// The entry, next offset, and whether it spans two slots.
fn read_utf8(bytes: &[u8], pos: usize) -> (CpEntry, usize, bool) {
    let len = usize::from(be_u16(bytes, pos).unwrap_or(0));
    let start = pos + 2;
    let end = start.saturating_add(len);
    let text = bytes.get(start..end).map(|s| String::from_utf8_lossy(s).to_string());
    (CpEntry::Utf8(text.unwrap_or_default()), end, false)
}

/// Reads a name-and-type constant-pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the two indices.
///
/// # Returns
///
/// The entry, next offset, and whether it spans two slots.
fn read_name_and_type(bytes: &[u8], pos: usize) -> (CpEntry, usize, bool) {
    let name = be_u16(bytes, pos).unwrap_or(0);
    let desc = be_u16(bytes, pos + 2).unwrap_or(0);
    (CpEntry::NameAndType(name, desc), pos + 4, false)
}

/// Reads one constant-pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the entry tag.
///
/// # Returns
///
/// The entry, next offset, and whether it spans two slots.
fn read_cp_entry(bytes: &[u8], pos: usize) -> Option<(CpEntry, usize, bool)> {
    let tag = *bytes.get(pos)?;
    Some(match tag {
        1 => read_utf8(bytes, pos + 1),
        7 => read_class_ref(bytes, pos + 1),
        12 => read_name_and_type(bytes, pos + 1),
        _ => skip_entry(pos, tag),
    })
}

/// Builds an entry for a constant-pool tag that is skipped.
///
/// # Arguments
///
/// * `pos` - Offset of the entry tag.
/// * `tag` - Constant-pool tag.
///
/// # Returns
///
/// The entry, next offset, and whether it spans two slots.
fn skip_entry(pos: usize, tag: u8) -> (CpEntry, usize, bool) {
    let next = pos + 1 + skip_size(tag);
    (CpEntry::Other, next, tag == 5 || tag == 6)
}

/// Reads a class-reference constant-pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the name index.
///
/// # Returns
///
/// The entry, next offset, and whether it spans two slots.
fn read_class_ref(bytes: &[u8], pos: usize) -> (CpEntry, usize, bool) {
    let name_index = be_u16(bytes, pos).unwrap_or(0);
    (CpEntry::ClassRef(name_index), pos + 2, false)
}

/// Advances a constant-pool index past a wide entry.
///
/// # Arguments
///
/// * `index` - Current one-based index.
/// * `wide` - Whether the entry spans two slots.
///
/// # Returns
///
/// The next index.
fn advance(index: usize, wide: bool) -> usize {
    index + if wide { 2 } else { 1 }
}

/// Appends one constant-pool entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the entry tag.
/// * `pool` - Accumulated entries.
/// * `index` - One-based index, advanced past the entry.
///
/// # Returns
///
/// The next entry offset.
fn push_entry(bytes: &[u8], pos: usize, pool: &mut Vec<CpEntry>, index: &mut usize) -> usize {
    let Some((entry, next, wide)) = read_cp_entry(bytes, pos) else {
        return bytes.len();
    };
    pool.push(entry);
    *index = advance(*index, wide);
    next
}

/// Parses the constant pool.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the first entry.
/// * `count` - Declared pool count.
///
/// # Returns
///
/// The entries and the offset following the pool.
fn parse_pool(bytes: &[u8], mut pos: usize, count: usize) -> (Vec<CpEntry>, usize) {
    let mut pool = Vec::new();
    let mut index = 1;
    while index < count && pos < bytes.len() {
        pos = push_entry(bytes, pos, &mut pool, &mut index);
    }
    (pool, pos)
}

/// Resolves a UTF-8 constant by index.
///
/// # Arguments
///
/// * `pool` - Constant pool.
/// * `index` - One-based pool index.
///
/// # Returns
///
/// The string, or an empty string.
fn utf8(pool: &[CpEntry], index: u16) -> String {
    let slot = usize::from(index).checked_sub(1);
    match slot.and_then(|i| pool.get(i)) {
        Some(CpEntry::Utf8(text)) => text.clone(),
        _ => String::new(),
    }
}

/// Resolves a class name by index.
///
/// # Arguments
///
/// * `pool` - Constant pool.
/// * `index` - One-based pool index.
///
/// # Returns
///
/// The class name, or an empty string.
fn class_name(pool: &[CpEntry], index: u16) -> String {
    let slot = usize::from(index).checked_sub(1).and_then(|i| pool.get(i));
    if let Some(CpEntry::ClassRef(name_index)) = slot {
        return utf8(pool, *name_index);
    }
    String::new()
}

/// Collects every UTF-8 constant.
///
/// # Arguments
///
/// * `pool` - Constant pool.
///
/// # Returns
///
/// All string constants.
fn pool_strings(pool: &[CpEntry]) -> Vec<String> {
    let strings = pool.iter().filter_map(|e| match e {
        CpEntry::Utf8(text) => Some(text.clone()),
        _ => None,
    });
    strings.collect()
}

/// Skips a member attribute table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the first attribute.
/// * `count` - Attribute count.
///
/// # Returns
///
/// The offset past the attributes, or `None`.
fn skip_attributes(bytes: &[u8], mut pos: usize, count: usize) -> Option<usize> {
    for _ in 0..count {
        let len = usize::try_from(be_u32(bytes, pos + 2)?).ok()?;
        pos += 6 + len;
    }
    Some(pos)
}

/// Reads one field or method.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the member.
/// * `pool` - Constant pool.
///
/// # Returns
///
/// The member and the next offset, or `None`.
fn read_member(bytes: &[u8], pos: usize, pool: &[CpEntry]) -> Option<(JavaMember, usize)> {
    let access = be_u16(bytes, pos)?;
    let name = utf8(pool, be_u16(bytes, pos + 2)?);
    let descriptor = utf8(pool, be_u16(bytes, pos + 4)?);
    let attrs = usize::from(be_u16(bytes, pos + 6)?);
    let end = skip_attributes(bytes, pos + 8, attrs)?;
    Some((JavaMember { name, descriptor, access }, end))
}

/// Appends one field or method at a cursor.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `cursor` - Cursor advanced past the member.
/// * `pool` - Constant pool.
/// * `members` - Accumulated members.
///
/// # Returns
///
/// `true` when a member was read.
fn push_member(
    bytes: &[u8],
    cursor: &mut usize,
    pool: &[CpEntry],
    members: &mut Vec<JavaMember>,
) -> bool {
    let Some((member, next)) = read_member(bytes, *cursor, pool) else {
        return false;
    };
    members.push(member);
    *cursor = next;
    true
}

/// Parses a field or method table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the count.
/// * `pool` - Constant pool.
///
/// # Returns
///
/// The members and the offset following the table.
fn parse_members(bytes: &[u8], pos: usize, pool: &[CpEntry]) -> (Vec<JavaMember>, usize) {
    let count = usize::from(be_u16(bytes, pos).unwrap_or(0));
    let mut cursor = pos + 2;
    let mut members = Vec::new();
    while members.len() < count && push_member(bytes, &mut cursor, pool, &mut members) {}
    (members, cursor)
}

/// Parses the implemented-interface table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the count.
/// * `pool` - Constant pool.
///
/// # Returns
///
/// The class names and the offset following the table.
fn parse_interfaces(bytes: &[u8], pos: usize, pool: &[CpEntry]) -> (Vec<String>, usize) {
    let count = usize::from(be_u16(bytes, pos).unwrap_or(0));
    let mut names = Vec::new();
    for i in 0..count {
        let index = be_u16(bytes, pos + 2 + i * 2).unwrap_or(0);
        names.push(class_name(pool, index));
    }
    (names, pos + 2 + count * 2)
}

/// Parses the interfaces, fields, and methods of a class.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the interface count.
/// * `pool` - Constant pool.
///
/// # Returns
///
/// Interfaces, fields, and methods.
fn parse_body(
    bytes: &[u8],
    pos: usize,
    pool: &[CpEntry],
) -> (Vec<String>, Vec<JavaMember>, Vec<JavaMember>) {
    let (interfaces, after_ifaces) = parse_interfaces(bytes, pos, pool);
    let (fields, after_fields) = parse_members(bytes, after_ifaces, pool);
    let (methods, _) = parse_members(bytes, after_fields, pool);
    (interfaces, fields, methods)
}

/// Builds the header fields of a class.
///
/// # Arguments
///
/// * `pool` - Constant pool.
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the access flags.
///
/// # Returns
///
/// A partially populated class.
fn head(pool: &[CpEntry], bytes: &[u8], pos: usize) -> JavaClass {
    JavaClass {
        access: be_u16(bytes, pos).unwrap_or(0),
        this_class: class_name(pool, be_u16(bytes, pos + 2).unwrap_or(0)),
        super_class: class_name(pool, be_u16(bytes, pos + 4).unwrap_or(0)),
        strings: pool_strings(pool),
        ..Default::default()
    }
}

/// Assembles a class inventory from its parts.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Offset of the access flags.
/// * `pool` - Constant pool.
/// * `major` - Major version.
/// * `minor` - Minor version.
///
/// # Returns
///
/// The recovered class.
fn assemble(bytes: &[u8], pos: usize, pool: &[CpEntry], major: u16, minor: u16) -> JavaClass {
    let (interfaces, fields, methods) = parse_body(bytes, pos + 6, pool);
    let mut class = head(pool, bytes, pos);
    class.major = major;
    class.minor = minor;
    class.interfaces = interfaces;
    class.fields = fields;
    class.methods = methods;
    class
}

/// Parses a Java class file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered class, or `None`.
#[must_use]
pub fn parse_class(bytes: &[u8]) -> Option<JavaClass> {
    if !is_class(bytes) {
        return None;
    }
    let count = usize::from(be_u16(bytes, 8)?);
    let (pool, pos) = parse_pool(bytes, 10, count);
    let major = be_u16(bytes, 6)?;
    let minor = be_u16(bytes, 4)?;
    Some(assemble(bytes, pos, &pool, major, minor))
}

/// Formats a Java class as a human-readable text block.
///
/// # Arguments
///
/// * `class` - Recovered class.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_java_summary(class: &JavaClass) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Java class {}", class.this_class);
    let _ = writeln!(out, "Version: {}.{}", class.major, class.minor);
    let _ = writeln!(out, "Super: {}", class.super_class);
    let _ = writeln!(out, "Interfaces: {}", class.interfaces.len());
    let _ = writeln!(out, "Fields: {}", class.fields.len());
    let _ = writeln!(out, "Methods: {}", class.methods.len());
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal class file with `this` and `super` names.
    ///
    /// # Returns
    ///
    /// The encoded class bytes.
    fn sample_class() -> Vec<u8> {
        let mut data = CLASS_MAGIC.to_vec();
        data.extend_from_slice(&[0, 0, 0, 52, 0, 5, 1, 0, 4]);
        data.extend_from_slice(b"Test");
        data.extend_from_slice(&[7, 0, 1, 1, 0, 16]);
        data.extend_from_slice(b"java/lang/Object");
        data.extend_from_slice(&[7, 0, 3, 0, 0x21, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0]);
        data
    }

    /// Verifies Java class detection.
    #[test]
    fn test_java_detection() {
        assert!(is_class(&CLASS_MAGIC));
        assert!(!is_class(b"MZ"));
    }

    /// Verifies this-class and super-class resolution.
    #[test]
    fn test_java_names() {
        let class = parse_class(&sample_class()).unwrap();
        assert_eq!(class.this_class, "Test");
        assert_eq!(class.super_class, "java/lang/Object");
        assert_eq!(class.major, 52);
        assert!(class.strings.contains(&"Test".to_string()));
    }

    /// Verifies summary formatting.
    #[test]
    fn test_java_summary() {
        let class = parse_class(&sample_class()).unwrap();
        assert!(format_java_summary(&class).contains("Java class Test"));
    }
}
