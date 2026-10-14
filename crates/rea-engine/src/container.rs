//! Firmware Container Header Decoding
//!
//! **File:** `container.rs`
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
use std::collections::BTreeMap;

// Import dependencies from std
use std::fmt::Write as _;

/// Offset of the descriptor record table in a container header.
const RECORD_TABLE: usize = 0x10;

/// Maximum descriptor records parsed from a container header.
const MAX_RECORDS: usize = 32;

/// Start of the pointer window scanned for the runtime load base.
const BASE_WINDOW_START: usize = 0x100;

/// End of the pointer window scanned for the runtime load base.
const BASE_WINDOW_END: usize = 0x200;

/// A parsed firmware container header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Container {
    /// Header checksum or version word.
    pub checksum: u32,

    /// Entry point expressed as a file offset.
    pub entry: u32,

    /// Declared total image size in bytes.
    pub size: u32,

    /// Descriptor records in header order.
    pub records: Vec<Record>,

    /// Deduced runtime load base, when determinable.
    pub base: Option<u32>,
}

/// One container descriptor record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    /// Record kind discriminator.
    pub kind: u8,

    /// File offset of the record payload.
    pub offset: u32,
}

/// Parses a firmware container header from raw image bytes.
///
/// Detects the flash-container layout used by embedded storage firmware, whose
/// header carries an entry point and a table of `{kind, offset}` records
/// terminated by a record whose offset equals the total image size.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
///
/// # Returns
///
/// The parsed container, or `None` when the header does not match.
#[must_use]
pub fn parse(bytes: &[u8]) -> Option<Container> {
    let size = header_size(bytes)?;
    let records = records(bytes, size)?;
    let checksum = read_u32(bytes, 0)?;
    let entry = read_u32(bytes, 4)?;
    Some(Container { checksum, entry, size, records, base: load_base(bytes, size) })
}

/// Validates the container size and returns it as a `u32`.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
///
/// # Returns
///
/// The declared size, or `None`.
fn header_size(bytes: &[u8]) -> Option<u32> {
    let size = u32::try_from(bytes.len()).ok()?;
    (bytes.len() >= RECORD_TABLE && bytes.len().is_power_of_two()).then_some(size)
}

/// Parses the descriptor records up to the size terminator.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
/// * `size` - Declared image size.
///
/// # Returns
///
/// The records, or `None`.
fn records(bytes: &[u8], size: u32) -> Option<Vec<Record>> {
    let words = (0..MAX_RECORDS).map_while(|step| read_u32(bytes, RECORD_TABLE + step * 4));
    let out: Vec<Record> = words.take_while(|word| *word & 0xff != 0).map(record).collect();
    let end = read_u32(bytes, RECORD_TABLE + out.len() * 4)?;
    (end >> 8 == size && !out.is_empty()).then_some(out)
}

/// Splits a packed record word into its kind and offset.
///
/// # Arguments
///
/// * `word` - Packed `{kind, offset}` word.
///
/// # Returns
///
/// The decoded record.
fn record(word: u32) -> Record {
    Record { kind: u8::try_from(word & 0xff).unwrap_or(0), offset: word >> 8 }
}

/// Deduces the runtime load base from pointer literals in the header region.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
/// * `size` - Declared image size.
///
/// # Returns
///
/// The most common load base, when at least two pointers agree.
fn load_base(bytes: &[u8], size: u32) -> Option<u32> {
    let pointers = (BASE_WINDOW_START..BASE_WINDOW_END).step_by(4);
    let candidates = pointers.filter_map(|offset| pointer(bytes, size, offset));
    mode(candidates)
}

/// Extracts a candidate load base from one pointer word.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
/// * `size` - Declared image size.
/// * `offset` - Word offset.
///
/// # Returns
///
/// A candidate base, or `None`.
fn pointer(bytes: &[u8], size: u32, offset: usize) -> Option<u32> {
    let word = read_u32(bytes, offset)?;
    ((word >> 24) == 0x04 && (word & 0x00ff_ffff) < size).then_some(word & 0xff00_0000)
}

/// Returns the most frequent candidate that occurs at least twice.
///
/// # Arguments
///
/// * `candidates` - Candidate base values.
///
/// # Returns
///
/// The dominant base, or `None`.
fn mode(candidates: impl Iterator<Item = u32>) -> Option<u32> {
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for base in candidates {
        *counts.entry(base).or_default() += 1;
    }
    dominant(&counts)
}

/// Returns the most frequent base that occurs at least twice.
///
/// # Arguments
///
/// * `counts` - Candidate base frequencies.
///
/// # Returns
///
/// The dominant base, or `None`.
fn dominant(counts: &BTreeMap<u32, usize>) -> Option<u32> {
    let best = counts.iter().max_by_key(|(_, count)| **count);
    best.filter(|(_, count)| **count >= 2).map(|(base, _)| *base)
}

/// Reads a little-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source bytes.
/// * `offset` - Word offset.
///
/// # Returns
///
/// The word, or `None`.
fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

/// Formats a container as a single summary line.
///
/// # Arguments
///
/// * `container` - Parsed container.
///
/// # Returns
///
/// The summary text.
#[must_use]
pub fn format_container(container: &Container) -> String {
    let mut out = format!("entry {:#x}, size {:#x}", container.entry, container.size);
    let _ = write!(
        out,
        ", load base {}, {} records",
        base_label(container),
        container.records.len()
    );
    out
}

/// Formats the load base or a placeholder.
///
/// # Arguments
///
/// * `container` - Parsed container.
///
/// # Returns
///
/// The base text.
fn base_label(container: &Container) -> String {
    container
        .base
        .map_or_else(|| "unknown".to_string(), |base| format!("{base:#x}"))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a synthetic container image.
    ///
    /// # Returns
    ///
    /// The container bytes.
    fn sample() -> Vec<u8> {
        let mut bytes = vec![0u8; 0x1000];
        put(&mut bytes, 0x04, 0x40);
        put(&mut bytes, 0x10, 0x0001_0002);
        put(&mut bytes, 0x14, 0x0002_0005);
        put(&mut bytes, 0x18, 0x0010_0000);
        put(&mut bytes, 0x100, 0x0400_0080);
        put(&mut bytes, 0x104, 0x0400_0200);
        bytes
    }

    /// Writes a little-endian word into a test buffer.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Destination buffer.
    /// * `offset` - Field offset.
    /// * `value` - Value to write.
    fn put(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Verifies parsing of a synthetic container header.
    #[test]
    fn test_parse_container() {
        let parsed = parse(&sample()).unwrap();
        assert_eq!(parsed.entry, 0x40);
        assert_eq!(parsed.size, 0x1000);
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.base, Some(0x0400_0000));
    }

    /// Verifies a flat image is not mistaken for a container.
    #[test]
    fn test_parse_rejects_flat_image() {
        assert!(parse(b"plain data").is_none());
        assert!(parse(&[0u8; 0x1000]).is_none());
    }

    /// Verifies container formatting.
    #[test]
    fn test_format_container() {
        let text = format_container(&parse(&sample()).unwrap());
        assert!(text.contains("load base 0x4000000"));
    }
}
