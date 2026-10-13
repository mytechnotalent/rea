//! Windows PDB (MSF) Container Parsing
//!
//! **File:** `pdb.rs`
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

/// MSF 7.0 superblock magic.
pub const MSF_MAGIC: &[u8; 32] = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0";

/// Recovered PDB container metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PdbInfo {
    /// PDB info stream version.
    pub version: u32,
    /// PDB age counter.
    pub age: u32,
    /// GUID as lowercase hexadecimal.
    pub guid: String,
    /// Number of streams in the container.
    pub stream_count: usize,
}

/// Reports whether a buffer is an MSF container.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the superblock magic matches.
#[must_use]
pub fn is_pdb(bytes: &[u8]) -> bool {
    bytes.starts_with(MSF_MAGIC)
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

/// Returns a page slice by number.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `page` - Page number.
///
/// # Returns
///
/// The page slice, or `None`.
fn page_span(bytes: &[u8], page_size: usize, page: u32) -> Option<&[u8]> {
    let start = usize::try_from(page).ok()?.checked_mul(page_size)?;
    let end = start.checked_add(page_size)?;
    bytes.get(start..end)
}

/// Computes the page count for a stream size.
///
/// # Arguments
///
/// * `size` - Stream size, or `0xFFFFFFFF` for an absent stream.
/// * `page_size` - Page size in bytes.
///
/// # Returns
///
/// The number of pages.
fn stream_pages(size: u32, page_size: usize) -> usize {
    if size == u32::MAX || page_size == 0 {
        return 0;
    }
    usize::try_from(size).unwrap_or(0).div_ceil(page_size)
}

/// Reads a list of little-endian words.
///
/// # Arguments
///
/// * `data` - Source slice.
/// * `count` - Word count.
///
/// # Returns
///
/// The decoded words.
fn read_list(data: &[u8], count: usize) -> Vec<u32> {
    (0..count).map(|i| read_u32_le(data, i * 4).unwrap_or(0)).collect()
}

/// Concatenates pages into a contiguous buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `pages` - Page numbers.
///
/// # Returns
///
/// The gathered bytes.
fn gather_pages(bytes: &[u8], page_size: usize, pages: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &page in pages {
        if let Some(span) = page_span(bytes, page_size, page) {
            out.extend_from_slice(span);
        }
    }
    out
}

/// Reads the MSF superblock fields.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The page size, directory size, and block map page.
fn superblock(bytes: &[u8]) -> Option<(usize, usize, usize)> {
    if !is_pdb(bytes) {
        return None;
    }
    let page_size = usize::try_from(read_u32_le(bytes, 32)?).ok()?;
    let directory_size = usize::try_from(read_u32_le(bytes, 44)?).ok()?;
    let block_map = usize::try_from(read_u32_le(bytes, 52)?).ok()?;
    Some((page_size, directory_size, block_map))
}

/// Reads the stream directory page list.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `block_map` - Offset of the block map page.
/// * `directory_size` - Directory size in bytes.
///
/// # Returns
///
/// The directory page numbers, or `None`.
fn dir_page_list(
    bytes: &[u8],
    page_size: usize,
    block_map: usize,
    directory_size: usize,
) -> Option<Vec<u32>> {
    let page = u32::try_from(block_map).ok()?;
    let span = page_span(bytes, page_size, page)?;
    let pages = stream_pages(u32::try_from(directory_size).ok()?, page_size);
    Some(read_list(span, pages))
}

/// Appends one stream's page list.
///
/// # Arguments
///
/// * `directory` - Directory bytes.
/// * `cursor` - Current offset.
/// * `size` - Stream size.
/// * `page_size` - Page size in bytes.
/// * `lists` - Accumulated page lists.
///
/// # Returns
///
/// The next offset.
fn push_stream_list(
    directory: &[u8],
    cursor: usize,
    size: u32,
    page_size: usize,
    lists: &mut Vec<Vec<u32>>,
) -> usize {
    let pages = stream_pages(size, page_size);
    lists.push(read_list(&directory[cursor..], pages));
    cursor + pages * 4
}

/// Reads the page list of every stream.
///
/// # Arguments
///
/// * `directory` - Directory bytes.
/// * `count` - Stream count.
/// * `page_size` - Page size in bytes.
///
/// # Returns
///
/// One page list per stream.
fn stream_lists(directory: &[u8], count: usize, page_size: usize) -> Vec<Vec<u32>> {
    let sizes = read_list(&directory[4..], count);
    let mut cursor = 4 + count * 4;
    let mut lists = Vec::new();
    for &size in &sizes {
        cursor = push_stream_list(directory, cursor, size, page_size, &mut lists);
    }
    lists
}

/// Formats a GUID as lowercase hexadecimal.
///
/// # Arguments
///
/// * `info` - PDB info stream.
/// * `at` - GUID offset.
///
/// # Returns
///
/// The hexadecimal GUID.
fn guid_text(info: &[u8], at: usize) -> String {
    let raw = info.get(at..at + 16).unwrap_or_default();
    let mut out = String::new();
    for byte in raw {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parses the PDB info stream.
///
/// # Arguments
///
/// * `info` - PDB info stream.
/// * `stream_count` - Number of streams.
///
/// # Returns
///
/// The recovered metadata.
fn parse_info(info: &[u8], stream_count: usize) -> PdbInfo {
    PdbInfo {
        version: read_u32_le(info, 0).unwrap_or(0),
        age: read_u32_le(info, 8).unwrap_or(0),
        guid: guid_text(info, 12),
        stream_count,
    }
}

/// Parses the stream directory and info stream.
///
/// # Arguments
///
/// * `directory` - Directory bytes.
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
///
/// # Returns
///
/// The recovered metadata, or `None`.
fn parse_directory(directory: &[u8], bytes: &[u8], page_size: usize) -> Option<PdbInfo> {
    let count = usize::try_from(read_u32_le(directory, 0)?).ok()?;
    let lists = stream_lists(directory, count, page_size);
    let info = lists
        .get(1)
        .map_or_else(Vec::new, |pages| gather_pages(bytes, page_size, pages));
    Some(parse_info(&info, count))
}

/// Parses an MSF container's PDB metadata.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered metadata, or `None`.
#[must_use]
pub fn parse_pdb(bytes: &[u8]) -> Option<PdbInfo> {
    let (page_size, directory_size, block_map) = superblock(bytes)?;
    let dir_pages = dir_page_list(bytes, page_size, block_map, directory_size)?;
    let directory = gather_pages(bytes, page_size, &dir_pages);
    parse_directory(&directory, bytes, page_size)
}

/// Formats PDB metadata as a human-readable text block.
///
/// # Arguments
///
/// * `pdb` - Recovered metadata.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_pdb_summary(pdb: &PdbInfo) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "PDB version {}", pdb.version);
    let _ = writeln!(out, "Age: {}", pdb.age);
    let _ = writeln!(out, "Streams: {}", pdb.stream_count);
    let _ = writeln!(out, "GUID: {}", pdb.guid);
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds the MSF superblock.
    ///
    /// # Returns
    ///
    /// The superblock bytes.
    fn super_block() -> Vec<u8> {
        let mut data = MSF_MAGIC.to_vec();
        data.extend_from_slice(&512u32.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data
    }

    /// Directory bytes used by tests.
    const DIRECTORY: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 3, 0, 0, 0];

    /// PDB info stream head used by tests.
    const INFO_HEAD: [u8; 12] = [1, 0, 0, 0, 90, 0x2f, 0x30, 0x01, 7, 0, 0, 0];

    /// Builds a minimal MSF container.
    ///
    /// # Returns
    ///
    /// The container bytes.
    fn sample_pdb() -> Vec<u8> {
        let mut data = super_block();
        data.resize(1024, 0);
        data[512..516].copy_from_slice(&2u32.to_le_bytes());
        data.resize(1536, 0);
        data[1024..1040].copy_from_slice(&DIRECTORY);
        data.resize(2048, 0);
        data[1536..1548].copy_from_slice(&INFO_HEAD);
        data
    }

    /// Verifies MSF detection.
    #[test]
    fn test_pdb_detection() {
        assert!(is_pdb(&sample_pdb()));
        assert!(!is_pdb(b"not a pdb"));
    }

    /// Verifies superblock and info stream recovery.
    #[test]
    fn test_pdb_info() {
        let pdb = parse_pdb(&sample_pdb()).unwrap();
        assert_eq!(pdb.version, 1);
        assert_eq!(pdb.age, 7);
        assert_eq!(pdb.stream_count, 2);
        assert_eq!(pdb.guid.len(), 32);
    }

    /// Verifies summary formatting.
    #[test]
    fn test_pdb_summary() {
        let pdb = parse_pdb(&sample_pdb()).unwrap();
        assert!(format_pdb_summary(&pdb).contains("Age: 7"));
    }
}
