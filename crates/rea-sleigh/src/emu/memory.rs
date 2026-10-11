//! Emulator Memory
//!
//! **File:** `memory.rs`
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
use std::collections::HashMap;

/// Maximum width, in bytes, of a word read or written as an integer.
pub(crate) const WORD: usize = 8;

/// Size of a storage page in bytes.
const PAGE: usize = 0x1000;

/// Allocates a fresh zeroed page.
///
/// # Returns
///
/// A boxed page of zero bytes.
fn zero_page() -> Box<[u8; PAGE]> {
    Box::new([0u8; PAGE])
}

/// Splits a byte offset into its page number and in-page index.
///
/// # Arguments
///
/// * `offset` - Byte offset in a space.
///
/// # Returns
///
/// The `(page, index)` pair.
fn locate(offset: u64) -> (u64, usize) {
    let page = offset / PAGE as u64;
    let index = usize::try_from(offset % PAGE as u64).unwrap_or(0);
    (page, index)
}

/// Finds the offsets where a needle occurs in a byte slice.
///
/// # Arguments
///
/// * `hay` - Bytes to search.
/// * `needle` - Pattern to find.
///
/// # Returns
///
/// The starting offsets of each match.
fn positions(hay: &[u8], needle: &[u8]) -> Vec<u64> {
    if needle.is_empty() || hay.len() < needle.len() {
        return Vec::new();
    }
    let last = hay.len() - needle.len();
    (0..=last)
        .filter(|&i| &hay[i..i + needle.len()] == needle)
        .map(|i| i as u64)
        .collect()
}

/// Paged byte-addressable storage spanning every p-code address space.
///
/// Pages are keyed by their space index and page number, so registers, unique
/// temporaries, and RAM share one uniform store while large images stay
/// compact. Unmapped bytes read as zero.
#[derive(Debug, Default)]
pub(crate) struct Memory {
    /// Live pages keyed by `(space index, page number)`.
    pages: HashMap<(usize, u64), Box<[u8; PAGE]>>,

    /// Whether multi-byte accesses use big-endian byte order.
    big_endian: bool,
}

impl Memory {
    /// Creates empty storage with a fixed byte order.
    ///
    /// # Arguments
    ///
    /// * `big_endian` - Whether multi-byte accesses are big-endian.
    ///
    /// # Returns
    ///
    /// The empty memory.
    pub(crate) fn new(big_endian: bool) -> Self {
        Self { pages: HashMap::new(), big_endian }
    }

    /// Reads one byte, defaulting unmapped cells to zero.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `offset` - Byte offset in the space.
    ///
    /// # Returns
    ///
    /// The stored byte, or zero.
    pub(crate) fn read_byte(&self, space: usize, offset: u64) -> u8 {
        let (page, index) = locate(offset);
        self.pages.get(&(space, page)).map_or(0, |bytes| bytes[index])
    }

    /// Writes one byte, allocating its page on first touch.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `offset` - Byte offset in the space.
    /// * `value` - Byte to store.
    pub(crate) fn write_byte(&mut self, space: usize, offset: u64, value: u8) {
        let (page, index) = locate(offset);
        self.pages.entry((space, page)).or_insert_with(zero_page)[index] = value;
    }

    /// Loads a byte slice into a space at a base offset.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `base` - Starting byte offset.
    /// * `bytes` - Bytes to load.
    pub(crate) fn load(&mut self, space: usize, base: u64, bytes: &[u8]) {
        for (index, byte) in bytes.iter().enumerate() {
            self.write_byte(space, base.wrapping_add(index as u64), *byte);
        }
    }

    /// Finds every occurrence of a byte pattern in a space.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `needle` - Byte pattern to search for.
    ///
    /// # Returns
    ///
    /// The byte offsets where the pattern begins, in ascending order.
    pub(crate) fn find(&self, space: usize, needle: &[u8]) -> Vec<u64> {
        let mut pages = self.page_numbers(space);
        pages.sort_unstable();
        let mut hits = Vec::new();
        for page in pages {
            self.scan_page(space, page, needle, &mut hits);
        }
        hits
    }

    /// Collects the mapped page numbers for a space.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    ///
    /// # Returns
    ///
    /// The page numbers, in arbitrary order.
    fn page_numbers(&self, space: usize) -> Vec<u64> {
        self.pages.keys().filter(|(s, _)| *s == space).map(|(_, page)| *page).collect()
    }

    /// Returns the base addresses of mapped pages in a space, ascending.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    ///
    /// # Returns
    ///
    /// The page base addresses.
    pub(crate) fn page_bases(&self, space: usize) -> Vec<u64> {
        let mut pages = self.page_numbers(space);
        pages.sort_unstable();
        pages.iter().map(|page| page * PAGE as u64).collect()
    }

    /// Scans one page plus its boundary into the next for a pattern.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `page` - Page number to scan.
    /// * `needle` - Byte pattern to search for.
    /// * `hits` - Accumulator receiving match offsets.
    fn scan_page(&self, space: usize, page: u64, needle: &[u8], hits: &mut Vec<u64>) {
        let base = page * PAGE as u64;
        let span = PAGE + needle.len().saturating_sub(1);
        let bytes: Vec<u8> = (0..span).map(|i| self.read_byte(space, base + i as u64)).collect();
        for offset in positions(&bytes, needle) {
            hits.push(base + offset);
        }
    }

    /// Reads a little- or big-endian word up to [`WORD`] bytes wide.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `offset` - Byte offset in the space.
    /// * `size` - Width in bytes, clamped to [`WORD`].
    ///
    /// # Returns
    ///
    /// The assembled value, right-aligned.
    pub(crate) fn read(&self, space: usize, offset: u64, size: usize) -> u64 {
        let width = size.min(WORD);
        (0..width).fold(0u64, |value, index| {
            self.fold_read(value, space, offset, index, width)
        })
    }

    /// Folds one byte into a word being assembled from memory.
    ///
    /// # Arguments
    ///
    /// * `value` - Word assembled so far.
    /// * `space` - Address space index.
    /// * `offset` - Base byte offset.
    /// * `index` - Byte position within the word.
    /// * `width` - Total width in bytes.
    ///
    /// # Returns
    ///
    /// The word with this byte placed at its endian position.
    fn fold_read(&self, value: u64, space: usize, offset: u64, index: usize, width: usize) -> u64 {
        let byte = u64::from(self.read_byte(space, offset.wrapping_add(index as u64)));
        value | (byte << self.shift_of(index, width))
    }

    /// Writes a little- or big-endian word up to [`WORD`] bytes wide.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `offset` - Byte offset in the space.
    /// * `size` - Width in bytes, clamped to [`WORD`].
    /// * `value` - Value to store, right-aligned.
    pub(crate) fn write(&mut self, space: usize, offset: u64, size: usize, value: u64) {
        let width = size.min(WORD);
        for index in 0..width {
            self.write_one(space, offset, index, width, value);
        }
    }

    /// Writes one byte of a word at its endian position.
    ///
    /// # Arguments
    ///
    /// * `space` - Address space index.
    /// * `offset` - Base byte offset.
    /// * `index` - Byte position within the word.
    /// * `width` - Total width in bytes.
    /// * `value` - Word being stored.
    fn write_one(&mut self, space: usize, offset: u64, index: usize, width: usize, value: u64) {
        let byte = u8::try_from((value >> self.shift_of(index, width)) & 0xff).unwrap_or(0);
        self.write_byte(space, offset.wrapping_add(index as u64), byte);
    }

    /// Returns the bit shift for a byte position under the active byte order.
    ///
    /// # Arguments
    ///
    /// * `index` - Byte position within the word.
    /// * `width` - Total width in bytes.
    ///
    /// # Returns
    ///
    /// The left shift that places this byte at its endian position.
    fn shift_of(&self, index: usize, width: usize) -> u32 {
        let position = if self.big_endian {
            width - 1 - index
        } else {
            index
        };
        u32::try_from(position * 8).unwrap_or(0)
    }

    /// Copies a byte range between two locations in any spaces.
    ///
    /// # Arguments
    ///
    /// * `src` - Source `(space, offset)`.
    /// * `dst` - Destination `(space, offset)`.
    /// * `size` - Number of bytes to copy.
    pub(crate) fn copy(&mut self, src: (usize, u64), dst: (usize, u64), size: usize) {
        for index in 0..size {
            let byte = self.read_byte(src.0, src.1.wrapping_add(index as u64));
            self.write_byte(dst.0, dst.1.wrapping_add(index as u64), byte);
        }
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies little-endian round-trips preserve the value.
    #[test]
    fn test_little_endian_round_trip() {
        let mut memory = Memory::new(false);
        memory.write(0, 0x100, 4, 0x1122_3344);
        assert_eq!(memory.read(0, 0x100, 4), 0x1122_3344);
        assert_eq!(memory.read_byte(0, 0x100), 0x44);
    }

    /// Verifies big-endian storage places the high byte first.
    #[test]
    fn test_big_endian_byte_order() {
        let mut memory = Memory::new(true);
        memory.write(0, 0, 2, 0xaabb);
        assert_eq!(memory.read_byte(0, 0), 0xaa);
        assert_eq!(memory.read_byte(0, 1), 0xbb);
    }

    /// Verifies segment loading and wide byte copies.
    #[test]
    fn test_load_and_copy() {
        let mut memory = Memory::new(false);
        memory.load(1, 0, &[1, 2, 3, 4]);
        memory.copy((1, 0), (2, 0x10), 4);
        assert_eq!(memory.read(2, 0x10, 4), 0x0403_0201);
    }

    /// Verifies unwritten cells read as zero.
    #[test]
    fn test_unmapped_reads_zero() {
        let memory = Memory::new(false);
        assert_eq!(memory.read(0, 0xdead, 8), 0);
    }

    /// Verifies the generic byte search finds an arbitrary needle, including
    /// one that straddles a page boundary. The needle is neutral test data.
    #[test]
    fn test_find_pattern() {
        let mut memory = Memory::new(false);
        memory.load(0, 0x2000, b"junkWxyz-abcjunk");
        memory.load(0, 0x0ffe, b"Wxyz-x");
        assert_eq!(memory.find(0, b"Wxyz"), vec![0x0ffe, 0x2004]);
    }
}
