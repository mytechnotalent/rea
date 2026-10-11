//! String Extraction
//!
//! **File:** `strings.rs`
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

// Import crate::numeric
use crate::numeric::to_u64;

/// Minimum run length, in characters, for a string to be reported.
pub const MIN_LEN: usize = 5;

/// Maximum number of strings retained per encoding.
pub const MAX_STRINGS: usize = 50_000;

/// Maximum characters kept from one string.
pub const MAX_TEXT: usize = 300;

/// Character encodings recognized by the extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Single-byte printable ASCII.
    Ascii,

    /// Little-endian UTF-16 restricted to printable ASCII code points.
    Utf16Le,
}

/// A printable string located in the target file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundString {
    /// File offset of the first character.
    pub offset: u64,

    /// String text, truncated to [`MAX_TEXT`] characters.
    pub text: String,

    /// Encoding the string was found in.
    pub encoding: Encoding,
}

/// Extracts ASCII and UTF-16LE strings, ordered by file offset.
///
/// # Arguments
///
/// * `bytes` - Target file contents.
///
/// # Returns
///
/// Every string of at least [`MIN_LEN`] characters, within the caps.
#[must_use]
pub fn extract(bytes: &[u8]) -> Vec<FoundString> {
    let mut found = ascii(bytes);
    found.extend(utf16(bytes));
    found.sort_by_key(|s| s.offset);
    found
}

/// Extracts single-byte strings.
///
/// # Arguments
///
/// * `bytes` - Target file contents.
///
/// # Returns
///
/// ASCII strings in offset order.
fn ascii(bytes: &[u8]) -> Vec<FoundString> {
    let mut scanner = Scanner::new(1, Encoding::Ascii);
    for (index, byte) in bytes.iter().enumerate() {
        scanner.feed(index, u32::from(*byte));
    }
    scanner.finish()
}

/// Extracts two-byte little-endian strings at even offsets.
///
/// # Arguments
///
/// * `bytes` - Target file contents.
///
/// # Returns
///
/// UTF-16LE strings in offset order.
fn utf16(bytes: &[u8]) -> Vec<FoundString> {
    let mut scanner = Scanner::new(2, Encoding::Utf16Le);
    for (index, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        scanner.feed(index, u32::from(u16::from_le_bytes([pair[0], pair[1]])));
    }
    scanner.finish()
}

/// Maps a code unit to a printable character.
///
/// # Arguments
///
/// * `unit` - Code unit value.
///
/// # Returns
///
/// The character for tab or printable ASCII, otherwise `None`.
fn printable(unit: u32) -> Option<char> {
    let ok = unit == 0x09 || (0x20..0x7f).contains(&unit);
    ok.then(|| char::from_u32(unit)).flatten()
}

/// Incremental run detector for one encoding.
struct Scanner {
    /// Bytes per code unit.
    width: u64,

    /// Encoding tag attached to results.
    encoding: Encoding,

    /// Unit index where the current run started.
    start: usize,

    /// Characters seen in the current run.
    length: usize,

    /// Retained text of the current run.
    run: String,

    /// Completed strings.
    found: Vec<FoundString>,
}

impl Scanner {
    /// Creates a scanner for one encoding.
    ///
    /// # Arguments
    ///
    /// * `width` - Bytes per code unit.
    /// * `encoding` - Encoding tag for results.
    ///
    /// # Returns
    ///
    /// An idle scanner.
    fn new(width: u64, encoding: Encoding) -> Self {
        Self { width, encoding, start: 0, length: 0, run: String::new(), found: Vec::new() }
    }

    /// Consumes one code unit.
    ///
    /// # Arguments
    ///
    /// * `index` - Unit index within the file.
    /// * `unit` - Code unit value.
    fn feed(&mut self, index: usize, unit: u32) {
        match printable(unit) {
            Some(c) => self.extend(index, c),
            None => self.flush(),
        }
    }

    /// Appends a character to the current run.
    ///
    /// # Arguments
    ///
    /// * `index` - Unit index of the character.
    /// * `c` - Printable character.
    fn extend(&mut self, index: usize, c: char) {
        if self.length == 0 {
            self.start = index;
        }
        if self.length < MAX_TEXT {
            self.run.push(c);
        }
        self.length += 1;
    }

    /// Ends the current run, keeping it when long enough.
    fn flush(&mut self) {
        if self.length >= MIN_LEN && self.found.len() < MAX_STRINGS {
            let text = std::mem::take(&mut self.run);
            let found = self.completed(text);
            self.found.push(found);
        }
        self.run.clear();
        self.length = 0;
    }

    /// Wraps the current run's text as a located string.
    ///
    /// # Arguments
    ///
    /// * `text` - Retained run text.
    ///
    /// # Returns
    ///
    /// The string with its file offset and encoding.
    fn completed(&self, text: String) -> FoundString {
        FoundString { offset: to_u64(self.start) * self.width, text, encoding: self.encoding }
    }

    /// Flushes the final run and returns all strings.
    ///
    /// # Returns
    ///
    /// Every completed string.
    fn finish(mut self) -> Vec<FoundString> {
        self.flush();
        self.found
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies ASCII extraction with offsets and the length threshold.
    #[test]
    fn test_extract_ascii_with_offsets() {
        let found = extract(b"\x00\x01hello world\x00abc\x00");
        assert_eq!(found.len(), 1);
        assert_eq!(
            (found[0].offset, found[0].text.as_str()),
            (2, "hello world")
        );
    }

    /// Verifies UTF-16LE extraction.
    #[test]
    fn test_extract_utf16() {
        let data: Vec<u8> = "kernel32".encode_utf16().flat_map(u16::to_le_bytes).collect();
        let found = extract(&data);
        assert!(found.iter().any(|s| s.encoding == Encoding::Utf16Le && s.text == "kernel32"));
    }

    /// Verifies that very long runs are truncated but still reported.
    #[test]
    fn test_extract_truncates_long_runs() {
        let data = vec![b'A'; MAX_TEXT * 3];
        let found = extract(&data);
        assert_eq!(found[0].text.len(), MAX_TEXT);
    }

    /// Verifies that tabs count as printable.
    #[test]
    fn test_printable_accepts_tab() {
        assert_eq!(printable(0x09), Some('\t'));
        assert_eq!(printable(0x7f), None);
    }
}
