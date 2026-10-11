//! Runtime String Discovery
//!
//! **File:** `discover.rs`
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

// Import dependencies from rea_sleigh
use rea_sleigh::Emulator;

/// Bytes scanned per mapped page.
const PAGE_SPAN: usize = 0x1000;

/// Trailing bytes read past a page so tokens may cross its boundary.
const OVERLAP: usize = 128;

/// Maximum bytes between an opening and closing brace.
const MAX_TOKEN: usize = 96;

/// Maximum findings reported from one run.
const MAX_FINDINGS: usize = 16;

/// Minimum identifier characters before the opening brace.
const MIN_PREFIX: usize = 2;

/// A flag-shaped token recovered from emulated memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Address where the token begins.
    pub address: u64,

    /// The recovered token text.
    pub text: String,
}

/// Scans emulated memory for flag-shaped tokens absent from the file.
///
/// A token is an identifier followed by a braced, printable body, in the shape
/// used by capture-the-flag flags of any prefix. Tokens already present in the
/// static file are ignored, so only runtime-produced strings are reported.
///
/// # Arguments
///
/// * `emu` - Emulator whose memory is scanned.
/// * `static_bytes` - Original file contents, used to reject known strings.
///
/// # Returns
///
/// The recovered tokens, newest discoveries first and capped in number.
#[must_use]
pub fn scan(emu: &Emulator, static_bytes: &[u8]) -> Vec<Finding> {
    let mut out = Vec::new();
    for base in emu.mapped_pages() {
        window(emu, base, static_bytes, &mut out);
        if out.len() >= MAX_FINDINGS {
            break;
        }
    }
    out
}

/// Scans one mapped page window for tokens.
///
/// # Arguments
///
/// * `emu` - Emulator whose memory is read.
/// * `base` - Page base address.
/// * `static_bytes` - Original file contents.
/// * `out` - Accumulator receiving findings.
fn window(emu: &Emulator, base: u64, static_bytes: &[u8], out: &mut Vec<Finding>) {
    let bytes = emu.read_bytes(base, PAGE_SPAN + OVERLAP);
    for (offset, text) in tokens(&bytes) {
        consider(base + offset as u64, text, static_bytes, out);
    }
}

/// Records a token when it is novel and not in the static file.
///
/// # Arguments
///
/// * `address` - Address where the token begins.
/// * `text` - Token text.
/// * `static_bytes` - Original file contents.
/// * `out` - Accumulator receiving findings.
fn consider(address: u64, text: String, static_bytes: &[u8], out: &mut Vec<Finding>) {
    let known = contains(static_bytes, text.as_bytes());
    let seen = out.iter().any(|finding| finding.text == text);
    if !known && !seen {
        out.push(Finding { address, text });
    }
}

/// Finds flag-shaped tokens in a byte slice.
///
/// # Arguments
///
/// * `bytes` - Bytes to search.
///
/// # Returns
///
/// The `(offset, text)` of each token.
fn tokens(bytes: &[u8]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for position in brace_positions(bytes) {
        if let Some(found) = token_at(bytes, position) {
            out.push(found);
        }
    }
    out
}

/// Lists the positions of opening braces in a slice.
///
/// # Arguments
///
/// * `bytes` - Bytes to search.
///
/// # Returns
///
/// The indices of each `{` byte.
fn brace_positions(bytes: &[u8]) -> Vec<usize> {
    bytes.iter().enumerate().filter(|(_, &b)| b == b'{').map(|(i, _)| i).collect()
}

/// Extracts a token anchored at an opening brace.
///
/// # Arguments
///
/// * `bytes` - Bytes to search.
/// * `brace` - Index of the opening brace.
///
/// # Returns
///
/// The `(offset, text)` of the token, or `None` when the shape does not match.
fn token_at(bytes: &[u8], brace: usize) -> Option<(usize, String)> {
    let start = prefix_start(bytes, brace)?;
    let close = close_brace(bytes, brace)?;
    let text = String::from_utf8_lossy(&bytes[start..=close]).into_owned();
    Some((start, text))
}

/// Finds the start of the identifier preceding a brace.
///
/// # Arguments
///
/// * `bytes` - Bytes to search.
/// * `brace` - Index of the opening brace.
///
/// # Returns
///
/// The identifier start, or `None` when the prefix is too short.
fn prefix_start(bytes: &[u8], brace: usize) -> Option<usize> {
    let mut start = brace;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    (brace - start >= MIN_PREFIX).then_some(start)
}

/// Finds the closing brace of a printable token body.
///
/// # Arguments
///
/// * `bytes` - Bytes to search.
/// * `brace` - Index of the opening brace.
///
/// # Returns
///
/// The index of the closing brace, or `None` when the body is not printable.
fn close_brace(bytes: &[u8], brace: usize) -> Option<usize> {
    let end = (brace + 1 + MAX_TOKEN).min(bytes.len());
    let body = bytes.get(brace + 1..end)?;
    let stop = body.iter().position(|&b| b == b'}')?;
    body[..stop].iter().all(|&b| is_print(b)).then_some(brace + 1 + stop)
}

/// Reports whether a byte may appear in an identifier.
///
/// # Arguments
///
/// * `byte` - Byte to classify.
///
/// # Returns
///
/// `true` for alphanumerics and underscore.
fn is_ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Reports whether a byte is printable ASCII.
///
/// # Arguments
///
/// * `byte` - Byte to classify.
///
/// # Returns
///
/// `true` for the printable ASCII range.
fn is_print(byte: u8) -> bool {
    (0x20..=0x7e).contains(&byte)
}

/// Reports whether a byte slice contains a sub-slice.
///
/// # Arguments
///
/// * `hay` - Bytes to search.
/// * `needle` - Sub-slice to find.
///
/// # Returns
///
/// `true` when `needle` occurs in `hay`.
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return false;
    }
    memchr::memmem::find(hay, needle).is_some()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies tokens of an arbitrary prefix are detected, proving the scan
    /// has no built-in prefix list. The prefix here is deliberately nonsense.
    #[test]
    fn test_tokens_detects_any_prefix() {
        let found = tokens(b"junk Wxyz{abc_123} tail");
        assert_eq!(found, vec![(5, "Wxyz{abc_123}".to_string())]);
    }

    /// Verifies a one-character prefix is rejected as too short.
    #[test]
    fn test_tokens_rejects_short_prefix() {
        assert!(tokens(b"a{bc}").is_empty());
    }

    /// Verifies a non-printable body is rejected.
    #[test]
    fn test_tokens_rejects_binary_body() {
        assert!(tokens(b"AB{a\x00b}").is_empty());
    }

    /// Verifies the sub-slice search matches and rejects correctly.
    #[test]
    fn test_contains_matches() {
        assert!(contains(b"hello world", b"o w"));
        assert!(!contains(b"hello", b"xyz"));
    }
}
