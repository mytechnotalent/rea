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

/// Reports whether an ASCII character is a vowel.
///
/// # Arguments
///
/// * `c` - Character to inspect.
///
/// # Returns
///
/// `true` for ASCII vowels in either case.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

/// Reports whether text resembles a dotted version number.
///
/// # Arguments
///
/// * `s` - Candidate text.
///
/// # Returns
///
/// `true` when segments are numeric with at least one dot.
fn is_version(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() > 1 && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// Reports whether letters and digits interleave unnaturally as in opcode noise.
///
/// # Arguments
///
/// * `s` - Candidate text.
///
/// # Returns
///
/// `true` for noise patterns like `b3K4I` or `O6L`.
fn has_interleaved_digits(s: &str) -> bool {
    let b = s.as_bytes();
    if s.starts_with("0x") || s.starts_with("boot") || s.starts_with("pico") {
        return false;
    }
    b.windows(4).any(|w| {
        (w[0].is_ascii_alphabetic() && w[1].is_ascii_digit() && w[2].is_ascii_alphabetic())
            || (w[0].is_ascii_digit() && w[1].is_ascii_alphabetic() && w[2].is_ascii_digit())
    })
}

/// Reports whether a string containing percent signs is a plausible format string.
///
/// # Arguments
///
/// * `s` - Candidate text containing `%`.
///
/// # Returns
///
/// `true` for genuine format strings like `age: %d`.
fn is_valid_format_str(s: &str) -> bool {
    let Some(idx) = s.find('%') else { return false };
    let spec = s[idx + 1..]
        .chars()
        .find(|c| !c.is_ascii_digit() && !matches!(c, '-' | '+' | ' ' | '#' | '.'));
    let valid_spec = spec
        .is_some_and(|c| matches!(c, 'd' | 'i' | 'u' | 'o' | 'x' | 'X' | 'f' | 's' | 'c' | 'p'));
    let pfx = s[..idx].trim_matches(|c: char| !c.is_ascii_alphabetic());
    valid_spec && (pfx.is_empty() || pfx.chars().any(is_vowel))
}

/// Advances consonant run count for a character.
///
/// # Arguments
///
/// * `c` - Current character.
/// * `run` - Previous consonant run.
///
/// # Returns
///
/// Updated consonant run count.
fn next_consonant_run(c: char, run: usize) -> usize {
    if c.is_ascii_alphabetic() && !is_vowel(c) {
        run + 1
    } else {
        0
    }
}

/// Reports whether a consonant sequence reaches four characters.
///
/// # Arguments
///
/// * `s` - Candidate text.
///
/// # Returns
///
/// `true` when a consonant sequence reaches four characters.
fn has_excessive_consonants(s: &str) -> bool {
    let mut run = 0;
    for c in s.chars() {
        run = next_consonant_run(c, run);
        if run >= 4 {
            return true;
        }
    }
    false
}

/// Reports whether a single letter dominates the text as typical of opcode repetitions.
///
/// # Arguments
///
/// * `s` - Candidate text.
/// * `alphas` - Total alphabetic count.
///
/// # Returns
///
/// `true` when one letter appears three or more times and forms at least 40% of letters.
fn has_repeated_letter(s: &str, alphas: usize) -> bool {
    let mut counts = [0u8; 26];
    for c in s.chars().filter(char::is_ascii_alphabetic) {
        counts[(c.to_ascii_lowercase() as usize) - ('a' as usize)] += 1;
    }
    counts.iter().any(|&cnt| cnt >= 3 && usize::from(cnt) * 5 >= alphas * 2)
}

/// Reports whether a character is an internal symbol noise delimiter.
///
/// # Arguments
///
/// * `c` - Character to test.
///
/// # Returns
///
/// `true` for non-standard punctuation noise.
fn is_noise_char(c: char) -> bool {
    b"@!+*)('`~^&|<>;\"'".contains(&(c as u8))
}

/// Reports whether unexpected symbols appear inside words.
///
/// # Arguments
///
/// * `s` - Candidate text.
///
/// # Returns
///
/// `true` when internal noise characters appear.
fn has_symbol_noise(s: &str) -> bool {
    let clean = s.trim_matches(|c| matches!(c, '*' | '-' | '=' | '#' | '"' | '\''));
    let inner = clean.trim_matches(|c| matches!(c, ' ' | '_' | '-' | '.' | ':' | '/' | ','));
    inner.chars().any(is_noise_char)
}

/// Reports whether letter casing flips unnaturally back and forth.
///
/// # Arguments
///
/// * `s` - Candidate text.
/// * `alphas` - Total alphabetic count.
///
/// # Returns
///
/// `true` for alternating casing noise like `aKiCa`.
fn has_alternating_case(s: &str, alphas: usize) -> bool {
    let letters: Vec<char> = s.chars().filter(char::is_ascii_alphabetic).collect();
    let flips = letters
        .windows(2)
        .filter(|w| w[0].is_ascii_uppercase() != w[1].is_ascii_uppercase())
        .count();
    flips >= 3 && flips * 2 >= alphas
}

/// Reports whether the text contains at least one word with a vowel.
///
/// # Arguments
///
/// * `s` - Candidate text.
///
/// # Returns
///
/// `true` when a token contains vowels and normal characters.
fn has_valid_word(s: &str) -> bool {
    s.split([' ', '_', '-', '.', ':', '/', ','])
        .any(|w| w.len() >= 2 && w.chars().any(is_vowel))
}

/// Reports whether text structure has normal natural vowel and consonant rhythms.
///
/// # Arguments
///
/// * `t` - Trimmed candidate text.
/// * `alphas` - Count of alphabetic characters.
///
/// # Returns
///
/// `true` if text follows natural rhythm.
fn has_natural_rhythm(t: &str, alphas: usize) -> bool {
    let vowels = t.chars().filter(|c| is_vowel(*c)).count();
    vowels > 0
        && vowels * 4 >= alphas
        && !has_excessive_consonants(t)
        && !has_repeated_letter(t, alphas)
        && !has_alternating_case(t, alphas)
        && has_valid_word(t)
}

/// Checks if string matches version or format string patterns.
///
/// # Arguments
///
/// * `t` - Trimmed candidate text.
///
/// # Returns
///
/// Plausibility decision if recognized as specialized format.
fn is_special_text(t: &str) -> Option<bool> {
    if is_version(t) {
        return Some(true);
    }
    if t.contains('%') {
        return Some(is_valid_format_str(t));
    }
    None
}

/// Reports whether candidate string satisfies basic length and noise checks.
fn is_plausible_candidate(t: &str) -> bool {
    !(t.len() < 3 || has_interleaved_digits(t) || has_symbol_noise(t))
}

/// Checks whether alphabetic characters have natural linguistic rhythm.
fn check_natural_rhythm(t: &str) -> bool {
    let alphas = t.chars().filter(char::is_ascii_alphabetic).count();
    alphas >= 3 && has_natural_rhythm(t, alphas)
}

/// Reports whether an extracted string is plausible text rather than opcode noise.
///
/// # Arguments
///
/// * `s` - Candidate string text.
///
/// # Returns
///
/// `true` when the string has natural formatting, words, and vowels.
#[must_use]
pub fn is_plausible(s: &str) -> bool {
    let t = s.trim();
    if !is_plausible_candidate(t) {
        return false;
    }
    is_special_text(t).unwrap_or_else(|| check_natural_rhythm(t))
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

    /// Verifies that plausible text strings pass validation.
    #[test]
    fn test_plausible_strings_accepted() {
        assert!(is_plausible("age: %d"));
        assert!(is_plausible("*** PANIC ***"));
        assert!(is_plausible("2.2.0"));
        assert!(is_plausible("pico2"));
    }

    /// Verifies that opcode sequence noise is rejected.
    #[test]
    fn test_opcode_noise_rejected() {
        assert!(!is_plausible("FKFAF  "));
        assert!(!is_plausible("BFKFQF  "));
        assert!(!is_plausible("RFKFAF  "));
        assert!(!is_plausible("b3K4I"));
        assert!(!is_plausible("aKiCa"));
    }
}
