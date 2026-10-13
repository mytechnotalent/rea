//! Cipher Recovery
//!
//! **File:** `decrypt.rs`
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
use std::collections::BTreeSet;

/// Minimum recovered run length.
const MIN_LEN: usize = 6;

/// Maximum recovered strings returned.
const MAX_RESULTS: usize = 64;

/// Maximum bytes scanned per target.
const MAX_SCAN: usize = 4 << 20;

/// Byte ciphers attempted during recovery.
const CIPHERS: [Cipher; 5] = [
    Cipher::Xor,
    Cipher::Add,
    Cipher::Sub,
    Cipher::Rol,
    Cipher::Ror,
];

/// A simple reversible byte cipher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cipher {
    /// Exclusive-or with a single-byte key.
    Xor,

    /// Modular addition of a single-byte key.
    Add,

    /// Modular subtraction of a single-byte key.
    Sub,

    /// Bit rotation left by a key amount.
    Rol,

    /// Bit rotation right by a key amount.
    Ror,
}

impl Cipher {
    /// Returns the cipher's short name.
    ///
    /// # Returns
    ///
    /// The cipher name.
    fn name(self) -> &'static str {
        match self {
            Self::Xor => "xor",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Rol => "rol",
            Self::Ror => "ror",
        }
    }

    /// Applies the cipher to one byte.
    ///
    /// # Arguments
    ///
    /// * `byte` - Input byte.
    /// * `key` - Single-byte key.
    ///
    /// # Returns
    ///
    /// The transformed byte.
    fn apply(self, byte: u8, key: u8) -> u8 {
        match self {
            Self::Xor => byte ^ key,
            Self::Add => byte.wrapping_add(key),
            Self::Sub => byte.wrapping_sub(key),
            Self::Rol => byte.rotate_left(u32::from(key) & 7),
            Self::Ror => byte.rotate_right(u32::from(key) & 7),
        }
    }
}

/// A string recovered by applying a simple cipher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// Address where the recovered string begins.
    pub address: u64,

    /// Cipher that revealed the string.
    pub cipher: &'static str,

    /// Single-byte key that revealed the string.
    pub key: u64,

    /// The recovered text.
    pub text: String,
}

/// A printable run accumulated during scanning.
#[derive(Default)]
struct Run {
    /// Offset where the run begins.
    start: usize,

    /// Bytes in the run.
    bytes: Vec<u8>,
}

impl Run {
    /// Appends a byte, recording the start offset on the first byte.
    ///
    /// # Arguments
    ///
    /// * `index` - Byte offset in the buffer.
    /// * `byte` - Decoded byte.
    fn push(&mut self, index: usize, byte: u8) {
        if self.bytes.is_empty() {
            self.start = index;
        }
        self.bytes.push(byte);
    }

    /// Clears the accumulated run.
    fn clear(&mut self) {
        self.bytes.clear();
    }
}

/// Scanner state for one cipher and key.
struct Scanner<'a> {
    /// Base address added to recovered offsets.
    base: u64,

    /// Cipher being applied.
    cipher: Cipher,

    /// Key being applied.
    key: u8,

    /// Accumulated results.
    out: &'a mut Vec<Decoded>,

    /// Deduplication set of recovered texts.
    seen: &'a mut BTreeSet<String>,

    /// Current printable run.
    run: Run,
}

impl Scanner<'_> {
    /// Feeds one decoded byte, emitting a run when text ends.
    ///
    /// # Arguments
    ///
    /// * `index` - Byte offset in the buffer.
    /// * `byte` - Decoded byte.
    fn feed(&mut self, index: usize, byte: u8) {
        if is_text(byte) {
            self.run.push(index, byte);
        } else {
            self.emit();
        }
    }

    /// Emits the current run when it is a plausible string.
    fn emit(&mut self) {
        if self.run.bytes.len() >= MIN_LEN {
            self.record();
        }
        self.run.clear();
    }

    /// Records the current run when it is meaningful.
    fn record(&mut self) {
        let text = String::from_utf8_lossy(&self.run.bytes).into_owned();
        let address = self.base + self.run.start as u64;
        if interesting(&text) && self.seen.insert(text.clone()) {
            self.push(address, text);
        }
    }

    /// Stores a recovered string.
    ///
    /// # Arguments
    ///
    /// * `address` - Address of the recovered string.
    /// * `text` - Recovered text.
    fn push(&mut self, address: u64, text: String) {
        let key = u64::from(self.key);
        self.out.push(Decoded { address, cipher: self.cipher.name(), key, text });
    }
}

/// Recovers printable strings hidden by single-byte ciphers.
///
/// # Arguments
///
/// * `bytes` - Target contents.
/// * `base` - Base address added to reported offsets.
///
/// # Returns
///
/// The recovered strings, capped at [`MAX_RESULTS`].
#[must_use]
pub fn recover(bytes: &[u8], base: u64) -> Vec<Decoded> {
    let scanned = &bytes[..bytes.len().min(MAX_SCAN)];
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for cipher in CIPHERS {
        sweep(scanned, base, cipher, &mut out, &mut seen);
    }
    out
}

/// Sweeps every key for one cipher.
///
/// # Arguments
///
/// * `bytes` - Target contents.
/// * `base` - Base address added to reported offsets.
/// * `cipher` - Cipher to apply.
/// * `out` - Accumulated results.
/// * `seen` - Deduplication set.
fn sweep(
    bytes: &[u8],
    base: u64,
    cipher: Cipher,
    out: &mut Vec<Decoded>,
    seen: &mut BTreeSet<String>,
) {
    for key in 1..=u8::MAX {
        scan(bytes, base, cipher, key, out, seen);
        if out.len() >= MAX_RESULTS {
            return;
        }
    }
}

/// Scans the buffer for printable runs under one cipher and key.
///
/// # Arguments
///
/// * `bytes` - Target contents.
/// * `base` - Base address added to reported offsets.
/// * `cipher` - Cipher to apply.
/// * `key` - Key to apply.
/// * `out` - Accumulated results.
/// * `seen` - Deduplication set.
fn scan(
    bytes: &[u8],
    base: u64,
    cipher: Cipher,
    key: u8,
    out: &mut Vec<Decoded>,
    seen: &mut BTreeSet<String>,
) {
    let mut scanner = Scanner { base, cipher, key, out, seen, run: Run::default() };
    for (index, byte) in bytes.iter().enumerate() {
        scanner.feed(index, cipher.apply(*byte, key));
    }
    scanner.emit();
}

/// Reports whether a byte is printable text.
///
/// # Arguments
///
/// * `byte` - Byte to test.
///
/// # Returns
///
/// `true` for printable bytes.
fn is_text(byte: u8) -> bool {
    byte.is_ascii_graphic() || byte == b' ' || byte == b'\t'
}

/// Reports whether a recovered run looks like meaningful text.
///
/// # Arguments
///
/// * `text` - Recovered run.
///
/// # Returns
///
/// `true` for runs with enough letters.
fn interesting(text: &str) -> bool {
    let letters = text.chars().filter(char::is_ascii_alphabetic).count();
    letters >= 3 && text.chars().any(|c| c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that an XOR-obfuscated string is recovered.
    #[test]
    fn test_recover_xor_string() {
        let plain = b"http://c2.example/beacon";
        let encoded: Vec<u8> = plain.iter().map(|b| b ^ 0x5a).collect();
        let decoded = recover(&encoded, 0x1000);
        assert!(decoded.iter().any(|d| d.text.contains("http://c2.example")));
    }

    /// Verifies that an ADD-obfuscated string is recovered.
    #[test]
    fn test_recover_add_string() {
        let plain = b"CMO{xor_and_roll_recovered}";
        let encoded: Vec<u8> = plain.iter().map(|b| b.wrapping_add(0x11)).collect();
        let decoded = recover(&encoded, 0);
        assert!(decoded.iter().any(|d| d.text.contains("CMO{")));
    }

    /// Verifies that plaintext does not produce noisy false positives.
    #[test]
    fn test_recover_deduplicates() {
        let decoded = recover(b"hello world hello world", 0);
        let texts: BTreeSet<&str> = decoded.iter().map(|d| d.text.as_str()).collect();
        assert_eq!(texts.len(), decoded.len());
    }
}
