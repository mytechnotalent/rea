//! Cryptographic Constant Detection
//!
//! **File:** `crypto.rs`
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

/// Maximum hits reported per constant.
const MAX_HITS: usize = 8;

/// Known cryptographic constants and their identifying bytes.
const CONSTANTS: &[(&str, &str, &[u8])] = &[
    (
        "AES S-box",
        "block-cipher",
        &[
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7,
            0xab, 0x76,
        ],
    ),
    (
        "AES inverse S-box",
        "block-cipher",
        &[
            0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3,
            0xd7, 0xfb,
        ],
    ),
    (
        "MD5 initial state",
        "hash",
        &[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ],
    ),
    (
        "SHA-1 initial state",
        "hash",
        &[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10, 0xf0, 0xe1, 0xd2, 0xc3,
        ],
    ),
    (
        "SHA-256 initial state",
        "hash",
        &[
            0x67, 0xe6, 0x09, 0x6a, 0x85, 0xae, 0x67, 0xbb, 0x72, 0xf3, 0x6e, 0x3c, 0x3a, 0xf5,
            0x4f, 0xa5,
        ],
    ),
    (
        "CRC32 table",
        "checksum",
        &[
            0x00, 0x00, 0x00, 0x00, 0x96, 0x30, 0x07, 0x77, 0x2c, 0x61, 0x0e, 0xee,
        ],
    ),
    ("TEA/XTEA delta", "block-cipher", &[0xb9, 0x79, 0x37, 0x9e]),
    (
        "Blowfish P-array",
        "block-cipher",
        &[
            0x24, 0x3f, 0x6a, 0x88, 0x85, 0xa3, 0x08, 0xd3, 0x13, 0x19, 0x8a, 0x2e, 0x03, 0x70,
            0x73, 0x44, 0xa4, 0x09, 0x38, 0x22, 0x29, 0x9f, 0x31, 0xd0, 0x08, 0x2e, 0xfa, 0x98,
            0xec, 0x4e, 0x6c, 0x89,
        ],
    ),
    (
        "DES S-box 1",
        "block-cipher",
        &[
            0x0e, 0x04, 0x0d, 0x01, 0x02, 0x0f, 0x0b, 0x08, 0x03, 0x0a, 0x06, 0x0c, 0x05, 0x09,
            0x00, 0x07,
        ],
    ),
    (
        "RC5/RC6 key schedule",
        "block-cipher",
        &[0xb7, 0xe1, 0x51, 0x63, 0x9e, 0x37, 0x79, 0xb9],
    ),
    ("ChaCha sigma", "stream-cipher", b"expand 32-byte k"),
    ("bcrypt salt prefix", "hash", b"$2a$"),
    (
        "base64 alphabet",
        "encoding",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
    ),
];

/// One detected cryptographic constant.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CryptoHit {
    /// Algorithm or table name.
    pub name: String,

    /// Classification, such as `block-cipher` or `hash`.
    pub kind: String,

    /// Offset of the constant.
    pub address: u64,
}

/// Detects known cryptographic constants in a buffer.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The detected constants.
#[must_use]
pub fn detect(bytes: &[u8]) -> Vec<CryptoHit> {
    CONSTANTS
        .iter()
        .flat_map(|(name, kind, pattern)| find_all(bytes, name, kind, pattern))
        .collect()
}

/// Finds every occurrence of one constant.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `name` - Constant name.
/// * `kind` - Constant classification.
/// * `pattern` - Byte pattern.
///
/// # Returns
///
/// The matching hits, capped at [`MAX_HITS`].
fn find_all(bytes: &[u8], name: &str, kind: &str, pattern: &[u8]) -> Vec<CryptoHit> {
    if pattern.is_empty() {
        return Vec::new();
    }
    let windows = bytes.windows(pattern.len()).enumerate();
    let hits = windows.filter(|(_, w)| *w == pattern).take(MAX_HITS);
    hits.map(|(i, _)| hit(name, kind, i)).collect()
}

/// Builds a hit record.
///
/// # Arguments
///
/// * `name` - Constant name.
/// * `kind` - Constant classification.
/// * `offset` - Byte offset.
///
/// # Returns
///
/// The hit record.
fn hit(name: &str, kind: &str, offset: usize) -> CryptoHit {
    CryptoHit {
        name: name.to_string(),
        kind: kind.to_string(),
        address: u64::try_from(offset).unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the AES S-box is detected.
    #[test]
    fn test_detect_aes_sbox() {
        let mut buffer = vec![0u8; 32];
        buffer[8..24].copy_from_slice(&[
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7,
            0xab, 0x76,
        ]);
        let hits = detect(&buffer);
        assert!(hits.iter().any(|h| h.name == "AES S-box" && h.address == 8));
    }

    /// Verifies that clean data yields no hits.
    #[test]
    fn test_detect_clean() {
        assert_eq!(detect(b"no crypto constants present here at all").len(), 0);
    }

    /// Verifies that the Blowfish P-array is detected.
    #[test]
    fn test_detect_blowfish() {
        let mut buffer = vec![0u8; 40];
        buffer[4..36].copy_from_slice(&blowfish_bytes());
        assert!(detect(&buffer).iter().any(|h| h.name == "Blowfish P-array"));
    }

    /// Returns the first four Blowfish P-array words.
    ///
    /// # Returns
    ///
    /// The leading P-array bytes.
    fn blowfish_bytes() -> [u8; 32] {
        [
            0x24, 0x3f, 0x6a, 0x88, 0x85, 0xa3, 0x08, 0xd3, 0x13, 0x19, 0x8a, 0x2e, 0x03, 0x70,
            0x73, 0x44, 0xa4, 0x09, 0x38, 0x22, 0x29, 0x9f, 0x31, 0xd0, 0x08, 0x2e, 0xfa, 0x98,
            0xec, 0x4e, 0x6c, 0x89,
        ]
    }
}
