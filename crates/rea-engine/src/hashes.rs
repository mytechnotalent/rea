//! Cryptographic Fingerprints
//!
//! **File:** `hashes.rs`
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

// Import dependencies from md5
use md5::Md5;

// Import dependencies from sha1
use sha1::Sha1;

// Import dependencies from sha2
use sha2::{Digest, Sha256};

/// Lowercase hexadecimal digits.
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Standard cryptographic fingerprints of a file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hashes {
    /// MD5 digest in lowercase hex.
    pub md5: String,

    /// SHA-1 digest in lowercase hex.
    pub sha1: String,

    /// SHA-256 digest in lowercase hex.
    pub sha256: String,
}

/// Computes MD5, SHA-1, and SHA-256 of a buffer.
///
/// # Arguments
///
/// * `bytes` - Data to fingerprint.
///
/// # Returns
///
/// All three digests as lowercase hex.
#[must_use]
pub fn compute(bytes: &[u8]) -> Hashes {
    Hashes {
        md5: md5_hex(bytes),
        sha1: hex(&Sha1::digest(bytes)),
        sha256: hex(&Sha256::digest(bytes)),
    }
}

/// Computes the MD5 digest of a buffer.
///
/// # Arguments
///
/// * `bytes` - Data to fingerprint.
///
/// # Returns
///
/// The digest as lowercase hex.
#[must_use]
pub fn md5_hex(bytes: &[u8]) -> String {
    hex(&Md5::digest(bytes))
}

/// Encodes bytes as lowercase hexadecimal.
///
/// # Arguments
///
/// * `bytes` - Bytes to encode.
///
/// # Returns
///
/// Two hex digits per input byte.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let digits = bytes.iter().flat_map(|b| [b >> 4, b & 0x0f]);
    digits.map(|d| char::from(HEX[usize::from(d)])).collect()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the digests of the empty input against published values.
    #[test]
    fn test_compute_empty_vectors() {
        let hashes = compute(b"");
        assert_eq!(hashes.md5, "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hashes.sha1, "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        let sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(hashes.sha256, sha);
    }

    /// Verifies hex encoding of boundary byte values.
    #[test]
    fn test_hex_encodes_bytes() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }
}
