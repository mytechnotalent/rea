//! Shannon Entropy
//!
//! **File:** `entropy.rs`
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
use crate::numeric::ratio_value;

/// Computes the Shannon entropy of a buffer in bits per byte.
///
/// # Arguments
///
/// * `bytes` - Data to measure.
///
/// # Returns
///
/// A value from `0.0` (constant) to `8.0` (uniformly random); `0.0` when empty.
#[must_use]
pub fn shannon(bytes: &[u8]) -> f64 {
    let mut counts = [0usize; 256];
    for byte in bytes {
        counts[usize::from(*byte)] += 1;
    }
    let total = ratio_value(bytes.len());
    let used = counts.iter().filter(|c| **c > 0);
    used.map(|c| information(ratio_value(*c) / total)).sum()
}

/// Returns the information contribution of one symbol probability.
///
/// # Arguments
///
/// * `p` - Symbol probability in `(0, 1]`.
///
/// # Returns
///
/// `-p * log2(p)`.
fn information(p: f64) -> f64 {
    -p * p.log2()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that constant data has zero entropy.
    #[test]
    fn test_shannon_constant_is_zero() {
        assert!(shannon(&[7u8; 64]).abs() < 1e-9);
        assert!(shannon(&[]).abs() < 1e-9);
    }

    /// Verifies that all 256 byte values once yield eight bits.
    #[test]
    fn test_shannon_uniform_is_eight() {
        let data: Vec<u8> = (0..=255).collect();
        assert!((shannon(&data) - 8.0).abs() < 1e-9);
    }

    /// Verifies that two equally likely symbols yield one bit.
    #[test]
    fn test_shannon_two_symbols_is_one() {
        assert!((shannon(&[0, 1, 0, 1]) - 1.0).abs() < 1e-9);
    }
}
