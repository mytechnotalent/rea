//! Numeric Conversions
//!
//! **File:** `numeric.rs`
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

/// Converts a count to `f64` for statistics, saturating at `u32::MAX`.
///
/// # Arguments
///
/// * `count` - Count to convert.
///
/// # Returns
///
/// The count as a float; inputs beyond `u32::MAX` saturate.
pub fn ratio_value(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Converts a 64-bit file quantity to `usize`, saturating on narrow targets.
///
/// # Arguments
///
/// * `value` - Offset or size to convert.
///
/// # Returns
///
/// The value as `usize`, or `usize::MAX` when it does not fit.
pub fn to_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Converts a `usize` quantity to `u64`.
///
/// # Arguments
///
/// * `value` - Offset or size to convert.
///
/// # Returns
///
/// The value as `u64`, saturating in the impossible overflow case.
pub fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Returns the bytes in `[start, start + len)`, clamped to the buffer.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `start` - First offset.
/// * `len` - Requested length.
///
/// # Returns
///
/// The available part of the range; empty when `start` is out of bounds.
pub fn window(bytes: &[u8], start: u64, len: u64) -> &[u8] {
    let start = to_usize(start).min(bytes.len());
    let end = start.saturating_add(to_usize(len)).min(bytes.len());
    &bytes[start..end]
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that counts convert exactly in the common range.
    #[test]
    fn test_ratio_value_converts() {
        assert!((ratio_value(10) - 10.0).abs() < f64::EPSILON);
    }

    /// Verifies that windows clamp to the buffer bounds.
    #[test]
    fn test_window_clamps() {
        let data = [1u8, 2, 3, 4];
        assert_eq!(window(&data, 1, 2), &[2, 3]);
        assert_eq!(window(&data, 3, 10), &[4]);
        assert_eq!(window(&data, 9, 1).len(), 0);
    }

    /// Verifies lossless round trips between offset types.
    #[test]
    fn test_offset_conversions_round_trip() {
        assert_eq!(to_usize(to_u64(77)), 77);
    }
}
