//! Packed Relative Relocations
//!
//! **File:** `relr.rs`
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

/// Bytes per relocated pointer in 64-bit RELR tables.
const WORD: u64 = 8;

/// Locations covered by one bitmap entry beyond the base location.
const BITMAP_SPAN: u64 = 63;

/// Decodes a 64-bit `DT_RELR` table into the addresses it relocates.
///
/// Even words name an address to relocate; odd words are bitmaps whose bits
/// 1 through 63 mark the following 63 pointer slots.
///
/// # Arguments
///
/// * `words` - Table entries in file order.
///
/// # Returns
///
/// Every relocated address.
#[must_use]
pub fn locations(words: &[u64]) -> Vec<u64> {
    let mut decoder = Decoder::default();
    for word in words {
        decoder.feed(*word);
    }
    decoder.found
}

/// Incremental RELR decoding state.
#[derive(Default)]
struct Decoder {
    /// Address of the next slot a bitmap describes.
    next: u64,

    /// Relocated addresses decoded so far.
    found: Vec<u64>,
}

impl Decoder {
    /// Consumes one table word.
    ///
    /// # Arguments
    ///
    /// * `word` - Address or bitmap entry.
    fn feed(&mut self, word: u64) {
        if word & 1 == 0 {
            self.address(word);
        } else {
            self.bitmap(word);
        }
    }

    /// Records an address entry.
    ///
    /// # Arguments
    ///
    /// * `address` - Address to relocate.
    fn address(&mut self, address: u64) {
        self.found.push(address);
        self.next = address.wrapping_add(WORD);
    }

    /// Records every slot marked in a bitmap entry.
    ///
    /// # Arguments
    ///
    /// * `bitmap` - Bitmap entry with its low bit set.
    fn bitmap(&mut self, bitmap: u64) {
        let base = self.next;
        let marked = (1..=BITMAP_SPAN).filter(|bit| bitmap >> bit & 1 == 1);
        self.found.extend(marked.map(|bit| base.wrapping_add((bit - 1) * WORD)));
        self.next = base.wrapping_add(BITMAP_SPAN * WORD);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies a lone address entry.
    #[test]
    fn test_locations_address_entry() {
        assert_eq!(locations(&[0x1000]), vec![0x1000]);
    }

    /// Verifies that a bitmap continues after its address entry.
    #[test]
    fn test_locations_bitmap_entry() {
        assert_eq!(locations(&[0x1000, 0b1011]), vec![0x1000, 0x1008, 0x1018]);
    }

    /// Verifies that consecutive bitmaps advance by 63 slots.
    #[test]
    fn test_locations_consecutive_bitmaps() {
        let found = locations(&[0x1000, 0b11, 0b11]);
        assert_eq!(found, vec![0x1000, 0x1008, 0x1008 + 63 * 8]);
    }
}
