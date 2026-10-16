//! Cryptographic Constants
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

// Import dependencies from rea_engine
use rea_engine::crypto::{self, CryptoHit};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Cryptographic constant table columns.
const HEADERS: [&str; 3] = ["Address", "Constant", "Kind"];

/// Explains the cryptographic constant table.
const NOTE: &str = "Derived: cryptographic constants identified by table signature in the target.";

/// Builds the cryptographic constants section.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The section, or an empty vector when nothing matched.
pub fn sections(bytes: &[u8]) -> Vec<Section> {
    let hits = crypto::detect(bytes);
    if hits.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<String>> = hits.iter().map(row).collect();
    let note = text::paragraph(NOTE);
    let table = Block::Table(text::table(&HEADERS, rows));
    vec![Section::new("Cryptographic Constants").with(note).with(table)]
}

/// Formats one constant row.
///
/// # Arguments
///
/// * `hit` - Detected constant.
///
/// # Returns
///
/// The row cells.
fn row(hit: &CryptoHit) -> Vec<String> {
    vec![text::hex(hit.address), hit.name.clone(), hit.kind.clone()]
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that clean data yields no crypto section.
    #[test]
    fn test_sections_skip_clean_data() {
        assert_eq!(sections(b"no crypto constants here at all").len(), 0);
    }
}
