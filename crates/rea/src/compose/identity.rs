//! Identity and Fingerprints
//!
//! **File:** `identity.rs`
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
use rea_engine::{model::Binary, Analysis};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// One labeled value read from a source of type `T`.
type Row<T> = (&'static str, fn(&T) -> String);

/// File-level identity rows.
const FILE_ROWS: [Row<Analysis>; 7] = [
    ("File name", |a| a.name.clone()),
    ("Size", |a| text::size(a.size)),
    ("Format", |a| a.format.label().to_string()),
    ("Entropy (bits/byte)", |a| text::entropy(a.entropy)),
    ("MD5", |a| a.hashes.md5.clone()),
    ("SHA-1", |a| a.hashes.sha1.clone()),
    ("SHA-256", |a| a.hashes.sha256.clone()),
];

/// Executable identity rows.
const BINARY_ROWS: [Row<Binary>; 5] = [
    ("Image kind", |b| b.kind.clone()),
    ("Architecture", |b| b.arch_name.clone()),
    ("Address width", |b| format!("{} bits", b.bits)),
    ("Entry point", |b| text::hex(b.entry)),
    ("Image base", |b| text::hex(b.image_base)),
];

/// Builds the identity and fingerprint section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The identity section.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let rows = identity_rows(analysis);
    let table = Block::Table(text::pairs(["Property", "Value"], rows));
    let section = Section::new("Identity and Fingerprints").with(table);
    vec![with_facts(section, analysis.binary.as_ref())]
}

/// Collects every identity row.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// Label and value pairs.
fn identity_rows(analysis: &Analysis) -> Vec<(String, String)> {
    let mut rows = labeled(&FILE_ROWS, analysis);
    rows.extend(analysis.imphash.iter().map(|h| imphash_row(h)));
    let binary = analysis.binary.as_ref();
    let extra = binary.map(|b| labeled(&BINARY_ROWS, b));
    rows.extend(extra.unwrap_or_default());
    rows
}

/// Reads labeled rows from a source.
///
/// # Arguments
///
/// * `table` - Labels and readers.
/// * `source` - Value to read from.
///
/// # Returns
///
/// Label and value pairs.
fn labeled<T>(table: &[Row<T>], source: &T) -> Vec<(String, String)> {
    table.iter().map(|(label, read)| (label.to_string(), read(source))).collect()
}

/// Formats the import hash row.
///
/// # Arguments
///
/// * `hash` - Import hash.
///
/// # Returns
///
/// The label and value pair.
fn imphash_row(hash: &str) -> (String, String) {
    ("Import hash (imphash)".to_string(), hash.to_string())
}

/// Appends the executable's metadata facts.
///
/// # Arguments
///
/// * `section` - Identity section.
/// * `binary` - Parsed executable, when available.
///
/// # Returns
///
/// The section with a facts table when facts exist.
fn with_facts(section: Section, binary: Option<&Binary>) -> Section {
    let facts = binary.map(|b| b.facts.as_slice()).unwrap_or_default();
    if facts.is_empty() {
        return section;
    }
    let rows = facts.iter().map(|f| (f.label.clone(), f.value.clone()));
    let table = text::pairs(["Fact", "Value"], rows.collect());
    let section = section.with(text::subheading("Build and Metadata Facts"));
    section.with(Block::Table(table))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that fingerprints appear for any data.
    #[test]
    fn test_identity_rows_include_hashes() {
        let analysis = rea_engine::analyze("x.bin", b"abc");
        let rows = identity_rows(&analysis);
        assert!(rows.iter().any(|(label, _)| label == "SHA-256"));
    }

    /// Verifies that sections without facts gain no facts table.
    #[test]
    fn test_with_facts_skips_empty() {
        let section = with_facts(Section::new("S"), None);
        assert!(section.blocks.is_empty());
    }
}
