//! Vulnerabilities
//!
//! **File:** `vulns.rs`
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
use rea_engine::{
    vulns::{self, Vulnerability},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Vulnerability table columns.
const HEADERS: [&str; 5] = ["Severity", "CWE", "Title", "Function", "Address"];

/// Explains the vulnerability table.
const NOTE: &str = "Derived: dangerous API sinks reachable from recovered code. Confirm \
reachability and argument bounds before reporting.";

/// Builds the vulnerability section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The section, or an empty vector when nothing matched.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let findings = vulns::audit(analysis);
    if findings.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<String>> = findings.iter().map(row).collect();
    let note = text::paragraph(NOTE);
    let table = Block::Table(text::table(&HEADERS, rows));
    vec![Section::new("Vulnerabilities").with(note).with(table)]
}

/// Formats one finding row.
///
/// # Arguments
///
/// * `item` - Vulnerability finding.
///
/// # Returns
///
/// The row cells.
fn row(item: &Vulnerability) -> Vec<String> {
    vec![
        item.severity.to_string(),
        item.cwe.clone(),
        item.title.clone(),
        item.function.clone(),
        text::hex(item.address),
    ]
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that clean data yields no vulnerability section.
    #[test]
    fn test_sections_skip_clean_data() {
        let analysis = rea_engine::analyze("blob", b"no sinks here");
        assert!(sections(&analysis).is_empty());
    }
}
