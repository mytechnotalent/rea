//! Capabilities
//!
//! **File:** `rules.rs`
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
    rules::{self, RuleMatch},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Capability table columns.
const HEADERS: [&str; 4] = ["Capability", "Namespace", "Scope", "ATT&CK / MBC"];

/// Explains the capability table.
const NOTE: &str = "Derived: capabilities matched from the embedded rule pack against imports, \
strings, and recovered instructions.";

/// Builds the capability section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The section, or an empty vector when nothing matched.
pub fn sections(analysis: &Analysis, bytes: &[u8]) -> Vec<Section> {
    let found = rules::evaluate(analysis, bytes);
    if found.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<String>> = found.iter().map(row).collect();
    let note = text::paragraph(NOTE);
    let table = Block::Table(text::table(&HEADERS, rows));
    vec![Section::new("Detected Capabilities").with(note).with(table)]
}

/// Formats one capability row.
///
/// # Arguments
///
/// * `item` - Matched capability.
///
/// # Returns
///
/// The row cells.
fn row(item: &RuleMatch) -> Vec<String> {
    vec![
        item.name.clone(),
        item.namespace.clone(),
        item.scope.to_string(),
        techniques(item),
    ]
}

/// Joins a match's technique labels.
///
/// # Arguments
///
/// * `item` - Matched capability.
///
/// # Returns
///
/// The comma-separated labels, or `none`.
fn techniques(item: &RuleMatch) -> String {
    if item.techniques.is_empty() {
        "none".to_string()
    } else {
        item.techniques.join("; ")
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that clean data yields no capability section.
    #[test]
    fn test_sections_skip_clean_data() {
        let analysis = rea_engine::analyze("blob", b"no capabilities here");
        assert!(sections(&analysis, &[]).is_empty());
    }

    /// Verifies technique label joining.
    #[test]
    fn test_techniques_join() {
        let item = RuleMatch {
            name: "x".into(),
            namespace: "y".into(),
            scope: "file",
            address: 0,
            techniques: vec!["A".into(), "B".into()],
        };
        assert_eq!(techniques(&item), "A; B");
    }
}
