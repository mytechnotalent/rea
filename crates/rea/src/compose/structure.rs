//! Sections and Mitigations
//!
//! **File:** `structure.rs`
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
    model::{Binary, Protection, Section as BinarySection},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Section table columns.
const LAYOUT_HEADERS: [&str; 6] = [
    "Name",
    "Address",
    "Virtual size",
    "Raw size",
    "Entropy",
    "Access",
];

/// Mitigation table columns.
const MITIGATION_HEADERS: [&str; 3] = ["Mitigation", "Status", "Evidence"];

/// Explanation placed above the mitigation table.
const MITIGATION_NOTE: &str =
    "Observed: each verdict cites the header field or import that decides it.";

/// Warning added when code sections are writable.
const WX_RISK: &str = " Inferred: writable code sections allow self-modifying or unpacked code.";

/// Builds the memory layout and mitigation sections.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// Both sections for executables, otherwise none.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let binary = analysis.binary.as_ref();
    binary.map_or_else(Vec::new, |b| vec![layout(b), mitigations(b)])
}

/// Builds the memory layout section.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The section table with a summary.
fn layout(binary: &Binary) -> Section {
    let rows = binary.sections.iter().map(section_row).collect();
    let table = text::table(&LAYOUT_HEADERS, rows);
    let section = Section::new("Sections and Memory Layout");
    let section = section.with(Block::Paragraph(layout_note(binary)));
    section.with(Block::Table(table))
}

/// Summarizes the section layout, flagging writable code.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One paragraph.
fn layout_note(binary: &Binary) -> String {
    let count = |f: fn(&BinarySection) -> bool| binary.sections.iter().filter(|s| f(s)).count();
    let (code, writable) = (count(|s| s.executable), count(|s| s.writable));
    let both = count(|s| s.executable && s.writable);
    let total = text::counted(binary.sections.len(), "section");
    let risk = text::when(both > 0, WX_RISK);
    format!("Observed: {total} ({code} executable, {writable} writable, {both} both).{risk}")
}

/// Formats one section table row.
///
/// # Arguments
///
/// * `section` - Binary section.
///
/// # Returns
///
/// The row cells.
fn section_row(section: &BinarySection) -> Vec<String> {
    vec![
        section.name.clone(),
        text::hex(section.address),
        text::hex(section.virtual_size),
        text::hex(section.file_size),
        text::entropy(section.entropy),
        access(section),
    ]
}

/// Formats memory access rights.
///
/// # Arguments
///
/// * `section` - Binary section.
///
/// # Returns
///
/// `R`, `W`, and `X` flags, such as `R-X`.
fn access(section: &BinarySection) -> String {
    let write = flag(section.writable, 'W');
    let execute = flag(section.executable, 'X');
    format!("R{write}{execute}")
}

/// Shows a permission letter or a dash.
///
/// # Arguments
///
/// * `granted` - Whether the permission is granted.
/// * `letter` - Permission letter.
///
/// # Returns
///
/// `letter` when granted, otherwise `-`.
fn flag(granted: bool, letter: char) -> char {
    if granted {
        letter
    } else {
        '-'
    }
}

/// Builds the security mitigation section.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The mitigation table.
fn mitigations(binary: &Binary) -> Section {
    let rows = binary.protections.iter().map(protection_row).collect();
    let table = text::table(&MITIGATION_HEADERS, rows);
    let section = Section::new("Security Mitigations");
    let section = section.with(text::paragraph(MITIGATION_NOTE));
    section.with(Block::Table(table))
}

/// Formats one mitigation row.
///
/// # Arguments
///
/// * `protection` - Mitigation verdict.
///
/// # Returns
///
/// The row cells.
fn protection_row(protection: &Protection) -> Vec<String> {
    let status = if protection.enabled {
        "Enabled"
    } else {
        "Missing"
    };
    let (name, evidence) = (protection.name.clone(), protection.evidence.clone());
    vec![name, status.to_string(), evidence]
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies access flag formatting.
    #[test]
    fn test_access_flags() {
        let code = BinarySection::mapped(".text", 0, 1).with_access(true, false);
        assert_eq!(access(&code), "R-X");
    }

    /// Verifies that data files produce no layout sections.
    #[test]
    fn test_sections_skip_data() {
        let analysis = rea_engine::analyze("x.bin", b"abc");
        assert_eq!(sections(&analysis).len(), 0);
    }

    /// Verifies that writable code is flagged.
    #[test]
    fn test_layout_note_flags_writable_code() {
        let section = BinarySection::mapped("x", 0, 1).with_access(true, true);
        let binary = Binary { sections: vec![section], ..Binary::default() };
        assert!(layout_note(&binary).contains("Inferred"));
    }
}
