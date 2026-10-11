//! Report Composition
//!
//! **File:** `mod.rs`
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

/// Code structure and key functions.
mod code;

/// Dynamic analysis section built from entry-point emulation.
mod dynamic;

/// Identity and fingerprints.
mod identity;

/// Imports, exports, and capabilities.
mod imports;

/// Firmware filesystem and artifacts.
mod firmware;

/// Methodology and limitations.
mod method;

/// Packing assessment.
mod packing;

/// Sections and mitigations.
mod structure;

/// Strings and indicators.
mod strings;

/// Executive summary.
mod summary;

/// Shared formatting helpers.
pub mod text;

// Import dependencies from rea_engine
use rea_engine::Analysis;

// Import dependencies from rea_report
use rea_report::{Block, Document, Section};

/// Report section builders, in reading order.
const SECTIONS: [fn(&Analysis) -> Vec<Section>; 10] = [
    summary::sections,
    identity::sections,
    structure::sections,
    firmware::sections,
    imports::sections,
    strings::sections,
    code::sections,
    dynamic::sections,
    packing::sections,
    method::sections,
];

/// Appends a paragraph or subheading chunk.
///
/// # Arguments
///
/// * `section` - Destination section.
/// * `chunk` - Text chunk.
///
/// # Returns
///
/// Updated section.
fn append_chunk(section: Section, chunk: &str) -> Section {
    if chunk.is_empty() {
        section
    } else if chunk.starts_with('#') {
        let title = chunk.trim_start_matches('#').trim();
        section.with(text::subheading(title))
    } else {
        section.with(Block::Paragraph(chunk.to_string()))
    }
}

/// Parses the assessment markdown text into a report section.
///
/// # Arguments
///
/// * `text` - Assessment markdown.
///
/// # Returns
///
/// Populated section.
fn assessment_section(text: &str) -> Section {
    let mut section = Section::new("Autonomous Security Assessment");
    for chunk in text.split("\n\n") {
        section = append_chunk(section, chunk.trim());
    }
    section
}

/// Composes the complete report for one analysis.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `assessment` - AI analyst narrative, when available.
///
/// # Returns
///
/// The format-neutral report document.
#[must_use]
pub fn build(analysis: &Analysis, assessment: Option<&str>) -> Document {
    let title = format!("Reverse Engineering Report: {}", analysis.name);
    let mut document = Document::new(&title, &subtitle(analysis));
    document.sections = SECTIONS.iter().flat_map(|b| b(analysis)).collect();
    if let Some(text) = assessment {
        document.sections.insert(1, assessment_section(text));
    }
    document
}

/// Describes the target on the cover.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// Format, size, and SHA-256.
fn subtitle(analysis: &Analysis) -> String {
    let size = text::size(analysis.size);
    let sha = &analysis.hashes.sha256;
    format!("{} | {size} | SHA-256 {sha}", analysis.format.label())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that unknown data still yields a complete report.
    #[test]
    fn test_build_unknown_data() {
        let analysis = rea_engine::analyze("blob.bin", b"plain text with http://example.test");
        let document = build(&analysis, None);
        assert!(document.title.ends_with("blob.bin"));
        assert!(document.sections.len() >= 4);
    }

    /// Verifies that a real executable yields every section.
    #[test]
    fn test_build_native_executable() {
        let path = std::env::current_exe().unwrap();
        let analysis = rea_engine::analyze_path(&path).unwrap();
        let headings: Vec<String> =
            build(&analysis, None).sections.into_iter().map(|s| s.heading).collect();
        assert!(headings.iter().any(|h| h == "Key Functions"));
    }
}
