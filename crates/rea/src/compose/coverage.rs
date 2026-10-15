//! Coverage and Limitations
//!
//! **File:** `coverage.rs`
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
    regions::{self, RegionKind},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::Section;

// Import crate::compose::text
use crate::compose::text;

/// Note appended to the coverage section describing the evidence bundle.
const BUNDLE_NOTE: &str = "An audit-grade evidence bundle (`re-<name>.evidence.json`) lists every \
finding with a stable content-addressed ID and can be checked offline with `rea verify`.";

/// Builds the coverage and limitations section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The section.
pub fn sections(analysis: &Analysis, bytes: &[u8]) -> Vec<Section> {
    let note = coverage_note(analysis, bytes);
    vec![Section::new("Coverage & Limitations").with(text::paragraph(&note))]
}

/// Composes the coverage and limitations note.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note text.
fn coverage_note(analysis: &Analysis, bytes: &[u8]) -> String {
    let mut items = Vec::new();
    push_truncation(&mut items, analysis);
    push_encrypted(&mut items, bytes);
    push_header(&mut items, analysis);
    format!("{} {} {BUNDLE_NOTE}", analyzed(analysis), summarize(&items))
}

/// Describes what was analyzed.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The coverage summary.
fn analyzed(analysis: &Analysis) -> String {
    let functions = analysis.disassembly.as_ref().map_or(0, |d| d.functions.len());
    let instructions = analysis.disassembly.as_ref().map_or(0, |d| d.instructions);
    format!(
        "Coverage: {} parsed as {}; {functions} functions and {instructions} instructions recovered across {} strings.",
        analysis.name,
        analysis.format.label(),
        analysis.strings.len()
    )
}

/// Records a partial-disassembly limitation.
///
/// # Arguments
///
/// * `items` - Accumulated limitations.
/// * `analysis` - Completed analysis.
fn push_truncation(items: &mut Vec<String>, analysis: &Analysis) {
    if analysis.disassembly.as_ref().is_some_and(|d| d.truncated) {
        items
            .push("disassembly stopped at a resource cap, so code coverage is partial".to_string());
    }
}

/// Records an encrypted-region limitation.
///
/// # Arguments
///
/// * `items` - Accumulated limitations.
/// * `bytes` - Raw target bytes.
fn push_encrypted(items: &mut Vec<String>, bytes: &[u8]) {
    let size = encrypted_span(bytes);
    if size > 0 {
        items.push(format!(
            "{size} bytes of high-entropy (encrypted or compressed) regions were not decrypted"
        ));
    }
}

/// Records a missing-header limitation.
///
/// # Arguments
///
/// * `items` - Accumulated limitations.
/// * `analysis` - Completed analysis.
fn push_header(items: &mut Vec<String>, analysis: &Analysis) {
    if analysis.binary.is_none() {
        items.push(
            "no executable header was parsed, so imports and mitigations are unavailable"
                .to_string(),
        );
    }
}

/// Summarizes the collected limitations.
///
/// # Arguments
///
/// * `items` - Accumulated limitations.
///
/// # Returns
///
/// The limitation summary.
fn summarize(items: &[String]) -> String {
    if items.is_empty() {
        return "Limitations: none detected; findings are complete for the surfaces above."
            .to_string();
    }
    format!("Limitations: {}.", items.join("; "))
}

/// Sums the sizes of encrypted and compressed regions.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The total size in bytes.
fn encrypted_span(bytes: &[u8]) -> usize {
    let opaque = |kind: RegionKind| matches!(kind, RegionKind::Encrypted | RegionKind::Compressed);
    regions::segment(bytes).iter().filter(|r| opaque(r.kind)).map(|r| r.size).sum()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that plain data yields a coverage section.
    #[test]
    fn test_coverage_section_renders() {
        let analysis = rea_engine::analyze("blob.bin", b"plain data");
        assert_eq!(sections(&analysis, b"plain data").len(), 1);
    }
}
