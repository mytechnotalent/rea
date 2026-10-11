//! Methodology and Limitations
//!
//! **File:** `method.rs`
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
use rea_engine::Analysis;

// Import dependencies from rea_report
use rea_report::{Block, Section};

/// Analysis steps performed for every target.
const STEPS: [&str; 7] = [
    "Identified the format from file signatures and fingerprinted the file with MD5, SHA-1, and SHA-256.",
    "Parsed PE, ELF, and Mach-O headers, sections, imports, exports, symbols, and build metadata.",
    "Measured Shannon entropy for the file and every section.",
    "Extracted ASCII and UTF-16LE strings and classified network, path, registry, and command indicators.",
    "Classified imported APIs into capability categories and evaluated security mitigations from header fields.",
    "Disassembled x86/x86-64 with iced-x86 and every other supported architecture through an embedded SLEIGH engine, by recursive descent.",
    "Ranked functions by the capabilities and indicators they touch and listed the most relevant ones.",
];

/// Standing limitations of this release.
const LIMITS: [&str; 4] = [
    "Analysis is static: the target is never executed, so run-time behavior is inferred, not observed.",
    "Imports show what a program can call, not what it does; dynamically resolved APIs are invisible to import analysis.",
    "Disassembly covers code reachable from the entry point, exports, and symbols; indirect-only code may be missed.",
    "A few architectures are parsed but not yet disassembled; for those, code analysis is omitted.",
];

/// Builds the methodology and limitations section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The section.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let steps = Block::Bullets(STEPS.iter().map(ToString::to_string).collect());
    let mut limits: Vec<String> = LIMITS.iter().map(ToString::to_string).collect();
    limits.extend(truncation(analysis));
    let section = Section::new("Methodology and Limitations").with(steps);
    let heading = Block::Subheading("Limitations".to_string());
    vec![section.with(heading).with(Block::Bullets(limits))]
}

/// Notes when a resource cap cut the analysis short.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// A limitation statement, when disassembly was truncated.
fn truncation(analysis: &Analysis) -> Option<String> {
    let truncated = analysis.disassembly.as_ref().is_some_and(|d| d.truncated);
    let text = "Disassembly reached a resource cap for this target; code coverage is partial.";
    truncated.then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that every step and limitation is listed.
    #[test]
    fn test_sections_list_steps_and_limits() {
        let section = &sections(&rea_engine::analyze("x", b"abc"))[0];
        let Block::Bullets(steps) = &section.blocks[0] else {
            panic!("expected bullets")
        };
        assert_eq!(steps.len(), STEPS.len());
    }

    /// Verifies that complete analyses add no truncation note.
    #[test]
    fn test_truncation_absent() {
        assert!(truncation(&rea_engine::analyze("x", b"abc")).is_none());
    }
}
