//! Dynamic Analysis
//!
//! **File:** `dynamic.rs`
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
use rea_engine::{Analysis, Emulation, Finding, Stop};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Explanation placed above the emulation findings.
const INTRO: &str = "Observed: the entry point was executed in a sandboxed, pure-software \
p-code interpreter. The target's own instructions ran without any native execution on the host, \
so this reflects the program's real behavior rather than its static form.";

/// Builds the dynamic analysis section when emulation ran.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// One section when emulation executed instructions, otherwise nothing.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    match &analysis.emulation {
        Some(emulation) if emulation.steps > 0 => vec![section(emulation)],
        _ => Vec::new(),
    }
}

/// Builds the dynamic analysis section for an emulation outcome.
///
/// # Arguments
///
/// * `emulation` - Entry-point emulation outcome.
///
/// # Returns
///
/// The dynamic analysis section.
fn section(emulation: &Emulation) -> Section {
    let section = Section::new("Dynamic Analysis (Emulated)").with(text::paragraph(INTRO));
    let section = section.with(Block::Paragraph(summary(emulation)));
    let section = with_findings(section, "Recovered Runtime Strings", &emulation.found);
    with_findings(section, "Decrypted Strings", &emulation.strings)
}

/// Appends recovered strings when any were found.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `title` - Subheading title.
/// * `found` - Recovered strings.
///
/// # Returns
///
/// The section, extended when findings exist.
fn with_findings(section: Section, title: &str, found: &[Finding]) -> Section {
    if found.is_empty() {
        return section;
    }
    let section = section.with(text::subheading(title));
    section.with(Block::Bullets(found.iter().map(finding_line).collect()))
}

/// Formats one recovered string as a bullet line.
///
/// # Arguments
///
/// * `finding` - Recovered string.
///
/// # Returns
///
/// The bullet text with the token and its address.
fn finding_line(finding: &Finding) -> String {
    format!("`{}` at {}", finding.text, text::hex(finding.address))
}

/// Summarizes an emulation outcome in one paragraph.
///
/// # Arguments
///
/// * `emulation` - Entry-point emulation outcome.
///
/// # Returns
///
/// The summary paragraph text.
fn summary(emulation: &Emulation) -> String {
    let count = text::counted(usize::try_from(emulation.steps).unwrap_or(0), "instruction");
    let entry = text::hex(emulation.entry);
    let exit = text::hex(emulation.exit);
    let reason = reason(emulation.stop);
    format!("Derived: emulation from {entry} executed {count} before stopping at {exit}; {reason}.")
}

/// Describes why an emulation run stopped.
///
/// # Arguments
///
/// * `stop` - Stop reason.
///
/// # Returns
///
/// A human-readable explanation.
fn reason(stop: Stop) -> &'static str {
    match stop {
        Stop::LeftMappedCode => "control left the mapped image on a return or external call",
        Stop::Budget => "the instruction budget was reached",
        Stop::DecodeError => "an instruction could not be decoded",
        Stop::NoArch => "no emulator was available for the architecture",
        Stop::TooLarge => "the image was too large to emulate",
        Stop::Faulted => "emulation stopped after an internal fault",
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies an emulation that executed instructions yields a section.
    #[test]
    fn test_section_present_when_steps_run() {
        let emulation = emulation_fixture();
        let analysis = with_emulation(Some(emulation));
        assert_eq!(sections(&analysis).len(), 1);
    }

    /// Verifies no section appears when emulation did not run.
    #[test]
    fn test_section_absent_without_emulation() {
        assert_eq!(sections(&with_emulation(None)).len(), 0);
    }

    /// Builds an emulation outcome with one recovered string.
    ///
    /// # Returns
    ///
    /// The fixture emulation.
    fn emulation_fixture() -> Emulation {
        let found = vec![Finding { address: 0x2100, text: "Sample{placeholder}".to_string() }];
        let strings = vec![Finding { address: 0x2200, text: "decrypted-config".to_string() }];
        Emulation { entry: 0x1000, steps: 12, exit: 0x2000, stop: Stop::Budget, found, strings }
    }

    /// Builds an analysis carrying a chosen emulation outcome.
    ///
    /// # Arguments
    ///
    /// * `emulation` - Emulation to attach.
    ///
    /// # Returns
    ///
    /// The analysis.
    fn with_emulation(emulation: Option<Emulation>) -> Analysis {
        let mut analysis = rea_engine::analyze("x", b"data");
        analysis.emulation = emulation;
        analysis
    }
}
