//! Executive Summary
//!
//! **File:** `summary.rs`
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
    capabilities::Capability, detect::Format, insights::KeyFunction, iocs::Indicator,
    model::Binary, Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Explanation of the evidence labels used throughout the report.
const EVIDENCE_NOTE: &str = "Evidence labels: Observed facts are read directly from the file; \
Derived facts are computed deterministically from observed data; Inferred statements are \
interpretations that an analyst should confirm.";

/// Finding used when no capability-relevant imports exist.
const NO_CAPABILITIES: &str = "Derived: no capability-relevant imports were found.";

/// Finding used when every mitigation is enabled.
const ALL_ENABLED: &str = "Observed: every checked mitigation is enabled.";

/// Reason attached to the entry-point function.
const ENTRY_REASON: &str = "Program entry point";

/// Summary findings, in reading order.
const FINDINGS: [fn(&Analysis) -> Option<String>; 6] = [
    capability_finding,
    packing_finding,
    protection_finding,
    indicator_finding,
    code_finding,
    key_finding,
];

/// Builds the executive summary.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The summary section.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let findings = FINDINGS.iter().filter_map(|f| f(analysis));
    let section = Section::new("Executive Summary");
    let section = section.with(Block::Paragraph(overview(analysis)));
    let section = section.with(Block::Bullets(findings.collect()));
    vec![section.with(text::paragraph(EVIDENCE_NOTE))]
}

/// Describes what the target is.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// One overview paragraph.
fn overview(analysis: &Analysis) -> String {
    match (&analysis.binary, &analysis.load_error) {
        (Some(binary), _) => native_overview(&analysis.name, binary, analysis.size),
        (None, Some(error)) => broken_overview(analysis, error),
        (None, None) => data_overview(analysis),
    }
}

/// Describes a parsed executable.
///
/// # Arguments
///
/// * `name` - Target name.
/// * `binary` - Parsed executable.
/// * `size` - File size.
///
/// # Returns
///
/// The overview sentence.
fn native_overview(name: &str, binary: &Binary, size: u64) -> String {
    let kind = binary.kind.to_lowercase();
    let platform = platform(binary.format);
    let what = format!("{}-bit {platform} {kind}", binary.bits);
    let (arch, container) = (&binary.arch_name, binary.format.label());
    let (article, size) = (text::article(container), text::size(size));
    format!("Observed: {name} is a {what} for {arch}, {article} {container} file of {size}.")
}

/// Names the operating system family a container format targets.
///
/// # Arguments
///
/// * `format` - Container format.
///
/// # Returns
///
/// `Windows`, `Unix-like`, `Apple`, or `native`.
fn platform(format: Format) -> &'static str {
    match format {
        Format::Pe => "Windows",
        Format::Elf => "Unix-like",
        Format::MachO | Format::MachOUniversal => "Apple",
        _ => "native",
    }
}

/// Describes a file that claims a native format but failed to parse.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `error` - Parser failure.
///
/// # Returns
///
/// The overview sentence.
fn broken_overview(analysis: &Analysis, error: &str) -> String {
    let (name, format) = (&analysis.name, analysis.format.label());
    let size = text::size(analysis.size);
    format!("Observed: {name} has a {format} signature ({size}) but is malformed: {error}.")
}

/// Describes non-executable data.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The overview sentence.
fn data_overview(analysis: &Analysis) -> String {
    let (name, format) = (&analysis.name, analysis.format.label());
    let size = text::size(analysis.size);
    format!("Observed: {name} is {format} ({size}); native code analysis does not apply.")
}

/// Summarizes import-based capabilities.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when the target is an executable.
fn capability_finding(analysis: &Analysis) -> Option<String> {
    analysis.binary.as_ref()?;
    let names = capability_names(&analysis.capabilities);
    let found = format!("Derived: imports support {}.", text::listed(&names, 8));
    let found = (!names.is_empty()).then_some(found);
    Some(text::or(found, NO_CAPABILITIES))
}

/// Lists capability category names.
///
/// # Arguments
///
/// * `capabilities` - Classified capabilities.
///
/// # Returns
///
/// The category names.
fn capability_names(capabilities: &[Capability]) -> Vec<String> {
    capabilities.iter().map(|c| c.category.to_string()).collect()
}

/// Summarizes the packing verdict.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when there is an executable or any signal.
fn packing_finding(analysis: &Analysis) -> Option<String> {
    let packing = &analysis.packing;
    let (verdict, score) = (text::packing_verdict(packing.likely), packing.score);
    let found = format!("Inferred: the file is {verdict} (packing score {score}).");
    let relevant = analysis.binary.is_some() || score > 0;
    relevant.then_some(found)
}

/// Summarizes missing mitigations.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when the target is an executable.
fn protection_finding(analysis: &Analysis) -> Option<String> {
    let binary = analysis.binary.as_ref()?;
    let missing = missing_protections(binary).join(", ");
    let found = format!("Observed: missing mitigations: {missing}.");
    let found = (!missing.is_empty()).then_some(found);
    Some(text::or(found, ALL_ENABLED))
}

/// Lists disabled mitigations.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The names of mitigations that are not enabled.
fn missing_protections(binary: &Binary) -> Vec<String> {
    let missing = binary.protections.iter().filter(|p| !p.enabled);
    missing.map(|p| p.name.clone()).collect()
}

/// Summarizes extracted indicators.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when any indicator was found.
fn indicator_finding(analysis: &Analysis) -> Option<String> {
    let count = analysis.indicators.len();
    let all = text::counted(count, "indicator");
    let urls = url_note(&analysis.indicators);
    let found = format!("Observed: {all} extracted{urls}.");
    (count > 0).then_some(found)
}

/// Mentions how many indicators are URLs.
///
/// # Arguments
///
/// * `indicators` - Extracted indicators.
///
/// # Returns
///
/// `, including N URLs`, or nothing when there are none.
fn url_note(indicators: &[Indicator]) -> String {
    let urls = indicators.iter().filter(|i| i.kind == "URL").count();
    let note = format!(", including {}", text::counted(urls, "URL"));
    if urls == 0 {
        String::new()
    } else {
        note
    }
}

/// Summarizes recovered code structure.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when disassembly ran.
fn code_finding(analysis: &Analysis) -> Option<String> {
    let code = analysis.disassembly.as_ref()?;
    let functions = text::counted(code.functions.len(), "function");
    let instructions = text::counted(code.instructions, "instruction");
    Some(format!("Derived: recovered {functions} ({instructions})."))
}

/// Highlights the most relevant function.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The finding, when a key function has a behavioral reason.
fn key_finding(analysis: &Analysis) -> Option<String> {
    let top = top_key(&analysis.key_functions)?;
    let reason = top.reasons.iter().find(|r| *r != ENTRY_REASON)?;
    let (name, address) = (&top.name, text::hex(top.address));
    Some(format!(
        "Inferred: start with {name} at {address}: {reason}."
    ))
}

/// Finds the best-ranked function with a behavioral reason.
///
/// # Arguments
///
/// * `keys` - Ranked key functions.
///
/// # Returns
///
/// The first function with more than one reason.
fn top_key(keys: &[KeyFunction]) -> Option<&KeyFunction> {
    keys.iter().find(|k| k.reasons.len() > 1)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the overview of non-executable data.
    #[test]
    fn test_data_overview() {
        let analysis = rea_engine::analyze("notes.txt", b"just some notes");
        assert!(overview(&analysis).contains("does not apply"));
    }

    /// Verifies the overview of a malformed executable.
    #[test]
    fn test_broken_overview() {
        let mut bad = vec![0u8; 0x44];
        bad[0] = b'M';
        bad[1] = b'Z';
        bad[0x3c] = 0x40;
        bad[0x40..0x44].copy_from_slice(b"PE\x00\x00");
        let analysis = rea_engine::analyze("bad.exe", &bad);
        assert!(overview(&analysis).contains("malformed"));
    }

    /// Verifies that executable-only findings are skipped for data.
    #[test]
    fn test_findings_skip_for_data() {
        let analysis = rea_engine::analyze("notes.txt", b"just some notes");
        assert!(capability_finding(&analysis).is_none());
        assert!(code_finding(&analysis).is_none());
    }

    /// Verifies platform naming.
    #[test]
    fn test_platform_names() {
        assert_eq!(platform(Format::Pe), "Windows");
        assert_eq!(platform(Format::Zip), "native");
    }
}
