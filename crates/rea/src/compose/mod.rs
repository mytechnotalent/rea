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

/// Top-level plain-English description of program behavior.
mod program_summary;

/// Sink-based vulnerability findings.
mod vulns;

/// Capability rule matches.
mod rules;

/// Cryptographic constant findings.
mod crypto;

/// Shared formatting helpers.
pub mod text;

/// Executive Threat Dossier.
pub mod dossier;

/// Static Reverse Engineering Record.
pub mod record;

// Import dependencies from rea_engine
use rea_engine::Analysis;

// Import dependencies from rea_report
use rea_report::{Block, Document, Section, Table};

/// Artifact format findings.
mod artifacts;

/// Firmware container findings.
mod container;

/// Coverage and limitations.
mod coverage;

/// Report section builders, in reading order.
const SECTIONS: [fn(&Analysis) -> Vec<Section>; 12] = [
    summary::sections,
    program_summary::sections,
    identity::sections,
    structure::sections,
    firmware::sections,
    imports::sections,
    strings::sections,
    code::sections,
    vulns::sections,
    dynamic::sections,
    packing::sections,
    method::sections,
];

/// Note placed above the deterministic evidence appendix.
const APPENDIX_NOTE: &str = "Deterministic engine findings supporting the assessment above: \
identity, structure, imports, strings, code inventory, capabilities, vulnerabilities, and \
artifacts. This is raw evidence, not analysis.";

/// Extracts the content of a bullet or numbered list line.
///
/// # Arguments
///
/// * `line` - Raw line of text.
///
/// # Returns
///
/// Inner item text when the line starts with a list marker.
fn bullet_line(line: &str) -> Option<&str> {
    let t = line.trim();
    if let Some(r) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")) {
        return Some(r);
    }
    let (n, r) = t.split_once(". ")?;
    (n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty()).then_some(r)
}

/// Appends a line to the list items or extends the prior item.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `line` - Next line of text.
fn push_bullet(items: &mut Vec<String>, line: &str) {
    if let Some(item) = bullet_line(line) {
        items.push(item.to_string());
    } else if let Some(last) = items.last_mut() {
        last.push(' ');
        last.push_str(line.trim());
    }
}

/// Tracks partially assembled report blocks during markdown streaming.
enum PendingBlock {
    /// No block currently open.
    None,
    /// Accumulating multi-line paragraph text.
    Paragraph(String),
    /// Accumulating bullet list items.
    Bullets(Vec<String>),
    /// Accumulating table rows.
    Table(Vec<String>),
    /// Accumulating fenced code lines.
    Code(Vec<String>),
}

/// Splits a markdown table row into trimmed cell values.
///
/// # Arguments
///
/// * `line` - Table row line.
///
/// # Returns
///
/// Extracted cell strings.
fn parse_cells(line: &str) -> Vec<String> {
    line.split('|')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Parses a markdown pipe table into a structured table.
///
/// # Arguments
///
/// * `chunk` - Table text chunk.
///
/// # Returns
///
/// Parsed table when valid headers and rows exist.
fn parse_table(chunk: &str) -> Option<Table> {
    let mut lines = chunk.lines().filter(|l| !l.contains("---"));
    let headers = parse_cells(lines.next()?);
    let rows: Vec<_> = lines.map(parse_cells).filter(|r| !r.is_empty()).collect();
    (!headers.is_empty() && !rows.is_empty()).then_some(Table { headers, rows })
}

/// Converts a completed pending block into a document block.
///
/// # Arguments
///
/// * `pending` - Completed block accumulator.
///
/// # Returns
///
/// Final block when content exists.
fn flush_block(pending: PendingBlock) -> Option<Block> {
    match pending {
        PendingBlock::None => None,
        PendingBlock::Paragraph(t) => (!t.is_empty()).then_some(Block::Paragraph(t)),
        PendingBlock::Bullets(b) => (!b.is_empty()).then_some(Block::Bullets(b)),
        PendingBlock::Table(lines) => parse_table(&lines.join("\n")).map(Block::Table),
        PendingBlock::Code(lines) => (!lines.is_empty()).then_some(Block::Code(lines.join("\n"))),
    }
}

/// Flushes the active pending block onto the section.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
fn push_flush(section: &mut Section, pending: &mut PendingBlock) {
    let old = std::mem::replace(pending, PendingBlock::None);
    if let Some(block) = flush_block(old) {
        section.blocks.push(block);
    }
}

/// Appends a subheading block after flushing prior content.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `t` - Heading line text.
fn handle_heading(section: &mut Section, pending: &mut PendingBlock, t: &str) {
    push_flush(section, pending);
    let title = t.trim_matches(|c| c == '#' || c == ' ');
    section.blocks.push(text::subheading(title));
}

/// Adds a table row line to the active table block.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `line` - Table row line.
fn handle_table(section: &mut Section, pending: &mut PendingBlock, line: &str) {
    if !matches!(pending, PendingBlock::Table(_)) {
        push_flush(section, pending);
        *pending = PendingBlock::Table(Vec::new());
    }
    if let PendingBlock::Table(lines) = pending {
        lines.push(line.to_string());
    }
}

/// Adds a bullet item to the active bullet list block.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `line` - Bullet line.
fn handle_bullet(section: &mut Section, pending: &mut PendingBlock, line: &str) {
    if !matches!(pending, PendingBlock::Bullets(_)) {
        push_flush(section, pending);
        *pending = PendingBlock::Bullets(Vec::new());
    }
    if let PendingBlock::Bullets(items) = pending {
        push_bullet(items, line);
    }
}

/// Appends trimmed text to an active paragraph.
///
/// # Arguments
///
/// * `pending` - Block accumulator.
/// * `trimmed` - Text line.
fn append_paragraph(pending: &mut PendingBlock, trimmed: &str) {
    if let PendingBlock::Paragraph(p) = pending {
        if !p.is_empty() {
            p.push(' ');
        }
        p.push_str(trimmed);
    }
}

/// Ensures the pending block is an open paragraph.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
fn ensure_paragraph(section: &mut Section, pending: &mut PendingBlock) {
    if !matches!(pending, PendingBlock::Paragraph(_)) {
        push_flush(section, pending);
        *pending = PendingBlock::Paragraph(String::new());
    }
}

/// Adds regular text to a paragraph or indented bullet.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `line` - Content line.
fn handle_text(section: &mut Section, pending: &mut PendingBlock, line: &str) {
    if let (PendingBlock::Bullets(items), true) = (&mut *pending, line.starts_with(' ')) {
        push_bullet(items, line);
    } else {
        ensure_paragraph(section, pending);
        append_paragraph(pending, line.trim());
    }
}

/// Routes content lines to table, bullet, or paragraph handlers.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `line` - Line of text.
fn route_content(section: &mut Section, pending: &mut PendingBlock, line: &str) {
    if line.trim().starts_with('|') {
        handle_table(section, pending, line);
    } else if bullet_line(line).is_some() {
        handle_bullet(section, pending, line);
    } else {
        handle_text(section, pending, line);
    }
}

/// Dispatches one markdown line to the appropriate handler.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `line` - Line of text.
fn process_line(section: &mut Section, pending: &mut PendingBlock, line: &str) {
    if line.trim().starts_with("```") {
        toggle_code(section, pending);
    } else if !append_code(pending, line) {
        route_line(section, pending, line.trim(), line);
    }
}

/// Toggles fenced code-block accumulation.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
fn toggle_code(section: &mut Section, pending: &mut PendingBlock) {
    if matches!(pending, PendingBlock::Code(_)) {
        push_flush(section, pending);
    } else {
        push_flush(section, pending);
        *pending = PendingBlock::Code(Vec::new());
    }
}

/// Appends a line to an open code block.
///
/// # Arguments
///
/// * `pending` - Block accumulator.
/// * `line` - Verbatim code line.
///
/// # Returns
///
/// `true` when the line was captured by an open code block.
fn append_code(pending: &mut PendingBlock, line: &str) -> bool {
    if let PendingBlock::Code(lines) = pending {
        lines.push(line.to_string());
        return true;
    }
    false
}

/// Routes a non-code line to a separator, heading, or content handler.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `t` - Trimmed line text.
/// * `line` - Original line text.
fn route_line(section: &mut Section, pending: &mut PendingBlock, t: &str, line: &str) {
    if is_separator(t) {
        push_flush(section, pending);
    } else {
        route_inline(section, pending, t, line);
    }
}

/// Routes a content line to a heading or inline handler.
///
/// # Arguments
///
/// * `section` - Target section.
/// * `pending` - Block accumulator.
/// * `t` - Trimmed line text.
/// * `line` - Original line text.
fn route_inline(section: &mut Section, pending: &mut PendingBlock, t: &str, line: &str) {
    if t.starts_with('#') {
        handle_heading(section, pending, t);
    } else {
        route_content(section, pending, line);
    }
}

/// Reports whether a line ends the current block.
///
/// # Arguments
///
/// * `t` - Trimmed line text.
///
/// # Returns
///
/// `true` for blank lines and horizontal rules.
fn is_separator(t: &str) -> bool {
    t.is_empty() || t == "---"
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
    let mut pending = PendingBlock::None;
    for line in text.lines() {
        process_line(&mut section, &mut pending, line);
    }
    push_flush(&mut section, &mut pending);
    section
}

/// Composes the complete report for one analysis.
///
/// When an AI assessment is present, the report is authored by the analyst and
/// the deterministic findings become an evidence appendix.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
/// * `assessment` - AI analyst narrative, when available.
///
/// # Returns
///
/// The format-neutral report document.
#[must_use]
pub fn build(analysis: &Analysis, bytes: &[u8], assessment: Option<&str>) -> Document {
    let title = format!("Reverse Engineering Report: {}", analysis.name);
    let mut document = Document::new(&title, &subtitle(analysis));
    document.sections = assemble(analysis, bytes, assessment);
    document
}

/// Assembles the report body: authored narrative then evidence appendix.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
/// * `assessment` - AI analyst narrative, when available.
///
/// # Returns
///
/// The ordered report sections.
fn assemble(analysis: &Analysis, bytes: &[u8], assessment: Option<&str>) -> Vec<Section> {
    let mut sections = Vec::new();
    if let Some(text) = assessment {
        sections.push(assessment_section(text));
        sections.push(appendix_heading());
    }
    sections.extend(evidence(analysis, bytes));
    sections
}

/// Builds the deterministic evidence sections.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The evidence sections.
fn evidence(analysis: &Analysis, bytes: &[u8]) -> Vec<Section> {
    let mut sections: Vec<Section> = SECTIONS.iter().flat_map(|b| b(analysis)).collect();
    sections.extend(rules::sections(analysis, bytes));
    sections.extend(crypto::sections(bytes));
    sections.extend(artifacts::sections(analysis, bytes));
    sections.extend(container::sections(bytes));
    sections.extend(coverage::sections(analysis, bytes));
    sections
}

/// Builds the evidence appendix heading.
///
/// # Returns
///
/// The heading section.
fn appendix_heading() -> Section {
    let note = text::paragraph(APPENDIX_NOTE);
    Section::new("Evidence Appendix").with(note)
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

    /// Verifies that an AI assessment leads the report and the evidence follows.
    #[test]
    fn test_assessment_leads_report() {
        let analysis = rea_engine::analyze("blob.bin", b"plain text");
        let doc = build(&analysis, &[], Some("## BLUF\n\nThe target is a blob."));
        let first = doc.sections.first().map(|s| s.heading.clone());
        assert_eq!(first.as_deref(), Some("Autonomous Security Assessment"));
        assert!(doc.sections.iter().any(|s| s.heading == "Evidence Appendix"));
    }

    /// Verifies that unknown data still yields a complete report.
    #[test]
    fn test_build_unknown_data() {
        let analysis = rea_engine::analyze("blob.bin", b"plain text with http://example.test");
        let document = build(&analysis, &[], None);
        assert!(document.title.ends_with("blob.bin"));
        assert!(document.sections.len() >= 4);
    }

    /// Verifies that a real executable yields every section.
    #[test]
    fn test_build_native_executable() {
        let path = std::env::current_exe().unwrap();
        let analysis = rea_engine::analyze_path(&path).unwrap();
        let headings: Vec<String> =
            build(&analysis, &[], None).sections.into_iter().map(|s| s.heading).collect();
        assert!(headings.iter().any(|h| h == "Key Functions"));
    }

    /// Verifies that assessment markdown parses into structured blocks.
    #[test]
    fn test_assessment_section_parses_blocks() {
        let md =
            "### Title\n\nParagraph text\n\n- Item 1\n- Item 2\n\n| A | B |\n| - | - |\n| 1 | 2 |";
        let section = assessment_section(md);
        assert_eq!(section.blocks.len(), 4);
        assert!(matches!(section.blocks[0], Block::Subheading(_)));
        assert!(matches!(section.blocks[1], Block::Paragraph(_)));
        assert!(matches!(section.blocks[2], Block::Bullets(_)));
        assert!(matches!(section.blocks[3], Block::Table(_)));
    }

    /// Verifies that fenced code keeps one instruction per line.
    #[test]
    fn test_assessment_preserves_code_lines() {
        let md = "## Listing\n\n```\n0x1  mov r0, r1\n0x2  bx lr\n```";
        let section = assessment_section(md);
        let code = section.blocks.iter().find_map(|b| match b {
            Block::Code(text) => Some(text.clone()),
            _ => None,
        });
        assert_eq!(code.as_deref(), Some("0x1  mov r0, r1\n0x2  bx lr"));
    }
}
