//! Top-Level Program Behavior Summary
//!
//! **File:** `program_summary.rs`
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
use rea_engine::{narrative, Analysis};

// Import dependencies from rea_report
use rea_report::{Block, Section};

/// Builds the top-level plain-English program summary section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// A section summarizing what the program does, when possible.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let Some(binary) = &analysis.binary else {
        return Vec::new();
    };
    let intro = describe_identity(&analysis.name, binary);
    let section = Section::new("What This Program Does").with(Block::Paragraph(intro));
    let story = narrative::detailed_program_story(analysis);
    vec![compose_summary(section, &story)]
}

/// Title for execution lifecycle subsection.
const BOOT_TITLE: &str = "Execution Lifecycle & Boot Sequence";

/// Title for main logic subsection.
const MAIN_TITLE: &str = "Main Application Logic & Program Loop";

/// Title for compiler optimization subsection.
const OPT_TITLE: &str = "Variable Storage & Compiler Optimization";

/// Title for reverse engineering vectors subsection.
const PATCH_TITLE: &str = "Binary Patching & Reverse Engineering Vectors";

/// Appends core logic and optimization narrative subsections.
///
/// # Arguments
///
/// * `section` - Section to extend.
/// * `story` - Detailed program narrative.
///
/// # Returns
///
/// Updated section with core logic subsections.
fn append_core_summary(mut section: Section, story: &narrative::DetailedStory) -> Section {
    section = append_part(section, MAIN_TITLE, &story.main_logic);
    append_part(section, OPT_TITLE, &story.optimization)
}

/// Composes narrative subsections into the target summary section.
///
/// # Arguments
///
/// * `section` - Section to extend.
/// * `story` - Detailed program narrative.
///
/// # Returns
///
/// Completed section with headings and paragraphs.
fn compose_summary(mut section: Section, story: &narrative::DetailedStory) -> Section {
    section = append_part(section, BOOT_TITLE, &story.boot);
    section = append_core_summary(section, story);
    append_part(section, PATCH_TITLE, &story.patching)
}

/// Appends a subheading and its paragraph blocks to a section when non-empty.
///
/// # Arguments
///
/// * `section` - Section to extend.
/// * `title` - Subheading title.
/// * `paragraphs` - Paragraph lines.
///
/// # Returns
///
/// The extended section.
fn append_part(section: Section, title: &str, paragraphs: &[String]) -> Section {
    if paragraphs.is_empty() {
        return section;
    }
    let section = section.with(Block::Subheading(title.to_string()));
    paragraphs.iter().fold(section, |s, p| s.with(Block::Paragraph(p.clone())))
}

/// Describes the target identity and architecture.
///
/// # Arguments
///
/// * `name` - Target name.
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One sentence of plain-English description.
fn describe_identity(name: &str, binary: &rea_engine::model::Binary) -> String {
    let kind = binary.kind.to_lowercase();
    let arch = &binary.arch_name;
    let bits = binary.bits;
    format!("{name} is a {bits}-bit {arch} {kind}.")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Tests program summary sections with empty analysis.
    #[test]
    fn test_sections_empty() {
        let a = Analysis::default();
        let s = sections(&a);
        assert!(s.is_empty());
    }
}
