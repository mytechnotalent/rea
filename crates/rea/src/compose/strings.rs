//! Strings and Indicators
//!
//! **File:** `strings.rs`
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
    iocs::Indicator,
    model::Binary,
    strings::{Encoding, FoundString},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Maximum indicator rows listed.
const MAX_INDICATORS: usize = 400;

/// Maximum strings shown in the sample.
const MAX_SAMPLE: usize = 150;

/// Minimum length of a sampled string.
const SAMPLE_MIN: usize = 8;

/// Maximum characters shown per sampled string.
const SAMPLE_CHARS: usize = 110;

/// Indicator table columns.
const INDICATOR_HEADERS: [&str; 3] = ["Type", "Value", "File offset"];

/// Statement used when no indicators exist.
const NO_INDICATORS: &str = "No network, path, registry, or command indicators were found.";

/// Builds the strings and indicators section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The section.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let summary = Block::Paragraph(summary(&analysis.strings));
    let section = Section::new("Strings and Indicators").with(summary);
    let section = with_indicators(section, &analysis.indicators);
    let sample = sample(&analysis.strings, analysis.binary.as_ref());
    vec![with_sample(section, sample)]
}

/// Summarizes string counts by encoding.
///
/// # Arguments
///
/// * `strings` - Extracted strings.
///
/// # Returns
///
/// One paragraph.
fn summary(strings: &[FoundString]) -> String {
    let wide = strings.iter().filter(|s| s.encoding == Encoding::Utf16Le).count();
    let ascii = strings.len() - wide;
    let total = text::counted(strings.len(), "printable string");
    format!("Observed: {total} ({ascii} ASCII, {wide} UTF-16LE).")
}

/// Appends the indicator table, or a statement that none were found.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `indicators` - Extracted indicators.
///
/// # Returns
///
/// The extended section.
fn with_indicators(section: Section, indicators: &[Indicator]) -> Section {
    if indicators.is_empty() {
        return section.with(text::paragraph(NO_INDICATORS));
    }
    let rows = indicators.iter().map(indicator_row).collect();
    let (table, omitted) = text::capped(&INDICATOR_HEADERS, rows, MAX_INDICATORS);
    let section = section.with(text::subheading("Indicators"));
    text::with_omission(section.with(Block::Table(table)), omitted, "indicator")
}

/// Formats one indicator row.
///
/// # Arguments
///
/// * `indicator` - Extracted indicator.
///
/// # Returns
///
/// The row cells.
fn indicator_row(indicator: &Indicator) -> Vec<String> {
    let offset = text::hex(indicator.offset);
    vec![indicator.kind.to_string(), indicator.value.clone(), offset]
}

/// Appends the string sample when it is not empty.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `sample` - Formatted sample lines.
///
/// # Returns
///
/// The extended section.
fn with_sample(section: Section, sample: String) -> Section {
    if sample.is_empty() {
        return section;
    }
    let section = section.with(text::subheading("String Sample"));
    section.with(Block::Code(sample))
}

/// Selects a readable sample of longer strings outside code.
///
/// # Arguments
///
/// * `strings` - Extracted strings.
/// * `binary` - Parsed executable, when available.
///
/// # Returns
///
/// One `offset  text` line per sampled string.
fn sample(strings: &[FoundString], binary: Option<&Binary>) -> String {
    let long = strings.iter().filter(|s| s.text.trim().len() >= SAMPLE_MIN);
    let data = long.filter(|s| !in_code(binary, s.offset));
    let lines: Vec<String> = data.take(MAX_SAMPLE).map(sample_line).collect();
    lines.join("\n")
}

/// Reports whether a file offset lies inside an executable section.
///
/// # Arguments
///
/// * `binary` - Parsed executable, when available.
/// * `offset` - File offset.
///
/// # Returns
///
/// `true` when the bytes belong to code, where printable runs are noise.
fn in_code(binary: Option<&Binary>, offset: u64) -> bool {
    let address = binary.and_then(|b| b.address_of(offset));
    let section = address.and_then(|a| binary?.section_at(a));
    section.is_some_and(|s| s.executable)
}

/// Formats one sampled string.
///
/// # Arguments
///
/// * `string` - Extracted string.
///
/// # Returns
///
/// The offset and clipped text.
fn sample_line(string: &FoundString) -> String {
    let text = text::clip(&string.text, SAMPLE_CHARS);
    format!("{:08x}  {text}", string.offset)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds an extracted string.
    ///
    /// # Arguments
    ///
    /// * `text` - String text.
    /// * `encoding` - Encoding.
    ///
    /// # Returns
    ///
    /// The string at offset 16.
    fn found(text: &str, encoding: Encoding) -> FoundString {
        let text = text.to_string();
        FoundString { offset: 16, text, encoding }
    }

    /// Verifies counting by encoding.
    #[test]
    fn test_summary_counts_encodings() {
        let strings = [
            found("hello", Encoding::Ascii),
            found("world", Encoding::Utf16Le),
        ];
        assert!(summary(&strings).contains("1 ASCII, 1 UTF-16LE"));
    }

    /// Verifies that the sample skips short strings and formats offsets.
    #[test]
    fn test_sample_filters_short() {
        let strings = [
            found("tiny", Encoding::Ascii),
            found("long enough text", Encoding::Ascii),
        ];
        assert_eq!(sample(&strings, None), "00000010  long enough text");
    }
}
