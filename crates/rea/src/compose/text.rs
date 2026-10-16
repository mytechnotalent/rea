//! Report Text Formatting
//!
//! **File:** `text.rs`
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

// Import dependencies from rea_report
use rea_report::{Block, Section, Table};

/// Binary size units used for human-readable sizes.
const UNITS: [(&str, u64); 4] = [
    ("B", 1),
    ("KiB", 1 << 10),
    ("MiB", 1 << 20),
    ("GiB", 1 << 30),
];

/// Formats an address or size as lowercase hex.
///
/// # Arguments
///
/// * `value` - Value to format.
///
/// # Returns
///
/// Text such as `0x401000`.
#[must_use]
pub fn hex(value: u64) -> String {
    format!("0x{value:x}")
}

/// Formats a byte count with grouping and a scaled unit.
///
/// # Arguments
///
/// * `bytes` - Byte count.
///
/// # Returns
///
/// Text such as `1,234,567 bytes (1.1 MiB)`.
#[must_use]
pub fn size(bytes: u64) -> String {
    format!("{} bytes ({})", grouped(bytes), scaled(bytes))
}

/// Groups decimal digits in threes.
///
/// # Arguments
///
/// * `value` - Number to format.
///
/// # Returns
///
/// Text such as `1,234,567`.
#[must_use]
pub fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let chunks = digits.as_bytes().rchunks(3).rev();
    let parts: Vec<String> = chunks.map(|c| String::from_utf8_lossy(c).into()).collect();
    parts.join(",")
}

/// Scales a byte count to the largest fitting binary unit.
///
/// # Arguments
///
/// * `bytes` - Byte count.
///
/// # Returns
///
/// Text with one decimal place, such as `1.5 KiB`.
fn scaled(bytes: u64) -> String {
    let fitting = UNITS.iter().rev().find(|(_, factor)| bytes >= *factor);
    let (unit, factor) = fitting.copied().unwrap_or(UNITS[0]);
    let tenths = bytes.saturating_mul(10) / factor;
    format!("{}.{} {unit}", tenths / 10, tenths % 10)
}

/// Formats entropy with two decimals.
///
/// # Arguments
///
/// * `value` - Entropy in bits per byte.
///
/// # Returns
///
/// Text such as `6.42`.
#[must_use]
pub fn entropy(value: f64) -> String {
    format!("{value:.2}")
}

/// Shortens text to a maximum number of characters.
///
/// # Arguments
///
/// * `text` - Text to shorten.
/// * `max` - Maximum characters kept.
///
/// # Returns
///
/// The text, with an ellipsis when cut.
#[must_use]
pub fn clip(text: &str, max: usize) -> String {
    let short: String = text.chars().take(max).collect();
    let cut = short.len() < text.len();
    if cut {
        short + "..."
    } else {
        short
    }
}

/// Joins items, listing at most `max` of them.
///
/// # Arguments
///
/// * `items` - Items to list.
/// * `max` - Maximum items shown.
///
/// # Returns
///
/// Comma-separated items with a `(+N more)` suffix when cut.
#[must_use]
pub fn listed(items: &[String], max: usize) -> String {
    let shown = items[..items.len().min(max)].join(", ");
    let hidden = items.len().saturating_sub(max);
    let more = format!("{shown} (+{hidden} more)");
    if hidden == 0 {
        shown
    } else {
        more
    }
}

/// Pluralizes a counted noun.
///
/// # Arguments
///
/// * `count` - Quantity.
/// * `noun` - Singular noun.
///
/// # Returns
///
/// Text such as `1 section` or `3 libraries`.
#[must_use]
pub fn counted(count: usize, noun: &str) -> String {
    let word = if count == 1 {
        noun.to_string()
    } else {
        plural(noun)
    };
    format!("{} {word}", grouped(to_u64(count)))
}

/// Forms the plural of an English noun.
///
/// # Arguments
///
/// * `noun` - Singular noun.
///
/// # Returns
///
/// The noun with `ies` for a consonant before `y`, otherwise with `s`.
#[must_use]
pub fn plural(noun: &str) -> String {
    let stem = noun.strip_suffix('y').filter(|s| !s.ends_with(['a', 'e', 'o', 'u']));
    stem.map_or_else(|| format!("{noun}s"), |s| format!("{s}ies"))
}

/// Chooses the indefinite article for a word.
///
/// # Arguments
///
/// * `word` - Word that follows the article.
///
/// # Returns
///
/// `an` before a vowel letter, otherwise `a`.
#[must_use]
pub fn article(word: &str) -> &'static str {
    let vowel = word.starts_with(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']);
    if vowel {
        "an"
    } else {
        "a"
    }
}

/// Converts a count to `u64`.
///
/// # Arguments
///
/// * `count` - Count to convert.
///
/// # Returns
///
/// The count, saturating in the impossible overflow case.
#[must_use]
pub fn to_u64(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

/// Returns text only when a condition holds.
///
/// # Arguments
///
/// * `condition` - Whether to keep the text.
/// * `text` - Text to keep.
///
/// # Returns
///
/// `text`, or an empty string.
#[must_use]
pub fn when(condition: bool, text: &'static str) -> &'static str {
    if condition {
        text
    } else {
        ""
    }
}

/// Uses a value or falls back to fixed text.
///
/// # Arguments
///
/// * `value` - Preferred text, when available.
/// * `fallback` - Text used otherwise.
///
/// # Returns
///
/// The chosen text.
#[must_use]
pub fn or(value: Option<String>, fallback: &str) -> String {
    value.unwrap_or_else(|| fallback.to_string())
}

/// Describes a packing verdict.
///
/// # Arguments
///
/// * `likely` - Whether packing is likely.
///
/// # Returns
///
/// A short verdict phrase.
#[must_use]
pub fn packing_verdict(likely: bool) -> &'static str {
    if likely {
        "likely packed, compressed, or encrypted"
    } else {
        "not packed"
    }
}

/// Builds a paragraph block.
///
/// # Arguments
///
/// * `text` - Paragraph text.
///
/// # Returns
///
/// The block.
#[must_use]
pub fn paragraph(text: &str) -> Block {
    Block::Paragraph(text.to_string())
}

/// Builds a subheading block.
///
/// # Arguments
///
/// * `text` - Heading text.
///
/// # Returns
///
/// The block.
#[must_use]
pub fn subheading(text: &str) -> Block {
    Block::Subheading(text.to_string())
}

/// Builds a table from headings and rows.
///
/// # Arguments
///
/// * `headers` - Column headings.
/// * `rows` - Data rows.
///
/// # Returns
///
/// The table.
#[must_use]
pub fn table(headers: &[&str], rows: Vec<Vec<String>>) -> Table {
    rows.into_iter().fold(Table::new(headers), Table::row)
}

/// Builds a two-column table from label and value pairs.
///
/// # Arguments
///
/// * `headers` - The two column headings.
/// * `rows` - Label and value pairs.
///
/// # Returns
///
/// The table.
#[must_use]
pub fn pairs(headers: [&str; 2], rows: Vec<(String, String)>) -> Table {
    let cells = rows.into_iter().map(|(label, value)| vec![label, value]);
    table(&headers, cells.collect())
}

/// Builds a table from rows, keeping at most `max`.
///
/// # Arguments
///
/// * `headers` - Column headings.
/// * `rows` - Data rows.
/// * `max` - Maximum rows kept.
///
/// # Returns
///
/// The table and how many rows were omitted.
#[must_use]
pub fn capped(headers: &[&str], rows: Vec<Vec<String>>, max: usize) -> (Table, usize) {
    let omitted = rows.len().saturating_sub(max);
    let kept = rows.into_iter().take(max).collect();
    (table(headers, kept), omitted)
}

/// Appends a note about omitted rows when any were left out.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `omitted` - Rows left out.
/// * `noun` - What the rows are.
///
/// # Returns
///
/// The section, with a note when rows were omitted.
#[must_use]
pub fn with_omission(section: Section, omitted: usize, noun: &str) -> Section {
    let count = counted(omitted, noun);
    let note = format!("{count} not shown; the table lists the first entries.");
    let notes = (omitted > 0).then_some(Block::Paragraph(note));
    notes.into_iter().fold(section, Section::with)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies digit grouping.
    #[test]
    fn test_grouped_digits() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    /// Verifies unit scaling.
    #[test]
    fn test_size_scales_units() {
        assert_eq!(size(1536), "1,536 bytes (1.5 KiB)");
        assert_eq!(size(10), "10 bytes (10.0 B)");
    }

    /// Verifies list truncation and pluralization.
    #[test]
    fn test_listed_and_counted() {
        let items: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(listed(&items, 2), "a, b (+1 more)");
        assert_eq!(counted(1, "file"), "1 file");
        assert_eq!(counted(2, "library"), "2 libraries");
        assert_eq!(plural("key"), "keys");
    }

    /// Verifies capped tables and omission notes.
    #[test]
    fn test_capped_and_omission() {
        let rows = vec![vec!["1".to_string()], vec!["2".to_string()]];
        let (table, omitted) = capped(&["N"], rows, 1);
        assert_eq!((table.rows.len(), omitted), (1, 1));
        assert_eq!(with_omission(Section::new("S"), 0, "row").blocks.len(), 0);
    }

    /// Verifies clipping and hex formatting.
    #[test]
    fn test_clip_and_hex() {
        assert_eq!(clip("abcdef", 3), "abc...");
        assert_eq!(hex(4096), "0x1000");
    }

    /// Verifies conditional and fallback text.
    #[test]
    fn test_when_and_or() {
        assert_eq!(when(false, "x"), "");
        assert_eq!(or(None, "fallback"), "fallback");
        assert_eq!(packing_verdict(false), "not packed");
        assert_eq!(article("Executable"), "an");
    }
}
