//! Target Region Segmentation and Crypto Classification
//!
//! **File:** `regions.rs`
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

// Import dependencies from std
use std::fmt::Write as _;

// Import crate::entropy
use crate::entropy::shannon;

// Import crate::numeric
use crate::numeric::ratio_value;

/// Segmentation granularity in bytes.
pub const BLOCK: usize = 4096;

/// Entropy at or above which a block is treated as protected.
const PROTECTED_ENTROPY: f64 = 7.5;

/// Entropy at or above which a block is treated as code-like.
const CODE_ENTROPY: f64 = 6.5;

/// Fraction of zero bytes classifying a block as padding.
const ZERO_RATIO: f64 = 0.9;

/// Fraction of printable bytes classifying a block as a string table.
const PRINTABLE_RATIO: f64 = 0.85;

/// Fraction of peripheral-looking words classifying a block as MMIO.
const MMIO_RATIO: f64 = 0.25;

/// Evidence note for zero padding.
const ZERO_NOTE: &str = "All-zero (>= 90%) padding.";

/// Evidence note for dense text.
const STRINGS_NOTE: &str = "Dense printable text (>= 85%).";

/// Evidence note for a recognized container.
const COMPRESSED_NOTE: &str = "Begins with a known compression container magic.";

/// Evidence note for protected data.
const ENCRYPTED_NOTE: &str = "Entropy >= 7.5 with no standard container magic.";

/// Evidence note for MMIO pointers.
const MMIO_NOTE: &str = "Contains peripheral/MMIO address words.";

/// Evidence note for code.
const CODE_NOTE: &str = "Instruction-dense entropy band.";

/// Evidence note for tables.
const TABLE_NOTE: &str = "Low-entropy structured data.";

/// Evidence note for unclassified data.
const UNKNOWN_NOTE: &str = "Unclassified.";

/// Known compression container magics and their labels.
const MAGICS: [(&[u8], &str); 8] = [
    (b"\x78\x01", "zlib"),
    (b"\x78\x9c", "zlib"),
    (b"\x78\xda", "zlib"),
    (b"\x1f\x8b", "gzip"),
    (b"\x5d\x00\x00", "lzma"),
    (b"\xfd7zXZ", "xz"),
    (b"BZh", "bzip2"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
];

/// Human-readable labels indexed by region kind.
const KIND_LABELS: [(RegionKind, &str); 9] = [
    (RegionKind::Unknown, "unknown"),
    (RegionKind::Code, "code"),
    (RegionKind::Data, "data"),
    (RegionKind::Table, "table"),
    (RegionKind::Strings, "strings"),
    (RegionKind::Mmio, "mmio"),
    (RegionKind::Encrypted, "encrypted"),
    (RegionKind::Compressed, "compressed"),
    (RegionKind::Zero, "padding"),
];

/// Classification of a contiguous region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionKind {
    /// Not classified.
    #[default]
    Unknown,
    /// Executable-looking code.
    Code,
    /// Mixed data.
    Data,
    /// Structured records or tables.
    Table,
    /// Dense printable text.
    Strings,
    /// Memory-mapped peripheral pointers.
    Mmio,
    /// High-entropy data with no standard container.
    Encrypted,
    /// A recognized compressed container.
    Compressed,
    /// Zero padding.
    Zero,
}

impl RegionKind {
    /// Returns the human-readable label.
    ///
    /// # Returns
    ///
    /// The region kind label.
    #[must_use]
    pub fn label(self) -> &'static str {
        let found = KIND_LABELS.iter().find(|(kind, _)| *kind == self);
        found.map_or("unknown", |(_, label)| label)
    }
}

/// A classified contiguous region of the target.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Region {
    /// Region start offset.
    pub start: usize,
    /// Region length in bytes.
    pub size: usize,
    /// Classification.
    pub kind: RegionKind,
    /// Mean entropy of the block at the region start.
    pub entropy: f64,
    /// One-line evidence for the classification.
    pub note: String,
}

impl Region {
    /// Returns the exclusive end offset.
    ///
    /// # Returns
    ///
    /// The end offset.
    #[must_use]
    pub fn end(&self) -> usize {
        self.start.saturating_add(self.size)
    }
}

/// Segments and classifies a target.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The classified regions in file order.
#[must_use]
pub fn segment(bytes: &[u8]) -> Vec<Region> {
    let blocks: Vec<Region> =
        (0..bytes.len()).step_by(BLOCK).map(|s| classify_block(bytes, s)).collect();
    merge(blocks)
}

/// Classifies one block starting at an offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `start` - Block start offset.
///
/// # Returns
///
/// The classified block as a region.
fn classify_block(bytes: &[u8], start: usize) -> Region {
    let end = start.saturating_add(BLOCK).min(bytes.len());
    let block = bytes.get(start..end).unwrap_or_default();
    let entropy = shannon(block);
    let kind = classify(block, entropy);
    Region { start, size: end - start, kind, entropy, note: note_for(kind).to_string() }
}

/// Chooses a region kind from block features.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `entropy` - Block entropy.
///
/// # Returns
///
/// The classification.
fn classify(block: &[u8], entropy: f64) -> RegionKind {
    if zero_ratio(block) > ZERO_RATIO {
        return RegionKind::Zero;
    }
    classify_text(block, entropy)
}

/// Classifies a non-padding block from text density and container magic.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `entropy` - Block entropy.
///
/// # Returns
///
/// The classification.
fn classify_text(block: &[u8], entropy: f64) -> RegionKind {
    if printable_ratio(block) > PRINTABLE_RATIO {
        return RegionKind::Strings;
    }
    match magic_of(block) {
        Some(_) => RegionKind::Compressed,
        None => classify_entropy(block, entropy),
    }
}

/// Classifies a block by entropy and peripheral density.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `entropy` - Block entropy.
///
/// # Returns
///
/// The classification.
fn classify_entropy(block: &[u8], entropy: f64) -> RegionKind {
    if entropy >= PROTECTED_ENTROPY {
        return RegionKind::Encrypted;
    }
    classify_structured(block, entropy)
}

/// Classifies a non-protected, non-text block.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `entropy` - Block entropy.
///
/// # Returns
///
/// The classification.
fn classify_structured(block: &[u8], entropy: f64) -> RegionKind {
    if looks_like_mmio(block) {
        return RegionKind::Mmio;
    }
    if entropy >= CODE_ENTROPY {
        return RegionKind::Code;
    }
    RegionKind::Table
}

/// Returns the compression container label for a block.
///
/// # Arguments
///
/// * `block` - Block bytes.
///
/// # Returns
///
/// The container label when the block begins with a known magic.
fn magic_of(block: &[u8]) -> Option<&'static str> {
    MAGICS
        .iter()
        .find(|(magic, _)| block.starts_with(magic))
        .map(|(_, label)| *label)
}

/// Returns the evidence note for a classification.
///
/// # Arguments
///
/// * `kind` - Region kind.
///
/// # Returns
///
/// The evidence note.
fn note_for(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Zero => ZERO_NOTE,
        RegionKind::Strings => STRINGS_NOTE,
        RegionKind::Compressed => COMPRESSED_NOTE,
        RegionKind::Encrypted => ENCRYPTED_NOTE,
        _ => other_note(kind),
    }
}

/// Returns the evidence note for a structured classification.
///
/// # Arguments
///
/// * `kind` - Region kind.
///
/// # Returns
///
/// The evidence note.
fn other_note(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Mmio => MMIO_NOTE,
        RegionKind::Code => CODE_NOTE,
        RegionKind::Table => TABLE_NOTE,
        _ => UNKNOWN_NOTE,
    }
}

/// Computes the fraction of zero bytes.
///
/// # Arguments
///
/// * `block` - Block bytes.
///
/// # Returns
///
/// The zero fraction.
fn zero_ratio(block: &[u8]) -> f64 {
    ratio(block, |b| b == 0)
}

/// Computes the fraction of printable bytes.
///
/// # Arguments
///
/// * `block` - Block bytes.
///
/// # Returns
///
/// The printable fraction.
fn printable_ratio(block: &[u8]) -> f64 {
    ratio(block, |b| {
        b.is_ascii_graphic() || matches!(b, b' ' | b'\n' | b'\r' | b'\t')
    })
}

/// Computes the fraction of bytes matching a predicate.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `pred` - Byte predicate.
///
/// # Returns
///
/// The fraction, or zero for an empty block.
fn ratio(block: &[u8], pred: impl Fn(u8) -> bool) -> f64 {
    if block.is_empty() {
        return 0.0;
    }
    let count = block.iter().filter(|&&b| pred(b)).count();
    fraction(count, block.len())
}

/// Reports whether a block is dominated by peripheral pointers.
///
/// # Arguments
///
/// * `block` - Block bytes.
///
/// # Returns
///
/// `true` when at least a quarter of the words are MMIO addresses.
fn looks_like_mmio(block: &[u8]) -> bool {
    let words = block.len() / 4;
    let count = (0..words).filter(|i| is_mmio(word(block, i * 4))).count();
    words > 0 && fraction(count, words) >= MMIO_RATIO
}

/// Computes a count-to-total fraction.
///
/// # Arguments
///
/// * `count` - Matching count.
/// * `total` - Total count.
///
/// # Returns
///
/// The fraction.
fn fraction(count: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    ratio_value(count) / ratio_value(total)
}

/// Reports whether a word is an ARM peripheral address.
///
/// # Arguments
///
/// * `value` - 32-bit word.
///
/// # Returns
///
/// `true` for peripheral or private-peripheral ranges.
fn is_mmio(value: u32) -> bool {
    (0x4000_0000..0x5000_0000).contains(&value) || (0xE000_0000..0xE100_0000).contains(&value)
}

/// Reads a little-endian 32-bit word at an offset.
///
/// # Arguments
///
/// * `block` - Block bytes.
/// * `offset` - Word offset.
///
/// # Returns
///
/// The word, or zero when truncated.
fn word(block: &[u8], offset: usize) -> u32 {
    let slice = block.get(offset..offset + 4).unwrap_or_default();
    let mut buf = [0u8; 4];
    buf[..slice.len()].copy_from_slice(slice);
    u32::from_le_bytes(buf)
}

/// Merges adjacent blocks of the same kind.
///
/// # Arguments
///
/// * `blocks` - Classified blocks.
///
/// # Returns
///
/// The merged regions.
fn merge(blocks: Vec<Region>) -> Vec<Region> {
    let mut out: Vec<Region> = Vec::new();
    for block in blocks {
        push_merged(&mut out, block);
    }
    out
}

/// Appends a block, extending the previous region when contiguous and equal.
///
/// # Arguments
///
/// * `out` - Accumulated regions.
/// * `block` - Block to append.
fn push_merged(out: &mut Vec<Region>, block: Region) {
    if let Some(last) = out.last_mut() {
        if last.kind == block.kind && last.end() == block.start {
            last.size += block.size;
            return;
        }
    }
    out.push(block);
}

/// Reports the byte total of each region kind.
///
/// # Arguments
///
/// * `regions` - Classified regions.
///
/// # Returns
///
/// A `(kind label, bytes)` list in decreasing size.
#[must_use]
pub fn summary(regions: &[Region]) -> Vec<(&'static str, usize)> {
    let mut totals: Vec<(&'static str, usize)> = Vec::new();
    for region in regions {
        add_total(&mut totals, region.kind.label(), region.size);
    }
    totals.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    totals
}

/// Adds a byte total to a label accumulator.
///
/// # Arguments
///
/// * `totals` - Accumulated totals.
/// * `label` - Region kind label.
/// * `size` - Byte count.
fn add_total(totals: &mut Vec<(&'static str, usize)>, label: &'static str, size: usize) {
    if let Some(entry) = totals.iter_mut().find(|(name, _)| *name == label) {
        entry.1 += size;
    } else {
        totals.push((label, size));
    }
}

/// Formats the region map as a human-readable text block.
///
/// # Arguments
///
/// * `regions` - Classified regions.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_regions(regions: &[Region]) -> String {
    let mut out = String::new();
    for region in regions {
        append_region(&mut out, region);
    }
    out
}

/// Appends one region line.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `region` - Region to describe.
fn append_region(out: &mut String, region: &Region) {
    let kind = region.kind.label();
    let size = region.size;
    let _ = writeln!(
        out,
        "0x{:06x} {kind} +{size} e{:.2}",
        region.start, region.entropy
    );
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that zero padding and text are separated.
    #[test]
    fn test_segment_padding_and_text() {
        let mut data = vec![0u8; 8192];
        let text = b"Shanghai 8051 Calibration Failed".repeat(200);
        data.extend_from_slice(&text);
        let regions = segment(&data);
        assert!(regions.iter().any(|r| r.kind == RegionKind::Zero));
        assert!(regions.iter().any(|r| r.kind == RegionKind::Strings));
    }

    /// Verifies that high-entropy data is classified as protected.
    #[test]
    fn test_segment_encrypted() {
        let data: Vec<u8> =
            (0..8192u32).map(|i| i.wrapping_mul(2_654_435_761).to_le_bytes()[2]).collect();
        let regions = segment(&data);
        assert!(regions.iter().any(|r| r.kind == RegionKind::Encrypted));
    }

    /// Verifies that MMIO-dense data is detected.
    #[test]
    fn test_segment_mmio() {
        let words: Vec<u8> = (0..1024).flat_map(|_| [0x00u8, 0x10, 0x0d, 0x40]).collect();
        let regions = segment(&words);
        assert!(regions.iter().any(|r| r.kind == RegionKind::Mmio));
    }

    /// Verifies that adjacent equal blocks merge.
    #[test]
    fn test_segment_merges() {
        let data = vec![0u8; 12_288];
        let regions = segment(&data);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].size, 12_288);
    }
}
