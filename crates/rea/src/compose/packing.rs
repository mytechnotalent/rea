//! Packing Assessment
//!
//! **File:** `packing.rs`
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
    packer::{Assessment, Signal, LIKELY_SCORE},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Builds the packing and obfuscation section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The section.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let packing = &analysis.packing;
    let verdict = Block::Paragraph(verdict_text(packing));
    let section = Section::new("Packing and Obfuscation").with(verdict);
    vec![section.with(signal_block(&packing.signals))]
}

/// States the packing verdict with its score.
///
/// # Arguments
///
/// * `packing` - Packing assessment.
///
/// # Returns
///
/// One paragraph.
fn verdict_text(packing: &Assessment) -> String {
    let verdict = text::packing_verdict(packing.likely);
    let (score, threshold) = (packing.score, LIKELY_SCORE);
    format!("Inferred: the target is {verdict} (score {score}; threshold {threshold}).")
}

/// Lists the signals that fired.
///
/// # Arguments
///
/// * `signals` - Packing signals.
///
/// # Returns
///
/// A bullet list, or a statement that none fired.
fn signal_block(signals: &[Signal]) -> Block {
    if signals.is_empty() {
        return text::paragraph("No packing signals fired.");
    }
    Block::Bullets(signals.iter().map(signal_text).collect())
}

/// Formats one signal.
///
/// # Arguments
///
/// * `signal` - Packing signal.
///
/// # Returns
///
/// The reason with its weight.
fn signal_text(signal: &Signal) -> String {
    format!("{} (weight {})", signal.reason, signal.weight)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that low-entropy data is reported as not packed.
    #[test]
    fn test_verdict_not_packed() {
        let analysis = rea_engine::analyze("x", b"aaaaaaaaaaaaaaaa");
        assert!(verdict_text(&analysis.packing).contains("not packed"));
    }

    /// Verifies the empty signal statement.
    #[test]
    fn test_signal_block_empty() {
        assert!(matches!(signal_block(&[]), Block::Paragraph(_)));
    }
}
