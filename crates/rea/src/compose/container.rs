//! Firmware Container Findings
//!
//! **File:** `container.rs`
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
use rea_engine::container::{self, Container};

// Import dependencies from rea_report
use rea_report::Section;

// Import crate::compose::text
use crate::compose::text;

/// Builds the firmware container section.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// One section when a container header is detected.
pub fn sections(bytes: &[u8]) -> Vec<Section> {
    container::parse(bytes).map_or_else(Vec::new, |found| vec![build(&found)])
}

/// Builds the container section from a parsed header.
///
/// # Arguments
///
/// * `found` - Parsed container.
///
/// # Returns
///
/// The section.
fn build(found: &Container) -> Section {
    Section::new("Firmware Container").with(text::paragraph(&note(found)))
}

/// Composes the container note text.
///
/// # Arguments
///
/// * `found` - Parsed container.
///
/// # Returns
///
/// The note text.
fn note(found: &Container) -> String {
    let summary = container::format_container(found);
    let kinds: Vec<String> = found.records.iter().map(|r| format!("{:#04x}", r.kind)).collect();
    format!(
        "Multi-record firmware container, not a flat image. {summary}. Record kinds observed: {}.",
        kinds.join(", ")
    )
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a flat image yields no container section.
    #[test]
    fn test_container_section_skips_flat_image() {
        assert_eq!(sections(b"plain data").len(), 0);
    }
}
