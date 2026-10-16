//! Build Provenance Section
//!
//! **File:** `provenance.rs`
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
    provenance::{recover_detailed, Provenance},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::Section;

// Import crate::compose::text
use crate::compose::text;

/// Builds the build-provenance section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The section, or nothing when no provenance was recovered.
pub fn sections(analysis: &Analysis, bytes: &[u8]) -> Vec<Section> {
    let provenance = recover_detailed(analysis, bytes);
    if provenance.components.is_empty()
        && provenance.toolchain.is_none()
        && provenance.build_path.is_none()
    {
        return Vec::new();
    }
    vec![Section::new("Provenance").with(text::paragraph(&note(&provenance)))]
}

/// Composes the provenance note.
///
/// # Arguments
///
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The note text.
fn note(provenance: &Provenance) -> String {
    let mut parts = Vec::new();
    push_labeled(&mut parts, "toolchain", provenance.toolchain.as_deref());
    push_labeled(&mut parts, "built at", provenance.build_path.as_deref());
    push_components(&mut parts, provenance);
    format!("Recovered build provenance: {}.", parts.join("; "))
}

/// Appends a labelled value when present.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `label` - Value label.
/// * `value` - Value text.
fn push_labeled(parts: &mut Vec<String>, label: &str, value: Option<&str>) {
    if let Some(value) = value {
        parts.push(format!("{label} {value}"));
    }
}

/// Appends the component summary.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `provenance` - Recovered provenance.
fn push_components(parts: &mut Vec<String>, provenance: &Provenance) {
    let names: Vec<&str> = provenance.components.iter().map(|c| c.name.as_str()).collect();
    if !names.is_empty() {
        parts.push(format!(
            "{} component(s): {}",
            names.len(),
            names.join(", ")
        ));
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a plain target yields no provenance section.
    #[test]
    fn test_provenance_skips_plain() {
        let analysis = rea_engine::analyze("blob.bin", b"plain data");
        assert_eq!(sections(&analysis, b"").len(), 0);
    }
}
