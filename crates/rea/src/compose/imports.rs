//! Imports, Capabilities, and Exports
//!
//! **File:** `imports.rs`
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
use std::collections::BTreeMap;

// Import dependencies from rea_engine
use rea_engine::{
    capabilities::Capability,
    model::{Binary, Import, Symbol},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section, Table};

// Import crate::compose::text
use crate::compose::text;

/// Maximum import rows listed.
const MAX_IMPORTS: usize = 600;

/// Maximum export rows listed.
const MAX_EXPORTS: usize = 300;

/// Maximum APIs listed per capability.
const MAX_APIS: usize = 12;

/// Capability table columns.
const CAPABILITY_HEADERS: [&str; 3] = ["Capability", "What it enables", "Imported APIs"];

/// Import table columns.
const IMPORT_HEADERS: [&str; 3] = ["Library", "Function", "Slot"];

/// Explanation placed above the capability table.
const CAPABILITY_NOTE: &str = "Derived: capabilities come from the APIs the program imports; \
an import shows ability, not proof of use.";

/// Statement used when no capability-relevant imports exist.
const NO_CAPABILITIES: &str = "No capability-relevant imports were found.";

/// Builds the capability, import, and export sections.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The sections for executables, otherwise none.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let Some(binary) = &analysis.binary else {
        return Vec::new();
    };
    let mut found = vec![capabilities(&analysis.capabilities), imports(binary)];
    found.extend(exports(binary));
    found
}

/// Builds the capability section.
///
/// # Arguments
///
/// * `capabilities` - Classified capabilities.
///
/// # Returns
///
/// The capability table or a statement that none were found.
fn capabilities(capabilities: &[Capability]) -> Section {
    let note = text::paragraph(CAPABILITY_NOTE);
    let section = Section::new("Capabilities").with(note);
    if capabilities.is_empty() {
        return section.with(text::paragraph(NO_CAPABILITIES));
    }
    let rows = capabilities.iter().map(capability_row).collect();
    let table = text::table(&CAPABILITY_HEADERS, rows);
    section.with(Block::Table(table))
}

/// Formats one capability row.
///
/// # Arguments
///
/// * `capability` - Classified capability.
///
/// # Returns
///
/// The row cells.
fn capability_row(capability: &Capability) -> Vec<String> {
    let apis = text::listed(&capability.apis, MAX_APIS);
    let category = capability.category.to_string();
    vec![category, capability.description.to_string(), apis]
}

/// Builds the import section with per-library counts and the full table.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The import section.
fn imports(binary: &Binary) -> Section {
    let summary = Block::Paragraph(import_summary(binary));
    let section = Section::new("Imports").with(summary);
    let section = section.with(Block::Table(library_table(binary)));
    let section = section.with(text::subheading("Import Table"));
    let rows = binary.imports.iter().map(import_row).collect();
    let (table, omitted) = text::capped(&IMPORT_HEADERS, rows, MAX_IMPORTS);
    text::with_omission(section.with(Block::Table(table)), omitted, "import")
}

/// Summarizes the import counts.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One paragraph.
fn import_summary(binary: &Binary) -> String {
    let imports = text::counted(binary.imports.len(), "imported function");
    let libraries = text::counted(binary.libraries.len(), "library");
    format!("Observed: {imports} from {libraries}.")
}

/// Formats one import row.
///
/// # Arguments
///
/// * `import` - Imported function.
///
/// # Returns
///
/// The row cells.
fn import_row(import: &Import) -> Vec<String> {
    let slot = text::hex(import.address);
    vec![import.library.clone(), import.name.clone(), slot]
}

/// Tabulates imported functions per library.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Libraries and their import counts, or "not recorded" for formats without attribution.
fn library_table(binary: &Binary) -> Table {
    let counts = library_counts(binary);
    let attributed = binary.imports.iter().any(|i| !i.library.is_empty());
    let named = counts.into_iter().filter(|(library, _)| !library.is_empty());
    let rows = named.map(|(l, n)| vec![l.to_string(), count_text(n, attributed)]);
    text::table(&["Library", "Imported functions"], rows.collect())
}

/// Counts imports per library, including libraries with none attributed.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Library name to import count.
fn library_counts(binary: &Binary) -> BTreeMap<&str, usize> {
    let libraries = binary.libraries.iter().map(|l| (l.as_str(), 0));
    let mut counts: BTreeMap<&str, usize> = libraries.collect();
    for import in &binary.imports {
        *counts.entry(import.library.as_str()).or_default() += 1;
    }
    counts
}

/// Formats an import count.
///
/// # Arguments
///
/// * `count` - Imports attributed to a library.
/// * `attributed` - Whether the format records providing libraries.
///
/// # Returns
///
/// The count, or `not recorded`.
fn count_text(count: usize, attributed: bool) -> String {
    let text = count.to_string();
    if attributed {
        text
    } else {
        "not recorded".to_string()
    }
}

/// Builds the export section when the binary exports symbols.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The export section, or nothing.
fn exports(binary: &Binary) -> Option<Section> {
    let rows = binary.exports.iter().map(export_row).collect();
    let (table, omitted) = text::capped(&["Export", "Address"], rows, MAX_EXPORTS);
    let section = Section::new("Exports").with(Block::Table(table));
    let section = text::with_omission(section, omitted, "export");
    (!binary.exports.is_empty()).then_some(section)
}

/// Formats one export row.
///
/// # Arguments
///
/// * `symbol` - Exported symbol.
///
/// # Returns
///
/// The row cells.
fn export_row(symbol: &Symbol) -> Vec<String> {
    vec![symbol.name.clone(), text::hex(symbol.address)]
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the empty capability case is stated.
    #[test]
    fn test_capabilities_empty() {
        assert_eq!(capabilities(&[]).blocks.len(), 2);
    }

    /// Verifies per-library import counting.
    #[test]
    fn test_library_table_counts() {
        let imports = vec![Import::new("A.dll", "x", 0), Import::new("A.dll", "y", 0)];
        let binary = Binary { imports, ..Binary::default() };
        let want = vec![vec!["A.dll".to_string(), "2".to_string()]];
        assert_eq!(library_table(&binary).rows, want);
    }

    /// Verifies that exports are omitted when absent and listed when present.
    #[test]
    fn test_exports_presence() {
        let symbol = Symbol::new("DllMain", 0x1000);
        let mut binary = Binary { exports: Vec::new(), ..Binary::default() };
        assert!(exports(&binary).is_none());
        binary.exports.push(symbol);
        assert!(exports(&binary).is_some());
    }

    /// Verifies unattributed import counts.
    #[test]
    fn test_count_text_unattributed() {
        assert_eq!(count_text(0, false), "not recorded");
        assert_eq!(count_text(3, true), "3");
    }
}
