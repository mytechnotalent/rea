//! Code Analysis
//!
//! **File:** `code.rs`
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
use std::cmp::Reverse;

// Import dependencies from rea_engine
use rea_engine::{
    disasm::{Disassembly, Function},
    insights::KeyFunction,
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section, Table};

// Import crate::compose::text
use crate::compose::text;

/// Number of largest functions tabulated.
const TOP_FUNCTIONS: usize = 25;

/// Maximum imports listed per function row.
const ROW_IMPORTS: usize = 4;

/// Function table columns.
const FUNCTION_HEADERS: [&str; 6] = [
    "Function",
    "Address",
    "Instructions",
    "Callers",
    "Callees",
    "Imports used",
];

/// Note appended when a resource cap stopped disassembly.
const TRUNCATED: &str = " Analysis stopped at a resource cap; results are partial.";

/// Explanation placed above the key function listings.
const KEY_NOTE: &str = "Inferred: functions are ranked by the capability-relevant APIs they \
call and the indicators they reference. Listings use Intel syntax with imports, functions, and \
strings resolved in comments.";

/// Builds the code analysis and key function sections.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The sections when disassembly ran, an explanation for other executables, or nothing.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    match (&analysis.binary, &analysis.disassembly) {
        (_, Some(code)) => vec![overview(code), key_functions(&analysis.key_functions)],
        (Some(binary), None) => vec![unsupported(&binary.arch_name)],
        (None, None) => Vec::new(),
    }
}

/// Explains why no disassembly is present.
///
/// # Arguments
///
/// * `arch` - Architecture name.
///
/// # Returns
///
/// The code analysis section.
fn unsupported(arch: &str) -> Section {
    let text = format!(
        "Observed: the architecture is {arch}, which this build does not yet disassemble. \
         Identity, sections, imports, strings, and indicators above still apply."
    );
    Section::new("Code Analysis").with(Block::Paragraph(text))
}

/// Builds the code overview with the largest functions.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The code analysis section.
fn overview(code: &Disassembly) -> Section {
    let note = Block::Paragraph(overview_note(code));
    let section = Section::new("Code Analysis").with(note);
    let section = section.with(text::subheading("Largest Functions"));
    section.with(Block::Table(largest(code)))
}

/// Summarizes recovered code coverage.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// One paragraph.
fn overview_note(code: &Disassembly) -> String {
    let functions = text::counted(code.functions.len(), "function");
    let instructions = text::counted(code.instructions, "instruction");
    let capped = text::when(code.truncated, TRUNCATED);
    let bits = code.bitness;
    format!(
        "Derived: {bits}-bit recursive descent recovered {functions} and {instructions}.{capped}"
    )
}

/// Tabulates the largest functions by instruction count.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The function table.
fn largest(code: &Disassembly) -> Table {
    let mut functions: Vec<&Function> = code.functions.values().collect();
    functions.sort_by_key(|f| Reverse(f.instructions));
    let top = functions.into_iter().take(TOP_FUNCTIONS);
    let rows = top.map(|f| function_row(code, f)).collect();
    text::table(&FUNCTION_HEADERS, rows)
}

/// Formats one function row.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `function` - Function to describe.
///
/// # Returns
///
/// The row cells.
fn function_row(code: &Disassembly, function: &Function) -> Vec<String> {
    vec![
        code.name_of(function.address),
        text::hex(function.address),
        function.instructions.to_string(),
        code.callers(function.address).to_string(),
        function.calls.len().to_string(),
        imports_text(function),
    ]
}

/// Lists the imports a function uses.
///
/// # Arguments
///
/// * `function` - Function to describe.
///
/// # Returns
///
/// Up to [`ROW_IMPORTS`] import names.
fn imports_text(function: &Function) -> String {
    let imports: Vec<String> = function.imports.iter().cloned().collect();
    text::listed(&imports, ROW_IMPORTS)
}

/// Builds the key function section with reasons and listings.
///
/// # Arguments
///
/// * `keys` - Ranked key functions.
///
/// # Returns
///
/// The key function section.
fn key_functions(keys: &[KeyFunction]) -> Section {
    let note = text::paragraph(KEY_NOTE);
    let section = Section::new("Key Functions").with(note);
    keys.iter().fold(section, with_key)
}

/// Appends one key function.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `key` - Key function.
///
/// # Returns
///
/// The extended section.
fn with_key(section: Section, key: &KeyFunction) -> Section {
    let title = format!("{} at {}", key.name, text::hex(key.address));
    let section = section.with(Block::Subheading(title));
    let section = section.with(Block::Bullets(key.reasons.clone()));
    section.with(Block::Code(key.listing.join("\n")))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the explanation for unsupported architectures.
    #[test]
    fn test_unsupported_names_arch() {
        let section = unsupported("aarch64");
        assert_eq!(section.blocks.len(), 1);
    }

    /// Verifies that key functions render heading, reasons, and listing.
    #[test]
    fn test_key_functions_blocks() {
        let reasons = vec!["Program entry point".to_string()];
        let key = KeyFunction::new(0x1000, "entry".into(), 3, reasons);
        assert_eq!(key_functions(&[key]).blocks.len(), 4);
    }

    /// Verifies that data files produce no code sections.
    #[test]
    fn test_sections_skip_data() {
        assert!(sections(&rea_engine::analyze("x", b"abc")).is_empty());
    }
}
