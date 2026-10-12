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
    disasm::{DataEntry, Disassembly, Function},
    insights::KeyFunction,
    narrative::{self, Evidence},
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

/// Note placed above the full walkthrough.
const WALKTHROUGH_NOTE: &str =
    "Observed: complete disassembly of every recovered function. Each function shows the full \
raw assembler listing first, followed by a plain-language summary of what the function actually \
does, and dual code/data analysis of any embedded literal pools.";

/// Data pool table columns.
const DATA_HEADERS: [&str; 5] = [
    "Address",
    "Hex Data",
    "As Data",
    "As Candidate Code",
    "Security Discussion",
];

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
        (_, Some(code)) => code_sections(analysis, code),
        (Some(binary), None) => vec![unsupported(&binary.arch_name)],
        (None, None) => Vec::new(),
    }
}

/// Builds all code analysis sections for recovered disassembly.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `code` - Disassembly result.
///
/// # Returns
///
/// List of sections.
fn code_sections(analysis: &Analysis, code: &Disassembly) -> Vec<Section> {
    let mut sections = vec![overview(code), key_functions(&analysis.key_functions)];
    if !code.functions.is_empty() {
        sections.push(walkthrough(analysis, code));
    }
    sections
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

/// Builds the exhaustive assembly walkthrough for all recovered functions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The assembly walkthrough section.
fn walkthrough(analysis: &Analysis, code: &Disassembly) -> Section {
    let note = text::paragraph(WALKTHROUGH_NOTE);
    let section = Section::new("Assembly Walkthrough & Instruction Analysis").with(note);
    let ev = build_walkthrough_evidence(analysis, code);
    let mut funcs: Vec<&Function> = code.functions.values().collect();
    funcs.sort_by_key(|f| f.address);
    funcs.into_iter().fold(section, |s, f| {
        with_function_walkthrough(s, ev.as_ref(), code, f)
    })
}

/// Builds the evidence resolver precomputed with the recovered call graph.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `code` - Disassembly result.
///
/// # Returns
///
/// The resolver when binary metadata exists.
fn build_walkthrough_evidence<'a>(
    analysis: &'a Analysis,
    code: &Disassembly,
) -> Option<Evidence<'a>> {
    let bin = analysis.binary.as_ref()?;
    Some(Evidence::new(bin, &analysis.strings).with_callers(code))
}

/// Appends one function's walkthrough blocks.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `ev` - Evidence resolver, when a parsed binary exists.
/// * `code` - Recovered code structure.
/// * `func` - Function to append.
///
/// # Returns
///
/// The extended section.
fn with_function_walkthrough(
    section: Section,
    ev: Option<&Evidence<'_>>,
    code: &Disassembly,
    func: &Function,
) -> Section {
    let section = append_func_heading(section, code, func);
    let section = append_raw_listing(section, func);
    let section = append_func_summary(section, ev, code, func);
    append_data_pools(section, code, func)
}

/// Appends the function heading.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
///
/// # Returns
///
/// The extended section.
fn append_func_heading(section: Section, code: &Disassembly, func: &Function) -> Section {
    let name = code.name_of(func.address);
    let title = format!(
        "{name} at {} ({} instructions)",
        text::hex(func.address),
        func.instructions
    );
    section.with(Block::Subheading(title))
}

/// Appends the complete raw assembler listing for a function.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `func` - Function whose listing is rendered.
///
/// # Returns
///
/// The extended section.
fn append_raw_listing(section: Section, func: &Function) -> Section {
    if func.lines.is_empty() {
        return section;
    }
    let listing: Vec<String> = func.lines.iter().map(format_line).collect();
    section.with(Block::Code(listing.join("\n")))
}

/// Formats one raw assembler line with its address.
///
/// # Arguments
///
/// * `line` - Decoded instruction line.
///
/// # Returns
///
/// Text such as `0x1000023a: movs r1, #0x2b`.
fn format_line(line: &rea_engine::disasm::InstructionLine) -> String {
    format!("{}: {}", text::hex(line.address), line.text)
}

/// Appends the plain-language summary of what the function does.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `ev` - Evidence resolver, when a parsed binary exists.
/// * `code` - Recovered code structure.
/// * `func` - Function to summarize.
///
/// # Returns
///
/// The extended section.
fn append_func_summary(
    section: Section,
    ev: Option<&Evidence<'_>>,
    code: &Disassembly,
    func: &Function,
) -> Section {
    let Some(ev) = ev else { return section };
    let story = narrative::function_story(ev, code, func);
    section.with(Block::Paragraph(story.join(" ")))
}

/// Appends literal pool and data analysis table.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `code` - Recovered code structure.
/// * `func` - Function whose data pools are rendered.
///
/// # Returns
///
/// The extended section.
fn append_data_pools(section: Section, code: &Disassembly, func: &Function) -> Section {
    if func.data_pools.is_empty() {
        return section;
    }
    let (title, table) = make_data_pool_blocks(code, func);
    section.with(Block::Subheading(title)).with(Block::Table(table))
}

/// Builds the subheading title and formatted table for literal pools.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function whose data pools are rendered.
///
/// # Returns
///
/// Pair of title string and formatted table.
fn make_data_pool_blocks(code: &Disassembly, func: &Function) -> (String, rea_report::Table) {
    let name = code.name_of(func.address);
    let title = format!("{name} - Literal Pools & Embedded Data");
    let rows: Vec<Vec<String>> = func.data_pools.iter().map(data_row).collect();
    (title, text::table(&DATA_HEADERS, rows))
}

/// Formats one data pool row.
///
/// # Arguments
///
/// * `entry` - Data entry.
///
/// # Returns
///
/// Row cells.
fn data_row(entry: &DataEntry) -> Vec<String> {
    let as_code = entry.as_code.as_deref().unwrap_or("none");
    vec![
        text::hex(entry.address),
        entry.bytes_hex.clone(),
        entry.as_data.clone(),
        as_code.to_string(),
        entry.discussion.clone(),
    ]
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

    /// Verifies that walkthrough renders when disassembly exists.
    #[test]
    fn test_walkthrough_section() {
        let analysis = rea_engine::analyze("x", b"abc");
        let section = walkthrough(&analysis, &Disassembly::default());
        assert_eq!(
            section.heading,
            "Assembly Walkthrough & Instruction Analysis"
        );
    }
}
