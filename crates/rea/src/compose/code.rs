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
use std::{cmp::Reverse, collections::BTreeSet};

// Import dependencies from rea_engine
use rea_engine::{
    decompile,
    disasm::{DataEntry, Disassembly, Function},
    insights::KeyFunction,
    model::Arch,
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

/// Note placed above the function analysis.
const WALKTHROUGH_NOTE: &str =
    "Observed: the full instruction stream for every recovered function is written to \
    re-<name>.asm. This section summarizes what each recovered function does in plain language, \
    with its call relationships and embedded literal pools. Read the .asm artifact when you need \
    the raw assembly.";

/// Data pool table columns.
const DATA_HEADERS: [&str; 5] = [
    "Address",
    "Hex Data",
    "As Data",
    "As Candidate Code",
    "Security Discussion",
];

/// Largest function, in instructions, rendered as recovered pseudo-C.
const DECOMP_CAP: usize = 200;

/// Most functions per report rendered as recovered pseudo-C.
const DECOMP_LIMIT: usize = 64;

/// Most functions rendered in the function analysis.
const WALKTHROUGH_FUNCTIONS: usize = 16;

/// Most literal-pool rows shown per function.
const DATA_POOL_CAP: usize = 12;

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
    let mut sections = vec![overview(code), key_functions(analysis)];
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
fn key_functions(analysis: &Analysis) -> Section {
    let note = text::paragraph(KEY_NOTE);
    let section = Section::new("Key Functions").with(note);
    analysis
        .key_functions
        .iter()
        .fold(section, |acc, key| with_key(acc, analysis, key))
}

/// Appends one key function.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `analysis` - Completed analysis.
/// * `key` - Key function.
///
/// # Returns
///
/// The extended section.
fn with_key(section: Section, analysis: &Analysis, key: &KeyFunction) -> Section {
    let title = format!("{} at {}", key.name, text::hex(key.address));
    let section = section.with(Block::Subheading(title));
    let section = section.with(Block::Bullets(key.reasons.clone()));
    let section = section.with(Block::Code(key.listing.join("\n")));
    append_pseudocode(section, analysis, key.address)
}

/// Appends the recovered pseudo-C for a function.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `analysis` - Completed analysis.
/// * `address` - Function entry address.
///
/// # Returns
///
/// The extended section.
fn append_pseudocode(section: Section, analysis: &Analysis, address: u64) -> Section {
    match decompiled(analysis, address) {
        Some(code) => {
            let titled = section.with(Block::Subheading("Recovered Pseudo-C".to_string()));
            titled.with(Block::Code(code))
        }
        None => section,
    }
}

/// Decompiles the function at an address, when available.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `address` - Function entry address.
///
/// # Returns
///
/// The pseudo-C listing, or `None`.
fn decompiled(analysis: &Analysis, address: u64) -> Option<String> {
    let binary = analysis.binary.as_ref()?;
    let func = analysis.disassembly.as_ref()?.functions.get(&address)?;
    decompilable(binary.arch, func).then(|| decompile(func, binary.arch))
}

/// Reports whether a function may be rendered as pseudo-C.
///
/// # Arguments
///
/// * `arch` - Target architecture.
/// * `func` - Function to check.
///
/// # Returns
///
/// `true` for supported, small functions.
fn decompilable(arch: Arch, func: &Function) -> bool {
    matches!(arch, Arch::X86 | Arch::X86_64) && func.instructions <= DECOMP_CAP
}

/// Appends pseudo-C for a walkthrough function when it is selected.
///
/// # Arguments
///
/// * `section` - Section being built.
/// * `analysis` - Completed analysis.
/// * `func` - Function to decompile.
/// * `selected` - Addresses chosen for decompilation.
///
/// # Returns
///
/// The extended section.
fn append_selected(
    section: Section,
    analysis: &Analysis,
    func: &Function,
    selected: &BTreeSet<u64>,
) -> Section {
    if selected.contains(&func.address) {
        return append_pseudocode(section, analysis, func.address);
    }
    section
}

/// Selects the first functions to render as pseudo-C within the caps.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The selected function entry addresses.
fn decompile_targets(code: &Disassembly) -> BTreeSet<u64> {
    let mut funcs: Vec<&Function> =
        code.functions.values().filter(|f| f.instructions <= DECOMP_CAP).collect();
    funcs.sort_by_key(|f| f.address);
    funcs.into_iter().take(DECOMP_LIMIT).map(|f| f.address).collect()
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
    let section = walkthrough_heading();
    let ev = build_walkthrough_evidence(analysis, code);
    let selected = decompile_targets(code);
    let funcs = ranked_functions(analysis, code);
    funcs.into_iter().fold(section, |s, f| {
        with_function_walkthrough(s, analysis, ev.as_ref(), code, f, &selected)
    })
}

/// Lists the analyst-ranked functions, resolved to recovered functions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The ranked functions in descending relevance.
fn ranked_functions<'a>(analysis: &'a Analysis, code: &'a Disassembly) -> Vec<&'a Function> {
    analysis
        .key_functions
        .iter()
        .take(WALKTHROUGH_FUNCTIONS)
        .filter_map(|key| code.functions.get(&key.address))
        .collect()
}

/// Builds the walkthrough section heading.
///
/// # Returns
///
/// The section with its explanatory note.
fn walkthrough_heading() -> Section {
    let note = text::paragraph(WALKTHROUGH_NOTE);
    Section::new("Function Analysis").with(note)
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
    analysis: &Analysis,
    ev: Option<&Evidence<'_>>,
    code: &Disassembly,
    func: &Function,
    selected: &BTreeSet<u64>,
) -> Section {
    let section = append_func_heading(section, code, func);
    let section = append_func_summary(section, ev, code, func);
    let section = append_selected(section, analysis, func, selected);
    let section = append_call_graph(section, code, func);
    append_data_pools(section, code, func)
}

/// Appends the function's callers, callees, and imported calls.
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
fn append_call_graph(section: Section, code: &Disassembly, func: &Function) -> Section {
    let targets: Vec<String> = func.calls.iter().map(|a| code.name_of(*a)).collect();
    let imports: Vec<String> = func.imports.iter().cloned().collect();
    let text = graph_text(code, func, &targets, &imports);
    section.with(Block::Paragraph(text))
}

/// Builds the call-graph sentence for a function.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
/// * `targets` - Callee names.
/// * `imports` - Imported call names.
///
/// # Returns
///
/// The sentence.
fn graph_text(
    code: &Disassembly,
    func: &Function,
    targets: &[String],
    imports: &[String],
) -> String {
    let inbound = callers_of(code, func.address);
    let (calls, used) = (join_names(targets), join_names(imports));
    format!(
        "Calls: {calls}. Called by: {}. Imports used: {used}.",
        join_names(&inbound)
    )
}

/// Joins names or returns `none` when empty.
///
/// # Arguments
///
/// * `names` - Names to join.
///
/// # Returns
///
/// The joined text.
fn join_names(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// Collects the callers of a function.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `target` - Callee address.
///
/// # Returns
///
/// The caller names.
fn callers_of(code: &Disassembly, target: u64) -> Vec<String> {
    code.functions
        .values()
        .filter(|f| f.calls.contains(&target))
        .map(|f| code.name_of(f.address))
        .collect()
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
    let count = instructions_label(func.instructions);
    let title = format!("{name} at {} ({count})", text::hex(func.address));
    section.with(Block::Subheading(title))
}

/// Formats an instruction count with correct pluralization.
///
/// # Arguments
///
/// * `count` - Instruction count.
///
/// # Returns
///
/// The label text.
fn instructions_label(count: usize) -> String {
    if count == 1 {
        "1 instruction".to_string()
    } else {
        format!("{count} instructions")
    }
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
    let story = match ev {
        Some(ev) => narrative::function_story(ev, code, func),
        None => vec![structural_summary(code, func)],
    };
    section.with(Block::Paragraph(story.join(" ")))
}

/// Builds a structural summary when no parsed binary is available.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to summarize.
///
/// # Returns
///
/// The summary sentence.
fn structural_summary(code: &Disassembly, func: &Function) -> String {
    let name = code.name_of(func.address);
    let (calls, imports, refs) = (func.calls.len(), func.imports.len(), func.references.len());
    let branches = func.lines.iter().filter(|l| l.kind == "Control Flow").count();
    format!(
        "{name} is a {}-instruction routine making {calls} internal and {imports} imported \
call(s), with {branches} branch(es) and {refs} referenced address(es).",
        func.instructions
    )
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
    let rows: Vec<Vec<String>> = func.data_pools.iter().take(DATA_POOL_CAP).map(data_row).collect();
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
        let mut analysis = rea_engine::analyze("x", b"abc");
        analysis.key_functions = vec![key];
        assert_eq!(key_functions(&analysis).blocks.len(), 4);
    }

    /// Verifies that decompilation is skipped without a parsed binary.
    #[test]
    fn test_decompiled_without_binary() {
        let analysis = rea_engine::analyze("x", b"abc");
        assert!(decompiled(&analysis, 0x1000).is_none());
    }

    /// Verifies that data files produce no code sections.
    #[test]
    fn test_sections_skip_data() {
        assert_eq!(sections(&rea_engine::analyze("x", b"abc")).len(), 0);
    }

    /// Verifies that function analysis renders when disassembly exists.
    #[test]
    fn test_walkthrough_section() {
        let analysis = rea_engine::analyze("x", b"abc");
        let section = walkthrough(&analysis, &Disassembly::default());
        assert_eq!(section.heading, "Function Analysis");
    }
}
