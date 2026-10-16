//! Key Function Ranking
//!
//! **File:** `insights.rs`
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
use std::{cmp::Reverse, collections::HashMap};

// Import crate items
use crate::{
    analysis::Analysis,
    capabilities::category_of,
    disasm::{CodeMap, Disassembly, Function},
    iocs::Indicator,
    listing,
    model::{Arch, Binary},
    multiarch,
    strings::FoundString,
};

/// Maximum functions highlighted in the report.
pub const MAX_KEY_FUNCTIONS: usize = 8;

/// Maximum instructions listed per highlighted function.
pub const LISTING_LINES: usize = 60;

/// Maximum characters of a string shown in a listing comment.
const NOTE_CHARS: usize = 60;

/// Maximum reasons recorded per function.
const MAX_REASONS: usize = 8;

/// A function worth an analyst's attention, with evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyFunction {
    /// Entry address.
    pub address: u64,

    /// Display name.
    pub name: String,

    /// Relevance score.
    pub score: u32,

    /// Evidence-backed reasons for the score.
    pub reasons: Vec<String>,

    /// Annotated disassembly listing.
    pub listing: Vec<String>,
}

/// Address-to-comment lookup used to annotate listings.
pub struct Annotator {
    /// Import slot address to `library!name`.
    slots: HashMap<u64, String>,

    /// Function address to name.
    names: HashMap<u64, String>,

    /// String address to text.
    strings: HashMap<u64, String>,

    /// String address to indicator description.
    indicators: HashMap<u64, String>,
}

impl Annotator {
    /// Builds the lookup tables for one analysis.
    ///
    /// # Arguments
    ///
    /// * `analysis` - Completed analysis.
    /// * `binary` - Parsed executable.
    /// * `disassembly` - Recovered code structure.
    ///
    /// # Returns
    ///
    /// The annotator.
    #[must_use]
    pub fn new(analysis: &Analysis, binary: &Binary, disassembly: &Disassembly) -> Self {
        Self {
            slots: slot_names(binary),
            names: function_names(disassembly),
            strings: string_addresses(analysis, binary),
            indicators: indicator_addresses(analysis, binary),
        }
    }

    /// Returns a comment for an address: import, function, or string.
    ///
    /// # Arguments
    ///
    /// * `address` - Referenced address.
    ///
    /// # Returns
    ///
    /// The comment text, when the address is known.
    #[must_use]
    pub fn note(&self, address: u64) -> Option<String> {
        let slot = self.slots.get(&address);
        let named = slot.or_else(|| self.names.get(&address)).cloned();
        let string = self.strings.get(&address).map(|s| quote(s));
        named.or(string)
    }

    /// Describes the indicator stored at an address.
    ///
    /// # Arguments
    ///
    /// * `address` - Referenced address.
    ///
    /// # Returns
    ///
    /// The indicator description, when any.
    #[must_use]
    pub fn indicator(&self, address: u64) -> Option<&String> {
        self.indicators.get(&address)
    }
}

/// Maps import slots to qualified import names.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Slot address to `library!name`.
fn slot_names(binary: &Binary) -> HashMap<u64, String> {
    let slots = binary.imports.iter().filter(|i| i.address != 0);
    slots.map(|i| (i.address, slot_name(&i.library, &i.name))).collect()
}

/// Maps function addresses to their known names.
///
/// # Arguments
///
/// * `disassembly` - Recovered code structure.
///
/// # Returns
///
/// Address to name for named functions.
fn function_names(disassembly: &Disassembly) -> HashMap<u64, String> {
    let functions = disassembly.functions.values();
    functions.filter_map(|f| Some((f.address, f.name.clone()?))).collect()
}

/// Maps string virtual addresses to their text.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Address to text for strings inside mapped sections.
fn string_addresses(analysis: &Analysis, binary: &Binary) -> HashMap<u64, String> {
    let place = |s: &FoundString| Some((binary.address_of(s.offset)?, s.text.clone()));
    analysis.strings.iter().filter_map(place).collect()
}

/// Maps indicator virtual addresses to descriptions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Address to a `kind` and `value` description for mapped indicators.
fn indicator_addresses(analysis: &Analysis, binary: &Binary) -> HashMap<u64, String> {
    let describe = |i: &Indicator| format!("{} \"{}\"", i.kind, i.value);
    let place = |i: &Indicator| Some((binary.address_of(i.offset)?, describe(i)));
    analysis.indicators.iter().filter_map(place).collect()
}

/// Quotes a string for a listing comment.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The clipped text in double quotes.
fn quote(text: &str) -> String {
    format!("\"{}\"", clip(text))
}

/// Formats an import as `library!name`.
///
/// # Arguments
///
/// * `library` - Providing library, possibly empty.
/// * `name` - Imported function.
///
/// # Returns
///
/// The qualified name.
fn slot_name(library: &str, name: &str) -> String {
    if library.is_empty() {
        name.to_string()
    } else {
        format!("{library}!{name}")
    }
}

/// Shortens text for a one-line comment.
///
/// # Arguments
///
/// * `text` - Text to shorten.
///
/// # Returns
///
/// At most [`NOTE_CHARS`] characters, with an ellipsis when cut.
fn clip(text: &str) -> String {
    let short: String = text.chars().take(NOTE_CHARS).collect();
    if short.len() < text.len() {
        short + "..."
    } else {
        short
    }
}

/// Ranks functions by evidence of interesting behavior and lists the top ones.
///
/// # Arguments
///
/// * `analysis` - Analysis with binary, strings, indicators, and disassembly.
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Up to [`MAX_KEY_FUNCTIONS`] functions, highest score first.
#[must_use]
pub fn rank(analysis: &Analysis, bytes: &[u8]) -> Vec<KeyFunction> {
    let (Some(binary), Some(disassembly)) = (&analysis.binary, &analysis.disassembly) else {
        return Vec::new();
    };
    let annotator = Annotator::new(analysis, binary, disassembly);
    let ranked = candidates(binary, disassembly, &annotator);
    listed(ranked, binary, bytes, &annotator)
}

/// Attaches listings to ranked functions.
///
/// # Arguments
///
/// * `ranked` - Functions without listings.
/// * `code` - Executable regions.
/// * `bitness` - Decoder bitness.
/// * `annotator` - Comment source.
///
/// # Returns
///
/// The functions with listings.
fn listed(
    ranked: Vec<KeyFunction>,
    binary: &Binary,
    bytes: &[u8],
    annotator: &Annotator,
) -> Vec<KeyFunction> {
    let keys = ranked.into_iter();
    keys.map(|k| with_listing(k, binary, bytes, annotator)).collect()
}

/// Scores every function and keeps the best.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `disassembly` - Recovered code structure.
/// * `annotator` - Indicator lookup.
///
/// # Returns
///
/// The top functions without listings.
fn candidates(
    binary: &Binary,
    disassembly: &Disassembly,
    annotator: &Annotator,
) -> Vec<KeyFunction> {
    let functions = disassembly.functions.values();
    let scored = functions.map(|f| evaluate(f, binary, disassembly, annotator));
    let mut ranked: Vec<KeyFunction> = scored.filter(|k| k.score > 0).collect();
    ranked.sort_by_key(|k| (Reverse(k.score), k.address));
    ranked.truncate(MAX_KEY_FUNCTIONS);
    ranked
}

/// Scores one function.
///
/// # Arguments
///
/// * `function` - Function to score.
/// * `binary` - Parsed executable.
/// * `disassembly` - Recovered code structure.
/// * `annotator` - Indicator lookup.
///
/// # Returns
///
/// The scored function without a listing.
fn evaluate(
    function: &Function,
    binary: &Binary,
    disassembly: &Disassembly,
    annotator: &Annotator,
) -> KeyFunction {
    let mut reasons = collect_reasons(function, binary.entry, annotator);
    let score = score_reasons(&reasons);
    reasons.truncate(MAX_REASONS);
    let name = disassembly.name_of(function.address);
    KeyFunction::new(function.address, name, score, reasons)
}

/// Collects all relevance reasons for a function.
///
/// # Arguments
///
/// * `func` - Function to evaluate.
/// * `entry` - Program entry address.
/// * `annotator` - Symbol annotator.
///
/// # Returns
///
/// List of discovered reasons.
fn collect_reasons(func: &Function, entry: u64, annotator: &Annotator) -> Vec<String> {
    let mut r = loop_reasons(func, annotator);
    r.extend(entry_reason(func.address == entry));
    r.extend(api_reasons(func));
    r.extend(indicator_reasons(func, annotator));
    r.extend(string_reasons(func, annotator));
    r
}

/// Computes prioritized relevance score from discovered reasons.
///
/// # Arguments
///
/// * `reasons` - Discovered function reasons.
///
/// # Returns
///
/// Weighted score, boosted for main loops.
fn score_reasons(reasons: &[String]) -> u32 {
    let base = weight(reasons.len());
    if reasons.iter().any(|r| r.starts_with("Contains a loop that prints")) {
        return base.saturating_add(100);
    }
    if reasons.iter().any(|r| r.starts_with("Contains a backward-branching loop")) {
        return base.saturating_add(50);
    }
    base
}

/// Converts a reason count into a relevance score.
///
/// # Arguments
///
/// * `count` - Number of reasons.
///
/// # Returns
///
/// Two points per reason, saturating.
fn weight(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX).saturating_mul(2)
}

impl KeyFunction {
    /// Creates a key function without a listing.
    ///
    /// # Arguments
    ///
    /// * `address` - Entry address.
    /// * `name` - Display name.
    /// * `score` - Relevance score.
    /// * `reasons` - Evidence-backed reasons.
    ///
    /// # Returns
    ///
    /// The key function.
    #[must_use]
    pub fn new(address: u64, name: String, score: u32, reasons: Vec<String>) -> Self {
        Self { address, name, score, reasons, listing: Vec::new() }
    }
}

/// Notes when a function is the program entry point.
///
/// # Arguments
///
/// * `is_entry` - Whether the function is the entry point.
///
/// # Returns
///
/// The entry-point reason, or nothing.
fn entry_reason(is_entry: bool) -> Vec<String> {
    let reason = "Program entry point".to_string();
    if is_entry {
        vec![reason]
    } else {
        Vec::new()
    }
}

/// Explains the capability-relevant imports a function calls.
///
/// # Arguments
///
/// * `function` - Function to inspect.
///
/// # Returns
///
/// One reason per relevant import.
fn api_reasons(function: &Function) -> Vec<String> {
    let imports = function.imports.iter();
    let relevant = imports.filter_map(|name| Some((name, category_of(name)?)));
    relevant.map(|(name, category)| format!("Calls {name} ({category})")).collect()
}

/// Explains the indicators a function references.
///
/// # Arguments
///
/// * `function` - Function to inspect.
/// * `annotator` - Indicator lookup.
///
/// # Returns
///
/// One reason per referenced indicator.
fn indicator_reasons(function: &Function, annotator: &Annotator) -> Vec<String> {
    let references = function.references.iter();
    let found = references.filter_map(|a| annotator.indicator(*a));
    found.map(|i| format!("References {i}")).collect()
}

/// Explains strings a function references directly or via literal pools.
///
/// # Arguments
///
/// * `function` - Function to inspect.
/// * `annotator` - Symbol lookup.
///
/// # Returns
///
/// Reasons for referenced strings.
fn string_reasons(function: &Function, annotator: &Annotator) -> Vec<String> {
    let direct = function.references.iter().filter_map(|a| annotator.strings.get(a));
    let pool = function.data_pools.iter().filter_map(|p| pool_string(p, annotator));
    let valid = direct.chain(pool).filter(|s| crate::strings::is_plausible(s));
    let mut all: Vec<String> = valid.map(|s| format!("References text {}", quote(s))).collect();
    all.dedup();
    all
}

/// Resolves a string referenced inside a literal pool entry.
///
/// # Arguments
///
/// * `entry` - Data pool entry.
/// * `annotator` - Symbol lookup.
///
/// # Returns
///
/// Referenced string text if found.
fn pool_string<'a>(
    entry: &crate::disasm::DataEntry,
    annotator: &'a Annotator,
) -> Option<&'a String> {
    let target = entry.as_data.split('[').next()?.trim();
    let hex = target.strip_prefix("0x")?;
    let addr = u64::from_str_radix(hex, 16).ok()?;
    annotator.strings.get(&addr)
}

/// Finds the first format string referenced by a function.
///
/// # Arguments
///
/// * `function` - Function to inspect.
/// * `annotator` - Symbol lookup.
///
/// # Returns
///
/// Referenced format string text if present.
fn find_format_string(function: &Function, annotator: &Annotator) -> Option<String> {
    let direct = function.references.iter().filter_map(|a| annotator.strings.get(a));
    let pool = function.data_pools.iter().filter_map(|p| pool_string(p, annotator));
    direct.chain(pool).find(|s| s.contains('%')).cloned()
}

/// Reports whether the function does not terminate in a breakpoint trap loop.
///
/// # Arguments
///
/// * `function` - Function to inspect.
///
/// # Returns
///
/// `true` if the second-to-last instruction is not a trap.
fn is_not_trap(function: &Function) -> bool {
    let len = function.lines.len();
    if len < 2 {
        return true;
    }
    let prev = &function.lines[len - 2].text;
    !prev.starts_with("bkpt") && !prev.starts_with("udf") && !prev.starts_with("wfi")
}

/// Reports whether function exhibits a non-trap backward self-loop.
fn is_loop_candidate(function: &Function) -> bool {
    function.lines.last().is_some_and(|l| is_self_loop(l, function.address))
        && is_not_trap(function)
}

/// Fallback reason when no format string is identified in loop.
fn fallback_loop_reasons(function: &Function) -> Vec<String> {
    (!function.calls.is_empty())
        .then(|| "Contains a backward-branching loop".into())
        .into_iter()
        .collect()
}

/// Formats reason when format string was discovered in loop.
fn format_loop_reason(fmt: &str) -> Vec<String> {
    vec![format!("Contains a loop that prints {}", quote(fmt))]
}

/// Explains main loop self-branching behavior.
///
/// # Arguments
///
/// * `function` - Function to inspect.
/// * `annotator` - Symbol lookup.
///
/// # Returns
///
/// Reason when function is a main loop candidate.
fn loop_reasons(function: &Function, annotator: &Annotator) -> Vec<String> {
    if !is_loop_candidate(function) {
        return Vec::new();
    }
    find_format_string(function, annotator).map_or_else(
        || fallback_loop_reasons(function),
        |fmt| format_loop_reason(&fmt),
    )
}

/// Reports whether an instruction line is a backward branch into the function.
///
/// # Arguments
///
/// * `line` - Instruction line.
/// * `entry` - Function entry address.
///
/// # Returns
///
/// `true` for branches to addresses >= entry and < instruction address.
fn is_self_loop(line: &crate::disasm::InstructionLine, entry: u64) -> bool {
    let target = line.text.strip_prefix("b ").and_then(strip_hex);
    target.is_some_and(|t| t >= entry && t < line.address)
}

/// Parses a hexadecimal address token.
///
/// # Arguments
///
/// * `token` - Token text.
///
/// # Returns
///
/// Parsed address when valid.
fn strip_hex(token: &str) -> Option<u64> {
    let hex = token.trim().strip_prefix("0x")?;
    u64::from_str_radix(hex, 16).ok()
}

/// Attaches an annotated listing to a key function.
///
/// # Arguments
///
/// * `key` - Function without a listing.
/// * `code` - Executable regions.
/// * `bitness` - Decoder bitness.
/// * `annotator` - Comment source.
///
/// # Returns
///
/// The function with its listing.
fn with_listing(
    mut key: KeyFunction,
    binary: &Binary,
    bytes: &[u8],
    annotator: &Annotator,
) -> KeyFunction {
    key.listing = render_listing(binary, bytes, key.address, annotator);
    key
}

/// Renders a function listing with the decoder for the binary's architecture.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
/// * `address` - Function entry address.
/// * `annotator` - Comment source.
///
/// # Returns
///
/// The annotated listing lines.
fn render_listing(
    binary: &Binary,
    bytes: &[u8],
    address: u64,
    annotator: &Annotator,
) -> Vec<String> {
    let note = |a: u64| annotator.note(a);
    match binary.arch {
        Arch::X86 | Arch::X86_64 => x86_listing(binary, bytes, address, &note),
        _ => multiarch::render_listing(binary, bytes, address, LISTING_LINES, &note)
            .unwrap_or_default(),
    }
}

/// Renders a listing with the x86 decoder.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
/// * `address` - Function entry address.
/// * `note` - Comment source.
///
/// # Returns
///
/// The annotated listing lines.
fn x86_listing(
    binary: &Binary,
    bytes: &[u8],
    address: u64,
    note: &listing::Annotate<'_>,
) -> Vec<String> {
    let code = CodeMap::new(binary, bytes);
    listing::render(&code, binary.bits, address, LISTING_LINES, note)
}

/// Deterministic summary of the recovered call graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CallGraph {
    /// Total function count.
    pub functions: usize,

    /// Total internal call-edge count.
    pub edges: usize,

    /// Entry-point function name, when known.
    pub entry: String,

    /// Most-called functions as name and caller count.
    pub hubs: Vec<(String, usize)>,

    /// Leaf-function count.
    pub leaves: usize,

    /// Recursive function names.
    pub recursive: Vec<String>,
}

/// Builds a deterministic call-graph summary.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The summary, or `None` when no disassembly exists.
#[must_use]
pub fn call_graph(analysis: &Analysis) -> Option<CallGraph> {
    let code = analysis.disassembly.as_ref()?;
    let mut graph = CallGraph::default();
    fill_shape(&mut graph, code);
    fill_names(&mut graph, analysis, code);
    Some(graph)
}

/// Fills structural call-graph counts.
///
/// # Arguments
///
/// * `graph` - Summary to populate.
/// * `code` - Recovered code structure.
fn fill_shape(graph: &mut CallGraph, code: &Disassembly) {
    graph.functions = code.functions.len();
    graph.edges = code.functions.values().map(|f| f.calls.len()).sum();
    graph.leaves = code.functions.values().filter(|f| f.calls.is_empty()).count();
}

/// Fills named call-graph fields.
///
/// # Arguments
///
/// * `graph` - Summary to populate.
/// * `analysis` - Completed analysis.
/// * `code` - Recovered code structure.
fn fill_names(graph: &mut CallGraph, analysis: &Analysis, code: &Disassembly) {
    graph.entry = entry_name(analysis, code);
    graph.hubs = hubs(code);
    graph.recursive = recursive(code);
}

/// Names the entry-point function.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The entry-point name, or an empty string.
fn entry_name(analysis: &Analysis, code: &Disassembly) -> String {
    let entry = analysis.binary.as_ref().map_or(0, |b| b.entry);
    if entry == 0 || !code.functions.contains_key(&entry) {
        return String::new();
    }
    code.name_of(entry)
}

/// Ranks the most-called internal functions.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// Up to three `(name, caller count)` pairs.
fn hubs(code: &Disassembly) -> Vec<(String, usize)> {
    let mut degree: HashMap<u64, usize> = HashMap::new();
    for function in code.functions.values() {
        bump(&mut degree, &function.calls);
    }
    ranked(&degree, code)
}

/// Increments the caller count for each call target.
///
/// # Arguments
///
/// * `degree` - Callee to caller count.
/// * `targets` - Call targets.
fn bump(degree: &mut HashMap<u64, usize>, targets: &std::collections::BTreeSet<u64>) {
    for &target in targets {
        *degree.entry(target).or_default() += 1;
    }
}

/// Sorts and truncates the caller-degree list.
///
/// # Arguments
///
/// * `degree` - Callee to caller count.
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The ranked hubs.
fn ranked(degree: &HashMap<u64, usize>, code: &Disassembly) -> Vec<(String, usize)> {
    let mut pairs: Vec<(u64, usize)> = degree.iter().map(|(&a, &n)| (a, n)).collect();
    pairs.sort_by_key(|&(_, n)| Reverse(n));
    pairs
        .iter()
        .filter(|&&(_, n)| n >= 2)
        .take(3)
        .map(|&(a, n)| (code.name_of(a), n))
        .collect()
}

/// Lists recursive functions.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// The recursive function names.
fn recursive(code: &Disassembly) -> Vec<String> {
    code.functions
        .values()
        .filter(|f| f.calls.contains(&f.address))
        .map(|f| code.name_of(f.address))
        .collect()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies qualified import naming.
    #[test]
    fn test_slot_name_qualifies_library() {
        assert_eq!(slot_name("KERNEL32.dll", "Sleep"), "KERNEL32.dll!Sleep");
        assert_eq!(slot_name("", "puts"), "puts");
    }

    /// Verifies that long strings are clipped with an ellipsis.
    #[test]
    fn test_clip_long_text() {
        let long = "x".repeat(NOTE_CHARS + 5);
        assert_eq!(clip(&long).len(), NOTE_CHARS + 3);
        assert_eq!(clip("short"), "short");
    }

    /// Verifies that capability-relevant imports produce reasons.
    #[test]
    fn test_api_reasons_flag_relevant_imports() {
        let mut function = Function::default();
        function.imports.insert("CreateRemoteThread".into());
        function.imports.insert("HeapFree".into());
        assert_eq!(
            api_reasons(&function),
            vec!["Calls CreateRemoteThread (Process injection)"]
        );
    }

    /// Verifies that analyses without code yield no key functions.
    #[test]
    fn test_rank_without_code_is_empty() {
        assert_eq!(rank(&Analysis::default(), &[]).len(), 0);
    }
}
