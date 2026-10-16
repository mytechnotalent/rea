//! Differential Binary Comparison
//!
//! **File:** `compare.rs`
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
use std::collections::BTreeSet;

// Import dependencies from std
use std::fmt::Write as _;

// Import dependencies from crate
use crate::{crypto, rules, Analysis};

/// Maximum items listed per difference.
const LIMIT: usize = 100;

/// Maximum strings considered per side.
const STRING_LIMIT: usize = 20_000;

/// A differential comparison between two analyzed targets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Comparison {
    /// First target name.
    pub a_name: String,

    /// Second target name.
    pub b_name: String,

    /// Whether the two targets share a SHA-256.
    pub identical: bool,

    /// Format difference summary.
    pub format: String,

    /// Architecture difference summary.
    pub arch: String,

    /// Size difference summary.
    pub size: String,

    /// Entropy difference summary.
    pub entropy: String,

    /// Function-count difference summary.
    pub functions: String,

    /// Instruction-count difference summary.
    pub instructions: String,

    /// Imports present only in the second target.
    pub added_imports: Vec<String>,

    /// Imports present only in the first target.
    pub removed_imports: Vec<String>,

    /// Strings present only in the second target.
    pub added_strings: Vec<String>,

    /// Strings present only in the first target.
    pub removed_strings: Vec<String>,

    /// Capabilities present only in the second target.
    pub added_capabilities: Vec<String>,

    /// Capabilities present only in the first target.
    pub removed_capabilities: Vec<String>,

    /// Rules matched only in the second target.
    pub added_rules: Vec<String>,

    /// Rules matched only in the first target.
    pub removed_rules: Vec<String>,

    /// Crypto constants present only in the second target.
    pub added_crypto: Vec<String>,

    /// Crypto constants present only in the first target.
    pub removed_crypto: Vec<String>,
}

/// Compares two analyzed targets.
///
/// # Arguments
///
/// * `a` - First analysis.
/// * `b` - Second analysis.
/// * `ab` - First target bytes.
/// * `bb` - Second target bytes.
///
/// # Returns
///
/// The comparison.
#[must_use]
pub fn compare(a: &Analysis, b: &Analysis, ab: &[u8], bb: &[u8]) -> Comparison {
    let mut c = Comparison::default();
    identity(&mut c, a, b);
    code(&mut c, a, b);
    string_diff(&mut c, a, b);
    set_diff(&mut c, a, b, ab, bb);
    c
}

/// Fills identity and fingerprint differences.
///
/// # Arguments
///
/// * `c` - Comparison to fill.
/// * `a` - First analysis.
/// * `b` - Second analysis.
fn identity(c: &mut Comparison, a: &Analysis, b: &Analysis) {
    c.a_name.clone_from(&a.name);
    c.b_name.clone_from(&b.name);
    c.identical = a.hashes.sha256 == b.hashes.sha256;
    c.format = changed(a.format.label(), b.format.label());
    c.arch = changed(arch(a), arch(b));
    c.size = pair_u64(a.size, b.size);
    c.entropy = pair_f64(a.entropy, b.entropy);
}

/// Fills function and instruction count differences.
///
/// # Arguments
///
/// * `c` - Comparison to fill.
/// * `a` - First analysis.
/// * `b` - Second analysis.
fn code(c: &mut Comparison, a: &Analysis, b: &Analysis) {
    c.functions = pair_usize(count_functions(a), count_functions(b));
    c.instructions = pair_usize(count_instructions(a), count_instructions(b));
}

/// Fills string differences.
///
/// # Arguments
///
/// * `c` - Comparison to fill.
/// * `a` - First analysis.
/// * `b` - Second analysis.
fn string_diff(c: &mut Comparison, a: &Analysis, b: &Analysis) {
    c.added_strings = added(&string_set(a), &string_set(b));
    c.removed_strings = added(&string_set(b), &string_set(a));
}

/// Fills import, capability, rule, and crypto differences.
///
/// # Arguments
///
/// * `c` - Comparison to fill.
/// * `a` - First analysis.
/// * `b` - Second analysis.
/// * `ab` - First target bytes.
/// * `bb` - Second target bytes.
fn set_diff(c: &mut Comparison, a: &Analysis, b: &Analysis, ab: &[u8], bb: &[u8]) {
    c.added_imports = added(&imports(a), &imports(b));
    c.removed_imports = added(&imports(b), &imports(a));
    c.added_capabilities = added(&caps(a), &caps(b));
    c.removed_capabilities = added(&caps(b), &caps(a));
    c.added_rules = added(&rule_set(a, ab), &rule_set(b, bb));
    c.removed_rules = added(&rule_set(b, bb), &rule_set(a, ab));
    c.added_crypto = added(&crypto_set(ab), &crypto_set(bb));
    c.removed_crypto = added(&crypto_set(bb), &crypto_set(ab));
}

/// Returns the target architecture name.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The architecture name.
fn arch(a: &Analysis) -> &str {
    a.binary.as_ref().map_or("unknown", |b| b.arch_name.as_str())
}

/// Counts recovered functions.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The function count.
fn count_functions(a: &Analysis) -> usize {
    a.disassembly.as_ref().map_or(0, |d| d.functions.len())
}

/// Counts recovered instructions.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The instruction count.
fn count_instructions(a: &Analysis) -> usize {
    a.disassembly.as_ref().map_or(0, |d| d.instructions)
}

/// Collects imported function names.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The import names.
fn imports(a: &Analysis) -> BTreeSet<String> {
    a.binary.as_ref().map_or_else(BTreeSet::new, |b| {
        b.imports.iter().map(|i| i.name.clone()).collect()
    })
}

/// Collects recovered strings.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The string set.
fn string_set(a: &Analysis) -> BTreeSet<String> {
    a.strings.iter().take(STRING_LIMIT).map(|s| s.text.clone()).collect()
}

/// Collects capability categories.
///
/// # Arguments
///
/// * `a` - Completed analysis.
///
/// # Returns
///
/// The capability set.
fn caps(a: &Analysis) -> BTreeSet<String> {
    a.capabilities.iter().map(|c| c.category.to_string()).collect()
}

/// Collects matched rule names.
///
/// # Arguments
///
/// * `a` - Completed analysis.
/// * `bytes` - Target bytes.
///
/// # Returns
///
/// The rule set.
fn rule_set(a: &Analysis, bytes: &[u8]) -> BTreeSet<String> {
    rules::evaluate(a, bytes).iter().map(|m| m.name.clone()).collect()
}

/// Collects detected crypto constant names.
///
/// # Arguments
///
/// * `bytes` - Target bytes.
///
/// # Returns
///
/// The crypto set.
fn crypto_set(bytes: &[u8]) -> BTreeSet<String> {
    crypto::detect(bytes).iter().map(|h| h.name.clone()).collect()
}

/// Returns items present in `b` but not `a`.
///
/// # Arguments
///
/// * `a` - First set.
/// * `b` - Second set.
///
/// # Returns
///
/// The difference.
fn added(a: &BTreeSet<String>, b: &BTreeSet<String>) -> Vec<String> {
    b.difference(a).take(LIMIT).cloned().collect()
}

/// Formats a value pair that is unchanged or marked changed.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The difference summary.
fn changed(a: &str, b: &str) -> String {
    if a == b {
        a.to_string()
    } else {
        format!("{a} -> {b}")
    }
}

/// Formats a 64-bit value pair.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The difference summary.
fn pair_u64(a: u64, b: u64) -> String {
    changed(&a.to_string(), &b.to_string())
}

/// Formats a usize value pair.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The difference summary.
fn pair_usize(a: usize, b: usize) -> String {
    changed(&a.to_string(), &b.to_string())
}

/// Formats a floating-point value pair.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The difference summary.
fn pair_f64(a: f64, b: f64) -> String {
    changed(&format!("{a:.4}"), &format!("{b:.4}"))
}

/// Renders a human-readable differential report.
///
/// # Arguments
///
/// * `c` - The comparison.
///
/// # Returns
///
/// The report text.
#[must_use]
pub fn format(c: &Comparison) -> String {
    let mut out = String::new();
    header(&mut out, c);
    sets_a(&mut out, c);
    sets_b(&mut out, c);
    out
}

/// Appends the import, string, and capability sections.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `c` - The comparison.
fn sets_a(out: &mut String, c: &Comparison) {
    section(out, "Imports", &c.added_imports, &c.removed_imports);
    section(out, "Strings", &c.added_strings, &c.removed_strings);
    section(
        out,
        "Capabilities",
        &c.added_capabilities,
        &c.removed_capabilities,
    );
}

/// Appends the rule and crypto sections.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `c` - The comparison.
fn sets_b(out: &mut String, c: &Comparison) {
    section(out, "Rules", &c.added_rules, &c.removed_rules);
    section(out, "Crypto", &c.added_crypto, &c.removed_crypto);
}

/// Appends the report header lines.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `c` - The comparison.
fn header(out: &mut String, c: &Comparison) {
    let _ = writeln!(out, "# Differential Analysis: {} vs {}", c.a_name, c.b_name);
    let _ = writeln!(out, "\nIdentical: {}", c.identical);
    kv(out, "Format", &c.format);
    kv(out, "Architecture", &c.arch);
    kv(out, "Size", &c.size);
    kv(out, "Entropy", &c.entropy);
    kv(out, "Functions", &c.functions);
    kv(out, "Instructions", &c.instructions);
}

/// Appends one key-value line.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `key` - Label.
/// * `value` - Value.
fn kv(out: &mut String, key: &str, value: &str) {
    let _ = writeln!(out, "- {key}: {value}");
}

/// Appends one difference section.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `title` - Section title.
/// * `added` - Added items.
/// * `removed` - Removed items.
fn section(out: &mut String, title: &str, added: &[String], removed: &[String]) {
    let _ = writeln!(out, "\n## {title}");
    let _ = writeln!(out, "- Added ({}): {}", added.len(), joined(added));
    let _ = writeln!(out, "- Removed ({}): {}", removed.len(), joined(removed));
}

/// Joins items or returns `none`.
///
/// # Arguments
///
/// * `items` - Items to join.
///
/// # Returns
///
/// The joined text.
fn joined(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies identical inputs compare as identical.
    #[test]
    fn test_compare_identical() {
        let a = crate::analyze("a.bin", b"GET /x HTTP/1.1");
        let b = crate::analyze("a.bin", b"GET /x HTTP/1.1");
        let c = compare(&a, &b, b"GET /x HTTP/1.1", b"GET /x HTTP/1.1");
        assert!(c.identical);
        assert_eq!(c.added_strings.len(), 0);
    }

    /// Verifies a new string is reported as added.
    #[test]
    fn test_compare_added_string() {
        let a = crate::analyze("a.bin", b"alpha string here");
        let b = crate::analyze("b.bin", b"beta string here");
        let c = compare(&a, &b, b"alpha string here", b"beta string here");
        assert!(!c.identical);
        assert!(format(&c).contains("beta string here"));
    }
}
