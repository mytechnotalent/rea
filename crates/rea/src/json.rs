//! Machine-Readable JSON Report
//!
//! **File:** `json.rs`
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
    crypto,
    regions::{self, RegionKind},
    rules, vulns, Analysis,
};

// Import dependencies from serde_json
use serde_json::{json, Map, Value};

/// Renders a machine-readable JSON report.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The pretty-printed JSON document.
#[must_use]
pub fn report(analysis: &Analysis, bytes: &[u8]) -> String {
    let mut map = Map::new();
    identity(&mut map, analysis);
    structure(&mut map, analysis);
    findings(&mut map, analysis, bytes);
    coverage(&mut map, analysis, bytes);
    serde_json::to_string_pretty(&Value::Object(map)).unwrap_or_default()
}

/// Inserts one key-value pair into the report object.
///
/// # Arguments
///
/// * `map` - Report object.
/// * `key` - Field name.
/// * `value` - Field value.
fn put(map: &mut Map<String, Value>, key: &str, value: Value) {
    map.insert(key.to_string(), value);
}

/// Inserts a coverage object describing analysis completeness.
///
/// # Arguments
///
/// * `map` - Report object.
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
fn coverage(map: &mut Map<String, Value>, analysis: &Analysis, bytes: &[u8]) {
    put(map, "coverage", coverage_value(analysis, bytes));
}

/// Builds the coverage object.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The coverage value.
fn coverage_value(analysis: &Analysis, bytes: &[u8]) -> Value {
    json!({
        "functions": count_functions(analysis),
        "instructions": count_instructions(analysis),
        "strings": analysis.strings.len(),
        "truncated": truncated(analysis),
        "opaque_bytes": opaque_bytes(bytes),
    })
}

/// Counts recovered functions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The function count.
fn count_functions(analysis: &Analysis) -> usize {
    analysis.disassembly.as_ref().map_or(0, |d| d.functions.len())
}

/// Counts recovered instructions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The instruction count.
fn count_instructions(analysis: &Analysis) -> usize {
    analysis.disassembly.as_ref().map_or(0, |d| d.instructions)
}

/// Reports whether disassembly stopped early.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// `true` when truncated.
fn truncated(analysis: &Analysis) -> bool {
    analysis.disassembly.as_ref().is_some_and(|d| d.truncated)
}

/// Sums encrypted and compressed region bytes.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The opaque byte count.
fn opaque_bytes(bytes: &[u8]) -> usize {
    let opaque = |kind: RegionKind| matches!(kind, RegionKind::Encrypted | RegionKind::Compressed);
    regions::segment(bytes).iter().filter(|r| opaque(r.kind)).map(|r| r.size).sum()
}

/// Inserts identity and fingerprint fields.
///
/// # Arguments
///
/// * `map` - Report object.
/// * `analysis` - Completed analysis.
fn identity(map: &mut Map<String, Value>, analysis: &Analysis) {
    put(map, "name", json!(analysis.name));
    put(map, "format", json!(analysis.format.label()));
    put(map, "size", json!(analysis.size));
    put(map, "entropy", json!(analysis.entropy));
    put(map, "sha256", json!(analysis.hashes.sha256));
}

/// Inserts architecture, mitigation, import, and capability fields.
///
/// # Arguments
///
/// * `map` - Report object.
/// * `analysis` - Completed analysis.
fn structure(map: &mut Map<String, Value>, analysis: &Analysis) {
    let arch = analysis.binary.as_ref().map_or("unknown", |b| b.arch_name.as_str());
    put(map, "architecture", json!(arch));
    put(map, "mitigations", json!(mitigations(analysis)));
    put(map, "imports", json!(import_names(analysis)));
    put(map, "capabilities", json!(capabilities(analysis)));
}

/// Inserts rule, vulnerability, and crypto findings.
///
/// # Arguments
///
/// * `map` - Report object.
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
fn findings(map: &mut Map<String, Value>, analysis: &Analysis, bytes: &[u8]) {
    put(map, "rules", json!(rule_names(analysis, bytes)));
    put(map, "vulnerabilities", json!(vuln_cwes(analysis)));
    put(map, "crypto", json!(crypto_names(bytes)));
}

/// Lists enabled mitigation names.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The mitigation names.
fn mitigations(analysis: &Analysis) -> Vec<String> {
    analysis.binary.as_ref().map_or_else(Vec::new, |b| {
        b.protections.iter().filter(|p| p.enabled).map(|p| p.name.clone()).collect()
    })
}

/// Lists imported function names.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The import names.
fn import_names(analysis: &Analysis) -> Vec<String> {
    analysis.binary.as_ref().map_or_else(Vec::new, |b| {
        b.imports.iter().map(|i| i.name.clone()).collect()
    })
}

/// Lists import-based capability categories.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The capability categories.
fn capabilities(analysis: &Analysis) -> Vec<String> {
    analysis.capabilities.iter().map(|c| c.category.to_string()).collect()
}

/// Lists matched capability rule names.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The rule names.
fn rule_names(analysis: &Analysis, bytes: &[u8]) -> Vec<String> {
    rules::evaluate(analysis, bytes).iter().map(|m| m.name.clone()).collect()
}

/// Lists vulnerability CWE identifiers.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The CWE identifiers.
fn vuln_cwes(analysis: &Analysis) -> Vec<String> {
    vulns::audit(analysis).iter().map(|v| v.cwe.clone()).collect()
}

/// Lists detected cryptographic constant names.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The constant names.
fn crypto_names(bytes: &[u8]) -> Vec<String> {
    crypto::detect(bytes).iter().map(|h| h.name.clone()).collect()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the report renders valid JSON with identity fields.
    #[test]
    fn test_report_has_name() {
        let analysis = rea_engine::analyze("blob.bin", b"plain data");
        let text = report(&analysis, b"plain data");
        assert!(text.contains("\"name\": \"blob.bin\""));
        assert!(text.contains("\"coverage\""));
    }
}
