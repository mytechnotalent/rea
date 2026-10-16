//! Audit-Grade Evidence Bundles
//!
//! **File:** `evidence.rs`
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

// Import dependencies from std
use std::fmt::Write as _;

// Import dependencies from crate
use crate::{
    crypto, hashes,
    regions::{self, RegionKind},
    rules, vulns, Analysis,
};

// Import dependencies from serde_json
use serde_json::{json, Value};

/// Evidence bundle schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// Maximum items collected per evidence source.
const ITEM_LIMIT: usize = 500;

/// Unknown text for partial disassembly.
const TRUNCATED: &str = "Disassembly stopped at a resource cap; some functions may be missing.";

/// Unknown text for a missing executable header.
const NO_HEADER: &str = "No executable header was parsed; imports and mitigations are unavailable.";

/// Confidence level of an evidence item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    /// Directly observed in the target.
    High,

    /// Derived deterministically.
    Medium,

    /// Inferred; confirm before relying on it.
    Low,
}

impl Confidence {
    /// Returns the lowercase label.
    ///
    /// # Returns
    ///
    /// The label text.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        }
    }
}

/// One addressable evidence item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// Content-addressed identifier.
    pub id: String,

    /// Evidence kind, such as `rule` or `crypto`.
    pub kind: String,

    /// Short subject.
    pub subject: String,

    /// Supporting detail.
    pub detail: String,

    /// Address the item refers to, when known.
    pub address: Option<u64>,

    /// Confidence level.
    pub confidence: Confidence,
}

/// Target identity within a bundle.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Target {
    /// File name.
    pub name: String,

    /// SHA-256 digest.
    pub sha256: String,

    /// File size in bytes.
    pub size: u64,

    /// Detected format label.
    pub format: String,
}

/// Coverage facts for a bundle.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Coverage {
    /// Recovered function count.
    pub functions: usize,

    /// Recovered instruction count.
    pub instructions: usize,

    /// Extracted string count.
    pub strings: usize,

    /// Whether disassembly stopped early.
    pub truncated: bool,

    /// Bytes in encrypted or compressed regions.
    pub opaque_bytes: usize,
}

/// A canonical evidence bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    /// Schema version.
    pub schema_version: u32,

    /// Producing tool version.
    pub tool_version: String,

    /// Target identity.
    pub target: Target,

    /// Coverage facts.
    pub coverage: Coverage,

    /// Addressable evidence items.
    pub items: Vec<Item>,

    /// Explicit unknowns and limitations.
    pub unknowns: Vec<String>,
}

/// Verification result for a bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyReport {
    /// Parsed schema version.
    pub schema_version: u32,

    /// Total item count.
    pub items: usize,

    /// Items whose identifier matched their content.
    pub verified: usize,

    /// Identifiers that failed the content check.
    pub tampered: Vec<String>,
}

/// Builds an evidence bundle for a target.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The evidence bundle.
#[must_use]
pub fn build(analysis: &Analysis, bytes: &[u8]) -> Bundle {
    Bundle {
        schema_version: SCHEMA_VERSION,
        tool_version: tool_version(),
        target: target(analysis),
        coverage: coverage(analysis, bytes),
        items: collect(analysis, bytes),
        unknowns: unknowns(analysis, bytes),
    }
}

/// Collects every evidence item.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The evidence items.
fn collect(analysis: &Analysis, bytes: &[u8]) -> Vec<Item> {
    let mut items = Vec::new();
    push_capabilities(&mut items, analysis);
    push_indicators(&mut items, analysis);
    push_rules(&mut items, analysis, bytes);
    push_vulns(&mut items, analysis);
    push_crypto(&mut items, bytes);
    push_code(&mut items, analysis);
    items
}

/// Appends function and provenance-component evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_code(items: &mut Vec<Item>, analysis: &Analysis) {
    push_functions(items, analysis);
    push_components(items, analysis);
}

/// Appends recovered provenance-component evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_components(items: &mut Vec<Item>, analysis: &Analysis) {
    for component in crate::provenance::recover(analysis).components.iter().take(ITEM_LIMIT) {
        items.push(medium("component", &component.name, &component.version));
    }
}

/// Appends capability evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_capabilities(items: &mut Vec<Item>, analysis: &Analysis) {
    for cap in analysis.capabilities.iter().take(ITEM_LIMIT) {
        items.push(high("capability", cap.category, ""));
    }
}

/// Appends indicator evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_indicators(items: &mut Vec<Item>, analysis: &Analysis) {
    for ind in analysis.indicators.iter().take(ITEM_LIMIT) {
        items.push(high_at("indicator", &ind.value, ind.kind, ind.offset));
    }
}

/// Appends capability-rule evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
fn push_rules(items: &mut Vec<Item>, analysis: &Analysis, bytes: &[u8]) {
    for hit in rules::evaluate(analysis, bytes).iter().take(ITEM_LIMIT) {
        items.push(medium("rule", &hit.name, &hit.namespace));
    }
}

/// Appends vulnerability evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_vulns(items: &mut Vec<Item>, analysis: &Analysis) {
    for vuln in vulns::audit(analysis).iter().take(ITEM_LIMIT) {
        items.push(vuln_item(vuln));
    }
}

/// Builds one vulnerability evidence item.
///
/// # Arguments
///
/// * `vuln` - Vulnerability finding.
///
/// # Returns
///
/// The evidence item.
fn vuln_item(vuln: &vulns::Vulnerability) -> Item {
    let detail = format!("{} {}", vuln.cwe, vuln.severity);
    medium_at("vulnerability", &vuln.title, &detail, vuln.address)
}

/// Appends crypto-constant evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `bytes` - Raw target bytes.
fn push_crypto(items: &mut Vec<Item>, bytes: &[u8]) {
    for hit in crypto::detect(bytes).iter().take(ITEM_LIMIT) {
        items.push(high_at("crypto", &hit.name, &hit.kind, hit.address));
    }
}

/// Appends key-function evidence.
///
/// # Arguments
///
/// * `items` - Accumulated items.
/// * `analysis` - Completed analysis.
fn push_functions(items: &mut Vec<Item>, analysis: &Analysis) {
    for key in analysis.key_functions.iter().take(ITEM_LIMIT) {
        let detail = format!("{} instructions", key.score);
        items.push(high_at("function", &key.name, &detail, key.address));
    }
}

/// Builds a high-confidence item without an address.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
///
/// # Returns
///
/// The evidence item.
fn high(kind: &str, subject: &str, detail: &str) -> Item {
    item(kind, subject, detail, None, Confidence::High)
}

/// Builds a medium-confidence item without an address.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
///
/// # Returns
///
/// The evidence item.
fn medium(kind: &str, subject: &str, detail: &str) -> Item {
    item(kind, subject, detail, None, Confidence::Medium)
}

/// Builds a high-confidence item with an address.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
/// * `address` - Referenced address.
///
/// # Returns
///
/// The evidence item.
fn high_at(kind: &str, subject: &str, detail: &str, address: u64) -> Item {
    item(kind, subject, detail, Some(address), Confidence::High)
}

/// Builds a medium-confidence item with an address.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
/// * `address` - Referenced address.
///
/// # Returns
///
/// The evidence item.
fn medium_at(kind: &str, subject: &str, detail: &str, address: u64) -> Item {
    item(kind, subject, detail, Some(address), Confidence::Medium)
}

/// Builds one content-addressed item.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
/// * `address` - Referenced address, when known.
/// * `confidence` - Confidence level.
///
/// # Returns
///
/// The evidence item.
fn item(
    kind: &str,
    subject: &str,
    detail: &str,
    address: Option<u64>,
    confidence: Confidence,
) -> Item {
    let id = item_id(kind, subject, detail, address);
    let (k, s, d) = (kind.to_string(), subject.to_string(), detail.to_string());
    Item { id, kind: k, subject: s, detail: d, address, confidence }
}

/// Computes a content-addressed identifier.
///
/// # Arguments
///
/// * `kind` - Evidence kind.
/// * `subject` - Short subject.
/// * `detail` - Supporting detail.
/// * `address` - Referenced address, when known.
///
/// # Returns
///
/// The identifier.
#[must_use]
pub fn item_id(kind: &str, subject: &str, detail: &str, address: Option<u64>) -> String {
    let seed = format!("{kind}|{subject}|{detail}|{}", address.unwrap_or(0));
    let digest = hashes::compute(seed.as_bytes()).sha256;
    format!("ev_{}", &digest[..16])
}

/// Builds the target identity.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The target identity.
fn target(analysis: &Analysis) -> Target {
    Target {
        name: analysis.name.clone(),
        sha256: analysis.hashes.sha256.clone(),
        size: analysis.size,
        format: analysis.format.label().to_string(),
    }
}

/// Builds the coverage facts.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The coverage facts.
fn coverage(analysis: &Analysis, bytes: &[u8]) -> Coverage {
    Coverage {
        functions: count_functions(analysis),
        instructions: count_instructions(analysis),
        strings: analysis.strings.len(),
        truncated: truncated(analysis),
        opaque_bytes: opaque_bytes(bytes),
    }
}

/// Builds the explicit unknowns.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The unknown statements.
fn unknowns(analysis: &Analysis, bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    push_truncation(&mut out, analysis);
    push_opaque(&mut out, bytes);
    push_header(&mut out, analysis);
    out
}

/// Records a partial-disassembly unknown.
///
/// # Arguments
///
/// * `out` - Accumulated unknowns.
/// * `analysis` - Completed analysis.
fn push_truncation(out: &mut Vec<String>, analysis: &Analysis) {
    if truncated(analysis) {
        out.push(TRUNCATED.to_string());
    }
}

/// Records an encrypted-region unknown.
///
/// # Arguments
///
/// * `out` - Accumulated unknowns.
/// * `bytes` - Raw target bytes.
fn push_opaque(out: &mut Vec<String>, bytes: &[u8]) {
    let opaque = opaque_bytes(bytes);
    if opaque > 0 {
        out.push(format!(
            "{opaque} bytes are high-entropy (encrypted or compressed) and were not decrypted."
        ));
    }
}

/// Records a missing-header unknown.
///
/// # Arguments
///
/// * `out` - Accumulated unknowns.
/// * `analysis` - Completed analysis.
fn push_header(out: &mut Vec<String>, analysis: &Analysis) {
    if analysis.binary.is_none() {
        out.push(NO_HEADER.to_string());
    }
}

/// Counts recovered functions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The count.
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
/// The count.
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
    regions::segment(bytes)
        .iter()
        .filter(|r| is_opaque(r.kind))
        .map(|r| r.size)
        .sum()
}

/// Reports whether a region kind is opaque.
///
/// # Arguments
///
/// * `kind` - Region kind.
///
/// # Returns
///
/// `true` for encrypted or compressed regions.
fn is_opaque(kind: RegionKind) -> bool {
    matches!(kind, RegionKind::Encrypted | RegionKind::Compressed)
}

/// Returns the producing tool version.
///
/// # Returns
///
/// The version string.
fn tool_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Renders a bundle as pretty JSON.
///
/// # Arguments
///
/// * `bundle` - Evidence bundle.
///
/// # Returns
///
/// The JSON document.
#[must_use]
pub fn to_json(bundle: &Bundle) -> String {
    serde_json::to_string_pretty(&to_value(bundle)).unwrap_or_default()
}

/// Converts a bundle to a JSON value.
///
/// # Arguments
///
/// * `bundle` - Evidence bundle.
///
/// # Returns
///
/// The JSON value.
fn to_value(bundle: &Bundle) -> Value {
    json!({
        "schema_version": bundle.schema_version,
        "tool_version": bundle.tool_version,
        "target": target_value(&bundle.target),
        "coverage": coverage_value(&bundle.coverage),
        "items": items_value(&bundle.items),
        "unknowns": bundle.unknowns,
    })
}

/// Converts the target identity to a JSON value.
///
/// # Arguments
///
/// * `target` - Target identity.
///
/// # Returns
///
/// The JSON value.
fn target_value(target: &Target) -> Value {
    json!({"name": target.name, "sha256": target.sha256, "size": target.size, "format": target.format})
}

/// Converts coverage facts to a JSON value.
///
/// # Arguments
///
/// * `coverage` - Coverage facts.
///
/// # Returns
///
/// The JSON value.
fn coverage_value(coverage: &Coverage) -> Value {
    json!({
        "functions": coverage.functions,
        "instructions": coverage.instructions,
        "strings": coverage.strings,
        "truncated": coverage.truncated,
        "opaque_bytes": coverage.opaque_bytes,
    })
}

/// Converts evidence items to a JSON array.
///
/// # Arguments
///
/// * `items` - Evidence items.
///
/// # Returns
///
/// The JSON array.
fn items_value(items: &[Item]) -> Value {
    Value::Array(items.iter().map(item_value).collect())
}

/// Converts one evidence item to a JSON value.
///
/// # Arguments
///
/// * `item` - Evidence item.
///
/// # Returns
///
/// The JSON value.
fn item_value(item: &Item) -> Value {
    json!({
        "id": item.id,
        "kind": item.kind,
        "subject": item.subject,
        "detail": item.detail,
        "address": item.address,
        "confidence": item.confidence.label(),
    })
}

/// Verifies a bundle's integrity and content-addressed identities.
///
/// # Arguments
///
/// * `text` - Bundle JSON.
///
/// # Returns
///
/// The verification report.
///
/// # Errors
///
/// Returns an error when the JSON is invalid.
pub fn verify(text: &str) -> Result<VerifyReport, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let items = items_of(&value);
    let tampered: Vec<String> = items.iter().filter_map(tampered_id).collect();
    let verified = items.len() - tampered.len();
    Ok(report(&value, items.len(), verified, tampered))
}

/// Extracts the items array from a bundle value.
///
/// # Arguments
///
/// * `value` - Bundle JSON value.
///
/// # Returns
///
/// The item values.
fn items_of(value: &Value) -> Vec<Value> {
    value.get("items").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// Builds a verification report.
///
/// # Arguments
///
/// * `value` - Bundle JSON value.
/// * `items` - Item count.
/// * `verified` - Verified count.
/// * `tampered` - Tampered identifiers.
///
/// # Returns
///
/// The report.
fn report(value: &Value, items: usize, verified: usize, tampered: Vec<String>) -> VerifyReport {
    let version = value.get("schema_version").and_then(Value::as_u64).unwrap_or(0);
    let schema_version = u32::try_from(version).unwrap_or(0);
    VerifyReport { schema_version, items, verified, tampered }
}

/// A bundle-to-bundle differential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff {
    /// Subjects present only in the second bundle.
    pub added: Vec<String>,

    /// Subjects present only in the first bundle.
    pub removed: Vec<String>,

    /// Count of shared items.
    pub shared: usize,
}

/// Compares two evidence bundles by content-addressed identity.
///
/// # Arguments
///
/// * `a_text` - First bundle JSON.
/// * `b_text` - Second bundle JSON.
///
/// # Returns
///
/// The differential, or `None` when either input is not a bundle.
#[must_use]
pub fn diff(a_text: &str, b_text: &str) -> Option<Diff> {
    let (a, b) = (parse_bundle(a_text)?, parse_bundle(b_text)?);
    Some(diff_subjects(&subjects(&a), &subjects(&b)))
}

/// Parses bundle JSON and returns its items, or `None` when not a bundle.
///
/// # Arguments
///
/// * `text` - Bundle JSON.
///
/// # Returns
///
/// The item values, or `None`.
fn parse_bundle(text: &str) -> Option<Vec<Value>> {
    let value: Value = serde_json::from_str(text).ok()?;
    value.get("schema_version")?;
    Some(items_of(&value))
}

/// Maps item identifiers to subjects.
///
/// # Arguments
///
/// * `items` - Item values.
///
/// # Returns
///
/// The identifier-to-subject map.
fn subjects(items: &[Value]) -> BTreeMap<String, String> {
    items.iter().map(subject_pair).collect()
}

/// Maps one item value to an identifier-subject pair.
///
/// # Arguments
///
/// * `item` - Item value.
///
/// # Returns
///
/// The identifier and subject.
fn subject_pair(item: &Value) -> (String, String) {
    let (id, subject) = (str_field(item, "id"), str_field(item, "subject"));
    (id.to_string(), subject.to_string())
}

/// Diffs two identifier maps.
///
/// # Arguments
///
/// * `a` - First map.
/// * `b` - Second map.
///
/// # Returns
///
/// The differential.
fn diff_subjects(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> Diff {
    let added = labels(b, a);
    let removed = labels(a, b);
    let shared = a.len() - removed.len();
    Diff { added, removed, shared }
}

/// Lists subjects present in one map but not another.
///
/// # Arguments
///
/// * `from` - Source map.
/// * `other` - Map to exclude.
///
/// # Returns
///
/// The subjects.
fn labels(from: &BTreeMap<String, String>, other: &BTreeMap<String, String>) -> Vec<String> {
    from.iter()
        .filter(|(id, _)| !other.contains_key(*id))
        .map(|(_, s)| s.clone())
        .collect()
}

/// Renders a bundle differential.
///
/// # Arguments
///
/// * `diff` - The differential.
///
/// # Returns
///
/// The report text.
#[must_use]
pub fn format_diff(diff: &Diff) -> String {
    let mut out = String::from("# Evidence Diff\n");
    kv(&mut out, "shared items", &diff.shared.to_string());
    kv(&mut out, "added", &group(&diff.added));
    kv(&mut out, "removed", &group(&diff.removed));
    out
}

/// Formats a subject group with its count.
///
/// # Arguments
///
/// * `subjects` - Subjects.
///
/// # Returns
///
/// The formatted group.
fn group(subjects: &[String]) -> String {
    format!("({}) {}", subjects.len(), joined(subjects))
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

/// Joins subjects or returns `none`.
///
/// # Arguments
///
/// * `items` - Subjects.
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

/// Returns the identifier of an item whose content does not match its id.
///
/// # Arguments
///
/// * `item` - Evidence item value.
///
/// # Returns
///
/// The identifier when tampered, otherwise `None`.
fn tampered_id(item: &Value) -> Option<String> {
    let expected = item_id(
        str_field(item, "kind"),
        str_field(item, "subject"),
        str_field(item, "detail"),
        item.get("address").and_then(Value::as_u64),
    );
    let actual = str_field(item, "id");
    (expected != actual).then(|| actual.to_string())
}

/// Reads a string field or returns an empty string.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `key` - Field name.
///
/// # Returns
///
/// The field text.
fn str_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies identifiers are stable and content-addressed.
    #[test]
    fn test_item_id_stable() {
        let a = item_id("rule", "crypto", "execution", Some(0x1000));
        let b = item_id("rule", "crypto", "execution", Some(0x1000));
        assert_eq!(a, b);
        assert!(a.starts_with("ev_"));
        assert_ne!(a, item_id("rule", "crypto", "execution", Some(0x2000)));
    }

    /// Verifies a built bundle round-trips and verifies.
    #[test]
    fn test_bundle_verify() {
        let bytes = b"GET http://example.test/ HTTP/1.1 setup";
        let analysis = crate::analyze("blob.bin", bytes);
        let bundle = build(&analysis, bytes);
        assert_eq!(bundle.schema_version, SCHEMA_VERSION);
        let report = verify(&to_json(&bundle)).unwrap();
        assert_eq!(report.items, report.verified);
        assert_eq!(report.tampered.len(), 0);
    }

    /// Verifies tampering is detected.
    #[test]
    fn test_verify_detects_tampering() {
        let bytes = b"plain data sample";
        let analysis = crate::analyze("blob.bin", bytes);
        let mut parsed: Value = serde_json::from_str(&to_json(&build(&analysis, bytes))).unwrap();
        parsed["items"] = json!([{"id": "ev_deadbeef", "kind": "rule", "subject": "x", "detail": "y", "address": null}]);
        let report = verify(&parsed.to_string()).unwrap();
        assert_eq!(report.tampered.len(), 1);
    }
}
