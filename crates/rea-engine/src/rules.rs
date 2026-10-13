//! Capability Rule Engine
//!
//! **File:** `rules.rs`
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

// Import dependencies from serde_json
use serde_json::{Map, Value};

// Import crate::disasm
use crate::disasm::Function;

// Import crate::analysis
use crate::Analysis;

/// Embedded capability rule pack (capa-class, JSON DSL).
const RULES: &str = include_str!("rules_pack.json");

/// Maximum instructions scanned per function.
const MAX_LINES: usize = 4096;

/// The scope a rule is evaluated over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Scope {
    /// The whole file.
    #[default]
    File,

    /// One function.
    Function,
}

/// A single rule feature or combinator.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Feature {
    /// Imported API name substring.
    Api(String),

    /// Extracted string substring.
    Text(String),

    /// Instruction mnemonic.
    Mnemonic(String),

    /// Immediate constant.
    Number(u64),

    /// Raw byte pattern.
    Bytes(Vec<u8>),

    /// All sub-features match.
    All(Vec<Self>),

    /// Any sub-feature matches.
    Any(Vec<Self>),

    /// At least `n` items match a sub-feature.
    Count(usize, Box<Self>),
}

impl Default for Feature {
    /// Builds an empty disjunction.
    ///
    /// # Returns
    ///
    /// An empty `Any` feature.
    fn default() -> Self {
        Self::Any(Vec::new())
    }
}

/// A capability rule.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Rule {
    /// Rule name.
    name: String,

    /// Rule namespace.
    namespace: String,

    /// Evaluation scope.
    scope: Scope,

    /// Combined ATT&CK and MBC technique labels.
    techniques: Vec<String>,

    /// Root feature.
    features: Feature,
}

/// A matched capability with evidence.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuleMatch {
    /// Rule name.
    pub name: String,

    /// Rule namespace.
    pub namespace: String,

    /// Evaluation scope.
    pub scope: &'static str,

    /// Function address, or zero for file scope.
    pub address: u64,

    /// Combined ATT&CK and MBC technique labels.
    pub techniques: Vec<String>,
}

/// Evaluation context built from a target or function.
struct Context<'a> {
    /// Imported API names.
    apis: Vec<String>,

    /// Extracted strings.
    strings: Vec<String>,

    /// Instruction mnemonics.
    mnemonics: Vec<String>,

    /// Immediate constants.
    numbers: Vec<u64>,

    /// Raw target bytes.
    bytes: &'a [u8],
}

/// Evaluates every embedded rule against a target.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The matched capabilities.
#[must_use]
pub fn evaluate(analysis: &Analysis, bytes: &[u8]) -> Vec<RuleMatch> {
    let rules = parse_rules(RULES);
    let file = file_context(analysis, bytes);
    let mut out = Vec::new();
    for rule in &rules {
        collect(&mut out, rule, analysis, &file);
    }
    out
}

/// Appends matches for one rule in its scope.
///
/// # Arguments
///
/// * `out` - Accumulated matches.
/// * `rule` - Rule to evaluate.
/// * `analysis` - Completed analysis.
/// * `file` - File-scope context.
fn collect(out: &mut Vec<RuleMatch>, rule: &Rule, analysis: &Analysis, file: &Context<'_>) {
    match rule.scope {
        Scope::File => push_if(out, rule, file, 0),
        Scope::Function => collect_functions(out, rule, analysis),
    }
}

/// Evaluates a rule over every recovered function.
///
/// # Arguments
///
/// * `out` - Accumulated matches.
/// * `rule` - Rule to evaluate.
/// * `analysis` - Completed analysis.
fn collect_functions(out: &mut Vec<RuleMatch>, rule: &Rule, analysis: &Analysis) {
    let Some(code) = &analysis.disassembly else {
        return;
    };
    for function in code.functions.values() {
        let ctx = function_context(function, analysis);
        push_if(out, rule, &ctx, function.address);
    }
}

/// Records a match when the rule fires.
///
/// # Arguments
///
/// * `out` - Accumulated matches.
/// * `rule` - Rule being evaluated.
/// * `ctx` - Evaluation context.
/// * `address` - Function address, or zero.
fn push_if(out: &mut Vec<RuleMatch>, rule: &Rule, ctx: &Context<'_>, address: u64) {
    if matches(&rule.features, ctx) {
        out.push(rule_match(rule, address));
    }
}

/// Builds a match record from a rule.
///
/// # Arguments
///
/// * `rule` - Fired rule.
/// * `address` - Function address, or zero.
///
/// # Returns
///
/// The match record.
fn rule_match(rule: &Rule, address: u64) -> RuleMatch {
    RuleMatch {
        name: rule.name.clone(),
        namespace: rule.namespace.clone(),
        scope: scope_name(rule.scope),
        address,
        techniques: rule.techniques.clone(),
    }
}

/// Returns the display name of a scope.
///
/// # Arguments
///
/// * `scope` - Evaluation scope.
///
/// # Returns
///
/// The scope name.
fn scope_name(scope: Scope) -> &'static str {
    if scope == Scope::File {
        "file"
    } else {
        "function"
    }
}

/// Builds the whole-file evaluation context.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The context.
fn file_context<'a>(analysis: &'a Analysis, bytes: &'a [u8]) -> Context<'a> {
    let mut ctx = empty_context(bytes);
    fill_common(&mut ctx, analysis);
    if let Some(code) = &analysis.disassembly {
        for function in code.functions.values() {
            extend(&mut ctx, function);
        }
    }
    ctx
}

/// Builds an empty evaluation context over a buffer.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The empty context.
fn empty_context(bytes: &[u8]) -> Context<'_> {
    Context {
        apis: Vec::new(),
        strings: Vec::new(),
        mnemonics: Vec::new(),
        numbers: Vec::new(),
        bytes,
    }
}

/// Fills the API and string facts shared by every context.
///
/// # Arguments
///
/// * `ctx` - Context to fill.
/// * `analysis` - Completed analysis.
fn fill_common(ctx: &mut Context<'_>, analysis: &Analysis) {
    if let Some(binary) = &analysis.binary {
        ctx.apis = binary.imports.iter().map(|i| i.name.clone()).collect();
    }
    ctx.strings = analysis.strings.iter().map(|s| s.text.clone()).collect();
}

/// Builds a function evaluation context.
///
/// # Arguments
///
/// * `function` - Function to evaluate.
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The context.
fn function_context<'a>(function: &Function, analysis: &'a Analysis) -> Context<'a> {
    let mut ctx = empty_context(&[]);
    ctx.apis = function.imports.iter().cloned().collect();
    ctx.strings = analysis.strings.iter().map(|s| s.text.clone()).collect();
    extend(&mut ctx, function);
    ctx
}

/// Adds a function's instruction facts to a context.
///
/// # Arguments
///
/// * `ctx` - Context to extend.
/// * `function` - Function whose instructions are scanned.
fn extend(ctx: &mut Context<'_>, function: &Function) {
    for line in function.lines.iter().take(MAX_LINES) {
        ctx.mnemonics.push(mnemonic_of(&line.text));
        ctx.numbers.extend(numbers_in(&line.text));
    }
}

/// Returns the mnemonic of an instruction line.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// The first whitespace-delimited token.
fn mnemonic_of(text: &str) -> String {
    text.split_whitespace().next().unwrap_or_default().to_string()
}

/// Extracts immediate constants from instruction text.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// The constants found.
fn numbers_in(text: &str) -> Vec<u64> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(parse_number)
        .collect()
}

/// Parses a hexadecimal or decimal token.
///
/// # Arguments
///
/// * `token` - Token text.
///
/// # Returns
///
/// The value, or `None`.
fn parse_number(token: &str) -> Option<u64> {
    match token.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => token.parse::<u64>().ok(),
    }
}

/// Evaluates a feature against a context.
///
/// # Arguments
///
/// * `feature` - Feature to evaluate.
/// * `ctx` - Evaluation context.
///
/// # Returns
///
/// `true` when the feature matches.
fn matches(feature: &Feature, ctx: &Context<'_>) -> bool {
    match feature {
        Feature::All(features) => features.iter().all(|f| matches(f, ctx)),
        Feature::Any(features) => features.iter().any(|f| matches(f, ctx)),
        Feature::Count(threshold, inner) => count(inner, ctx) >= *threshold,
        leaf => matches_leaf(leaf, ctx),
    }
}

/// Evaluates a leaf feature against a context.
///
/// # Arguments
///
/// * `feature` - Leaf feature.
/// * `ctx` - Evaluation context.
///
/// # Returns
///
/// `true` when the leaf matches.
fn matches_leaf(feature: &Feature, ctx: &Context<'_>) -> bool {
    match feature {
        Feature::Api(name) => ctx.apis.iter().any(|a| contains(a, name)),
        Feature::Text(name) => ctx.strings.iter().any(|s| contains(s, name)),
        Feature::Mnemonic(name) => ctx.mnemonics.iter().any(|m| m.eq_ignore_ascii_case(name)),
        Feature::Number(value) => ctx.numbers.contains(value),
        Feature::Bytes(pattern) => find(ctx.bytes, pattern),
        _ => false,
    }
}

/// Counts how many items match a leaf feature.
///
/// # Arguments
///
/// * `feature` - Leaf feature.
/// * `ctx` - Evaluation context.
///
/// # Returns
///
/// The matching item count.
fn count(feature: &Feature, ctx: &Context<'_>) -> usize {
    match feature {
        Feature::Api(name) => matching(&ctx.apis, name),
        Feature::Text(name) => matching(&ctx.strings, name),
        Feature::Mnemonic(name) => mnemonics_matching(&ctx.mnemonics, name),
        Feature::Number(value) => ctx.numbers.iter().filter(|n| *n == value).count(),
        _ => usize::from(matches(feature, ctx)),
    }
}

/// Counts list items containing a needle.
///
/// # Arguments
///
/// * `items` - Text list.
/// * `needle` - Substring to find.
///
/// # Returns
///
/// The matching count.
fn matching(items: &[String], needle: &str) -> usize {
    items.iter().filter(|item| contains(item, needle)).count()
}

/// Counts list items equal to a mnemonic.
///
/// # Arguments
///
/// * `items` - Mnemonic list.
/// * `needle` - Mnemonic to find.
///
/// # Returns
///
/// The matching count.
fn mnemonics_matching(items: &[String], needle: &str) -> usize {
    items.iter().filter(|item| item.eq_ignore_ascii_case(needle)).count()
}

/// Reports whether a haystack contains a needle, case-insensitively.
///
/// # Arguments
///
/// * `haystack` - Text to search.
/// * `needle` - Substring to find.
///
/// # Returns
///
/// `true` when the needle is present.
fn contains(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

/// Reports whether a byte pattern is present.
///
/// # Arguments
///
/// * `bytes` - Buffer to search.
/// * `pattern` - Byte pattern.
///
/// # Returns
///
/// `true` when the pattern is present.
fn find(bytes: &[u8], pattern: &[u8]) -> bool {
    !pattern.is_empty() && bytes.windows(pattern.len()).any(|window| window == pattern)
}

/// Parses the embedded rule pack.
///
/// # Arguments
///
/// * `json` - Rule pack JSON.
///
/// # Returns
///
/// The parsed rules.
fn parse_rules(json: &str) -> Vec<Rule> {
    let Ok(root) = serde_json::from_str::<Value>(json) else {
        return Vec::new();
    };
    let Some(list) = root.get("rules").and_then(Value::as_array) else {
        return Vec::new();
    };
    list.iter().filter_map(parse_rule).collect()
}

/// Parses one rule object.
///
/// # Arguments
///
/// * `value` - Rule JSON object.
///
/// # Returns
///
/// The rule, or `None`.
fn parse_rule(value: &Value) -> Option<Rule> {
    let obj = value.as_object()?;
    build_rule(obj)
}

/// Builds a rule from its JSON object.
///
/// # Arguments
///
/// * `obj` - Rule JSON object.
///
/// # Returns
///
/// The rule, or `None`.
fn build_rule(obj: &Map<String, Value>) -> Option<Rule> {
    let scope = scope_of(obj.get("scope").and_then(Value::as_str));
    Some(Rule {
        name: text_field(obj, "name"),
        namespace: text_field(obj, "namespace"),
        scope,
        techniques: techniques_from(obj),
        features: parse_feature(obj.get("features")?)?,
    })
}

/// Combines a rule's ATT&CK and MBC labels.
///
/// # Arguments
///
/// * `obj` - Rule JSON object.
///
/// # Returns
///
/// The combined technique labels.
fn techniques_from(obj: &Map<String, Value>) -> Vec<String> {
    let mut list = string_list(obj.get("attack"));
    list.extend(string_list(obj.get("mbc")));
    list
}

/// Reads a string field, defaulting to empty.
///
/// # Arguments
///
/// * `obj` - JSON object.
/// * `key` - Field name.
///
/// # Returns
///
/// The field value.
fn text_field(obj: &Map<String, Value>, key: &str) -> String {
    obj.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

/// Reads a string array field.
///
/// # Arguments
///
/// * `value` - JSON value.
///
/// # Returns
///
/// The string list.
fn string_list(value: Option<&Value>) -> Vec<String> {
    value.and_then(Value::as_array).map_or_else(Vec::new, |list| {
        list.iter().filter_map(Value::as_str).map(str::to_string).collect()
    })
}

/// Maps a scope name to a scope.
///
/// # Arguments
///
/// * `name` - Scope name.
///
/// # Returns
///
/// The scope.
fn scope_of(name: Option<&str>) -> Scope {
    if name == Some("function") {
        Scope::Function
    } else {
        Scope::File
    }
}

/// Parses a feature object.
///
/// # Arguments
///
/// * `value` - Feature JSON.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_feature(value: &Value) -> Option<Feature> {
    let obj = value.as_object()?;
    parse_leaf(obj).or_else(|| parse_combinator(obj))
}

/// Parses a leaf feature.
///
/// # Arguments
///
/// * `obj` - Feature JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_leaf(obj: &Map<String, Value>) -> Option<Feature> {
    parse_text_leaf(obj).or_else(|| parse_data_leaf(obj))
}

/// Parses an API, string, or mnemonic feature.
///
/// # Arguments
///
/// * `obj` - Feature JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_text_leaf(obj: &Map<String, Value>) -> Option<Feature> {
    if let Some(feature) = parse_named_leaf(obj) {
        return Some(feature);
    }
    obj.get("mnemonic")
        .and_then(Value::as_str)
        .map(|m| Feature::Mnemonic(m.to_string()))
}

/// Parses an API or string feature.
///
/// # Arguments
///
/// * `obj` - Feature JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_named_leaf(obj: &Map<String, Value>) -> Option<Feature> {
    if let Some(name) = obj.get("api").and_then(Value::as_str) {
        return Some(Feature::Api(name.to_string()));
    }
    obj.get("string").and_then(Value::as_str).map(|s| Feature::Text(s.to_string()))
}

/// Parses a number or byte feature.
///
/// # Arguments
///
/// * `obj` - Feature JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_data_leaf(obj: &Map<String, Value>) -> Option<Feature> {
    if let Some(number) = obj.get("number").and_then(Value::as_u64) {
        return Some(Feature::Number(number));
    }
    obj.get("bytes").and_then(Value::as_str).and_then(parse_hex).map(Feature::Bytes)
}

/// Parses a combinator feature.
///
/// # Arguments
///
/// * `obj` - Feature JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_combinator(obj: &Map<String, Value>) -> Option<Feature> {
    if let Some(list) = obj.get("and").and_then(Value::as_array) {
        return Some(Feature::All(parse_list(list)));
    }
    if let Some(list) = obj.get("or").and_then(Value::as_array) {
        return Some(Feature::Any(parse_list(list)));
    }
    obj.get("count").and_then(Value::as_object).and_then(parse_count)
}

/// Parses a list of features.
///
/// # Arguments
///
/// * `list` - Feature JSON array.
///
/// # Returns
///
/// The parsed features.
fn parse_list(list: &[Value]) -> Vec<Feature> {
    list.iter().filter_map(parse_feature).collect()
}

/// Parses a count feature.
///
/// # Arguments
///
/// * `obj` - Count JSON object.
///
/// # Returns
///
/// The feature, or `None`.
fn parse_count(obj: &Map<String, Value>) -> Option<Feature> {
    let threshold = usize::try_from(obj.get("n").and_then(Value::as_u64)?).ok()?;
    let inner = parse_feature(obj.get("of")?)?;
    Some(Feature::Count(threshold, Box::new(inner)))
}

/// Parses a hex byte pattern string.
///
/// # Arguments
///
/// * `text` - Hex text, such as `4d5a`.
///
/// # Returns
///
/// The bytes, or `None`.
fn parse_hex(text: &str) -> Option<Vec<u8>> {
    let digits: Vec<u8> = text.bytes().filter(u8::is_ascii_hexdigit).collect();
    if digits.len() < 2 || !digits.len().is_multiple_of(2) {
        return None;
    }
    digits.chunks(2).map(|pair| hex_byte(pair[0], pair[1])).collect()
}

/// Combines two hex digits into a byte.
///
/// # Arguments
///
/// * `high` - High nibble character.
/// * `low` - Low nibble character.
///
/// # Returns
///
/// The byte, or `None`.
fn hex_byte(high: u8, low: u8) -> Option<u8> {
    let h = char::from(high).to_digit(16)?;
    let l = char::from(low).to_digit(16)?;
    u8::try_from(h * 16 + l).ok()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the embedded rule pack parses.
    #[test]
    fn test_rules_pack_parses() {
        let rules = parse_rules(RULES);
        assert!(rules.len() >= 30);
        assert!(rules.iter().any(|r| r.namespace.starts_with("communication")));
    }

    /// Verifies API matching against a synthetic context.
    #[test]
    fn test_api_match() {
        let ctx = api_context(&["CreateProcessW"]);
        assert!(matches(&Feature::Api("CreateProcess".into()), &ctx));
        assert!(!matches(&Feature::Api("RegSetValue".into()), &ctx));
    }

    /// Verifies hex byte pattern parsing.
    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_hex("4d5a"), Some(vec![0x4d, 0x5a]));
        assert_eq!(parse_hex("xyz"), None);
    }

    /// Verifies combinator evaluation.
    #[test]
    fn test_combinators() {
        let ctx = api_context(&["VirtualAlloc", "VirtualProtect"]);
        assert!(matches(&both_apis(), &ctx));
        assert!(matches(
            &Feature::Count(2, Box::new(Feature::Api("Virtual".into()))),
            &ctx
        ));
    }

    /// Builds a conjunction of two allocator APIs.
    ///
    /// # Returns
    ///
    /// The combined feature.
    fn both_apis() -> Feature {
        Feature::All(vec![
            Feature::Api("VirtualAlloc".into()),
            Feature::Api("VirtualProtect".into()),
        ])
    }

    /// Builds a context with the given API names.
    ///
    /// # Arguments
    ///
    /// * `apis` - API names.
    ///
    /// # Returns
    ///
    /// The context.
    fn api_context(apis: &[&str]) -> Context<'static> {
        Context {
            apis: apis.iter().map(|a| (*a).to_string()).collect(),
            strings: Vec::new(),
            mnemonics: Vec::new(),
            numbers: Vec::new(),
            bytes: &[],
        }
    }
}
