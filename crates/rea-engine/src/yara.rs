//! YARA-Subset Signature Matching
//!
//! **File:** `yara.rs`
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
use std::collections::HashMap;

/// Comparison operator tokens and their meanings.
const CMP_TOKENS: [(&str, Cmp); 6] = [
    (">", Cmp::Gt),
    ("<", Cmp::Lt),
    (">=", Cmp::Ge),
    ("<=", Cmp::Le),
    ("==", Cmp::Eq),
    ("!=", Cmp::Ne),
];

/// A byte pattern with wildcard mask and case option.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Matcher {
    /// Pattern bytes.
    pattern: Vec<u8>,
    /// Per-byte match mask, `true` when the byte must match.
    mask: Vec<bool>,
    /// Whether matching is ASCII case-insensitive.
    nocase: bool,
}

/// A named pattern within a rule.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RuleString {
    /// Pattern identifier, such as `$a`.
    id: String,
    /// Compiled matcher.
    matcher: Matcher,
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Cmp {
    /// Greater than.
    #[default]
    Gt,
    /// Less than.
    Lt,
    /// Greater than or equal.
    Ge,
    /// Less than or equal.
    Le,
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
}

/// A condition expression.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Expr {
    /// Always true.
    True,
    /// Always false.
    False,
    /// String presence by identifier.
    Id(String),
    /// Logical negation.
    Not(Box<Expr>),
    /// Logical conjunction.
    And(Box<Expr>, Box<Expr>),
    /// Logical disjunction.
    Or(Box<Expr>, Box<Expr>),
    /// String occurrence comparison.
    Count(String, Cmp, u64),
    /// File size comparison.
    Filesize(Cmp, u64),
    /// All strings present.
    All,
    /// Any string present.
    Any,
    /// At least a number of strings present.
    AtLeast(usize),
}

/// Default condition expression.
impl Default for Expr {
    /// Returns the always-true expression.
    fn default() -> Self {
        Expr::True
    }
}

/// A parsed rule.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Rule {
    /// Rule name.
    name: String,
    /// Rule tags.
    tags: Vec<String>,
    /// Rule patterns.
    strings: Vec<RuleString>,
    /// Rule condition.
    condition: Expr,
}

/// A matched rule.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Match {
    /// Matched rule name.
    pub name: String,
    /// Matched rule tags.
    pub tags: Vec<String>,
}

/// Scans a buffer with a YARA-subset rule set.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `rules` - Rule source text.
///
/// # Returns
///
/// Every matched rule.
#[must_use]
pub fn scan(bytes: &[u8], rules: &str) -> Vec<Match> {
    parse_rules(rules)
        .into_iter()
        .filter(|rule| evaluate(bytes, rule))
        .map(|rule| Match { name: rule.name, tags: rule.tags })
        .collect()
}

/// Parses every rule in a rule set.
///
/// # Arguments
///
/// * `text` - Rule source text.
///
/// # Returns
///
/// The parsed rules.
fn parse_rules(text: &str) -> Vec<Rule> {
    let mut rules = Vec::new();
    let mut rest = text;
    while let Some((rule, tail)) = next_rule(rest) {
        rules.push(rule);
        rest = tail;
    }
    rules
}

/// Parses the next rule block.
///
/// # Arguments
///
/// * `text` - Remaining source text.
///
/// # Returns
///
/// The rule and the remaining text, or `None`.
fn next_rule(text: &str) -> Option<(Rule, &str)> {
    let at = text.find("rule ")? + 5;
    let after = &text[at..];
    let brace = after.find('{')?;
    let (name, tags) = parse_header(&after[..brace]);
    let open = at + brace;
    let close = matching_brace(text, open)?;
    let body = &text[open + 1..close];
    Some((build_rule(name, tags, body), &text[close + 1..]))
}

/// Finds the brace matching an opening brace.
///
/// # Arguments
///
/// * `text` - Source text.
/// * `open` - Offset of the opening brace.
///
/// # Returns
///
/// The matching close offset, or `None`.
fn matching_brace(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in text.char_indices().skip_while(|(i, _)| *i < open) {
        depth += brace_delta(ch);
        if depth == 0 {
            return Some(i);
        }
    }
    None
}

/// Returns the brace depth change for a character.
///
/// # Arguments
///
/// * `ch` - Character.
///
/// # Returns
///
/// One for `{`, minus one for `}`, otherwise zero.
fn brace_delta(ch: char) -> i32 {
    match ch {
        '{' => 1,
        '}' => -1,
        _ => 0,
    }
}

/// Parses a rule header into a name and tags.
///
/// # Arguments
///
/// * `header` - Header text between `rule` and `{`.
///
/// # Returns
///
/// The name and tags.
fn parse_header(header: &str) -> (String, Vec<String>) {
    let trimmed = header.trim();
    let (name, tags) = trimmed.split_once(':').unwrap_or((trimmed, ""));
    (
        name.trim().to_string(),
        tags.split_whitespace().map(str::to_string).collect(),
    )
}

/// Builds a rule from its parts.
///
/// # Arguments
///
/// * `name` - Rule name.
/// * `tags` - Rule tags.
/// * `body` - Rule body text.
///
/// # Returns
///
/// The parsed rule.
fn build_rule(name: String, tags: Vec<String>, body: &str) -> Rule {
    let (strings_text, condition_text) = split_sections(body);
    Rule {
        name,
        tags,
        strings: parse_strings(strings_text),
        condition: parse_condition(condition_text),
    }
}

/// Splits a rule body into string and condition sections.
///
/// # Arguments
///
/// * `body` - Rule body text.
///
/// # Returns
///
/// The strings text and the condition text.
fn split_sections(body: &str) -> (&str, &str) {
    let Some(at) = body.find("condition:") else {
        return (body, "true");
    };
    let strings = body.find("strings:").map_or("", |s| &body[s + 8..at]);
    (strings, &body[at + 10..])
}

/// Parses the string definitions of a rule.
///
/// # Arguments
///
/// * `text` - Strings section text.
///
/// # Returns
///
/// The parsed patterns.
fn parse_strings(text: &str) -> Vec<RuleString> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some((item, tail)) = next_string_def(rest) {
        out.push(item);
        rest = tail;
    }
    out
}

/// Parses the next `$id = value` definition.
///
/// # Arguments
///
/// * `text` - Remaining strings text.
///
/// # Returns
///
/// The pattern and the remaining text, or `None`.
fn next_string_def(text: &str) -> Option<(RuleString, &str)> {
    let start = text.find('$')?;
    let rest = &text[start..];
    let end = rest[1..].find('$').map_or(rest.len(), |i| i + 1);
    let (id, value) = rest[..end].split_once('=')?;
    let matcher = parse_matcher(value.trim())?;
    let item = RuleString { id: id.trim().to_string(), matcher };
    Some((item, &rest[end..]))
}

/// Parses a pattern value with optional modifiers.
///
/// # Arguments
///
/// * `value` - Value text, such as `"text" nocase`.
///
/// # Returns
///
/// The compiled matcher, or `None`.
fn parse_matcher(value: &str) -> Option<Matcher> {
    if let Some(rest) = value.strip_prefix('{') {
        let end = rest.find('}')?;
        let hex = parse_hex(rest[..end].trim())?;
        return Some(hex);
    }
    let literal = value.strip_prefix('"')?.split_once('"')?.0;
    let modifiers = value.rsplit_once('"')?.1;
    Some(literal_matcher(literal, modifiers))
}

/// Compiles a literal pattern with modifiers.
///
/// # Arguments
///
/// * `literal` - Pattern text.
/// * `modifiers` - Modifier text.
///
/// # Returns
///
/// The compiled matcher.
fn literal_matcher(literal: &str, modifiers: &str) -> Matcher {
    let nocase = modifiers.contains("nocase");
    let bytes = if modifiers.contains("wide") {
        wide_bytes(literal.as_bytes())
    } else {
        literal.as_bytes().to_vec()
    };
    Matcher { mask: vec![true; bytes.len()], pattern: bytes, nocase }
}

/// Encodes bytes as a UTF-16LE wide pattern.
///
/// # Arguments
///
/// * `bytes` - ASCII bytes.
///
/// # Returns
///
/// The wide pattern.
fn wide_bytes(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| [b, 0]).collect()
}

/// Parses a hex pattern with optional `??` wildcards.
///
/// # Arguments
///
/// * `text` - Hex text.
///
/// # Returns
///
/// The compiled matcher, or `None` on a malformed token.
fn parse_hex(text: &str) -> Option<Matcher> {
    let mut pattern = Vec::new();
    let mut mask = Vec::new();
    for token in text.split_whitespace() {
        push_hex_byte(token, &mut pattern, &mut mask)?;
    }
    Some(Matcher { pattern, mask, nocase: false })
}

/// Appends one hex token to a pattern.
///
/// # Arguments
///
/// * `token` - Hex token or wildcard.
/// * `pattern` - Accumulated pattern bytes.
/// * `mask` - Accumulated match mask.
///
/// # Returns
///
/// `None` when the token is malformed.
fn push_hex_byte(token: &str, pattern: &mut Vec<u8>, mask: &mut Vec<bool>) -> Option<()> {
    if token == "??" || token == "?" {
        pattern.push(0);
        mask.push(false);
        return Some(());
    }
    push_literal_byte(token, pattern, mask)
}

/// Appends a literal hex byte to a pattern.
///
/// # Arguments
///
/// * `token` - Hex token.
/// * `pattern` - Accumulated pattern bytes.
/// * `mask` - Accumulated match mask.
///
/// # Returns
///
/// `None` when the token is malformed.
fn push_literal_byte(token: &str, pattern: &mut Vec<u8>, mask: &mut Vec<bool>) -> Option<()> {
    pattern.push(u8::from_str_radix(token, 16).ok()?);
    mask.push(true);
    Some(())
}

/// Parses a condition expression.
///
/// # Arguments
///
/// * `text` - Condition text.
///
/// # Returns
///
/// The parsed expression.
fn parse_condition(text: &str) -> Expr {
    let mut parser = Parser { tokens: tokenize(text), pos: 0 };
    parser.parse_or().unwrap_or(Expr::True)
}

/// Splits condition text into lowercase tokens.
///
/// # Arguments
///
/// * `text` - Condition text.
///
/// # Returns
///
/// The token list.
fn tokenize(text: &str) -> Vec<String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    spaced.split_whitespace().map(str::to_ascii_lowercase).collect()
}

/// A recursive-descent condition parser.
struct Parser {
    /// Token stream.
    tokens: Vec<String>,
    /// Current position.
    pos: usize,
}

impl Parser {
    /// Returns the current token.
    ///
    /// # Returns
    ///
    /// The token, or `None`.
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    /// Consumes and returns the current token.
    ///
    /// # Returns
    ///
    /// The token, or `None`.
    fn next(&mut self) -> Option<String> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        token
    }

    /// Parses a disjunction.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_or(&mut self) -> Option<Expr> {
        let mut left = self.parse_and()?;
        while self.peek() == Some("or") {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    /// Parses a conjunction.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_and(&mut self) -> Option<Expr> {
        let mut left = self.parse_unary()?;
        while self.peek() == Some("and") {
            self.pos += 1;
            let right = self.parse_unary()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    /// Parses a unary expression.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_unary(&mut self) -> Option<Expr> {
        if self.peek() == Some("not") {
            self.pos += 1;
            return Some(Expr::Not(Box::new(self.parse_unary()?)));
        }
        self.parse_primary()
    }

    /// Parses a primary expression.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_primary(&mut self) -> Option<Expr> {
        let token = self.next()?;
        match token.as_str() {
            "(" => self.parse_group(),
            "all" => self.parse_quantifier(Expr::All),
            "any" => self.parse_quantifier(Expr::Any),
            _ => parse_atom(self, &token),
        }
    }

    /// Parses a parenthesized expression.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_group(&mut self) -> Option<Expr> {
        let inner = self.parse_or()?;
        if self.next().as_deref() == Some(")") {
            return Some(inner);
        }
        None
    }

    /// Parses a quantifier such as `all of them`.
    ///
    /// # Arguments
    ///
    /// * `expr` - The quantifier expression.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_quantifier(&mut self, expr: Expr) -> Option<Expr> {
        if self.next().as_deref() == Some("of") && self.next().as_deref() == Some("them") {
            return Some(expr);
        }
        None
    }

    /// Parses a `filesize` comparison.
    ///
    /// # Returns
    ///
    /// The parsed expression, or `None`.
    fn parse_filesize(&mut self) -> Option<Expr> {
        let op = parse_cmp(&self.next()?)?;
        let value = self.next()?.parse().ok()?;
        Some(Expr::Filesize(op, value))
    }
}

/// Parses a string or count term.
///
/// # Arguments
///
/// * `token` - Leading token.
/// * `parser` - Active parser.
///
/// # Returns
///
/// The parsed expression, or `None`.
fn parse_term(token: &str, parser: &mut Parser) -> Option<Expr> {
    if let Some(id) = token.strip_prefix('#') {
        return parse_count(id, parser);
    }
    if token.starts_with('$') {
        return Some(Expr::Id(token.to_string()));
    }
    parse_quantifier_number(token, parser)
}

/// Parses a trailing atom such as `true`, `false`, or `filesize`.
///
/// # Arguments
///
/// * `parser` - Active parser.
/// * `token` - Atom token.
///
/// # Returns
///
/// The parsed expression, or `None`.
fn parse_atom(parser: &mut Parser, token: &str) -> Option<Expr> {
    match token {
        "true" => Some(Expr::True),
        "false" => Some(Expr::False),
        "filesize" => parser.parse_filesize(),
        _ => parse_term(token, parser),
    }
}

/// Parses an occurrence comparison term.
///
/// # Arguments
///
/// * `id` - Identifier without the `#` prefix.
/// * `parser` - Active parser.
///
/// # Returns
///
/// The parsed expression, or `None`.
fn parse_count(id: &str, parser: &mut Parser) -> Option<Expr> {
    let op = parse_cmp(&parser.next()?)?;
    let value = parser.next()?.parse().ok()?;
    Some(Expr::Count(format!("${id}"), op, value))
}

/// Parses a `N of them` quantifier.
///
/// # Arguments
///
/// * `token` - Leading token.
/// * `parser` - Active parser.
///
/// # Returns
///
/// The parsed expression, or `None`.
fn parse_quantifier_number(token: &str, parser: &mut Parser) -> Option<Expr> {
    let count = token.parse::<usize>().ok()?;
    parser.parse_quantifier(Expr::AtLeast(count))
}

/// Parses a comparison operator.
///
/// # Arguments
///
/// * `token` - Operator token.
///
/// # Returns
///
/// The operator, or `None`.
fn parse_cmp(token: &str) -> Option<Cmp> {
    CMP_TOKENS.iter().find(|(text, _)| *text == token).map(|(_, op)| *op)
}

/// Evaluates a rule against a buffer.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `rule` - Parsed rule.
///
/// # Returns
///
/// `true` when the rule matches.
fn evaluate(bytes: &[u8], rule: &Rule) -> bool {
    let counts = counts_for(bytes, &rule.strings);
    let total = rule.strings.len();
    eval(&rule.condition, &counts, total, bytes.len() as u64)
}

/// Counts occurrences of every pattern in a buffer.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `strings` - Rule patterns.
///
/// # Returns
///
/// The occurrence count per identifier.
fn counts_for(bytes: &[u8], strings: &[RuleString]) -> HashMap<String, usize> {
    strings
        .iter()
        .map(|s| (s.id.clone(), count_matches(bytes, &s.matcher)))
        .collect()
}

/// Evaluates a condition expression.
///
/// # Arguments
///
/// * `expr` - Expression.
/// * `counts` - Occurrence counts by identifier.
/// * `total` - Number of strings in the rule.
/// * `size` - Buffer length.
///
/// # Returns
///
/// `true` when the expression holds.
fn eval(expr: &Expr, counts: &HashMap<String, usize>, total: usize, size: u64) -> bool {
    match expr {
        Expr::True => true,
        Expr::False => false,
        Expr::Id(id) => presence(counts, id) > 0,
        Expr::Not(inner) => !eval(inner, counts, total, size),
        _ => eval_quantifier(expr, counts, total, size),
    }
}

/// Evaluates a compound or comparison expression.
///
/// # Arguments
///
/// * `expr` - Expression.
/// * `counts` - Occurrence counts by identifier.
/// * `total` - Number of strings in the rule.
/// * `size` - Buffer length.
///
/// # Returns
///
/// `true` when the expression holds.
fn eval_quantifier(expr: &Expr, counts: &HashMap<String, usize>, total: usize, size: u64) -> bool {
    match expr {
        Expr::And(a, b) => eval(a, counts, total, size) && eval(b, counts, total, size),
        Expr::Or(a, b) => eval(a, counts, total, size) || eval(b, counts, total, size),
        Expr::Count(id, cmp, n) => compare(presence(counts, id) as u64, *cmp, *n),
        Expr::Filesize(cmp, n) => compare(size, *cmp, *n),
        _ => eval_presence(expr, counts, total),
    }
}

/// Evaluates a presence quantifier.
///
/// # Arguments
///
/// * `expr` - Expression.
/// * `counts` - Occurrence counts by identifier.
/// * `total` - Number of strings in the rule.
///
/// # Returns
///
/// `true` when the quantifier holds.
fn eval_presence(expr: &Expr, counts: &HashMap<String, usize>, total: usize) -> bool {
    match expr {
        Expr::All => present(counts) == total,
        Expr::Any => present(counts) > 0,
        Expr::AtLeast(n) => present(counts) >= *n,
        _ => false,
    }
}

/// Returns the occurrence count for an identifier.
///
/// # Arguments
///
/// * `counts` - Occurrence counts.
/// * `id` - Pattern identifier.
///
/// # Returns
///
/// The occurrence count.
fn presence(counts: &HashMap<String, usize>, id: &str) -> usize {
    counts.get(id).copied().unwrap_or(0)
}

/// Counts identifiers with at least one occurrence.
///
/// # Arguments
///
/// * `counts` - Occurrence counts.
///
/// # Returns
///
/// The number of present identifiers.
fn present(counts: &HashMap<String, usize>) -> usize {
    counts.values().filter(|count| **count > 0).count()
}

/// Applies a comparison operator.
///
/// # Arguments
///
/// * `left` - Left operand.
/// * `op` - Comparison operator.
/// * `right` - Right operand.
///
/// # Returns
///
/// The comparison result.
fn compare(left: u64, op: Cmp, right: u64) -> bool {
    match op {
        Cmp::Gt => left > right,
        Cmp::Lt => left < right,
        Cmp::Ge => left >= right,
        Cmp::Le => left <= right,
        Cmp::Eq => left == right,
        Cmp::Ne => left != right,
    }
}

/// Counts non-overlapping pattern matches in a buffer.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `matcher` - Compiled pattern.
///
/// # Returns
///
/// The number of matches.
fn count_matches(bytes: &[u8], matcher: &Matcher) -> usize {
    let len = matcher.pattern.len();
    if len == 0 || bytes.len() < len {
        return 0;
    }
    (0..=bytes.len() - len).filter(|&i| matches_at(bytes, matcher, i)).count()
}

/// Reports whether a pattern matches at an offset.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
/// * `matcher` - Compiled pattern.
/// * `at` - Candidate offset.
///
/// # Returns
///
/// `true` on a match.
fn matches_at(bytes: &[u8], matcher: &Matcher, at: usize) -> bool {
    matcher.mask.iter().enumerate().all(|(j, &required)| {
        if !required {
            return true;
        }
        byte_eq(bytes[at + j], matcher.pattern[j], matcher.nocase)
    })
}

/// Compares two bytes with an optional case fold.
///
/// # Arguments
///
/// * `left` - Source byte.
/// * `right` - Pattern byte.
/// * `nocase` - Whether to fold ASCII case.
///
/// # Returns
///
/// `true` when the bytes are equal.
fn byte_eq(left: u8, right: u8, nocase: bool) -> bool {
    if nocase {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies a literal string match.
    #[test]
    fn test_yara_literal() {
        let rules = "rule pe { strings: $a = \"MZ\" condition: $a }";
        assert_eq!(scan(b"MZ\x90\x00payload", rules).len(), 1);
        assert!(scan(b"no match here", rules).is_empty());
    }

    /// Verifies quantifier and negation conditions.
    #[test]
    fn test_yara_condition() {
        let rules = "rule r { strings: $a = \"one\" $b = \"two\" condition: all of them }";
        assert_eq!(scan(b"one and two", rules).len(), 1);
        assert!(scan(b"only one", rules).is_empty());
    }

    /// Verifies hex patterns with wildcards.
    #[test]
    fn test_yara_hex() {
        let rules = "rule h { strings: $a = { 4d 5a ?? 00 } condition: $a }";
        assert_eq!(scan(b"MZ\x90\x00", rules).len(), 1);
    }

    /// Verifies case-insensitive matching.
    #[test]
    fn test_yara_nocase() {
        let rules = "rule n { strings: $a = \"flags\" nocase condition: $a }";
        assert_eq!(scan(b"FLAGS", rules).len(), 1);
    }

    /// Verifies size and count conditions.
    #[test]
    fn test_yara_counts() {
        let rules = "rule c { strings: $a = \"x\" condition: #a > 2 }";
        assert_eq!(scan(b"x x x", rules).len(), 1);
        assert!(scan(b"x x", rules).is_empty());
    }
}
