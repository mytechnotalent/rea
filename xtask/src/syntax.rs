//! Syntax-Aware Convention Checks
//!
//! **File:** `syntax.rs`
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

// Import dependencies from syn
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
    Attribute, Block, ImplItem, Item, ItemFn, ItemUse, TraitItem, UseTree, Visibility,
};

/// Violation reported when a file with functions lacks a tests module.
const NO_TESTS: &str = "defines functions but has no `#[cfg(test)] mod tests` with `use super::*;`";

/// Checks blank lines, import comments, test docs, and the tests module.
///
/// # Arguments
///
/// * `source` - Full source text.
///
/// # Returns
///
/// Every violation found, or one parse failure.
pub fn check(source: &str) -> Vec<String> {
    let Ok(file) = syn::parse_file(source) else {
        return vec!["file does not parse as Rust".to_string()];
    };
    let mut inspector = Inspector::new(source);
    inspector.visit_file(&file);
    inspector.found.extend(tests_rule(&file.items));
    inspector.found
}

/// Applies the tests-module rule to a file's top-level items.
///
/// # Arguments
///
/// * `items` - Top-level items of the file.
///
/// # Returns
///
/// One violation when functions exist without a tests module.
fn tests_rule(items: &[Item]) -> Vec<String> {
    let ok = !items.iter().any(defines_fn) || items.iter().any(is_tests_module);
    crate::violation(ok, NO_TESTS)
}

/// Reports whether an item defines at least one function body.
///
/// # Arguments
///
/// * `item` - Top-level item to inspect.
///
/// # Returns
///
/// `true` for functions, impls with methods, and traits with default methods.
fn defines_fn(item: &Item) -> bool {
    match item {
        Item::Fn(_) => true,
        Item::Impl(i) => i.items.iter().any(|x| matches!(x, ImplItem::Fn(_))),
        Item::Trait(t) => t.items.iter().any(is_default_fn),
        _ => false,
    }
}

/// Reports whether a trait item is a method with a default body.
///
/// # Arguments
///
/// * `item` - Trait item to inspect.
///
/// # Returns
///
/// `true` when the item is a function with a default body.
fn is_default_fn(item: &TraitItem) -> bool {
    matches!(item, TraitItem::Fn(f) if f.default.is_some())
}

/// Reports whether an item is `#[cfg(test)] mod tests` importing `super::*`.
///
/// # Arguments
///
/// * `item` - Top-level item to inspect.
///
/// # Returns
///
/// `true` when the item is a conforming tests module.
fn is_tests_module(item: &Item) -> bool {
    let Item::Mod(module) = item else {
        return false;
    };
    let body = module.content.as_ref().map(|(_, items)| items.as_slice());
    let imports = body.unwrap_or_default().iter().any(is_super_glob);
    module.ident == "tests" && is_cfg_test(&module.attrs) && imports
}

/// Reports whether an item is `use super::*;`.
///
/// # Arguments
///
/// * `item` - Item inside a tests module.
///
/// # Returns
///
/// `true` for a glob import of the parent module.
fn is_super_glob(item: &Item) -> bool {
    let Item::Use(ItemUse { tree: UseTree::Path(path), .. }) = item else {
        return false;
    };
    path.ident == "super" && matches!(*path.tree, UseTree::Glob(_))
}

/// Reports whether attributes contain `#[cfg(test)]`.
///
/// # Arguments
///
/// * `attrs` - Attributes to search.
///
/// # Returns
///
/// `true` when a `cfg(test)` attribute is present.
fn is_cfg_test(attrs: &[Attribute]) -> bool {
    let is_test = |a: &Attribute| a.parse_args::<syn::Ident>().is_ok_and(|i| i == "test");
    attrs.iter().any(|a| a.path().is_ident("cfg") && is_test(a))
}

/// Reports whether attributes contain one with the given name.
///
/// # Arguments
///
/// * `attrs` - Attributes to search.
/// * `name` - Attribute path identifier, such as `doc` or `test`.
///
/// # Returns
///
/// `true` when a matching attribute is present.
fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|a| a.path().is_ident(name))
}

/// Syntax visitor that records line-level convention violations.
struct Inspector<'a> {
    /// Source lines indexed from zero.
    lines: Vec<&'a str>,

    /// Violations found so far.
    found: Vec<String>,
}

impl<'a> Inspector<'a> {
    /// Creates an inspector over the given source.
    ///
    /// # Arguments
    ///
    /// * `source` - Full source text.
    ///
    /// # Returns
    ///
    /// An inspector with no recorded violations.
    fn new(source: &'a str) -> Self {
        Self { lines: source.lines().collect(), found: Vec::new() }
    }

    /// Records blank lines strictly between two 1-based brace lines.
    ///
    /// # Arguments
    ///
    /// * `open` - Line holding the opening brace.
    /// * `close` - Line holding the closing brace.
    fn check_blank(&mut self, open: usize, close: usize) {
        let inner = self.lines.iter().enumerate().take(close.saturating_sub(1));
        let blank = inner.skip(open).filter(|(_, l)| l.trim().is_empty());
        let lines: Vec<usize> = blank.map(|(i, _)| i + 1).collect();
        for line in lines {
            self.found.push(format!("line {line}: blank line inside a function"));
        }
    }

    /// Records a `use` item that lacks an `// Import` comment directly above.
    ///
    /// # Arguments
    ///
    /// * `line` - 1-based line where the `use` item starts.
    fn check_import(&mut self, line: usize) {
        let above = line.checked_sub(2).and_then(|i| self.lines.get(i));
        if !above.is_some_and(|t| t.trim_start().starts_with("// Import")) {
            let text = format!("line {line}: `use` without an `// Import` comment above");
            self.found.push(text);
        }
    }

    /// Records a `#[test]` function that has no doc comment.
    ///
    /// # Arguments
    ///
    /// * `item` - Function item to inspect.
    fn check_test_doc(&mut self, item: &ItemFn) {
        if has_attr(&item.attrs, "test") && !has_attr(&item.attrs, "doc") {
            let line = item.span().start().line;
            self.found.push(format!("line {line}: test without a doc comment"));
        }
    }
}

impl<'ast> Visit<'ast> for Inspector<'_> {
    /// Checks a block for blank lines; nested blocks are already covered.
    ///
    /// # Arguments
    ///
    /// * `block` - Block to inspect.
    fn visit_block(&mut self, block: &'ast Block) {
        let span = block.brace_token.span;
        self.check_blank(span.open().start().line, span.close().start().line);
    }

    /// Checks that a private `use` item carries its import comment.
    ///
    /// Public re-exports are exempt, matching the crate-root style.
    ///
    /// # Arguments
    ///
    /// * `item` - Use item to inspect.
    fn visit_item_use(&mut self, item: &'ast ItemUse) {
        if matches!(item.vis, Visibility::Inherited) {
            self.check_import(item.span().start().line);
        }
    }

    /// Checks test documentation, then visits the function body.
    ///
    /// # Arguments
    ///
    /// * `item` - Function item to inspect.
    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        self.check_test_doc(item);
        visit::visit_item_fn(self, item);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a blank line inside a function is reported.
    #[test]
    fn test_check_reports_blank_line_in_function() {
        let found = check("fn a() {\n    let x = 1;\n\n    let _ = x;\n}\n");
        assert!(found.iter().any(|v| v.starts_with("line 3: blank")));
    }

    /// Verifies that a `use` without an import comment is reported.
    #[test]
    fn test_check_reports_missing_import_comment() {
        let found = check("use std::fs;\n");
        assert!(found.iter().any(|v| v.contains("`// Import`")));
    }

    /// Verifies that a commented `use` passes.
    #[test]
    fn test_check_accepts_commented_import() {
        assert_eq!(
            check("// Import dependencies from std\nuse std::fs;\n"),
            Vec::<String>::new()
        );
    }

    /// Verifies that functions without a tests module are reported.
    #[test]
    fn test_check_requires_tests_module() {
        assert!(check("fn a() {}\n").contains(&NO_TESTS.to_string()));
    }

    /// Verifies that a conforming tests module satisfies the rule.
    #[test]
    fn test_check_accepts_tests_module() {
        let src = "fn a() {}\n#[cfg(test)]\nmod tests {\n    // Import all\n    use super::*;\n}\n";
        assert_eq!(check(src), Vec::<String>::new());
    }

    /// Verifies that an undocumented test is reported.
    #[test]
    fn test_check_reports_undocumented_test() {
        let found = check("#[test]\nfn t() {}\n");
        assert!(found.iter().any(|v| v.contains("test without a doc")));
    }
}
