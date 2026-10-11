//! MIT File Header Validation
//!
//! **File:** `header.rs`
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
use std::{ffi::OsStr, path::Path};

/// MIT license text every file header carries, without the `//! ` prefix.
const LICENSE: [&str; 21] = [
    "MIT License",
    "",
    "Copyright (c) 2026 Kevin Thomas",
    "",
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    "of this software and associated documentation files (the \"Software\"), to deal",
    "in the Software without restriction, including without limitation the rights",
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
    "copies of the Software, and to permit persons to whom the Software is",
    "furnished to do so, subject to the following conditions:",
    "",
    "The above copyright notice and this permission notice shall be included in",
    "all copies or substantial portions of the Software.",
    "",
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR",
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,",
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE",
    "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER",
    "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,",
    "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE",
    "SOFTWARE.",
];

/// Checks that a file opens with the exact required header.
///
/// # Arguments
///
/// * `path` - Source path whose file name must appear in the header.
/// * `source` - Full source text.
///
/// # Returns
///
/// One violation when the header is missing or malformed, otherwise none.
pub fn check(path: &Path, source: &str) -> Vec<String> {
    let want = expected(file_name(path));
    let mut lines = source.lines();
    let title_ok = lines.next().is_some_and(is_title);
    let head: Vec<&str> = lines.by_ref().take(want.len()).collect();
    let gap_ok = lines.next() == Some("");
    let ok = title_ok && head == want && gap_ok;
    crate::violation(ok, "missing or malformed MIT file header")
}

/// Extracts the file name of a path as text.
///
/// # Arguments
///
/// * `path` - Path to inspect.
///
/// # Returns
///
/// The final path component, or an empty string when it is not UTF-8.
fn file_name(path: &Path) -> &str {
    let name = path.file_name().and_then(OsStr::to_str);
    name.unwrap_or_default()
}

/// Builds the required header lines that follow the title line.
///
/// # Arguments
///
/// * `name` - File name shown on the `**File:**` line.
///
/// # Returns
///
/// The exact expected lines, each with its `//!` prefix.
fn expected(name: &str) -> Vec<String> {
    let file = format!("**File:** `{name}`");
    let meta = ["", &file, "**Author:** Kevin Thomas", "**Date:** 2026", ""];
    meta.iter().chain(LICENSE.iter()).map(|t| prefixed(t)).collect()
}

/// Prefixes one header line with the inner doc marker.
///
/// # Arguments
///
/// * `text` - Line text without a prefix.
///
/// # Returns
///
/// `//!` for empty text, otherwise `//! ` followed by the text.
fn prefixed(text: &str) -> String {
    if text.is_empty() {
        "//!".to_string()
    } else {
        format!("//! {text}")
    }
}

/// Reports whether a line is a non-empty `//!` title.
///
/// # Arguments
///
/// * `line` - First line of the file.
///
/// # Returns
///
/// `true` when the line is `//! ` followed by visible text.
fn is_title(line: &str) -> bool {
    line.strip_prefix("//! ").is_some_and(|t| !t.trim().is_empty())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a source file carrying a valid header for `name`.
    ///
    /// # Arguments
    ///
    /// * `name` - File name placed on the `**File:**` line.
    ///
    /// # Returns
    ///
    /// Source text with title, header, blank separator, and one item.
    fn sample(name: &str) -> String {
        format!(
            "//! Title\n{}\n\nconst X: u8 = 0;\n",
            expected(name).join("\n")
        )
    }

    /// Verifies that a correct header passes.
    #[test]
    fn test_check_accepts_valid_header() {
        assert!(check(Path::new("a.rs"), &sample("a.rs")).is_empty());
    }

    /// Verifies that a header naming another file fails.
    #[test]
    fn test_check_rejects_wrong_file_name() {
        assert_eq!(check(Path::new("b.rs"), &sample("a.rs")).len(), 1);
    }

    /// Verifies that a file without a header fails.
    #[test]
    fn test_check_rejects_missing_header() {
        assert_eq!(check(Path::new("a.rs"), "fn main() {}\n").len(), 1);
    }

    /// Verifies that the title must contain text.
    #[test]
    fn test_is_title_requires_text() {
        assert!(is_title("//! Engine"));
        assert!(!is_title("//! "));
    }
}
