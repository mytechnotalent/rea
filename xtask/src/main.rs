//! Developer Task Runner
//!
//! **File:** `main.rs`
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

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

/// Exact MIT file-header validation.
mod header;

/// External cargo steps run by the `check` task.
mod steps;

/// Syntax-aware convention checks.
mod syntax;

/// Workspace source discovery.
mod walk;

// Import dependencies from std
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

/// Runs the developer task named by the first command-line argument.
///
/// # Returns
///
/// Success when the task passes, failure otherwise.
fn main() -> ExitCode {
    let task = std::env::args().nth(1).unwrap_or_default();
    let root = workspace_root();
    let passed = match task.as_str() {
        "check" => conventions_pass(&root) && steps::run_all(&root),
        "conventions" => conventions_pass(&root),
        _ => usage(),
    };
    exit_code(passed)
}

/// Locates the workspace root relative to this crate.
///
/// # Returns
///
/// The directory containing the workspace `Cargo.toml`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Runs the custom convention checks over every workspace source file.
///
/// # Arguments
///
/// * `root` - Workspace root directory.
///
/// # Returns
///
/// `true` when no violations were found.
fn conventions_pass(root: &Path) -> bool {
    let files = walk::rust_files(root);
    let violations: Vec<String> = files.iter().flat_map(|p| check_file(root, p)).collect();
    for violation in &violations {
        eprintln!("{violation}");
    }
    let total = violations.len();
    eprintln!("conventions: {} files, {total} violations", files.len());
    violations.is_empty()
}

/// Checks one source file against every custom convention.
///
/// # Arguments
///
/// * `root` - Workspace root used to shorten reported paths.
/// * `path` - Source file to check.
///
/// # Returns
///
/// Every violation, each prefixed with the file path.
fn check_file(root: &Path, path: &Path) -> Vec<String> {
    let label = path.strip_prefix(root).unwrap_or(path).display();
    let source = std::fs::read_to_string(path).unwrap_or_default();
    let mut found = header::check(path, &source);
    found.extend(syntax::check(&source));
    found.into_iter().map(|v| format!("{label}: {v}")).collect()
}

/// Turns a pass flag into zero or one violation messages.
///
/// # Arguments
///
/// * `ok` - Whether the rule passed.
/// * `message` - Violation text reported on failure.
///
/// # Returns
///
/// An empty list when `ok`, otherwise a list holding `message`.
fn violation(ok: bool, message: &str) -> Vec<String> {
    (!ok).then(|| message.to_string()).into_iter().collect()
}

/// Prints the supported tasks.
///
/// # Returns
///
/// Always `false`, because no task ran.
fn usage() -> bool {
    eprintln!("usage: cargo xtask <check|conventions>");
    false
}

/// Converts a pass flag into a process exit code.
///
/// # Arguments
///
/// * `passed` - Whether the task succeeded.
///
/// # Returns
///
/// `ExitCode::SUCCESS` or `ExitCode::FAILURE`.
fn exit_code(passed: bool) -> ExitCode {
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the workspace root holds the workspace manifest.
    #[test]
    fn test_workspace_root_has_manifest() {
        assert!(workspace_root().join("Cargo.toml").is_file());
    }

    /// Verifies that the checker's own sources satisfy every convention.
    #[test]
    fn test_xtask_sources_pass_conventions() {
        let root = workspace_root();
        let path = root.join("xtask").join("src").join("main.rs");
        assert!(check_file(&root, &path).is_empty());
    }

    /// Verifies that an unknown task is reported as a failure.
    #[test]
    fn test_usage_reports_failure() {
        assert!(!usage());
    }
}
