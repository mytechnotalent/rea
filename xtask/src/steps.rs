//! Cargo Quality Steps
//!
//! **File:** `steps.rs`
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
use std::{path::Path, process::Command};

/// Cargo invocations that must all succeed, in order.
const STEPS: [&[&str]; 3] = [
    &["fmt", "--all", "--check"],
    &[
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ],
    &["test", "--workspace", "--quiet"],
];

/// Runs every cargo quality step, stopping at the first failure.
///
/// # Arguments
///
/// * `root` - Workspace root used as the working directory.
///
/// # Returns
///
/// `true` when every step succeeded.
pub fn run_all(root: &Path) -> bool {
    STEPS.iter().all(|args| run(root, args))
}

/// Runs one cargo invocation and reports whether it succeeded.
///
/// # Arguments
///
/// * `root` - Working directory for the command.
/// * `args` - Arguments passed to cargo.
///
/// # Returns
///
/// `true` when cargo exited successfully.
fn run(root: &Path, args: &[&str]) -> bool {
    eprintln!("cargo {}", args.join(" "));
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(cargo).args(args).current_dir(root).status();
    status.is_ok_and(|s| s.success())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that clippy treats every warning as an error.
    #[test]
    fn test_clippy_step_denies_warnings() {
        let clippy = STEPS.iter().find(|s| s[0] == "clippy").unwrap();
        assert!(clippy.ends_with(&["-D", "warnings"]));
    }

    /// Verifies that a trivial cargo invocation succeeds.
    #[test]
    fn test_run_reports_success() {
        assert!(run(Path::new(env!("CARGO_MANIFEST_DIR")), &["--version"]));
    }
}
