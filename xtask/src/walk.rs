//! Workspace Source Discovery
//!
//! **File:** `walk.rs`
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
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Directory names never scanned for sources.
const SKIPPED: [&str; 3] = ["target", ".git", ".cargo"];

/// Lists every Rust source file in the workspace, sorted.
///
/// # Arguments
///
/// * `root` - Workspace root directory.
///
/// # Returns
///
/// Paths of all `.rs` files outside skipped directories.
pub fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    files
}

/// Recursively collects `.rs` files below a directory.
///
/// # Arguments
///
/// * `dir` - Directory to scan.
/// * `files` - Accumulator receiving discovered source paths.
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).into_iter().flatten().flatten();
    for path in entries.map(|entry| entry.path()) {
        if path.is_dir() && !is_skipped(&path) {
            collect(&path, files);
        } else if is_rust(&path) {
            files.push(path);
        }
    }
}

/// Reports whether a directory is excluded from scanning.
///
/// # Arguments
///
/// * `path` - Directory path to test.
///
/// # Returns
///
/// `true` when the directory name is in [`SKIPPED`].
fn is_skipped(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str());
    name.is_some_and(|n| SKIPPED.contains(&n))
}

/// Reports whether a path names a Rust source file.
///
/// # Arguments
///
/// * `path` - File path to test.
///
/// # Returns
///
/// `true` when the extension is `rs`.
fn is_rust(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "rs")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that build output directories are skipped.
    #[test]
    fn test_is_skipped_matches_target() {
        assert!(is_skipped(Path::new("/work/target")));
        assert!(!is_skipped(Path::new("/work/crates")));
    }

    /// Verifies that only `.rs` files are treated as sources.
    #[test]
    fn test_is_rust_checks_extension() {
        assert!(is_rust(Path::new("lib.rs")));
        assert!(!is_rust(Path::new("Cargo.toml")));
    }

    /// Verifies that discovery finds this very file.
    #[test]
    fn test_rust_files_finds_walk_module() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let files = rust_files(dir);
        assert!(files.iter().any(|p| p.ends_with("walk.rs")));
    }
}
