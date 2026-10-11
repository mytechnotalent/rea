//! Report Output Paths
//!
//! **File:** `paths.rs`
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
use std::path::{Path, PathBuf};

/// Where the two report files for one target are written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportPaths {
    /// Markdown report path.
    pub markdown: PathBuf,

    /// PDF report path.
    pub pdf: PathBuf,
}

/// Computes report paths beside the target: `re-<stem>.md` and `re-<stem>.pdf`.
///
/// # Arguments
///
/// * `target` - Analyzed file.
///
/// # Returns
///
/// The Markdown and PDF paths in the target's directory.
#[must_use]
pub fn report_paths(target: &Path) -> ReportPaths {
    let directory = target.parent().unwrap_or_else(|| Path::new(""));
    let base = format!("re-{}", stem(target));
    let markdown = directory.join(format!("{base}.md"));
    let pdf = directory.join(format!("{base}.pdf"));
    ReportPaths { markdown, pdf }
}

/// Returns the file name without its final extension.
///
/// # Arguments
///
/// * `target` - Analyzed file.
///
/// # Returns
///
/// The stem, or `target` when the path has none.
fn stem(target: &Path) -> String {
    let stem = target.file_stem().map(|s| s.to_string_lossy().into_owned());
    stem.unwrap_or_else(|| "target".to_string())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the extension is replaced and the directory kept.
    #[test]
    fn test_report_paths_beside_target() {
        let paths = report_paths(Path::new("samples/foo.bin"));
        assert_eq!(paths.markdown, Path::new("samples").join("re-foo.md"));
        assert_eq!(paths.pdf, Path::new("samples").join("re-foo.pdf"));
    }

    /// Verifies files without an extension.
    #[test]
    fn test_report_paths_without_extension() {
        let paths = report_paths(Path::new("foo"));
        assert_eq!(paths.markdown, Path::new("re-foo.md"));
    }

    /// Verifies that only the final extension is removed.
    #[test]
    fn test_stem_keeps_inner_dots() {
        assert_eq!(stem(Path::new("archive.tar.gz")), "archive.tar");
    }
}
