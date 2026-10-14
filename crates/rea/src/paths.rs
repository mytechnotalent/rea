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

/// Where the report files and artifacts for one target are written.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReportPaths {
    /// Primary markdown report path.
    pub markdown: PathBuf,
    /// Primary PDF report path.
    pub pdf: PathBuf,
    /// Machine-readable report JSON path.
    pub json: PathBuf,
    /// Full assembler listing path.
    pub asm: PathBuf,
    /// Threat dossier markdown path.
    pub analysis_md: PathBuf,
    /// Threat dossier PDF path.
    pub analysis_pdf: PathBuf,
    /// Static RE record markdown path.
    pub record_md: PathBuf,
    /// Static RE record PDF path.
    pub record_pdf: PathBuf,
    /// Machine-readable findings JSON.
    pub findings: PathBuf,
    /// Directory containing verification `PoCs`.
    pub pocs: PathBuf,
    /// Directory containing Nuclei templates.
    pub nuclei: PathBuf,
    /// Directory containing decompiled sources.
    pub decompiled: PathBuf,
}

/// Constructs core report paths beside target.
///
/// # Arguments
///
/// * `d` - Parent directory.
/// * `stem` - File name stem.
///
/// # Returns
///
/// Initialized report paths.
fn base_paths(d: &Path, stem: &str) -> ReportPaths {
    ReportPaths {
        markdown: d.join(format!("re-{stem}.md")),
        pdf: d.join(format!("re-{stem}.pdf")),
        json: d.join(format!("re-{stem}.json")),
        asm: d.join(format!("re-{stem}.asm")),
        findings: d.join("findings.json"),
        ..Default::default()
    }
}

/// Populates dossier output paths.
///
/// # Arguments
///
/// * `p` - Report paths to populate.
/// * `d` - Parent directory.
/// * `s` - Stem string.
fn fill_dossier_paths(p: &mut ReportPaths, d: &Path, s: &str) {
    p.analysis_md = d.join(format!("re-{s}-analysis.md"));
    p.analysis_pdf = d.join(format!("re-{s}-analysis.pdf"));
    p.pocs = d.join("pocs");
}

/// Populates static RE record and tooling paths.
///
/// # Arguments
///
/// * `p` - Report paths to populate.
/// * `d` - Parent directory.
/// * `s` - Stem string.
fn fill_record_paths(p: &mut ReportPaths, d: &Path, s: &str) {
    p.record_md = d.join(format!("re-{s}-re-record.md"));
    p.record_pdf = d.join(format!("re-{s}-re-record.pdf"));
    p.nuclei = d.join("nuclei");
    p.decompiled = d.join("decompiled");
}

/// Assembles complete report and artifact paths.
///
/// # Arguments
///
/// * `d` - Parent directory.
/// * `stem` - File name stem.
///
/// # Returns
///
/// Populated report paths.
fn make_report_paths(d: &Path, stem: &str) -> ReportPaths {
    let mut paths = base_paths(d, stem);
    fill_dossier_paths(&mut paths, d, stem);
    fill_record_paths(&mut paths, d, stem);
    paths
}

/// Computes report paths beside the target.
///
/// # Arguments
///
/// * `target` - Analyzed file.
///
/// # Returns
///
/// The report and artifact paths in the target's directory.
#[must_use]
pub fn report_paths(target: &Path) -> ReportPaths {
    let directory = target.parent().unwrap_or_else(|| Path::new(""));
    make_report_paths(directory, &stem(target))
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
        let p = report_paths(Path::new("samples/foo.bin"));
        let dir = Path::new("samples");
        assert_eq!(p.markdown, dir.join("re-foo.md"));
        assert_eq!(p.pdf, dir.join("re-foo.pdf"));
        assert_eq!(p.analysis_md, dir.join("re-foo-analysis.md"));
        assert_eq!(p.record_md, dir.join("re-foo-re-record.md"));
    }

    /// Verifies files without an extension.
    #[test]
    fn test_report_paths_without_extension() {
        let paths = report_paths(Path::new("foo"));
        assert_eq!(paths.markdown, Path::new("re-foo.md"));
        assert_eq!(paths.findings, Path::new("findings.json"));
    }

    /// Verifies that only the final extension is removed.
    #[test]
    fn test_stem_keeps_inner_dots() {
        assert_eq!(stem(Path::new("archive.tar.gz")), "archive.tar");
    }
}
