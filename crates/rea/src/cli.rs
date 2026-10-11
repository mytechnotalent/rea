//! Command-Line Interface
//!
//! **File:** `cli.rs`
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
use std::{fmt, fs, path::Path, process::ExitCode};

// Import dependencies from rea_engine
use rea_engine::{analyze_path, Analysis, EngineError};

// Import dependencies from rea_report
use rea_report::{to_markdown, to_markup, to_pdf, Document, ReportError};

// Import crate items
use crate::{
    compose,
    paths::{report_paths, ReportPaths},
};

/// Help text shown for `--help` and usage errors.
const USAGE: &str = "Reverse Engineering Assistant

usage: rea <file>

Reverse engineers <file> and writes re-<name>.md and re-<name>.pdf beside it.

options:
  -h, --help     show this help
  -V, --version  show the version";

/// Exit status for command-line usage errors.
const USAGE_ERROR: u8 = 2;

/// Failures that stop a report from being produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    /// The target is a directory, which this version does not analyze.
    Directory(String),

    /// The engine could not read or analyze the target.
    Engine(EngineError),

    /// The PDF could not be typeset.
    Report(ReportError),

    /// A report file could not be written.
    Write(String),
}

impl fmt::Display for CliError {
    /// Formats the error for terminal output.
    ///
    /// # Arguments
    ///
    /// * `f` - Destination formatter.
    ///
    /// # Errors
    ///
    /// Returns an error when the formatter fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Directory(path) => write!(f, "{path} is a directory; pass a file"),
            Self::Engine(error) => write!(f, "{error}"),
            Self::Report(error) => write!(f, "{error}"),
            Self::Write(message) => write!(f, "cannot write report: {message}"),
        }
    }
}

/// Runs the command line.
///
/// # Arguments
///
/// * `args` - Arguments after the program name.
///
/// # Returns
///
/// The process exit status.
#[must_use]
pub fn main(args: &[String]) -> ExitCode {
    match args {
        [flag] if flag == "-h" || flag == "--help" => print(USAGE),
        [flag] if flag == "-V" || flag == "--version" => print(&version()),
        [target] => finish(run(Path::new(target))),
        _ => usage_error(),
    }
}

/// Returns the program name and version.
///
/// # Returns
///
/// Text such as `rea 0.1.0`.
#[must_use]
pub fn version() -> String {
    format!("rea {}", env!("CARGO_PKG_VERSION"))
}

/// Prints text and reports success.
///
/// # Arguments
///
/// * `text` - Text to print.
///
/// # Returns
///
/// `ExitCode::SUCCESS`.
fn print(text: &str) -> ExitCode {
    println!("{text}");
    ExitCode::SUCCESS
}

/// Prints usage to standard error and reports a usage error.
///
/// # Returns
///
/// Exit status 2.
fn usage_error() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(USAGE_ERROR)
}

/// Reports the outcome of a run.
///
/// # Arguments
///
/// * `result` - Report paths or the failure.
///
/// # Returns
///
/// Success when both reports were written.
fn finish(result: Result<ReportPaths, CliError>) -> ExitCode {
    match result {
        Ok(paths) => print_paths(&paths),
        Err(error) => fail(&error),
    }
}

/// Formats and prints written report paths.
///
/// # Arguments
///
/// * `p` - Generated report paths.
///
/// # Returns
///
/// Exit code success.
fn print_paths(p: &ReportPaths) -> ExitCode {
    let msg = format!(
        "{}\n{}\n{}\n{}",
        p.markdown.display(),
        p.pdf.display(),
        p.analysis_pdf.display(),
        p.record_pdf.display()
    );
    print(&msg)
}

/// Prints a failure and reports it.
///
/// # Arguments
///
/// * `error` - What went wrong.
///
/// # Returns
///
/// `ExitCode::FAILURE`.
fn fail(error: &CliError) -> ExitCode {
    eprintln!("rea: {error}");
    ExitCode::FAILURE
}

/// Renders a document to bytes for disk writing.
///
/// # Arguments
///
/// * `doc` - Composed report document.
///
/// # Errors
///
/// Returns [`CliError::Report`] when typesetting fails.
fn render_pair(doc: &Document) -> Result<(Vec<u8>, Vec<u8>), CliError> {
    let (md, pdf) = render(doc)?;
    Ok((md.into_bytes(), pdf))
}

/// Writes both markdown and PDF files for one report document.
///
/// # Arguments
///
/// * `md_path` - Markdown destination.
/// * `pdf_path` - PDF destination.
/// * `doc` - Composed report document.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_doc_pair(md_path: &Path, pdf_path: &Path, doc: &Document) -> Result<(), CliError> {
    let (md, pdf) = render_pair(doc)?;
    write(md_path, &md)?;
    write(pdf_path, &pdf)
}

/// Writes the executive threat dossier document.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_dossier_doc(paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    write_doc_pair(
        &paths.analysis_md,
        &paths.analysis_pdf,
        &compose::dossier::build(analysis),
    )
}

/// Writes the static reverse engineering record document.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_record_doc(paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    write_doc_pair(
        &paths.record_md,
        &paths.record_pdf,
        &compose::record::build(analysis),
    )
}

/// Writes threat dossier and static RE record documents.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_aux_docs(paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    write_dossier_doc(paths, analysis)?;
    write_record_doc(paths, analysis)
}

/// Writes all three report documents to disk.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
/// * `assess` - AI narrative assessment.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_all_docs(
    paths: &ReportPaths,
    analysis: &Analysis,
    assess: Option<&str>,
) -> Result<(), CliError> {
    write_doc_pair(
        &paths.markdown,
        &paths.pdf,
        &compose::build(analysis, assess),
    )?;
    write_aux_docs(paths, analysis)
}

/// Executes document generation and writes all accompanying artifacts.
///
/// # Arguments
///
/// * `target` - Target file path.
/// * `paths` - Output paths.
/// * `analysis` - Analysis result.
///
/// # Errors
///
/// Returns [`CliError`] on write failure.
fn emit_reports(target: &Path, paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    write_all_docs(paths, analysis, crate::agent::evaluate(analysis).as_deref())?;
    crate::artifacts::write_all(target, analysis)
}

/// Analyzes a target and writes all reports and artifacts beside it.
///
/// # Arguments
///
/// * `target` - File to reverse engineer.
///
/// # Returns
///
/// Where the reports were written.
///
/// # Errors
///
/// Returns a [`CliError`] when analysis, typesetting, or writing fails.
pub fn run(target: &Path) -> Result<ReportPaths, CliError> {
    reject_directory(target)?;
    eprintln!("rea: reverse engineering {}", target.display());
    let analysis = analyze_path(target).map_err(CliError::Engine)?;
    let paths = report_paths(target);
    emit_reports(target, &paths, &analysis)?;
    Ok(paths)
}

/// Refuses directory targets.
///
/// # Arguments
///
/// * `target` - Requested target.
///
/// # Errors
///
/// Returns [`CliError::Directory`] when the target is a directory.
fn reject_directory(target: &Path) -> Result<(), CliError> {
    let directory = CliError::Directory(target.display().to_string());
    if target.is_dir() {
        Err(directory)
    } else {
        Ok(())
    }
}

/// Renders a document as Markdown and PDF.
///
/// # Arguments
///
/// * `document` - Composed report.
///
/// # Returns
///
/// The Markdown text and PDF bytes.
///
/// # Errors
///
/// Returns [`CliError::Report`] when typesetting fails.
fn render(document: &Document) -> Result<(String, Vec<u8>), CliError> {
    let pdf = to_pdf(&to_markup(document)).map_err(CliError::Report)?;
    Ok((to_markdown(document), pdf))
}

/// Writes one report file.
///
/// # Arguments
///
/// * `path` - Destination.
/// * `bytes` - File contents.
///
/// # Errors
///
/// Returns [`CliError::Write`] when the file cannot be written.
fn write(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let failure = |e: std::io::Error| CliError::Write(format!("{}: {e}", path.display()));
    fs::write(path, bytes).map_err(failure)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the help and version flags.
    #[test]
    fn test_main_flags_succeed() {
        assert_eq!(main(&["--help".into()]), ExitCode::SUCCESS);
        assert_eq!(main(&["-V".into()]), ExitCode::SUCCESS);
    }

    /// Verifies that missing arguments are a usage error.
    #[test]
    fn test_main_requires_target() {
        assert_eq!(main(&[]), ExitCode::from(USAGE_ERROR));
    }

    /// Verifies that directories are refused.
    #[test]
    fn test_run_rejects_directories() {
        assert!(matches!(run(Path::new(".")), Err(CliError::Directory(_))));
    }

    /// Verifies that missing files surface engine errors.
    #[test]
    fn test_run_reports_missing_file() {
        let result = run(Path::new("no/such/file.exe"));
        assert!(matches!(result, Err(CliError::Engine(EngineError::Io(_)))));
    }

    /// Verifies that the version names the program.
    #[test]
    fn test_version_names_program() {
        assert!(version().starts_with("rea "));
    }
}
