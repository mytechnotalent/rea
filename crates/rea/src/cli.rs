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
const USAGE: &str = "Reverse Engineering Assistant (REA)

usage:
  rea <file>                  one-shot nation-state analysis and dossier generation
  rea serve | rea mcp         start Model Context Protocol (MCP) server over stdio
  rea mcp --config            print MCP config for Claude, Cursor, Windsurf, Gemini, OpenCode
  rea mcp --setup             turnkey configure Claude Code and OpenCode MCP
  rea info <file>             inspect file headers, hashes, mitigations, and sections
  rea strings <file>          extract printable strings and security indicators
  rea decompile <file> [sym]  decompile target or function into pseudo-C
  rea xrefs <file> <addr>     query cross-references to address or symbol
  rea analyze <file>          alias for one-shot analysis

options:
  -h, --help                  show this help
  -V, --version               show the version";

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

/// Checks if flag is a help request.
///
/// # Arguments
///
/// * `f` - Argument flag string.
///
/// # Returns
///
/// `true` if `-h` or `--help`.
fn is_help(f: &str) -> bool {
    f == "-h" || f == "--help"
}

/// Checks if flag is a version request.
///
/// # Arguments
///
/// * `f` - Argument flag string.
///
/// # Returns
///
/// `true` if `-V` or `--version`.
fn is_version(f: &str) -> bool {
    f == "-V" || f == "--version"
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
        [flag] if is_help(flag) => print(USAGE),
        [flag] if is_version(flag) => print(&version()),
        [cmd, rest @ ..] => dispatch_subcommand(cmd, rest),
        _ => usage_error(),
    }
}

/// Dispatches interactive inspection commands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "info" => Some(dispatch_info(rest)),
        "strings" => Some(dispatch_strings(rest)),
        "decompile" => Some(dispatch_decompile(rest)),
        "decrypt" => Some(dispatch_decrypt(rest)),
        "xrefs" => Some(dispatch_xrefs(rest)),
        _ => None,
    }
}

/// Dispatches named service or target subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Command execution exit code.
fn dispatch_named(cmd: &str, rest: &[String]) -> ExitCode {
    match cmd {
        "serve" => crate::mcp::serve_cmd(),
        "mcp" => dispatch_mcp(rest),
        "analyze" | "analysis" => dispatch_analyze(rest),
        _ => dispatch_target(cmd, rest),
    }
}

/// Dispatches subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Command execution exit code.
fn dispatch_subcommand(cmd: &str, rest: &[String]) -> ExitCode {
    dispatch_inspect(cmd, rest)
        .or_else(|| dispatch_inspect2(cmd, rest))
        .or_else(|| dispatch_inspect3(cmd, rest))
        .or_else(|| dispatch_inspect4(cmd, rest))
        .or_else(|| dispatch_inspect5(cmd, rest))
        .or_else(|| dispatch_inspect6(cmd, rest))
        .or_else(|| dispatch_inspect7(cmd, rest))
        .unwrap_or_else(|| dispatch_named(cmd, rest))
}

/// Dispatches emulation and format subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect2(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "dotnet" => Some(dispatch_dotnet(rest)),
        "emulate" => Some(dispatch_emulate(rest)),
        "unpack" => Some(dispatch_unpack(rest)),
        "disasm" => Some(dispatch_disasm(rest)),
        "regions" => Some(dispatch_regions(rest)),
        _ => None,
    }
}

/// Dispatches analysis subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect3(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "rules" => Some(dispatch_rules(rest)),
        "audit" => Some(dispatch_audit(rest)),
        "crypto" => Some(dispatch_crypto(rest)),
        _ => None,
    }
}

/// Dispatches artifact subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect4(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "wasm" => Some(dispatch_wasm(rest)),
        "sqlite" => Some(dispatch_sqlite(rest)),
        "coredump" | "core" => Some(dispatch_core(rest)),
        "evm" => Some(dispatch_evm(rest)),
        "plist" | "bundle" => Some(dispatch_plist(rest)),
        _ => None,
    }
}

/// Dispatches network and script subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect5(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "sourcemap" => Some(dispatch_sourcemap(rest)),
        "pcap" | "har" => Some(dispatch_pcap(rest)),
        "flutter" => Some(dispatch_flutter(rest)),
        "dos" => Some(dispatch_dos(rest)),
        "java" | "jar" | "class" => Some(dispatch_java(rest)),
        _ => None,
    }
}

/// Dispatches mobile resource subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect6(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "arsc" => Some(dispatch_arsc(rest)),
        "dwarf" => Some(dispatch_dwarf(rest)),
        "yara" => Some(dispatch_yara(rest)),
        "pdb" => Some(dispatch_pdb(rest)),
        "8051" => Some(dispatch_i8051(rest)),
        _ => None,
    }
}

/// Dispatches external provider subcommands.
///
/// # Arguments
///
/// * `cmd` - Subcommand keyword.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Exit code if matched, otherwise `None`.
fn dispatch_inspect7(cmd: &str, rest: &[String]) -> Option<ExitCode> {
    match cmd {
        "provider" => Some(dispatch_provider(rest)),
        _ => None,
    }
}

/// Dispatches provider subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_provider(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::provider_cmd(Path::new(t)))
}

/// Dispatches mcp server or config subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_mcp(rest: &[String]) -> ExitCode {
    match rest.first().map(String::as_str) {
        Some("--config" | "config") => crate::mcp::config_cmd(),
        Some("--setup" | "setup") => crate::mcp::setup_cmd(),
        _ => crate::mcp::serve_cmd(),
    }
}

/// Dispatches info subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_info(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::info_cmd(Path::new(t)))
}

/// Dispatches strings subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_strings(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::strings_cmd(Path::new(t)))
}

/// Dispatches decompile subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_decompile(rest: &[String]) -> ExitCode {
    match rest {
        [target] => crate::inspect::decompile_cmd(Path::new(target), None),
        [target, sym] => crate::inspect::decompile_cmd(Path::new(target), Some(sym)),
        _ => usage_error(),
    }
}

/// Dispatches xrefs subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_xrefs(rest: &[String]) -> ExitCode {
    match rest {
        [target, addr] => crate::inspect::xrefs_cmd(Path::new(target), addr),
        _ => usage_error(),
    }
}

/// Dispatches decrypt subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_decrypt(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::decrypt_cmd(Path::new(t)))
}

/// Dispatches dotnet subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_dotnet(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::dotnet_cmd(Path::new(t)))
}

/// Dispatches disasm subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_disasm(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::disasm_cmd(Path::new(t)))
}

/// Dispatches regions subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_regions(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::regions_cmd(Path::new(t)))
}

/// Dispatches emulate subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_emulate(rest: &[String]) -> ExitCode {
    match rest {
        [target] => crate::inspect::emulate_cmd(Path::new(target), None),
        [target, addr] => crate::inspect::emulate_cmd(Path::new(target), Some(addr)),
        _ => usage_error(),
    }
}

/// Dispatches unpack subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_unpack(rest: &[String]) -> ExitCode {
    match rest {
        [target] => crate::inspect::unpack_cmd(Path::new(target), None),
        [target, addr] => crate::inspect::unpack_cmd(Path::new(target), Some(addr)),
        _ => usage_error(),
    }
}

/// Dispatches rules subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_rules(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::rules_cmd(Path::new(t)))
}

/// Dispatches audit subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_audit(rest: &[String]) -> ExitCode {
    match rest {
        [target] => crate::inspect::audit_cmd(Path::new(target), false),
        [target, flag] if flag == "--sarif" => crate::inspect::audit_cmd(Path::new(target), true),
        _ => usage_error(),
    }
}

/// Dispatches crypto subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_crypto(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::crypto_cmd(Path::new(t)))
}

/// Dispatches wasm subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_wasm(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::wasm_cmd(Path::new(t)))
}

/// Dispatches sqlite subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_sqlite(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::sqlite_cmd(Path::new(t)))
}

/// Dispatches coredump subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_core(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::core_cmd(Path::new(t)))
}

/// Dispatches evm subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_evm(rest: &[String]) -> ExitCode {
    rest.first().map_or_else(usage_error, |t| crate::inspect::evm_cmd(Path::new(t)))
}

/// Dispatches plist subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_plist(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::plist_cmd(Path::new(t)))
}

/// Dispatches sourcemap subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_sourcemap(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::sourcemap_cmd(Path::new(t)))
}

/// Dispatches pcap subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_pcap(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::pcap_cmd(Path::new(t)))
}

/// Dispatches flutter subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_flutter(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::flutter_cmd(Path::new(t)))
}

/// Dispatches dos subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_dos(rest: &[String]) -> ExitCode {
    rest.first().map_or_else(usage_error, |t| crate::inspect::dos_cmd(Path::new(t)))
}

/// Dispatches java subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_java(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::java_cmd(Path::new(t)))
}

/// Dispatches arsc subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_arsc(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::arsc_cmd(Path::new(t)))
}

/// Dispatches dwarf subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_dwarf(rest: &[String]) -> ExitCode {
    rest.first()
        .map_or_else(usage_error, |t| crate::inspect::dwarf_cmd(Path::new(t)))
}

/// Dispatches yara subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_yara(rest: &[String]) -> ExitCode {
    match rest {
        [target, rules] => crate::inspect::yara_cmd(Path::new(target), Path::new(rules)),
        _ => usage_error(),
    }
}

/// Dispatches pdb subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_pdb(rest: &[String]) -> ExitCode {
    rest.first().map_or_else(usage_error, |t| crate::inspect::pdb_cmd(Path::new(t)))
}

/// Dispatches 8051 subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_i8051(rest: &[String]) -> ExitCode {
    match rest {
        [target] => crate::inspect::i8051_cmd(Path::new(target), None, None),
        [target, offset] => crate::inspect::i8051_cmd(Path::new(target), Some(offset), None),
        [target, offset, base] => {
            crate::inspect::i8051_cmd(Path::new(target), Some(offset), Some(base))
        }
        _ => usage_error(),
    }
}

/// Dispatches analyze subcommand.
///
/// # Arguments
///
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_analyze(rest: &[String]) -> ExitCode {
    rest.first().map_or_else(usage_error, |t| finish(run(Path::new(t))))
}

/// Dispatches direct target file invocation.
///
/// # Arguments
///
/// * `target` - Target file path string.
/// * `rest` - Trailing command-line arguments.
///
/// # Returns
///
/// Process exit code.
fn dispatch_target(target: &str, rest: &[String]) -> ExitCode {
    if rest.is_empty() {
        finish(run(Path::new(target)))
    } else {
        usage_error()
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
    let mut out = vec![
        p.markdown.display().to_string(),
        p.pdf.display().to_string(),
    ];
    append_existing_pdf(&mut out, &p.analysis_pdf);
    append_existing_pdf(&mut out, &p.record_pdf);
    print(&out.join("\n"))
}

/// Appends a PDF path if it exists on disk.
///
/// # Arguments
///
/// * `out` - Destination list.
/// * `p` - Path to check.
fn append_existing_pdf(out: &mut Vec<String>, p: &std::path::Path) {
    if p.exists() {
        out.push(p.display().to_string());
    }
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
/// # Returns
///
/// Tuple of markdown bytes and compiled PDF bytes.
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
/// # Returns
///
/// Unit when successful.
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
/// # Returns
///
/// Unit when successful.
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
/// # Returns
///
/// Unit when successful.
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
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_aux_docs(paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    write_dossier_doc(paths, analysis)?;
    write_record_doc(paths, analysis)
}

/// Writes primary report document pair to disk.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
/// * `assess` - AI narrative assessment.
///
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError`] on failure.
fn write_primary_doc(
    paths: &ReportPaths,
    analysis: &Analysis,
    bytes: &[u8],
    assess: Option<&str>,
) -> Result<(), CliError> {
    let doc = compose::build(analysis, bytes, assess);
    write_doc_pair(&paths.markdown, &paths.pdf, &doc)
}

/// Executes document generation and writes all accompanying artifacts.
///
/// # Arguments
///
/// * `target` - Target file path.
/// * `paths` - Output paths.
/// * `analysis` - Analysis result.
///
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError`] on write failure.
fn emit_reports(target: &Path, paths: &ReportPaths, analysis: &Analysis) -> Result<(), CliError> {
    let bytes = fs::read(target).unwrap_or_default();
    let assess = crate::agent::evaluate(analysis, &bytes);
    write_primary_doc(paths, analysis, &bytes, assess.as_deref())?;
    if analysis.firmware.is_some() {
        write_aux_docs(paths, analysis)?;
        crate::artifacts::write_all(target, analysis)?;
    }
    Ok(())
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
    let bytes = fs::read(target).unwrap_or_default();
    let paths = report_paths(target);
    write_artifacts(&paths, &analysis, &bytes)?;
    emit_reports(target, &paths, &analysis)?;
    Ok(paths)
}

/// Writes the machine-readable JSON and assembler artifacts.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError::Write`] when a file cannot be written.
fn write_artifacts(paths: &ReportPaths, analysis: &Analysis, bytes: &[u8]) -> Result<(), CliError> {
    write_json(paths, analysis, bytes)?;
    write_asm(paths, analysis, bytes)
}

/// Writes the machine-readable JSON report.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError::Write`] when the file cannot be written.
fn write_json(paths: &ReportPaths, analysis: &Analysis, bytes: &[u8]) -> Result<(), CliError> {
    write(&paths.json, crate::json::report(analysis, bytes).as_bytes())
}

/// Most instructions written to the assembler listing.
const MAX_ASM: usize = 4_000_000;

/// Writes the full assembler listing beside the report.
///
/// # Arguments
///
/// * `paths` - Output paths.
/// * `analysis` - Analysis results.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Unit when successful.
///
/// # Errors
///
/// Returns [`CliError::Write`] when the file cannot be written.
fn write_asm(paths: &ReportPaths, analysis: &Analysis, bytes: &[u8]) -> Result<(), CliError> {
    let Some(binary) = analysis.binary.as_ref() else {
        return Ok(());
    };
    let lines = rea_engine::listing::full(binary, bytes, MAX_ASM);
    write(&paths.asm, lines.join("\n").as_bytes())
}

/// Refuses directory targets.
///
/// # Arguments
///
/// * `target` - Requested target.
///
/// # Returns
///
/// Unit when target is not a directory.
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
/// # Returns
///
/// Unit when successful.
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

    /// Verifies that mcp config flag returns success.
    #[test]
    fn test_mcp_config_flag() {
        assert_eq!(main(&["mcp".into(), "--config".into()]), ExitCode::SUCCESS);
    }
}
