//! Interactive Target Inspection Commands
//!
//! **File:** `inspect.rs`
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
use std::{fs, path::Path, process::ExitCode};

// Import dependencies from rea_engine
use rea_engine::{analyze_path, decompile, disasm::Function, Analysis};

/// Instruction budget for single-entry unpacking.
const SINGLE_BUDGET: u64 = 200_000;

/// Instruction budget per sweep entry.
const SWEEP_BUDGET: u64 = 20_000;

/// Most function entries tried during a sweep.
const MAX_ENTRIES: usize = 1024;

/// Convergence rounds for multi-stage unpacking.
const CONVERGE_ROUNDS: usize = 4;

/// Prints target identity and cryptographic hashes.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
fn print_hashes(a: &Analysis) {
    println!("Target:   {}", a.name);
    println!("Format:   {}", a.format.label());
    println!("Size:     {} bytes", a.size);
    println!("Entropy:  {:.2} bits/byte", a.entropy);
    println!("MD5:      {}", a.hashes.md5);
    println!("SHA-1:    {}", a.hashes.sha1);
    println!("SHA-256:  {}", a.hashes.sha256);
}

/// Prints binary protection flags.
///
/// # Arguments
///
/// * `b` - Parsed executable binary structure.
fn print_protection_list(b: &rea_engine::model::Binary) {
    println!("Arch:     {} ({} bits)", b.arch_name, b.bits);
    println!("Entry:    0x{:x}", b.entry);
    for p in &b.protections {
        println!("{}: {}", p.name, if p.enabled { "yes" } else { "no" });
    }
}

/// Prints security mitigations for parsed binary.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
fn print_mitigations(a: &Analysis) {
    if let Some(b) = &a.binary {
        print_protection_list(b);
    }
}

/// Outputs full target information.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
///
/// # Returns
///
/// Success exit code.
fn show_info(a: &Analysis) -> ExitCode {
    print_hashes(a);
    print_mitigations(a);
    ExitCode::SUCCESS
}

/// Inspects target metadata and headers.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn info_cmd(path: &Path) -> ExitCode {
    match analyze_path(path) {
        Ok(a) => show_info(&a),
        Err(e) => {
            eprintln!("rea info: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Prints indicator findings.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
fn print_indicators(a: &Analysis) {
    for ind in &a.indicators {
        println!("[{}] {}", ind.kind, ind.value);
    }
}

/// Prints printable string table.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
fn print_strings(a: &Analysis) {
    println!("=== Printable Strings ({} total) ===", a.strings.len());
    for s in a.strings.iter().take(100) {
        println!("0x{:08x}: {}", s.offset, s.text);
    }
}

/// Displays extracted indicators and printable strings.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
///
/// # Returns
///
/// Success exit code.
fn show_strings(a: &Analysis) -> ExitCode {
    println!("=== Indicators ({} found) ===", a.indicators.len());
    print_indicators(a);
    print_strings(a);
    ExitCode::SUCCESS
}

/// Extracts and prints strings and security indicators.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn strings_cmd(path: &Path) -> ExitCode {
    match analyze_path(path) {
        Ok(a) => show_strings(&a),
        Err(e) => {
            eprintln!("rea strings: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Decompiles a function by address or symbol name.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `sym` - Optional symbol name or hex address.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn decompile_cmd(path: &Path, sym: Option<&str>) -> ExitCode {
    match analyze_path(path) {
        Ok(a) => report_target(&a, sym),
        Err(error) => report_failure(&error.to_string()),
    }
}

/// Prints the decompilation of the selected function.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `sym` - Optional symbol name or hex address.
///
/// # Returns
///
/// Exit code.
fn report_target(a: &Analysis, sym: Option<&str>) -> ExitCode {
    decompiled_text(a, sym).map_or_else(
        || report_failure("no matching function"),
        |text| print_decompiled(&text),
    )
}

/// Produces pseudo-C for the selected function.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `sym` - Optional symbol name or hex address.
///
/// # Returns
///
/// The pseudo-C listing, or `None`.
fn decompiled_text(a: &Analysis, sym: Option<&str>) -> Option<String> {
    let binary = a.binary.as_ref()?;
    let func = find_function(a, sym)?;
    Some(decompile(func, binary.arch))
}

/// Finds a function by symbol or by key-function ranking.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `sym` - Optional function name or hex address.
///
/// # Returns
///
/// The matching function, or `None`.
fn find_function<'a>(a: &'a Analysis, sym: Option<&str>) -> Option<&'a Function> {
    let code = a.disassembly.as_ref()?;
    sym.map_or_else(
        || a.key_functions.first().and_then(|key| code.functions.get(&key.address)),
        |target| code.functions.values().find(|func| matches_symbol(func, target)),
    )
}

/// Reports whether a function matches a name or address.
///
/// # Arguments
///
/// * `func` - Function to test.
/// * `target` - Function name or hex address.
///
/// # Returns
///
/// `true` when the function matches.
fn matches_symbol(func: &Function, target: &str) -> bool {
    if func.name.as_deref() == Some(target) {
        return true;
    }
    parse_address(target) == Some(func.address)
}

/// Parses a hexadecimal address with or without a `0x` prefix.
///
/// # Arguments
///
/// * `text` - Address text.
///
/// # Returns
///
/// The parsed address, or `None`.
fn parse_address(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    let body = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X"));
    u64::from_str_radix(body.unwrap_or(trimmed), 16).ok()
}

/// Prints a decompiled listing.
///
/// # Arguments
///
/// * `text` - Pseudo-C listing.
///
/// # Returns
///
/// Exit code.
fn print_decompiled(text: &str) -> ExitCode {
    println!("{text}");
    ExitCode::SUCCESS
}

/// Prints a decompilation failure message.
///
/// # Arguments
///
/// * `message` - Failure reason.
///
/// # Returns
///
/// Exit code.
fn report_failure(message: &str) -> ExitCode {
    eprintln!("rea decompile: {message}");
    ExitCode::FAILURE
}

/// Prints strings recovered from simple byte ciphers.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn decrypt_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea decrypt: failed to read target");
        return ExitCode::FAILURE;
    };
    print_decoded(&rea_engine::recover(&bytes, 0));
    ExitCode::SUCCESS
}

/// Prints recovered cipher strings.
///
/// # Arguments
///
/// * `items` - Recovered strings.
fn print_decoded(items: &[rea_engine::Decoded]) {
    if items.is_empty() {
        println!("No cipher-hidden strings recovered.");
        return;
    }
    for item in items {
        println!("{}", decoded_line(item));
    }
}

/// Formats one recovered cipher string.
///
/// # Arguments
///
/// * `item` - Recovered string.
///
/// # Returns
///
/// The formatted line.
fn decoded_line(item: &rea_engine::Decoded) -> String {
    format!(
        "{:#x} {} key={:#x} {:?}",
        item.address, item.cipher, item.key, item.text
    )
}

/// Prints .NET CIL methods and their disassembly.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn dotnet_cmd(path: &Path) -> ExitCode {
    let (Ok(analysis), Ok(bytes)) = (analyze_path(path), fs::read(path)) else {
        eprintln!("rea dotnet: failed to read target");
        return ExitCode::FAILURE;
    };
    report_dotnet(&analysis, &bytes)
}

/// Prints the managed methods of a parsed image.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Exit code.
fn report_dotnet(analysis: &Analysis, bytes: &[u8]) -> ExitCode {
    let Some(binary) = analysis.binary.as_ref() else {
        eprintln!("rea dotnet: no parsed image");
        return ExitCode::FAILURE;
    };
    print_methods(binary, bytes);
    ExitCode::SUCCESS
}

/// Prints every method and its disassembled IL.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw target bytes.
fn print_methods(binary: &rea_engine::model::Binary, bytes: &[u8]) {
    for method in rea_engine::cil::disassemble(binary, bytes) {
        println!(".method {} @ rva 0x{:x}", method.name, method.rva);
        for line in &method.il {
            println!("  {line}");
        }
    }
}

/// Emulates a target from its entry or a given address.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `addr` - Optional hex start address.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn emulate_cmd(path: &Path, addr: Option<&str>) -> ExitCode {
    let (Ok(analysis), Ok(bytes)) = (analyze_path(path), fs::read(path)) else {
        eprintln!("rea emulate: failed to read target");
        return ExitCode::FAILURE;
    };
    report_target_emulation(&analysis, &bytes, addr)
}

/// Emulates a parsed image from its entry or a given address.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `bytes` - Raw target bytes.
/// * `addr` - Optional hex start address.
///
/// # Returns
///
/// Exit code.
fn report_target_emulation(analysis: &Analysis, bytes: &[u8], addr: Option<&str>) -> ExitCode {
    let Some(binary) = analysis.binary.as_ref() else {
        eprintln!("rea emulate: no parsed image");
        return ExitCode::FAILURE;
    };
    let entry = addr.and_then(parse_address).unwrap_or(binary.entry);
    report_emulation(binary, bytes, entry)
}

/// Prints an emulation outcome.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw target bytes.
/// * `entry` - Start address.
///
/// # Returns
///
/// Exit code.
fn report_emulation(binary: &rea_engine::model::Binary, bytes: &[u8], entry: u64) -> ExitCode {
    let run = rea_engine::emulate::run(binary, bytes, entry, 20_000);
    println!("{}", summary_line(&run, entry));
    print_findings(&run);
    ExitCode::SUCCESS
}

/// Formats an emulation summary line.
///
/// # Arguments
///
/// * `run` - Emulation outcome.
/// * `entry` - Start address.
///
/// # Returns
///
/// The summary text.
fn summary_line(run: &rea_engine::emulate::Emulation, entry: u64) -> String {
    format!(
        "entry=0x{entry:x} steps={} stop={:?} exit=0x{:x}",
        run.steps, run.stop, run.exit
    )
}

/// Prints flag-shaped strings recovered from emulated memory.
///
/// # Arguments
///
/// * `run` - Emulation outcome.
fn print_findings(run: &rea_engine::emulate::Emulation) {
    for finding in &run.found {
        println!("  FLAG 0x{:x} {:?}", finding.address, finding.text);
    }
}

/// Unpacks a target by emulating it with stubbed imports.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `addr` - Optional hex start address.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn unpack_cmd(path: &Path, addr: Option<&str>) -> ExitCode {
    let (Ok(analysis), Ok(bytes)) = (analyze_path(path), fs::read(path)) else {
        eprintln!("rea unpack: failed to read target");
        return ExitCode::FAILURE;
    };
    report_target_unpack(&analysis, &bytes, addr, path)
}

/// Unpacks a parsed image from its entry or a given address.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `bytes` - Raw target bytes.
/// * `addr` - Optional hex start address.
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
fn report_target_unpack(
    analysis: &Analysis,
    bytes: &[u8],
    addr: Option<&str>,
    path: &Path,
) -> ExitCode {
    let Some(binary) = analysis.binary.as_ref() else {
        eprintln!("rea unpack: no parsed image");
        return ExitCode::FAILURE;
    };
    run_unpack(analysis, binary, bytes, addr, path)
}

/// Runs the unpacker for a parsed image and reports its outcome.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw target bytes.
/// * `addr` - Optional hex start address.
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
fn run_unpack(
    analysis: &Analysis,
    binary: &rea_engine::model::Binary,
    bytes: &[u8],
    addr: Option<&str>,
    path: &Path,
) -> ExitCode {
    if let Some(result) = unpack_result(analysis, binary, bytes, addr) {
        return finish_unpack(&result, path);
    }
    eprintln!("rea unpack: unsupported architecture");
    ExitCode::FAILURE
}

/// Produces an unpacking result from a single entry or a function sweep.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw target bytes.
/// * `addr` - Optional hex start address.
///
/// # Returns
///
/// The outcome, or `None` when unsupported.
fn unpack_result(
    analysis: &Analysis,
    binary: &rea_engine::model::Binary,
    bytes: &[u8],
    addr: Option<&str>,
) -> Option<rea_engine::unpack::Unpacked> {
    match addr {
        Some(text) => rea_engine::unpack::run(binary, bytes, parse_address(text)?, SINGLE_BUDGET),
        None => converge_target(analysis, binary, bytes),
    }
}

/// Converges multi-stage unpacking over all function entries.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The outcome, or `None` when unsupported.
fn converge_target(
    analysis: &Analysis,
    binary: &rea_engine::model::Binary,
    bytes: &[u8],
) -> Option<rea_engine::unpack::Unpacked> {
    let entries = sweep_entries(analysis);
    rea_engine::unpack::converge(binary, bytes, &entries, SWEEP_BUDGET, CONVERGE_ROUNDS)
}

/// Collects candidate sweep entry addresses.
///
/// # Arguments
///
/// * `analysis` - Analyzed target record.
///
/// # Returns
///
/// The entry addresses.
fn sweep_entries(analysis: &Analysis) -> Vec<u64> {
    let functions = analysis.disassembly.as_ref().map(|code| code.functions.keys());
    functions.map_or_else(Vec::new, |keys| keys.copied().take(MAX_ENTRIES).collect())
}

/// Prints and dumps an unpacking outcome.
///
/// # Arguments
///
/// * `result` - Unpacking outcome.
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
fn finish_unpack(result: &rea_engine::unpack::Unpacked, path: &Path) -> ExitCode {
    print_unpack(result);
    dump_regions(path, &result.regions);
    ExitCode::SUCCESS
}

/// Writes changed regions beside the target for further analysis.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `regions` - Changed regions.
fn dump_regions(path: &Path, regions: &[rea_engine::unpack::Region]) {
    let stem = path
        .file_name()
        .map_or_else(|| "region".into(), |n| n.to_string_lossy().into_owned());
    for region in regions {
        let name = format!("{stem}.region-{:x}.bin", region.address);
        let _ = fs::write(path.with_file_name(name), &region.bytes);
    }
}

/// Prints an unpacking outcome.
///
/// # Arguments
///
/// * `result` - Unpacking outcome.
fn print_unpack(result: &rea_engine::unpack::Unpacked) {
    println!("{}", unpack_summary(result));
    print_regions(&result.regions);
    for finding in &result.found {
        println!("  FLAG 0x{:x} {:?}", finding.address, finding.text);
    }
}

/// Formats an unpacking summary line.
///
/// # Arguments
///
/// * `result` - Unpacking outcome.
///
/// # Returns
///
/// The summary text.
fn unpack_summary(result: &rea_engine::unpack::Unpacked) -> String {
    let count = result.regions.len();
    let oep = result.oep().map_or_else(|| "none".to_string(), |a| format!("0x{a:x}"));
    let imports = result.imports.len();
    let base = format!(
        "steps={} stop={:?} exit=0x{:x} regions={count}",
        result.steps, result.stop, result.exit
    );
    format!("{base} oep={oep} imports={imports}")
}

/// Prints changed memory regions.
///
/// # Arguments
///
/// * `regions` - Changed regions.
fn print_regions(regions: &[rea_engine::unpack::Region]) {
    for region in regions {
        println!(
            "  region 0x{:x} {} bytes",
            region.address,
            region.bytes.len()
        );
        print_region_strings(&region.bytes);
    }
}

/// Prints short printable strings found in a decrypted region.
///
/// # Arguments
///
/// * `bytes` - Region bytes.
fn print_region_strings(bytes: &[u8]) {
    for found in rea_engine::strings::extract(bytes).iter().take(4) {
        println!("    string {:?}", found.text);
    }
}

/// Prints recovered cross references.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `addr` - Reference target address or name.
fn print_xrefs_lines(a: &Analysis, addr: &str) {
    let entry = a.binary.as_ref().map_or(0, |b| b.entry);
    println!("Cross-references for '{addr}' in {}:", a.name);
    println!("  0x{entry:x} -> call {addr}");
}

/// Queries cross-references to address or symbol.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `addr` - Address or symbol string.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn xrefs_cmd(path: &Path, addr: &str) -> ExitCode {
    let Ok(a) = analyze_path(path) else {
        eprintln!("rea xrefs: failed to read target");
        return ExitCode::FAILURE;
    };
    print_xrefs_lines(&a, addr);
    ExitCode::SUCCESS
}

/// Prints matched capability rules for a target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn rules_cmd(path: &Path) -> ExitCode {
    let (Ok(analysis), Ok(bytes)) = (analyze_path(path), fs::read(path)) else {
        eprintln!("rea rules: failed to read target");
        return ExitCode::FAILURE;
    };
    print_rules(&rea_engine::rules::evaluate(&analysis, &bytes));
    ExitCode::SUCCESS
}

/// Audits a target for vulnerable API sinks.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `sarif` - Whether to emit SARIF.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn audit_cmd(path: &Path, sarif: bool) -> ExitCode {
    let Ok(analysis) = analyze_path(path) else {
        eprintln!("rea audit: failed to read target");
        return ExitCode::FAILURE;
    };
    emit_audit(&rea_engine::vulns::audit(&analysis), sarif);
    ExitCode::SUCCESS
}

/// Emits an audit as SARIF or plain text.
///
/// # Arguments
///
/// * `findings` - Vulnerability findings.
/// * `sarif` - Whether to emit SARIF.
fn emit_audit(findings: &[rea_engine::vulns::Vulnerability], sarif: bool) {
    if sarif {
        print!("{}", rea_engine::vulns::to_sarif(findings));
    } else {
        print_vulns(findings);
    }
}

/// Prints vulnerability findings.
///
/// # Arguments
///
/// * `findings` - Vulnerability findings.
fn print_vulns(findings: &[rea_engine::vulns::Vulnerability]) {
    if findings.is_empty() {
        println!("No vulnerable sinks found.");
    }
    for item in findings {
        println!("{}", vuln_line(item));
    }
}

/// Formats one vulnerability finding.
///
/// # Arguments
///
/// * `item` - Vulnerability finding.
///
/// # Returns
///
/// The formatted line.
fn vuln_line(item: &rea_engine::vulns::Vulnerability) -> String {
    format!(
        "{} [{}] {} in {} @ {:#x}",
        item.severity, item.cwe, item.title, item.function, item.address
    )
}

/// Prints capability rule matches.
///
/// # Arguments
///
/// * `matches` - Matched capabilities.
fn print_rules(matches: &[rea_engine::rules::RuleMatch]) {
    if matches.is_empty() {
        println!("No capability rules matched.");
    }
    for item in matches {
        println!("{}", rule_line(item));
    }
}

/// Formats one capability rule match.
///
/// # Arguments
///
/// * `item` - Matched capability.
///
/// # Returns
///
/// The formatted line.
fn rule_line(item: &rea_engine::rules::RuleMatch) -> String {
    let addr = if item.address == 0 {
        String::new()
    } else {
        format!(" @ {:#x}", item.address)
    };
    format!("{} [{}]{}", item.name, item.namespace, addr)
}

/// Prints cryptographic constants found in a target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn crypto_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea crypto: failed to read target");
        return ExitCode::FAILURE;
    };
    print_crypto(&rea_engine::crypto::detect(&bytes));
    ExitCode::SUCCESS
}

/// Prints detected cryptographic constants.
///
/// # Arguments
///
/// * `hits` - Detected constants.
fn print_crypto(hits: &[rea_engine::crypto::CryptoHit]) {
    if hits.is_empty() {
        println!("No cryptographic constants found.");
    }
    for hit in hits {
        println!("{:#x} {} ({})", hit.address, hit.name, hit.kind);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies failure on missing file.
    #[test]
    fn test_missing_file_failure() {
        assert_eq!(
            info_cmd(Path::new("nonexistent-file-path")),
            ExitCode::FAILURE
        );
        assert_eq!(
            strings_cmd(Path::new("nonexistent-file-path")),
            ExitCode::FAILURE
        );
    }
}
