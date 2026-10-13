//! Agent Tool Execution
//!
//! **File:** `tools.rs`
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
use std::fmt::Write as _;

// Import dependencies from rea_engine
use rea_engine::{decompile, disasm::Function, emulate, Analysis};

/// Instruction budget for the `emulate` tool.
const EMU_BUDGET: u64 = 20_000;

/// Maximum findings or rows returned by one tool.
const LIMIT: usize = 40;

/// Analysis and bytes available to agent tools.
pub struct ToolContext<'a> {
    /// Completed engine analysis.
    pub analysis: &'a Analysis,

    /// Raw target bytes.
    pub bytes: &'a [u8],
}

/// Executes a named tool with a text argument.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `name` - Tool name.
/// * `args` - Tool argument text.
///
/// # Returns
///
/// The tool result text.
#[must_use]
pub fn run(ctx: &ToolContext<'_>, name: &str, args: &str) -> String {
    match name {
        "functions" => list_functions(ctx),
        "decrypt" => recover_strings(ctx),
        "rules" => run_rules(ctx),
        "audit" => run_audit(ctx),
        "crypto" => run_crypto(ctx),
        _ => run_arg(ctx, name, args.trim()),
    }
}

/// Detects cryptographic constants in the target.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The detected constants.
fn run_crypto(ctx: &ToolContext<'_>) -> String {
    let hits = rea_engine::crypto::detect(ctx.bytes);
    let lines: Vec<String> = hits.iter().take(LIMIT).map(crypto_line).collect();
    if lines.is_empty() {
        return "no cryptographic constants found".to_string();
    }
    lines.join("\n")
}

/// Formats one cryptographic constant.
///
/// # Arguments
///
/// * `hit` - Detected constant.
///
/// # Returns
///
/// The formatted line.
fn crypto_line(hit: &rea_engine::crypto::CryptoHit) -> String {
    format!("{:#x} {} ({})", hit.address, hit.name, hit.kind)
}

/// Audits the target for vulnerable API sinks.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The vulnerability findings.
fn run_audit(ctx: &ToolContext<'_>) -> String {
    let findings = rea_engine::vulns::audit(ctx.analysis);
    let lines: Vec<String> = findings.iter().take(LIMIT).map(vuln_line).collect();
    if lines.is_empty() {
        return "no vulnerable sinks found".to_string();
    }
    lines.join("\n")
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

/// Evaluates the capability rule pack against the target.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The matched capabilities.
fn run_rules(ctx: &ToolContext<'_>) -> String {
    let matches = rea_engine::rules::evaluate(ctx.analysis, ctx.bytes);
    let lines: Vec<String> = matches.iter().take(LIMIT).map(rule_line).collect();
    if lines.is_empty() {
        return "no capability rules matched".to_string();
    }
    lines.join("\n")
}

/// Formats one rule match.
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

/// Executes a tool that takes a text argument.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `name` - Tool name.
/// * `arg` - Trimmed argument text.
///
/// # Returns
///
/// The tool result text.
fn run_arg(ctx: &ToolContext<'_>, name: &str, arg: &str) -> String {
    match name {
        "decompile" => decompile_at(ctx, arg),
        "strings" => find_strings(ctx, arg),
        "emulate" => emulate_at(ctx, arg),
        "read" => read_bytes(ctx, arg),
        "unpack" => unpack_at(ctx, arg),
        _ => format!("unknown tool '{name}'"),
    }
}

/// Unpacks the target from an address or across all function entries.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `arg` - Optional hex start address.
///
/// # Returns
///
/// The unpacking summary.
fn unpack_at(ctx: &ToolContext<'_>, arg: &str) -> String {
    let Some(binary) = ctx.analysis.binary.as_ref() else {
        return "no parsed image".to_string();
    };
    let result = match parse_address(arg) {
        Some(address) => rea_engine::unpack::run(binary, ctx.bytes, address, 200_000),
        None => rea_engine::unpack::converge(binary, ctx.bytes, &entries(ctx), 20_000, 4),
    };
    format_unpacked(result.as_ref())
}

/// Collects candidate unpack entry addresses.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The function entry addresses.
fn entries(ctx: &ToolContext<'_>) -> Vec<u64> {
    let code = ctx.analysis.disassembly.as_ref().map(|d| d.functions.keys());
    code.map_or_else(Vec::new, |keys| keys.copied().take(512).collect())
}

/// Formats an unpacking result.
///
/// # Arguments
///
/// * `result` - Unpacking outcome, when supported.
///
/// # Returns
///
/// The summary text.
fn format_unpacked(result: Option<&rea_engine::unpack::Unpacked>) -> String {
    match result {
        Some(run) => unpack_summary(run),
        None => "unsupported architecture".to_string(),
    }
}

/// Formats an unpacking summary.
///
/// # Arguments
///
/// * `run` - Unpacking outcome.
///
/// # Returns
///
/// The summary text.
fn unpack_summary(run: &rea_engine::unpack::Unpacked) -> String {
    let regions = run.regions.len();
    let imports = run.imports.len();
    let base = format!("steps={} regions={regions} stop={:?}", run.steps, run.stop);
    format!("{base} oep={} imports={imports}", oep_label(run))
}

/// Formats an unpacking outcome's OEP label.
///
/// # Arguments
///
/// * `run` - Unpacking outcome.
///
/// # Returns
///
/// The OEP text.
fn oep_label(run: &rea_engine::unpack::Unpacked) -> String {
    run.oep().map_or_else(|| "none".to_string(), |a| format!("0x{a:x}"))
}

/// Recovers strings hidden by simple byte ciphers.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The recovered strings.
fn recover_strings(ctx: &ToolContext<'_>) -> String {
    let lines: Vec<String> =
        rea_engine::recover(ctx.bytes, 0).iter().take(LIMIT).map(decoded_line).collect();
    if lines.is_empty() {
        return "no cipher-hidden strings recovered".to_string();
    }
    lines.join("\n")
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
        "  {:#x} {} key={:#x} {:?}",
        item.address, item.cipher, item.key, item.text
    )
}

/// Lists recovered and key functions.
///
/// # Arguments
///
/// * `ctx` - Tool context.
///
/// # Returns
///
/// The function list text.
fn list_functions(ctx: &ToolContext<'_>) -> String {
    let Some(code) = &ctx.analysis.disassembly else {
        return "no disassembly available".to_string();
    };
    let mut out = functions_header(code.functions.len());
    append_key_functions(&mut out, ctx);
    out
}

/// Builds the function list header.
///
/// # Arguments
///
/// * `count` - Number of recovered functions.
///
/// # Returns
///
/// The header text.
fn functions_header(count: usize) -> String {
    format!("{count} functions recovered; key functions:\n")
}

/// Appends key function entries to output text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `ctx` - Tool context.
fn append_key_functions(out: &mut String, ctx: &ToolContext<'_>) {
    for key in ctx.analysis.key_functions.iter().take(LIMIT) {
        let _ = writeln!(out, "  {:#x} {}", key.address, key.name);
    }
}

/// Decompiles a function by name or address.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `target` - Function name or hex address.
///
/// # Returns
///
/// The pseudo-C listing or an error message.
fn decompile_at(ctx: &ToolContext<'_>, target: &str) -> String {
    let Some(binary) = &ctx.analysis.binary else {
        return "no executable parsed".to_string();
    };
    match find_function(ctx.analysis, target) {
        Some(func) => decompile(func, binary.arch),
        None => format!("no function matching '{target}'"),
    }
}

/// Finds a function by name or address.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `target` - Function name or hex address.
///
/// # Returns
///
/// The matching function, or `None`.
fn find_function<'a>(analysis: &'a Analysis, target: &str) -> Option<&'a Function> {
    let code = analysis.disassembly.as_ref()?;
    let address = parse_address(target);
    code.functions.values().find(|func| matches_function(func, target, address))
}

/// Reports whether a function matches a name or address.
///
/// # Arguments
///
/// * `func` - Function to test.
/// * `target` - Function name.
/// * `address` - Parsed address, when available.
///
/// # Returns
///
/// `true` when the function matches.
fn matches_function(func: &Function, target: &str, address: Option<u64>) -> bool {
    func.name.as_deref() == Some(target) || address == Some(func.address)
}

/// Searches recovered strings for a substring.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `query` - Substring to search.
///
/// # Returns
///
/// Matching strings.
fn find_strings(ctx: &ToolContext<'_>, query: &str) -> String {
    let mut out = String::new();
    for text in ctx.analysis.strings.iter().filter(|s| s.text.contains(query)).take(LIMIT) {
        let _ = writeln!(out, "  {:#x} {:?}", text.offset, text.text);
    }
    if out.is_empty() {
        out.push_str("no matching strings");
    }
    out
}

/// Emulates from an address and reports recovered flags.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `args` - Optional hex entry address.
///
/// # Returns
///
/// The emulation outcome.
fn emulate_at(ctx: &ToolContext<'_>, args: &str) -> String {
    let Some(binary) = &ctx.analysis.binary else {
        return "no executable parsed".to_string();
    };
    let address = parse_address(args).unwrap_or(binary.entry);
    format_emulation(&emulate::run(binary, ctx.bytes, address, EMU_BUDGET))
}

/// Formats an emulation outcome.
///
/// # Arguments
///
/// * `run` - Emulation outcome.
///
/// # Returns
///
/// The formatted outcome.
fn format_emulation(run: &emulate::Emulation) -> String {
    let mut out = format!(
        "steps={} stop={:?} exit={:#x}\n",
        run.steps, run.stop, run.exit
    );
    for finding in &run.found {
        let _ = writeln!(out, "  FLAG {:#x} {:?}", finding.address, finding.text);
    }
    out
}

/// Reads bytes at a virtual address.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `args` - Hex address and optional length.
///
/// # Returns
///
/// The hex and ASCII dump.
fn read_bytes(ctx: &ToolContext<'_>, args: &str) -> String {
    let (addr_text, len_text) = args.split_once(char::is_whitespace).unwrap_or((args, "64"));
    let address = parse_address(addr_text);
    let length = len_text.trim().parse::<usize>().unwrap_or(64);
    match address.and_then(|addr| bytes_at(ctx, addr, length)) {
        Some(slice) => hex_ascii(slice),
        None => "address not mapped".to_string(),
    }
}

/// Slices target bytes for a virtual address range.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `address` - Virtual address.
/// * `length` - Requested byte count.
///
/// # Returns
///
/// The mapped byte slice, or `None`.
fn bytes_at<'a>(ctx: &ToolContext<'a>, address: u64, length: usize) -> Option<&'a [u8]> {
    let binary = ctx.analysis.binary.as_ref()?;
    let section = binary.sections.iter().find(|s| s.contains(address))?;
    let offset = usize::try_from(section.file_offset_of(address)?).ok()?;
    let end = offset.checked_add(length)?.min(ctx.bytes.len());
    ctx.bytes.get(offset..end)
}

/// Formats bytes as hex and ASCII.
///
/// # Arguments
///
/// * `data` - Bytes to format.
///
/// # Returns
///
/// The hex and ASCII dump.
fn hex_ascii(data: &[u8]) -> String {
    let hex: Vec<String> = data.iter().map(|b| format!("{b:02x}")).collect();
    let text: String = data.iter().map(|b| printable(*b)).collect();
    format!("{} |{text}|", hex.join(" "))
}

/// Maps a byte to a printable character.
///
/// # Arguments
///
/// * `byte` - Byte to map.
///
/// # Returns
///
/// The printable character or a dot.
fn printable(byte: u8) -> char {
    if byte.is_ascii_graphic() || byte == b' ' {
        char::from(byte)
    } else {
        '.'
    }
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

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies address parsing with and without a prefix.
    #[test]
    fn test_parse_address() {
        assert_eq!(parse_address("0x140001770"), Some(0x1_4000_1770));
        assert_eq!(parse_address("140001770"), Some(0x1_4000_1770));
        assert_eq!(parse_address("main"), None);
    }

    /// Verifies that the functions tool lists a recovered target.
    #[test]
    fn test_list_functions_on_self() {
        let path = std::env::current_exe().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let analysis = rea_engine::analyze_path(&path).unwrap();
        let ctx = ToolContext { analysis: &analysis, bytes: &bytes };
        assert!(list_functions(&ctx).contains("functions recovered"));
    }
}
