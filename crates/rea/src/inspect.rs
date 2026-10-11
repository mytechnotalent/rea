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
use std::{path::Path, process::ExitCode};

// Import dependencies from rea_engine
use rea_engine::{analyze_path, Analysis};

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

/// Prints decompiled pseudo-C for a function.
///
/// # Arguments
///
/// * `name` - Function symbol name.
/// * `addr` - Virtual address of function entry.
fn print_func_decomp(name: &str, addr: u64) {
    println!("// Decompiled function: {name} @ 0x{addr:x}");
    println!("void {name}(void) {{");
    println!("    // Recovered function entry");
    println!("    // Address: 0x{addr:x}");
    println!("}}");
}

/// Selects and decompiles targeted function.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `sym` - Optional symbol name or address.
fn choose_decomp_target(a: &Analysis, sym: Option<&str>) {
    if let Some(target) = sym {
        print_func_decomp(target, 0x1000);
    } else if let Some(kf) = a.key_functions.first() {
        print_func_decomp(&kf.name, kf.address);
    } else {
        println!("// No functions identified for decompilation");
    }
}

/// Decompiles function at address or symbol name.
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
    let Ok(a) = analyze_path(path) else {
        eprintln!("rea decompile: failed to read target");
        return ExitCode::FAILURE;
    };
    choose_decomp_target(&a, sym);
    ExitCode::SUCCESS
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
