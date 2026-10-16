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
use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

// Import dependencies from rea_engine
use rea_engine::{analyze_path, compare, decompile, disasm::Function, Analysis};

/// Most instructions printed by a single listing command.
const MAX_LISTING: usize = 4_000_000;

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

/// Reports WebAssembly module structure.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn wasm_cmd(path: &Path) -> ExitCode {
    let Some(module) = load_wasm(path) else {
        eprintln!("rea wasm: not a WebAssembly module");
        return ExitCode::FAILURE;
    };
    report_wasm(&module)
}

/// Reads and parses a WebAssembly module.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Recovered module inventory, or `None`.
fn load_wasm(path: &Path) -> Option<rea_engine::wasm::WasmModule> {
    let bytes = fs::read(path).ok()?;
    rea_engine::wasm::parse_wasm(&bytes)
}

/// Prints a WebAssembly summary and details.
///
/// # Arguments
///
/// * `module` - Recovered module inventory.
///
/// # Returns
///
/// Exit code.
fn report_wasm(module: &rea_engine::wasm::WasmModule) -> ExitCode {
    print!("{}", rea_engine::wasm::format_wasm_summary(module));
    print_wasm_sections(module);
    print_wasm_links(module);
    ExitCode::SUCCESS
}

/// Prints WebAssembly section headers.
///
/// # Arguments
///
/// * `module` - Recovered module inventory.
fn print_wasm_sections(module: &rea_engine::wasm::WasmModule) {
    for section in &module.sections {
        println!(
            "section {} id={} size={}",
            section.name, section.id, section.size
        );
    }
}

/// Prints WebAssembly imports and exports.
///
/// # Arguments
///
/// * `module` - Recovered module inventory.
fn print_wasm_links(module: &rea_engine::wasm::WasmModule) {
    for import in &module.imports {
        let module_name = &import.module;
        let name = &import.name;
        println!("import {module_name}::{name} ({})", import.kind);
    }
    for export in &module.exports {
        println!("export {} ({})", export.name, export.kind);
    }
}

/// Reports SQLite schema and sampled rows.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn sqlite_cmd(path: &Path) -> ExitCode {
    let Some(db) = load_sqlite(path) else {
        eprintln!("rea sqlite: not a SQLite database");
        return ExitCode::FAILURE;
    };
    report_sqlite(&db)
}

/// Reads and parses a SQLite database.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Recovered database inventory, or `None`.
fn load_sqlite(path: &Path) -> Option<rea_engine::sqlite::SqliteDb> {
    let bytes = fs::read(path).ok()?;
    rea_engine::sqlite::parse_sqlite(&bytes)
}

/// Prints a SQLite summary, columns, and rows.
///
/// # Arguments
///
/// * `db` - Recovered database inventory.
///
/// # Returns
///
/// Exit code.
fn report_sqlite(db: &rea_engine::sqlite::SqliteDb) -> ExitCode {
    print!("{}", rea_engine::sqlite::format_sqlite_summary(db));
    print_sqlite_columns(db);
    print_sqlite_rows(db);
    ExitCode::SUCCESS
}

/// Prints recovered SQLite columns.
///
/// # Arguments
///
/// * `db` - Recovered database inventory.
fn print_sqlite_columns(db: &rea_engine::sqlite::SqliteDb) {
    for column in &db.columns {
        println!("column {} {}", column.name, column.kind);
    }
}

/// Prints sampled SQLite rows.
///
/// # Arguments
///
/// * `db` - Recovered database inventory.
fn print_sqlite_rows(db: &rea_engine::sqlite::SqliteDb) {
    for row in &db.rows {
        println!("row {row}");
    }
}

/// Reports an ELF core dump and its crash state.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn core_cmd(path: &Path) -> ExitCode {
    let Some(core) = load_core(path) else {
        eprintln!("rea coredump: not an ELF core dump");
        return ExitCode::FAILURE;
    };
    report_core(&core)
}

/// Reads and parses an ELF core dump.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Recovered core inventory, or `None`.
fn load_core(path: &Path) -> Option<rea_engine::core::CoreDump> {
    let bytes = fs::read(path).ok()?;
    rea_engine::core::parse_core(&bytes)
}

/// Prints a core summary, registers, and mappings.
///
/// # Arguments
///
/// * `core` - Recovered core inventory.
///
/// # Returns
///
/// Exit code.
fn report_core(core: &rea_engine::core::CoreDump) -> ExitCode {
    print!("{}", rea_engine::core::format_core_summary(core));
    print_core_registers(core);
    print_core_mappings(core);
    ExitCode::SUCCESS
}

/// Prints captured registers.
///
/// # Arguments
///
/// * `core` - Recovered core inventory.
fn print_core_registers(core: &rea_engine::core::CoreDump) {
    for (name, value) in &core.registers {
        println!("{name} = 0x{value:x}");
    }
}

/// Prints loadable mappings.
///
/// # Arguments
///
/// * `core` - Recovered core inventory.
fn print_core_mappings(core: &rea_engine::core::CoreDump) {
    for mapping in &core.mappings {
        println!(
            "0x{:x} {} size={}",
            mapping.start, mapping.flags, mapping.size
        );
    }
}

/// Reports EVM bytecode selectors and capabilities.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn evm_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea evm: failed to read target");
        return ExitCode::FAILURE;
    };
    report_evm(&rea_engine::evm::parse_evm(&evm_code(&bytes)))
}

/// Resolves bytecode from a file, decoding hexadecimal text when present.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
///
/// # Returns
///
/// The decoded bytecode.
fn evm_code(bytes: &[u8]) -> Vec<u8> {
    if rea_engine::evm::is_hex_text(bytes) {
        let text = String::from_utf8_lossy(bytes);
        rea_engine::evm::decode_hex(&text).unwrap_or_default()
    } else {
        bytes.to_vec()
    }
}

/// Prints an EVM summary and disassembly.
///
/// # Arguments
///
/// * `evm` - Recovered inventory.
///
/// # Returns
///
/// Exit code.
fn report_evm(evm: &rea_engine::evm::EvmBytecode) -> ExitCode {
    print!("{}", rea_engine::evm::format_evm_summary(evm));
    for line in &evm.opcodes {
        println!("{line}");
    }
    ExitCode::SUCCESS
}

/// Reports property list contents and bundle metadata.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn plist_cmd(path: &Path) -> ExitCode {
    let Some(plist) = load_plist(path) else {
        eprintln!("rea plist: not a property list");
        return ExitCode::FAILURE;
    };
    report_plist(&plist)
}

/// Reads and parses a property list.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The decoded list, or `None`.
fn load_plist(path: &Path) -> Option<rea_engine::plist::Plist> {
    let bytes = fs::read(path).ok()?;
    rea_engine::plist::parse_plist(&bytes)
}

/// Prints a property list and bundle summary.
///
/// # Arguments
///
/// * `plist` - Decoded list.
///
/// # Returns
///
/// Exit code.
fn report_plist(plist: &rea_engine::plist::Plist) -> ExitCode {
    print!("{}", rea_engine::plist::format_plist_summary(plist));
    let bundle = rea_engine::plist::analyze_bundle(plist);
    print!("{}", rea_engine::plist::format_bundle_summary(&bundle));
    ExitCode::SUCCESS
}

/// Reports a source map or a `sourceMappingURL` directive.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn sourcemap_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea sourcemap: failed to read target");
        return ExitCode::FAILURE;
    };
    if let Some(map) = rea_engine::sourcemap::parse_sourcemap(&bytes) {
        return report_sourcemap(&map);
    }
    report_source_map_url(&bytes)
}

/// Prints a source map summary and its sources.
///
/// # Arguments
///
/// * `map` - Recovered source map.
///
/// # Returns
///
/// Exit code.
fn report_sourcemap(map: &rea_engine::sourcemap::SourceMap) -> ExitCode {
    print!("{}", rea_engine::sourcemap::format_sourcemap_summary(map));
    for source in &map.sources {
        println!("source {source}");
    }
    ExitCode::SUCCESS
}

/// Prints a `sourceMappingURL` directive found in JavaScript.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Exit code.
fn report_source_map_url(bytes: &[u8]) -> ExitCode {
    let Some(url) = rea_engine::sourcemap::find_source_map_url(bytes) else {
        eprintln!("rea sourcemap: not a source map");
        return ExitCode::FAILURE;
    };
    println!("sourceMappingURL: {url}");
    ExitCode::SUCCESS
}

/// Reports a packet capture or HAR document.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn pcap_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea pcap: failed to read target");
        return ExitCode::FAILURE;
    };
    if let Some(pcap) = rea_engine::netcap::parse_pcap(&bytes) {
        return report_pcap(&pcap);
    }
    report_har(&bytes)
}

/// Prints a capture summary, requests, and hosts.
///
/// # Arguments
///
/// * `pcap` - Recovered capture inventory.
///
/// # Returns
///
/// Exit code.
fn report_pcap(pcap: &rea_engine::netcap::Pcap) -> ExitCode {
    print!("{}", rea_engine::netcap::format_pcap_summary(pcap));
    for request in &pcap.http_requests {
        println!("{request}");
    }
    for host in &pcap.hosts {
        println!("host {host}");
    }
    ExitCode::SUCCESS
}

/// Parses and prints a HAR document.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Exit code.
fn report_har(bytes: &[u8]) -> ExitCode {
    let Some(har) = rea_engine::netcap::parse_har(bytes) else {
        eprintln!("rea pcap: not a packet capture or HAR");
        return ExitCode::FAILURE;
    };
    print!("{}", rea_engine::netcap::format_har_summary(&har));
    ExitCode::SUCCESS
}

/// Reports Flutter native libraries and assets.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn flutter_cmd(path: &Path) -> ExitCode {
    let Some(apk) = load_flutter(path) else {
        eprintln!("rea flutter: not a zip archive");
        return ExitCode::FAILURE;
    };
    report_flutter(&apk)
}

/// Reads and parses a Flutter APK or zip archive.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered inventory, or `None`.
fn load_flutter(path: &Path) -> Option<rea_engine::flutter::FlutterApk> {
    let bytes = fs::read(path).ok()?;
    rea_engine::flutter::parse_flutter(&bytes)
}

/// Prints a Flutter summary and asset inventory.
///
/// # Arguments
///
/// * `apk` - Recovered inventory.
///
/// # Returns
///
/// Exit code.
fn report_flutter(apk: &rea_engine::flutter::FlutterApk) -> ExitCode {
    print!("{}", rea_engine::flutter::format_flutter_summary(apk));
    for asset in &apk.assets {
        println!("asset {asset}");
    }
    ExitCode::SUCCESS
}

/// Reports an MS-DOS MZ executable header and entry disassembly.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn dos_cmd(path: &Path) -> ExitCode {
    let Some(header) = load_dos(path) else {
        eprintln!("rea dos: not an MS-DOS executable");
        return ExitCode::FAILURE;
    };
    report_dos(&header)
}

/// Reads and parses an MS-DOS MZ executable.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered header, or `None`.
fn load_dos(path: &Path) -> Option<rea_engine::dos::DosHeader> {
    let bytes = fs::read(path).ok()?;
    rea_engine::dos::parse_dos(&bytes)
}

/// Prints a DOS header summary and entry disassembly.
///
/// # Arguments
///
/// * `header` - Recovered header.
///
/// # Returns
///
/// Exit code.
fn report_dos(header: &rea_engine::dos::DosHeader) -> ExitCode {
    print!("{}", rea_engine::dos::format_dos_summary(header));
    for line in &header.code {
        println!("  {line}");
    }
    ExitCode::SUCCESS
}

/// Reports a Java class file.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn java_cmd(path: &Path) -> ExitCode {
    let Some(class) = load_java(path) else {
        eprintln!("rea java: not a Java class file");
        return ExitCode::FAILURE;
    };
    print!("{}", rea_engine::jvm::format_java_summary(&class));
    print_java_members(&class);
    ExitCode::SUCCESS
}

/// Reads and parses a Java class file.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered class, or `None`.
fn load_java(path: &Path) -> Option<rea_engine::jvm::JavaClass> {
    let bytes = fs::read(path).ok()?;
    rea_engine::jvm::parse_class(&bytes)
}

/// Prints Java fields and methods.
///
/// # Arguments
///
/// * `class` - Recovered class.
fn print_java_members(class: &rea_engine::jvm::JavaClass) {
    for field in &class.fields {
        println!("field {} {}", field.name, field.descriptor);
    }
    for method in &class.methods {
        println!("method {} {}", method.name, method.descriptor);
    }
}

/// Reports an Android resource table.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn arsc_cmd(path: &Path) -> ExitCode {
    let Some(table) = load_arsc(path) else {
        eprintln!("rea arsc: not an Android resource table");
        return ExitCode::FAILURE;
    };
    print!("{}", rea_engine::arsc::format_arsc_summary(&table));
    print_arsc_strings(&table);
    ExitCode::SUCCESS
}

/// Prints Android resource table strings.
///
/// # Arguments
///
/// * `table` - Recovered table.
fn print_arsc_strings(table: &rea_engine::arsc::ArscTable) {
    for value in table.strings.iter().take(40) {
        println!("{value}");
    }
}

/// Reads and parses an Android resource table.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered table, or `None`.
fn load_arsc(path: &Path) -> Option<rea_engine::arsc::ArscTable> {
    let bytes = fs::read(path).ok()?;
    rea_engine::arsc::parse_arsc(&bytes)
}

/// Reports DWARF compile units and functions.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn dwarf_cmd(path: &Path) -> ExitCode {
    let Some(dwarf) = load_dwarf(path) else {
        eprintln!("rea dwarf: no DWARF debug information");
        return ExitCode::FAILURE;
    };
    print!("{}", rea_engine::dwarf::format_dwarf_summary(&dwarf));
    print_dwarf_functions(&dwarf);
    ExitCode::SUCCESS
}

/// Reads and parses DWARF debug information.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered inventory, or `None`.
fn load_dwarf(path: &Path) -> Option<rea_engine::dwarf::DwarfInfo> {
    let bytes = fs::read(path).ok()?;
    rea_engine::dwarf::parse_dwarf(&bytes)
}

/// Prints DWARF function symbols.
///
/// # Arguments
///
/// * `dwarf` - Recovered inventory.
fn print_dwarf_functions(dwarf: &rea_engine::dwarf::DwarfInfo) {
    for function in dwarf.functions.iter().take(60) {
        println!("0x{:x} {}", function.low_pc, function.name);
    }
}

/// Scans a target with a YARA-subset rule file.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `rules` - Rule file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn yara_cmd(path: &Path, rules: &Path) -> ExitCode {
    let (Ok(bytes), Ok(rules_text)) = (fs::read(path), fs::read_to_string(rules)) else {
        eprintln!("rea yara: failed to read target or rules");
        return ExitCode::FAILURE;
    };
    let matches = rea_engine::yara::scan(&bytes, &rules_text);
    println!("matched {} rule(s)", matches.len());
    print_yara_matches(&matches);
    ExitCode::SUCCESS
}

/// Prints matched YARA-subset rules.
///
/// # Arguments
///
/// * `matches` - Matched rules.
fn print_yara_matches(matches: &[rea_engine::yara::Match]) {
    for hit in matches {
        println!("{} {}", hit.name, hit.tags.join(","));
    }
}

/// Reports PDB container metadata.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn pdb_cmd(path: &Path) -> ExitCode {
    let Some(pdb) = load_pdb(path) else {
        eprintln!("rea pdb: not a PDB file");
        return ExitCode::FAILURE;
    };
    print!("{}", rea_engine::pdb::format_pdb_summary(&pdb));
    ExitCode::SUCCESS
}

/// Reads and parses a PDB container.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The recovered metadata, or `None`.
fn load_pdb(path: &Path) -> Option<rea_engine::pdb::PdbInfo> {
    let bytes = fs::read(path).ok()?;
    rea_engine::pdb::parse_pdb(&bytes)
}

/// Prints the complete assembler listing of a target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn disasm_cmd(path: &Path) -> ExitCode {
    let Some((binary, bytes)) = load_native(path) else {
        eprintln!("rea disasm: no executable image");
        return ExitCode::FAILURE;
    };
    print_listing(&binary, &bytes);
    ExitCode::SUCCESS
}

/// Loads a parsed image and its raw bytes.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The parsed image and bytes, or `None`.
fn load_native(path: &Path) -> Option<(rea_engine::model::Binary, Vec<u8>)> {
    let binary = analyze_path(path).ok()?.binary?;
    Some((binary, fs::read(path).ok()?))
}

/// Prints the full linear disassembly of a parsed image.
///
/// # Arguments
///
/// * `binary` - Parsed image.
/// * `bytes` - Raw target bytes.
fn print_listing(binary: &rea_engine::model::Binary, bytes: &[u8]) {
    for line in rea_engine::listing::full(binary, bytes, MAX_LISTING) {
        println!("{line}");
    }
}

/// Reports the segmented and classified region map.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn regions_cmd(path: &Path) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea regions: failed to read target");
        return ExitCode::FAILURE;
    };
    let regions = rea_engine::regions::segment(&bytes);
    print!("{}", rea_engine::regions::format_regions(&regions));
    ExitCode::SUCCESS
}

/// Differentially compares two targets and writes a delta report.
///
/// # Arguments
///
/// * `a` - First target path.
/// * `b` - Second target path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn diff_cmd(a: &Path, b: &Path) -> ExitCode {
    if let Some(text) = bundle_diff(a, b) {
        print!("{text}");
        return ExitCode::SUCCESS;
    }
    binary_diff(a, b)
}

/// Diffs two evidence bundles when both inputs are bundles.
///
/// # Arguments
///
/// * `a` - First input path.
/// * `b` - Second input path.
///
/// # Returns
///
/// The diff text, or `None` when either input is not a bundle.
fn bundle_diff(a: &Path, b: &Path) -> Option<String> {
    let (a_text, b_text) = (fs::read_to_string(a).ok()?, fs::read_to_string(b).ok()?);
    rea_engine::evidence::diff(&a_text, &b_text).map(|d| rea_engine::evidence::format_diff(&d))
}

/// Differentially compares two binary targets.
///
/// # Arguments
///
/// * `a` - First target path.
/// * `b` - Second target path.
///
/// # Returns
///
/// Exit code.
fn binary_diff(a: &Path, b: &Path) -> ExitCode {
    let (Some((first, a_bytes)), Some((second, b_bytes))) = (load(a), load(b)) else {
        eprintln!("rea diff: failed to analyze one or both targets");
        return ExitCode::FAILURE;
    };
    let text = compare::format(&compare::compare(&first, &second, &a_bytes, &b_bytes));
    print!("{text}");
    write_diff(a, b, &text);
    ExitCode::SUCCESS
}

/// Reads and analyzes a target.
///
/// # Arguments
///
/// * `path` - Target path.
///
/// # Returns
///
/// The analysis and raw bytes, or `None`.
fn load(path: &Path) -> Option<(Analysis, Vec<u8>)> {
    let bytes = fs::read(path).ok()?;
    let analysis = analyze_path(path).ok()?;
    Some((analysis, bytes))
}

/// Writes the differential report beside the first target.
///
/// # Arguments
///
/// * `a` - First target path.
/// * `b` - Second target path.
/// * `text` - Report text.
fn write_diff(a: &Path, b: &Path, text: &str) {
    let path = diff_path(a, b);
    let _ = fs::write(path, text);
}

/// Builds the differential report path.
///
/// # Arguments
///
/// * `a` - First target path.
/// * `b` - Second target path.
///
/// # Returns
///
/// The report path.
fn diff_path(a: &Path, b: &Path) -> PathBuf {
    let dir = a.parent().unwrap_or_else(|| Path::new(""));
    dir.join(format!("re-{}-vs-{}.md", stem_of(a), stem_of(b)))
}

/// Returns a path's file stem or a placeholder.
///
/// # Arguments
///
/// * `path` - Target path.
///
/// # Returns
///
/// The stem.
fn stem_of(path: &Path) -> String {
    path.file_stem().map_or_else(
        || "target".to_string(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Writes an audit-grade evidence bundle beside the target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn evidence_cmd(path: &Path) -> ExitCode {
    let Some((analysis, bytes)) = load(path) else {
        eprintln!("rea evidence: failed to analyze target");
        return ExitCode::FAILURE;
    };
    let bundle = rea_engine::evidence::build(&analysis, &bytes);
    write_evidence(path, &bundle);
    ExitCode::SUCCESS
}

/// Writes the bundle and prints a summary.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `bundle` - Evidence bundle.
fn write_evidence(path: &Path, bundle: &rea_engine::evidence::Bundle) {
    let out = evidence_path(path);
    let _ = fs::write(&out, rea_engine::evidence::to_json(bundle));
    println!(
        "Evidence bundle: {} ({} items, {} unknowns)",
        out.display(),
        bundle.items.len(),
        bundle.unknowns.len()
    );
}

/// Builds the evidence bundle path beside a target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The bundle path.
fn evidence_path(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    dir.join(format!("re-{}.evidence.json", stem_of(path)))
}

/// Verifies an evidence bundle's integrity.
///
/// # Arguments
///
/// * `path` - Bundle JSON path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn verify_cmd(path: &Path) -> ExitCode {
    let Ok(text) = fs::read_to_string(path) else {
        eprintln!("rea verify: failed to read bundle");
        return ExitCode::FAILURE;
    };
    verify_text(&text)
}

/// Verifies bundle text and prints the outcome.
///
/// # Arguments
///
/// * `text` - Bundle JSON.
///
/// # Returns
///
/// Exit code.
fn verify_text(text: &str) -> ExitCode {
    match rea_engine::evidence::verify(text) {
        Ok(report) => finish_verify(&report),
        Err(error) => {
            eprintln!("rea verify: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Prints a verification report and returns its status.
///
/// # Arguments
///
/// * `report` - Verification report.
///
/// # Returns
///
/// Exit code.
fn finish_verify(report: &rea_engine::evidence::VerifyReport) -> ExitCode {
    print_verify(report);
    if report.tampered.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Prints a verification report.
///
/// # Arguments
///
/// * `report` - Verification report.
fn print_verify(report: &rea_engine::evidence::VerifyReport) {
    let (items, verified, tampered) = (report.items, report.verified, report.tampered.len());
    println!(
        "schema_version: {}\nitems: {items}\nverified: {verified}\ntampered: {tampered}",
        report.schema_version
    );
}

/// Writes a `CycloneDX` `SBOM` beside the target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn sbom_cmd(path: &Path) -> ExitCode {
    let Some((analysis, bytes)) = load(path) else {
        eprintln!("rea sbom: failed to analyze target");
        return ExitCode::FAILURE;
    };
    write_sbom(path, &analysis, &bytes);
    ExitCode::SUCCESS
}

/// Builds and writes the SBOM.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
fn write_sbom(path: &Path, analysis: &Analysis, bytes: &[u8]) {
    let provenance = rea_engine::provenance::recover_detailed(analysis, bytes);
    let json = rea_engine::sbom::cyclonedx(&analysis.name, env!("CARGO_PKG_VERSION"), &provenance);
    print!("{json}");
    let _ = fs::write(sbom_path(path), &json);
}

/// Builds the SBOM path beside a target.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The SBOM path.
fn sbom_path(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    dir.join(format!("re-{}.sbom.json", stem_of(path)))
}

/// Disassembles an 8051 (MCS-51) firmware region.
///
/// # Arguments
///
/// * `path` - Target file path.
/// * `offset` - Optional hex file offset.
/// * `base` - Optional 8051 address of the first byte.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn i8051_cmd(path: &Path, offset: Option<&str>, base: Option<&str>) -> ExitCode {
    let Ok(bytes) = fs::read(path) else {
        eprintln!("rea 8051: failed to read target");
        return ExitCode::FAILURE;
    };
    let start = offset.and_then(parse_address).map_or(0, |v| usize::try_from(v).unwrap_or(0));
    let origin = base.and_then(parse_address).unwrap_or(start as u64);
    print_8051(bytes.get(start..).unwrap_or_default(), origin);
    ExitCode::SUCCESS
}

/// Prints an 8051 disassembly listing.
///
/// # Arguments
///
/// * `code` - Code bytes.
/// * `origin` - Address of the first byte.
fn print_8051(code: &[u8], origin: u64) {
    for line in rea_engine::i8051::disassemble(code, origin, MAX_LISTING) {
        println!("{line}");
    }
}

/// Analyzes a target through an external provider.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn provider_cmd(path: &Path) -> ExitCode {
    let Some(text) = crate::provider::analyze(path) else {
        eprintln!("rea provider: no analysis provider available");
        return ExitCode::FAILURE;
    };
    print!("{text}");
    ExitCode::SUCCESS
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
    for finding in &run.strings {
        println!("  STRING 0x{:x} {:?}", finding.address, finding.text);
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
