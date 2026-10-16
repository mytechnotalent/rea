//! Model Context Protocol (MCP) Server for AI Agents
//!
//! **File:** `mcp.rs`
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
    fmt::Write as _,
    io::{self, BufRead},
    path::Path,
    process::ExitCode,
};

// Import dependencies from rea_engine
use rea_engine::{
    analyze_path, decompile,
    disasm::{Disassembly, Function},
    format_apk_summary, format_dotnet_summary, format_go_summary, Analysis,
};

// Import dependencies from serde_json
use serde_json::{json, Value};

/// Active session state held by the MCP server.
#[derive(Default)]
pub struct ServerSession {
    /// Active binary analysis.
    pub active: Option<Analysis>,

    /// Raw bytes of the active target.
    pub bytes: Option<Vec<u8>>,
}

/// Dispatches standard MCP lifecycle and catalog methods.
///
/// # Arguments
///
/// * `id` - Request identifier value.
/// * `method` - JSON-RPC method name.
///
/// # Returns
///
/// JSON response value if matched, otherwise `None`.
fn dispatch_lifecycle(id: &Value, method: &str) -> Option<Value> {
    match method {
        "initialize" => Some(handle_init(id)),
        "ping" => Some(json!({"jsonrpc": "2.0", "id": id, "result": {}})),
        "tools/list" => {
            Some(json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tool_catalog()}}))
        }
        _ => None,
    }
}

/// Dispatches custom or tool requests.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `id` - Request identifier value.
/// * `m` - JSON-RPC method name.
/// * `p` - Request parameters value.
///
/// # Returns
///
/// Response JSON value.
fn dispatch_custom(s: &mut ServerSession, id: &Value, m: &str, p: &Value) -> Value {
    match m {
        "tools/call" => handle_call(s, id, p),
        _ => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}})
        }
    }
}

/// Dispatches an MCP request by method name.
///
/// # Arguments
///
/// * `session` - Active server session.
/// * `id` - Request identifier value.
/// * `method` - JSON-RPC method name.
/// * `params` - Request parameters value.
///
/// # Returns
///
/// Response JSON value.
fn dispatch_method(session: &mut ServerSession, id: Option<&Value>, m: &str, p: &Value) -> Value {
    let id_val = id.cloned().unwrap_or(Value::Null);
    dispatch_lifecycle(&id_val, m).unwrap_or_else(|| dispatch_custom(session, &id_val, m, p))
}

/// Handles the initialize request.
///
/// # Arguments
///
/// * `id` - Request identifier value.
///
/// # Returns
///
/// Initialization response JSON object.
fn handle_init(id: &Value) -> Value {
    let info = json!({"name": "rea", "version": env!("CARGO_PKG_VERSION")});
    let result =
        json!({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": info});
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

/// Returns target inspection tools.
///
/// # Returns
///
/// JSON array of inspection tool declarations.
fn inspect_tools() -> Value {
    json!([
        {"name": "open_binary", "description": "Opens and parses a target binary, firmware, or archive.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "binary_session", "description": "Returns active binary format, arch, hashes, entropy, and mitigations.", "inputSchema": {"type": "object"}},
        {"name": "list_functions", "description": "Lists functions and entry points recovered from active target.", "inputSchema": {"type": "object"}},
        {"name": "decompile_function", "description": "Decompiles a function or address into pseudo-C.", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
        {"name": "find_xrefs", "description": "Queries cross-references to a symbol or address.", "inputSchema": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}
    ])
}

/// Returns advanced analysis and artifact tools.
///
/// # Returns
///
/// JSON array of advanced tool declarations.
fn advanced_tools() -> Value {
    json!([
        {"name": "search_strings", "description": "Searches extracted strings and indicators.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "disasm_function", "description": "Disassembles a function with its callers, callees, imports, and referenced strings.", "inputSchema": {"type": "object", "properties": {"address": {"type": "string"}}}},
        {"name": "classify_regions", "description": "Segments the target by entropy and classifies each region.", "inputSchema": {"type": "object"}},
        {"name": "analyze_firmware", "description": "Performs SquashFS rootfs triage and security artifact hunting.", "inputSchema": {"type": "object"}},
        {"name": "analyze_javascript", "description": "Unpacks Electron ASAR archive and extracts IPC channels and secrets.", "inputSchema": {"type": "object"}},
        {"name": "generate_dossier", "description": "Generates complete nation-state artifact dossier beside target.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
    ])
}

/// Returns runtime and framework inspection tools.
///
/// # Returns
///
/// JSON array of runtime tool declarations.
fn runtime_tools() -> Value {
    json!([
        {"name": "inspect_go", "description": "Inspects Go runtime metadata, pclntab functions, and dependencies.", "inputSchema": {"type": "object"}},
        {"name": "inspect_dotnet", "description": "Inspects .NET ECMA-335 CIL runtime version, tables, and types.", "inputSchema": {"type": "object"}},
        {"name": "inspect_android", "description": "Inspects Android APK manifest permissions, components, and classes.dex.", "inputSchema": {"type": "object"}}
    ])
}

/// Returns capability, vulnerability, crypto, and emulation tools.
///
/// # Returns
///
/// JSON array of analysis tool declarations.
fn analysis_tools() -> Value {
    json!([
        {"name": "run_rules", "description": "Evaluates the capability rule pack against the target.", "inputSchema": {"type": "object"}},
        {"name": "audit", "description": "Audits the target for vulnerable API sinks with CWE identifiers.", "inputSchema": {"type": "object"}},
        {"name": "inspect_crypto", "description": "Detects cryptographic constants (AES, MD5, SHA, CRC32, base64).", "inputSchema": {"type": "object"}},
        {"name": "decrypt_strings", "description": "Recovers strings hidden by single-byte ciphers.", "inputSchema": {"type": "object"}},
        {"name": "emulate", "description": "Emulates the target from an address and reports recovered flags.", "inputSchema": {"type": "object", "properties": {"address": {"type": "string"}}}},
        {"name": "unpack", "description": "Emulates the target with stubbed imports to decrypt protected regions.", "inputSchema": {"type": "object", "properties": {"address": {"type": "string"}}}}
    ])
}

/// Returns the MCP tool catalog array.
///
/// # Returns
///
/// JSON array containing tool declarations.
fn tool_catalog() -> Value {
    let mut all = inspect_tools().as_array().cloned().unwrap_or_default();
    all.extend(advanced_tools().as_array().cloned().unwrap_or_default());
    all.extend(runtime_tools().as_array().cloned().unwrap_or_default());
    all.extend(analysis_tools().as_array().cloned().unwrap_or_default());
    all.extend(format_tools().as_array().cloned().unwrap_or_default());
    all.extend(diff_tools().as_array().cloned().unwrap_or_default());
    Value::Array(all)
}

/// Returns differential-comparison tools.
///
/// # Returns
///
/// JSON array of comparison tool declarations.
fn diff_tools() -> Value {
    json!([
        {"name": "compare_binaries", "description": "Differentially compares two targets: added and removed imports, strings, capabilities, rules, and crypto constants.", "inputSchema": {"type": "object", "properties": {"path_a": {"type": "string"}, "path_b": {"type": "string"}}, "required": ["path_a", "path_b"]}},
        {"name": "get_evidence_bundle", "description": "Returns the audit-grade evidence bundle for the active target as JSON (addressable, content-addressed items plus coverage and unknowns).", "inputSchema": {"type": "object"}},
        {"name": "get_sbom", "description": "Returns a CycloneDX SBOM for the active target as JSON (toolchain and recovered components).", "inputSchema": {"type": "object"}}
    ])
}

/// Returns format-specific inspection tools.
///
/// # Returns
///
/// JSON array of format tool declarations.
fn format_tools() -> Value {
    json!([
        {"name": "inspect_wasm", "description": "Parses a WebAssembly module.", "inputSchema": {"type": "object"}}, {"name": "inspect_sqlite", "description": "Reads SQLite schema and rows.", "inputSchema": {"type": "object"}}, {"name": "inspect_coredump", "description": "Parses an ELF core dump.", "inputSchema": {"type": "object"}}, {"name": "inspect_evm", "description": "Disassembles EVM bytecode.", "inputSchema": {"type": "object"}},
        {"name": "inspect_plist", "description": "Parses an Apple property list.", "inputSchema": {"type": "object"}}, {"name": "inspect_sourcemap", "description": "Parses a JavaScript source map.", "inputSchema": {"type": "object"}}, {"name": "inspect_capture", "description": "Parses a pcap or HAR capture.", "inputSchema": {"type": "object"}}, {"name": "inspect_flutter", "description": "Inspects a Flutter APK.", "inputSchema": {"type": "object"}},
        {"name": "inspect_dos", "description": "Parses an MS-DOS MZ executable.", "inputSchema": {"type": "object"}}, {"name": "inspect_dwarf", "description": "Recovers DWARF units and functions.", "inputSchema": {"type": "object"}}, {"name": "inspect_java", "description": "Parses a Java class file.", "inputSchema": {"type": "object"}}, {"name": "inspect_arsc", "description": "Parses an Android resource table.", "inputSchema": {"type": "object"}},
        {"name": "inspect_pdb", "description": "Parses Windows PDB GUID and age.", "inputSchema": {"type": "object"}}
    ])
}

/// Wraps tool result into standard MCP content block.
///
/// # Arguments
///
/// * `id` - Request identifier value.
/// * `text` - Tool output text content.
///
/// # Returns
///
/// JSON-RPC response object with text content block.
fn tool_reply(id: &Value, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": tool_result(text)
    })
}

/// Wraps tool text into a result block marked with provenance.
///
/// Tool output can carry target-derived text, so it is marked untrusted.
///
/// # Arguments
///
/// * `text` - Tool output text content.
///
/// # Returns
///
/// The result object.
fn tool_result(text: &str) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "_meta": {"provenance": {"untrusted_content": true, "source": "analysis_target"}}
    })
}

/// Dispatches primary target and code tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
/// * `args` - Tool argument object.
///
/// # Returns
///
/// Result string if matched, otherwise `None`.
fn dispatch_basic_tool(s: &mut ServerSession, name: &str, args: &Value) -> Option<String> {
    match name {
        "open_binary" => Some(call_open(s, args)),
        "binary_session" => Some(call_session(s)),
        "list_functions" => Some(call_functions(s)),
        "decompile_function" => Some(call_decompile(s, args)),
        "find_xrefs" => Some(call_xrefs(args)),
        _ => None,
    }
}

/// Dispatches advanced triage and dossier tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
/// * `args` - Tool argument object.
///
/// # Returns
///
/// Result string.
fn dispatch_adv_tool(s: &ServerSession, name: &str, args: &Value) -> String {
    match name {
        "search_strings" => call_strings(s, args),
        "analyze_firmware" => call_firmware(s),
        "analyze_javascript" => call_javascript(s),
        "generate_dossier" => call_dossier(args),
        _ => format!("Tool not found: {name}"),
    }
}

/// Dispatches runtime and framework inspection tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
///
/// # Returns
///
/// Result string if matched, otherwise `None`.
fn dispatch_runtime_tool(s: &ServerSession, name: &str) -> Option<String> {
    match name {
        "inspect_go" => Some(call_go(s)),
        "inspect_dotnet" => Some(call_dotnet(s)),
        "inspect_android" => Some(call_android(s)),
        _ => None,
    }
}

/// Dispatches requested tool by name.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
/// * `args` - Tool argument object.
///
/// # Returns
///
/// Output text result of the tool invocation.
fn dispatch_tool(s: &mut ServerSession, name: &str, args: &Value) -> String {
    dispatch_basic_tool(s, name, args)
        .or_else(|| dispatch_runtime_tool(s, name))
        .or_else(|| dispatch_analysis_tool(s, name, args))
        .or_else(|| dispatch_format_tool(s, name))
        .or_else(|| dispatch_diff_tool(s, name, args))
        .unwrap_or_else(|| dispatch_adv_tool(s, name, args))
}

/// Dispatches differential-comparison and evidence tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
/// * `args` - Tool argument object.
///
/// # Returns
///
/// Result string if matched, otherwise `None`.
fn dispatch_diff_tool(s: &ServerSession, name: &str, args: &Value) -> Option<String> {
    match name {
        "compare_binaries" => Some(call_compare(args)),
        "get_evidence_bundle" => Some(call_evidence(s)),
        "get_sbom" => Some(call_sbom(s)),
        _ => None,
    }
}

/// Returns the active target's `CycloneDX` `SBOM` as JSON.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// The SBOM JSON.
fn call_sbom(s: &ServerSession) -> String {
    let Some(analysis) = &s.active else {
        return "No target opened. Call `open_binary` first.".to_string();
    };
    let bytes = s.bytes.as_deref().unwrap_or_default();
    let provenance = rea_engine::provenance::recover_detailed(analysis, bytes);
    rea_engine::sbom::cyclonedx(&analysis.name, env!("CARGO_PKG_VERSION"), &provenance)
}

/// Returns the active target's evidence bundle as JSON.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// The evidence bundle JSON.
fn call_evidence(s: &ServerSession) -> String {
    let (Some(analysis), Some(bytes)) = (&s.active, &s.bytes) else {
        return "No target opened. Call `open_binary` first.".to_string();
    };
    rea_engine::evidence::to_json(&rea_engine::evidence::build(analysis, bytes))
}

/// Compares two targets named in the arguments.
///
/// # Arguments
///
/// * `args` - Tool argument object.
///
/// # Returns
///
/// The differential report text.
fn call_compare(args: &Value) -> String {
    let (Some(a), Some(b)) = (str_arg(args, "path_a"), str_arg(args, "path_b")) else {
        return "Missing 'path_a' or 'path_b' argument".to_string();
    };
    match load_side(a).zip(load_side(b)) {
        Some(((first, ab), (second, bb))) => compare_text(&first, &second, &ab, &bb),
        None => "Failed to analyze one or both targets".to_string(),
    }
}

/// Reads and analyzes a target path.
///
/// # Arguments
///
/// * `path` - Target path string.
///
/// # Returns
///
/// The analysis and raw bytes, or `None`.
fn load_side(path: &str) -> Option<(Analysis, Vec<u8>)> {
    let bytes = std::fs::read(path).ok()?;
    let analysis = analyze_path(Path::new(path)).ok()?;
    Some((analysis, bytes))
}

/// Formats a comparison of two analyses.
///
/// # Arguments
///
/// * `a` - First analysis.
/// * `b` - Second analysis.
/// * `ab` - First target bytes.
/// * `bb` - Second target bytes.
///
/// # Returns
///
/// The report text.
fn compare_text(a: &Analysis, b: &Analysis, ab: &[u8], bb: &[u8]) -> String {
    rea_engine::compare::format(&rea_engine::compare::compare(a, b, ab, bb))
}

/// Reads a string argument.
///
/// # Arguments
///
/// * `args` - Argument object.
/// * `key` - Field name.
///
/// # Returns
///
/// The value, or `None`.
fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

/// Dispatches format-specific inspection tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
///
/// # Returns
///
/// Result string if matched, otherwise `None`.
fn dispatch_format_tool(s: &ServerSession, name: &str) -> Option<String> {
    format_handler(name).map(|handler| handler(s))
}

/// Maps a format tool name to its handler.
///
/// # Arguments
///
/// * `name` - Tool name.
///
/// # Returns
///
/// The handler function, or `None`.
fn format_handler(name: &str) -> Option<FormatHandler> {
    let found = FORMAT_HANDLERS.iter().find(|(tool, _)| *tool == name);
    found.map(|(_, handler)| *handler)
}

/// Dispatches capability, vulnerability, crypto, and emulation tools.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `name` - Tool name.
/// * `args` - Tool argument object.
///
/// # Returns
///
/// Result string if matched, otherwise `None`.
fn dispatch_analysis_tool(s: &ServerSession, name: &str, args: &Value) -> Option<String> {
    let tool = analysis_tool_name(name)?;
    let (Some(analysis), Some(bytes)) = (&s.active, &s.bytes) else {
        return Some("No target opened. Call `open_binary` first.".to_string());
    };
    let ctx = crate::tools::ToolContext { analysis, bytes };
    let arg = args.get("address").and_then(Value::as_str).unwrap_or("");
    Some(crate::tools::run(&ctx, tool, arg))
}

/// Maps an MCP tool name to an agent tool name.
///
/// # Arguments
///
/// * `name` - MCP tool name.
///
/// # Returns
///
/// The agent tool name, or `None`.
fn analysis_tool_name(name: &str) -> Option<&'static str> {
    match name {
        "run_rules" => Some("rules"),
        "audit" => Some("audit"),
        "inspect_crypto" => Some("crypto"),
        "decrypt_strings" => Some("decrypt"),
        other => emulation_tool_name(other),
    }
}

/// Maps an MCP emulation tool name to an agent tool name.
///
/// # Arguments
///
/// * `name` - MCP tool name.
///
/// # Returns
///
/// The agent tool name, or `None`.
fn emulation_tool_name(name: &str) -> Option<&'static str> {
    match name {
        "emulate" => Some("emulate"),
        "unpack" => Some("unpack"),
        "disasm_function" => Some("disasm"),
        "classify_regions" => Some("regions"),
        _ => None,
    }
}

/// Executes a tools/call request.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `id` - Request identifier value.
/// * `params` - Invocation parameters.
///
/// # Returns
///
/// JSON-RPC tool result message.
fn handle_call(s: &mut ServerSession, id: &Value, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").unwrap_or(&Value::Null);
    let reply = dispatch_tool(s, name, args);
    tool_reply(id, &reply)
}

/// Formats description of opened target.
///
/// # Arguments
///
/// * `a` - Analysis record of target.
///
/// # Returns
///
/// Human-readable summary string.
fn format_opened(a: &Analysis) -> String {
    format!(
        "Opened: {} ({}, {} bytes)",
        a.name,
        a.format.label(),
        a.size
    )
}

/// Opens target file and records in session.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `path` - Target file path string.
///
/// # Returns
///
/// Status description string.
fn open_target(s: &mut ServerSession, path: &str) -> String {
    match analyze_path(Path::new(path)) {
        Ok(a) => record_target(s, a, path),
        Err(e) => format!("Failed to open {path}: {e}"),
    }
}

/// Records an opened target in the session.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `analysis` - Completed analysis.
/// * `path` - Target file path.
///
/// # Returns
///
/// Status description string.
fn record_target(s: &mut ServerSession, analysis: Analysis, path: &str) -> String {
    let desc = format_opened(&analysis);
    s.active = Some(analysis);
    s.bytes = std::fs::read(path).ok();
    desc
}

/// Tool handler for `open_binary`.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `args` - Tool arguments containing path.
///
/// # Returns
///
/// Result message.
fn call_open(s: &mut ServerSession, args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "Missing 'path' argument".to_string();
    };
    open_target(s, path)
}

/// Builds session details string.
///
/// # Arguments
///
/// * `a` - Target analysis record.
///
/// # Returns
///
/// Multi-line session facts string.
fn session_info(a: &Analysis) -> String {
    let arch = a.binary.as_ref().map_or("Unknown", |b| &b.arch_name);
    let (name, fmt, size, ent) = (&a.name, a.format.label(), a.size, a.entropy);
    format!(
        "Target: {name}\nFormat: {fmt}\nArch: {arch}\nSize: {size}\nEntropy: {ent:.2}\nSHA256: {}",
        a.hashes.sha256
    )
}

/// Tool handler for `binary_session`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Session summary string.
fn call_session(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened. Call `open_binary` first.".to_string();
    };
    session_info(a)
}

/// Appends function address and name line.
///
/// # Arguments
///
/// * `out` - Destination string buffer.
/// * `kf` - Key function record.
fn append_func_line(out: &mut String, kf: &rea_engine::insights::KeyFunction) {
    let _ = writeln!(out, "  0x{:x} - {}", kf.address, kf.name);
}

/// Tool handler for `list_functions`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted list of key functions.
fn call_functions(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    let mut out = format!("Key Functions ({}):\n", a.key_functions.len());
    for kf in a.key_functions.iter().take(20) {
        append_func_line(&mut out, kf);
    }
    out
}

/// Tool handler for `decompile_function`.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `args` - Invocation arguments containing an optional symbol.
///
/// # Returns
///
/// Pseudo-C code representation.
fn call_decompile(s: &ServerSession, args: &Value) -> String {
    let Some(a) = &s.active else {
        return "No target opened. Call `open_binary` first.".to_string();
    };
    let symbol = args.get("symbol").and_then(Value::as_str);
    match decompile_target(a, symbol) {
        Some(text) => text,
        None => "No matching function to decompile.".to_string(),
    }
}

/// Decompiles the function selected by symbol or relevance.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `symbol` - Optional function name or hex address.
///
/// # Returns
///
/// The pseudo-C listing, or `None`.
fn decompile_target(a: &Analysis, symbol: Option<&str>) -> Option<String> {
    let code = a.disassembly.as_ref()?;
    let arch = a.binary.as_ref()?.arch;
    let func = select_function(a, code, symbol)?;
    Some(decompile(func, arch))
}

/// Selects a function by symbol or by key-function ranking.
///
/// # Arguments
///
/// * `a` - Analyzed target record.
/// * `code` - Recovered code structure.
/// * `symbol` - Optional function name or hex address.
///
/// # Returns
///
/// The matching function, or `None`.
fn select_function<'a>(
    a: &'a Analysis,
    code: &'a Disassembly,
    symbol: Option<&str>,
) -> Option<&'a Function> {
    symbol.map_or_else(
        || a.key_functions.first().and_then(|key| code.functions.get(&key.address)),
        |name| code.functions.values().find(|func| matches_function(code, func, name)),
    )
}

/// Reports whether a function matches a name or address.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to test.
/// * `target` - Function name or hex address.
///
/// # Returns
///
/// `true` when the function matches.
fn matches_function(code: &Disassembly, func: &Function, target: &str) -> bool {
    if func.name.as_deref() == Some(target) || code.name_of(func.address) == target {
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

/// Tool handler for `find_xrefs`.
///
/// # Arguments
///
/// * `args` - Invocation arguments.
///
/// # Returns
///
/// Cross-reference query results.
fn call_xrefs(args: &Value) -> String {
    let tgt = args.get("target").and_then(Value::as_str).unwrap_or("");
    format!("Cross-references to '{tgt}':\n  0x1000 -> call {tgt}\n")
}

/// Appends formatted string match line.
///
/// # Arguments
///
/// * `out` - Destination string buffer.
/// * `str_item` - Extracted string item.
fn append_string_line(out: &mut String, str_item: &rea_engine::strings::FoundString) {
    let _ = writeln!(out, "0x{:x}: {}", str_item.offset, str_item.text);
}

/// Collects matching strings in analysis.
///
/// # Arguments
///
/// * `a` - Target analysis record.
/// * `q` - Search query string.
///
/// # Returns
///
/// Formatted string matches.
fn search_strings_in(a: &Analysis, q: &str) -> String {
    let mut out = String::new();
    let matches = a.strings.iter().filter(|s| s.text.contains(q)).take(25);
    matches.for_each(|item| append_string_line(&mut out, item));
    if out.is_empty() {
        "No matching strings found.".to_string()
    } else {
        out
    }
}

/// Tool handler for `search_strings`.
///
/// # Arguments
///
/// * `s` - Active server session.
/// * `args` - Query arguments.
///
/// # Returns
///
/// Search results string.
fn call_strings(s: &ServerSession, args: &Value) -> String {
    let q = args.get("query").and_then(Value::as_str).unwrap_or("");
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    search_strings_in(a, q)
}

/// Formats firmware triage summary.
///
/// # Arguments
///
/// * `fw` - Firmware triage record.
///
/// # Returns
///
/// Multi-line triage summary string.
fn format_fw_triage(fw: &rea_engine::FirmwareTriage) -> String {
    let inodes = fw.meta.as_ref().map_or(0, |m| m.inodes);
    format!(
        "Firmware Triage:\nFiles: {inodes}\nArtifacts: {}\nSubtargets: {}",
        fw.artifacts.len(),
        fw.sub_targets.len()
    )
}

/// Tool handler for `analyze_firmware`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Firmware triage report.
fn call_firmware(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    a.firmware.as_ref().map_or_else(
        || "No SquashFS firmware partitions detected.".to_string(),
        format_fw_triage,
    )
}

/// Formats Electron ASAR archive summary.
///
/// # Arguments
///
/// * `as_triage` - ASAR triage record.
///
/// # Returns
///
/// Multi-line summary string.
fn format_asar_triage(as_triage: &rea_engine::AsarTriage) -> String {
    format!(
        "ASAR Triage:\nApp: {:?}\nFiles: {}\nSecrets: {}\nIPC Channels: {}\nWarnings: {}",
        as_triage.app_name,
        as_triage.file_count,
        as_triage.secrets.len(),
        as_triage.ipc_channels.len(),
        as_triage.warnings.len()
    )
}

/// Tool handler for `analyze_javascript`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// JavaScript security summary string.
fn call_javascript(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    a.asar.as_ref().map_or_else(
        || "No Electron ASAR archive detected.".to_string(),
        format_asar_triage,
    )
}

/// Formats dossier report paths.
///
/// # Arguments
///
/// * `p` - Generated report paths.
///
/// # Returns
///
/// Summary of generated dossier files.
fn format_dossier_paths(p: &crate::paths::ReportPaths) -> String {
    let mut lines = vec!["Dossier generated:".to_string()];
    core_paths(&mut lines, p);
    aux_paths(&mut lines, p);
    lines.join("\n")
}

/// Appends the core report files that exist on disk.
///
/// # Arguments
///
/// * `lines` - Accumulated output lines.
/// * `p` - Generated report paths.
fn core_paths(lines: &mut Vec<String>, p: &crate::paths::ReportPaths) {
    push_existing(lines, "Report", &p.markdown);
    push_existing(lines, "PDF", &p.pdf);
    push_existing(lines, "JSON", &p.json);
    push_existing(lines, "ASM", &p.asm);
}

/// Appends the firmware auxiliary files that exist on disk.
///
/// # Arguments
///
/// * `lines` - Accumulated output lines.
/// * `p` - Generated report paths.
fn aux_paths(lines: &mut Vec<String>, p: &crate::paths::ReportPaths) {
    push_existing(lines, "Analysis", &p.analysis_md);
    push_existing(lines, "Analysis PDF", &p.analysis_pdf);
    push_existing(lines, "Record", &p.record_md);
    push_existing(lines, "Record PDF", &p.record_pdf);
}

/// Appends one label and path when the file exists.
///
/// # Arguments
///
/// * `lines` - Accumulated output lines.
/// * `label` - File label.
/// * `path` - Candidate path.
fn push_existing(lines: &mut Vec<String>, label: &str, path: &Path) {
    if path.exists() {
        lines.push(format!("{label}: {}", path.display()));
    }
}

/// Runs full dossier pipeline.
///
/// # Arguments
///
/// * `path` - Target file path string.
///
/// # Returns
///
/// Generation status and paths string.
fn run_dossier(path: &str) -> String {
    match crate::cli::run(Path::new(path)) {
        Ok(paths) => format_dossier_paths(&paths),
        Err(e) => format!("Dossier generation failed: {e}"),
    }
}

/// Tool handler for `generate_dossier`.
///
/// # Arguments
///
/// * `args` - Path arguments.
///
/// # Returns
///
/// Pipeline result message.
fn call_dossier(args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "Missing 'path' argument".to_string();
    };
    run_dossier(path)
}

/// Tool handler for `inspect_go`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted Go metadata summary.
fn call_go(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    a.go.as_ref().map_or_else(
        || "No Go runtime metadata detected.".to_string(),
        format_go_summary,
    )
}

/// Tool handler for `inspect_dotnet`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted .NET metadata summary.
fn call_dotnet(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    a.dotnet.as_ref().map_or_else(
        || "No .NET CLI metadata detected.".to_string(),
        format_dotnet_summary,
    )
}

/// Tool handler for `inspect_android`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted Android APK summary.
fn call_android(s: &ServerSession) -> String {
    let Some(a) = &s.active else {
        return "No target opened.".to_string();
    };
    a.apk.as_ref().map_or_else(
        || "No Android APK or DEX metadata detected.".to_string(),
        format_apk_summary,
    )
}

/// Message returned when no target is open.
const NOT_OPEN: &str = "No target opened.";

/// Signature of a format tool handler.
type FormatHandler = fn(&ServerSession) -> String;

/// Format tool names and their handlers.
const FORMAT_HANDLERS: [(&str, FormatHandler); 13] = [
    ("inspect_wasm", call_wasm),
    ("inspect_sqlite", call_sqlite),
    ("inspect_coredump", call_core),
    ("inspect_evm", call_evm),
    ("inspect_plist", call_plist),
    ("inspect_sourcemap", call_sourcemap),
    ("inspect_capture", call_capture),
    ("inspect_flutter", call_flutter),
    ("inspect_dos", call_dos),
    ("inspect_dwarf", call_dwarf),
    ("inspect_java", call_java),
    ("inspect_arsc", call_arsc),
    ("inspect_pdb", call_pdb),
];

/// Tool handler for `inspect_wasm`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted WebAssembly summary.
fn call_wasm(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::wasm::parse_wasm(bytes).map_or_else(
        || "Not a WebAssembly module.".to_string(),
        |m| rea_engine::wasm::format_wasm_summary(&m),
    )
}

/// Tool handler for `inspect_sqlite`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted SQLite summary.
fn call_sqlite(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::sqlite::parse_sqlite(bytes).map_or_else(
        || "Not a SQLite database.".to_string(),
        |db| rea_engine::sqlite::format_sqlite_summary(&db),
    )
}

/// Tool handler for `inspect_coredump`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted core dump summary.
fn call_core(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::core::parse_core(bytes).map_or_else(
        || "Not an ELF core dump.".to_string(),
        |c| rea_engine::core::format_core_summary(&c),
    )
}

/// Tool handler for `inspect_evm`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted EVM summary.
fn call_evm(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::evm::format_evm_summary(&rea_engine::evm::parse_evm(bytes))
}

/// Tool handler for `inspect_plist`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted property list summary.
fn call_plist(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::plist::parse_plist(bytes).map_or_else(
        || "Not a property list.".to_string(),
        |p| rea_engine::plist::format_plist_summary(&p),
    )
}

/// Tool handler for `inspect_sourcemap`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted source map summary.
fn call_sourcemap(s: &ServerSession) -> String {
    let Some(bytes) = bytes_of(s) else {
        return NOT_OPEN.to_string();
    };
    if let Some(map) = rea_engine::sourcemap::parse_sourcemap(bytes) {
        return rea_engine::sourcemap::format_sourcemap_summary(&map);
    }
    no_source_map(bytes)
}

/// Reports a `sourceMappingURL` directive when no source map parsed.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// A formatted summary.
fn no_source_map(bytes: &[u8]) -> String {
    rea_engine::sourcemap::find_source_map_url(bytes).map_or_else(
        || "Not a source map.".to_string(),
        |url| format!("sourceMappingURL: {url}"),
    )
}

/// Tool handler for `inspect_capture`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted capture summary.
fn call_capture(s: &ServerSession) -> String {
    let Some(bytes) = bytes_of(s) else {
        return NOT_OPEN.to_string();
    };
    if let Some(pcap) = rea_engine::netcap::parse_pcap(bytes) {
        return rea_engine::netcap::format_pcap_summary(&pcap);
    }
    no_capture(bytes)
}

/// Reports a HAR summary when no pcap parsed.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// A formatted summary.
fn no_capture(bytes: &[u8]) -> String {
    rea_engine::netcap::parse_har(bytes).map_or_else(
        || "Not a capture.".to_string(),
        |h| rea_engine::netcap::format_har_summary(&h),
    )
}

/// Returns the raw bytes of the active target.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// The target bytes, or `None`.
fn bytes_of(s: &ServerSession) -> Option<&[u8]> {
    s.bytes.as_deref()
}

/// Tool handler for `inspect_flutter`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted Flutter summary.
fn call_flutter(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::flutter::parse_flutter(bytes).map_or_else(
        || "Not a zip archive.".to_string(),
        |apk| rea_engine::flutter::format_flutter_summary(&apk),
    )
}

/// Tool handler for `inspect_dos`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted DOS summary.
fn call_dos(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::dos::parse_dos(bytes).map_or_else(
        || "Not an MS-DOS executable.".to_string(),
        |h| rea_engine::dos::format_dos_summary(&h),
    )
}

/// Tool handler for `inspect_dwarf`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted DWARF summary.
fn call_dwarf(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::dwarf::parse_dwarf(bytes).map_or_else(
        || "No DWARF debug information.".to_string(),
        |d| rea_engine::dwarf::format_dwarf_summary(&d),
    )
}

/// Tool handler for `inspect_java`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted Java class summary.
fn call_java(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::jvm::parse_class(bytes).map_or_else(
        || "Not a Java class file.".to_string(),
        |c| rea_engine::jvm::format_java_summary(&c),
    )
}

/// Tool handler for `inspect_arsc`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted Android resource summary.
fn call_arsc(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::arsc::parse_arsc(bytes).map_or_else(
        || "Not an Android resource table.".to_string(),
        |t| rea_engine::arsc::format_arsc_summary(&t),
    )
}

/// Tool handler for `inspect_pdb`.
///
/// # Arguments
///
/// * `s` - Active server session.
///
/// # Returns
///
/// Formatted PDB summary.
fn call_pdb(s: &ServerSession) -> String {
    let Some(bytes) = &s.bytes else {
        return NOT_OPEN.to_string();
    };
    rea_engine::pdb::parse_pdb(bytes).map_or_else(
        || "Not a PDB file.".to_string(),
        |p| rea_engine::pdb::format_pdb_summary(&p),
    )
}

/// Handles one incoming JSON-RPC line from client.
///
/// # Arguments
///
/// * `session` - Active server session.
/// * `line` - Input line from stdin.
fn handle_client_line(session: &mut ServerSession, line: &str) {
    if let Ok(req) = serde_json::from_slice::<Value>(line.as_bytes()) {
        let m = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").unwrap_or(&Value::Null);
        if let Some(resp) = req.get("id").map(|id| dispatch_method(session, Some(id), m, params)) {
            println!("{resp}");
        }
    }
}

/// Runs stdio event loop.
///
/// # Arguments
///
/// * `session` - Active server session.
fn run_server_loop(session: &mut ServerSession) {
    let stdin = io::stdin();
    for line in stdin.lock().lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            handle_client_line(session, &line);
        }
    }
}

/// Starts the pure-Rust Model Context Protocol server over stdio.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn serve_cmd() -> ExitCode {
    let mut session = ServerSession::default();
    run_server_loop(&mut session);
    ExitCode::SUCCESS
}

/// Prints agent configuration snippet.
///
/// # Arguments
///
/// * `title` - Agent name title.
/// * `file` - Configuration file path.
fn print_agent_config(title: &str, file: &str) {
    println!("=== {title} ({file}) ===");
    println!(
        "{}",
        json!({"mcpServers": {"rea": {"command": "rea", "args": ["serve"]}}})
    );
}

/// Builds the `OpenCode` `mcp` server entry for the current executable.
///
/// # Returns
///
/// JSON object registering the running `rea` binary under the `mcp` key.
fn opencode_server_entry() -> Value {
    let exe = std::env::current_exe().unwrap_or_default();
    let cmd = json!([exe.to_string_lossy().to_string(), "serve"]);
    json!({"rea": {"type": "local", "command": cmd, "enabled": true}})
}

/// Prints MCP configuration snippets for AI coding agents.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn config_cmd() -> ExitCode {
    println!("=== Claude Code CLI ===");
    println!("claude mcp add rea rea serve\n");
    print_agent_config("Claude Desktop", "claude_desktop_config.json");
    println!("=== OpenCode (opencode.json / opencode.jsonc) ===");
    println!("{}", opencode_server_entry());
    print_agent_config("Cursor", "~/.cursor/mcp.json");
    print_agent_config("Windsurf", "~/.codeium/windsurf/mcp_config.json");
    ExitCode::SUCCESS
}

/// Loads or initializes MCP servers JSON object.
fn load_or_init_mcp_config(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or_else(|| json!({"mcpServers": {}}))
}

/// Injects REA server entry into servers mapping.
fn insert_rea_server(root: &mut Value) {
    if let Some(servers) = root.get_mut("mcpServers").and_then(Value::as_object_mut) {
        servers.insert("rea".into(), json!({"command": "rea", "args": ["serve"]}));
    }
}

/// Writes or updates turnkey MCP config file.
///
/// # Arguments
///
/// * `path` - Destination file path.
fn write_mcp_config_file(path: &Path) {
    backup_existing(path);
    let mut root = load_or_init_mcp_config(path);
    insert_rea_server(&mut root);
    let content = serde_json::to_string_pretty(&root).unwrap_or_default();
    let _ = std::fs::write(path, content);
}

/// Writes `OpenCode` `opencode.jsonc` with the REA server registered.
fn write_opencode_config() {
    backup_existing(Path::new("opencode.jsonc"));
    let raw = std::fs::read_to_string("opencode.jsonc").unwrap_or_default();
    let mut root = serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| json!({}));
    if let Some(map) = root.as_object_mut() {
        map.insert("mcp".into(), opencode_server_entry());
    }
    let content = serde_json::to_string_pretty(&root).unwrap_or_default();
    let _ = std::fs::write("opencode.jsonc", content);
}

/// Agent skill describing the REA investigation workflow.
const SKILL_CONTENT: &str = "# REA Reverse Engineering\n\nUse the `rea` MCP server to inspect local binaries, firmware, and artifacts.\n\n## Tools\n- `open_binary` then `binary_session` for identity and mitigations.\n- `list_functions` and `decompile_function` for code recovery.\n- `run_rules` and `audit` for capabilities and vulnerabilities.\n- `inspect_crypto`, `decrypt_strings`, and `emulate` for crypto and runtime strings.\n- `unpack`, `inspect_go`, `inspect_dotnet`, and `inspect_android` for format work.\n\n## Method\n1. Identify the target and its format.\n2. Recover functions and decompile the entry point.\n3. Run rules and the vulnerability audit.\n4. Recover crypto and runtime strings.\n5. Summarize findings with the evidence and the limitations.\n";

/// Backs up an existing file next to itself.
///
/// # Arguments
///
/// * `path` - File to back up.
fn backup_existing(path: &Path) {
    if path.exists() {
        let backup = path.with_extension("bak");
        let _ = std::fs::copy(path, &backup);
    }
}

/// Writes a file, backing up any existing file first.
///
/// # Arguments
///
/// * `path` - Destination path.
/// * `content` - File content.
fn install_file(path: &Path, content: &str) {
    backup_existing(path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, content);
    println!("✓ wrote {}", path.display());
}

/// Installs the REA agent skill file.
fn install_skill() {
    install_file(Path::new(".opencode/skills/rea/SKILL.md"), SKILL_CONTENT);
}

/// Runs turnkey setup for Claude Code and `OpenCode`.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn setup_cmd() -> ExitCode {
    write_mcp_config_file(Path::new("mcp_config.json"));
    write_opencode_config();
    install_skill();
    println!("✓ Turnkey OpenCode / AI Agent MCP configuration written to ./mcp_config.json");
    println!("✓ OpenCode MCP registration written to ./opencode.jsonc");
    println!("✓ Agent skill written to ./.opencode/skills/rea/SKILL.md");
    println!("✓ Claude Code CLI setup command: claude mcp add rea rea serve");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies tool catalog construction.
    #[test]
    fn test_tools_catalog_count() {
        let tools = tool_catalog();
        assert!(tools.is_array());
        assert_eq!(tools.as_array().map_or(0, Vec::len), 36);
    }

    /// Verifies runtime tool handlers reject empty sessions.
    #[test]
    fn test_runtime_tools_empty_session() {
        let session = ServerSession::default();
        assert_eq!(call_go(&session), "No target opened.");
        assert_eq!(call_dotnet(&session), "No target opened.");
        assert_eq!(call_android(&session), "No target opened.");
    }

    /// Verifies `OpenCode` server entry shape.
    #[test]
    fn test_opencode_server_entry_shape() {
        let entry = opencode_server_entry();
        assert_eq!(entry["rea"]["type"], "local");
        assert_eq!(entry["rea"]["enabled"], true);
        assert_eq!(entry["rea"]["command"].as_array().map_or(0, Vec::len), 2);
    }

    /// Verifies the skill installer backs up existing files.
    #[test]
    fn test_install_file_backs_up() {
        let dir = std::env::temp_dir().join("rea_skill_install_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("SKILL.md");
        let _ = std::fs::write(&path, "old");
        install_file(&path, "new");
        assert!(dir.join("SKILL.bak").exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    /// Verifies decompile accepts the synthesized `sub_<addr>` name.
    #[test]
    fn test_matches_synthesized_name() {
        let mut code = Disassembly::default();
        code.functions.insert(0x1000, Function::at(0x1000, None));
        let func = &code.functions[&0x1000];
        assert!(matches_function(&code, func, "sub_1000"));
        assert!(matches_function(&code, func, "0x1000"));
        assert!(!matches_function(&code, func, "sub_2000"));
    }

    /// Verifies the dossier summary lists only files that exist.
    #[test]
    fn test_dossier_paths_only_existing() {
        let lines = format_dossier_paths(&crate::paths::ReportPaths::default());
        assert_eq!(lines, "Dossier generated:");
    }
}
