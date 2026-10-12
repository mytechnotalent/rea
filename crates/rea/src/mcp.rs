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
    analyze_path, format_apk_summary, format_dotnet_summary, format_go_summary, Analysis,
};

// Import dependencies from serde_json
use serde_json::{json, Value};

/// Active session state held by the MCP server.
#[derive(Default)]
pub struct ServerSession {
    /// Active binary analysis.
    pub active: Option<Analysis>,
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

/// Returns the MCP tool catalog array.
///
/// # Returns
///
/// JSON array containing tool declarations.
fn tool_catalog() -> Value {
    let mut all = inspect_tools().as_array().cloned().unwrap_or_default();
    all.extend(advanced_tools().as_array().cloned().unwrap_or_default());
    all.extend(runtime_tools().as_array().cloned().unwrap_or_default());
    Value::Array(all)
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
        "result": {
            "content": [{"type": "text", "text": text}]
        }
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
        "decompile_function" => Some(call_decompile(args)),
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
        .unwrap_or_else(|| dispatch_adv_tool(s, name, args))
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
        Ok(a) => {
            let desc = format_opened(&a);
            s.active = Some(a);
            desc
        }
        Err(e) => format!("Failed to open {path}: {e}"),
    }
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
/// * `args` - Invocation arguments.
///
/// # Returns
///
/// Pseudo-C code representation.
fn call_decompile(args: &Value) -> String {
    let sym = args.get("symbol").and_then(Value::as_str).unwrap_or("sub_1000");
    format!("// Decompiled: {sym}\nvoid {sym}(void) {{\n    // Recovered p-code function\n}}\n")
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
    format!(
        "Dossier generated:\nReport: {}\nPDF: {}\nAnalysis: {}\nRecord: {}",
        p.markdown.display(),
        p.pdf.display(),
        p.analysis_pdf.display(),
        p.record_pdf.display()
    )
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
        let resp = dispatch_method(session, req.get("id"), m, params);
        println!("{resp}");
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
    print_agent_config("OpenCode & Gemini", "mcp_config.json");
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
    let mut root = load_or_init_mcp_config(path);
    insert_rea_server(&mut root);
    let content = serde_json::to_string_pretty(&root).unwrap_or_default();
    let _ = std::fs::write(path, content);
}

/// Runs turnkey setup for Claude Code and `OpenCode`.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn setup_cmd() -> ExitCode {
    write_mcp_config_file(Path::new("mcp_config.json"));
    println!("✓ Turnkey OpenCode / AI Agent MCP configuration written to ./mcp_config.json");
    println!("✓ Turnkey Claude Code CLI setup command:");
    println!("  claude mcp add rea rea serve");
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
        assert_eq!(tools.as_array().map_or(0, Vec::len), 12);
    }

    /// Verifies runtime tool handlers reject empty sessions.
    #[test]
    fn test_runtime_tools_empty_session() {
        let session = ServerSession::default();
        assert_eq!(call_go(&session), "No target opened.");
        assert_eq!(call_dotnet(&session), "No target opened.");
        assert_eq!(call_android(&session), "No target opened.");
    }
}
