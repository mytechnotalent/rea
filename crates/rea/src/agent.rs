//! Autonomous Reverse Engineering Analyst Loop
//!
//! **File:** `agent.rs`
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
use std::fmt::Write;

// Import dependencies from rea_engine
use rea_engine::{Analysis, FirmwareTriage};

// Import dependencies from serde_json
use serde_json::json;

// Import crate::tools
use crate::tools::{self, ToolContext};

/// System prompt for the autonomous tool-using analyst agent.
const AGENT_PROMPT: &str = "You are a principal reverse engineering analyst with tools to inspect \
the target. Investigate autonomously and never fabricate. To use a tool, reply with a single line \
'ACTION: <tool> <argument>', where <tool> is one of: functions; decompile <name|0xaddr>; \
strings <query>; emulate <0xaddr>; read <0xaddr> <len>; decrypt; rules; audit; crypto. You then \
receive an OBSERVATION. When you have the answer, reply with 'FINAL:' followed by an authoritative \
report with the sections \
Executive Overview, Architecture Analysis, Program Logic, Recovered Secrets (flags, keys, C2, \
credentials), and Recommendations. Ground every claim in observations and disassembly.";

/// Instruction appended to the evidence dossier describing the tool protocol.
const TOOL_NOTE: &str = "You may call tools to gather more evidence before answering.";

/// Maximum agent turns before stopping.
const MAX_TURNS: usize = 4;

/// Reads the API key from environment variables.
///
/// # Returns
///
/// Found key or `None`.
fn read_env_var() -> Option<String> {
    let key = std::env::var("OPENROUTER_API_KEY").ok()?;
    (!key.is_empty()).then_some(key)
}

/// Parses an API key definition line from `.env`.
///
/// # Arguments
///
/// * `line` - File line.
///
/// # Returns
///
/// Found key or `None`.
fn parse_env_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let val = trimmed.strip_prefix("OPENROUTER_API_KEY=")?;
    let clean = val.trim_matches('"').trim_matches('\'').trim();
    (!clean.is_empty()).then(|| clean.to_string())
}

/// Searches ancestor directories for `.env` contents.
///
/// # Arguments
///
/// * `dir` - Starting directory.
///
/// # Returns
///
/// File contents if found.
fn search_env(mut dir: std::path::PathBuf) -> Option<String> {
    loop {
        if let Ok(c) = std::fs::read_to_string(dir.join(".env")) {
            return Some(c);
        }
        dir.pop().then_some(())?;
    }
}

/// Reads the API key from a local `.env` file in current or parent dirs.
///
/// # Returns
///
/// Found key or `None`.
fn read_env_file() -> Option<String> {
    let dir = std::env::current_dir().ok()?;
    let content = search_env(dir)?;
    content.lines().find_map(parse_env_line)
}

/// Discovers the `OpenRouter` API key from environment or `.env`.
///
/// # Returns
///
/// The API key when available.
#[must_use]
pub fn load_api_key() -> Option<String> {
    read_env_var().or_else(read_env_file)
}

/// Resolves the model to request from `OpenRouter`.
///
/// # Returns
///
/// Model identifier string.
fn model_name() -> String {
    std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "nvidia/nemotron-3.5-lightning:free".to_string())
}

/// Constructs a chat completion JSON payload.
///
/// # Arguments
///
/// * `model` - Model identifier.
/// * `system` - System prompt.
/// * `prompt` - User prompt.
///
/// # Returns
///
/// Serialized request body.
fn build_payload(model: &str, system: &str, prompt: &str) -> serde_json::Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    })
}

/// Dispatches an HTTP POST request to `OpenRouter`.
///
/// # Arguments
///
/// * `key` - Bearer API key.
/// * `body` - Request JSON.
///
/// # Returns
///
/// Parsed response JSON if successful.
///
/// # Errors
///
/// Returns [`ureq::Error`] on network or deserialization failure.
fn send_request(key: &str, body: &serde_json::Value) -> Result<serde_json::Value, ureq::Error> {
    let mut res = ureq::post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", &format!("Bearer {key}"))
        .header("HTTP-Referer", "https://github.com/mytechnotalent/rea")
        .header("X-Title", "Reverse Engineering Assistant")
        .send_json(body)?;
    res.body_mut().read_json()
}

/// Extracts text content from a chat completion response.
///
/// # Arguments
///
/// * `val` - API response.
///
/// # Returns
///
/// Text content string.
fn extract_response(val: &serde_json::Value) -> Option<String> {
    let choice = val.get("choices")?.get(0)?;
    let content = choice.get("message")?.get("content")?.as_str()?;
    Some(content.to_string())
}

/// Executes a single chat completion turn.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `system` - System prompt.
/// * `prompt` - User prompt.
///
/// # Returns
///
/// Generated message content.
fn chat(key: &str, model: &str, system: &str, prompt: &str) -> Option<String> {
    let body = build_payload(model, system, prompt);
    let val = send_request(key, &body).ok()?;
    extract_response(&val)
}

/// Appends target identity facts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_identity(out: &mut String, a: &Analysis) {
    let (name, fmt, size, ent) = (&a.name, a.format.label(), a.size, a.entropy);
    let arch = a.binary.as_ref().map_or("Unknown", |b| &b.arch_name);
    let sha = &a.hashes.sha256;
    let _ = writeln!(
        out,
        "TARGET: {name}\nFORMAT: {fmt}\nARCH: {arch}\nSIZE: {size}\nENTROPY: {ent:.2}\nSHA256: {sha}"
    );
}

/// Appends capability facts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_capabilities(out: &mut String, a: &Analysis) {
    for c in &a.capabilities {
        let apis = c.apis.join(", ");
        let _ = writeln!(
            out,
            "CAPABILITY: {} ({}) APIs: {apis}",
            c.category, c.description
        );
    }
}

/// Appends firmware security artifacts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `fw` - Firmware triage details.
fn append_artifacts(out: &mut String, fw: &FirmwareTriage) {
    for art in &fw.artifacts {
        let _ = writeln!(
            out,
            "SECURITY ARTIFACT: {} [{}] - {}",
            art.path, art.kind, art.detail
        );
    }
}

/// Appends subtarget binary facts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `fw` - Firmware triage details.
fn append_subtargets(out: &mut String, fw: &FirmwareTriage) {
    for sub in &fw.sub_targets {
        let _ = writeln!(
            out,
            "SUBTARGET BINARY: {} [ELF] caps: {}",
            sub.name,
            sub.capabilities.len()
        );
    }
}

/// Appends firmware rootfs facts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_firmware(out: &mut String, a: &Analysis) {
    if let Some(fw) = &a.firmware {
        append_artifacts(out, fw);
        append_subtargets(out, fw);
    }
}

/// Appends binary facts to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_facts(out: &mut String, a: &Analysis) {
    if let Some(b) = &a.binary {
        for f in &b.facts {
            let _ = writeln!(out, "FACT: {} = {}", f.label, f.value);
        }
    }
}

/// Reports whether an extracted string is plausible text rather than opcode noise.
///
/// Appends extracted printable strings to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_strings_evidence(out: &mut String, a: &Analysis) {
    let valid = a.strings.iter().filter(|s| rea_engine::strings::is_plausible(&s.text));
    for s in valid.take(25) {
        let _ = writeln!(out, "STRING: offset={:#x} text={:?}", s.offset, s.text);
    }
}

/// Appends disassembly recovery header to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `code` - Disassembly summary.
fn append_code_header(out: &mut String, code: &rea_engine::disasm::Disassembly) {
    let (n_fn, ins, bit) = (code.functions.len(), code.instructions, code.bitness);
    let _ = writeln!(
        out,
        "CODE RECOVERY: {n_fn} functions, {ins} instructions recovered ({bit}-bit)."
    );
}

/// Appends disassembly and key function listings to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_code_evidence(out: &mut String, a: &Analysis) {
    let Some(code) = &a.disassembly else { return };
    append_code_header(out, code);
    for key in a.key_functions.iter().take(4) {
        append_key_func_listing(out, key);
    }
}

/// Appends one key function's listing to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `key` - Key function.
fn append_key_func_listing(out: &mut String, key: &rea_engine::insights::KeyFunction) {
    let _ = writeln!(out, "KEY FUNCTION: {} at {:#x}", key.name, key.address);
    for line in key.listing.iter().take(20) {
        let _ = writeln!(out, "  {line}");
    }
}

/// Appends deterministic behavioral narrative to evidence text.
/// Flattens structured story sections into a contiguous list of facts.
fn collect_story_facts(story: rea_engine::narrative::DetailedStory) -> Vec<String> {
    [
        story.boot,
        story.main_logic,
        story.optimization,
        story.patching,
    ]
    .concat()
}

/// Appends detailed deterministic story facts to evidence output.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_story_evidence(out: &mut String, a: &Analysis) {
    let facts = collect_story_facts(rea_engine::narrative::detailed_program_story(a));
    for p in facts {
        let _ = writeln!(out, "DETERMINISTIC FACT: {p}");
    }
}

/// Appends identity, facts, and capabilities to evidence text.
///
/// # Arguments
///
/// * `out` - Destination string.
/// * `a` - Completed analysis.
fn append_base_evidence(out: &mut String, a: &Analysis) {
    append_identity(out, a);
    append_facts(out, a);
    append_capabilities(out, a);
}

/// Compiles deterministic engine facts into an evidence dossier.
///
/// # Arguments
///
/// * `a` - Target analysis.
///
/// # Returns
///
/// Formatted evidence string.
fn build_evidence(a: &Analysis) -> String {
    let mut out = String::new();
    append_base_evidence(&mut out, a);
    append_strings_evidence(&mut out, a);
    append_code_evidence(&mut out, a);
    append_story_evidence(&mut out, a);
    append_firmware(&mut out, a);
    out
}

/// The agent's next step.
enum Action {
    /// Call a tool with an argument.
    Tool(String, String),

    /// Produce the final report.
    Final(String),

    /// No structured action; treat the reply as final.
    None,
}

/// Parses the agent's next action from a reply.
///
/// # Arguments
///
/// * `reply` - Model reply text.
///
/// # Returns
///
/// The parsed action.
fn parse_action(reply: &str) -> Action {
    for line in reply.lines() {
        if let Some(rest) = line.trim().strip_prefix("ACTION:") {
            return tool_action(rest.trim());
        }
    }
    reply.find("FINAL:").map_or(Action::None, |i| {
        Action::Final(reply[i + 6..].trim().to_string())
    })
}

/// Builds a tool action from the text after `ACTION:`.
///
/// # Arguments
///
/// * `rest` - Tool name and argument text.
///
/// # Returns
///
/// The tool action.
fn tool_action(rest: &str) -> Action {
    let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    Action::Tool(name.to_string(), args.trim().to_string())
}

/// Executes one agent turn, updating the scratchpad.
///
/// # Arguments
///
/// * `ctx` - Tool context.
/// * `reply` - Model reply text.
/// * `scratch` - Growing conversation scratchpad.
///
/// # Returns
///
/// The final report when produced, otherwise `None`.
fn step(ctx: &ToolContext<'_>, reply: &str, scratch: &mut String) -> Option<String> {
    match parse_action(reply) {
        Action::Final(text) => Some(text),
        Action::Tool(name, args) => {
            append_observation(scratch, reply, &tools::run(ctx, &name, &args));
            None
        }
        Action::None => Some(reply.to_string()),
    }
}

/// Appends the reply and a tool observation to the scratchpad.
///
/// # Arguments
///
/// * `scratch` - Growing conversation scratchpad.
/// * `reply` - Model reply text.
/// * `observation` - Tool result text.
fn append_observation(scratch: &mut String, reply: &str, observation: &str) {
    let _ = write!(scratch, "\n\n{reply}\nOBSERVATION:\n{observation}");
}

/// Runs the tool-calling agent loop.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `ctx` - Tool context.
/// * `evidence` - Evidence dossier.
///
/// # Returns
///
/// The final report, or `None`.
fn run_loop(key: &str, model: &str, ctx: &ToolContext<'_>, evidence: &str) -> Option<String> {
    let mut scratch = format!("{evidence}\n\n{TOOL_NOTE}");
    for _ in 0..MAX_TURNS {
        let reply = chat(key, model, AGENT_PROMPT, &scratch)?;
        if let Some(text) = step(ctx, &reply, &mut scratch) {
            return Some(text);
        }
    }
    None
}

/// Runs the autonomous tool-calling analyst loop when configured.
///
/// # Arguments
///
/// * `analysis` - Completed engine analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// Final narrative assessment if API key is present and calls succeed.
#[must_use]
pub fn evaluate(analysis: &Analysis, bytes: &[u8]) -> Option<String> {
    let key = load_api_key()?;
    let model = model_name();
    eprintln!("rea: running autonomous AI analyst loop ({model})");
    let evidence = build_evidence(analysis);
    let ctx = ToolContext { analysis, bytes };
    run_loop(&key, &model, &ctx, &evidence)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies parsing plain key lines.
    #[test]
    fn test_parse_env_line_plain() {
        let expected = Some("sk-123".to_string());
        assert_eq!(parse_env_line("OPENROUTER_API_KEY=sk-123"), expected);
        assert_eq!(parse_env_line("OTHER=val"), None);
    }

    /// Verifies parsing quoted key lines.
    #[test]
    fn test_parse_env_line_quoted() {
        let expected = Some("sk-456".to_string());
        assert_eq!(parse_env_line("OPENROUTER_API_KEY=\"sk-456\""), expected);
    }

    /// Verifies model fallback default.
    #[test]
    fn test_model_name_default() {
        assert!(!model_name().is_empty());
    }

    /// Verifies evidence assembly includes identity facts.
    #[test]
    fn test_build_evidence_contains_target() {
        let analysis = rea_engine::analyze("demo.bin", b"test data");
        let evidence = build_evidence(&analysis);
        assert!(evidence.contains("TARGET: demo.bin"));
    }
}
