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
use std::{
    fmt::Write,
    time::{Duration, Instant},
};

// Import dependencies from rea_engine
use rea_engine::{Analysis, FirmwareTriage};

// Import dependencies from serde_json
use serde_json::json;

// Import crate::tools
use crate::tools::{self, ToolContext};

/// Default model used when `OPENROUTER_MODEL` is unset.
const DEFAULT_MODEL: &str = "deepseek/deepseek-v4.1-flash";

/// System prompt for the autonomous tool-using analyst agent.
const AGENT_PROMPT: &str = "You are a principal reverse engineering analyst and the AUTHOR of the \
final security assessment report. You have tools to inspect the target. Investigate autonomously \
and ground every claim in tool observations and the provided evidence; never fabricate strings, \
offsets, keys, or behavior. Call tools as often as needed, then write the complete report \
yourself. Your report MUST contain the sections: Executive Overview; Target Summary; Architecture \
Analysis; Program Logic; Recovered Secrets (flags, keys, C2, credentials); Indicators of \
Compromise; and Recommendations. Cite concrete addresses and strings from the evidence. When a \
fact is unknown or encrypted, state that plainly rather than guessing.";

/// Instruction appended to the evidence dossier describing the model's role.
const TOOL_NOTE: &str =
    "You are the report's author. Use the tools to gather additional evidence, \
then produce the full narrative assessment. The evidence dossier is source material, not the \
report: expand it into an authoritative write-up.";

/// Maximum agent turns before stopping.
const MAX_TURNS: usize = 90;

/// Tool specifications exposed to the model: name and description.
const TOOL_SPECS: [(&str, &str); 11] = [
    ("functions", "List recovered and key functions."),
    ("decompile", "Decompile a function by name or hex address."),
    ("strings", "Search recovered strings for a substring."),
    ("read", "Hex dump bytes at an address: '0xADDR LEN'."),
    (
        "emulate",
        "Emulate from a hex address and report recovered flags.",
    ),
    (
        "unpack",
        "Emulate with stubbed imports to decrypt protected regions.",
    ),
    ("decrypt", "Recover strings hidden by simple byte ciphers."),
    (
        "rules",
        "Evaluate capability rules (ATT&CK and MBC mapping).",
    ),
    (
        "audit",
        "Audit for vulnerable API sinks with CWE identifiers.",
    ),
    (
        "crypto",
        "Detect cryptographic constants such as AES, SHA, CRC32.",
    ),
    (
        "regions",
        "Segment the target by entropy and classify each region.",
    ),
];

/// Minimum quotation length considered by the grounding validator.
const GROUNDING_MIN: usize = 6;

/// Maximum unverified quotations listed in the grounding note.
const GROUNDING_LIMIT: usize = 8;

/// Wall-clock budget for the analyst loop before forcing a final answer.
const AI_BUDGET: Duration = Duration::from_secs(420);

/// Per-request network timeout for `OpenRouter` calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// `OpenRouter` chat completions endpoint.
const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Maximum characters of one tool observation fed back into the conversation.
const OBS_LIMIT: usize = 4000;

/// Instruction that forces the model to stop investigating and write the report.
const FINAL_NUDGE: &str = "Stop investigating now and write the complete final report using the \
evidence gathered so far. Do not call any more tools.";

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
    std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Constructs a tool-enabled chat completion JSON payload.
///
/// # Arguments
///
/// * `model` - Model identifier.
/// * `messages` - Conversation message array.
///
/// # Returns
///
/// Serialized request body.
fn build_payload(model: &str, messages: &serde_json::Value) -> serde_json::Value {
    json!({
        "model": model,
        "messages": messages,
        "tools": tool_defs(),
        "tool_choice": "auto",
        "temperature": 0.2
    })
}

/// Builds the JSON tool definition array from [`TOOL_SPECS`].
///
/// # Returns
///
/// Array of function tool definitions.
fn tool_defs() -> serde_json::Value {
    let list = TOOL_SPECS.iter().map(|(name, desc)| tool_def(name, desc)).collect();
    serde_json::Value::Array(list)
}

/// Builds one function tool definition.
///
/// # Arguments
///
/// * `name` - Tool name.
/// * `description` - Tool description.
///
/// # Returns
///
/// A single function tool definition.
fn tool_def(name: &str, description: &str) -> serde_json::Value {
    let function = json!({
        "name": name,
        "description": description,
        "parameters": tool_params()
    });
    json!({"type": "function", "function": function})
}

/// Builds the shared single-argument tool parameter schema.
///
/// # Returns
///
/// An object schema with one optional `arg` string.
fn tool_params() -> serde_json::Value {
    json!({"type": "object", "properties": {"arg": {"type": "string"}}})
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
    let mut res = post_json(key, body)?;
    res.body_mut().read_json()
}

/// Sends a JSON POST to `OpenRouter` with a bounded timeout.
///
/// # Arguments
///
/// * `key` - Bearer API key.
/// * `body` - Request JSON.
///
/// # Returns
///
/// The HTTP response.
///
/// # Errors
///
/// Returns [`ureq::Error`] on network failure.
fn post_json(
    key: &str,
    body: &serde_json::Value,
) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
    ureq::post(ENDPOINT)
        .config()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .build()
        .header("Authorization", &format!("Bearer {key}"))
        .header("HTTP-Referer", "https://github.com/mytechnotalent/rea")
        .header("X-Title", "Reverse Engineering Assistant")
        .send_json(body)
}

/// Extracts the assistant message from a chat completion response.
///
/// # Arguments
///
/// * `val` - API response.
///
/// # Returns
///
/// The assistant message object.
fn extract_message(val: &serde_json::Value) -> Option<serde_json::Value> {
    val.get("choices")?.get(0)?.get("message").cloned()
}

/// Executes a single chat completion turn.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `messages` - Conversation message array.
///
/// # Returns
///
/// The assistant message object.
fn chat(key: &str, model: &str, messages: &serde_json::Value) -> Option<serde_json::Value> {
    let body = build_payload(model, messages);
    let val = send_request(key, &body).ok()?;
    extract_message(&val)
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

/// Builds the initial system and user conversation messages.
///
/// # Arguments
///
/// * `evidence` - Evidence dossier text.
///
/// # Returns
///
/// The initial message array.
fn initial_messages(evidence: &str) -> serde_json::Value {
    let user = format!("{evidence}\n\n{TOOL_NOTE}");
    json!([
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user}
    ])
}

/// Reads a string field from a JSON value.
///
/// # Arguments
///
/// * `value` - Source JSON value.
/// * `key` - Field name.
///
/// # Returns
///
/// The field text, or an empty string.
fn text_field(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Returns the tool calls carried by an assistant message.
///
/// # Arguments
///
/// * `message` - Assistant message object.
///
/// # Returns
///
/// The tool call objects, possibly empty.
fn calls_of(message: &serde_json::Value) -> Vec<serde_json::Value> {
    let calls = message.get("tool_calls").and_then(serde_json::Value::as_array);
    calls.cloned().unwrap_or_default()
}

/// Extracts the identifier, name, and argument from a tool call.
///
/// # Arguments
///
/// * `call` - Tool call object.
///
/// # Returns
///
/// A `(id, name, argument)` tuple.
fn parse_call(call: &serde_json::Value) -> (String, String, String) {
    let function = call.get("function").cloned().unwrap_or_default();
    let id = text_field(call, "id");
    let name = text_field(&function, "name");
    (id, name, tool_arg(&function))
}

/// Extracts the `arg` parameter from a tool call function object.
///
/// # Arguments
///
/// * `function` - Tool call function object.
///
/// # Returns
///
/// The argument text, or an empty string.
fn tool_arg(function: &serde_json::Value) -> String {
    let args = text_field(function, "arguments");
    let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
    text_field(&parsed, "arg")
}

/// Appends one tool result to the conversation.
///
/// # Arguments
///
/// * `messages` - Conversation message array.
/// * `id` - Tool call identifier.
/// * `content` - Tool result text.
fn append_tool(messages: &mut serde_json::Value, id: &str, content: &str) {
    if let Some(array) = messages.as_array_mut() {
        array.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
    }
}

/// Conversation state for one analyst run.
struct Session {
    /// Conversation message array.
    messages: serde_json::Value,

    /// Evidence corpus for grounding.
    seen: String,
}

impl Session {
    /// Creates a session seeded with the evidence dossier.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence dossier text.
    ///
    /// # Returns
    ///
    /// A new session.
    fn new(evidence: &str) -> Self {
        Self { messages: initial_messages(evidence), seen: evidence.to_string() }
    }

    /// Advances one turn, executing tools or returning the final report.
    ///
    /// # Arguments
    ///
    /// * `message` - Assistant message object.
    /// * `ctx` - Tool context.
    ///
    /// # Returns
    ///
    /// The grounded final report, or `None` to continue.
    fn finish(&mut self, message: &serde_json::Value, ctx: &ToolContext<'_>) -> Option<String> {
        let calls = calls_of(message);
        self.messages.as_array_mut()?.push(message.clone());
        if calls.is_empty() {
            return final_report(message, &self.seen);
        }
        self.run_calls(&calls, ctx);
        None
    }

    /// Runs the loop until a report is produced, the budget expires, or turns run out.
    ///
    /// # Arguments
    ///
    /// * `key` - API key.
    /// * `model` - Model identifier.
    /// * `ctx` - Tool context.
    /// * `deadline` - Wall-clock budget deadline.
    ///
    /// # Returns
    ///
    /// The final report, or `None` when the budget or turn limit is reached.
    fn run(
        &mut self,
        key: &str,
        model: &str,
        ctx: &ToolContext<'_>,
        deadline: Instant,
    ) -> Option<String> {
        let mut turn = 0;
        while !expired(deadline) && turn < MAX_TURNS {
            turn += 1;
            if let Some(text) = self.turn(turn, key, model, ctx) {
                return Some(text);
            }
        }
        None
    }

    /// Runs one chat turn and folds its result into the session.
    ///
    /// # Arguments
    ///
    /// * `turn` - One-based turn number.
    /// * `key` - API key.
    /// * `model` - Model identifier.
    /// * `ctx` - Tool context.
    ///
    /// # Returns
    ///
    /// The final report when the model stops calling tools.
    fn turn(
        &mut self,
        turn: usize,
        key: &str,
        model: &str,
        ctx: &ToolContext<'_>,
    ) -> Option<String> {
        eprintln!("rea: agent turn {turn}");
        let message = chat(key, model, &self.messages)?;
        self.finish(&message, ctx)
    }

    /// Forces a final report after the loop stops without one.
    ///
    /// # Arguments
    ///
    /// * `key` - API key.
    /// * `model` - Model identifier.
    ///
    /// # Returns
    ///
    /// The grounded report, or `None`.
    fn finalize(&mut self, key: &str, model: &str) -> Option<String> {
        push_user(&mut self.messages, FINAL_NUDGE);
        let message = chat_none(key, model, &self.messages)?;
        final_report(&message, &self.seen)
    }

    /// Executes tool calls, folding observations into the corpus.
    ///
    /// # Arguments
    ///
    /// * `calls` - Tool call objects.
    /// * `ctx` - Tool context.
    fn run_calls(&mut self, calls: &[serde_json::Value], ctx: &ToolContext<'_>) {
        for call in calls {
            let (id, name, arg) = parse_call(call);
            eprintln!("rea: tool {name} {arg}");
            let observation = tools::run(ctx, &name, &arg);
            self.seen.push_str(&observation);
            append_tool(&mut self.messages, &id, &clip(&observation));
        }
    }
}

/// Reports whether a wall-clock deadline has passed.
///
/// # Arguments
///
/// * `deadline` - Wall-clock deadline.
///
/// # Returns
///
/// `true` when the deadline has passed.
fn expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

/// Truncates a tool observation to [`OBS_LIMIT`] characters.
///
/// # Arguments
///
/// * `text` - Observation text.
///
/// # Returns
///
/// The bounded text.
fn clip(text: &str) -> String {
    text.chars().take(OBS_LIMIT).collect()
}

/// Appends a user message to the conversation.
///
/// # Arguments
///
/// * `messages` - Conversation message array.
/// * `content` - User message text.
fn push_user(messages: &mut serde_json::Value, content: &str) {
    if let Some(array) = messages.as_array_mut() {
        array.push(json!({"role": "user", "content": content}));
    }
}

/// Executes a chat turn that forbids further tool calls.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `messages` - Conversation message array.
///
/// # Returns
///
/// The assistant message object.
fn chat_none(key: &str, model: &str, messages: &serde_json::Value) -> Option<serde_json::Value> {
    let mut body = build_payload(model, messages);
    body["tool_choice"] = json!("none");
    send_request(key, &body).ok().and_then(|val| extract_message(&val))
}

/// Produces the final grounded report from an assistant message.
///
/// # Arguments
///
/// * `message` - Assistant message object.
/// * `seen` - Evidence dossier plus tool observations.
///
/// # Returns
///
/// The grounded report, or `None` when the reply is empty.
fn final_report(message: &serde_json::Value, seen: &str) -> Option<String> {
    let content = text_field(message, "content");
    (!content.trim().is_empty()).then(|| ground(&content, seen))
}

/// Flags report quotations that are absent from the evidence corpus.
///
/// # Arguments
///
/// * `text` - Report text.
/// * `seen` - Evidence dossier plus tool observations.
///
/// # Returns
///
/// The report, annotated when unverified quotations are found.
fn ground(text: &str, seen: &str) -> String {
    let bad = unverified(text, seen);
    if bad.is_empty() {
        return text.to_string();
    }
    let note = format!("{} unverified quotation(s): {}", bad.len(), sample(&bad));
    format!("{text}\n\n[Grounding] {note}")
}

/// Selects report quotations absent from the evidence corpus.
///
/// # Arguments
///
/// * `text` - Report text.
/// * `seen` - Evidence dossier plus tool observations.
///
/// # Returns
///
/// The unverified quotations.
fn unverified(text: &str, seen: &str) -> Vec<String> {
    let kept = |q: &String| q.len() >= GROUNDING_MIN && !seen.contains(q.as_str());
    quotes(text).into_iter().filter(kept).collect()
}

/// Builds a bounded comma-separated sample of strings.
///
/// # Arguments
///
/// * `items` - Source strings.
///
/// # Returns
///
/// The sampled, joined text.
fn sample(items: &[String]) -> String {
    items.iter().take(GROUNDING_LIMIT).cloned().collect::<Vec<_>>().join(", ")
}

/// Collects double-quoted tokens from report text.
///
/// # Arguments
///
/// * `text` - Report text.
///
/// # Returns
///
/// The trimmed quoted tokens.
fn quotes(text: &str) -> Vec<String> {
    let clean = |part: &str| part.trim().trim_end_matches([',', '.', ';', ':']).to_string();
    text.split('"').skip(1).step_by(2).map(clean).collect()
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
    let mut session = Session::new(evidence);
    let deadline = Instant::now() + ai_budget();
    let draft = session.run(key, model, ctx, deadline);
    draft.or_else(|| session.finalize(key, model))
}

/// Resolves the analyst wall-clock budget from `REA_AGENT_BUDGET_SECS`.
///
/// # Returns
///
/// The configured budget, defaulting to [`AI_BUDGET`].
fn ai_budget() -> Duration {
    let secs = std::env::var("REA_AGENT_BUDGET_SECS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(secs.unwrap_or(AI_BUDGET.as_secs()))
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

    /// Verifies tool argument extraction from a function object.
    #[test]
    fn test_tool_arg_parses() {
        let function = json!({"name": "decompile", "arguments": "{\"arg\":\"0x40\"}"});
        assert_eq!(tool_arg(&function), "0x40");
    }

    /// Verifies grounding flags quotations absent from evidence.
    #[test]
    fn test_ground_flags_unverified() {
        let text = "Recovered flag \"ABC_FAKE_FLAG\" from data.";
        assert!(ground(text, "nothing here").contains("unverified"));
    }

    /// Verifies grounding accepts quotations present in evidence.
    #[test]
    fn test_ground_accepts_evidence() {
        let text = "Recovered flag \"REAL_FLAG\" from data.";
        assert!(!ground(text, "the REAL_FLAG lives here").contains("unverified"));
    }

    /// Verifies tool definitions expose one schema per spec.
    #[test]
    fn test_tool_defs_shape() {
        let defs = tool_defs();
        assert_eq!(defs.as_array().unwrap().len(), TOOL_SPECS.len());
    }
}
