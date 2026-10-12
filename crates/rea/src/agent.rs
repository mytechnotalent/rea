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

/// System prompt for the analyst agent proposing hypotheses.
const ANALYST_PROMPT: &str = "You are a principal binary reverse engineering analyst. \
Given deterministic facts, memory maps, vector tables, strings, and recovered disassembly listings, \
conduct an exhaustive analysis: \
(1) Identify the exact microcontroller / SoC platform (e.g., Raspberry Pi RP2040/RP2350, STM32, ESP32, nRF52, etc.) \
and hardware architecture from memory ranges, vector tables, and MMIO peripheral access. \
(2) Trace program execution: boot sequence, silicon/clock init, peripheral configuration (GPIO, UART, SIO, Timers), \
and core application control loops (strictly ground your main function in the identified main loop candidate with formatted prints). \
(3) Explain variable storage (.rodata, .data, .bss, registers) and compiler optimizations (dead-store elimination, \
constant folding/propagation where immediate operands replace stack variables in registers). \
(4) Detail binary patching & reverse engineering vectors: exact instruction, file offset, patch values, and hardware flashing procedures. \
Do not hallucinate meaning for accidental opcode byte sequences (e.g., FKFAF, BFKFQF, RFKFAF). Only analyze authentic application strings.";

/// System prompt for the reviewer agent auditing proposals.
const REVIEWER_PROMPT: &str = "You are a strict technical peer reviewer. \
Verify the Analyst's hypotheses against the deterministic evidence. Eliminate unsupported claims, \
verify microcontroller identification against the vector table and memory map, and ensure disassembly citations are accurate.";

/// System prompt for the adjudicator agent synthesizing final output.
const ADJUDICATOR_PROMPT: &str = "You are the lead security adjudicator. \
Synthesize findings into an authoritative reverse engineering report matching professional malware/embedded research with markdown sections: \
### Executive Overview\n\
### Architecture & Microcontroller Analysis\n\
### Program Logic & Control Flow\n\
### Variable Storage & Compiler Optimization\n\
### Binary Patching & Exploitation Vectors\n\
### Security Recommendations\n\
Do not duplicate heading markers. Ground every finding in the provided DETERMINISTIC FACT and disassembly listings. Identify the exact microcontroller model and hardware peripherals from the evidence. \
Never cite or hallucinate about opcode byte noise fragments (such as FKFAF, BFKFQF, RFKFAF). Only reference genuine application strings.";

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

/// Stage 1: Analyst proposes hypotheses grounded in facts.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `evidence` - Evidence dossier.
///
/// # Returns
///
/// Analyst proposal text.
fn propose(key: &str, model: &str, evidence: &str) -> Option<String> {
    chat(key, model, ANALYST_PROMPT, evidence)
}

/// Stage 2: Reviewer audits hypotheses against evidence.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `evidence` - Evidence dossier.
/// * `draft` - Analyst proposal.
///
/// # Returns
///
/// Review critique text.
fn review(key: &str, model: &str, evidence: &str, draft: &str) -> Option<String> {
    let prompt = format!("EVIDENCE:\n{evidence}\n\nANALYST PROPOSAL:\n{draft}");
    chat(key, model, REVIEWER_PROMPT, &prompt)
}

/// Stage 3: Adjudicator produces final authoritative narrative.
///
/// # Arguments
///
/// * `key` - API key.
/// * `model` - Model identifier.
/// * `evidence` - Evidence dossier.
/// * `draft` - Analyst proposal.
/// * `critique` - Reviewer audit.
///
/// # Returns
///
/// Final synthesized assessment narrative.
fn adjudicate(
    key: &str,
    model: &str,
    evidence: &str,
    draft: &str,
    critique: &str,
) -> Option<String> {
    let prompt = format!("EVIDENCE:\n{evidence}\n\nANALYST DRAFT:\n{draft}\n\nAUDIT:\n{critique}");
    chat(key, model, ADJUDICATOR_PROMPT, &prompt)
}

/// Runs the autonomous propose-review-adjudicate loop when configured.
///
/// # Arguments
///
/// * `analysis` - Completed engine analysis.
///
/// # Returns
///
/// Final narrative assessment if API key is present and calls succeed.
#[must_use]
pub fn evaluate(analysis: &Analysis) -> Option<String> {
    let key = load_api_key()?;
    let model = model_name();
    eprintln!("rea: running autonomous AI analyst loop ({model})");
    let evidence = build_evidence(analysis);
    let draft = propose(&key, &model, &evidence)?;
    let audit = review(&key, &model, &evidence, &draft)?;
    adjudicate(&key, &model, &evidence, &draft, &audit)
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
