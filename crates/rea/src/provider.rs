//! Binary Ninja MCP Provider
//!
//! **File:** `provider.rs`
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
use std::{fs, path::Path};

// Import dependencies from serde_json
use serde_json::{json, Value};

/// Binary Ninja MCP endpoint.
const ENDPOINT: &str = "http://127.0.0.1:24642/mcp";

/// MCP protocol version requested.
const PROTOCOL: &str = "2025-11-25";

/// Tool name for the view triage summary.
const TRIAGE: &str = "bn_binary_view_triage";

/// Tool name for the function list.
const LIST: &str = "bn_function_list";

/// Tool name for function decompilation.
const DECOMPILE: &str = "bn_function_decompile";

/// An HTTP response carrying a JSON-RPC result.
type Reply = ureq::http::Response<ureq::Body>;

/// A connected external analysis provider.
pub struct Provider {
    /// HTTP agent.
    agent: ureq::Agent,
    /// Session id returned by the server.
    session: Option<String>,
    /// Request counter.
    id: u64,
}

impl Provider {
    /// Connects to the provider, or returns `None` when unavailable.
    ///
    /// # Returns
    ///
    /// A connected provider, or `None`.
    #[must_use]
    pub fn connect() -> Option<Provider> {
        let mut provider = Provider { agent: ureq::agent(), session: None, id: 0 };
        let params = json!({"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "rea", "version": "0"}});
        provider.request("initialize", &params)?;
        provider.notify("notifications/initialized")?;
        Some(provider)
    }

    /// Calls a tool and returns its text result.
    ///
    /// # Arguments
    ///
    /// * `tool` - Tool name.
    /// * `args` - Tool arguments.
    ///
    /// # Returns
    ///
    /// The tool result text, or `None`.
    pub fn call(&mut self, tool: &str, args: &Value) -> Option<String> {
        let params = json!({"name": tool, "arguments": args});
        let result = self.request("tools/call", &params)?;
        Some(extract_text(&result))
    }

    /// Posts a JSON-RPC request and returns its result.
    ///
    /// # Arguments
    ///
    /// * `method` - JSON-RPC method.
    /// * `params` - Method parameters.
    ///
    /// # Returns
    ///
    /// The result value, or `None`.
    fn request(&mut self, method: &str, params: &Value) -> Option<Value> {
        self.id += 1;
        let payload = json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params});
        self.post(&payload)?.get("result").cloned()
    }

    /// Posts a JSON-RPC notification.
    ///
    /// # Arguments
    ///
    /// * `method` - JSON-RPC method.
    ///
    /// # Returns
    ///
    /// `Some` when the notification was sent.
    fn notify(&mut self, method: &str) -> Option<()> {
        let payload = json!({"jsonrpc": "2.0", "method": method});
        self.send(&payload)?;
        Some(())
    }

    /// Posts a payload and parses the JSON response.
    ///
    /// # Arguments
    ///
    /// * `payload` - JSON-RPC payload.
    ///
    /// # Returns
    ///
    /// The parsed response, or `None`.
    fn post(&mut self, payload: &Value) -> Option<Value> {
        let response = self.send(payload)?;
        let text = response.into_body().read_to_string().ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Sends a payload, remembering any returned session id.
    ///
    /// # Arguments
    ///
    /// * `payload` - JSON-RPC payload.
    ///
    /// # Returns
    ///
    /// The HTTP response, or `None`.
    fn send(&mut self, payload: &Value) -> Option<Reply> {
        let session = self.session.clone();
        let request = self.build(session.as_deref());
        let response = request.send_json(payload.clone()).ok()?;
        self.remember(&response);
        Some(response)
    }

    /// Builds an authorized request.
    ///
    /// # Arguments
    ///
    /// * `session` - Optional session id header.
    ///
    /// # Returns
    ///
    /// The configured request.
    fn build(&self, session: Option<&str>) -> ureq::RequestBuilder<ureq::typestate::WithBody> {
        let mut request = self.agent.post(ENDPOINT).header("Content-Type", "application/json");
        request = request.header("Accept", "application/json, text/event-stream");
        match session {
            Some(id) => request.header("MCP-Session-Id", id),
            None => request,
        }
    }

    /// Stores the session id from a response.
    ///
    /// # Arguments
    ///
    /// * `response` - HTTP response.
    fn remember(&mut self, response: &Reply) {
        if let Some(value) = response.headers().get("mcp-session-id").and_then(|v| v.to_str().ok())
        {
            self.session = Some(value.to_string());
        }
    }
}

/// Extracts text from an MCP result value.
///
/// # Arguments
///
/// * `result` - Tool result value.
///
/// # Returns
///
/// The concatenated content text or the serialized value.
fn extract_text(result: &Value) -> String {
    match result.get("content").and_then(Value::as_array) {
        Some(items) => items.iter().filter_map(|i| i.get("text").and_then(Value::as_str)).collect(),
        None => result.to_string(),
    }
}

/// Analyzes a target through the provider, wrapping raw images when needed.
///
/// # Arguments
///
/// * `target` - Target file path.
///
/// # Returns
///
/// The provider findings, or `None` when no provider is available.
#[must_use]
pub fn analyze(target: &Path) -> Option<String> {
    let bytes = fs::read(target).ok()?;
    let path = prepare(&bytes, target)?;
    let mut provider = Provider::connect()?;
    let open = json!({"path": path, "kind": "file", "setActive": true});
    provider.call("bn_open_item_open", &open);
    provider.call("bn_analysis_update_and_wait", &json!({}));
    Some(collect(&mut provider))
}

/// Collects the provider's findings into one text block.
///
/// # Arguments
///
/// * `provider` - Connected provider.
///
/// # Returns
///
/// The concatenated findings.
fn collect(provider: &mut Provider) -> String {
    let mut out = String::new();
    let decompile = json!({"function": "0x40", "limit": 180});
    append(&mut out, provider, TRIAGE, &json!({}));
    append(&mut out, provider, LIST, &json!({"limit": 25}));
    append(&mut out, provider, DECOMPILE, &decompile);
    out
}

/// Appends one provider tool result.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `provider` - Connected provider.
/// * `tool` - Tool name.
/// * `args` - Tool arguments.
fn append(out: &mut String, provider: &mut Provider, tool: &str, args: &Value) {
    if let Some(text) = provider.call(tool, args) {
        out.push_str(&text);
        out.push('\n');
    }
}

/// Resolves the path to hand a provider, wrapping raw images as ELF.
///
/// # Arguments
///
/// * `bytes` - Target bytes.
/// * `target` - Target path.
///
/// # Returns
///
/// The path to analyze, or `None`.
fn prepare(bytes: &[u8], target: &Path) -> Option<String> {
    if rea_engine::detect::detect(bytes).is_native() {
        return target.to_str().map(str::to_string);
    }
    wrap_path(bytes)
}

/// Writes a wrapped ELF for a raw image and returns its path.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
///
/// # Returns
///
/// The wrapped ELF path, or `None`.
fn wrap_path(bytes: &[u8]) -> Option<String> {
    let entry = u32::try_from(rea_engine::detect::arm_entry(bytes).unwrap_or(0)).unwrap_or(0);
    let wrapped = rea_engine::elfwrap::wrap_arm_at(bytes, load_base(bytes), entry);
    let path = std::env::temp_dir().join("rea-provider.elf");
    fs::write(&path, &wrapped.bytes).ok()?;
    path.to_str().map(str::to_string)
}

/// Resolves the raw-image load base from the environment or container header.
///
/// `REA_LOAD_BASE` overrides container detection when set to a hex value.
///
/// # Arguments
///
/// * `bytes` - Raw firmware bytes.
///
/// # Returns
///
/// The load base address, defaulting to zero.
fn load_base(bytes: &[u8]) -> u32 {
    std::env::var("REA_LOAD_BASE")
        .ok()
        .and_then(|text| parse_hex(&text))
        .unwrap_or_else(|| {
            rea_engine::container::parse(bytes).and_then(|found| found.base).unwrap_or(0)
        })
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
fn parse_hex(text: &str) -> Option<u32> {
    let body = text.trim().trim_start_matches("0x").trim_start_matches("0X");
    u32::from_str_radix(body, 16).ok()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies content-text extraction from an MCP result.
    #[test]
    fn test_extract_text_content() {
        let result = json!({"content": [{"type": "text", "text": "hello"}, {"text": " world"}]});
        assert_eq!(extract_text(&result), "hello world");
    }

    /// Verifies fallback serialization of a non-content result.
    #[test]
    fn test_extract_text_fallback() {
        let result = json!({"openItem": {"handle": "item_1"}});
        assert!(extract_text(&result).contains("item_1"));
    }

    /// Verifies that native images are passed through unwrapped.
    #[test]
    fn test_prepare_native() {
        let elf = b"\x7fELF\x02\x01\x01";
        let path = prepare(elf, Path::new("native.bin")).unwrap();
        assert_eq!(path, "native.bin");
    }
}
