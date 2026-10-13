//! Network Capture Analysis for PCAP and HAR
//!
//! **File:** `netcap.rs`
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

// Import dependencies from serde_json
use serde_json::Value;

/// Most HTTP entries retained.
const MAX_HTTP: usize = 200;

/// Recovered packet capture inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pcap {
    /// Link-layer type.
    pub linktype: u32,
    /// Snap length.
    pub snaplen: u32,
    /// Packet count.
    pub packet_count: usize,
    /// Total captured bytes.
    pub total_bytes: u64,
    /// Recovered HTTP request lines.
    pub http_requests: Vec<String>,
    /// Recovered HTTP host headers.
    pub hosts: Vec<String>,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// Recovered HAR inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Har {
    /// Number of log entries.
    pub entry_count: usize,
    /// Request URLs.
    pub urls: Vec<String>,
    /// Response status codes.
    pub statuses: Vec<u64>,
}

/// Reads a little or big-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The word, or zero when truncated.
fn read_u32(bytes: &[u8], offset: usize, little: bool) -> u32 {
    let word = bytes.get(offset..offset + 4).unwrap_or_default();
    let mut buf = [0u8; 4];
    buf[..word.len().min(4)].copy_from_slice(&word[..word.len().min(4)]);
    if little {
        u32::from_le_bytes(buf)
    } else {
        u32::from_be_bytes(buf)
    }
}

/// Resolves the endianness of a classic pcap file.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` for little-endian, or `None` when not a classic pcap.
fn pcap_endian(bytes: &[u8]) -> Option<bool> {
    match bytes.get(..4)? {
        [0xd4, 0xc3, 0xb2, 0xa1] | [0x4d, 0x3c, 0xb2, 0xa1] => Some(true),
        [0xa1, 0xb2, 0xc3, 0xd4] | [0xa1, 0xb2, 0x3c, 0x4d] => Some(false),
        _ => None,
    }
}

/// Reports whether a buffer is a classic pcap file.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the pcap magic matches.
#[must_use]
pub fn is_pcap(bytes: &[u8]) -> bool {
    pcap_endian(bytes).is_some()
}

/// Parses a classic pcap file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered capture, or `None`.
#[must_use]
pub fn parse_pcap(bytes: &[u8]) -> Option<Pcap> {
    let little = pcap_endian(bytes)?;
    let mut pcap = Pcap {
        linktype: read_u32(bytes, 20, little),
        snaplen: read_u32(bytes, 16, little),
        ..Default::default()
    };
    parse_packets(bytes, little, &mut pcap);
    Some(pcap)
}

/// Iterates the packet records of a pcap file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `little` - Whether the encoding is little-endian.
/// * `pcap` - Output capture inventory.
fn parse_packets(bytes: &[u8], little: bool, pcap: &mut Pcap) {
    let mut pos = 24;
    while let Some((start, end)) = packet_span(bytes, pos, little) {
        record_packet(bytes.get(start..end).unwrap_or_default(), pcap);
        pos = end;
    }
}

/// Computes the span of one packet record.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Record offset.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The payload start and end offsets, or `None`.
fn packet_span(bytes: &[u8], pos: usize, little: bool) -> Option<(usize, usize)> {
    if pos + 16 > bytes.len() {
        return None;
    }
    let incl = usize::try_from(read_u32(bytes, pos + 8, little)).ok()?;
    let start = pos + 16;
    let end = start.saturating_add(incl).min(bytes.len());
    Some((start, end))
}

/// Records one packet and scans it for HTTP.
///
/// # Arguments
///
/// * `data` - Packet payload.
/// * `pcap` - Output capture inventory.
fn record_packet(data: &[u8], pcap: &mut Pcap) {
    pcap.packet_count += 1;
    pcap.total_bytes += u64::try_from(data.len()).unwrap_or(0);
    scan_http(data, pcap);
}

/// Scans a packet for HTTP request lines and host headers.
///
/// # Arguments
///
/// * `data` - Packet payload.
/// * `pcap` - Output capture inventory.
fn scan_http(data: &[u8], pcap: &mut Pcap) {
    for line in data.split(|&b| b == b'\n') {
        let text = String::from_utf8_lossy(line).trim_end_matches('\r').to_string();
        record_http_line(&text, pcap);
    }
}

/// Records an HTTP request line or host header.
///
/// # Arguments
///
/// * `line` - Candidate line.
/// * `pcap` - Output capture inventory.
fn record_http_line(line: &str, pcap: &mut Pcap) {
    if is_request_line(line) {
        push_unique(&mut pcap.http_requests, line.to_string());
    } else if let Some(host) = line.strip_prefix("Host: ") {
        push_unique(&mut pcap.hosts, host.trim().to_string());
    }
}

/// Reports whether a line is an HTTP request line.
///
/// # Arguments
///
/// * `line` - Candidate line.
///
/// # Returns
///
/// `true` when the line is a well-formed request line.
fn is_request_line(line: &str) -> bool {
    let methods = [
        "GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ", "PATCH ",
    ];
    methods.iter().any(|m| line.starts_with(m)) && line.contains("HTTP/")
}

/// Pushes a value to a list when new and under the cap.
///
/// # Arguments
///
/// * `list` - Destination list.
/// * `value` - Value to add.
fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.iter().any(|v| v == &value) && list.len() < MAX_HTTP {
        list.push(value);
    }
}

/// Reports whether a buffer is a HAR document.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when a `log` object is present.
#[must_use]
pub fn is_har(bytes: &[u8]) -> bool {
    let parsed = serde_json::from_slice::<Value>(bytes);
    parsed.is_ok_and(|v| v.get("log").is_some())
}

/// Parses a HAR document.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered HAR inventory, or `None`.
#[must_use]
pub fn parse_har(bytes: &[u8]) -> Option<Har> {
    if !is_har(bytes) {
        return None;
    }
    let value: Value = serde_json::from_slice(bytes).ok()?;
    Some(build_har(&value))
}

/// Builds the HAR inventory from parsed JSON.
///
/// # Arguments
///
/// * `value` - Parsed JSON value.
///
/// # Returns
///
/// The recovered HAR inventory.
fn build_har(value: &Value) -> Har {
    let list = value.pointer("/log/entries").and_then(Value::as_array);
    let entries: &[Value] = list.map_or(&[], Vec::as_slice);
    let mut har = Har { entry_count: entries.len(), ..Default::default() };
    for entry in entries.iter().take(MAX_HTTP) {
        har.urls.push(request_url(entry));
        har.statuses.push(response_status(entry));
    }
    har
}

/// Reads a HAR request URL.
///
/// # Arguments
///
/// * `entry` - HAR log entry.
///
/// # Returns
///
/// The request URL.
fn request_url(entry: &Value) -> String {
    let url = entry.pointer("/request/url").and_then(Value::as_str).unwrap_or_default();
    url.to_string()
}

/// Reads a HAR response status code.
///
/// # Arguments
///
/// * `entry` - HAR log entry.
///
/// # Returns
///
/// The response status code, or zero.
fn response_status(entry: &Value) -> u64 {
    entry.pointer("/response/status").and_then(Value::as_u64).unwrap_or(0)
}

/// Formats a capture inventory as a human-readable text block.
///
/// # Arguments
///
/// * `pcap` - Recovered capture inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_pcap_summary(pcap: &Pcap) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "PCAP linktype {}", pcap.linktype);
    let _ = writeln!(out, "Snaplen: {}", pcap.snaplen);
    let _ = writeln!(out, "Packets: {}", pcap.packet_count);
    let _ = writeln!(out, "Bytes: {}", pcap.total_bytes);
    let _ = writeln!(out, "HTTP requests: {}", pcap.http_requests.len());
    let _ = writeln!(out, "Hosts: {}", pcap.hosts.len());
    out
}

/// Formats a HAR inventory as a human-readable text block.
///
/// # Arguments
///
/// * `har` - Recovered HAR inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_har_summary(har: &Har) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "HAR entries: {}", har.entry_count);
    for (url, status) in har.urls.iter().zip(&har.statuses) {
        let _ = writeln!(out, "  {status} {url}");
    }
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal little-endian pcap with one HTTP packet.
    ///
    /// # Returns
    ///
    /// The capture bytes.
    fn sample_pcap() -> Vec<u8> {
        let payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let mut data = vec![0xd4, 0xc3, 0xb2, 0xa1, 0x02, 0x00, 0x04, 0x00];
        data.extend_from_slice(&[0; 8]);
        data.extend_from_slice(&65535u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[0; 8]);
        append_packet(&mut data, payload);
        data
    }

    /// Appends a packet record with a length prefix.
    ///
    /// # Arguments
    ///
    /// * `data` - Capture buffer.
    /// * `payload` - Packet payload.
    fn append_packet(data: &mut Vec<u8>, payload: &[u8]) {
        let len = u32::try_from(payload.len()).unwrap();
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(payload);
    }

    /// Verifies pcap detection and parsing.
    #[test]
    fn test_pcap_parse() {
        let data = sample_pcap();
        assert!(is_pcap(&data));
        let pcap = parse_pcap(&data).unwrap();
        assert_eq!(pcap.packet_count, 1);
        assert_eq!(pcap.http_requests.len(), 1);
        assert_eq!(pcap.hosts, vec!["example.com".to_string()]);
    }

    /// Verifies request-line detection.
    #[test]
    fn test_pcap_request_line() {
        assert!(is_request_line("GET / HTTP/1.1"));
        assert!(!is_request_line("Host: example.com"));
    }

    /// Verifies HAR detection and parsing.
    #[test]
    fn test_har_parse() {
        let json =
            r#"{"log":{"entries":[{"request":{"url":"https://x/y"},"response":{"status":200}}]}}"#;
        assert!(is_har(json.as_bytes()));
        let har = parse_har(json.as_bytes()).unwrap();
        assert_eq!(har.entry_count, 1);
        assert_eq!(har.urls, vec!["https://x/y".to_string()]);
        assert_eq!(har.statuses, vec![200]);
    }

    /// Verifies summary formatting.
    #[test]
    fn test_pcap_summary() {
        let pcap = parse_pcap(&sample_pcap()).unwrap();
        assert!(format_pcap_summary(&pcap).contains("Packets: 1"));
    }
}
