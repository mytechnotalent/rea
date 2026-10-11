//! Indicator Extraction
//!
//! **File:** `iocs.rs`
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
use std::{collections::HashSet, net::Ipv4Addr};

// Import crate::strings
use crate::strings::FoundString;

/// Maximum indicators retained per target.
pub const MAX_INDICATORS: usize = 5_000;

/// A potential indicator found in a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indicator {
    /// Indicator kind, such as `URL`.
    pub kind: &'static str,

    /// Extracted indicator value.
    pub value: String,

    /// File offset of the containing string.
    pub offset: u64,
}

/// Signature of one indicator detector.
type Detector = fn(&str) -> Option<String>;

/// Indicator detectors evaluated against every string.
const DETECTORS: [(&str, Detector); 8] = [
    ("URL", url),
    ("IPv4 address", ipv4),
    ("Email address", email),
    ("Registry key", registry),
    ("Windows path", windows_path),
    ("Unix path", unix_path),
    ("Command or interpreter", command),
    ("User agent", user_agent),
];

/// Interpreters and administrative tools worth flagging.
const COMMANDS: [&str; 16] = [
    "cmd.exe",
    "powershell",
    "pwsh",
    "/bin/sh",
    "/bin/bash",
    "wscript",
    "cscript",
    "rundll32",
    "regsvr32",
    "schtasks",
    "vssadmin",
    "bcdedit",
    "wmic",
    "certutil",
    "mshta",
    "bitsadmin",
];

/// Directory prefixes that mark Unix paths.
const UNIX_ROOTS: [&str; 8] = [
    "/etc/", "/tmp/", "/proc/", "/dev/", "/usr/", "/var/", "/home/", "/root/",
];

/// URL schemes recognized in strings.
const SCHEMES: [&str; 4] = ["https://", "http://", "ftp://", "wss://"];

/// Scans strings for indicators, deduplicating by kind and value.
///
/// # Arguments
///
/// * `strings` - Extracted strings.
///
/// # Returns
///
/// Unique indicators in string order, capped at [`MAX_INDICATORS`].
#[must_use]
pub fn scan(strings: &[FoundString]) -> Vec<Indicator> {
    let mut seen = HashSet::new();
    let all = strings.iter().flat_map(classify);
    let unique = all.filter(|i| seen.insert((i.kind, i.value.clone())));
    unique.take(MAX_INDICATORS).collect()
}

/// Runs every detector on one string.
///
/// # Arguments
///
/// * `string` - Extracted string.
///
/// # Returns
///
/// All indicators the string contains.
fn classify(string: &FoundString) -> Vec<Indicator> {
    let hit = |(kind, detect): &(&'static str, Detector)| {
        detect(&string.text).map(|value| Indicator { kind, value, offset: string.offset })
    };
    DETECTORS.iter().filter_map(hit).collect()
}

/// Extracts the first URL.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The URL up to the first delimiter.
fn url(text: &str) -> Option<String> {
    let start = SCHEMES.iter().filter_map(|s| text.find(s)).min()?;
    let tail = &text[start..];
    let end = tail.find(|c: char| c.is_whitespace() || "\"'<>".contains(c));
    Some(tail[..end.unwrap_or(tail.len())].to_string())
}

/// Extracts the first dotted-quad IPv4 address that is not a placeholder.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The address text.
fn ipv4(text: &str) -> Option<String> {
    let tokens = text.split(|c: char| !(c.is_ascii_digit() || c == '.'));
    let quads = tokens.filter(|t| t.split('.').count() == 4);
    let mut parsed = quads.filter_map(|t| t.parse::<Ipv4Addr>().ok());
    parsed.find(|ip| is_plausible_ip(*ip)).map(|ip| ip.to_string())
}

/// Rejects placeholder and version-number shaped addresses.
///
/// # Arguments
///
/// * `ip` - Parsed address.
///
/// # Returns
///
/// `false` for unspecified, broadcast, zero-leading, and `a.b.0.0` addresses.
fn is_plausible_ip(ip: Ipv4Addr) -> bool {
    let [first, _, third, fourth] = ip.octets();
    let version_like = third == 0 && fourth == 0;
    first != 0 && !ip.is_broadcast() && !version_like
}

/// Extracts the first email address.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The address, when one is well formed.
fn email(text: &str) -> Option<String> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || "._%+-@".contains(c);
    let mut tokens = text.split(|c: char| !allowed(c)).map(|t| t.trim_end_matches('.'));
    tokens.find(|t| is_email(t)).map(ToString::to_string)
}

/// Reports whether a token is shaped like an email address.
///
/// # Arguments
///
/// * `token` - Candidate token.
///
/// # Returns
///
/// `true` for `local@domain.tld` with non-empty parts.
fn is_email(token: &str) -> bool {
    let (local, domain) = token.split_once('@').unwrap_or_default();
    let (host, tld) = domain.rsplit_once('.').unwrap_or_default();
    let alphabetic = tld.chars().all(|c| c.is_ascii_alphabetic());
    !local.is_empty() && !host.is_empty() && tld.len() >= 2 && alphabetic
}

/// Detects registry keys.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The string, when it names a registry key.
fn registry(text: &str) -> Option<String> {
    let upper = text.to_ascii_uppercase();
    let found = REGISTRY_ROOTS.iter().any(|r| upper.contains(r));
    found.then(|| text.to_string())
}

/// Uppercase fragments that mark registry keys.
const REGISTRY_ROOTS: [&str; 5] = [
    "HKEY_",
    "HKLM\\",
    "HKCU\\",
    "SOFTWARE\\",
    "SYSTEM\\CURRENTCONTROLSET",
];

/// Detects Windows file system paths.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The string, when it contains a drive, UNC, or environment path.
fn windows_path(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let drive = bytes.windows(3).any(|w| w[0].is_ascii_alphabetic() && &w[1..] == b":\\");
    let special = text.starts_with("\\\\") || (text.starts_with('%') && text.contains("%\\"));
    (drive || special).then(|| text.to_string())
}

/// Detects Unix file system paths.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The string, when it starts with a well-known root directory.
fn unix_path(text: &str) -> Option<String> {
    UNIX_ROOTS.iter().any(|r| text.starts_with(r)).then(|| text.to_string())
}

/// Detects interpreters and administrative tools.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The string, when it mentions a flagged command.
fn command(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    COMMANDS.iter().any(|c| lower.contains(c)).then(|| text.to_string())
}

/// Detects HTTP user agent strings.
///
/// # Arguments
///
/// * `text` - String text.
///
/// # Returns
///
/// The string, when it is a browser-style user agent.
fn user_agent(text: &str) -> Option<String> {
    text.starts_with("Mozilla/").then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::strings
    use crate::strings::Encoding;

    /// Builds an extracted string.
    ///
    /// # Arguments
    ///
    /// * `text` - String text.
    ///
    /// # Returns
    ///
    /// An ASCII string at offset zero.
    fn found(text: &str) -> FoundString {
        FoundString { offset: 0, text: text.into(), encoding: Encoding::Ascii }
    }

    /// Verifies URL extraction stops at delimiters.
    #[test]
    fn test_url_stops_at_delimiters() {
        assert_eq!(
            url("get \"https://example.test/a?b=1\" now"),
            Some("https://example.test/a?b=1".into())
        );
    }

    /// Verifies IPv4 extraction ignores placeholders and versions with extra parts.
    #[test]
    fn test_ipv4_filters_placeholders() {
        assert_eq!(ipv4("connect 10.1.2.3:443"), Some("10.1.2.3".into()));
        assert_eq!(ipv4("0.0.0.0"), None);
        assert_eq!(ipv4("version 6.1.0.0"), None);
        assert_eq!(ipv4("1.2.3.4.5"), None);
    }

    /// Verifies email shape checks.
    #[test]
    fn test_email_requires_domain() {
        assert_eq!(
            email("mail admin@example.org."),
            Some("admin@example.org".into())
        );
        assert!(is_email("a@b.io"));
        assert!(!is_email("a@b"));
    }

    /// Verifies path and registry detection.
    #[test]
    fn test_paths_and_registry() {
        assert!(windows_path("C:\\Windows\\system32").is_some());
        assert!(unix_path("/etc/passwd").is_some());
        assert!(registry("Software\\Microsoft\\Windows\\CurrentVersion\\Run").is_some());
    }

    /// Verifies that scanning deduplicates repeated values.
    #[test]
    fn test_scan_deduplicates() {
        let strings = [found("cmd.exe /c dir"), found("cmd.exe /c dir")];
        assert_eq!(scan(&strings).len(), 1);
    }
}
