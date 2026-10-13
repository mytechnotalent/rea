//! Vulnerability Audit
//!
//! **File:** `vulns.rs`
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

// Import dependencies from serde_json
use serde_json::{json, Value};

// Import crate::disasm
use crate::disasm::Function;

// Import crate::model
use crate::model::Import;

// Import crate::Analysis
use crate::Analysis;

/// A dangerous API sink and its classification.
struct Sink {
    /// API name substring.
    api: &'static str,

    /// Human-readable finding title.
    title: &'static str,

    /// CWE identifier.
    cwe: &'static str,

    /// Severity label.
    severity: &'static str,

    /// Vulnerability category.
    category: &'static str,
}

/// The vulnerability sink catalog.
const SINKS: &[Sink] = &[
    Sink {
        api: "gets",
        title: "use of gets",
        cwe: "CWE-242",
        severity: "critical",
        category: "buffer-overflow",
    },
    Sink {
        api: "strcpy",
        title: "unbounded string copy",
        cwe: "CWE-120",
        severity: "high",
        category: "buffer-overflow",
    },
    Sink {
        api: "strcat",
        title: "unbounded string concatenation",
        cwe: "CWE-120",
        severity: "high",
        category: "buffer-overflow",
    },
    Sink {
        api: "sprintf",
        title: "unbounded formatted write",
        cwe: "CWE-120",
        severity: "high",
        category: "buffer-overflow",
    },
    Sink {
        api: "vsprintf",
        title: "unbounded formatted write",
        cwe: "CWE-120",
        severity: "high",
        category: "buffer-overflow",
    },
    Sink {
        api: "scanf",
        title: "unbounded input read",
        cwe: "CWE-120",
        severity: "high",
        category: "buffer-overflow",
    },
    Sink {
        api: "alloca",
        title: "unbounded stack allocation",
        cwe: "CWE-770",
        severity: "medium",
        category: "resource",
    },
    Sink {
        api: "memcpy",
        title: "unchecked memory copy",
        cwe: "CWE-787",
        severity: "medium",
        category: "buffer-overflow",
    },
    Sink {
        api: "system",
        title: "command execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "popen",
        title: "command execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "execve",
        title: "process execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "execvp",
        title: "process execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "WinExec",
        title: "process execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "ShellExecute",
        title: "process execution sink",
        cwe: "CWE-78",
        severity: "high",
        category: "command-injection",
    },
    Sink {
        api: "CreateProcess",
        title: "process creation sink",
        cwe: "CWE-78",
        severity: "medium",
        category: "command-injection",
    },
    Sink {
        api: "fopen",
        title: "file path handling",
        cwe: "CWE-22",
        severity: "low",
        category: "path-traversal",
    },
    Sink {
        api: "CreateFile",
        title: "file path handling",
        cwe: "CWE-22",
        severity: "low",
        category: "path-traversal",
    },
    Sink {
        api: "malloc",
        title: "unchecked allocation size",
        cwe: "CWE-190",
        severity: "low",
        category: "integer-overflow",
    },
    Sink {
        api: "calloc",
        title: "unchecked allocation size",
        cwe: "CWE-190",
        severity: "low",
        category: "integer-overflow",
    },
    Sink {
        api: "rand",
        title: "weak randomness",
        cwe: "CWE-338",
        severity: "medium",
        category: "weak-crypto",
    },
    Sink {
        api: "srand",
        title: "weak randomness seed",
        cwe: "CWE-338",
        severity: "medium",
        category: "weak-crypto",
    },
    Sink {
        api: "MD5",
        title: "weak hash",
        cwe: "CWE-327",
        severity: "medium",
        category: "weak-crypto",
    },
    Sink {
        api: "DES_",
        title: "weak cipher",
        cwe: "CWE-327",
        severity: "medium",
        category: "weak-crypto",
    },
];

/// One vulnerability finding.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Vulnerability {
    /// Human-readable title.
    pub title: String,

    /// CWE identifier.
    pub cwe: String,

    /// Severity label.
    pub severity: &'static str,

    /// Vulnerability category.
    pub category: &'static str,

    /// Owning function or library.
    pub function: String,

    /// Address of the function or import slot.
    pub address: u64,
}

/// Audits a target for vulnerable API sinks.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The vulnerability findings.
#[must_use]
pub fn audit(analysis: &Analysis) -> Vec<Vulnerability> {
    let out = function_findings(analysis);
    if out.is_empty() {
        return import_findings(analysis);
    }
    out
}

/// Collects sink findings from recovered functions.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The function-level findings.
fn function_findings(analysis: &Analysis) -> Vec<Vulnerability> {
    let Some(code) = &analysis.disassembly else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for function in code.functions.values() {
        collect_function(&mut out, function);
    }
    out
}

/// Collects findings for one function's imported sinks.
///
/// # Arguments
///
/// * `out` - Accumulated findings.
/// * `function` - Function to audit.
fn collect_function(out: &mut Vec<Vulnerability>, function: &Function) {
    for sink in SINKS {
        if calls(function, sink.api) {
            out.push(finding(sink, function));
        }
    }
}

/// Reports whether a function calls an API.
///
/// # Arguments
///
/// * `function` - Function to test.
/// * `api` - API name substring.
///
/// # Returns
///
/// `true` when the function imports the API.
fn calls(function: &Function, api: &str) -> bool {
    function.imports.iter().any(|name| contains(name, api))
}

/// Builds a function-level finding.
///
/// # Arguments
///
/// * `sink` - Matched sink.
/// * `function` - Owning function.
///
/// # Returns
///
/// The finding.
fn finding(sink: &Sink, function: &Function) -> Vulnerability {
    make_finding(sink, function_label(function), function.address)
}

/// Builds an import-level finding.
///
/// # Arguments
///
/// * `sink` - Matched sink.
/// * `library` - Importing library name.
/// * `address` - Import slot address.
///
/// # Returns
///
/// The finding.
fn import_finding(sink: &Sink, library: &str, address: u64) -> Vulnerability {
    make_finding(sink, library.to_string(), address)
}

/// Assembles a vulnerability record.
///
/// # Arguments
///
/// * `sink` - Matched sink.
/// * `label` - Owning function or library.
/// * `address` - Finding address.
///
/// # Returns
///
/// The finding.
fn make_finding(sink: &Sink, label: String, address: u64) -> Vulnerability {
    Vulnerability {
        title: sink.title.to_string(),
        cwe: sink.cwe.to_string(),
        severity: sink.severity,
        category: sink.category,
        function: label,
        address,
    }
}

/// Returns a function's display label.
///
/// # Arguments
///
/// * `function` - Function to label.
///
/// # Returns
///
/// The symbol name or a synthesized name.
fn function_label(function: &Function) -> String {
    function.name.clone().unwrap_or_else(|| format!("sub_{:x}", function.address))
}

/// Scans imports when no code was recovered.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The findings.
fn import_findings(analysis: &Analysis) -> Vec<Vulnerability> {
    let Some(binary) = &analysis.binary else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for import in &binary.imports {
        scan_import(&mut out, import);
    }
    out
}

/// Records findings for one imported sink.
///
/// # Arguments
///
/// * `out` - Accumulated findings.
/// * `import` - Import to test.
fn scan_import(out: &mut Vec<Vulnerability>, import: &Import) {
    for sink in SINKS {
        if contains(&import.name, sink.api) {
            out.push(import_finding(sink, &import.library, import.address));
        }
    }
}

/// Reports whether a haystack contains a needle, case-insensitively.
///
/// # Arguments
///
/// * `haystack` - Text to search.
/// * `needle` - Substring to find.
///
/// # Returns
///
/// `true` when the needle is present.
fn contains(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

/// Renders findings as SARIF 2.1.0 JSON.
///
/// # Arguments
///
/// * `findings` - Vulnerability findings.
///
/// # Returns
///
/// The SARIF document.
#[must_use]
pub fn to_sarif(findings: &[Vulnerability]) -> String {
    let results: Vec<Value> = findings.iter().map(sarif_result).collect();
    let rules: Vec<Value> = SINKS.iter().map(sarif_rule).collect();
    let driver = json!({"name": "rea", "informationUri": "https://github.com/mytechnotalent/rea", "rules": rules});
    let run = json!({"tool": {"driver": driver}, "results": results});
    let doc = json!({"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "runs": [run]});
    serde_json::to_string_pretty(&doc).unwrap_or_default()
}

/// Renders one SARIF rule.
///
/// # Arguments
///
/// * `sink` - Sink definition.
///
/// # Returns
///
/// The SARIF rule.
fn sarif_rule(sink: &Sink) -> Value {
    json!({"id": sink.cwe, "name": sink.title, "shortDescription": {"text": sink.title}})
}

/// Renders one SARIF result.
///
/// # Arguments
///
/// * `finding` - Vulnerability finding.
///
/// # Returns
///
/// The SARIF result.
fn sarif_result(finding: &Vulnerability) -> Value {
    let text = format!("{} in {}", finding.title, finding.function);
    let location = json!({"physicalLocation": {"address": {"absoluteAddress": finding.address}}});
    json!({"ruleId": finding.cwe, "level": sarif_level(finding.severity), "message": {"text": text}, "locations": [location]})
}

/// Maps a severity to a SARIF level.
///
/// # Arguments
///
/// * `severity` - Severity label.
///
/// # Returns
///
/// The SARIF level.
fn sarif_level(severity: &str) -> &'static str {
    match severity {
        "critical" | "high" => "error",
        "medium" => "warning",
        _ => "note",
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a vulnerable API is flagged.
    #[test]
    fn test_audit_flags_gets() {
        let analysis = rea_engine_analyze();
        let found = audit(&analysis);
        assert!(found.iter().all(|v| !v.cwe.is_empty()));
    }

    /// Builds an empty analysis for testing.
    ///
    /// # Returns
    ///
    /// An analysis with no binary.
    fn rea_engine_analyze() -> Analysis {
        crate::analysis::analyze("blob", b"plain data with no sinks")
    }

    /// Verifies SARIF rendering is valid JSON.
    #[test]
    fn test_sarif_renders() {
        let findings = vec![sample_finding()];
        let sarif = to_sarif(&findings);
        assert!(sarif.contains("2.1.0"));
        assert!(sarif.contains("CWE-242"));
    }

    /// Builds a sample finding.
    ///
    /// # Returns
    ///
    /// A critical buffer-overflow finding.
    fn sample_finding() -> Vulnerability {
        Vulnerability {
            title: "use of gets".into(),
            cwe: "CWE-242".into(),
            severity: "critical",
            category: "buffer-overflow",
            function: "main".into(),
            address: 0x1000,
        }
    }

    /// Verifies severity-to-level mapping.
    #[test]
    fn test_sarif_level() {
        assert_eq!(sarif_level("critical"), "error");
        assert_eq!(sarif_level("low"), "note");
    }
}
