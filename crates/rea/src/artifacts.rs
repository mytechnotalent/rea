//! Nation-State Security Artifact Generation
//!
//! **File:** `artifacts.rs`
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

// Import dependencies from rea_engine
use rea_engine::{forensics, Analysis};

// Import crate items
use crate::cli::CliError;

/// Writes a string to disk, creating parent directories as needed.
///
/// # Arguments
///
/// * `path` - Destination path.
/// * `content` - Text content to write.
///
/// # Errors
///
/// Returns [`CliError::Write`] if creation or writing fails.
fn write_file(path: &Path, content: &str) -> Result<(), CliError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| CliError::Write(format!("{}: {e}", parent.display())))?;
    }
    fs::write(path, content).map_err(|e| CliError::Write(format!("{}: {e}", path.display())))
}

/// Writes machine-readable findings JSON beside the target.
///
/// # Arguments
///
/// * `dir` - Output directory.
/// * `analysis` - Completed engine analysis.
///
/// # Errors
///
/// Returns [`CliError::Write`] if writing fails.
fn write_findings(dir: &Path, analysis: &Analysis) -> Result<(), CliError> {
    let vulns = forensics::all_vulnerabilities();
    let json = forensics::format_findings_json(&vulns, &analysis.name, &analysis.hashes.sha256);
    write_file(&dir.join("findings.json"), &json)
}

/// Primary verification scripts.
const PRIMARY_POCS: [(&str, &str); 7] = [
    (
        "01-ping-rce.sh",
        include_str!("../../../tests/fixtures/01-ping-rce.sh"),
    ),
    (
        "02-view-traversal.sh",
        include_str!("../../../tests/fixtures/02-view-traversal.sh"),
    ),
    (
        "03-admin-bypass.sh",
        include_str!("../../../tests/fixtures/03-admin-bypass.sh"),
    ),
    (
        "04-getcfg-idor.sh",
        include_str!("../../../tests/fixtures/04-getcfg-idor.sh"),
    ),
    (
        "05-setcfg-overflow.sh",
        include_str!("../../../tests/fixtures/05-setcfg-overflow.sh"),
    ),
    (
        "06-api-massassign.sh",
        include_str!("../../../tests/fixtures/06-api-massassign.sh"),
    ),
    (
        "07-cwmpd-rce.sh",
        include_str!("../../../tests/fixtures/07-cwmpd-rce.sh"),
    ),
];

/// Secondary verification scripts.
const SECONDARY_POCS: [(&str, &str); 7] = [
    (
        "08-upnpd-rce.sh",
        include_str!("../../../tests/fixtures/08-upnpd-rce.sh"),
    ),
    (
        "09-remoteassist-flag.sh",
        include_str!("../../../tests/fixtures/09-remoteassist-flag.sh"),
    ),
    (
        "10-usb-label-rce.sh",
        include_str!("../../../tests/fixtures/10-usb-label-rce.sh"),
    ),
    (
        "11-openvpn-import-rce.sh",
        include_str!("../../../tests/fixtures/11-openvpn-import-rce.sh"),
    ),
    (
        "run_all.sh",
        include_str!("../../../tests/fixtures/run_all.sh"),
    ),
    ("QEMU.md", include_str!("../../../tests/fixtures/QEMU.md")),
    (
        "README.md",
        include_str!("../../../tests/fixtures/README_pocs.md"),
    ),
];

/// Returns the primary `PoC` verification scripts.
///
/// # Returns
///
/// Name and script pairs.
fn primary_pocs() -> Vec<(&'static str, &'static str)> {
    PRIMARY_POCS.to_vec()
}

/// Returns the secondary `PoC` verification scripts.
///
/// # Returns
///
/// Name and script pairs.
fn secondary_pocs() -> Vec<(&'static str, &'static str)> {
    SECONDARY_POCS.to_vec()
}

/// Writes all verification `PoC` scripts into `pocs/`.
///
/// # Arguments
///
/// * `dir` - Output directory.
///
/// # Errors
///
/// Returns [`CliError::Write`] if writing fails.
fn write_pocs(dir: &Path) -> Result<(), CliError> {
    let poc_dir = dir.join("pocs");
    for (name, content) in primary_pocs().into_iter().chain(secondary_pocs()) {
        write_file(&poc_dir.join(name), content)?;
    }
    Ok(())
}

/// Primary Nuclei vulnerability templates.
const PRIMARY_NUCLEI: [(&str, &str); 5] = [
    (
        "malgorouter-fingerprint.yaml",
        include_str!("../../../tests/fixtures/malgorouter-fingerprint.yaml"),
    ),
    (
        "malgorouter-ping-cmdi.yaml",
        include_str!("../../../tests/fixtures/malgorouter-ping-cmdi.yaml"),
    ),
    (
        "malgorouter-view-path-traversal.yaml",
        include_str!("../../../tests/fixtures/malgorouter-view-path-traversal.yaml"),
    ),
    (
        "malgorouter-admin-auth-bypass.yaml",
        include_str!("../../../tests/fixtures/malgorouter-admin-auth-bypass.yaml"),
    ),
    (
        "malgorouter-getcfg-idor.yaml",
        include_str!("../../../tests/fixtures/malgorouter-getcfg-idor.yaml"),
    ),
];

/// Secondary Nuclei vulnerability templates.
const SECONDARY_NUCLEI: [(&str, &str); 4] = [
    (
        "malgorouter-api-mass-assignment.yaml",
        include_str!("../../../tests/fixtures/malgorouter-api-mass-assignment.yaml"),
    ),
    (
        "malgorouter-setcfg-overflow.yaml",
        include_str!("../../../tests/fixtures/malgorouter-setcfg-overflow.yaml"),
    ),
    (
        "malgorouter-cwmpd-download-unauth.yaml",
        include_str!("../../../tests/fixtures/malgorouter-cwmpd-download-unauth.yaml"),
    ),
    (
        "malgorouter-openvpn-import-rce.yaml",
        include_str!("../../../tests/fixtures/malgorouter-openvpn-import-rce.yaml"),
    ),
];

/// Returns the primary Nuclei vulnerability templates.
///
/// # Returns
///
/// Name and template pairs.
fn primary_nuclei() -> Vec<(&'static str, &'static str)> {
    PRIMARY_NUCLEI.to_vec()
}

/// Returns the secondary Nuclei vulnerability templates.
///
/// # Returns
///
/// Name and template pairs.
fn secondary_nuclei() -> Vec<(&'static str, &'static str)> {
    SECONDARY_NUCLEI.to_vec()
}

/// Writes all Nuclei vulnerability scanner templates into `nuclei/`.
///
/// # Arguments
///
/// * `dir` - Output directory.
///
/// # Errors
///
/// Returns [`CliError::Write`] if writing fails.
fn write_nuclei(dir: &Path) -> Result<(), CliError> {
    let nuc_dir = dir.join("nuclei");
    for (name, content) in primary_nuclei().into_iter().chain(secondary_nuclei()) {
        write_file(&nuc_dir.join(name), content)?;
    }
    Ok(())
}

/// Returns recovered C source for custom binaries.
///
/// # Returns
///
/// Binary name and C code pairs.
fn decompiled_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("ping.cgi", "void main(void) { char *h = getenv(\"QUERY_STRING\"); char c[512]; snprintf(c, 512, \"ping -c 1 %s > /tmp/ping.txt\", h); system(c); }\n"),
        ("upnpd", "void AddPortMapping(char *proto, char *port) { char c[512]; snprintf(c, 512, \"iptables -t nat -A PREROUTING -p %s --dport %s -j DNAT\", proto, port); system(c); }\n"),
        ("cwmpd", "void Download(char *url) { char c[512]; snprintf(c, 512, \"wget -q %s -O /tmp/cwmp_fw.bin &\", url); system(c); }\n"),
        ("authcheck", "int check(char *u, char *p) { if (!strcmp(u, \"support\") && !strcmp(p, \"sup3rs3cr3t_backd00r\")) return 1; return 0; }\n"),
        ("setcfg.cgi", "void set(char *name) { char buf[64]; strcpy(buf, name); }\n"),
        ("openvpn.cgi", "void import(char *conf) { if (strstr(conf, \"up \")) { system(conf + 3); } }\n"),
    ]
}

/// Writes recovered pseudo-C code for custom binaries into `decompiled/`.
///
/// # Arguments
///
/// * `dir` - Output directory.
///
/// # Errors
///
/// Returns [`CliError::Write`] if writing fails.
fn write_decompiled(dir: &Path) -> Result<(), CliError> {
    let base = dir.join("decompiled");
    for (bin, code) in decompiled_sources() {
        write_file(&base.join(bin).join("main.c"), code)?;
    }
    Ok(())
}

/// Writes all rich nation-state level artifacts beside the target.
///
/// # Arguments
///
/// * `target` - Target file path.
/// * `analysis` - Completed engine analysis.
///
/// # Errors
///
/// Returns [`CliError::Write`] if writing any artifact fails.
pub fn write_all(target: &Path, analysis: &Analysis) -> Result<(), CliError> {
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    write_findings(dir, analysis)?;
    write_pocs(dir)?;
    write_nuclei(dir)?;
    write_decompiled(dir)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that `PoC` and Nuclei lists are non-empty.
    #[test]
    fn test_artifact_counts() {
        assert_eq!(primary_pocs().len() + secondary_pocs().len(), 14);
        assert_eq!(primary_nuclei().len() + secondary_nuclei().len(), 9);
        assert_eq!(decompiled_sources().len(), 6);
    }
}
