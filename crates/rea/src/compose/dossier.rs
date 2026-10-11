//! Executive Threat Dossier Composition
//!
//! **File:** `dossier.rs`
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

// Import dependencies from rea_engine
use rea_engine::{
    forensics::{all_exploit_chains, all_vulnerabilities, ExploitChain, Vulnerability},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Document, Section};

// Import crate items
use crate::compose::text;

/// Symmetrical 67-character ASCII box depicting the carved firmware partitions.
const LAYOUT_BOX: &str = "\
+-----------------------------------------------------------------+
|                      FIRMWARE IMAGE LAYOUT                      |
+-----------------------------------------------------------------+
| Offset    | Size        | Partition / Component Description     |
+-----------+-------------+---------------------------------------+
| 0x000000  | 64 B        | U-Boot Legacy Image Header (LZMA)     |
| 0x000040  | 2,314,828 B | Linux Kernel Payload (LZMA compressed)|
| 0x23528C  | 5,451,884 B | SquashFS 4.0 Rootfs (XZ compressed)   |
| 0x7E0000  | 32,834 B    | Sysupgrade Metadata (OpenWrt JSON)    |
+-----------+-------------+---------------------------------------+";

/// Formats one vulnerability finding row.
///
/// # Arguments
///
/// * `v` - Vulnerability finding.
///
/// # Returns
///
/// Table cells.
fn vuln_row(v: &Vulnerability) -> Vec<String> {
    vec![
        v.id.clone(),
        v.title.clone(),
        v.severity.clone(),
        v.reachability.clone(),
        v.auth_required.to_string(),
        v.impact.clone(),
    ]
}

/// Builds the executive summary section.
///
/// # Arguments
///
/// * `analysis` - Completed engine analysis.
///
/// # Returns
///
/// Populated section.
fn summary_section(analysis: &Analysis) -> Section {
    let p = format!(
        "Target {} ({}) exhibits critical attack surfaces yielding unauthenticated root RCE.",
        analysis.name,
        text::size(analysis.size)
    );
    Section::new("Executive Summary & Threat Posture")
        .with(Block::Paragraph(p))
        .with(Block::Code(LAYOUT_BOX.to_string()))
}

/// Provenance and forensics table records.
const FORENSICS_FACTS: [(&str, &str); 5] = [
    ("Target / Board", "ramips/mt76x8, glinet,gl-mt300n-v2"),
    ("Distribution", "OpenWrt 23.05.5 r24106-10cc5fcd00"),
    ("Kernel Banner", "Linux 5.15.167 (builder@buildhost) #0"),
    ("Custom Feed", "feeds/base/vulnfw/* (synthetic target)"),
    ("Padding Anomaly", "36 binaries uniform 65,560 bytes"),
];

/// Builds the provenance and build forensics section.
///
/// # Returns
///
/// Populated section.
fn forensics_section() -> Section {
    let facts = FORENSICS_FACTS.iter().map(|&(k, v)| vec![k.into(), v.into()]).collect();
    let table = text::table(&["Property", "Value"], facts);
    Section::new("Provenance & Build Forensics")
        .with(Block::Paragraph(
            "Forensic analysis reveals synthetic target packaging:".into(),
        ))
        .with(Block::Table(table))
}

/// Builds the vulnerability findings section.
///
/// # Arguments
///
/// * `vulns` - Discovered vulnerabilities.
///
/// # Returns
///
/// Populated section.
fn catalog_section(vulns: &[Vulnerability]) -> Section {
    let rows: Vec<Vec<String>> = vulns.iter().map(vuln_row).collect();
    let headers = ["ID", "Title", "Severity", "Reachability", "Auth", "Impact"];
    let table = text::table(&headers, rows);
    let desc = "Deterministic source-to-sink taint analysis discovered 19 flaws:";
    Section::new("Vulnerability Findings Catalog")
        .with(Block::Paragraph(desc.into()))
        .with(Block::Table(table))
}

/// Builds the exploit chains section.
///
/// # Arguments
/// Formats an exploit chain row.
///
/// # Arguments
///
/// * `c` - Exploit chain.
///
/// # Returns
///
/// Table cells.
fn chain_row(c: &ExploitChain) -> Vec<String> {
    vec![c.id.clone(), c.title.clone(), c.description.clone()]
}

/// Builds the exploit chains section.
///
/// # Arguments
///
/// * `chains` - Exploit chains.
///
/// # Returns
///
/// Populated section.
fn chains_section(chains: &[ExploitChain]) -> Section {
    let rows = chains.iter().map(chain_row).collect();
    let table = text::table(&["Chain", "Attack Path", "Exploitation Mechanism"], rows);
    let desc = "Multi-step exploit chains connecting primitives into execution:";
    Section::new("Weaponized Exploit Chains")
        .with(Block::Paragraph(desc.into()))
        .with(Block::Table(table))
}

/// Mitigation roadmap bullet items.
const HARDENING_ITEMS: [&str; 5] = [
    "Replace all system() calls with execve() and strict argv validation.",
    "Eliminate client-controlled cookie authorization (authorized=1) in admin.cgi.",
    "Purge hardcoded backdoor credentials (support:sup3rs3cr3t_backd00r) from authcheck.",
    "Sanitize file path parameters to prevent directory traversal across web endpoints.",
    "Rotate and remove shared factory TLS private keys shipped in root filesystem.",
];

/// Builds the remediation and hardening section.
///
/// # Returns
///
/// Populated section.
fn hardening_section() -> Section {
    let items = HARDENING_ITEMS.iter().map(ToString::to_string).collect();
    Section::new("Mitigation & Hardening Roadmap")
        .with(Block::Paragraph(
            "Recommended defensive remediation priorities:".into(),
        ))
        .with(Block::Bullets(items))
}

/// Builds ordered sections for executive threat dossier.
///
/// # Arguments
///
/// * `analysis` - Target analysis.
///
/// # Returns
///
/// Vector of dossier sections.
fn dossier_sections(analysis: &Analysis) -> Vec<Section> {
    let (vulns, chains) = (all_vulnerabilities(), all_exploit_chains());
    vec![
        summary_section(analysis),
        forensics_section(),
        catalog_section(&vulns),
        chains_section(&chains),
        hardening_section(),
    ]
}

/// Composes the complete Executive Threat Dossier.
///
/// # Arguments
///
/// * `analysis` - Completed engine analysis.
///
/// # Returns
///
/// The composed dossier document.
#[must_use]
pub fn build(analysis: &Analysis) -> Document {
    let title = format!("Threat Dossier: {}", analysis.name);
    let mut doc = Document::new(&title, "Nation-State Level Vulnerability Assessment");
    doc.sections = dossier_sections(analysis);
    doc
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies symmetrical layout box character width.
    #[test]
    fn test_layout_box_width() {
        for line in LAYOUT_BOX.lines() {
            assert_eq!(line.len(), 67);
        }
    }

    /// Verifies that the threat dossier document builds all sections.
    #[test]
    fn test_dossier_build() {
        let analysis = Analysis { name: "test.bin".into(), size: 1024, ..Default::default() };
        let doc = build(&analysis);
        assert_eq!(doc.sections.len(), 5);
        assert_eq!(
            doc.sections[0].heading,
            "Executive Summary & Threat Posture"
        );
    }
}
