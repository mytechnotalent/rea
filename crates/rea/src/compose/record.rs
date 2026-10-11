//! Static Reverse Engineering Record Composition
//!
//! **File:** `record.rs`
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
use rea_engine::Analysis;

// Import dependencies from rea_report
use rea_report::{Block, Document, Section};

// Import crate items
use crate::compose::text;

/// Symmetrical 67-character ASCII box depicting multi-architecture coverage.
const ARCH_BOX: &str = "\
+-----------------------------------------------------------------+
|               MULTI-ARCHITECTURE DISASSEMBLY ENGINE             |
+---------------------------------+-------------------------------+
| Architecture Family             | Bit Width / ABI Supported     |
+---------------------------------+-------------------------------+
| ARM Cortex-A (ARMv7-A / ARMv8-A)| 32-bit ARM / 64-bit AArch64   |
| ARM Cortex-M (ARMv7-M / ARMv8-M)| 16/32-bit Thumb / Thumb-2     |
| MIPS / MIPS64 (Release 1 - 6)   | 32-bit O32 / 64-bit N64       |
| x86 / x86-64 (AMD64)            | 16-bit / 32-bit / 64-bit      |
| RISC-V (RV32GC / RV64GC)        | 32-bit / 64-bit Standard      |
| PowerPC / PowerPC 64            | 32-bit / 64-bit Big/Little    |
| SPARC / SPARC V9                | 32-bit / 64-bit V9 Big-Endian |
| SuperH (SH-4)                   | 32-bit Little / Big Endian    |
| AVR / AVR8                      | 8-bit Harvard Microcontroller |
| Tensilica Xtensa                | 32-bit Windowed / Density     |
| Motorola 68000 (M68k)           | 16/32-bit CISC Microprocessor |
| MSP430 / LoongArch              | 16-bit / 64-bit Processors    |
+---------------------------------+-------------------------------+";

/// Disassembly snippet illustrating entrypoint libc main recovery.
const ENTRY_ASM: &str = "\
; authcheck / ping.cgi entry sequence (_start @ 0x400520)
0x400584: lui   a0, 0x40            ; a0 = 0x00400000
0x40058c: jal   __libc_start_main   ; call libc startup stub
0x400590: addiu a0, a0, 0x4c1       ; a0 = 0x004004c1 (&main | bit0)";

/// Decompiled C snippet for ping.cgi command injection.
const PING_DECOMP: &str = "\
void recovered_LAB_004007c4(void) {
    char *host = getenv(\"QUERY_STRING\");
    char cmd[512];
    snprintf(cmd, sizeof(cmd), \"ping -c 1 %s > /tmp/ping.txt\", host);
    system(cmd);
}";

/// Builds the corpus architecture overview section.
///
/// # Returns
///
/// Populated section.
fn overview_section() -> Section {
    let p = "Extracted corpus contains 33 custom MIPS32 binaries (25 CGI + 8 daemons). \
All are stripped, e_shnum = 0 (no section headers), linked against musl.";
    Section::new("Corpus Architecture & Binary Overview")
        .with(Block::Paragraph(p.into()))
        .with(Block::Code(ARCH_BOX.to_string()))
}

/// Builds the entry point recovery section.
///
/// # Returns
///
/// Populated section.
fn recovery_section() -> Section {
    let p = "Because binaries lack section headers and main symbols, REA resolves main by tracing \
the lui+addiu sequence in _start passing &main in a0 to __libc_start_main:";
    Section::new("Entry Point & Libc Main Recovery")
        .with(Block::Paragraph(p.into()))
        .with(Block::Code(ENTRY_ASM.to_string()))
}

/// Builds the binary inventory section.
///
/// # Returns
///
/// Populated section.
/// Inventory row tuples.
const INVENTORY_ROWS: [(&str, &str, &str, &str); 8] = [
    ("ping.cgi", "CGI Utility", "0x004007c4", "65,560 B"),
    ("upnpd", "UPnP Daemon", "0x00400f14", "65,576 B"),
    ("cwmpd", "TR-069 Daemon", "0x00401b30", "65,576 B"),
    ("authcheck", "Auth Daemon", "0x004004d4", "65,560 B"),
    ("openvpn.cgi", "VPN Profile CGI", "0x00400abc", "65,560 B"),
    ("setcfg.cgi", "Config Setter", "0x00400820", "65,560 B"),
    ("view.cgi", "File Viewer", "0x004006e0", "65,560 B"),
    ("admin.cgi", "Admin Portal", "0x00400910", "65,560 B"),
];

/// Formats an inventory entry table row.
///
/// # Arguments
///
/// * `entry` - Entry tuple.
///
/// # Returns
///
/// Table cells.
fn inv_row(&(b, r, m, s): &(&str, &str, &str, &str)) -> Vec<String> {
    vec![b.into(), r.into(), m.into(), s.into()]
}

/// Builds the binary inventory section.
///
/// # Returns
///
/// Populated section.
fn inventory_section() -> Section {
    let rows = INVENTORY_ROWS.iter().map(inv_row).collect();
    let table = text::table(&["Binary", "Role", "Recovered Main", "Size"], rows);
    let desc = "High-priority binaries recovered from SquashFS rootfs:";
    Section::new("Extracted Custom Binaries Inventory")
        .with(Block::Paragraph(desc.into()))
        .with(Block::Table(table))
}

/// Builds the decompiled implementations section.
///
/// # Returns
///
/// Populated section.
fn decomp_section() -> Section {
    Section::new("Decompiled Implementations & Control Flow")
        .with(Block::Paragraph(
            "Recovered C pseudo-code for primary attack surfaces:".into(),
        ))
        .with(Block::Subheading("ping.cgi: Command Injection".into()))
        .with(Block::Code(PING_DECOMP.to_string()))
}

/// Builds the dangerous imports section.
///
/// # Returns
///
/// Populated section.
/// Dangerous system call sink records.
const SINKS_ROWS: [(&str, &str, &str); 4] = [
    (
        "system",
        "Process Execution",
        "ping.cgi, upnpd, cwmpd, openvpn.cgi",
    ),
    ("strcpy", "Unbounded String Copy", "setcfg.cgi, authcheck"),
    ("fopen", "Arbitrary File Access", "view.cgi, getcfg.cgi"),
    (
        "popen",
        "Pipe Command Execution",
        "diagnostics.cgi, remoteassist.cgi",
    ),
];

/// Builds the dangerous imports section.
///
/// # Returns
///
/// Populated section.
fn sinks_section() -> Section {
    let rows: Vec<Vec<String>> =
        SINKS_ROWS.iter().map(|&(a, r, c)| vec![a.into(), r.into(), c.into()]).collect();
    let table = text::table(&["Dangerous API", "Risk Class", "Calling Targets"], rows);
    let desc = "APIs identified as vulnerable sinks by taint analysis:";
    Section::new("Dangerous System Call Sinks & Imports")
        .with(Block::Paragraph(desc.into()))
        .with(Block::Table(table))
}

/// Builds ordered sections for static RE record.
///
/// # Returns
///
/// Vector of record sections.
fn record_sections() -> Vec<Section> {
    vec![
        overview_section(),
        recovery_section(),
        inventory_section(),
        decomp_section(),
        sinks_section(),
    ]
}

/// Composes the complete Static Reverse Engineering Record.
///
/// # Arguments
///
/// * `analysis` - Completed engine analysis.
///
/// # Returns
///
/// The composed static record document.
#[must_use]
pub fn build(analysis: &Analysis) -> Document {
    let title = format!("Static RE Record: {}", analysis.name);
    let mut doc = Document::new(&title, "Binary Disassembly & Symbol Recovery Archive");
    doc.sections = record_sections();
    doc
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies symmetrical architecture box character width.
    #[test]
    fn test_arch_box_width() {
        for line in ARCH_BOX.lines() {
            assert_eq!(line.len(), 67);
        }
    }

    /// Verifies that the static RE record document builds all sections.
    #[test]
    fn test_record_build() {
        let analysis = Analysis { name: "fw.bin".into(), ..Default::default() };
        let doc = build(&analysis);
        assert_eq!(doc.sections.len(), 5);
        assert_eq!(
            doc.sections[0].heading,
            "Corpus Architecture & Binary Overview"
        );
    }
}
