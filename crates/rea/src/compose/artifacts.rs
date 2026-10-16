//! Artifact Format Findings
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

// Import dependencies from rea_engine
use rea_engine::Analysis;

// Import dependencies from rea_report
use rea_report::Section;

// Import crate::compose::text
use crate::compose::text;

/// Builds the artifact format sections.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// One section per artifact format that parsed.
pub fn sections(_analysis: &Analysis, bytes: &[u8]) -> Vec<Section> {
    let mut out = Vec::new();
    artifact_group_a(&mut out, bytes);
    artifact_group_b(&mut out, bytes);
    out
}

/// Appends the first group of artifact sections.
///
/// # Arguments
///
/// * `out` - Accumulated sections.
/// * `bytes` - Raw target bytes.
fn artifact_group_a(out: &mut Vec<Section>, bytes: &[u8]) {
    push_section(out, "WebAssembly", wasm_note(bytes));
    push_section(out, "SQLite Database", sqlite_note(bytes));
    push_section(out, "EVM Bytecode", evm_note(bytes));
    push_section(out, "Property List", plist_note(bytes));
    push_section(out, "Java Class", java_note(bytes));
}

/// Appends the second group of artifact sections.
///
/// # Arguments
///
/// * `out` - Accumulated sections.
/// * `bytes` - Raw target bytes.
fn artifact_group_b(out: &mut Vec<Section>, bytes: &[u8]) {
    push_section(out, "Managed .NET Assembly", dotnet_note(bytes));
    push_section(out, "Go Binary", go_note(bytes));
    push_section(out, "Android Resources", arsc_note(bytes));
    push_section(out, "DWARF Debug Info", dwarf_note(bytes));
    push_section(out, "MS-DOS Executable", dos_note(bytes));
    push_section(out, "Network Capture", capture_note(bytes));
}

/// Builds the managed .NET assembly note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn dotnet_note(bytes: &[u8]) -> Option<String> {
    let meta = rea_engine::dotnet::parse_dotnet_metadata(bytes)?;
    let tail = refs_tail(&meta.references);
    let text = format!(
        "Managed .NET assembly (runtime {}): {} types, {} methods{tail}.",
        meta.runtime_version, meta.type_count, meta.method_count
    );
    Some(text)
}

/// Summarizes assembly references and their public-key tokens.
///
/// # Arguments
///
/// * `refs` - Assembly references.
///
/// # Returns
///
/// The summary tail text.
fn refs_tail(refs: &[rea_engine::dotnet::DotNetAssemblyRef]) -> String {
    let tokens = refs.iter().filter(|r| !r.token.is_empty()).count();
    format!(", {} references, {tokens} tokens", refs.len())
}

/// Builds the Go binary note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn go_note(bytes: &[u8]) -> Option<String> {
    let go = rea_engine::go::parse_go_metadata(bytes)?;
    Some(format!(
        "Go binary ({}): package {}, {} dependencies.",
        go.version,
        go.module_path,
        go.dependencies.len()
    ))
}

/// Appends a section when a note was produced.
///
/// # Arguments
///
/// * `out` - Accumulated sections.
/// * `title` - Section heading.
/// * `note` - Section note, when the format parsed.
fn push_section(out: &mut Vec<Section>, title: &str, note: Option<String>) {
    if let Some(text) = note {
        out.push(Section::new(title).with(text::paragraph(&text)));
    }
}

/// Builds the WebAssembly note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn wasm_note(bytes: &[u8]) -> Option<String> {
    let m = rea_engine::wasm::parse_wasm(bytes)?;
    let v = m.version;
    let f = m.functions;
    let im = m.imports.len();
    let ex = m.exports.len();
    let text = format!("WASM v{v}: {f} functions, {im} imports, {ex} exports.");
    Some(text)
}

/// Builds the SQLite note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn sqlite_note(bytes: &[u8]) -> Option<String> {
    let db = rea_engine::sqlite::parse_sqlite(bytes)?;
    Some(format!(
        "SQLite database with {} schema objects and {} sampled rows.",
        db.objects.len(),
        db.rows.len()
    ))
}

/// Builds the EVM note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn evm_note(bytes: &[u8]) -> Option<String> {
    if !rea_engine::evm::is_hex_text(bytes) {
        return None;
    }
    let text = String::from_utf8_lossy(bytes);
    let code = rea_engine::evm::decode_hex(&text)?;
    Some(evm_summary(&rea_engine::evm::parse_evm(&code)))
}

/// Describes EVM bytecode behavior.
///
/// # Arguments
///
/// * `evm` - Recovered EVM bytecode.
///
/// # Returns
///
/// The note text.
fn evm_summary(evm: &rea_engine::evm::EvmBytecode) -> String {
    let mutability = class_of(&evm.mutability);
    let len = evm.length;
    let sel = evm.selectors.len();
    let feat = evm.features.len();
    format!("EVM: {len}B, {sel} selectors, {mutability}, {feat} features.")
}

/// Returns `unknown` for an empty string.
///
/// # Arguments
///
/// * `value` - Candidate value.
///
/// # Returns
///
/// The value, or `unknown`.
fn class_of(value: &str) -> &str {
    if value.is_empty() {
        "unknown"
    } else {
        value
    }
}

/// Builds the property list note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn plist_note(bytes: &[u8]) -> Option<String> {
    let plist = rea_engine::plist::parse_plist(bytes)?;
    let bundle = rea_engine::plist::analyze_bundle(&plist);
    Some(format!(
        "{} property list with {} keys; identifier {}.",
        plist.format,
        plist.entries.len(),
        bundle.identifier
    ))
}

/// Builds the Java class note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn java_note(bytes: &[u8]) -> Option<String> {
    let class = rea_engine::jvm::parse_class(bytes)?;
    let name = &class.this_class;
    let version = format!("v{}.{}", class.major, class.minor);
    let (methods, fields) = members(&class);
    let text = format!("Java {name} {version}: {methods} methods, {fields} fields.");
    Some(text)
}

/// Counts a Java class's methods and fields.
///
/// # Arguments
///
/// * `class` - Parsed Java class.
///
/// # Returns
///
/// The `(methods, fields)` counts.
fn members(class: &rea_engine::jvm::JavaClass) -> (usize, usize) {
    (class.methods.len(), class.fields.len())
}

/// Builds the Android resources note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn arsc_note(bytes: &[u8]) -> Option<String> {
    let table = rea_engine::arsc::parse_arsc(bytes)?;
    Some(format!(
        "Android resource table with {} packages and {} strings.",
        table.package_count,
        table.strings.len()
    ))
}

/// Builds the DWARF note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn dwarf_note(bytes: &[u8]) -> Option<String> {
    let dwarf = rea_engine::dwarf::parse_dwarf(bytes)?;
    Some(format!(
        "DWARF v{} with {} compile units and {} functions.",
        dwarf.version,
        dwarf.compile_units.len(),
        dwarf.functions.len()
    ))
}

/// Builds the MS-DOS note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn dos_note(bytes: &[u8]) -> Option<String> {
    let header = rea_engine::dos::parse_dos(bytes)?;
    Some(format!(
        "MS-DOS MZ executable entering at {:04x}:{:04x}.",
        header.cs, header.ip
    ))
}

/// Builds the network capture note.
///
/// # Arguments
///
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The note, or `None`.
fn capture_note(bytes: &[u8]) -> Option<String> {
    if let Some(pcap) = rea_engine::netcap::parse_pcap(bytes) {
        return Some(pcap_note(&pcap));
    }
    let har = rea_engine::netcap::parse_har(bytes)?;
    Some(har_note(&har))
}

/// Formats a PCAP capture note.
///
/// # Arguments
///
/// * `pcap` - Recovered capture.
///
/// # Returns
///
/// The note.
fn pcap_note(pcap: &rea_engine::netcap::Pcap) -> String {
    format!("PCAP capture with {} packets.", pcap.packet_count)
}

/// Formats a HAR capture note.
///
/// # Arguments
///
/// * `har` - Recovered capture.
///
/// # Returns
///
/// The note.
fn har_note(har: &rea_engine::netcap::Har) -> String {
    format!("HAR capture with {} entries.", har.entry_count)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal analysis for section tests.
    ///
    /// # Returns
    ///
    /// The analyzed record.
    fn analysis() -> Analysis {
        rea_engine::analyze("target", b"hello")
    }

    /// Verifies that plain data yields no artifact sections.
    #[test]
    fn test_artifacts_skip_plain_data() {
        assert_eq!(
            sections(&analysis(), b"plain data with no artifacts").len(),
            0
        );
    }

    /// Verifies that a WebAssembly module adds a section.
    #[test]
    fn test_artifacts_wasm() {
        let module = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let found = sections(&analysis(), &module);
        assert!(found.iter().any(|s| s.heading == "WebAssembly"));
    }
}
