//! Packing Assessment
//!
//! **File:** `packer.rs`
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

// Import crate::model
use crate::model::{Binary, Section};

/// Score at which packing is reported as likely.
pub const LIKELY_SCORE: u32 = 3;

/// Entropy, in bits per byte, above which data looks compressed or encrypted.
pub const HIGH_ENTROPY: f64 = 7.2;

/// Import count below which a table counts as minimal.
const THIN_IMPORTS: usize = 10;

/// Section names left behind by well-known packers and protectors.
const PACKER_SECTIONS: [(&str, &str); 14] = [
    ("UPX0", "UPX"),
    ("UPX1", "UPX"),
    ("UPX2", "UPX"),
    (".aspack", "ASPack"),
    (".adata", "ASPack"),
    ("MPRESS1", "MPRESS"),
    ("MPRESS2", "MPRESS"),
    (".petite", "Petite"),
    (".themida", "Themida"),
    (".vmp0", "VMProtect"),
    (".vmp1", "VMProtect"),
    (".enigma1", "Enigma Protector"),
    ("PEC2", "PECompact"),
    (".nsp0", "NsPack"),
];

/// A weighted reason contributing to the packing score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signal {
    /// Contribution to the score.
    pub weight: u32,

    /// Evidence-backed explanation.
    pub reason: String,
}

/// Packing and obfuscation verdict.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Assessment {
    /// Whether packing is likely.
    pub likely: bool,

    /// Sum of signal weights.
    pub score: u32,

    /// Signals that fired.
    pub signals: Vec<Signal>,
}

impl Assessment {
    /// Builds a verdict from its signals.
    ///
    /// # Arguments
    ///
    /// * `signals` - Signals that fired.
    ///
    /// # Returns
    ///
    /// The assessment with its score and likelihood.
    #[must_use]
    pub fn new(signals: Vec<Signal>) -> Self {
        let score = signals.iter().map(|s| s.weight).sum();
        let likely = score >= LIKELY_SCORE;
        Self { likely, score, signals }
    }
}

/// Binary-level checks run when the target parsed as an executable.
const CHECKS: [fn(&Binary) -> Vec<Signal>; 7] = [
    packer_sections,
    packed_code,
    entry_placement,
    thin_imports,
    unusual_section_names,
    writable_executable,
    crowded_sections,
];

/// Section-name substrings that identify specific protectors.
const PROTECTOR_NAMES: [(&str, &str); 3] = [
    (".binsafe", "Alcatraz"),
    (".themida", "Themida/WinLicense"),
    (".enigma", "Enigma Protector"),
];

/// Section count above which the layout is treated as unusual.
const CROWDED_SECTIONS: usize = 16;

/// Assesses whether the target is packed, compressed, or encrypted.
///
/// # Arguments
///
/// * `binary` - Parsed executable, when available.
/// * `file_entropy` - Entropy of the whole file.
///
/// # Returns
///
/// The weighted assessment.
#[must_use]
pub fn assess(binary: Option<&Binary>, file_entropy: f64) -> Assessment {
    let mut signals = file_signal(file_entropy);
    signals.extend(binary.map(binary_signals).unwrap_or_default());
    Assessment::new(signals)
}

/// Runs every binary-level check.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// All signals from all checks.
fn binary_signals(binary: &Binary) -> Vec<Signal> {
    CHECKS.iter().flat_map(|check| check(binary)).collect()
}

/// Builds zero or one signal.
///
/// # Arguments
///
/// * `fired` - Whether the check fired.
/// * `weight` - Score contribution.
/// * `reason` - Explanation.
///
/// # Returns
///
/// The signal when `fired`, otherwise nothing.
fn signal_if(fired: bool, weight: u32, reason: String) -> Vec<Signal> {
    let signal = Signal { weight, reason };
    if fired {
        vec![signal]
    } else {
        Vec::new()
    }
}

/// Flags near-random whole-file entropy.
///
/// # Arguments
///
/// * `entropy` - Whole-file entropy.
///
/// # Returns
///
/// One signal when entropy is high.
fn file_signal(entropy: f64) -> Vec<Signal> {
    let reason = format!("Whole-file entropy is {entropy:.2} bits/byte, typical of packed data");
    signal_if(entropy > HIGH_ENTROPY, 2, reason)
}

/// Flags section names left by known packers.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal per matching section.
fn packer_sections(binary: &Binary) -> Vec<Signal> {
    let hits = binary.sections.iter().filter_map(|s| packer_of(&s.name));
    hits.flat_map(|(name, packer)| signal_if(true, 3, packer_reason(name, packer)))
        .collect()
}

/// Finds the packer associated with a section name.
///
/// # Arguments
///
/// * `name` - Section name.
///
/// # Returns
///
/// The matching `(section, packer)` pair, when any.
fn packer_of(name: &str) -> Option<(&'static str, &'static str)> {
    PACKER_SECTIONS.iter().find(|(section, _)| *section == name).copied()
}

/// Explains a packer section match.
///
/// # Arguments
///
/// * `name` - Section name.
/// * `packer` - Packer name.
///
/// # Returns
///
/// The explanation.
fn packer_reason(name: &str, packer: &str) -> String {
    format!("Section `{name}` is characteristic of {packer}")
}

/// Flags non-standard section names, including named protectors.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal per non-ASCII or protector-named section.
fn unusual_section_names(binary: &Binary) -> Vec<Signal> {
    let named = binary.sections.iter().flat_map(|s| protector_signal(&s.name));
    let odd = binary.sections.iter().filter(|s| has_odd_name(&s.name));
    let emoji = odd.flat_map(|s| signal_if(true, 3, odd_reason(&s.name)));
    named.chain(emoji).collect()
}

/// Names the protector a section belongs to, if known.
///
/// # Arguments
///
/// * `name` - Section name.
///
/// # Returns
///
/// One signal naming the protector when the section matches.
fn protector_signal(name: &str) -> Vec<Signal> {
    let found = PROTECTOR_NAMES.iter().find(|(section, _)| *section == name);
    let reason = |p| format!("Section `{name}` is the hallmark of the {p} protector");
    found.into_iter().flat_map(|(_, p)| signal_if(true, 4, reason(p))).collect()
}

/// Reports whether a section name contains non-standard characters.
///
/// # Arguments
///
/// * `name` - Section name.
///
/// # Returns
///
/// `true` when any character is not ASCII graphic or space.
fn has_odd_name(name: &str) -> bool {
    name.chars().any(|c| !c.is_ascii_graphic() && c != ' ')
}

/// Explains a non-standard section name.
///
/// # Arguments
///
/// * `name` - Section name.
///
/// # Returns
///
/// The explanation, escaping the raw name.
fn odd_reason(name: &str) -> String {
    let shown: String = name.chars().map(escape_char).collect();
    format!("Section name `{shown}` contains non-standard characters, typical of a protector")
}

/// Renders a character literally when graphic, otherwise as a Unicode escape.
///
/// # Arguments
///
/// * `c` - Character from a section name.
///
/// # Returns
///
/// The character, or `U+XXXX` for non-graphic characters.
fn escape_char(c: char) -> String {
    if c.is_ascii_graphic() {
        c.to_string()
    } else {
        format!("U+{:04X}", c as u32)
    }
}

/// Flags sections that are both writable and executable.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal per writable-executable section.
fn writable_executable(binary: &Binary) -> Vec<Signal> {
    let wx = binary.sections.iter().filter(|s| s.executable && s.writable);
    let reason = |s: &Section| format!("Section `{}` is writable and executable", s.name);
    wx.flat_map(|s| signal_if(true, 2, reason(s))).collect()
}

/// Flags an unusually high section count.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal when the section count exceeds the threshold.
fn crowded_sections(binary: &Binary) -> Vec<Signal> {
    let count = binary.sections.len();
    let reason =
        format!("{count} sections is unusually many, typical of a protector's added segments");
    signal_if(count > CROWDED_SECTIONS, 1, reason)
}

/// Flags executable sections whose data looks compressed or encrypted.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal per high-entropy executable section.
fn packed_code(binary: &Binary) -> Vec<Signal> {
    let code = binary.sections.iter().filter(|s| s.executable);
    let hot = code.filter(|s| s.entropy > HIGH_ENTROPY);
    hot.flat_map(|s| signal_if(true, 2, packed_reason(s))).collect()
}

/// Explains a high-entropy executable section.
///
/// # Arguments
///
/// * `section` - Section that fired.
///
/// # Returns
///
/// The explanation.
fn packed_reason(section: &Section) -> String {
    let (name, entropy) = (&section.name, section.entropy);
    format!("Executable section `{name}` has entropy {entropy:.2} bits/byte")
}

/// Flags entry points in writable or unmapped locations.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal when the entry point placement is unusual.
fn entry_placement(binary: &Binary) -> Vec<Signal> {
    let problem = entry_problem(binary.entry, binary.section_at(binary.entry));
    problem.map_or_else(Vec::new, |reason| signal_if(true, 2, reason))
}

/// Explains an unusual entry point location.
///
/// # Arguments
///
/// * `entry` - Entry-point address.
/// * `section` - Section containing it, if any.
///
/// # Returns
///
/// The explanation, when the placement is unusual.
fn entry_problem(entry: u64, section: Option<&Section>) -> Option<String> {
    match section {
        None if entry != 0 => Some(format!("Entry point 0x{entry:x} is outside every section")),
        Some(s) if s.writable => Some(format!("Entry point is in writable section `{}`", s.name)),
        _ => None,
    }
}

/// Flags tiny import tables that resolve everything at run time.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// One signal when the import table is minimal but includes a resolver.
fn thin_imports(binary: &Binary) -> Vec<Signal> {
    let count = binary.imports.len();
    let resolver = binary.imports.iter().any(|i| is_resolver(&i.name));
    let reason = format!("Only {count} imports, including a run-time API resolver");
    signal_if(count < THIN_IMPORTS && resolver, 2, reason)
}

/// Reports whether an import resolves other APIs at run time.
///
/// # Arguments
///
/// * `name` - Imported function name.
///
/// # Returns
///
/// `true` for `GetProcAddress` and `dlsym`.
fn is_resolver(name: &str) -> bool {
    name.starts_with("GetProcAddress") || name == "dlsym"
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::model
    use crate::model::Import;

    /// Verifies that UPX section names make packing likely.
    #[test]
    fn test_assess_detects_upx() {
        let sections = vec![Section::mapped("UPX0", 0x1000, 0x100)];
        let binary = Binary { sections, ..Binary::default() };
        let verdict = assess(Some(&binary), 5.0);
        assert!(verdict.likely);
        assert!(verdict.signals[0].reason.contains("UPX"));
    }

    /// Verifies that ordinary binaries are not flagged.
    #[test]
    fn test_assess_clean_binary() {
        let text = Section::mapped(".text", 0x1000, 0x100).with_access(true, false);
        let binary = Binary { sections: vec![text], entry: 0x1000, ..Binary::default() };
        assert!(!assess(Some(&binary), 6.0).likely);
    }

    /// Verifies the minimal-import signal.
    #[test]
    fn test_thin_imports_with_resolver() {
        let imports = vec![Import::new("k", "GetProcAddress", 0)];
        let binary = Binary { imports, ..Binary::default() };
        assert_eq!(thin_imports(&binary).len(), 1);
    }

    /// Verifies the whole-file entropy signal threshold.
    #[test]
    fn test_file_signal_threshold() {
        assert_eq!(file_signal(7.9).len(), 1);
        assert_eq!(file_signal(6.0).len(), 0);
    }

    /// Verifies entry placement explanations.
    #[test]
    fn test_entry_problem_cases() {
        let writable = Section::mapped(".data", 0, 1).with_access(false, true);
        assert!(entry_problem(0x10, None).is_some());
        assert!(entry_problem(0, None).is_none());
        assert!(entry_problem(1, Some(&writable)).is_some());
    }

    /// Verifies that an Alcatraz section and an emoji section are flagged.
    #[test]
    fn test_unusual_section_names_detects_protector() {
        let odd = Section::mapped("\u{1f480}", 0x1000, 0x10);
        let sections = vec![Section::mapped(".binsafe", 0, 1), odd];
        let signals = unusual_section_names(&Binary { sections, ..Default::default() });
        let has = |t: &str| signals.iter().any(|s| s.reason.contains(t));
        assert!(has("Alcatraz") && has("U+1F480"));
    }

    /// Verifies the writable-executable and crowded-section checks.
    #[test]
    fn test_wx_and_crowded_sections() {
        let wx = Section::mapped(".x", 0, 1).with_access(true, true);
        let mut binary = Binary { sections: vec![wx], ..Binary::default() };
        assert_eq!(writable_executable(&binary).len(), 1);
        binary.sections = (0..20).map(|i| Section::mapped(".s", i, 1)).collect();
        assert_eq!(crowded_sections(&binary).len(), 1);
    }
}
