//! Whole-Target Analysis
//!
//! **File:** `analysis.rs`
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

// Import crate items
use crate::{
    asar::{self, AsarTriage},
    capabilities::{self, Capability},
    detect::{detect, Format},
    disasm::{self, Disassembly},
    emulate::{self, Emulation},
    entropy::shannon,
    error::EngineError,
    hashes::{self, Hashes},
    insights::{self, KeyFunction},
    iocs::{self, Indicator},
    loader,
    model::{Arch, Binary, Import},
    multiarch,
    numeric::to_u64,
    packer::{self, Assessment},
    squashfs::{self, FirmwareTriage},
    strings::{self, FoundString},
};

/// Largest target accepted, in bytes.
pub const MAX_INPUT: u64 = 1 << 30;

/// Instruction budget for entry-point emulation.
const EMU_BUDGET: u64 = 20_000;

/// Library suffixes removed when computing an import hash.
const LIBRARY_SUFFIXES: [&str; 3] = [".dll", ".ocx", ".sys"];

/// Everything learned about one target.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Analysis {
    /// File name of the target.
    pub name: String,

    /// File size in bytes.
    pub size: u64,

    /// Detected format.
    pub format: Format,

    /// Cryptographic fingerprints.
    pub hashes: Hashes,

    /// Whole-file entropy in bits per byte.
    pub entropy: f64,

    /// Parsed executable, when the format is native and well formed.
    pub binary: Option<Binary>,

    /// Why a native executable failed to parse.
    pub load_error: Option<String>,

    /// Import hash for PE files.
    pub imphash: Option<String>,

    /// Printable strings.
    pub strings: Vec<FoundString>,

    /// Indicators found in strings.
    pub indicators: Vec<Indicator>,

    /// Capabilities supported by imports.
    pub capabilities: Vec<Capability>,

    /// Packing assessment.
    pub packing: Assessment,

    /// Recovered code structure.
    pub disassembly: Option<Disassembly>,

    /// Functions ranked by analyst relevance.
    pub key_functions: Vec<KeyFunction>,

    /// Entry-point emulation outcome, when a binary was parsed.
    pub emulation: Option<Emulation>,

    /// Firmware root filesystem triage, when a `SquashFS` container was unpacked.
    pub firmware: Option<FirmwareTriage>,

    /// Electron ASAR archive triage, when an ASAR container was unpacked.
    pub asar: Option<AsarTriage>,
}

/// Reads and analyzes a target file.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The completed analysis.
///
/// # Errors
///
/// Returns [`EngineError::Io`] when the file cannot be read and
/// [`EngineError::TooLarge`] when it exceeds [`MAX_INPUT`].
pub fn analyze_path(path: &Path) -> Result<Analysis, EngineError> {
    let io = |e: std::io::Error| EngineError::Io(format!("{}: {e}", path.display()));
    let size = fs::metadata(path).map_err(io)?.len();
    if size > MAX_INPUT {
        return Err(EngineError::TooLarge(size));
    }
    Ok(analyze(&display_name(path), &fs::read(path).map_err(io)?))
}

/// Chooses the display name of a target path.
///
/// # Arguments
///
/// * `path` - Target file path.
///
/// # Returns
///
/// The file name, or the whole path when it has none.
fn display_name(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    name.unwrap_or_else(|| path.display().to_string())
}

/// Analyzes a target held in memory.
///
/// # Arguments
///
/// * `name` - Display name of the target.
/// * `bytes` - Target contents.
///
/// # Returns
///
/// The completed analysis.
#[must_use]
pub fn analyze(name: &str, bytes: &[u8]) -> Analysis {
    let mut analysis = Analysis::identify(name, bytes);
    analysis.load(bytes);
    analysis.strings = strings::extract(bytes);
    analysis.indicators = iocs::scan(&analysis.strings);
    analysis.assess();
    analysis.triage_all(bytes);
    analysis.study(bytes);
    analysis
}

impl Analysis {
    /// Records identity, size, fingerprints, and entropy.
    ///
    /// # Arguments
    ///
    /// * `name` - Display name of the target.
    /// * `bytes` - Target contents.
    ///
    /// # Returns
    ///
    /// An analysis with only identity fields set.
    fn identify(name: &str, bytes: &[u8]) -> Self {
        let mut analysis = Self::default();
        analysis.fingerprint(bytes);
        analysis.name = name.to_string();
        analysis
    }

    /// Records size, format, fingerprints, and entropy.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Target contents.
    fn fingerprint(&mut self, bytes: &[u8]) {
        self.size = to_u64(bytes.len());
        self.format = detect(bytes);
        self.hashes = hashes::compute(bytes);
        self.entropy = shannon(bytes);
    }

    /// Recovers code structure and ranks key functions.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Target contents.
    fn study(&mut self, bytes: &[u8]) {
        let binary = self.binary.as_ref();
        self.disassembly = binary.and_then(|b| recover_code(b, bytes));
        self.key_functions = insights::rank(self, bytes);
        self.emulation = self.emulate_entry(bytes);
    }

    /// Emulates the entry point when a binary with an entry is present.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Target contents.
    ///
    /// # Returns
    ///
    /// The emulation outcome, or `None` when there is no entry to run.
    fn emulate_entry(&self, bytes: &[u8]) -> Option<Emulation> {
        let binary = self.binary.as_ref().filter(|b| b.entry != 0)?;
        Some(emulate::from_entry(binary, bytes, EMU_BUDGET))
    }

    /// Parses native executables and derives the import hash.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Target contents.
    fn load(&mut self, bytes: &[u8]) {
        match loader::load(bytes, self.format) {
            Some(Ok(binary)) => self.binary = Some(with_entropy(binary, bytes)),
            Some(Err(error)) => self.load_error = Some(error.to_string()),
            None => {}
        }
        self.imphash = self.binary.as_ref().and_then(imphash);
    }

    /// Classifies capabilities and assesses packing.
    fn assess(&mut self) {
        let imports = self.binary.as_ref().map_or(&[][..], |b| b.imports.as_slice());
        self.capabilities = capabilities::classify(imports);
        self.packing = packer::assess(self.binary.as_ref(), self.entropy);
    }

    /// Performs firmware filesystem triage if a `SquashFS` rootfs is present.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Target contents.
    fn triage_firmware(&mut self, bytes: &[u8]) {
        if self.format == Format::Squashfs {
            self.firmware = squashfs::triage(bytes);
        } else if let Some(binary) = &self.binary {
            let rootfs = binary.sections.iter().find(|s| s.name == "rootfs_squashfs");
            self.firmware = rootfs.and_then(|s| squashfs::triage(s.data(bytes)));
        }
    }

    /// Performs Electron ASAR archive triage if an ASAR archive is present.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Whole target file.
    fn triage_asar(&mut self, bytes: &[u8]) {
        if self.format == Format::Asar || crate::detect::is_asar(bytes) {
            self.asar = asar::parse(bytes);
        }
    }

    /// Triages embedded filesystem containers.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Whole target file.
    fn triage_all(&mut self, bytes: &[u8]) {
        self.triage_firmware(bytes);
        self.triage_asar(bytes);
    }
}

/// Recovers code structure with the decoder for the binary's architecture.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered code, or `None` when the architecture is unsupported.
fn recover_code(binary: &Binary, bytes: &[u8]) -> Option<Disassembly> {
    match binary.arch {
        Arch::X86 | Arch::X86_64 => disasm::disassemble(binary, bytes),
        _ => multiarch::disassemble(binary, bytes),
    }
}

/// Fills in per-section entropy.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The binary with section entropy set.
fn with_entropy(mut binary: Binary, bytes: &[u8]) -> Binary {
    for section in &mut binary.sections {
        section.entropy = shannon(section.data(bytes));
    }
    binary
}

/// Computes the PE import hash (MD5 of `library.function` pairs).
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The import hash for PE files with imports.
fn imphash(binary: &Binary) -> Option<String> {
    let eligible = binary.format == Format::Pe && !binary.imports.is_empty();
    let entries: Vec<String> = binary.imports.iter().map(imphash_entry).collect();
    eligible.then(|| hashes::md5_hex(entries.join(",").as_bytes()))
}

/// Formats one import-hash entry.
///
/// # Arguments
///
/// * `import` - Imported function.
///
/// # Returns
///
/// Lowercase `library.function`, with ordinals as `ordN`.
fn imphash_entry(import: &Import) -> String {
    let lower = import.library.to_ascii_lowercase();
    let suffix = LIBRARY_SUFFIXES.iter().find_map(|s| lower.strip_suffix(s));
    let library = suffix.unwrap_or(&lower);
    format!("{library}.{}", imphash_name(&import.name))
}

/// Normalizes an imported name for the import hash.
///
/// # Arguments
///
/// * `name` - Imported function name.
///
/// # Returns
///
/// The lowercase name, or `ordN` for ordinal imports.
fn imphash_name(name: &str) -> String {
    match name.strip_prefix("ORDINAL ") {
        Some(number) => format!("ord{number}"),
        None => name.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that unknown data still gets identity and strings.
    #[test]
    fn test_analyze_unknown_data() {
        let analysis = analyze("blob", b"hello analyst, visit https://example.test now");
        assert_eq!(analysis.format, Format::Unknown);
        assert!(analysis.binary.is_none());
        assert_eq!(analysis.indicators[0].kind, "URL");
    }

    /// Verifies that malformed executables record a load error.
    #[test]
    fn test_analyze_records_load_error() {
        let analysis = analyze("bad.exe", b"MZ\x00\x01");
        assert!(analysis.load_error.is_some());
    }

    /// Verifies import-hash entry normalization.
    #[test]
    fn test_imphash_entry_normalizes() {
        let named = Import::new("KERNEL32.dll", "Sleep", 0);
        let ordinal = Import::new("WS2_32.dll", "ORDINAL 115", 0);
        assert_eq!(imphash_entry(&named), "kernel32.sleep");
        assert_eq!(imphash_entry(&ordinal), "ws2_32.ord115");
    }

    /// Verifies that missing files report an I/O error.
    #[test]
    fn test_analyze_path_missing_file() {
        let result = analyze_path(Path::new("definitely/not/here.bin"));
        assert!(matches!(result, Err(EngineError::Io(_))));
    }

    /// Verifies that this test executable analyzes end to end.
    #[test]
    fn test_analyze_path_own_executable() {
        let path = std::env::current_exe().unwrap();
        let analysis = analyze_path(&path).unwrap();
        assert!(analysis.binary.is_some());
        assert!(analysis.disassembly.is_some_and(|d| !d.functions.is_empty()));
    }
}
