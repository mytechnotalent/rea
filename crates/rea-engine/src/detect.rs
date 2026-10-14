//! File Format Detection
//!
//! **File:** `detect.rs`
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

/// File formats recognized by signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// Windows Portable Executable.
    Pe,

    /// Executable and Linkable Format.
    Elf,

    /// Single-architecture Mach-O.
    MachO,

    /// Multi-architecture Mach-O container.
    MachOUniversal,

    /// Java class file.
    JavaClass,

    /// ZIP container, including APK and JAR packages.
    Zip,

    /// Android Dalvik executable.
    Dex,

    /// Script with an interpreter line.
    Script,

    /// PDF document.
    Pdf,

    /// Raw ARM Cortex-M firmware image (headerless, vector table at offset 0).
    CortexM,

    /// Raw ARM-mode firmware with a boot preamble.
    ArmFirmware,

    /// U-Boot legacy image container.
    UImage,

    /// `SquashFS` filesystem.
    Squashfs,

    /// Electron archive (ASAR).
    Asar,

    /// WebAssembly binary module.
    Wasm,

    /// SQLite database file.
    Sqlite,

    /// Apple property list.
    Plist,

    /// Packet capture file.
    Pcap,

    /// MS-DOS MZ executable.
    Dos,

    /// Unrecognized content.
    #[default]
    Unknown,
}

/// Leading-byte signatures checked in order.
const SIGNATURES: [(&[u8], Format); 18] = [
    (b"MZ", Format::Pe),
    (b"\x7fELF", Format::Elf),
    (&[0xfe, 0xed, 0xfa, 0xce], Format::MachO),
    (&[0xce, 0xfa, 0xed, 0xfe], Format::MachO),
    (&[0xfe, 0xed, 0xfa, 0xcf], Format::MachO),
    (&[0xcf, 0xfa, 0xed, 0xfe], Format::MachO),
    (&[0xca, 0xfe, 0xba, 0xbe], Format::MachOUniversal),
    (b"PK\x03\x04", Format::Zip),
    (b"dex\n", Format::Dex),
    (b"#!", Format::Script),
    (b"%PDF", Format::Pdf),
    (&[0x27, 0x05, 0x19, 0x56], Format::UImage),
    (b"hsqs", Format::Squashfs),
    (b"sqsh", Format::Squashfs),
    (b"\x00asm", Format::Wasm),
    (b"SQLite format 3\x00", Format::Sqlite),
    (b"bplist00", Format::Plist),
    (b"\xd4\xc3\xb2\xa1", Format::Pcap),
];

/// Human-readable format names.
const LABELS: [(Format, &str); 20] = [
    (Format::Pe, "Portable Executable (PE)"),
    (Format::Elf, "Executable and Linkable Format (ELF)"),
    (Format::MachO, "Mach-O"),
    (Format::MachOUniversal, "Mach-O universal binary"),
    (Format::JavaClass, "Java class file"),
    (Format::Zip, "ZIP container"),
    (Format::Dex, "Dalvik executable (DEX)"),
    (Format::Script, "Script"),
    (Format::Pdf, "PDF document"),
    (Format::CortexM, "Raw ARM Cortex-M firmware"),
    (Format::ArmFirmware, "Raw ARM firmware"),
    (Format::UImage, "U-Boot legacy image (uImage)"),
    (Format::Squashfs, "SquashFS filesystem"),
    (Format::Asar, "Electron Archive (ASAR)"),
    (Format::Wasm, "WebAssembly module"),
    (Format::Sqlite, "SQLite database"),
    (Format::Plist, "Apple property list"),
    (Format::Pcap, "Packet capture (pcap)"),
    (Format::Dos, "MS-DOS executable"),
    (Format::Unknown, "Unrecognized data"),
];

/// Largest architecture count accepted before `0xCAFEBABE` is read as Java.
const MAX_FAT_ARCHES: u32 = 30;

impl Format {
    /// Returns the human-readable name of the format.
    ///
    /// # Returns
    ///
    /// A descriptive label.
    #[must_use]
    pub fn label(self) -> &'static str {
        let found = LABELS.iter().find(|(format, _)| *format == self);
        found.map_or("Unrecognized data", |(_, label)| label)
    }

    /// Reports whether the format is a native executable the engine loads.
    ///
    /// # Returns
    ///
    /// `true` for PE, ELF, and Mach-O.
    #[must_use]
    pub fn is_native(self) -> bool {
        let all = [Self::Pe, Self::Elf, Self::MachO, Self::MachOUniversal];
        all.contains(&self) || self == Self::CortexM || self == Self::UImage
    }
}

/// Identifies the format of a buffer from its leading bytes.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// The detected format, or [`Format::Unknown`].
#[must_use]
pub fn detect(bytes: &[u8]) -> Format {
    let found = SIGNATURES.iter().find(|(magic, _)| bytes.starts_with(magic));
    let format = refine(bytes, found.map_or(Format::Unknown, |(_, format)| *format));
    check_fallback(bytes, format)
}

/// Fallback detection for headerless or structured container formats.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `format` - Initial format classification.
///
/// # Returns
///
/// Refined format or unchanged input format.
fn check_fallback(bytes: &[u8], format: Format) -> Format {
    match format {
        Format::Unknown if is_cortex_m(bytes) => Format::CortexM,
        Format::Unknown if is_asar(bytes) => Format::Asar,
        Format::Unknown if crate::plist::is_xml_plist(bytes) => Format::Plist,
        Format::Unknown if arm_entry(bytes).is_some() => Format::ArmFirmware,
        _ => format,
    }
}

/// Locates the ARM-mode boot preamble of a raw firmware image.
///
/// An ARM reset preamble sets a processor mode and stack for each mode with a
/// repeated `mov r0, #imm` / `msr cpsr` pair, so the first such pair marks the
/// start of executable code.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The byte offset of the boot code, or `None`.
#[must_use]
pub fn arm_entry(bytes: &[u8]) -> Option<usize> {
    let limit = bytes.len().min(0x400);
    (0..limit).step_by(4).find(|offset| is_arm_boot(bytes, *offset))
}

/// Reports whether ARM boot code begins at an offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `offset` - Candidate 4-byte-aligned offset.
///
/// # Returns
///
/// `true` for a `mov r0, #imm` followed by two or more `msr cpsr` words.
fn is_arm_boot(bytes: &[u8], offset: usize) -> bool {
    let Some(first) = le_u32(bytes, offset) else {
        return false;
    };
    let pairs = (1..8)
        .filter_map(|i| le_u32(bytes, offset + i * 4))
        .filter(|w| is_msr_cpsr(*w))
        .count();
    is_mov_imm(first) && pairs >= 2
}

/// Reports whether a word is an unconditional `mov r0, #imm`.
///
/// # Arguments
///
/// * `word` - Instruction word.
///
/// # Returns
///
/// `true` for the `mov r0, #imm` encoding.
fn is_mov_imm(word: u32) -> bool {
    word >> 28 == 0xE && (word & 0x0FE0_0000) == 0x03A0_0000
}

/// Reports whether a word is a `msr cpsr, rN` instruction.
///
/// # Arguments
///
/// * `word` - Instruction word.
///
/// # Returns
///
/// `true` for the conditional `msr cpsr` encoding.
fn is_msr_cpsr(word: u32) -> bool {
    word >> 28 == 0xE && (word & 0x0FFF_FFF0) == 0x0121_F000
}

/// Reports whether a buffer is an Electron ASAR archive.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// `true` when the header matches an ASAR pickle header.
#[must_use]
pub fn is_asar(bytes: &[u8]) -> bool {
    bytes.len() >= 24 && bytes.starts_with(&[4, 0, 0, 0]) && bytes[16..].starts_with(b"{\"files\"")
}

/// Reports whether a headerless buffer looks like Cortex-M firmware.
///
/// A Cortex-M image begins with a vector table: an initial stack pointer in
/// SRAM followed by an odd (Thumb) reset handler inside the image.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// `true` when the first two vectors are plausible.
#[must_use]
pub fn is_cortex_m(bytes: &[u8]) -> bool {
    let (Some(sp), Some(reset)) = (le_u32(bytes, 0), le_u32(bytes, 4)) else {
        return false;
    };
    plausible_stack(sp) && plausible_reset(reset, bytes.len())
}

/// Reports whether a word is a plausible initial stack pointer.
///
/// # Arguments
///
/// * `sp` - First vector-table word.
///
/// # Returns
///
/// `true` for an aligned pointer into a RAM-like region.
fn plausible_stack(sp: u32) -> bool {
    let region = (0x2000_0000..=0x3000_0000).contains(&sp);
    sp.trailing_zeros() >= 2 && region
}

/// Reports whether a word is a plausible reset handler within the image.
///
/// # Arguments
///
/// * `reset` - Second vector-table word.
/// * `len` - File length in bytes.
///
/// # Returns
///
/// `true` when the handler is Thumb and lies inside the loaded image.
fn plausible_reset(reset: u32, len: usize) -> bool {
    let base = reset & 0xFFF0_0000;
    let offset = (reset & !1).wrapping_sub(base);
    reset & 1 == 1 && base != 0 && (offset as usize) < len
}

/// Reads a little-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, when four bytes are available.
fn le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let word = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}

/// Separates Java class files from universal Mach-O, which share a magic.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
/// * `format` - Signature-based guess.
///
/// # Returns
///
/// The refined format.
fn refine(bytes: &[u8], format: Format) -> Format {
    if format == Format::Pe && !crate::dos::is_pe_image(bytes) {
        return Format::Dos;
    }
    refine_fat(bytes, format)
}

/// Separates Java class files from universal Mach-O, which share a magic.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
/// * `format` - Signature-based guess.
///
/// # Returns
///
/// [`Format::JavaClass`] when the architecture count is implausible.
fn refine_fat(bytes: &[u8], format: Format) -> Format {
    let fat = format == Format::MachOUniversal;
    let java = fat && be_u32(bytes, 4).is_some_and(|c| c > MAX_FAT_ARCHES);
    if java {
        Format::JavaClass
    } else {
        format
    }
}

/// Reads a big-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, when four bytes are available.
fn be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let word = bytes.get(offset..offset + 4)?;
    Some(u32::from_be_bytes([word[0], word[1], word[2], word[3]]))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies detection of the main executable formats.
    #[test]
    fn test_detect_executables() {
        assert_eq!(detect(b"MZ\x90\x00"), Format::Dos);
        assert_eq!(detect(b"\x7fELF\x02"), Format::Elf);
        assert_eq!(detect(&[0xcf, 0xfa, 0xed, 0xfe]), Format::MachO);
    }

    /// Verifies detection of `uImage` and `SquashFS` container signatures.
    #[test]
    fn test_detect_containers() {
        assert_eq!(detect(&[0x27, 0x05, 0x19, 0x56, 0, 0]), Format::UImage);
        assert_eq!(detect(b"hsqs\x00\x00"), Format::Squashfs);
        assert_eq!(detect(b"sqsh\x00\x00"), Format::Squashfs);
    }

    /// Verifies that `0xCAFEBABE` is split between Mach-O and Java.
    #[test]
    fn test_detect_cafebabe_disambiguation() {
        assert_eq!(
            detect(&[0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 2]),
            Format::MachOUniversal
        );
        assert_eq!(
            detect(&[0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 52]),
            Format::JavaClass
        );
    }

    /// Verifies that unknown data is reported as such.
    #[test]
    fn test_detect_unknown() {
        assert_eq!(detect(b"hello"), Format::Unknown);
        assert_eq!(detect(b""), Format::Unknown);
    }

    /// Verifies WebAssembly detection.
    #[test]
    fn test_detect_wasm() {
        assert_eq!(detect(b"\x00asm\x01\x00\x00\x00"), Format::Wasm);
    }

    /// Verifies SQLite detection.
    #[test]
    fn test_detect_sqlite() {
        assert_eq!(detect(b"SQLite format 3\x00"), Format::Sqlite);
    }

    /// Verifies property list detection.
    #[test]
    fn test_detect_plist() {
        assert_eq!(detect(b"bplist00\xd1\x01"), Format::Plist);
        assert_eq!(
            detect(b"<?xml version=\"1.0\"?><plist version=\"1.0\">"),
            Format::Plist
        );
    }

    /// Verifies labels and native classification.
    #[test]
    fn test_label_and_is_native() {
        assert_eq!(Format::Elf.label(), "Executable and Linkable Format (ELF)");
        assert!(Format::Pe.is_native());
        assert!(!Format::Zip.is_native());
    }
}
