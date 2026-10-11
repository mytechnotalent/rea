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

    /// U-Boot legacy image container.
    UImage,

    /// `SquashFS` filesystem.
    Squashfs,

    /// Electron archive (ASAR).
    Asar,

    /// Unrecognized content.
    #[default]
    Unknown,
}

/// Leading-byte signatures checked in order.
const SIGNATURES: [(&[u8], Format); 14] = [
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
];

/// Human-readable format names.
const LABELS: [(Format, &str); 14] = [
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
    (Format::UImage, "U-Boot legacy image (uImage)"),
    (Format::Squashfs, "SquashFS filesystem"),
    (Format::Asar, "Electron Archive (ASAR)"),
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
        _ => format,
    }
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
/// [`Format::JavaClass`] when the architecture count is implausible.
fn refine(bytes: &[u8], format: Format) -> Format {
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
        assert_eq!(detect(b"MZ\x90\x00"), Format::Pe);
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

    /// Verifies labels and native classification.
    #[test]
    fn test_label_and_is_native() {
        assert_eq!(Format::Elf.label(), "Executable and Linkable Format (ELF)");
        assert!(Format::Pe.is_native());
        assert!(!Format::Zip.is_native());
    }
}
