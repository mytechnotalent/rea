//! U-Boot Legacy Image Loader
//!
//! **File:** `uimage.rs`
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

// Import crate items
use crate::{
    detect::Format,
    error::EngineError,
    model::{Arch, Binary, Fact, Section},
};

/// Reads a big-endian 32-bit unsigned integer from a byte slice.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Byte offset to read from.
///
/// # Returns
///
/// The 32-bit integer, or `None` if out of bounds.
fn be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Architecture mappings for U-Boot architecture identifiers.
const UIMAGE_ARCHES: [(u8, Arch); 12] = [
    (2, Arch::Arm),
    (3, Arch::X86),
    (5, Arch::Mips),
    (6, Arch::Mips64),
    (7, Arch::PowerPc),
    (9, Arch::SuperH),
    (10, Arch::Sparc),
    (12, Arch::M68k),
    (22, Arch::Arm64),
    (24, Arch::X86_64),
    (25, Arch::Riscv),
    (28, Arch::Xtensa),
];

/// Resolves a U-Boot architecture byte to an engine architecture.
///
/// # Arguments
///
/// * `arch` - Architecture byte.
///
/// # Returns
///
/// The matched architecture.
fn arch_of(arch: u8) -> Arch {
    let found = UIMAGE_ARCHES.iter().find(|(a, _)| *a == arch);
    found.map_or(Arch::Other, |(_, a)| *a)
}

/// Returns the operating system name for a U-Boot OS code.
///
/// # Arguments
///
/// * `os` - OS byte code.
///
/// # Returns
///
/// Operating system label.
fn os_name(os: u8) -> &'static str {
    match os {
        5 => "Linux",
        1 => "OpenBSD",
        2 => "NetBSD",
        3 => "FreeBSD",
        17 => "U-Boot",
        _ => "Unknown OS",
    }
}

/// Returns the compression algorithm name for a U-Boot compression code.
///
/// # Arguments
///
/// * `comp` - Compression byte code.
///
/// # Returns
///
/// Compression name.
fn comp_name(comp: u8) -> &'static str {
    match comp {
        0 => "none",
        1 => "gzip",
        2 => "bzip2",
        3 => "lzma",
        4 => "lzo",
        _ => "other",
    }
}

/// Returns the image type description for a U-Boot type code.
///
/// # Arguments
///
/// * `type_code` - Type byte code.
///
/// # Returns
///
/// Image type name.
fn type_name(type_code: u8) -> &'static str {
    match type_code {
        2 => "OS Kernel Image",
        1 => "Standalone Program",
        3 => "RAMDisk Image",
        5 => "Firmware",
        _ => "Other Image",
    }
}

/// Extracts the image name string from the 32-byte header field.
///
/// # Arguments
///
/// * `bytes` - Header bytes.
///
/// # Returns
///
/// Extracted UTF-8 string.
fn image_name(bytes: &[u8]) -> String {
    let slice = bytes.get(32..64).unwrap_or(&[]);
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).into_owned()
}

/// Verifies that a buffer begins with a valid U-Boot legacy header.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
///
/// # Returns
///
/// `true` when length is at least 64 and magic matches.
fn is_valid_uimage(bytes: &[u8]) -> bool {
    bytes.len() >= 64 && bytes.starts_with(&[0x27, 0x05, 0x19, 0x56])
}

/// Extracts header metadata tuple from header bytes.
///
/// # Arguments
///
/// * `bytes` - Header bytes.
///
/// # Returns
///
/// Architecture, compression name, and image type.
fn header_meta(bytes: &[u8]) -> (Arch, &'static str, &'static str) {
    (
        arch_of(bytes[29]),
        comp_name(bytes[31]),
        type_name(bytes[30]),
    )
}

/// Extracts load address, entry point, and payload size.
///
/// # Arguments
///
/// * `bytes` - Header bytes.
///
/// # Returns
///
/// Tuple of load address, entry point, and payload size.
fn header_addrs(bytes: &[u8]) -> (u64, u64, u64) {
    let load = u64::from(be_u32(bytes, 16).unwrap_or(0));
    let ep = u64::from(be_u32(bytes, 20).unwrap_or(0));
    let size = u64::from(be_u32(bytes, 12).unwrap_or(0));
    (load, ep, size)
}

/// Checks for an appended `SquashFS` filesystem partition and builds its section.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `addr` - Virtual address for the section.
/// * `offset` - File offset where rootfs starts.
///
/// # Returns
///
/// Mapped section if `SquashFS` magic is present.
fn squashfs_section(bytes: &[u8], addr: u64, offset: u64) -> Option<Section> {
    let start = usize::try_from(offset).ok()?;
    let tail = bytes.get(start..)?;
    if !tail.starts_with(b"hsqs") && !tail.starts_with(b"sqsh") {
        return None;
    }
    let size = tail.len() as u64;
    Some(Section::mapped("rootfs_squashfs", addr, size).with_file(offset, size))
}

/// Appends the rootfs partition when detected.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `addr` - Virtual address.
/// * `offset` - File offset.
/// * `sections` - Sections list to append to.
fn push_rootfs(bytes: &[u8], addr: u64, offset: u64, sections: &mut Vec<Section>) {
    if let Some(rootfs) = squashfs_section(bytes, addr, offset) {
        sections.push(rootfs);
    }
}

/// Constructs sections for a U-Boot legacy image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `load` - Kernel load address.
/// * `size` - Kernel size in bytes.
///
/// # Returns
///
/// List of mapped sections.
fn uimage_sections(bytes: &[u8], load: u64, size: u64) -> Vec<Section> {
    let hdr = Section::mapped("uimage_header", 0, 64).with_file(0, 64);
    let kern = Section::mapped("kernel", load, size)
        .with_file(64, size)
        .with_access(true, false);
    let mut sections = vec![hdr, kern];
    push_rootfs(bytes, load + size, 64 + size, &mut sections);
    sections
}

/// Reports whether an offset contains `SquashFS` magic bytes.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `offset` - Byte offset to inspect.
///
/// # Returns
///
/// `true` if `SquashFS` magic is present.
fn has_squashfs(bytes: &[u8], offset: u64) -> bool {
    let start = usize::try_from(offset).ok().unwrap_or(bytes.len());
    let tail = bytes.get(start..).unwrap_or(&[]);
    tail.starts_with(b"hsqs") || tail.starts_with(b"sqsh")
}

/// Appends `SquashFS` partition facts when detected.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `size` - Kernel payload size.
/// * `facts` - Facts list to append to.
fn append_rootfs_fact(bytes: &[u8], size: u64, facts: &mut Vec<Fact>) {
    let offset = 64 + size;
    if has_squashfs(bytes, offset) {
        facts.push(Fact::new("Filesystem", "SquashFS 4.0"));
        facts.push(Fact::new("Rootfs Offset", format!("0x{offset:x}")));
    }
}

/// Generates metadata facts from a U-Boot header.
///
/// # Arguments
///
/// * `bytes` - Header bytes.
/// * `size` - Kernel payload size.
///
/// # Returns
///
/// List of metadata facts.
fn uimage_facts(bytes: &[u8], size: u64) -> Vec<Fact> {
    let mut facts = vec![
        Fact::new("Image Name", image_name(bytes)),
        Fact::new("Operating System", os_name(bytes[28])),
        Fact::new("Compression", comp_name(bytes[31])),
    ];
    append_rootfs_fact(bytes, size, &mut facts);
    facts
}

/// Constructs a Binary model from extracted U-Boot image properties.
///
/// # Arguments
///
/// * `arch` - Architecture family.
/// * `comp` - Compression name.
/// * `kind` - Image type description.
/// * `ep` - Entry point address.
/// * `load` - Load address.
/// * `sections` - Mapped sections.
/// * `facts` - Metadata facts.
///
/// # Returns
///
/// Initialized Binary model.
fn build_binary(
    arch: Arch,
    comp: &str,
    kind: &str,
    ep: u64,
    load: u64,
    sections: Vec<Section>,
    facts: Vec<Fact>,
) -> Binary {
    let arch_name = format!("{arch:?} (kernel {comp})");
    let mut binary = Binary::new(Format::UImage, arch, &arch_name, 32);
    (binary.kind, binary.entry, binary.image_base) = (kind.to_string(), ep, load);
    (binary.sections, binary.facts) = (sections, facts);
    binary
}

/// Parses a U-Boot legacy image into a Binary model.
///
/// # Arguments
///
/// * `bytes` - Target file bytes.
///
/// # Returns
///
/// The parsed binary model.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when the header is malformed or invalid.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    if !is_valid_uimage(bytes) {
        return Err(EngineError::Parse("invalid uImage header".to_string()));
    }
    let (arch, comp, kind) = header_meta(bytes);
    let (load, ep, size) = header_addrs(bytes);
    let sections = uimage_sections(bytes, load, size);
    let facts = uimage_facts(bytes, size);
    Ok(build_binary(arch, comp, kind, ep, load, sections, facts))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that short buffers return a parse error.
    #[test]
    fn test_parse_short_buffer() {
        assert!(parse(b"too short").is_err());
    }

    /// Verifies that invalid magic bytes return a parse error.
    #[test]
    fn test_parse_invalid_magic() {
        let fake = [0u8; 64];
        assert!(parse(&fake).is_err());
    }

    /// Builds a minimal 68-byte test uImage buffer.
    fn sample_uimage() -> [u8; 68] {
        let mut hdr = [0u8; 68];
        hdr[..4].copy_from_slice(&[0x27, 0x05, 0x19, 0x56]);
        hdr[12..16].copy_from_slice(&4u32.to_be_bytes());
        (hdr[28], hdr[29], hdr[30], hdr[31]) = (5, 5, 2, 3);
        hdr
    }

    /// Verifies that a valid uImage header parses correctly with facts and sections.
    #[test]
    fn test_parse_valid_uimage() {
        let parsed = parse(&sample_uimage()).unwrap();
        assert_eq!(parsed.arch, Arch::Mips);
        assert_eq!(parsed.sections.len(), 2);
    }
}
