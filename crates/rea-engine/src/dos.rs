//! MS-DOS MZ Executable Analysis and 16-bit Disassembly
//!
//! **File:** `dos.rs`
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
use std::fmt::Write as _;

// Import dependencies from iced_x86
use iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};

/// Most 16-bit instructions decoded at the entry point.
const MAX_INSTRUCTIONS: usize = 64;

/// Recovered MS-DOS MZ executable inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DosHeader {
    /// Header size in paragraphs.
    pub header_paragraphs: u16,
    /// Minimum extra paragraphs requested.
    pub min_alloc: u16,
    /// Maximum extra paragraphs requested.
    pub max_alloc: u16,
    /// Initial stack segment.
    pub ss: u16,
    /// Initial stack pointer.
    pub sp: u16,
    /// Initial instruction pointer.
    pub ip: u16,
    /// Initial code segment.
    pub cs: u16,
    /// Relocation table entry count.
    pub reloc_count: u16,
    /// Relocation table offset.
    pub reloc_offset: u16,
    /// Offset of the extended header.
    pub lfanew: u32,
    /// File offset of the entry point.
    pub entry_offset: usize,
    /// Relocation segment and offset pairs.
    pub relocations: Vec<(u16, u16)>,
    /// Disassembled entry instructions.
    pub code: Vec<String>,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// Reports whether a buffer carries a PE header.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when `PE\0\0` is present at the extended header offset.
#[must_use]
pub fn is_pe_image(bytes: &[u8]) -> bool {
    let at = usize::try_from(read_u32(bytes, 60).unwrap_or(0)).unwrap_or(0);
    bytes.get(at..at + 4) == Some(b"PE\x00\x00")
}

/// Reports whether a buffer is an MS-DOS MZ executable.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` for MZ files without a PE header.
#[must_use]
pub fn is_dos_exe(bytes: &[u8]) -> bool {
    bytes.starts_with(b"MZ") && !is_pe_image(bytes)
}

/// Reads a little-endian 16-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, or `None`.
fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let b = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
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
/// The word, or `None`.
fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Reads the header fields stored in the fixed prefix.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// A partially populated header.
fn base_fields(bytes: &[u8]) -> DosHeader {
    DosHeader {
        header_paragraphs: read_u16(bytes, 8).unwrap_or(0),
        min_alloc: read_u16(bytes, 10).unwrap_or(0),
        max_alloc: read_u16(bytes, 12).unwrap_or(0),
        reloc_count: read_u16(bytes, 6).unwrap_or(0),
        reloc_offset: read_u16(bytes, 24).unwrap_or(0),
        ..Default::default()
    }
}

/// Builds the full DOS header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The populated header.
fn build_header(bytes: &[u8]) -> DosHeader {
    let mut h = base_fields(bytes);
    h.ss = read_u16(bytes, 14).unwrap_or(0);
    h.sp = read_u16(bytes, 16).unwrap_or(0);
    h.ip = read_u16(bytes, 20).unwrap_or(0);
    h.cs = read_u16(bytes, 22).unwrap_or(0);
    h.lfanew = read_u32(bytes, 60).unwrap_or(0);
    h.entry_offset = usize::from(h.header_paragraphs) * 16 + usize::from(h.ip);
    h
}

/// Reads one relocation table entry.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Entry offset.
///
/// # Returns
///
/// The segment and offset, or `None`.
fn reloc_at(bytes: &[u8], pos: usize) -> Option<(u16, u16)> {
    Some((read_u16(bytes, pos)?, read_u16(bytes, pos + 2)?))
}

/// Parses the relocation table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `h` - Header fields.
///
/// # Returns
///
/// Every relocation pair.
fn parse_relocations(bytes: &[u8], h: &DosHeader) -> Vec<(u16, u16)> {
    let start = usize::from(h.reloc_offset);
    (0..usize::from(h.reloc_count))
        .filter_map(|i| reloc_at(bytes, start + i * 4))
        .collect()
}

/// Formats one decoded instruction.
///
/// # Arguments
///
/// * `decoder` - Active decoder.
/// * `formatter` - Instruction formatter.
///
/// # Returns
///
/// The formatted instruction.
fn format_instruction(decoder: &mut Decoder, formatter: &mut NasmFormatter) -> String {
    let instruction = decoder.decode();
    let mut text = String::new();
    formatter.format(&instruction, &mut text);
    text
}

/// Decodes 16-bit instructions.
///
/// # Arguments
///
/// * `code` - Code bytes.
/// * `limit` - Most instructions to decode.
///
/// # Returns
///
/// One text line per instruction.
fn decode16(code: &[u8], limit: usize) -> Vec<String> {
    let mut decoder = Decoder::with_ip(16, code, 0, DecoderOptions::NONE);
    let mut formatter = NasmFormatter::new();
    let mut out = Vec::new();
    while out.len() < limit && decoder.can_decode() {
        out.push(format_instruction(&mut decoder, &mut formatter));
    }
    out
}

/// Disassembles the entry point of a DOS executable.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `h` - Header fields.
///
/// # Returns
///
/// One text line per instruction.
fn disassemble16(bytes: &[u8], h: &DosHeader) -> Vec<String> {
    let start = h.entry_offset.min(bytes.len());
    let code = bytes.get(start..).unwrap_or_default();
    decode16(code, MAX_INSTRUCTIONS)
}

/// Parses an MS-DOS MZ executable.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered header, or `None`.
#[must_use]
pub fn parse_dos(bytes: &[u8]) -> Option<DosHeader> {
    if !is_dos_exe(bytes) {
        return None;
    }
    let mut h = build_header(bytes);
    h.relocations = parse_relocations(bytes, &h);
    h.code = disassemble16(bytes, &h);
    Some(h)
}

/// Formats a DOS header as a human-readable text block.
///
/// # Arguments
///
/// * `h` - Recovered header.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_dos_summary(h: &DosHeader) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "MS-DOS MZ executable");
    let _ = writeln!(out, "CS:IP: {:04x}:{:04x}", h.cs, h.ip);
    let _ = writeln!(out, "SS:SP: {:04x}:{:04x}", h.ss, h.sp);
    append_dos_fields(&mut out, h);
    out
}

/// Appends DOS header fields.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `h` - Recovered header.
fn append_dos_fields(out: &mut String, h: &DosHeader) {
    let _ = writeln!(out, "Header: {} paragraphs", h.header_paragraphs);
    let _ = writeln!(out, "Min alloc: {}", h.min_alloc);
    let _ = writeln!(out, "Max alloc: {}", h.max_alloc);
    let _ = writeln!(out, "Relocations: {}", h.relocations.len());
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal MZ header with a `RET` at the entry.
    ///
    /// # Returns
    ///
    /// The executable bytes.
    fn sample_dos() -> Vec<u8> {
        let mut data = vec![0u8; 64];
        data[0] = b'M';
        data[1] = b'Z';
        data[8..10].copy_from_slice(&2u16.to_le_bytes());
        data[16..18].copy_from_slice(&0x0100u16.to_le_bytes());
        data[32] = 0xc3;
        data
    }

    /// Verifies DOS versus PE classification.
    #[test]
    fn test_dos_detection() {
        let dos = sample_dos();
        assert!(is_dos_exe(&dos));
        assert!(!is_pe_image(&dos));
    }

    /// Verifies header parsing.
    #[test]
    fn test_dos_header() {
        let h = parse_dos(&sample_dos()).unwrap();
        assert_eq!(h.header_paragraphs, 2);
        assert_eq!(h.sp, 0x0100);
        assert_eq!(h.entry_offset, 32);
    }

    /// Verifies 16-bit decoding of a return instruction.
    #[test]
    fn test_dos_disassembly() {
        let h = parse_dos(&sample_dos()).unwrap();
        assert_eq!(h.code.first().map(String::as_str), Some("ret"));
    }

    /// Verifies summary formatting.
    #[test]
    fn test_dos_summary() {
        let h = parse_dos(&sample_dos()).unwrap();
        assert!(format_dos_summary(&h).contains("MS-DOS MZ"));
    }
}
