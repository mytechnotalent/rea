//! Raw Firmware ELF Wrapper
//!
//! **File:** `elfwrap.rs`
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

/// Size of an ELF32 file header.
const EHDR: usize = 52;

/// Size of an ELF32 program header.
const PHDR: usize = 32;

/// File offset at which the raw image is placed.
pub const LOAD_OFFSET: usize = 0x1000;

/// Program header type for a loadable segment.
const PT_LOAD: u32 = 1;

/// Machine id for 32-bit ARM.
const EM_ARM: u16 = 40;

/// Result of wrapping a raw image as an ELF.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Wrapped {
    /// The synthesized ELF image.
    pub bytes: Vec<u8>,
    /// Entry address inside the wrapped image.
    pub entry: u32,
}

/// Wraps a raw 32-bit ARM image as a loadable ELF.
///
/// The raw bytes are mapped at virtual address zero, so an analysis engine
/// that only auto-loads known formats can decompile headerless firmware.
///
/// # Arguments
///
/// * `raw` - Raw firmware bytes.
/// * `entry` - Entry address within the raw image.
///
/// # Returns
///
/// The wrapped ELF and its entry address.
#[must_use]
pub fn wrap_arm(raw: &[u8], entry: u32) -> Wrapped {
    let mut bytes = elf_header(entry);
    bytes.extend_from_slice(&program_header(raw.len()));
    bytes.resize(LOAD_OFFSET, 0);
    bytes.extend_from_slice(raw);
    Wrapped { bytes, entry }
}

/// Builds the ELF header for a raw ARM image.
///
/// # Arguments
///
/// * `entry` - Entry address.
///
/// # Returns
///
/// The 52-byte ELF header.
fn elf_header(entry: u32) -> Vec<u8> {
    let mut header = elf_ident();
    put_u16(&mut header, 16, 2);
    put_u16(&mut header, 18, EM_ARM);
    put_u32(&mut header, 20, 1);
    put_u32(&mut header, 24, entry);
    put_u32(&mut header, 28, 52);
    header
}

/// Builds the ELF identification and remaining fixed fields.
///
/// # Returns
///
/// A zeroed header with the identification fields set.
fn elf_ident() -> Vec<u8> {
    let mut header = vec![0u8; EHDR];
    header[0..4].copy_from_slice(b"\x7fELF");
    header[4] = 1;
    header[5] = 1;
    header[6] = 1;
    elf_meta(&mut header);
    header
}

/// Writes the ELF flags and table-size fields.
///
/// # Arguments
///
/// * `header` - Header buffer.
fn elf_meta(header: &mut [u8]) {
    put_u32(header, 36, 0x0500_0000);
    put_u16(header, 40, 52);
    put_u16(header, 42, 32);
    put_u16(header, 44, 1);
}

/// Builds a single read-execute program header.
///
/// # Arguments
///
/// * `size` - Raw image length.
///
/// # Returns
///
/// The 32-byte program header.
fn program_header(size: usize) -> Vec<u8> {
    let mut header = vec![0u8; PHDR];
    let len = u32::try_from(size).unwrap_or(u32::MAX);
    put_u32(&mut header, 0, PT_LOAD);
    put_u32(&mut header, 4, 0x1000);
    put_u32(&mut header, 16, len);
    put_u32(&mut header, 20, len);
    program_flags(&mut header);
    header
}

/// Writes the program header permission and alignment fields.
///
/// # Arguments
///
/// * `header` - Program header buffer.
fn program_flags(header: &mut [u8]) {
    put_u32(header, 24, 5);
    put_u32(header, 28, 0x1000);
}

/// Writes a little-endian 16-bit field.
///
/// # Arguments
///
/// * `bytes` - Destination buffer.
/// * `offset` - Field offset.
/// * `value` - Value to write.
fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Writes a little-endian 32-bit field.
///
/// # Arguments
///
/// * `bytes` - Destination buffer.
/// * `offset` - Field offset.
/// * `value` - Value to write.
fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the ELF header fields and image placement.
    #[test]
    fn test_wrap_arm_header() {
        let wrapped = wrap_arm(&[0xaau8; 64], 0x40);
        let bytes = &wrapped.bytes;
        assert_eq!(&bytes[0..4], b"\x7fELF");
        assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), EM_ARM);
        assert_eq!(wrapped.entry, 0x40);
        assert!(bytes[LOAD_OFFSET..LOAD_OFFSET + 4].iter().all(|b| *b == 0xaa));
    }

    /// Verifies the program header maps virtual zero to the load offset.
    #[test]
    fn test_wrap_arm_phdr() {
        let wrapped = wrap_arm(&[0u8; 16], 0x40);
        let ph = EHDR;
        let kind = u32::from_le_bytes(wrapped.bytes[ph..ph + 4].try_into().unwrap());
        let offset = u32::from_le_bytes(wrapped.bytes[ph + 4..ph + 8].try_into().unwrap());
        assert_eq!(kind, PT_LOAD);
        assert_eq!(offset as usize, LOAD_OFFSET);
    }
}
