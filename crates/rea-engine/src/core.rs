//! ELF Core Dump and Recorded Crash Analysis
//!
//! **File:** `core.rs`
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

/// ELF core file type.
const ET_CORE: u16 = 4;

/// Program header type for loadable segments.
const PT_LOAD: u32 = 1;

/// Program header type for note segments.
const PT_NOTE: u32 = 4;

/// Note type for a thread status record.
const NT_PRSTATUS: u32 = 1;

/// Note type for a process info record.
const NT_PRPSINFO: u32 = 3;

/// Note type for the mapped file table.
const NT_FILE: u32 = 0x4649_4c45;

/// Machine id for x86-64.
const EM_X86_64: u16 = 62;

/// Machine id for the ARM 64-bit architecture.
const EM_AARCH64: u16 = 183;

/// Register names in the 64-bit x86 status order.
const X86_64_REGS: [&str; 27] = [
    "r15", "r14", "r13", "r12", "rbp", "rbx", "r11", "r10", "r9", "r8", "rax", "rcx", "rdx", "rsi",
    "rdi", "orig_rax", "rip", "cs", "eflags", "rsp", "ss", "fs_base", "gs_base", "ds", "es", "fs",
    "gs",
];

/// Register names in the 64-bit ARM status order.
const AARCH64_REGS: [&str; 34] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "sp", "pc", "pstate",
];

/// Byte offset of the register set inside `elf_prstatus`.
const PRSTATUS_REGS: usize = 112;

/// Parsed ELF header fields used for program header traversal.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Ehdr {
    /// Whether the file is 64-bit.
    class64: bool,
    /// Whether the file is little-endian.
    little: bool,
    /// Machine identifier.
    machine: u16,
    /// Program header table offset.
    phoff: usize,
    /// Program header entry size.
    phentsize: usize,
    /// Program header count.
    count: usize,
}

/// A loadable mapping from a core file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoreMapping {
    /// Start virtual address.
    pub start: u64,
    /// File offset of the mapping.
    pub offset: u64,
    /// Mapped length in bytes.
    pub size: u64,
    /// Permission flags text.
    pub flags: String,
}

/// Recovered core dump inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoreDump {
    /// Architecture label.
    pub arch: String,
    /// Terminating signal number.
    pub signal: Option<u32>,
    /// Crashing process id.
    pub pid: u32,
    /// Process name.
    pub process: String,
    /// Captured registers.
    pub registers: Vec<(String, u64)>,
    /// Loadable mappings.
    pub mappings: Vec<CoreMapping>,
    /// Mapped file paths.
    pub files: Vec<String>,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// Reports whether a buffer is an ELF core dump.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` for an ELF file whose type is `ET_CORE`.
#[must_use]
pub fn is_core(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF") && read_u16(bytes, 16, is_little(bytes)) == Some(ET_CORE)
}

/// Reports whether a buffer is little-endian ELF.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the data encoding is little-endian.
fn is_little(bytes: &[u8]) -> bool {
    bytes.get(5).copied().unwrap_or(1) == 1
}

/// Reads a little or big-endian 16-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The word, when two bytes are available.
fn read_u16(bytes: &[u8], offset: usize, little: bool) -> Option<u16> {
    let b = bytes.get(offset..offset + 2)?;
    let arr = [b[0], b[1]];
    Some(if little {
        u16::from_le_bytes(arr)
    } else {
        u16::from_be_bytes(arr)
    })
}

/// Reads a little or big-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The word, when four bytes are available.
fn read_u32(bytes: &[u8], offset: usize, little: bool) -> Option<u32> {
    let b = bytes.get(offset..offset + 4)?;
    let arr = [b[0], b[1], b[2], b[3]];
    Some(if little {
        u32::from_le_bytes(arr)
    } else {
        u32::from_be_bytes(arr)
    })
}

/// Reads a little or big-endian 64-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The word, when eight bytes are available.
fn read_u64(bytes: &[u8], offset: usize, little: bool) -> Option<u64> {
    let b = bytes.get(offset..offset + 8)?;
    let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    Some(if little {
        u64::from_le_bytes(arr)
    } else {
        u64::from_be_bytes(arr)
    })
}

/// Reads a null-terminated ASCII string.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the string.
///
/// # Returns
///
/// The trimmed string up to the terminator.
fn read_cstr(bytes: &[u8], offset: usize) -> String {
    let slice = bytes.get(offset..).unwrap_or_default();
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).trim().to_string()
}

/// Rounds a length up to a four-byte boundary.
///
/// # Arguments
///
/// * `value` - Length to round.
///
/// # Returns
///
/// The padded length.
fn pad4(value: usize) -> usize {
    value.wrapping_add(3) & !3
}

/// Parses the ELF header fields of a core file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Parsed header fields, or `None` when malformed.
fn parse_ehdr(bytes: &[u8]) -> Option<Ehdr> {
    let class64 = *bytes.get(4)? == 2;
    let little = is_little(bytes);
    Some(Ehdr {
        class64,
        little,
        machine: read_u16(bytes, 18, little)?,
        ..ph_table(bytes, class64, little)?
    })
}

/// Reads the program header table descriptor.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `class64` - Whether the file is 64-bit.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// A header carrying table fields, or `None`.
fn ph_table(bytes: &[u8], class64: bool, little: bool) -> Option<Ehdr> {
    Some(Ehdr {
        phoff: ehdr_phoff(bytes, class64, little)?,
        phentsize: usize::from(read_u16(bytes, 54, little)?),
        count: usize::from(read_u16(bytes, 56, little)?),
        ..Default::default()
    })
}

/// Resolves the program header table offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `class64` - Whether the file is 64-bit.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The table offset, or `None`.
fn ehdr_phoff(bytes: &[u8], class64: bool, little: bool) -> Option<usize> {
    let raw = if class64 {
        read_u64(bytes, 32, little)
    } else {
        read_u32(bytes, 28, little).map(u64::from)
    };
    raw.and_then(|v| usize::try_from(v).ok())
}

/// Parses a core dump from a buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Recovered core inventory, or `None` when not a core file.
#[must_use]
pub fn parse_core(bytes: &[u8]) -> Option<CoreDump> {
    if !is_core(bytes) {
        return None;
    }
    let ehdr = parse_ehdr(bytes)?;
    let mut core = CoreDump { arch: arch_label(ehdr.machine), ..Default::default() };
    parse_program_headers(bytes, &ehdr, &mut core);
    Some(core)
}

/// Resolves an architecture label from a machine id.
///
/// # Arguments
///
/// * `machine` - ELF machine id.
///
/// # Returns
///
/// A canonical architecture label.
fn arch_label(machine: u16) -> String {
    match machine {
        EM_X86_64 => "x86-64",
        EM_AARCH64 => "aarch64",
        3 => "i386",
        40 => "arm",
        _ => "unknown",
    }
    .to_string()
}

/// Iterates the program headers of a core file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
fn parse_program_headers(bytes: &[u8], ehdr: &Ehdr, core: &mut CoreDump) {
    for index in 0..ehdr.count {
        let Some(ph) = read_phdr(bytes, ehdr, index) else {
            break;
        };
        dispatch_phdr(bytes, ehdr, &ph, core);
    }
}

/// Routes a program header to the matching handler.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `ehdr` - Parsed header fields.
/// * `ph` - Program header fields.
/// * `core` - Output core inventory.
fn dispatch_phdr(bytes: &[u8], ehdr: &Ehdr, ph: &Phdr, core: &mut CoreDump) {
    match ph.p_type {
        PT_LOAD => core.mappings.push(mapping_of(ph)),
        PT_NOTE => parse_notes(bytes, ph, ehdr, core),
        _ => {}
    }
}

/// A program header subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Phdr {
    /// Header type.
    p_type: u32,
    /// Permission flags.
    flags: u32,
    /// File offset.
    offset: u64,
    /// Virtual address.
    vaddr: u64,
    /// Segment size.
    filesz: u64,
}

/// Builds a mapping description from a program header.
///
/// # Arguments
///
/// * `ph` - Program header fields.
///
/// # Returns
///
/// The mapping description.
fn mapping_of(ph: &Phdr) -> CoreMapping {
    CoreMapping { start: ph.vaddr, offset: ph.offset, size: ph.filesz, flags: flag_text(ph.flags) }
}

/// Renders program header permission flags.
///
/// # Arguments
///
/// * `flags` - Permission bit field.
///
/// # Returns
///
/// A three-character permission string.
fn flag_text(flags: u32) -> String {
    let read = if flags & 4 != 0 { 'r' } else { '-' };
    let write = if flags & 2 != 0 { 'w' } else { '-' };
    let exec = if flags & 1 != 0 { 'x' } else { '-' };
    format!("{read}{write}{exec}")
}

/// Reads one program header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `ehdr` - Parsed header fields.
/// * `index` - Program header index.
///
/// # Returns
///
/// The header fields, or `None`.
fn read_phdr(bytes: &[u8], ehdr: &Ehdr, index: usize) -> Option<Phdr> {
    let base = ehdr.phoff.checked_add(index.checked_mul(ehdr.phentsize)?)?;
    if ehdr.class64 {
        read_phdr64(bytes, base, ehdr.little)
    } else {
        read_phdr32(bytes, base, ehdr.little)
    }
}

/// Reads a 64-bit program header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `base` - Header offset.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The header fields, or `None`.
fn read_phdr64(bytes: &[u8], base: usize, little: bool) -> Option<Phdr> {
    Some(Phdr {
        p_type: read_u32(bytes, base, little)?,
        flags: read_u32(bytes, base + 4, little)?,
        offset: read_u64(bytes, base + 8, little)?,
        vaddr: read_u64(bytes, base + 16, little)?,
        filesz: read_u64(bytes, base + 32, little)?,
    })
}

/// Reads a 32-bit program header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `base` - Header offset.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The header fields, or `None`.
fn read_phdr32(bytes: &[u8], base: usize, little: bool) -> Option<Phdr> {
    Some(Phdr {
        p_type: read_u32(bytes, base, little)?,
        offset: u64::from(read_u32(bytes, base + 4, little)?),
        vaddr: u64::from(read_u32(bytes, base + 8, little)?),
        filesz: u64::from(read_u32(bytes, base + 16, little)?),
        flags: read_u32(bytes, base + 24, little)?,
    })
}

/// Parses the notes of a `PT_NOTE` segment.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `ph` - Note segment header.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
fn parse_notes(bytes: &[u8], ph: &Phdr, ehdr: &Ehdr, core: &mut CoreDump) {
    let start = usize::try_from(ph.offset).unwrap_or(0);
    let end = start.saturating_add(usize::try_from(ph.filesz).unwrap_or(0)).min(bytes.len());
    let mut pos = start;
    while pos + 12 <= end {
        if !parse_one_note(bytes, &mut pos, end, ehdr, core) {
            break;
        }
    }
}

/// Parses one note record at a cursor.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Cursor advanced past the note.
/// * `end` - Segment end offset.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
///
/// # Returns
///
/// `true` when a note was parsed.
fn parse_one_note(
    bytes: &[u8],
    pos: &mut usize,
    end: usize,
    ehdr: &Ehdr,
    core: &mut CoreDump,
) -> bool {
    let Some(note) = read_note(bytes, *pos, end, ehdr.little) else {
        return false;
    };
    apply_note(note.kind, note.desc, ehdr, core);
    *pos = note.next;
    true
}

/// A decoded note record.
struct Note<'a> {
    /// Note type.
    kind: u32,
    /// Description bytes.
    desc: &'a [u8],
    /// Offset of the next note.
    next: usize,
}

/// Reads one note record at an offset.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Note offset.
/// * `end` - Segment end offset.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The decoded note, or `None`.
fn read_note(bytes: &[u8], pos: usize, end: usize, little: bool) -> Option<Note<'_>> {
    let (namesz, descsz, kind) = note_header(bytes, pos, little)?;
    let desc_start = pos.saturating_add(12).saturating_add(pad4(namesz));
    let next = desc_start.saturating_add(pad4(descsz));
    if next > end {
        return None;
    }
    Some(Note { kind, desc: bytes.get(desc_start..desc_start + descsz)?, next })
}

/// Reads the three-word note header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `pos` - Note offset.
/// * `little` - Whether the encoding is little-endian.
///
/// # Returns
///
/// The name size, description size, and note type.
fn note_header(bytes: &[u8], pos: usize, little: bool) -> Option<(usize, usize, u32)> {
    let namesz = usize::try_from(read_u32(bytes, pos, little)?).ok()?;
    let descsz = usize::try_from(read_u32(bytes, pos + 4, little)?).ok()?;
    let kind = read_u32(bytes, pos + 8, little)?;
    Some((namesz, descsz, kind))
}

/// Applies a decoded note to the core inventory.
///
/// # Arguments
///
/// * `kind` - Note type.
/// * `desc` - Note description bytes.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
fn apply_note(kind: u32, desc: &[u8], ehdr: &Ehdr, core: &mut CoreDump) {
    match kind {
        NT_PRSTATUS => parse_prstatus(desc, ehdr, core),
        NT_PRPSINFO => core.process = read_cstr(desc, 40),
        NT_FILE => parse_file_note(desc, ehdr, core),
        _ => {}
    }
}

/// Parses a thread status note.
///
/// # Arguments
///
/// * `desc` - Note description bytes.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
fn parse_prstatus(desc: &[u8], ehdr: &Ehdr, core: &mut CoreDump) {
    core.signal = read_u16(desc, 12, ehdr.little).map(u32::from).or(core.signal);
    core.pid = read_u32(desc, 32, ehdr.little).unwrap_or(core.pid);
    if core.registers.is_empty() {
        core.registers = read_registers(desc, ehdr);
    }
}

/// Reads the register set of a thread status note.
///
/// # Arguments
///
/// * `desc` - Note description bytes.
/// * `ehdr` - Parsed header fields.
///
/// # Returns
///
/// Register name and value pairs.
fn read_registers(desc: &[u8], ehdr: &Ehdr) -> Vec<(String, u64)> {
    let names = register_names(ehdr.machine);
    let regs = names.iter().enumerate().filter_map(|(i, name)| {
        let value = read_u64(desc, PRSTATUS_REGS + i * 8, ehdr.little)?;
        Some(((*name).to_string(), value))
    });
    regs.collect()
}

/// Resolves register names for a machine.
///
/// # Arguments
///
/// * `machine` - ELF machine id.
///
/// # Returns
///
/// The register name table.
fn register_names(machine: u16) -> &'static [&'static str] {
    match machine {
        EM_X86_64 => &X86_64_REGS,
        EM_AARCH64 => &AARCH64_REGS,
        _ => &[],
    }
}

/// Parses the mapped file table note.
///
/// # Arguments
///
/// * `desc` - Note description bytes.
/// * `ehdr` - Parsed header fields.
/// * `core` - Output core inventory.
fn parse_file_note(desc: &[u8], ehdr: &Ehdr, core: &mut CoreDump) {
    let Some(count) = read_u64(desc, 0, ehdr.little) else {
        return;
    };
    let names_start = 16 + usize::try_from(count).unwrap_or(0) * 24;
    let names = desc.get(names_start.min(desc.len())..).unwrap_or_default();
    for name in split_cstrings(names) {
        core.files.push(name);
    }
}

/// Splits a buffer into non-empty null-terminated strings.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
///
/// # Returns
///
/// Every string in order.
fn split_cstrings(bytes: &[u8]) -> Vec<String> {
    let strings = bytes.split(|&b| b == 0).filter(|s| !s.is_empty());
    strings.map(|s| String::from_utf8_lossy(s).to_string()).collect()
}

/// Formats a core inventory as a human-readable text block.
///
/// # Arguments
///
/// * `core` - Recovered core inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_core_summary(core: &CoreDump) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Core dump: {}", core.arch);
    let _ = writeln!(out, "Process: {} (pid {})", core.process, core.pid);
    let _ = writeln!(out, "Signal: {:?}", core.signal);
    let _ = writeln!(out, "Mappings: {}", core.mappings.len());
    let _ = writeln!(out, "Files: {}", core.files.len());
    let _ = writeln!(out, "Registers: {}", core.registers.len());
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal 64-bit little-endian core header.
    ///
    /// # Returns
    ///
    /// A 64-byte ELF header buffer.
    fn core_header() -> Vec<u8> {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[4] = 2;
        data[5] = 1;
        data[16] = u8::try_from(ET_CORE).unwrap();
        data[18] = u8::try_from(EM_X86_64).unwrap();
        data
    }

    /// Verifies core detection against a regular ELF.
    #[test]
    fn test_core_detection() {
        assert!(is_core(&core_header()));
        let mut exec = core_header();
        exec[16] = 2;
        assert!(!is_core(&exec));
    }

    /// Verifies architecture label resolution.
    #[test]
    fn test_core_arch_label() {
        assert_eq!(arch_label(EM_X86_64), "x86-64");
        assert_eq!(arch_label(EM_AARCH64), "aarch64");
    }

    /// Verifies permission flag rendering.
    #[test]
    fn test_core_flags() {
        assert_eq!(flag_text(7), "rwx");
        assert_eq!(flag_text(4), "r--");
    }

    /// Verifies program header parsing.
    #[test]
    fn test_core_phdr() {
        let data = phdr_data();
        let ehdr = parse_ehdr(&data).unwrap();
        let ph = read_phdr(&data, &ehdr, 0).unwrap();
        assert_eq!(ph.p_type, PT_LOAD);
        assert_eq!(flag_text(ph.flags), "r-x");
    }

    /// Builds a core header carrying a single loadable segment.
    ///
    /// # Returns
    ///
    /// The header and program header bytes.
    fn phdr_data() -> Vec<u8> {
        let mut data = core_header();
        data[54] = 56;
        data[56] = 1;
        data[32] = 64;
        data.resize(64 + 56, 0);
        data[64] = u8::try_from(PT_LOAD).unwrap();
        data[68] = 5;
        data
    }

    /// Verifies summary formatting.
    #[test]
    fn test_core_summary() {
        let core = CoreDump { arch: "x86-64".into(), pid: 42, ..Default::default() };
        assert!(format_core_summary(&core).contains("pid 42"));
    }
}
