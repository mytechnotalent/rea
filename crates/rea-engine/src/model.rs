//! Binary Model
//!
//! **File:** `model.rs`
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

// Import crate::detect
use crate::detect::Format;

// Import crate::numeric
use crate::numeric::window;

/// Processor architecture families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Arch {
    /// 32-bit x86.
    X86,

    /// 64-bit x86.
    X86_64,

    /// 32-bit ARM.
    Arm,

    /// 64-bit ARM.
    Arm64,

    /// 32-bit MIPS.
    Mips,

    /// 64-bit MIPS.
    Mips64,

    /// 32-bit PowerPC.
    PowerPc,

    /// 64-bit PowerPC.
    PowerPc64,

    /// 32-bit RISC-V.
    Riscv,

    /// 64-bit RISC-V.
    Riscv64,

    /// SPARC.
    Sparc,

    /// `SuperH`.
    SuperH,

    /// Atmel AVR.
    Avr,

    /// Motorola 68000.
    M68k,

    /// TI MSP430.
    Msp430,

    /// Tensilica Xtensa.
    Xtensa,

    /// `LoongArch`.
    LoongArch,

    /// Any other architecture.
    #[default]
    Other,
}

/// Display names of known architecture families.
const ARCH_LABELS: [(Arch, &str); 17] = [
    (Arch::X86, "x86 (32-bit)"),
    (Arch::X86_64, "x86-64 (AMD64)"),
    (Arch::Arm, "ARM (32-bit)"),
    (Arch::Arm64, "ARM64 (AArch64)"),
    (Arch::Mips, "MIPS (32-bit)"),
    (Arch::Mips64, "MIPS64 (64-bit)"),
    (Arch::PowerPc, "PowerPC (32-bit)"),
    (Arch::PowerPc64, "PowerPC 64-bit"),
    (Arch::Riscv, "RISC-V (32-bit)"),
    (Arch::Riscv64, "RISC-V 64-bit"),
    (Arch::Sparc, "SPARC"),
    (Arch::SuperH, "SuperH (SH-4)"),
    (Arch::Avr, "AVR (8-bit)"),
    (Arch::M68k, "Motorola 68000 (M68k)"),
    (Arch::Msp430, "MSP430 (16-bit)"),
    (Arch::Xtensa, "Tensilica Xtensa"),
    (Arch::LoongArch, "LoongArch (64-bit)"),
];

impl Arch {
    /// Returns a readable name for known architecture families.
    ///
    /// # Returns
    ///
    /// The display name, or `None` for [`Arch::Other`].
    #[must_use]
    pub fn label(self) -> Option<&'static str> {
        let found = ARCH_LABELS.iter().find(|(arch, _)| *arch == self);
        found.map(|(_, label)| *label)
    }

    /// Returns the decoder bitness for x86 family architectures.
    ///
    /// # Returns
    ///
    /// `Some(32)` or `Some(64)` for x86, otherwise `None`.
    #[must_use]
    pub fn x86_bitness(self) -> Option<u32> {
        match self {
            Self::X86 => Some(32),
            Self::X86_64 => Some(64),
            _ => None,
        }
    }
}

/// One section or loadable segment of a binary.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Section {
    /// Section name as stored in the file.
    pub name: String,

    /// Virtual address where the section is mapped.
    pub address: u64,

    /// Size in memory.
    pub virtual_size: u64,

    /// File offset of the section data.
    pub file_offset: u64,

    /// Size of the section data in the file.
    pub file_size: u64,

    /// Whether the section is mapped executable.
    pub executable: bool,

    /// Whether the section is mapped writable.
    pub writable: bool,

    /// Shannon entropy of the file data in bits per byte.
    pub entropy: f64,
}

impl Section {
    /// Creates a mapped section without file data or access rights.
    ///
    /// # Arguments
    ///
    /// * `name` - Section name.
    /// * `address` - Virtual address.
    /// * `virtual_size` - Size in memory.
    ///
    /// # Returns
    ///
    /// The section, ready for the `with_*` builders.
    #[must_use]
    pub fn mapped(name: &str, address: u64, virtual_size: u64) -> Self {
        let name = name.to_string();
        Self { name, address, virtual_size, ..Self::default() }
    }

    /// Sets where the section's data lives in the file.
    ///
    /// # Arguments
    ///
    /// * `offset` - File offset.
    /// * `size` - Size in the file.
    ///
    /// # Returns
    ///
    /// The updated section.
    #[must_use]
    pub fn with_file(mut self, offset: u64, size: u64) -> Self {
        (self.file_offset, self.file_size) = (offset, size);
        self
    }

    /// Sets the section's memory access rights.
    ///
    /// # Arguments
    ///
    /// * `executable` - Whether the section is executable.
    /// * `writable` - Whether the section is writable.
    ///
    /// # Returns
    ///
    /// The updated section.
    #[must_use]
    pub fn with_access(mut self, executable: bool, writable: bool) -> Self {
        (self.executable, self.writable) = (executable, writable);
        self
    }

    /// Returns the section's bytes from the file, clamped to the buffer.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Whole target file.
    ///
    /// # Returns
    ///
    /// The file-backed section data.
    #[must_use]
    pub fn data<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        window(bytes, self.file_offset, self.file_size)
    }

    /// Maps a virtual address to its file offset within the section.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// The file offset, when the address is backed by file data.
    #[must_use]
    pub fn file_offset_of(&self, address: u64) -> Option<u64> {
        let delta = address.checked_sub(self.address)?;
        let backed = self.address != 0 && delta < self.file_size;
        backed.then(|| self.file_offset + delta)
    }

    /// Reports whether a virtual address falls inside the section.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address to test.
    ///
    /// # Returns
    ///
    /// `true` when the address is within the mapped size.
    #[must_use]
    pub fn contains(&self, address: u64) -> bool {
        let size = self.virtual_size.max(self.file_size);
        address >= self.address && address - self.address < size
    }
}

/// A function imported from another module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Import {
    /// Providing library, when the format records it.
    pub library: String,

    /// Imported symbol name.
    pub name: String,

    /// Virtual address of the import slot, or zero when unknown.
    pub address: u64,
}

impl Import {
    /// Creates an import record.
    ///
    /// # Arguments
    ///
    /// * `library` - Providing library, or empty when unknown.
    /// * `name` - Imported symbol name.
    /// * `address` - Import slot address, or zero.
    ///
    /// # Returns
    ///
    /// The import.
    #[must_use]
    pub fn new(library: &str, name: &str, address: u64) -> Self {
        Self { library: library.to_string(), name: name.to_string(), address }
    }
}

/// A named code or data location.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Symbol {
    /// Symbol name.
    pub name: String,

    /// Virtual address.
    pub address: u64,
}

impl Symbol {
    /// Creates a symbol.
    ///
    /// # Arguments
    ///
    /// * `name` - Symbol name.
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// The symbol.
    #[must_use]
    pub fn new(name: &str, address: u64) -> Self {
        let name = name.to_string();
        Self { name, address }
    }
}

/// One security mitigation and whether it is enabled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Protection {
    /// Mitigation name.
    pub name: String,

    /// Whether the mitigation is enabled.
    pub enabled: bool,

    /// Evidence for the verdict.
    pub evidence: String,
}

impl Protection {
    /// Creates a protection verdict.
    ///
    /// # Arguments
    ///
    /// * `name` - Mitigation name.
    /// * `enabled` - Whether it is enabled.
    /// * `evidence` - Where the verdict comes from.
    ///
    /// # Returns
    ///
    /// The protection record.
    #[must_use]
    pub fn new(name: &str, enabled: bool, evidence: &str) -> Self {
        Self { name: name.to_string(), enabled, evidence: evidence.to_string() }
    }
}

/// A labeled metadata fact read from the binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    /// Fact label.
    pub label: String,

    /// Fact value.
    pub value: String,
}

impl Fact {
    /// Creates a fact.
    ///
    /// # Arguments
    ///
    /// * `label` - Fact label.
    /// * `value` - Fact value.
    ///
    /// # Returns
    ///
    /// The fact record.
    #[must_use]
    pub fn new(label: &str, value: impl Into<String>) -> Self {
        Self { label: label.to_string(), value: value.into() }
    }
}

/// Format-neutral description of a parsed executable.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Binary {
    /// Container format.
    pub format: Format,

    /// Architecture family.
    pub arch: Arch,

    /// Detailed architecture name from the file.
    pub arch_name: String,

    /// Address width in bits.
    pub bits: u32,

    /// Whether the target stores multi-byte values big-endian.
    pub big_endian: bool,

    /// Whether code runs in a sub-mode, such as ARM Thumb throughout.
    pub thumb: bool,

    /// Image kind, such as executable or shared library.
    pub kind: String,

    /// Entry-point virtual address, or zero when absent.
    pub entry: u64,

    /// Preferred load address.
    pub image_base: u64,

    /// Sections or loadable segments.
    pub sections: Vec<Section>,

    /// Imported functions.
    pub imports: Vec<Import>,

    /// Exported symbols.
    pub exports: Vec<Symbol>,

    /// Local function symbols used to name code.
    pub symbols: Vec<Symbol>,

    /// Function entry points recorded by the format, such as PE unwind data.
    pub function_starts: Vec<u64>,

    /// Required libraries.
    pub libraries: Vec<String>,

    /// Security mitigations.
    pub protections: Vec<Protection>,

    /// Additional metadata facts.
    pub facts: Vec<Fact>,
}

/// Returns the address width for a 64-bit flag.
///
/// # Arguments
///
/// * `is_64` - Whether the binary is 64-bit.
///
/// # Returns
///
/// `64` or `32`.
#[must_use]
pub fn bits_of(is_64: bool) -> u32 {
    if is_64 {
        64
    } else {
        32
    }
}

impl Binary {
    /// Creates a binary with its identity fields set.
    ///
    /// # Arguments
    ///
    /// * `format` - Container format.
    /// * `arch` - Architecture family.
    /// * `arch_name` - Detailed architecture name.
    /// * `bits` - Address width.
    ///
    /// # Returns
    ///
    /// A binary with every other field empty.
    #[must_use]
    pub fn new(format: Format, arch: Arch, arch_name: &str, bits: u32) -> Self {
        let name = if arch_name.is_empty() {
            arch.label().unwrap_or("Unknown").to_string()
        } else {
            arch_name.to_string()
        };
        Self { format, arch, arch_name: name, bits, ..Self::default() }
    }

    /// Finds the section containing a virtual address.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// The containing section, when any.
    #[must_use]
    pub fn section_at(&self, address: u64) -> Option<&Section> {
        self.sections.iter().find(|s| s.contains(address))
    }

    /// Maps a virtual address to the file offset holding its bytes.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// The file offset, when a section backs the address with file data.
    #[must_use]
    pub fn offset_of(&self, address: u64) -> Option<u64> {
        self.sections.iter().find_map(|s| s.file_offset_of(address))
    }

    /// Maps a file offset to the virtual address where it is loaded.
    ///
    /// # Arguments
    ///
    /// * `offset` - File offset.
    ///
    /// # Returns
    ///
    /// The virtual address, when the offset lies in a section.
    #[must_use]
    pub fn address_of(&self, offset: u64) -> Option<u64> {
        let inside = |s: &&Section| offset >= s.file_offset && offset - s.file_offset < s.file_size;
        let section = self.sections.iter().find(inside)?;
        Some(section.address + (offset - section.file_offset))
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a section mapping file offset 0x400 to address 0x1000.
    ///
    /// # Returns
    ///
    /// A 0x200-byte executable section.
    fn text() -> Section {
        let mapped = Section::mapped(".text", 0x1000, 0x200);
        mapped.with_file(0x400, 0x200).with_access(true, false)
    }

    /// Verifies address containment at the section edges.
    #[test]
    fn test_section_contains_edges() {
        assert!(text().contains(0x1000));
        assert!(text().contains(0x11ff));
        assert!(!text().contains(0x1200));
    }

    /// Verifies offset-to-address translation.
    #[test]
    fn test_address_of_maps_offsets() {
        let mut binary = Binary::new(Format::Pe, Arch::X86, "i386", 32);
        binary.sections = vec![text()];
        assert_eq!(binary.address_of(0x410), Some(0x1010));
        assert_eq!(binary.address_of(0x10), None);
    }

    /// Verifies that section data is clamped to the file.
    #[test]
    fn test_section_data_clamps() {
        assert_eq!(text().data(&[0u8; 0x500]).len(), 0x100);
    }

    /// Verifies x86 bitness mapping.
    #[test]
    fn test_x86_bitness() {
        assert_eq!(Arch::X86_64.x86_bitness(), Some(64));
        assert_eq!(Arch::Arm64.x86_bitness(), None);
    }

    /// Verifies the record constructors.
    #[test]
    fn test_constructors() {
        assert_eq!(Import::new("l", "n", 4).address, 4);
        assert_eq!(Symbol::new("s", 8).name, "s");
        assert_eq!(bits_of(true), 64);
    }
}
