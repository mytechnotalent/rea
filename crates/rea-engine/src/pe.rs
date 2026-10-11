//! PE Loader
//!
//! **File:** `pe.rs`
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
    model::{bits_of, Arch, Binary, Fact, Import, Protection, Section, Symbol},
    numeric::to_u64,
    time::utc_timestamp,
};

// Import dependencies from goblin
use goblin::pe::{
    dll_characteristic as dll,
    export::Export,
    header::{self, machine_to_str},
    import::Import as PeImport,
    section_table::SectionTable,
    PE,
};

/// Section characteristic: contains code.
const SCN_CNT_CODE: u32 = 0x0000_0020;

/// Section characteristic: executable.
const SCN_MEM_EXECUTE: u32 = 0x2000_0000;

/// Section characteristic: writable.
const SCN_MEM_WRITE: u32 = 0x8000_0000;

/// Mitigations controlled by `DllCharacteristics` bits.
const FLAG_PROTECTIONS: [(&str, u16, &str); 5] = [
    (
        "ASLR",
        dll::IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE,
        "DYNAMIC_BASE",
    ),
    (
        "High-entropy ASLR",
        dll::IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA,
        "HIGH_ENTROPY_VA",
    ),
    (
        "DEP / NX",
        dll::IMAGE_DLLCHARACTERISTICS_NX_COMPAT,
        "NX_COMPAT",
    ),
    (
        "Control Flow Guard",
        dll::IMAGE_DLLCHARACTERISTICS_GUARD_CF,
        "GUARD_CF",
    ),
    (
        "Forced integrity checks",
        dll::IMAGE_DLLCHARACTERISTICS_FORCE_INTEGRITY,
        "FORCE_INTEGRITY",
    ),
];

/// Windows subsystem names by identifier.
const SUBSYSTEMS: [(u16, &str); 8] = [
    (1, "Native (driver or boot component)"),
    (2, "Windows GUI"),
    (3, "Windows console"),
    (7, "POSIX console"),
    (9, "Windows CE GUI"),
    (10, "EFI application"),
    (11, "EFI boot service driver"),
    (16, "Windows boot application"),
];

/// Optional metadata extractors run against every PE.
const FACTS: [fn(&PE) -> Option<Fact>; 7] =
    [subsystem, linker, pdb, managed, rich, tls, module_name];

/// Parses a PE image into the format-neutral model.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The parsed binary.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when the PE structures are malformed.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    let pe = PE::parse(bytes)?;
    let (mut binary, base) = (identity(&pe), pe.image_base);
    binary.sections = pe.sections.iter().map(|s| section(s, base)).collect();
    binary.imports = pe.imports.iter().map(|i| import(i, base)).collect();
    binary.exports = pe.exports.iter().filter_map(|e| export(e, base)).collect();
    populate_pe_tables(&mut binary, bytes, &pe, base);
    (binary.protections, binary.facts) = (protections(&pe), facts(&pe));
    Ok(binary)
}

/// Lists function entry points from the exception directory (x64 unwind data).
///
/// # Arguments
///
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
///
/// # Returns
///
/// Every runtime function's start address.
fn function_starts(pe: &PE, base: u64) -> Vec<u64> {
    let Some(data) = &pe.exception_data else {
        return Vec::new();
    };
    let entries = data.functions().filter_map(Result::ok);
    entries.map(|f| base + u64::from(f.begin_address)).collect()
}

/// Collects metadata facts.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The link timestamp followed by every applicable optional fact.
fn facts(pe: &PE) -> Vec<Fact> {
    let mut found = vec![timestamp(pe)];
    found.extend(FACTS.iter().filter_map(|fact| fact(pe)));
    found
}

/// Builds the identity fields of the model.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// A binary with format, architecture, kind, entry, and libraries set.
fn identity(pe: &PE) -> Binary {
    let machine = pe.header.coff_header.machine;
    let name = machine_to_str(machine);
    let mut binary = Binary::new(Format::Pe, arch(machine), name, bits_of(pe.is_64));
    binary.kind = kind(pe).to_string();
    binary.entry = entry_address(pe);
    binary.image_base = pe.image_base;
    binary.libraries = pe.libraries.iter().map(ToString::to_string).collect();
    binary
}

/// Describes the image kind.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// `Dynamic-link library` or `Executable`.
fn kind(pe: &PE) -> &'static str {
    if pe.is_lib {
        "Dynamic-link library"
    } else {
        "Executable"
    }
}

/// Computes the entry point's virtual address.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The entry address, or zero when the image has none.
fn entry_address(pe: &PE) -> u64 {
    if pe.entry == 0 {
        0
    } else {
        pe.image_base + u64::from(pe.entry)
    }
}

/// Architecture mappings for known COFF machine types.
const PE_ARCHES: [(u16, Arch); 16] = [
    (header::COFF_MACHINE_X86, Arch::X86),
    (header::COFF_MACHINE_X86_64, Arch::X86_64),
    (header::COFF_MACHINE_ARM64, Arch::Arm64),
    (header::COFF_MACHINE_ARM, Arch::Arm),
    (header::COFF_MACHINE_ARMNT, Arch::Arm),
    (header::COFF_MACHINE_THUMB, Arch::Arm),
    (header::COFF_MACHINE_POWERPC, Arch::PowerPc),
    (header::COFF_MACHINE_POWERPCFP, Arch::PowerPc),
    (header::COFF_MACHINE_MIPS16, Arch::Mips),
    (header::COFF_MACHINE_MIPSFPU, Arch::Mips),
    (header::COFF_MACHINE_R4000, Arch::Mips),
    (header::COFF_MACHINE_RISCV32, Arch::Riscv),
    (header::COFF_MACHINE_RISCV64, Arch::Riscv64),
    (header::COFF_MACHINE_SH3, Arch::SuperH),
    (header::COFF_MACHINE_SH4, Arch::SuperH),
    (header::COFF_MACHINE_LOONGARCH64, Arch::LoongArch),
];

/// Maps a COFF machine identifier to an architecture family.
///
/// # Arguments
///
/// * `machine` - COFF machine field.
///
/// # Returns
///
/// The architecture family.
fn arch(machine: u16) -> Arch {
    let found = PE_ARCHES.iter().find(|(m, _)| *m == machine);
    found.map_or(Arch::Other, |(_, a)| *a)
}

/// Converts one section header.
///
/// # Arguments
///
/// * `table` - Section header.
/// * `base` - Image base address.
///
/// # Returns
///
/// The format-neutral section.
fn section(table: &SectionTable, base: u64) -> Section {
    let flags = table.characteristics;
    let name = table.name().unwrap_or("?");
    let address = base + u64::from(table.virtual_address);
    let mapped = Section::mapped(name, address, u64::from(table.virtual_size));
    let offset = u64::from(table.pointer_to_raw_data);
    let executable = flags & (SCN_CNT_CODE | SCN_MEM_EXECUTE) != 0;
    let backed = mapped.with_file(offset, u64::from(table.size_of_raw_data));
    backed.with_access(executable, flags & SCN_MEM_WRITE != 0)
}

/// Converts one import, locating its import address table slot.
///
/// # Arguments
///
/// * `item` - Parsed import.
/// * `base` - Image base address.
///
/// # Returns
///
/// The format-neutral import.
fn import(item: &PeImport, base: u64) -> Import {
    Import {
        library: item.dll.to_string(),
        name: item.name.to_string(),
        address: base + to_u64(item.offset),
    }
}

/// Converts one named export.
///
/// # Arguments
///
/// * `item` - Parsed export.
/// * `base` - Image base address.
///
/// # Returns
///
/// The exported symbol, or `None` for ordinal-only exports.
fn export(item: &Export, base: u64) -> Option<Symbol> {
    let name = item.name?.to_string();
    Some(Symbol { name, address: base + to_u64(item.rva) })
}

/// Evaluates the image's security mitigations.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// One verdict per mitigation.
fn protections(pe: &PE) -> Vec<Protection> {
    let header = pe.header.optional_header;
    let flags = header.map_or(0, |o| o.windows_fields.dll_characteristics);
    let table = FLAG_PROTECTIONS.iter().filter(|e| applies(pe, e));
    let mut found: Vec<Protection> = table.map(|e| flag_protection(flags, e)).collect();
    found.push(signature(pe));
    found
}

/// Evaluates Authenticode signing.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The signature verdict.
fn signature(pe: &PE) -> Protection {
    let signed = !pe.certificates.is_empty();
    Protection::new("Authenticode signature", signed, "Certificate table")
}

/// Reports whether a `DllCharacteristics` mitigation applies to the image.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
/// * `entry` - Mitigation name, bit, and flag name.
///
/// # Returns
///
/// `false` only for high-entropy ASLR on 32-bit images.
fn applies(pe: &PE, entry: &(&str, u16, &str)) -> bool {
    pe.is_64 || entry.1 != dll::IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA
}

/// Evaluates one `DllCharacteristics` mitigation.
///
/// # Arguments
///
/// * `flags` - `DllCharacteristics` value.
/// * `entry` - Mitigation name, bit, and flag name.
///
/// # Returns
///
/// The verdict.
fn flag_protection(flags: u16, &(name, bit, field): &(&str, u16, &str)) -> Protection {
    let evidence = format!("OptionalHeader.DllCharacteristics {field}");
    Protection::new(name, flags & bit != 0, &evidence)
}

/// Reports the link timestamp.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The timestamp fact.
fn timestamp(pe: &PE) -> Fact {
    let stamp = pe.header.coff_header.time_date_stamp;
    let note = stamp_note(stamp);
    let date = utc_timestamp(u64::from(stamp));
    let text = format!("{date} (raw 0x{stamp:08x}{note})");
    Fact::new("Link timestamp", text)
}

/// Annotates a link timestamp outside the plausible build era.
///
/// # Arguments
///
/// * `stamp` - Raw `TimeDateStamp`.
///
/// # Returns
///
/// An empty string, or [`HASH_NOTE`] for implausible values.
fn stamp_note(stamp: u32) -> &'static str {
    if plausible_stamp(stamp) {
        ""
    } else {
        HASH_NOTE
    }
}

/// Reports whether a link timestamp falls in the plausible build era.
///
/// # Arguments
///
/// * `stamp` - Raw `TimeDateStamp`.
///
/// # Returns
///
/// `true` between 1992 and 2035.
fn plausible_stamp(stamp: u32) -> bool {
    (PLAUSIBLE_STAMPS.0..PLAUSIBLE_STAMPS.1).contains(&stamp)
}

/// Note attached to timestamps outside the plausible build era.
const HASH_NOTE: &str = "; implausible, likely a reproducible-build hash";

/// Raw timestamps bounding 1992-01-01 and 2035-01-01.
const PLAUSIBLE_STAMPS: (u32, u32) = (0x2961_0480, 0x7A43_2B80);

/// Reports the Windows subsystem.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The subsystem fact, when an optional header exists.
fn subsystem(pe: &PE) -> Option<Fact> {
    let id = pe.header.optional_header?.windows_fields.subsystem;
    let name = SUBSYSTEMS.iter().find(|(key, _)| *key == id).map_or("Unknown", |(_, n)| n);
    Some(Fact::new("Subsystem", format!("{name} ({id})")))
}

/// Reports the linker version.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The linker version fact, when an optional header exists.
fn linker(pe: &PE) -> Option<Fact> {
    let fields = pe.header.optional_header?.standard_fields;
    let version = format!(
        "{}.{}",
        fields.major_linker_version, fields.minor_linker_version
    );
    Some(Fact::new("Linker version", version))
}

/// Reports the `CodeView` debug database path and identity.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The PDB fact, when a PDB 7.0 record exists.
fn pdb(pe: &PE) -> Option<Fact> {
    let info = pe.debug_data.as_ref()?.codeview_pdb70_debug_info.as_ref()?;
    let raw = String::from_utf8_lossy(info.filename);
    let path = raw.trim_end_matches('\0');
    let guid = crate::hashes::hex(&info.signature);
    let text = format!("{path} (GUID {guid}, age {})", info.age);
    Some(Fact::new("Debug database (PDB)", text))
}

/// Reports whether the image is a managed .NET assembly.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The CLR fact, when a CLR runtime header exists.
fn managed(pe: &PE) -> Option<Fact> {
    pe.clr_data.as_ref()?;
    Some(Fact::new(
        "Managed code",
        "CLR runtime header present (.NET assembly)",
    ))
}

/// Reports the presence of a Rich header.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The Rich header fact, when present.
fn rich(pe: &PE) -> Option<Fact> {
    let header = pe.header.rich_header.as_ref()?;
    let text = format!(
        "Present (Microsoft toolchain fingerprint, key 0x{:08x})",
        header.key
    );
    Some(Fact::new("Rich header", text))
}

/// Reports TLS callbacks, which run before the entry point.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The TLS fact, when callbacks exist.
fn tls(pe: &PE) -> Option<Fact> {
    let callbacks = &pe.tls_data.as_ref()?.callbacks;
    let list: Vec<String> = callbacks.iter().map(|c| format!("0x{c:x}")).collect();
    (!list.is_empty()).then(|| Fact::new("TLS callbacks", list.join(", ")))
}

/// Reports the module name recorded in the export directory.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
///
/// # Returns
///
/// The internal name fact, when present.
fn module_name(pe: &PE) -> Option<Fact> {
    pe.name.map(|name| Fact::new("Internal module name", name))
}

/// Populates symbols and function start addresses into the binary model.
///
/// # Arguments
///
/// * `binary` - Target binary model.
/// * `bytes` - Raw file bytes.
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
fn populate_pe_tables(binary: &mut Binary, bytes: &[u8], pe: &PE, base: u64) {
    binary.symbols = coff_symbols(bytes, pe, base);
    binary.function_starts = function_starts(pe, base);
}

/// Extracts COFF symbols defined in sections from the symbol table.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
///
/// # Returns
///
/// Defined symbols with computed virtual addresses.
fn coff_symbols(bytes: &[u8], pe: &PE, base: u64) -> Vec<Symbol> {
    let Ok(Some(symtab)) = pe.header.coff_header.symbols(bytes) else {
        return Vec::new();
    };
    let Ok(Some(strtab)) = pe.header.coff_header.strings(bytes) else {
        return Vec::new();
    };
    collect_coff_symbols(&symtab, &strtab, pe, base)
}

/// Collects defined section symbols from the COFF symbol table.
///
/// # Arguments
///
/// * `symtab` - Parsed symbol table.
/// * `strtab` - String table for out-of-line names.
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
///
/// # Returns
///
/// Parsed symbols.
fn collect_coff_symbols(
    symtab: &goblin::pe::symbol::SymbolTable<'_>,
    strtab: &goblin::strtab::Strtab<'_>,
    pe: &PE,
    base: u64,
) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    for (_idx, _inline, sym) in symtab.iter() {
        if let Some(entry) = coff_entry(&sym, strtab, pe, base) {
            symbols.push(entry);
        }
    }
    symbols
}

/// Converts a single COFF symbol to the model if defined in a section.
///
/// # Arguments
///
/// * `sym` - COFF symbol record.
/// * `strtab` - String table.
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
///
/// # Returns
///
/// Valid named symbol with virtual address, if defined.
fn coff_entry(
    sym: &goblin::pe::symbol::Symbol,
    strtab: &goblin::strtab::Strtab<'_>,
    pe: &PE,
    base: u64,
) -> Option<Symbol> {
    let name = sym.name(strtab).ok()?;
    if name.starts_with('.') || name.is_empty() {
        return None;
    }
    let addr = coff_address(pe, base, sym.section_number, sym.value)?;
    Some(Symbol::new(name, addr))
}

/// Computes the virtual address of a COFF section symbol.
///
/// # Arguments
///
/// * `pe` - Parsed PE.
/// * `base` - Image base address.
/// * `sec_num` - 1-based section index.
/// * `val` - Section-relative value.
///
/// # Returns
///
/// Virtual address when section index is valid.
fn coff_address(pe: &PE, base: u64, sec_num: i16, val: u32) -> Option<u64> {
    let idx = usize::try_from(sec_num.checked_sub(1)?).ok()?;
    let sec = pe.sections.get(idx)?;
    Some(base + u64::from(sec.virtual_address) + u64::from(val))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies architecture mapping for common machines.
    #[test]
    fn test_arch_maps_machines() {
        assert_eq!(arch(header::COFF_MACHINE_X86_64), Arch::X86_64);
        assert_eq!(arch(header::COFF_MACHINE_X86), Arch::X86);
        assert_eq!(arch(0xffff), Arch::Other);
    }

    /// Verifies that garbage is rejected as malformed.
    #[test]
    fn test_parse_rejects_garbage() {
        let result = parse(b"MZ\x00\x00garbage");
        assert!(matches!(result, Err(EngineError::Parse(_))));
    }

    /// Verifies section conversion of flags and addresses.
    #[test]
    fn test_section_converts_header() {
        let table = SectionTable {
            name: *b".text\0\0\0",
            virtual_address: 0x1000,
            characteristics: SCN_MEM_EXECUTE,
            ..SectionTable::default()
        };
        let converted = section(&table, 0x40_0000);
        assert_eq!((converted.address, converted.executable), (0x40_1000, true));
    }

    /// Verifies `DllCharacteristics` mitigation evaluation.
    #[test]
    fn test_flag_protection_reads_bits() {
        let aslr = &FLAG_PROTECTIONS[0];
        assert!(flag_protection(aslr.1, aslr).enabled);
        assert!(!flag_protection(0, aslr).enabled);
    }
}
