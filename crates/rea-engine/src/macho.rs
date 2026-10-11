//! Mach-O Loader
//!
//! **File:** `macho.rs`
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
};

// Import dependencies from goblin
use goblin::mach::{
    constants::{
        cputype, SECTION_TYPE, S_ATTR_PURE_INSTRUCTIONS, S_ATTR_SOME_INSTRUCTIONS, S_ZEROFILL,
    },
    header,
    load_command::CommandVariant,
    segment::{Section as MachSection, Segment},
    symbols::Nlist,
    Mach, MachO, MultiArch, SingleArch,
};

/// VM protection bit: writable.
const VM_PROT_WRITE: u32 = 0x2;

/// Symbol type mask.
const N_TYPE: u8 = 0x0e;

/// Symbol type: defined in a section.
const N_SECT: u8 = 0x0e;

/// Section attributes marking machine code.
const CODE_FLAGS: u32 = S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS;

/// Mach-O file type names.
const FILE_TYPES: [(u32, &str); 6] = [
    (header::MH_OBJECT, "Relocatable object"),
    (header::MH_EXECUTE, "Executable"),
    (header::MH_DYLIB, "Dynamic library"),
    (header::MH_DYLINKER, "Dynamic linker"),
    (header::MH_BUNDLE, "Loadable bundle"),
    (header::MH_KEXT_BUNDLE, "Kernel extension"),
];

/// Mitigations controlled by header flags.
const FLAG_PROTECTIONS: [(&str, u32, &str); 2] = [
    ("PIE", header::MH_PIE, "MH_PIE"),
    (
        "Non-executable heap",
        header::MH_NO_HEAP_EXECUTION,
        "MH_NO_HEAP_EXECUTION",
    ),
];

/// Parses a single or universal Mach-O; universal files use their first slice.
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
/// Returns [`EngineError::Parse`] when the Mach-O structures are malformed.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    match Mach::parse(bytes)? {
        Mach::Binary(macho) => Ok(convert(&macho)),
        Mach::Fat(multi) => first_slice(&multi),
    }
}

/// Converts the first slice of a universal binary.
///
/// # Arguments
///
/// * `multi` - Universal container.
///
/// # Returns
///
/// The first slice, annotated with the slice count.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when the slice is malformed or an archive.
fn first_slice(multi: &MultiArch) -> Result<Binary, EngineError> {
    let SingleArch::MachO(macho) = multi.get(0)? else {
        return Err(EngineError::Parse("universal slice is an archive".into()));
    };
    Ok(universal(convert(&macho), multi.narches))
}

/// Records that the binary came from a universal container.
///
/// # Arguments
///
/// * `binary` - Binary converted from the first slice.
/// * `count` - Number of architecture slices.
///
/// # Returns
///
/// The binary with a universal-container fact.
fn universal(mut binary: Binary, count: usize) -> Binary {
    let text = format!("{count} slices; analyzed the first ({})", binary.arch_name);
    binary.facts.push(Fact::new("Universal binary", text));
    binary
}

/// Converts one Mach-O slice.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// The format-neutral binary.
fn convert(macho: &MachO) -> Binary {
    let mut binary = identity(macho);
    binary.sections = macho.segments.iter().flat_map(segment_sections).collect();
    binary.imports = imports(macho);
    binary.exports = exports(macho, binary.image_base);
    binary.symbols = symbols(macho);
    binary.protections = protections(macho, &binary.imports);
    binary
}

/// Builds the identity fields of the model.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// A binary with format, architecture, kind, base, entry, and libraries set.
fn identity(macho: &MachO) -> Binary {
    let (cpu, sub) = (macho.header.cputype, macho.header.cpusubtype);
    let bits = bits_of(macho.is_64);
    let mut binary = Binary::new(Format::MachO, arch(cpu), arch_name(cpu, sub), bits);
    binary.kind = file_type(macho.header.filetype).to_string();
    binary.image_base = text_base(macho);
    binary.entry = entry_address(macho, binary.image_base);
    binary.libraries = libraries(macho);
    binary
}

/// Maps a CPU type to an architecture family.
///
/// # Arguments
///
/// * `cpu` - Mach-O CPU type.
///
/// # Returns
///
/// The architecture family.
fn arch(cpu: u32) -> Arch {
    match cpu {
        cputype::CPU_TYPE_X86 => Arch::X86,
        cputype::CPU_TYPE_X86_64 => Arch::X86_64,
        cputype::CPU_TYPE_ARM => Arch::Arm,
        cputype::CPU_TYPE_ARM64 => Arch::Arm64,
        _ => Arch::Other,
    }
}

/// Names a CPU type and subtype.
///
/// # Arguments
///
/// * `cpu` - Mach-O CPU type.
/// * `sub` - Mach-O CPU subtype.
///
/// # Returns
///
/// The architecture name, or `unknown`.
fn arch_name(cpu: u32, sub: u32) -> &'static str {
    cputype::get_arch_name_from_types(cpu, sub).unwrap_or("unknown")
}

/// Names a Mach-O file type.
///
/// # Arguments
///
/// * `filetype` - Header file type.
///
/// # Returns
///
/// A human-readable image kind.
fn file_type(filetype: u32) -> &'static str {
    let found = FILE_TYPES.iter().find(|(id, _)| *id == filetype);
    found.map_or("Unknown file type", |(_, name)| name)
}

/// Finds the virtual address of the `__TEXT` segment.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// The `__TEXT` load address, or zero.
fn text_base(macho: &MachO) -> u64 {
    let text = macho.segments.iter().find(|s| is_text(s));
    text.map_or(0, |s| s.vmaddr)
}

/// Reports whether a segment is `__TEXT`.
///
/// # Arguments
///
/// * `segment` - Load segment.
///
/// # Returns
///
/// `true` for the `__TEXT` segment.
fn is_text(segment: &Segment) -> bool {
    segment.name().is_ok_and(|name| name == "__TEXT")
}

/// Computes the entry point's virtual address.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
/// * `base` - `__TEXT` load address.
///
/// # Returns
///
/// The entry address; `LC_MAIN` offsets are rebased onto `__TEXT`.
fn entry_address(macho: &MachO, base: u64) -> u64 {
    if macho.old_style_entry {
        macho.entry
    } else {
        base + macho.entry
    }
}

/// Lists linked libraries.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// Library install names, excluding the `self` placeholder.
fn libraries(macho: &MachO) -> Vec<String> {
    let linked = macho.libs.iter().filter(|l| **l != "self");
    linked.map(ToString::to_string).collect()
}

/// Converts the sections of one segment.
///
/// # Arguments
///
/// * `segment` - Load segment.
///
/// # Returns
///
/// The segment's sections.
fn segment_sections(segment: &Segment) -> Vec<Section> {
    let writable = segment.initprot & VM_PROT_WRITE != 0;
    let parsed = segment.sections().unwrap_or_default();
    parsed.iter().map(|(s, _)| convert_section(s, writable)).collect()
}

/// Converts one section.
///
/// # Arguments
///
/// * `section` - Mach-O section.
/// * `writable` - Whether its segment is writable.
///
/// # Returns
///
/// The format-neutral section named `segment,section`.
fn convert_section(section: &MachSection, writable: bool) -> Section {
    let name = section_name(section);
    let zero_fill = section.flags & SECTION_TYPE == S_ZEROFILL;
    let file_size = if zero_fill { 0 } else { section.size };
    let mapped = Section::mapped(&name, section.addr, section.size);
    let backed = mapped.with_file(u64::from(section.offset), file_size);
    backed.with_access(section.flags & CODE_FLAGS != 0, writable)
}

/// Formats a section's qualified name.
///
/// # Arguments
///
/// * `section` - Mach-O section.
///
/// # Returns
///
/// `segment,section`.
fn section_name(section: &MachSection) -> String {
    let segment = section.segname().unwrap_or("?");
    format!("{segment},{}", section.name().unwrap_or("?"))
}

/// Lists bound imports.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// Imports with libraries; slot addresses are not tracked.
fn imports(macho: &MachO) -> Vec<Import> {
    let bound = macho.imports().unwrap_or_default();
    bound.iter().map(|i| Import::new(i.dylib, i.name, 0)).collect()
}

/// Lists exported symbols.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
/// * `base` - Image base used to rebase export offsets.
///
/// # Returns
///
/// Exported symbols with virtual addresses.
fn exports(macho: &MachO, base: u64) -> Vec<Symbol> {
    let found = macho.exports().unwrap_or_default();
    found.iter().map(|e| Symbol::new(&e.name, base + e.offset)).collect()
}

/// Lists symbols defined in sections.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// Section-defined symbols with nonzero addresses.
fn symbols(macho: &MachO) -> Vec<Symbol> {
    let parsed = macho.symbols().filter_map(Result::ok);
    let defined = parsed.filter(|(_, n)| is_defined(n));
    defined.map(|(name, n)| Symbol::new(name, n.n_value)).collect()
}

/// Reports whether a symbol is defined in a section at a real address.
///
/// # Arguments
///
/// * `nlist` - Symbol table entry.
///
/// # Returns
///
/// `true` for section-defined symbols with nonzero values.
fn is_defined(nlist: &Nlist) -> bool {
    nlist.n_type & N_TYPE == N_SECT && nlist.n_value != 0
}

/// Evaluates the image's security mitigations.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
/// * `imports` - Imported symbols.
///
/// # Returns
///
/// One verdict per mitigation.
fn protections(macho: &MachO, imports: &[Import]) -> Vec<Protection> {
    let flags = macho.header.flags;
    let table = FLAG_PROTECTIONS.iter();
    let mut found: Vec<Protection> = table.map(|e| flag_protection(flags, e)).collect();
    found.push(canary(imports));
    found.push(signature(macho));
    found
}

/// Evaluates one header-flag mitigation.
///
/// # Arguments
///
/// * `flags` - Header flags.
/// * `entry` - Mitigation name, bit, and flag name.
///
/// # Returns
///
/// The verdict.
fn flag_protection(flags: u32, &(name, bit, field): &(&str, u32, &str)) -> Protection {
    Protection::new(name, flags & bit != 0, &format!("Header flag {field}"))
}

/// Evaluates stack-protector instrumentation.
///
/// # Arguments
///
/// * `imports` - Imported symbols.
///
/// # Returns
///
/// The canary verdict.
fn canary(imports: &[Import]) -> Protection {
    let found = imports.iter().any(|i| i.name.contains("stack_chk_fail"));
    Protection::new("Stack canary", found, "Imports ___stack_chk_fail")
}

/// Evaluates code signing.
///
/// # Arguments
///
/// * `macho` - Parsed Mach-O.
///
/// # Returns
///
/// The code signature verdict.
fn signature(macho: &MachO) -> Protection {
    let mut commands = macho.load_commands.iter();
    let signed = commands.any(|c| matches!(c.command, CommandVariant::CodeSignature(_)));
    Protection::new("Code signature", signed, "LC_CODE_SIGNATURE load command")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies architecture mapping for common CPU types.
    #[test]
    fn test_arch_maps_cpu_types() {
        assert_eq!(arch(cputype::CPU_TYPE_ARM64), Arch::Arm64);
        assert_eq!(arch(cputype::CPU_TYPE_X86_64), Arch::X86_64);
        assert_eq!(arch(0), Arch::Other);
    }

    /// Verifies file type naming.
    #[test]
    fn test_file_type_names() {
        assert_eq!(file_type(header::MH_DYLIB), "Dynamic library");
        assert_eq!(file_type(0xff), "Unknown file type");
    }

    /// Verifies that truncated input is rejected as malformed.
    #[test]
    fn test_parse_rejects_truncated() {
        let result = parse(&[0xcf, 0xfa, 0xed, 0xfe, 7]);
        assert!(matches!(result, Err(EngineError::Parse(_))));
    }

    /// Verifies header-flag mitigation evaluation.
    #[test]
    fn test_flag_protection_reads_bits() {
        let verdict = flag_protection(header::MH_PIE, &FLAG_PROTECTIONS[0]);
        assert!(verdict.enabled);
        assert!(!flag_protection(0, &FLAG_PROTECTIONS[1]).enabled);
    }
}
