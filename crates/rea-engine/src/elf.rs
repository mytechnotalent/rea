//! ELF Loader
//!
//! **File:** `elf.rs`
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
use std::collections::HashMap;

// Import crate items
use crate::{
    detect::Format,
    error::EngineError,
    model::{bits_of, Arch, Binary, Fact, Import, Protection, Section, Symbol},
    numeric::window,
    relr,
};

// Import dependencies from goblin
use goblin::{
    elf::{
        dynamic::{Dyn, DF_1_NOW, DF_BIND_NOW, DT_BIND_NOW, DT_FLAGS, DT_FLAGS_1},
        header::{self, machine_to_str},
        program_header::{ProgramHeader, PF_W, PF_X, PT_GNU_RELRO, PT_GNU_STACK, PT_LOAD},
        reloc::R_X86_64_RELATIVE,
        section_header::{SectionHeader, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS, SHT_NULL},
        Elf, Symtab,
    },
    strtab::Strtab,
};

/// Optional metadata extractors run against every ELF.
const FACTS: [fn(&Elf) -> Option<Fact>; 3] = [interpreter, soname, search_paths];

/// Symbols whose import shows stack-protector instrumentation.
const CANARY_SYMBOLS: [&str; 2] = ["__stack_chk_fail", "__stack_chk_guard"];

/// Parses an ELF image into the format-neutral model.
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
/// Returns [`EngineError::Parse`] when the ELF structures are malformed.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    let elf = Elf::parse(bytes)?;
    let mut binary = identity(&elf);
    binary.sections = sections(&elf);
    binary.imports = imports(&elf);
    (binary.exports, binary.symbols) = symbols(&elf);
    binary.function_starts = code_pointers(&elf, bytes, &binary);
    (binary.protections, binary.facts) = (protections(&elf, &binary.imports), facts(&elf));
    Ok(binary)
}

/// Builds the identity fields of the model.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// A binary with format, architecture, kind, entry, and libraries set.
fn identity(elf: &Elf) -> Binary {
    let machine = elf.header.e_machine;
    let mut binary = base_binary(elf, machine);
    binary.big_endian = !elf.little_endian;
    binary.thumb = machine == header::EM_ARM && elf.entry & 1 == 1;
    binary.kind = kind(elf).to_string();
    binary.entry = elf.entry & !u64::from(binary.thumb);
    binary.libraries = elf.libraries.iter().map(ToString::to_string).collect();
    binary
}

/// Builds the format, architecture, and width of an ELF binary.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `machine` - `e_machine` field.
///
/// # Returns
///
/// A binary with identity set and other fields empty.
fn base_binary(elf: &Elf, machine: u16) -> Binary {
    let name = machine_to_str(machine);
    Binary::new(
        Format::Elf,
        arch(machine, elf.is_64),
        name,
        bits_of(elf.is_64),
    )
}

/// Maps an ELF machine identifier to an architecture family.
///
/// # Arguments
///
/// * `machine` - `e_machine` field.
///
/// # Returns
///
/// The architecture family.
fn arch(machine: u16, is_64: bool) -> Arch {
    let found = ELF_ARCHES.iter().find(|(id, _)| *id == machine);
    found.map_or(Arch::Other, |(_, pick)| pick(is_64))
}

/// Picks an architecture family from a 64-bit flag.
type ArchPick = fn(bool) -> Arch;

/// ELF machine identifiers mapped to an architecture selector.
const ELF_ARCHES: [(u16, ArchPick); 13] = [
    (header::EM_386, |_| Arch::X86),
    (header::EM_X86_64, |_| Arch::X86_64),
    (header::EM_ARM, |_| Arch::Arm),
    (header::EM_AARCH64, |_| Arch::Arm64),
    (
        header::EM_MIPS,
        |w| if w { Arch::Mips64 } else { Arch::Mips },
    ),
    (header::EM_PPC, |_| Arch::PowerPc),
    (header::EM_PPC64, |_| Arch::PowerPc64),
    (header::EM_RISCV, |w| {
        if w {
            Arch::Riscv64
        } else {
            Arch::Riscv
        }
    }),
    (header::EM_SPARC, |_| Arch::Sparc),
    (header::EM_SPARCV9, |_| Arch::Sparc),
    (header::EM_SH, |_| Arch::SuperH),
    (header::EM_AVR, |_| Arch::Avr),
    (header::EM_68K, |_| Arch::M68k),
];

/// Describes the object type.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// A human-readable image kind.
fn kind(elf: &Elf) -> &'static str {
    match (elf.header.e_type, elf.interpreter.is_some()) {
        (header::ET_EXEC, _) => "Executable",
        (header::ET_DYN, true) => "Position-independent executable",
        (header::ET_DYN, false) => "Shared object",
        (header::ET_REL, _) => "Relocatable object",
        (header::ET_CORE, _) => "Core dump",
        _ => "Unknown object type",
    }
}

/// Lists sections, falling back to loadable segments when none exist.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The format-neutral sections.
fn sections(elf: &Elf) -> Vec<Section> {
    let headers = elf.section_headers.iter();
    let real = headers.filter(|h| h.sh_type != SHT_NULL);
    let named: Vec<Section> = real.map(|h| from_header(elf, h)).collect();
    if named.is_empty() {
        segments(elf)
    } else {
        named
    }
}

/// Lists loadable segments as pseudo-sections.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// One pseudo-section per `PT_LOAD` header.
fn segments(elf: &Elf) -> Vec<Section> {
    let headers = elf.program_headers.iter().enumerate();
    headers.filter_map(from_segment).collect()
}

/// Converts one section header.
///
/// # Arguments
///
/// * `elf` - Parsed ELF, for the section name table.
/// * `h` - Section header.
///
/// # Returns
///
/// The format-neutral section.
fn from_header(elf: &Elf, h: &SectionHeader) -> Section {
    let name = elf.shdr_strtab.get_at(h.sh_name).unwrap_or("?");
    let nobits = h.sh_type == SHT_NOBITS;
    let file_size = if nobits { 0 } else { h.sh_size };
    let executable = h.sh_flags & u64::from(SHF_EXECINSTR) != 0;
    let writable = h.sh_flags & u64::from(SHF_WRITE) != 0;
    let mapped = Section::mapped(name, h.sh_addr, h.sh_size);
    let backed = mapped.with_file(h.sh_offset, file_size);
    backed.with_access(executable, writable)
}

/// Converts one loadable program header into a pseudo-section.
///
/// # Arguments
///
/// * `entry` - Index and program header.
///
/// # Returns
///
/// The pseudo-section for `PT_LOAD` headers, otherwise `None`.
fn from_segment((index, p): (usize, &ProgramHeader)) -> Option<Section> {
    let name = format!("LOAD{index}");
    let mapped = Section::mapped(&name, p.p_vaddr, p.p_memsz);
    let backed = mapped.with_file(p.p_offset, p.p_filesz);
    let access = (p.p_flags & PF_X != 0, p.p_flags & PF_W != 0);
    let section = backed.with_access(access.0, access.1);
    (p.p_type == PT_LOAD).then_some(section)
}

/// Lists imported symbols with their relocation slots.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// Every named undefined dynamic symbol.
fn imports(elf: &Elf) -> Vec<Import> {
    let slots = relocation_slots(elf);
    let address = |index: usize| slots.get(&index).copied().unwrap_or(0);
    let named = undefined_symbols(elf).into_iter();
    named.map(|(i, name)| Import::new("", name, address(i))).collect()
}

/// Lists named undefined dynamic symbols with their indices.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// `(index, name)` pairs.
fn undefined_symbols<'a>(elf: &'a Elf) -> Vec<(usize, &'a str)> {
    let symbols = elf.dynsyms.iter().enumerate();
    let undefined = symbols.filter(|(_, s)| s.is_import());
    let named = undefined.map(|(i, s)| (i, dynamic_name(elf, s.st_name)));
    named.filter(|(_, name)| !name.is_empty()).collect()
}

/// Reads a name from the dynamic string table.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `index` - String table offset.
///
/// # Returns
///
/// The name, or an empty string when out of range.
fn dynamic_name<'a>(elf: &'a Elf, index: usize) -> &'a str {
    elf.dynstrtab.get_at(index).unwrap_or_default()
}

/// Maps dynamic symbol indices to the addresses their relocations patch.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// Symbol index to relocated slot address.
fn relocation_slots(elf: &Elf) -> HashMap<usize, u64> {
    let plt = elf.pltrelocs.iter();
    let relocs = plt.chain(elf.dynrelas.iter()).chain(elf.dynrels.iter());
    relocs.map(|r| (r.r_sym, r.r_offset)).collect()
}

/// Lists exported and local function symbols.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// `(exports, symbols)` from the dynamic and static tables.
fn symbols(elf: &Elf) -> (Vec<Symbol>, Vec<Symbol>) {
    let exports = functions(&elf.dynsyms, &elf.dynstrtab);
    (exports, functions(&elf.syms, &elf.strtab))
}

/// Lists defined function symbols from one symbol table.
///
/// # Arguments
///
/// * `symtab` - Symbol table.
/// * `strtab` - Matching string table.
///
/// # Returns
///
/// Named functions with nonzero addresses.
fn functions(symtab: &Symtab, strtab: &Strtab) -> Vec<Symbol> {
    let defined = symtab.iter().filter(|s| s.is_function() && s.st_value != 0);
    let named = defined.map(|s| (strtab.get_at(s.st_name).unwrap_or_default(), s.st_value));
    let real = named.filter(|(name, _)| !name.is_empty());
    real.map(|(name, address)| Symbol::new(name, address)).collect()
}

/// Evaluates the image's security mitigations.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `imports` - Imported symbols.
///
/// # Returns
///
/// One verdict per mitigation.
fn protections(elf: &Elf, imports: &[Import]) -> Vec<Protection> {
    let mut found = vec![nx(elf), relro(elf), canary(imports), fortify(imports)];
    if is_program(elf) {
        found.insert(1, pie(elf));
    }
    found
}

/// Reports whether the object is a program rather than a library or object file.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// `true` for `ET_EXEC` and for `ET_DYN` with a program interpreter.
fn is_program(elf: &Elf) -> bool {
    let dynamic = elf.header.e_type == header::ET_DYN && elf.interpreter.is_some();
    elf.header.e_type == header::ET_EXEC || dynamic
}

/// Dynamic tag holding the size of the packed relative relocation table.
const DT_RELRSZ: u64 = 35;

/// Dynamic tag holding the address of the packed relative relocation table.
const DT_RELR: u64 = 36;

/// Lists code pointers fixed up by every supported relative relocation form.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `bytes` - Whole target file.
/// * `binary` - Binary with sections already converted.
///
/// # Returns
///
/// Relocated pointer values; callers keep only those inside code.
fn code_pointers(elf: &Elf, bytes: &[u8], binary: &Binary) -> Vec<u64> {
    let mut targets = relative_targets(elf);
    targets.extend(relr_targets(elf, bytes, binary));
    targets
}

/// Lists pointer values relocated by a 64-bit little-endian `DT_RELR` table.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `bytes` - Whole target file.
/// * `binary` - Binary with sections already converted.
///
/// # Returns
///
/// The pointer stored at every relocated slot.
fn relr_targets(elf: &Elf, bytes: &[u8], binary: &Binary) -> Vec<u64> {
    if !(elf.is_64 && elf.little_endian) {
        return Vec::new();
    }
    let table = relr_table(elf, bytes, binary);
    let words: Vec<u64> = table.as_chunks::<8>().0.iter().map(|chunk| le_u64(chunk)).collect();
    let slots = relr::locations(&words).into_iter();
    slots.filter_map(|s| pointer_at(binary, bytes, s)).collect()
}

/// Locates the bytes of the `DT_RELR` table.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `bytes` - Whole target file.
/// * `binary` - Binary with sections already converted.
///
/// # Returns
///
/// The table bytes, or an empty slice when absent.
fn relr_table<'a>(elf: &Elf, bytes: &'a [u8], binary: &Binary) -> &'a [u8] {
    let size = dynamic_value(elf, DT_RELRSZ).unwrap_or(0);
    let address = dynamic_value(elf, DT_RELR);
    let offset = address.and_then(|a| binary.offset_of(a));
    offset.map_or(&[], |o| window(bytes, o, size))
}

/// Reads one value from the dynamic section.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
/// * `tag` - Dynamic tag.
///
/// # Returns
///
/// The value of the first entry with the tag.
fn dynamic_value(elf: &Elf, tag: u64) -> Option<u64> {
    let dynamic = elf.dynamic.as_ref()?;
    dynamic.dyns.iter().find(|d| d.d_tag == tag).map(|d| d.d_val)
}

/// Reads the little-endian pointer stored at a virtual address.
///
/// # Arguments
///
/// * `binary` - Binary with sections.
/// * `bytes` - Whole target file.
/// * `address` - Virtual address of the pointer.
///
/// # Returns
///
/// The pointer value, when the address is file-backed.
fn pointer_at(binary: &Binary, bytes: &[u8], address: u64) -> Option<u64> {
    let offset = binary.offset_of(address)?;
    let word = window(bytes, offset, 8);
    (word.len() == 8).then(|| le_u64(word))
}

/// Decodes an eight-byte little-endian word.
///
/// # Arguments
///
/// * `chunk` - Exactly eight bytes.
///
/// # Returns
///
/// The word.
fn le_u64(chunk: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(chunk);
    u64::from_le_bytes(word)
}

/// Lists code pointers fixed up by x86-64 `R_X86_64_RELATIVE` relocations.
///
/// These include function tables such as `.init_array` and dispatch tables.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// Relocation targets; callers keep only those inside code.
fn relative_targets(elf: &Elf) -> Vec<u64> {
    let x86_64 = elf.header.e_machine == header::EM_X86_64;
    let relative = elf.dynrelas.iter().filter(|r| r.r_type == R_X86_64_RELATIVE);
    let targets = relative.filter_map(|r| u64::try_from(r.r_addend?).ok());
    targets.filter(|_| x86_64).collect()
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
    let found = imports.iter().any(|i| CANARY_SYMBOLS.contains(&i.name.as_str()));
    Protection::new(
        "Stack canary",
        found,
        "Imports __stack_chk_fail or __stack_chk_guard",
    )
}

/// Evaluates source fortification.
///
/// # Arguments
///
/// * `imports` - Imported symbols.
///
/// # Returns
///
/// The `FORTIFY_SOURCE` verdict.
fn fortify(imports: &[Import]) -> Protection {
    let checked = imports.iter().any(|i| is_fortified(&i.name));
    Protection::new(
        "FORTIFY_SOURCE",
        checked,
        "Imports checked __*_chk functions",
    )
}

/// Reports whether a symbol is a fortified libc variant.
///
/// # Arguments
///
/// * `name` - Symbol name.
///
/// # Returns
///
/// `true` for names like `__memcpy_chk`.
fn is_fortified(name: &str) -> bool {
    name.starts_with("__") && name.ends_with("_chk")
}

/// Evaluates the non-executable stack.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The NX verdict.
fn nx(elf: &Elf) -> Protection {
    let stack = elf.program_headers.iter().find(|p| p.p_type == PT_GNU_STACK);
    let enabled = stack.is_some_and(|p| p.p_flags & PF_X == 0);
    Protection::new("NX (non-executable stack)", enabled, "PT_GNU_STACK flags")
}

/// Evaluates position independence.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The PIE verdict.
fn pie(elf: &Elf) -> Protection {
    let enabled = elf.header.e_type == header::ET_DYN && elf.interpreter.is_some();
    Protection::new("PIE", enabled, "ET_DYN with a program interpreter")
}

/// Evaluates read-only relocations.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The RELRO verdict, distinguishing partial and full.
fn relro(elf: &Elf) -> Protection {
    let segment = elf.program_headers.iter().any(|p| p.p_type == PT_GNU_RELRO);
    let state = relro_state(segment, bind_now(elf));
    let evidence = format!("{state} (PT_GNU_RELRO and BIND_NOW)");
    Protection::new("RELRO", segment, &evidence)
}

/// Names the RELRO level.
///
/// # Arguments
///
/// * `segment` - Whether `PT_GNU_RELRO` exists.
/// * `now` - Whether immediate binding is requested.
///
/// # Returns
///
/// `None`, `Partial`, or `Full`.
fn relro_state(segment: bool, now: bool) -> &'static str {
    match (segment, now) {
        (false, _) => "None",
        (true, false) => "Partial",
        (true, true) => "Full",
    }
}

/// Reports whether the dynamic section requests immediate binding.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// `true` when any dynamic entry requests it.
fn bind_now(elf: &Elf) -> bool {
    let dynamic = elf.dynamic.as_ref();
    dynamic.is_some_and(|d| d.dyns.iter().any(is_bind_now))
}

/// Reports whether a dynamic entry requests immediate binding.
///
/// # Arguments
///
/// * `entry` - Dynamic section entry.
///
/// # Returns
///
/// `true` for `DT_BIND_NOW`, `DF_BIND_NOW`, or `DF_1_NOW`.
fn is_bind_now(entry: &Dyn) -> bool {
    let flags = entry.d_tag == DT_FLAGS && entry.d_val & DF_BIND_NOW != 0;
    let flags_1 = entry.d_tag == DT_FLAGS_1 && entry.d_val & DF_1_NOW != 0;
    flags || flags_1 || entry.d_tag == DT_BIND_NOW
}

/// Collects metadata facts.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// Every applicable fact, ending with the symbol table state.
fn facts(elf: &Elf) -> Vec<Fact> {
    let mut found: Vec<Fact> = FACTS.iter().filter_map(|fact| fact(elf)).collect();
    found.push(symbol_table(elf));
    found
}

/// Reports the program interpreter.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The interpreter fact, when present.
fn interpreter(elf: &Elf) -> Option<Fact> {
    elf.interpreter.map(|path| Fact::new("Program interpreter", path))
}

/// Reports the shared object name.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The soname fact, when present.
fn soname(elf: &Elf) -> Option<Fact> {
    elf.soname.map(|name| Fact::new("Shared object name", name))
}

/// Reports library search paths embedded in the binary.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The search path fact, when any RPATH or RUNPATH exists.
fn search_paths(elf: &Elf) -> Option<Fact> {
    let all = elf.rpaths.iter().chain(elf.runpaths.iter());
    let paths: Vec<&str> = all.copied().collect();
    let joined = paths.join(":");
    (!paths.is_empty()).then(|| Fact::new("Library search paths", joined))
}

/// Reports whether the static symbol table survived stripping.
///
/// # Arguments
///
/// * `elf` - Parsed ELF.
///
/// # Returns
///
/// The symbol table fact.
fn symbol_table(elf: &Elf) -> Fact {
    let count = elf.syms.len();
    let text = format!("{count} static symbols");
    let state = if count == 0 {
        "Stripped (no static symbols)".to_string()
    } else {
        text
    };
    Fact::new("Symbol table", state)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a dynamic section entry.
    ///
    /// # Arguments
    ///
    /// * `d_tag` - Entry tag.
    /// * `d_val` - Entry value.
    ///
    /// # Returns
    ///
    /// The entry.
    fn entry(d_tag: u64, d_val: u64) -> Dyn {
        Dyn { d_tag, d_val }
    }

    /// Verifies architecture mapping for common machines.
    #[test]
    fn test_arch_maps_machines() {
        assert_eq!(arch(header::EM_X86_64, true), Arch::X86_64);
        assert_eq!(arch(header::EM_AARCH64, true), Arch::Arm64);
        assert_eq!(arch(header::EM_MIPS, true), Arch::Mips64);
        assert_eq!(arch(0xfff0, false), Arch::Other);
    }

    /// Verifies that truncated input is rejected as malformed.
    #[test]
    fn test_parse_rejects_truncated() {
        let result = parse(b"\x7fELF\x02\x01");
        assert!(matches!(result, Err(EngineError::Parse(_))));
    }

    /// Verifies detection of immediate binding entries.
    #[test]
    fn test_is_bind_now() {
        assert!(is_bind_now(&entry(DT_BIND_NOW, 0)));
        assert!(is_bind_now(&entry(DT_FLAGS, DF_BIND_NOW)));
        assert!(!is_bind_now(&entry(DT_FLAGS, 0)));
    }

    /// Verifies that only loadable segments become pseudo-sections.
    #[test]
    fn test_from_segment_keeps_load_only() {
        let mut header = ProgramHeader::default();
        (header.p_type, header.p_flags) = (PT_LOAD, PF_X);
        assert!(from_segment((0, &header)).is_some_and(|s| s.executable));
        header.p_type = PT_GNU_STACK;
        assert!(from_segment((1, &header)).is_none());
    }

    /// Verifies RELRO level naming and fortified-symbol detection.
    #[test]
    fn test_relro_state_and_fortify() {
        assert_eq!(relro_state(true, true), "Full");
        assert_eq!(relro_state(false, true), "None");
        assert!(is_fortified("__memcpy_chk"));
    }
}
