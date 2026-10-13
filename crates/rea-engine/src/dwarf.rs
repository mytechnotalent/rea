//! DWARF Debug Information Recovery
//!
//! **File:** `dwarf.rs`
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
use std::{collections::HashMap, fmt::Write as _};

// Import dependencies from goblin
use goblin::elf::{Elf, SectionHeader};

/// Debugging information entry tag for a compilation unit.
const TAG_COMPILE_UNIT: u64 = 0x11;

/// Debugging information entry tag for a subprogram.
const TAG_SUBPROGRAM: u64 = 0x2e;

/// Attribute naming a debugging information entry.
const AT_NAME: u64 = 0x03;

/// Attribute holding the low program counter of a subprogram.
const AT_LOW_PC: u64 = 0x11;

/// Attribute holding a compilation-directory path.
const AT_COMP_DIR: u64 = 0x1b;

/// Attribute holding a language-level linkage name.
const AT_LINKAGE_NAME: u64 = 0x6e;

/// Most functions retained.
const MAX_FUNCTIONS: usize = 20_000;

/// A recovered compilation unit.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompileUnit {
    /// Primary source file or unit name.
    pub name: String,
    /// Compilation directory.
    pub comp_dir: String,
}

/// A recovered function symbol.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DwarfFunction {
    /// Function name.
    pub name: String,
    /// Low program counter.
    pub low_pc: u64,
}

/// Recovered DWARF inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DwarfInfo {
    /// Debug information version.
    pub version: u16,
    /// Recovered compilation units.
    pub compile_units: Vec<CompileUnit>,
    /// Recovered functions.
    pub functions: Vec<DwarfFunction>,
}

/// A decoded attribute value.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    /// A string value.
    Str(String),
    /// An unsigned integer value.
    Uint(u64),
    /// A value skipped without interpretation.
    Skip,
}

impl Value {
    /// Returns the value as an unsigned integer.
    ///
    /// # Returns
    ///
    /// The integer, or zero for non-integer values.
    fn uint(&self) -> u64 {
        match self {
            Value::Uint(value) => *value,
            _ => 0,
        }
    }
}

/// A decoded abbreviation declaration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Abbrev {
    /// Entry tag.
    tag: u64,
    /// Attribute and form pairs.
    attrs: Vec<(u64, u64)>,
}

/// Attributes recovered from one entry.
#[derive(Debug, Clone, Default)]
struct DieAttrs {
    /// Entry name.
    name: Option<String>,
    /// Compilation directory.
    comp_dir: String,
    /// Low program counter.
    low_pc: u64,
}

/// Byte cursor over a debug section.
struct Reader<'a> {
    /// Source bytes.
    data: &'a [u8],
    /// Current offset.
    pos: usize,
    /// Exclusive limit for reads.
    end: usize,
}

impl<'a> Reader<'a> {
    /// Creates a reader over a section.
    ///
    /// # Arguments
    ///
    /// * `data` - Section bytes.
    /// * `pos` - Initial offset.
    ///
    /// # Returns
    ///
    /// The reader.
    fn new(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos, end: data.len() }
    }

    /// Sets the exclusive read limit.
    ///
    /// # Arguments
    ///
    /// * `end` - New limit.
    fn set_end(&mut self, end: usize) {
        self.end = end;
    }

    /// Returns the current offset.
    ///
    /// # Returns
    ///
    /// The offset.
    fn pos(&self) -> usize {
        self.pos
    }

    /// Reads one byte.
    ///
    /// # Returns
    ///
    /// The byte, or `None`.
    fn u8(&mut self) -> Option<u8> {
        let value = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(value)
    }

    /// Reads a little-endian 16-bit word.
    ///
    /// # Returns
    ///
    /// The word, or `None`.
    fn u16(&mut self) -> Option<u16> {
        let b = self.data.get(self.pos..self.pos + 2)?;
        self.pos += 2;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    /// Reads a little-endian 32-bit word.
    ///
    /// # Returns
    ///
    /// The word, or `None`.
    fn u32(&mut self) -> Option<u32> {
        let b = self.data.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads a little-endian 64-bit word.
    ///
    /// # Returns
    ///
    /// The word, or `None`.
    fn u64(&mut self) -> Option<u64> {
        let b = self.data.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Advances past a number of bytes.
    ///
    /// # Arguments
    ///
    /// * `n` - Byte count.
    ///
    /// # Returns
    ///
    /// The skipped bytes, or `None`.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.data.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(b)
    }

    /// Reads an unsigned LEB128 integer.
    ///
    /// # Returns
    ///
    /// The value, or `None`.
    fn uleb(&mut self) -> Option<u64> {
        let end = self.leb_end()?;
        let raw = self.data.get(self.pos..end)?;
        self.pos = end;
        decode_uleb(raw)
    }

    /// Reads a signed LEB128 integer.
    ///
    /// # Returns
    ///
    /// The value, or `None`.
    fn sleb(&mut self) -> Option<i64> {
        let end = self.leb_end()?;
        let raw = self.data.get(self.pos..end)?;
        self.pos = end;
        decode_sleb(raw)
    }

    /// Locates the end of the LEB128 integer at the cursor.
    ///
    /// # Returns
    ///
    /// The end offset, or `None`.
    fn leb_end(&self) -> Option<usize> {
        let rest = self.data.get(self.pos..)?;
        let continuation = rest.iter().take_while(|b| **b & 0x80 != 0).count();
        if continuation >= 9 {
            return None;
        }
        Some(self.pos + continuation + 1)
    }
}

/// Decodes an unsigned LEB128 value.
///
/// # Arguments
///
/// * `bytes` - Encoded bytes.
///
/// # Returns
///
/// The decoded value, or `None`.
fn decode_uleb(bytes: &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for (i, &byte) in bytes.iter().enumerate() {
        value |= u64::from(byte & 0x7f) << (7 * u32::try_from(i).ok()?);
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// Decodes a signed LEB128 value.
///
/// # Arguments
///
/// * `bytes` - Encoded bytes.
///
/// # Returns
///
/// The decoded value, or `None`.
fn decode_sleb(bytes: &[u8]) -> Option<i64> {
    let last = *bytes.last()?;
    let raw = decode_uleb(bytes)?;
    let bits = 7 * bytes.len();
    if bits < 64 && last & 0x40 != 0 {
        return Some(sign_extend(raw, bits));
    }
    Some(i64::from_ne_bytes(raw.to_ne_bytes()))
}

/// Sign-extends the low bits of a value.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `bits` - Number of meaningful bits.
///
/// # Returns
///
/// The sign-extended value.
fn sign_extend(value: u64, bits: usize) -> i64 {
    let shift = 64 - bits;
    i64::from_ne_bytes((value << shift).to_ne_bytes()) >> shift
}

/// Reads a section body by header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `shdr` - Section header.
///
/// # Returns
///
/// The section bytes, or `None`.
fn section<'a>(bytes: &'a [u8], shdr: &SectionHeader) -> Option<&'a [u8]> {
    let start = usize::try_from(shdr.sh_offset).ok()?;
    let end = start.checked_add(usize::try_from(shdr.sh_size).ok()?)?;
    bytes.get(start..end)
}

/// Finds a named section body.
///
/// # Arguments
///
/// * `elf` - Parsed ELF file.
/// * `bytes` - Whole target file.
/// * `name` - Section name.
///
/// # Returns
///
/// The section bytes, or `None`.
fn find_section<'a>(elf: &Elf, bytes: &'a [u8], name: &str) -> Option<&'a [u8]> {
    for shdr in &elf.section_headers {
        if elf.shdr_strtab.get_at(shdr.sh_name) == Some(name) {
            return section(bytes, shdr);
        }
    }
    None
}

/// Parses the debug sections of an ELF file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The abbreviation, information, and string sections.
fn dwarf_sections(bytes: &[u8]) -> Option<(&[u8], &[u8], &[u8])> {
    if !bytes.starts_with(b"\x7fELF") {
        return None;
    }
    let elf = Elf::parse(bytes).ok()?;
    let abbrev = find_section(&elf, bytes, ".debug_abbrev")?;
    let info = find_section(&elf, bytes, ".debug_info")?;
    let strs = find_section(&elf, bytes, ".debug_str").unwrap_or_default();
    Some((abbrev, info, strs))
}

/// Parses DWARF debug information from an ELF file.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered inventory, or `None` when DWARF is absent.
#[must_use]
pub fn parse_dwarf(bytes: &[u8]) -> Option<DwarfInfo> {
    let (abbrev_bytes, info, strs) = dwarf_sections(bytes)?;
    let abbrev = parse_abbrev(abbrev_bytes);
    let mut dwarf = DwarfInfo::default();
    parse_units(info, &abbrev, strs, &mut dwarf);
    Some(dwarf)
}

/// Parses every abbreviation declaration.
///
/// # Arguments
///
/// * `bytes` - Abbreviation section.
///
/// # Returns
///
/// The declarations keyed by code.
fn parse_abbrev(bytes: &[u8]) -> HashMap<u64, Abbrev> {
    let mut map = HashMap::new();
    let mut reader = Reader::new(bytes, 0);
    while reader.pos() < bytes.len() {
        if parse_abbrev_decl(&mut reader, &mut map).is_none() {
            break;
        }
    }
    map
}

/// Parses one abbreviation declaration.
///
/// # Arguments
///
/// * `reader` - Abbreviation cursor.
/// * `map` - Output declaration map.
///
/// # Returns
///
/// `Some` when a declaration or separator was read.
fn parse_abbrev_decl(reader: &mut Reader, map: &mut HashMap<u64, Abbrev>) -> Option<()> {
    let code = reader.uleb()?;
    if code == 0 {
        return Some(());
    }
    map.insert(code, read_abbrev(reader)?);
    Some(())
}

/// Reads one abbreviation declaration body.
///
/// # Arguments
///
/// * `reader` - Abbreviation cursor.
///
/// # Returns
///
/// The declaration, or `None`.
fn read_abbrev(reader: &mut Reader) -> Option<Abbrev> {
    let tag = reader.uleb()?;
    let _has_children = reader.u8()?;
    let attrs = read_attr_specs(reader)?;
    Some(Abbrev { tag, attrs })
}

/// Reads an abbreviation attribute specification list.
///
/// # Arguments
///
/// * `reader` - Abbreviation cursor.
///
/// # Returns
///
/// The attribute and form pairs.
fn read_attr_specs(reader: &mut Reader) -> Option<Vec<(u64, u64)>> {
    let mut attrs = Vec::new();
    while push_attr_spec(reader, &mut attrs)? {}
    Some(attrs)
}

/// Reads one attribute specification.
///
/// # Arguments
///
/// * `reader` - Abbreviation cursor.
/// * `attrs` - Accumulated pairs.
///
/// # Returns
///
/// `false` when the specification list ended.
fn push_attr_spec(reader: &mut Reader, attrs: &mut Vec<(u64, u64)>) -> Option<bool> {
    let attr = reader.uleb()?;
    let form = reader.uleb()?;
    if attr == 0 && form == 0 {
        return Some(false);
    }
    attrs.push((attr, form));
    Some(true)
}

/// Walks every compilation unit in the information section.
///
/// # Arguments
///
/// * `info` - Information section.
/// * `abbrev` - Abbreviation declarations.
/// * `strs` - String section.
/// * `dwarf` - Output inventory.
fn parse_units(info: &[u8], abbrev: &HashMap<u64, Abbrev>, strs: &[u8], dwarf: &mut DwarfInfo) {
    let mut pos = 0;
    while pos + 11 <= info.len() {
        let next = parse_unit(info, pos, abbrev, strs, dwarf);
        if next <= pos {
            break;
        }
        pos = next;
    }
}

/// Parses one compilation unit.
///
/// # Arguments
///
/// * `info` - Information section.
/// * `start` - Unit offset.
/// * `abbrev` - Abbreviation declarations.
/// * `strs` - String section.
/// * `dwarf` - Output inventory.
///
/// # Returns
///
/// The next unit offset.
fn parse_unit(
    info: &[u8],
    start: usize,
    abbrev: &HashMap<u64, Abbrev>,
    strs: &[u8],
    dwarf: &mut DwarfInfo,
) -> usize {
    let Some((version, end, addr_size)) = unit_header(info, start) else {
        return info.len();
    };
    dwarf.version = version;
    walk_dies(info, start + 11, end, abbrev, strs, addr_size, dwarf);
    end
}

/// Reads a compilation unit header.
///
/// # Arguments
///
/// * `info` - Information section.
/// * `start` - Unit offset.
///
/// # Returns
///
/// The version, unit end, and address size, or `None`.
fn unit_header(info: &[u8], start: usize) -> Option<(u16, usize, u8)> {
    let len = u64::from(read_u32(info, start)?);
    if len == 0 {
        return None;
    }
    let end = start.saturating_add(4 + usize::try_from(len).ok()?).min(info.len());
    let version = read_u16(info, start + 4)?;
    let addr_size = *info.get(start + 10).unwrap_or(&8);
    Some((version, end, addr_size))
}

/// Walks the debugging information entries of a unit.
///
/// # Arguments
///
/// * `info` - Information section.
/// * `start` - First entry offset.
/// * `end` - Unit end offset.
/// * `abbrev` - Abbreviation declarations.
/// * `strs` - String section.
/// * `addr_size` - Target address size.
/// * `dwarf` - Output inventory.
fn walk_dies(
    info: &[u8],
    start: usize,
    end: usize,
    abbrev: &HashMap<u64, Abbrev>,
    strs: &[u8],
    addr_size: u8,
    dwarf: &mut DwarfInfo,
) {
    let mut reader = Reader::new(info, start);
    reader.set_end(end);
    while reader.pos() < end {
        if parse_die(&mut reader, abbrev, strs, addr_size, dwarf).is_none() {
            break;
        }
    }
}

/// Parses one debugging information entry.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `abbrev` - Abbreviation declarations.
/// * `strs` - String section.
/// * `addr_size` - Target address size.
/// * `dwarf` - Output inventory.
///
/// # Returns
///
/// `Some` when an entry was parsed.
fn parse_die(
    reader: &mut Reader,
    abbrev: &HashMap<u64, Abbrev>,
    strs: &[u8],
    addr_size: u8,
    dwarf: &mut DwarfInfo,
) -> Option<()> {
    let code = reader.uleb()?;
    if code == 0 {
        return Some(());
    }
    let decl = abbrev.get(&code)?;
    let attrs = read_die_attrs(reader, &decl.attrs, addr_size, strs)?;
    record_die(decl.tag, attrs, dwarf);
    Some(())
}

/// Reads the attributes of one entry.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `attrs` - Attribute and form pairs.
/// * `addr_size` - Target address size.
/// * `strs` - String section.
///
/// # Returns
///
/// The recovered attributes, or `None`.
fn read_die_attrs(
    reader: &mut Reader,
    attrs: &[(u64, u64)],
    addr_size: u8,
    strs: &[u8],
) -> Option<DieAttrs> {
    let mut out = DieAttrs::default();
    for &(attr, form) in attrs {
        let value = read_form(reader, form, addr_size, strs)?;
        absorb_attr(attr, &value, &mut out);
    }
    Some(out)
}

/// Applies a decoded attribute to an entry.
///
/// # Arguments
///
/// * `attr` - Attribute code.
/// * `value` - Decoded value.
/// * `out` - Output attributes.
fn absorb_attr(attr: u64, value: &Value, out: &mut DieAttrs) {
    match attr {
        AT_NAME | AT_LINKAGE_NAME => out.name = name_of(value),
        AT_COMP_DIR => out.comp_dir = name_of(value).unwrap_or_default(),
        AT_LOW_PC => out.low_pc = value.uint(),
        _ => {}
    }
}

/// Extracts a string from a value.
///
/// # Arguments
///
/// * `value` - Decoded value.
///
/// # Returns
///
/// The string, or `None`.
fn name_of(value: &Value) -> Option<String> {
    match value {
        Value::Str(text) => Some(text.clone()),
        _ => None,
    }
}

/// Records an entry in the output inventory.
///
/// # Arguments
///
/// * `tag` - Entry tag.
/// * `attrs` - Recovered attributes.
/// * `dwarf` - Output inventory.
fn record_die(tag: u64, attrs: DieAttrs, dwarf: &mut DwarfInfo) {
    match tag {
        TAG_COMPILE_UNIT => push_unit(attrs, dwarf),
        TAG_SUBPROGRAM => push_function(attrs, dwarf),
        _ => {}
    }
}

/// Records a compilation unit.
///
/// # Arguments
///
/// * `attrs` - Recovered attributes.
/// * `dwarf` - Output inventory.
fn push_unit(attrs: DieAttrs, dwarf: &mut DwarfInfo) {
    let name = attrs.name.unwrap_or_default();
    let unit = CompileUnit { name, comp_dir: attrs.comp_dir };
    dwarf.compile_units.push(unit);
}

/// Records a function.
///
/// # Arguments
///
/// * `attrs` - Recovered attributes.
/// * `dwarf` - Output inventory.
fn push_function(attrs: DieAttrs, dwarf: &mut DwarfInfo) {
    if let Some(name) = attrs.name {
        if dwarf.functions.len() < MAX_FUNCTIONS {
            dwarf.functions.push(DwarfFunction { name, low_pc: attrs.low_pc });
        }
    }
}

/// Reads one attribute value.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
/// * `addr_size` - Target address size.
/// * `strs` - String section.
///
/// # Returns
///
/// The decoded value, or `None` for unsupported forms.
fn read_form(reader: &mut Reader, form: u64, addr_size: u8, strs: &[u8]) -> Option<Value> {
    if let Some(value) = read_text_form(reader, form, strs) {
        return Some(value);
    }
    read_number_form(reader, form, addr_size)
}

/// Reads a string form value.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
/// * `strs` - String section.
///
/// # Returns
///
/// The decoded value, or `None` for non-string forms.
fn read_text_form(reader: &mut Reader, form: u64, strs: &[u8]) -> Option<Value> {
    match form {
        0x08 => Some(Value::Str(read_cstr(reader)?)),
        0x0e => Some(Value::Str(strp(reader, strs)?)),
        _ => None,
    }
}

/// Reads a numeric form value.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
/// * `addr_size` - Target address size.
///
/// # Returns
///
/// The decoded value, or `None`.
fn read_number_form(reader: &mut Reader, form: u64, addr_size: u8) -> Option<Value> {
    if let Some(value) = read_fixed_form(reader, form) {
        return Some(value);
    }
    read_variable_form(reader, form, addr_size)
}

/// Reads a fixed-width numeric form.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
///
/// # Returns
///
/// The decoded value, or `None`.
fn read_fixed_form(reader: &mut Reader, form: u64) -> Option<Value> {
    match form {
        0x0b | 0x0c | 0x11 => Some(Value::Uint(u64::from(reader.u8()?))),
        0x05 | 0x12 => Some(Value::Uint(u64::from(reader.u16()?))),
        0x06 | 0x13 | 0x17 => Some(Value::Uint(u64::from(reader.u32()?))),
        0x07 | 0x14 => Some(Value::Uint(reader.u64()?)),
        _ => None,
    }
}

/// Reads a variable-width numeric form.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
/// * `addr_size` - Target address size.
///
/// # Returns
///
/// The decoded value, or `None`.
fn read_variable_form(reader: &mut Reader, form: u64, addr_size: u8) -> Option<Value> {
    match form {
        0x01 => Some(Value::Uint(read_addr(reader, addr_size)?)),
        0x0f | 0x15 => Some(Value::Uint(reader.uleb()?)),
        0x0d | 0x1b => Some(signed_value(reader)?),
        0x19 => Some(Value::Uint(1)),
        _ => skip_form(reader, form),
    }
}

/// Reads a signed integer as an unsigned bit pattern.
///
/// # Arguments
///
/// * `reader` - Information cursor.
///
/// # Returns
///
/// The value, or `None`.
fn signed_value(reader: &mut Reader) -> Option<Value> {
    let raw = reader.sleb()?;
    Some(Value::Uint(u64::from_ne_bytes(raw.to_ne_bytes())))
}

/// Skips a block or expression form.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
///
/// # Returns
///
/// A skipped value, or `None` for unsupported forms.
fn skip_form(reader: &mut Reader, form: u64) -> Option<Value> {
    let len = block_len(reader, form)?;
    reader.take(len)?;
    Some(Value::Skip)
}

/// Reads the length prefix of a block form.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `form` - Attribute form code.
///
/// # Returns
///
/// The block length, or `None`.
fn block_len(reader: &mut Reader, form: u64) -> Option<usize> {
    match form {
        0x03 => Some(usize::from(reader.u16()?)),
        0x04 => usize::try_from(reader.u32()?).ok(),
        0x0a | 0x18 => usize::try_from(reader.uleb()?).ok(),
        _ => None,
    }
}

/// Reads a target-address-sized value.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `addr_size` - Target address size.
///
/// # Returns
///
/// The value, or `None`.
fn read_addr(reader: &mut Reader, addr_size: u8) -> Option<u64> {
    match addr_size {
        4 => Some(u64::from(reader.u32()?)),
        _ => reader.u64(),
    }
}

/// Reads a null-terminated string.
///
/// # Arguments
///
/// * `reader` - Information cursor.
///
/// # Returns
///
/// The string, or `None`.
fn read_cstr(reader: &mut Reader) -> Option<String> {
    let start = reader.pos();
    let slice = reader.data.get(start..)?;
    let end = slice.iter().position(|&b| b == 0)?;
    reader.take(end + 1)?;
    Some(String::from_utf8_lossy(&slice[..end]).to_string())
}

/// Reads a string at an offset in the string section.
///
/// # Arguments
///
/// * `reader` - Information cursor.
/// * `strs` - String section.
///
/// # Returns
///
/// The string, or `None`.
fn strp(reader: &mut Reader, strs: &[u8]) -> Option<String> {
    let offset = usize::try_from(reader.u32()?).ok()?;
    let slice = strs.get(offset..)?;
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    Some(String::from_utf8_lossy(&slice[..end]).to_string())
}

/// Reads a little-endian 16-bit word from a buffer.
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

/// Reads a little-endian 32-bit word from a buffer.
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

/// Formats DWARF inventory as a human-readable text block.
///
/// # Arguments
///
/// * `dwarf` - Recovered inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_dwarf_summary(dwarf: &DwarfInfo) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "DWARF version {}", dwarf.version);
    let _ = writeln!(out, "Compile units: {}", dwarf.compile_units.len());
    let _ = writeln!(out, "Functions: {}", dwarf.functions.len());
    for unit in dwarf.compile_units.iter().take(8) {
        let _ = writeln!(out, "  {} {}", unit.name, unit.comp_dir);
    }
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds an abbreviation section with one compile unit.
    ///
    /// # Returns
    ///
    /// The encoded abbreviations.
    fn abbrev_bytes() -> Vec<u8> {
        vec![0x01, 0x11, 0x01, 0x03, 0x0e, 0x1b, 0x0e, 0x00, 0x00]
    }

    /// Builds the header of a compilation unit.
    ///
    /// # Returns
    ///
    /// The encoded unit header.
    fn unit_header_bytes() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&17u32.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.push(8);
        data
    }

    /// Builds an information section with one compile unit.
    ///
    /// # Returns
    ///
    /// The encoded information section.
    fn info_bytes() -> Vec<u8> {
        let mut data = unit_header_bytes();
        data.push(0x01);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&6u32.to_le_bytes());
        data.push(0);
        data
    }

    /// Builds a string section.
    ///
    /// # Returns
    ///
    /// The encoded strings.
    fn strs_bytes() -> Vec<u8> {
        let mut data = vec![0u8];
        data.extend_from_slice(b"Test\0");
        data.extend_from_slice(b"/tmp\0");
        data
    }

    /// Verifies LEB128 decoding.
    #[test]
    fn test_dwarf_leb() {
        let mut reader = Reader::new(&[0x81, 0x01], 0);
        assert_eq!(reader.uleb(), Some(129));
        let mut signed = Reader::new(&[0x7f], 0);
        assert_eq!(signed.sleb(), Some(-1));
    }

    /// Verifies compilation-unit recovery.
    #[test]
    fn test_dwarf_unit() {
        let abbrev = parse_abbrev(&abbrev_bytes());
        let mut dwarf = DwarfInfo::default();
        parse_units(&info_bytes(), &abbrev, &strs_bytes(), &mut dwarf);
        assert_eq!(dwarf.version, 4);
        assert_eq!(dwarf.compile_units.len(), 1);
        assert_eq!(dwarf.compile_units[0].name, "Test");
        assert_eq!(dwarf.compile_units[0].comp_dir, "/tmp");
    }

    /// Verifies that non-ELF data is rejected.
    #[test]
    fn test_dwarf_rejects_non_elf() {
        assert!(parse_dwarf(b"not an elf file").is_none());
    }
}
