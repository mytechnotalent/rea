//! .NET CIL Disassembly
//!
//! **File:** `cil.rs`
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

// Import dependencies from crate::dotnet
use crate::dotnet::BSJB_MAGIC;

// Import dependencies from crate::model
use crate::model::Binary;

/// Operand encoding of a CIL opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    /// No operand.
    None,

    /// One-byte signed immediate.
    I1,

    /// Four-byte immediate.
    I4,

    /// Eight-byte immediate.
    I8,

    /// Four-byte float.
    R4,

    /// Eight-byte float.
    R8,

    /// Four-byte `#US` string token.
    Str,

    /// Four-byte metadata token.
    Tok,

    /// One-byte variable index.
    Var1,

    /// Two-byte variable index.
    Var2,

    /// One-byte signed branch.
    BrS,

    /// Four-byte signed branch.
    Br,

    /// Switch table.
    Switch,
}

/// The CIL opcode table, keyed by one-byte opcode or `0xFE`-prefixed code.
const OPCODES: &[(u16, &str, Op)] = &[
    (0x00, "nop", Op::None),
    (0x01, "break", Op::None),
    (0x02, "ldarg.0", Op::None),
    (0x03, "ldarg.1", Op::None),
    (0x04, "ldarg.2", Op::None),
    (0x05, "ldarg.3", Op::None),
    (0x06, "ldloc.0", Op::None),
    (0x07, "ldloc.1", Op::None),
    (0x08, "ldloc.2", Op::None),
    (0x09, "ldloc.3", Op::None),
    (0x0a, "stloc.0", Op::None),
    (0x0b, "stloc.1", Op::None),
    (0x0c, "stloc.2", Op::None),
    (0x0d, "stloc.3", Op::None),
    (0x0e, "ldarg.s", Op::Var1),
    (0x0f, "ldarga.s", Op::Var1),
    (0x10, "starg.s", Op::Var1),
    (0x11, "ldloc.s", Op::Var1),
    (0x12, "ldloca.s", Op::Var1),
    (0x13, "stloc.s", Op::Var1),
    (0x14, "ldnull", Op::None),
    (0x15, "ldc.i4.m1", Op::None),
    (0x16, "ldc.i4.0", Op::None),
    (0x17, "ldc.i4.1", Op::None),
    (0x18, "ldc.i4.2", Op::None),
    (0x19, "ldc.i4.3", Op::None),
    (0x1a, "ldc.i4.4", Op::None),
    (0x1b, "ldc.i4.5", Op::None),
    (0x1c, "ldc.i4.6", Op::None),
    (0x1d, "ldc.i4.7", Op::None),
    (0x1e, "ldc.i4.8", Op::None),
    (0x1f, "ldc.i4.s", Op::I1),
    (0x20, "ldc.i4", Op::I4),
    (0x21, "ldc.i8", Op::I8),
    (0x22, "ldc.r4", Op::R4),
    (0x23, "ldc.r8", Op::R8),
    (0x25, "dup", Op::None),
    (0x26, "pop", Op::None),
    (0x27, "jmp", Op::Tok),
    (0x28, "call", Op::Tok),
    (0x29, "calli", Op::Tok),
    (0x2a, "ret", Op::None),
    (0x2b, "br.s", Op::BrS),
    (0x2c, "brfalse.s", Op::BrS),
    (0x2d, "brtrue.s", Op::BrS),
    (0x2e, "beq.s", Op::BrS),
    (0x2f, "bge.s", Op::BrS),
    (0x30, "bgt.s", Op::BrS),
    (0x31, "ble.s", Op::BrS),
    (0x32, "blt.s", Op::BrS),
    (0x33, "bne.un.s", Op::BrS),
    (0x34, "bge.un.s", Op::BrS),
    (0x35, "bgt.un.s", Op::BrS),
    (0x36, "ble.un.s", Op::BrS),
    (0x37, "blt.un.s", Op::BrS),
    (0x38, "br", Op::Br),
    (0x39, "brfalse", Op::Br),
    (0x3a, "brtrue", Op::Br),
    (0x3b, "beq", Op::Br),
    (0x3c, "bge", Op::Br),
    (0x3d, "bgt", Op::Br),
    (0x3e, "ble", Op::Br),
    (0x3f, "blt", Op::Br),
    (0x40, "bne.un", Op::Br),
    (0x41, "bge.un", Op::Br),
    (0x42, "bgt.un", Op::Br),
    (0x43, "ble.un", Op::Br),
    (0x44, "blt.un", Op::Br),
    (0x45, "switch", Op::Switch),
    (0x46, "ldind.i1", Op::None),
    (0x47, "ldind.u1", Op::None),
    (0x48, "ldind.i2", Op::None),
    (0x49, "ldind.u2", Op::None),
    (0x4a, "ldind.i4", Op::None),
    (0x4b, "ldind.u4", Op::None),
    (0x4c, "ldind.i8", Op::None),
    (0x4d, "ldind.i", Op::None),
    (0x4e, "ldind.r4", Op::None),
    (0x4f, "ldind.r8", Op::None),
    (0x50, "ldind.ref", Op::None),
    (0x51, "stind.ref", Op::None),
    (0x52, "stind.i1", Op::None),
    (0x53, "stind.i2", Op::None),
    (0x54, "stind.i4", Op::None),
    (0x55, "stind.i8", Op::None),
    (0x56, "stind.r4", Op::None),
    (0x57, "stind.r8", Op::None),
    (0x58, "add", Op::None),
    (0x59, "sub", Op::None),
    (0x5a, "mul", Op::None),
    (0x5b, "div", Op::None),
    (0x5c, "div.un", Op::None),
    (0x5d, "rem", Op::None),
    (0x5e, "rem.un", Op::None),
    (0x5f, "and", Op::None),
    (0x60, "or", Op::None),
    (0x61, "xor", Op::None),
    (0x62, "shl", Op::None),
    (0x63, "shr", Op::None),
    (0x64, "shr.un", Op::None),
    (0x65, "neg", Op::None),
    (0x66, "not", Op::None),
    (0x67, "conv.i1", Op::None),
    (0x68, "conv.i2", Op::None),
    (0x69, "conv.i4", Op::None),
    (0x6a, "conv.i8", Op::None),
    (0x6b, "conv.r4", Op::None),
    (0x6c, "conv.r8", Op::None),
    (0x6d, "conv.u4", Op::None),
    (0x6e, "conv.u8", Op::None),
    (0x6f, "callvirt", Op::Tok),
    (0x70, "cpobj", Op::Tok),
    (0x71, "ldobj", Op::Tok),
    (0x72, "ldstr", Op::Str),
    (0x73, "newobj", Op::Tok),
    (0x74, "castclass", Op::Tok),
    (0x75, "isinst", Op::Tok),
    (0x76, "conv.r.un", Op::None),
    (0x79, "unbox", Op::Tok),
    (0x7a, "throw", Op::None),
    (0x7b, "ldfld", Op::Tok),
    (0x7c, "ldflda", Op::Tok),
    (0x7d, "stfld", Op::Tok),
    (0x7e, "ldsfld", Op::Tok),
    (0x7f, "ldsflda", Op::Tok),
    (0x80, "stsfld", Op::Tok),
    (0x81, "stobj", Op::Tok),
    (0x8c, "box", Op::Tok),
    (0x8d, "newarr", Op::Tok),
    (0x8e, "ldlen", Op::None),
    (0x8f, "ldelema", Op::Tok),
    (0x90, "ldelem.i1", Op::None),
    (0x91, "ldelem.u1", Op::None),
    (0x92, "ldelem.i2", Op::None),
    (0x93, "ldelem.u2", Op::None),
    (0x94, "ldelem.i4", Op::None),
    (0x95, "ldelem.u4", Op::None),
    (0x96, "ldelem.i8", Op::None),
    (0x97, "ldelem.i", Op::None),
    (0x98, "ldelem.r4", Op::None),
    (0x99, "ldelem.r8", Op::None),
    (0x9a, "ldelem.ref", Op::None),
    (0x9b, "stelem.i", Op::None),
    (0x9c, "stelem.i1", Op::None),
    (0x9d, "stelem.i2", Op::None),
    (0x9e, "stelem.i4", Op::None),
    (0x9f, "stelem.i8", Op::None),
    (0xa0, "stelem.r4", Op::None),
    (0xa1, "stelem.r8", Op::None),
    (0xa2, "stelem.ref", Op::None),
    (0xa3, "ldelem", Op::Tok),
    (0xa4, "stelem", Op::Tok),
    (0xa5, "unbox.any", Op::Tok),
    (0xc2, "refanyval", Op::Tok),
    (0xc3, "ckfinite", Op::None),
    (0xc6, "mkrefany", Op::Tok),
    (0xd0, "ldtoken", Op::Tok),
    (0xd1, "conv.u2", Op::None),
    (0xd2, "conv.u1", Op::None),
    (0xd3, "conv.i", Op::None),
    (0xdc, "endfinally", Op::None),
    (0xdd, "leave", Op::Br),
    (0xde, "leave.s", Op::BrS),
    (0xdf, "stind.i", Op::None),
    (0xe0, "conv.u", Op::None),
    (0xfe00, "arglist", Op::None),
    (0xfe01, "ceq", Op::None),
    (0xfe02, "cgt", Op::None),
    (0xfe03, "cgt.un", Op::None),
    (0xfe04, "clt", Op::None),
    (0xfe05, "clt.un", Op::None),
    (0xfe06, "ldftn", Op::Tok),
    (0xfe07, "ldvirtftn", Op::Tok),
    (0xfe09, "ldarg", Op::Var2),
    (0xfe0a, "ldarga", Op::Var2),
    (0xfe0b, "starg", Op::Var2),
    (0xfe0c, "ldloc", Op::Var2),
    (0xfe0d, "ldloca", Op::Var2),
    (0xfe0e, "stloc", Op::Var2),
    (0xfe0f, "localloc", Op::None),
    (0xfe11, "endfilter", Op::None),
    (0xfe12, "unaligned.", Op::I1),
    (0xfe13, "volatile.", Op::None),
    (0xfe14, "tail.", Op::None),
    (0xfe15, "initobj", Op::Tok),
    (0xfe16, "constrained.", Op::Tok),
    (0xfe17, "cpblk", Op::None),
    (0xfe18, "initblk", Op::None),
    (0xfe1a, "rethrow", Op::None),
    (0xfe1c, "sizeof", Op::Tok),
    (0xfe1d, "refanytype", Op::None),
    (0xfe1e, "readonly.", Op::None),
];

/// Offsets of the metadata streams used for disassembly.
#[derive(Default)]
struct Streams {
    /// Offset of the `#~` tables stream.
    tables: usize,

    /// Size of the `#~` tables stream.
    tables_size: usize,

    /// Offset of the `#Strings` heap.
    strings: usize,

    /// Offset of the `#US` user-string heap.
    user_strings: usize,
}

/// One disassembled method.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CilMethod {
    /// Method name.
    pub name: String,

    /// Method body RVA.
    pub rva: u64,

    /// Disassembled IL lines.
    pub il: Vec<String>,
}

/// Disassembles every method body in a managed assembly.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The disassembled methods.
#[must_use]
pub fn disassemble(binary: &Binary, bytes: &[u8]) -> Vec<CilMethod> {
    let Some((streams, schema)) = metadata(bytes) else {
        return Vec::new();
    };
    let names = build_names(bytes, &streams, &schema);
    read_methods(bytes, &streams, &schema)
        .iter()
        .map(|row| disassemble_method(binary, bytes, &streams, &names, row))
        .collect()
}

/// Locates the metadata streams and tables schema.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The streams and schema, or `None`.
fn metadata(bytes: &[u8]) -> Option<(Streams, Schema)> {
    let found = streams(bytes)?;
    let schema = table_schema(bytes, &found)?;
    Some((found, schema))
}

/// One `MethodDef` row: name and body RVA.
struct MethodRow {
    /// Method name.
    name: String,

    /// Body RVA.
    rva: u64,
}

/// Locates the metadata streams needed for disassembly.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The stream offsets, or `None`.
fn streams(bytes: &[u8]) -> Option<Streams> {
    magic_offsets(bytes).into_iter().find_map(|root| parse_root(bytes, root))
}

/// Finds every `BSJB` metadata-root candidate offset.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The candidate offsets.
fn magic_offsets(bytes: &[u8]) -> Vec<usize> {
    let hits = bytes.windows(4).enumerate();
    hits.filter(|(_, window)| *window == BSJB_MAGIC)
        .map(|(index, _)| index)
        .collect()
}

/// Parses a metadata root candidate into streams.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `root` - Candidate root offset.
///
/// # Returns
///
/// The streams, or `None` when the candidate is not a metadata root.
fn parse_root(bytes: &[u8], root: usize) -> Option<Streams> {
    let version_len = read_u32(bytes, root + 12)? as usize;
    if version_len > 256 {
        return None;
    }
    let (start, count) = stream_directory(bytes, root, version_len)?;
    let mut streams = read_streams(bytes, start, count)?;
    absolutize(&mut streams, root);
    (streams.tables != 0).then_some(streams)
}

/// Reads the stream directory start and count.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `root` - Metadata root offset.
/// * `version_len` - Version string length.
///
/// # Returns
///
/// The header start and stream count, or `None`.
fn stream_directory(bytes: &[u8], root: usize, version_len: usize) -> Option<(usize, usize)> {
    let header = (root + 16 + version_len + 3) & !3;
    let count = usize::from(u16::from_le_bytes([
        *bytes.get(header + 2)?,
        *bytes.get(header + 3)?,
    ]));
    Some((header + 4, count))
}

/// Reads a sequence of stream headers.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `start` - First header offset.
/// * `count` - Number of headers.
///
/// # Returns
///
/// The collected streams, or `None`.
fn read_streams(bytes: &[u8], start: usize, count: usize) -> Option<Streams> {
    let mut cur = start;
    let mut streams = Streams::default();
    for _ in 0..count.min(16) {
        cur = read_stream_header(bytes, cur, &mut streams)?;
    }
    Some(streams)
}

/// Converts root-relative stream offsets to absolute file offsets.
///
/// # Arguments
///
/// * `streams` - Stream offsets being adjusted.
/// * `root` - Metadata root offset.
fn absolutize(streams: &mut Streams, root: usize) {
    add_base(&mut streams.tables, root);
    add_base(&mut streams.strings, root);
    add_base(&mut streams.user_strings, root);
}

/// Adds the metadata root to a non-zero stream offset.
///
/// # Arguments
///
/// * `value` - Stream offset.
/// * `root` - Metadata root offset.
fn add_base(value: &mut usize, root: usize) {
    if *value != 0 {
        *value += root;
    }
}

/// Reads one stream header and records its offset.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `cur` - Current header offset.
/// * `streams` - Stream offsets being collected.
///
/// # Returns
///
/// The next header offset, or `None`.
fn read_stream_header(bytes: &[u8], cur: usize, streams: &mut Streams) -> Option<usize> {
    let offset = read_u32(bytes, cur)? as usize;
    let size = read_u32(bytes, cur + 4)? as usize;
    let name = read_cstring(bytes, cur + 8);
    record_stream(streams, &name, offset, size);
    Some(next_header(cur, &name))
}

/// Computes the offset of the stream header following a name.
///
/// # Arguments
///
/// * `cur` - Current header offset.
/// * `name` - Stream name.
///
/// # Returns
///
/// The next header offset.
fn next_header(cur: usize, name: &str) -> usize {
    cur + 8 + name.len() + 1 + pad_to_four(name.len() + 1)
}

/// Records a known stream's offset and size.
///
/// # Arguments
///
/// * `streams` - Stream offsets being collected.
/// * `name` - Stream name.
/// * `offset` - Stream offset.
/// * `size` - Stream size.
fn record_stream(streams: &mut Streams, name: &str, offset: usize, size: usize) {
    match name {
        "#~" => set_tables(streams, offset, size),
        "#Strings" => streams.strings = offset,
        "#US" => streams.user_strings = offset,
        _ => {}
    }
}

/// Records the tables stream offset and size.
///
/// # Arguments
///
/// * `streams` - Stream offsets being collected.
/// * `offset` - Stream offset.
/// * `size` - Stream size.
fn set_tables(streams: &mut Streams, offset: usize, size: usize) {
    streams.tables = offset;
    streams.tables_size = size;
}

/// Rounds a length up to the next multiple of four.
///
/// # Arguments
///
/// * `len` - Length in bytes.
///
/// # Returns
///
/// The padded length minus the input.
fn pad_to_four(len: usize) -> usize {
    (4 - (len % 4)) % 4
}

/// Layout information for the tables stream.
struct Schema {
    /// Offset where table data begins.
    data: usize,

    /// Row counts indexed by table id.
    rows: [u32; 64],

    /// Byte size of a string heap index.
    string_size: usize,

    /// Byte size of a blob heap index.
    blob_size: usize,

    /// Byte size of a GUID heap index.
    guid_size: usize,

    /// Byte offset of each table's rows.
    offsets: [usize; 64],
}

/// Parses the tables stream header into a schema.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
///
/// # Returns
///
/// The schema, or `None`.
fn table_schema(bytes: &[u8], streams: &Streams) -> Option<Schema> {
    let base = streams.tables;
    let heap = *bytes.get(base + 6)?;
    let valid = read_u64(bytes, base + 8)?;
    let mut rows = [0u32; 64];
    let mut cursor = base + 24;
    fill_rows(bytes, valid, &mut rows, &mut cursor)?;
    Some(build_schema(heap, cursor, rows))
}

/// Builds a schema from heap sizes and row counts.
///
/// # Arguments
///
/// * `heap` - Heap-size flags byte.
/// * `data` - Offset where table data begins.
/// * `rows` - Row counts per table.
///
/// # Returns
///
/// The schema.
fn build_schema(heap: u8, data: usize, rows: [u32; 64]) -> Schema {
    let mut schema = raw_schema(heap, data, rows);
    schema.offsets = table_offsets(&schema);
    schema
}

/// Builds a schema with heap sizes but no table offsets.
///
/// # Arguments
///
/// * `heap` - Heap-size flags byte.
/// * `data` - Offset where table data begins.
/// * `rows` - Row counts per table.
///
/// # Returns
///
/// The schema.
fn raw_schema(heap: u8, data: usize, rows: [u32; 64]) -> Schema {
    Schema {
        data,
        rows,
        string_size: heap_size(heap, 0),
        blob_size: heap_size(heap, 2),
        guid_size: heap_size(heap, 1),
        offsets: [0; 64],
    }
}

/// Computes the byte offset of each table's rows.
///
/// # Arguments
///
/// * `schema` - Tables schema with row counts and heap sizes.
///
/// # Returns
///
/// The table offsets.
fn table_offsets(schema: &Schema) -> [usize; 64] {
    let mut offsets = [0usize; 64];
    let mut cursor = schema.data;
    for (table, slot) in offsets.iter_mut().enumerate().take(0x0B) {
        *slot = cursor;
        cursor += advance(table, schema);
    }
    offsets
}

/// Returns the byte span of a table's rows.
///
/// # Arguments
///
/// * `table` - Table id.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row-count times row size.
fn advance(table: usize, schema: &Schema) -> usize {
    schema.rows[table] as usize * row_size(table, schema).unwrap_or(0)
}

/// Reads the row count of every present table.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `valid` - Bitmask of present tables.
/// * `rows` - Row counts being filled.
/// * `cursor` - Current read offset.
///
/// # Returns
///
/// Unit when successful, or `None`.
fn fill_rows(bytes: &[u8], valid: u64, rows: &mut [u32; 64], cursor: &mut usize) -> Option<()> {
    for (table, row) in rows.iter_mut().enumerate() {
        if valid & (1u64 << table) != 0 {
            *row = read_u32(bytes, *cursor)?;
            *cursor += 4;
        }
    }
    Some(())
}

/// Returns the index size for a heap flag bit.
///
/// # Arguments
///
/// * `heap` - Heap-size flags byte.
/// * `bit` - Flag bit index.
///
/// # Returns
///
/// Two or four bytes.
fn heap_size(heap: u8, bit: u32) -> usize {
    if heap & (1 << bit) != 0 {
        4
    } else {
        2
    }
}

/// Reads every `MethodDef` row.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The method rows.
fn read_methods(bytes: &[u8], streams: &Streams, schema: &Schema) -> Vec<MethodRow> {
    let offset = schema.offsets[0x06];
    let count = schema.rows[0x06] as usize;
    let size = method_row_size(schema);
    (0..count)
        .filter_map(|i| read_method_row(bytes, streams, offset + i * size, schema))
        .collect()
}

/// Returns the byte size of a table row.
///
/// # Arguments
///
/// * `table` - Table id.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row size, or `None` for unknown tables.
fn row_size(table: usize, schema: &Schema) -> Option<usize> {
    match table {
        0x00 => Some(2 + schema.string_size + 3 * schema.guid_size),
        0x01 => Some(coded_size(schema, 0) + 2 * schema.string_size),
        0x02 => Some(typedef_row(schema)),
        0x04 => Some(2 + schema.string_size + schema.blob_size),
        0x06 => Some(method_row_size(schema)),
        other => other_row(other, schema),
    }
}

/// Returns the byte size of a `TypeDef` row.
///
/// # Arguments
///
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row size.
fn typedef_row(schema: &Schema) -> usize {
    let s = schema.string_size;
    4 + 2 * s + coded_size(schema, 1) + index_size(schema, 0x04) + index_size(schema, 0x06)
}

/// Returns the byte size of a secondary table row.
///
/// # Arguments
///
/// * `table` - Table id.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row size, or `None`.
fn other_row(table: usize, schema: &Schema) -> Option<usize> {
    match table {
        0x03 => Some(index_size(schema, 0x04)),
        0x05 => Some(index_size(schema, 0x06)),
        0x07 => Some(index_size(schema, 0x08)),
        other => member_row(other, schema),
    }
}

/// Returns the byte size of a member or parameter table row.
///
/// # Arguments
///
/// * `table` - Table id.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row size, or `None`.
fn member_row(table: usize, schema: &Schema) -> Option<usize> {
    match table {
        0x08 => Some(4 + schema.string_size),
        0x09 => Some(index_size(schema, 0x02) + coded_size(schema, 1)),
        0x0A => Some(coded_size(schema, 2) + schema.string_size + schema.blob_size),
        _ => None,
    }
}

/// Returns the size of a simple table index.
///
/// # Arguments
///
/// * `schema` - Tables schema.
/// * `table` - Referenced table id.
///
/// # Returns
///
/// Two or four bytes.
fn index_size(schema: &Schema, table: usize) -> usize {
    if schema.rows[table] >= 0x1_0000 {
        4
    } else {
        2
    }
}

/// Returns the size of a coded index.
///
/// # Arguments
///
/// * `schema` - Tables schema.
/// * `kind` - Coded index kind.
///
/// # Returns
///
/// Two or four bytes.
fn coded_size(schema: &Schema, kind: usize) -> usize {
    let max = referenced_tables(kind).iter().map(|t| schema.rows[*t]).max().unwrap_or(0);
    let bits = if kind == 2 { 13 } else { 14 };
    if max >= (1 << bits) {
        4
    } else {
        2
    }
}

/// Returns the tables referenced by a coded index kind.
///
/// # Arguments
///
/// * `kind` - Coded index kind.
///
/// # Returns
///
/// The referenced table ids.
fn referenced_tables(kind: usize) -> &'static [usize] {
    match kind {
        0 => &[0x00, 0x1a, 0x23, 0x01],
        1 => &[0x02, 0x01, 0x1b],
        2 => &[0x02, 0x01, 0x1a, 0x06, 0x1b],
        _ => &[],
    }
}

/// Returns the byte size of a `MethodDef` row.
///
/// # Arguments
///
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The row size.
fn method_row_size(schema: &Schema) -> usize {
    8 + schema.string_size + schema.blob_size + index_size(schema, 0x08)
}

/// Reads one `MethodDef` row.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `offset` - Row offset.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The method row, or `None`.
fn read_method_row(
    bytes: &[u8],
    streams: &Streams,
    offset: usize,
    schema: &Schema,
) -> Option<MethodRow> {
    let rva = u64::from(read_u32(bytes, offset)?);
    let name_at = offset + 8;
    let name_index = read_index(bytes, name_at, schema.string_size)?;
    let name = read_cstring(bytes, streams.strings + name_index);
    Some(MethodRow { name, rva })
}

/// A per-row name reader.
type RowReader = fn(&[u8], &Streams, &Schema, usize) -> String;

/// Resolved metadata names for token rendering.
struct Names {
    /// `TypeDef` names, indexed by row.
    type_defs: Vec<String>,

    /// `TypeRef` names, indexed by row.
    type_refs: Vec<String>,

    /// `MethodDef` names, indexed by row.
    methods: Vec<String>,

    /// `MemberRef` names, indexed by row.
    members: Vec<String>,

    /// `Field` names, indexed by row.
    fields: Vec<String>,
}

/// Resolves metadata table rows to readable names.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
///
/// # Returns
///
/// The resolved names.
fn build_names(bytes: &[u8], streams: &Streams, schema: &Schema) -> Names {
    let external = table_names(bytes, streams, schema, 0x01, type_ref_name);
    let local = table_names(bytes, streams, schema, 0x02, type_def_name);
    let methods = table_names(bytes, streams, schema, 0x06, method_name);
    let fields = table_names(bytes, streams, schema, 0x04, field_name);
    let members = member_names(bytes, streams, schema, (&external, &local, &methods));
    Names { type_defs: local, type_refs: external, methods, members, fields }
}

/// Reads every row of a table into names.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `table` - Table id.
/// * `read` - Per-row name reader.
///
/// # Returns
///
/// The row names.
fn table_names(
    bytes: &[u8],
    streams: &Streams,
    schema: &Schema,
    table: usize,
    read: RowReader,
) -> Vec<String> {
    let size = row_size(table, schema).unwrap_or(0);
    if size == 0 {
        return Vec::new();
    }
    let base = schema.offsets[table];
    (0..schema.rows[table] as usize)
        .map(|i| read(bytes, streams, schema, base + i * size))
        .collect()
}

/// Reads a `TypeRef` row into a qualified name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `off` - Row offset.
///
/// # Returns
///
/// The qualified name.
fn type_ref_name(bytes: &[u8], streams: &Streams, schema: &Schema, off: usize) -> String {
    let stride = coded_size(schema, 0);
    let name = field_string(bytes, streams, schema, off + stride);
    let space = field_string(bytes, streams, schema, off + stride + schema.string_size);
    qualify(&space, &name)
}

/// Reads a string heap index from a row field.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `off` - Field offset.
///
/// # Returns
///
/// The referenced string.
fn field_string(bytes: &[u8], streams: &Streams, schema: &Schema, off: usize) -> String {
    string_at(bytes, streams, read_index(bytes, off, schema.string_size))
}

/// Reads a `TypeDef` row into a qualified name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `off` - Row offset.
///
/// # Returns
///
/// The qualified name.
fn type_def_name(bytes: &[u8], streams: &Streams, schema: &Schema, off: usize) -> String {
    let name = field_string(bytes, streams, schema, off + 4);
    let space = field_string(bytes, streams, schema, off + 4 + schema.string_size);
    qualify(&space, &name)
}

/// Reads a `MethodDef` row into a name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `off` - Row offset.
///
/// # Returns
///
/// The method name.
fn method_name(bytes: &[u8], streams: &Streams, schema: &Schema, off: usize) -> String {
    string_at(
        bytes,
        streams,
        read_index(bytes, off + 8, schema.string_size),
    )
}

/// Reads a `Field` row into a name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `off` - Row offset.
///
/// # Returns
///
/// The field name.
fn field_name(bytes: &[u8], streams: &Streams, schema: &Schema, off: usize) -> String {
    string_at(
        bytes,
        streams,
        read_index(bytes, off + 2, schema.string_size),
    )
}

/// Reads every `MemberRef` row into an owner-qualified name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `type_refs` - Resolved type reference names.
/// * `type_defs` - Resolved type definition names.
/// * `methods` - Resolved method names.
///
/// # Returns
///
/// The member names.
fn member_names(
    bytes: &[u8],
    streams: &Streams,
    schema: &Schema,
    types: (&[String], &[String], &[String]),
) -> Vec<String> {
    let size = row_size(0x0A, schema).unwrap_or(0);
    if size == 0 {
        return Vec::new();
    }
    let base = schema.offsets[0x0A];
    (0..schema.rows[0x0A] as usize)
        .map(|i| member_at(bytes, streams, schema, types, base + i * size))
        .collect()
}

/// Reads one `MemberRef` row into an owner-qualified name.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `schema` - Tables schema.
/// * `types` - Resolved type reference, type definition, and method names.
/// * `off` - Row offset.
///
/// # Returns
///
/// The qualified member name.
fn member_at(
    bytes: &[u8],
    streams: &Streams,
    schema: &Schema,
    types: (&[String], &[String], &[String]),
    off: usize,
) -> String {
    let stride = coded_size(schema, 2);
    let owner = read_index(bytes, off, stride).unwrap_or(0);
    let name = string_at(
        bytes,
        streams,
        read_index(bytes, off + stride, schema.string_size),
    );
    format!("{}::{name}", owner_name(owner, types))
}

/// Resolves a `MemberRefParent` coded index to an owner name.
///
/// # Arguments
///
/// * `coded` - Coded index value.
/// * `types` - Resolved type reference, type definition, and method names.
///
/// # Returns
///
/// The owner name, or an empty string.
fn owner_name(coded: usize, types: (&[String], &[String], &[String])) -> String {
    let rid = (coded >> 3).wrapping_sub(1);
    owner_list(coded & 7, types).get(rid).cloned().unwrap_or_default()
}

/// Selects the owner table for a `MemberRefParent` tag.
///
/// # Arguments
///
/// * `tag` - Coded index tag.
/// * `types` - Resolved type reference, type definition, and method names.
///
/// # Returns
///
/// The owner name table.
fn owner_list<'a>(tag: usize, types: (&'a [String], &'a [String], &'a [String])) -> &'a [String] {
    let (external, local, methods) = types;
    match tag {
        0 => local,
        1 => external,
        3 => methods,
        _ => &[],
    }
}

/// Resolves a string heap index to text.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `index` - String heap index, when present.
///
/// # Returns
///
/// The string text.
fn string_at(bytes: &[u8], streams: &Streams, index: Option<usize>) -> String {
    index.map_or_else(String::new, |i| read_cstring(bytes, streams.strings + i))
}

/// Qualifies a name with its namespace.
///
/// # Arguments
///
/// * `space` - Namespace text.
/// * `name` - Type name.
///
/// # Returns
///
/// The qualified name.
fn qualify(space: &str, name: &str) -> String {
    if space.is_empty() {
        name.to_string()
    } else {
        format!("{space}.{name}")
    }
}

/// Resolves a metadata token to a readable name.
///
/// # Arguments
///
/// * `token` - Metadata token.
/// * `names` - Resolved metadata names.
///
/// # Returns
///
/// The resolved name or a token fallback.
fn resolve(token: u32, names: &Names) -> String {
    let rid = (token & 0x00ff_ffff) as usize;
    match token_list((token >> 24) & 0xff, names) {
        Some(list) => list.get(rid.wrapping_sub(1)).cloned().unwrap_or_else(|| fallback(token)),
        None => fallback(token),
    }
}

/// Selects the name table for a metadata token table id.
///
/// # Arguments
///
/// * `table` - Metadata table id.
/// * `names` - Resolved metadata names.
///
/// # Returns
///
/// The name table, or `None`.
fn token_list(table: u32, names: &Names) -> Option<&[String]> {
    match table {
        0x01 => Some(&names.type_refs),
        0x02 => Some(&names.type_defs),
        0x04 => Some(&names.fields),
        0x06 => Some(&names.methods),
        0x0A => Some(&names.members),
        _ => None,
    }
}

/// Formats a metadata token fallback.
///
/// # Arguments
///
/// * `token` - Metadata token.
///
/// # Returns
///
/// The token text.
fn fallback(token: u32) -> String {
    format!("token 0x{token:08x}")
}

/// Disassembles one method body.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
/// * `row` - Method row.
///
/// # Returns
///
/// The disassembled method.
fn disassemble_method(
    binary: &Binary,
    bytes: &[u8],
    streams: &Streams,
    names: &Names,
    row: &MethodRow,
) -> CilMethod {
    let il = method_body(binary, bytes, row.rva)
        .map_or_else(Vec::new, |body| decode(body, bytes, streams, names));
    CilMethod { name: row.name.clone(), rva: row.rva, il }
}

/// Extracts the IL bytes of a method body at an RVA.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `rva` - Method body RVA.
///
/// # Returns
///
/// The IL bytes, or `None`.
fn method_body<'a>(binary: &Binary, bytes: &'a [u8], rva: u64) -> Option<&'a [u8]> {
    let address = binary.image_base + rva;
    let section = binary.sections.iter().find(|s| s.contains(address))?;
    let offset = usize::try_from(section.file_offset_of(address)?).ok()?;
    body_slice(bytes, offset)
}

/// Slices the IL bytes from a method body header.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Body header offset.
///
/// # Returns
///
/// The IL bytes, or `None`.
fn body_slice(bytes: &[u8], offset: usize) -> Option<&[u8]> {
    let header = *bytes.get(offset)?;
    match header & 3 {
        2 => code_slice(bytes, offset + 1, usize::from(header >> 2)),
        3 => fat_slice(bytes, offset),
        _ => None,
    }
}

/// Slices the code of a fat method body.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Body header offset.
///
/// # Returns
///
/// The IL bytes, or `None`.
fn fat_slice(bytes: &[u8], offset: usize) -> Option<&[u8]> {
    let flags = u16::from_le_bytes([*bytes.get(offset)?, *bytes.get(offset + 1)?]);
    let header_size = usize::from(flags >> 12) * 4;
    let code_size = read_u32(bytes, offset + 4)? as usize;
    code_slice(bytes, offset + header_size, code_size)
}

/// Slices a byte range of IL code.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `start` - Code start offset.
/// * `size` - Code size.
///
/// # Returns
///
/// The IL bytes, or `None`.
fn code_slice(bytes: &[u8], start: usize, size: usize) -> Option<&[u8]> {
    bytes.get(start..start.checked_add(size)?)
}

/// Disassembles a stream of IL bytes.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
///
/// # Returns
///
/// The disassembled lines.
fn decode(il: &[u8], bytes: &[u8], streams: &Streams, names: &Names) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some((line, next)) = decode_one(il, cursor, bytes, streams, names) {
        out.push(line);
        cursor = next;
    }
    out
}

/// Disassembles one instruction.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `cursor` - Current offset.
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
///
/// # Returns
///
/// The rendered line and next offset, or `None`.
fn decode_one(
    il: &[u8],
    cursor: usize,
    bytes: &[u8],
    streams: &Streams,
    names: &Names,
) -> Option<(String, usize)> {
    let (code, after) = opcode(il, cursor)?;
    let (name, operand) = lookup(code)?;
    let (text, next) = render(il, after, operand, bytes, streams, names)?;
    Some((
        format!("{cursor:04x}: {name} {text}").trim_end().to_string(),
        next,
    ))
}

/// Reads an opcode, consuming a `0xFE` prefix when present.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `cursor` - Current offset.
///
/// # Returns
///
/// The opcode value and the offset after it.
fn opcode(il: &[u8], cursor: usize) -> Option<(u16, usize)> {
    let first = u16::from(*il.get(cursor)?);
    if first == 0xFE {
        Some((0xFE00 | u16::from(*il.get(cursor + 1)?), cursor + 2))
    } else {
        Some((first, cursor + 1))
    }
}

/// Looks up an opcode's mnemonic and operand kind.
///
/// # Arguments
///
/// * `code` - Opcode value.
///
/// # Returns
///
/// The mnemonic and operand kind, or `None`.
fn lookup(code: u16) -> Option<(&'static str, Op)> {
    OPCODES
        .iter()
        .find(|(value, _, _)| *value == code)
        .map(|(_, name, operand)| (*name, *operand))
}

/// Renders an instruction's operand text.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
/// * `operand` - Operand kind.
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
///
/// # Returns
///
/// The operand text and next offset, or `None`.
fn render(
    il: &[u8],
    after: usize,
    operand: Op,
    bytes: &[u8],
    streams: &Streams,
    names: &Names,
) -> Option<(String, usize)> {
    match operand {
        Op::Str => render_string(il, after, bytes, streams),
        Op::Tok => render_token(il, after, names),
        Op::Switch => render_switch(il, after),
        Op::BrS => render_small_branch(il, after),
        Op::Br => render_big_branch(il, after),
        _ => render_data(il, after, operand),
    }
}

/// Renders a metadata token operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
/// * `names` - Resolved metadata names.
///
/// # Returns
///
/// The token text and next offset, or `None`.
fn render_token(il: &[u8], after: usize, names: &Names) -> Option<(String, usize)> {
    let token = read_u32(il, after)?;
    Some((resolve(token, names), after + 4))
}

/// Renders a one-byte branch operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The target text and next offset, or `None`.
fn render_small_branch(il: &[u8], after: usize) -> Option<(String, usize)> {
    let delta = i64::from(i8::from_le_bytes([*il.get(after)?]));
    Some(render_branch(after, delta, 1))
}

/// Renders a four-byte branch operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The target text and next offset, or `None`.
fn render_big_branch(il: &[u8], after: usize) -> Option<(String, usize)> {
    Some(render_branch(after, i64::from(read_i32(il, after)?), 4))
}

/// Renders a one-byte signed immediate.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The text and next offset, or `None`.
fn render_i1(il: &[u8], after: usize) -> Option<(String, usize)> {
    Some((
        format!("{}", i8::from_le_bytes([*il.get(after)?])),
        after + 1,
    ))
}

/// Renders a four-byte immediate.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The text and next offset, or `None`.
fn render_i4(il: &[u8], after: usize) -> Option<(String, usize)> {
    Some((format!("{}", read_i32(il, after)?), after + 4))
}

/// Renders an eight-byte immediate.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The text and next offset, or `None`.
fn render_i8(il: &[u8], after: usize) -> Option<(String, usize)> {
    Some((format!("{}", read_i64(il, after)?), after + 8))
}

/// Renders an immediate or variable operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
/// * `operand` - Operand kind.
///
/// # Returns
///
/// The operand text and next offset, or `None`.
fn render_data(il: &[u8], after: usize, operand: Op) -> Option<(String, usize)> {
    match operand {
        Op::None => Some((String::new(), after)),
        Op::I1 => render_i1(il, after),
        Op::I4 => render_i4(il, after),
        Op::I8 => render_i8(il, after),
        _ => render_bytes(il, after, operand),
    }
}

/// Renders float or variable-index operands.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
/// * `operand` - Operand kind.
///
/// # Returns
///
/// The operand text and next offset, or `None`.
fn render_bytes(il: &[u8], after: usize, operand: Op) -> Option<(String, usize)> {
    match operand {
        Op::R4 => Some((format!("0x{:08x}", read_u32(il, after)?), after + 4)),
        Op::R8 => Some((format!("0x{:016x}", read_u64(il, after)?), after + 8)),
        Op::Var1 => Some((format!("{}", il.get(after)?), after + 1)),
        Op::Var2 => Some((format!("{}", var2(il, after)?), after + 2)),
        _ => None,
    }
}

/// Reads a two-byte variable index.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The index, or `None`.
fn var2(il: &[u8], after: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*il.get(after)?, *il.get(after + 1)?]))
}

/// Renders a branch target operand.
///
/// # Arguments
///
/// * `after` - Offset after the opcode.
/// * `delta` - Signed branch displacement.
/// * `width` - Operand width in bytes.
///
/// # Returns
///
/// The target text and next offset.
fn render_branch(after: usize, delta: i64, width: usize) -> (String, usize) {
    let next = after + width;
    let target = to_i64(next) + delta;
    (format!("IL_{target:04x}"), next)
}

/// Converts a `usize` offset to `i64`.
///
/// # Arguments
///
/// * `value` - Offset value.
///
/// # Returns
///
/// The value as `i64`.
fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Renders a switch table operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
///
/// # Returns
///
/// The target list and next offset, or `None`.
fn render_switch(il: &[u8], after: usize) -> Option<(String, usize)> {
    let count = read_u32(il, after)? as usize;
    let start = after + 4;
    let end = start.checked_add(count.checked_mul(4)?)?;
    Some((switch_targets(il, start, count, end), end))
}

/// Renders every switch target.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `start` - Offset of the target table.
/// * `count` - Number of targets.
/// * `end` - Offset after the target table.
///
/// # Returns
///
/// The comma-separated targets.
fn switch_targets(il: &[u8], start: usize, count: usize, end: usize) -> String {
    let targets: Vec<String> = (0..count).map(|i| switch_target(il, start + i * 4, end)).collect();
    targets.join(", ")
}

/// Renders one switch target.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `at` - Offset of the target entry.
/// * `end` - Offset after the target table.
///
/// # Returns
///
/// The target text.
fn switch_target(il: &[u8], at: usize, end: usize) -> String {
    let delta = i64::from(read_i32(il, at).unwrap_or(0));
    format!("IL_{:04x}", to_i64(end) + delta)
}

/// Renders a `#US` string operand.
///
/// # Arguments
///
/// * `il` - IL bytes.
/// * `after` - Offset after the opcode.
/// * `bytes` - Raw file contents.
/// * `streams` - Metadata streams.
///
/// # Returns
///
/// The quoted string and next offset, or `None`.
fn render_string(
    il: &[u8],
    after: usize,
    bytes: &[u8],
    streams: &Streams,
) -> Option<(String, usize)> {
    let token = read_u32(il, after)? as usize & 0x00ff_ffff;
    let text = user_string(bytes, streams.user_strings + token);
    Some((format!("{text:?}"), after + 4))
}

/// Reads a `#US` user string by index.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Offset of the string entry.
///
/// # Returns
///
/// The decoded string.
fn user_string(bytes: &[u8], offset: usize) -> String {
    let Some((length, start)) = compressed_uint(bytes, offset) else {
        return String::new();
    };
    utf16_string(bytes, start, length.saturating_sub(1))
}

/// Decodes a UTF-16LE string of `length` bytes.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `start` - String start offset.
/// * `length` - Byte length.
///
/// # Returns
///
/// The decoded string.
fn utf16_string(bytes: &[u8], start: usize, length: usize) -> String {
    let Some(slice) = bytes.get(start..start.saturating_add(length)) else {
        return String::new();
    };
    String::from_utf16_lossy(&units(slice))
}

/// Decodes little-endian UTF-16 units from bytes.
///
/// # Arguments
///
/// * `slice` - UTF-16LE bytes.
///
/// # Returns
///
/// The decoded code units.
fn units(slice: &[u8]) -> Vec<u16> {
    slice
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Reads a compressed unsigned integer.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Offset of the integer.
///
/// # Returns
///
/// The value and the offset after it.
fn compressed_uint(bytes: &[u8], offset: usize) -> Option<(usize, usize)> {
    let first = *bytes.get(offset)?;
    if first & 0x80 == 0 {
        return Some((usize::from(first), offset + 1));
    }
    if first & 0xC0 == 0x80 {
        return two_byte(bytes, offset, first);
    }
    four_byte(bytes, offset, first)
}

/// Reads a two-byte compressed integer.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Offset of the integer.
/// * `first` - First byte.
///
/// # Returns
///
/// The value and next offset, or `None`.
fn two_byte(bytes: &[u8], offset: usize, first: u8) -> Option<(usize, usize)> {
    let second = *bytes.get(offset + 1)?;
    let value = (usize::from(first & 0x3f) << 8) | usize::from(second);
    Some((value, offset + 2))
}

/// Reads a four-byte compressed integer.
///
/// # Arguments
///
/// * `bytes` - Raw file contents.
/// * `offset` - Offset of the integer.
/// * `first` - First byte.
///
/// # Returns
///
/// The value and next offset, or `None`.
fn four_byte(bytes: &[u8], offset: usize, first: u8) -> Option<(usize, usize)> {
    let rest = bytes.get(offset + 1..offset + 4)?;
    let value = (usize::from(first & 0x1f) << 24)
        | (usize::from(rest[0]) << 16)
        | (usize::from(rest[1]) << 8)
        | usize::from(rest[2]);
    Some((value, offset + 4))
}

/// Reads a little-endian `u32`.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - Read offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Reads a little-endian `u64`.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - Read offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let slice = bytes.get(offset..offset + 8)?;
    Some(u64::from_le_bytes(slice.try_into().ok()?))
}

/// Reads a little-endian `i32`.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - Read offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Reads a little-endian `i64`.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - Read offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_i64(bytes: &[u8], offset: usize) -> Option<i64> {
    let slice = bytes.get(offset..offset + 8)?;
    Some(i64::from_le_bytes(slice.try_into().ok()?))
}

/// Reads a heap index of the given size.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - Read offset.
/// * `size` - Index size in bytes.
///
/// # Returns
///
/// The index value, or `None`.
fn read_index(bytes: &[u8], offset: usize, size: usize) -> Option<usize> {
    if size == 4 {
        read_u32(bytes, offset).map(|v| v as usize)
    } else {
        let slice = bytes.get(offset..offset + 2)?;
        Some(usize::from(u16::from_le_bytes([slice[0], slice[1]])))
    }
}

/// Reads a null-terminated string.
///
/// # Arguments
///
/// * `bytes` - Byte buffer.
/// * `offset` - String offset.
///
/// # Returns
///
/// The string content.
fn read_cstring(bytes: &[u8], offset: usize) -> String {
    let Some(slice) = bytes.get(offset..) else {
        return String::new();
    };
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).to_string()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies opcode lookup for one- and two-byte opcodes.
    #[test]
    fn test_lookup_opcodes() {
        assert_eq!(lookup(0x28), Some(("call", Op::Tok)));
        assert_eq!(lookup(0x72), Some(("ldstr", Op::Str)));
        assert_eq!(lookup(0xFE01), Some(("ceq", Op::None)));
    }

    /// Verifies compressed integer decoding across widths.
    #[test]
    fn test_compressed_uint() {
        assert_eq!(compressed_uint(&[0x05], 0), Some((5, 1)));
        assert_eq!(compressed_uint(&[0x80, 0x40], 0), Some((0x40, 2)));
    }

    /// Verifies branch target rendering.
    #[test]
    fn test_render_branch() {
        let (text, next) = render_branch(1, -2, 1);
        assert_eq!(next, 2);
        assert_eq!(text, "IL_0000");
    }

    /// Verifies CIL decoding of a simple `ldstr`/`ret` body.
    #[test]
    fn test_decode_simple_body() {
        let il = [0x72, 0x01, 0x00, 0x00, 0x70, 0x2a];
        let streams = Streams::default();
        let lines = decode(&il, &[], &streams, &empty_names());
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("ldstr"));
        assert!(lines[1].contains("ret"));
    }

    /// Builds an empty metadata name set.
    ///
    /// # Returns
    ///
    /// The empty names.
    fn empty_names() -> Names {
        Names {
            type_defs: Vec::new(),
            type_refs: Vec::new(),
            methods: Vec::new(),
            members: Vec::new(),
            fields: Vec::new(),
        }
    }
}
