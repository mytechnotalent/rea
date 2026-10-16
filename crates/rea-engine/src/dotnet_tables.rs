//! .NET ECMA-335 Metadata Table Walker
//!
//! **File:** `dotnet_tables.rs`
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

// Import dependencies from crate
use crate::dotnet::{DotNetAssemblyRef, DotNetMetadata};

/// A metadata table column type.
enum Col {
    /// Two-byte unsigned integer.
    U16,

    /// Four-byte unsigned integer.
    U32,

    /// `#Strings` heap index.
    Str,

    /// `#GUID` heap index.
    Guid,

    /// `#Blob` heap index.
    Blob,

    /// Simple index into the given table id.
    Simple(u8),

    /// Coded index with the given coded-index id.
    Coded(u8),
}

/// Tag-bit widths for each coded index id.
const CODED_BITS: [u8; 13] = [2, 2, 5, 1, 2, 3, 1, 1, 1, 2, 3, 2, 1];

/// Referenced table sets for each coded index id.
///
/// Ids follow `CODED_BITS` order: `TypeDefOrRef`, `HasConstant`,
/// `HasCustomAttribute`, `HasFieldMarshal`, `HasDeclSecurity`,
/// `MemberRefParent`, `HasSemantics`, `MethodDefOrRef`, `MemberForwarded`,
/// `Implementation`, `CustomAttributeType`, `ResolutionScope`, `TypeOrMethodDef`.
const CODED_TABLES: [&[u8]; 13] = [
    &[0x02, 0x01, 0x1b],
    &[0x04, 0x08, 0x17],
    &[
        0x06, 0x04, 0x01, 0x02, 0x08, 0x09, 0x0a, 0x00, 0x0e, 0x17, 0x14, 0x11, 0x1a, 0x1b, 0x20,
        0x23, 0x26, 0x27, 0x28, 0x2a, 0x2c, 0x2b,
    ],
    &[0x04, 0x08],
    &[0x02, 0x06, 0x20],
    &[0x02, 0x01, 0x1a, 0x06, 0x1b],
    &[0x14, 0x17],
    &[0x06, 0x0a],
    &[0x04, 0x06],
    &[0x26, 0x23, 0x27],
    &[0x06, 0x0a],
    &[0x00, 0x1a, 0x23, 0x01],
    &[0x02, 0x06],
];

/// Empty column list for unused table ids.
const EMPTY: &[Col] = &[];

/// Column schemas indexed by metadata table id (`0x00`-`0x2C`).
#[rustfmt::skip]
const SCHEMAS: [&[Col]; 0x2d] = [
    &[Col::U16, Col::Str, Col::Guid, Col::Guid, Col::Guid],
    &[Col::Coded(11), Col::Str, Col::Str],
    &[Col::U32, Col::Str, Col::Str, Col::Coded(0), Col::Simple(0x04), Col::Simple(0x06)],
    &[Col::Simple(0x04)],
    &[Col::U16, Col::Str, Col::Blob],
    &[Col::Simple(0x06)],
    &[Col::U32, Col::U16, Col::U16, Col::Str, Col::Blob, Col::Simple(0x08)],
    &[Col::Simple(0x08)],
    &[Col::U16, Col::U16, Col::Str],
    &[Col::Simple(0x02), Col::Coded(0)],
    &[Col::Coded(5), Col::Str, Col::Blob],
    &[Col::U16, Col::Coded(1), Col::Blob],
    &[Col::Coded(2), Col::Coded(10), Col::Blob],
    &[Col::Coded(3), Col::Blob],
    &[Col::U16, Col::Coded(4), Col::Blob],
    &[Col::U16, Col::U32, Col::Simple(0x02)],
    &[Col::U32, Col::Simple(0x04)],
    &[Col::Blob],
    &[Col::Simple(0x02), Col::Simple(0x14)],
    &[Col::Simple(0x14)],
    &[Col::U16, Col::Str, Col::Coded(0)],
    &[Col::Simple(0x02), Col::Simple(0x17)],
    &[Col::Simple(0x17)],
    &[Col::U16, Col::Str, Col::Blob],
    &[Col::U16, Col::Simple(0x06), Col::Coded(6)],
    &[Col::Simple(0x02), Col::Coded(7), Col::Coded(7)],
    &[Col::Str],
    &[Col::Blob],
    &[Col::U16, Col::Coded(8), Col::Str, Col::Simple(0x1a)],
    &[Col::U32, Col::Simple(0x04)],
    &[Col::U32, Col::U32],
    &[Col::U32],
    &[Col::U32, Col::U16, Col::U16, Col::U16, Col::U16, Col::U32, Col::Blob, Col::Str, Col::Str],
    &[Col::U32],
    &[Col::U32, Col::U32, Col::U32],
    &[Col::U16, Col::U16, Col::U16, Col::U16, Col::U32, Col::Blob, Col::Str, Col::Str, Col::Blob],
    &[Col::U32, Col::Simple(0x23)],
    &[Col::U32, Col::U32, Col::U32, Col::Simple(0x23)],
    &[Col::U32, Col::Str, Col::Blob],
    &[Col::U32, Col::U32, Col::Str, Col::Str, Col::Coded(9)],
    &[Col::U32, Col::U32, Col::Str, Col::Coded(9)],
    &[Col::Simple(0x02), Col::Simple(0x02)],
    &[Col::U16, Col::U16, Col::Coded(12), Col::Str],
    &[Col::Coded(7), Col::Blob],
    &[Col::Simple(0x2a), Col::Coded(0)],
];

/// Located metadata stream offsets.
#[derive(Default)]
struct StreamLoc {
    /// Tables (`#~`/`#-`) stream offset and size.
    tables: Option<(usize, usize)>,

    /// `#Strings` heap offset.
    strings: Option<usize>,

    /// `#Blob` heap offset.
    blob: Option<usize>,
}

impl StreamLoc {
    /// Records a named stream location.
    ///
    /// # Arguments
    ///
    /// * `name` - Stream name.
    /// * `off` - Absolute stream offset.
    /// * `size` - Stream size in bytes.
    fn assign(&mut self, name: &str, off: usize, size: usize) {
        match name {
            "#~" | "#-" => self.tables = Some((off, size)),
            "#Strings" => self.strings = Some(off),
            "#Blob" => self.blob = Some(off),
            _ => {}
        }
    }
}

/// Heap index widths derived from the heap-size flags.
struct Heaps {
    /// `#Strings` index width.
    strings: usize,

    /// `#GUID` index width.
    guid: usize,

    /// `#Blob` index width.
    blob: usize,
}

impl Heaps {
    /// Builds heap widths from the heap-size flags.
    ///
    /// # Arguments
    ///
    /// * `flags` - Heap-size flags byte.
    ///
    /// # Returns
    ///
    /// The heap widths.
    fn new(flags: u16) -> Heaps {
        Heaps { strings: width(flags, 0), guid: width(flags, 1), blob: width(flags, 2) }
    }
}

/// Parsed metadata tables stream context.
struct Tables<'a> {
    /// Raw file bytes.
    bytes: &'a [u8],

    /// Row counts indexed by table id.
    rows: [u32; 64],

    /// Heap index widths.
    heaps: Heaps,

    /// Offset of the first present table's rows.
    data: usize,

    /// `#Strings` heap offset.
    strings: usize,

    /// `#Blob` heap offset.
    blobs: usize,
}

impl Tables<'_> {
    /// Parses a tables stream into a row-counted context.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw file bytes.
    /// * `toff` - Tables stream offset.
    /// * `soff` - `#Strings` heap offset.
    ///
    /// # Returns
    ///
    /// The parsed context, or `None`.
    fn parse(bytes: &[u8], toff: usize, soff: usize, boff: usize) -> Option<Tables<'_>> {
        let flags = read_u16(bytes, toff + 6)?;
        let valid = read_u64(bytes, toff + 8)?;
        let rows = read_rows(bytes, toff + 24, valid)?;
        let data = toff + 24 + 4 * valid.count_ones() as usize;
        let heaps = Heaps::new(flags);
        Some(Tables { bytes, rows, heaps, data, strings: soff, blobs: boff })
    }

    /// Recovers all `AssemblyRef` entries.
    ///
    /// # Returns
    ///
    /// The recovered assembly references.
    fn assembly_refs(&self) -> Vec<DotNetAssemblyRef> {
        let off = self.table_offset(0x23);
        let n = self.rows[0x23] as usize;
        let size = self.row_size(0x23);
        (0..n).filter_map(|r| self.assembly_ref(off + r * size)).collect()
    }

    /// Computes the offset of a table's rows.
    ///
    /// # Arguments
    ///
    /// * `target` - Target table id.
    ///
    /// # Returns
    ///
    /// The row offset.
    fn table_offset(&self, target: u8) -> usize {
        let mut off = self.data;
        for table in 0..target {
            off += self.row_size(table) * self.rows[table as usize] as usize;
        }
        off
    }

    /// Computes the byte size of one table row.
    ///
    /// # Arguments
    ///
    /// * `table` - Table id.
    ///
    /// # Returns
    ///
    /// The row size in bytes.
    fn row_size(&self, table: u8) -> usize {
        columns(table).iter().map(|col| self.col_width(col)).sum()
    }

    /// Computes the byte width of one column.
    ///
    /// # Arguments
    ///
    /// * `col` - Column type.
    ///
    /// # Returns
    ///
    /// The column width in bytes.
    fn col_width(&self, col: &Col) -> usize {
        match col {
            Col::Simple(table) => simple_width(*table, &self.rows),
            Col::Coded(id) => coded_width(*id, &self.rows),
            other => fixed_width(other, &self.heaps),
        }
    }

    /// Reads one `AssemblyRef` row.
    ///
    /// # Arguments
    ///
    /// * `off` - Row offset.
    ///
    /// # Returns
    ///
    /// The assembly reference, or `None`.
    fn assembly_ref(&self, off: usize) -> Option<DotNetAssemblyRef> {
        let token = self.read_blob_at(off + 12);
        let name_col = off + 12 + self.heaps.blob;
        let name_idx = self.read_index(name_col)?;
        let name = self.read_string(name_idx)?;
        let version = version_text(self.bytes, off)?;
        Some(DotNetAssemblyRef { name, version, token })
    }

    /// Reads a `#Strings` heap index at an offset.
    ///
    /// # Arguments
    ///
    /// * `at` - Index offset.
    ///
    /// # Returns
    ///
    /// The heap-relative index, or `None`.
    fn read_index(&self, at: usize) -> Option<usize> {
        self.read_indices(at, self.heaps.strings)
    }

    /// Reads a heap index of the given width at an offset.
    ///
    /// # Arguments
    ///
    /// * `at` - Index offset.
    /// * `width` - Index width in bytes.
    ///
    /// # Returns
    ///
    /// The index value, or `None`.
    fn read_indices(&self, at: usize, width: usize) -> Option<usize> {
        let value = if width == 4 {
            read_u32(self.bytes, at)? as usize
        } else {
            read_u16(self.bytes, at)? as usize
        };
        Some(value)
    }

    /// Reads and hex-encodes a `#Blob` entry addressed by an index column.
    ///
    /// # Arguments
    ///
    /// * `index_col` - Offset of the blob index.
    ///
    /// # Returns
    ///
    /// The hex token, or an empty string.
    fn read_blob_at(&self, index_col: usize) -> String {
        let index = self.read_indices(index_col, self.heaps.blob);
        let Some(index) = index else {
            return String::new();
        };
        match self.read_blob(index) {
            Some(bytes) => crate::hashes::hex(&bytes),
            None => String::new(),
        }
    }

    /// Reads a `#Blob` heap entry by heap-relative offset.
    ///
    /// # Arguments
    ///
    /// * `index` - Heap-relative offset.
    ///
    /// # Returns
    ///
    /// The blob bytes, or `None`.
    fn read_blob(&self, index: usize) -> Option<Vec<u8>> {
        let base = self.blobs + index;
        let (len, size) = compressed_uint(self.bytes, base)?;
        let start = base + size;
        Some(self.bytes.get(start..start + len)?.to_vec())
    }

    /// Reads a `#Strings` heap entry by offset.
    ///
    /// # Arguments
    ///
    /// * `at` - Heap-relative offset.
    ///
    /// # Returns
    ///
    /// The string, or `None`.
    fn read_string(&self, at: usize) -> Option<String> {
        let text = read_cstring(self.bytes, self.strings + at)?;
        (!text.is_empty()).then_some(text)
    }
}

/// Recovers managed assembly references into metadata.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `bsjb` - Offset of the `BSJB` metadata root.
/// * `meta` - Output .NET metadata.
pub fn recover_references(bytes: &[u8], bsjb: usize, meta: &mut DotNetMetadata) {
    let loc = find_streams(bytes, bsjb);
    let streams = loc.as_ref().and_then(stream_pair);
    let parsed = streams.and_then(|(t, s, b)| Tables::parse(bytes, t, s, b));
    if let Some(tables) = parsed {
        meta.references = tables.assembly_refs();
    }
}

/// Extracts the tables and strings stream offsets.
///
/// # Arguments
///
/// * `loc` - Located streams.
///
/// # Returns
///
/// The tables offset and strings offset.
fn stream_pair(loc: &StreamLoc) -> Option<(usize, usize, usize)> {
    let (toff, _) = loc.tables?;
    Some((toff, loc.strings?, loc.blob?))
}

/// Locates the metadata streams of interest.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `bsjb` - Offset of the `BSJB` metadata root.
///
/// # Returns
///
/// The stream locations, or `None`.
fn find_streams(bytes: &[u8], bsjb: usize) -> Option<StreamLoc> {
    let end = bsjb + 16 + read_u32(bytes, bsjb + 12)? as usize;
    let count = read_u16(bytes, end + 2)?.min(16) as usize;
    let mut state = (StreamLoc::default(), end + 4);
    for _ in 0..count {
        state = stream_step(bytes, bsjb, state)?;
    }
    Some(state.0)
}

/// Advances over one stream header entry.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `bsjb` - Offset of the `BSJB` metadata root.
/// * `state` - Current locations and cursor.
///
/// # Returns
///
/// The updated state, or `None`.
fn stream_step(bytes: &[u8], bsjb: usize, state: (StreamLoc, usize)) -> Option<(StreamLoc, usize)> {
    let (mut loc, p) = state;
    let (name, off, size, next) = stream_entry(bytes, bsjb, p)?;
    loc.assign(&name, off, size);
    Some((loc, next))
}

/// Parses one stream header entry.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `bsjb` - Offset of the `BSJB` metadata root.
/// * `p` - Entry offset.
///
/// # Returns
///
/// The name, absolute offset, size, and next entry offset.
fn stream_entry(bytes: &[u8], bsjb: usize, p: usize) -> Option<(String, usize, usize, usize)> {
    let off = read_u32(bytes, p)? as usize;
    let size = read_u32(bytes, p + 4)? as usize;
    let name = read_stream_name(bytes, p + 8)?;
    let next = p + 8 + ((name.len() + 4) & !3);
    Some((name, bsjb + off, size, next))
}

/// Reads a null-terminated stream name.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `p` - Name offset.
///
/// # Returns
///
/// The stream name, or `None`.
fn read_stream_name(bytes: &[u8], p: usize) -> Option<String> {
    let slice = bytes.get(p..p + 32)?;
    let end = slice.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&slice[..end]).to_string())
}

/// Reads a null-terminated string from a heap.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
///
/// # Returns
///
/// The string, or `None`.
fn read_cstring(bytes: &[u8], at: usize) -> Option<String> {
    let slice = bytes.get(at..)?;
    let end = slice.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&slice[..end]).to_string())
}

/// Formats the four-part version of an `AssemblyRef` row.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `off` - Row offset.
///
/// # Returns
///
/// The dotted version, or `None`.
fn version_text(bytes: &[u8], off: usize) -> Option<String> {
    let major = read_u16(bytes, off)?;
    let minor = read_u16(bytes, off + 2)?;
    let build = read_u16(bytes, off + 4)?;
    let rev = read_u16(bytes, off + 6)?;
    Some(format!("{major}.{minor}.{build}.{rev}"))
}

/// Returns a table's column schema.
///
/// # Arguments
///
/// * `table` - Table id.
///
/// # Returns
///
/// The column list.
fn columns(table: u8) -> &'static [Col] {
    SCHEMAS.get(table as usize).copied().unwrap_or(EMPTY)
}

/// Computes the width of a heap index.
///
/// # Arguments
///
/// * `flags` - Heap-size flags byte.
/// * `bit` - Heap bit index.
///
/// # Returns
///
/// The index width in bytes.
fn width(flags: u16, bit: u32) -> usize {
    if flags & (1 << bit) != 0 {
        4
    } else {
        2
    }
}

/// Computes the width of a simple table index.
///
/// # Arguments
///
/// * `table` - Referenced table id.
/// * `rows` - Row counts.
///
/// # Returns
///
/// The index width in bytes.
fn simple_width(table: u8, rows: &[u32; 64]) -> usize {
    if rows[table as usize] >= 1 << 16 {
        4
    } else {
        2
    }
}

/// Computes the width of a coded index.
///
/// # Arguments
///
/// * `id` - Coded index id.
/// * `rows` - Row counts.
///
/// # Returns
///
/// The index width in bytes.
fn coded_width(id: u8, rows: &[u32; 64]) -> usize {
    let bits = CODED_BITS[id as usize];
    let max = CODED_TABLES[id as usize].iter().map(|&t| rows[t as usize]).max().unwrap_or(0);
    if max >= 1 << (16 - bits) {
        4
    } else {
        2
    }
}

/// Computes the width of a fixed-width column.
///
/// # Arguments
///
/// * `col` - Column type.
/// * `heaps` - Heap widths.
///
/// # Returns
///
/// The column width in bytes.
fn fixed_width(col: &Col, heaps: &Heaps) -> usize {
    match col {
        Col::U16 => 2,
        Col::U32 => 4,
        Col::Str => heaps.strings,
        Col::Guid => heaps.guid,
        Col::Blob => heaps.blob,
        _ => 0,
    }
}

/// Reads the row counts for present tables.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `p` - Row-count offset.
/// * `valid` - Present-table bitmask.
///
/// # Returns
///
/// The row counts, or `None`.
fn read_rows(bytes: &[u8], p: usize, valid: u64) -> Option<[u32; 64]> {
    let mut rows = [0u32; 64];
    let mut p = p;
    for i in set_bits(valid) {
        rows[i] = read_u32(bytes, p)?;
        p += 4;
    }
    Some(rows)
}

/// Iterates the set bit indices of a mask.
///
/// # Arguments
///
/// * `valid` - Mask value.
///
/// # Returns
///
/// An iterator over set bit indices.
fn set_bits(valid: u64) -> impl Iterator<Item = usize> {
    (0..64).filter(move |i| valid & (1 << i) != 0)
}

/// Reads an ECMA-335 compressed unsigned integer.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
///
/// # Returns
///
/// The value and its encoded length, or `None`.
fn compressed_uint(bytes: &[u8], at: usize) -> Option<(usize, usize)> {
    let first = *bytes.get(at)? as usize;
    if first & 0x80 == 0 {
        return Some((first, 1));
    }
    wide_uint(bytes, at, first)
}

/// Reads a two- or four-byte compressed unsigned integer.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
/// * `first` - First byte.
///
/// # Returns
///
/// The value and its encoded length, or `None`.
fn wide_uint(bytes: &[u8], at: usize, first: usize) -> Option<(usize, usize)> {
    let second = *bytes.get(at + 1)? as usize;
    if first & 0x40 == 0 {
        return Some((((first & 0x3f) << 8) | second, 2));
    }
    let high = ((first & 0x1f) << 24) | (second << 16);
    Some((compressed_tail(bytes, at, high)?, 4))
}

/// Reads the trailing bytes of a four-byte compressed integer.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
/// * `high` - Pre-accumulated high bytes.
///
/// # Returns
///
/// The value, or `None`.
fn compressed_tail(bytes: &[u8], at: usize, high: usize) -> Option<usize> {
    let second = *bytes.get(at + 2)? as usize;
    let third = *bytes.get(at + 3)? as usize;
    Some(high | (second << 8) | third)
}

/// Reads a little-endian `u16`.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let b = bytes.get(at..at + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

/// Reads a little-endian `u32`.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Reads a little-endian `u64`.
///
/// # Arguments
///
/// * `bytes` - Raw file bytes.
/// * `at` - Source offset.
///
/// # Returns
///
/// The value, or `None`.
fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let b = bytes.get(at..at + 8)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(b);
    Some(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Pushes a little-endian `u16`.
    ///
    /// # Arguments
    ///
    /// * `buf` - Output buffer.
    /// * `value` - Value to write.
    fn push_u16(buf: &mut Vec<u8>, value: u16) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Pushes a little-endian `u32`.
    ///
    /// # Arguments
    ///
    /// * `buf` - Output buffer.
    /// * `value` - Value to write.
    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Pushes a little-endian `u64`.
    ///
    /// # Arguments
    ///
    /// * `buf` - Output buffer.
    /// * `value` - Value to write.
    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Appends a stream header entry.
    ///
    /// # Arguments
    ///
    /// * `buf` - Output buffer.
    /// * `name` - Stream name.
    /// * `off` - Stream offset.
    /// * `size` - Stream size.
    fn add_stream(buf: &mut Vec<u8>, name: &str, off: u32, size: u32) {
        push_u32(buf, off);
        push_u32(buf, size);
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
    }

    /// Stream headers for the synthetic blob.
    ///
    /// # Returns
    ///
    /// The encoded stream headers.
    fn stream_headers() -> Vec<u8> {
        let mut buf = Vec::new();
        add_stream(&mut buf, "#Strings", 80, 16);
        add_stream(&mut buf, "#Blob", 96, 12);
        add_stream(&mut buf, "#~", 108, 48);
        buf
    }

    /// `BSJB` prefix with version and stream count.
    ///
    /// # Returns
    ///
    /// The encoded prefix.
    fn prefix() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::dotnet::BSJB_MAGIC);
        buf.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
        push_u32(&mut buf, 12);
        buf.extend_from_slice(b"v4.0.30319\0\0");
        push_u32(&mut buf, 0x0003_0000);
        buf
    }

    /// `#Strings` heap for the synthetic blob.
    ///
    /// # Returns
    ///
    /// The encoded heap.
    fn strings_heap() -> Vec<u8> {
        b"\0System.Runtime\0".to_vec()
    }

    /// `#Blob` heap for the synthetic blob.
    ///
    /// # Returns
    ///
    /// The encoded heap.
    fn blob_heap() -> Vec<u8> {
        let mut buf = vec![0u8, 8];
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        buf.extend_from_slice(&[0, 0]);
        buf
    }

    /// Appends one `AssemblyRef` row.
    ///
    /// # Arguments
    ///
    /// * `buf` - Output buffer.
    fn push_assembly_ref_row(buf: &mut Vec<u8>) {
        push_u32(buf, 9);
        push_u32(buf, 0);
        push_u32(buf, 0);
        push_u16(buf, 1);
        push_u16(buf, 1);
        push_u16(buf, 0);
        push_u16(buf, 0);
    }

    /// `#~` tables stream for the synthetic blob.
    ///
    /// # Returns
    ///
    /// The encoded tables stream.
    fn tables() -> Vec<u8> {
        let mut buf = vec![0u8; 4];
        buf.extend_from_slice(&[2, 0, 0, 0]);
        push_u64(&mut buf, 1 << 0x23);
        push_u64(&mut buf, 1 << 0x23);
        push_u32(&mut buf, 1);
        push_assembly_ref_row(&mut buf);
        buf
    }

    /// Assembles a minimal managed metadata blob.
    ///
    /// # Returns
    ///
    /// The encoded blob.
    fn blob() -> Vec<u8> {
        let mut buf = prefix();
        buf.extend_from_slice(&stream_headers());
        buf.extend_from_slice(&strings_heap());
        buf.extend_from_slice(&blob_heap());
        buf.extend_from_slice(&tables());
        buf
    }

    /// Verifies `AssemblyRef` recovery from metadata tables.
    #[test]
    fn test_recovers_assembly_ref() {
        let bytes = blob();
        let meta = crate::dotnet::parse_dotnet_metadata(&bytes).expect("metadata");
        assert_eq!(meta.references.len(), 1);
        assert_eq!(meta.references[0].name, "System.Runtime");
        assert_eq!(meta.references[0].version, "9.0.0.0");
        assert_eq!(meta.references[0].token, "0102030405060708");
    }

    /// Verifies coded and simple index width computation.
    #[test]
    fn test_index_widths() {
        let mut rows = [0u32; 64];
        rows[0x02] = 100;
        assert_eq!(simple_width(0x02, &rows), 2);
        assert_eq!(coded_width(0, &rows), 2);
        rows[0x02] = 70_000;
        assert_eq!(simple_width(0x02, &rows), 4);
    }
}
