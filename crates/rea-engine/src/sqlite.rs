//! SQLite Database Schema and Row Recovery
//!
//! **File:** `sqlite.rs`
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

/// Database file magic string.
pub const SQLITE_MAGIC: &[u8; 16] = b"SQLite format 3\x00";

/// B-tree page type for a table leaf.
const LEAF_TABLE: u8 = 0x0d;

/// B-tree page type for a table interior node.
const INTERIOR_TABLE: u8 = 0x05;

/// Byte widths of serial types zero through eleven.
const SERIAL_SIZES: [usize; 12] = [0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0];

/// Most rows sampled per table.
const MAX_ROWS: usize = 20;

/// Parsed database header fields.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SqliteHeader {
    /// Page size in bytes.
    pub page_size: u32,
    /// Total page count.
    pub page_count: u32,
    /// Text encoding label.
    pub encoding: &'static str,
    /// Schema cookie version.
    pub schema_version: u32,
    /// Application-defined user version.
    pub user_version: u32,
    /// First freelist trunk page.
    pub freelist: u32,
}

/// A schema object from `sqlite_master`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SqliteObject {
    /// Object kind: table, index, view, or trigger.
    pub kind: String,
    /// Object name.
    pub name: String,
    /// Owning table name.
    pub table: String,
    /// Root page number.
    pub root: u32,
    /// Original creation SQL.
    pub sql: String,
}

/// A table column and its declared type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SqliteColumn {
    /// Column name.
    pub name: String,
    /// Declared type text.
    pub kind: String,
}

/// Recovered database inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SqliteDb {
    /// Parsed header fields.
    pub header: SqliteHeader,
    /// Schema objects.
    pub objects: Vec<SqliteObject>,
    /// Table columns.
    pub columns: Vec<SqliteColumn>,
    /// Sampled row renderings.
    pub rows: Vec<String>,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// Accumulator that splits text on top-level commas.
#[derive(Default)]
struct Splitter {
    /// Completed parts.
    parts: Vec<String>,
    /// Current part buffer.
    current: String,
    /// Parenthesis nesting depth.
    depth: i32,
}

impl Splitter {
    /// Consumes one character, breaking parts at top-level commas.
    ///
    /// # Arguments
    ///
    /// * `ch` - Character to consume.
    fn push(&mut self, ch: char) {
        if ch == ',' && self.depth == 0 {
            self.flush();
        } else {
            self.depth += paren_delta(ch);
            self.current.push(ch);
        }
    }

    /// Pushes the current buffer as a trimmed part.
    fn flush(&mut self) {
        self.parts.push(self.current.trim().to_string());
        self.current.clear();
    }

    /// Finalizes the splitter and returns the parts.
    ///
    /// # Returns
    ///
    /// Every trimmed part.
    fn finish(mut self) -> Vec<String> {
        self.flush();
        self.parts
    }
}

/// Reports whether a buffer begins with the SQLite magic.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when the magic prefix matches.
#[must_use]
pub fn is_sqlite(bytes: &[u8]) -> bool {
    bytes.starts_with(SQLITE_MAGIC)
}

/// Reports the nesting change for a parenthesis character.
///
/// # Arguments
///
/// * `ch` - Character to classify.
///
/// # Returns
///
/// One for `(`, minus one for `)`, otherwise zero.
fn paren_delta(ch: char) -> i32 {
    match ch {
        '(' => 1,
        ')' => -1,
        _ => 0,
    }
}

/// Reads a big-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Offset of the word.
///
/// # Returns
///
/// The word, when four bytes are available.
fn read_be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let word = bytes.get(offset..offset + 4)?;
    Some(u32::from_be_bytes([word[0], word[1], word[2], word[3]]))
}

/// Reads an unsigned varint at a cursor.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `pos` - Cursor advanced past the varint.
///
/// # Returns
///
/// The decoded value, or `None` on truncation.
fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let slice = bytes.get(*pos..)?;
    let (value, used) = decode_varint(slice)?;
    *pos += used;
    Some(value)
}

/// Decodes an unsigned varint and its encoded length.
///
/// # Arguments
///
/// * `bytes` - Buffer beginning at the varint.
///
/// # Returns
///
/// The value and the number of bytes consumed.
fn decode_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut result = 0u64;
    for (i, &byte) in bytes.iter().enumerate().take(9) {
        result = combine_varint(result, byte, i);
        if byte & 0x80 == 0 || i == 8 {
            return Some((result, i + 1));
        }
    }
    None
}

/// Folds one varint byte into the accumulator.
///
/// # Arguments
///
/// * `result` - Accumulated value.
/// * `byte` - Next byte.
/// * `index` - Zero-based byte index.
///
/// # Returns
///
/// The updated accumulator.
fn combine_varint(result: u64, byte: u8, index: usize) -> u64 {
    if index == 8 {
        (result << 8) | u64::from(byte)
    } else {
        (result << 7) | u64::from(byte & 0x7f)
    }
}

/// Resolves the byte width of a serial type.
///
/// # Arguments
///
/// * `serial` - Serial type code.
///
/// # Returns
///
/// The stored byte width.
fn serial_size(serial: u64) -> usize {
    if serial < 12 {
        SERIAL_SIZES.get(usize::try_from(serial).unwrap_or(0)).copied().unwrap_or(0)
    } else {
        usize::try_from((serial - 12) / 2).unwrap_or(0)
    }
}

/// Reports whether a serial type holds text.
///
/// # Arguments
///
/// * `serial` - Serial type code.
///
/// # Returns
///
/// `true` for text serial types.
fn is_text(serial: u64) -> bool {
    serial >= 13 && serial % 2 == 1
}

/// Resolves the page size from the header.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The page size, defaulting to 4096.
fn header_page_size(bytes: &[u8]) -> u32 {
    let raw = bytes.get(16..18).map(|b| u32::from(u16::from_be_bytes([b[0], b[1]])));
    match raw {
        Some(1) => 65536,
        Some(size) => size,
        None => 4096,
    }
}

/// Resolves a text encoding label.
///
/// # Arguments
///
/// * `code` - Encoding code.
///
/// # Returns
///
/// A canonical label or `unknown`.
fn encoding_label(code: u32) -> &'static str {
    match code {
        1 => "UTF-8",
        2 => "UTF-16le",
        3 => "UTF-16be",
        _ => "unknown",
    }
}

/// Parses the database header fields.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The parsed header.
fn parse_header(bytes: &[u8]) -> SqliteHeader {
    SqliteHeader {
        page_size: header_page_size(bytes),
        page_count: read_be_u32(bytes, 28).unwrap_or(0),
        encoding: encoding_label(read_be_u32(bytes, 56).unwrap_or(1)),
        schema_version: read_be_u32(bytes, 40).unwrap_or(0),
        user_version: read_be_u32(bytes, 60).unwrap_or(0),
        freelist: read_be_u32(bytes, 36).unwrap_or(0),
    }
}

/// Parses a SQLite database from a buffer.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// Recovered database inventory, or `None` when the magic is absent.
#[must_use]
pub fn parse_sqlite(bytes: &[u8]) -> Option<SqliteDb> {
    if !is_sqlite(bytes) {
        return None;
    }
    let mut db = SqliteDb { header: parse_header(bytes), ..Default::default() };
    let size = usize::try_from(db.header.page_size).unwrap_or(4096);
    parse_schema(bytes, size, &mut db);
    parse_table_rows(bytes, size, &mut db);
    Some(finish_db(db))
}

/// Records warnings and returns the finished inventory.
///
/// # Arguments
///
/// * `db` - Output database inventory.
///
/// # Returns
///
/// The finished inventory.
fn finish_db(mut db: SqliteDb) -> SqliteDb {
    record_warnings(&mut db);
    db
}

/// Reads a page by number.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `page` - One-based page number.
///
/// # Returns
///
/// The page slice, or an empty slice.
fn read_page(bytes: &[u8], page_size: usize, page: u32) -> &[u8] {
    let start = usize::try_from(page.saturating_sub(1)).unwrap_or(0) * page_size;
    let end = start.saturating_add(page_size).min(bytes.len());
    bytes.get(start..end).unwrap_or_default()
}

/// Reads a cell pointer from a page.
///
/// # Arguments
///
/// * `page` - Page bytes.
/// * `index` - Cell index.
///
/// # Returns
///
/// The cell offset, or `None`.
fn cell_pointer(page: &[u8], index: usize) -> Option<usize> {
    let at = 8 + index * 2;
    let b = page.get(at..at + 2)?;
    Some(usize::from(u16::from_be_bytes([b[0], b[1]])))
}

/// Reads the cell count from a page header.
///
/// # Arguments
///
/// * `page` - Page bytes.
///
/// # Returns
///
/// The declared cell count.
fn cell_count(page: &[u8]) -> usize {
    page.get(3..5).map_or(0, |b| usize::from(u16::from_be_bytes([b[0], b[1]])))
}

/// Lists the cell offsets of a table leaf page.
///
/// # Arguments
///
/// * `page` - Page bytes.
///
/// # Returns
///
/// Offsets of every cell.
fn leaf_cell_offsets(page: &[u8]) -> Vec<usize> {
    if page.first().copied() != Some(LEAF_TABLE) || page.len() < 8 {
        return Vec::new();
    }
    let count = cell_count(page);
    (0..count).filter_map(|i| cell_pointer(page, i)).collect()
}

/// Reads the record header and serial types.
///
/// # Arguments
///
/// * `payload` - Record payload.
///
/// # Returns
///
/// The serial types and the header length.
fn read_record_header(payload: &[u8]) -> Option<(Vec<u64>, usize)> {
    let mut pos = 0;
    let header = read_varint(payload, &mut pos).and_then(|v| usize::try_from(v).ok())?;
    let mut types = Vec::new();
    while pos < header && pos < payload.len() {
        types.push(read_varint(payload, &mut pos)?);
    }
    Some((types, header))
}

/// Decodes one serial value.
///
/// # Arguments
///
/// * `payload` - Record payload.
/// * `offset` - Value offset.
/// * `serial` - Serial type code.
///
/// # Returns
///
/// The value bytes and the width consumed.
fn decode_value(payload: &[u8], offset: usize, serial: u64) -> (Option<Vec<u8>>, usize) {
    let len = serial_size(serial);
    let end = offset.saturating_add(len);
    let raw = payload.get(offset..end).unwrap_or_default().to_vec();
    let value = if serial == 0 { None } else { Some(raw) };
    (value, len)
}

/// Decodes every value of a record.
///
/// # Arguments
///
/// * `payload` - Record payload.
/// * `header` - Record header length.
/// * `types` - Serial types.
///
/// # Returns
///
/// Decoded values in column order.
fn decode_values(payload: &[u8], header: usize, types: &[u64]) -> Vec<Option<Vec<u8>>> {
    let mut values = Vec::new();
    let mut offset = header;
    for &serial in types {
        let (value, used) = decode_value(payload, offset, serial);
        values.push(value);
        offset += used;
    }
    values
}

/// Renders one record cell as text.
///
/// # Arguments
///
/// * `serial` - Serial type code.
/// * `value` - Decoded value bytes.
///
/// # Returns
///
/// A human-readable cell rendering.
fn render_cell(serial: u64, value: Option<Vec<u8>>) -> String {
    match value {
        None => "NULL".to_string(),
        Some(raw) if is_text(serial) => String::from_utf8_lossy(&raw).to_string(),
        Some(raw) => hex_string(&raw),
    }
}

/// Renders bytes as lowercase hexadecimal.
///
/// # Arguments
///
/// * `bytes` - Bytes to render.
///
/// # Returns
///
/// The hexadecimal string.
fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Renders a record payload into text cells.
///
/// # Arguments
///
/// * `payload` - Record payload.
///
/// # Returns
///
/// One rendering per column.
fn render_record(payload: &[u8]) -> Vec<String> {
    let (types, header) = read_record_header(payload).unwrap_or_default();
    let values = decode_values(payload, header, &types);
    types.iter().zip(values).map(|(s, v)| render_cell(*s, v)).collect()
}

/// Reads the cells of one leaf table cell.
///
/// # Arguments
///
/// * `page` - Page bytes.
/// * `cell` - Cell offset.
///
/// # Returns
///
/// The rendered row cells, or `None`.
fn leaf_row_cells(page: &[u8], cell: usize) -> Option<Vec<String>> {
    let mut pos = cell;
    let payload = read_varint(page, &mut pos)?;
    let _rowid = read_varint(page, &mut pos)?;
    let len = usize::try_from(payload).ok()?;
    let end = pos.saturating_add(len).min(page.len());
    Some(render_record(page.get(pos..end)?))
}

/// Collects rows from a table leaf page.
///
/// # Arguments
///
/// * `page` - Page bytes.
/// * `out` - Accumulated rows.
fn collect_leaf(page: &[u8], out: &mut Vec<Vec<String>>) {
    for cell in leaf_cell_offsets(page) {
        if out.len() >= MAX_ROWS {
            break;
        }
        if let Some(cells) = leaf_row_cells(page, cell) {
            out.push(cells);
        }
    }
}

/// Reads an interior child pointer.
///
/// # Arguments
///
/// * `page` - Page bytes.
/// * `index` - Cell index.
///
/// # Returns
///
/// The child page number, or `None`.
fn child_at(page: &[u8], index: usize) -> Option<u32> {
    let cell = cell_pointer(page, index)?;
    read_be_u32(page, cell)
}

/// Lists the children of an interior table page.
///
/// # Arguments
///
/// * `page` - Page bytes.
///
/// # Returns
///
/// Child page numbers in order.
fn interior_children(page: &[u8]) -> Vec<u32> {
    let count = cell_count(page);
    let mut children: Vec<u32> = (0..count).filter_map(|i| child_at(page, i)).collect();
    if let Some(right) = read_be_u32(page, 8) {
        children.push(right);
    }
    children
}

/// Routes a page to its leaf or interior collector.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `page` - Page bytes.
/// * `out` - Accumulated rows.
fn dispatch_page(bytes: &[u8], page_size: usize, page: &[u8], out: &mut Vec<Vec<String>>) {
    match page.first().copied() {
        Some(LEAF_TABLE) => collect_leaf(page, out),
        Some(INTERIOR_TABLE) => collect_interior(bytes, page_size, page, out),
        _ => {}
    }
}

/// Recursively collects rows from a table b-tree.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `root` - Root page number.
/// * `out` - Accumulated rows.
fn collect_rows(bytes: &[u8], page_size: usize, root: u32, out: &mut Vec<Vec<String>>) {
    if out.len() >= MAX_ROWS || root == 0 {
        return;
    }
    let page = read_page(bytes, page_size, root);
    dispatch_page(bytes, page_size, page, out);
}

/// Collects rows from an interior page's children.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `page` - Interior page bytes.
/// * `out` - Accumulated rows.
fn collect_interior(bytes: &[u8], page_size: usize, page: &[u8], out: &mut Vec<Vec<String>>) {
    for child in interior_children(page) {
        collect_rows(bytes, page_size, child, out);
    }
}

/// Reads one cell of a rendered row.
///
/// # Arguments
///
/// * `cells` - Rendered cells.
/// * `index` - Column index.
///
/// # Returns
///
/// The cell text, or an empty string.
fn cell(cells: &[String], index: usize) -> String {
    cells.get(index).cloned().unwrap_or_default()
}

/// Builds a schema object from a `sqlite_master` row.
///
/// # Arguments
///
/// * `cells` - Rendered schema row.
///
/// # Returns
///
/// The parsed schema object.
fn parse_object(cells: &[String]) -> SqliteObject {
    SqliteObject {
        kind: cell(cells, 0),
        name: cell(cells, 1),
        table: cell(cells, 2),
        root: cell(cells, 3).parse().unwrap_or(0),
        sql: cell(cells, 4),
    }
}

/// Reports whether a column token is a table constraint.
///
/// # Arguments
///
/// * `word` - First token of a column definition.
///
/// # Returns
///
/// `true` for constraint keywords.
fn is_constraint(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "PRIMARY" | "UNIQUE" | "CHECK" | "FOREIGN" | "CONSTRAINT"
    )
}

/// Parses one column definition.
///
/// # Arguments
///
/// * `text` - Column definition text.
///
/// # Returns
///
/// The parsed column, or `None` for constraints.
fn parse_column(text: &str) -> Option<SqliteColumn> {
    let mut parts = text.split_whitespace();
    let name = parts.next()?;
    if is_constraint(name) {
        return None;
    }
    let kind = parts.next().unwrap_or("").trim_end_matches(',');
    Some(SqliteColumn { name: name.trim_matches('"').to_string(), kind: kind.to_string() })
}

/// Splits text on top-level commas.
///
/// # Arguments
///
/// * `text` - Text to split.
///
/// # Returns
///
/// The trimmed parts.
fn split_top_level(text: &str) -> Vec<String> {
    let mut splitter = Splitter::default();
    for ch in text.chars() {
        splitter.push(ch);
    }
    splitter.finish()
}

/// Extracts columns from a `CREATE TABLE` statement.
///
/// # Arguments
///
/// * `sql` - Creation SQL.
///
/// # Returns
///
/// The parsed columns.
fn extract_columns(sql: &str) -> Vec<SqliteColumn> {
    let (Some(open), Some(close)) = (sql.find('('), sql.rfind(')')) else {
        return Vec::new();
    };
    if close <= open {
        return Vec::new();
    }
    let parts = split_top_level(&sql[open + 1..close]);
    parts.into_iter().filter_map(|t| parse_column(&t)).collect()
}

/// Parses the schema from the `sqlite_master` b-tree.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `db` - Output database inventory.
fn parse_schema(bytes: &[u8], page_size: usize, db: &mut SqliteDb) {
    let page = read_page(bytes, page_size, 1);
    let mut rows = Vec::new();
    collect_leaf(page, &mut rows);
    for cells in &rows {
        let object = parse_object(cells);
        db.columns.extend(extract_columns(&object.sql));
        db.objects.push(object);
    }
}

/// Lists root pages of ordinary tables.
///
/// # Arguments
///
/// * `objects` - Schema objects.
///
/// # Returns
///
/// Root page numbers of table objects.
fn table_roots(objects: &[SqliteObject]) -> Vec<u32> {
    objects.iter().filter(|o| o.kind == "table").map(|o| o.root).collect()
}

/// Samples rows from every ordinary table.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `page_size` - Page size in bytes.
/// * `db` - Output database inventory.
fn parse_table_rows(bytes: &[u8], page_size: usize, db: &mut SqliteDb) {
    let roots = table_roots(&db.objects);
    for root in roots {
        let mut rows = Vec::new();
        collect_rows(bytes, page_size, root, &mut rows);
        db.rows.extend(rows.into_iter().map(|cells| cells.join(" | ")));
    }
}

/// Records validation warnings on the database inventory.
///
/// # Arguments
///
/// * `db` - Output database inventory.
fn record_warnings(db: &mut SqliteDb) {
    if db.header.page_size == 0 {
        db.warnings.push("invalid page size".to_string());
    }
}

/// Formats a database inventory as a human-readable text block.
///
/// # Arguments
///
/// * `db` - Recovered database inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_sqlite_summary(db: &SqliteDb) -> String {
    let mut out = String::new();
    append_sqlite_head(&mut out, db);
    append_sqlite_objects(&mut out, db);
    out
}

/// Appends SQLite header fields.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `db` - Recovered database inventory.
fn append_sqlite_head(out: &mut String, db: &SqliteDb) {
    let _ = writeln!(out, "SQLite pages: {}", db.header.page_count);
    let _ = writeln!(out, "Page size: {} bytes", db.header.page_size);
    let _ = writeln!(out, "Encoding: {}", db.header.encoding);
    let _ = writeln!(out, "Objects: {}", db.objects.len());
    let _ = writeln!(out, "Columns: {}", db.columns.len());
}

/// Appends SQLite schema objects.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `db` - Recovered database inventory.
fn append_sqlite_objects(out: &mut String, db: &SqliteDb) {
    for object in &db.objects {
        let _ = writeln!(out, "  {} {}", object.kind, object.name);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal database header.
    ///
    /// # Returns
    ///
    /// A 100-byte header buffer.
    fn header_bytes() -> Vec<u8> {
        let mut data = SQLITE_MAGIC.to_vec();
        data.resize(100, 0);
        data[16] = 0x10;
        data[18] = 1;
        data[19] = 1;
        data[31] = 4;
        data[59] = 1;
        data
    }

    /// Verifies database magic detection.
    #[test]
    fn test_sqlite_detection() {
        assert!(is_sqlite(SQLITE_MAGIC));
        assert!(!is_sqlite(b"MZ"));
    }

    /// Verifies header field parsing.
    #[test]
    fn test_sqlite_header() {
        let header = parse_header(&header_bytes());
        assert_eq!(header.page_size, 4096);
        assert_eq!(header.page_count, 4);
        assert_eq!(header.encoding, "UTF-8");
    }

    /// Verifies varint decoding.
    #[test]
    fn test_sqlite_varint() {
        assert_eq!(decode_varint(&[0x81, 0x01]), Some((129, 2)));
        assert_eq!(decode_varint(&[0x7f]), Some((127, 1)));
    }

    /// Verifies serial widths and text classification.
    #[test]
    fn test_sqlite_serial() {
        assert_eq!(serial_size(1), 1);
        assert_eq!(serial_size(5), 6);
        assert_eq!(serial_size(17), 2);
        assert!(is_text(17));
    }

    /// Verifies record rendering.
    #[test]
    fn test_sqlite_record() {
        let cells = render_record(&[0x02, 0x11, b'h', b'i']);
        assert_eq!(cells, vec!["hi".to_string()]);
    }

    /// Verifies top-level comma splitting.
    #[test]
    fn test_sqlite_split() {
        let parts = split_top_level("a INTEGER, b TEXT(10,2)");
        assert_eq!(
            parts,
            vec!["a INTEGER".to_string(), "b TEXT(10,2)".to_string()]
        );
    }

    /// Verifies column extraction from creation SQL.
    #[test]
    fn test_sqlite_columns() {
        let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, PRIMARY KEY (id))";
        let columns = extract_columns(sql);
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].name, "id");
        assert_eq!(columns[1].kind, "TEXT");
    }

    /// Verifies summary formatting.
    #[test]
    fn test_sqlite_summary() {
        let db = SqliteDb::default();
        assert!(format_sqlite_summary(&db).contains("SQLite pages"));
    }
}
