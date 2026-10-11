//! Report Document Model
//!
//! **File:** `document.rs`
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

/// A complete report, independent of its output format.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Document {
    /// Report title shown on the cover.
    pub title: String,

    /// One-line subtitle shown under the title.
    pub subtitle: String,

    /// Ordered report sections.
    pub sections: Vec<Section>,
}

/// One titled section of a report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Section {
    /// Section heading text.
    pub heading: String,

    /// Ordered content blocks.
    pub blocks: Vec<Block>,
}

/// One content block inside a section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// A paragraph of plain text.
    Paragraph(String),

    /// An unordered list of plain-text items.
    Bullets(Vec<String>),

    /// A table with a header row.
    Table(Table),

    /// A preformatted listing such as disassembly.
    Code(String),

    /// A second-level heading inside a section.
    Subheading(String),
}

/// A table with a header row and plain-text cells.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Table {
    /// Column headings.
    pub headers: Vec<String>,

    /// Data rows, each holding one cell per column.
    pub rows: Vec<Vec<String>>,
}

impl Document {
    /// Creates an empty document with a title and subtitle.
    ///
    /// # Arguments
    ///
    /// * `title` - Cover title.
    /// * `subtitle` - Cover subtitle.
    ///
    /// # Returns
    ///
    /// A document without sections.
    #[must_use]
    pub fn new(title: &str, subtitle: &str) -> Self {
        Self { title: title.to_string(), subtitle: subtitle.to_string(), sections: Vec::new() }
    }

    /// Appends a section, returning the document for chaining.
    ///
    /// # Arguments
    ///
    /// * `section` - Section appended after existing sections.
    ///
    /// # Returns
    ///
    /// The extended document.
    #[must_use]
    pub fn with_section(mut self, section: Section) -> Self {
        self.sections.push(section);
        self
    }
}

impl Section {
    /// Creates an empty section.
    ///
    /// # Arguments
    ///
    /// * `heading` - Section heading text.
    ///
    /// # Returns
    ///
    /// A section without blocks.
    #[must_use]
    pub fn new(heading: &str) -> Self {
        Self { heading: heading.to_string(), blocks: Vec::new() }
    }

    /// Appends a block, returning the section for chaining.
    ///
    /// # Arguments
    ///
    /// * `block` - Block appended after existing blocks.
    ///
    /// # Returns
    ///
    /// The extended section.
    #[must_use]
    pub fn with(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }
}

impl Table {
    /// Creates a table with the given column headings and no rows.
    ///
    /// # Arguments
    ///
    /// * `headers` - Column headings.
    ///
    /// # Returns
    ///
    /// An empty table.
    #[must_use]
    pub fn new(headers: &[&str]) -> Self {
        let headers = headers.iter().map(ToString::to_string).collect();
        Self { headers, rows: Vec::new() }
    }

    /// Appends a row, returning the table for chaining.
    ///
    /// # Arguments
    ///
    /// * `cells` - One cell per column.
    ///
    /// # Returns
    ///
    /// The extended table.
    #[must_use]
    pub fn row(mut self, cells: Vec<String>) -> Self {
        self.rows.push(cells);
        self
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that sections accumulate blocks in order.
    #[test]
    fn test_section_with_appends_in_order() {
        let first = Block::Paragraph("a".into());
        let second = Block::Code("b".into());
        let section = Section::new("S").with(first.clone()).with(second.clone());
        assert_eq!(section.blocks, vec![first, second]);
    }

    /// Verifies that tables keep headings and rows.
    #[test]
    fn test_table_row_appends_rows() {
        let table = Table::new(&["A", "B"]).row(vec!["1".into(), "2".into()]);
        assert_eq!(table.headers, vec!["A", "B"]);
        assert_eq!(table.rows.len(), 1);
    }

    /// Verifies that a new document starts without sections.
    #[test]
    fn test_document_new_is_empty() {
        let doc = Document::new("T", "S");
        assert_eq!((doc.title.as_str(), doc.sections.len()), ("T", 0));
    }

    /// Verifies that `with_section` appends sections.
    #[test]
    fn test_document_with_section() {
        let doc = Document::new("T", "S").with_section(Section::new("Sec"));
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].heading, "Sec");
    }
}
