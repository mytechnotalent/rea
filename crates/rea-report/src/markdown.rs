//! Markdown Report Renderer
//!
//! **File:** `markdown.rs`
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

// Import crate::document
use crate::document::{Block, Document, Section, Table};

/// Renders a document as GitHub-flavored Markdown.
///
/// # Arguments
///
/// * `doc` - Document to render.
///
/// # Returns
///
/// The complete Markdown text.
#[must_use]
pub fn render(doc: &Document) -> String {
    let mut out = format!("# {}\n\n_{}_\n\n", doc.title, doc.subtitle);
    for section in &doc.sections {
        out.push_str(&section_text(section));
    }
    out
}

/// Renders one section with its heading and blocks.
///
/// # Arguments
///
/// * `section` - Section to render.
///
/// # Returns
///
/// Markdown text for the section.
fn section_text(section: &Section) -> String {
    let mut out = format!("## {}\n\n", section.heading);
    for block in &section.blocks {
        out.push_str(&block_text(block));
        out.push('\n');
    }
    out
}

/// Renders one block.
///
/// # Arguments
///
/// * `block` - Block to render.
///
/// # Returns
///
/// Markdown text for the block, ending in a newline.
fn block_text(block: &Block) -> String {
    match block {
        Block::Paragraph(text) => format!("{text}\n"),
        Block::Bullets(items) => items.iter().map(|i| ["- ", i, "\n"].concat()).collect(),
        Block::Table(table) => table_text(table),
        Block::Code(code) => format!("{FENCE}text\n{code}\n{FENCE}\n"),
        Block::Subheading(text) => format!("### {text}\n"),
    }
}

/// Fence used around preformatted listings.
const FENCE: &str = "````";

/// Renders a table with a header separator row.
///
/// # Arguments
///
/// * `table` - Table to render.
///
/// # Returns
///
/// Markdown pipe-table text.
fn table_text(table: &Table) -> String {
    let rule = vec!["---".to_string(); table.headers.len()];
    let mut out = row_text(&table.headers) + &row_text(&rule);
    for row in &table.rows {
        out.push_str(&row_text(row));
    }
    out
}

/// Renders one table row, escaping cell content.
///
/// # Arguments
///
/// * `cells` - Cells in column order.
///
/// # Returns
///
/// One pipe-delimited Markdown line.
fn row_text(cells: &[String]) -> String {
    let escaped: Vec<String> = cells.iter().map(|c| cell_text(c)).collect();
    format!("| {} |\n", escaped.join(" | "))
}

/// Escapes text so it stays inside one table cell.
///
/// # Arguments
///
/// * `text` - Raw cell text.
///
/// # Returns
///
/// Text with pipes escaped and line breaks flattened.
fn cell_text(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the title, subtitle, and section heading layout.
    #[test]
    fn test_render_includes_title_and_heading() {
        let mut doc = Document::new("Report", "sub");
        doc.sections.push(Section::new("Intro"));
        assert_eq!(render(&doc), "# Report\n\n_sub_\n\n## Intro\n\n");
    }

    /// Verifies that tables render with a separator row.
    #[test]
    fn test_table_text_has_separator() {
        let table = Table::new(&["A"]).row(vec!["x".into()]);
        assert_eq!(table_text(&table), "| A |\n| --- |\n| x |\n");
    }

    /// Verifies that pipes and newlines cannot break a cell.
    #[test]
    fn test_cell_text_escapes_pipes_and_newlines() {
        assert_eq!(cell_text("a|b\nc"), "a\\|b c");
    }

    /// Verifies that bullets render one item per line.
    #[test]
    fn test_block_text_renders_bullets() {
        let block = Block::Bullets(vec!["a".into(), "b".into()]);
        assert_eq!(block_text(&block), "- a\n- b\n");
    }
}
