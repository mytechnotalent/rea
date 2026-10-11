//! PDF Report Compiler
//!
//! **File:** `pdf.rs`
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
use std::{fmt, path::PathBuf};

// Import dependencies from typst
use typst::{
    diag::{FileError, FileResult, SourceDiagnostic},
    ecow::EcoVec,
    foundations::{Bytes, Datetime, Duration},
    syntax::{FileId, RootedPath, Source, VirtualPath, VirtualRoot},
    text::{Font, FontBook},
    utils::LazyHash,
    Library, LibraryExt, World,
};

// Import dependencies from typst_layout
use typst_layout::PagedDocument;

// Import dependencies from typst_pdf
use typst_pdf::PdfOptions;

/// Errors raised while turning markup into a PDF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportError {
    /// Typst rejected the markup or failed to lay it out.
    Typeset(String),
}

impl fmt::Display for ReportError {
    /// Formats the error for terminal output.
    ///
    /// # Arguments
    ///
    /// * `f` - Destination formatter.
    ///
    /// # Errors
    ///
    /// Returns an error when the formatter fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Typeset(message) = self;
        write!(f, "PDF typesetting failed: {message}")
    }
}

impl std::error::Error for ReportError {}

/// Compiles Typst markup into PDF bytes using only embedded fonts.
///
/// # Arguments
///
/// * `markup` - Complete Typst source.
///
/// # Returns
///
/// The encoded PDF file.
///
/// # Errors
///
/// Returns [`ReportError::Typeset`] when compilation or export fails.
pub fn compile(markup: &str) -> Result<Vec<u8>, ReportError> {
    let world = ReportWorld::new(markup);
    let warned = typst::compile::<PagedDocument>(&world);
    let document = warned.output.map_err(|d| typeset_error(&d))?;
    typst_pdf::pdf(&document, &PdfOptions::default()).map_err(|d| typeset_error(&d))
}

/// Joins Typst diagnostics into one error.
///
/// # Arguments
///
/// * `diagnostics` - Diagnostics reported by Typst.
///
/// # Returns
///
/// A [`ReportError::Typeset`] listing every message.
fn typeset_error(diagnostics: &EcoVec<SourceDiagnostic>) -> ReportError {
    let messages: Vec<&str> = diagnostics.iter().map(|d| d.message.as_str()).collect();
    ReportError::Typeset(messages.join("; "))
}

/// In-memory Typst world holding one source file and the bundled fonts.
struct ReportWorld {
    /// Standard library used for evaluation.
    library: LazyHash<Library>,

    /// Metadata describing every available font.
    book: LazyHash<FontBook>,

    /// Loaded font faces indexed like `book`.
    fonts: Vec<Font>,

    /// The single main source file.
    source: Source,
}

impl ReportWorld {
    /// Creates a world whose main file holds the given markup.
    ///
    /// # Arguments
    ///
    /// * `markup` - Typst source text.
    ///
    /// # Returns
    ///
    /// A world ready for compilation.
    fn new(markup: &str) -> Self {
        let fonts = bundled_fonts();
        Self {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(FontBook::from_fonts(&fonts)),
            fonts,
            source: Source::new(main_id(), markup.to_string()),
        }
    }
}

/// Loads every font face embedded in the binary.
///
/// # Returns
///
/// All bundled font faces.
fn bundled_fonts() -> Vec<Font> {
    typst_assets::fonts().flat_map(|data| Font::iter(Bytes::new(data))).collect()
}

/// Builds the file identifier of the main report source.
///
/// # Returns
///
/// The identifier of `/report.typ` in the project root.
///
/// # Panics
///
/// Never in practice; the constant path is always valid.
fn main_id() -> FileId {
    let path = VirtualPath::new("report.typ").expect("constant path is valid");
    FileId::new(RootedPath::new(VirtualRoot::Project, path))
}

impl World for ReportWorld {
    /// Returns the standard library.
    ///
    /// # Returns
    ///
    /// The shared library.
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    /// Returns the font metadata book.
    ///
    /// # Returns
    ///
    /// The shared font book.
    fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    /// Returns the identifier of the main file.
    ///
    /// # Returns
    ///
    /// The report source identifier.
    fn main(&self) -> FileId {
        self.source.id()
    }

    /// Looks up a source file; only the report itself exists.
    ///
    /// # Arguments
    ///
    /// * `id` - Requested file.
    ///
    /// # Errors
    ///
    /// Returns [`FileError::NotFound`] for any other file.
    fn source(&self, id: FileId) -> FileResult<Source> {
        let found = (id == self.source.id()).then(|| self.source.clone());
        found.ok_or_else(|| FileError::NotFound(PathBuf::new()))
    }

    /// Refuses all binary file access; reports never read files.
    ///
    /// # Arguments
    ///
    /// * `_id` - Requested file.
    ///
    /// # Errors
    ///
    /// Always returns [`FileError::AccessDenied`].
    fn file(&self, _id: FileId) -> FileResult<Bytes> {
        Err(FileError::AccessDenied)
    }

    /// Returns a font face by index.
    ///
    /// # Arguments
    ///
    /// * `index` - Index into the font book.
    ///
    /// # Returns
    ///
    /// The face, when the index is valid.
    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.get(index).cloned()
    }

    /// Reports no current date, keeping output reproducible.
    ///
    /// # Arguments
    ///
    /// * `_offset` - Requested UTC offset.
    ///
    /// # Returns
    ///
    /// Always `None`.
    fn today(&self, _offset: Option<Duration>) -> Option<Datetime> {
        None
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that valid markup compiles to a PDF file.
    #[test]
    fn test_compile_produces_pdf() {
        let pdf = compile("= Hello\nWorld").unwrap();
        assert!(pdf.starts_with(b"%PDF"));
    }

    /// Verifies that invalid markup reports a typesetting error.
    #[test]
    fn test_compile_rejects_invalid_markup() {
        assert!(matches!(
            compile("#undefined_name"),
            Err(ReportError::Typeset(_))
        ));
    }

    /// Verifies that fonts are embedded and loadable.
    #[test]
    fn test_bundled_fonts_are_present() {
        assert!(!bundled_fonts().is_empty());
    }

    /// Verifies that a full report document compiles.
    #[test]
    fn test_compile_full_report_markup() {
        let doc = crate::Document::new("Sample", "subtitle");
        assert!(compile(&crate::to_markup(&doc)).is_ok());
    }
}
