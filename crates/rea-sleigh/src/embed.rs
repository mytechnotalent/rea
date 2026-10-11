//! Embedded Specifications
//!
//! **File:** `embed.rs`
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

// Import dependencies from include_dir
use include_dir::{include_dir, Dir, DirEntry};

// Import dependencies from sleigh
use sleigh::{FileId, SourceDb};

/// Every embedded SLEIGH specification file, keyed by family-relative path.
static SPECS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/processors");

/// Builds a source database holding every embedded specification file.
///
/// # Returns
///
/// A database whose include paths resolve entirely in memory.
#[must_use]
pub fn source_db() -> SourceDb {
    let mut db = SourceDb::new();
    add_dir(&SPECS, &mut db);
    db
}

/// Adds every file in a directory tree to the database.
///
/// # Arguments
///
/// * `dir` - Embedded directory to add.
/// * `db` - Database receiving the files.
fn add_dir(dir: &Dir, db: &mut SourceDb) {
    for entry in dir.entries() {
        match entry {
            DirEntry::Dir(child) => add_dir(child, db),
            DirEntry::File(file) => add_file(file, db),
        }
    }
}

/// Adds one embedded text file to the database under its path.
///
/// # Arguments
///
/// * `file` - Embedded file to add.
/// * `db` - Database receiving the file.
fn add_file(file: &include_dir::File, db: &mut SourceDb) {
    if let Some(text) = file.contents_utf8() {
        db.add_file(file.path().to_path_buf(), text.to_string());
    }
}

/// Finds the file identifier of a specification by its path.
///
/// # Arguments
///
/// * `db` - Database already populated by [`source_db`].
/// * `path` - Family-relative path, such as `ARM/ARM7_le.slaspec`.
///
/// # Returns
///
/// The identifier when the path is present.
#[must_use]
pub fn file_id(db: &SourceDb, path: &str) -> Option<FileId> {
    db.file_by_path(std::path::Path::new(path))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that the ARM specification is embedded and findable.
    #[test]
    fn test_source_db_contains_arm() {
        let db = source_db();
        assert!(file_id(&db, "ARM/ARM7_le.slaspec").is_some());
    }

    /// Verifies that an unknown path is reported as absent.
    #[test]
    fn test_file_id_missing() {
        let db = source_db();
        assert!(file_id(&db, "nope/none.slaspec").is_none());
    }
}
