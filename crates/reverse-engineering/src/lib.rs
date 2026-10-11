//! Reverse Engineering Assistant Framework
//!
//! **File:** `lib.rs`
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

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

// Import dependencies from std
use std::{path::Path, process::ExitCode};

// Re-export core engine and report crates
pub use rea;
pub use rea_engine as engine;
pub use rea_report as report;

// Re-export essential engine types
pub use rea_engine::{
    analyze, analyze_path, Analysis, AsarTriage, EngineError, FirmwareTriage, Format,
};

/// Analyzes a target binary or firmware image at the given path.
///
/// # Arguments
///
/// * `path` - Filesystem path to target binary.
///
/// # Returns
///
/// The complete [`Analysis`] record.
///
/// # Errors
///
/// Returns [`EngineError`] if reading or parsing fails.
pub fn inspect(path: &Path) -> Result<Analysis, EngineError> {
    analyze_path(path)
}

/// Starts the Model Context Protocol (MCP) server over standard I/O.
///
/// # Returns
///
/// Exit code.
#[must_use]
pub fn serve_mcp() -> ExitCode {
    rea::mcp::serve_cmd()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Tests format identification through re-export.
    #[test]
    fn test_reexport_format() {
        assert_eq!(Format::Pe.label(), "Portable Executable (PE)");
        assert_eq!(Format::Asar.label(), "Electron Archive (ASAR)");
    }
}
