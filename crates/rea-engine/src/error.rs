//! Engine Errors
//!
//! **File:** `error.rs`
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
use std::fmt;

/// Errors returned by the analysis engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The target could not be read.
    Io(String),

    /// The target exceeds the maximum supported size in bytes.
    TooLarge(u64),

    /// The target is not a well-formed executable of its detected format.
    Parse(String),
}

impl fmt::Display for EngineError {
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
        match self {
            Self::Io(message) => write!(f, "cannot read target: {message}"),
            Self::TooLarge(size) => write!(f, "target is too large: {size} bytes"),
            Self::Parse(message) => write!(f, "malformed executable: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<goblin::error::Error> for EngineError {
    /// Wraps a container parsing failure.
    ///
    /// # Arguments
    ///
    /// * `error` - Parser error.
    ///
    /// # Returns
    ///
    /// [`EngineError::Parse`] carrying the parser's message.
    fn from(error: goblin::error::Error) -> Self {
        Self::Parse(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that every variant renders a readable message.
    #[test]
    fn test_display_messages() {
        let too_large = EngineError::TooLarge(9).to_string();
        let io = EngineError::Io("x".into()).to_string();
        let parse = EngineError::Parse("y".into()).to_string();
        assert_eq!(too_large, "target is too large: 9 bytes");
        assert!(io.contains("cannot read") && parse.contains("malformed"));
    }
}
