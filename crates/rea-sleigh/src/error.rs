//! SLEIGH Errors
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

/// Errors raised while loading a specification or decoding an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SleighError {
    /// A required specification file was missing from the embedded set.
    MissingSpec(String),

    /// The specification failed to compile.
    Compile(String),

    /// A context field named by an architecture does not exist.
    MissingField(String),
}

impl fmt::Display for SleighError {
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
            Self::MissingSpec(path) => write!(f, "embedded specification not found: {path}"),
            Self::Compile(message) => write!(f, "specification failed to compile: {message}"),
            Self::MissingField(name) => write!(f, "specification has no context field: {name}"),
        }
    }
}

impl std::error::Error for SleighError {}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that each variant renders a readable message.
    #[test]
    fn test_display_messages() {
        let missing = SleighError::MissingSpec("x".into()).to_string();
        let compile = SleighError::Compile("y".into()).to_string();
        let field = SleighError::MissingField("z".into()).to_string();
        assert!(missing.contains("not found") && compile.contains("compile"));
        assert!(field.contains("context field"));
    }
}
