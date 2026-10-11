//! Emulator Errors
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

/// Errors raised while emulating p-code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmuError {
    /// The bytes at an address did not decode to an instruction.
    Decode(u64),

    /// The step budget was exhausted before execution halted.
    Budget(u64),

    /// A named register was not present in the specification.
    MissingRegister(String),
}

impl fmt::Display for EmuError {
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
            Self::Decode(address) => write!(f, "could not decode instruction at {address:#x}"),
            Self::Budget(limit) => write!(f, "emulation exceeded its {limit}-step budget"),
            Self::MissingRegister(name) => write!(f, "specification has no register: {name}"),
        }
    }
}

impl std::error::Error for EmuError {}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that each variant renders a readable message.
    #[test]
    fn test_display_messages() {
        let decode = EmuError::Decode(0x1000).to_string();
        let budget = EmuError::Budget(64).to_string();
        let register = EmuError::MissingRegister("pc".into()).to_string();
        assert!(decode.contains("decode") && budget.contains("budget"));
        assert!(register.contains("register"));
    }
}
