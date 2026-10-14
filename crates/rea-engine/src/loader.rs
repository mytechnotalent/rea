//! Executable Format Dispatch
//!
//! **File:** `loader.rs`
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

// Import crate items
use crate::{
    armfw, detect::Format, elf, error::EngineError, firmware, macho, model::Binary, pe, uimage,
};

/// Dispatches raw or container firmware loaders.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `format` - Detected format.
///
/// # Returns
///
/// The parsed binary if the format is supported firmware, otherwise `None`.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] if firmware structure is malformed.
fn load_firmware(bytes: &[u8], format: Format) -> Option<Result<Binary, EngineError>> {
    match format {
        Format::CortexM => Some(firmware::parse(bytes)),
        Format::ArmFirmware => Some(armfw::parse(bytes)),
        Format::UImage => Some(uimage::parse(bytes)),
        _ => None,
    }
}

/// Parses a native executable with the loader for its format.
///
/// # Arguments
///
/// * `bytes` - Whole target file.
/// * `format` - Detected format.
///
/// # Returns
///
/// The parsed binary, or `None` for non-native formats.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when a native image is malformed.
pub fn load(bytes: &[u8], format: Format) -> Option<Result<Binary, EngineError>> {
    match format {
        Format::Pe => Some(pe::parse(bytes)),
        Format::Elf => Some(elf::parse(bytes)),
        Format::MachO | Format::MachOUniversal => Some(macho::parse(bytes)),
        _ => load_firmware(bytes, format),
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that non-native formats are not loaded.
    #[test]
    fn test_load_skips_non_native() {
        assert!(load(b"PK\x03\x04", Format::Zip).is_none());
    }

    /// Verifies that malformed native images report errors.
    #[test]
    fn test_load_reports_malformed() {
        assert!(matches!(load(b"MZ", Format::Pe), Some(Err(_))));
    }

    /// Verifies that uimage format dispatches to the uimage loader.
    #[test]
    fn test_load_dispatches_uimage() {
        assert!(load(b"too short", Format::UImage).is_some());
    }
}
