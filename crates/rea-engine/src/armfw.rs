//! Raw ARM Firmware Loader
//!
//! **File:** `armfw.rs`
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
    detect::{self, Format},
    error::EngineError,
    model::{Arch, Binary, Fact, Section, Symbol},
    numeric::to_u64,
};

/// Bytes per ARM instruction.
const WORD: usize = 4;

/// Most recovered call targets retained.
const MAX_CALLS: usize = 4096;

/// Parses a raw ARM-mode firmware image into the model.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// A 32-bit ARM binary mapped at address zero.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when no boot preamble is present.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    let entry =
        detect::arm_entry(bytes).ok_or_else(|| EngineError::Parse("no ARM boot code".into()))?;
    let mut binary = identity();
    populate(bytes, entry, &mut binary);
    Ok(binary)
}

/// Builds the base identity for a raw ARM image.
///
/// # Returns
///
/// A 32-bit ARM-mode binary mapped at address zero.
fn identity() -> Binary {
    let mut binary = Binary::new(Format::ArmFirmware, Arch::Arm, "ARM (32-bit, ARM mode)", 32);
    binary.thumb = false;
    binary.image_base = 0;
    binary.kind = "Raw ARM firmware image".to_string();
    binary
}

/// Fills the recovered structure into the binary.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `entry` - Byte offset of the boot code.
/// * `binary` - Binary to populate.
fn populate(bytes: &[u8], entry: usize, binary: &mut Binary) {
    binary.entry = to_u64(entry);
    binary.sections = vec![image_section(bytes)];
    binary.symbols = symbols(entry);
    binary.function_starts = function_starts(bytes, entry);
    binary.facts = facts(bytes, entry);
}

/// Builds the single executable section covering the whole image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// A read-execute section mapped at address zero.
fn image_section(bytes: &[u8]) -> Section {
    let size = to_u64(bytes.len());
    let mapped = Section::mapped(".text", 0, size);
    mapped.with_file(0, size).with_access(true, false)
}

/// Builds symbols for the recovered entry point.
///
/// # Arguments
///
/// * `entry` - Byte offset of the boot code.
///
/// # Returns
///
/// The entry symbol.
fn symbols(entry: usize) -> Vec<Symbol> {
    vec![Symbol::new("entry", to_u64(entry))]
}

/// Collects function-start candidates.
///
/// The entry point and every direct `bl` target inside the image seed the
/// recursive-descent walk.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `entry` - Byte offset of the boot code.
///
/// # Returns
///
/// Deduplicated candidate entry addresses.
fn function_starts(bytes: &[u8], entry: usize) -> Vec<u64> {
    let mut starts = vec![to_u64(entry)];
    starts.extend(bl_targets(bytes));
    starts.sort_unstable();
    starts.dedup();
    starts.truncate(MAX_CALLS);
    starts
}

/// Scans the image for direct `bl` call targets.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// In-range targets of every `bl` instruction.
fn bl_targets(bytes: &[u8]) -> Vec<u64> {
    let end = to_u64(bytes.len());
    (0..bytes.len() / WORD).filter_map(|i| scan_bl(bytes, i, end)).collect()
}

/// Resolves one candidate instruction's `bl` target.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `index` - Word index.
/// * `end` - Exclusive image end address.
///
/// # Returns
///
/// The in-range target, or `None`.
fn scan_bl(bytes: &[u8], index: usize, end: u64) -> Option<u64> {
    let pc = to_u64(index * WORD);
    let target = bl_target(word(bytes, index * WORD)?, pc)?;
    (target < end).then_some(target)
}

/// Resolves the target of an ARM `bl` instruction.
///
/// # Arguments
///
/// * `word` - Instruction word.
/// * `pc` - Address of the instruction.
///
/// # Returns
///
/// The branch target, or `None` when the word is not a `bl`.
fn bl_target(word: u32, pc: u64) -> Option<u64> {
    if word & 0xFF00_0000 != 0xEB00_0000 {
        return None;
    }
    let imm = (word & 0x00FF_FFFF) << 2;
    let signed = i64::from(i32::from_ne_bytes((imm << 4).to_ne_bytes()) >> 4);
    Some(pc.wrapping_add(8).wrapping_add(u64::from_ne_bytes(signed.to_ne_bytes())))
}

/// Builds metadata facts for the image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `entry` - Byte offset of the boot code.
/// * `starts` - Recovered function starts.
///
/// # Returns
///
/// Identity and layout facts.
fn facts(bytes: &[u8], entry: usize) -> Vec<Fact> {
    let f1 = Fact::new("Image base", "0x00000000");
    let f2 = Fact::new("Entry offset", format!("0x{entry:08x}"));
    let f3 = Fact::new("Code size", format!("{} bytes", bytes.len()));
    let f4 = Fact::new(
        "Recovered entry points",
        function_starts(bytes, entry).len().to_string(),
    );
    vec![f1, f2, f3, f4]
}

/// Reads a little-endian 32-bit word.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Byte offset of the word.
///
/// # Returns
///
/// The word, when four bytes are available.
fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(WORD)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// ARM boot preamble words used by tests.
    const SAMPLE_WORDS: [u32; 7] = [
        0xE3A0_00D1,
        0xE121_F000,
        0xE59F_D0D4,
        0xE3A0_00D2,
        0xE121_F000,
        0xE59F_D0CC,
        0xEB00_0004,
    ];

    /// Builds a small ARM boot preamble with a `bl` call.
    ///
    /// # Returns
    ///
    /// Firmware bytes beginning with `mov r0`/`msr` pairs.
    fn image() -> Vec<u8> {
        let mut words = SAMPLE_WORDS.to_vec();
        words.resize(20, 0);
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Verifies boot-preamble detection.
    #[test]
    fn test_arm_entry() {
        assert_eq!(detect::arm_entry(&image()), Some(0));
    }

    /// Verifies entry and call-target recovery.
    #[test]
    fn test_parse_recovers_entry_and_calls() {
        let binary = parse(&image()).unwrap();
        assert_eq!(binary.arch, Arch::Arm);
        assert_eq!(binary.entry, 0);
        assert_eq!(binary.function_starts.len(), 2);
    }

    /// Verifies `bl` target resolution.
    #[test]
    fn test_bl_target() {
        assert_eq!(bl_target(0xEB00_0004, 0), Some(0x18));
        assert_eq!(bl_target(0xE3A0_0000, 0), None);
    }
}
