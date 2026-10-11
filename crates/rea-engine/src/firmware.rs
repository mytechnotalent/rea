//! Raw Cortex-M Firmware Loader
//!
//! **File:** `firmware.rs`
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
    detect::Format,
    error::EngineError,
    model::{Arch, Binary, Fact, Section, Symbol},
    numeric::to_u64,
};

/// Bytes per 32-bit vector-table entry.
const ENTRY: usize = 4;

/// Largest vector table scanned for handler entries.
const MAX_VECTORS: usize = 64;

/// Named exception vectors in their fixed table order after the reset handler.
const EXCEPTIONS: [&str; 14] = [
    "NMI_Handler",
    "HardFault_Handler",
    "MemManage_Handler",
    "BusFault_Handler",
    "UsageFault_Handler",
    "SecureFault_Handler",
    "Reserved7",
    "Reserved8",
    "Reserved9",
    "SVC_Handler",
    "DebugMon_Handler",
    "Reserved12",
    "PendSV_Handler",
    "SysTick_Handler",
];

/// Parses a raw Cortex-M firmware image into the model.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// A Thumb binary mapped at its inferred flash base.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when the vector table is unreadable.
pub fn parse(bytes: &[u8]) -> Result<Binary, EngineError> {
    let reset = reset_vector(bytes)?;
    let base = u64::from(reset & 0xFFF0_0000);
    let mut binary = identity(base, reset);
    binary.sections = vec![flash(base, bytes)];
    binary.symbols = handlers(bytes, base);
    binary.function_starts = discovered_starts(bytes, base, &binary.symbols);
    binary.facts = facts(bytes, base);
    Ok(binary)
}

/// Reads the reset-vector word from the image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// The raw reset-vector word.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] when the vector table is unreadable.
fn reset_vector(bytes: &[u8]) -> Result<u32, EngineError> {
    word(bytes, ENTRY).ok_or_else(|| EngineError::Parse("missing reset vector".into()))
}

/// Collects function-start candidates for firmware discovery.
///
/// Vector handlers, code pointers held in literal pools and data, and Thumb
/// prologues together recover functions reached only through registers.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `base` - Flash base address.
/// * `handlers` - Named vector-table handlers.
///
/// # Returns
///
/// Deduplicated candidate entry addresses.
fn discovered_starts(bytes: &[u8], base: u64, handlers: &[Symbol]) -> Vec<u64> {
    let mut starts: Vec<u64> = handlers.iter().map(|s| s.address).collect();
    starts.extend(pointer_scan(bytes, base));
    starts.extend(prologue_scan(bytes, base));
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Scans the image for stored Thumb code pointers.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `base` - Flash base address.
///
/// # Returns
///
/// Aligned addresses of every odd word pointing into the image.
fn pointer_scan(bytes: &[u8], base: u64) -> Vec<u64> {
    let end = base + to_u64(bytes.len());
    let words = (0..bytes.len() / ENTRY).filter_map(|i| word(bytes, i * ENTRY));
    let pointers = words.filter(|w| w & 1 == 1).map(|w| u64::from(w & !1));
    pointers.filter(|a| *a >= base && *a < end).collect()
}

/// Scans executable bytes for common Thumb function prologues.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `base` - Flash base address.
///
/// # Returns
///
/// Addresses of `push {..., lr}` half-words at even offsets.
fn prologue_scan(bytes: &[u8], base: u64) -> Vec<u64> {
    let halfwords = (0..bytes.len() / 2).map(|i| (i * 2, bytes[i * 2 + 1]));
    let prologues = halfwords.filter(|(_, high)| *high == 0xb5);
    prologues.map(|(offset, _)| base + to_u64(offset)).collect()
}

/// Builds the identity fields for a firmware image.
///
/// # Arguments
///
/// * `base` - Inferred flash base address.
/// * `reset` - Raw reset-vector word.
///
/// # Returns
///
/// A 32-bit Thumb ARM binary.
fn identity(base: u64, reset: u32) -> Binary {
    let mut binary = Binary::new(Format::CortexM, Arch::Arm, "ARM Cortex-M (Thumb)", 32);
    binary.thumb = true;
    binary.image_base = base;
    binary.entry = u64::from(reset & !1);
    binary.kind = "Bare-metal firmware image".to_string();
    binary
}

/// Builds the single executable section covering the whole image.
///
/// # Arguments
///
/// * `base` - Flash base address.
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// A read-execute section mapped at the base.
fn flash(base: u64, bytes: &[u8]) -> Section {
    let size = to_u64(bytes.len());
    let mapped = Section::mapped(".flash", base, size);
    mapped.with_file(0, size).with_access(true, false)
}

/// Reads handler addresses from the vector table as symbols.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `base` - Flash base address.
///
/// # Returns
///
/// Named reset and exception handlers plus numbered interrupt handlers.
fn handlers(bytes: &[u8], base: u64) -> Vec<Symbol> {
    let entries = vector_entries(bytes);
    let named = entries
        .iter()
        .enumerate()
        .filter_map(|(index, value)| handler(index, *value, base));
    named.collect()
}

/// Reads the raw vector-table words that point into the image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
///
/// # Returns
///
/// Vector words from index 1 onward, capped at [`MAX_VECTORS`].
fn vector_entries(bytes: &[u8]) -> Vec<u32> {
    let count = (bytes.len() / ENTRY).min(MAX_VECTORS);
    let offsets = (1..count).map(|index| index * ENTRY);
    offsets.filter_map(|offset| word(bytes, offset)).collect()
}

/// Converts one vector entry into a named handler symbol.
///
/// # Arguments
///
/// * `index` - Zero-based index after the stack pointer.
/// * `value` - Raw vector word.
/// * `base` - Flash base address.
///
/// # Returns
///
/// A symbol when the entry is a Thumb pointer into flash.
fn handler(index: usize, value: u32, base: u64) -> Option<Symbol> {
    let address = u64::from(value & !1);
    let inside = value & 1 == 1 && address >= base;
    inside.then(|| Symbol::new(&handler_name(index), address))
}

/// Names a handler by its vector-table index.
///
/// # Arguments
///
/// * `index` - Zero-based index after the stack pointer.
///
/// # Returns
///
/// The reset, a named exception, or a numbered interrupt handler name.
fn handler_name(index: usize) -> String {
    match index {
        0 => "Reset_Handler".to_string(),
        n if n <= EXCEPTIONS.len() => EXCEPTIONS[n - 1].to_string(),
        n => format!("IRQ{}_Handler", n - 1 - EXCEPTIONS.len()),
    }
}

/// Builds metadata facts for a firmware image.
///
/// # Arguments
///
/// * `bytes` - Whole firmware image.
/// * `base` - Flash base address.
///
/// # Returns
///
/// The inferred base, stack pointer, and handler count.
fn facts(bytes: &[u8], base: u64) -> Vec<Fact> {
    let sp = word(bytes, 0).unwrap_or(0);
    let count = handlers(bytes, base).len();
    vec![
        Fact::new("Inferred flash base", format!("0x{base:08x}")),
        Fact::new("Initial stack pointer", format!("0x{sp:08x}")),
        Fact::new("Vector-table handlers", count.to_string()),
    ]
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
    let end = offset.checked_add(ENTRY)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a minimal vector table: SP, reset, and one IRQ.
    ///
    /// # Returns
    ///
    /// Firmware bytes based at 0x10000000.
    fn image() -> Vec<u8> {
        let words = [0x2008_2000u32, 0x1000_0101, 0x1000_0201, 0, 0];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Verifies that the base, entry, and reset symbol are recovered.
    #[test]
    fn test_parse_recovers_entry_and_reset() {
        let binary = parse(&image()).unwrap();
        assert_eq!(binary.image_base, 0x1000_0000);
        assert_eq!(binary.entry, 0x1000_0100);
        assert!(binary.symbols.iter().any(|s| s.name == "Reset_Handler"));
    }

    /// Verifies that handler names follow the exception table order.
    #[test]
    fn test_handler_name_order() {
        assert_eq!(handler_name(0), "Reset_Handler");
        assert_eq!(handler_name(1), "NMI_Handler");
        assert_eq!(handler_name(15), "IRQ0_Handler");
    }

    /// Verifies that even (ARM-mode) vector words are rejected as handlers.
    #[test]
    fn test_handler_rejects_even() {
        assert!(handler(0, 0x1000_0100, 0x1000_0000).is_none());
        assert!(handler(0, 0x1000_0101, 0x1000_0000).is_some());
    }
}
