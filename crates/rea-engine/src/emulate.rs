//! Emulated Execution
//!
//! **File:** `emulate.rs`
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
use std::panic::{catch_unwind, AssertUnwindSafe};

// Import dependencies from rea_sleigh
use rea_sleigh::Emulator;

// Import crate items
use crate::{
    discover::{self, Finding},
    model::{Arch, Binary, Section},
    multiarch,
};

/// Stack gap placed above the image, in bytes.
const STACK_GAP: u64 = 0x0010_0000;

/// Largest image, in bytes of file data, that emulation will load.
const MAX_IMAGE: u64 = 16 << 20;

/// Stack registers for architectures that do not name theirs `sp`.
const STACK_REGS: [(Arch, &str); 4] = [
    (Arch::X86, "ESP"),
    (Arch::X86_64, "RSP"),
    (Arch::PowerPc, "r1"),
    (Arch::PowerPc64, "r1"),
];

/// Why an emulation run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// No SLEIGH architecture or specification was available.
    NoArch,

    /// Control flow left the mapped image, as on a return or import call.
    LeftMappedCode,

    /// The instruction budget was exhausted.
    Budget,

    /// An instruction failed to decode.
    DecodeError,

    /// The image was too large to emulate within memory limits.
    TooLarge,

    /// Emulation was aborted after an internal fault.
    Faulted,
}

/// The result of emulating from an entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emulation {
    /// Entry address the run started from.
    pub entry: u64,

    /// Number of instructions executed.
    pub steps: u64,

    /// Program counter where the run stopped.
    pub exit: u64,

    /// Why the run stopped.
    pub stop: Stop,

    /// Flag-shaped strings recovered from emulated memory.
    pub found: Vec<Finding>,
}

/// Emulates a binary from its declared entry point.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents backing the sections.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The emulation outcome.
#[must_use]
pub fn from_entry(binary: &Binary, bytes: &[u8], budget: u64) -> Emulation {
    run(binary, bytes, binary.entry, budget)
}

/// Emulates a binary from an explicit entry address.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents backing the sections.
/// * `entry` - Address to begin execution at.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The emulation outcome.
#[must_use]
pub fn run(binary: &Binary, bytes: &[u8], entry: u64, budget: u64) -> Emulation {
    let mut emu = match start(binary) {
        Ok(emu) => emu,
        Err(stop) => return halted(entry, 0, entry, stop),
    };
    prepare(&mut emu, binary, bytes, entry);
    let range = code_range(binary);
    guard(&mut emu, bytes, entry, range, budget)
}

/// Builds an emulator after the image passes its size check.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The emulator, or the stop reason that prevents emulation.
///
/// # Errors
///
/// Returns [`Stop::TooLarge`] or [`Stop::NoArch`] when emulation cannot start.
fn start(binary: &Binary) -> Result<Emulator, Stop> {
    if oversized(binary) {
        return Err(Stop::TooLarge);
    }
    build(binary).ok_or(Stop::NoArch)
}

/// Runs the emulation under a panic guard so no input can crash a report.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `bytes` - Raw file contents, scanned for runtime strings.
/// * `entry` - Entry address the run started from.
/// * `range` - Mapped `(low, high)` bounds.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The emulation outcome, or a faulted outcome if an internal panic occurred.
fn guard(
    emu: &mut Emulator,
    bytes: &[u8],
    entry: u64,
    range: (u64, u64),
    budget: u64,
) -> Emulation {
    let attempt = AssertUnwindSafe(|| execute(emu, bytes, entry, range, budget));
    catch_unwind(attempt).unwrap_or_else(|_| halted(entry, 0, entry, Stop::Faulted))
}

/// Drives the emulation and scans memory for recovered strings.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `bytes` - Raw file contents, scanned for runtime strings.
/// * `entry` - Entry address the run started from.
/// * `range` - Mapped `(low, high)` bounds.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The emulation outcome with any findings attached.
fn execute(
    emu: &mut Emulator,
    bytes: &[u8],
    entry: u64,
    range: (u64, u64),
    budget: u64,
) -> Emulation {
    let mut outcome = drive(emu, entry, range, budget);
    outcome.found = discover::scan(emu, bytes);
    outcome
}

/// Reports whether an image is too large to emulate.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// `true` when the summed section file data exceeds [`MAX_IMAGE`].
fn oversized(binary: &Binary) -> bool {
    binary.sections.iter().map(|section| section.file_size).sum::<u64>() > MAX_IMAGE
}

/// Builds an emulator for a binary's architecture.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// A ready emulator, or `None` when the architecture is unavailable.
fn build(binary: &Binary) -> Option<Emulator> {
    let arch = multiarch::emulation_arch(binary)?;
    Emulator::new(arch).ok()
}

/// Loads sections, sets the stack, and positions the program counter.
///
/// # Arguments
///
/// * `emu` - Emulator to prepare.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents backing the sections.
/// * `entry` - Address to begin execution at.
fn prepare(emu: &mut Emulator, binary: &Binary, bytes: &[u8], entry: u64) {
    load_sections(emu, binary, bytes);
    let (_, hi) = code_range(binary);
    let _ = emu.write_reg(stack_register(binary.arch), hi.saturating_add(STACK_GAP));
    emu.set_pc(entry);
}

/// Loads every section's file bytes at its virtual address.
///
/// # Arguments
///
/// * `emu` - Emulator receiving the bytes.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents backing the sections.
fn load_sections(emu: &mut Emulator, binary: &Binary, bytes: &[u8]) {
    for section in &binary.sections {
        if let Some(data) = section_bytes(section, bytes) {
            emu.load(section.address, data);
        }
    }
}

/// Slices a section's bytes from the file image.
///
/// # Arguments
///
/// * `section` - Section to slice.
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The section's file bytes, or `None` when the range is invalid.
fn section_bytes<'a>(section: &Section, bytes: &'a [u8]) -> Option<&'a [u8]> {
    let start = usize::try_from(section.file_offset).ok()?;
    let len = usize::try_from(section.file_size).ok()?;
    bytes.get(start..start.checked_add(len)?)
}

/// Computes the mapped address range of the image.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The inclusive-low, exclusive-high bounds of mapped code and data.
fn code_range(binary: &Binary) -> (u64, u64) {
    let lo = binary.sections.iter().map(|s| s.address).min();
    let hi = binary.sections.iter().map(section_end).max();
    (
        lo.unwrap_or(binary.image_base),
        hi.unwrap_or(binary.image_base),
    )
}

/// Returns the exclusive end of a section's mapped range.
///
/// # Arguments
///
/// * `section` - Section to measure.
///
/// # Returns
///
/// The address just past the section.
fn section_end(section: &Section) -> u64 {
    section.address.saturating_add(section.virtual_size.max(section.file_size))
}

/// Chooses the stack-pointer register name for an architecture.
///
/// # Arguments
///
/// * `arch` - Architecture family.
///
/// # Returns
///
/// The register name, defaulting to `sp`.
fn stack_register(arch: Arch) -> &'static str {
    STACK_REGS
        .iter()
        .find(|(family, _)| *family == arch)
        .map_or("sp", |(_, name)| *name)
}

/// Runs the step loop until control leaves the image or the budget ends.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `entry` - Entry address the run started from.
/// * `range` - Mapped `(low, high)` bounds.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The emulation outcome.
fn drive(emu: &mut Emulator, entry: u64, range: (u64, u64), budget: u64) -> Emulation {
    for steps in 0..budget {
        if let Some(stop) = tick(emu, range) {
            return halted(entry, steps + 1, emu.pc(), stop);
        }
    }
    halted(entry, budget, emu.pc(), Stop::Budget)
}

/// Executes one instruction and reports a stop reason when the run should end.
///
/// # Arguments
///
/// * `emu` - Emulator to step.
/// * `range` - Mapped `(low, high)` bounds.
///
/// # Returns
///
/// A stop reason, or `None` to continue.
fn tick(emu: &mut Emulator, range: (u64, u64)) -> Option<Stop> {
    match emu.step() {
        Ok(step) => (!in_range(step.pc, range)).then_some(Stop::LeftMappedCode),
        Err(_) => Some(Stop::DecodeError),
    }
}

/// Reports whether an address lies within the mapped image.
///
/// # Arguments
///
/// * `pc` - Program counter, possibly carrying an interworking bit.
/// * `range` - Mapped `(low, high)` bounds.
///
/// # Returns
///
/// `true` when the masked address is within the range.
fn in_range(pc: u64, range: (u64, u64)) -> bool {
    let address = pc & !1;
    address >= range.0 && address < range.1
}

/// Builds an emulation outcome.
///
/// # Arguments
///
/// * `entry` - Entry address the run started from.
/// * `steps` - Instructions executed.
/// * `exit` - Program counter where the run stopped.
/// * `stop` - Reason the run stopped.
///
/// # Returns
///
/// The outcome.
fn halted(entry: u64, steps: u64, exit: u64, stop: Stop) -> Emulation {
    Emulation { entry, steps, exit, stop, found: Vec::new() }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the stack register defaults and overrides by architecture.
    #[test]
    fn test_stack_register_selection() {
        assert_eq!(stack_register(Arch::X86_64), "RSP");
        assert_eq!(stack_register(Arch::Arm), "sp");
    }

    /// Verifies the mapped-range membership test masks the Thumb bit.
    #[test]
    fn test_in_range_masks_thumb_bit() {
        assert!(in_range(0x1001, (0x1000, 0x2000)));
        assert!(!in_range(0x2000, (0x1000, 0x2000)));
    }

    /// Verifies a binary with no mapped architecture reports `NoArch`.
    #[test]
    fn test_run_without_arch_is_noarch() {
        let binary = Binary { arch: Arch::Other, ..Binary::default() };
        let result = from_entry(&binary, b"", 16);
        assert_eq!(result.stop, Stop::NoArch);
    }

    /// Verifies an image larger than the limit is skipped, not loaded.
    #[test]
    fn test_oversized_image_is_skipped() {
        let big = Section { file_size: MAX_IMAGE + 1, ..Section::default() };
        let binary = Binary { arch: Arch::X86_64, sections: vec![big], ..Binary::default() };
        assert_eq!(from_entry(&binary, b"", 16).stop, Stop::TooLarge);
    }
}
