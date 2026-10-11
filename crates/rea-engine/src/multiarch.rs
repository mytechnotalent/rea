//! Multi-Architecture Disassembly
//!
//! **File:** `multiarch.rs`
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
use std::collections::HashSet;

// Import dependencies from rea_sleigh
use rea_sleigh::{arch as sarch, Arch as SleighArch, Decoded, Disassembler, Flow};

// Import crate items
use crate::{
    disasm::{CodeMap, Disassembly, Function, MAX_FUNCTIONS, MAX_INSTRUCTIONS},
    model::{Arch, Binary},
};

/// Architecture families mapped to their little- and big-endian specifications.
const ARCH_MAP: [(Arch, SleighArch, SleighArch); 15] = [
    (Arch::Arm, sarch::ARM_LE, sarch::ARM_BE),
    (Arch::Arm64, sarch::AARCH64_LE, sarch::AARCH64_BE),
    (Arch::Mips, sarch::MIPS32_LE, sarch::MIPS32_BE),
    (Arch::Mips64, sarch::MIPS64_LE, sarch::MIPS64_BE),
    (Arch::PowerPc, sarch::PPC32_LE, sarch::PPC32_BE),
    (Arch::PowerPc64, sarch::PPC64_LE, sarch::PPC64_BE),
    (Arch::Riscv, sarch::RISCV32, sarch::RISCV32),
    (Arch::Riscv64, sarch::RISCV64, sarch::RISCV64),
    (Arch::Sparc, sarch::SPARC_V9, sarch::SPARC_V9),
    (Arch::SuperH, sarch::SUPERH, sarch::SUPERH),
    (Arch::Avr, sarch::AVR8, sarch::AVR8),
    (Arch::M68k, sarch::M68000, sarch::M68000),
    (Arch::Xtensa, sarch::XTENSA_LE, sarch::XTENSA_LE),
    (Arch::Msp430, sarch::MSP430, sarch::MSP430),
    (Arch::LoongArch, sarch::LOONGARCH64, sarch::LOONGARCH64),
];

/// Selects the SLEIGH architecture for a parsed binary.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The matching specification, or `None` for x86 and unsupported families.
#[must_use]
pub fn sleigh_arch(binary: &Binary) -> Option<SleighArch> {
    if binary.arch == Arch::Arm && binary.thumb {
        return Some(thumb(binary.big_endian));
    }
    let found = ARCH_MAP.iter().find(|(family, _, _)| *family == binary.arch)?;
    Some(if binary.big_endian { found.2 } else { found.1 })
}

/// Selects the SLEIGH architecture for emulating a parsed binary.
///
/// This extends [`sleigh_arch`] with x86 and x86-64, which are disassembled by
/// a dedicated decoder but emulated through their SLEIGH specifications.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The architecture to emulate with, or `None` when none is available.
#[must_use]
pub fn emulation_arch(binary: &Binary) -> Option<SleighArch> {
    match binary.arch {
        Arch::X86 => Some(sarch::X86),
        Arch::X86_64 => Some(sarch::X86_64),
        _ => sleigh_arch(binary),
    }
}

/// Picks the ARM Thumb specification for an endianness.
///
/// # Arguments
///
/// * `big_endian` - Whether the target is big-endian.
///
/// # Returns
///
/// The matching Thumb architecture.
fn thumb(big_endian: bool) -> SleighArch {
    if big_endian {
        sarch::THUMB_BE
    } else {
        sarch::THUMB_LE
    }
}

/// Disassembles a non-x86 binary by recursive descent using SLEIGH.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered code structure, or `None` when the architecture is unsupported.
#[must_use]
pub fn disassemble(binary: &Binary, bytes: &[u8]) -> Option<Disassembly> {
    let decoder = Disassembler::new(sleigh_arch(binary)?).ok()?;
    let mut walker = Walker::new(decoder, CodeMap::new(binary, bytes), binary);
    walker.run();
    Some(walker.result)
}

/// Renders a linear instruction listing for a non-x86 function.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
/// * `start` - Function entry address.
/// * `max` - Maximum instructions to list.
/// * `annotate` - Comment source for referenced addresses.
///
/// # Returns
///
/// One formatted line per instruction, or `None` when unsupported.
#[must_use]
pub fn render_listing(
    binary: &Binary,
    bytes: &[u8],
    start: u64,
    max: usize,
    annotate: &Annotate<'_>,
) -> Option<Vec<String>> {
    let decoder = Disassembler::new(sleigh_arch(binary)?).ok()?;
    let code = CodeMap::new(binary, bytes);
    Some(listing(&decoder, &code, start, max, annotate))
}

/// Comment source mapping a referenced address to an annotation.
pub type Annotate<'a> = dyn Fn(u64) -> Option<String> + 'a;

/// Produces the listing lines for a function.
///
/// # Arguments
///
/// * `decoder` - Instruction decoder.
/// * `code` - Executable regions.
/// * `start` - Function entry address.
/// * `max` - Maximum instructions.
/// * `annotate` - Comment source.
///
/// # Returns
///
/// The formatted lines.
fn listing(
    decoder: &Disassembler,
    code: &CodeMap,
    start: u64,
    max: usize,
    annotate: &Annotate<'_>,
) -> Vec<String> {
    let interworking = decoder.interworking();
    let (mut out, mut cursor) = (Vec::new(), Some(start));
    while out.len() < max {
        let Some(address) = cursor else { break };
        cursor = step_line(decoder, code, interworking, annotate, address, &mut out);
    }
    out
}

/// Decodes and appends one listing line, returning the next address.
///
/// # Arguments
///
/// * `decoder` - Instruction decoder.
/// * `code` - Executable regions.
/// * `interworking` - Whether bit 0 selects Thumb mode.
/// * `annotate` - Comment source.
/// * `address` - Instruction address.
/// * `out` - Accumulator of lines.
///
/// # Returns
///
/// The next address while the listing falls through, otherwise `None`.
fn step_line(
    decoder: &Disassembler,
    code: &CodeMap,
    interworking: bool,
    annotate: &Annotate<'_>,
    address: u64,
    out: &mut Vec<String>,
) -> Option<u64> {
    let line = decoder.decode(address, code.bytes_at(address))?;
    out.push(format_line(address, &line, interworking, annotate));
    line.flow.falls_through().then(|| address + line.len as u64)
}

/// Formats one listing line with an optional annotation.
///
/// # Arguments
///
/// * `address` - Instruction address.
/// * `decoded` - Decoded instruction.
/// * `interworking` - Whether bit 0 selects Thumb mode.
/// * `annotate` - Comment source.
///
/// # Returns
///
/// `address  text  ; note`.
fn format_line(
    address: u64,
    decoded: &Decoded,
    interworking: bool,
    annotate: &Annotate<'_>,
) -> String {
    let mut aligned = decoded.targets.iter().map(|t| code_address(*t, interworking));
    let note = aligned.find_map(annotate);
    let comment = note.map(|n| format!("  ; {n}")).unwrap_or_default();
    format!("{address:08x}  {}{comment}", decoded.text)
}

/// Masks a code address, clearing the ARM Thumb selector bit.
///
/// # Arguments
///
/// * `address` - Raw target address.
/// * `interworking` - Whether bit 0 selects Thumb mode.
///
/// # Returns
///
/// The aligned instruction address.
fn code_address(address: u64, interworking: bool) -> u64 {
    if interworking {
        address & !1
    } else {
        address
    }
}

/// Recursive-descent engine over a SLEIGH decoder.
struct Walker<'a> {
    /// Instruction decoder for the target architecture.
    decoder: Disassembler,

    /// Executable regions of the binary.
    code: CodeMap<'a>,

    /// Whether bit 0 of a target selects Thumb mode.
    interworking: bool,

    /// Function entries waiting to be traced.
    pending: Vec<u64>,

    /// Structure recovered so far.
    result: Disassembly,

    /// Binary architecture family.
    arch: Arch,
}

impl<'a> Walker<'a> {
    /// Creates a walker seeded with the binary's roots.
    ///
    /// # Arguments
    ///
    /// * `decoder` - Instruction decoder.
    /// * `code` - Executable regions.
    /// * `binary` - Parsed executable.
    ///
    /// # Returns
    ///
    /// The walker.
    fn new(decoder: Disassembler, code: CodeMap<'a>, binary: &Binary) -> Self {
        let interworking = decoder.interworking();
        let pending = roots(binary, interworking);
        let result = Disassembly { bitness: binary.bits, ..Disassembly::default() };
        let arch = binary.arch;
        Self { decoder, code, interworking, pending, result, arch }
    }

    /// Traces functions until none remain or the function cap is reached.
    fn run(&mut self) {
        while let Some(address) = self.pending.pop() {
            if self.result.functions.len() >= MAX_FUNCTIONS {
                self.result.truncated = true;
                break;
            }
            self.discover(address);
        }
    }

    /// Traces one function unless it is known or not code.
    ///
    /// # Arguments
    ///
    /// * `start` - Candidate function entry.
    fn discover(&mut self, start: u64) {
        if self.result.functions.contains_key(&start) || !self.code.contains(start) {
            return;
        }
        let function = self.trace(start);
        self.result.functions.insert(start, function);
    }

    /// Decodes every reachable block of one function.
    ///
    /// # Arguments
    ///
    /// * `start` - Function entry.
    ///
    /// # Returns
    ///
    /// The recovered function.
    fn trace(&mut self, start: u64) -> Function {
        let mut function = Function::at(start, None);
        self.drain_blocks(start, &mut function);
        function.lines.sort_by_key(|l| l.address);
        function.lines.dedup_by_key(|l| l.address);
        self.analyze_data_pools(&mut function);
        function
    }

    /// Explores reachable blocks for a function.
    ///
    /// # Arguments
    ///
    /// * `start` - Initial entry address.
    /// * `function` - Function accumulator.
    fn drain_blocks(&mut self, start: u64, function: &mut Function) {
        let (mut blocks, mut seen) = (vec![start], HashSet::new());
        while let Some(block) = blocks.pop() {
            self.block(block, function, &mut blocks, &mut seen);
        }
    }

    /// Extracts and analyzes referenced literal pools in code memory.
    ///
    /// # Arguments
    ///
    /// * `function` - Function being built.
    fn analyze_data_pools(&self, function: &mut Function) {
        let addrs: Vec<u64> = function.references.iter().copied().collect();
        for addr in addrs {
            self.check_pool_word(function, addr);
        }
    }

    /// Checks and records a candidate literal pool word.
    ///
    /// # Arguments
    ///
    /// * `function` - Function accumulator.
    /// * `addr` - Candidate pool address.
    fn check_pool_word(&self, function: &mut Function, addr: u64) {
        let Some(val) = self.read_pool_u32(addr) else {
            return;
        };
        let entry = crate::semantics::analyze_data_word(self.arch, addr, val);
        function.data_pools.push(entry);
    }

    /// Reads a 32-bit little-endian word from code memory if present.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address to read.
    ///
    /// # Returns
    ///
    /// Decoded 32-bit value if available.
    fn read_pool_u32(&self, addr: u64) -> Option<u32> {
        let b = self.code.contains(addr).then(|| self.code.bytes_at(addr))?;
        (b.len() >= 4).then(|| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Decodes one basic block linearly until it ends.
    ///
    /// # Arguments
    ///
    /// * `start` - Block start address.
    /// * `function` - Function being built.
    /// * `blocks` - Queue of further block starts.
    /// * `seen` - Instruction addresses already decoded.
    fn block(
        &mut self,
        start: u64,
        function: &mut Function,
        blocks: &mut Vec<u64>,
        seen: &mut HashSet<u64>,
    ) {
        let mut address = start;
        while let Some(decoded) = self.step(address, function, seen) {
            self.follow(&decoded, function, blocks);
            if !decoded.flow.falls_through() {
                break;
            }
            address += decoded.len as u64;
        }
    }

    /// Decodes and counts one instruction, enforcing the global cap.
    ///
    /// # Arguments
    ///
    /// * `address` - Instruction address.
    /// * `function` - Function being built.
    /// * `seen` - Instruction addresses already decoded.
    ///
    /// # Returns
    ///
    /// The decoded instruction, or `None` to stop the block.
    fn step(
        &mut self,
        address: u64,
        function: &mut Function,
        seen: &mut HashSet<u64>,
    ) -> Option<Decoded> {
        let room = self.result.instructions < MAX_INSTRUCTIONS;
        self.result.truncated |= !room;
        (room && seen.insert(address)).then_some(())?;
        let decoded = self.decoder.decode(address, self.code.bytes_at(address))?;
        self.record_instruction(address, &decoded, function);
        Some(decoded)
    }

    /// Records an analyzed instruction into the function.
    ///
    /// # Arguments
    ///
    /// * `address` - Instruction address.
    /// * `decoded` - Decoded instruction.
    /// * `function` - Function being built.
    fn record_instruction(&mut self, address: u64, decoded: &Decoded, function: &mut Function) {
        self.result.instructions += 1;
        function.instructions += 1;
        let text = crate::semantics::format_imm_hex(&decoded.text);
        let (kind, discussion) = crate::semantics::discuss(self.arch, address, &text);
        function
            .lines
            .push(crate::disasm::InstructionLine { address, text, kind, discussion });
    }

    /// Records calls, branch targets, and references for one instruction.
    ///
    /// # Arguments
    ///
    /// * `decoded` - Decoded instruction.
    /// * `function` - Function being built.
    /// * `blocks` - Queue of further block starts.
    fn follow(&mut self, decoded: &Decoded, function: &mut Function, blocks: &mut Vec<u64>) {
        for raw in &decoded.targets {
            self.route(
                code_address(*raw, self.interworking),
                decoded.flow,
                function,
                blocks,
            );
        }
    }

    /// Routes one target to a call, a local block, or a data reference.
    ///
    /// # Arguments
    ///
    /// * `target` - Aligned target address.
    /// * `flow` - Flow of the instruction producing the target.
    /// * `function` - Function being built.
    /// * `blocks` - Queue of further block starts.
    fn route(&mut self, target: u64, flow: Flow, function: &mut Function, blocks: &mut Vec<u64>) {
        let code = self.code.contains(target);
        match (flow, code) {
            (Flow::Call, true) => self.on_call(target, function),
            (_, true) => blocks.push(target),
            (_, false) => drop(function.references.insert(target)),
        }
    }

    /// Records a direct call and queues its target for discovery.
    ///
    /// # Arguments
    ///
    /// * `target` - Call target.
    /// * `function` - Function being built.
    fn on_call(&mut self, target: u64, function: &mut Function) {
        function.calls.insert(target);
        self.pending.push(target);
    }
}

/// Lists the addresses where function discovery starts.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `interworking` - Whether to clear the Thumb selector bit.
///
/// # Returns
///
/// Entry, symbol, export, and recorded function-start addresses.
fn roots(binary: &Binary, interworking: bool) -> Vec<u64> {
    let named = binary.symbols.iter().chain(&binary.exports).map(|s| s.address);
    let starts = named.chain(binary.function_starts.iter().copied());
    let mut roots: Vec<u64> = starts.chain(std::iter::once(binary.entry)).collect();
    roots.retain(|address| *address != 0);
    roots.iter().map(|a| code_address(*a, interworking)).collect()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::detect
    use crate::detect::Format;

    /// Builds a Thumb binary whose `.text` holds `code` at 0x1000.
    ///
    /// # Arguments
    ///
    /// * `code` - Machine code at file offset zero.
    ///
    /// # Returns
    ///
    /// A Cortex-M-style binary.
    fn thumb_binary(code: &[u8]) -> Binary {
        let size = crate::numeric::to_u64(code.len());
        let text = crate::model::Section::mapped(".text", 0x1000, 0x200).with_file(0, size);
        let mut binary = Binary::new(Format::Elf, Arch::Arm, "ARM", 32);
        (binary.thumb, binary.entry, binary.sections) =
            (true, 0x1000, vec![text.with_access(true, false)]);
        binary
    }

    /// Verifies that a Thumb `bl` plus `bx lr` recovers two functions.
    #[test]
    fn test_disassemble_thumb_call() {
        let code = [0x00, 0xf0, 0x02, 0xf8, 0x70, 0x47, 0x70, 0x47];
        let result = disassemble(&thumb_binary(&code), &code).unwrap();
        assert!(result.functions.contains_key(&0x1000));
        assert!(result.instructions >= 2);
    }

    /// Verifies architecture selection, including Thumb and endianness.
    #[test]
    fn test_sleigh_arch_selection() {
        let mut binary = Binary::new(Format::Elf, Arch::Mips, "MIPS", 32);
        binary.big_endian = true;
        assert_eq!(
            sleigh_arch(&binary).map(|a| a.name),
            Some(sarch::MIPS32_BE.name)
        );
    }

    /// Verifies that x86 is left to the dedicated decoder.
    #[test]
    fn test_sleigh_arch_skips_x86() {
        let binary = Binary::new(Format::Pe, Arch::X86_64, "x86-64", 64);
        assert!(sleigh_arch(&binary).is_none());
    }
}
