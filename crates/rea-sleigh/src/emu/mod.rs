//! P-code Emulator
//!
//! **File:** `mod.rs`
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

/// Emulator error types.
pub mod error;

/// P-code operation execution.
mod exec;

/// Sparse multi-space memory.
mod memory;

/// Machine-word p-code value semantics.
mod value;

// Import dependencies from std
use std::sync::Arc;

// Import dependencies from sleigh
use sleigh::{CompiledSpec, ContextBytes, Decoder, PcodeOp};

// Import crate items
use crate::{arch::Arch, decode, error::SleighError};

pub use error::EmuError;

/// Bytes fetched per instruction, covering the widest supported encoding.
const FETCH: usize = 16;

/// Upper bound on p-code operations executed within one instruction.
const OP_GUARD: u32 = 100_000;

/// How a single executed instruction affected control flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// Execution advanced within the current function.
    Flow,

    /// Execution entered a subroutine.
    Call,

    /// Execution returned from a subroutine.
    Return,
}

/// The outcome of emulating one instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// Program counter after the instruction.
    pub pc: u64,

    /// How control flow was affected.
    pub kind: StepKind,
}

/// The outcome of a bounded emulation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Halted {
    /// Program counter where emulation stopped.
    pub pc: u64,

    /// Number of instructions executed before stopping.
    pub steps: u64,
}

/// A p-code interpreter over one compiled specification and sparse memory.
pub struct Emulator {
    /// Compiled specification driving decode and lowering.
    spec: Arc<CompiledSpec>,

    /// Decode context carrying mode fields such as ARM `TMode`.
    context: ContextBytes,

    /// Sparse byte memory spanning every address space.
    memory: memory::Memory,

    /// Current program counter.
    pc: u64,

    /// Index of the default address space used for code and data.
    ram: usize,

    /// Whether bit 0 of a code address selects a sub-mode.
    interworking: bool,
}

impl Emulator {
    /// Builds an emulator for an architecture, compiling its spec if needed.
    ///
    /// # Arguments
    ///
    /// * `arch` - Target architecture.
    ///
    /// # Returns
    ///
    /// A ready emulator with empty memory.
    ///
    /// # Errors
    ///
    /// Returns [`SleighError`] when the spec is missing or fails to compile.
    pub fn new(arch: Arch) -> Result<Self, SleighError> {
        let spec = crate::spec::load(arch.root)?;
        let context = decode::build_context(&spec, arch.context)?;
        let ram = usize::from(spec.default_space());
        let memory = memory::Memory::new(arch.big_endian);
        Ok(Self { spec, context, memory, pc: 0, ram, interworking: arch.interworking })
    }

    /// Loads bytes into the default space at an address.
    ///
    /// # Arguments
    ///
    /// * `address` - Base virtual address.
    /// * `bytes` - Bytes to load.
    pub fn load(&mut self, address: u64, bytes: &[u8]) {
        self.memory.load(self.ram, address, bytes);
    }

    /// Returns the current program counter.
    ///
    /// # Returns
    ///
    /// The program counter.
    #[must_use]
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Sets the program counter.
    ///
    /// # Arguments
    ///
    /// * `address` - New program counter.
    pub fn set_pc(&mut self, address: u64) {
        self.pc = address;
    }

    /// Reads a value from the default space.
    ///
    /// # Arguments
    ///
    /// * `address` - Byte offset in the default space.
    /// * `size` - Width in bytes.
    ///
    /// # Returns
    ///
    /// The value, right-aligned.
    #[must_use]
    pub fn read_mem(&self, address: u64, size: usize) -> u64 {
        self.memory.read(self.ram, address, size)
    }

    /// Writes a value to the default space.
    ///
    /// # Arguments
    ///
    /// * `address` - Byte offset in the default space.
    /// * `size` - Width in bytes.
    /// * `value` - Value to store.
    pub fn write_mem(&mut self, address: u64, size: usize, value: u64) {
        self.memory.write(self.ram, address, size, value);
    }

    /// Reads a named register.
    ///
    /// # Arguments
    ///
    /// * `name` - Register name in the specification.
    ///
    /// # Returns
    ///
    /// The register value.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::MissingRegister`] when the name is unknown.
    pub fn read_reg(&self, name: &str) -> Result<u64, EmuError> {
        let (space, offset, size) = self.register_location(name)?;
        Ok(self.memory.read(space, offset, size))
    }

    /// Searches the default space for a byte pattern.
    ///
    /// # Arguments
    ///
    /// * `needle` - Byte pattern to search for.
    ///
    /// # Returns
    ///
    /// The addresses where the pattern begins.
    #[must_use]
    pub fn scan(&self, needle: &[u8]) -> Vec<u64> {
        self.memory.find(self.ram, needle)
    }

    /// Returns the base addresses of mapped pages in the default space.
    ///
    /// # Returns
    ///
    /// The page base addresses, ascending.
    #[must_use]
    pub fn mapped_pages(&self) -> Vec<u64> {
        self.memory.page_bases(self.ram)
    }

    /// Reads a byte range from the default space.
    ///
    /// # Arguments
    ///
    /// * `address` - Starting address.
    /// * `len` - Number of bytes to read.
    ///
    /// # Returns
    ///
    /// The bytes, with unmapped positions read as zero.
    #[must_use]
    pub fn read_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| self.memory.read_byte(self.ram, address.wrapping_add(i as u64)))
            .collect()
    }

    /// Writes a named register.
    ///
    /// # Arguments
    ///
    /// * `name` - Register name in the specification.
    /// * `value` - Value to store.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::MissingRegister`] when the name is unknown.
    pub fn write_reg(&mut self, name: &str, value: u64) -> Result<(), EmuError> {
        let (space, offset, size) = self.register_location(name)?;
        self.memory.write(space, offset, size, value);
        Ok(())
    }

    /// Resolves a register name to its storage location.
    ///
    /// # Arguments
    ///
    /// * `name` - Register name in the specification.
    ///
    /// # Returns
    ///
    /// The `(space index, byte offset, byte width)` triple.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::MissingRegister`] when the name is unknown.
    fn register_location(&self, name: &str) -> Result<(usize, u64, usize), EmuError> {
        let register = self.spec.register(name).ok_or_else(|| missing(name))?;
        let offset = u64::try_from(register.offset()).unwrap_or(0);
        Ok((usize::from(register.space()), offset, register.size()))
    }

    /// Emulates one instruction at the program counter.
    ///
    /// # Returns
    ///
    /// The step outcome.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::Decode`] when the bytes do not decode.
    pub fn step(&mut self) -> Result<Step, EmuError> {
        let (ops, next) = self.decode_here()?;
        let flow = self.run_ops(&ops);
        Ok(self.apply(flow, next))
    }

    /// Emulates instructions until a halt, a top-level return, or the budget.
    ///
    /// # Arguments
    ///
    /// * `budget` - Maximum number of instructions to execute.
    ///
    /// # Returns
    ///
    /// Where and after how many steps emulation stopped.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::Budget`] when the budget is exhausted, or a decode
    /// error from [`step`](Self::step).
    pub fn run(&mut self, budget: u64) -> Result<Halted, EmuError> {
        let mut depth = 0i64;
        for steps in 0..budget {
            let step = self.step()?;
            if let Some(halt) = settle(&mut depth, step, steps) {
                return Ok(halt);
            }
        }
        Err(EmuError::Budget(budget))
    }

    /// Decodes the instruction at the program counter into flat p-code.
    ///
    /// # Returns
    ///
    /// The lowered operations and the fall-through address.
    ///
    /// # Errors
    ///
    /// Returns [`EmuError::Decode`] when the bytes do not decode.
    fn decode_here(&self) -> Result<(Vec<PcodeOp>, u64), EmuError> {
        let address = self.fetch_address();
        let bytes = self.fetch(address);
        let decoder = Decoder::new(&self.spec);
        let result = decoder.decode_one(address, &bytes, &self.context);
        let insn = result.map_err(|_| EmuError::Decode(self.pc))?;
        let ops = insn.pcode_ops().map(|p| p.ops).unwrap_or_default();
        Ok((ops, insn.next_address()))
    }

    /// Returns the address to fetch from, masking the interworking bit.
    ///
    /// # Returns
    ///
    /// The program counter with bit 0 cleared on interworking targets.
    fn fetch_address(&self) -> u64 {
        if self.interworking {
            self.pc & !1
        } else {
            self.pc
        }
    }

    /// Reads the instruction bytes at an address from the default space.
    ///
    /// # Arguments
    ///
    /// * `address` - Address to fetch from.
    ///
    /// # Returns
    ///
    /// Up to [`FETCH`] bytes.
    fn fetch(&self, address: u64) -> Vec<u8> {
        (0..FETCH)
            .map(|index| self.memory.read_byte(self.ram, address.wrapping_add(index as u64)))
            .collect()
    }

    /// Executes an instruction's operations, following p-code branches.
    ///
    /// # Arguments
    ///
    /// * `ops` - Flat p-code for the instruction.
    ///
    /// # Returns
    ///
    /// The control effect that ends the instruction.
    fn run_ops(&mut self, ops: &[PcodeOp]) -> exec::OpFlow {
        let mut index = 0usize;
        for _ in 0..OP_GUARD {
            match self.advance(ops, index) {
                Ok(next) => index = next,
                Err(flow) => return flow,
            }
        }
        exec::OpFlow::Next
    }

    /// Executes one operation and reports the next index or a branch effect.
    ///
    /// # Arguments
    ///
    /// * `ops` - Flat p-code for the instruction.
    /// * `index` - Index of the operation to execute.
    ///
    /// # Returns
    ///
    /// The next index to execute, or a control effect to return.
    ///
    /// # Errors
    ///
    /// Returns the escaping control effect as the error variant.
    fn advance(&mut self, ops: &[PcodeOp], index: usize) -> Result<usize, exec::OpFlow> {
        let next = match self.op_flow(ops, index)? {
            exec::OpFlow::Next => index + 1,
            exec::OpFlow::Relative(delta) => relative_index(index, delta, ops.len()),
            other => return Err(other),
        };
        (next < ops.len()).then_some(next).ok_or(exec::OpFlow::Next)
    }

    /// Executes the operation at an index, ending the run when out of range.
    ///
    /// # Arguments
    ///
    /// * `ops` - Flat p-code for the instruction.
    /// * `index` - Index of the operation to execute.
    ///
    /// # Returns
    ///
    /// The control effect of the operation.
    ///
    /// # Errors
    ///
    /// Returns [`exec::OpFlow::Next`] when the index is past the operations,
    /// which falls the instruction through.
    fn op_flow(&mut self, ops: &[PcodeOp], index: usize) -> Result<exec::OpFlow, exec::OpFlow> {
        match ops.get(index) {
            Some(op) => Ok(self.execute_op(op)),
            None => Err(exec::OpFlow::Next),
        }
    }

    /// Applies an instruction's control effect to the program counter.
    ///
    /// # Arguments
    ///
    /// * `flow` - Control effect from the instruction's p-code.
    /// * `next` - Fall-through address.
    ///
    /// # Returns
    ///
    /// The resulting step.
    fn apply(&mut self, flow: exec::OpFlow, next: u64) -> Step {
        match flow {
            exec::OpFlow::Jump(address) => self.go(address, StepKind::Flow),
            exec::OpFlow::Call(address) => self.go(address, StepKind::Call),
            exec::OpFlow::Return(address) => self.go(address, StepKind::Return),
            _ => self.go(next, StepKind::Flow),
        }
    }

    /// Moves the program counter and builds a step of a given kind.
    ///
    /// # Arguments
    ///
    /// * `address` - New program counter.
    /// * `kind` - Control-flow classification.
    ///
    /// # Returns
    ///
    /// The step.
    fn go(&mut self, address: u64, kind: StepKind) -> Step {
        self.pc = address;
        Step { pc: address, kind }
    }
}

/// Builds a missing-register error for a name.
///
/// # Arguments
///
/// * `name` - Register name that was not found.
///
/// # Returns
///
/// The error.
fn missing(name: &str) -> EmuError {
    EmuError::MissingRegister(name.to_string())
}

/// Resolves a relative p-code branch to an operation index.
///
/// # Arguments
///
/// * `index` - Index of the branching operation.
/// * `delta` - Signed operation delta.
/// * `len` - Number of operations in the instruction.
///
/// # Returns
///
/// The target index, or `len` to fall through past the instruction.
fn relative_index(index: usize, delta: i64, len: usize) -> usize {
    let target = i64::try_from(index).ok().and_then(|base| base.checked_add(delta));
    match target {
        Some(value) if value >= 0 => usize::try_from(value).unwrap_or(len),
        _ => len,
    }
}

/// Updates call depth and reports whether a run should stop.
///
/// # Arguments
///
/// * `depth` - Current call depth, updated in place.
/// * `step` - The step just executed.
/// * `steps` - Number of instructions executed so far.
///
/// # Returns
///
/// A halt outcome when the run should stop, otherwise `None`.
fn settle(depth: &mut i64, step: Step, steps: u64) -> Option<Halted> {
    match step.kind {
        StepKind::Return if *depth == 0 => Some(Halted { pc: step.pc, steps }),
        StepKind::Return => drop_depth(depth),
        StepKind::Call => raise_depth(depth),
        StepKind::Flow => None,
    }
}

/// Decrements the call depth and continues the run.
///
/// # Arguments
///
/// * `depth` - Call depth to decrement.
///
/// # Returns
///
/// Always `None`, signalling the run continues.
fn drop_depth(depth: &mut i64) -> Option<Halted> {
    *depth -= 1;
    None
}

/// Increments the call depth and continues the run.
///
/// # Arguments
///
/// * `depth` - Call depth to increment.
///
/// # Returns
///
/// Always `None`, signalling the run continues.
fn raise_depth(depth: &mut i64) -> Option<Halted> {
    *depth += 1;
    None
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::arch
    use crate::arch;

    /// Verifies a backward branch past the start falls through to the end.
    #[test]
    fn test_relative_index_falls_through() {
        assert_eq!(relative_index(1, -5, 4), 4);
        assert_eq!(relative_index(1, 1, 4), 2);
    }

    /// Verifies the call-depth settler stops at a top-level return.
    #[test]
    fn test_settle_returns_at_top() {
        let mut depth = 0i64;
        let step = Step { pc: 0x10, kind: StepKind::Return };
        assert!(settle(&mut depth, step, 3).is_some());
    }

    /// Verifies a simple `AArch64` add program computes the expected sum.
    #[test]
    fn test_emulate_aarch64_add() {
        let mut emu = Emulator::new(arch::AARCH64_LE).unwrap();
        emu.write_reg("x0", 40).unwrap();
        emu.write_reg("x1", 2).unwrap();
        emu.load(0x1000, &[0x00, 0x00, 0x01, 0x8b]);
        emu.set_pc(0x1000);
        emu.step().unwrap();
        assert_eq!(emu.read_reg("x0").unwrap(), 42);
    }

    /// Verifies a missing register is reported rather than panicking.
    #[test]
    fn test_missing_register_errors() {
        let emu = Emulator::new(arch::AARCH64_LE).unwrap();
        assert!(emu.read_reg("not_a_register").is_err());
    }

    /// Verifies a two-instruction ARM Thumb program computes its result.
    #[test]
    fn test_emulate_thumb_immediate_add() {
        let mut emu = Emulator::new(arch::THUMB_LE).unwrap();
        emu.load(0x1000, &[0x28, 0x20, 0x02, 0x30]);
        emu.set_pc(0x1000);
        emu.step().unwrap();
        emu.step().unwrap();
        assert_eq!(emu.read_reg("r0").unwrap(), 42);
    }
}
