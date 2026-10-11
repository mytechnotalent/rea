//! P-code Operation Execution
//!
//! **File:** `exec.rs`
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

// Import dependencies from sleigh
use sleigh::{Opcode, PcodeOp, Varnode};

// Import crate::emu items
use crate::emu::{memory, value, Emulator};

/// How one p-code operation affects control flow within an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpFlow {
    /// Continue with the next operation.
    Next,

    /// Jump to a relative operation index inside this instruction.
    Relative(i64),

    /// Transfer to a machine address without recording a call.
    Jump(u64),

    /// Transfer to a machine address as a subroutine call.
    Call(u64),

    /// Return to a machine address held in a varnode.
    Return(u64),
}

/// Reinterprets a word as a signed branch delta without lossy casts.
///
/// # Arguments
///
/// * `value` - Relative offset encoded as a word.
///
/// # Returns
///
/// The same bit pattern as a signed delta.
fn as_delta(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

/// Returns an operation's input, or a zero constant when it is absent.
///
/// # Arguments
///
/// * `op` - Operation whose input is read.
/// * `index` - Input position.
///
/// # Returns
///
/// The input varnode, defaulting to a zero constant for malformed p-code.
fn operand(op: &PcodeOp, index: usize) -> Varnode {
    op.inputs.get(index).copied().unwrap_or(Varnode::constant(0, 8))
}

impl Emulator {
    /// Executes one p-code operation and reports its control effect.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to execute.
    ///
    /// # Returns
    ///
    /// The control effect, [`OpFlow::Next`] for pure data operations.
    pub(crate) fn execute_op(&mut self, op: &PcodeOp) -> OpFlow {
        if let Some(flow) = self.control_op(op) {
            return flow;
        }
        self.data_op(op);
        OpFlow::Next
    }

    /// Reads a varnode as a machine word, resolving constants directly.
    ///
    /// # Arguments
    ///
    /// * `node` - Varnode to read.
    ///
    /// # Returns
    ///
    /// The value, truncated to the varnode width.
    pub(crate) fn read_varnode(&self, node: Varnode) -> u64 {
        if node.is_constant() {
            value::trunc(node.offset, node.size)
        } else {
            self.memory.read(usize::from(node.space), node.offset, node.size)
        }
    }

    /// Writes a machine word to a non-constant varnode.
    ///
    /// # Arguments
    ///
    /// * `node` - Destination varnode.
    /// * `word` - Value to store.
    fn write_varnode(&mut self, node: Varnode, word: u64) {
        if !node.is_constant() {
            self.memory.write(usize::from(node.space), node.offset, node.size, word);
        }
    }

    /// Classifies an operation that may branch, delegating calls.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to inspect.
    ///
    /// # Returns
    ///
    /// The control effect, or `None` for data operations.
    fn control_op(&mut self, op: &PcodeOp) -> Option<OpFlow> {
        match op.opcode {
            Opcode::Branch => Some(Self::branch(op)),
            Opcode::CBranch => Some(self.cbranch(op)),
            Opcode::BranchInd => Some(OpFlow::Jump(self.read_varnode(operand(op, 0)))),
            _ => self.call_op(op),
        }
    }

    /// Classifies a call or return operation.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to inspect.
    ///
    /// # Returns
    ///
    /// The control effect, or `None` for data operations.
    fn call_op(&mut self, op: &PcodeOp) -> Option<OpFlow> {
        match op.opcode {
            Opcode::Call => Some(OpFlow::Call(operand(op, 0).offset)),
            Opcode::CallInd => Some(OpFlow::Call(self.read_varnode(operand(op, 0)))),
            Opcode::Return => Some(OpFlow::Return(self.read_varnode(operand(op, 0)))),
            _ => None,
        }
    }

    /// Resolves an unconditional branch into a relative or absolute target.
    ///
    /// # Arguments
    ///
    /// * `op` - Branch operation.
    ///
    /// # Returns
    ///
    /// A relative hop for a constant target, otherwise a machine jump.
    fn branch(op: &PcodeOp) -> OpFlow {
        let target = operand(op, 0);
        if target.is_constant() {
            OpFlow::Relative(as_delta(target.offset))
        } else {
            OpFlow::Jump(target.offset)
        }
    }

    /// Resolves a conditional branch using its predicate input.
    ///
    /// # Arguments
    ///
    /// * `op` - Conditional branch operation.
    ///
    /// # Returns
    ///
    /// The branch effect when taken, otherwise [`OpFlow::Next`].
    fn cbranch(&mut self, op: &PcodeOp) -> OpFlow {
        if self.read_varnode(operand(op, 1)) & 1 == 0 {
            return OpFlow::Next;
        }
        Self::branch(op)
    }

    /// Executes a data operation, trying memory, compute, then concat forms.
    ///
    /// # Arguments
    ///
    /// * `op` - Data operation to execute.
    fn data_op(&mut self, op: &PcodeOp) {
        if self.memory_op(op) || self.compute_op(op) {
            return;
        }
        self.piece_op(op);
    }

    /// Executes load, store, and wide-copy operations.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to execute.
    ///
    /// # Returns
    ///
    /// `true` when this operation was a memory move.
    fn memory_op(&mut self, op: &PcodeOp) -> bool {
        match op.opcode {
            Opcode::Load => self.exec_load(op),
            Opcode::Store => self.exec_store(op),
            Opcode::Copy if Self::is_wide(op) => self.exec_wide_copy(op),
            _ => return false,
        }
        true
    }

    /// Reports whether a copy exceeds the machine-word width.
    ///
    /// # Arguments
    ///
    /// * `op` - Copy operation.
    ///
    /// # Returns
    ///
    /// `true` when the output is wider than a machine word.
    fn is_wide(op: &PcodeOp) -> bool {
        op.output.is_some_and(|out| out.size > memory::WORD)
    }

    /// Executes a `LOAD` from the space named by its first operand.
    ///
    /// # Arguments
    ///
    /// * `op` - Load operation.
    fn exec_load(&mut self, op: &PcodeOp) {
        let Some(out) = op.output else { return };
        let space = usize::try_from(operand(op, 0).offset).unwrap_or(self.ram);
        let address = self.read_varnode(operand(op, 1));
        self.load_value(out, space, address);
    }

    /// Stores a loaded value, copying bytes for wide destinations.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination varnode.
    /// * `space` - Source space index.
    /// * `address` - Source byte offset.
    fn load_value(&mut self, out: Varnode, space: usize, address: u64) {
        if out.size <= memory::WORD {
            let word = self.memory.read(space, address, out.size);
            return self.write_varnode(out, word);
        }
        let to = (usize::from(out.space), out.offset);
        self.memory.copy((space, address), to, out.size);
    }

    /// Executes a `STORE` to the space named by its first operand.
    ///
    /// # Arguments
    ///
    /// * `op` - Store operation.
    fn exec_store(&mut self, op: &PcodeOp) {
        let space = usize::try_from(operand(op, 0).offset).unwrap_or(self.ram);
        let address = self.read_varnode(operand(op, 1));
        self.store_value(operand(op, 2), space, address);
    }

    /// Stores a source value, copying bytes for wide sources.
    ///
    /// # Arguments
    ///
    /// * `src` - Source varnode.
    /// * `space` - Destination space index.
    /// * `address` - Destination byte offset.
    fn store_value(&mut self, src: Varnode, space: usize, address: u64) {
        if src.size <= memory::WORD {
            let word = self.read_varnode(src);
            return self.memory.write(space, address, src.size, word);
        }
        let from = (usize::from(src.space), src.offset);
        self.memory.copy(from, (space, address), src.size);
    }

    /// Copies a value wider than a machine word byte for byte.
    ///
    /// # Arguments
    ///
    /// * `op` - Wide copy operation.
    fn exec_wide_copy(&mut self, op: &PcodeOp) {
        let Some(out) = op.output else { return };
        let src = operand(op, 0);
        let from = (usize::from(src.space), src.offset);
        self.memory.copy(from, (usize::from(out.space), out.offset), out.size);
    }

    /// Executes unary and binary arithmetic through the value tables.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to execute.
    ///
    /// # Returns
    ///
    /// `true` when a table evaluated the operation.
    fn compute_op(&mut self, op: &PcodeOp) -> bool {
        if let Some(evaluate) = value::unary_op(op.opcode) {
            self.apply_unary(op, evaluate);
        } else if let Some(evaluate) = value::binary_op(op.opcode) {
            self.apply_binary(op, evaluate);
        } else {
            return false;
        }
        true
    }

    /// Applies a unary evaluator to an operation's input and output.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation whose input feeds the evaluator.
    /// * `evaluate` - Unary machine-word evaluator.
    fn apply_unary(&mut self, op: &PcodeOp, evaluate: value::Unary) {
        let Some(out) = op.output else { return };
        let input = operand(op, 0);
        let result = evaluate(self.read_varnode(input), input.size, out.size);
        self.write_varnode(out, result);
    }

    /// Applies a binary evaluator to an operation's inputs and output.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation whose inputs feed the evaluator.
    /// * `evaluate` - Binary machine-word evaluator.
    fn apply_binary(&mut self, op: &PcodeOp, evaluate: value::Binary) {
        let Some(out) = op.output else { return };
        let (left, right) = (
            self.read_varnode(operand(op, 0)),
            self.read_varnode(operand(op, 1)),
        );
        let result = evaluate(left, right, operand(op, 0).size);
        self.write_varnode(out, result);
    }

    /// Executes sub-range extraction and concatenation operations.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation to execute.
    fn piece_op(&mut self, op: &PcodeOp) {
        match op.opcode {
            Opcode::SubPiece => self.exec_subpiece(op),
            Opcode::Piece => self.exec_piece(op),
            _ => {}
        }
    }

    /// Executes a `SUBPIECE`, slicing bytes above a constant offset.
    ///
    /// # Arguments
    ///
    /// * `op` - Sub-piece operation.
    fn exec_subpiece(&mut self, op: &PcodeOp) {
        let Some(out) = op.output else { return };
        let offset = usize::try_from(operand(op, 1).offset).unwrap_or(0);
        let result = value::subpiece(self.read_varnode(operand(op, 0)), offset, out.size);
        self.write_varnode(out, result);
    }

    /// Executes a `PIECE`, concatenating a high and low operand.
    ///
    /// # Arguments
    ///
    /// * `op` - Concatenation operation.
    fn exec_piece(&mut self, op: &PcodeOp) {
        let Some(out) = op.output else { return };
        let (high, low) = (operand(op, 0), operand(op, 1));
        let result = value::piece(self.read_varnode(high), self.read_varnode(low), low.size);
        self.write_varnode(out, result);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::arch
    use crate::arch;

    /// Verifies the signed delta reinterpretation round-trips.
    #[test]
    fn test_as_delta_round_trips() {
        assert_eq!(as_delta(u64::MAX), -1);
        assert_eq!(as_delta(5), 5);
    }

    /// Verifies a store followed by a load returns the stored word.
    #[test]
    fn test_store_then_load_round_trip() {
        let mut emu = Emulator::new(arch::AARCH64_LE).unwrap();
        emu.write_mem(0x4000, 4, 0xdead_beef);
        assert_eq!(emu.read_mem(0x4000, 4), 0xdead_beef);
    }
}
