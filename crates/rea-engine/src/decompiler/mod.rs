//! Structured Decompilation
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

/// Control-flow graph construction from lifted IR.
mod cfg;

/// Pseudo-C emission from the IR and control-flow graph.
mod emit;

/// Architecture-neutral intermediate representation and lifting.
mod ir;

/// Instruction-line simplification and copy propagation.
mod optimize;

/// Natural-loop and control-flow structuring.
mod structurer;

// Import dependencies from crate::disasm
use crate::disasm::Function;

// Import dependencies from crate::model
use crate::model::Arch;

pub use cfg::Graph;
pub use ir::{Expr, Inst, Line};

/// Decompiles a recovered function into readable pseudo-C.
///
/// # Arguments
///
/// * `func` - Recovered function with decoded instruction lines.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The recovered pseudo-C listing.
#[must_use]
pub fn decompile(func: &Function, arch: Arch) -> String {
    let address = func.address;
    let name = func.name.clone().unwrap_or_else(|| format!("sub_{address:x}"));
    let lifted: Vec<Line> = func.lines.iter().map(|line| ir::lift(line, arch)).collect();
    let analysis = cfg::build(&lifted);
    let lines = optimize::run(&lifted, arch, &analysis);
    let graph = cfg::build(&lines);
    emit::render(&name, &lines, &graph)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    // Import dependencies from crate::disasm
    use crate::disasm::InstructionLine;

    /// Builds a decoded instruction line for tests.
    ///
    /// # Arguments
    ///
    /// * `address` - Instruction address.
    /// * `text` - Assembly text.
    ///
    /// # Returns
    ///
    /// The synthetic instruction line.
    fn instr(address: u64, text: &str) -> InstructionLine {
        InstructionLine {
            address,
            text: text.to_string(),
            kind: String::new(),
            discussion: String::new(),
        }
    }

    /// Verifies that a function with no lines still renders a stub.
    #[test]
    fn test_decompile_empty_function() {
        let func = Function::at(0x1000, None);
        let text = decompile(&func, Arch::X86_64);
        assert!(text.contains("void sub_1000(void) {"));
    }

    /// Verifies that a move is decompiled into an assignment.
    #[test]
    fn test_decompile_move() {
        let mut func = Function::at(0x1000, None);
        func.lines = vec![instr(0x1000, "mov rax, rbx")];
        assert!(decompile(&func, Arch::X86_64).contains("rax = rbx;"));
    }

    /// Verifies that a dead store does not break decompilation.
    #[test]
    fn test_decompile_dead_store() {
        let mut func = Function::at(0x1000, None);
        func.lines = vec![
            instr(0x1000, "mov rax, rbx"),
            instr(0x1002, "mov rax, rcx"),
            instr(0x1004, "ret"),
        ];
        assert!(decompile(&func, Arch::X86_64).contains("void sub_1000"));
    }
}
