//! Decompiler Control-Flow Graph
//!
//! **File:** `cfg.rs`
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
use std::collections::{BTreeMap, BTreeSet};

// Import dependencies from super::ir
use super::ir::{Inst, Line};

/// A basic block: a maximal run of instructions with one entry and one exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Start address of the block.
    pub start: u64,

    /// Index of the first line in the block.
    pub first: usize,

    /// Index one past the last line in the block.
    pub end: usize,

    /// Successor block start addresses.
    pub succ: Vec<u64>,

    /// Whether the block ends in a return.
    pub returns: bool,
}

/// A control-flow graph over one function's lifted lines.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// Blocks in address order.
    pub blocks: Vec<Block>,

    /// Map from block start address to its index.
    pub index: BTreeMap<u64, usize>,

    /// Addresses that are explicit branch targets and need labels.
    pub labels: BTreeSet<u64>,
}

/// Builds the control-flow graph for a function body.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines of one function.
///
/// # Returns
///
/// The populated control-flow graph.
#[must_use]
pub fn build(lines: &[Line]) -> Graph {
    let starts = mark_starts(lines);
    let blocks = split_blocks(lines, &starts);
    let index = index_blocks(&blocks);
    let linked = link(lines, blocks, &index);
    Graph { blocks: linked, index, labels: branch_targets(lines) }
}

/// Reports whether an instruction ends a basic block.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
///
/// # Returns
///
/// `true` for returns, jumps, and conditional jumps.
fn is_terminator(inst: &Inst) -> bool {
    matches!(inst, Inst::Ret | Inst::Jump(_) | Inst::JumpIf { .. })
}

/// Collects the explicit branch-target addresses in a body.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
///
/// # Returns
///
/// The set of jump targets.
fn branch_targets(lines: &[Line]) -> BTreeSet<u64> {
    lines.iter().filter_map(target_of_line).collect()
}

/// Extracts the branch target of one lifted line.
///
/// # Arguments
///
/// * `line` - Lifted instruction line.
///
/// # Returns
///
/// The target address for jumps, or `None`.
fn target_of_line(line: &Line) -> Option<u64> {
    match line.inst {
        Inst::Jump(target) | Inst::JumpIf { target, .. } => Some(target),
        _ => None,
    }
}

/// Marks the first line index of every basic block.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
///
/// # Returns
///
/// A flag per line, `true` at each block entry.
fn mark_starts(lines: &[Line]) -> Vec<bool> {
    let targets = branch_targets(lines);
    let range = 0..lines.len();
    let mut starts: Vec<bool> = range.map(|index| block_start(lines, index, &targets)).collect();
    if let Some(first) = starts.first_mut() {
        *first = true;
    }
    starts
}

/// Reports whether a line index begins a basic block.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `index` - Line index to test.
/// * `targets` - Set of branch-target addresses.
///
/// # Returns
///
/// `true` when the line begins a block.
fn block_start(lines: &[Line], index: usize, targets: &BTreeSet<u64>) -> bool {
    if index == 0 {
        return true;
    }
    is_terminator(&lines[index - 1].inst) || targets.contains(&lines[index].address)
}

/// Splits the body into blocks at the marked starts.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `starts` - Block-entry flags per line.
///
/// # Returns
///
/// The blocks in address order.
fn split_blocks(lines: &[Line], starts: &[bool]) -> Vec<Block> {
    let entries = block_entries(starts);
    let count = entries.len();
    (0..count).map(|n| make_block(lines, entries[n], entries.get(n + 1))).collect()
}

/// Collects the line indices that begin basic blocks.
///
/// # Arguments
///
/// * `starts` - Block-entry flags per line.
///
/// # Returns
///
/// The block-entry line indices.
fn block_entries(starts: &[bool]) -> Vec<usize> {
    let flags = starts.iter().enumerate();
    flags.filter(|(_, flag)| **flag).map(|(index, _)| index).collect()
}

/// Builds one block from its line range.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `first` - First line index.
/// * `next` - Optional index of the following block entry.
///
/// # Returns
///
/// The block with empty successors.
fn make_block(lines: &[Line], first: usize, next: Option<&usize>) -> Block {
    let end = next.copied().unwrap_or(lines.len());
    let start = lines[first].address;
    Block { start, first, end, succ: Vec::new(), returns: false }
}

/// Maps block start addresses to indices.
///
/// # Arguments
///
/// * `blocks` - Blocks in address order.
///
/// # Returns
///
/// The address-to-index map.
fn index_blocks(blocks: &[Block]) -> BTreeMap<u64, usize> {
    blocks.iter().enumerate().map(|(index, block)| (block.start, index)).collect()
}

/// Computes successors and return flags for every block.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `blocks` - Blocks in address order.
/// * `index` - Address-to-index map.
///
/// # Returns
///
/// The blocks with successors set.
fn link(lines: &[Line], blocks: Vec<Block>, index: &BTreeMap<u64, usize>) -> Vec<Block> {
    blocks.into_iter().map(|block| link_block(lines, block, index)).collect()
}

/// Computes successors and the return flag for one block.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `block` - Block to link.
/// * `index` - Address-to-index map.
///
/// # Returns
///
/// The linked block.
fn link_block(lines: &[Line], mut block: Block, index: &BTreeMap<u64, usize>) -> Block {
    let last = &lines[block.end - 1].inst;
    let next = lines.get(block.end).map(|line| line.address);
    let (succ, returns) = successors(last, next, index);
    block.succ = succ;
    block.returns = returns;
    block
}

/// Computes the successors and return flag of a terminator.
///
/// # Arguments
///
/// * `last` - Last instruction in the block.
/// * `next` - Address of the following line, when present.
/// * `index` - Address-to-index map.
///
/// # Returns
///
/// The successor addresses and whether the block returns.
fn successors(last: &Inst, next: Option<u64>, index: &BTreeMap<u64, usize>) -> (Vec<u64>, bool) {
    match last {
        Inst::Ret => (Vec::new(), true),
        Inst::Jump(target) => (existing(index, *target), false),
        Inst::JumpIf { target, .. } => (pair(index, *target, next), false),
        _ => (fallthrough(index, next), false),
    }
}

/// Returns a target address when it starts a block.
///
/// # Arguments
///
/// * `index` - Address-to-index map.
/// * `target` - Candidate target address.
///
/// # Returns
///
/// A one-element vector when the target begins a block.
fn existing(index: &BTreeMap<u64, usize>, target: u64) -> Vec<u64> {
    index.contains_key(&target).then_some(target).into_iter().collect()
}

/// Returns the branch target and fallthrough successors.
///
/// # Arguments
///
/// * `index` - Address-to-index map.
/// * `target` - Branch target address.
/// * `next` - Fallthrough address, when present.
///
/// # Returns
///
/// The distinct successor addresses.
fn pair(index: &BTreeMap<u64, usize>, target: u64, next: Option<u64>) -> Vec<u64> {
    let mut out = existing(index, target);
    if let Some(addr) = next {
        if addr != target && index.contains_key(&addr) {
            out.push(addr);
        }
    }
    out
}

/// Returns the fallthrough successor when it starts a block.
///
/// # Arguments
///
/// * `index` - Address-to-index map.
/// * `next` - Fallthrough address, when present.
///
/// # Returns
///
/// A one-element vector when the fallthrough begins a block.
fn fallthrough(index: &BTreeMap<u64, usize>, next: Option<u64>) -> Vec<u64> {
    next.and_then(|addr| index.contains_key(&addr).then_some(addr))
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    // Import dependencies from super::ir
    use super::super::ir::Expr;

    /// Builds a lifted line with the given instruction.
    ///
    /// # Arguments
    ///
    /// * `address` - Instruction address.
    /// * `inst` - Lifted instruction.
    ///
    /// # Returns
    ///
    /// The synthetic line.
    fn line(address: u64, inst: Inst) -> Line {
        Line { address, inst, asm: String::new() }
    }

    /// Builds a block-ending conditional jump.
    ///
    /// # Arguments
    ///
    /// * `target` - Jump target address.
    ///
    /// # Returns
    ///
    /// The conditional jump instruction.
    fn jump_if(target: u64) -> Inst {
        Inst::JumpIf { cond: "!=".into(), target }
    }

    /// Builds a body whose first instruction branches to its last.
    ///
    /// # Returns
    ///
    /// The synthetic instruction lines.
    fn conditional_body() -> Vec<Line> {
        vec![
            line(0x1000, jump_if(0x1010)),
            line(
                0x1004,
                Inst::Assign { dst: Expr::Reg("a".into()), src: Expr::Imm(1) },
            ),
            line(0x1010, Inst::Ret),
        ]
    }

    /// Verifies that a conditional branch yields two successors.
    #[test]
    fn test_build_conditional_successors() {
        let graph = build(&conditional_body());
        assert_eq!(graph.blocks[0].succ, vec![0x1010, 0x1004]);
    }

    /// Verifies that a return block reports no successors.
    #[test]
    fn test_build_return_block() {
        let body = vec![line(0x1000, Inst::Ret)];
        let graph = build(&body);
        assert_eq!(graph.blocks[0].succ.len(), 0);
        assert!(graph.blocks[0].returns);
    }

    /// Verifies that branch targets are recorded for labels.
    #[test]
    fn test_build_records_labels() {
        let body = vec![line(0x1000, Inst::Jump(0x1008)), line(0x1008, Inst::Ret)];
        let graph = build(&body);
        assert!(graph.labels.contains(&0x1008));
    }
}
