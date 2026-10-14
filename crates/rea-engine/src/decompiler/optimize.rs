//! Decompiler IR Simplification
//!
//! **File:** `optimize.rs`
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
use std::collections::BTreeMap;

// Import dependencies from crate::model
use crate::model::Arch;

// Import dependencies from super::cfg
use super::cfg::{Block, Graph};

// Import dependencies from super::ir
use super::ir::{Expr, Inst, Line};

/// Stack and frame pointer registers recognized across architectures.
const STACK_REGS: [&str; 6] = ["rsp", "esp", "rbp", "ebp", "sp", "fp"];

/// Simplifies and optimizes lifted instruction lines with cross-block dataflow.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `arch` - Architecture of the target.
/// * `graph` - Control-flow graph over the lines.
///
/// # Returns
///
/// The optimized lines.
#[must_use]
pub fn run(lines: &[Line], arch: Arch, graph: &Graph) -> Vec<Line> {
    let states = block_states(lines, graph);
    let optimized = optimize_blocks(lines, arch, graph, &states);
    eliminate(&optimized)
}

/// Optimizes each block using its entry state.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `arch` - Architecture of the target.
/// * `graph` - Control-flow graph.
/// * `states` - Per-block entry states.
///
/// # Returns
///
/// The optimized lines.
fn optimize_blocks(
    lines: &[Line],
    arch: Arch,
    graph: &Graph,
    states: &BTreeMap<u64, BTreeMap<String, String>>,
) -> Vec<Line> {
    let mut out = Vec::with_capacity(lines.len());
    for block in &graph.blocks {
        let mut map = states.get(&block.start).cloned().unwrap_or_default();
        let body = &lines[block.first..block.end];
        out.extend(body.iter().map(|line| optimize_line(line, arch, &mut map)));
    }
    out
}

/// Removes assignments overwritten before their value is read.
///
/// # Arguments
///
/// * `lines` - Optimized instruction lines.
///
/// # Returns
///
/// The lines without dead stores.
fn eliminate(lines: &[Line]) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::with_capacity(lines.len());
    for line in lines {
        if dead_before(&out, line) {
            continue;
        }
        out.push(line.clone());
    }
    out
}

/// Reports whether the previous line is a dead store.
///
/// # Arguments
///
/// * `out` - Lines kept so far.
/// * `line` - Current line.
///
/// # Returns
///
/// `true` when the previous line's result is overwritten unread.
fn dead_before(out: &[Line], line: &Line) -> bool {
    let Some(prev) = out.last() else {
        return false;
    };
    if is_terminator(&prev.inst) {
        return false;
    }
    overwrites(&prev.inst, &line.inst)
}

/// Reports whether a line overwrites the previous line's destination.
///
/// # Arguments
///
/// * `prev` - Previous instruction.
/// * `next` - Current instruction.
///
/// # Returns
///
/// `true` when the previous destination is written again without being read.
fn overwrites(prev: &Inst, next: &Inst) -> bool {
    match (prev, next) {
        (Inst::Assign { dst: Expr::Reg(old), .. }, Inst::Assign { dst: Expr::Reg(new), src }) => {
            old == new && !reads_reg(src, new)
        }
        _ => false,
    }
}

/// Reports whether an expression reads a register.
///
/// # Arguments
///
/// * `expr` - Expression to scan.
/// * `name` - Register name.
///
/// # Returns
///
/// `true` when the register is read.
fn reads_reg(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Reg(reg) => reg == name,
        Expr::Mem(inner) => reads_reg(inner, name),
        Expr::Bin(_, lhs, rhs) => reads_reg(lhs, name) || reads_reg(rhs, name),
        _ => false,
    }
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

/// Computes the copy-propagation state entering each block.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `graph` - Control-flow graph.
///
/// # Returns
///
/// The per-block entry state, keyed by block start address.
fn block_states(lines: &[Line], graph: &Graph) -> BTreeMap<u64, BTreeMap<String, String>> {
    let mut outs = initial_outs(graph);
    let budget = graph.blocks.len() + 1;
    for _ in 0..budget {
        if !propagate(lines, graph, &mut outs) {
            break;
        }
    }
    graph.blocks.iter().map(|b| (b.start, meet(graph, &outs, b))).collect()
}

/// Initializes empty out-states for every block.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
///
/// # Returns
///
/// The empty out-state per block start address.
fn initial_outs(graph: &Graph) -> BTreeMap<u64, BTreeMap<String, String>> {
    graph.blocks.iter().map(|b| (b.start, BTreeMap::new())).collect()
}

/// Runs one fixpoint sweep over the block states.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `graph` - Control-flow graph.
/// * `outs` - Block out-states to update.
///
/// # Returns
///
/// `true` when any state changed.
fn propagate(
    lines: &[Line],
    graph: &Graph,
    outs: &mut BTreeMap<u64, BTreeMap<String, String>>,
) -> bool {
    let mut changed = false;
    for block in &graph.blocks {
        changed |= update_block(lines, graph, outs, block);
    }
    changed
}

/// Updates one block's out-state from its entry state.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `graph` - Control-flow graph.
/// * `outs` - Block out-states.
/// * `block` - Block to update.
///
/// # Returns
///
/// `true` when the state changed.
fn update_block(
    lines: &[Line],
    graph: &Graph,
    outs: &mut BTreeMap<u64, BTreeMap<String, String>>,
    block: &Block,
) -> bool {
    let incoming = meet(graph, outs, block);
    let produced = transfer(lines, block, &incoming);
    if outs.get(&block.start) == Some(&produced) {
        return false;
    }
    outs.insert(block.start, produced);
    true
}

/// Intersects the out-states of a block's predecessors.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `outs` - Block out-states.
/// * `block` - Target block.
///
/// # Returns
///
/// The entry state valid on every incoming path.
fn meet(
    graph: &Graph,
    outs: &BTreeMap<u64, BTreeMap<String, String>>,
    block: &Block,
) -> BTreeMap<String, String> {
    let preds = predecessors(graph, block);
    let Some(&first) = preds.first() else {
        return BTreeMap::new();
    };
    let acc = outs[&graph.blocks[first].start].clone();
    intersect_all(graph, outs, &preds[1..], acc)
}

/// Intersects a state with every predecessor's out-state.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `outs` - Block out-states.
/// * `preds` - Remaining predecessor indices.
/// * `acc` - Accumulated intersection.
///
/// # Returns
///
/// The intersected state.
fn intersect_all(
    graph: &Graph,
    outs: &BTreeMap<u64, BTreeMap<String, String>>,
    preds: &[usize],
    mut acc: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    for &pred in preds {
        acc.retain(|key, value| outs[&graph.blocks[pred].start].get(key) == Some(value));
    }
    acc
}

/// Collects the predecessor block indices of a block.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `block` - Target block.
///
/// # Returns
///
/// The predecessor indices.
fn predecessors(graph: &Graph, block: &Block) -> Vec<usize> {
    let target = block.start;
    graph
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.succ.contains(&target))
        .map(|(i, _)| i)
        .collect()
}

/// Applies a block's definitions to an entry state.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `block` - Block to transfer over.
/// * `incoming` - Entry state.
///
/// # Returns
///
/// The exit state.
fn transfer(
    lines: &[Line],
    block: &Block,
    incoming: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut map = incoming.clone();
    for line in &lines[block.first..block.end] {
        update_map(&line.inst, &mut map);
    }
    map
}

/// Optimizes one line and updates the copy-propagation state.
///
/// # Arguments
///
/// * `line` - Lifted instruction line.
/// * `arch` - Architecture of the target.
/// * `map` - Copy-propagation state.
///
/// # Returns
///
/// The optimized line.
fn optimize_line(line: &Line, arch: Arch, map: &mut BTreeMap<String, String>) -> Line {
    let inst = rewrite(&line.inst, arch, map);
    update_map(&line.inst, map);
    Line { address: line.address, inst, asm: line.asm.clone() }
}

/// Applies copy propagation and simplification to an instruction.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
/// * `arch` - Architecture of the target.
/// * `map` - Copy-propagation state.
///
/// # Returns
///
/// The rewritten instruction.
fn rewrite(inst: &Inst, arch: Arch, map: &BTreeMap<String, String>) -> Inst {
    simplify_inst(&substitute_inst(inst, map), arch)
}

/// Substitutes copied registers in the sources of an instruction.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
/// * `map` - Copy-propagation state.
///
/// # Returns
///
/// The substituted instruction.
fn substitute_inst(inst: &Inst, map: &BTreeMap<String, String>) -> Inst {
    match inst {
        Inst::Assign { dst, src } => Inst::Assign { dst: dst.clone(), src: subst(src, map) },
        Inst::Store { addr, src } => Inst::Store { addr: subst(addr, map), src: subst(src, map) },
        Inst::Call { target } => Inst::Call { target: subst(target, map) },
        other => other.clone(),
    }
}

/// Substitutes copied registers within an expression.
///
/// # Arguments
///
/// * `expr` - Expression to rewrite.
/// * `map` - Copy-propagation state.
///
/// # Returns
///
/// The substituted expression.
fn subst(expr: &Expr, map: &BTreeMap<String, String>) -> Expr {
    match expr {
        Expr::Reg(name) => Expr::Reg(resolve(map, name)),
        Expr::Mem(inner) => Expr::Mem(Box::new(subst(inner, map))),
        Expr::Bin(op, lhs, rhs) => {
            Expr::Bin(op, Box::new(subst(lhs, map)), Box::new(subst(rhs, map)))
        }
        other => other.clone(),
    }
}

/// Resolves a register to its copied source.
///
/// # Arguments
///
/// * `map` - Copy-propagation state.
/// * `name` - Register name.
///
/// # Returns
///
/// The ultimate source register.
fn resolve(map: &BTreeMap<String, String>, name: &str) -> String {
    let mut current = name.to_string();
    for _ in 0..map.len() {
        match map.get(&current) {
            Some(next) if next != &current => current = next.clone(),
            _ => break,
        }
    }
    current
}

/// Updates copy-propagation state from an instruction's definition.
///
/// # Arguments
///
/// * `inst` - Original lifted instruction.
/// * `map` - Copy-propagation state.
fn update_map(inst: &Inst, map: &mut BTreeMap<String, String>) {
    match inst {
        Inst::Assign { dst: Expr::Reg(name), src } => record(name, src, map),
        Inst::Call { .. } | Inst::Ret | Inst::Jump(_) | Inst::JumpIf { .. } => map.clear(),
        _ => {}
    }
}

/// Records or clears a register definition.
///
/// # Arguments
///
/// * `name` - Defined register.
/// * `src` - Source expression.
/// * `map` - Copy-propagation state.
fn record(name: &str, src: &Expr, map: &mut BTreeMap<String, String>) {
    match src {
        Expr::Reg(source) => record_reg(name, source, map),
        _ => {
            map.remove(name);
        }
    }
}

/// Records a register-to-register copy, avoiding self or cyclic mappings.
///
/// # Arguments
///
/// * `name` - Defined register.
/// * `source` - Source register.
/// * `map` - Copy-propagation state.
fn record_reg(name: &str, source: &str, map: &mut BTreeMap<String, String>) {
    let resolved = resolve(map, source);
    if resolved == name {
        map.remove(name);
    } else {
        map.insert(name.to_string(), resolved);
    }
}

/// Simplifies the expressions of an instruction.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The simplified instruction.
fn simplify_inst(inst: &Inst, arch: Arch) -> Inst {
    match inst {
        Inst::Assign { dst, src } => simplified_assign(dst, src, arch),
        Inst::Store { addr, src } => simplified_store(addr, src, arch),
        Inst::Call { target } => Inst::Call { target: simplify_expr(target, arch) },
        other => other.clone(),
    }
}

/// Builds a simplified assignment instruction.
///
/// # Arguments
///
/// * `dst` - Destination expression.
/// * `src` - Source expression.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The simplified assignment.
fn simplified_assign(dst: &Expr, src: &Expr, arch: Arch) -> Inst {
    Inst::Assign { dst: dst.clone(), src: simplify_expr(src, arch) }
}

/// Builds a simplified store instruction.
///
/// # Arguments
///
/// * `addr` - Address expression.
/// * `src` - Source expression.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The simplified store.
fn simplified_store(addr: &Expr, src: &Expr, arch: Arch) -> Inst {
    Inst::Store { addr: simplify_addr(addr, arch), src: simplify_expr(src, arch) }
}

/// Simplifies an expression and names stack slots.
///
/// # Arguments
///
/// * `expr` - Expression to simplify.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The simplified expression.
fn simplify_expr(expr: &Expr, arch: Arch) -> Expr {
    match expr {
        Expr::Mem(inner) => simplify_mem(inner, arch),
        Expr::Bin(op, lhs, rhs) => fold(op, simplify_expr(lhs, arch), simplify_expr(rhs, arch)),
        other => other.clone(),
    }
}

/// Simplifies a memory address, naming stack slots.
///
/// # Arguments
///
/// * `expr` - Address expression.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// A named slot or the simplified address.
fn simplify_addr(expr: &Expr, arch: Arch) -> Expr {
    let simplified = simplify_expr(expr, arch);
    slot_name(&simplified).map_or(simplified, Expr::Var)
}

/// Simplifies a memory dereference, naming stack slots.
///
/// # Arguments
///
/// * `inner` - Address expression.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// A named slot or a memory dereference.
fn simplify_mem(inner: &Expr, arch: Arch) -> Expr {
    let simplified = simplify_expr(inner, arch);
    match slot_name(&simplified) {
        Some(name) => Expr::Var(name),
        None => Expr::Mem(Box::new(simplified)),
    }
}

/// Names a stack or frame slot from an address expression.
///
/// # Arguments
///
/// * `expr` - Address expression.
///
/// # Returns
///
/// The slot name, or `None`.
fn slot_name(expr: &Expr) -> Option<String> {
    let Expr::Bin("+", base, offset) = expr else {
        return None;
    };
    let (Expr::Reg(reg), Expr::Imm(value)) = (base.as_ref(), offset.as_ref()) else {
        return None;
    };
    STACK_REGS.contains(&reg.as_str()).then(|| format!("var_{value:x}"))
}

/// Folds a binary operation with constant operands and identities.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `lhs` - Left operand.
/// * `rhs` - Right operand.
///
/// # Returns
///
/// The folded expression.
fn fold(op: &'static str, lhs: Expr, rhs: Expr) -> Expr {
    if let (Expr::Imm(a), Expr::Imm(b)) = (&lhs, &rhs) {
        if let Some(value) = eval(op, *a, *b) {
            return Expr::Imm(value);
        }
    }
    identity(op, lhs, rhs)
}

/// Evaluates an arithmetic operator on constants.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `a` - Left value.
/// * `b` - Right value.
///
/// # Returns
///
/// The result, or `None`.
fn eval(op: &str, a: u64, b: u64) -> Option<u64> {
    match op {
        "+" => a.checked_add(b),
        "-" => a.checked_sub(b),
        "*" => a.checked_mul(b),
        other => eval_bits(other, a, b),
    }
}

/// Evaluates a bitwise or shift operator on constants.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `a` - Left value.
/// * `b` - Right value.
///
/// # Returns
///
/// The result, or `None`.
fn eval_bits(op: &str, a: u64, b: u64) -> Option<u64> {
    match op {
        "&" => Some(a & b),
        "|" => Some(a | b),
        "^" => Some(a ^ b),
        "<<" => a.checked_shl(u32::try_from(b).ok()?),
        ">>" => a.checked_shr(u32::try_from(b).ok()?),
        _ => None,
    }
}

/// Removes identity operations such as adding zero.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `lhs` - Left operand.
/// * `rhs` - Right operand.
///
/// # Returns
///
/// The simplified expression.
fn identity(op: &'static str, lhs: Expr, rhs: Expr) -> Expr {
    if drop_right(op, &rhs) {
        return lhs;
    }
    if drop_left(op, &lhs) {
        return rhs;
    }
    Expr::Bin(op, Box::new(lhs), Box::new(rhs))
}

/// Reports whether the right operand is an identity to drop.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `rhs` - Right operand.
///
/// # Returns
///
/// `true` when the right operand is neutral.
fn drop_right(op: &str, rhs: &Expr) -> bool {
    let additive = is_zero(rhs) && matches!(op, "+" | "-" | "|" | "^" | "<<" | ">>");
    additive || (is_one(rhs) && op == "*")
}

/// Reports whether the left operand is an identity to drop.
///
/// # Arguments
///
/// * `op` - Operator.
/// * `lhs` - Left operand.
///
/// # Returns
///
/// `true` when the left operand is neutral.
fn drop_left(op: &str, lhs: &Expr) -> bool {
    is_zero(lhs) && matches!(op, "+" | "|" | "^")
}

/// Reports whether an expression is the constant zero.
///
/// # Arguments
///
/// * `expr` - Expression to test.
///
/// # Returns
///
/// `true` for the constant zero.
fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(0))
}

/// Reports whether an expression is the constant one.
///
/// # Arguments
///
/// * `expr` - Expression to test.
///
/// # Returns
///
/// `true` for the constant one.
fn is_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(1))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds a lifted line with the given instruction.
    ///
    /// # Arguments
    ///
    /// * `inst` - Lifted instruction.
    ///
    /// # Returns
    ///
    /// The synthetic line.
    fn line(inst: Inst) -> Line {
        Line { address: 0x1000, inst, asm: String::new() }
    }

    /// Verifies constant folding of arithmetic.
    #[test]
    fn test_fold_constants() {
        let expr = Expr::Bin("+", Box::new(Expr::Imm(2)), Box::new(Expr::Imm(3)));
        assert_eq!(simplify_expr(&expr, Arch::X86_64), Expr::Imm(5));
    }

    /// Verifies that adding zero is removed.
    #[test]
    fn test_identity_add_zero() {
        let expr = Expr::Bin(
            "+",
            Box::new(Expr::Reg("rax".into())),
            Box::new(Expr::Imm(0)),
        );
        assert_eq!(simplify_expr(&expr, Arch::X86_64), Expr::Reg("rax".into()));
    }

    /// Builds a memory operand of a base register plus offset.
    ///
    /// # Arguments
    ///
    /// * `reg` - Base register name.
    /// * `offset` - Byte offset.
    ///
    /// # Returns
    ///
    /// The memory expression.
    fn mem_of(reg: &str, offset: u64) -> Expr {
        let base = Expr::Reg(reg.to_string());
        let displacement = Expr::Imm(offset);
        Expr::Mem(Box::new(Expr::Bin(
            "+",
            Box::new(base),
            Box::new(displacement),
        )))
    }

    /// Verifies stack slot naming.
    #[test]
    fn test_stack_slot_naming() {
        assert_eq!(
            simplify_expr(&mem_of("rsp", 0x28), Arch::X86_64),
            Expr::Var("var_28".into())
        );
    }

    /// Verifies within-block copy propagation.
    #[test]
    fn test_copy_propagation() {
        let lines = vec![
            line(Inst::Assign { dst: Expr::Reg("rbx".into()), src: Expr::Reg("rcx".into()) }),
            line(Inst::Assign { dst: Expr::Reg("rax".into()), src: Expr::Reg("rbx".into()) }),
        ];
        let graph = super::super::cfg::build(&lines);
        let optimized = run(&lines, Arch::X86_64, &graph);
        let expected = Inst::Assign { dst: Expr::Reg("rax".into()), src: Expr::Reg("rcx".into()) };
        assert_eq!(optimized[1].inst, expected);
    }

    /// Verifies a self-move does not create a cyclic copy mapping.
    #[test]
    fn test_self_move_does_not_cycle() {
        let mut map = BTreeMap::new();
        record("r0", &Expr::Reg("r0".into()), &mut map);
        assert_eq!(resolve(&map, "r0"), "r0");
    }

    /// Verifies a two-register cycle cannot make resolution spin.
    #[test]
    fn test_cyclic_copy_resolves() {
        let mut map = BTreeMap::new();
        map.insert("r0".to_string(), "r1".to_string());
        map.insert("r1".to_string(), "r0".to_string());
        assert!(!resolve(&map, "r0").is_empty());
    }
}
