//! Decompiler Pseudo-C Emission
//!
//! **File:** `emit.rs`
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

// Import dependencies from super::cfg
use super::cfg::Graph;

// Import dependencies from super::ir
use super::ir::{Expr, Inst, Line};

// Import dependencies from super::structurer
use super::structurer::{self, Diamond, Structure};

/// Shared context for pseudo-C emission.
struct Ctx<'a> {
    /// Lifted instruction lines.
    lines: &'a [Line],

    /// Control-flow graph.
    graph: &'a Graph,

    /// Control-flow structure.
    structure: &'a Structure,
}

/// Renders a recovered function as pseudo-C.
///
/// # Arguments
///
/// * `name` - Function name.
/// * `lines` - Lifted instruction lines.
/// * `graph` - Control-flow graph over the lines.
///
/// # Returns
///
/// The pseudo-C listing.
#[must_use]
pub fn render(name: &str, lines: &[Line], graph: &Graph) -> String {
    let structure = structurer::analyze(graph, lines);
    let ctx = Ctx { lines, graph, structure: &structure };
    let mut out = vec![format!("void {name}(void) {{")];
    out.extend(render_blocks(&ctx));
    out.push("}".to_string());
    out.join("\n")
}

/// Renders every visible block with loop wrapping.
///
/// # Arguments
///
/// * `ctx` - Emission context.
///
/// # Returns
///
/// The rendered body lines.
fn render_blocks(ctx: &Ctx<'_>) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 1usize;
    for index in visible(ctx) {
        depth = emit_block(&mut out, ctx, index, depth);
    }
    out
}

/// Lists the block indices not consumed by a diamond arm.
///
/// # Arguments
///
/// * `ctx` - Emission context.
///
/// # Returns
///
/// The visible block indices.
fn visible(ctx: &Ctx<'_>) -> Vec<usize> {
    let count = ctx.graph.blocks.len();
    (0..count).filter(|index| !ctx.structure.consumed.contains(index)).collect()
}

/// Emits one block with any loops that open or close around it.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `depth` - Current indentation depth.
///
/// # Returns
///
/// The indentation depth after the block.
fn emit_block(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, depth: usize) -> usize {
    let opens = ctx.structure.opens_at(index);
    let guard = loop_guard(ctx, index, opens);
    let inner = open_region(out, depth, opens, guard.as_deref());
    emit_body(out, ctx, index, depth + inner, guard.is_some());
    let closes = ctx.structure.closes_at(index);
    close_loops(out, closes, depth + inner);
    depth + inner - closes
}

/// Emits a loop or region opener and returns its nesting depth.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `depth` - Current indentation depth.
/// * `opens` - Number of loops opening.
/// * `guard` - Pre-tested loop condition, when present.
///
/// # Returns
///
/// The number of levels opened.
fn open_region(out: &mut Vec<String>, depth: usize, opens: usize, guard: Option<&str>) -> usize {
    if let Some(text) = guard {
        out.push(format!("{}while ({text}) {{", indent(depth)));
        return 1;
    }
    open_loops(out, opens, depth);
    opens
}

/// Recovers a pre-tested loop condition from a header exit test.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Header block index.
/// * `opens` - Number of loops opening at the block.
///
/// # Returns
///
/// The inverted loop condition, or `None`.
fn loop_guard(ctx: &Ctx<'_>, index: usize, opens: usize) -> Option<String> {
    let loop_id = single_open_loop(ctx, index, opens)?;
    let nodes = &ctx.structure.loops.get(loop_id)?.nodes;
    if exits_loop(ctx, index, nodes) {
        guard_text(ctx, index)
    } else {
        None
    }
}

/// Returns the loop index when exactly one loop opens without a diamond.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `opens` - Number of loops opening at the block.
///
/// # Returns
///
/// The loop index, or `None`.
fn single_open_loop(ctx: &Ctx<'_>, index: usize, opens: usize) -> Option<usize> {
    if opens != 1 || ctx.structure.diamonds.contains_key(&index) {
        return None;
    }
    ctx.structure.opens.get(&index)?.first().copied()
}

/// Reports whether a header's branch target leaves the loop.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Header block index.
/// * `nodes` - Loop member block indices.
///
/// # Returns
///
/// `true` when the branch target is outside the loop.
fn exits_loop(ctx: &Ctx<'_>, index: usize, nodes: &std::collections::BTreeSet<usize>) -> bool {
    match exit_target(ctx, index) {
        Some(exit) => !nodes.contains(&exit),
        None => false,
    }
}

/// Resolves a header block's conditional branch target to a block index.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The target block index, or `None`.
fn exit_target(ctx: &Ctx<'_>, index: usize) -> Option<usize> {
    let block = &ctx.graph.blocks[index];
    let line = ctx.lines.get(block.end.checked_sub(1)?)?;
    let Inst::JumpIf { target, .. } = line.inst else {
        return None;
    };
    block_at(ctx, target)
}

/// Builds the inverted loop condition from the header's exit test.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Header block index.
///
/// # Returns
///
/// The inverted condition, or `None`.
fn guard_text(ctx: &Ctx<'_>, index: usize) -> Option<String> {
    let block = &ctx.graph.blocks[index];
    let line = ctx.lines.get(block.end.checked_sub(1)?)?;
    let Inst::JumpIf { cond, .. } = &line.inst else {
        return None;
    };
    let prev = ctx.lines.get(block.end.wrapping_sub(2))?.asm.as_str();
    compare_guard(&negate_operator(cond)?, prev)
}

/// Finds a block index by start address.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `address` - Block start address.
///
/// # Returns
///
/// The block index, or `None`.
fn block_at(ctx: &Ctx<'_>, address: u64) -> Option<usize> {
    ctx.graph.blocks.iter().position(|block| block.start == address)
}

/// Returns the logical negation of a comparison operator.
///
/// # Arguments
///
/// * `cond` - Comparison operator.
///
/// # Returns
///
/// The negated operator, or `None`.
fn negate_operator(cond: &str) -> Option<String> {
    let neg = opposition(cond);
    let neg = if neg.is_empty() {
        equality_opposition(cond)
    } else {
        neg
    };
    (!neg.is_empty()).then(|| neg.to_string())
}

/// Returns the ordering negation of a comparison operator.
///
/// # Arguments
///
/// * `cond` - Comparison operator.
///
/// # Returns
///
/// The negated operator, or an empty string.
fn opposition(cond: &str) -> &'static str {
    match cond {
        "<" => ">=",
        "<=" => ">",
        ">" => "<=",
        ">=" => "<",
        _ => "",
    }
}

/// Returns the equality negation of a comparison operator.
///
/// # Arguments
///
/// * `cond` - Comparison operator.
///
/// # Returns
///
/// The negated operator, or an empty string.
fn equality_opposition(cond: &str) -> &'static str {
    match cond {
        "==" => "!=",
        "!=" => "==",
        _ => "",
    }
}

/// Emits a block body, structured as a diamond when possible.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `level` - Indentation depth.
/// * `drop_last` - Whether to omit the block's final line.
fn emit_body(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, level: usize, drop_last: bool) {
    match ctx.structure.diamonds.get(&index) {
        Some(diamond) => append_diamond(out, ctx, index, level, diamond),
        None => append_block(out, ctx, index, level, drop_last),
    }
}

/// Emits `while (true) {` lines for loops opening at a block.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `count` - Number of loops opening.
/// * `depth` - Current indentation depth.
fn open_loops(out: &mut Vec<String>, count: usize, depth: usize) {
    for _ in 0..count {
        out.push(format!("{}while (true) {{", indent(depth)));
    }
}

/// Emits closing braces for loops ending at a block.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `count` - Number of loops closing.
/// * `depth` - Indentation depth inside the loops.
fn close_loops(out: &mut Vec<String>, count: usize, depth: usize) {
    for _ in 0..count {
        out.push(format!("{}}}", indent(depth - 1)));
    }
}

/// Appends one basic block's lines to the output.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `level` - Indentation depth.
fn append_block(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, level: usize, drop_last: bool) {
    if needs_label(ctx, index) {
        let start = ctx.graph.blocks[index].start;
        out.push(format!("{}{}", indent(level), label(start)));
    }
    append_statements(out, ctx, body_slice(ctx, index, drop_last), index, level);
}

/// Reports whether a block still needs an emitted label.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// `true` when a non-back-edge jump targets the block.
fn needs_label(ctx: &Ctx<'_>, index: usize) -> bool {
    let target = ctx.graph.blocks[index].start;
    if !ctx.graph.labels.contains(&target) {
        return false;
    }
    (0..ctx.graph.blocks.len()).any(|source| external_jump(ctx, source, index, target))
}

/// Reports whether a block jumps to a target outside a loop back edge.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `source` - Source block index.
/// * `index` - Target block index.
/// * `target` - Target block start address.
///
/// # Returns
///
/// `true` for a non-back-edge jump to the target.
fn external_jump(ctx: &Ctx<'_>, source: usize, index: usize, target: u64) -> bool {
    let block = &ctx.graph.blocks[source];
    let jumps = jumps_to(&ctx.lines[block.end - 1].inst, target);
    jumps && !ctx.structure.is_back_edge(source, index)
}

/// Reports whether an instruction jumps to an address.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
/// * `target` - Target address.
///
/// # Returns
///
/// `true` for a direct jump to the target.
fn jumps_to(inst: &Inst, target: u64) -> bool {
    match inst {
        Inst::Jump(address) | Inst::JumpIf { target: address, .. } => *address == target,
        _ => false,
    }
}

/// Returns the instruction lines of a block, optionally dropping its terminator.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `drop_last` - Whether to exclude the terminator line.
///
/// # Returns
///
/// The block's instruction line slice.
fn body_slice<'a>(ctx: &Ctx<'a>, index: usize, drop_last: bool) -> &'a [Line] {
    let block = &ctx.graph.blocks[index];
    let end = if drop_last { block.end - 1 } else { block.end };
    &ctx.lines[block.first..end]
}

/// Returns the assembly text of a condition block's final line.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The final line assembly text, or an empty string.
fn cond_block_tail<'a>(ctx: &Ctx<'a>, index: usize) -> &'a str {
    let block = &ctx.graph.blocks[index];
    block
        .end
        .checked_sub(1)
        .and_then(|i| ctx.lines.get(i))
        .map_or("", |l| l.asm.as_str())
}

/// Returns the assembly text of a condition block's penultimate line.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The penultimate line assembly text, or an empty string.
fn cond_block_prev<'a>(ctx: &Ctx<'a>, index: usize) -> &'a str {
    let block = &ctx.graph.blocks[index];
    block
        .end
        .checked_sub(2)
        .and_then(|i| ctx.lines.get(i))
        .map_or("", |l| l.asm.as_str())
}

/// Appends a recovered if/else or if-then diamond.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Condition block index.
/// * `level` - Indentation depth.
/// * `diamond` - Diamond to render.
fn append_diamond(
    out: &mut Vec<String>,
    ctx: &Ctx<'_>,
    index: usize,
    level: usize,
    diamond: &Diamond,
) {
    append_statements(out, ctx, body_slice(ctx, index, true), index, level);
    let tail = cond_block_tail(ctx, index);
    let prev = cond_block_prev(ctx, index);
    let cond = cond_guard(&diamond.cond, tail, prev);
    out.push(format!("{}if ({cond}) {{", indent(level)));
    append_arm(out, ctx, diamond.then_block, level + 1);
    append_else(out, ctx, level, diamond);
    out.push(format!("{}}}", indent(level)));
}

/// Appends a diamond's else arm when present.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `level` - Indentation depth.
/// * `diamond` - Diamond to render.
fn append_else(out: &mut Vec<String>, ctx: &Ctx<'_>, level: usize, diamond: &Diamond) {
    if let Some(other) = diamond.else_block {
        out.push(format!("{}}} else {{", indent(level)));
        append_arm(out, ctx, other, level + 1);
    }
}

/// Appends a single-block diamond arm without its join jump.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Arm block index.
/// * `level` - Indentation depth.
fn append_arm(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, level: usize) {
    append_statements(out, ctx, body_slice(ctx, index, true), index, level);
    if ends_in_return(ctx, index) {
        out.push(format!("{}return;", indent(level)));
    }
}

/// Reports whether a block's terminator is a return.
///
/// # Arguments
///
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// `true` when the block ends in a return.
fn ends_in_return(ctx: &Ctx<'_>, index: usize) -> bool {
    let block = &ctx.graph.blocks[index];
    matches!(ctx.lines[block.end - 1].inst, Inst::Ret)
}

/// Appends rendered statements for a line range.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `body` - Instruction lines to render.
/// * `index` - Block index.
/// * `level` - Indentation depth.
fn append_statements(
    out: &mut Vec<String>,
    ctx: &Ctx<'_>,
    body: &[Line],
    index: usize,
    level: usize,
) {
    for (position, line) in body.iter().enumerate() {
        out.push(statement(line, prev_of(body, position), ctx, index, level));
    }
}

/// Returns the assembly text of a body line's predecessor.
///
/// # Arguments
///
/// * `body` - Block body lines.
/// * `position` - Line position.
///
/// # Returns
///
/// The predecessor assembly text, or an empty string.
fn prev_of(body: &[Line], position: usize) -> &str {
    body.get(position.wrapping_sub(1)).map_or("", |l| l.asm.as_str())
}

/// Formats a block label.
///
/// # Arguments
///
/// * `address` - Block start address.
///
/// # Returns
///
/// The label text.
fn label(address: u64) -> String {
    format!("lbl_{address:x}:")
}

/// Builds an indentation string for a depth.
///
/// # Arguments
///
/// * `level` - Indentation depth.
///
/// # Returns
///
/// Four spaces per level.
fn indent(level: usize) -> String {
    "    ".repeat(level)
}

/// Renders one lifted line as a C statement with its assembly comment.
///
/// # Arguments
///
/// * `line` - Lifted instruction line.
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `level` - Indentation depth.
///
/// # Returns
///
/// The indented statement.
fn statement(line: &Line, prev: &str, ctx: &Ctx<'_>, index: usize, level: usize) -> String {
    let pad = indent(level);
    if let Inst::Other(text) = &line.inst {
        return format!("{pad}// {text}");
    }
    render_line(&pad, line, prev, ctx, index)
}

/// Renders a non-comment statement with its assembly comment.
///
/// # Arguments
///
/// * `pad` - Indentation prefix.
/// * `line` - Lifted instruction line.
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The indented statement.
fn render_line(pad: &str, line: &Line, prev: &str, ctx: &Ctx<'_>, index: usize) -> String {
    let asm = &line.asm;
    let body = control_text(&line.inst, asm, prev, ctx, index);
    let text = body.unwrap_or_else(|| render_value(&line.inst));
    format!("{pad}{text}  // {asm}")
}

/// Renders control-flow instructions as statements.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The statement text, or `None` for non-control instructions.
fn control_text(inst: &Inst, asm: &str, prev: &str, ctx: &Ctx<'_>, index: usize) -> Option<String> {
    match inst {
        Inst::Ret => Some("return;".to_string()),
        Inst::Jump(target) => Some(jump_text(*target, ctx, index)),
        Inst::JumpIf { cond, target } => Some(cond_text(cond, *target, asm, prev, ctx, index)),
        _ => None,
    }
}

/// Renders an unconditional jump as a `continue` or `goto`.
///
/// # Arguments
///
/// * `target` - Jump target address.
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The statement text.
fn jump_text(target: u64, ctx: &Ctx<'_>, index: usize) -> String {
    if is_back_edge(target, ctx, index) {
        return "continue;".to_string();
    }
    format!("goto lbl_{target:x};")
}

/// Renders a conditional jump as a guarded `continue` or `goto`.
///
/// # Arguments
///
/// * `cond` - Comparison condition text.
/// * `target` - Jump target address.
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// The statement text.
fn cond_text(
    cond: &str,
    target: u64,
    asm: &str,
    prev: &str,
    ctx: &Ctx<'_>,
    index: usize,
) -> String {
    let guard = cond_guard(cond, asm, prev);
    if is_back_edge(target, ctx, index) {
        return format!("if ({guard}) continue;");
    }
    format!("if ({guard}) goto lbl_{target:x};")
}

/// Builds a condition from a zero-test or a preceding compare.
///
/// # Arguments
///
/// * `cond` - Branch condition text.
/// * `asm` - Branch instruction assembly.
/// * `prev` - Preceding instruction assembly.
///
/// # Returns
///
/// The C condition expression.
fn cond_guard(cond: &str, asm: &str, prev: &str) -> String {
    let guard = zero_test(asm).or_else(|| compare_operand(cond, prev));
    guard.unwrap_or_else(|| condition_expr(cond))
}

/// Builds a condition from the preceding compare operands.
///
/// # Arguments
///
/// * `cond` - Branch condition text.
/// * `prev` - Preceding instruction assembly.
///
/// # Returns
///
/// The condition, or `None`.
fn compare_operand(cond: &str, prev: &str) -> Option<String> {
    is_operator(cond).then(|| compare_guard(cond, prev)).flatten()
}

/// Builds a condition from a `cbz`/`cbnz` zero test.
///
/// # Arguments
///
/// * `asm` - Branch instruction assembly.
///
/// # Returns
///
/// The condition, or `None`.
fn zero_test(asm: &str) -> Option<String> {
    let (mnemonic, rest) = asm.split_once(char::is_whitespace)?;
    let reg = rest.split(',').next()?.trim();
    match mnemonic.to_ascii_lowercase().as_str() {
        "cbz" => Some(format!("{reg} == 0")),
        "cbnz" => Some(format!("{reg} != 0")),
        _ => None,
    }
}

/// Reports whether a condition is a bare comparison operator.
///
/// # Arguments
///
/// * `cond` - Branch condition text.
///
/// # Returns
///
/// `true` for a comparison operator.
fn is_operator(cond: &str) -> bool {
    matches!(cond, "==" | "!=" | "<" | "<=" | ">" | ">=")
}

/// Builds a condition from a preceding `cmp` or `test`.
///
/// # Arguments
///
/// * `cond` - Branch condition operator.
/// * `prev` - Preceding instruction assembly.
///
/// # Returns
///
/// The C condition, or `None`.
fn compare_guard(cond: &str, prev: &str) -> Option<String> {
    let (mnemonic, rest) = prev.split_once(char::is_whitespace)?;
    let (lhs, rhs) = rest.split_once(',')?;
    let (lhs, rhs) = (lhs.trim(), rhs.trim());
    match mnemonic.to_ascii_lowercase().as_str() {
        "cmp" => Some(format!("{lhs} {cond} {rhs}")),
        "test" => Some(format!("{lhs} {cond} 0")),
        _ => None,
    }
}

/// Builds a valid C condition from an IR branch condition.
///
/// # Arguments
///
/// * `cond` - Branch condition text.
///
/// # Returns
///
/// The C condition expression.
fn condition_expr(cond: &str) -> String {
    if is_operator(cond) {
        return format!("flags {cond} 0");
    }
    cond.to_string()
}

/// Reports whether a branch target closes an enclosing loop.
///
/// # Arguments
///
/// * `target` - Branch target address.
/// * `ctx` - Emission context.
/// * `index` - Block index.
///
/// # Returns
///
/// `true` when the target is an enclosing loop header.
fn is_back_edge(target: u64, ctx: &Ctx<'_>, index: usize) -> bool {
    ctx.graph
        .index
        .get(&target)
        .is_some_and(|header| ctx.structure.is_back_edge(index, *header))
}

/// Renders value-producing instructions as statements.
///
/// # Arguments
///
/// * `inst` - Lifted instruction.
///
/// # Returns
///
/// The statement text.
fn render_value(inst: &Inst) -> String {
    match inst {
        Inst::Assign { dst, src } => format!("{} = {};", render_expr(dst), render_expr(src)),
        Inst::Store { addr, src } => format!("{} = {};", render_lvalue(addr), render_expr(src)),
        Inst::Call { target } => format!("{}(...);", render_expr(target)),
        Inst::Other(text) => format!("// {text}"),
        _ => String::new(),
    }
}

/// Renders a store destination as an lvalue.
///
/// # Arguments
///
/// * `expr` - Destination expression.
///
/// # Returns
///
/// The rendered lvalue.
fn render_lvalue(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        other => format!("*({})", render_expr(other)),
    }
}

/// Renders an expression as C text.
///
/// # Arguments
///
/// * `expr` - Expression to render.
///
/// # Returns
///
/// The rendered expression.
fn render_expr(expr: &Expr) -> String {
    match expr {
        Expr::Reg(name) | Expr::Var(name) => name.clone(),
        Expr::Imm(value) => format!("0x{value:x}"),
        Expr::Mem(inner) => format!("*({})", render_expr(inner)),
        Expr::Bin(op, lhs, rhs) => format!("({} {op} {})", render_expr(lhs), render_expr(rhs)),
        Expr::Raw(text) => text.clone(),
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    // Import dependencies from super::super::cfg
    use super::super::cfg::build;

    // Import dependencies from super::super::ir
    use super::super::ir::{Inst, Line};

    /// Builds a lifted line with the given instruction.
    ///
    /// # Arguments
    ///
    /// * `address` - Instruction address.
    /// * `inst` - Lifted instruction.
    /// * `asm` - Original assembly text.
    ///
    /// # Returns
    ///
    /// The synthetic line.
    fn line(address: u64, inst: Inst, asm: &str) -> Line {
        Line { address, inst, asm: asm.to_string() }
    }

    /// Builds a two-arm if/else body converging on a return.
    ///
    /// # Returns
    ///
    /// The synthetic instruction lines.
    fn if_else_body() -> Vec<Line> {
        let jump = Inst::JumpIf { cond: "!=".into(), target: 0x1008 };
        vec![
            line(0x1000, jump, "jne 0x1008"),
            line(0x1004, Inst::Jump(0x100c), "jmp 0x100c"),
            line(0x1008, Inst::Jump(0x100c), "jmp 0x100c"),
            line(0x100c, Inst::Ret, "ret"),
        ]
    }

    /// Verifies that a return function renders a header and footer.
    #[test]
    fn test_render_returns_function() {
        let lines = vec![line(0x1000, Inst::Ret, "ret")];
        let graph = build(&lines);
        let text = render("sub_1000", &lines, &graph);
        assert!(text.starts_with("void sub_1000(void) {"));
        assert!(text.ends_with('}'));
        assert!(text.contains("return;"));
    }

    /// Verifies that branch conditions reuse preceding compare operands.
    #[test]
    fn test_cond_guard_uses_compare() {
        let less = compare_guard("<", "cmp rax,rbx");
        let zero = compare_guard("==", "test esi,esi");
        assert_eq!(less.as_deref(), Some("rax < rbx"));
        assert_eq!(zero.as_deref(), Some("esi == 0"));
        assert_eq!(cond_guard("!=", "jne 0x1008", ""), "flags != 0");
        assert_eq!(cond_guard("!=", "cbnz r3,0x1008", ""), "r3 != 0");
    }

    /// Builds a pre-tested loop header with an exit test.
    ///
    /// # Returns
    ///
    /// The synthetic instruction lines.
    fn guard_body() -> Vec<Line> {
        let jump = Inst::JumpIf { cond: "<".into(), target: 0x1010 };
        vec![
            line(0x1000, Inst::Other("cmp rax, rbx".into()), "cmp rax,rbx"),
            line(0x1004, jump, "jb 0x1010"),
            line(0x1008, Inst::Jump(0x1000), "jmp 0x1000"),
            line(0x1010, Inst::Ret, "ret"),
        ]
    }

    /// Verifies that a pre-tested header hoists a real while condition.
    #[test]
    fn test_render_loop_guard() {
        let lines = guard_body();
        let graph = build(&lines);
        let text = render("f", &lines, &graph);
        assert!(text.contains("while (rax >= rbx) {"));
        assert!(!text.contains("// jb 0x1010"));
    }

    /// Verifies that a self-loop renders as a while loop with continue.
    #[test]
    fn test_render_loop() {
        let lines = vec![line(0x1000, Inst::Jump(0x1000), "b 0x1000")];
        let graph = build(&lines);
        let text = render("f", &lines, &graph);
        assert!(text.contains("while (true) {"));
        assert!(text.contains("continue;"));
    }

    /// Verifies that an if/else diamond renders as an if/else block.
    #[test]
    fn test_render_if_else() {
        let lines = if_else_body();
        let graph = build(&lines);
        let text = render("f", &lines, &graph);
        assert!(text.contains("if (flags != 0) {"));
        assert!(text.contains("} else {"));
    }
}
