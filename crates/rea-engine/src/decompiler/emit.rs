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
    open_loops(out, opens, depth);
    emit_body(out, ctx, index, depth + opens);
    let closes = ctx.structure.closes_at(index);
    close_loops(out, closes, depth + opens);
    depth + opens - closes
}

/// Emits a block body, structured as a diamond when possible.
///
/// # Arguments
///
/// * `out` - Accumulated output lines.
/// * `ctx` - Emission context.
/// * `index` - Block index.
/// * `level` - Indentation depth.
fn emit_body(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, level: usize) {
    match ctx.structure.diamonds.get(&index) {
        Some(diamond) => append_diamond(out, ctx, index, level, diamond),
        None => append_block(out, ctx, index, level),
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
fn append_block(out: &mut Vec<String>, ctx: &Ctx<'_>, index: usize, level: usize) {
    if needs_label(ctx, index) {
        let start = ctx.graph.blocks[index].start;
        out.push(format!("{}{}", indent(level), label(start)));
    }
    append_statements(out, ctx, body_slice(ctx, index, false), index, level);
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
    out.push(format!("{}if ({}) {{", indent(level), diamond.cond));
    append_arm(out, ctx, diamond.then_block, level + 1);
    if let Some(other) = diamond.else_block {
        out.push(format!("{}}} else {{", indent(level)));
        append_arm(out, ctx, other, level + 1);
    }
    out.push(format!("{}}}", indent(level)));
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
    for line in body {
        out.push(statement(line, ctx, index, level));
    }
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
fn statement(line: &Line, ctx: &Ctx<'_>, index: usize, level: usize) -> String {
    let pad = indent(level);
    if let Inst::Other(text) = &line.inst {
        return format!("{pad}// {text}");
    }
    render_line(&pad, line, ctx, index)
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
fn render_line(pad: &str, line: &Line, ctx: &Ctx<'_>, index: usize) -> String {
    let asm = &line.asm;
    let body = control_text(&line.inst, ctx, index);
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
fn control_text(inst: &Inst, ctx: &Ctx<'_>, index: usize) -> Option<String> {
    match inst {
        Inst::Ret => Some("return;".to_string()),
        Inst::Jump(target) => Some(jump_text(*target, ctx, index)),
        Inst::JumpIf { cond, target } => Some(cond_text(cond, *target, ctx, index)),
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
fn cond_text(cond: &str, target: u64, ctx: &Ctx<'_>, index: usize) -> String {
    if is_back_edge(target, ctx, index) {
        return format!("if ({cond}) continue;");
    }
    format!("if ({cond}) goto lbl_{target:x};")
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
        assert!(text.contains("if (!=) {"));
        assert!(text.contains("} else {"));
    }
}
