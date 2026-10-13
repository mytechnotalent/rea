//! Decompiler Control-Flow Structuring
//!
//! **File:** `structurer.rs`
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

// Import dependencies from super::cfg
use super::cfg::{Block, Graph};

// Import dependencies from super::ir
use super::ir::{Inst, Line};

/// Largest block count that is structured; larger functions keep flat control flow.
const MAX_BLOCKS: usize = 512;

/// A natural loop identified by its header and member blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop {
    /// Header block index.
    pub header: usize,

    /// Member block indices, including the header.
    pub nodes: BTreeSet<usize>,
}

/// How a single-block branch arm exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArmExit {
    /// The arm returns to the caller.
    Returns,

    /// The arm continues at a block index.
    Next(usize),
}

/// A recoverable if/else or if-then region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diamond {
    /// Condition text.
    pub cond: String,

    /// Then-branch block index.
    pub then_block: usize,

    /// Else-branch block index, when present.
    pub else_block: Option<usize>,
}

/// Control-flow structuring: loops, diamonds, and their emission positions.
#[derive(Debug, Clone, Default)]
pub struct Structure {
    /// Natural loops.
    pub loops: Vec<Loop>,

    /// Loop indices that open at each block index.
    pub opens: BTreeMap<usize, Vec<usize>>,

    /// Loop indices that close at each block index.
    pub closes: BTreeMap<usize, Vec<usize>>,

    /// Recoverable diamonds keyed by condition block index.
    pub diamonds: BTreeMap<usize, Diamond>,

    /// Blocks rendered inside a diamond arm.
    pub consumed: BTreeSet<usize>,
}

/// Shared context for structuring analysis.
struct Ctx<'a> {
    /// Control-flow graph.
    graph: &'a Graph,

    /// Lifted instruction lines.
    lines: &'a [Line],

    /// Predecessor index lists.
    preds: &'a [Vec<usize>],

    /// Loop structure.
    structure: &'a Structure,
}

impl Structure {
    /// Builds a structure from detected loops.
    ///
    /// # Arguments
    ///
    /// * `loops` - Detected natural loops.
    ///
    /// # Returns
    ///
    /// The structure with loop open and close positions.
    fn new(loops: Vec<Loop>) -> Self {
        let opens = opens_map(&loops);
        let closes = closes_map(&loops);
        Self { loops, opens, closes, ..Self::default() }
    }

    /// Reports whether an edge is a back edge to an enclosing loop header.
    ///
    /// # Arguments
    ///
    /// * `from` - Source block index.
    /// * `target` - Target block index.
    ///
    /// # Returns
    ///
    /// `true` when the edge closes an enclosing natural loop.
    #[must_use]
    pub fn is_back_edge(&self, from: usize, target: usize) -> bool {
        self.loops.iter().any(|lp| lp.header == target && lp.nodes.contains(&from))
    }

    /// Counts loops that open at a block index.
    ///
    /// # Arguments
    ///
    /// * `index` - Block index.
    ///
    /// # Returns
    ///
    /// The number of loops opening there.
    #[must_use]
    pub fn opens_at(&self, index: usize) -> usize {
        self.opens.get(&index).map_or(0, Vec::len)
    }

    /// Counts loops that close at a block index.
    ///
    /// # Arguments
    ///
    /// * `index` - Block index.
    ///
    /// # Returns
    ///
    /// The number of loops closing there.
    #[must_use]
    pub fn closes_at(&self, index: usize) -> usize {
        self.closes.get(&index).map_or(0, Vec::len)
    }
}

/// Analyzes the graph for natural loops, diamonds, and emission positions.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `lines` - Lifted instruction lines.
///
/// # Returns
///
/// The derived structure, or an empty one for oversized functions.
#[must_use]
pub fn analyze(graph: &Graph, lines: &[Line]) -> Structure {
    if graph.blocks.len() > MAX_BLOCKS {
        return Structure::default();
    }
    structure_function(graph, lines)
}

/// Builds loops and diamonds for a graph.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `lines` - Lifted instruction lines.
///
/// # Returns
///
/// The derived structure.
fn structure_function(graph: &Graph, lines: &[Line]) -> Structure {
    let preds = predecessors(graph);
    let dom = dominators(&preds);
    let mut structure = Structure::new(natural_loops(graph, &dom, &preds));
    let ctx = Ctx { graph, lines, preds: &preds, structure: &structure };
    let (diamonds, consumed) = find_diamonds(&ctx);
    apply_diamonds(&mut structure, diamonds, consumed);
    structure
}

/// Records recovered diamonds and consumed blocks into a structure.
///
/// # Arguments
///
/// * `structure` - Structure to update.
/// * `diamonds` - Recovered diamonds.
/// * `consumed` - Consumed arm blocks.
fn apply_diamonds(
    structure: &mut Structure,
    diamonds: BTreeMap<usize, Diamond>,
    consumed: BTreeSet<usize>,
) {
    structure.diamonds = diamonds;
    structure.consumed = consumed;
}

/// Builds predecessor index lists for every block.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
///
/// # Returns
///
/// One predecessor list per block index.
fn predecessors(graph: &Graph) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); graph.blocks.len()];
    for (source, block) in graph.blocks.iter().enumerate() {
        for target in succ_indices(block, graph) {
            preds[target].push(source);
        }
    }
    preds
}

/// Collects successor block indices of a block.
///
/// # Arguments
///
/// * `block` - Source block.
/// * `graph` - Control-flow graph.
///
/// # Returns
///
/// The successor indices.
fn succ_indices(block: &Block, graph: &Graph) -> Vec<usize> {
    block.succ.iter().filter_map(|addr| graph.index.get(addr).copied()).collect()
}

/// Computes dominator sets for every block.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
///
/// # Returns
///
/// One dominator set per block index.
fn dominators(preds: &[Vec<usize>]) -> Vec<BTreeSet<usize>> {
    let mut dom = initial(preds.len());
    while sweep(preds, &mut dom) {}
    dom
}

/// Builds the initial dominator sets for a block count.
///
/// # Arguments
///
/// * `count` - Number of blocks.
///
/// # Returns
///
/// Every set initialized to all blocks, with the entry set to itself.
fn initial(count: usize) -> Vec<BTreeSet<usize>> {
    let universe: BTreeSet<usize> = (0..count).collect();
    let mut dom = vec![universe; count];
    if let Some(first) = dom.first_mut() {
        *first = BTreeSet::from([0]);
    }
    dom
}

/// Performs one dominator dataflow sweep.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
/// * `dom` - Dominator sets to update.
///
/// # Returns
///
/// `true` when any set changed.
fn sweep(preds: &[Vec<usize>], dom: &mut [BTreeSet<usize>]) -> bool {
    let updated: Vec<BTreeSet<usize>> = (0..dom.len()).map(|idx| meet(preds, dom, idx)).collect();
    let changed = updated != *dom;
    dom.clone_from_slice(&updated);
    changed
}

/// Computes the dominator set of one block from its predecessors.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
/// * `dom` - Current dominator sets.
/// * `index` - Block index.
///
/// # Returns
///
/// The recomputed dominator set.
fn meet(preds: &[Vec<usize>], dom: &[BTreeSet<usize>], index: usize) -> BTreeSet<usize> {
    let incoming = &preds[index];
    if incoming.is_empty() {
        return BTreeSet::from([index]);
    }
    let mut acc = intersect(dom, incoming);
    acc.insert(index);
    acc
}

/// Intersects the dominator sets of predecessor blocks.
///
/// # Arguments
///
/// * `dom` - Current dominator sets.
/// * `preds` - Predecessor block indices.
///
/// # Returns
///
/// The intersection of predecessor dominator sets.
fn intersect(dom: &[BTreeSet<usize>], preds: &[usize]) -> BTreeSet<usize> {
    let mut iter = preds.iter();
    let first = iter.next().copied().unwrap_or(0);
    let mut acc = dom[first].clone();
    for &pred in iter {
        acc = acc.intersection(&dom[pred]).copied().collect();
    }
    acc
}

/// Collects the back edges of the graph.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `dom` - Dominator sets.
///
/// # Returns
///
/// Source and header index pairs.
fn back_edges(graph: &Graph, dom: &[BTreeSet<usize>]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for (source, block) in graph.blocks.iter().enumerate() {
        for header in succ_indices(block, graph) {
            push_back_edge(&mut edges, dom, source, header);
        }
    }
    edges
}

/// Records a back edge when the header dominates the source.
///
/// # Arguments
///
/// * `edges` - Accumulated back edges.
/// * `dom` - Dominator sets.
/// * `source` - Source block index.
/// * `header` - Target block index.
fn push_back_edge(
    edges: &mut Vec<(usize, usize)>,
    dom: &[BTreeSet<usize>],
    source: usize,
    header: usize,
) {
    if dom[source].contains(&header) {
        edges.push((source, header));
    }
}

/// Detects natural loops from the graph's back edges.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `dom` - Dominator sets.
/// * `preds` - Predecessor index lists.
///
/// # Returns
///
/// The detected loops.
fn natural_loops(graph: &Graph, dom: &[BTreeSet<usize>], preds: &[Vec<usize>]) -> Vec<Loop> {
    let mut loops: Vec<Loop> = Vec::new();
    for (source, header) in back_edges(graph, dom) {
        let nodes = loop_nodes(preds, source, header);
        push_loop(&mut loops, header, nodes);
    }
    loops
}

/// Adds a loop when its header is not already present.
///
/// # Arguments
///
/// * `loops` - Accumulated loops.
/// * `header` - Loop header index.
/// * `nodes` - Loop member indices.
fn push_loop(loops: &mut Vec<Loop>, header: usize, nodes: BTreeSet<usize>) {
    if loops.iter().any(|lp| lp.header == header) {
        return;
    }
    loops.push(Loop { header, nodes });
}

/// Computes the member blocks of a natural loop.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
/// * `source` - Back-edge source index.
/// * `header` - Loop header index.
///
/// # Returns
///
/// The loop member indices.
fn loop_nodes(preds: &[Vec<usize>], source: usize, header: usize) -> BTreeSet<usize> {
    let mut nodes = BTreeSet::from([header]);
    let mut stack = vec![source];
    while let Some(node) = stack.pop() {
        extend_loop(preds, &mut nodes, &mut stack, node, header);
    }
    nodes
}

/// Extends a loop with a node and its predecessors.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
/// * `nodes` - Loop member indices.
/// * `stack` - Pending worklist.
/// * `node` - Node being visited.
/// * `header` - Loop header index.
fn extend_loop(
    preds: &[Vec<usize>],
    nodes: &mut BTreeSet<usize>,
    stack: &mut Vec<usize>,
    node: usize,
    header: usize,
) {
    if !nodes.insert(node) {
        return;
    }
    for &pred in &preds[node] {
        if pred != header {
            stack.push(pred);
        }
    }
}

/// Maps loop indices by their opening block index.
///
/// # Arguments
///
/// * `loops` - Detected loops.
///
/// # Returns
///
/// Opening loop indices per block.
fn opens_map(loops: &[Loop]) -> BTreeMap<usize, Vec<usize>> {
    let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, lp) in loops.iter().enumerate() {
        map.entry(lp.header).or_default().push(index);
    }
    map
}

/// Maps loop indices by their closing block index.
///
/// # Arguments
///
/// * `loops` - Detected loops.
///
/// # Returns
///
/// Closing loop indices per block.
fn closes_map(loops: &[Loop]) -> BTreeMap<usize, Vec<usize>> {
    let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, lp) in loops.iter().enumerate() {
        map.entry(last_node(lp)).or_default().push(index);
    }
    map
}

/// Returns the highest block index in a loop.
///
/// # Arguments
///
/// * `lp` - Loop.
///
/// # Returns
///
/// The last member block index.
fn last_node(lp: &Loop) -> usize {
    lp.nodes.iter().next_back().copied().unwrap_or(lp.header)
}

/// Finds recoverable single-block diamonds across the graph.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
///
/// # Returns
///
/// The diamonds and the blocks they consume.
fn find_diamonds(ctx: &Ctx<'_>) -> (BTreeMap<usize, Diamond>, BTreeSet<usize>) {
    let mut diamonds = BTreeMap::new();
    let mut consumed = BTreeSet::new();
    for index in 0..ctx.graph.blocks.len() {
        record_diamond(ctx, index, &mut diamonds, &mut consumed);
    }
    (diamonds, consumed)
}

/// Records one recoverable diamond when present.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `index` - Condition block index.
/// * `diamonds` - Accumulated diamonds.
/// * `consumed` - Accumulated consumed blocks.
fn record_diamond(
    ctx: &Ctx<'_>,
    index: usize,
    diamonds: &mut BTreeMap<usize, Diamond>,
    consumed: &mut BTreeSet<usize>,
) {
    if consumed.contains(&index) {
        return;
    }
    let Some(diamond) = diamond_at(ctx, index, consumed) else {
        return;
    };
    apply_diamond(diamonds, consumed, index, diamond);
}

/// Records a diamond and marks its arms consumed.
///
/// # Arguments
///
/// * `diamonds` - Accumulated diamonds.
/// * `consumed` - Accumulated consumed blocks.
/// * `index` - Condition block index.
/// * `diamond` - Recovered diamond.
fn apply_diamond(
    diamonds: &mut BTreeMap<usize, Diamond>,
    consumed: &mut BTreeSet<usize>,
    index: usize,
    diamond: Diamond,
) {
    consumed.insert(diamond.then_block);
    if let Some(other) = diamond.else_block {
        consumed.insert(other);
    }
    diamonds.insert(index, diamond);
}

/// Derives a diamond from a conditional block, when clean.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `index` - Condition block index.
/// * `consumed` - Already consumed blocks.
///
/// # Returns
///
/// The diamond, or `None`.
fn diamond_at(ctx: &Ctx<'_>, index: usize, consumed: &BTreeSet<usize>) -> Option<Diamond> {
    let (cond, target) = conditional_at(ctx.lines, ctx.graph, index)?;
    let then_block = *ctx.graph.index.get(&target)?;
    let else_block = index + 1;
    if usable(ctx, consumed, then_block, else_block) {
        return classify(ctx, cond, then_block, else_block);
    }
    None
}

/// Reports whether a candidate diamond is well formed.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `consumed` - Already consumed blocks.
/// * `then_block` - Then-branch block index.
/// * `else_block` - Else-branch block index.
///
/// # Returns
///
/// `true` when the arms are forward, distinct, and consumable.
fn usable(ctx: &Ctx<'_>, consumed: &BTreeSet<usize>, then_block: usize, else_block: usize) -> bool {
    then_block > else_block
        && else_block < ctx.graph.blocks.len()
        && !blocked(ctx, consumed, then_block, else_block)
}

/// Reports whether the candidate arms cannot be safely consumed.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `consumed` - Already consumed blocks.
/// * `then_block` - Then-branch block index.
/// * `else_block` - Else-branch block index.
///
/// # Returns
///
/// `true` when the diamond must be skipped.
fn blocked(
    ctx: &Ctx<'_>,
    consumed: &BTreeSet<usize>,
    then_block: usize,
    else_block: usize,
) -> bool {
    let reused = consumed.contains(&then_block) || consumed.contains(&else_block);
    let in_loop = ctx.structure.opens_at(then_block) > 0 || ctx.structure.opens_at(else_block) > 0;
    let shared = !single(ctx.preds, then_block) || !single(ctx.preds, else_block);
    reused || in_loop || shared
}

/// Reports whether a block has exactly one predecessor.
///
/// # Arguments
///
/// * `preds` - Predecessor index lists.
/// * `index` - Block index.
///
/// # Returns
///
/// `true` for single-predecessor blocks.
fn single(preds: &[Vec<usize>], index: usize) -> bool {
    preds[index].len() == 1
}

/// Classifies a conditional into an if/else or if-then diamond.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `cond` - Condition text.
/// * `then_block` - Then-branch block index.
/// * `else_block` - Else-branch block index.
///
/// # Returns
///
/// The diamond, or `None` when the arms do not reconverge.
fn classify(ctx: &Ctx<'_>, cond: String, then_block: usize, else_block: usize) -> Option<Diamond> {
    let then_exit = arm_exit(ctx, then_block)?;
    let else_exit = arm_exit(ctx, else_block)?;
    reconverge(cond, then_block, else_block, then_exit, else_exit)
}

/// Builds a diamond from two arm exits.
///
/// # Arguments
///
/// * `cond` - Condition text.
/// * `then_block` - Then-branch block index.
/// * `else_block` - Else-branch block index.
/// * `then_exit` - Then-arm exit.
/// * `else_exit` - Else-arm exit.
///
/// # Returns
///
/// The diamond, or `None` when the arms do not reconverge.
fn reconverge(
    cond: String,
    then_block: usize,
    else_block: usize,
    then_exit: ArmExit,
    else_exit: ArmExit,
) -> Option<Diamond> {
    if converges(then_exit, else_exit) {
        return Some(if_else(cond, then_block, else_block));
    }
    if then_only(then_exit, else_block) {
        return Some(if_then(cond, then_block));
    }
    None
}

/// Reports whether both arms converge on the same block.
///
/// # Arguments
///
/// * `then_exit` - Then-arm exit.
/// * `else_exit` - Else-arm exit.
///
/// # Returns
///
/// `true` when the arms share a successor.
fn converges(then_exit: ArmExit, else_exit: ArmExit) -> bool {
    matches!((then_exit, else_exit), (ArmExit::Next(a), ArmExit::Next(b)) if a == b)
}

/// Reports whether the then-arm alone rejoins at the fallthrough.
///
/// # Arguments
///
/// * `then_exit` - Then-arm exit.
/// * `else_block` - Else-branch block index.
///
/// # Returns
///
/// `true` for a returning or rejoining then-arm.
fn then_only(then_exit: ArmExit, else_block: usize) -> bool {
    match then_exit {
        ArmExit::Returns => true,
        ArmExit::Next(join) => join == else_block,
    }
}

/// Builds an if-then diamond.
///
/// # Arguments
///
/// * `cond` - Condition text.
/// * `then_block` - Then-branch block index.
///
/// # Returns
///
/// The diamond without an else branch.
fn if_then(cond: String, then_block: usize) -> Diamond {
    Diamond { cond, then_block, else_block: None }
}

/// Builds an if/else diamond.
///
/// # Arguments
///
/// * `cond` - Condition text.
/// * `then_block` - Then-branch block index.
/// * `else_block` - Else-branch block index.
///
/// # Returns
///
/// The diamond with both branches.
fn if_else(cond: String, then_block: usize, else_block: usize) -> Diamond {
    Diamond { cond, then_block, else_block: Some(else_block) }
}

/// Reads the conditional of a block's terminator.
///
/// # Arguments
///
/// * `lines` - Lifted instruction lines.
/// * `graph` - Control-flow graph.
/// * `index` - Block index.
///
/// # Returns
///
/// The condition text and target address, or `None`.
fn conditional_at(lines: &[Line], graph: &Graph, index: usize) -> Option<(String, u64)> {
    let block = &graph.blocks[index];
    match &lines[block.end - 1].inst {
        Inst::JumpIf { cond, target } => Some((cond.clone(), *target)),
        _ => None,
    }
}

/// Computes how a single-block arm exits.
///
/// # Arguments
///
/// * `ctx` - Structuring context.
/// * `index` - Arm block index.
///
/// # Returns
///
/// The arm exit, or `None` when the arm is not a single block.
fn arm_exit(ctx: &Ctx<'_>, index: usize) -> Option<ArmExit> {
    let block = &ctx.graph.blocks[index];
    match &ctx.lines[block.end - 1].inst {
        Inst::Ret => Some(ArmExit::Returns),
        Inst::Jump(target) => Some(jump_exit(ctx.graph, *target)),
        Inst::JumpIf { .. } => None,
        _ => Some(ArmExit::Next(index + 1)),
    }
}

/// Resolves a jump exit to a block index.
///
/// # Arguments
///
/// * `graph` - Control-flow graph.
/// * `target` - Jump target address.
///
/// # Returns
///
/// The arm exit for the target.
fn jump_exit(graph: &Graph, target: u64) -> ArmExit {
    graph.index.get(&target).map_or(ArmExit::Returns, |join| ArmExit::Next(*join))
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
    ///
    /// # Returns
    ///
    /// The synthetic line.
    fn line(address: u64, inst: Inst) -> Line {
        Line { address, inst, asm: String::new() }
    }

    /// Builds a conditional jump instruction.
    ///
    /// # Arguments
    ///
    /// * `target` - Jump target address.
    ///
    /// # Returns
    ///
    /// The conditional jump.
    fn jump_if(target: u64) -> Inst {
        Inst::JumpIf { cond: "!=".into(), target }
    }

    /// Builds a two-arm if/else body converging on a return.
    ///
    /// # Returns
    ///
    /// The synthetic instruction lines.
    fn if_else_body() -> Vec<Line> {
        vec![
            line(0x1000, jump_if(0x1008)),
            line(0x1004, Inst::Jump(0x100c)),
            line(0x1008, Inst::Jump(0x100c)),
            line(0x100c, Inst::Ret),
        ]
    }

    /// Verifies that a self-loop is detected.
    #[test]
    fn test_analyze_detects_loop() {
        let lines = vec![line(0x1000, Inst::Jump(0x1000))];
        let structure = analyze(&build(&lines), &lines);
        assert_eq!(structure.loops.len(), 1);
        assert!(structure.is_back_edge(0, 0));
    }

    /// Verifies that a straight-line function has no loops.
    #[test]
    fn test_analyze_no_loop() {
        let lines = vec![line(0x1000, Inst::Ret)];
        let structure = analyze(&build(&lines), &lines);
        assert!(structure.loops.is_empty());
    }

    /// Verifies that an if/else diamond is recovered.
    #[test]
    fn test_analyze_recovers_if_else() {
        let lines = if_else_body();
        let structure = analyze(&build(&lines), &lines);
        assert_eq!(structure.diamonds.len(), 1);
        assert!(structure.consumed.contains(&1) && structure.consumed.contains(&2));
    }
}
