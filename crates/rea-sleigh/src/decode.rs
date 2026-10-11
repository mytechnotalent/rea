//! Generic Instruction Decoder
//!
//! **File:** `decode.rs`
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
use std::sync::Arc;

// Import dependencies from sleigh
use sleigh::{
    CompiledSpec, ContextBytes, Decoder, Instruction, InstructionPcode, Opcode, PcodeAst,
    PcodeStatementKind, SpaceId, Varnode,
};

// Import crate items
use crate::{arch::Arch, error::SleighError, spec};

/// How execution leaves one instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Execution continues at the next instruction.
    Fallthrough,

    /// Unconditional direct branch; execution does not fall through.
    Branch,

    /// Conditional branch; execution may fall through or take a target.
    CondBranch,

    /// Direct call; execution falls through on return.
    Call,

    /// Return from a function.
    Return,

    /// Indirect branch with no static target.
    Indirect,

    /// Indirect call; execution falls through on return.
    IndirectCall,
}

impl Flow {
    /// Reports whether execution continues at the following instruction.
    ///
    /// # Returns
    ///
    /// `true` for fall-through, conditional branches, and calls.
    #[must_use]
    pub fn falls_through(self) -> bool {
        matches!(
            self,
            Self::Fallthrough | Self::CondBranch | Self::Call | Self::IndirectCall
        )
    }

    /// Reports whether the instruction ends a basic block.
    ///
    /// # Returns
    ///
    /// `true` for every flow except plain fall-through.
    #[must_use]
    pub fn ends_block(self) -> bool {
        !matches!(self, Self::Fallthrough)
    }
}

/// One decoded instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// Instruction length in bytes.
    pub len: usize,

    /// Assembly text.
    pub text: String,

    /// How execution leaves the instruction.
    pub flow: Flow,

    /// Direct branch and call targets.
    pub targets: Vec<u64>,
}

/// A multi-architecture decoder bound to one compiled specification.
pub struct Disassembler {
    /// Compiled specification backing the decoder.
    spec: Arc<CompiledSpec>,

    /// Decode context carrying mode fields such as ARM `TMode`.
    context: ContextBytes,

    /// Whether bit 0 of a target selects a sub-mode.
    interworking: bool,
}

impl Disassembler {
    /// Builds a decoder for an architecture, compiling its spec if needed.
    ///
    /// # Arguments
    ///
    /// * `arch` - Target architecture.
    ///
    /// # Returns
    ///
    /// A ready decoder.
    ///
    /// # Errors
    ///
    /// Returns [`SleighError`] when the spec is missing, fails to compile, or
    /// names an unknown context field.
    pub fn new(arch: Arch) -> Result<Self, SleighError> {
        let spec = spec::load(arch.root)?;
        let context = build_context(&spec, arch.context)?;
        let interworking = arch.interworking;
        Ok(Self { spec, context, interworking })
    }

    /// Reports whether bit 0 of a code address selects a sub-mode.
    ///
    /// # Returns
    ///
    /// `true` for interworking architectures such as ARM.
    #[must_use]
    pub fn interworking(&self) -> bool {
        self.interworking
    }

    /// Decodes one instruction at an address.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address of the instruction.
    /// * `bytes` - Bytes beginning at the instruction.
    ///
    /// # Returns
    ///
    /// The decoded instruction, or `None` when the bytes do not decode.
    #[must_use]
    pub fn decode(&self, address: u64, bytes: &[u8]) -> Option<Decoded> {
        let decoder = Decoder::new(&self.spec);
        let instruction = decoder.decode_one(address, bytes, &self.context).ok()?;
        let (text, len) = (instruction.display().ok()?, instruction.len());
        let (flow, targets) = self.control_flow(&instruction, &text);
        Some(Decoded { len, text, flow, targets })
    }

    /// Determines control flow, preferring p-code and falling back to the AST.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    /// * `text` - Assembly text, used to resolve a fallback target.
    ///
    /// # Returns
    ///
    /// The flow kind and any direct targets.
    fn control_flow(&self, instruction: &Instruction, text: &str) -> (Flow, Vec<u64>) {
        match instruction.pcode_ops() {
            Ok(pcode) => classify(&pcode, self.spec.default_space()),
            Err(_) => fallback_flow(instruction, text),
        }
    }
}

/// Classifies flow from the AST when p-code lowering fails.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
/// * `text` - Assembly text holding the resolved target.
///
/// # Returns
///
/// The flow kind and a direct target parsed from the text when applicable.
fn fallback_flow(instruction: &Instruction, text: &str) -> (Flow, Vec<u64>) {
    let flow = instruction.pcode_ast().map_or(Flow::Fallthrough, |ast| ast_flow(&ast));
    let direct = matches!(flow, Flow::Branch | Flow::CondBranch | Flow::Call);
    let targets = direct.then(|| parse_target(text)).flatten();
    (flow, targets.into_iter().collect())
}

/// Classifies flow from a p-code AST by folding its statement kinds.
///
/// # Arguments
///
/// * `ast` - Instruction p-code AST.
///
/// # Returns
///
/// The strongest control flow among the statements.
fn ast_flow(ast: &PcodeAst) -> Flow {
    ast.statements
        .iter()
        .fold(Flow::Fallthrough, |flow, stmt| fold_ast(flow, &stmt.ty))
}

/// Folds one AST statement's control effect into the running flow.
///
/// # Arguments
///
/// * `flow` - Flow accumulated so far.
/// * `kind` - Statement kind being folded in.
///
/// # Returns
///
/// The combined flow.
fn fold_ast(flow: Flow, kind: &PcodeStatementKind) -> Flow {
    match ast_statement_flow(kind) {
        Some(effect) => stronger(flow, effect),
        None => flow,
    }
}

/// Maps an AST statement kind to the flow it induces.
///
/// # Arguments
///
/// * `kind` - Statement kind.
///
/// # Returns
///
/// The induced flow, or `None` for non-control statements.
fn ast_statement_flow(kind: &PcodeStatementKind) -> Option<Flow> {
    match kind {
        PcodeStatementKind::Call { .. } => Some(Flow::Call),
        PcodeStatementKind::CallIndirect { .. } => Some(Flow::IndirectCall),
        PcodeStatementKind::Return { .. } => Some(Flow::Return),
        PcodeStatementKind::BranchIndirect { .. } => Some(Flow::Indirect),
        _ => direct_ast_flow(kind),
    }
}

/// Maps the direct branch AST statements to their flow.
///
/// # Arguments
///
/// * `kind` - Statement kind.
///
/// # Returns
///
/// The branch flow, or `None` for non-branch statements.
fn direct_ast_flow(kind: &PcodeStatementKind) -> Option<Flow> {
    match kind {
        PcodeStatementKind::Branch { .. } => Some(Flow::Branch),
        PcodeStatementKind::ConditionalBranch { .. } => Some(Flow::CondBranch),
        _ => None,
    }
}

/// Parses a resolved branch or call target from assembly text.
///
/// # Arguments
///
/// * `text` - Assembly text whose last token is the target.
///
/// # Returns
///
/// The target address, when the last token is numeric.
fn parse_target(text: &str) -> Option<u64> {
    let token = text.split([' ', ',']).next_back()?.trim();
    match token.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => token.parse().ok(),
    }
}

/// Builds a decode context with an architecture's fields applied.
///
/// # Arguments
///
/// * `spec` - Compiled specification.
/// * `fields` - Field name and value pairs to set.
///
/// # Returns
///
/// The configured context.
///
/// # Errors
///
/// Returns [`SleighError::MissingField`] when a field is unknown.
pub(crate) fn build_context(
    spec: &CompiledSpec,
    fields: &[(&str, u32)],
) -> Result<ContextBytes, SleighError> {
    let mut context = spec.new_context();
    for (name, value) in fields {
        let field =
            spec.field(name).ok_or_else(|| SleighError::MissingField((*name).to_string()))?;
        let _ = spec.set_context_field(&mut context, field.id, u64::from(*value));
    }
    Ok(context)
}

/// Classifies control flow from an instruction's p-code.
///
/// # Arguments
///
/// * `pcode` - Lowered p-code for the instruction.
/// * `default_space` - Address space whose offsets are code addresses.
///
/// # Returns
///
/// The flow kind and any direct targets.
fn classify(pcode: &InstructionPcode, default_space: SpaceId) -> (Flow, Vec<u64>) {
    let mut targets = Vec::new();
    let mut flow = Flow::Fallthrough;
    for op in &pcode.ops {
        flow = fold(flow, op.opcode);
        collect_target(op.opcode, op.inputs.first(), default_space, &mut targets);
    }
    (flow, targets)
}

/// Combines the running flow with one operation's control effect.
///
/// # Arguments
///
/// * `flow` - Flow accumulated so far.
/// * `opcode` - Operation being folded in.
///
/// # Returns
///
/// The strongest control effect seen so far.
fn fold(flow: Flow, opcode: Opcode) -> Flow {
    match branch_flow(opcode) {
        Some(effect) => stronger(flow, effect),
        None => flow,
    }
}

/// Maps a control opcode to the flow it induces.
///
/// # Arguments
///
/// * `opcode` - Operation being inspected.
///
/// # Returns
///
/// The induced flow, or `None` for non-control operations.
fn branch_flow(opcode: Opcode) -> Option<Flow> {
    let found = CONTROL_FLOWS.iter().find(|(op, _)| *op == opcode);
    found.map(|(_, flow)| *flow)
}

/// Control opcodes paired with the flow they induce.
const CONTROL_FLOWS: [(Opcode, Flow); 6] = [
    (Opcode::Call, Flow::Call),
    (Opcode::CallInd, Flow::IndirectCall),
    (Opcode::Return, Flow::Return),
    (Opcode::BranchInd, Flow::Indirect),
    (Opcode::Branch, Flow::Branch),
    (Opcode::CBranch, Flow::CondBranch),
];

/// Keeps the stronger of two branch-like flows.
///
/// # Arguments
///
/// * `flow` - Flow accumulated so far.
/// * `candidate` - New branch flow.
///
/// # Returns
///
/// `flow` when it is already a call or return, otherwise `candidate`.
fn stronger(flow: Flow, candidate: Flow) -> Flow {
    match flow {
        Flow::Call | Flow::IndirectCall | Flow::Return => flow,
        _ => candidate,
    }
}

/// Records an absolute code target for branch and call operations.
///
/// # Arguments
///
/// * `opcode` - Operation being inspected.
/// * `dest` - First input varnode, if any.
/// * `default_space` - Address space whose offsets are code addresses.
/// * `targets` - Accumulator receiving targets.
fn collect_target(
    opcode: Opcode,
    dest: Option<&Varnode>,
    default_space: SpaceId,
    targets: &mut Vec<u64>,
) {
    let direct = matches!(opcode, Opcode::Branch | Opcode::CBranch | Opcode::Call);
    if let Some(node) = dest.filter(|_| direct) {
        if node.space == default_space && !node.is_constant() {
            targets.push(node.offset);
        }
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate::arch
    use crate::arch;

    /// Verifies Thumb decoding of a known RP2350 reset-handler instruction.
    #[test]
    fn test_decode_thumb_movw() {
        let dis = Disassembler::new(arch::THUMB_LE).unwrap();
        let out = dis.decode(0x1000_015a, &[0x4f, 0xf0, 0x50, 0x40]).unwrap();
        assert_eq!(out.len, 4);
        assert!(out.text.contains("mov"));
    }

    /// Verifies that a Thumb unconditional branch reports its target.
    #[test]
    fn test_decode_thumb_branch_target() {
        let dis = Disassembler::new(arch::THUMB_LE).unwrap();
        let out = dis.decode(0x1000, &[0xfe, 0xe7]).unwrap();
        assert_eq!(out.flow, Flow::Branch);
        assert_eq!(out.targets, vec![0x1000]);
    }

    /// Verifies flow fall-through classification.
    #[test]
    fn test_flow_properties() {
        assert!(Flow::Call.falls_through());
        assert!(!Flow::Branch.falls_through());
        assert!(Flow::Return.ends_block());
    }
}
