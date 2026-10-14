//! Decompiler Intermediate Representation
//!
//! **File:** `ir.rs`
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

// Import dependencies from crate::disasm
use crate::disasm::InstructionLine;

// Import dependencies from crate::model
use crate::model::Arch;

/// x86 operand size prefixes stripped before expression parsing.
const SIZE_PREFIXES: [&str; 4] = ["byte ptr ", "word ptr ", "dword ptr ", "qword ptr "];

/// Mnemonics that return control to the caller.
const RETURNS: &[&str] = &[
    "ret", "retn", "retf", "rts", "rte", "reti", "blr", "bctr", "rti", "eret", "iret", "rfe",
    "ret.w",
];

/// Mnemonics that call a subroutine.
const CALLS: &[&str] = &[
    "call", "callq", "bl", "blx", "jal", "jalr", "jalx", "jsr", "bsr", "rcall", "call0", "call4",
    "call8", "call12", "lcall", "bctrl",
];

/// Mnemonics that transfer control unconditionally.
const JUMPS: &[&str] = &[
    "jmp", "b", "bra", "j", "rjmp", "jmpf", "br", "b.w", "braf", "ba", "ijmp", "br.w",
];

/// Conditional-branch mnemonics shared across architectures.
const CONDITIONS: &[&str] = &[
    "je", "jz", "jne", "jnz", "jg", "jge", "jl", "jle", "ja", "jb", "jae", "jbe", "js", "jns",
    "jo", "jno", "jp", "jnp", "jcxz", "jecxz", "jrcxz", "loop", "loope", "loopne", "beq", "bne",
    "blt", "ble", "bgt", "bge", "blo", "bhi", "bls", "bhs", "cbz", "cbnz", "tbz", "tbnz", "bgtz",
    "blez", "bltz", "bgez", "beqz", "bnez", "bt", "bf", "bt.s", "bf.s", "bdnz", "bdz", "bc",
    "bgtu", "bleu", "bltu", "bgeu", "brbs", "brbc", "breq", "brne", "brcs", "brcc", "brsh", "brlo",
    "b.n",
];

/// Register-move mnemonics.
const MOVES: &[&str] = &[
    "mov", "movq", "movl", "movw", "movb", "movs", "movz", "movn", "movzx", "movsx", "movsxd",
    "movabs", "move", "mr", "li", "mv", "movi", "lui", "auipc", "mflr", "mtlr", "mfhi", "mflo",
    "mfcr", "mtcr", "mfspr", "mtspr", "mfc0", "mtc0", "csel", "csinc", "cset",
];

/// Load mnemonics.
const LOADS: &[&str] = &[
    "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldur", "ldurb", "ldurh", "ldursb", "ldursh",
    "ldursw", "lw", "lwz", "lwu", "lwzu", "lh", "lhu", "lha", "lhau", "lb", "lbu", "ld", "ldb",
    "ldh", "ldw", "lmw", "lwl", "lwr", "lwc1", "ldc1",
];

/// Store mnemonics.
const STORES: &[&str] = &[
    "str", "strb", "strh", "stur", "sturb", "sturh", "sw", "swz", "swu", "stw", "stwu", "sd", "sb",
    "sh", "st", "stb", "sth", "stmw", "swr", "swl", "sdc1", "swc1",
];

/// Address-materializing mnemonics.
const ADDRESSES: &[&str] = &["lea", "adr", "adrp", "la", "auipc", "larl", "llilf", "lgfi"];

/// Arithmetic mnemonic groups mapped to their IR operator.
const ARITH: &[(&str, &[&str])] = &[
    (
        "+",
        &[
            "add", "adds", "addi", "addiu", "addu", "addw", "adc", "adcs", "addis", "addic",
        ],
    ),
    (
        "-",
        &[
            "sub", "subs", "subi", "subiu", "subu", "subw", "sbc", "sbcs", "rsb", "rsc", "neg",
        ],
    ),
    (
        "*",
        &[
            "mul", "muls", "mullo", "imul", "mulw", "mult", "multu", "mullw",
        ],
    ),
    (
        "&",
        &["and", "ands", "andi", "andw", "andi.", "andis.", "bic"],
    ),
    ("|", &["or", "orrs", "orr", "ori", "orw", "oris", "orri"]),
    ("^", &["xor", "xors", "eor", "xori", "xorw", "xoris"]),
    (
        "<<",
        &[
            "shl", "sal", "lsl", "sll", "slli", "sllw", "slliw", "slwi", "rlwinm",
        ],
    ),
    (
        ">>",
        &[
            "shr", "lsr", "srl", "srli", "sra", "srai", "sar", "sraw", "srlw", "sraw", "srwi",
        ],
    ),
    ("<", &["slt", "sltu", "slti", "sltiu"]),
];

/// An expression or destination operand in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Register, such as `rax`.
    Reg(String),

    /// Immediate constant.
    Imm(u64),

    /// Memory dereference of an address expression.
    Mem(Box<Self>),

    /// Named stack or frame slot, such as `var_28`.
    Var(String),

    /// Binary operation `lhs op rhs`.
    Bin(&'static str, Box<Self>, Box<Self>),

    /// Unparsed operand text.
    Raw(String),
}

/// A lifted, architecture-neutral instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    /// Assigns `src` to `dst`.
    Assign {
        /// Destination operand.
        dst: Expr,
        /// Source expression.
        src: Expr,
    },

    /// Writes `src` to the memory address `addr`.
    Store {
        /// Destination address expression.
        addr: Expr,
        /// Source expression.
        src: Expr,
    },

    /// Calls the target expression.
    Call {
        /// Call target.
        target: Expr,
    },

    /// Returns from the function.
    Ret,

    /// Unconditional jump to an address.
    Jump(u64),

    /// Conditional jump to an address.
    JumpIf {
        /// Comparison condition text.
        cond: String,
        /// Jump target address.
        target: u64,
    },

    /// Instruction with no structured effect.
    Other(String),
}

/// One lifted instruction with its address and original assembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// Virtual address of the instruction.
    pub address: u64,

    /// Lifted instruction.
    pub inst: Inst,

    /// Original assembly text.
    pub asm: String,
}

/// Lifts a decoded instruction line into the IR.
///
/// # Arguments
///
/// * `line` - Decoded instruction with address and assembly text.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The lifted IR line.
#[must_use]
pub fn lift(line: &InstructionLine, arch: Arch) -> Line {
    let (mnemonic, operands) = split_asm(&line.text);
    let inst = lift_inst(&mnemonic, operands, arch);
    Line { address: line.address, inst, asm: line.text.clone() }
}

/// Splits assembly text into its mnemonic and operand text.
///
/// # Arguments
///
/// * `text` - Full instruction text.
///
/// # Returns
///
/// The lowercased mnemonic and the trimmed operand text.
fn split_asm(text: &str) -> (String, &str) {
    let trimmed = text.trim();
    match trimmed.split_once(char::is_whitespace) {
        Some((m, rest)) => (base_mnemonic(m), rest.trim()),
        None => (base_mnemonic(trimmed), ""),
    }
}

/// Lowercases a mnemonic and drops an ARM width suffix.
///
/// # Arguments
///
/// * `text` - Raw mnemonic text.
///
/// # Returns
///
/// The lowercased base mnemonic.
fn base_mnemonic(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    match lower.strip_suffix(".w").or_else(|| lower.strip_suffix(".n")) {
        Some(base) => base.to_string(),
        None => lower,
    }
}

/// Classifies and lifts one mnemonic with its operands.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The lifted instruction.
fn lift_inst(mnemonic: &str, operands: &str, arch: Arch) -> Inst {
    control(mnemonic, operands, arch).unwrap_or_else(|| assign(mnemonic, operands))
}

/// Lifts control-flow mnemonics when recognized.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The lifted control-flow instruction, or `None`.
fn control(mnemonic: &str, operands: &str, arch: Arch) -> Option<Inst> {
    if is_return(mnemonic, operands) {
        return Some(Inst::Ret);
    }
    if CALLS.contains(&mnemonic) {
        return Some(Inst::Call { target: call_target(operands) });
    }
    branch(mnemonic, operands, arch)
}

/// Lifts conditional and unconditional jump mnemonics.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// The lifted jump instruction, or `None`.
fn branch(mnemonic: &str, operands: &str, arch: Arch) -> Option<Inst> {
    if is_conditional(mnemonic, arch) {
        return Some(conditional(mnemonic, operands));
    }
    if JUMPS.contains(&mnemonic) {
        return Some(unconditional(operands));
    }
    None
}

/// Reports whether the instruction returns to the caller.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
///
/// # Returns
///
/// `true` for returns, including link-register returns.
fn is_return(mnemonic: &str, operands: &str) -> bool {
    if RETURNS.contains(&mnemonic) {
        return true;
    }
    (mnemonic == "bx" && operands.contains("lr"))
        || (mnemonic == "jr" && operands.contains("ra"))
        || (mnemonic == "pop" && operands.contains("pc"))
}

/// Reports whether the instruction is a conditional branch.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// `true` when the mnemonic denotes a conditional branch.
fn is_conditional(mnemonic: &str, arch: Arch) -> bool {
    CONDITIONS.contains(&mnemonic) || arch_conditional(mnemonic, arch)
}

/// Reports whether the architecture treats the mnemonic as conditional.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `arch` - Architecture of the target.
///
/// # Returns
///
/// `true` for architecture-specific conditional branches.
fn arch_conditional(mnemonic: &str, arch: Arch) -> bool {
    match arch {
        Arch::X86 | Arch::X86_64 => mnemonic.starts_with('j') && mnemonic != "jmp",
        Arch::Arm | Arch::Arm64 => mnemonic.starts_with("b."),
        Arch::Avr => mnemonic.starts_with("br") && mnemonic.len() > 2,
        _ => false,
    }
}

/// Builds a conditional jump when a target address is present.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
///
/// # Returns
///
/// A conditional jump, or an unstructured fallback.
fn conditional(mnemonic: &str, operands: &str) -> Inst {
    match target_of(operands) {
        Some(target) => Inst::JumpIf { cond: condition_alias(mnemonic).to_string(), target },
        None => Inst::Other(joined(mnemonic, operands)),
    }
}

/// Builds an unconditional jump when a target address is present.
///
/// # Arguments
///
/// * `operands` - Trimmed operand text.
///
/// # Returns
///
/// An unconditional jump, or an unstructured fallback.
fn unconditional(operands: &str) -> Inst {
    match target_of(operands) {
        Some(target) => Inst::Jump(target),
        None => Inst::Other(joined("jmp", operands)),
    }
}

/// Builds the call-target expression.
///
/// # Arguments
///
/// * `operands` - Trimmed operand text.
///
/// # Returns
///
/// A named target for direct calls, otherwise raw operand text.
fn call_target(operands: &str) -> Expr {
    match target_of(operands) {
        Some(target) => Expr::Raw(format!("sub_{target:x}")),
        None => Expr::Raw(operands.trim().to_string()),
    }
}

/// Lifts a non-control-flow instruction into an assignment or store.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Trimmed operand text.
///
/// # Returns
///
/// The lifted assignment, store, or unstructured fallback.
fn assign(mnemonic: &str, operands: &str) -> Inst {
    let (dst, src, extra) = split_operands(operands);
    match (dst, src) {
        (Some(d), Some(s)) => assign_pair(mnemonic, d, s, extra),
        _ => Inst::Other(joined(mnemonic, operands)),
    }
}

/// Lifts a two- or three-operand instruction into an assignment or store.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `dst` - Destination operand text.
/// * `src` - Source operand text.
/// * `extra` - Optional third operand text.
///
/// # Returns
///
/// The lifted assignment, store, or unstructured fallback.
fn assign_pair(mnemonic: &str, dst: &str, src: &str, extra: Option<&str>) -> Inst {
    if ADDRESSES.contains(&mnemonic) {
        return Inst::Assign { dst: parse_expr(dst), src: address_of(src) };
    }
    if STORES.contains(&mnemonic) {
        return store_from(dst, src);
    }
    value_pair(mnemonic, dst, src, extra)
}

/// Lifts a store, ordering the memory and value operands correctly.
///
/// Handles both x86 `store [mem], reg` and ARM/RISC-V `store reg, [mem]`.
///
/// # Arguments
///
/// * `dst` - First operand text.
/// * `src` - Second operand text.
///
/// # Returns
///
/// The store instruction.
fn store_from(dst: &str, src: &str) -> Inst {
    if is_memory(dst) {
        return Inst::Store { addr: address_of(dst), src: parse_expr(src) };
    }
    Inst::Store { addr: address_of(src), src: parse_expr(dst) }
}

/// Lifts a value-producing two- or three-operand instruction.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `dst` - Destination operand text.
/// * `src` - Source operand text.
/// * `extra` - Optional third operand text.
///
/// # Returns
///
/// The lifted assignment or unstructured fallback.
fn value_pair(mnemonic: &str, dst: &str, src: &str, extra: Option<&str>) -> Inst {
    if let Some(op) = arith_op(mnemonic) {
        return self_update(op, dst, src, extra);
    }
    data_transfer(mnemonic, dst, src)
}

/// Lifts a move or load into an assignment or a memory store.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `dst` - Destination operand text.
/// * `src` - Source operand text.
///
/// # Returns
///
/// The lifted assignment, store, or unstructured fallback.
fn data_transfer(mnemonic: &str, dst: &str, src: &str) -> Inst {
    if MOVES.contains(&mnemonic) && is_memory(dst) {
        return Inst::Store { addr: address_of(dst), src: parse_expr(src) };
    }
    if LOADS.contains(&mnemonic) || MOVES.contains(&mnemonic) {
        return Inst::Assign { dst: parse_expr(dst), src: parse_expr(src) };
    }
    Inst::Other(joined_pair(mnemonic, dst, src))
}

/// Reports whether operand text denotes a memory location.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// `true` for bracketed or base-plus-offset operands.
fn is_memory(text: &str) -> bool {
    let trimmed = strip_size(text.trim());
    bracketed(trimmed).is_some() || trimmed.contains('(')
}

/// Builds `dst = dst op src`.
///
/// # Arguments
///
/// * `op` - IR binary operator.
/// * `dst` - Destination operand text.
/// * `src` - Source operand text.
///
/// # Returns
///
/// The self-updating assignment.
fn self_update(op: &'static str, dst: &str, src: &str, extra: Option<&str>) -> Inst {
    let target = parse_expr(dst);
    let (lhs, rhs) = match extra {
        Some(third) => (parse_expr(src), parse_expr(third)),
        None => (target.clone(), parse_expr(src)),
    };
    Inst::Assign { dst: target, src: Expr::Bin(op, Box::new(lhs), Box::new(rhs)) }
}

/// Resolves the arithmetic operator for a mnemonic.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
///
/// # Returns
///
/// The operator, or `None` when the mnemonic is not arithmetic.
fn arith_op(mnemonic: &str) -> Option<&'static str> {
    ARITH.iter().find_map(|(op, names)| names.contains(&mnemonic).then_some(*op))
}

/// Parses an operand into an expression.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The parsed expression.
fn parse_expr(text: &str) -> Expr {
    let trimmed = strip_size(text.trim()).trim_start_matches('#');
    if let Some(inner) = bracketed(trimmed) {
        return Expr::Mem(Box::new(drop_zero(parse_sum(inner))));
    }
    if let Some(inner) = parenthesized(trimmed) {
        return Expr::Mem(Box::new(inner));
    }
    parse_atom(trimmed)
}

/// Removes a zero offset from an address expression.
///
/// # Arguments
///
/// * `expr` - Address expression.
///
/// # Returns
///
/// The base expression when the offset is zero, otherwise the expression.
fn drop_zero(expr: Expr) -> Expr {
    match expr {
        Expr::Bin("+", base, rhs) if *rhs == Expr::Imm(0) => *base,
        other => other,
    }
}

/// Removes a leading x86 operand size prefix.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The operand text without a size prefix.
fn strip_size(text: &str) -> &str {
    SIZE_PREFIXES.iter().find_map(|p| text.strip_prefix(p)).unwrap_or(text).trim()
}

/// Extracts the inner text of a bracketed operand.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The bracket contents when the operand is fully bracketed.
fn bracketed(text: &str) -> Option<&str> {
    text.strip_prefix('[').and_then(|r| r.strip_suffix(']'))
}

/// Parses a MIPS-style `offset(base)` operand.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The base-plus-offset expression, or `None`.
fn parenthesized(text: &str) -> Option<Expr> {
    let open = text.find('(')?;
    let base = text[open + 1..].strip_suffix(')')?;
    let offset = text[..open].trim();
    Some(address_sum(parse_expr(base), offset))
}

/// Adds a non-zero offset to a base expression.
///
/// # Arguments
///
/// * `base` - Parsed base expression.
/// * `offset` - Offset operand text.
///
/// # Returns
///
/// The base plus offset expression, or the base alone.
fn address_sum(base: Expr, offset: &str) -> Expr {
    if offset.is_empty() || offset == "0" {
        return base;
    }
    Expr::Bin("+", Box::new(base), Box::new(parse_expr(offset)))
}

/// Parses a sum of terms inside a memory operand.
///
/// # Arguments
///
/// * `text` - Inner operand text.
///
/// # Returns
///
/// The parsed expression.
fn parse_sum(text: &str) -> Expr {
    match find_add(text) {
        Some((index, sign)) => combine(text, index, sign),
        None => parse_term(text),
    }
}

/// Combines the two sides of a sum or difference.
///
/// # Arguments
///
/// * `text` - Full operand text.
/// * `index` - Byte index of the operator.
/// * `sign` - The `+` or `-` character.
///
/// # Returns
///
/// The binary expression.
fn combine(text: &str, index: usize, sign: char) -> Expr {
    let lhs = parse_sum(&text[..index]);
    let rhs = parse_sum(&text[index + 1..]);
    Expr::Bin(
        if sign == '-' { "-" } else { "+" },
        Box::new(lhs),
        Box::new(rhs),
    )
}

/// Finds the first additive operator past any leading sign.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The operator index and character, or `None`.
fn find_add(text: &str) -> Option<(usize, char)> {
    text.char_indices().skip(1).find(|(_, c)| *c == '+' || *c == '-' || *c == ',')
}

/// Parses a product term inside a memory operand.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The parsed expression.
fn parse_term(text: &str) -> Expr {
    match text.split_once('*') {
        Some((lhs, rhs)) => Expr::Bin("*", Box::new(parse_atom(lhs)), Box::new(parse_atom(rhs))),
        None => parse_atom(text),
    }
}

/// Parses a leaf operand into a register, immediate, or raw text.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The leaf expression.
fn parse_atom(text: &str) -> Expr {
    let trimmed = text.trim().trim_start_matches('#');
    if let Some(value) = parse_imm(trimmed) {
        return Expr::Imm(value);
    }
    if is_register(trimmed) {
        return Expr::Reg(trimmed.to_string());
    }
    Expr::Raw(trimmed.to_string())
}

/// Parses an immediate constant in decimal or hexadecimal.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The constant value, or `None`.
fn parse_imm(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    if let Some(hex) = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = trimmed.strip_suffix('h').or_else(|| trimmed.strip_suffix('H')) {
        return u64::from_str_radix(hex, 16).ok();
    }
    trimmed.parse::<u64>().ok()
}

/// Reports whether the text looks like a register name.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// `true` for short alphanumeric tokens that start with a letter.
fn is_register(text: &str) -> bool {
    let alpha = text.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    alpha && !text.is_empty() && text.len() <= 8 && text.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Extracts the first hexadecimal target address in operand text.
///
/// # Arguments
///
/// * `operands` - Operand text.
///
/// # Returns
///
/// The target address, or `None`.
fn target_of(operands: &str) -> Option<u64> {
    if let Some(start) = operands.find("0x") {
        return parse_hex_run(&operands[start + 2..]);
    }
    suffixed_target(operands)
}

/// Parses a leading run of hexadecimal digits.
///
/// # Arguments
///
/// * `text` - Text beginning with hexadecimal digits.
///
/// # Returns
///
/// The parsed value, or `None`.
fn parse_hex_run(text: &str) -> Option<u64> {
    let digits: String = text.chars().take_while(char::is_ascii_hexdigit).collect();
    u64::from_str_radix(&digits, 16).ok()
}

/// Parses an x86 `h`-suffixed hexadecimal target, such as `1400025C0h`.
///
/// # Arguments
///
/// * `operands` - Operand text.
///
/// # Returns
///
/// The parsed value, or `None`.
fn suffixed_target(operands: &str) -> Option<u64> {
    let token = operands.split(is_separator).find(|part| is_hex_token(part))?;
    u64::from_str_radix(token.trim_end_matches(['h', 'H']), 16).ok()
}

/// Reports whether a character separates operand tokens.
///
/// # Arguments
///
/// * `character` - Character to test.
///
/// # Returns
///
/// `true` when the character is not alphanumeric.
fn is_separator(character: char) -> bool {
    !character.is_ascii_alphanumeric()
}

/// Reports whether a token is a hexadecimal address literal.
///
/// # Arguments
///
/// * `token` - Operand token.
///
/// # Returns
///
/// `true` for tokens of four or more hexadecimal digits.
fn is_hex_token(token: &str) -> bool {
    let digits = token.trim_end_matches(['h', 'H']);
    digits.len() >= 4 && digits.chars().all(|c| c.is_ascii_hexdigit())
}

/// Splits operand text into its first two operands.
///
/// # Arguments
///
/// * `operands` - Operand text.
///
/// # Returns
///
/// The destination and source operands when present.
fn split_operands(operands: &str) -> (Option<&str>, Option<&str>, Option<&str>) {
    let cuts = top_level_commas(operands);
    let parts = slice_parts(operands, &cuts);
    let dst = parts.first().copied().and_then(nonempty);
    let src = parts.get(1).copied().and_then(nonempty);
    let extra = parts.get(2).copied().and_then(nonempty);
    (dst, src, extra)
}

/// Finds top-level comma positions, ignoring brackets.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// Byte indices of commas outside any `[]`, `()`, or `{}`.
fn top_level_commas(text: &str) -> Vec<usize> {
    let mut state = (0i32, Vec::new());
    for (index, c) in text.char_indices() {
        step_depth(&mut state, index, c);
    }
    state.1
}

/// Updates nesting depth and records top-level commas.
///
/// # Arguments
///
/// * `state` - Running depth and comma positions.
/// * `index` - Current byte index.
/// * `c` - Current character.
fn step_depth(state: &mut (i32, Vec<usize>), index: usize, c: char) {
    match c {
        '[' | '(' | '{' => state.0 += 1,
        ']' | ')' | '}' => state.0 -= 1,
        ',' if state.0 == 0 => state.1.push(index),
        _ => {}
    }
}

/// Slices text at the given byte positions.
///
/// # Arguments
///
/// * `text` - Source text.
/// * `cuts` - Byte indices of separators.
///
/// # Returns
///
/// The substrings between the separators.
fn slice_parts<'a>(text: &'a str, cuts: &[usize]) -> Vec<&'a str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for &cut in cuts {
        parts.push(&text[start..cut]);
        start = cut + 1;
    }
    parts.push(&text[start..]);
    parts
}

/// Returns trimmed operand text when non-empty.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The trimmed text, or `None` when empty.
fn nonempty(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Maps a conditional mnemonic to a readable comparison operator.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
///
/// # Returns
///
/// The comparison text.
fn condition_alias(mnemonic: &str) -> &'static str {
    match mnemonic {
        "je" | "jz" | "beq" | "beqz" | "jeq" => "==",
        "jne" | "jnz" | "bne" | "bnez" | "brne" => "!=",
        other => ordering_alias(other),
    }
}

/// Maps an ordering conditional mnemonic to a comparison operator.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
///
/// # Returns
///
/// The comparison text.
fn ordering_alias(mnemonic: &str) -> &'static str {
    match mnemonic {
        "jl" | "jlt" | "blt" | "bltz" | "brlt" | "jb" | "jnae" | "blo" | "brlo" => "<",
        "jle" | "ble" | "jbe" | "bls" | "brls" => "<=",
        "jg" | "jgt" | "bgt" | "bgtz" | "brgt" | "ja" | "jnbe" | "bhi" | "brhi" => ">",
        "jge" | "bge" | "bgez" | "brge" | "jae" | "jnb" | "bhs" | "brhs" => ">=",
        other => sign_alias(other),
    }
}

/// Maps a sign-test conditional mnemonic to a comparison operator.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
///
/// # Returns
///
/// The comparison text.
fn sign_alias(mnemonic: &str) -> &'static str {
    match mnemonic {
        "js" | "brmi" => "<0",
        "jns" | "brpl" => ">=0",
        _ => "flags",
    }
}

/// Joins a mnemonic and operands into fallback text.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `operands` - Operand text.
///
/// # Returns
///
/// The trimmed combined text.
fn joined(mnemonic: &str, operands: &str) -> String {
    format!("{mnemonic} {operands}").trim().to_string()
}

/// Joins a mnemonic and two operands into fallback text.
///
/// # Arguments
///
/// * `mnemonic` - Lowercased mnemonic.
/// * `dst` - Destination operand.
/// * `src` - Source operand.
///
/// # Returns
///
/// The trimmed combined text.
fn joined_pair(mnemonic: &str, dst: &str, src: &str) -> String {
    format!("{mnemonic} {dst}, {src}").trim().to_string()
}

/// Materializes the address of a memory operand.
///
/// # Arguments
///
/// * `text` - Operand text.
///
/// # Returns
///
/// The address expression without a dereference.
fn address_of(text: &str) -> Expr {
    let inner = bracketed(strip_size(text.trim()));
    inner.map_or_else(|| parse_expr(text), |sum| drop_zero(parse_sum(sum)))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

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
    fn line(address: u64, text: &str) -> InstructionLine {
        InstructionLine {
            address,
            text: text.to_string(),
            kind: String::new(),
            discussion: String::new(),
        }
    }

    /// Verifies that a move lifts to an assignment.
    #[test]
    fn test_lift_move_assignment() {
        let lifted = lift(&line(0x1000, "mov rax, rbx"), Arch::X86_64);
        let expected = Inst::Assign { dst: Expr::Reg("rax".into()), src: Expr::Reg("rbx".into()) };
        assert_eq!(lifted.inst, expected);
    }

    /// Verifies that a memory store lifts to a store instruction.
    #[test]
    fn test_lift_store() {
        let lifted = lift(&line(0x1000, "mov dword ptr [rbp+0x8], eax"), Arch::X86_64);
        match lifted.inst {
            Inst::Store { .. } => {}
            other => panic!("expected store, got {other:?}"),
        }
    }

    /// Verifies that a conditional branch captures its target.
    #[test]
    fn test_lift_conditional_branch() {
        let lifted = lift(&line(0x1000, "jne 0x140001234"), Arch::X86_64);
        assert_eq!(
            lifted.inst,
            Inst::JumpIf { cond: "!=".into(), target: 0x1_4000_1234 }
        );
    }

    /// Verifies that a direct call resolves to a named target.
    #[test]
    fn test_lift_call_named_target() {
        let lifted = lift(&line(0x1000, "call 0x140001770"), Arch::X86_64);
        assert_eq!(
            lifted.inst,
            Inst::Call { target: Expr::Raw("sub_140001770".into()) }
        );
    }

    /// Verifies that an ARM return lifts to a return.
    #[test]
    fn test_lift_thumb_return() {
        let lifted = lift(&line(0x1000, "bx lr"), Arch::Arm);
        assert_eq!(lifted.inst, Inst::Ret);
    }

    /// Verifies bracketed ARM operands are not split on the inner comma.
    #[test]
    fn test_split_operands_bracket() {
        assert_eq!(
            split_operands("r3,[r3,#0]"),
            (Some("r3"), Some("[r3,#0]"), None)
        );
    }

    /// Verifies an ARM load dereferences its base register.
    #[test]
    fn test_lift_arm_load() {
        let lifted = lift(&line(0x1000, "ldrb r3,[r3,#0]"), Arch::Arm);
        let expected = Inst::Assign {
            dst: Expr::Reg("r3".into()),
            src: Expr::Mem(Box::new(Expr::Reg("r3".into()))),
        };
        assert_eq!(lifted.inst, expected);
    }

    /// Verifies an ARM store orders memory as the destination.
    #[test]
    fn test_lift_arm_store_order() {
        let lifted = lift(&line(0x1000, "str r3,[r6,#0]"), Arch::Arm);
        let expected = Inst::Store { addr: Expr::Reg("r6".into()), src: Expr::Reg("r3".into()) };
        assert_eq!(lifted.inst, expected);
    }

    /// Verifies an ARM width suffix is stripped before matching.
    #[test]
    fn test_lift_arm_width_suffix() {
        let lifted = lift(&line(0x1000, "strb.w r3,[sp,#7]"), Arch::Arm);
        assert!(matches!(lifted.inst, Inst::Store { .. }));
    }

    /// Verifies that a Thumb conditional branch is recognized.
    #[test]
    fn test_lift_thumb_conditional() {
        let lifted = lift(&line(0x1000, "bne 0x10000300"), Arch::Arm);
        assert_eq!(
            lifted.inst,
            Inst::JumpIf { cond: "!=".into(), target: 0x1000_0300 }
        );
    }

    /// Verifies that a MIPS load lifts to a memory assignment.
    #[test]
    fn test_lift_mips_load() {
        let lifted = lift(&line(0x1000, "lw ra, 0x1c(sp)"), Arch::Mips);
        match lifted.inst {
            Inst::Assign { src: Expr::Mem(_), .. } => {}
            other => panic!("expected memory assignment, got {other:?}"),
        }
    }

    /// Verifies immediate parsing across bases.
    #[test]
    fn test_parse_imm_bases() {
        assert_eq!(parse_imm("0x2b"), Some(0x2b));
        assert_eq!(parse_imm("43"), Some(43));
        assert_eq!(parse_imm("zzz"), None);
    }

    /// Verifies parsing of x86 `h`-suffixed hexadecimal immediates.
    #[test]
    fn test_parse_imm_h_suffix() {
        assert_eq!(parse_imm("38h"), Some(0x38));
        assert_eq!(parse_imm("0FFFFFFFFh"), Some(0xffff_ffff));
    }

    /// Verifies target extraction from x86 `h`-suffixed branch operands.
    #[test]
    fn test_target_h_suffix() {
        assert_eq!(target_of("near ptr 0000000140001900h"), Some(0x1_4000_1900));
        assert_eq!(target_of("short 14000181Bh"), Some(0x1_4000_181b));
        assert_eq!(target_of("rax"), None);
    }
}
