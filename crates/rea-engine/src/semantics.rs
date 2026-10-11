//! Semantic Instruction Analysis and Disambiguation
//!
//! **File:** `semantics.rs`
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
use std::fmt::Write;

// Import crate items
use crate::{
    disasm::{DataEntry, Function},
    model::Arch,
};

/// Normalizes decimal literals into clean hexadecimal numbers.
///
/// # Arguments
///
/// * `text` - Raw disassembly instruction text.
///
/// # Returns
///
/// Disassembly text with numbers >= 10 formatted in hexadecimal.
#[must_use]
pub fn format_imm_hex(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        process_char(c, &mut chars, &mut out);
    }
    out
}

/// Processes one character during immediate normalization.
///
/// # Arguments
///
/// * `c` - Current character.
/// * `chars` - Peekable iterator of remaining characters.
/// * `out` - Destination string buffer.
fn process_char(c: char, chars: &mut std::iter::Peekable<std::str::Chars>, out: &mut String) {
    if c.is_ascii_digit() {
        let num_str = collect_digits(c, chars);
        append_num(out, &num_str);
    } else {
        out.push(c);
    }
}

/// Collects consecutive digit characters from the stream.
///
/// # Arguments
///
/// * `first` - Initial digit character.
/// * `chars` - Peekable iterator of remaining characters.
///
/// # Returns
///
/// Collected string of digits.
fn collect_digits(first: char, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut s = String::from(first);
    while chars.peek().is_some_and(char::is_ascii_digit) {
        if let Some(d) = chars.next() {
            s.push(d);
        }
    }
    s
}

/// Appends a parsed numeric token formatted in hex if >= 10.
///
/// # Arguments
///
/// * `out` - Destination string buffer.
/// * `num_str` - Digit token string.
fn append_num(out: &mut String, num_str: &str) {
    match num_str.parse::<u64>() {
        Ok(v) if v >= 10 => {
            let _ = write!(out, "0x{v:x}");
        }
        _ => out.push_str(num_str),
    }
}

/// Classifies an instruction mnemonic into a functional category.
///
/// # Arguments
///
/// * `mnemonic` - Base instruction mnemonic.
///
/// # Returns
///
/// Category label string slice.
#[must_use]
pub fn classify_mnemonic(mnemonic: &str) -> &'static str {
    let m = mnemonic.to_ascii_lowercase();
    if is_stack_op(&m) {
        "Stack Frame"
    } else {
        classify_non_stack(&m)
    }
}

/// Classifies non-stack instruction mnemonics.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// Category label string slice.
fn classify_non_stack(m: &str) -> &'static str {
    if let Some(mem) = check_mem_op(m) {
        mem
    } else if is_branch_mnemonic(m) {
        "Control Flow"
    } else {
        classify_calc_or_system(m)
    }
}

/// Reports whether mnemonic is a stack operation.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// `true` for push or pop.
fn is_stack_op(m: &str) -> bool {
    m.starts_with("push") || m.starts_with("pop")
}

/// Categorizes memory load, store, or move operations.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// Category if matched.
fn check_mem_op(m: &str) -> Option<&'static str> {
    if m.starts_with("ldr") || (m.starts_with("mov") && m.contains('[')) {
        Some("Memory Load")
    } else if m.starts_with("str") {
        Some("Memory Store")
    } else {
        check_move_op(m)
    }
}

/// Categorizes register move operations.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// Category if matched.
fn check_move_op(m: &str) -> Option<&'static str> {
    (m.starts_with("mov") || m.starts_with("mvn")).then_some("Register Move")
}

/// Categorizes arithmetic, logic, or general system operations.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// Category name.
fn classify_calc_or_system(m: &str) -> &'static str {
    if is_alu_mnemonic(m) {
        "Arithmetic / Logic"
    } else {
        "General / System"
    }
}

/// Reports whether a mnemonic represents a branch or call.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// `true` for branch, jump, call, or return instructions.
fn is_branch_mnemonic(m: &str) -> bool {
    m.starts_with('b')
        || m.starts_with("call")
        || m.starts_with("jmp")
        || m.starts_with("ret")
        || m.starts_with("cb")
}

/// Reports whether a mnemonic represents arithmetic or logic.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// `true` for arithmetic, logic, comparison, or shift operations.
fn is_alu_mnemonic(m: &str) -> bool {
    is_basic_arith(m) || is_logic_or_shift(m)
}

/// Reports whether a mnemonic is basic arithmetic or comparison.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// `true` for add, sub, mul, or cmp.
fn is_basic_arith(m: &str) -> bool {
    m.starts_with("add") || m.starts_with("sub") || m.starts_with("mul") || m.starts_with("cmp")
}

/// Reports whether a mnemonic is logic or shift.
///
/// # Arguments
///
/// * `m` - Lowercase mnemonic.
///
/// # Returns
///
/// `true` for logic or shift instructions.
fn is_logic_or_shift(m: &str) -> bool {
    m.starts_with("and")
        || m.starts_with("orr")
        || m.starts_with("eor")
        || m.starts_with("xor")
        || m.starts_with("tst")
        || m.starts_with("lsl")
        || m.starts_with("lsr")
}

/// Produces discussion for stack frame operations.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Semantic discussion string.
fn discuss_stack(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if lower.contains("push") {
        "Function prologue: saves callee registers and return address to stack frame.".into()
    } else if lower.contains("pop") {
        discuss_pop(&lower)
    } else {
        discuss_stack_adjust(&lower)
    }
}

/// Discusses pop operations.
///
/// # Arguments
///
/// * `lower` - Lowercase instruction text.
///
/// # Returns
///
/// Discussion string.
fn discuss_pop(lower: &str) -> String {
    if lower.contains("pc") || lower.contains("ret") {
        "Function epilogue: restores registers and returns control to caller.".into()
    } else {
        "Restores preserved register state from stack frame.".into()
    }
}

/// Discusses stack adjustment operations.
///
/// # Arguments
///
/// * `lower` - Lowercase instruction text.
///
/// # Returns
///
/// Discussion string.
fn discuss_stack_adjust(lower: &str) -> String {
    if lower.contains("sub") && lower.contains("sp") {
        "Allocates local stack frame storage for variables and buffers.".into()
    } else {
        "Adjusts stack frame pointer for local storage.".into()
    }
}

/// Produces discussion for memory load operations.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Semantic discussion string.
fn discuss_load(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if lower.contains("[sp") {
        "Loads local variable from stack frame storage slot into register.".into()
    } else if lower.contains("[pc") {
        "Loads literal pool constant or target address relative to program counter.".into()
    } else {
        discuss_general_load(text)
    }
}

/// Discusses MMIO or general loads.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Discussion string.
fn discuss_general_load(text: &str) -> String {
    if is_mmio_text(text) {
        "Reads hardware status or data from memory-mapped peripheral I/O register.".into()
    } else {
        "Loads memory value into register.".into()
    }
}

/// Reports whether text references memory-mapped I/O ranges.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// `true` when peripheral address patterns are detected.
fn is_mmio_text(text: &str) -> bool {
    text.contains("0x400")
        || text.contains("0x500")
        || text.contains("0xd00")
        || text.contains("0xe00")
}

/// Produces discussion for memory store operations.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Semantic discussion string.
fn discuss_store(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if lower.contains("[sp") {
        "Stores local variable from register into stack frame storage slot.".into()
    } else if is_mmio_text(text) {
        "Writes control word or data to memory-mapped hardware peripheral register.".into()
    } else {
        "Stores register value into target memory address.".into()
    }
}

/// Produces discussion for control flow operations.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Semantic discussion string.
fn discuss_branch(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if is_call_or_ret(&lower) {
        discuss_call_or_ret(&lower)
    } else if lower.starts_with("cb") {
        discuss_compare_branch(&lower)
    } else {
        discuss_direct_branch(&lower)
    }
}

/// Reports whether mnemonic is a call or return.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// `true` for call or return.
fn is_call_or_ret(lower: &str) -> bool {
    lower.starts_with("bl ")
        || lower.starts_with("blx")
        || lower.starts_with("call")
        || lower.starts_with("bx lr")
        || lower.starts_with("ret")
}

/// Discusses call or return instructions.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_call_or_ret(lower: &str) -> String {
    if lower.starts_with("bx lr") || lower.starts_with("ret") {
        "Function return: branches to return address held in link register.".into()
    } else {
        "Subroutine call: branches to target function saving return address in link register."
            .into()
    }
}

/// Discusses compare-and-branch instructions.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_compare_branch(lower: &str) -> String {
    if lower.starts_with("cbz") {
        "Compare and branch on zero: conditional jump guarded on register zero value.".into()
    } else {
        "Compare and branch on non-zero: conditional jump guarded on non-zero register value."
            .into()
    }
}

/// Discusses unconditional or conditional branches.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_direct_branch(lower: &str) -> String {
    if (lower.starts_with('b') && !lower.starts_with("bl")) || lower.starts_with("jmp") {
        "Unconditional branch transferring execution flow to target block or loop.".into()
    } else {
        "Conditional branch evaluated from arithmetic flags.".into()
    }
}

/// Produces discussion for arithmetic, logic, and comparison operations.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// Semantic discussion string.
fn discuss_alu(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if is_compare_or_add(&lower) {
        discuss_cmp_add(&lower)
    } else if is_sub_or_mul(&lower) {
        discuss_sub_mul(&lower)
    } else {
        discuss_bitwise(&lower)
    }
}

/// Reports whether operation is compare or addition.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// `true` for cmp, tst, or add.
fn is_compare_or_add(lower: &str) -> bool {
    lower.starts_with("cmp") || lower.starts_with("tst") || lower.starts_with("add")
}

/// Discusses compare or addition.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_cmp_add(lower: &str) -> String {
    if lower.starts_with("add") {
        "Performs arithmetic addition and updates destination register.".into()
    } else {
        "Compares operands and updates processor ALU condition flags.".into()
    }
}

/// Reports whether operation is subtraction or multiplication.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// `true` for sub or mul.
fn is_sub_or_mul(lower: &str) -> bool {
    lower.starts_with("sub") || lower.starts_with("mul")
}

/// Discusses subtraction or multiplication.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_sub_mul(lower: &str) -> String {
    if lower.starts_with("sub") {
        "Performs arithmetic subtraction and updates destination register.".into()
    } else {
        "Performs integer multiplication of operands.".into()
    }
}

/// Discusses bitwise or other ALU computations.
///
/// # Arguments
///
/// * `lower` - Lowercase text.
///
/// # Returns
///
/// Discussion string.
fn discuss_bitwise(lower: &str) -> String {
    if lower.starts_with("and") || lower.starts_with("bic") {
        "Bitwise masking operation on register operand.".into()
    } else if lower.starts_with("orr") || lower.starts_with("eor") || lower.starts_with("xor") {
        "Bitwise logic operation updating register bits.".into()
    } else {
        "Arithmetic or shift computation on register operands.".into()
    }
}

/// Analyzes an instruction and returns its category and semantic discussion.
///
/// # Arguments
///
/// * `_arch` - Target architecture.
/// * `_address` - Virtual memory address.
/// * `text` - Disassembly text.
///
/// # Returns
///
/// Tuple of `(Category, Discussion)`.
#[must_use]
pub fn discuss(_arch: Arch, _address: u64, text: &str) -> (String, String) {
    let mnemonic = text.split_whitespace().next().unwrap_or_default();
    let kind = classify_mnemonic(mnemonic).to_string();
    let discussion = category_discussion(kind.as_str(), text);
    (kind, discussion)
}

/// Maps category and text to semantic discussion.
///
/// # Arguments
///
/// * `category` - Functional category.
/// * `text` - Raw instruction text.
///
/// # Returns
///
/// Discussion string.
fn category_discussion(category: &str, text: &str) -> String {
    match category {
        "Stack Frame" => discuss_stack(text),
        "Memory Load" => discuss_load(text),
        "Memory Store" => discuss_store(text),
        "Control Flow" => discuss_branch(text),
        "Arithmetic / Logic" => discuss_alu(text),
        _ => discuss_other_category(category),
    }
}

/// Discusses register moves or general system operations.
///
/// # Arguments
///
/// * `category` - Functional category.
///
/// # Returns
///
/// Discussion string.
fn discuss_other_category(category: &str) -> String {
    if category == "Register Move" {
        "Copies immediate value or register into destination register.".into()
    } else {
        "Executes hardware instruction or processor control operation.".into()
    }
}

/// Analyzes a 32-bit data word evaluating data and candidate instruction interpretations.
///
/// # Arguments
///
/// * `_arch` - Target architecture.
/// * `address` - Address where the data resides.
/// * `val` - 32-bit value.
///
/// # Returns
///
/// Data entry describing the value in both data and candidate code representations.
#[must_use]
pub fn analyze_data_word(_arch: Arch, address: u64, val: u32) -> DataEntry {
    DataEntry {
        address,
        bytes_hex: format_le_hex(val),
        as_data: format_data_val(val),
        as_code: None,
        discussion: format_data_discussion(val),
    }
}

/// Formats 4 bytes in little-endian hex space-separated.
///
/// # Arguments
///
/// * `val` - 32-bit integer.
///
/// # Returns
///
/// Hex string.
fn format_le_hex(val: u32) -> String {
    format!(
        "{:02x} {:02x} {:02x} {:02x}",
        val & 0xff,
        (val >> 8) & 0xff,
        (val >> 16) & 0xff,
        (val >> 24) & 0xff
    )
}

/// Formats data representation of a 32-bit word.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Description string.
fn format_data_val(val: u32) -> String {
    if let Some(desc) = check_hw_data_val(val) {
        desc
    } else {
        check_mem_data_val(val)
    }
}

/// Checks hardware MMIO ranges.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Hardware description if matched.
fn check_hw_data_val(val: u32) -> Option<String> {
    if (0xd000_0000..=0xd000_0200).contains(&val) {
        Some(format!("0x{val:08x} [RP2350 SIO Base Register]"))
    } else if (0x4000_0000..=0x5008_0000).contains(&val) {
        Some(format!("0x{val:08x} [Hardware Peripheral MMIO]"))
    } else {
        None
    }
}

/// Checks memory ranges (Flash, SRAM) or returns integer constant.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Memory description or integer constant.
fn check_mem_data_val(val: u32) -> String {
    if (0x1000_0000..=0x1fff_ffff).contains(&val) {
        format!("0x{val:08x} [Flash Pointer / Code Target]")
    } else if (0x2000_0000..=0x2008_2000).contains(&val) {
        format!("0x{val:08x} [SRAM Memory Address]")
    } else {
        format!("0x{val:08x} [Integer Constant]")
    }
}

/// Produces security discussion for a 32-bit data word.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Discussion string.
fn format_data_discussion(val: u32) -> String {
    if let Some(desc) = check_hw_data_disc(val) {
        desc
    } else {
        check_mem_data_disc(val)
    }
}

/// Checks hardware MMIO discussion.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Hardware discussion if matched.
fn check_hw_data_disc(val: u32) -> Option<String> {
    if (0xd000_0000..=0xd000_0200).contains(&val) {
        Some("Literal pool holds base address of RP2350 Single-Cycle IO peripheral block.".into())
    } else if (0x4000_0000..=0x5008_0000).contains(&val) {
        Some("Literal pool holds hardware peripheral MMIO control address.".into())
    } else {
        None
    }
}

/// Checks memory discussion (Flash, SRAM) or immediate mask.
///
/// # Arguments
///
/// * `val` - 32-bit word.
///
/// # Returns
///
/// Memory or immediate discussion.
fn check_mem_data_disc(val: u32) -> String {
    if (0x1000_0000..=0x1fff_ffff).contains(&val) {
        "Literal pool holds pointer to flash code target or callback routine.".into()
    } else if (0x2000_0000..=0x2008_2000).contains(&val) {
        "Literal pool holds RAM buffer pointer for data structure manipulation.".into()
    } else {
        "Literal pool holds numeric immediate constant or configuration mask.".into()
    }
}

/// Synthesizes a cohesive behavioral summary for an analyzed function.
///
/// # Arguments
///
/// * `func` - Function to summarize.
///
/// # Returns
///
/// High-level behavioral walkthrough summary.
#[must_use]
pub fn summarize_function(func: &Function) -> String {
    let mut parts = vec![func_intro(func)];
    describe_calls(func, &mut parts);
    describe_stack_vars(func, &mut parts);
    parts.join(" ")
}

/// Builds the opening sentence of a function behavioral summary.
///
/// # Arguments
///
/// * `func` - Function being summarized.
///
/// # Returns
///
/// Introductory sentence string.
fn func_intro(func: &Function) -> String {
    let name = func.name.as_deref().unwrap_or("subroutine");
    let count = func.lines.len().max(func.instructions);
    format!("Routine `{name}` contains {count} instructions.")
}

/// Appends description of subroutine calls made by the function.
///
/// # Arguments
///
/// * `func` - Function being summarized.
/// * `parts` - Output sentence parts.
fn describe_calls(func: &Function, parts: &mut Vec<String>) {
    if func.calls.is_empty() {
        parts.push("Operates as a leaf function with no outbound subroutine calls.".into());
    } else {
        parts.push(format_call_targets(func));
    }
}

/// Formats a description of outbound subroutine call targets.
///
/// # Arguments
///
/// * `func` - Function being summarized.
///
/// # Returns
///
/// Formatted sentence string.
fn format_call_targets(func: &Function) -> String {
    let targets: Vec<String> = func.calls.iter().take(3).map(|c| format!("0x{c:x}")).collect();
    let sample = targets.join(", ");
    format!(
        "Invokes {} subroutines including {sample}.",
        func.calls.len()
    )
}

/// Appends description of stack frame and local variables in the function.
///
/// # Arguments
///
/// * `func` - Function being summarized.
/// * `parts` - Output sentence parts.
fn describe_stack_vars(func: &Function, parts: &mut Vec<String>) {
    let has_stack = func.lines.iter().any(|l| l.text.contains("[sp"));
    if has_stack {
        parts.push("Allocates and manipulates local variables within its stack frame.".into());
    } else {
        parts.push("Maintains state predominantly within processor registers.".into());
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies hexadecimal immediate normalization.
    #[test]
    fn test_format_imm_hex() {
        assert_eq!(format_imm_hex("mov r0, #268435818"), "mov r0, #0x1000016a");
        assert_eq!(format_imm_hex("mov r0, 5"), "mov r0, 5");
    }

    /// Verifies classification of stack mnemonics.
    #[test]
    fn test_classify_stack() {
        assert_eq!(classify_mnemonic("push"), "Stack Frame");
        assert_eq!(classify_mnemonic("pop"), "Stack Frame");
    }

    /// Verifies classification of memory load and store.
    #[test]
    fn test_classify_mem() {
        assert_eq!(classify_mnemonic("ldr"), "Memory Load");
        assert_eq!(classify_mnemonic("str"), "Memory Store");
    }

    /// Verifies function summarization.
    #[test]
    fn test_summarize_function() {
        let func = Function::at(0x1000, Some("test_fn".into()));
        let summary = summarize_function(&func);
        assert!(summary.contains("test_fn"));
        assert!(summary.contains("leaf function"));
    }
}
