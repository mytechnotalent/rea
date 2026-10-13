//! Ethereum Virtual Machine Bytecode Analysis
//!
//! **File:** `evm.rs`
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
use std::fmt::Write as _;

/// Simple opcode names for bytes `0x00` through `0x5f`.
const LOW: [&str; 96] = [
    "STOP",
    "ADD",
    "MUL",
    "SUB",
    "DIV",
    "SDIV",
    "MOD",
    "SMOD",
    "ADDMOD",
    "MULMOD",
    "EXP",
    "SIGNEXTEND",
    "",
    "",
    "",
    "",
    "LT",
    "GT",
    "SLT",
    "SGT",
    "EQ",
    "ISZERO",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "BYTE",
    "SHL",
    "SHR",
    "SAR",
    "",
    "",
    "SHA3",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "ADDRESS",
    "BALANCE",
    "ORIGIN",
    "CALLER",
    "CALLVALUE",
    "CALLDATALOAD",
    "CALLDATASIZE",
    "CALLDATACOPY",
    "CODESIZE",
    "CODECOPY",
    "GASPRICE",
    "EXTCODESIZE",
    "EXTCODECOPY",
    "RETURNDATASIZE",
    "RETURNDATACOPY",
    "EXTCODEHASH",
    "BLOCKHASH",
    "COINBASE",
    "TIMESTAMP",
    "NUMBER",
    "DIFFICULTY",
    "GASLIMIT",
    "CHAINID",
    "SELFBALANCE",
    "BASEFEE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "POP",
    "MLOAD",
    "MSTORE",
    "MSTORE8",
    "SLOAD",
    "SSTORE",
    "JUMP",
    "JUMPI",
    "PC",
    "MSIZE",
    "GAS",
    "JUMPDEST",
    "TLOAD",
    "TSTORE",
    "MCOPY",
    "PUSH0",
];

/// Simple opcode names for bytes `0xf0` through `0xff`.
const HIGH: [&str; 16] = [
    "CREATE",
    "CALL",
    "CALLCODE",
    "RETURN",
    "DELEGATECALL",
    "CREATE2",
    "",
    "",
    "",
    "",
    "STATICCALL",
    "",
    "",
    "REVERT",
    "INVALID",
    "SELFDESTRUCT",
];

/// Notable opcodes used for capability inference.
const NOTABLE: [(u8, &str); 12] = [
    (0x54, "SLOAD"),
    (0x55, "SSTORE"),
    (0x35, "CALLDATALOAD"),
    (0x34, "CALLVALUE"),
    (0xf1, "CALL"),
    (0xf4, "DELEGATECALL"),
    (0xfa, "STATICCALL"),
    (0xf0, "CREATE"),
    (0xf5, "CREATE2"),
    (0xff, "SELFDESTRUCT"),
    (0xa0, "LOG0"),
    (0xfd, "REVERT"),
];

/// A recovered function selector and its jump target.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvmSelector {
    /// Four-byte selector.
    pub selector: [u8; 4],
    /// Destination program counter.
    pub destination: u64,
}

/// Recovered EVM bytecode inventory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvmBytecode {
    /// Bytecode length in bytes.
    pub length: usize,
    /// Recovered function selectors.
    pub selectors: Vec<EvmSelector>,
    /// Disassembled instructions.
    pub opcodes: Vec<String>,
    /// Inferred mutability class.
    pub mutability: String,
    /// Notable capabilities.
    pub features: Vec<String>,
    /// Validation problems.
    pub warnings: Vec<String>,
}

/// Reports whether a buffer is plain hexadecimal text.
///
/// # Arguments
///
/// * `bytes` - Start of the target file.
///
/// # Returns
///
/// `true` when every non-prefix character is a hex digit.
#[must_use]
pub fn is_hex_text(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let trimmed = text.trim().trim_start_matches("0x").trim();
    !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_hexdigit())
}

/// Decodes a hexadecimal string into bytes.
///
/// # Arguments
///
/// * `text` - Hexadecimal text, with optional `0x` prefix.
///
/// # Returns
///
/// The decoded bytes, or `None` when malformed.
#[must_use]
pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    let body = text.trim().trim_start_matches("0x");
    let clean: String = body.chars().filter(char::is_ascii_hexdigit).collect();
    if !clean.len().is_multiple_of(2) {
        return None;
    }
    (0..clean.len() / 2).map(|i| decode_byte(&clean[i * 2..i * 2 + 2])).collect()
}

/// Decodes one two-digit hexadecimal byte.
///
/// # Arguments
///
/// * `pair` - Two hexadecimal characters.
///
/// # Returns
///
/// The decoded byte, or `None`.
fn decode_byte(pair: &str) -> Option<u8> {
    u8::from_str_radix(pair, 16).ok()
}

/// Resolves the immediate size of a push opcode.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The number of immediate bytes, or zero.
fn push_size(op: u8) -> u8 {
    if (0x60..=0x7f).contains(&op) {
        op - 0x5f
    } else {
        0
    }
}

/// Resolves an opcode name.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The mnemonic, or an empty string.
fn opcode_name(op: u8) -> String {
    ranged_name(op).unwrap_or_else(|| simple_name(op).to_string())
}

/// Resolves a ranged opcode name.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The ranged mnemonic, or `None`.
fn ranged_name(op: u8) -> Option<String> {
    match op {
        0x60..=0x7f => Some(format!("PUSH{}", op - 0x5f)),
        0x80..=0x8f => Some(format!("DUP{}", op - 0x7f)),
        0x90..=0x9f => Some(format!("SWAP{}", op - 0x8f)),
        0xa0..=0xa4 => Some(format!("LOG{}", op - 0xa0)),
        _ => None,
    }
}

/// Resolves a simple opcode name.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The simple mnemonic, or an empty string.
fn simple_name(op: u8) -> &'static str {
    match op {
        0x00..=0x5f => LOW.get(usize::from(op)).copied().unwrap_or(""),
        0xf0..=0xff => HIGH.get(usize::from(op - 0xf0)).copied().unwrap_or(""),
        _ => "",
    }
}

/// Disassembles EVM bytecode.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
///
/// # Returns
///
/// One text line per instruction.
fn disassemble(bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        pos = disasm_one(bytes, pos, &mut out);
    }
    out
}

/// Disassembles one instruction.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
/// * `pos` - Instruction offset.
/// * `out` - Accumulated lines.
///
/// # Returns
///
/// The next instruction offset.
fn disasm_one(bytes: &[u8], pos: usize, out: &mut Vec<String>) -> usize {
    let op = bytes[pos];
    let size = usize::from(push_size(op));
    let imm = bytes.get(pos + 1..pos + 1 + size).unwrap_or_default();
    out.push(format!("{pos:04x}: {}", instruction_text(op, imm)));
    pos + 1 + size
}

/// Formats one instruction mnemonic with its immediate.
///
/// # Arguments
///
/// * `op` - Opcode byte.
/// * `imm` - Immediate bytes.
///
/// # Returns
///
/// The instruction text.
fn instruction_text(op: u8, imm: &[u8]) -> String {
    let name = opcode_name(op);
    if imm.is_empty() {
        name
    } else {
        format!("{name} 0x{}", hex(imm))
    }
}

/// Renders bytes as lowercase hexadecimal.
///
/// # Arguments
///
/// * `bytes` - Bytes to render.
///
/// # Returns
///
/// The hexadecimal string.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Extracts function selectors from a dispatch table.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
///
/// # Returns
///
/// Every selector and its destination.
fn extract_selectors(bytes: &[u8]) -> Vec<EvmSelector> {
    (0..bytes.len()).filter_map(|pos| match_selector(bytes, pos)).collect()
}

/// Matches a `PUSH4 <selector> EQ PUSHn <dest> JUMPI` dispatch arm.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
/// * `pos` - Candidate offset.
///
/// # Returns
///
/// The selector and destination, or `None`.
fn match_selector(bytes: &[u8], pos: usize) -> Option<EvmSelector> {
    let window = bytes.get(pos..pos + 7)?;
    if window[0] != 0x63 || window[5] != 0x14 {
        return None;
    }
    let destination = selector_dest(bytes, pos + 6)?;
    Some(EvmSelector { selector: [window[1], window[2], window[3], window[4]], destination })
}

/// Reads the jump destination of a dispatch arm.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
/// * `pos` - Offset of the push opcode.
///
/// # Returns
///
/// The destination, or `None`.
fn selector_dest(bytes: &[u8], pos: usize) -> Option<u64> {
    let op = *bytes.get(pos)?;
    let size = usize::from(push_size(op));
    if size == 0 || size > 2 {
        return None;
    }
    let imm = bytes.get(pos + 1..pos + 1 + size)?;
    Some(u64::from_be_bytes(padded8(imm)))
}

/// Right-aligns up to eight bytes into a word.
///
/// # Arguments
///
/// * `bytes` - Source bytes.
///
/// # Returns
///
/// The padded word.
fn padded8(bytes: &[u8]) -> [u8; 8] {
    let mut buf = [0u8; 8];
    let start = 8 - bytes.len().min(8);
    buf[start..].copy_from_slice(&bytes[bytes.len().saturating_sub(8)..]);
    buf
}

/// Scans bytecode for notable capabilities.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
///
/// # Returns
///
/// Distinct notable capability names.
fn scan_features(bytes: &[u8]) -> Vec<String> {
    let mut seen = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        record_op(bytes[pos], &mut seen);
        pos += 1 + usize::from(push_size(bytes[pos]));
    }
    seen
}

/// Records a notable opcode into the capability list.
///
/// # Arguments
///
/// * `op` - Opcode byte.
/// * `seen` - Accumulated capabilities.
fn record_op(op: u8, seen: &mut Vec<String>) {
    if let Some(name) = notable(op) {
        if !seen.iter().any(|s| s == name) {
            seen.push(name.to_string());
        }
    }
}

/// Resolves the notable name of an opcode.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The capability name, or `None`.
fn notable(op: u8) -> Option<&'static str> {
    NOTABLE.iter().find(|(code, _)| *code == op).map(|(_, name)| *name)
}

/// Infers a mutability class from capabilities.
///
/// # Arguments
///
/// * `features` - Recovered capabilities.
///
/// # Returns
///
/// One of `pure`, `view`, or `nonpayable`.
fn infer_mutability(features: &[String]) -> String {
    let has = |name: &str| features.iter().any(|f| f == name);
    let state = ["SSTORE", "CREATE", "CREATE2", "SELFDESTRUCT"].iter().any(|n| has(n));
    match (state, has("SLOAD")) {
        (true, _) => "nonpayable",
        (false, true) => "view",
        (false, false) => "pure",
    }
    .to_string()
}

/// Parses EVM bytecode into a recovered inventory.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
///
/// # Returns
///
/// The recovered inventory.
#[must_use]
pub fn parse_evm(bytes: &[u8]) -> EvmBytecode {
    let features = scan_features(bytes);
    build_evm(bytes, features)
}

/// Builds the recovered inventory from bytecode and capabilities.
///
/// # Arguments
///
/// * `bytes` - Bytecode.
/// * `features` - Recovered capabilities.
///
/// # Returns
///
/// The recovered inventory.
fn build_evm(bytes: &[u8], features: Vec<String>) -> EvmBytecode {
    EvmBytecode {
        length: bytes.len(),
        selectors: extract_selectors(bytes),
        opcodes: disassemble(bytes),
        mutability: infer_mutability(&features),
        features,
        ..Default::default()
    }
}

/// Formats an EVM inventory as a human-readable text block.
///
/// # Arguments
///
/// * `evm` - Recovered inventory.
///
/// # Returns
///
/// Formatted summary string.
#[must_use]
pub fn format_evm_summary(evm: &EvmBytecode) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "EVM bytecode: {} bytes", evm.length);
    let _ = writeln!(out, "Mutability: {}", evm.mutability);
    let _ = writeln!(out, "Selectors: {}", evm.selectors.len());
    let _ = writeln!(out, "Features: {}", evm.features.len());
    append_selectors(&mut out, &evm.selectors);
    out
}

/// Appends selector dispatch lines.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `selectors` - Recovered selectors.
fn append_selectors(out: &mut String, selectors: &[EvmSelector]) {
    for selector in selectors {
        let hex_selector = hex(&selector.selector);
        let _ = writeln!(out, "  0x{hex_selector} -> {:#06x}", selector.destination);
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies hexadecimal decoding and detection.
    #[test]
    fn test_evm_hex() {
        assert!(is_hex_text(b"0x6001600101"));
        assert_eq!(decode_hex("0x6001"), Some(vec![0x60, 0x01]));
        assert_eq!(decode_hex("601"), None);
    }

    /// Verifies push sizing and opcode names.
    #[test]
    fn test_evm_opcodes() {
        assert_eq!(push_size(0x60), 1);
        assert_eq!(push_size(0x7f), 32);
        assert_eq!(opcode_name(0x01), "ADD");
        assert_eq!(opcode_name(0x61), "PUSH2");
        assert_eq!(opcode_name(0x80), "DUP1");
    }

    /// Verifies selector dispatch extraction.
    #[test]
    fn test_evm_selectors() {
        let mut bytes = vec![0x63];
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        bytes.extend_from_slice(&[0x14, 0x61, 0x00, 0x10, 0x57]);
        let selectors = extract_selectors(&bytes);
        assert_eq!(selectors.len(), 1);
        assert_eq!(selectors[0].selector, [0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(selectors[0].destination, 0x10);
    }

    /// Verifies feature scanning and mutability inference.
    #[test]
    fn test_evm_features() {
        let bytes = [0x54, 0x55, 0x01];
        let evm = parse_evm(&bytes);
        assert!(evm.features.contains(&"SSTORE".to_string()));
        assert_eq!(evm.mutability, "nonpayable");
    }

    /// Verifies summary formatting.
    #[test]
    fn test_evm_summary() {
        let evm = parse_evm(&[0x00]);
        assert!(format_evm_summary(&evm).contains("EVM bytecode"));
    }
}
