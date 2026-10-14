//! Intel 8051 (MCS-51) Disassembly
//!
//! **File:** `i8051.rs`
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

/// Opcode templates and byte lengths for the base 8051 instruction set.
///
/// Tokens: `$1` first operand byte, `$2` second operand byte, `$w` big-endian
/// 16-bit address, `$r` relative branch target, `$a` 11-bit page address.
const OPS: [(&str, u8); 256] = [
    ("NOP", 1),
    ("AJMP $a", 2),
    ("LJMP $w", 3),
    ("RR A", 1),
    ("INC A", 1),
    ("INC $1", 2),
    ("INC @R0", 1),
    ("INC @R1", 1),
    ("INC R0", 1),
    ("INC R1", 1),
    ("INC R2", 1),
    ("INC R3", 1),
    ("INC R4", 1),
    ("INC R5", 1),
    ("INC R6", 1),
    ("INC R7", 1),
    ("JBC $1,$r", 3),
    ("ACALL $a", 2),
    ("LCALL $w", 3),
    ("RRC A", 1),
    ("DEC A", 1),
    ("DEC $1", 2),
    ("DEC @R0", 1),
    ("DEC @R1", 1),
    ("DEC R0", 1),
    ("DEC R1", 1),
    ("DEC R2", 1),
    ("DEC R3", 1),
    ("DEC R4", 1),
    ("DEC R5", 1),
    ("DEC R6", 1),
    ("DEC R7", 1),
    ("JB $1,$r", 3),
    ("AJMP $a", 2),
    ("RET", 1),
    ("RL A", 1),
    ("ADD A,#$1", 2),
    ("ADD A,$1", 2),
    ("ADD A,@R0", 1),
    ("ADD A,@R1", 1),
    ("ADD A,R0", 1),
    ("ADD A,R1", 1),
    ("ADD A,R2", 1),
    ("ADD A,R3", 1),
    ("ADD A,R4", 1),
    ("ADD A,R5", 1),
    ("ADD A,R6", 1),
    ("ADD A,R7", 1),
    ("JNB $1,$r", 3),
    ("ACALL $a", 2),
    ("RETI", 1),
    ("RLC A", 1),
    ("ADDC A,#$1", 2),
    ("ADDC A,$1", 2),
    ("ADDC A,@R0", 1),
    ("ADDC A,@R1", 1),
    ("ADDC A,R0", 1),
    ("ADDC A,R1", 1),
    ("ADDC A,R2", 1),
    ("ADDC A,R3", 1),
    ("ADDC A,R4", 1),
    ("ADDC A,R5", 1),
    ("ADDC A,R6", 1),
    ("ADDC A,R7", 1),
    ("JC $r", 2),
    ("AJMP $a", 2),
    ("ORL $1,A", 2),
    ("ORL $1,#$2", 3),
    ("ORL A,#$1", 2),
    ("ORL A,$1", 2),
    ("ORL A,@R0", 1),
    ("ORL A,@R1", 1),
    ("ORL A,R0", 1),
    ("ORL A,R1", 1),
    ("ORL A,R2", 1),
    ("ORL A,R3", 1),
    ("ORL A,R4", 1),
    ("ORL A,R5", 1),
    ("ORL A,R6", 1),
    ("ORL A,R7", 1),
    ("JNC $r", 2),
    ("ACALL $a", 2),
    ("ANL $1,A", 2),
    ("ANL $1,#$2", 3),
    ("ANL A,#$1", 2),
    ("ANL A,$1", 2),
    ("ANL A,@R0", 1),
    ("ANL A,@R1", 1),
    ("ANL A,R0", 1),
    ("ANL A,R1", 1),
    ("ANL A,R2", 1),
    ("ANL A,R3", 1),
    ("ANL A,R4", 1),
    ("ANL A,R5", 1),
    ("ANL A,R6", 1),
    ("ANL A,R7", 1),
    ("JZ $r", 2),
    ("AJMP $a", 2),
    ("XRL $1,A", 2),
    ("XRL $1,#$2", 3),
    ("XRL A,#$1", 2),
    ("XRL A,$1", 2),
    ("XRL A,@R0", 1),
    ("XRL A,@R1", 1),
    ("XRL A,R0", 1),
    ("XRL A,R1", 1),
    ("XRL A,R2", 1),
    ("XRL A,R3", 1),
    ("XRL A,R4", 1),
    ("XRL A,R5", 1),
    ("XRL A,R6", 1),
    ("XRL A,R7", 1),
    ("JNZ $r", 2),
    ("ACALL $a", 2),
    ("ORL C,$1", 2),
    ("JMP @A+DPTR", 1),
    ("MOV A,#$1", 2),
    ("MOV $1,#$2", 3),
    ("MOV @R0,#$1", 2),
    ("MOV @R1,#$1", 2),
    ("MOV R0,#$1", 2),
    ("MOV R1,#$1", 2),
    ("MOV R2,#$1", 2),
    ("MOV R3,#$1", 2),
    ("MOV R4,#$1", 2),
    ("MOV R5,#$1", 2),
    ("MOV R6,#$1", 2),
    ("MOV R7,#$1", 2),
    ("SJMP $r", 2),
    ("AJMP $a", 2),
    ("ANL C,$1", 2),
    ("MOVC A,@A+PC", 1),
    ("DIV AB", 1),
    ("MOV $1,$2", 3),
    ("MOV $1,@R0", 2),
    ("MOV $1,@R1", 2),
    ("MOV $1,R0", 2),
    ("MOV $1,R1", 2),
    ("MOV $1,R2", 2),
    ("MOV $1,R3", 2),
    ("MOV $1,R4", 2),
    ("MOV $1,R5", 2),
    ("MOV $1,R6", 2),
    ("MOV $1,R7", 2),
    ("MOV DPTR,#$w", 3),
    ("ACALL $a", 2),
    ("MOV $1,C", 2),
    ("MOVC A,@A+DPTR", 1),
    ("SUBB A,#$1", 2),
    ("SUBB A,$1", 2),
    ("SUBB A,@R0", 1),
    ("SUBB A,@R1", 1),
    ("SUBB A,R0", 1),
    ("SUBB A,R1", 1),
    ("SUBB A,R2", 1),
    ("SUBB A,R3", 1),
    ("SUBB A,R4", 1),
    ("SUBB A,R5", 1),
    ("SUBB A,R6", 1),
    ("SUBB A,R7", 1),
    ("ORL C,/$1", 2),
    ("AJMP $a", 2),
    ("MOV C,$1", 2),
    ("INC DPTR", 1),
    ("MUL AB", 1),
    ("RESERVED", 1),
    ("MOV @R0,$1", 2),
    ("MOV @R1,$1", 2),
    ("MOV R0,$1", 2),
    ("MOV R1,$1", 2),
    ("MOV R2,$1", 2),
    ("MOV R3,$1", 2),
    ("MOV R4,$1", 2),
    ("MOV R5,$1", 2),
    ("MOV R6,$1", 2),
    ("MOV R7,$1", 2),
    ("ANL C,/$1", 2),
    ("ACALL $a", 2),
    ("CPL $1", 2),
    ("CPL C", 1),
    ("CJNE A,#$1,$r", 3),
    ("CJNE A,$1,$r", 3),
    ("CJNE @R0,#$1,$r", 3),
    ("CJNE @R1,#$1,$r", 3),
    ("CJNE R0,#$1,$r", 3),
    ("CJNE R1,#$1,$r", 3),
    ("CJNE R2,#$1,$r", 3),
    ("CJNE R3,#$1,$r", 3),
    ("CJNE R4,#$1,$r", 3),
    ("CJNE R5,#$1,$r", 3),
    ("CJNE R6,#$1,$r", 3),
    ("CJNE R7,#$1,$r", 3),
    ("PUSH $1", 2),
    ("AJMP $a", 2),
    ("CLR $1", 2),
    ("CLR C", 1),
    ("SWAP A", 1),
    ("XCH A,$1", 2),
    ("XCH A,@R0", 1),
    ("XCH A,@R1", 1),
    ("XCH A,R0", 1),
    ("XCH A,R1", 1),
    ("XCH A,R2", 1),
    ("XCH A,R3", 1),
    ("XCH A,R4", 1),
    ("XCH A,R5", 1),
    ("XCH A,R6", 1),
    ("XCH A,R7", 1),
    ("POP $1", 2),
    ("ACALL $a", 2),
    ("SETB $1", 2),
    ("SETB C", 1),
    ("DA A", 1),
    ("DJNZ $1,$r", 3),
    ("XCHD A,@R0", 1),
    ("XCHD A,@R1", 1),
    ("DJNZ R0,$r", 2),
    ("DJNZ R1,$r", 2),
    ("DJNZ R2,$r", 2),
    ("DJNZ R3,$r", 2),
    ("DJNZ R4,$r", 2),
    ("DJNZ R5,$r", 2),
    ("DJNZ R6,$r", 2),
    ("DJNZ R7,$r", 2),
    ("MOVX A,@DPTR", 1),
    ("AJMP $a", 2),
    ("MOVX A,@R0", 1),
    ("MOVX A,@R1", 1),
    ("CLR A", 1),
    ("MOV A,$1", 2),
    ("MOV A,@R0", 1),
    ("MOV A,@R1", 1),
    ("MOV A,R0", 1),
    ("MOV A,R1", 1),
    ("MOV A,R2", 1),
    ("MOV A,R3", 1),
    ("MOV A,R4", 1),
    ("MOV A,R5", 1),
    ("MOV A,R6", 1),
    ("MOV A,R7", 1),
    ("MOVX @DPTR,A", 1),
    ("ACALL $a", 2),
    ("MOVX @R0,A", 1),
    ("MOVX @R1,A", 1),
    ("CPL A", 1),
    ("MOV $1,A", 2),
    ("MOV @R0,A", 1),
    ("MOV @R1,A", 1),
    ("MOV R0,A", 1),
    ("MOV R1,A", 1),
    ("MOV R2,A", 1),
    ("MOV R3,A", 1),
    ("MOV R4,A", 1),
    ("MOV R5,A", 1),
    ("MOV R6,A", 1),
    ("MOV R7,A", 1),
];

/// Returns the opcode table entry for a byte.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The template and instruction length.
#[must_use]
pub fn opcode(op: u8) -> (&'static str, u8) {
    OPS[usize::from(op)]
}

/// Disassembles a byte range as 8051 code.
///
/// # Arguments
///
/// * `bytes` - Code bytes.
/// * `base` - Address of the first byte.
/// * `max` - Maximum instructions to decode.
///
/// # Returns
///
/// One formatted line per instruction.
#[must_use]
pub fn disassemble(bytes: &[u8], base: u64, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() && out.len() < max {
        pos += emit_one(bytes, base, pos, &mut out);
    }
    out
}

/// Decodes and appends one instruction.
///
/// # Arguments
///
/// * `bytes` - Code bytes.
/// * `base` - Address of the first byte.
/// * `pos` - Instruction offset.
/// * `out` - Accumulated lines.
///
/// # Returns
///
/// The instruction length in bytes.
fn emit_one(bytes: &[u8], base: u64, pos: usize, out: &mut Vec<String>) -> usize {
    let op = get(bytes, pos);
    let (template, len) = OPS[usize::from(op)];
    let text = render(template, bytes, base, pos, op, usize::from(len));
    out.push(format!("{:04x}  {text}", base + pos as u64));
    usize::from(len)
}

/// Renders an instruction template with resolved operands.
///
/// # Arguments
///
/// * `template` - Opcode template.
/// * `bytes` - Code bytes.
/// * `base` - Base address.
/// * `pos` - Instruction offset.
/// * `op` - Opcode byte.
/// * `len` - Instruction length.
///
/// # Returns
///
/// The rendered instruction text.
fn render(template: &str, bytes: &[u8], base: u64, pos: usize, op: u8, len: usize) -> String {
    let tokens = [
        ("$a", hex16(addr11(bytes, base, pos, op))),
        ("$w", hex16(word16(bytes, pos))),
        ("$r", hex16(rel_target(bytes, base, pos, len, op))),
        ("$2", hex8(get(bytes, pos + 2))),
        ("$1", hex8(get(bytes, pos + 1))),
    ];
    substitute(template, &tokens)
}

/// Substitutes operand tokens into a template.
///
/// # Arguments
///
/// * `template` - Opcode template.
/// * `tokens` - Token and value pairs.
///
/// # Returns
///
/// The rendered instruction text.
fn substitute(template: &str, tokens: &[(&str, String)]) -> String {
    tokens.iter().fold(template.to_string(), |text, (token, value)| {
        text.replace(token, value)
    })
}

/// Reads a byte at an offset, defaulting to zero.
///
/// # Arguments
///
/// * `bytes` - Source buffer.
/// * `offset` - Byte offset.
///
/// # Returns
///
/// The byte value.
fn get(bytes: &[u8], offset: usize) -> u8 {
    bytes.get(offset).copied().unwrap_or(0)
}

/// Formats a byte as two hexadecimal digits.
///
/// # Arguments
///
/// * `value` - Byte value.
///
/// # Returns
///
/// The hexadecimal text.
fn hex8(value: u8) -> String {
    format!("{value:02x}")
}

/// Formats a word as four hexadecimal digits.
///
/// # Arguments
///
/// * `value` - Word value.
///
/// # Returns
///
/// The hexadecimal text.
fn hex16(value: u16) -> String {
    format!("{value:04x}")
}

/// Reads a big-endian 16-bit operand.
///
/// # Arguments
///
/// * `bytes` - Code bytes.
/// * `pos` - Instruction offset.
///
/// # Returns
///
/// The operand word.
fn word16(bytes: &[u8], pos: usize) -> u16 {
    (u16::from(get(bytes, pos + 1)) << 8) | u16::from(get(bytes, pos + 2))
}

/// Resolves an 11-bit page address for `ajmp`/`acall`.
///
/// # Arguments
///
/// * `bytes` - Code bytes.
/// * `base` - Base address.
/// * `pos` - Instruction offset.
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The target address.
fn addr11(bytes: &[u8], base: u64, pos: usize, op: u8) -> u16 {
    let page = page_of(base + pos as u64 + 2);
    page | (u16::from(op >> 5) << 8) | u16::from(get(bytes, pos + 1))
}

/// Masks an address to its 2 KiB page.
///
/// # Arguments
///
/// * `address` - Target address.
///
/// # Returns
///
/// The page base.
fn page_of(address: u64) -> u16 {
    u16::try_from(address & 0xF800).unwrap_or(0)
}

/// Resolves a relative branch target.
///
/// # Arguments
///
/// * `bytes` - Code bytes.
/// * `base` - Base address.
/// * `pos` - Instruction offset.
/// * `len` - Instruction length.
/// * `op` - Opcode byte.
///
/// # Returns
///
/// The target address.
fn rel_target(bytes: &[u8], base: u64, pos: usize, len: usize, op: u8) -> u16 {
    let byte = get(bytes, pos + rel_index(op));
    let rel = i64::from(i8::from_ne_bytes([byte]));
    let next = base + pos as u64 + len as u64;
    u16::try_from(next.wrapping_add_signed(rel) & 0xFFFF).unwrap_or(0)
}

/// Returns the relative-operand byte index for an opcode.
///
/// # Arguments
///
/// * `op` - Opcode byte.
///
/// # Returns
///
/// One when the relative byte is the first operand, otherwise two.
fn rel_index(op: u8) -> usize {
    if matches!(op, 0x10 | 0x20 | 0x30 | 0xB4..=0xBF | 0xD5) {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies decoding of the Shanghai 8051 prologue.
    #[test]
    fn test_disassemble_prologue() {
        let code = [
            0x7c, 0x00, 0x7d, 0x00, 0x7b, 0xff, 0x12, 0x01, 0xe0, 0x90, 0xc1, 0x05,
        ];
        let lines = disassemble(&code, 0x1000, 32);
        assert!(lines[0].contains("MOV R4,#00"));
        assert!(lines[2].contains("MOV R3,#ff"));
        assert!(lines[3].contains("LCALL 01e0"));
        assert!(lines[4].contains("MOV DPTR,#c105"));
    }

    /// Verifies relative and page branch resolution.
    #[test]
    fn test_branches() {
        let code = [0x80, 0xfe, 0x21, 0x00];
        let lines = disassemble(&code, 0x1000, 4);
        assert!(lines[0].contains("SJMP 1000"));
        assert!(lines[1].contains("AJMP 1100"));
    }

    /// Verifies the length of every opcode entry is sane.
    #[test]
    fn test_lengths() {
        assert!(OPS.iter().all(|(_, len)| (1..=3).contains(len)));
    }
}
