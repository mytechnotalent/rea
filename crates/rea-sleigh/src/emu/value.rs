//! P-code Value Semantics
//!
//! **File:** `value.rs`
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

// Import dependencies from sleigh
use sleigh::Opcode;

/// Signature of a size-aware binary p-code operation over machine words.
pub(crate) type Binary = fn(u64, u64, usize) -> u64;

/// Signature of a width-changing unary p-code operation over machine words.
pub(crate) type Unary = fn(u64, usize, usize) -> u64;

/// Binary opcodes paired with the machine-word function that evaluates them.
const BINARY_OPS: [(Opcode, Binary); 22] = [
    (Opcode::IntAdd, int_add),
    (Opcode::IntSub, int_sub),
    (Opcode::IntMult, int_mult),
    (Opcode::IntAnd, int_and),
    (Opcode::IntOr, int_or),
    (Opcode::IntXor, int_xor),
    (Opcode::IntLeft, int_left),
    (Opcode::IntRight, int_right),
    (Opcode::IntSRight, int_sright),
    (Opcode::IntDiv, int_div),
    (Opcode::IntSDiv, int_sdiv),
    (Opcode::IntRem, int_rem),
    (Opcode::IntSRem, int_srem),
    (Opcode::IntEqual, int_equal),
    (Opcode::IntNotEqual, int_not_equal),
    (Opcode::IntLess, int_less),
    (Opcode::IntLessEqual, int_less_equal),
    (Opcode::IntSLess, int_sless),
    (Opcode::IntSLessEqual, int_sless_equal),
    (Opcode::IntCarry, int_carry),
    (Opcode::IntSCarry, int_scarry),
    (Opcode::IntSBorrow, int_sborrow),
];

/// Unary opcodes paired with the machine-word function that evaluates them.
const UNARY_OPS: [(Opcode, Unary); 8] = [
    (Opcode::Copy, copy),
    (Opcode::IntZext, zero_extend),
    (Opcode::IntSext, sign_extend_op),
    (Opcode::Int2Comp, two_complement),
    (Opcode::IntNegate, bitwise_negate),
    (Opcode::BoolNegate, bool_negate),
    (Opcode::PopCount, pop_count),
    (Opcode::LzCount, leading_zeros),
];

/// Looks up the binary evaluator for an opcode.
///
/// # Arguments
///
/// * `opcode` - Operation to evaluate.
///
/// # Returns
///
/// The evaluator, or `None` when the opcode is not a binary operation.
pub(crate) fn binary_op(opcode: Opcode) -> Option<Binary> {
    BINARY_OPS.iter().find(|(op, _)| *op == opcode).map(|(_, f)| *f)
}

/// Looks up the unary evaluator for an opcode.
///
/// # Arguments
///
/// * `opcode` - Operation to evaluate.
///
/// # Returns
///
/// The evaluator, or `None` when the opcode is not a unary operation.
pub(crate) fn unary_op(opcode: Opcode) -> Option<Unary> {
    UNARY_OPS.iter().find(|(op, _)| *op == opcode).map(|(_, f)| *f)
}

/// Returns the bit width of a byte size, capped at 64.
///
/// # Arguments
///
/// * `size` - Width in bytes.
///
/// # Returns
///
/// The width in bits, never exceeding 64.
pub(crate) fn bits(size: usize) -> u32 {
    u32::try_from(size.saturating_mul(8)).unwrap_or(64).min(64)
}

/// Returns the low `size`-byte mask.
///
/// # Arguments
///
/// * `size` - Width in bytes.
///
/// # Returns
///
/// A mask with the low `size * 8` bits set.
pub(crate) fn mask(size: usize) -> u64 {
    let width = bits(size);
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Truncates a value to its low `size` bytes.
///
/// # Arguments
///
/// * `value` - Value to truncate.
/// * `size` - Width in bytes.
///
/// # Returns
///
/// The value masked to `size` bytes.
pub(crate) fn trunc(value: u64, size: usize) -> u64 {
    value & mask(size)
}

/// Sign-extends the low `size` bytes of a value to the full word.
///
/// # Arguments
///
/// * `value` - Value whose low bytes carry the sign.
/// * `size` - Signed width in bytes.
///
/// # Returns
///
/// The sign-extended value.
pub(crate) fn sext(value: u64, size: usize) -> u64 {
    let width = bits(size);
    if width >= 64 {
        return value;
    }
    let shift = 64 - width;
    as_unsigned(as_signed(value.wrapping_shl(shift)) >> shift)
}

/// Reinterprets a word as a signed integer without lossy casts.
///
/// # Arguments
///
/// * `value` - Unsigned word.
///
/// # Returns
///
/// The same bit pattern as a signed integer.
fn as_signed(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

/// Reinterprets a signed integer as a word without lossy casts.
///
/// # Arguments
///
/// * `value` - Signed integer.
///
/// # Returns
///
/// The same bit pattern as an unsigned word.
fn as_unsigned(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

/// Reports whether a signed value fits in a `size`-byte field.
///
/// # Arguments
///
/// * `value` - Candidate signed value.
/// * `size` - Field width in bytes.
///
/// # Returns
///
/// `true` when the value is representable in two's complement.
fn fits_signed(value: i128, size: usize) -> bool {
    let shift = bits(size).saturating_sub(1);
    let max = (1i128 << shift) - 1;
    value >= -(1i128 << shift) && value <= max
}

/// Evaluates a truncated two's-complement addition.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a + b` truncated to `size` bytes.
fn int_add(a: u64, b: u64, size: usize) -> u64 {
    trunc(a.wrapping_add(b), size)
}

/// Evaluates a truncated two's-complement subtraction.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a - b` truncated to `size` bytes.
fn int_sub(a: u64, b: u64, size: usize) -> u64 {
    trunc(a.wrapping_sub(b), size)
}

/// Evaluates a truncated two's-complement multiplication.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a * b` truncated to `size` bytes.
fn int_mult(a: u64, b: u64, size: usize) -> u64 {
    trunc(a.wrapping_mul(b), size)
}

/// Evaluates a truncated bitwise AND.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a & b` truncated to `size` bytes.
fn int_and(a: u64, b: u64, size: usize) -> u64 {
    trunc(a & b, size)
}

/// Evaluates a truncated bitwise OR.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a | b` truncated to `size` bytes.
fn int_or(a: u64, b: u64, size: usize) -> u64 {
    trunc(a | b, size)
}

/// Evaluates a truncated bitwise XOR.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a ^ b` truncated to `size` bytes.
fn int_xor(a: u64, b: u64, size: usize) -> u64 {
    trunc(a ^ b, size)
}

/// Evaluates a logical left shift with p-code out-of-range semantics.
///
/// # Arguments
///
/// * `a` - Value to shift.
/// * `b` - Shift amount.
/// * `size` - Value width in bytes.
///
/// # Returns
///
/// `a << b`, or zero when the amount reaches the bit width.
fn int_left(a: u64, b: u64, size: usize) -> u64 {
    match u32::try_from(b) {
        Ok(amount) if amount < bits(size) => trunc(a.wrapping_shl(amount), size),
        _ => 0,
    }
}

/// Evaluates a logical right shift with p-code out-of-range semantics.
///
/// # Arguments
///
/// * `a` - Value to shift.
/// * `b` - Shift amount.
/// * `size` - Value width in bytes.
///
/// # Returns
///
/// `a >> b` unsigned, or zero when the amount reaches the bit width.
fn int_right(a: u64, b: u64, size: usize) -> u64 {
    match u32::try_from(b) {
        Ok(amount) if amount < bits(size) => trunc(trunc(a, size).wrapping_shr(amount), size),
        _ => 0,
    }
}

/// Evaluates an arithmetic right shift with p-code out-of-range semantics.
///
/// # Arguments
///
/// * `a` - Value to shift.
/// * `b` - Shift amount.
/// * `size` - Value width in bytes.
///
/// # Returns
///
/// `a >> b` with sign fill, saturating to the sign bit past the width.
fn int_sright(a: u64, b: u64, size: usize) -> u64 {
    let value = as_signed(sext(a, size));
    let amount = u32::try_from(b).unwrap_or(64).min(bits(size).saturating_sub(1));
    trunc(as_unsigned(value >> amount), size)
}

/// Evaluates an unsigned division, yielding zero on a zero divisor.
///
/// # Arguments
///
/// * `a` - Dividend.
/// * `b` - Divisor.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a / b` unsigned, or zero when `b` is zero.
fn int_div(a: u64, b: u64, size: usize) -> u64 {
    let quotient = trunc(a, size).checked_div(trunc(b, size)).unwrap_or(0);
    trunc(quotient, size)
}

/// Evaluates an unsigned remainder, yielding zero on a zero divisor.
///
/// # Arguments
///
/// * `a` - Dividend.
/// * `b` - Divisor.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a % b` unsigned, or zero when `b` is zero.
fn int_rem(a: u64, b: u64, size: usize) -> u64 {
    let remainder = trunc(a, size).checked_rem(trunc(b, size)).unwrap_or(0);
    trunc(remainder, size)
}

/// Evaluates a signed division, guarding zero and overflow.
///
/// # Arguments
///
/// * `a` - Dividend.
/// * `b` - Divisor.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a / b` signed, or a defined value for zero or overflowing inputs.
fn int_sdiv(a: u64, b: u64, size: usize) -> u64 {
    let (x, y) = (as_signed(sext(a, size)), as_signed(sext(b, size)));
    if y == 0 {
        return 0;
    }
    trunc(as_unsigned(x.wrapping_div(y)), size)
}

/// Evaluates a signed remainder, guarding zero and overflow.
///
/// # Arguments
///
/// * `a` - Dividend.
/// * `b` - Divisor.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// `a % b` signed, or a defined value for zero or overflowing inputs.
fn int_srem(a: u64, b: u64, size: usize) -> u64 {
    let (x, y) = (as_signed(sext(a, size)), as_signed(sext(b, size)));
    if y == 0 {
        return 0;
    }
    trunc(as_unsigned(x.wrapping_rem(y)), size)
}

/// Evaluates an equality comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when the truncated operands are equal, otherwise zero.
fn int_equal(a: u64, b: u64, size: usize) -> u64 {
    u64::from(trunc(a, size) == trunc(b, size))
}

/// Evaluates an inequality comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when the truncated operands differ, otherwise zero.
fn int_not_equal(a: u64, b: u64, size: usize) -> u64 {
    u64::from(trunc(a, size) != trunc(b, size))
}

/// Evaluates an unsigned less-than comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a < b` unsigned, otherwise zero.
fn int_less(a: u64, b: u64, size: usize) -> u64 {
    u64::from(trunc(a, size) < trunc(b, size))
}

/// Evaluates an unsigned less-than-or-equal comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a <= b` unsigned, otherwise zero.
fn int_less_equal(a: u64, b: u64, size: usize) -> u64 {
    u64::from(trunc(a, size) <= trunc(b, size))
}

/// Evaluates a signed less-than comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a < b` signed, otherwise zero.
fn int_sless(a: u64, b: u64, size: usize) -> u64 {
    u64::from(as_signed(sext(a, size)) < as_signed(sext(b, size)))
}

/// Evaluates a signed less-than-or-equal comparison.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a <= b` signed, otherwise zero.
fn int_sless_equal(a: u64, b: u64, size: usize) -> u64 {
    u64::from(as_signed(sext(a, size)) <= as_signed(sext(b, size)))
}

/// Evaluates the unsigned addition carry flag.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a + b` overflows the field, otherwise zero.
fn int_carry(a: u64, b: u64, size: usize) -> u64 {
    let sum = u128::from(trunc(a, size)) + u128::from(trunc(b, size));
    u64::from(sum > u128::from(mask(size)))
}

/// Evaluates the signed addition overflow flag.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a + b` overflows the signed field, otherwise zero.
fn int_scarry(a: u64, b: u64, size: usize) -> u64 {
    let sum = i128::from(as_signed(sext(a, size))) + i128::from(as_signed(sext(b, size)));
    u64::from(!fits_signed(sum, size))
}

/// Evaluates the signed subtraction overflow flag.
///
/// # Arguments
///
/// * `a` - Left operand.
/// * `b` - Right operand.
/// * `size` - Operand width in bytes.
///
/// # Returns
///
/// One when `a - b` overflows the signed field, otherwise zero.
fn int_sborrow(a: u64, b: u64, size: usize) -> u64 {
    let diff = i128::from(as_signed(sext(a, size))) - i128::from(as_signed(sext(b, size)));
    u64::from(!fits_signed(diff, size))
}

/// Copies a value into an output width.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `_in_size` - Source width in bytes, unused.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The value truncated to the destination width.
fn copy(value: u64, _in_size: usize, out_size: usize) -> u64 {
    trunc(value, out_size)
}

/// Zero-extends a value into a wider output.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `in_size` - Source width in bytes.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The low `in_size` bytes placed in a zeroed destination.
fn zero_extend(value: u64, in_size: usize, out_size: usize) -> u64 {
    trunc(trunc(value, in_size), out_size)
}

/// Sign-extends a value into a wider output.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `in_size` - Source width in bytes.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The sign-extended source truncated to the destination width.
fn sign_extend_op(value: u64, in_size: usize, out_size: usize) -> u64 {
    trunc(sext(value, in_size), out_size)
}

/// Negates a value in two's complement.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `_in_size` - Source width in bytes, unused.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// `-value` truncated to the destination width.
fn two_complement(value: u64, _in_size: usize, out_size: usize) -> u64 {
    trunc(0u64.wrapping_sub(value), out_size)
}

/// Inverts every bit of a value.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `_in_size` - Source width in bytes, unused.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The bitwise complement truncated to the destination width.
fn bitwise_negate(value: u64, _in_size: usize, out_size: usize) -> u64 {
    trunc(!value, out_size)
}

/// Inverts a boolean value.
///
/// # Arguments
///
/// * `value` - Source value whose low bit is the boolean.
/// * `_in_size` - Source width in bytes, unused.
/// * `_out_size` - Destination width in bytes, unused.
///
/// # Returns
///
/// One when the low bit is clear, otherwise zero.
fn bool_negate(value: u64, _in_size: usize, _out_size: usize) -> u64 {
    (value & 1) ^ 1
}

/// Counts the set bits of a value.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `in_size` - Source width in bytes.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The population count truncated to the destination width.
fn pop_count(value: u64, in_size: usize, out_size: usize) -> u64 {
    trunc(u64::from(trunc(value, in_size).count_ones()), out_size)
}

/// Counts the leading zero bits within the source width.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `in_size` - Source width in bytes.
/// * `out_size` - Destination width in bytes.
///
/// # Returns
///
/// The leading-zero count truncated to the destination width.
fn leading_zeros(value: u64, in_size: usize, out_size: usize) -> u64 {
    let width = bits(in_size);
    let masked = trunc(value, in_size);
    if masked == 0 {
        return trunc(u64::from(width), out_size);
    }
    trunc(u64::from(masked.leading_zeros() - (64 - width)), out_size)
}

/// Extracts a sub-range of a value beginning at a byte offset.
///
/// # Arguments
///
/// * `value` - Source value.
/// * `byte_offset` - Least-significant byte of the slice.
/// * `out_size` - Width of the slice in bytes.
///
/// # Returns
///
/// The requested byte slice, right-aligned.
pub(crate) fn subpiece(value: u64, byte_offset: usize, out_size: usize) -> u64 {
    let shift = bits(byte_offset);
    let shifted = if shift >= 64 { 0 } else { value >> shift };
    trunc(shifted, out_size)
}

/// Concatenates a high and low value into one word.
///
/// # Arguments
///
/// * `high` - Most-significant value.
/// * `low` - Least-significant value.
/// * `low_size` - Width of the low value in bytes.
///
/// # Returns
///
/// `high` shifted above `low`, combined into one word.
pub(crate) fn piece(high: u64, low: u64, low_size: usize) -> u64 {
    let shift = bits(low_size);
    let upper = if shift >= 64 { 0 } else { high << shift };
    upper | trunc(low, low_size)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies masking and truncation at a sub-word width.
    #[test]
    fn test_trunc_masks_low_bytes() {
        assert_eq!(trunc(0x1234, 1), 0x34);
        assert_eq!(mask(2), 0xffff);
    }

    /// Verifies sign extension of a negative byte.
    #[test]
    fn test_sext_byte() {
        assert_eq!(sext(0x80, 1), 0xffff_ffff_ffff_ff80);
        assert_eq!(sext(0x7f, 1), 0x7f);
    }

    /// Verifies wrapping addition through the binary table.
    #[test]
    fn test_binary_add_wraps() {
        let add = binary_op(Opcode::IntAdd).unwrap();
        assert_eq!(add(0xff, 0x01, 1), 0x00);
    }

    /// Verifies signed comparison through the binary table.
    #[test]
    fn test_binary_signed_less() {
        let less = binary_op(Opcode::IntSLess).unwrap();
        assert_eq!(less(0xff, 0x01, 1), 1);
    }

    /// Verifies out-of-range shifts resolve to zero.
    #[test]
    fn test_shift_out_of_range() {
        assert_eq!(int_left(1, 64, 8), 0);
        assert_eq!(int_right(0x80, 8, 1), 0);
    }

    /// Verifies the carry flag detects unsigned overflow.
    #[test]
    fn test_carry_flag() {
        assert_eq!(int_carry(0xff, 0x01, 1), 1);
        assert_eq!(int_carry(0x01, 0x01, 1), 0);
    }

    /// Verifies the unary table zero-extends and sign-extends.
    #[test]
    fn test_unary_extends() {
        let zext = unary_op(Opcode::IntZext).unwrap();
        let sxt = unary_op(Opcode::IntSext).unwrap();
        assert_eq!(zext(0x80, 1, 4), 0x80);
        assert_eq!(sxt(0x80, 1, 4), 0xffff_ff80);
    }

    /// Verifies sub-range extraction and concatenation.
    #[test]
    fn test_subpiece_and_piece() {
        assert_eq!(subpiece(0x1122_3344, 1, 1), 0x33);
        assert_eq!(piece(0xaa, 0xbb, 1), 0xaabb);
    }
}
