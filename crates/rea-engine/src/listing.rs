//! Instruction Listings
//!
//! **File:** `listing.rs`
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

// Import crate::disasm
use crate::disasm::{ends_block, referenced_addresses, CodeMap};

// Import dependencies from iced_x86
use iced_x86::{Decoder, DecoderOptions, FlowControl, Formatter, Instruction, IntelFormatter};

/// Farthest forward branch, in bytes, still treated as part of the listed function.
const LOCAL_SPAN: u64 = 0x4000;

/// Produces an annotated Intel-syntax listing of one function.
///
/// Decoding is linear and stops at the first block terminator past every
/// forward branch seen so far, or after `max` instructions.
///
/// # Arguments
///
/// * `code` - Executable regions.
/// * `bitness` - Decoder bitness.
/// * `start` - Function entry.
/// * `max` - Maximum instructions to list.
/// * `annotate` - Comment source for referenced addresses.
///
/// # Returns
///
/// One formatted line per instruction.
#[must_use]
pub fn render(
    code: &CodeMap,
    bitness: u32,
    start: u64,
    max: usize,
    annotate: &Annotate<'_>,
) -> Vec<String> {
    let mut formatter = configured_formatter();
    let mut extent = Extent::new(start);
    let decoder = Decoder::with_ip(bitness, code.bytes_at(start), start, DecoderOptions::NONE);
    let body = decoder.into_iter().take(max);
    let kept = body.take_while(|i| extent.proceed(i));
    let width = format!("{start:x}").len().max(8);
    kept.map(|i| line(&mut formatter, &i, width, annotate)).collect()
}

/// Comment source mapping a referenced address to an annotation.
pub type Annotate<'a> = dyn Fn(u64) -> Option<String> + 'a;

/// Builds the formatter used for every listing.
///
/// # Returns
///
/// An Intel formatter with `0x` hex and absolute RIP-relative addresses.
fn configured_formatter() -> IntelFormatter {
    let mut formatter = IntelFormatter::new();
    let options = formatter.options_mut();
    options.set_hex_prefix("0x");
    options.set_hex_suffix("");
    options.set_uppercase_hex(false);
    options.set_space_after_operand_separator(true);
    formatter
}

/// Formats one listing line.
///
/// # Arguments
///
/// * `formatter` - Instruction formatter.
/// * `instruction` - Decoded instruction.
/// * `width` - Address width in hex digits.
/// * `annotate` - Comment source for referenced addresses.
///
/// # Returns
///
/// `address  mnemonic operands  ; comment`.
fn line(
    formatter: &mut IntelFormatter,
    instruction: &Instruction,
    width: usize,
    annotate: &Annotate<'_>,
) -> String {
    let mut text = String::new();
    formatter.format(instruction, &mut text);
    let mut targets = referenced_addresses(instruction);
    targets.push(instruction.near_branch_target());
    let note = targets.into_iter().filter(|t| *t != 0).find_map(annotate);
    let comment = note.map(|n| format!("  ; {n}")).unwrap_or_default();
    format!("{:0width$x}  {text}{comment}", instruction.ip())
}

/// Tracks how far a linearly listed function extends.
struct Extent {
    /// Function entry.
    start: u64,

    /// Farthest local branch target seen.
    furthest: u64,

    /// Whether the function end has been passed.
    done: bool,
}

impl Extent {
    /// Starts tracking a function.
    ///
    /// # Arguments
    ///
    /// * `start` - Function entry.
    ///
    /// # Returns
    ///
    /// An extent covering only the entry.
    fn new(start: u64) -> Self {
        Self { start, furthest: start, done: false }
    }

    /// Decides whether to list an instruction and updates the extent.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Next decoded instruction.
    ///
    /// # Returns
    ///
    /// `true` while the instruction belongs to the function.
    fn proceed(&mut self, instruction: &Instruction) -> bool {
        let keep = !self.done && !instruction.is_invalid();
        self.extend(instruction);
        let past = instruction.next_ip() > self.furthest;
        self.done = ends_block(instruction) && past;
        keep
    }

    /// Grows the extent to cover a nearby forward branch target.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    fn extend(&mut self, instruction: &Instruction) {
        let target = instruction.near_branch_target();
        let nearby = target > self.furthest && target - self.start < LOCAL_SPAN;
        if nearby && is_local_branch(instruction) {
            self.furthest = target;
        }
    }
}

/// Reports whether an instruction is a direct jump that may stay in the function.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
///
/// # Returns
///
/// `true` for conditional and unconditional direct branches.
fn is_local_branch(instruction: &Instruction) -> bool {
    matches!(
        instruction.flow_control(),
        FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
    )
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate items
    use crate::{
        detect::Format,
        model::{Arch, Binary, Section},
    };

    /// Lists `code` mapped at 0x1000.
    ///
    /// # Arguments
    ///
    /// * `code` - Machine code.
    /// * `annotate` - Comment source.
    ///
    /// # Returns
    ///
    /// The listing lines.
    fn list(code: &[u8], annotate: &Annotate<'_>) -> Vec<String> {
        let size = crate::numeric::to_u64(code.len());
        let text = Section::mapped(".text", 0x1000, 0x100).with_file(0, size);
        let mut binary = Binary::new(Format::Unknown, Arch::X86_64, "x86-64", 64);
        binary.sections = vec![text.with_access(true, false)];
        let map = CodeMap::new(&binary, code);
        render(&map, 64, 0x1000, 50, annotate)
    }

    /// Verifies that listing continues past a branch target and stops at the end.
    #[test]
    fn test_render_spans_forward_branches() {
        let lines = list(&[0x74, 0x01, 0xc3, 0x90, 0xc3, 0xcc, 0xcc], &|_| None);
        assert_eq!(lines.len(), 4);
        assert!(lines[3].ends_with("ret"));
    }

    /// Verifies address formatting and annotation comments.
    #[test]
    fn test_render_annotates_targets() {
        let note = |a: u64| (a == 0x1005).then(|| "helper".to_string());
        let lines = list(&[0xe8, 0x00, 0, 0, 0, 0xc3], &note);
        assert!(lines[0].starts_with("00001000  call"));
        assert!(lines[0].ends_with("; helper"));
    }
}
