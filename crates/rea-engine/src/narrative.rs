//! Behavioral Narrative Synthesis
//!
//! **File:** `narrative.rs`
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
use std::collections::HashMap;

// Import crate items
use crate::{
    disasm::{Disassembly, Function, InstructionLine},
    model::Binary,
    semantics,
    strings::FoundString,
    Analysis,
};

/// Evidence anchors resolving raw addresses to human meaning.
pub struct Evidence<'a> {
    /// Parsed executable metadata.
    binary: &'a Binary,
    /// Virtual address and text of each string in the image.
    strings: HashMap<u64, &'a str>,
    /// Precomputed inverted call graph mapping callee to callers.
    callers: HashMap<u64, Vec<u64>>,
}

impl<'a> Evidence<'a> {
    /// Builds evidence anchors from parsed metadata and extracted strings.
    ///
    /// # Arguments
    ///
    /// * `binary` - Parsed executable.
    /// * `strings` - Strings extracted from the file.
    ///
    /// # Returns
    ///
    /// Evidence resolver for narrative synthesis.
    #[must_use]
    pub fn new(binary: &'a Binary, strings: &'a [FoundString]) -> Self {
        let mapped = strings.iter().filter_map(|s| binary.address_of(s.offset).zip(Some(&*s.text)));
        Self { binary, strings: mapped.collect(), callers: HashMap::new() }
    }

    /// Indexes the call graph from disassembly for fast caller lookups.
    ///
    /// # Arguments
    ///
    /// * `code` - Disassembly result.
    ///
    /// # Returns
    ///
    /// Updated resolver with precomputed callers.
    #[must_use]
    pub fn with_callers(mut self, code: &Disassembly) -> Self {
        for f in code.functions.values() {
            for target in &f.calls {
                self.callers.entry(*target).or_default().push(f.address);
            }
        }
        self
    }

    /// Finds the string stored at a virtual address.
    ///
    /// # Arguments
    ///
    /// * `va` - Virtual address referenced by code.
    ///
    /// # Returns
    ///
    /// The string text when the address marks a string start.
    #[must_use]
    pub fn string_at(&self, va: u64) -> Option<&'a str> {
        self.strings.get(&va).copied()
    }

    /// Strings a function refers to, directly or through literal-pool pointers.
    ///
    /// # Arguments
    ///
    /// * `func` - Function to inspect.
    ///
    /// # Returns
    ///
    /// Referenced string texts, deduplicated in address order.
    #[must_use]
    pub fn strings_of(&self, func: &Function) -> Vec<&'a str> {
        let mut found: Vec<&'a str> =
            func.references.iter().filter_map(|a| self.string_at(*a)).collect();
        for entry in &func.data_pools {
            found.extend(pool_word_value(&entry.bytes_hex).and_then(|va| self.string_at(va)));
        }
        found.retain(|s| crate::strings::is_plausible(s));
        found.dedup();
        found
    }
}

/// Decodes a literal-pool `bytes_hex` field into a 32-bit value.
///
/// # Arguments
///
/// * `bytes_hex` - Space-separated little-endian hex bytes.
///
/// # Returns
///
/// The decoded word value.
fn pool_word_value(bytes_hex: &str) -> Option<u64> {
    let bytes: Vec<u8> = bytes_hex
        .split_whitespace()
        .filter_map(|b| u8::from_str_radix(b, 16).ok())
        .collect();
    (bytes.len() == 4)
        .then(|| u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])))
}

/// Produces a behavioral story for one function as plain-English sentences.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
///
/// # Returns
///
/// Ordered sentences explaining what the function actually does.
#[must_use]
pub fn function_story(ev: &Evidence<'_>, code: &Disassembly, func: &Function) -> Vec<String> {
    let mut out = vec![role_sentence(ev, code, func)];
    out.extend(io_sentences(ev, func));
    out.extend(loop_sentences(func));
    out.push(shape_sentence(code, func));
    out
}

/// Sentence naming the function and its architectural role.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
///
/// # Returns
///
/// Introductory sentence.
fn role_sentence(ev: &Evidence<'_>, code: &Disassembly, func: &Function) -> String {
    let name = code.name_of(func.address);
    if func.address == ev.binary.entry {
        return format!("`{name}` is the reset handler: the first code that runs after boot.");
    }
    non_entry_role(ev, code, func, &name)
}

/// Role sentence for a non-entry function.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
/// * `name` - Function display name.
///
/// # Returns
///
/// Caller list or orphan-role sentence.
fn non_entry_role(ev: &Evidence<'_>, code: &Disassembly, func: &Function, name: &str) -> String {
    let callers = caller_names(ev, code, func);
    if callers.is_empty() {
        orphan_role(code, func, name)
    } else {
        format!("`{name}` is called by {}.", callers.join(", "))
    }
}

/// Role sentence for a function with no recorded callers.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
/// * `name` - Function display name.
///
/// # Returns
///
/// A sentence noting vector-table or main candidacy.
fn orphan_role(code: &Disassembly, func: &Function, name: &str) -> String {
    let hint = orphan_hint(code, func);
    format!("`{name}` has no recorded callers; {hint}.")
}

/// Chooses the orphan-role explanation phrase.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
///
/// # Returns
///
/// Explanation phrase.
fn orphan_hint(code: &Disassembly, func: &Function) -> &'static str {
    let reaches_leaf = func.calls.iter().filter(|c| code.functions.contains_key(c)).count();
    if ends_in_infinite_loop(func) && !func.calls.is_empty() {
        "its self-looping shape and outbound calls mark it as a main loop candidate"
    } else if reaches_leaf == 0 && func.calls.is_empty() {
        "it is a leaf stub, likely reached through the vector table"
    } else {
        "it is likely reached through the vector table or an indirect jump"
    }
}

/// Text naming a function's callers from indexed evidence.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Recovered code structure.
/// * `func` - Function to inspect.
///
/// # Returns
///
/// Caller names in address order.
fn caller_names(ev: &Evidence<'_>, code: &Disassembly, func: &Function) -> Vec<String> {
    let Some(callers) = ev.callers.get(&func.address) else {
        return Vec::new();
    };
    callers.iter().map(|a| code.name_of(*a)).collect()
}

/// Sentences about I/O behavior: strings printed and hardware touched.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `func` - Function to inspect.
///
/// # Returns
///
/// Zero or more behavioral sentences.
fn io_sentences(ev: &Evidence<'_>, func: &Function) -> Vec<String> {
    let mut out = Vec::new();
    out.extend(string_sentence(ev, func));
    out.extend(gpio_sentence(func));
    out.extend(mmio_sentence(func));
    out
}

/// Sentence quoting user-visible strings the function emits or reads.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `func` - Function to inspect.
///
/// # Returns
///
/// A sentence when the function references strings.
fn string_sentence(ev: &Evidence<'_>, func: &Function) -> Option<String> {
    let strings = ev.strings_of(func);
    let quoted: Vec<String> =
        strings.iter().take(3).map(|s| format!("\"{}\"", escape(s))).collect();
    (!quoted.is_empty()).then(|| format!("It works with the text {}.", quoted.join(", ")))
}

/// Sentence describing fast coprocessor GPIO use and the pin involved.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// A sentence when coprocessor GPIO instructions are present.
fn gpio_sentence(func: &Function) -> Option<String> {
    let coproc = func.lines.iter().any(|l| is_gpio_coproc(&l.text));
    let pin = coproc.then(|| pin_of(func)).flatten();
    let pin_text = pin.map_or(String::new(), |p| format!(" pin {p}"));
    coproc.then(|| format!("It drives GPIO{pin_text} through fast coprocessor instructions (`mcrr p0`), switching pin state."))
}

/// Reports whether an instruction uses coprocessor fast I/O.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// `true` for coprocessor fast I/O instructions.
fn is_gpio_coproc(text: &str) -> bool {
    text.starts_with("mcrr p0")
}

/// Pin number loaded next to coprocessor use.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// The immediate loaded into `r4`, the GPIO pin register, when present.
fn pin_of(func: &Function) -> Option<u64> {
    func.lines
        .iter()
        .filter(|l| l.text.starts_with("movs r4,"))
        .find_map(|l| immediate(&l.text))
}

/// Sentence describing peripheral register access.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// A sentence when MMIO addresses are referenced.
fn mmio_sentence(func: &Function) -> Option<String> {
    let mmio = func.references.iter().find(|&&a| a >= 0x4000_0000 && a < 0xf000_0000)?;
    Some(format!("It accesses memory-mapped peripheral registers at {:#x}.", mmio))
}

/// Sentences describing loops: infinite main loops and busy-wait delays.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// Zero or more loop sentences.
fn loop_sentences(func: &Function) -> Vec<String> {
    let mut out = Vec::new();
    out.extend(infinite_sentence(func));
    out.extend(delay_sentence(func));
    out
}

/// Sentence describing a self-branching infinite loop.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// A sentence when the function loops forever.
fn infinite_sentence(func: &Function) -> Option<String> {
    ends_in_infinite_loop(func).then(|| {
        "Its final instruction branches back to its own body, so once entered it runs forever."
            .to_string()
    })
}

/// Reports whether the function ends with an unconditional branch into itself.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// `true` when the last instruction loops back.
fn ends_in_infinite_loop(func: &Function) -> bool {
    let Some(last) = func.lines.last() else {
        return false;
    };
    let target = last.text.strip_prefix("b ").and_then(|t| t.trim().strip_prefix("0x"));
    target
        .and_then(|h| u64::from_str_radix(h, 16).ok())
        .is_some_and(|t| t >= func.address && t < last.address)
}

/// Reports whether the function ends in a trap or breakpoint hang.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// `true` when the function ends in a terminal trap loop.
fn is_terminal_trap(func: &Function) -> bool {
    let len = func.lines.len();
    if len < 2 {
        return false;
    }
    let prev = &func.lines[len - 2].text;
    prev.starts_with("bkpt") || prev.starts_with("udf") || prev.starts_with("wfi")
}

/// Reports whether a function is a viable candidate for the main application loop.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// `true` when the function self-loops without being a terminal trap.
fn is_main_candidate(func: &Function) -> bool {
    ends_in_infinite_loop(func) && !is_terminal_trap(func)
}

/// Sentence describing a busy-wait countdown loop.
///
/// # Arguments
///
/// * `func` - Function to inspect.
///
/// # Returns
///
/// A sentence when the body is a countdown loop with no calls.
fn delay_sentence(func: &Function) -> Option<String> {
    let counts = func.lines.iter().any(|l| l.text.starts_with("subs"));
    let loops = func.lines.iter().any(is_backward);
    (counts && loops && func.calls.is_empty()).then(|| {
        "It is a busy-wait delay loop, counting a register down to zero to burn time.".to_string()
    })
}

/// Reports whether a line branches to an earlier address.
///
/// # Arguments
///
/// * `line` - Instruction line.
///
/// # Returns
///
/// `true` for backward branches.
fn is_backward(line: &InstructionLine) -> bool {
    line.text.starts_with('b') && branch_target(&line.text).is_some_and(|t| t < line.address)
}

/// Extracts a hexadecimal branch target from instruction text.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// The target address when present.
fn branch_target(text: &str) -> Option<u64> {
    let start = text.rfind("0x")?;
    let hex: String = text[start + 2..].chars().take_while(char::is_ascii_hexdigit).collect();
    u64::from_str_radix(&hex, 16).ok()
}

/// Extracts a `#imm` operand value from instruction text.
///
/// # Arguments
///
/// * `text` - Instruction text.
///
/// # Returns
///
/// The immediate value when present.
fn immediate(text: &str) -> Option<u64> {
    let start = text.find('#')? + 1;
    let raw: String = text[start..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit() || *c == 'x')
        .collect();
    raw.strip_prefix("0x")
        .map_or_else(|| raw.parse().ok(), |h| u64::from_str_radix(h, 16).ok())
}

/// Sentence describing the function's call shape.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function to describe.
///
/// # Returns
///
/// Closing sentence about callees.
fn shape_sentence(code: &Disassembly, func: &Function) -> String {
    if func.calls.is_empty() {
        return "It calls nothing else.".to_string();
    }
    let names: Vec<String> = func.calls.iter().take(3).map(|a| code.name_of(*a)).collect();
    format!(
        "It delegates to {} ({} callees).",
        names.join(", "),
        func.calls.len()
    )
}

/// Escapes a string for inline display.
///
/// # Arguments
///
/// * `text` - Raw string text.
///
/// # Returns
///
/// Text with control characters shown as escapes.
fn escape(text: &str) -> String {
    text.replace('\r', "\\r").replace('\n', "\\n")
}

/// Detailed narrative breakdown of program behavior and architecture.
#[derive(Debug, Clone, Default)]
pub struct DetailedStory {
    /// Boot path and hardware initialization narrative.
    pub boot: Vec<String>,
    /// Main application logic and control loop narrative.
    pub main_logic: Vec<String>,
    /// Compiler optimization and variable lifecycle analysis.
    pub optimization: Vec<String>,
    /// Binary patching and reverse engineering vector analysis.
    pub patching: Vec<String>,
}

/// Builds the comprehensive, structured story of what the program does.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The detailed story structure.
#[must_use]
pub fn detailed_program_story(analysis: &Analysis) -> DetailedStory {
    let (Some(binary), Some(code)) = (&analysis.binary, &analysis.disassembly) else {
        return DetailedStory::default();
    };
    let ev = Evidence::new(binary, &analysis.strings);
    let main_func = find_main_func(&ev, code);
    let boot = boot_narrative(binary);
    let main_logic = main_logic_narrative(&ev, code, main_func);
    let optimization = optimization_narrative(&ev, main_func);
    let patching = patching_narrative(binary, main_func);
    DetailedStory { boot, main_logic, optimization, patching }
}

/// Selects the primary main application function from recovered code.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Disassembly result.
///
/// # Returns
///
/// The identified main application function.
fn find_main_func<'a>(ev: &Evidence<'_>, code: &'a Disassembly) -> Option<&'a Function> {
    let funcs = code.functions.values().filter(|f| is_main_candidate(f));
    let mut cand: Vec<&Function> = funcs.collect();
    if let Some(&f) = cand.iter().find(|f| format_string_of(ev, f).is_some()) {
        return Some(f);
    }
    cand.retain(|f| !f.calls.is_empty());
    cand.first().copied()
}

/// Builds the top-level plain-English story of what the program does.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// Ordered narrative sentences describing program behavior end to end.
#[must_use]
pub fn program_story(analysis: &Analysis) -> Vec<String> {
    let detailed = detailed_program_story(analysis);
    let mut out = detailed.boot;
    out.extend(detailed.main_logic);
    out.extend(detailed.optimization);
    out.extend(detailed.patching);
    out
}

/// Builds the boot path and hardware initialization narrative.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Boot narrative sentences.
fn boot_narrative(binary: &Binary) -> Vec<String> {
    let arch = &binary.arch_name;
    let s1 = format!(
        "On {arch}, the hardware loads the initial stack pointer from the vector table at {:#x} and jumps to the reset handler at {:#x}.",
        binary.image_base, binary.entry
    );
    let s2 = "The reset handler performs silicon initialization, configures system clocks, copies initialized data from Flash to SRAM, zeroes uninitialized BSS memory, and transfers control to the main application.".to_string();
    vec![s1, s2]
}

/// Synthesizes narrative describing the main application logic and loop.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Disassembly result.
/// * `main_func` - Identified main function.
///
/// # Returns
///
/// Application logic narrative sentences.
fn main_logic_narrative(
    ev: &Evidence<'_>,
    code: &Disassembly,
    main_func: Option<&Function>,
) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(f) = main_func {
        out.extend(main_func_sentences(ev, code, f));
    }
    out.extend(gpio_behavior(code));
    out.extend(delay_behavior(code));
    out
}

/// Generates sentences describing the main loop function execution.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Disassembly result.
/// * `func` - Main loop function.
///
/// # Returns
///
/// Execution walkthrough sentences.
fn main_func_sentences(ev: &Evidence<'_>, code: &Disassembly, func: &Function) -> Vec<String> {
    let name = code.name_of(func.address);
    let mut s = vec![format!(
        "Application function `{name}` at {:#x} implements the core firmware lifecycle.",
        func.address
    )];
    s.extend(init_call_sentence(code, func));
    s.extend(loop_print_sentence(ev, func));
    s
}

/// Generates sentence for early initialization calls before the loop.
///
/// # Arguments
///
/// * `code` - Disassembly result.
/// * `func` - Main function.
///
/// # Returns
///
/// Sentence if an initialization call exists.
fn init_call_sentence(code: &Disassembly, func: &Function) -> Option<String> {
    let first_call = func.calls.iter().next()?;
    let name = code.name_of(*first_call);
    Some(format!("Before entering the loop, it invokes `{name}` to initialize standard I/O and peripheral drivers."))
}

/// Generates sentence for printing and looping inside the main function.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `func` - Main function.
///
/// # Returns
///
/// Sentences describing loop output and branch.
fn loop_print_sentence(ev: &Evidence<'_>, func: &Function) -> Vec<String> {
    let strings = ev.strings_of(func);
    let str_hit = strings.iter().find(|s| s.contains('%'));
    let imm = func.lines.iter().find_map(|l| semantics::parse_imm_dec(&l.text));
    let mut out = Vec::new();
    if let (Some(fmt), Some(val)) = (str_hit, imm) {
        out.push(format!("Inside the loop, it loads immediate value {val} into parameter register r1 and prints \"{}\" via formatted serial output.", escape(fmt)));
    }
    out.push("At the bottom of the loop body, an unconditional backward branch loops execution back to repeat indefinitely.".into());
    out
}

/// Synthesizes compiler optimization and dead-code elimination analysis.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `code` - Disassembly result.
///
/// # Returns
///
/// Optimization analysis sentences.
fn optimization_narrative(ev: &Evidence<'_>, main_func: Option<&Function>) -> Vec<String> {
    let Some(f) = main_func else { return standard_storage_sentences(); };
    let has_fmt = format_string_of(ev, f).is_some();
    let imm = f.lines.iter().find_map(|l| semantics::parse_imm_dec(&l.text));
    if has_fmt && imm.is_some() {
        constant_prop_sentences(imm.unwrap_or_default())
    } else {
        standard_storage_sentences()
    }
}

/// Sentences explaining constant propagation and dead store elimination.
///
/// # Arguments
///
/// * `val` - Formatted immediate token.
///
/// # Returns
///
/// Explanation sentences.
fn constant_prop_sentences(val: String) -> Vec<String> {
    vec![
        "Compiler Optimization Analysis: In C source code, declaring a variable with an initial value and immediately overwriting it triggers dead-store elimination and constant folding.".into(),
        format!("Because the initial assignment is never read prior to reassignment, the compiler eliminates it entirely. Rather than reserving a memory slot in SRAM or stack frame storage, the compiler propagates the final constant {val} directly into machine code."),
        "The variable exists purely inside processor registers as an immediate operand in Flash memory, eliminating all RAM memory overhead.".into(),
    ]
}

/// Standard storage layout narrative for typical binaries.
///
/// # Returns
///
/// Default memory storage explanation sentences.
fn standard_storage_sentences() -> Vec<String> {
    vec![
        "Variable Storage Analysis: Read-only data resides in Flash memory (.rodata), initialized global variables are stored in Flash and copied to SRAM (.data) at boot, and uninitialized globals are allocated in SRAM (.bss) and zeroed at startup.".into(),
        "Local function variables are allocated dynamically within processor registers and thread stack frames.".into(),
    ]
}

/// Represents a concrete instruction patch target.
struct PatchPoint {
    /// Virtual address of the instruction.
    address: u64,
    /// Raw byte offset within the binary file.
    file_offset: u64,
    /// Immediate byte value.
    byte_val: u8,
    /// Instruction text.
    text: String,
}

/// Synthesizes binary patching and reverse engineering vectors.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `main_func` - Identified main function.
///
/// # Returns
///
/// Patching guidance sentences.
fn patching_narrative(binary: &Binary, main_func: Option<&Function>) -> Vec<String> {
    let Some(patch) = main_func.and_then(|f| find_patch_point(binary, f)) else {
        return vec!["Binary Patching: Analysis revealed no direct immediate constants in the main execution loop.".into()];
    };
    patch_sentences(binary, &patch)
}

/// Finds the prime immediate operand patch target in the main loop.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `func` - Main application function.
///
/// # Returns
///
/// Identified patch target if found.
fn find_patch_point(binary: &Binary, func: &Function) -> Option<PatchPoint> {
    let line = func.lines.iter().find(|l| l.text.starts_with("movs ") && l.text.contains('#'))?;
    let imm = immediate(&line.text)?;
    let byte_val = u8::try_from(imm).ok()?;
    let file_offset = line.address.checked_sub(binary.image_base)?;
    Some(PatchPoint { address: line.address, file_offset, byte_val, text: line.text.clone() })
}

/// Formats patching tutorial sentences for a patch point.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `p` - Patch point.
///
/// # Returns
///
/// Explanatory patching sentences.
fn patch_sentences(binary: &Binary, p: &PatchPoint) -> Vec<String> {
    let arch = &binary.arch_name;
    let s1 = format!("Reverse Engineering & Binary Patching Vector: The program's runtime behavior is controlled by the immediate operand at virtual address {:#x} (`{}`).", p.address, p.text);
    let s2 = format!("At raw binary file offset {:#x} (address {:#x} minus image base {:#x}), byte {:#04x} represents the loaded constant (decimal {}).", p.file_offset, p.address, binary.image_base, p.byte_val, p.byte_val);
    let s3 = format!("To alter program execution, patch byte {:#04x} at file offset {:#x} to any desired byte (e.g., 0x46 for decimal 70).", p.byte_val, p.file_offset);
    let s4 = format!("For {arch} deployment, package the patched binary using the target platform flasher and flash to the device.");
    vec![s1, s2, s3, s4]
}

/// Finds a format string referenced by a function.
///
/// # Arguments
///
/// * `ev` - Evidence resolver.
/// * `func` - Function to inspect.
///
/// # Returns
///
/// The format string text and the function address, when present.
fn format_string_of<'a>(ev: &Evidence<'a>, func: &Function) -> Option<(&'a str, u64)> {
    ev.strings_of(func).iter().find(|s| s.contains('%')).map(|s| (*s, func.address))
}

/// Sentence describing GPIO manipulation and the pin involved.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// A sentence when any function drives the GPIO coprocessor.
fn gpio_behavior(code: &Disassembly) -> Option<String> {
    let func = code
        .functions
        .values()
        .find(|f| f.lines.iter().any(|l| is_gpio_coproc(&l.text)));
    func.map(|f| gpio_text(code, f))
}

/// Formats the GPIO behavior sentence for one function.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
/// * `func` - Function driving GPIO.
///
/// # Returns
///
/// The formatted sentence.
fn gpio_text(code: &Disassembly, func: &Function) -> String {
    let pin = pin_of(func).map_or("?".into(), |p| p.to_string());
    let name = code.name_of(func.address);
    format!("It toggles GPIO pin {pin} using single-cycle I/O coprocessor instructions (in `{name}`).")
}

/// Sentence describing delay loops that pace the behavior.
///
/// # Arguments
///
/// * `code` - Recovered code structure.
///
/// # Returns
///
/// A sentence when a busy-wait delay exists.
fn delay_behavior(code: &Disassembly) -> Option<String> {
    let delay = code.functions.values().any(|f| delay_sentence(f).is_some());
    delay.then(|| "It paces itself with busy-wait delay loops, counting down registers to create the pause between actions.".to_string())
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Tests program story generation with empty analysis.
    #[test]
    fn test_program_story_empty() {
        let a = Analysis::default();
        let story = program_story(&a);
        assert!(story.is_empty());
    }

    /// Tests detailed program story with empty analysis.
    #[test]
    fn test_detailed_story_empty() {
        let a = Analysis::default();
        let detailed = detailed_program_story(&a);
        assert!(detailed.boot.is_empty());
    }
}

