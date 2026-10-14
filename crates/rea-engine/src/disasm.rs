//! Recursive-Descent Disassembler
//!
//! **File:** `disasm.rs`
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// Import crate items
use crate::{
    detect::Format,
    model::{Arch, Binary},
    numeric::{to_u64, to_usize},
};

// Import dependencies from iced_x86
use iced_x86::{
    Decoder, DecoderOptions, FlowControl as Flow, Instruction, Mnemonic, OpKind, Register,
};

/// Upper bound on decoded instructions across the whole program.
pub const MAX_INSTRUCTIONS: usize = 4_000_000;

/// Upper bound on discovered functions.
pub const MAX_FUNCTIONS: usize = 100_000;

/// Largest body, in instructions, of a function treated as an import thunk.
const THUNK_SIZE: usize = 2;

/// An analyzed instruction with disassembly and semantic discussion.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InstructionLine {
    /// Virtual memory address of the instruction.
    pub address: u64,
    /// Disassembled instruction mnemonic and operands.
    pub text: String,
    /// Functional category of the instruction.
    pub kind: String,
    /// Detailed analysis and discussion of the instruction operation.
    pub discussion: String,
}

/// An analyzed data entry or literal pool embedded in an executable region.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataEntry {
    /// Virtual memory address of the data.
    pub address: u64,
    /// Raw byte sequence formatted as hex.
    pub bytes_hex: String,
    /// Representation when interpreted as data.
    pub as_data: String,
    /// Representation when disassembled as candidate instructions.
    pub as_code: Option<String>,
    /// Security and contextual discussion of this data block.
    pub discussion: String,
}

/// A function recovered by recursive descent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Function {
    /// Entry address.
    pub address: u64,

    /// Symbol or synthesized name, when known.
    pub name: Option<String>,

    /// Instructions decoded in the function body.
    pub instructions: usize,

    /// Direct call targets inside the program.
    pub calls: BTreeSet<u64>,

    /// Imported functions called directly or through thunks.
    pub imports: BTreeSet<String>,

    /// Addresses referenced by memory operands and immediates.
    pub references: BTreeSet<u64>,

    /// Analyzed assembly instruction sequence.
    pub lines: Vec<InstructionLine>,

    /// Embedded literal pools and data blocks accessed by this function.
    pub data_pools: Vec<DataEntry>,
}

/// The program's recovered code structure.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Disassembly {
    /// Decoder bitness (32 or 64).
    pub bitness: u32,

    /// Functions keyed by entry address.
    pub functions: BTreeMap<u64, Function>,

    /// Total instructions decoded.
    pub instructions: usize,

    /// Whether a resource cap stopped the analysis early.
    pub truncated: bool,
}

impl Function {
    /// Creates an empty function at an address.
    ///
    /// # Arguments
    ///
    /// * `address` - Entry address.
    /// * `name` - Known name, if any.
    ///
    /// # Returns
    ///
    /// A function with no recorded body.
    #[must_use]
    pub fn at(address: u64, name: Option<String>) -> Self {
        Self { address, name, ..Self::default() }
    }
}

impl Disassembly {
    /// Counts the functions that call an address directly.
    ///
    /// # Arguments
    ///
    /// * `address` - Callee address.
    ///
    /// # Returns
    ///
    /// The number of distinct callers.
    #[must_use]
    pub fn callers(&self, address: u64) -> usize {
        self.functions.values().filter(|f| f.calls.contains(&address)).count()
    }

    /// Names a function, synthesizing `sub_<address>` when unnamed.
    ///
    /// # Arguments
    ///
    /// * `address` - Function address.
    ///
    /// # Returns
    ///
    /// The display name.
    #[must_use]
    pub fn name_of(&self, address: u64) -> String {
        let named = self.functions.get(&address).and_then(|f| f.name.clone());
        named.unwrap_or_else(|| format!("sub_{address:x}"))
    }
}

/// The executable regions of a binary, addressable by virtual address.
pub struct CodeMap<'a> {
    /// Start address and file-backed bytes of each executable section.
    regions: Vec<(u64, &'a [u8])>,
}

impl<'a> CodeMap<'a> {
    /// Collects the executable sections of a binary.
    ///
    /// # Arguments
    ///
    /// * `binary` - Parsed executable.
    /// * `bytes` - Whole target file.
    ///
    /// # Returns
    ///
    /// The code map.
    #[must_use]
    pub fn new(binary: &Binary, bytes: &'a [u8]) -> Self {
        let code = binary.sections.iter().filter(|s| s.executable);
        let regions = code.map(|s| (s.address, s.data(bytes))).collect();
        Self { regions }
    }

    /// Returns the code bytes from an address to the end of its region.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// The remaining bytes, or an empty slice when the address is not code.
    #[must_use]
    pub fn bytes_at(&self, address: u64) -> &'a [u8] {
        let inside = |(start, data): &&(u64, &[u8])| {
            address >= *start && address - start < to_u64(data.len())
        };
        let region = self.regions.iter().find(inside);
        region.map_or(&[], |(start, data)| &data[to_usize(address - start)..])
    }

    /// Reports whether an address lies in executable code.
    ///
    /// # Arguments
    ///
    /// * `address` - Virtual address.
    ///
    /// # Returns
    ///
    /// `true` when code bytes exist at the address.
    #[must_use]
    pub fn contains(&self, address: u64) -> bool {
        !self.bytes_at(address).is_empty()
    }

    /// Returns the executable regions as start address and file bytes.
    ///
    /// # Returns
    ///
    /// The regions in discovery order.
    #[must_use]
    pub fn regions(&self) -> &[(u64, &'a [u8])] {
        &self.regions
    }
}

/// Disassembles an x86 or x86-64 binary from its entry point, exports, and symbols.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Whole target file.
///
/// # Returns
///
/// The recovered code structure, or `None` for other architectures.
#[must_use]
pub fn disassemble(binary: &Binary, bytes: &[u8]) -> Option<Disassembly> {
    let bitness = binary.arch.x86_bitness()?;
    let mut walker = Walker::new(binary, bytes, bitness);
    walker.run();
    Some(walker.finish())
}

/// Lists the addresses an instruction's operands refer to.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
///
/// # Returns
///
/// Nonzero memory and immediate operand addresses.
#[must_use]
pub fn referenced_addresses(instruction: &Instruction) -> Vec<u64> {
    let kinds = (0..instruction.op_count()).map(|i| (i, instruction.op_kind(i)));
    let found = kinds.filter_map(|(i, kind)| operand_address(instruction, i, kind));
    found.filter(|address| *address != 0).collect()
}

/// Resolves one operand to an address.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
/// * `index` - Operand index.
/// * `kind` - Operand kind.
///
/// # Returns
///
/// The address for memory and wide immediate operands.
fn operand_address(instruction: &Instruction, index: u32, kind: OpKind) -> Option<u64> {
    match kind {
        OpKind::Memory => memory_address(instruction),
        OpKind::Immediate32 | OpKind::Immediate64 | OpKind::Immediate32to64 => {
            Some(instruction.immediate(index))
        }
        _ => None,
    }
}

/// Resolves a memory operand with a static address.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
///
/// # Returns
///
/// The RIP-relative or absolute address, when register-free.
fn memory_address(instruction: &Instruction) -> Option<u64> {
    if instruction.is_ip_rel_memory_operand() {
        return Some(instruction.ip_rel_memory_address());
    }
    let absolute =
        instruction.memory_base() == Register::None && instruction.memory_index() == Register::None;
    absolute.then(|| instruction.memory_displacement64())
}

/// Reports whether an instruction ends a basic block.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
///
/// # Returns
///
/// `true` for returns, unconditional jumps, traps, and `int3` padding.
#[must_use]
pub fn ends_block(instruction: &Instruction) -> bool {
    let terminal = TERMINALS.contains(&instruction.flow_control());
    terminal || instruction.mnemonic() == Mnemonic::Int3
}

/// Mnemonics whose operands commonly take the address of a callback.
const POINTER_TAKERS: [Mnemonic; 2] = [Mnemonic::Lea, Mnemonic::Push];

/// Control-flow kinds after which execution does not fall through.
const TERMINALS: [Flow; 4] = [
    Flow::Return,
    Flow::UnconditionalBranch,
    Flow::IndirectBranch,
    Flow::Exception,
];

/// Read-only facts the walker consults while decoding.
struct Context<'a> {
    /// Executable regions.
    code: CodeMap<'a>,

    /// Decoder bitness.
    bitness: u32,

    /// Import slot address to imported function name.
    slots: HashMap<u64, String>,

    /// Known function names by address.
    names: HashMap<u64, String>,

    /// Whether to treat taken code addresses as function starts.
    guess_pointers: bool,
}

impl<'a> Context<'a> {
    /// Gathers the decoding context for a binary.
    ///
    /// # Arguments
    ///
    /// * `binary` - Parsed executable.
    /// * `bytes` - Whole target file.
    /// * `bitness` - Decoder bitness.
    ///
    /// # Returns
    ///
    /// The context.
    fn new(binary: &Binary, bytes: &'a [u8], bitness: u32) -> Self {
        Self {
            code: CodeMap::new(binary, bytes),
            bitness,
            slots: import_slots(binary),
            names: symbol_names(binary),
            guess_pointers: binary.function_starts.is_empty() || binary.format != Format::Pe,
        }
    }
}

/// Maps import slot addresses to imported function names.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Slot address to name.
fn import_slots(binary: &Binary) -> HashMap<u64, String> {
    let known = binary.imports.iter().filter(|i| i.address != 0);
    known.map(|i| (i.address, i.name.clone())).collect()
}

/// Maps code addresses to symbol names, naming the entry point.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Address to name.
fn symbol_names(binary: &Binary) -> HashMap<u64, String> {
    let all = binary.symbols.iter().chain(&binary.exports);
    let mut names: HashMap<u64, String> = all.map(|s| (s.address, s.name.clone())).collect();
    names.entry(binary.entry).or_insert_with(|| "entry".to_string());
    names
}

/// Lists the addresses where function discovery starts.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// Symbol and export addresses, with the entry point last so it is traced first.
fn roots(binary: &Binary) -> Vec<u64> {
    let symbols = binary.symbols.iter().chain(&binary.exports);
    let mut roots: Vec<u64> = symbols.map(|s| s.address).collect();
    roots.extend(&binary.function_starts);
    roots.push(binary.entry);
    roots.retain(|address| *address != 0);
    roots
}

/// Per-function tracing state.
struct Trace {
    /// Function being built.
    function: Function,

    /// Basic blocks waiting to be decoded.
    blocks: Vec<u64>,

    /// Instruction addresses already decoded in this function.
    seen: HashSet<u64>,
}

impl Trace {
    /// Starts tracing a function.
    ///
    /// # Arguments
    ///
    /// * `address` - Function entry.
    /// * `name` - Known name, if any.
    ///
    /// # Returns
    ///
    /// Tracing state with the entry block queued.
    fn new(address: u64, name: Option<String>) -> Self {
        Self {
            function: Function::at(address, name),
            blocks: vec![address],
            seen: HashSet::new(),
        }
    }

    /// Records a referenced address as an import call or a data reference.
    ///
    /// # Arguments
    ///
    /// * `address` - Referenced address.
    /// * `slot` - Import name when the address is an import slot.
    fn record(&mut self, address: u64, slot: Option<&String>) {
        if let Some(name) = slot {
            self.function.imports.insert(name.clone());
        } else {
            self.function.references.insert(address);
        }
    }
}

/// Recursive-descent function discovery engine.
struct Walker<'a> {
    /// Decoding context.
    context: Context<'a>,

    /// Function entries waiting to be traced.
    pending: Vec<u64>,

    /// Structure recovered so far.
    result: Disassembly,
}

impl<'a> Walker<'a> {
    /// Creates a walker seeded with the binary's roots.
    ///
    /// # Arguments
    ///
    /// * `binary` - Parsed executable.
    /// * `bytes` - Whole target file.
    /// * `bitness` - Decoder bitness.
    ///
    /// # Returns
    ///
    /// The walker.
    fn new(binary: &Binary, bytes: &'a [u8], bitness: u32) -> Self {
        Self {
            context: Context::new(binary, bytes, bitness),
            pending: roots(binary),
            result: Disassembly { bitness, ..Disassembly::default() },
        }
    }

    /// Traces functions until none remain or the function cap is reached.
    fn run(&mut self) {
        while let Some(address) = self.pending.pop() {
            if self.result.functions.len() >= MAX_FUNCTIONS {
                self.result.truncated = true;
                break;
            }
            self.discover(address);
        }
    }

    /// Traces one function unless it is known or not code.
    ///
    /// # Arguments
    ///
    /// * `start` - Candidate function entry.
    fn discover(&mut self, start: u64) {
        if self.result.functions.contains_key(&start) || !self.context.code.contains(start) {
            return;
        }
        let function = self.trace(start);
        self.result.functions.insert(start, function);
    }

    /// Decodes every reachable block of one function.
    ///
    /// # Arguments
    ///
    /// * `start` - Function entry.
    ///
    /// # Returns
    ///
    /// The recovered function.
    fn trace(&mut self, start: u64) -> Function {
        let mut trace = Trace::new(start, self.context.names.get(&start).cloned());
        while let Some(block) = trace.blocks.pop() {
            self.trace_block(block, &mut trace);
        }
        trace.function.lines.sort_by_key(|l| l.address);
        trace.function.lines.dedup_by_key(|l| l.address);
        trace.function
    }

    /// Decodes one basic block.
    ///
    /// # Arguments
    ///
    /// * `address` - Block start.
    /// * `trace` - Function tracing state.
    fn trace_block(&mut self, address: u64, trace: &mut Trace) {
        let bytes = self.context.code.bytes_at(address);
        let decoder = Decoder::with_ip(self.context.bitness, bytes, address, DecoderOptions::NONE);
        for instruction in decoder {
            if !self.accept(&instruction, trace) || self.step(&instruction, trace) {
                break;
            }
        }
    }

    /// Counts a freshly decoded instruction, enforcing the global cap.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    /// * `trace` - Function tracing state.
    ///
    /// # Returns
    ///
    /// `false` when decoding should stop for this block.
    fn accept(&mut self, instruction: &Instruction, trace: &mut Trace) -> bool {
        let room = self.result.instructions < MAX_INSTRUCTIONS;
        self.result.truncated |= !room;
        let fresh = room && !instruction.is_invalid() && trace.seen.insert(instruction.ip());
        if fresh {
            self.record_line(instruction, trace);
        }
        fresh
    }

    /// Records a disassembled line into trace state.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    /// * `trace` - Function tracing state.
    fn record_line(&mut self, instruction: &Instruction, trace: &mut Trace) {
        self.result.instructions += 1;
        trace.function.instructions += 1;
        trace.function.lines.push(make_instruction_line(instruction));
    }

    /// Follows control flow and records references for one instruction.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    /// * `trace` - Function tracing state.
    ///
    /// # Returns
    ///
    /// `true` when the instruction ends the block.
    fn step(&mut self, instruction: &Instruction, trace: &mut Trace) -> bool {
        self.note(instruction, trace);
        let target = instruction.near_branch_target();
        match instruction.flow_control() {
            Flow::Call => self.on_call(target, trace),
            Flow::ConditionalBranch | Flow::UnconditionalBranch => trace.blocks.push(target),
            _ => {}
        }
        ends_block(instruction)
    }

    /// Records every address an instruction references.
    ///
    /// Without format-provided function starts, code addresses taken by
    /// `lea` or `push` are queued as likely callbacks.
    ///
    /// # Arguments
    ///
    /// * `instruction` - Decoded instruction.
    /// * `trace` - Function tracing state.
    fn note(&mut self, instruction: &Instruction, trace: &mut Trace) {
        let taker = POINTER_TAKERS.contains(&instruction.mnemonic());
        let guess = taker && self.context.guess_pointers;
        for address in referenced_addresses(instruction) {
            trace.record(address, self.context.slots.get(&address));
            if guess && self.context.code.contains(address) {
                self.pending.push(address);
            }
        }
    }

    /// Records a direct call and queues its target for discovery.
    ///
    /// # Arguments
    ///
    /// * `target` - Call target.
    /// * `trace` - Function tracing state.
    fn on_call(&mut self, target: u64, trace: &mut Trace) {
        if self.context.code.contains(target) {
            trace.function.calls.insert(target);
            self.pending.push(target);
        }
    }

    /// Propagates imports through thunks and returns the result.
    ///
    /// # Returns
    ///
    /// The completed disassembly.
    fn finish(mut self) -> Disassembly {
        let thunks = thunk_names(&self.result);
        for function in self.result.functions.values_mut() {
            inherit(function, &thunks);
        }
        self.result
    }
}

/// Finds functions that only jump to an import.
///
/// # Arguments
///
/// * `result` - Recovered structure.
///
/// # Returns
///
/// Thunk address to imported function name.
fn thunk_names(result: &Disassembly) -> HashMap<u64, String> {
    let tiny = result
        .functions
        .values()
        .filter(|f| f.instructions <= THUNK_SIZE && f.calls.is_empty());
    let single = tiny.filter(|f| f.imports.len() == 1);
    single.filter_map(|f| Some((f.address, f.imports.first()?.clone()))).collect()
}

/// Attributes thunk targets to callers and names the thunks themselves.
///
/// # Arguments
///
/// * `function` - Function to update.
/// * `thunks` - Thunk address to imported function name.
fn inherit(function: &mut Function, thunks: &HashMap<u64, String>) {
    let called = function.calls.iter().filter_map(|c| thunks.get(c));
    let names: Vec<String> = called.cloned().collect();
    function.imports.extend(names);
    if function.name.is_none() {
        let thunk = thunks.get(&function.address);
        function.name = thunk.map(|name| format!("j_{name}"));
    }
}

/// Builds an instruction line with semantic classification and discussion.
///
/// # Arguments
///
/// * `instruction` - Decoded instruction.
///
/// # Returns
///
/// Constructed instruction line.
fn make_instruction_line(instruction: &Instruction) -> InstructionLine {
    let (ip, text) = (instruction.ip(), format!("{instruction}"));
    let (kind, discussion) = crate::semantics::discuss(Arch::X86_64, ip, &text);
    InstructionLine { address: ip, text, kind, discussion }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;
    // Import crate items
    use crate::{
        detect::Format,
        model::{Arch, Import, Section},
    };

    /// Builds a 64-bit binary whose `.text` holds `code` at 0x1000.
    ///
    /// # Arguments
    ///
    /// * `code` - Machine code placed at file offset zero.
    ///
    /// # Returns
    ///
    /// The binary with an import slot at 0x3000.
    fn sample(code: &[u8]) -> Binary {
        let size = to_u64(code.len());
        let text = Section::mapped(".text", 0x1000, 0x100).with_file(0, size);
        let sections = vec![text.with_access(true, false)];
        let imports = vec![Import::new("KERNEL32.dll", "ExitProcess", 0x3000)];
        let mut binary = Binary::new(Format::Pe, Arch::X86_64, "x86-64", 64);
        (binary.entry, binary.sections, binary.imports) = (0x1000, sections, imports);
        binary
    }

    /// Verifies call discovery, import attribution, and termination at `ret`.
    #[test]
    fn test_disassemble_follows_calls_and_imports() {
        let code = [
            0xe8, 0x07, 0, 0, 0, 0xff, 0x15, 0xf5, 0x1f, 0, 0, 0xc3, 0xc3,
        ];
        let result = disassemble(&sample(&code), &code).unwrap();
        let entry = &result.functions[&0x1000];
        assert!(entry.calls.contains(&0x100c));
        assert!(entry.imports.contains("ExitProcess"));
        assert_eq!(result.functions.len(), 2);
    }

    /// Verifies that non-x86 binaries are skipped.
    #[test]
    fn test_disassemble_skips_other_arches() {
        let binary = Binary::new(Format::Elf, Arch::Arm64, "aarch64", 64);
        assert!(disassemble(&binary, &[]).is_none());
    }

    /// Verifies that conditional branches keep both paths in one function.
    #[test]
    fn test_conditional_branch_stays_local() {
        let code = [0x74, 0x01, 0xc3, 0x90, 0xc3];
        let result = disassemble(&sample(&code), &code).unwrap();
        assert_eq!(result.functions[&0x1000].instructions, 4);
    }

    /// Verifies synthesized names and caller counts.
    #[test]
    fn test_name_of_and_callers() {
        let code = [0xe8, 0x01, 0, 0, 0, 0xc3, 0xc3];
        let result = disassemble(&sample(&code), &code).unwrap();
        assert_eq!(result.name_of(0x1006), "sub_1006");
        assert_eq!(result.callers(0x1006), 1);
    }

    /// Verifies code map lookups at region boundaries.
    #[test]
    fn test_code_map_bounds() {
        let code = [0x90u8; 4];
        let binary = sample(&code);
        let map = CodeMap::new(&binary, &code);
        assert_eq!(map.bytes_at(0x1003).len(), 1);
        assert!(!map.contains(0x1004));
    }

    /// Verifies that import thunks are named and attributed to callers.
    #[test]
    fn test_thunks_propagate_imports() {
        let code = [0xe8, 0x01, 0, 0, 0, 0xc3, 0xff, 0x25, 0xf4, 0x1f, 0, 0];
        let result = disassemble(&sample(&code), &code).unwrap();
        assert_eq!(result.name_of(0x1006), "j_ExitProcess");
        assert!(result.functions[&0x1000].imports.contains("ExitProcess"));
    }
}
