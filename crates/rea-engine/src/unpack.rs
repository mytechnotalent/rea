//! Emulation-Driven Unpacking
//!
//! **File:** `unpack.rs`
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
use std::collections::BTreeMap;

// Import dependencies from rea_sleigh
use rea_sleigh::Emulator;

// Import crate items
use crate::{
    discover::{self, Finding},
    emulate::Stop,
    model::{Arch, Binary, Section},
    multiarch,
};

/// Page size used for memory snapshots.
const PAGE: usize = 0x1000;

/// Base offset of synthetic import stubs above the image base.
const STUB_OFFSET: u64 = 0x1000_0000;

/// Spacing between synthetic import stubs.
const STUB_STRIDE: u64 = 0x10;

/// Base offset of synthetic allocations above the image base.
const ALLOC_OFFSET: u64 = 0x2000_0000;

/// Stack gap placed above the image, in bytes.
const STACK_GAP: u64 = 0x0010_0000;

/// Largest number of changed regions reported.
const MAX_REGIONS: usize = 64;

/// A memory region whose contents changed during emulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Page address of the region.
    pub address: u64,

    /// Page bytes after emulation.
    pub bytes: Vec<u8>,
}

/// The result of an unpacking run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unpacked {
    /// Instructions executed.
    pub steps: u64,

    /// Why the run stopped.
    pub stop: Stop,

    /// Program counter where the run stopped.
    pub exit: u64,

    /// Regions written during emulation.
    pub regions: Vec<Region>,

    /// Imports referenced by decrypted code.
    pub imports: Vec<String>,

    /// Flag-shaped strings recovered from emulated memory.
    pub found: Vec<Finding>,
}

impl Unpacked {
    /// Reports the original entry point when control reached a decrypted region.
    ///
    /// # Returns
    ///
    /// The original entry point, or `None`.
    #[must_use]
    pub fn oep(&self) -> Option<u64> {
        entry_point(&self.regions, self.exit)
    }
}

/// Unpacks a target by emulating every candidate entry and merging results.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `entries` - Candidate entry addresses.
/// * `budget` - Maximum instructions per entry.
///
/// # Returns
///
/// The merged outcome, or `None` when the architecture is unsupported.
#[must_use]
pub fn sweep(binary: &Binary, bytes: &[u8], entries: &[u64], budget: u64) -> Option<Unpacked> {
    let mut acc = Accum::default();
    for &entry in entries {
        accumulate(binary, bytes, entry, budget, &mut acc);
    }
    Some(acc.finish())
}

/// Unpacks a target across rounds, re-feeding decrypted pages until stable.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `entries` - Candidate entry addresses.
/// * `budget` - Maximum instructions per entry.
/// * `rounds` - Maximum convergence rounds.
///
/// # Returns
///
/// The merged outcome, or `None` when the architecture is unsupported.
#[must_use]
pub fn converge(
    binary: &Binary,
    bytes: &[u8],
    entries: &[u64],
    budget: u64,
    rounds: usize,
) -> Option<Unpacked> {
    let mut current = bytes.to_vec();
    let mut merged = BTreeMap::new();
    run_rounds(binary, &mut current, entries, budget, rounds, &mut merged);
    Some(converged(to_regions(merged)))
}

/// Builds a converged outcome.
///
/// # Arguments
///
/// * `regions` - Decrypted regions.
///
/// # Returns
///
/// The outcome.
fn converged(regions: Vec<Region>) -> Unpacked {
    outcome(0, Stop::Budget, 0, regions, Vec::new(), Vec::new())
}

/// Runs up to `rounds` convergence rounds.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `current` - Working image bytes.
/// * `entries` - Candidate entry addresses.
/// * `budget` - Maximum instructions per entry.
/// * `rounds` - Maximum convergence rounds.
/// * `merged` - Accumulated pages.
fn run_rounds(
    binary: &Binary,
    current: &mut [u8],
    entries: &[u64],
    budget: u64,
    rounds: usize,
    merged: &mut BTreeMap<u64, Vec<u8>>,
) {
    for _ in 0..rounds {
        if !step_converge(binary, current, entries, budget, merged) {
            break;
        }
    }
}

/// Runs one convergence round and applies new pages.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `current` - Working image bytes.
/// * `entries` - Candidate entry addresses.
/// * `budget` - Maximum instructions per entry.
/// * `merged` - Accumulated pages.
///
/// # Returns
///
/// `true` when new pages were found.
fn step_converge(
    binary: &Binary,
    current: &mut [u8],
    entries: &[u64],
    budget: u64,
    merged: &mut BTreeMap<u64, Vec<u8>>,
) -> bool {
    let Some(result) = sweep(binary, current, entries, budget) else {
        return false;
    };
    if !merge(merged, &result.regions) {
        return false;
    }
    apply(binary, current, &result.regions);
    true
}

/// Merges new regions, reporting whether anything changed.
///
/// # Arguments
///
/// * `merged` - Accumulated pages.
/// * `regions` - New regions.
///
/// # Returns
///
/// `true` when a page changed.
fn merge(merged: &mut BTreeMap<u64, Vec<u8>>, regions: &[Region]) -> bool {
    let mut changed = false;
    for region in regions {
        if merged.get(&region.address) != Some(&region.bytes) {
            merged.insert(region.address, region.bytes.clone());
            changed = true;
        }
    }
    changed
}

/// Writes decrypted regions back into the working image.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `current` - Working image bytes.
/// * `regions` - Decrypted regions.
fn apply(binary: &Binary, current: &mut [u8], regions: &[Region]) {
    for region in regions {
        if let Some(offset) = file_offset(binary, region.address) {
            write_at(current, offset, &region.bytes);
        }
    }
}

/// Maps a virtual address to a file offset.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `address` - Virtual address.
///
/// # Returns
///
/// The file offset, or `None`.
fn file_offset(binary: &Binary, address: u64) -> Option<usize> {
    let section = binary.sections.iter().find(|s| s.contains(address))?;
    usize::try_from(section.file_offset_of(address)?).ok()
}

/// Copies bytes into a buffer at an offset when in range.
///
/// # Arguments
///
/// * `current` - Working image bytes.
/// * `offset` - Destination offset.
/// * `bytes` - Source bytes.
fn write_at(current: &mut [u8], offset: usize, bytes: &[u8]) {
    let end = offset.saturating_add(bytes.len());
    if let Some(slice) = current.get_mut(offset..end) {
        slice.copy_from_slice(bytes);
    }
}

/// Accumulated results across sweep entries.
struct Accum {
    /// Merged changed pages.
    merged: BTreeMap<u64, Vec<u8>>,

    /// Instructions executed across entries.
    steps: u64,

    /// Last stop reason.
    stop: Stop,

    /// Last exit program counter.
    exit: u64,
}

impl Default for Accum {
    /// Builds an empty accumulator.
    ///
    /// # Returns
    ///
    /// The empty accumulator.
    fn default() -> Self {
        Self { merged: BTreeMap::new(), steps: 0, stop: Stop::Budget, exit: 0 }
    }
}

impl Accum {
    /// Builds the merged unpacking outcome.
    ///
    /// # Returns
    ///
    /// The outcome.
    fn finish(self) -> Unpacked {
        outcome(
            self.steps,
            self.stop,
            self.exit,
            to_regions(self.merged),
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Converts merged pages into regions.
///
/// # Arguments
///
/// * `merged` - Merged page bytes by address.
///
/// # Returns
///
/// The region list.
fn to_regions(merged: BTreeMap<u64, Vec<u8>>) -> Vec<Region> {
    merged.into_iter().map(|(address, bytes)| Region { address, bytes }).collect()
}

/// Runs one entry and merges its changed pages.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `entry` - Entry address.
/// * `budget` - Maximum instructions.
/// * `acc` - Accumulated results.
fn accumulate(binary: &Binary, bytes: &[u8], entry: u64, budget: u64, acc: &mut Accum) {
    let Some(result) = run(binary, bytes, entry, budget) else {
        return;
    };
    record(acc, &result);
}

/// Records one run's results into the accumulator.
///
/// # Arguments
///
/// * `acc` - Accumulated results.
/// * `result` - One run's outcome.
fn record(acc: &mut Accum, result: &Unpacked) {
    acc.steps += result.steps;
    acc.stop = result.stop;
    acc.exit = result.exit;
    for region in &result.regions {
        acc.merged.insert(region.address, region.bytes.clone());
    }
}

/// Mutable state carried across the emulation loop.
struct State {
    /// Address of the next synthetic allocation.
    alloc: u64,

    /// Pointer size in bytes.
    ptr: usize,

    /// Accumulator register name.
    acc_reg: &'static str,

    /// Stack-pointer register name.
    stack_reg: &'static str,

    /// Link register name, when returns use one instead of the stack.
    link_reg: Option<&'static str>,
}

/// Unpacks a target by emulating it with stubbed imports.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `entry` - Address to begin execution at.
/// * `budget` - Maximum instructions to execute.
///
/// # Returns
///
/// The unpacking outcome, or `None` when the architecture is unsupported.
#[must_use]
pub fn run(binary: &Binary, bytes: &[u8], entry: u64, budget: u64) -> Option<Unpacked> {
    let mut emu = build(binary)?;
    let state = State::new(binary)?;
    prepare(&mut emu, binary, bytes, entry, &state);
    let stubs = install_stubs(&mut emu, binary, state.ptr);
    let baseline = snapshot(&emu);
    Some(execute(
        &mut emu, &baseline, bytes, &stubs, &state, entry, budget,
    ))
}

/// Runs the emulation and assembles the outcome, including recovered imports.
///
/// # Arguments
///
/// * `emu` - Emulator after preparation.
/// * `baseline` - Pre-run page snapshot.
/// * `bytes` - Raw file contents.
/// * `stubs` - Installed import stubs.
/// * `state` - Emulation register state.
/// * `entry` - Entry address.
/// * `budget` - Maximum instructions.
///
/// # Returns
///
/// The outcome.
fn execute(
    emu: &mut Emulator,
    baseline: &BTreeMap<u64, Vec<u8>>,
    bytes: &[u8],
    stubs: &BTreeMap<u64, String>,
    state: &State,
    entry: u64,
    budget: u64,
) -> Unpacked {
    let (steps, stop, exit) = drive(emu, stubs, state, entry, budget);
    let regions = changed(baseline, emu);
    let imports = rebuild_imports(&regions, stubs, state.ptr);
    let found = discover::scan(emu, bytes);
    outcome(steps, stop, exit, regions, imports, found)
}

/// Builds an unpacking outcome.
///
/// # Arguments
///
/// * `steps` - Instructions executed.
/// * `stop` - Stop reason.
/// * `exit` - Exit program counter.
/// * `regions` - Regions written during emulation.
/// * `imports` - Imports referenced by decrypted code.
/// * `found` - Flag-shaped strings recovered from memory.
///
/// # Returns
///
/// The outcome.
fn outcome(
    steps: u64,
    stop: Stop,
    exit: u64,
    regions: Vec<Region>,
    imports: Vec<String>,
    found: Vec<Finding>,
) -> Unpacked {
    Unpacked { steps, stop, exit, regions, imports, found }
}

/// Recovers the imports referenced by decrypted code.
///
/// # Arguments
///
/// * `regions` - Regions written during emulation.
/// * `stubs` - Installed import stubs.
/// * `ptr` - Pointer size in bytes.
///
/// # Returns
///
/// The referenced import names.
fn rebuild_imports(regions: &[Region], stubs: &BTreeMap<u64, String>, ptr: usize) -> Vec<String> {
    let mut names = Vec::new();
    for region in regions {
        collect_imports(&mut names, &region.bytes, stubs, ptr);
    }
    names
}

/// Collects import names referenced within one region.
///
/// # Arguments
///
/// * `names` - Accumulated import names.
/// * `bytes` - Region bytes.
/// * `stubs` - Installed import stubs.
/// * `ptr` - Pointer size in bytes.
fn collect_imports(
    names: &mut Vec<String>,
    bytes: &[u8],
    stubs: &BTreeMap<u64, String>,
    ptr: usize,
) {
    for chunk in bytes.chunks(ptr).filter(|c| c.len() == ptr) {
        if let Some(name) = stub_name(chunk, stubs) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
}

/// Resolves a little-endian pointer to an import name.
///
/// # Arguments
///
/// * `chunk` - Pointer-sized bytes.
/// * `stubs` - Installed import stubs.
///
/// # Returns
///
/// The import name, or `None`.
fn stub_name(chunk: &[u8], stubs: &BTreeMap<u64, String>) -> Option<String> {
    stubs.get(&read_le(chunk)).cloned()
}

/// Reads a little-endian pointer value.
///
/// # Arguments
///
/// * `chunk` - Pointer-sized bytes.
///
/// # Returns
///
/// The pointer value.
fn read_le(chunk: &[u8]) -> u64 {
    chunk.iter().rev().fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte))
}

/// Loads the image, sets the stack, and positions the program counter.
///
/// # Arguments
///
/// * `emu` - Emulator to prepare.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
/// * `entry` - Entry address.
/// * `state` - Emulation register state.
fn prepare(emu: &mut Emulator, binary: &Binary, bytes: &[u8], entry: u64, state: &State) {
    load(emu, binary, bytes);
    set_stack(emu, binary, state);
    emu.set_pc(entry);
}

/// Positions the stack pointer above the image.
///
/// # Arguments
///
/// * `emu` - Emulator to prepare.
/// * `binary` - Parsed executable.
/// * `state` - Emulation register state.
fn set_stack(emu: &mut Emulator, binary: &Binary, state: &State) {
    let hi = binary.sections.iter().map(section_end).max().unwrap_or(binary.image_base);
    let _ = emu.write_reg(state.stack_reg, hi.saturating_add(STACK_GAP));
}

/// Returns the exclusive end of a section's mapped range.
///
/// # Arguments
///
/// * `section` - Section to measure.
///
/// # Returns
///
/// The address just past the section.
fn section_end(section: &Section) -> u64 {
    section.address.saturating_add(section.virtual_size.max(section.file_size))
}

/// Records the original entry point when control reached a decrypted region.
///
/// # Arguments
///
/// * `regions` - Regions written during emulation.
/// * `exit` - Program counter where the run stopped.
///
/// # Returns
///
/// The original entry point, or `None`.
fn entry_point(regions: &[Region], exit: u64) -> Option<u64> {
    regions.iter().any(|region| within(region, exit)).then_some(exit)
}

/// Reports whether an address lies within a region.
///
/// # Arguments
///
/// * `region` - Memory region.
/// * `address` - Candidate address.
///
/// # Returns
///
/// `true` when the address is inside the region.
fn within(region: &Region, address: u64) -> bool {
    let end = region.address.saturating_add(u64::try_from(region.bytes.len()).unwrap_or(0));
    address >= region.address && address < end
}

impl State {
    /// Builds emulation register state for a binary.
    ///
    /// # Arguments
    ///
    /// * `binary` - Parsed executable.
    ///
    /// # Returns
    ///
    /// The state, or `None` for unsupported architectures.
    fn new(binary: &Binary) -> Option<Self> {
        let (acc_reg, stack_reg, link_reg) = registers(binary.arch)?;
        let ptr = if binary.bits == 64 { 8 } else { 4 };
        Some(Self { alloc: binary.image_base + ALLOC_OFFSET, ptr, acc_reg, stack_reg, link_reg })
    }
}

/// Returns the accumulator and stack register names for an architecture.
///
/// # Arguments
///
/// * `arch` - Architecture family.
///
/// # Returns
///
/// Accumulator and stack register names, or `None`.
fn registers(arch: Arch) -> Option<(&'static str, &'static str, Option<&'static str>)> {
    match arch {
        Arch::X86 => Some(("EAX", "ESP", None)),
        Arch::X86_64 => Some(("RAX", "RSP", None)),
        other => registers_other(other),
    }
}

/// Returns register names for non-x86 architectures.
///
/// # Arguments
///
/// * `arch` - Architecture family.
///
/// # Returns
///
/// Accumulator, stack, and link register names, or `None`.
fn registers_other(arch: Arch) -> Option<(&'static str, &'static str, Option<&'static str>)> {
    match arch {
        Arch::Arm => Some(("r0", "sp", Some("lr"))),
        Arch::Arm64 => Some(("x0", "sp", Some("x30"))),
        Arch::Mips | Arch::Mips64 => Some(("v0", "sp", Some("ra"))),
        Arch::Riscv | Arch::Riscv64 => Some(("a0", "sp", Some("ra"))),
        _ => None,
    }
}

/// Builds an emulator for a binary's architecture.
///
/// # Arguments
///
/// * `binary` - Parsed executable.
///
/// # Returns
///
/// The emulator, or `None`.
fn build(binary: &Binary) -> Option<Emulator> {
    Emulator::new(multiarch::emulation_arch(binary)?).ok()
}

/// Loads sections and sets the stack.
///
/// # Arguments
///
/// * `emu` - Emulator to prepare.
/// * `binary` - Parsed executable.
/// * `bytes` - Raw file contents.
fn load(emu: &mut Emulator, binary: &Binary, bytes: &[u8]) {
    for section in &binary.sections {
        if let Some(data) = section_bytes(section, bytes) {
            emu.load(section.address, data);
        }
    }
}

/// Slices a section's bytes from the file image.
///
/// # Arguments
///
/// * `section` - Section to slice.
/// * `bytes` - Raw file contents.
///
/// # Returns
///
/// The section bytes, or `None`.
fn section_bytes<'a>(section: &Section, bytes: &'a [u8]) -> Option<&'a [u8]> {
    let start = usize::try_from(section.file_offset).ok()?;
    let len = usize::try_from(section.file_size).ok()?;
    bytes.get(start..start.checked_add(len)?)
}

/// Installs synthetic import stubs into the IAT.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `binary` - Parsed executable.
/// * `ptr` - Pointer size in bytes.
///
/// # Returns
///
/// A map from stub address to import name.
fn install_stubs(emu: &mut Emulator, binary: &Binary, ptr: usize) -> BTreeMap<u64, String> {
    let mut stubs = BTreeMap::new();
    for (index, import) in binary.imports.iter().enumerate() {
        let stub = binary.image_base + STUB_OFFSET + index as u64 * STUB_STRIDE;
        emu.write_mem(import.address, ptr, stub);
        stubs.insert(stub, import.name.clone());
    }
    stubs
}

/// Snapshots every mapped page.
///
/// # Arguments
///
/// * `emu` - Emulator to snapshot.
///
/// # Returns
///
/// A map from page address to page bytes.
fn snapshot(emu: &Emulator) -> BTreeMap<u64, Vec<u8>> {
    let mut map = BTreeMap::new();
    for page in emu.mapped_pages() {
        map.insert(page, emu.read_bytes(page, PAGE));
    }
    map
}

/// Collects pages whose bytes changed since the snapshot.
///
/// # Arguments
///
/// * `baseline` - Pre-run page snapshot.
/// * `emu` - Emulator after the run.
///
/// # Returns
///
/// The changed regions, capped at [`MAX_REGIONS`].
fn changed(baseline: &BTreeMap<u64, Vec<u8>>, emu: &Emulator) -> Vec<Region> {
    let mut regions = Vec::new();
    for (page, before) in baseline {
        collect_change(&mut regions, *page, before, emu);
        if regions.len() >= MAX_REGIONS {
            break;
        }
    }
    regions
}

/// Records a page as a region when its bytes changed.
///
/// # Arguments
///
/// * `regions` - Accumulated regions.
/// * `page` - Page address.
/// * `before` - Page bytes before the run.
/// * `emu` - Emulator after the run.
fn collect_change(regions: &mut Vec<Region>, page: u64, before: &[u8], emu: &Emulator) {
    let after = emu.read_bytes(page, before.len());
    if after != before {
        regions.push(Region { address: page, bytes: after });
    }
}

/// Runs the step loop, servicing import stubs as they are reached.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `stubs` - Import stub map.
/// * `state` - Emulation register state.
/// * `entry` - Entry address.
/// * `budget` - Maximum instructions.
///
/// # Returns
///
/// The step count, stop reason, and exit program counter.
fn drive(
    emu: &mut Emulator,
    stubs: &BTreeMap<u64, String>,
    state: &State,
    entry: u64,
    budget: u64,
) -> (u64, Stop, u64) {
    let range = code_range(emu, entry);
    for steps in 0..budget {
        if let Some((stop, exit)) = tick(emu, stubs, state, range) {
            return (steps + 1, stop, exit);
        }
    }
    (budget, Stop::Budget, emu.pc())
}

/// Executes one step or services a stub, reporting a stop when the run ends.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `stubs` - Import stub map.
/// * `state` - Emulation register state.
/// * `range` - Mapped bounds.
///
/// # Returns
///
/// A stop reason and exit program counter, or `None` to continue.
fn tick(
    emu: &mut Emulator,
    stubs: &BTreeMap<u64, String>,
    state: &State,
    range: (u64, u64),
) -> Option<(Stop, u64)> {
    match emu.step() {
        Ok(step) => after_step(emu, stubs, state, range, step.pc),
        Err(_) => Some((Stop::DecodeError, emu.pc())),
    }
}

/// Services a stub or reports a stop after one step.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `stubs` - Import stub map.
/// * `state` - Emulation register state.
/// * `range` - Mapped bounds.
/// * `pc` - Program counter after the step.
///
/// # Returns
///
/// A stop reason and exit program counter, or `None` to continue.
fn after_step(
    emu: &mut Emulator,
    stubs: &BTreeMap<u64, String>,
    state: &State,
    range: (u64, u64),
    pc: u64,
) -> Option<(Stop, u64)> {
    if let Some(name) = stubs.get(&pc) {
        handle_stub(emu, name, state);
        return None;
    }
    (!in_range(pc, range)).then_some((Stop::LeftMappedCode, pc))
}

/// Services one import stub and returns to the caller.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `name` - Imported function name.
/// * `state` - Emulation register state.
fn handle_stub(emu: &mut Emulator, name: &str, state: &State) {
    let value = api_value(name, emu, state);
    let _ = emu.write_reg(state.acc_reg, value);
    return_to_caller(emu, state);
}

/// Returns control from a stub to its caller.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `state` - Emulation register state.
fn return_to_caller(emu: &mut Emulator, state: &State) {
    match state.link_reg {
        Some(link) => link_return(emu, link),
        None => stack_return(emu, state),
    }
}

/// Returns through a link register.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `link` - Link register name.
fn link_return(emu: &mut Emulator, link: &str) {
    let target = emu.read_reg(link).unwrap_or(0);
    emu.set_pc(target);
}

/// Returns by popping a stack return address.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `state` - Emulation register state.
fn stack_return(emu: &mut Emulator, state: &State) {
    let stack = emu.read_reg(state.stack_reg).unwrap_or(0);
    let target = emu.read_mem(stack, state.ptr);
    emu.set_pc(target);
    let _ = emu.write_reg(state.stack_reg, stack + state.ptr as u64);
}

/// Computes a plausible return value for a stubbed API.
///
/// # Arguments
///
/// * `name` - Imported function name.
/// * `emu` - Emulator to patch.
/// * `state` - Emulation register state.
///
/// # Returns
///
/// The accumulator return value.
fn api_value(name: &str, emu: &mut Emulator, state: &State) -> u64 {
    let lower = name.to_ascii_lowercase();
    if is_allocator(&lower) {
        return allocate(emu, state);
    }
    if is_boolean_api(&lower) {
        return 1;
    }
    0
}

/// Reports whether an API is an allocator.
///
/// # Arguments
///
/// * `name` - Lowercased import name.
///
/// # Returns
///
/// `true` for allocation APIs.
fn is_allocator(name: &str) -> bool {
    ["virtualalloc", "heapalloc", "malloc"].iter().any(|key| name.contains(key))
}

/// Reports whether an API returns a boolean success flag.
///
/// # Arguments
///
/// * `name` - Lowercased import name.
///
/// # Returns
///
/// `true` for boolean-returning APIs.
fn is_boolean_api(name: &str) -> bool {
    ["virtualprotect", "virtualquery", "loadlibrary"]
        .iter()
        .any(|key| name.contains(key))
}

/// Allocates a zeroed page for a stubbed allocator.
///
/// # Arguments
///
/// * `emu` - Emulator to patch.
/// * `state` - Emulation register state.
///
/// # Returns
///
/// The allocation address.
fn allocate(emu: &mut Emulator, state: &State) -> u64 {
    let address = state.alloc;
    emu.load(address, &[0u8; PAGE]);
    address
}

/// Computes the mapped address range of the image.
///
/// # Arguments
///
/// * `emu` - Prepared emulator.
/// * `entry` - Entry address.
///
/// # Returns
///
/// The inclusive-low, exclusive-high bounds.
fn code_range(emu: &Emulator, entry: u64) -> (u64, u64) {
    let pages = emu.mapped_pages();
    let lo = pages.first().copied().unwrap_or(entry);
    let hi = pages.last().map_or(entry, |page| page + PAGE as u64);
    (lo, hi)
}

/// Reports whether an address lies within the mapped image.
///
/// # Arguments
///
/// * `pc` - Program counter.
/// * `range` - Mapped bounds.
///
/// # Returns
///
/// `true` when the address is mapped.
fn in_range(pc: u64, range: (u64, u64)) -> bool {
    let address = pc & !1;
    address >= range.0 && address < range.1
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies register selection per architecture.
    #[test]
    fn test_registers() {
        assert_eq!(registers(Arch::X86_64), Some(("RAX", "RSP", None)));
        assert_eq!(registers(Arch::Arm), Some(("r0", "sp", Some("lr"))));
        assert_eq!(registers(Arch::Sparc), None);
    }

    /// Verifies stub API classification.
    #[test]
    fn test_api_classification() {
        assert!(is_allocator("virtualalloc"));
        assert!(is_boolean_api("virtualprotect"));
        assert!(!is_allocator("sleep"));
    }

    /// Verifies mapped-range checks.
    #[test]
    fn test_in_range() {
        assert!(in_range(0x1000, (0x1000, 0x2000)));
        assert!(!in_range(0x2000, (0x1000, 0x2000)));
    }

    /// Verifies section end computation.
    #[test]
    fn test_section_end() {
        let section = Section::mapped(".text", 0x1000, 0x100).with_file(0, 0x80);
        assert_eq!(section_end(&section), 0x1100);
    }
}
