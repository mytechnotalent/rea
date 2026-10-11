//! Architecture Registry
//!
//! **File:** `arch.rs`
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

/// A decode target: an embedded specification plus its decode context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    /// Short human-readable name.
    pub name: &'static str,

    /// Family-relative root specification path.
    pub root: &'static str,

    /// Whether the target is big-endian.
    pub big_endian: bool,

    /// Address width in bits.
    pub bits: u32,

    /// Context fields to set before decoding, as `(field, value)` pairs.
    pub context: &'static [(&'static str, u32)],

    /// Whether code addresses with bit 0 set select a sub-mode (ARM Thumb).
    pub interworking: bool,
}

/// ARM Thumb context selecting 16/32-bit Thumb decoding.
const THUMB: &[(&str, u32)] = &[("TMode", 1)];

/// RISC-V 64-bit machine-length context.
const RV64: &[(&str, u32)] = &[("MXL", 2)];

/// No additional context.
const NONE: &[(&str, u32)] = &[];

/// 32-bit ARM, little-endian (A32).
pub const ARM_LE: Arch = arm("ARM (A32, LE)", "ARM/ARM7_le.slaspec", false, NONE);

/// 32-bit ARM, big-endian (A32).
pub const ARM_BE: Arch = arm("ARM (A32, BE)", "ARM/ARM7_be.slaspec", true, NONE);

/// ARM Thumb, little-endian (Cortex-M and Thumb functions).
pub const THUMB_LE: Arch = arm("ARM Thumb (LE)", "ARM/ARM7_le.slaspec", false, THUMB);

/// ARM Thumb, big-endian.
pub const THUMB_BE: Arch = arm("ARM Thumb (BE)", "ARM/ARM7_be.slaspec", true, THUMB);

/// 64-bit ARM (`AArch64`), little-endian.
pub const AARCH64_LE: Arch = plain("AArch64 (LE)", "AARCH64/AARCH64.slaspec", false, 64);

/// 64-bit ARM (`AArch64`), big-endian.
pub const AARCH64_BE: Arch = plain("AArch64 (BE)", "AARCH64/AARCH64BE.slaspec", true, 64);

/// 32-bit MIPS, big-endian.
pub const MIPS32_BE: Arch = plain("MIPS32 (BE)", "MIPS/mips32be.slaspec", true, 32);

/// 32-bit MIPS, little-endian.
pub const MIPS32_LE: Arch = plain("MIPS32 (LE)", "MIPS/mips32le.slaspec", false, 32);

/// 64-bit MIPS, big-endian.
pub const MIPS64_BE: Arch = plain("MIPS64 (BE)", "MIPS/mips64be.slaspec", true, 64);

/// 64-bit MIPS, little-endian.
pub const MIPS64_LE: Arch = plain("MIPS64 (LE)", "MIPS/mips64le.slaspec", false, 64);

/// 32-bit PowerPC, big-endian.
pub const PPC32_BE: Arch = plain("PowerPC32 (BE)", "PowerPC/ppc_32_be.slaspec", true, 32);

/// 32-bit PowerPC, little-endian.
pub const PPC32_LE: Arch = plain("PowerPC32 (LE)", "PowerPC/ppc_32_le.slaspec", false, 32);

/// 64-bit PowerPC, big-endian.
pub const PPC64_BE: Arch = plain("PowerPC64 (BE)", "PowerPC/ppc_64_be.slaspec", true, 64);

/// 64-bit PowerPC, little-endian.
pub const PPC64_LE: Arch = plain("PowerPC64 (LE)", "PowerPC/ppc_64_le.slaspec", false, 64);

/// 32-bit RISC-V, little-endian.
pub const RISCV32: Arch = plain("RISC-V 32", "RISCV/riscv.ilp32d.slaspec", false, 32);

/// 64-bit RISC-V, little-endian.
pub const RISCV64: Arch = Arch {
    name: "RISC-V 64",
    root: "RISCV/riscv.lp64d.slaspec",
    big_endian: false,
    bits: 64,
    context: RV64,
    interworking: false,
};

/// 64-bit SPARC, big-endian.
pub const SPARC_V9: Arch = plain("SPARC V9", "Sparc/SparcV9_64.slaspec", true, 64);

/// `SuperH` SH-2, big-endian.
pub const SUPERH: Arch = plain("SuperH SH-2", "SuperH/sh-2.slaspec", true, 32);

/// 8-bit AVR, little-endian.
pub const AVR8: Arch = plain("AVR8", "Atmel/avr8.slaspec", false, 16);

/// 68000, big-endian.
pub const M68000: Arch = plain("Motorola 68000", "68000/68020.slaspec", true, 32);

/// MSP430, little-endian.
pub const MSP430: Arch = plain("TI MSP430", "TI_MSP430/TI_MSP430.slaspec", false, 16);

/// 64-bit `LoongArch`, little-endian.
pub const LOONGARCH64: Arch = plain(
    "LoongArch64",
    "Loongarch/loongarch64_f64.slaspec",
    false,
    64,
);

/// Xtensa, little-endian.
pub const XTENSA_LE: Arch = plain("Xtensa (LE)", "Xtensa/xtensa_le.slaspec", false, 32);

/// eBPF, little-endian.
pub const EBPF: Arch = plain("eBPF", "eBPF/eBPF_le.slaspec", false, 64);

/// x86-64 long-mode decode context.
const X64: &[(&str, u32)] = &[
    ("longMode", 1),
    ("bit64", 1),
    ("addrsize", 2),
    ("opsize", 1),
    ("protectedMode", 1),
];

/// 32-bit x86 protected-mode decode context.
const X86_32: &[(&str, u32)] = &[("addrsize", 1), ("opsize", 1), ("protectedMode", 1)];

/// Intel x86-64 (long mode), little-endian.
pub const X86_64: Arch = Arch {
    name: "x86-64",
    root: "x86/x86-64.slaspec",
    big_endian: false,
    bits: 64,
    context: X64,
    interworking: false,
};

/// Intel x86 (protected mode), little-endian.
pub const X86: Arch = Arch {
    name: "x86",
    root: "x86/x86.slaspec",
    big_endian: false,
    bits: 32,
    context: X86_32,
    interworking: false,
};

/// Builds an ARM-family architecture entry.
///
/// # Arguments
///
/// * `name` - Display name.
/// * `root` - Root specification path.
/// * `big_endian` - Whether the target is big-endian.
/// * `context` - Context fields to set.
///
/// # Returns
///
/// The architecture entry with 32-bit interworking enabled.
const fn arm(
    name: &'static str,
    root: &'static str,
    big_endian: bool,
    context: &'static [(&'static str, u32)],
) -> Arch {
    Arch { name, root, big_endian, bits: 32, context, interworking: true }
}

/// Builds a non-interworking architecture entry.
///
/// # Arguments
///
/// * `name` - Display name.
/// * `root` - Root specification path.
/// * `big_endian` - Whether the target is big-endian.
/// * `bits` - Address width.
///
/// # Returns
///
/// The architecture entry with no extra context.
const fn plain(name: &'static str, root: &'static str, big_endian: bool, bits: u32) -> Arch {
    Arch { name, root, big_endian, bits, context: NONE, interworking: false }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies ARM Thumb carries its mode context and interworking.
    #[test]
    fn test_thumb_entry() {
        let thumb = std::hint::black_box(THUMB_LE);
        assert_eq!(thumb.context, &[("TMode", 1)]);
        assert!(thumb.interworking);
    }

    /// Verifies plain entries have no context and no interworking.
    #[test]
    fn test_plain_entry() {
        let mips = std::hint::black_box(MIPS32_BE);
        assert!(mips.context.is_empty() && !mips.interworking);
        assert_eq!(mips.bits, 32);
    }
}
