//! Native Binary Analysis Engine
//!
//! **File:** `lib.rs`
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

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

/// Whole-target analysis orchestration.
pub mod analysis;

/// Import-based capability classification.
pub mod capabilities;

/// File format identification by signature.
pub mod detect;

/// Runtime string discovery from emulated memory.
pub mod discover;

/// Emulated execution over lifted p-code.
pub mod emulate;

/// Recursive-descent x86 and x86-64 disassembly.
pub mod disasm;

/// Shannon entropy measurement.
pub mod entropy;

/// Engine error type.
pub mod error;

/// Cryptographic file fingerprints.
pub mod hashes;

/// Function ranking by analyst relevance.
pub mod insights;

/// Indicator extraction from strings.
pub mod iocs;

/// Formatted instruction listings.
pub mod listing;

/// Executable format dispatch.
mod loader;

/// Mach-O loader.
mod macho;

/// Multi-architecture disassembly via SLEIGH.
pub mod multiarch;

/// Behavioral narrative synthesis from recovered code.
pub mod narrative;

/// Format-neutral binary model.
pub mod model;

/// ELF loader.
mod elf;

/// Raw Cortex-M firmware loader.
mod firmware;

/// U-Boot legacy image loader.
mod uimage;

/// Packing and obfuscation assessment.
pub mod packer;

/// PE loader.
mod pe;

/// Packed relative relocation decoding.
mod relr;

/// SquashFS rootfs extraction and binary triage.
pub mod squashfs;

/// Printable string extraction.
pub mod strings;

/// Calendar formatting of timestamps.
mod time;

/// Small numeric conversion helpers.
mod numeric;

/// Firmware security forensics and vulnerability catalog.
pub mod forensics;

/// Electron ASAR archive parsing and JavaScript triage.
pub mod asar;

/// Semantic instruction analysis, function summarization, and data disambiguation.
pub mod semantics;

/// Go runtime and build metadata extraction.
pub mod go;

/// .NET ECMA-335 CLI managed assembly metadata extraction.
pub mod dotnet;

/// Android DEX and APK package forensics.
pub mod dex;

pub use analysis::{analyze, analyze_path, Analysis};
pub use asar::{AsarIpcChannel, AsarSecret, AsarTriage, AsarWarning};
pub use detect::Format;
pub use dex::{format_apk_summary, parse_apk_metadata, parse_dex_header, ApkMetadata, DexMetadata};
pub use discover::Finding;
pub use dotnet::{format_dotnet_summary, parse_dotnet_metadata, DotNetAssemblyRef, DotNetMetadata};
pub use emulate::{Emulation, Stop};
pub use error::EngineError;
pub use forensics::{
    all_exploit_chains, all_vulnerabilities, format_findings_json, ExploitChain, Vulnerability,
};
pub use go::{format_go_summary, parse_go_metadata, GoMetadata, GoModule};
pub use squashfs::{FirmwareTriage, SecurityArtifact, SquashfsMeta};
