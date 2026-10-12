# reverse-engineering

Autonomous reverse engineering framework, pure-Rust multi-architecture disassembler, and Model Context Protocol (MCP) server.

[![Crates.io](https://img.shields.io/crates/v/reverse-engineering.svg)](https://crates.io/crates/reverse-engineering)
[![Documentation](https://docs.rs/reverse-engineering/badge.svg)](https://docs.rs/reverse-engineering)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`reverse-engineering` is a 100% pure-Rust, zero-dependency reverse engineering framework. It provides embedded SLEIGH-driven multi-architecture disassembly, sandboxed p-code software emulation, embedded firmware and SquashFS 4.0 rootfs carving, Electron `.asar` archive triage, and a native Model Context Protocol (MCP) server for integration with AI coding agents (Claude Code, Cursor, Windsurf, Gemini CLI, and OpenCode).

The crate enforces `#![forbid(unsafe_code)]` across its entire codebase, guaranteeing that hostile malware binaries or corrupted firmware images cannot cause memory corruption or compromise the host analysis system.

## Quick Start

Add `reverse-engineering` to your `Cargo.toml`:

```toml
[dependencies]
reverse-engineering = "4.0.0"
```

### Programmatic Binary Analysis

```rust
use std::path::Path;
use reverse_engineering::{inspect, Format};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let analysis = inspect(Path::new("firmware.bin"))?;

    println!("Target:   {}", analysis.name);
    println!("Format:   {}", analysis.format.label());
    println!("Entropy:  {:.2} bits/byte", analysis.entropy);
    println!("SHA-256:  {}", analysis.hashes.sha256);

    if let Some(binary) = &analysis.binary {
        println!("Arch:     {} ({} bits)", binary.arch_name, binary.bits);
        println!("Entry:    0x{:x}", binary.entry);
    }

    if let Some(fw) = &analysis.firmware {
        println!("RootFS Inodes:    {}", fw.squashfs.total_inodes);
        println!("Security Artifacts: {}", fw.artifacts.len());
    }

    Ok(())
}
```

### Running the Native MCP Server

Integrate directly with AI coding assistants over standard I/O:

```rust
use reverse_engineering::serve_mcp;

fn main() {
    serve_mcp();
}
```

## Architecture & Engine Specifications

```text
+-----------------------------------------------------------------+
|                    REVERSE-ENGINEERING ENGINE                   |
+-----------------------------------------------------------------+
|  [Formats]       PE, ELF, Mach-O, uImage, SquashFS, ASAR, Cortex-M|
|  [Disassembly]   x86, x86-64, ARM, AArch64, MIPS, RISC-V, PowerPC |
|  [Emulation]     Sandboxed P-Code interpreter & runtime strings |
|  [Forensics]     Firmware rootfs, shadow hashes, private keys   |
|  [Protocol]      Native Model Context Protocol (MCP) stdio JSON-RPC|
+-----------------------------------------------------------------+
```

### Supported Target Formats

| Format | Parsing Level | Capabilities |
| :--- | :--- | :--- |
| **PE (EXE, DLL, SYS)** | Full Native | Headers, sections, imports, exports, mitigations, imphash |
| **ELF (Linux, BSD)** | Full Native | Sections, segments, dynamic symbols, PLT/GOT, mitigations |
| **Mach-O & Universal** | Full Native | Fat architectures, load commands, code signatures |
| **SquashFS 4.0** | In-Memory | Pure-Rust decompression, rootfs inode tree, credential hunting |
| **U-Boot `uImage`** | Header & Body | Kernel identification, header checksums, payload carving |
| **Electron `.asar`** | In-Memory | Pickle JSON directory tree, JavaScript secret & IPC scanning |
| **Raw ARM Cortex-M** | Vector Table | Flash base identification, Thumb reset vectors |

### Multi-Architecture Coverage (SLEIGH)

Disassembly and p-code lifting are powered by embedded SLEIGH processor specifications compiled directly into pure Rust:
- **x86 / x86-64**: Native recursive-descent call graph recovery and iced-x86 decoding.
- **ARM**: ARMv7-A, ARMv8-A (AArch64), ARMv7-M / ARMv8-M (Thumb-2).
- **MIPS**: MIPS32, MIPS64, microMIPS (both big and little endian).
- **PowerPC**: PPC32, PPC64 (PPC and VLE).
- **RISC-V**: RV32I, RV32G, RV64I, RV64G with compressed instructions.
- **SPARC**: SPARC V8, SPARC V9.
- **SuperH**: SH-1, SH-2, SH-2A, SH-4.
- **Microcontrollers**: AVR (ATmega), TI MSP430, Motorola 68000, Tensilica Xtensa.

## Model Context Protocol (MCP) Integration

The native MCP server runs over standard I/O (`stdio`) using JSON-RPC 2.0 and supports standard MCP protocol versions (e.g. `2024-11-05`).

### Tool Catalog

| Tool Name | Description | Key Arguments |
| :--- | :--- | :--- |
| `open_binary` | Parses target binary or archive into active session | `path: String` |
| `binary_session` | Returns active target architecture, format, and mitigations | None |
| `list_functions` | Enumerates recovered functions and entry points | None |
| `decompile_function` | Generates recovered pseudo-C for a function | `symbol: String` |
| `find_xrefs` | Queries cross-references to a symbol or memory address | `target: String` |
| `search_strings` | Searches extracted printable strings and indicators | `query: String` |
| `analyze_firmware` | Unpacks and triages SquashFS rootfs partitions and credentials | None |
| `analyze_javascript` | Unpacks Electron ASAR archives, extracting IPC channels and secrets | None |
| `generate_dossier` | Emits complete unified reports and security dossier beside target | `path: String` |

### Configuration for AI Assistants

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "rea": {
      "command": "rea",
      "args": ["serve"]
    }
  }
}
```

#### Cursor (`~/.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "rea": {
      "command": "rea",
      "args": ["serve"]
    }
  }
}
```

#### Windsurf (`~/.codeium/windsurf/mcp_config.json`)
```json
{
  "mcpServers": {
    "rea": {
      "command": "rea",
      "args": ["serve"]
    }
  }
}
```

## Security & Safety Guarantees

1. **`#![forbid(unsafe_code)]`**: The entire codebase strictly forbids unsafe code blocks.
2. **Zero External C Dependencies**: No external C libraries, no Java runtime, and no Python bindings required.
3. **Sandboxed Emulation**: Target binaries are executed only within a sandboxed virtual software p-code interpreter; they are never executed natively on the host CPU.

## License

MIT &mdash; Copyright (c) 2026 Kevin Thomas.
