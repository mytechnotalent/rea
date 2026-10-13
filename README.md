![image](Reverse-Engineering-Assistant.png)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant (REA) v4.0.0

Autonomous Reverse Engineering Framework

```bash
rea suspect
```

Writes `re-suspect.md`, `re-suspect.pdf`, and `re-suspect.json` next to the target. That's the whole workflow.

**Targets in. Answers out.** One command. Hybrid autonomous multi-agent reasoning and deterministic engine. Built in 100% pure Rust.

<br>

# New in v4.0.0

REA v4.0.0 is a native pure-Rust reverse engineering engine with a first-class one-shot workflow and a Model Context Protocol (MCP) server for AI coding agents. Highlights:

- **Pure-Rust Native Decompiler.** Recovered pseudo-C for every function from a p-code intermediate representation and control-flow graph, with structured control flow (loops, if/else), constant folding, stack-slot naming, cross-block copy propagation, and dead-store elimination. The original assembly is cited per line.
- **Cross-Architecture.** x86/x86-64, ARM Cortex-M (Thumb-2) and Cortex-A (AArch64), MIPS, RISC-V, PowerPC, SPARC, SuperH, AVR, MSP430, 68000, Xtensa, LoongArch — no Ghidra, IDA, Hopper, or Java required.
- **Emulation-Driven Unpacking.** API-stubbed sandboxed emulation decrypts protected regions (OEP detection, IAT rebuild, region dump, multi-stage convergence) on x86, ARM, MIPS, and RISC-V.
- **Capability Rules (capa-class).** An embedded rule pack with MITRE ATT&CK and MBC mapping over imports, strings, and instructions.
- **Vulnerability Audit + SARIF.** Sink-based detection with CWE identifiers, severity, and SARIF 2.1.0 export (`rea audit --sarif`).
- **Cipher & Crypto Recovery.** Recovers strings hidden by single-byte ciphers and detects cryptographic constants (AES, MD5, SHA, CRC32, TEA, ChaCha, base64).
- **Pure-Rust Runtime Forensics.** Go `pclntab`/`modinfo`, .NET ECMA-335 CIL (with token→name resolution), and Android APK/DEX inspection.
- **MCP Server (18 tools).** `open_binary`, `binary_session`, `list_functions`, `decompile_function`, `find_xrefs`, `search_strings`, `analyze_firmware`, `analyze_javascript`, `generate_dossier`, `inspect_go`, `inspect_dotnet`, `inspect_android`, `run_rules`, `audit`, `inspect_crypto`, `decrypt_strings`, `emulate`, `unpack`.
- **Turnkey Agent Setup.** `rea mcp --setup` writes the OpenCode/agent configuration for the current binary; `rea mcp --config` prints ready-to-paste snippets for Claude Code, OpenCode, Claude Desktop, Cursor, and Windsurf.
- **Machine-Readable Output.** Every one-shot writes `re-<name>.md`, `re-<name>.pdf`, and `re-<name>.json`.

<br>

# Features

- **One command.** `rea <file>` produces a Markdown report, an executive typeset PDF report, and a machine-readable JSON report beside the target.
- **Any file.** Format is detected from contents, not the file extension.
- **Autonomous AI Security Assessment.** When configured with an API key, REA runs an autonomous tool-calling analyst loop that inspects the target (decompile, emulate, decrypt, audit, rules) and writes an Executive Overview, Architecture Analysis, Evidence-Validated Vulnerabilities, and Actionable Recommendations.
- **Deterministic Offline Fallback.** With no AI API key, REA runs completely offline, generating comprehensive deterministic technical reports with zero network requests.
- **Firmware & Filesystems.** U-Boot legacy images (`uImage`), SquashFS 4.0 root filesystems, embedded Linux partitions, and raw microcontroller firmware.
- **Formats.** PE (EXE/DLL/SYS), ELF, Mach-O, universal binaries, firmware bundles, and raw ARM Cortex-M images.
- **Static Analysis.** Cryptographic hashes (MD5, SHA-1, SHA-256), import hashes (imphash), Shannon entropy, sections, memory layouts, imports, exports, and mitigations (ASLR, DEP/NX, CFG, PIE, RELRO, stack canaries).
- **Disassembly.** Recursive-descent function and call-graph recovery with inline symbol and string resolution and per-instruction semantics.
- **P-Code Emulation.** Sandboxed software emulator executes target code without native host execution, unmasking runtime strings, keys, and decrypted payloads.
- **Protectors & Packing.** High-entropy mapping, abnormal memory layouts, W^X violations, and signature matching for commercial protectors.
- **Indicators.** Structured extraction of URLs, IPs, emails, registry keys, file paths, commands, and user agents.
- **Strict Safety.** `#![forbid(unsafe_code)]` enforced across the entire codebase. Malformed or hostile inputs can never compromise the host.

<br>

# Supported Targets

| Format | Parsing | Analysis & Triage |
| --- | --- | --- |
| PE (EXE, DLL, SYS) | Full | x86, x86-64, imports, mitigations, imphash |
| ELF (Linux, BSD, Android) | Full | x86/x86-64 + every SLEIGH architecture |
| Mach-O and universal | Full | x86/x86-64 + every SLEIGH architecture |
| Go Executables (ELF/PE/Mach-O) | Full | `pclntab` functions, `modinfo` dependencies, toolchain version |
| .NET CIL Assemblies (EXE/DLL) | Full | ECMA-335 CLI metadata, `BSJB` streams, CIL disassembly, token→name |
| Android APK & DEX Packages | In-memory | `AndroidManifest.xml` binary XML, permissions, `classes.dex` |
| Electron Applications (`.asar`) | In-memory | Archive unpacking, IPC channel extraction, hardcoded secrets |
| U-Boot images (`uImage`) | Full | Kernel identification, compression, rootfs offsets |
| SquashFS 4.0 Filesystems | In-memory | Unpacking, credential discovery, sub-target triage |
| Raw ARM Cortex-M firmware | Vector table, flash base | ARM Thumb disassembly and handler mapping |
| ARM Cortex-A (Linux, Android, iOS) | Full (ELF/Mach-O) | ARMv7-A and ARMv8-A/AArch64 disassembly, call graphs |
| Any raw or unknown file | Identity, hashes, entropy | String extraction, indicators, entropy mapping |

Non-x86 disassembly and emulation are powered by embedded SLEIGH processor specifications compiled directly by a pure-Rust engine. No Ghidra, Java, Python, or external command-line tools are required.

<br>

# One-Shot Usage

The one-shot is the primary workflow: point REA at a target and read the report.

```bash
rea <path-to-target>
```

REA writes three files beside the target:

- `re-<name>.md` — GitHub-flavored Markdown report.
- `re-<name>.pdf` — executive typeset PDF report.
- `re-<name>.json` — machine-readable summary (format, architecture, hashes, mitigations, imports, capabilities, rules, vulnerabilities, crypto).

**Windows x64 (PowerShell)**
```powershell
.\rea.exe .\sample.exe
```

**macOS Apple Silicon**
```bash
./rea ./sample
```

**Linux x64**
```bash
./rea ./sample
```

Focused subcommands are also available:

```bash
rea decompile <target> [symbol|0xaddr]   # recovered pseudo-C
rea disasm                              # (see decompile) assembly walkthrough
rea strings <target>                    # extracted strings
rea decrypt <target>                    # strings hidden by single-byte ciphers
rea crypto <target>                     # cryptographic constants
rea rules <target>                      # capability rules (ATT&CK/MBC)
rea audit <target> [--sarif]            # vulnerability sinks + SARIF
rea unpack <target> [0xaddr]            # emulation-driven unpacking
rea emulate <target> [0xaddr]           # sandboxed emulation
rea dotnet <assembly>                   # .NET CIL disassembly
rea analyze <target>                    # alias for the one-shot
```

<br>

# MCP Server Usage (Second Option)

The MCP server exposes the same engine as tools so AI coding agents can drive the analysis interactively. It runs over stdio (JSON-RPC 2.0) via:

```bash
rea serve
```

One-step setup writes the agent configuration for the current binary:

```bash
rea mcp --setup    # writes mcp_config.json and opencode.jsonc for this binary
rea mcp --config   # prints ready-to-paste snippets for every supported agent
```

### Claude Code

```bash
claude mcp add rea rea serve
```

Then ask, for example:

```text
Open ./sample.exe with rea, decompile the entry point, run the capability rules,
and explain what it does.
```

### OpenCode

Add REA to `opencode.json` (project) or `~/.config/opencode/opencode.jsonc` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rea": {
      "type": "local",
      "command": ["/absolute/path/to/rea", "serve"],
      "enabled": true
    }
  }
}
```

`rea mcp --setup` writes this for you with the correct absolute path. Restart OpenCode, then ask:

```text
Use rea to open ./sample.exe, list the key functions, and decompile the entry point.
```

### Per-Platform MCP Setup

**Windows x64 (PowerShell)**
```powershell
.\rea.exe mcp --setup
# Claude Code:
claude mcp add rea "$PWD\rea.exe" serve
```

**macOS Apple Silicon**
```bash
./rea mcp --setup
# Claude Code:
claude mcp add rea "$PWD/rea" serve
```

**Linux x64**
```bash
./rea mcp --setup
# Claude Code:
claude mcp add rea "$PWD/rea" serve
```

The server exposes 18 tools: `open_binary`, `binary_session`, `list_functions`, `decompile_function`, `find_xrefs`, `search_strings`, `analyze_firmware`, `analyze_javascript`, `generate_dossier`, `inspect_go`, `inspect_dotnet`, `inspect_android`, `run_rules`, `audit`, `inspect_crypto`, `decrypt_strings`, `emulate`, `unpack`.

<br>

# AI Analyst Configuration

To enable the autonomous tool-calling analyst loop, provide an [OpenRouter](https://openrouter.ai) API key. REA locates the key from environment variables or a `.env` file in the current or ancestor directories:

```bash
# In .env or shell environment:
OPENROUTER_API_KEY="sk-or-v1-..."

# Optional model selection (defaults to high-throughput free tier):
OPENROUTER_MODEL="nvidia/nemotron-3.5-lightning:free"
```

If no key is configured, REA operates in purely deterministic offline mode.

<br>

# Build from Source

Install the [stable Rust toolchain](https://rustup.rs), clone the repository, and build on your operating system:

```bash
git clone https://github.com/mytechnotalent/rea.git
cd rea
```

**Windows x64 (PowerShell)**
```powershell
rustup target add x86_64-pc-windows-msvc
cargo build --release -p rea --target x86_64-pc-windows-msvc
```

Output: `target\x86_64-pc-windows-msvc\release\rea.exe`

**macOS Apple Silicon**
```bash
rustup target add aarch64-apple-darwin
cargo build --release -p rea --target aarch64-apple-darwin
```

Output: `target/aarch64-apple-darwin/release/rea`

**Linux x64**
```bash
rustup target add x86_64-unknown-linux-gnu
cargo build --release -p rea --target x86_64-unknown-linux-gnu
```

Output: `target/x86_64-unknown-linux-gnu/release/rea`

Install a native build on your `PATH` with:

```bash
cargo install --path crates/rea
```

Prebuilt binaries for macOS Apple Silicon (`aarch64-apple-darwin`) are located in [`releases/`](releases). To build for Windows or Linux, follow the commands above.

<br>

# Quality & Verification

```bash
cargo xtask check
```

Enforces strict zero-violation coding standards (`CONVENTIONS.md`), runs `cargo fmt --check`, denies all Clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings`), and executes the complete workspace test suite.

<br>

# License

[MIT License](LICENSE)
