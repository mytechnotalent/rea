![image](Reverse-Engineering-Assistant.png)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant (REA) v6.0.2

**The reverse-engineering framework you can run anywhere, with nothing installed, and actually trust.**

```bash
rea suspect
```

Writes `re-suspect.md`, `re-suspect.pdf`, and `re-suspect.json` next to the target. That's the whole workflow.

**Targets in. Answers out.** One command. Hybrid autonomous multi-agent reasoning and a deterministic engine. Built in 100% pure Rust — run anywhere, install nothing, and every claim verified.

<br>

# New in v6.0.2

REA v6.0.2 makes the CI gate pass on clippy 1.99:

- **Empty-collection assertions.** Test assertions now use `assert_eq!(… .len(), 0)` / `assert_ne!(… .len(), 0)` instead of `assert!(… .is_empty())`, which the newer `clippy::assert_is_empty` pedantic lint rejects. No behavior change.

# Features

- **One command.** `rea <file>` produces a Markdown report, an executive typeset PDF report, and a machine-readable JSON report beside the target.
- **Any file.** Format is detected from contents, not the file extension.
- **Autonomous AI Security Assessment.** When configured with an API key, REA runs an autonomous, 90-turn tool-calling analyst loop with native function calling. The model is the report's author and inspects the target itself (decompile, emulate, decrypt, audit, rules, regions, 8051, and more), writing an Executive Overview, Target Summary, Architecture Analysis, Recovered Secrets, Indicators of Compromise, and Recommendations. A grounding validator flags any quotation not backed by engine evidence.
- **Deterministic Offline Fallback.** With no AI API key, REA runs completely offline, generating comprehensive deterministic technical reports with zero network requests.
- **Firmware & Filesystems.** U-Boot legacy images (`uImage`), SquashFS 4.0 root filesystems, embedded Linux partitions, and raw microcontroller firmware.
- **Formats.** PE (EXE/DLL/SYS), ELF, Mach-O, universal binaries, firmware bundles, and raw ARM Cortex-M images.
- **Static Analysis.** Cryptographic hashes (MD5, SHA-1, SHA-256), import hashes (imphash), Shannon entropy, sections, memory layouts, imports, exports, and mitigations (ASLR, DEP/NX, CFG, PIE, RELRO, stack canaries).
- **Disassembly.** Recursive-descent function and call-graph recovery with inline symbol and string resolution and per-instruction semantics.
- **P-Code Emulation.** Sandboxed software emulator executes target code without native host execution, unmasking runtime strings, keys, and decrypted payloads.
- **Protectors & Packing.** High-entropy mapping, abnormal memory layouts, W^X violations, and signature matching for commercial protectors.
- **Indicators.** Structured extraction of URLs, IPs, emails, registry keys, file paths, commands, and user agents.
- **Artifact Formats.** WebAssembly, SQLite, ELF core dumps, EVM bytecode, Apple property lists, Flutter APKs, PCAP/HAR captures, JavaScript source maps, and MS-DOS MZ executables.
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
| WebAssembly modules (`.wasm`) | Full | Sections, imports, exports, function counts |
| SQLite databases (`.db`/`.sqlite`) | Full | Header, tables, columns, views, triggers, sampled rows |
| ELF core dumps | Full | Signal, registers, loadable mappings, mapped files |
| EVM bytecode (raw or hex) | Full | Selectors, jump targets, notable opcodes, mutability |
| Apple property lists & bundles | Full | Binary/XML plist, bundle identifier and version |
| Flutter APKs & zip archives | Full | Per-ABI native libraries, digests, Flutter assets |
| PCAP and HAR captures | Full | Packets, HTTP requests, hosts, HAR entries |
| JavaScript source maps | Full | Sources, mapping statistics, `sourceMappingURL` |
| MS-DOS `MZ` executables | Full | Header, relocations, 16-bit entry disassembly |
| Java class files (`.class`) | Full | Constant pool, this/super, methods, fields |
| Android `resources.arsc` | Full | Packages, string pool, configuration locales |
| DWARF debug info (ELF) | Full | Compile units, directories, function symbols |
| Windows PDB (`.pdb`) | Full | MSF container, info-stream version, age, GUID |
| YARA-subset rules | Matching | Literal/wide/nocase/hex patterns with boolean conditions |
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
rea decompile <target> [symbol|0xaddr]  # recovered pseudo-C
rea strings <target>                    # extracted strings
rea decrypt <target>                    # strings hidden by single-byte ciphers
rea crypto <target>                     # cryptographic constants
rea rules <target>                      # capability rules (ATT&CK/MBC)
rea audit <target> [--sarif]            # vulnerability sinks + SARIF
rea unpack <target> [0xaddr]            # emulation-driven unpacking
rea emulate <target> [0xaddr]           # sandboxed emulation
rea dotnet <assembly>                   # .NET CIL disassembly
rea wasm <module.wasm>                  # WebAssembly sections and links
rea sqlite <database>                   # SQLite schema and rows
rea coredump <core>                     # ELF core signal, registers, mappings
rea evm <bytecode>                      # EVM selectors and mutability
rea plist <Info.plist>                  # property list and bundle metadata
rea sourcemap <app.js.map>              # source map sources and statistics
rea pcap <capture.pcap>                 # packets, HTTP requests, hosts
rea flutter <app.apk>                   # per-ABI native libraries and assets
rea dos <program.exe>                   # MZ header and 16-bit entry code
rea java <App.class>                    # Java class pool, methods, fields
rea arsc <resources.arsc>               # Android packages, strings, locales
rea dwarf <app.elf>                     # DWARF compile units and functions
rea pdb <app.pdb>                       # Windows PDB version, age, GUID
rea yara <target> <rules.yar>           # YARA-subset signature matching
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

The server exposes 31 tools: `open_binary`, `binary_session`, `list_functions`, `decompile_function`, `find_xrefs`, `search_strings`, `analyze_firmware`, `analyze_javascript`, `generate_dossier`, `inspect_go`, `inspect_dotnet`, `inspect_android`, `run_rules`, `audit`, `inspect_crypto`, `decrypt_strings`, `emulate`, `unpack`, `inspect_wasm`, `inspect_sqlite`, `inspect_coredump`, `inspect_evm`, `inspect_plist`, `inspect_sourcemap`, `inspect_capture`, `inspect_flutter`, `inspect_dos`, `inspect_dwarf`, `inspect_java`, `inspect_arsc`, `inspect_pdb`.

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
