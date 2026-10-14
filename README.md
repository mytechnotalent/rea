![image](Reverse-Engineering-Assistant.png)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant (REA) v5.3.0

Autonomous Reverse Engineering Framework

```bash
rea suspect
```

Writes `re-suspect.md`, `re-suspect.pdf`, and `re-suspect.json` next to the target. That's the whole workflow.

**Targets in. Answers out.** One command. Hybrid autonomous multi-agent reasoning and deterministic engine. Built in 100% pure Rust.

<br>

# New in v5.3.0

REA v5.3.0 adds firmware-container awareness and sharpens analysis quality:

- **Firmware Container Decoding.** Detects multi-record flash-container headers and reports the entry point, declared size, descriptor records, and the deduced runtime load base in a new report section, so firmware is analyzed at its true address instead of a flat offset map.
- **Correct Raw-Image Base.** Raw firmware handed to the provider (Binary Ninja) is wrapped at its detected load base, so addresses, cross-references, and symbols resolve at the real runtime address. Override with `REA_LOAD_BASE`.
- **Faster Analyst Loop.** The autonomous analyst skips duplicate tool calls and stops after repeated no-progress turns, cutting wasted turns and cost.
- **Stronger Grounding.** The grounding validator now checks quotations against the full extracted-string set, not just the dossier, reducing false positives.
- **Accurate Instructions.** Fixed register corruption in immediate formatting (`r12` no longer becomes `r0xc`) and corrected branch classification (`beq` is a conditional branch with its condition, not "unconditional"); block stores/loads (`stm`/`ldm`) are described as stack-frame operations.

# New in v5.2.1

REA v5.2.1 is a reliability release that fixes a decompiler hang and hardens the autonomous analyst:

- **Decompiler Hang Fixed.** Copy propagation could map a register to itself (`mov r0, r0`), and the dominator and copy-propagation fixpoints could oscillate on cyclic control flow, hanging `rea decompile` and the one-shot on some ARM Cortex-M targets. Self/cyclic mappings are now rejected and both fixpoints are iteration-bounded.
- **Bounded Analyst Loop.** The agent loop now enforces a wall-clock budget (default 420s, `REA_AGENT_BUDGET_SECS`), forces a final report when the budget expires (`tool_choice: none`), caps each observation, and applies a per-request network timeout so a one-shot always terminates with a report.
- **Live Progress.** The analyst prints per-turn and per-tool progress so long investigations are visible.

# New in v5.2.0

REA v5.2.0 rebuilds the autonomous analyst into a true tool-calling author, so the AI actually investigates the target instead of summarizing a static dossier:

- **Native Tool Calling.** The analyst uses OpenRouter function/tool calling (JSON schemas) instead of a brittle text protocol, eliminating mis-parsed actions and hallucinated free-writing.
- **90-Turn Investigation.** The agent loop runs up to 90 tool turns (up from 4), so it decompiles, emulates, and reads its way to an answer.
- **The LLM Authors the Report.** The model is the report's author; deterministic engine sections feed it as evidence, and the authored assessment leads the report.
- **Grounding Validator.** Every quoted string in the final report is checked against the evidence dossier and all tool observations; unverified quotations are flagged rather than shipped as fact.
- **Capable Default Model.** Defaults to `deepseek/deepseek-v4.1-flash`; override with `OPENROUTER_MODEL`.
- **Stronger Engine Tools.** The loop gains entropy `regions` segmentation and 8051 (`MCS-51`) disassembly. Every tool is pure-Rust and works one-shot with no external tools, servers, or licenses required.

Retained from v5.1.0: DWARF/PDB symbols, YARA-subset matching, JVM `.class`, Android `resources.arsc`, source-map VLQ, WASM code bodies, crypto expansion, artifact report sections, and the 31-tool MCP server.

<br>

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
rea decompile <target> [symbol|0xaddr]   # recovered pseudo-C
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
