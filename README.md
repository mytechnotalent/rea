![image](Reverse-Engineering-Assistant.png)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant (REA) v3.2.0

Autonomous Reverse Engineering Framework

```bash
rea suspect
```

Writes `re-suspect.md` and `re-suspect.pdf` next to the target. That's the whole workflow.

**Targets in. Answers out.** One command. Hybrid autonomous multi-agent reasoning and deterministic engine. Built in 100% pure Rust.

<br>

# New in v3.2.0

REA v3.2.0 delivers format-aware behavioral modeling, elimination of cross-platform narrative hallucinations, and specialized firmware and desktop binary triage:

- **Format-Aware Narrative Engine**: Dedicated architectural dispatch across Windows PE (`.exe`), Linux/POSIX ELF, Darwin Mach-O, U-Boot legacy images (`uImage`), and raw embedded microcontroller firmware (`CortexM`).
- **Zero Cross-Platform Hallucinations**: Eliminates bare-metal microcontroller vector table, Flash-to-SRAM, and clock divider narratives on desktop PE/ELF binaries and embedded Linux router images.
- **U-Boot & Kernel Lifecycle Triage**: Specialized bootloader decompression, `kernel_entry` register handover, virtual memory management, embedded SquashFS rootfs mounting, and U-Boot CRC32/TFTP recovery patching guidance.
- **Desktop & OS Binary Modeling**: Accurate Windows PE and Linux ELF operating system loader mapping, runtime startup (`_start`, CRT environment, SEH/TLS callbacks, stdio), and `.rdata`/`.rodata` constant layout.
- **Production-Validated Targets**: Validated against real-world protected Windows PE crackmes, MIPS OpenWrt gateway firmware, and ARM Cortex-M bare-metal binaries.

<br>

# Features

- **One command.** `rea <file>` produces a GitHub-flavored Markdown report and an executive typeset PDF report beside the target.
- **Any file.** Format is detected from contents, not the file extension.
- **Autonomous AI Security Assessment.** When configured with an API key, REA runs an autonomous propose-review-adjudicate loop to generate an Executive Overview, Architecture Analysis, Evidence-Validated Vulnerabilities, and Actionable Recommendations.
- **Deterministic Offline Fallback.** If no AI API key is configured, REA runs completely offline, generating comprehensive deterministic technical reports with zero network requests.
- **Firmware & Filesystems.** U-Boot legacy images (`uImage`), SquashFS 4.0 root filesystems, embedded Linux partitions, and raw microcontroller firmware.
- **Formats.** PE (EXE/DLL/SYS), ELF, Mach-O, universal binaries, firmware bundles, and raw ARM Cortex-M images.
- **Architectures.** Native x86 and x86-64, plus every SLEIGH family: ARM/Thumb, AArch64, MIPS, PowerPC, RISC-V, SPARC, SuperH, AVR, 68000, MSP430, Xtensa, LoongArch — both endiannesses.
- **Static Analysis.** Cryptographic hashes (MD5, SHA-1, SHA-256), import hashes (imphash), Shannon entropy, sections, memory layouts, imports, exports, and mitigations (ASLR, DEP/NX, CFG, PIE, RELRO, stack canaries).
- **Disassembly.** Recursive-descent function and call-graph recovery with inline symbol and string resolution.
- **P-Code Emulation.** Sandboxed software emulator executes target code without native host execution, unmasking runtime strings, keys, and decrypted payloads.
- **Protectors & Packing.** High-entropy mapping, abnormal memory layouts, W^X violations, and signature matching for commercial protectors.
- **Indicators.** Structured extraction of URLs, IPs, emails, registry keys, file paths, commands, and user agents.
- **Strict Safety.** `#![forbid(unsafe_code)]` enforced across the entire codebase. Malformed or hostile inputs can never compromise the host.

<br>

# AI Analyst Configuration

To enable the autonomous multi-agent analysis loop, provide an [OpenRouter](https://openrouter.ai) API key. REA automatically locates your key from environment variables or a `.env` file in the current or ancestor directories:

```bash
# In .env or shell environment:
OPENROUTER_API_KEY="sk-or-v1-..."

# Optional model selection (defaults to high-throughput free tier):
OPENROUTER_MODEL="nvidia/nemotron-3.5-lightning:free"
```

If no key is configured, REA operates in purely deterministic offline mode.

<br>

# Example Usage

Pass REA the path to any binary or firmware target. Reports are generated beside the input as `re-<name>.md` and `re-<name>.pdf`.

**Windows x64 (PowerShell)**
```powershell
.\rea.exe .\sample
```

**macOS Apple Silicon**
```bash
./rea ./sample
```

**Linux x64**
```bash
./rea ./sample
```

<br>

# Supported Targets

| Format | Parsing | Analysis & Triage |
| --- | --- | --- |
| PE (EXE, DLL, SYS) | Full | x86, x86-64, imports, mitigations, imphash |
| ELF (Linux, BSD, Android) | Full | x86/x86-64 + every SLEIGH architecture |
| Mach-O and universal | Full | x86/x86-64 + every SLEIGH architecture |
| U-Boot images (`uImage`) | Full | Kernel identification, compression, rootfs offsets |
| SquashFS 4.0 Filesystems | In-memory | Unpacking, credential discovery, sub-target triage |
| Raw ARM Cortex-M firmware | Vector table, flash base | ARM Thumb disassembly and handler mapping |
| Any raw or unknown file | Identity, hashes, entropy | String extraction, indicators, entropy mapping |

Non-x86 disassembly is powered by embedded SLEIGH processor specifications compiled directly by a pure-Rust engine. No Ghidra, Java, Python, or external command-line tools are required.

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

Prebuilt binaries for macOS Apple Silicon (`aarch64-apple-darwin`) are located in [`releases/`](releases). To build for Windows or Linux, follow the cross-compilation and native build commands above.

<br>

# Quality & Verification

```bash
cargo xtask check
```

Enforces strict zero-violation coding standards (`CONVENTIONS.md`), runs `cargo fmt --check`, denies all Clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings`), and executes the complete workspace test suite.

<br>

# License

[MIT License](LICENSE)
