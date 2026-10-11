![image](Reverse-Engineering-Assistant.png)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant

**Point it at a binary. Get a complete reverse engineering report.**

```bash
rea suspect.exe
```

Writes `re-suspect.md` and `re-suspect.pdf` next to the file. That's the whole workflow.

**Binaries in. Answers out.** One command. Fully offline. Built in Rust.

<br>

# Features
- **One command.** `rea <file>` produces a Markdown and a typeset PDF report beside the target.
- **Any file.** Format is detected from contents, not the extension.
- **Formats.** PE (EXE/DLL/SYS), ELF, Mach-O and universal binaries, and raw ARM Cortex-M firmware.
- **Architectures.** x86 and x86-64, plus every SLEIGH family: ARM/Thumb, AArch64, MIPS, PowerPC, RISC-V, SPARC, SuperH, AVR, 68000, MSP430, Xtensa, LoongArch — both endiannesses.
- **Static analysis.** Hashes, import hash, entropy, sections, imports/exports, mitigations (ASLR, DEP/NX, CFG, PIE, RELRO, canaries), and capability classification.
- **Disassembly.** Recursive-descent function and call-graph recovery with imports, strings, and functions resolved inline.
- **Emulation.** A sandboxed, pure-software p-code emulator runs the target's own code with no native execution, and reports strings decrypted at runtime.
- **Packing & protectors.** High entropy, writable-executable code, abnormal layouts, and named commercial protectors by their section signatures.
- **Indicators.** URLs, IPs, emails, registry keys, file paths, commands, and user agents.
- **Safe.** `#![forbid(unsafe_code)]` throughout; malformed or malicious inputs can never execute on the host or crash a report.

<br>

# Example Usage
Pass REA the path to a binary. Reports are written beside the input as `re-<name>.md` and `re-<name>.pdf`.

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

<br>

# Supported Targets
| Format | Parsing | Disassembly |
| --- | --- | --- |
| PE (EXE, DLL, SYS) | Full | x86, x86-64 |
| ELF (Linux, BSD, Android) | Full | x86/x86-64 + every SLEIGH architecture |
| Mach-O and universal | Full | x86/x86-64 + every SLEIGH architecture |
| Raw ARM Cortex-M firmware | Vector table, inferred flash base, handlers | ARM Thumb |
| Any other file | Identity, hashes, entropy, strings, indicators | — |

Non-x86 disassembly is driven by embedded [Ghidra](https://github.com/NationalSecurityAgency/ghidra) SLEIGH processor specifications compiled by a pure-Rust engine, so no Ghidra, Java, or external tools are needed.

<br>

# Build from Source
Install the [stable Rust toolchain](https://rustup.rs), clone the repository, and build on the target operating system. Windows builds require the Visual Studio C++ Build Tools.

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

Prebuilt binaries are in [`releases/`](releases).

<br>

# Verify
```bash
cargo xtask check
```

Enforces the coding standard, then runs `cargo fmt --check`, `cargo clippy` with every warning denied, and the full test suite.

<br>

# License
[MIT License](LICENSE)
