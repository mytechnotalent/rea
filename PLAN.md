# REA — Master Plan

**Project:** Reverse Engineering Assistant (REA)
**Repository:** `mytechnotalent/rea`
**Current release:** v5.6.3
**Document owner:** Kevin Thomas

---

## 1. Mission

Build the best **self-contained, pure-Rust** reverse engineering framework: one command in,
a complete analysis out. REA must own its engine — no Hopper, Ghidra, IDA, JEB, Java, or
external command-line tools. Everything runs locally, deterministically, and safely.

**Primary product:** `rea <target>` — the one-shot. Produces `re-<name>.md`, `re-<name>.pdf`,
and `re-<name>.json` beside the target.

**Secondary product:** `rea serve` — the Model Context Protocol (MCP) server for AI coding
agents (OpenCode, Claude Code, Cursor, Windsurf).

---

## 2. Non-negotiable principles

- **Pure Rust, std-only at runtime.** No external binaries, no JVM, no Python.
- **`#![forbid(unsafe_code)]`** across every crate.
- **Conventions gate** (`cargo xtask check`): MIT file header with matching file name; every
  item documented; function bodies ≤ 8 lines (post-`rustfmt`); no blank lines inside blocks;
  `// Import` comment above every private `use`; a `#[cfg(test)] mod tests { use super::*; }`
  in every file that defines functions; tests named `test_<behavior>`; no `#[allow(...)]`.
- **Deterministic, offline-first.** An API key enables the AI analyst loop; absence of a key
  yields a full deterministic report with zero network requests.
- **A+ or it does not ship.** Each capability must be complete, tested, and honest about
  limitations before a release is cut.

---

## 3. Release history

| Version | Headline |
| --- | --- |
| v4.0.0 | Native pure-Rust decompiler, multi-arch unpacking, rules/vulns/crypto, .NET CIL, 18-tool MCP. |
| v5.0.0 | WASM, SQLite, core dumps, EVM, plist, Flutter, PCAP/HAR, source maps, MS-DOS; JSON report; setup skill install. |
| v5.1.0 | DWARF, PDB, YARA-subset, Java `.class`, Android `resources.arsc`, crypto expansion, artifact report sections, 31-tool MCP. |
| v5.2.0 | Autonomous analyst rebuilt: native tool calling, 90-turn loop, LLM-authored grounded report, entropy regions + 8051 tools, DeepSeek V4.1 Flash default. |
| v5.2.1 | Fix decompiler infinite loop (self/cyclic copy map, oscillating fixpoints); bound the analyst loop with a wall-clock budget and forced final report. |
| v5.3.0 | Firmware container decoding + load-base re-basing (`REA_LOAD_BASE`), analyst dedupe/no-progress stop, full-string grounding, instruction-annotation accuracy fixes. |
| v5.3.1 | Fix silent AI failure (missing `max_tokens` caused HTTP 402); add `disasm` tool (function listing + call graph + strings); richer evidence; behavior-forcing prompt; surface API errors; automatic model failover. |
| v5.4.0 | Nation-state report redesign: LLM authors the full intelligence assessment (BLUF, capability/behavior/function analysis, attribution), deterministic findings become an Evidence Appendix, always-on citation verification plus adversarial reviewers and an adjudicator (`REA_VERIFY`). |
| v5.4.1 | Reliable model failover (`deepseek/deepseek-v4.1-flash` -> `openrouter/free`), case-insensitive citation verification, and documented analyst env knobs. |
| v5.5.0 | Provider-selectable one-shot: run the analyst on OpenCode Go (`REA_PROVIDER=opencode-go`, default `deepseek-v4.1-flash`) or OpenRouter, via configurable `REA_API_BASE`/`REA_API_KEY`/`REA_MODEL`. |
| v5.6.0 | Differential comparison (`rea diff`, MCP `compare_binaries`), Coverage & Limitations report section and JSON `coverage` block, and untrusted-content provenance marking in MCP results. |
| v5.6.1 | Report formatting/accuracy: preserve fenced code one instruction per line, `0x`/hex deterministic listings, and hedge the loop heuristic. |
| v5.6.2 | Fix overlapping text in PDF tables: long tokens wrap inside constrained columns. |
| v5.6.3 | New tagline positioning the pure-Rust, zero-dependency, verifiable core. |

---

## 3a. v5.4.0 — Nation-State Report Redesign (in progress)

**Goal:** the report is the product. An LLM authors a nation-state-grade intelligence assessment
end to end; the deterministic engine is the evidence appendix. No unproven claim enters the report.

**Architecture — Author, Adversarial Reviewers, Adjudicator:**

1. **Author** — the analyst loop writes the full narrative with inline citations (addresses,
   strings, offsets) and a confidence (High/Medium/Low) per non-obvious claim.
2. **Deterministic verifier (always on)** — every cited address and quoted string must exist in the
   evidence corpus (dossier + strings + all tool observations); anything absent is flagged.
3. **Adversarial reviewers (LLM, opt-in)** — independent passes attempt to falsify each claim
   against the evidence and report contradictions and unsupported statements.
4. **Adjudicator (LLM, opt-in)** — resolves verdicts; only proven claims survive into the final
   report, and everything rejected is listed in a verification appendix.

**Report structure (LLM-authored body, deterministic evidence appendix):**

1. BLUF — Bottom Line Up Front
2. Target Identification & Provenance
3. Capability Assessment (ATT&CK/MBC + crypto + network, each with confidence)
4. Behavioral Analysis (end-to-end, plain English)
5. Function Analysis (per key function: real summary, call graph, constants, pseudocode)
6. Network, C2 & IOCs
7. Secrets & Cryptography
8. Vulnerabilities & Exploitation
9. Anti-Analysis & Evasion
10. Attribution & Tooling
11. Assessment Confidence & Limitations
12. Evidence Appendix (function inventory, call graph, strings, sections, imports)

**Status:** report assembly now leads with the authored narrative and relegates deterministic
findings to an Evidence Appendix; the analyst prompt emits the structure above; deterministic
verification covers addresses and strings. Adversarial reviewers and the adjudicator are next.

---

## 4. Shipped capability map

**Engine (`crates/rea-engine`)**

Formats & loaders: PE, ELF, Mach-O (incl. universal), Java class, DEX/APK, .NET CIL, Go,
WASM, SQLite, MS-DOS MZ, Apple plist, Android `resources.arsc`, WASM, firmware (uImage,
SquashFS, raw Cortex-M), Electron ASAR, PCAP/HAR, ELF core dumps, EVM bytecode, JS source
maps, Flutter APKs.

Analysis: recursive-descent disassembly (SLEIGH + iced-x86), p-code emulation and unpacking
(OEP, IAT, multi-stage), a native decompiler (CFG, structurer, constant folding, copy
propagation, dead-store elimination), capability rules (capa-class, ATT&CK/MBC), vulnerability
audit (CWE + SARIF), crypto-constant detection, cipher recovery, DWARF, PDB, and YARA-subset.

**Frontends:** one-shot report (Markdown/PDF/JSON), focused CLI subcommands, MCP server.

---

## 5. v5.2.0 roadmap

Priority order. Release when every item is complete, tested, and the gate is green.

1. **Pure-Rust inflate.** Depend on `miniz_oxide` (library, pure Rust — not an external tool)
   to unlock compressed data everywhere.
2. **Rust binary metadata.** Decode the `cargo-auditable` `.dep-v0` zlib JSON plus `.rustc`
   and embedded registry-path markers: crate names, versions, sources, and `rustc` commit.
3. **Zip/APK/SAZ entry extraction.** Decompress STORED/DEFLATE entries: read `resources.arsc`
   and `classes.dex` from an APK, and Fiddler SAZ request/response text.
4. **PCAP protocol summaries.** Ethernet/IP/TCP/UDP decode, DNS names, TLS SNI, endpoints, and
   a protocol histogram.
5. **ELF/Mach-O surface.** Full symbols and relocations; Mach-O entitlements and code signature.
6. **Decompiler type inference.** Primitive/pointer type recovery and array detection.

Deferred from earlier: SAZ content (item 3) and decompiler types (item 6).

---

## 6. Quality gate & release process

1. `cargo fmt --all`
2. `cargo xtask check` — conventions (0 violations) + `fmt --check` + `clippy --workspace
   --all-targets -- -D warnings` + full workspace tests. Must exit 0.
3. Bump `version` in `Cargo.toml` (`workspace.package`).
4. Rewrite `README.md` for the new version (current release notes only; one-shot first, MCP
   second; Windows / macOS Apple Silicon / Linux instructions).
5. Stage only intended files; verify no secrets; commit `Release vX.Y.Z: <summary>`.
6. `git tag -f vX.Y.Z`; `git push origin HEAD:main`; `git push origin vX.Y.Z`.
7. `gh release create vX.Y.Z --title vX.Y.Z --notes <notes>`.

---

## 7. Active investigation

**Target:** `analysis/unknown/seagate.bin` (524,288 bytes; unrecognized by signature detector).
Initial triage: little-endian, ARM 32-bit exception vectors (ARM mode) at offset `0x40`
(`mov r0, <mode>` / `msr cpsr` / `ldr sp, [pc, …]`), with a header/table in the first `0x40`
bytes. Hypothesis: Seagate hard-disk firmware image.

**Goal:** recover format structure, entry points, and semantics; capture findings here and in
the REA report.

**Status update:** the first-pass report for `seagate.bin` was inadequate — it recovered ~0.3%
of the code, produced templated prose, invented semantics, and tied no string to any code. This
target exposed that REA's firmware analysis has no real foundation. The v5.2.0 work below is the
rebuild that fixes it.

---

## 8. Firmware analysis architecture (v5.2.0 rebuild)

The current pipeline (signatures → sections → strings → text) cannot prove behavior on opaque
firmware. The rebuild introduces four foundations that every firmware capability shares.

### 8.1 Region model

Every byte of a target belongs to exactly one classified region:

```
Region { start, size, kind, entropy, confidence, evidence }
kind  ∈ Code | Data | Table | Strings | Mmio | Encrypted | Compressed | Zero | Unknown
```

Regions are produced by entropy segmentation plus structure heuristics (magic records,
zero runs, pointer density, string density). The report reasons over regions, never over the
whole file at once. A high-entropy region is classified `Encrypted`/`Compressed` only after
magic scans and decompression attempts, never from entropy alone.

### 8.2 Analysis IR (operand-level)

Disassembly produces structured instructions, not text:

```
Insn { address, size, bytes, mnemonic, operands: Vec<Operand>, references: Vec<Reference> }
Operand  ∈ Reg(name) | Imm(value) | Mem(base, index, scale, disp, resolved: Option<u64>)
Reference { from, to, kind: Branch | Call | LiteralLoad | DataLoad | MmioWrite }
```

References are resolved from real operands (PC-relative literals, pool pointers, computed
base+offset), so `Reference.to` can name a target: a function, a string, a table entry, or an
MMIO register. This is the mechanism that lets the report prove "this routine uses `Bad CRC
Seg:`" instead of guessing.

### 8.3 Evidence contract

Every report statement cites its source:
- Observed — bytes/fields read directly.
- Derived — computed from the IR (counts, xrefs, regions).
- Inferred — interpretation, explicitly uncertain.

The narrative is generated only from IR/xref/region facts. No templated prose, no strings
invented, no offsets that do not exist. A statement that cannot cite evidence is not emitted.

### 8.4 Crypto / compression classification

Ordered checks, each recorded as evidence:
1. Magic scan: zlib (`78 01/9c/da`), gzip (`1f 8b`), LZMA (`5d 00 00`), xz, bzip2, LZ4, Zstd.
2. Decompression attempt on candidate regions (pure-Rust inflate and variants).
3. Entropy + byte-distribution statistics for remaining high-entropy regions → `Encrypted`
   with an explicit "no standard container detected" note.
4. Known vendor container markers (e.g. `0xdeadf00d`, `0xfeed…` record tags) reported as
   named structure.

### 8.5 Deliverables

New engine modules: `regions.rs` (8.1/8.4), `ir.rs`/`xrefs.rs` (8.2), extended multiarch
decode. New report sections: **Target Structure** (regions + container records + MMIO map) and
**Code ↔ Data References** (proven xrefs). CLI: `rea regions <target>`.

### 8.6 Sequence

1. Architecture (this section).
2. Foundation: region segmenter + crypto classification (`regions.rs`).
3. Container/record detection (magics, descriptor headers, MMIO clusters).
4. Operand IR + reference resolver (`xrefs.rs`); Code ↔ Data references.
5. ARM/Thumb structuring + decompilation; grounded narrative.
6. Regression gate on `analysis/unknown/seagate.bin` and the existing corpora.
