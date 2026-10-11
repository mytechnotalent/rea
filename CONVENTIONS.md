# Coding Conventions

Every Rust file in this repository follows the same standard, with no exceptions. `cargo xtask check` enforces it and fails on any violation.

<br>

# File header
Every `.rs` file starts with a `//!` header: a title line, then the file name, author, date, and the full MIT license.

```rust
//! Title Of The File
//!
//! **File:** `name.rs`
//! **Author:** Kevin Thomas
//! **Date:** 2026
//!
//! MIT License
//!
//! Copyright (c) 2026 Kevin Thomas
//! ...
```

A blank line follows the header.

<br>

# Crate attributes
Every crate root carries:

```rust
#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
```

<br>

# Imports
Each private `use` group has a comment directly above it, such as `// Import dependencies from std`. Public re-exports at the crate root need no comment.

<br>

# Documentation
Every item has a `///` doc comment: public and private functions, constants, types, struct fields, enum variants, trait methods, and tests. Function docs use these sections, in this order, when they apply:

1. `# Arguments`
2. `# Returns`
3. `# Errors`
4. `# Panics`
5. `# Example`

The doc comment comes first, then attributes such as `#[must_use]`, then the item.

<br>

# Functions
- A function body is at most 8 lines, as counted by `clippy::too_many_lines` with `too-many-lines-threshold = 8`.
- No blank lines inside a function.

<br>

# Tests
Every file that defines a function has a `#[cfg(test)] mod tests` that imports `use super::*;`. Each test has a doc comment saying what it verifies and is named `test_<behavior>`.

<br>

# Exemptions
Only generated build-time output under `OUT_DIR` is exempt from the custom checks.

<br>

# License
[MIT License](https://github.com/mytechnotalent/rea/blob/main/LICENSE)
