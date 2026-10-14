//! Reverse Engineering Assistant
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

/// Autonomous AI analyst loop.
pub mod agent;

/// Command-line entry point and argument handling.
pub mod cli;

/// Report composition from analysis results.
pub mod compose;

/// Report output locations.
pub mod paths;

/// Security artifact and threat dossier generation.
pub mod artifacts;

/// Interactive target inspection commands.
pub mod inspect;

/// Model Context Protocol (MCP) server for AI agents.
pub mod mcp;

/// Engine tools invoked by the autonomous agent loop.
pub mod tools;

/// External analysis provider (Binary Ninja over MCP).
pub mod provider;

/// Machine-readable JSON report.
pub mod json;
