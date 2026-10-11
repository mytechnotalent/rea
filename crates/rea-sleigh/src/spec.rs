//! Compiled-Specification Cache
//!
//! **File:** `spec.rs`
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

// Import dependencies from std
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

// Import dependencies from sleigh
use sleigh::{CompiledSpec, Compiler};

// Import crate items
use crate::{embed, error::SleighError};

/// Process-wide cache of compiled specifications, keyed by root path.
static CACHE: OnceLock<Mutex<HashMap<&'static str, Arc<CompiledSpec>>>> = OnceLock::new();

/// Loads and caches the compiled specification rooted at a path.
///
/// Compilation happens once per path; later calls clone the cached handle.
///
/// # Arguments
///
/// * `root` - Family-relative root path, such as `ARM/ARM7_le.slaspec`.
///
/// # Returns
///
/// A shared handle to the compiled specification.
///
/// # Errors
///
/// Returns [`SleighError`] when the file is missing or fails to compile.
pub fn load(root: &'static str) -> Result<Arc<CompiledSpec>, SleighError> {
    let mut guard = locked();
    if let Some(spec) = guard.get(root) {
        return Ok(Arc::clone(spec));
    }
    let spec = guard.entry(root).or_insert(Arc::new(compile(root)?));
    Ok(Arc::clone(spec))
}

/// Locks the process-wide specification cache.
///
/// # Returns
///
/// A guard over the cache map, recovering from a poisoned lock.
fn locked() -> std::sync::MutexGuard<'static, HashMap<&'static str, Arc<CompiledSpec>>> {
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Compiles one specification from the embedded source set.
///
/// # Arguments
///
/// * `root` - Family-relative root path.
///
/// # Returns
///
/// The compiled specification.
///
/// # Errors
///
/// Returns [`SleighError`] when the file is missing or fails to compile.
fn compile(root: &str) -> Result<CompiledSpec, SleighError> {
    let mut db = embed::source_db();
    let file =
        embed::file_id(&db, root).ok_or_else(|| SleighError::MissingSpec(root.to_string()))?;
    let result = Compiler::new(&mut db).compile(file);
    result.map_err(|error| SleighError::Compile(format!("{error:?}")))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that ARM compiles and is cached on the second call.
    #[test]
    fn test_load_arm_is_cached() {
        let first = load("ARM/ARM7_le.slaspec").unwrap();
        let second = load("ARM/ARM7_le.slaspec").unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    /// Verifies that a missing specification is reported.
    #[test]
    fn test_load_missing() {
        let result = load("nope/none.slaspec");
        assert!(matches!(result, Err(SleighError::MissingSpec(_))));
    }
}
