//! Build Provenance Recovery
//!
//! **File:** `provenance.rs`
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

// Import dependencies from crate
use crate::Analysis;

/// Kind of software component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentKind {
    /// The target application itself.
    Application,

    /// A linked or bundled library.
    Library,

    /// A managed or language runtime framework.
    Framework,
}

impl ComponentKind {
    /// Returns the `CycloneDX` component type label.
    ///
    /// # Returns
    ///
    /// The lowercase label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ComponentKind::Application => "application",
            ComponentKind::Library => "library",
            ComponentKind::Framework => "framework",
        }
    }
}

/// One recovered software component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// Component name or import path.
    pub name: String,

    /// Component version, when known.
    pub version: String,

    /// Component kind.
    pub kind: ComponentKind,
}

/// Recovered build provenance for a target.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Provenance {
    /// Toolchain or runtime version, when known.
    pub toolchain: Option<String>,

    /// Recovered components.
    pub components: Vec<Component>,
}

/// Recovers build provenance from an analysis.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The recovered provenance.
#[must_use]
pub fn recover(analysis: &Analysis) -> Provenance {
    let mut provenance = Provenance::default();
    from_go(&mut provenance, analysis);
    from_dotnet(&mut provenance, analysis);
    from_native(&mut provenance, analysis);
    provenance
}

/// Recovers a native toolchain fingerprint from strings.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_native(provenance: &mut Provenance, analysis: &Analysis) {
    if provenance.toolchain.is_some() {
        return;
    }
    provenance.toolchain = detect_toolchain(&analysis.strings);
}

/// Detects a toolchain from recovered strings.
///
/// # Arguments
///
/// * `strings` - Recovered strings.
///
/// # Returns
///
/// The toolchain label, or `None`.
fn detect_toolchain(strings: &[crate::strings::FoundString]) -> Option<String> {
    strings.iter().find_map(|s| toolchain_of(&s.text))
}

/// Classifies a string as a toolchain fingerprint.
///
/// # Arguments
///
/// * `text` - Candidate string.
///
/// # Returns
///
/// The toolchain label, or `None`.
fn toolchain_of(text: &str) -> Option<String> {
    if is_gcc(text) {
        return Some(clamp(text));
    }
    native_label(text).map(str::to_string)
}

/// Reports whether a string is a GCC or Clang fingerprint.
///
/// # Arguments
///
/// * `text` - Candidate string.
///
/// # Returns
///
/// `true` for GCC or Clang.
fn is_gcc(text: &str) -> bool {
    text.starts_with("GCC:") || text.contains("clang version")
}

/// Returns a fixed native toolchain label.
///
/// # Arguments
///
/// * `text` - Candidate string.
///
/// # Returns
///
/// The label, or `None`.
fn native_label(text: &str) -> Option<&'static str> {
    if text.contains("Microsoft") && text.contains("C++") {
        return Some("Microsoft Visual C++");
    }
    (text.contains("rustc") || text.contains("rust_begin_unwind")).then_some("Rust")
}

/// Truncates a toolchain string.
///
/// # Arguments
///
/// * `text` - Toolchain string.
///
/// # Returns
///
/// The bounded string.
fn clamp(text: &str) -> String {
    text.chars().take(64).collect()
}

/// Recovers component provenance from Go build metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_go(provenance: &mut Provenance, analysis: &Analysis) {
    let Some(go) = &analysis.go else { return };
    set_toolchain(provenance, &go.version);
    push_main(provenance, &go.module_path);
    push_deps(provenance, &go.dependencies);
}

/// Sets the toolchain when non-empty.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `version` - Toolchain version.
fn set_toolchain(provenance: &mut Provenance, version: &str) {
    if !version.is_empty() {
        provenance.toolchain = Some(version.to_string());
    }
}

/// Appends Go dependency components.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `deps` - Go modules.
fn push_deps(provenance: &mut Provenance, deps: &[crate::go::GoModule]) {
    for dep in deps {
        provenance
            .components
            .push(component(&dep.path, &dep.version, ComponentKind::Library));
    }
}

/// Recovers component provenance from .NET assembly metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_dotnet(provenance: &mut Provenance, analysis: &Analysis) {
    let Some(dotnet) = &analysis.dotnet else {
        return;
    };
    if provenance.toolchain.is_none() {
        provenance.toolchain = Some(format!(".NET CLR {}", dotnet.runtime_version));
    }
    push_main(provenance, &dotnet.assembly_name);
}

/// Pushes the root application component when named.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `name` - Root component name.
fn push_main(provenance: &mut Provenance, name: &str) {
    if !name.is_empty() {
        provenance.components.push(component(name, "", ComponentKind::Application));
    }
}

/// Builds one component.
///
/// # Arguments
///
/// * `name` - Component name.
/// * `version` - Component version.
/// * `kind` - Component kind.
///
/// # Returns
///
/// The component.
fn component(name: &str, version: &str, kind: ComponentKind) -> Component {
    Component { name: name.to_string(), version: version.to_string(), kind }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a plain target yields no components.
    #[test]
    fn test_recover_empty() {
        let analysis = crate::analyze("blob.bin", b"plain data");
        assert_eq!(recover(&analysis).components.len(), 0);
    }

    /// Verifies native toolchain detection from a GCC string.
    #[test]
    fn test_detect_gcc_toolchain() {
        assert!(detect_toolchain(&[]).is_none());
        assert_eq!(
            toolchain_of("GCC: (Rev5, Built by MSYS2 project) 16.1.0"),
            Some("GCC: (Rev5, Built by MSYS2 project) 16.1.0".to_string())
        );
        assert_eq!(toolchain_of("rust_begin_unwind"), Some("Rust".to_string()));
    }
}
