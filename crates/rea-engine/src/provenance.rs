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

// Import dependencies from serde_json
use serde_json::Value;

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

    /// Build-origin path from debug metadata, when known.
    pub build_path: Option<String>,

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
    from_imports(&mut provenance, analysis);
    from_debug(&mut provenance, analysis);
    from_native(&mut provenance, analysis);
    provenance
}

/// Recovers build provenance, including debug-derived build paths.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The recovered provenance.
#[must_use]
pub fn recover_detailed(analysis: &Analysis, bytes: &[u8]) -> Provenance {
    let mut provenance = recover(analysis);
    from_dwarf(&mut provenance, bytes);
    from_auditable(&mut provenance, analysis, bytes);
    provenance
}

/// Recovers Rust `cargo-auditable` crate dependencies.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
fn from_auditable(provenance: &mut Provenance, analysis: &Analysis, bytes: &[u8]) {
    let Some(json) = auditable_json(analysis, bytes) else {
        return;
    };
    parse_packages(provenance, &json);
}

/// Extracts and decompresses a `cargo-auditable` JSON document.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
/// * `bytes` - Raw target bytes.
///
/// # Returns
///
/// The decompressed JSON, or `None`.
fn auditable_json(analysis: &Analysis, bytes: &[u8]) -> Option<Vec<u8>> {
    let binary = analysis.binary.as_ref()?;
    let section = binary.sections.iter().find(|s| s.name.ends_with(".dep-v0"))?;
    miniz_oxide::inflate::decompress_to_vec_zlib(section.data(bytes)).ok()
}

/// Parses recovered `cargo-auditable` package metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `json` - Decompressed JSON document.
fn parse_packages(provenance: &mut Provenance, json: &[u8]) {
    let Ok(value) = serde_json::from_slice::<Value>(json) else {
        return;
    };
    let Some(packages) = value.get("packages").and_then(Value::as_array) else {
        return;
    };
    provenance.components.extend(packages.iter().filter_map(package_component));
}

/// Builds a library component from a `cargo-auditable` package object.
///
/// # Arguments
///
/// * `package` - Package JSON object.
///
/// # Returns
///
/// The component, or `None` when unnamed.
fn package_component(package: &Value) -> Option<Component> {
    let name = package.get("name").and_then(Value::as_str)?;
    let version = package.get("version").and_then(Value::as_str).unwrap_or("");
    Some(component(name, version, ComponentKind::Library))
}

/// Recovers a DWARF compilation directory as the build path.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `bytes` - Raw target bytes.
fn from_dwarf(provenance: &mut Provenance, bytes: &[u8]) {
    if provenance.build_path.is_some() {
        return;
    }
    let Some(dwarf) = crate::dwarf::parse_dwarf(bytes) else {
        return;
    };
    provenance.build_path = first_comp_dir(&dwarf.compile_units);
}

/// Returns the first non-empty compilation directory.
///
/// # Arguments
///
/// * `units` - Compilation units.
///
/// # Returns
///
/// The first non-empty `comp_dir`, or `None`.
fn first_comp_dir(units: &[crate::dwarf::CompileUnit]) -> Option<String> {
    units.iter().map(|u| u.comp_dir.clone()).find(|dir| !dir.is_empty())
}

/// Recovers the build-origin path from debug metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_debug(provenance: &mut Provenance, analysis: &Analysis) {
    let Some(binary) = &analysis.binary else {
        return;
    };
    provenance.build_path = binary.facts.iter().find_map(pdb_path);
}

/// Extracts a PDB path from a debug-database fact.
///
/// # Arguments
///
/// * `fact` - Binary fact.
///
/// # Returns
///
/// The build path, or `None`.
fn pdb_path(fact: &crate::model::Fact) -> Option<String> {
    if fact.label != "Debug database (PDB)" {
        return None;
    }
    let path = fact.value.split(" (GUID").next().unwrap_or("");
    (!path.is_empty()).then(|| path.to_string())
}

/// Recovers linked library components from import metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_imports(provenance: &mut Provenance, analysis: &Analysis) {
    let Some(binary) = &analysis.binary else {
        return;
    };
    let mut seen = std::collections::BTreeSet::new();
    for import in &binary.imports {
        push_library(provenance, &mut seen, &import.library);
    }
}

/// Pushes a distinct library component.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `seen` - Already-recorded library names.
/// * `library` - Library name.
fn push_library(
    provenance: &mut Provenance,
    seen: &mut std::collections::BTreeSet<String>,
    library: &str,
) {
    if !library.is_empty() && seen.insert(library.to_string()) {
        provenance.components.push(component(library, "", ComponentKind::Library));
    }
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
    push_refs(provenance, &dotnet.references);
}

/// Pushes managed assembly reference components.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `refs` - Managed assembly references.
fn push_refs(provenance: &mut Provenance, refs: &[crate::dotnet::DotNetAssemblyRef]) {
    for reference in refs {
        let item = component(&reference.name, &reference.version, ComponentKind::Library);
        provenance.components.push(item);
    }
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

    /// Verifies that linked libraries are deduplicated.
    #[test]
    fn test_push_library_dedupes() {
        let mut provenance = Provenance::default();
        let mut seen = std::collections::BTreeSet::new();
        push_library(&mut provenance, &mut seen, "KERNEL32.dll");
        push_library(&mut provenance, &mut seen, "KERNEL32.dll");
        assert_eq!(provenance.components.len(), 1);
    }

    /// Verifies PDB path extraction from a debug fact.
    #[test]
    fn test_pdb_path_strips_guid() {
        let fact = debug_fact(r"C:\app.pdb");
        assert_eq!(pdb_path(&fact).as_deref(), Some(r"C:\app.pdb"));
        let other = crate::model::Fact::new("Rich header", "Present");
        assert_eq!(pdb_path(&other), None);
    }

    /// Verifies cargo-auditable package parsing.
    #[test]
    fn test_auditable_json() {
        let json = br#"{"packages":[{"name":"libc","version":"0.2.190"}]}"#;
        let mut provenance = Provenance::default();
        parse_packages(&mut provenance, json);
        assert_eq!(provenance.components.len(), 1);
        assert_eq!(provenance.components[0].name, "libc");
    }

    /// Verifies DWARF compilation directory selection.
    #[test]
    fn test_first_comp_dir() {
        let empty = crate::dwarf::CompileUnit::default();
        let unit = crate::dwarf::CompileUnit { name: "b.c".into(), comp_dir: "/build".into() };
        assert_eq!(first_comp_dir(&[empty, unit]).as_deref(), Some("/build"));
    }

    /// Builds a synthetic debug-database fact.
    ///
    /// # Arguments
    ///
    /// * `path` - PDB path text.
    ///
    /// # Returns
    ///
    /// The fact.
    fn debug_fact(path: &str) -> crate::model::Fact {
        let value = format!("{path} (GUID abcd, age 3)");
        crate::model::Fact::new("Debug database (PDB)", &value)
    }
}
