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
    provenance
}

/// Recovers component provenance from Go build metadata.
///
/// # Arguments
///
/// * `provenance` - Provenance to populate.
/// * `analysis` - Completed analysis.
fn from_go(provenance: &mut Provenance, analysis: &Analysis) {
    let Some(go) = &analysis.go else { return };
    provenance.toolchain = Some(go.version.clone());
    push_main(provenance, &go.module_path);
    for dep in &go.dependencies {
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
}
