//! Software Bill of Materials Generation
//!
//! **File:** `sbom.rs`
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
use crate::{
    provenance::{Component, Provenance},
    time::utc_timestamp,
};

// Import dependencies from serde_json
use serde_json::{json, Value};

/// `CycloneDX` specification version emitted.
const CYCLONEDX_SPEC: &str = "1.5";

/// `SPDX` specification version emitted.
const SPDX_SPEC: &str = "SPDX-2.3";

/// Renders a `CycloneDX` SBOM document.
///
/// # Arguments
///
/// * `name` - Target name.
/// * `tool_version` - Producing tool version.
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The pretty JSON document.
#[must_use]
pub fn cyclonedx(name: &str, tool_version: &str, provenance: &Provenance) -> String {
    let metadata = metadata(name, tool_version, provenance);
    let components = components(provenance);
    let document = json!({"bomFormat": "CycloneDX", "specVersion": CYCLONEDX_SPEC, "version": 1, "metadata": metadata, "components": components});
    serde_json::to_string_pretty(&document).unwrap_or_default()
}

/// Builds the `CycloneDX` metadata object.
///
/// # Arguments
///
/// * `name` - Target name.
/// * `tool_version` - Producing tool version.
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The JSON object.
fn metadata(name: &str, tool_version: &str, provenance: &Provenance) -> Value {
    json!({
        "timestamp": now(),
        "tools": [{"vendor": "REA", "name": "rea", "version": tool_version}],
        "component": {"type": "application", "name": name},
        "properties": properties(provenance),
    })
}

/// Builds `CycloneDX` metadata properties.
///
/// # Arguments
///
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The JSON array.
fn properties(provenance: &Provenance) -> Value {
    let mut items = Vec::new();
    push_property(&mut items, "toolchain", provenance.toolchain.as_deref());
    push_property(&mut items, "build_path", provenance.build_path.as_deref());
    Value::Array(items)
}

/// Appends a provenance property when a value is present.
///
/// # Arguments
///
/// * `items` - Accumulated properties.
/// * `name` - Property name.
/// * `value` - Property value.
fn push_property(items: &mut Vec<Value>, name: &str, value: Option<&str>) {
    if let Some(value) = value {
        items.push(json!({"name": name, "value": value}));
    }
}

/// Renders an `SPDX` SBOM document.
///
/// # Arguments
///
/// * `name` - Target name.
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The pretty JSON document.
#[must_use]
pub fn spdx(name: &str, provenance: &Provenance) -> String {
    let info = creation_info();
    let packages = packages(provenance);
    let namespace = format!("https://github.com/mytechnotalent/rea/{name}");
    let document = json!({"spdxVersion": SPDX_SPEC, "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT", "name": name, "documentNamespace": namespace, "creationInfo": info, "packages": packages});
    serde_json::to_string_pretty(&document).unwrap_or_default()
}

/// Builds the `SPDX` creation info object.
///
/// # Returns
///
/// The JSON object.
fn creation_info() -> Value {
    json!({"created": now(), "creators": ["Tool: rea"]})
}

/// Builds the `CycloneDX` components array.
///
/// # Arguments
///
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The JSON array.
fn components(provenance: &Provenance) -> Value {
    Value::Array(provenance.components.iter().map(component_json).collect())
}

/// Builds one `CycloneDX` component object.
///
/// # Arguments
///
/// * `component` - Recovered component.
///
/// # Returns
///
/// The JSON object.
fn component_json(component: &Component) -> Value {
    json!({"type": component.kind.label(), "name": component.name, "version": component.version})
}

/// Builds the `SPDX` packages array.
///
/// # Arguments
///
/// * `provenance` - Recovered provenance.
///
/// # Returns
///
/// The JSON array.
fn packages(provenance: &Provenance) -> Value {
    let items = provenance.components.iter().enumerate();
    Value::Array(items.map(package_json).collect())
}

/// Builds one `SPDX` package object.
///
/// # Arguments
///
/// * `item` - Component index and value.
///
/// # Returns
///
/// The JSON object.
fn package_json(item: (usize, &Component)) -> Value {
    let (index, component) = item;
    json!({
        "name": component.name,
        "versionInfo": component.version,
        "SPDXID": format!("SPDXRef-Package-{index}"),
    })
}

/// Returns the current time as an ISO-8601 UTC timestamp.
///
/// # Returns
///
/// The timestamp string.
fn now() -> String {
    let elapsed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    utc_timestamp(elapsed.map_or(0, |d| d.as_secs()))
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds provenance with one component.
    ///
    /// # Returns
    ///
    /// The provenance.
    fn sample() -> Provenance {
        let components = vec![component()];
        Provenance {
            toolchain: Some("go1.22".to_string()),
            build_path: Some("/build/app".to_string()),
            components,
        }
    }

    /// Builds a sample library component.
    ///
    /// # Returns
    ///
    /// The component.
    fn component() -> Component {
        Component {
            name: "example/module".to_string(),
            version: "v1.0.0".to_string(),
            kind: crate::provenance::ComponentKind::Library,
        }
    }

    /// Verifies `CycloneDX` output shape.
    #[test]
    fn test_cyclonedx_shape() {
        let text = cyclonedx("app", "7.0.0", &sample());
        assert!(text.contains("\"bomFormat\": \"CycloneDX\""));
        assert!(text.contains("example/module"));
    }

    /// Verifies `SPDX` output shape.
    #[test]
    fn test_spdx_shape() {
        let text = spdx("app", &sample());
        assert!(text.contains("\"spdxVersion\": \"SPDX-2.3\""));
        assert!(text.contains("\"SPDXID\": \"SPDXRef-Package-0\""));
    }
}
