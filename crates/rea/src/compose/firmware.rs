//! Firmware Filesystem and Artifacts
//!
//! **File:** `firmware.rs`
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

// Import dependencies from rea_engine
use rea_engine::{
    capabilities::Capability, insights::KeyFunction, Analysis, FirmwareTriage, SecurityArtifact,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

// Import crate::compose::text
use crate::compose::text;

/// Formats one security artifact row.
///
/// # Arguments
///
/// * `a` - Security artifact.
///
/// # Returns
///
/// Table row columns.
fn artifact_row(a: &SecurityArtifact) -> Vec<String> {
    vec![
        a.path.clone(),
        a.kind.clone(),
        a.detail.clone(),
        text::size(a.size),
    ]
}

/// Formats one executable path row.
///
/// # Arguments
///
/// * `path` - Executable path.
///
/// # Returns
///
/// Table row columns.
fn exe_row(path: &str) -> Vec<String> {
    vec![path.to_owned(), "ELF Executable".to_string()]
}

/// Formats one sub-target capability row.
///
/// # Arguments
///
/// * `c` - Discovered capability.
///
/// # Returns
///
/// Table row columns.
fn cap_row(c: &Capability) -> Vec<String> {
    vec![
        c.category.to_string(),
        c.description.to_string(),
        text::listed(&c.apis, 8),
    ]
}

/// Formats one key function row.
///
/// # Arguments
///
/// * `f` - Ranked key function.
///
/// # Returns
///
/// Table row columns.
fn func_row(f: &KeyFunction) -> Vec<String> {
    vec![
        f.name.clone(),
        text::hex(f.address),
        f.score.to_string(),
        f.reasons.join(", "),
    ]
}

/// Extracts filesystem metadata properties.
///
/// # Arguments
///
/// * `fw` - Firmware triage result.
///
/// # Returns
///
/// Tuple of compression, files count, and block size.
fn meta_props(fw: &FirmwareTriage) -> (String, String, u32) {
    let meta = fw.meta.as_ref();
    let comp = meta.map_or("Unknown", |m| m.compression.as_str()).to_string();
    let files = meta.map_or(0, |m| m.file_count).to_string();
    (comp, files, meta.map_or(0, |m| m.block_size))
}

/// Compiles filesystem metadata rows.
///
/// # Arguments
///
/// * `fw` - Firmware triage result.
///
/// # Returns
///
/// Label and value pairs.
fn summary_rows(fw: &FirmwareTriage) -> Vec<(String, String)> {
    let (comp, files, block) = meta_props(fw);
    vec![
        ("Filesystem Format".to_string(), "SquashFS 4.0".to_string()),
        ("Compression".to_string(), comp),
        ("Regular Files".to_string(), files),
        ("Block Size".to_string(), text::size(u64::from(block))),
    ]
}

/// Adds the filesystem summary table to the section.
///
/// # Arguments
///
/// * `section` - Section to append to.
/// * `fw` - Firmware triage result.
///
/// # Returns
///
/// The updated section.
fn add_summary(section: Section, fw: &FirmwareTriage) -> Section {
    let rows = summary_rows(fw);
    let table = Block::Table(text::pairs(["Property", "Value"], rows));
    section.with(text::subheading("Filesystem Summary")).with(table)
}

/// Adds security artifacts table to the section.
///
/// # Arguments
///
/// * `section` - Section to append to.
/// * `artifacts` - Discovered artifacts.
///
/// # Returns
///
/// The updated section.
fn add_artifacts(section: Section, artifacts: &[SecurityArtifact]) -> Section {
    if artifacts.is_empty() {
        return section;
    }
    let rows: Vec<Vec<String>> = artifacts.iter().map(artifact_row).collect();
    let table = text::table(&["Path", "Classification", "Detail", "Size"], rows);
    section
        .with(text::subheading("Security Artifacts and Credentials"))
        .with(Block::Table(table))
}

/// Adds identified custom executables table to the section.
///
/// # Arguments
///
/// * `section` - Section to append to.
/// * `executables` - Executable paths.
///
/// # Returns
///
/// The updated section.
fn add_executables(section: Section, executables: &[String]) -> Section {
    if executables.is_empty() {
        return section;
    }
    let rows: Vec<Vec<String>> = executables.iter().take(25).map(|e| exe_row(e)).collect();
    let table = text::table(&["Executable Path", "Format"], rows);
    section
        .with(text::subheading("Identified Custom Executables"))
        .with(Block::Table(table))
}

/// Appends sub-target capabilities table.
///
/// # Arguments
///
/// * `section` - Destination section.
/// * `sub` - Sub-target analysis.
///
/// # Returns
///
/// Updated section.
fn append_sub_caps(section: Section, sub: &Analysis) -> Section {
    if sub.capabilities.is_empty() {
        return section;
    }
    let rows: Vec<Vec<String>> = sub.capabilities.iter().map(cap_row).collect();
    let table = text::table(&["Capability", "Description", "APIs"], rows);
    section
        .with(text::subheading("Sub-Target Capabilities"))
        .with(Block::Table(table))
}

/// Appends sub-target key functions table.
///
/// # Arguments
///
/// * `section` - Destination section.
/// * `sub` - Sub-target analysis.
///
/// # Returns
///
/// Updated section.
fn append_sub_funcs(section: Section, sub: &Analysis) -> Section {
    if sub.key_functions.is_empty() {
        return section;
    }
    let rows: Vec<Vec<String>> = sub.key_functions.iter().take(5).map(func_row).collect();
    let table = text::table(&["Function", "Address", "Relevance", "Reasons"], rows);
    section
        .with(text::subheading("Sub-Target Key Functions"))
        .with(Block::Table(table))
}

/// Appends one sub-target analysis subsection.
///
/// # Arguments
///
/// * `section` - Base section.
/// * `sub` - Sub-target analysis.
///
/// # Returns
///
/// Updated section.
fn append_sub_target(section: Section, sub: &Analysis) -> Section {
    let heading = text::subheading(&format!("Sub-Target Analysis: {}", sub.name));
    let intro = Block::Paragraph(format!(
        "Contained binary {} analyzed natively with SLEIGH engine. Detected format: {}.",
        sub.name,
        sub.format.label()
    ));
    let with_caps = append_sub_caps(section.with(heading).with(intro), sub);
    append_sub_funcs(with_caps, sub)
}

/// Appends sub-target analysis subsections to the firmware section.
///
/// # Arguments
///
/// * `section` - Base section.
/// * `targets` - Analyzed sub-targets.
///
/// # Returns
///
/// List of resulting sections.
fn add_sub_targets(mut section: Section, targets: &[Analysis]) -> Vec<Section> {
    for target in targets {
        section = append_sub_target(section, target);
    }
    vec![section]
}

/// Builds the firmware filesystem and artifacts section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// Sections when firmware triage is present, otherwise empty.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let Some(fw) = &analysis.firmware else {
        return vec![];
    };
    let mut section = Section::new("Embedded Filesystem and Artifacts");
    section = add_summary(section, fw);
    section = add_artifacts(section, &fw.artifacts);
    section = add_executables(section, &fw.executables);
    add_sub_targets(section, &fw.sub_targets)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that analyses without firmware yield no sections.
    #[test]
    fn test_sections_empty_when_no_firmware() {
        let analysis = Analysis::default();
        assert_eq!(sections(&analysis).len(), 0);
    }

    /// Constructs an analysis populated with test firmware data.
    fn sample_firmware_analysis() -> Analysis {
        let mut triage = FirmwareTriage::default();
        triage.artifacts.push(SecurityArtifact::new("/etc/key", "Key", "RSA", 100));
        triage.executables.push("/bin/app".to_string());
        Analysis { firmware: Some(triage), ..Default::default() }
    }

    /// Verifies that firmware triage produces a populated section.
    #[test]
    fn test_sections_populated_with_firmware() {
        let analysis = sample_firmware_analysis();
        let result = sections(&analysis);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].heading, "Embedded Filesystem and Artifacts");
    }
}
