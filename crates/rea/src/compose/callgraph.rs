//! Call Graph Summary Section
//!
//! **File:** `callgraph.rs`
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
    insights::{call_graph, CallGraph},
    Analysis,
};

// Import dependencies from rea_report
use rea_report::{Block, Section};

/// Minimum functions before a call-graph section is shown.
const MIN_FUNCTIONS: usize = 5;

/// Builds the call-graph summary section.
///
/// # Arguments
///
/// * `analysis` - Completed analysis.
///
/// # Returns
///
/// The section, or nothing when no meaningful graph exists.
pub fn sections(analysis: &Analysis) -> Vec<Section> {
    let Some(graph) = call_graph(analysis) else {
        return Vec::new();
    };
    if graph.functions < MIN_FUNCTIONS {
        return Vec::new();
    }
    vec![Section::new("Call Graph").with(Block::Paragraph(story(&graph)))]
}

/// Composes the call-graph narrative.
///
/// # Arguments
///
/// * `graph` - Recovered call-graph summary.
///
/// # Returns
///
/// The narrative text.
fn story(graph: &CallGraph) -> String {
    let mut parts = Vec::new();
    push_head(&mut parts, graph);
    push_entry(&mut parts, graph);
    push_hubs(&mut parts, graph);
    push_tail(&mut parts, graph);
    format!("Recovered call graph: {}.", parts.join("; "))
}

/// Appends the call-graph size summary.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `graph` - Recovered call-graph summary.
fn push_head(parts: &mut Vec<String>, graph: &CallGraph) {
    let text = format!(
        "{} functions and {} internal call edges",
        graph.functions, graph.edges
    );
    parts.push(text);
}

/// Appends the entry-point description.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `graph` - Recovered call-graph summary.
fn push_entry(parts: &mut Vec<String>, graph: &CallGraph) {
    if !graph.entry.is_empty() {
        parts.push(format!("entry `{}`", graph.entry));
    }
}

/// Appends the most-called functions.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `graph` - Recovered call-graph summary.
fn push_hubs(parts: &mut Vec<String>, graph: &CallGraph) {
    if graph.hubs.is_empty() {
        return;
    }
    let items: Vec<String> =
        graph.hubs.iter().map(|(n, c)| format!("`{n}` ({c} callers)")).collect();
    parts.push(format!("most-called {}", items.join(", ")));
}

/// Appends the leaf and recursion summary.
///
/// # Arguments
///
/// * `parts` - Accumulated note parts.
/// * `graph` - Recovered call-graph summary.
fn push_tail(parts: &mut Vec<String>, graph: &CallGraph) {
    parts.push(format!("{} leaf function(s)", graph.leaves));
    if !graph.recursive.is_empty() {
        parts.push(format!("recursive: {}", graph.recursive.join(", ")));
    }
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that a plain target yields no call-graph section.
    #[test]
    fn test_sections_empty() {
        let analysis = Analysis::default();
        assert_eq!(sections(&analysis).len(), 0);
    }

    /// Builds a sample call-graph summary.
    ///
    /// # Returns
    ///
    /// A populated summary.
    fn sample() -> CallGraph {
        CallGraph {
            functions: 10,
            entry: "main".into(),
            hubs: vec![("sub_1".into(), 3)],
            ..Default::default()
        }
    }

    /// Verifies that the narrative mentions size, entry, and hubs.
    #[test]
    fn test_story_mentions_hubs() {
        let text = story(&sample());
        assert!(text.contains("10 functions") && text.contains("main") && text.contains("sub_1"));
    }
}
