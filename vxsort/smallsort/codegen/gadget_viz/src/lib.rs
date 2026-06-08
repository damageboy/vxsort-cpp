use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use mermaid_rs_renderer::render as render_mermaid_to_svg;
use rayon::prelude::*;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputFormat {
    Mermaid,
    Markdown,
    Html,
}

#[derive(Debug)]
pub enum VizError {
    Io(std::io::Error),
    Json {
        line: usize,
        source: serde_json::Error,
    },
    IndexOutOfRange {
        index: usize,
        len: usize,
    },
    NonTemplateRecord {
        index: usize,
        kind: String,
    },
    MermaidRender {
        source: String,
    },
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Json { line, source } => write!(f, "invalid JSON on line {line}: {source}"),
            Self::IndexOutOfRange { index, len } => {
                write!(f, "record index {index} is out of range for {len} records")
            }
            Self::NonTemplateRecord { index, kind } => {
                write!(
                    f,
                    "record {index} has kind '{kind}', but gadget_viz only renders template records"
                )
            }
            Self::MermaidRender { source } => write!(f, "failed to render Mermaid SVG: {source}"),
        }
    }
}

impl std::error::Error for VizError {}

impl From<std::io::Error> for VizError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, VizError>;

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let text = fs::read_to_string(path)?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(idx, line)| {
            serde_json::from_str(line).map_err(|source| VizError::Json {
                line: idx + 1,
                source,
            })
        })
        .collect()
}

pub fn select_records(records: &[Value], index: Option<usize>) -> Result<Vec<(usize, Value)>> {
    match index {
        Some(index) => records
            .get(index)
            .cloned()
            .map(|record| vec![(index, record)])
            .ok_or(VizError::IndexOutOfRange {
                index,
                len: records.len(),
            }),
        None => Ok(records.iter().cloned().enumerate().collect()),
    }
}

pub fn render_records(
    records: &[(usize, Value)],
    format: OutputFormat,
    title: &str,
) -> Result<String> {
    ensure_template_records(records)?;
    match format {
        OutputFormat::Mermaid => Ok(render_mermaid_document(records)),
        OutputFormat::Markdown => Ok(render_markdown_document(records, title)),
        OutputFormat::Html => render_html_document(records, title),
    }
}

fn ensure_template_records(records: &[(usize, Value)]) -> Result<()> {
    for (index, record) in records {
        let kind = record
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if kind != "template" {
            return Err(VizError::NonTemplateRecord {
                index: *index,
                kind: kind.to_owned(),
            });
        }
    }
    Ok(())
}

fn render_mermaid_document(records: &[(usize, Value)]) -> String {
    records
        .iter()
        .map(|(index, record)| render_record_mermaid(*index, record))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_markdown_document(records: &[(usize, Value)], title: &str) -> String {
    let mut out = String::new();
    writeln!(out, "# {title}\n").unwrap();
    for (index, record) in records {
        writeln!(out, "## Record {index}\n").unwrap();
        writeln!(out, "{}\n", metadata_markdown(record)).unwrap();
        writeln!(out, "```mermaid").unwrap();
        writeln!(out, "{}", render_record_mermaid(*index, record)).unwrap();
        writeln!(out, "```\n").unwrap();
        writeln!(out, "<details><summary>JSON</summary>\n\n```json").unwrap();
        writeln!(out, "{}", serde_json::to_string_pretty(record).unwrap()).unwrap();
        writeln!(out, "```\n</details>\n").unwrap();
    }
    out
}

fn render_html_document(records: &[(usize, Value)], title: &str) -> Result<String> {
    let mut out = String::new();
    writeln!(out, "<!doctype html>").unwrap();
    writeln!(out, "<html><head><meta charset=\"utf-8\">").unwrap();
    writeln!(out, "<title>{}</title>", escape_html(title)).unwrap();
    writeln!(
        out,
        "<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:2rem;}}\
         section{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0;}}\
         .meta{{color:#333;font-size:0.9rem;line-height:1.35;}}\
         .diagram img{{max-width:100%;height:auto;}}\
         pre{{background:#f7f7f7;padding:0.75rem;overflow:auto;}}\
         summary{{cursor:pointer;}}</style>"
    )
    .unwrap();
    writeln!(out, "</head><body>").unwrap();
    writeln!(out, "<h1>{}</h1>", escape_html(title)).unwrap();

    let rendered_diagrams = records
        .par_iter()
        .map(|(index, record)| {
            let mermaid = render_record_mermaid(*index, record);
            let svg_base64 = render_mermaid_svg_base64(&mermaid)?;
            Ok((*index, mermaid, svg_base64))
        })
        .collect::<Result<Vec<_>>>()?;

    for ((index, record), (diagram_index, mermaid, svg_base64)) in
        records.iter().zip(rendered_diagrams)
    {
        debug_assert_eq!(*index, diagram_index);
        writeln!(out, "<section>").unwrap();
        writeln!(out, "<h2>Record {index}</h2>").unwrap();
        writeln!(out, "<div class=\"meta\">{}</div>", metadata_html(record)).unwrap();
        writeln!(
            out,
            "<div class=\"diagram\"><img src=\"data:image/svg+xml;base64,{svg_base64}\" alt=\"Record {index} gadget diagram\"></div>"
        )
        .unwrap();
        writeln!(
            out,
            "<details><summary>Mermaid source</summary><pre>{}</pre></details>",
            escape_html(&mermaid)
        )
        .unwrap();
        writeln!(
            out,
            "<details><summary>JSON</summary><pre>{}</pre></details>",
            escape_html(&serde_json::to_string_pretty(record).unwrap())
        )
        .unwrap();
        writeln!(out, "</section>").unwrap();
    }
    writeln!(out, "</body></html>").unwrap();
    Ok(out)
}

fn render_mermaid_svg_base64(mermaid: &str) -> Result<String> {
    let svg = render_mermaid_to_svg(mermaid).map_err(|source| VizError::MermaidRender {
        source: source.to_string(),
    })?;
    Ok(BASE64_STANDARD.encode(svg.as_bytes()))
}

pub fn render_record_mermaid(_index: usize, record: &Value) -> String {
    let mut graph = MermaidGraph::default();
    graph.line("flowchart TD");
    render_template_record(&mut graph, record);
    graph.finish()
}

fn render_template_record(graph: &mut MermaidGraph, record: &Value) {
    graph.node("top_in", "top", "top input", Shape::Rect);
    graph.node("bottom_in", "bottom", "bottom input", Shape::Rect);
    if let Some(top) = record.pointer("/graph/top") {
        let top_id = render_template_node(graph, top, "top_graph");
        graph.edge(&top_id, "top_out", None);
    } else {
        graph.edge("top_in", "top_out", None);
    }
    if let Some(bottom) = record.pointer("/graph/bottom") {
        let bottom_id = render_template_node(graph, bottom, "bottom_graph");
        graph.edge(&bottom_id, "bottom_out", None);
    } else {
        graph.edge("bottom_in", "bottom_out", None);
    }
    graph.node("top_out", "top out", "top output", Shape::Rect);
    graph.node("bottom_out", "bottom out", "bottom output", Shape::Rect);
}

fn render_template_node(graph: &mut MermaidGraph, value: &Value, hint: &str) -> String {
    if value.is_null() {
        return match hint.contains("bottom") {
            true => "bottom_in".to_owned(),
            false => "top_in".to_owned(),
        };
    }

    match value.get("kind").and_then(Value::as_str) {
        Some("input") => match value.get("name").and_then(Value::as_str) {
            Some("bottom") => "bottom_in".to_owned(),
            _ => "top_in".to_owned(),
        },
        Some("symbolic") => {
            let id = graph.fresh("sym");
            graph.node(&id, "symbolic", &symbolic_label(value), Shape::Rect);
            id
        }
        Some("mux") => {
            let id = graph.fresh("mux");
            let pruning = value
                .get("pruning")
                .and_then(Value::as_str)
                .unwrap_or("none");
            graph.node(
                &id,
                "mux",
                &format!("mux\npruning={pruning}"),
                Shape::Diamond,
            );
            if let Some(select) = value.get("select") {
                let select_id = render_template_node(graph, select, "mux_select");
                graph.edge(&select_id, &id, Some("select"));
            }
            if let Some(sources) = value.get("sources").and_then(Value::as_array) {
                for (idx, source) in sources.iter().enumerate() {
                    let src_id = render_template_node(graph, source, &format!("mux_src_{idx}"));
                    graph.edge(&src_id, &id, Some(&format!("src{idx}")));
                }
            }
            id
        }
        Some("intrinsic") => {
            let id = graph.fresh("inst");
            let name = value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("intrinsic");
            graph.node(&id, name, &format!("{name}\ntemplate"), Shape::Rect);
            if let Some(operands) = value.get("operands").and_then(Value::as_object) {
                for (key, operand) in operands {
                    let operand_id = render_template_node(graph, operand, key);
                    graph.edge(&operand_id, &id, Some(key));
                }
            }
            id
        }
        _ => {
            let id = graph.fresh("value");
            graph.node(&id, "value", &short_json(value), Shape::Rect);
            id
        }
    }
}

fn symbolic_label(value: &Value) -> String {
    let role = value.get("role").and_then(Value::as_str).unwrap_or("sym");
    let var_id = value.get("var_id").and_then(Value::as_str).unwrap_or("v?");
    let bits = value.get("bits").and_then(Value::as_u64).unwrap_or(0);
    format!("{role}\n{var_id}\n{bits} bits")
}

fn metadata_markdown(record: &Value) -> String {
    let mut parts = vec![format!(
        "- kind: `{}`",
        record.get("kind").and_then(Value::as_str).unwrap_or("?")
    )];
    for key in ["arch", "dtype", "gadget_depth", "fixture"] {
        if let Some(value) = record.get(key) {
            parts.push(format!("- {key}: `{}`", scalar_to_string(value)));
        }
    }
    parts.join("\n")
}

fn metadata_html(record: &Value) -> String {
    metadata_markdown(record)
        .lines()
        .map(|line| escape_html(line.trim_start_matches("- ")))
        .collect::<Vec<_>>()
        .join("<br>")
}

fn scalar_to_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "?".to_owned())
}

fn short_json(value: &Value) -> String {
    let text = compact_json(value);
    if text.len() > 80 {
        format!("{}…", &text[..80])
    } else {
        text
    }
}

#[derive(Default)]
struct MermaidGraph {
    next_id: usize,
    lines: Vec<String>,
}

impl MermaidGraph {
    fn fresh(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}_{}", self.next_id);
        self.next_id += 1;
        id
    }

    fn line(&mut self, line: &str) {
        self.lines.push(line.to_owned());
    }

    fn node(&mut self, id: &str, _fallback: &str, label: &str, shape: Shape) {
        let label = label
            .lines()
            .map(escape_mermaid_label_part)
            .collect::<Vec<_>>()
            .join("<br/>");
        let line = match shape {
            Shape::Rect => format!("  {id}[\"{label}\"]"),
            Shape::Diamond => format!("  {id}{{\"{label}\"}}"),
        };
        self.lines.push(line);
    }

    fn edge(&mut self, from: &str, to: &str, label: Option<&str>) {
        if let Some(label) = label {
            self.lines.push(format!(
                "  {from} -->|{}| {to}",
                escape_mermaid_edge_label(label)
            ));
        } else {
            self.lines.push(format!("  {from} --> {to}"));
        }
    }

    fn finish(self) -> String {
        self.lines.join("\n")
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Rect,
    Diamond,
}

fn escape_mermaid_label_part(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_mermaid_edge_label(value: &str) -> String {
    value.replace('|', "∣").replace('\n', " ")
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template_record() -> Value {
        serde_json::json!({
            "kind": "template",
            "arch": "avx2",
            "dtype": "i64",
            "gadget_depth": 2,
            "tier": "deep",
            "graph": {
                "top": {
                    "kind": "intrinsic",
                    "name": "_mm256_shuffle_pd",
                    "isomorphic_order": true,
                    "operands": {
                        "a": {"kind": "input", "name": "top"},
                        "b": {"kind": "input", "name": "bottom"},
                        "imm8": {"kind": "symbolic", "role": "imm8", "bits": 8, "var_id": "v0"}
                    }
                },
                "bottom": null
            }
        })
    }

    fn synthesized_record() -> Value {
        serde_json::json!({"kind": "synthesized"})
    }

    #[test]
    fn template_mermaid_shows_only_template_graph() {
        let diagram = render_record_mermaid(3, &template_record());
        assert!(!diagram.contains("record #3"));
        assert!(diagram.contains("top input"));
        assert!(diagram.contains("bottom input"));
        assert!(diagram.contains("_mm256_shuffle_pd"));
        assert!(diagram.contains("imm8"));
    }

    #[test]
    fn render_records_rejects_synthesized_records() {
        let records = vec![(7, synthesized_record())];
        assert!(matches!(
            render_records(&records, OutputFormat::Mermaid, "demo"),
            Err(VizError::NonTemplateRecord { index: 7, .. })
        ));
    }

    #[test]
    fn index_selection_is_checked() {
        let records = vec![template_record()];
        assert_eq!(select_records(&records, None).unwrap().len(), 1);
        assert_eq!(select_records(&records, Some(0)).unwrap().len(), 1);
        assert!(matches!(
            select_records(&records, Some(1)),
            Err(VizError::IndexOutOfRange { .. })
        ));
    }
}
