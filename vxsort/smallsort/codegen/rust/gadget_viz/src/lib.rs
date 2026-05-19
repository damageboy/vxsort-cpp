use std::fmt::Write as _;
use std::fs;
use std::path::Path;

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
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Json { line, source } => write!(f, "invalid JSON on line {line}: {source}"),
            Self::IndexOutOfRange { index, len } => {
                write!(f, "record index {index} is out of range for {len} records")
            }
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

pub fn render_records(records: &[(usize, Value)], format: OutputFormat, title: &str) -> String {
    match format {
        OutputFormat::Mermaid => render_mermaid_document(records),
        OutputFormat::Markdown => render_markdown_document(records, title),
        OutputFormat::Html => render_html_document(records, title),
    }
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

fn render_html_document(records: &[(usize, Value)], title: &str) -> String {
    let mut out = String::new();
    writeln!(out, "<!doctype html>").unwrap();
    writeln!(out, "<html><head><meta charset=\"utf-8\">").unwrap();
    writeln!(out, "<title>{}</title>", escape_html(title)).unwrap();
    writeln!(
        out,
        "<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:2rem;}}\
         section{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0;}}\
         .meta{{color:#333;font-size:0.9rem;line-height:1.35;}}\
         pre{{background:#f7f7f7;padding:0.75rem;overflow:auto;}}\
         summary{{cursor:pointer;}}</style>"
    )
    .unwrap();
    writeln!(
        out,
        "<script type=\"module\">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs'; mermaid.initialize({{startOnLoad:true,securityLevel:'loose'}});</script>"
    )
    .unwrap();
    writeln!(out, "</head><body>").unwrap();
    writeln!(out, "<h1>{}</h1>", escape_html(title)).unwrap();
    for (index, record) in records {
        writeln!(out, "<section>").unwrap();
        writeln!(out, "<h2>Record {index}</h2>").unwrap();
        writeln!(out, "<div class=\"meta\">{}</div>", metadata_html(record)).unwrap();
        writeln!(
            out,
            "<pre class=\"mermaid\">{}</pre>",
            escape_html(&render_record_mermaid(*index, record))
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
    out
}

pub fn render_record_mermaid(index: usize, record: &Value) -> String {
    let mut graph = MermaidGraph::default();
    graph.line("flowchart TD");
    graph.node(
        "record",
        "record #",
        &format!("record #{index}"),
        Shape::Rect,
    );
    graph.node("meta", "meta", &record_summary(record), Shape::Rect);
    graph.edge("record", "meta", None);

    match record.get("kind").and_then(Value::as_str) {
        Some("template") => render_template_record(&mut graph, record),
        Some("synthesized") => render_synthesized_record(&mut graph, record),
        _ => graph.node("unknown", "unknown", "unknown record kind", Shape::Rect),
    }

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

fn render_synthesized_record(graph: &mut MermaidGraph, record: &Value) {
    graph.node(
        "top_in",
        "top input",
        &format!(
            "top input\n{}",
            lane_array(record.pointer("/input_state/top"))
        ),
        Shape::Rect,
    );
    graph.node(
        "bottom_in",
        "bottom input",
        &format!(
            "bottom input\n{}",
            lane_array(record.pointer("/input_state/bottom"))
        ),
        Shape::Rect,
    );
    graph.node(
        "target",
        "target",
        &format!("target pairs\n{}", lane_array(record.get("target_pairs"))),
        Shape::Rect,
    );
    graph.edge("meta", "target", None);

    let top_out_source = render_instruction_chain(
        graph,
        "top",
        record.get("top_instructions").and_then(Value::as_array),
    );
    let bottom_out_source = render_instruction_chain(
        graph,
        "bottom",
        record.get("bottom_instructions").and_then(Value::as_array),
    );

    graph.node(
        "top_out",
        "top output",
        &format!(
            "top output\n{}",
            lane_array(record.pointer("/output_state/top"))
        ),
        Shape::Rect,
    );
    graph.node(
        "bottom_out",
        "bottom output",
        &format!(
            "bottom output\n{}",
            lane_array(record.pointer("/output_state/bottom"))
        ),
        Shape::Rect,
    );
    graph.edge(&top_out_source, "top_out", None);
    graph.edge(&bottom_out_source, "bottom_out", None);
}

fn render_instruction_chain(
    graph: &mut MermaidGraph,
    side: &str,
    instructions: Option<&Vec<Value>>,
) -> String {
    let input_id = if side == "bottom" {
        "bottom_in"
    } else {
        "top_in"
    };
    let Some(instructions) = instructions else {
        return input_id.to_owned();
    };
    if instructions.is_empty() {
        return input_id.to_owned();
    }

    let mut instruction_ids = Vec::new();
    for (idx, instruction) in instructions.iter().enumerate() {
        let id = format!("{side}_inst_{idx}");
        graph.node(&id, &id, &instruction_label(instruction), Shape::Rect);
        instruction_ids.push(id.clone());

        let mut saw_register_edge = false;
        if let Some(args) = instruction.get("args").and_then(Value::as_object) {
            for (key, value) in args {
                if let Some(src) = value
                    .as_str()
                    .and_then(|s| resolve_register_ref(s, input_id, &instruction_ids, idx))
                {
                    graph.edge(&src, &id, Some(key));
                    saw_register_edge = true;
                }
            }
        }
        if !saw_register_edge {
            let src = if idx == 0 {
                input_id.to_owned()
            } else {
                instruction_ids[idx - 1].clone()
            };
            graph.edge(&src, &id, None);
        }
    }

    instruction_ids
        .last()
        .cloned()
        .unwrap_or_else(|| input_id.to_owned())
}

fn resolve_register_ref(
    value: &str,
    current_input: &str,
    instruction_ids: &[String],
    current_idx: usize,
) -> Option<String> {
    match value {
        "top" => Some("top_in".to_owned()),
        "bottom" => Some("bottom_in".to_owned()),
        "prev" => Some(if current_idx == 0 {
            current_input.to_owned()
        } else {
            instruction_ids[current_idx - 1].clone()
        }),
        _ if value.starts_with("result_") => value[7..]
            .parse::<usize>()
            .ok()
            .and_then(|idx| instruction_ids.get(idx).cloned()),
        _ => None,
    }
}

fn instruction_label(instruction: &Value) -> String {
    let name = instruction
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("instruction");
    let mut lines = vec![name.to_owned()];
    if let Some(args) = instruction.get("args").and_then(Value::as_object) {
        for (key, value) in args {
            lines.push(format!("{key}={}", arg_label(value)));
        }
    }
    lines.join("\n")
}

fn arg_label(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_owned();
    }
    if let Some(n) = value.as_i64() {
        return n.to_string();
    }
    if let Some(n) = value.as_u64() {
        return n.to_string();
    }
    if value.get("kind").and_then(Value::as_str) == Some("bitvec") {
        let bits = value.get("bits").and_then(Value::as_u64).unwrap_or(0);
        let hex = value.get("hex").and_then(Value::as_str).unwrap_or("0x");
        return format!("{}b:{}", bits, shorten_hex(hex));
    }
    short_json(value)
}

fn symbolic_label(value: &Value) -> String {
    let role = value.get("role").and_then(Value::as_str).unwrap_or("sym");
    let var_id = value.get("var_id").and_then(Value::as_str).unwrap_or("v?");
    let bits = value.get("bits").and_then(Value::as_u64).unwrap_or(0);
    format!("{role}\n{var_id}\n{bits} bits")
}

fn record_summary(record: &Value) -> String {
    let kind = record.get("kind").and_then(Value::as_str).unwrap_or("?");
    let arch = record.get("arch").and_then(Value::as_str).unwrap_or("?");
    let dtype = record.get("dtype").and_then(Value::as_str).unwrap_or("?");
    let depth = record
        .get("gadget_depth")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let fixture = record.get("fixture").and_then(Value::as_str).unwrap_or("");
    if fixture.is_empty() {
        format!("{kind}\n{arch}/{dtype}\ndepth={depth}")
    } else {
        format!("{kind}\n{arch}/{dtype}\ndepth={depth}\nfixture={fixture}")
    }
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
    if let Some(input) = record.get("input_state") {
        parts.push(format!("- input_state: `{}`", compact_json(input)));
    }
    if let Some(output) = record.get("output_state") {
        parts.push(format!("- output_state: `{}`", compact_json(output)));
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

fn lane_array(value: Option<&Value>) -> String {
    value.map(compact_json).unwrap_or_else(|| "[]".to_owned())
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

fn shorten_hex(hex: &str) -> String {
    if hex.len() <= 18 {
        hex.to_owned()
    } else {
        format!("{}…{}", &hex[..10], &hex[hex.len() - 6..])
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

    fn synthesized_record() -> Value {
        serde_json::json!({
            "kind": "synthesized",
            "arch": "avx2",
            "dtype": "i64",
            "gadget_depth": 2,
            "fixture": "identity_pairs",
            "input_state": {"top": [0,1,2,3], "bottom": [4,5,6,7]},
            "target_pairs": [[0,4],[1,5],[2,6],[3,7]],
            "output_state": {"top": [0,1,2,3], "bottom": [4,5,6,7]},
            "top_instructions": [
                {"name":"_mm256_permute_pd", "args":{"a":"top", "imm8": 0}},
                {"name":"_mm256_shuffle_pd", "args":{"a":"prev", "b":"bottom", "imm8": 0}}
            ],
            "bottom_instructions": []
        })
    }

    #[test]
    fn synthesized_mermaid_shows_lanes_and_edges() {
        let diagram = render_record_mermaid(3, &synthesized_record());
        assert!(diagram.contains("record #3"));
        assert!(diagram.contains("top input"));
        assert!(diagram.contains("[0,1,2,3]"));
        assert!(diagram.contains("_mm256_shuffle_pd"));
        assert!(diagram.contains("-->|a| top_inst_1"));
    }

    #[test]
    fn index_selection_is_checked() {
        let records = vec![synthesized_record()];
        assert_eq!(select_records(&records, None).unwrap().len(), 1);
        assert_eq!(select_records(&records, Some(0)).unwrap().len(), 1);
        assert!(matches!(
            select_records(&records, Some(1)),
            Err(VizError::IndexOutOfRange { .. })
        ));
    }
}
