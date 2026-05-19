use std::collections::HashSet;
use std::fs;
use std::process::Command;

use gadget_synth::{
    Arch, DType, GadgetGraph, GadgetNode, GadgetSynthesizer, InputRef, InstructionArg,
    IntrinsicNode, MuxNode, MuxPruning, Symbolic, SynthesisOptions, VectorState,
};
use serde_json::{Value, json};

#[path = "../src/bin/dump_gadget_synth.rs"]
mod dump_gadget_synth_module;

fn dump_gadget_synth_bin() -> String {
    std::env::var("CARGO_BIN_EXE_dump_gadget_synth")
        .expect("dump_gadget_synth binary should be built by cargo test")
}

fn run_template_dump(output: &std::path::Path, intrinsic_filter: Option<&str>) {
    let mut command = Command::new(dump_gadget_synth_bin());
    command
        .args([
            "templates",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "1",
            "--output",
        ])
        .arg(output);
    if let Some(intrinsic_filter) = intrinsic_filter {
        command.args(["--intrinsic-filter", intrinsic_filter]);
    }
    let status = command.status().expect("dump_gadget_synth should run");

    assert!(status.success(), "dump_gadget_synth exited with {status}");
}

fn read_jsonl(path: &std::path::Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("dump should be readable")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("each JSONL record should parse"))
        .collect()
}

fn collect_symbolics<'a>(value: &'a Value, out: &mut Vec<&'a serde_json::Map<String, Value>>) {
    match value {
        Value::Object(map) => {
            if map.get("kind") == Some(&Value::String("symbolic".to_owned())) {
                out.push(map);
            }
            for child in map.values() {
                collect_symbolics(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_symbolics(child, out);
            }
        }
        _ => {}
    }
}

fn collect_intrinsic_names(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if map.get("kind") == Some(&Value::String("intrinsic".to_owned())) {
                out.push(
                    map.get("name")
                        .and_then(Value::as_str)
                        .expect("intrinsic node should have a string name")
                        .to_owned(),
                );
            }
            for child in map.values() {
                collect_intrinsic_names(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_intrinsic_names(child, out);
            }
        }
        _ => {}
    }
}

fn input_node(name: &'static str) -> GadgetNode {
    GadgetNode::Input(InputRef { name })
}

fn imm8_node(name: &str) -> GadgetNode {
    GadgetNode::Symbolic(Symbolic {
        name: name.to_owned(),
        bit_width: 8,
    })
}

fn permute_pd_node(input: GadgetNode, imm_name: &str) -> IntrinsicNode {
    IntrinsicNode::new("_mm256_permute_pd")
        .with_operand("a", input)
        .with_operand("imm8", imm8_node(imm_name))
}

#[test]
fn nested_graph_normalization_is_stable_for_single_to_single_shape() {
    let first = permute_pd_node(input_node("top"), "imm8_first_raw_name");
    let graph = GadgetGraph::new(
        Some(permute_pd_node(
            GadgetNode::Intrinsic(Box::new(first)),
            "imm8_second_raw_name",
        )),
        None,
    );

    let normalized = dump_gadget_synth_module::normalize_graph(&graph);
    assert_eq!(
        normalized,
        dump_gadget_synth_module::normalize_graph(&graph),
        "normalization should be stable for a hand-built nested graph"
    );
    assert_eq!(
        normalized,
        json!({
            "top": {
                "kind": "intrinsic",
                "name": "_mm256_permute_pd",
                "isomorphic_order": true,
                "operands": {
                    "a": {
                        "kind": "intrinsic",
                        "name": "_mm256_permute_pd",
                        "isomorphic_order": true,
                        "operands": {
                            "a": {"kind": "input", "name": "top"},
                            "imm8": {"kind": "symbolic", "role": "imm8", "bits": 8, "var_id": "v0"}
                        }
                    },
                    "imm8": {"kind": "symbolic", "role": "imm8", "bits": 8, "var_id": "v1"}
                }
            },
            "bottom": null
        })
    );
}

#[test]
fn mux_graph_normalization_includes_select_role_sources_and_pruning() {
    let graph = GadgetGraph::new(
        Some(
            IntrinsicNode::new("_mm256_permute_pd")
                .with_operand(
                    "a",
                    GadgetNode::Mux(MuxNode {
                        select: Symbolic {
                            name: "sel_raw_name".to_owned(),
                            bit_width: 2,
                        },
                        sources: vec![input_node("top"), input_node("bottom")],
                        pruning: Some(MuxPruning::ForceLast),
                    }),
                )
                .with_operand("imm8", imm8_node("imm8_raw_name")),
        ),
        None,
    );

    let normalized = dump_gadget_synth_module::normalize_graph(&graph);
    let mux = &normalized["top"]["operands"]["a"];
    assert_eq!(mux["kind"], "mux");
    assert_eq!(mux["pruning"], "force_last");
    assert_eq!(
        mux["select"],
        json!({"kind": "symbolic", "role": "mux_select", "bits": 2, "var_id": "v0"})
    );
    assert_eq!(
        mux["sources"],
        json!([
            {"kind": "input", "name": "top"},
            {"kind": "input", "name": "bottom"}
        ])
    );
}

#[test]
fn nested_intrinsic_synthesis_concretizes_parent_operand_as_prev() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);
    let input_state = VectorState::new(vec![0, 1, 2, 3], vec![4, 5, 6, 7]);
    let target_pairs = vec![(0, 4), (1, 5), (2, 6), (3, 7)];
    let first = permute_pd_node(input_node("top"), "imm8_first_eval");
    let graph = GadgetGraph::new(
        Some(permute_pd_node(
            GadgetNode::Intrinsic(Box::new(first)),
            "imm8_second_eval",
        )),
        None,
    );

    let results = synth
        .synthesize_graph(
            &graph,
            &input_state,
            &target_pairs,
            SynthesisOptions {
                max_unique_outputs: 1,
                allow_any_lane_order: true,
            },
        )
        .expect("nested intrinsic graph should synthesize");

    assert_eq!(results.len(), 1);
    let instructions = results[0].0.top_instructions();
    assert_eq!(instructions.len(), 2);
    assert_eq!(
        instructions[0].args().get("a"),
        Some(&InstructionArg::Input("top".to_owned()))
    );
    assert_eq!(
        instructions[1].args().get("a"),
        Some(&InstructionArg::Input("prev".to_owned()))
    );
}

#[test]
fn mux_evaluation_concretizes_selected_source() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);
    let input_state = VectorState::new(vec![0, 1, 2, 3], vec![4, 5, 6, 7]);
    let target_pairs = vec![(0, 4), (1, 5), (2, 6), (3, 7)];
    let graph = GadgetGraph::new(
        Some(
            IntrinsicNode::new("_mm256_permute_pd")
                .with_operand(
                    "a",
                    GadgetNode::Mux(MuxNode {
                        select: Symbolic {
                            name: "sel_eval".to_owned(),
                            bit_width: 2,
                        },
                        sources: vec![input_node("top"), input_node("bottom")],
                        pruning: None,
                    }),
                )
                .with_operand("imm8", imm8_node("imm8_mux_eval")),
        ),
        None,
    );

    let results = synth
        .synthesize_graph(
            &graph,
            &input_state,
            &target_pairs,
            SynthesisOptions {
                max_unique_outputs: 1,
                allow_any_lane_order: true,
            },
        )
        .expect("mux graph should synthesize");

    assert_eq!(results.len(), 1);
    let instructions = results[0].0.top_instructions();
    assert_eq!(instructions.len(), 1);
    assert_eq!(
        instructions[0].args().get("a"),
        Some(&InstructionArg::Input("top".to_owned()))
    );
}

#[test]
fn dump_schema_avx2_i64_depth1_templates_are_stable_and_normalized() {
    let temp_dir = std::env::temp_dir().join(format!(
        "gadget_synth_dump_schema_{}_{}",
        std::process::id(),
        "avx2_i64_depth1"
    ));
    fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let first = temp_dir.join("first.jsonl");
    let second = temp_dir.join("second.jsonl");

    run_template_dump(&first, None);
    run_template_dump(&second, None);

    let first_text = fs::read_to_string(&first).expect("first dump should be readable");
    let second_text = fs::read_to_string(&second).expect("second dump should be readable");
    assert_eq!(first_text, second_text, "template dump should be stable");
    assert!(!first_text.contains("0x"));
    assert!(!first_text.contains("object at"));
    assert!(!first_text.contains("imm8_permute"));
    assert!(!first_text.contains("ctrl_permutexvar"));

    let lines: Vec<&str> = first_text.lines().filter(|line| !line.is_empty()).collect();
    let mut sorted_lines = lines.clone();
    sorted_lines.sort_unstable();
    assert_eq!(lines, sorted_lines, "JSONL records should be sorted");
    let unique_lines: HashSet<&str> = lines.iter().copied().collect();
    assert_eq!(unique_lines.len(), lines.len(), "records should be unique");

    let records = read_jsonl(&first);
    assert_eq!(records.len(), 224);
    for record in &records {
        assert_eq!(record["kind"], "template");
        assert_eq!(record["arch"], "avx2");
        assert_eq!(record["dtype"], "i64");
        assert_eq!(record["gadget_depth"], 1);
        assert_eq!(record["tier"], "shallow");
        assert!(record.get("graph").is_some());

        let mut symbolics = Vec::new();
        collect_symbolics(record, &mut symbolics);
        for symbolic in symbolics {
            let keys: Vec<&str> = symbolic.keys().map(String::as_str).collect();
            assert_eq!(keys, ["bits", "kind", "role", "var_id"]);
            assert!(symbolic["bits"].is_u64());
            assert_eq!(symbolic["kind"], "symbolic");
            assert!(symbolic["role"].is_string());
            assert!(
                symbolic["var_id"].as_str().unwrap().starts_with('v'),
                "symbolic var_id should be normalized"
            );
        }
    }
}

#[test]
fn dump_templates_rejects_unsupported_gadget_depth_without_panic() {
    let temp_dir = std::env::temp_dir().join(format!(
        "gadget_synth_dump_schema_{}_{}",
        std::process::id(),
        "unsupported_depth"
    ));
    fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let output_path = temp_dir.join("unsupported.jsonl");
    let _ = fs::remove_file(&output_path);

    let output = Command::new(dump_gadget_synth_bin())
        .args([
            "templates",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "3",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("dump_gadget_synth should run");

    assert!(!output.status.success(), "unsupported depth should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("gadget depths above 2 are not supported yet"),
        "unexpected stderr: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "unexpected panic: {stderr}");
    assert!(
        !output_path.exists(),
        "unsupported depth should not create an output file"
    );
}

#[test]
fn dump_templates_accepts_intrinsic_filter_and_filters_avx2_i64_depth1() {
    let temp_dir = std::env::temp_dir().join(format!(
        "gadget_synth_dump_schema_{}_{}",
        std::process::id(),
        "avx2_i64_depth1_filter"
    ));
    fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let unfiltered = temp_dir.join("unfiltered.jsonl");
    let filtered = temp_dir.join("filtered.jsonl");

    run_template_dump(&unfiltered, None);
    run_template_dump(&filtered, Some("_mm256_permute_pd"));

    let unfiltered_records = read_jsonl(&unfiltered);
    let filtered_records = read_jsonl(&filtered);
    assert_eq!(unfiltered_records.len(), 224);
    assert_eq!(filtered_records.len(), 3);
    assert!(filtered_records.len() < unfiltered_records.len());

    for record in &filtered_records {
        assert_eq!(record["kind"], "template");
        assert_eq!(record["arch"], "avx2");
        assert_eq!(record["dtype"], "i64");
        assert_eq!(record["gadget_depth"], 1);
        assert!(record.get("intrinsic_filter").is_none());

        let mut names = Vec::new();
        collect_intrinsic_names(&record["graph"], &mut names);
        assert!(!names.is_empty());
        assert!(
            names.iter().all(|name| name == "_mm256_permute_pd"),
            "filtered records should only use _mm256_permute_pd: {names:?}"
        );
    }
}

#[test]
fn dump_rejects_synthesized_mode() {
    let temp_dir = std::env::temp_dir().join(format!(
        "gadget_synth_dump_schema_{}_{}",
        std::process::id(),
        "reject_synthesized"
    ));
    fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let output_path = temp_dir.join("synthesized.jsonl");

    let output = Command::new(dump_gadget_synth_bin())
        .args([
            "synthesized",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("dump_gadget_synth should run");

    assert!(!output.status.success(), "synthesized mode should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("templates"), "unexpected stderr: {stderr}");
    assert!(!output_path.exists());
}
