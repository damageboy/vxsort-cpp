use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};
use uica_data::{
    DATAPACK_MANIFEST_SCHEMA_VERSION, DATAPACK_SCHEMA_VERSION, DataPack, InstructionRecord,
    PerfRecord, UIPACK_VERSION, encode_uipack, read_uipack_header,
};
use vxsort_codegen::json_exporter::{
    SolutionJsonMetadata, solution_json_for_assigned_paths, solution_json_for_paths,
};
use vxsort_codegen::scoring::AssignedPath;
use vxsort_codegen::transition_table::{GadgetIndex, TransitionTable};
use vxsort_codegen::{ArchArg, DTypeArg, EstimateOptions, estimate_solution_json};

fn state(top: &[u64], bottom: &[u64]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn inst(name: &'static str, args: &[(&'static str, InstructionArg)]) -> InstructionSpec {
    InstructionSpec::new(name, args.iter().cloned().collect::<BTreeMap<_, _>>())
}

fn metadata() -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: true,
        arch: ArchArg::Avx2,
        dtype: DTypeArg::I64,
        num_vecs: 2,
    }
}

fn permute_gadget(imm8: u64) -> PermutationGadget {
    PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(imm8)),
            ],
        )],
        Vec::new(),
    )
}

fn double_permute_gadget() -> PermutationGadget {
    PermutationGadget::new(
        vec![
            inst(
                "_mm256_permute4x64_epi64",
                &[
                    ("a", InstructionArg::Input("top".to_owned())),
                    ("imm8", InstructionArg::U64(0x4e)),
                ],
            ),
            inst(
                "_mm256_permute4x64_epi64",
                &[
                    ("a", InstructionArg::Input("prev".to_owned())),
                    ("imm8", InstructionArg::U64(0x1b)),
                ],
            ),
        ],
        Vec::new(),
    )
}

fn unsupported_gadget() -> PermutationGadget {
    PermutationGadget::new(
        vec![inst(
            "_mm256_unknown_epi64",
            &[("a", InstructionArg::Input("top".to_owned()))],
        )],
        Vec::new(),
    )
}

fn assigned_json_with_two_paths() -> serde_json::Value {
    assigned_json_from_gadgets(vec![permute_gadget(0x4e), permute_gadget(0x1b)], &[0, 1])
}

fn assigned_json_for_ranking() -> serde_json::Value {
    assigned_json_from_gadgets(vec![permute_gadget(0x4e), double_permute_gadget()], &[0, 1])
}

fn assigned_json_with_unsupported_path() -> serde_json::Value {
    assigned_json_from_gadgets(vec![unsupported_gadget()], &[0])
}

fn assigned_json_selecting_second_gadget() -> serde_json::Value {
    let json = assigned_json_from_gadgets(
        vec![
            PermutationGadget::new(Vec::new(), Vec::new()),
            permute_gadget(0x4e),
        ],
        &[0, 1],
    );

    serde_json::json!({
        "natural_order": json["natural_order"],
        "vector_machine": json["vector_machine"],
        "primitive_type": json["primitive_type"],
        "num_vecs": json["num_vecs"],
        "roots": json["roots"],
        "nodes": json["nodes"],
        "paths": [json["paths"][1].clone()]
    })
}

fn roots_nodes_json_with_one_path() -> serde_json::Value {
    let input = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let output = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, permute_gadget(0x4e));
    let path = table
        .complete_path_from_zero_based_steps(&[(0, input.as_tuple(), output.as_tuple())])
        .expect("path should resolve");

    solution_json_for_paths(&metadata(), &table, &[path])
}

fn assigned_json_from_gadgets(
    gadgets: Vec<PermutationGadget>,
    selected_indices: &[u16],
) -> serde_json::Value {
    let input = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let output = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let mut table = TransitionTable::new(1);
    for gadget in gadgets {
        table.add_transition(0, &input, &output, gadget);
    }
    let path = table
        .complete_path_from_zero_based_steps(&[(0, input.as_tuple(), output.as_tuple())])
        .expect("path should resolve");
    let paths = selected_indices
        .iter()
        .map(|index| AssignedPath::new(path.clone(), vec![GadgetIndex(*index)]))
        .collect::<Vec<_>>();
    solution_json_for_assigned_paths(&metadata(), &table, &paths)
}

#[test]
fn selection_top_k_keeps_original_path_indices() {
    let input_path = write_json_fixture("selection-top-k", assigned_json_with_two_paths());
    let output_dir = temp_dir("selection-top-k-output");

    let report = estimate_solution_json(
        &input_path,
        estimate_options(fixture_uica_data_dir("selection-top-k-data"), output_dir),
    )
    .expect("estimate should import JSON");

    assert_eq!(report.imported_path_count, 2);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].path_index, 0);

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

#[test]
fn selection_uses_assigned_gadget_choice() {
    let input_path = write_json_fixture(
        "assigned-selection",
        assigned_json_selecting_second_gadget(),
    );
    let output_dir = temp_dir("assigned-selection-output");

    let report = estimate_solution_json(
        &input_path,
        estimate_options(fixture_uica_data_dir("assigned-selection-data"), output_dir),
    )
    .expect("estimate should import assigned JSON");

    assert_eq!(report.imported_path_count, 1);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].path_index, 0);
    assert_eq!(report.results[0].instruction_count, 1);

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

#[test]
fn selection_lowers_roots_nodes_json_paths() {
    let input_path = write_json_fixture("roots-nodes", roots_nodes_json_with_one_path());
    let output_dir = temp_dir("roots-nodes-output");

    let report = estimate_solution_json(
        &input_path,
        estimate_options(fixture_uica_data_dir("roots-nodes-data"), output_dir),
    )
    .expect("estimate should import roots/nodes JSON");

    assert_eq!(report.imported_path_count, 1);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].path_index, 0);
    assert_eq!(report.results[0].instruction_count, 1);

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

#[test]
fn simulation_writes_trace_html_for_successful_paths() {
    let input_path = write_json_fixture("trace-html", roots_nodes_json_with_one_path());
    let output_dir = temp_dir("trace-html-output");

    let report = estimate_solution_json(
        &input_path,
        EstimateOptions {
            write_traces: true,
            ..estimate_options(fixture_uica_data_dir("trace-html-data"), output_dir)
        },
    )
    .expect("estimate should run uiCA");

    let result = &report.results[0];
    assert!(
        result
            .throughput
            .expect("throughput should be present")
            .is_finite()
    );
    let trace_path = result
        .trace_path
        .as_ref()
        .expect("trace path should be recorded");
    let html = fs::read_to_string(trace_path).expect("trace html should be written");
    assert!(html.contains("Execution Trace"));
    assert_eq!(
        trace_path.file_name().and_then(|name| name.to_str()),
        Some("solution_000_trace.html")
    );

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

#[test]
fn simulation_ranks_successful_paths_by_uica_throughput() {
    let input_path = write_json_fixture("ranking", assigned_json_for_ranking());
    let output_dir = temp_dir("ranking-output");

    let report = estimate_solution_json(
        &input_path,
        EstimateOptions {
            top_k: None,
            ..estimate_options(fixture_uica_data_dir("ranking-data"), output_dir)
        },
    )
    .expect("estimate should run uiCA");

    assert_eq!(report.results.len(), 2);
    assert_eq!(report.results[0].path_index, 0);
    assert_eq!(report.results[0].rank, Some(1));
    assert_eq!(report.results[1].path_index, 1);
    assert_eq!(report.results[1].rank, Some(2));
    assert!(
        report.results[0].throughput.unwrap() <= report.results[1].throughput.unwrap(),
        "expected sorted throughput rows: {:?}",
        report.results
    );

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

#[test]
fn simulation_keeps_error_row_for_unsupported_instruction() {
    let input_path = write_json_fixture("unsupported", assigned_json_with_unsupported_path());
    let output_dir = temp_dir("unsupported-output");

    let report = estimate_solution_json(
        &input_path,
        estimate_options(fixture_uica_data_dir("unsupported-data"), output_dir),
    )
    .expect("estimate should keep per-path errors in the report");

    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].path_index, 0);
    assert_eq!(report.results[0].throughput, None);
    assert!(
        report.results[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("UNSUPPORTED"),
        "unexpected error row: {:?}",
        report.results[0]
    );

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(report.output_dir);
}

fn estimate_options(data_dir: PathBuf, output_dir: PathBuf) -> EstimateOptions {
    EstimateOptions {
        target_cpu: "SKL".to_owned(),
        uica_data_dir: data_dir,
        top_k: Some(1),
        output_dir: Some(output_dir),
        write_traces: false,
    }
}

fn write_json_fixture(name: &str, json: serde_json::Value) -> PathBuf {
    let path = temp_path(name, "json");
    fs::write(
        &path,
        serde_json::to_vec(&json).expect("fixture should serialize"),
    )
    .expect("fixture should be written");
    path
}

fn fixture_uica_data_dir(name: &str) -> PathBuf {
    let dir = temp_dir(name);
    fs::create_dir_all(dir.join("arch")).expect("fixture arch dir should be created");
    let pack = DataPack {
        schema_version: DATAPACK_SCHEMA_VERSION.to_owned(),
        all_ports: ports(),
        alu_ports: ports(),
        instructions: vec![record("VPERMQ_YMMqq_YMMqq_IMMb", "VPERMQ (YMM, YMM, I8)")],
    };
    let bytes = encode_uipack(&pack, "SKL").expect("fixture uipack should encode");
    let header = read_uipack_header(&bytes).expect("fixture uipack header should decode");
    fs::write(dir.join("arch/SKL.uipack"), &bytes).expect("fixture pack should be written");
    fs::write(
        dir.join("manifest.json"),
        format!(
            r#"{{
  "schema_version": "{}",
  "uipack_version": {},
  "architectures": {{
    "SKL": {{
      "path": "arch/SKL.uipack",
      "size": {},
      "checksum_kind": "fnv1a64",
      "checksum": "{:016x}",
      "record_count": {}
    }}
  }}
}}"#,
            DATAPACK_MANIFEST_SCHEMA_VERSION,
            UIPACK_VERSION,
            header.file_len,
            header.checksum,
            header.records_count
        ),
    )
    .expect("fixture manifest should be written");
    dir
}

fn record(iform: &str, string: &str) -> InstructionRecord {
    InstructionRecord {
        arch: "SKL".to_owned(),
        iform: iform.to_owned(),
        string: string.to_owned(),
        all_ports: ports(),
        alu_ports: ports(),
        locked: false,
        xml_attrs: BTreeMap::new(),
        imm_zero: false,
        perf: PerfRecord {
            operands: vec![],
            latencies: vec![],
            uops: 1,
            retire_slots: 1,
            uops_mite: 1,
            uops_ms: 0,
            tp: Some(1.0),
            ports: BTreeMap::from([("5".to_owned(), 1)]),
            variants: BTreeMap::new(),
            div_cycles: 0,
            may_be_eliminated: false,
            complex_decoder: false,
            n_available_simple_decoders: 0,
            lcp_stall: false,
            implicit_rsp_change: 0,
            can_be_used_by_lsd: true,
            cannot_be_in_dsb_due_to_jcc_erratum: false,
            no_micro_fusion: false,
            no_macro_fusion: false,
            macro_fusible_with: vec![],
        },
    }
}

fn ports() -> Vec<String> {
    ["0", "1", "5", "6"]
        .iter()
        .map(|port| (*port).to_owned())
        .collect()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vxsort-uica-estimator-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn temp_path(name: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vxsort-uica-estimator-{name}-{}.{}",
        std::process::id(),
        extension
    ))
}
