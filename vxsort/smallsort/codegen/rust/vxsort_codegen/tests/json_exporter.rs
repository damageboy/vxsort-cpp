use std::collections::BTreeMap;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};
use serde_json::Value;
use vxsort_codegen::json_exporter::{
    SolutionJsonMetadata, solution_json_for_assigned_paths, solution_json_for_paths,
};
use vxsort_codegen::scoring::AssignedPath;
use vxsort_codegen::transition_table::{GadgetIndex, TransitionTable};
use vxsort_codegen::{ArchArg, DTypeArg};

fn state(top: &[u64], bottom: &[u64]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn inst(name: &'static str, args: &[(&'static str, InstructionArg)]) -> InstructionSpec {
    InstructionSpec::new(name, args.iter().cloned().collect::<BTreeMap<_, _>>())
}

fn metadata() -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: false,
        arch: ArchArg::Avx2,
        dtype: DTypeArg::I64,
        num_vecs: 2,
    }
}

#[test]
fn solution_json_matches_python_roots_nodes_schema() {
    let input = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let output = state(&[0, 1, 4, 5], &[2, 3, 6, 7]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = table
        .complete_path_from_zero_based_steps(&[(0, input.as_tuple(), output.as_tuple())])
        .expect("path should resolve");

    let json = solution_json_for_paths(&metadata(), &table, &[path]);

    assert_eq!(json["natural_order"], Value::Bool(false));
    assert_eq!(json["vector_machine"], Value::String("AVX2".to_owned()));
    assert_eq!(json["primitive_type"], Value::String("i64".to_owned()));
    assert_eq!(json["num_vecs"], Value::Number(2.into()));
    assert_eq!(
        json["roots"].as_array().expect("roots should be an array"),
        &[Value::String("n0".to_owned())]
    );

    let node = &json["nodes"]["n0"];
    assert_eq!(node["stage"], Value::Number(0.into()));
    assert_eq!(node["input_state"]["top"], serde_json::json!([1, 3, 5, 7]));
    assert_eq!(
        node["output_state"]["bottom"],
        serde_json::json!([3, 4, 7, 8])
    );
    assert_eq!(node["children"], serde_json::json!([]));
    assert_eq!(node["gadget_count"], Value::Number(1.into()));

    let gadget_json = &node["gadgets"][0];
    assert_eq!(
        gadget_json["top_instructions"][0]["name"],
        "_mm256_permute4x64_epi64"
    );
    assert_eq!(gadget_json["top_instructions"][0]["args"]["a"], "top");
    assert_eq!(gadget_json["top_instructions"][0]["args"]["imm8"], 0x4e);
    assert_eq!(gadget_json["bottom_instructions"], serde_json::json!([]));
    assert_eq!(
        gadget_json["unified_instructions"][0]["name"],
        "_mm256_permute4x64_epi64"
    );
    assert_eq!(gadget_json["top_output_index"], 0);
    assert_eq!(gadget_json["bottom_output_index"], -1);
}

#[test]
fn solution_json_can_represent_empty_runs() {
    let table = TransitionTable::new(0);

    let json = solution_json_for_paths(&metadata(), &table, &[]);

    assert_eq!(json["roots"], serde_json::json!([]));
    assert_eq!(json["nodes"], serde_json::json!({}));
}

#[test]
fn assigned_solution_json_records_concrete_gadget_indices() {
    let input = state(&[0, 2, 4, 6], &[1, 3, 5, 7]);
    let output = state(&[0, 1, 4, 5], &[2, 3, 6, 7]);
    let identity = PermutationGadget::new(Vec::new(), Vec::new());
    let permute = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, identity);
    table.add_transition(0, &input, &output, permute);
    let path = table
        .complete_path_from_zero_based_steps(&[(0, input.as_tuple(), output.as_tuple())])
        .expect("path should resolve");
    let path_a = AssignedPath::new(path.clone(), vec![GadgetIndex(0)]);
    let path_b = AssignedPath::new(path, vec![GadgetIndex(1)]);

    let json = solution_json_for_assigned_paths(&metadata(), &table, &[path_a, path_b]);

    assert_eq!(json["nodes"]["n0"]["gadget_count"], 2);
    assert_eq!(
        json["paths"],
        serde_json::json!([
            {"steps": [{"node_id": "n0", "gadget_index": 0}]},
            {"steps": [{"node_id": "n0", "gadget_index": 1}]}
        ])
    );
}
