use std::collections::BTreeMap;

use bitonic_codegen::asm_exporter::{
    generate_solution_asm_for_assigned_paths, generate_solution_asm_for_paths,
};
use bitonic_codegen::json_exporter::{
    SolutionJsonMetadata, solution_json_for_assigned_paths, solution_json_for_paths,
};
use bitonic_codegen::json_importer::solution_json_from_value;
use bitonic_codegen::scoring::AssignedPath;
use bitonic_codegen::transition_table::{GadgetIndex, TransitionTable};
use bitonic_codegen::{ArchArg, DTypeArg};
use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};

fn state(top: &[u8], bottom: &[u8]) -> VectorState {
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
fn solution_json_import_reconstructs_paths_table_and_asm() {
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
    let json = solution_json_for_paths(&metadata(), &table, std::slice::from_ref(&path));

    let imported = solution_json_from_value(&json).expect("exported JSON should import");

    assert_eq!(imported.metadata, metadata());
    assert_eq!(imported.paths, vec![path]);
    assert_eq!(
        imported.transition_table.get_all_transitions(0).len(),
        table.get_all_transitions(0).len()
    );

    let asm = generate_solution_asm_for_paths(
        &imported.metadata,
        &imported.transition_table,
        &imported.paths,
    );
    assert!(asm.contains("vpermq"));
    assert!(asm.contains("ret"));
}

#[test]
fn solution_json_import_reconstructs_concrete_assigned_paths() {
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
    table.add_transition(0, &input, &output, permute.clone());
    let path = table
        .complete_path_from_zero_based_steps(&[(0, input.as_tuple(), output.as_tuple())])
        .expect("path should resolve");
    let selected = AssignedPath::new(path.clone(), vec![GadgetIndex(1)]);
    let unselected_first = AssignedPath::new(path, vec![GadgetIndex(0)]);
    let json = solution_json_for_assigned_paths(&metadata(), &table, &[unselected_first, selected]);
    let selected_json = serde_json::json!({
        "natural_order": json["natural_order"],
        "vector_machine": json["vector_machine"],
        "primitive_type": json["primitive_type"],
        "num_vecs": json["num_vecs"],
        "roots": json["roots"],
        "nodes": json["nodes"],
        "paths": [json["paths"][1].clone()]
    });

    let imported = solution_json_from_value(&selected_json).expect("assigned JSON should import");

    assert_eq!(imported.assigned_paths.len(), 1);
    assert_eq!(
        imported.assigned_paths[0].gadget_at_stage(0),
        GadgetIndex(1)
    );
    assert_eq!(
        imported.paths,
        vec![imported.assigned_paths[0].as_complete_path()]
    );

    let asm = generate_solution_asm_for_assigned_paths(
        &imported.metadata,
        &imported.transition_table,
        &imported.assigned_paths,
    );
    assert!(asm.contains("vpermq"));
    assert!(asm.contains("ret"));
}

#[test]
fn solution_json_import_rejects_missing_child_nodes() {
    let json = serde_json::json!({
        "natural_order": false,
        "vector_machine": "AVX2",
        "primitive_type": "i64",
        "num_vecs": 2,
        "roots": ["n0"],
        "nodes": {
            "n0": {
                "stage": 0,
                "input_state": {"top": [1], "bottom": [2]},
                "output_state": {"top": [1], "bottom": [2]},
                "gadgets": [],
                "gadget_count": 0,
                "children": ["missing"]
            }
        }
    });

    let error = solution_json_from_value(&json).expect_err("missing child should fail");

    assert!(error.contains("missing node"));
}

#[test]
fn solution_json_import_accepts_python_primitive_aliases_for_asm_export() {
    for (primitive_type, dtype) in [
        ("i16", DTypeArg::I16),
        ("u16", DTypeArg::U16),
        ("u32", DTypeArg::U32),
        ("f32", DTypeArg::F32),
        ("u64", DTypeArg::U64),
        ("f64", DTypeArg::F64),
    ] {
        let json = serde_json::json!({
            "natural_order": false,
            "vector_machine": "AVX2",
            "primitive_type": primitive_type,
            "num_vecs": 2,
            "roots": ["n0"],
            "nodes": {
                "n0": {
                    "stage": 0,
                    "input_state": {"top": [1, 3], "bottom": [2, 4]},
                    "output_state": {"top": [1, 2], "bottom": [3, 4]},
                    "gadgets": [{"top_instructions": [], "bottom_instructions": []}],
                    "gadget_count": 1,
                    "children": []
                }
            }
        });

        let imported = solution_json_from_value(&json)
            .unwrap_or_else(|error| panic!("{primitive_type} should import: {error}"));

        assert_eq!(imported.metadata.dtype, dtype);
    }
}

#[test]
fn solution_json_import_accepts_large_python_control_vector_integer() {
    let json: serde_json::Value = serde_json::from_str(
        r#"{
  "natural_order": false,
  "vector_machine": "AVX2",
  "primitive_type": "i64",
  "num_vecs": 2,
  "roots": ["n0"],
  "nodes": {
    "n0": {
      "stage": 0,
      "input_state": {"top": [1, 3, 5, 7], "bottom": [2, 4, 6, 8]},
      "output_state": {"top": [1, 3, 5, 7], "bottom": [2, 4, 6, 8]},
      "gadgets": [{
        "top_instructions": [{
          "name": "_mm256_permutexvar_epi64",
          "args": {
            "a": "top",
            "op_idx": 18831305206160042292187933003464876175252262292329349513216
          }
        }],
        "bottom_instructions": []
      }],
      "gadget_count": 1,
      "children": []
    }
  }
}"#,
    )
    .expect("large JSON integer should parse");

    let imported = solution_json_from_value(&json).expect("large integer should import");
    let transitions = imported.transition_table.get_all_transitions(0);
    let gadgets = transitions
        .values()
        .next()
        .expect("transition should exist");
    let args = gadgets[0].top_instructions()[0].args();

    assert_eq!(
        args.get("op_idx"),
        Some(&InstructionArg::BitVec {
            bits: 256,
            hex: "3000000000000000200000000000000010000000000000000".to_owned(),
        })
    );
}
