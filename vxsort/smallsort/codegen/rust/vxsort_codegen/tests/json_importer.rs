use std::collections::BTreeMap;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};
use vxsort_codegen::asm_exporter::generate_solution_asm_for_paths;
use vxsort_codegen::json_exporter::{SolutionJsonMetadata, solution_json_for_paths};
use vxsort_codegen::json_importer::solution_json_from_value;
use vxsort_codegen::transition_table::TransitionTable;
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
fn solution_json_import_reconstructs_paths_table_and_asm() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
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
    let path = vec![(0, input.as_tuple(), output.as_tuple())];
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
