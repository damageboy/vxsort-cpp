use std::collections::BTreeMap;

use gadget_synth::{InstructionSpec, PermutationGadget, VectorState};
use vxsort_codegen::scoring::{DummyScorer, Scorer};
use vxsort_codegen::transition_table::TransitionTable;

fn state(top: &[u64], bottom: &[u64]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn gadget_with_instruction_count(count: usize) -> PermutationGadget {
    PermutationGadget::new(
        (0..count)
            .map(|_| InstructionSpec::new("top_a", BTreeMap::new()))
            .collect(),
        Vec::new(),
    )
}

#[test]
fn dummy_scorer_counts_gadget_instructions() {
    let scorer = DummyScorer;
    let gadget = PermutationGadget::new(
        vec![InstructionSpec::new("top_a", BTreeMap::new())],
        vec![
            InstructionSpec::new("bottom_a", BTreeMap::new()),
            InstructionSpec::new("bottom_b", BTreeMap::new()),
        ],
    );

    let cost = scorer.score_gadget(&gadget);

    assert_eq!(cost.instruction_count(), 3);
    assert_eq!(cost.score(), 3.0);
}

#[test]
fn dummy_scorer_assigns_constant_path_score() {
    let scorer = DummyScorer;
    let table = TransitionTable::new(0);

    let cost = scorer.score_path(&Vec::new(), &table);

    assert_eq!(cost.score(), 10.0);
    assert_eq!(cost.instruction_count(), 0);
}

#[test]
fn dummy_scorer_assigns_lowest_cost_gadget_per_path_step() {
    let scorer = DummyScorer;
    let mut table = TransitionTable::new(1);
    let input = state(&[1], &[2]);
    let output = state(&[3], &[4]);
    let expensive = gadget_with_instruction_count(3);
    let cheap = gadget_with_instruction_count(1);

    table.add_transition(0, &input, &output, expensive);
    table.add_transition(0, &input, &output, cheap);
    let path = vec![(0, input.as_tuple(), output.as_tuple())];

    let assigned = scorer
        .assign_path_gadgets(&path, &table)
        .expect("path should have gadget assignments");

    assert_eq!(assigned.steps().len(), 1);
    assert_eq!(assigned.steps()[0].gadget_index(), 1);
    assert_eq!(
        scorer
            .score_gadget(assigned.steps()[0].gadget())
            .instruction_count(),
        1
    );
}
