use std::collections::BTreeMap;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};
use vxsort_codegen::transition_table::TransitionTable;

fn state(top: &[u64], bottom: &[u64]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn empty_gadget() -> PermutationGadget {
    PermutationGadget::new(Vec::new(), Vec::new())
}

fn gadget_with_imm(value: u64) -> PermutationGadget {
    let mut args = BTreeMap::new();
    args.insert("imm8", InstructionArg::U64(value));
    PermutationGadget::new(
        vec![InstructionSpec::new("_mm256_permute_pd", args)],
        Vec::new(),
    )
}

fn add_identity(
    table: &mut TransitionTable,
    stage_idx: usize,
    input: &VectorState,
    output: &VectorState,
) {
    table.add_transition(stage_idx, input, output, empty_gadget());
}

#[test]
fn add_transition_tracks_unique_inputs_outputs_and_deduplicates_gadgets() {
    let mut table = TransitionTable::new(1);
    let input = state(&[3, 2, 1, 0], &[7, 6, 5, 4]);
    let output = state(&[0, 1, 2, 3], &[4, 5, 6, 7]);
    let first = gadget_with_imm(1);
    let second = gadget_with_imm(2);

    assert!(table.add_transition(0, &input, &output, first.clone()));
    assert!(!table.add_transition(0, &input, &output, first));
    assert!(table.add_transition(0, &input, &output, second));

    let transitions = table.get_transitions(0, &input.as_tuple());
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[&output.as_tuple()].len(), 2);
    assert_eq!(table.unique_output_count(0), 1);
    assert!(table.stage(0).unique_inputs().contains(&input.as_tuple()));
    assert!(
        table
            .stage(0)
            .unique_outputs()
            .contains_key(&output.as_tuple())
    );
}

#[test]
fn records_attempts_attempted_pairs_and_forwarded_outputs() {
    let mut table = TransitionTable::new(2);
    let input = state(&[1, 2, 3, 4], &[5, 6, 7, 8]);
    let output_a = state(&[1, 3, 2, 4], &[5, 7, 6, 8]);
    let output_b = state(&[2, 1, 4, 3], &[6, 5, 8, 7]);

    table.add_transition(0, &input, &output_a, empty_gadget());
    table.add_transition(0, &input, &output_b, gadget_with_imm(7));
    table.record_attempt(0, 5);
    table.record_attempted_pair(0, &input.as_tuple(), 42);

    assert_eq!(table.stage_stats(0).attempts, 5);
    assert!(table.was_attempted(0, &input.as_tuple(), 42));
    assert!(!table.was_attempted(0, &input.as_tuple(), 43));
    assert!(table.was_input_attempted(0, &input.as_tuple()));
    assert!(!table.was_input_attempted(0, &output_a.as_tuple()));

    let unforwarded = table.get_unforwarded_outputs(0);
    assert_eq!(unforwarded.len(), 2);
    table.mark_forwarded(0, &[output_a.as_tuple()]);

    let remaining = table.get_unforwarded_outputs(0);
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].0, output_b.as_tuple());
}

#[test]
fn enumerates_complete_paths_through_every_stage() {
    let mut table = TransitionTable::new(3);
    let a = state(&[1], &[2]);
    let b = state(&[3], &[4]);
    let c = state(&[5], &[6]);
    let d = state(&[7], &[8]);
    let e = state(&[9], &[10]);
    let z = state(&[11], &[12]);

    add_identity(&mut table, 0, &a, &b);
    add_identity(&mut table, 0, &a, &c);
    add_identity(&mut table, 1, &b, &d);
    add_identity(&mut table, 1, &c, &e);
    add_identity(&mut table, 2, &d, &z);
    add_identity(&mut table, 2, &e, &z);

    let paths = table.enumerate_complete_paths(None, None, None);

    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&vec![
        (0, a.as_tuple(), b.as_tuple()),
        (1, b.as_tuple(), d.as_tuple()),
        (2, d.as_tuple(), z.as_tuple()),
    ]));
    assert!(paths.contains(&vec![
        (0, a.as_tuple(), c.as_tuple()),
        (1, c.as_tuple(), e.as_tuple()),
        (2, e.as_tuple(), z.as_tuple()),
    ]));
}

#[test]
fn traces_complete_paths_ending_with_terminal_transition() {
    let mut table = TransitionTable::new(3);
    let a = state(&[1], &[2]);
    let b = state(&[3], &[4]);
    let c = state(&[5], &[6]);
    let d = state(&[7], &[8]);
    let e = state(&[9], &[10]);
    let z = state(&[11], &[12]);

    add_identity(&mut table, 0, &a, &b);
    add_identity(&mut table, 0, &a, &c);
    add_identity(&mut table, 1, &b, &d);
    add_identity(&mut table, 1, &c, &e);
    add_identity(&mut table, 2, &d, &z);
    add_identity(&mut table, 2, &e, &z);

    let paths = table.trace_paths_ending_with(2, &e.as_tuple(), &z.as_tuple(), None, None);

    assert_eq!(
        paths,
        vec![vec![
            (0, a.as_tuple(), c.as_tuple()),
            (1, c.as_tuple(), e.as_tuple()),
            (2, e.as_tuple(), z.as_tuple()),
        ]]
    );
}
