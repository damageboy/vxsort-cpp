use std::collections::BTreeMap;

use bitonic_codegen::transition_table::{
    CompletePath, PathRegistry, StateInterner, TransitionIndex, TransitionRef, TransitionTable,
};
use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};

fn state(top: &[u8], bottom: &[u8]) -> VectorState {
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

fn path_for(
    table: &TransitionTable,
    steps: &[(usize, &VectorState, &VectorState)],
) -> CompletePath {
    let steps = steps
        .iter()
        .map(|(stage, input, output)| (*stage, input.as_tuple(), output.as_tuple()))
        .collect::<Vec<_>>();
    table
        .complete_path_from_zero_based_steps(&steps)
        .expect("test path should resolve")
}

#[test]
fn interning_same_state_returns_same_id() {
    let mut interner = StateInterner::new(4);
    let state = VectorState::from_top_bottom([0, 1, 2, 3], [4, 5, 6, 7]);

    let first = interner.intern(state.clone());
    let second = interner.intern(state);

    assert_eq!(first, second);
    assert_eq!(interner.len(), 1);
}

#[test]
fn path_registry_stores_complete_path_once_and_indexes_transitions_by_id() {
    let mut registry = PathRegistry::default();
    let path = CompletePath::new(vec![TransitionIndex(3), TransitionIndex(5)]);

    let first = registry.register(path.clone());
    let second = registry.register(path.clone());

    assert_eq!(first.id, second.id);
    assert!(first.was_new);
    assert!(!second.was_new);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.path(first.id), &path);
    assert_eq!(
        registry.paths_for_transition(TransitionRef {
            stage: 0,
            transition: TransitionIndex(3),
        }),
        &[first.id]
    );
    assert_eq!(
        registry.paths_for_transition(TransitionRef {
            stage: 1,
            transition: TransitionIndex(5),
        }),
        &[first.id]
    );
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
    let input_id = table.lookup_zero_based_tuple(&input.as_tuple()).unwrap();
    let output_id = table.lookup_zero_based_tuple(&output.as_tuple()).unwrap();
    assert!(table.stage(0).unique_input_ids().contains(&input_id));
    assert!(table.stage(0).unique_output_ids().contains(&output_id));
}

#[test]
fn cloned_transition_table_snapshots_keep_prior_gadget_lists() {
    let mut table = TransitionTable::new(1);
    let input = state(&[3, 2, 1, 0], &[7, 6, 5, 4]);
    let output = state(&[0, 1, 2, 3], &[4, 5, 6, 7]);

    assert!(table.add_transition(0, &input, &output, gadget_with_imm(1)));
    let snapshot = table.clone();
    assert!(table.add_transition(0, &input, &output, gadget_with_imm(2)));

    let live_transitions = table.get_transitions(0, &input.as_tuple());
    let snapshot_transitions = snapshot.get_transitions(0, &input.as_tuple());
    assert_eq!(live_transitions[&output.as_tuple()].len(), 2);
    assert_eq!(snapshot_transitions[&output.as_tuple()].len(), 1);
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
    let first_path = path_for(&table, &[(0, &a, &b), (1, &b, &d), (2, &d, &z)]);
    let second_path = path_for(&table, &[(0, &a, &c), (1, &c, &e), (2, &e, &z)]);

    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&first_path));
    assert!(paths.contains(&second_path));
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
        vec![path_for(&table, &[(0, &a, &c), (1, &c, &e), (2, &e, &z)])]
    );
}
