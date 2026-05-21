use vxsort_codegen::wave_engine::{WaveConfig, WaveEngine};
use vxsort_codegen::{ArchArg, DTypeArg};

fn state(top: &[u64], bottom: &[u64]) -> gadget_synth::VectorState {
    gadget_synth::VectorState::new(top.to_vec(), bottom.to_vec())
}

fn empty_gadget() -> gadget_synth::PermutationGadget {
    gadget_synth::PermutationGadget::new(Vec::new(), Vec::new())
}

fn gadget_with_instruction_count(count: usize) -> gadget_synth::PermutationGadget {
    gadget_synth::PermutationGadget::new(
        (0..count)
            .map(|_| {
                gadget_synth::InstructionSpec::new("placeholder", std::collections::BTreeMap::new())
            })
            .collect(),
        Vec::new(),
    )
}

fn fast_config() -> WaveConfig {
    WaveConfig {
        num_vecs: 2,
        arch: ArchArg::Avx2,
        dtype: DTypeArg::I64,
        gadget_depth: 1,
        natural_order: false,
        retroactive_input: false,
        top_k: None,
        worker_count: 1,
    }
}

#[test]
fn construction_matches_python_wave_engine_initial_state() {
    let engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    assert_eq!(engine.elements_per_vector(), 4);
    assert_eq!(engine.total_elements(), 8);
    assert_eq!(engine.initial_state().top(), &[1, 3, 5, 7]);
    assert_eq!(engine.initial_state().bottom(), &[2, 4, 6, 8]);
    assert_eq!(engine.stages().len(), 6);
    assert_eq!(
        engine.transition_table().stages().len(),
        engine.stages().len()
    );
}

#[test]
fn natural_order_appends_final_reorder_stage() {
    let without_natural = WaveEngine::new(fast_config()).expect("engine should initialize");
    let with_natural = WaveEngine::new(WaveConfig {
        natural_order: true,
        ..fast_config()
    })
    .expect("engine should initialize");

    assert_eq!(
        with_natural.stages().len(),
        without_natural.stages().len() + 1
    );
    assert_eq!(
        with_natural
            .stages()
            .last()
            .expect("natural stage exists")
            .pairs(),
        &[(1, 5), (2, 6), (3, 7), (4, 8)]
    );
}

#[test]
fn construction_precomputes_candidate_tiers() {
    let engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    assert_eq!(engine.shallow_candidates().len(), 224);
    assert!(engine.deep_candidates().is_empty());
}

#[test]
fn retroactive_input_prepopulates_stage_zero_with_free_identity_permutations() {
    let engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");

    let stage_zero = engine.transition_table().get_all_transitions(0);

    assert_eq!(stage_zero.len(), 24);
    assert_eq!(engine.transition_table().unique_output_count(0), 24);
    for ((input, output), gadgets) in stage_zero {
        assert_eq!(input, output);
        assert_eq!(gadgets.len(), 1);
        assert!(gadgets[0].top_instructions().is_empty());
        assert!(gadgets[0].bottom_instructions().is_empty());
    }
}

#[test]
fn retroactive_input_forwards_and_exhausts_stage_zero() {
    let engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");

    assert!(engine.exhausted_stages().contains(&0));
    assert!(
        engine
            .transition_table()
            .get_unforwarded_outputs(0)
            .is_empty()
    );
    assert_eq!(engine.select_target_stage(), Some(1));
}

#[test]
fn make_jobs_for_stage_zero_uses_initial_state_and_candidate_indices() {
    let engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let jobs = engine.make_jobs(0, None, Some(3));

    assert_eq!(jobs.len(), 3);
    assert_eq!(jobs[0].stage(), 0);
    assert_eq!(jobs[0].input_state(), engine.initial_state());
    assert_eq!(jobs[0].candidate_index(), 0);
    assert_eq!(jobs[1].candidate_index(), 1);
    assert_eq!(jobs[2].candidate_index(), 2);
}

#[test]
fn make_jobs_for_downstream_stage_uses_previous_outputs_and_skips_attempted_pairs() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let initial_state = engine.initial_state().clone();
    let output_a = gadget_synth::VectorState::new(vec![1, 3, 2, 4], vec![5, 7, 6, 8]);
    let output_b = gadget_synth::VectorState::new(vec![2, 1, 4, 3], vec![6, 5, 8, 7]);

    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_a,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_b,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine
        .transition_table_mut()
        .record_attempted_pair(1, &output_a.as_tuple(), 0);

    let jobs = engine.make_jobs(1, None, Some(226));

    assert_eq!(jobs.len(), 226);
    assert!(
        !jobs
            .iter()
            .any(|job| job.input_state() == &output_a && job.candidate_index() == 0)
    );
    assert!(
        jobs.iter()
            .any(|job| job.input_state() == &output_b && job.candidate_index() == 0)
    );
    assert!(jobs.iter().all(|job| job.stage() == 1));
}

#[test]
fn make_jobs_uses_candidate_windows_across_all_inputs() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let initial_state = engine.initial_state().clone();
    let output_a = gadget_synth::VectorState::new(vec![1, 3, 2, 4], vec![5, 7, 6, 8]);
    let output_b = gadget_synth::VectorState::new(vec![2, 1, 4, 3], vec![6, 5, 8, 7]);

    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_a,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_b,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );

    let jobs = engine.make_jobs(1, None, Some(4));

    assert_eq!(jobs.len(), 4);
    assert_eq!(jobs[0].candidate_index(), 0);
    assert_eq!(jobs[1].candidate_index(), 0);
    assert_eq!(jobs[2].candidate_index(), 1);
    assert_eq!(jobs[3].candidate_index(), 1);
    assert_ne!(jobs[0].input_state(), jobs[1].input_state());
    assert_eq!(jobs[0].input_state(), jobs[2].input_state());
    assert_eq!(jobs[1].input_state(), jobs[3].input_state());
}

#[test]
fn make_jobs_prioritizes_uncovered_inputs_in_each_candidate_window() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let initial_state = engine.initial_state().clone();
    let output_a = gadget_synth::VectorState::new(vec![1, 3, 2, 4], vec![5, 7, 6, 8]);
    let output_b = gadget_synth::VectorState::new(vec![2, 1, 4, 3], vec![6, 5, 8, 7]);

    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_a,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine.transition_table_mut().add_transition(
        0,
        &initial_state,
        &output_b,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine
        .transition_table_mut()
        .record_attempted_pair(1, &output_a.as_tuple(), 7);

    let jobs = engine.make_jobs(1, None, Some(2));

    assert_eq!(jobs.len(), 2);
    assert_eq!(jobs[0].candidate_index(), 0);
    assert_eq!(jobs[0].input_state(), &output_b);
    assert_eq!(jobs[1].candidate_index(), 0);
    assert_eq!(jobs[1].input_state(), &output_a);
}

#[test]
fn execute_job_records_attempt_and_successful_transitions() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let jobs = engine.make_jobs(0, None, Some(engine.shallow_candidates().len()));

    let mut successful = None;
    for job in jobs {
        let result = engine
            .execute_job(&job)
            .expect("job execution should not error");
        if result.transition_count() > 0 {
            successful = Some(result);
            break;
        }
    }

    let result = successful.expect("at least one shallow candidate should solve stage 0");
    assert_eq!(result.stage(), 0);
    assert_eq!(
        engine.transition_table().stage_stats(0).attempts,
        result.attempts()
    );
    assert!(engine.transition_table().unique_output_count(0) > 0);
    assert!(!engine.transition_table().get_all_transitions(0).is_empty());
}

#[test]
fn run_stage_sync_respects_attempt_budget_and_records_progress() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let result = engine
        .run_stage_sync(0, 5, 100)
        .expect("stage run should not error");

    assert_eq!(result.stage(), 0);
    assert_eq!(result.attempts(), 5);
    assert_eq!(engine.transition_table().stage_stats(0).attempts, 5);
}

#[test]
fn run_stage_sync_with_workers_matches_single_worker_results() {
    let mut sequential = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut parallel = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");

    let sequential_result = sequential
        .run_stage_sync(0, 20, 100)
        .expect("single-worker stage run should not error");
    let parallel_result = parallel
        .run_stage_sync(0, 20, 100)
        .expect("multi-worker stage run should not error");

    assert_eq!(parallel_result, sequential_result);
    assert_eq!(
        parallel.transition_table().stage_stats(0),
        sequential.transition_table().stage_stats(0)
    );
    assert_eq!(
        parallel.transition_table().get_all_transitions(0),
        sequential.transition_table().get_all_transitions(0)
    );
}

#[test]
fn worker_pool_stops_applying_results_after_output_budget() {
    let mut sequential = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut parallel = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");

    let sequential_result = sequential
        .run_stage_sync(0, 20, 1)
        .expect("single-worker stage run should not error");
    let parallel_result = parallel
        .run_stage_sync(0, 20, 1)
        .expect("multi-worker stage run should not error");

    assert_eq!(parallel_result, sequential_result);
    assert_eq!(
        parallel.transition_table().stage_stats(0),
        sequential.transition_table().stage_stats(0)
    );
    assert_eq!(
        parallel.transition_table().get_all_transitions(0),
        sequential.transition_table().get_all_transitions(0)
    );
}

#[test]
fn run_wave_sync_exhausts_stage_with_inputs_but_no_remaining_jobs() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let candidate_count = engine.shallow_candidates().len() + engine.deep_candidates().len();

    engine
        .run_stage_sync(0, candidate_count, usize::MAX)
        .expect("stage run should consume all stage 0 candidates");
    let result = engine
        .run_wave_sync(5, 1)
        .expect("wave run should not error");

    assert_eq!(result.target_stage(), 0);
    assert_eq!(result.target().attempts(), 0);
    assert!(engine.exhausted_stages().contains(&0));
    assert_ne!(engine.select_target_stage(), Some(0));
}

#[test]
fn select_target_stage_starts_at_stage_zero() {
    let engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    assert_eq!(engine.wave_count(), 0);
    assert_eq!(engine.select_target_stage(), Some(0));
}

#[test]
fn run_wave_sync_runs_selected_stage_and_increments_wave_count() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let result = engine
        .run_wave_sync(5, 100)
        .expect("wave run should not error");

    assert_eq!(result.wave(), 0);
    assert_eq!(result.target_stage(), 0);
    assert_eq!(result.target().attempts(), 5);
    assert_eq!(result.discovered_paths(), 0);
    assert_eq!(result.scored_paths(), 0);
    assert_eq!(engine.wave_count(), 1);
    assert_eq!(engine.transition_table().stage_stats(0).attempts, 5);
}

#[test]
fn run_wave_sync_records_zero_output_budget_for_target_stage() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let result = engine
        .run_wave_sync(0, 1)
        .expect("wave run should not error");

    assert_eq!(result.target_stage(), 0);
    assert_eq!(result.target().new_outputs(), 0);
    assert_eq!(
        engine
            .transition_table()
            .stage(0)
            .consecutive_zero_budgets(),
        1
    );
}

#[test]
fn run_wave_sync_propagates_new_outputs_to_downstream_stage() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let result = engine
        .run_wave_sync(engine.shallow_candidates().len(), 1)
        .expect("wave run should not error");

    assert_eq!(result.target_stage(), 0);
    assert!(
        result.target().new_outputs() > 0,
        "stage 0 should produce at least one output for downstream propagation"
    );
    assert!(
        result
            .propagation()
            .iter()
            .any(|stage| stage.stage() == 1 && stage.attempts() > 0),
        "new stage 0 outputs should be attempted by stage 1 in the same wave"
    );
}

#[test]
fn run_sync_honors_max_waves_and_records_wave_results() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    let result = engine
        .run_sync(Some(2), 5, 100)
        .expect("sync run should not error");

    assert_eq!(result.wave_count(), 2);
    assert_eq!(engine.wave_count(), 2);
    assert_eq!(result.waves().len(), 2);
    assert!(!result.search_exhausted());
}

#[test]
fn select_target_stage_bubbles_up_after_repeated_zero_output_budgets() {
    let mut engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let stage0_output = engine
        .transition_table()
        .get_unique_outputs(0)
        .values()
        .next()
        .expect("retroactive input should create stage 0 outputs")
        .clone();
    let stage1_output = gadget_synth::VectorState::new(vec![1, 2, 3, 4], vec![5, 6, 7, 8]);

    engine.transition_table_mut().add_transition(
        1,
        &stage0_output,
        &stage1_output,
        gadget_synth::PermutationGadget::new(Vec::new(), Vec::new()),
    );
    engine.transition_table_mut().record_zero_output_budget(2);
    engine.transition_table_mut().record_zero_output_budget(2);

    assert_eq!(engine.select_target_stage(), Some(1));
}

#[test]
fn run_wave_sync_marks_downstream_stage_without_inputs_as_stalled() {
    let mut engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    engine.transition_table_mut().record_attempt(1, 1);

    let result = engine
        .run_wave_sync(5, 1)
        .expect("wave run should not error");

    assert_eq!(result.target_stage(), 2);
    assert_eq!(result.target().attempts(), 0);
    assert!(engine.stalled_stages().contains(&2));
    assert_ne!(engine.select_target_stage(), Some(2));
}

#[test]
fn complete_path_discovery_scores_terminal_paths_with_dummy_score() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        state(&[1], &[2]),
        state(&[3], &[4]),
        state(&[5], &[6]),
        state(&[7], &[8]),
        state(&[9], &[10]),
        state(&[11], &[12]),
        state(&[13], &[14]),
    ];

    for stage in 0..engine.stages().len() {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }

    let last_stage = engine.stages().len() - 1;
    let discovered = engine.discover_paths_for_transition(
        last_stage,
        &states[last_stage].as_tuple(),
        &states[last_stage + 1].as_tuple(),
        None,
    );

    assert_eq!(discovered, 1);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.best_score(), Some(10.0));
    assert_eq!(engine.scored_paths()[0].cost().score(), 10.0);
    assert_eq!(
        engine.scored_paths()[0].assigned_path().steps().len(),
        engine.stages().len()
    );
    assert_eq!(engine.scored_paths()[0].path().len(), engine.stages().len());
}

#[test]
fn complete_path_discovery_deduplicates_already_scored_paths() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        state(&[1], &[2]),
        state(&[3], &[4]),
        state(&[5], &[6]),
        state(&[7], &[8]),
        state(&[9], &[10]),
        state(&[11], &[12]),
        state(&[13], &[14]),
    ];

    for stage in 0..engine.stages().len() {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }

    let last_stage = engine.stages().len() - 1;
    assert_eq!(
        engine.discover_paths_for_transition(
            last_stage,
            &states[last_stage].as_tuple(),
            &states[last_stage + 1].as_tuple(),
            None,
        ),
        1
    );
    assert_eq!(
        engine.discover_paths_for_transition(
            last_stage,
            &states[last_stage].as_tuple(),
            &states[last_stage + 1].as_tuple(),
            None,
        ),
        0
    );

    assert_eq!(engine.scored_path_count(), 1);
}

#[test]
fn recording_terminal_transition_discovers_and_scores_complete_paths() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        state(&[1], &[2]),
        state(&[3], &[4]),
        state(&[5], &[6]),
        state(&[7], &[8]),
        state(&[9], &[10]),
        state(&[11], &[12]),
        state(&[13], &[14]),
    ];
    let last_stage = engine.stages().len() - 1;

    for stage in 0..last_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }

    let result = engine.record_transition(
        last_stage,
        &states[last_stage],
        &states[last_stage + 1],
        empty_gadget(),
    );

    assert!(result.transition_added());
    assert_eq!(result.discovered_paths(), 1);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.best_score(), Some(10.0));
}

#[test]
fn scored_path_assignment_uses_lowest_cost_gadget_alternative() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        state(&[1], &[2]),
        state(&[3], &[4]),
        state(&[5], &[6]),
        state(&[7], &[8]),
        state(&[9], &[10]),
        state(&[11], &[12]),
        state(&[13], &[14]),
    ];
    let last_stage = engine.stages().len() - 1;

    engine.transition_table_mut().add_transition(
        0,
        &states[0],
        &states[1],
        gadget_with_instruction_count(3),
    );
    engine.transition_table_mut().add_transition(
        0,
        &states[0],
        &states[1],
        gadget_with_instruction_count(1),
    );
    for stage in 1..last_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }

    engine.record_transition(
        last_stage,
        &states[last_stage],
        &states[last_stage + 1],
        empty_gadget(),
    );

    let assigned = engine.scored_paths()[0].assigned_path();
    assert_eq!(assigned.steps()[0].gadget_index(), 1);
}

#[test]
fn scored_path_retention_respects_top_k_with_stable_ties() {
    let mut engine = WaveEngine::new(WaveConfig {
        top_k: Some(1),
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let first_path = [
        state(&[1], &[2]),
        state(&[3], &[4]),
        state(&[5], &[6]),
        state(&[7], &[8]),
        state(&[9], &[10]),
        state(&[11], &[12]),
        state(&[13], &[14]),
    ];
    let second_path = [
        state(&[21], &[22]),
        state(&[23], &[24]),
        state(&[25], &[26]),
        state(&[27], &[28]),
        state(&[29], &[30]),
        state(&[31], &[32]),
        state(&[33], &[34]),
    ];
    let last_stage = engine.stages().len() - 1;

    for stage in 0..last_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &first_path[stage],
            &first_path[stage + 1],
            empty_gadget(),
        );
        engine.transition_table_mut().add_transition(
            stage,
            &second_path[stage],
            &second_path[stage + 1],
            empty_gadget(),
        );
    }
    engine.record_transition(
        last_stage,
        &first_path[last_stage],
        &first_path[last_stage + 1],
        empty_gadget(),
    );
    engine.record_transition(
        last_stage,
        &second_path[last_stage],
        &second_path[last_stage + 1],
        empty_gadget(),
    );

    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(
        engine.scored_paths()[0].path()[0].1,
        first_path[0].as_tuple()
    );
}
