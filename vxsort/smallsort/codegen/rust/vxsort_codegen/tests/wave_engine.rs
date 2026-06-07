use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use vxsort_codegen::scoring::{AssignedPath, GadgetCost, PathCost, Scorer};
use vxsort_codegen::transition_table::{TransitionKey, TransitionTable};
use vxsort_codegen::wave_engine::{WaveConfig, WaveEngine};
use vxsort_codegen::{ArchArg, DTypeArg, WorkerBackendArg};
use vxsort_codegen::{runtime::NullRuntimeSession, runtime_trace::RuntimeTrace};

fn state(top: &[u64], bottom: &[u64]) -> gadget_synth::VectorState {
    gadget_synth::VectorState::new(top.to_vec(), bottom.to_vec())
}

fn chain_state(index: u64) -> gadget_synth::VectorState {
    let base = index * 8;
    state(
        &[base, base + 1, base + 2, base + 3],
        &[base + 4, base + 5, base + 6, base + 7],
    )
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

fn transitions_with_sorted_gadgets(
    table: &TransitionTable,
    stage: usize,
) -> std::collections::HashMap<TransitionKey, Vec<gadget_synth::PermutationGadget>> {
    table
        .get_all_transitions(stage)
        .into_iter()
        .map(|(key, mut gadgets)| {
            gadgets.sort_by_key(gadget_synth::PermutationGadget::sort_key);
            (key, gadgets)
        })
        .collect()
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
        worker_backend: WorkerBackendArg::InProcess,
        max_unique_outputs: 3,
    }
}

struct CountingScorer;

impl Scorer for CountingScorer {
    fn score_gadget(&self, gadget: &gadget_synth::PermutationGadget) -> GadgetCost {
        let instruction_count =
            (gadget.top_instructions().len() + gadget.bottom_instructions().len()) as u32;
        GadgetCost::new(
            instruction_count,
            instruction_count as f64,
            instruction_count as f64,
            instruction_count as f64,
        )
    }

    fn score_assigned_path(
        &self,
        _assigned_path: &AssignedPath,
        _table: &TransitionTable,
    ) -> PathCost {
        PathCost::new(0, 1.0, 1.0)
    }
}

struct BlockingScorer {
    started: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

impl Scorer for BlockingScorer {
    fn score_gadget(&self, gadget: &gadget_synth::PermutationGadget) -> GadgetCost {
        self.started.fetch_add(1, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }
        let instruction_count =
            (gadget.top_instructions().len() + gadget.bottom_instructions().len()) as u32;
        GadgetCost::new(
            instruction_count,
            instruction_count as f64,
            instruction_count as f64,
            instruction_count as f64,
        )
    }

    fn score_assigned_path(
        &self,
        _assigned_path: &AssignedPath,
        _table: &TransitionTable,
    ) -> PathCost {
        PathCost::new(0, 1.0, 1.0)
    }
}

struct ReleaseScorerOnDrop(Arc<AtomicBool>);

impl Drop for ReleaseScorerOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn construction_matches_python_wave_engine_initial_state() {
    let engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");

    assert_eq!(engine.elements_per_vector(), 4);
    assert_eq!(engine.total_elements(), 8);
    assert_eq!(engine.initial_state().top(), &[0, 2, 4, 6]);
    assert_eq!(engine.initial_state().bottom(), &[1, 3, 5, 7]);
    assert_eq!(engine.stages().len(), 6);
    assert_eq!(
        engine.transition_table().stages().len(),
        engine.stages().len()
    );
}

#[test]
fn scorer_factory_builds_one_shared_rough_scorer_per_engine() {
    let constructed = Arc::new(AtomicUsize::new(0));
    let constructed_by_factory = Arc::clone(&constructed);

    let engine = WaveEngine::with_scorer_factory(
        WaveConfig {
            worker_count: 3,
            ..fast_config()
        },
        move || {
            constructed_by_factory.fetch_add(1, Ordering::SeqCst);
            Box::new(CountingScorer) as Box<dyn Scorer>
        },
    )
    .expect("wave engine should initialize");

    assert_eq!(constructed.load(Ordering::SeqCst), 1);
    drop(engine);
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
        transitions_with_sorted_gadgets(parallel.transition_table(), 0),
        transitions_with_sorted_gadgets(sequential.transition_table(), 0)
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
    assert_eq!(parallel.transition_table().unique_output_count(0), 1);
    assert_eq!(parallel.transition_table().stage_stats(0).total_gadgets, 1);
}

#[test]
fn run_stage_sync_respects_max_unique_outputs_per_candidate() {
    let mut one_output = WaveEngine::new(WaveConfig {
        max_unique_outputs: 1,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut default_outputs =
        WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let candidate_count =
        one_output.shallow_candidates().len() + one_output.deep_candidates().len();

    one_output
        .run_stage_sync(0, candidate_count, usize::MAX)
        .expect("stage run should complete with one output per candidate");
    default_outputs
        .run_stage_sync(0, candidate_count, usize::MAX)
        .expect("stage run should complete with default output count");

    assert!(
        default_outputs.transition_table().unique_output_count(0)
            > one_output.transition_table().unique_output_count(0)
    );
}

#[test]
fn run_wave_sync_exhausts_stage_with_inputs_but_no_remaining_jobs() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let candidate_count = engine.shallow_candidates().len() + engine.deep_candidates().len();

    engine
        .run_stage_sync(0, candidate_count, usize::MAX)
        .expect("stage run should consume all stage 0 candidates");
    let mut session = NullRuntimeSession;
    let mut trace = RuntimeTrace::disabled();
    let result = engine
        .run_wave_sync(5, 1, &mut session, &mut trace)
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
    let mut session = NullRuntimeSession;
    let mut trace = RuntimeTrace::disabled();

    let result = engine
        .run_wave_sync(5, 100, &mut session, &mut trace)
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
    let mut session = NullRuntimeSession;
    let mut trace = RuntimeTrace::disabled();

    let result = engine
        .run_wave_sync(0, 1, &mut session, &mut trace)
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
    let mut session = NullRuntimeSession;
    let mut trace = RuntimeTrace::disabled();

    let result = engine
        .run_wave_sync(
            engine.shallow_candidates().len(),
            1,
            &mut session,
            &mut trace,
        )
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
    let mut session = NullRuntimeSession;
    let mut trace = RuntimeTrace::disabled();

    let result = engine
        .run_wave_sync(5, 1, &mut session, &mut trace)
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
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
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
    let terminal = engine
        .transition_table()
        .transition_ref_for_zero_based_tuples(
            last_stage,
            &states[last_stage].as_tuple(),
            &states[last_stage + 1].as_tuple(),
        )
        .expect("terminal transition should resolve");
    let discovered = engine.discover_paths_for_transition(terminal, None);

    assert_eq!(discovered, 1);
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.best_score(), Some(10.0));
    assert_eq!(engine.scored_paths()[0].cost().score(), 10.0);
    assert_eq!(
        engine.scored_paths()[0].assigned_path().path().len(),
        engine.stages().len()
    );
    assert_eq!(engine.scored_paths()[0].path().len(), engine.stages().len());
}

#[test]
fn terminal_path_discovery_enqueues_scoring_until_drain() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
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
    let terminal = engine
        .transition_table()
        .transition_ref_for_zero_based_tuples(
            last_stage,
            &states[last_stage].as_tuple(),
            &states[last_stage + 1].as_tuple(),
        )
        .expect("terminal transition should resolve");
    let discovered = engine.discover_paths_for_transition(terminal, None);

    assert_eq!(discovered, 1);
    assert_eq!(engine.pending_scoring_job_count(), 1);
    assert_eq!(engine.scored_path_count(), 0);

    let scored = engine.drain_scoring_jobs();

    assert_eq!(scored, 1);
    assert_eq!(engine.pending_scoring_job_count(), 0);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.best_score(), Some(10.0));
}

#[test]
fn terminal_path_burst_queues_backlog_without_waiting_for_scoring_capacity() {
    let started = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicBool::new(false));
    let _release_on_drop = ReleaseScorerOnDrop(Arc::clone(&release));
    let mut engine = WaveEngine::with_scorer(
        WaveConfig {
            worker_count: 1,
            ..fast_config()
        },
        BlockingScorer {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        },
    )
    .expect("wave engine should initialize");
    let start = chain_state(0);
    let layers = (0..5)
        .map(|layer| {
            (0..3)
                .map(|branch| chain_state(1 + layer * 3 + branch))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let terminal_output = chain_state(16);

    for state in &layers[0] {
        engine
            .transition_table_mut()
            .add_transition(0, &start, state, empty_gadget());
    }
    for stage in 1..5 {
        for input in &layers[stage - 1] {
            for output in &layers[stage] {
                engine
                    .transition_table_mut()
                    .add_transition(stage, input, output, empty_gadget());
            }
        }
    }
    let terminal_input = layers[4][0].clone();
    engine.transition_table_mut().add_transition(
        5,
        &terminal_input,
        &terminal_output,
        empty_gadget(),
    );
    let terminal = engine
        .transition_table()
        .transition_ref_for_zero_based_tuples(
            5,
            &terminal_input.as_tuple(),
            &terminal_output.as_tuple(),
        )
        .expect("terminal transition should resolve");

    let discovered = engine.discover_paths_for_transition(terminal, None);

    assert_eq!(discovered, 81);
    assert_eq!(engine.pending_scoring_job_count(), 81);
    assert_eq!(engine.scored_path_count(), 0);
    assert!(started.load(Ordering::SeqCst) <= 1);

    release.store(true, Ordering::SeqCst);
    assert_eq!(engine.drain_scoring_jobs(), 81);
    assert_eq!(engine.pending_scoring_job_count(), 0);
    assert_eq!(engine.scored_path_count(), 81);
}

#[test]
fn complete_path_discovery_deduplicates_already_scored_paths() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
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
    let terminal = engine
        .transition_table()
        .transition_ref_for_zero_based_tuples(
            last_stage,
            &states[last_stage].as_tuple(),
            &states[last_stage + 1].as_tuple(),
        )
        .expect("terminal transition should resolve");
    assert_eq!(engine.discover_paths_for_transition(terminal, None), 1);
    assert_eq!(engine.discover_paths_for_transition(terminal, None), 0);
    assert_eq!(engine.drain_scoring_jobs(), 1);

    assert_eq!(engine.scored_path_count(), 1);
}

#[test]
fn recording_terminal_transition_discovers_and_scores_complete_paths() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
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

    let input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[last_stage].as_tuple())
        .expect("last-stage input should already be interned");
    let result =
        engine.record_transition(last_stage, input, &states[last_stage + 1], empty_gadget());

    assert!(result.transition_added());
    assert_eq!(result.discovered_paths(), 1);
    assert_eq!(engine.pending_scoring_job_count(), 1);
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.best_score(), Some(10.0));
}

#[test]
fn recording_new_mid_stage_transition_discovers_paths_through_existing_prefix_and_suffix() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
    ];
    let mid_stage = 2;

    for stage in 0..mid_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }
    for stage in mid_stage + 1..engine.stages().len() {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }

    let input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[mid_stage].as_tuple())
        .expect("mid-stage input should already be interned");
    let result = engine.record_transition(mid_stage, input, &states[mid_stage + 1], empty_gadget());

    assert!(result.transition_added());
    assert_eq!(result.discovered_paths(), 1);
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.scored_paths()[0].path().len(), engine.stages().len());
}

#[test]
fn scored_path_assignment_uses_lowest_cost_gadget_alternative() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
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

    let input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[last_stage].as_tuple())
        .expect("last-stage input should already be interned");
    engine.record_transition(last_stage, input, &states[last_stage + 1], empty_gadget());
    assert_eq!(engine.drain_scoring_jobs(), 1);

    let assigned = engine.scored_paths()[0].assigned_path();
    assert_eq!(assigned.gadget_at_stage(0).0, 1);
}

#[test]
fn new_gadget_on_known_mid_stage_transition_resubmits_complete_path_for_scoring() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
    ];
    let last_stage = engine.stages().len() - 1;

    engine.transition_table_mut().add_transition(
        0,
        &states[0],
        &states[1],
        gadget_with_instruction_count(3),
    );
    for stage in 1..last_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }
    let terminal_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[last_stage].as_tuple())
        .expect("last-stage input should already be interned");
    engine.record_transition(
        last_stage,
        terminal_input,
        &states[last_stage + 1],
        empty_gadget(),
    );
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(
        engine.scored_paths()[0]
            .assigned_path()
            .gadget_at_stage(0)
            .0,
        0
    );

    let stage0_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[0].as_tuple())
        .expect("stage 0 input should already be interned");
    let result = engine.record_transition(
        0,
        stage0_input,
        &states[1],
        gadget_with_instruction_count(1),
    );

    assert!(result.transition_added());
    assert_eq!(result.discovered_paths(), 0);
    assert_eq!(engine.pending_scoring_job_count(), 0);
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(engine.scored_path_count(), 2);
    assert!(
        engine
            .scored_paths()
            .iter()
            .any(|path| path.assigned_path().gadget_at_stage(0).0 == 1)
    );
}

#[test]
fn repeated_new_gadgets_on_known_transition_coalesce_path_rescoring() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let states = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
    ];
    let last_stage = engine.stages().len() - 1;

    engine.transition_table_mut().add_transition(
        0,
        &states[0],
        &states[1],
        gadget_with_instruction_count(3),
    );
    for stage in 1..last_stage {
        engine.transition_table_mut().add_transition(
            stage,
            &states[stage],
            &states[stage + 1],
            empty_gadget(),
        );
    }
    let terminal_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[last_stage].as_tuple())
        .expect("last-stage input should already be interned");
    engine.record_transition(
        last_stage,
        terminal_input,
        &states[last_stage + 1],
        empty_gadget(),
    );
    assert_eq!(engine.drain_scoring_jobs(), 1);

    let stage0_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&states[0].as_tuple())
        .expect("stage 0 input should already be interned");
    engine.record_transition(
        0,
        stage0_input,
        &states[1],
        gadget_with_instruction_count(2),
    );
    engine.record_transition(
        0,
        stage0_input,
        &states[1],
        gadget_with_instruction_count(1),
    );

    assert_eq!(engine.pending_scoring_job_count(), 0);
    assert_eq!(engine.drain_scoring_jobs(), 1);
    assert_eq!(engine.scored_path_count(), 2);
    assert!(
        engine
            .scored_paths()
            .iter()
            .any(|path| path.assigned_path().gadget_at_stage(0).0 == 2)
    );
}

#[test]
fn scored_path_retention_respects_top_k_with_stable_ties() {
    let mut engine = WaveEngine::new(WaveConfig {
        top_k: Some(1),
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let first_path = [
        chain_state(0),
        chain_state(1),
        chain_state(2),
        chain_state(3),
        chain_state(4),
        chain_state(5),
        chain_state(6),
    ];
    let second_path = [
        chain_state(10),
        chain_state(11),
        chain_state(12),
        chain_state(13),
        chain_state(14),
        chain_state(15),
        chain_state(16),
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
    let first_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&first_path[last_stage].as_tuple())
        .expect("first last-stage input should be interned");
    engine.record_transition(
        last_stage,
        first_input,
        &first_path[last_stage + 1],
        empty_gadget(),
    );
    let second_input = engine
        .transition_table()
        .lookup_zero_based_tuple(&second_path[last_stage].as_tuple())
        .expect("second last-stage input should be interned");
    engine.record_transition(
        last_stage,
        second_input,
        &second_path[last_stage + 1],
        empty_gadget(),
    );

    assert_eq!(engine.drain_scoring_jobs(), 2);
    assert_eq!(engine.scored_path_count(), 1);
    assert_eq!(engine.scored_path_key_count(), engine.scored_path_count());
    assert_eq!(
        engine
            .transition_table()
            .resolve_complete_path(engine.scored_paths()[0].path())[0]
            .input_state,
        first_path[0].as_tuple()
    );
}
