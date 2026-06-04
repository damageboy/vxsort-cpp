use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

use serde_json::{Value, json};
use vxsort_codegen::runtime::{ChannelRuntimeSession, RuntimeEvent, RuntimeSession};
use vxsort_codegen::runtime_trace::RuntimeTrace;
use vxsort_codegen::wave_engine::{WaveConfig, WaveEngine};
use vxsort_codegen::{ArchArg, DTypeArg, WorkerBackendArg};

#[derive(Default)]
struct RecordingRuntimeSession {
    events: Vec<RuntimeEvent>,
}

impl RuntimeSession for RecordingRuntimeSession {
    fn on_event(&mut self, event: RuntimeEvent) {
        self.events.push(event);
    }
}

#[derive(Default)]
struct StopAfterWaveStarted {
    events: Vec<RuntimeEvent>,
    stop: bool,
}

impl RuntimeSession for StopAfterWaveStarted {
    fn on_event(&mut self, event: RuntimeEvent) {
        if matches!(event, RuntimeEvent::WaveStarted { .. }) {
            self.stop = true;
        }
        self.events.push(event);
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
}

#[derive(Default)]
struct StopAfterStageUpdated {
    events: Vec<RuntimeEvent>,
    stop: bool,
}

impl RuntimeSession for StopAfterStageUpdated {
    fn on_event(&mut self, event: RuntimeEvent) {
        if matches!(
            event,
            RuntimeEvent::StageUpdated(ref snapshot) if snapshot.attempts() > 0
        ) {
            self.stop = true;
        }
        self.events.push(event);
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
}

#[derive(Default)]
struct StopAfterActiveWorkerUpdate {
    events: Vec<RuntimeEvent>,
    stop: bool,
}

impl RuntimeSession for StopAfterActiveWorkerUpdate {
    fn on_event(&mut self, event: RuntimeEvent) {
        if matches!(
            event,
            RuntimeEvent::StageUpdated(ref snapshot) if snapshot.active_workers() > 0
        ) {
            self.stop = true;
        }
        self.events.push(event);
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
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

#[test]
fn run_sync_with_session_emits_lifecycle_and_stage_updates() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut session = RecordingRuntimeSession::default();

    let result = engine
        .run_sync_with_session(Some(1), 0, 1, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert!(matches!(
        session.events.first(),
        Some(RuntimeEvent::RunStarted { stage_count: 6 })
    ));
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::WaveStarted {
            wave: 0,
            target_stage: 0
        }
    )));
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::StageUpdated(snapshot)
            if snapshot.stage() == 0 && snapshot.attempts() == 0
    )));
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::WaveFinished {
            wave: 0,
            target_stage: 0,
            attempts: 0,
            ..
        }
    )));
    assert!(matches!(
        session.events.last(),
        Some(RuntimeEvent::RunFinished {
            wave_count: 1,
            search_exhausted: false,
            scored_paths: 0,
            best_score: None,
        })
    ));
}

#[test]
fn run_sync_with_session_emits_async_scoring_progress() {
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

    let mut session = RecordingRuntimeSession::default();
    engine
        .run_sync_with_session(Some(0), 0, 1, &mut session)
        .expect("run should complete");

    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::ScoringProgress(snapshot)
            if snapshot.total_paths() == 1
                && snapshot.pending_paths() == 1
                && snapshot.completed_paths() == 0
    )));
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::ScoringProgress(snapshot)
            if snapshot.total_paths() == 1
                && snapshot.pending_paths() == 0
                && snapshot.completed_paths() == 1
                && snapshot.scored_paths() == 1
    )));
}

#[test]
fn run_sync_with_retroactive_input_emits_initial_stage_zero_outputs_before_first_wave() {
    let mut engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = RecordingRuntimeSession::default();

    engine
        .run_sync_with_session(Some(1), 0, 1, &mut session)
        .expect("run should complete");

    let stage_zero_update = session
        .events
        .iter()
        .position(|event| {
            matches!(
                event,
                RuntimeEvent::StageUpdated(snapshot)
                    if snapshot.stage() == 0 && snapshot.distinct_outputs() == 24
            )
        })
        .expect("retroactive stage 0 outputs should be emitted");
    let first_wave = session
        .events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::WaveStarted { .. }))
        .expect("run should start a wave");

    assert!(
        stage_zero_update < first_wave,
        "UI should see retroactive stage 0 outputs before wave work starts"
    );
}

#[test]
fn run_sync_reports_stage_progress_total_from_queued_jobs() {
    let mut engine = WaveEngine::new(WaveConfig {
        retroactive_input: true,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = RecordingRuntimeSession::default();

    engine
        .run_sync_with_session(Some(1), 5, 1, &mut session)
        .expect("run should complete");

    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::StageUpdated(snapshot)
            if snapshot.stage() == 1 && snapshot.attempts() == 0 && snapshot.progress_total() == 5
    )));
}

#[test]
fn runtime_session_can_stop_run_between_waves() {
    struct StopAfterFirstWave {
        events: Vec<RuntimeEvent>,
    }

    impl RuntimeSession for StopAfterFirstWave {
        fn on_event(&mut self, event: RuntimeEvent) {
            self.events.push(event);
        }

        fn should_stop(&self) -> bool {
            self.events
                .iter()
                .any(|event| matches!(event, RuntimeEvent::WaveFinished { .. }))
        }
    }

    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut session = StopAfterFirstWave { events: Vec::new() };

    let result = engine
        .run_sync_with_session(Some(5), 0, 1, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert_eq!(
        session
            .events
            .iter()
            .filter(|event| matches!(event, RuntimeEvent::WaveStarted { .. }))
            .count(),
        1
    );
}

#[test]
fn runtime_session_can_stop_run_during_active_wave() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut session = StopAfterWaveStarted::default();

    let result = engine
        .run_sync_with_session(Some(1), 10, 1, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert_eq!(result.waves().len(), 1);
    assert_eq!(result.waves()[0].target().attempts(), 0);
    assert_eq!(engine.transition_table().stage_stats(0).attempts, 0);
    assert!(matches!(
        session.events.last(),
        Some(RuntimeEvent::RunFinished { .. })
    ));
}

#[test]
fn run_sync_with_session_emits_stage_progress_during_active_wave() {
    let mut engine = WaveEngine::new(fast_config()).expect("wave engine should initialize");
    let mut session = StopAfterStageUpdated::default();

    let result = engine
        .run_sync_with_session(Some(1), 5, 100, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert_eq!(result.waves().len(), 1);
    assert_eq!(result.waves()[0].target().attempts(), 1);
    assert_eq!(engine.transition_table().stage_stats(0).attempts, 1);
    assert!(session.events.iter().position(|event| matches!(
        event,
        RuntimeEvent::StageUpdated(snapshot) if snapshot.stage() == 0 && snapshot.attempts() == 1
    ))
    .is_some_and(|stage_update_index| session.events[stage_update_index + 1..]
        .iter()
        .any(|event| matches!(event, RuntimeEvent::WaveFinished { .. }))));
}

#[test]
fn worker_run_emits_stage_progress_during_active_wave() {
    let mut engine = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = StopAfterStageUpdated::default();

    let result = engine
        .run_sync_with_session(Some(1), 5, 100, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert_eq!(result.waves().len(), 1);
    assert_eq!(result.waves()[0].target().attempts(), 1);
    assert_eq!(engine.transition_table().stage_stats(0).attempts, 1);
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::StageUpdated(snapshot) if snapshot.stage() == 0 && snapshot.attempts() == 1
    )));
}

#[test]
fn worker_run_reports_active_worker_occupancy_before_results_apply() {
    let mut engine = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = StopAfterActiveWorkerUpdate::default();

    let result = engine
        .run_sync_with_session(Some(1), 5, 100, &mut session)
        .expect("run should complete");

    assert_eq!(result.wave_count(), 1);
    assert!(session.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::StageUpdated(snapshot)
            if snapshot.stage() == 0
                && snapshot.active_workers() > 0
                && snapshot.worker_capacity() == 2
                && snapshot.queued_jobs() > 0
    )));
}

#[test]
fn worker_run_does_not_report_submitted_jobs_as_active_workers() {
    let mut engine = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = RecordingRuntimeSession::default();

    engine
        .run_sync_with_session(Some(1), 5, 100, &mut session)
        .expect("run should complete");

    let first_pool_snapshot = session
        .events
        .iter()
        .find_map(|event| match event {
            RuntimeEvent::StageUpdated(snapshot)
                if snapshot.stage() == 0 && snapshot.worker_capacity() == 2 =>
            {
                Some(snapshot)
            }
            _ => None,
        })
        .expect("worker pool should emit a capacity snapshot");

    assert_eq!(first_pool_snapshot.active_workers(), 0);
    assert_eq!(first_pool_snapshot.queued_jobs(), 5);
}

#[test]
fn runtime_trace_records_worker_jobs_and_counters() {
    let mut engine = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");
    let mut session = RecordingRuntimeSession::default();
    let trace_path = std::env::temp_dir().join(format!(
        "vxsort-runtime-trace-worker-jobs-{}.jsonl",
        std::process::id()
    ));
    let _ = fs::remove_file(&trace_path);

    {
        let mut trace = RuntimeTrace::open(&trace_path).expect("trace file should open");
        engine
            .run_sync_with_session_and_trace(Some(1), 5, 100, &mut session, &mut trace)
            .expect("run should complete");
        trace.flush().expect("trace should flush");
    }

    let trace_contents = fs::read_to_string(&trace_path).expect("trace file should be readable");
    let events = trace_contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("trace line should be JSON"))
        .collect::<Vec<_>>();
    let _ = fs::remove_file(&trace_path);

    assert!(
        events
            .iter()
            .any(|event| event["event"] == "worker_pool_started")
    );
    assert!(events.iter().any(|event| event["event"] == "job_submitted"));
    assert!(events.iter().any(|event| event["event"] == "job_started"));
    assert!(events.iter().any(|event| event["event"] == "job_finished"));
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "job_apply_finished")
    );
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "worker_pool_finished")
    );
    assert!(events.iter().any(|event| {
        event["event"] == "worker_counters"
            && event["stage"] == 0
            && event["worker_capacity"] == 2
            && event["queued_jobs"].as_u64().is_some()
            && event["in_flight"].as_u64().is_some()
            && event["active_workers"].as_u64().is_some()
    }));
}

#[test]
fn runtime_trace_records_scoring_queue_events() {
    let mut engine = WaveEngine::new(WaveConfig {
        worker_count: 2,
        ..fast_config()
    })
    .expect("wave engine should initialize");
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
    let mut session = RecordingRuntimeSession::default();
    let trace_path = std::env::temp_dir().join(format!(
        "vxsort-runtime-trace-scoring-queue-{}.jsonl",
        std::process::id()
    ));
    let _ = fs::remove_file(&trace_path);

    {
        let mut trace = RuntimeTrace::open(&trace_path).expect("trace file should open");
        assert_eq!(
            engine.discover_paths_for_transition_with_trace(terminal, None, &mut trace),
            1
        );
        engine
            .run_sync_with_session_and_trace(Some(0), 0, 1, &mut session, &mut trace)
            .expect("run should complete");
        trace.flush().expect("trace should flush");
    }

    let trace_contents = fs::read_to_string(&trace_path).expect("trace file should be readable");
    let events = trace_contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("trace line should be JSON"))
        .collect::<Vec<_>>();
    let _ = fs::remove_file(&trace_path);

    assert!(events.iter().any(|event| {
        event["event"] == "scoring_job_queued"
            && event["reason"].as_str().is_some()
            && event["order"].as_u64().is_some()
            && event["path_length"].as_u64().is_some()
            && event["path_transitions"].as_array().is_some()
            && event["queued"].as_u64().is_some()
            && event["submitted"].as_u64().is_some()
            && event["completed"].as_u64().is_some()
            && event["pending"].as_u64().is_some()
            && event["scored"].as_u64().is_some()
    }));
    assert!(events.iter().any(|event| {
        event["event"] == "scoring_job_applied"
            && event["reason"].as_str().is_some()
            && event["order"].as_u64().is_some()
            && event["path_length"].as_u64().is_some()
            && event["assigned_candidates"].as_u64().is_some()
            && event["applied_candidates"].as_u64().is_some()
            && event["queued"].as_u64().is_some()
            && event["submitted"].as_u64().is_some()
            && event["completed"].as_u64().is_some()
            && event["pending"].as_u64().is_some()
            && event["scored"].as_u64().is_some()
    }));
    assert!(events.iter().any(|event| {
        event["event"] == "scoring_queue_snapshot"
            && event["reason"].as_str().is_some()
            && event["queued"].as_u64().is_some()
            && event["submitted"].as_u64().is_some()
            && event["completed"].as_u64().is_some()
            && event["pending"].as_u64().is_some()
            && event["scored"].as_u64().is_some()
    }));
}

#[test]
fn runtime_trace_writes_complete_line_without_explicit_flush() {
    let trace_path = std::env::temp_dir().join(format!(
        "vxsort-runtime-trace-live-{}.jsonl",
        std::process::id()
    ));
    let _ = fs::remove_file(&trace_path);

    let mut trace = RuntimeTrace::open(&trace_path).expect("trace file should open");
    trace.event("probe", json!({"value": 1}));

    let trace_contents = fs::read_to_string(&trace_path).expect("trace file should be readable");
    let _ = fs::remove_file(&trace_path);
    let line = trace_contents
        .lines()
        .next()
        .expect("trace event should be visible before explicit flush");
    let event = serde_json::from_str::<Value>(line).expect("visible trace line should be JSON");

    assert_eq!(event["event"], "probe");
    assert_eq!(event["value"], 1);
    assert_eq!(trace_contents.lines().count(), 1);
}

#[test]
fn channel_runtime_session_forwards_events_and_reads_cancel_flag() {
    let (sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut session = ChannelRuntimeSession::new(sender, Arc::clone(&cancel_flag));

    session.on_event(RuntimeEvent::WaveStarted {
        wave: 3,
        target_stage: 2,
    });

    assert_eq!(
        receiver.recv().expect("event should be sent"),
        RuntimeEvent::WaveStarted {
            wave: 3,
            target_stage: 2
        }
    );
    assert!(!session.should_stop());

    cancel_flag.store(true, Ordering::SeqCst);

    assert!(session.should_stop());
}
