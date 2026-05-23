use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use vxsort_codegen::runtime::{ChannelRuntimeSession, RuntimeEvent, RuntimeSession};
use vxsort_codegen::wave_engine::{WaveConfig, WaveEngine};
use vxsort_codegen::{ArchArg, DTypeArg};

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
        max_unique_outputs: 3,
    }
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
