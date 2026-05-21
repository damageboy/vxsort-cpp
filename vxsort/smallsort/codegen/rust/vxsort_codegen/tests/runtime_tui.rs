use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

use vxsort_codegen::runtime::{RuntimeEvent, StageProgressSnapshot};
use vxsort_codegen::runtime_tui::{
    FocusedPane, TUI_TICK_RATE, TuiApp, TuiControl, TuiState, render_progress_bar, render_tui,
    stage_progress_bar_width, stage_progress_total,
};
use vxsort_codegen::transition_table::StageStats;

fn snapshot(stage: usize, attempts: usize, outputs: usize) -> StageProgressSnapshot {
    snapshot_with_gadgets(stage, attempts, outputs, outputs + 2)
}

fn snapshot_with_gadgets(
    stage: usize,
    attempts: usize,
    outputs: usize,
    total_gadgets: usize,
) -> StageProgressSnapshot {
    StageProgressSnapshot::new(
        stage,
        StageStats {
            attempts,
            distinct_outputs: outputs,
            success_rate: 0.0,
            transition_count: outputs + 1,
            total_gadgets,
        },
    )
}

#[test]
fn tui_state_updates_wave_metadata_stage_snapshots_and_logs() {
    let mut state = TuiState::new(3);

    state.apply_event(RuntimeEvent::RunStarted { stage_count: 3 });
    state.apply_event(RuntimeEvent::WaveStarted {
        wave: 7,
        target_stage: 2,
    });
    state.apply_event(RuntimeEvent::StageUpdated(snapshot(2, 55, 8)));
    state.apply_event(RuntimeEvent::WaveFinished {
        wave: 7,
        target_stage: 2,
        attempts: 11,
        valid_outputs: 4,
        new_outputs: 2,
        discovered_paths: 1,
        scored_paths: 1,
    });

    assert_eq!(state.stage_count(), 3);
    assert_eq!(state.current_wave(), Some(7));
    assert_eq!(state.target_stage(), Some(2));
    assert_eq!(state.scored_paths(), 1);
    assert_eq!(state.stage_snapshots()[2].attempts(), 55);
    assert_eq!(state.stage_snapshots()[2].distinct_outputs(), 8);
    assert!(
        state
            .log_lines()
            .iter()
            .any(|line| line.contains("wave 7 finished"))
    );
}

#[test]
fn tui_tick_rate_is_relaxed_to_ten_frames_per_second() {
    assert_eq!(TUI_TICK_RATE, Duration::from_millis(100));
}

#[test]
fn tui_state_clamps_stage_selection_and_toggles_focus() {
    let mut state = TuiState::new(2);

    state.select_previous_stage();
    assert_eq!(state.selected_stage(), 0);

    state.select_next_stage();
    state.select_next_stage();
    assert_eq!(state.selected_stage(), 1);

    assert_eq!(state.focused_pane(), FocusedPane::Stages);
    state.toggle_focus();
    assert_eq!(state.focused_pane(), FocusedPane::Log);
}

#[test]
fn tui_state_keeps_log_bounded() {
    let mut state = TuiState::new(1);

    for wave in 0..150 {
        state.apply_event(RuntimeEvent::WaveStarted {
            wave,
            target_stage: 0,
        });
    }

    assert_eq!(state.log_lines().len(), TuiState::MAX_LOG_LINES);
    assert!(
        state
            .log_lines()
            .first()
            .expect("bounded log should contain entries")
            .contains("wave 50")
    );
}

#[test]
fn render_tui_draws_stage_wave_and_log_regions() {
    let backend = ratatui::backend::TestBackend::new(100, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let mut state = TuiState::new(2);
    state.apply_event(RuntimeEvent::WaveStarted {
        wave: 3,
        target_stage: 1,
    });
    state.apply_event(RuntimeEvent::StageUpdated(snapshot(1, 22, 5)));

    terminal
        .draw(|frame| render_tui(frame, &state))
        .expect("render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered = buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("Stage Progress"));
    assert!(rendered.contains("Wave 3"));
    assert!(rendered.contains("Runtime Log"));
}

#[test]
fn render_tui_draws_readable_stage_table_headers() {
    let backend = ratatui::backend::TestBackend::new(140, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let state = TuiState::new(2);

    terminal
        .draw(|frame| render_tui(frame, &state))
        .expect("render should succeed");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("Attempts"));
    assert!(rendered.contains("Unique"));
    assert!(rendered.contains("Transitions"));
    assert!(rendered.contains("Valid"));
}

#[test]
fn render_tui_draws_per_stage_attempt_rate() {
    let backend = ratatui::backend::TestBackend::new(120, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let mut state = TuiState::new(2);
    let start = Instant::now();

    state.apply_event_at(RuntimeEvent::StageUpdated(snapshot(1, 10, 1)), start);
    state.apply_event_at(
        RuntimeEvent::StageUpdated(snapshot(1, 35, 3)),
        start + Duration::from_secs(5),
    );

    terminal
        .draw(|frame| render_tui(frame, &state))
        .expect("render should succeed");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("Rate/s"));
    assert!(rendered.contains("5.0"));
    assert!(!rendered.contains("items/s"));
}

#[test]
fn render_tui_draws_per_stage_rate_since_stage_started_working() {
    let backend = ratatui::backend::TestBackend::new(120, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let mut state = TuiState::new(2);
    let start = Instant::now();

    state.apply_event_at(RuntimeEvent::StageUpdated(snapshot(1, 10, 1)), start);
    state.apply_event_at(
        RuntimeEvent::StageUpdated(snapshot(1, 35, 3)),
        start + Duration::from_secs(5),
    );
    state.apply_event_at(
        RuntimeEvent::StageUpdated(snapshot(1, 45, 4)),
        start + Duration::from_secs(10),
    );

    terminal
        .draw(|frame| render_tui(frame, &state))
        .expect("render should succeed");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("3.5"));
    assert!(!rendered.contains("items/s"));
}

#[test]
fn render_tui_draws_total_active_stage_gadget_completion_rate() {
    let backend = ratatui::backend::TestBackend::new(120, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let mut state = TuiState::new(2);
    let start = Instant::now();

    state.apply_event_at(RuntimeEvent::RunStarted { stage_count: 2 }, start);
    state.apply_event_at(
        RuntimeEvent::StageUpdated(snapshot_with_gadgets(0, 40, 4, 9)),
        start + Duration::from_secs(10),
    );
    state.apply_event_at(
        RuntimeEvent::StageUpdated(snapshot_with_gadgets(1, 35, 3, 12)),
        start + Duration::from_secs(10),
    );

    terminal
        .draw(|frame| render_tui(frame, &state))
        .expect("render should succeed");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("total gadget completions: 2.1/s"));
}

#[test]
fn cancelling_render_shows_engine_shutdown_wait_status() {
    let backend = ratatui::backend::TestBackend::new(100, 28);
    let mut terminal = ratatui::Terminal::new(backend).expect("test backend should initialize");
    let (_sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(2, receiver, Arc::clone(&cancel_flag));

    app.handle_control(TuiControl::Quit);

    terminal
        .draw(|frame| render_tui(frame, app.state()))
        .expect("render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered = buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("cancelling - waiting for engine shutdown"));
}

#[test]
fn render_progress_bar_uses_fractional_blocks_for_success_over_attempts() {
    let line = render_progress_bar(10, 3, 10, 10);
    let symbols = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(symbols, "[██████████]");
    assert_eq!(line.spans[1].style.fg, Some(ratatui::style::Color::Green));
    assert_eq!(line.spans[4].style.fg, Some(ratatui::style::Color::Yellow));
}

#[test]
fn render_progress_bar_represents_partial_attempts_and_empty_tail() {
    let line = render_progress_bar(1, 0, 4, 10);
    let symbols = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(symbols, "[██▌       ]");
    assert_eq!(line.spans[1].style.fg, Some(ratatui::style::Color::Yellow));
}

#[test]
fn stage_progress_bar_width_expands_with_panel_width() {
    assert_eq!(stage_progress_bar_width(50), 12);
    assert_eq!(stage_progress_bar_width(70), 17);
    assert_eq!(stage_progress_bar_width(100), 47);
}

#[test]
fn stage_progress_total_is_local_to_the_stage_snapshot() {
    let retroactive_stage_zero = snapshot(0, 0, 24);
    let active_stage_one = snapshot(1, 200, 7);

    assert_eq!(stage_progress_total(&retroactive_stage_zero), 24);
    assert_eq!(stage_progress_total(&active_stage_one), 200);
}

#[test]
fn stage_progress_total_uses_queued_work_from_snapshot() {
    let snapshot = StageProgressSnapshot::with_progress_total(
        1,
        StageStats {
            attempts: 75,
            distinct_outputs: 3,
            success_rate: 0.0,
            transition_count: 3,
            total_gadgets: 3,
        },
        5_000,
    );

    assert_eq!(stage_progress_total(&snapshot), 5_000);
}

#[test]
fn tui_app_drains_runtime_events_without_blocking() {
    let (sender, receiver) = mpsc::channel();
    sender
        .send(RuntimeEvent::RunStarted { stage_count: 4 })
        .expect("send should succeed");
    sender
        .send(RuntimeEvent::WaveStarted {
            wave: 9,
            target_stage: 3,
        })
        .expect("send should succeed");

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(1, receiver, Arc::clone(&cancel_flag));

    let drained = app.drain_events();

    assert_eq!(drained, 2);
    assert_eq!(app.state().stage_count(), 4);
    assert_eq!(app.state().current_wave(), Some(9));
    assert_eq!(app.state().target_stage(), Some(3));
    assert!(!app.is_finished());
    assert!(!cancel_flag.load(Ordering::SeqCst));
}

#[test]
fn cancel_control_sets_cancel_flag_without_finishing_app() {
    let (_sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(2, receiver, Arc::clone(&cancel_flag));

    app.handle_control(TuiControl::Quit);

    assert!(cancel_flag.load(Ordering::SeqCst));
    assert!(!app.is_finished());
}

#[test]
fn run_finished_after_cancel_marks_app_finished() {
    let (sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(2, receiver, Arc::clone(&cancel_flag));

    app.handle_control(TuiControl::Quit);
    sender
        .send(RuntimeEvent::RunFinished {
            wave_count: 2,
            search_exhausted: false,
            scored_paths: 0,
            best_score: None,
        })
        .expect("send should succeed");

    assert_eq!(app.drain_events(), 1);
    assert!(app.is_finished());
}

#[test]
fn cancel_control_exposes_cancel_requested_state() {
    let (_sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(2, receiver, Arc::clone(&cancel_flag));

    assert!(!app.state().cancel_requested());

    app.handle_control(TuiControl::Quit);

    assert!(app.state().cancel_requested());
}

#[test]
fn finished_app_stays_finished_when_quit_sets_cancel_flag() {
    let (sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = TuiApp::new(2, receiver, Arc::clone(&cancel_flag));

    sender
        .send(RuntimeEvent::RunFinished {
            wave_count: 2,
            search_exhausted: false,
            scored_paths: 0,
            best_score: None,
        })
        .expect("send should succeed");

    assert_eq!(app.drain_events(), 1);
    assert!(app.is_finished());

    app.handle_control(TuiControl::Quit);

    assert!(app.is_finished());
    assert!(cancel_flag.load(Ordering::SeqCst));
}
