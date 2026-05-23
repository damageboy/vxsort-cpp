use crate::runtime::{RuntimeEvent, StageProgressSnapshot};
use crate::transition_table::StageStats;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
        is_raw_mode_enabled,
    },
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};
use std::{
    io::{self, Stdout},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError},
    },
    time::{Duration, Instant},
};

pub const TUI_TICK_RATE: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FocusedPane {
    Stages,
    Log,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuiState {
    stage_snapshots: Vec<StageProgressSnapshot>,
    stage_rates: Vec<StageRate>,
    run_started_at: Option<Instant>,
    total_gadget_completion_rate: Option<f64>,
    selected_stage: usize,
    focused_pane: FocusedPane,
    current_wave: Option<usize>,
    target_stage: Option<usize>,
    wave_count: usize,
    search_exhausted: bool,
    scored_paths: usize,
    best_score: Option<f64>,
    current_phase: String,
    log_lines: Vec<String>,
    cancel_requested: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StageRate {
    baseline_attempts: usize,
    started_at: Option<Instant>,
    attempts_per_second: Option<f64>,
}

impl StageRate {
    fn new() -> Self {
        Self {
            baseline_attempts: 0,
            started_at: None,
            attempts_per_second: None,
        }
    }

    fn update(&mut self, attempts: usize, now: Instant) {
        if attempts < self.baseline_attempts {
            *self = Self::new();
        }

        if let Some(started_at) = self.started_at {
            let elapsed = now.saturating_duration_since(started_at);
            if elapsed > Duration::ZERO {
                let delta = attempts.saturating_sub(self.baseline_attempts);
                self.attempts_per_second = Some(delta as f64 / elapsed.as_secs_f64());
            }
        } else {
            self.baseline_attempts = attempts;
            self.started_at = Some(now);
        }
    }
}

impl TuiState {
    pub const MAX_LOG_LINES: usize = 100;

    pub fn new(stage_count: usize) -> Self {
        let stage_count = stage_count.max(1);
        Self {
            stage_snapshots: (0..stage_count)
                .map(|stage| StageProgressSnapshot::new(stage, empty_stage_stats()))
                .collect(),
            stage_rates: (0..stage_count).map(|_| StageRate::new()).collect(),
            run_started_at: None,
            total_gadget_completion_rate: None,
            selected_stage: 0,
            focused_pane: FocusedPane::Stages,
            current_wave: None,
            target_stage: None,
            wave_count: 0,
            search_exhausted: false,
            scored_paths: 0,
            best_score: None,
            current_phase: "starting".to_owned(),
            log_lines: Vec::new(),
            cancel_requested: false,
        }
    }

    pub fn apply_event(&mut self, event: RuntimeEvent) {
        self.apply_event_at(event, Instant::now());
    }

    pub fn apply_event_at(&mut self, event: RuntimeEvent, now: Instant) {
        match event {
            RuntimeEvent::RunStarted { stage_count } => {
                *self = Self::new(stage_count);
                self.run_started_at = Some(now);
                self.current_phase = "rough search".to_owned();
                self.push_log(format!("run started ({stage_count} stages)"));
            }
            RuntimeEvent::TargetStarted {
                target_cpu,
                rough_candidate_limit,
                final_top_k,
            } => {
                self.current_phase = format!("rough search {target_cpu}");
                self.push_log(format!(
                    "target {target_cpu}: rough candidates {rough_candidate_limit}, final top-k {final_top_k}"
                ));
            }
            RuntimeEvent::WaveStarted { wave, target_stage } => {
                self.current_wave = Some(wave);
                self.target_stage = Some(target_stage);
                self.stage_rates.fill(StageRate::new());
                self.push_log(format!("wave {wave} started (target stage {target_stage})"));
            }
            RuntimeEvent::StageUpdated(snapshot) => {
                let stage = snapshot.stage();
                if let Some(rate) = self.stage_rates.get_mut(stage) {
                    rate.update(snapshot.attempts(), now);
                }
                if let Some(slot) = self.stage_snapshots.get_mut(stage) {
                    *slot = snapshot;
                }
                self.refresh_total_gadget_completion_rate(now);
            }
            RuntimeEvent::WaveFinished {
                wave,
                target_stage,
                attempts,
                valid_outputs,
                new_outputs,
                discovered_paths,
                scored_paths,
            } => {
                self.scored_paths += scored_paths;
                self.push_log(format!(
                    "wave {wave} finished: target stage {target_stage}, attempts {attempts}, valid {valid_outputs}, new outputs {new_outputs}, discovered paths {discovered_paths}, scored paths {scored_paths}"
                ));
            }
            RuntimeEvent::RoughSearchFinished {
                target_cpu,
                wave_count,
                search_exhausted,
                rough_candidate_count,
                best_rough_score,
            } => {
                self.wave_count = wave_count;
                self.search_exhausted = search_exhausted;
                self.scored_paths = rough_candidate_count;
                self.best_score = best_rough_score;
                self.current_phase = format!("final scoring {target_cpu}");
                self.push_log(format!(
                    "rough search finished for {target_cpu}: waves {wave_count}, exhausted {search_exhausted}, candidates {rough_candidate_count}"
                ));
            }
            RuntimeEvent::FullScoringStarted {
                target_cpu,
                candidate_count,
                final_top_k,
            } => {
                self.current_phase = format!("final scoring {target_cpu}");
                self.push_log(format!(
                    "final scoring started for {target_cpu}: {candidate_count} candidates, keeping {final_top_k}"
                ));
            }
            RuntimeEvent::FullScoringCandidateStarted {
                target_cpu,
                candidate_index,
                candidate_count,
                rough_score,
            } => {
                self.current_phase =
                    format!("final scoring {target_cpu} {candidate_index}/{candidate_count}");
                self.push_log(format!(
                    "final scoring {target_cpu} candidate {candidate_index}/{candidate_count}, rough score {rough_score}"
                ));
            }
            RuntimeEvent::FullScoringProgress {
                target_cpu,
                scored_count,
                candidate_count,
                best_score,
            } => {
                self.scored_paths = scored_count;
                self.best_score = best_score;
                self.current_phase =
                    format!("final scoring {target_cpu} {scored_count}/{candidate_count}");
                self.push_log(format!(
                    "final scoring progress for {target_cpu}: {scored_count}/{candidate_count}"
                ));
            }
            RuntimeEvent::FullScoringFinished {
                target_cpu,
                scored_count,
                kept_count,
                best_score,
            } => {
                self.scored_paths = kept_count;
                self.best_score = best_score;
                self.current_phase = format!("export {target_cpu}");
                self.push_log(format!(
                    "final scoring finished for {target_cpu}: scored {scored_count}, kept {kept_count}"
                ));
            }
            RuntimeEvent::ExportStarted {
                target_cpu,
                kind,
                path,
            } => {
                self.current_phase = format!("export {target_cpu}");
                self.push_log(format!("writing {kind} for {target_cpu}: {path}"));
            }
            RuntimeEvent::ExportFinished {
                target_cpu,
                kind,
                path,
            } => {
                self.push_log(format!("wrote {kind} for {target_cpu}: {path}"));
            }
            RuntimeEvent::RunFinished {
                wave_count,
                search_exhausted,
                scored_paths,
                best_score,
            } => {
                self.wave_count = wave_count;
                self.search_exhausted = search_exhausted;
                self.scored_paths = scored_paths;
                self.best_score = best_score;
                self.current_phase = "finished".to_owned();
                self.push_log(format!(
                    "run finished: waves {wave_count}, exhausted {search_exhausted}, scored paths {scored_paths}"
                ));
            }
        }
    }

    pub fn stage_count(&self) -> usize {
        self.stage_snapshots.len()
    }

    pub fn stage_snapshots(&self) -> &[StageProgressSnapshot] {
        &self.stage_snapshots
    }

    pub fn stage_attempt_rate(&self, stage: usize) -> Option<f64> {
        self.stage_rates.get(stage)?.attempts_per_second
    }

    pub fn total_gadget_completion_rate(&self) -> Option<f64> {
        self.total_gadget_completion_rate
    }

    pub fn active_worker_count(&self) -> usize {
        self.stage_snapshots
            .iter()
            .map(StageProgressSnapshot::active_workers)
            .sum()
    }

    pub fn worker_capacity(&self) -> usize {
        self.stage_snapshots
            .iter()
            .map(StageProgressSnapshot::worker_capacity)
            .sum()
    }

    pub fn queued_job_count(&self) -> usize {
        self.stage_snapshots
            .iter()
            .map(StageProgressSnapshot::queued_jobs)
            .sum()
    }

    fn refresh_total_gadget_completion_rate(&mut self, now: Instant) {
        let Some(run_started_at) = self.run_started_at else {
            self.total_gadget_completion_rate = None;
            return;
        };

        let elapsed = now.saturating_duration_since(run_started_at);
        if elapsed == Duration::ZERO {
            self.total_gadget_completion_rate = None;
            return;
        }

        let total_gadgets = self
            .stage_snapshots
            .iter()
            .map(StageProgressSnapshot::total_gadgets)
            .sum::<usize>();
        self.total_gadget_completion_rate =
            (total_gadgets > 0).then_some(total_gadgets as f64 / elapsed.as_secs_f64());
    }

    pub fn selected_stage(&self) -> usize {
        self.selected_stage
    }

    pub fn focused_pane(&self) -> FocusedPane {
        self.focused_pane
    }

    pub fn current_wave(&self) -> Option<usize> {
        self.current_wave
    }

    pub fn target_stage(&self) -> Option<usize> {
        self.target_stage
    }

    pub fn wave_count(&self) -> usize {
        self.wave_count
    }

    pub fn search_exhausted(&self) -> bool {
        self.search_exhausted
    }

    pub fn scored_paths(&self) -> usize {
        self.scored_paths
    }

    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    pub fn current_phase(&self) -> &str {
        &self.current_phase
    }

    pub fn log_lines(&self) -> &[String] {
        &self.log_lines
    }

    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested
    }

    pub fn select_previous_stage(&mut self) {
        self.selected_stage = self.selected_stage.saturating_sub(1);
    }

    pub fn select_next_stage(&mut self) {
        self.selected_stage = (self.selected_stage + 1).min(self.stage_snapshots.len() - 1);
    }

    pub fn toggle_focus(&mut self) {
        self.focused_pane = match self.focused_pane {
            FocusedPane::Stages => FocusedPane::Log,
            FocusedPane::Log => FocusedPane::Stages,
        };
    }

    fn request_cancel(&mut self) {
        self.cancel_requested = true;
    }

    fn push_log(&mut self, line: String) {
        self.log_lines.push(line);
        while self.log_lines.len() > Self::MAX_LOG_LINES {
            self.log_lines.remove(0);
        }
    }
}

fn empty_stage_stats() -> StageStats {
    StageStats {
        attempts: 0,
        distinct_outputs: 0,
        success_rate: 0.0,
        transition_count: 0,
        total_gadgets: 0,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiControl {
    PreviousStage,
    NextStage,
    ToggleFocus,
    Quit,
}

pub struct TuiApp {
    state: TuiState,
    receiver: Receiver<RuntimeEvent>,
    cancel_flag: Arc<AtomicBool>,
    finished: bool,
}

impl TuiApp {
    pub fn new(
        stage_count: usize,
        receiver: Receiver<RuntimeEvent>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            state: TuiState::new(stage_count),
            receiver,
            cancel_flag,
            finished: false,
        }
    }

    pub fn state(&self) -> &TuiState {
        &self.state
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn drain_events(&mut self) -> usize {
        let mut drained = 0;
        loop {
            match self.receiver.try_recv() {
                Ok(event) => {
                    if matches!(event, RuntimeEvent::RunFinished { .. }) {
                        self.finished = true;
                    }
                    self.state.apply_event(event);
                    drained += 1;
                }
                Err(TryRecvError::Empty) => return drained,
                Err(TryRecvError::Disconnected) => {
                    self.finished = true;
                    return drained;
                }
            }
        }
    }

    pub fn handle_control(&mut self, control: TuiControl) {
        match control {
            TuiControl::PreviousStage => self.state.select_previous_stage(),
            TuiControl::NextStage => self.state.select_next_stage(),
            TuiControl::ToggleFocus => self.state.toggle_focus(),
            TuiControl::Quit => {
                self.cancel_flag.store(true, Ordering::SeqCst);
                self.state.request_cancel();
            }
        }
    }
}

pub fn run_tui_event_loop(
    receiver: Receiver<RuntimeEvent>,
    cancel_flag: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut terminal = TuiTerminal::new()?;
    let mut app = TuiApp::new(1, receiver, cancel_flag);
    terminal.draw(app.state())?;

    while !app.is_finished() {
        app.drain_events();
        while event::poll(TUI_TICK_RATE)? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => app.handle_control(TuiControl::Quit),
                KeyCode::Up => app.handle_control(TuiControl::PreviousStage),
                KeyCode::Down => app.handle_control(TuiControl::NextStage),
                KeyCode::Tab => app.handle_control(TuiControl::ToggleFocus),
                _ => {}
            }
        }
        terminal.draw(app.state())?;
    }

    Ok(())
}

struct TuiTerminal {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    raw_mode_enabled: bool,
    alternate_screen_enabled: bool,
}

impl TuiTerminal {
    fn new() -> io::Result<Self> {
        let mut guard = TerminalSetupGuard::default();
        enable_raw_mode()?;
        guard.raw_mode_enabled = true;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        guard.alternate_screen_enabled = true;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        guard.disarm();

        Ok(Self {
            terminal,
            raw_mode_enabled: true,
            alternate_screen_enabled: true,
        })
    }

    fn draw(&mut self, state: &TuiState) -> io::Result<()> {
        self.terminal.draw(|frame| render_tui(frame, state))?;
        Ok(())
    }
}

impl Drop for TuiTerminal {
    fn drop(&mut self) {
        if self.alternate_screen_enabled {
            let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        }
        let _ = self.terminal.show_cursor();
        if self.raw_mode_enabled && is_raw_mode_enabled().unwrap_or(false) {
            let _ = disable_raw_mode();
        }
    }
}

#[derive(Default)]
struct TerminalSetupGuard {
    raw_mode_enabled: bool,
    alternate_screen_enabled: bool,
}

impl TerminalSetupGuard {
    fn disarm(&mut self) {
        self.raw_mode_enabled = false;
        self.alternate_screen_enabled = false;
    }
}

impl Drop for TerminalSetupGuard {
    fn drop(&mut self) {
        if self.alternate_screen_enabled {
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
        }
        if self.raw_mode_enabled && is_raw_mode_enabled().unwrap_or(false) {
            let _ = disable_raw_mode();
        }
    }
}

pub fn render_tui(frame: &mut Frame<'_>, state: &TuiState) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(frame.area());

    render_summary(frame, root[0], state);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(root[1]);
    render_stage_table(frame, body[0], state);
    render_selected_stage(frame, body[1], state);
    render_log(frame, root[2], state);
    render_footer(frame, root[3], state);
}

fn render_summary(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let wave = state
        .current_wave()
        .map(|wave| format!("Wave {wave}"))
        .unwrap_or_else(|| "Wave pending".to_owned());
    let target = state
        .target_stage()
        .map(|stage| format!("target stage {stage}"))
        .unwrap_or_else(|| "target stage pending".to_owned());
    let best_score = state
        .best_score()
        .map(|score| score.to_string())
        .unwrap_or_else(|| "none".to_owned());
    let mut lines = vec![
        Line::from(vec![
            Span::styled(wave, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" | {target} | stages {}", state.stage_count())),
        ]),
        Line::from(format!(
            "phase {} | scored paths {} | best score {} | workers {}/{} active | queued {} | completed waves {} | exhausted {}",
            state.current_phase(),
            state.scored_paths(),
            best_score,
            state.active_worker_count(),
            state.worker_capacity(),
            state.queued_job_count(),
            state.wave_count(),
            state.search_exhausted()
        )),
    ];
    if state.cancel_requested() {
        lines.push(Line::from("cancelling - waiting for engine shutdown"));
    }
    let paragraph =
        Paragraph::new(lines).block(Block::default().title("Wave Engine").borders(Borders::ALL));
    frame.render_widget(paragraph, area);
}

fn render_stage_table(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let progress_bar_width = stage_progress_bar_width(area.width);
    let rows = state.stage_snapshots().iter().map(|snapshot| {
        let selected = snapshot.stage() == state.selected_stage();
        let style = if selected {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        Row::new(vec![
            right_cell(snapshot.stage().to_string()),
            right_cell(format_attempt_rate(
                state.stage_attempt_rate(snapshot.stage()),
            )),
            Cell::new(render_progress_bar(
                snapshot.attempts(),
                snapshot.distinct_outputs(),
                stage_progress_total(snapshot),
                progress_bar_width,
            )),
            right_cell(snapshot.attempts().to_string()),
            right_cell(snapshot.distinct_outputs().to_string()),
            right_cell(format!(
                "{}/{}",
                snapshot.active_workers(),
                snapshot.worker_capacity()
            )),
            right_cell(snapshot.queued_jobs().to_string()),
            right_cell(snapshot.transition_count().to_string()),
            right_cell(snapshot.total_gadgets().to_string()),
        ])
        .style(style)
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Min((progress_bar_width + 2) as u16),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Length(5),
        ],
    )
    .header(
        Row::new(vec![
            right_cell("Stage"),
            right_cell("Rate/s"),
            Cell::new("Progress"),
            right_cell("Attempts"),
            right_cell("Unique"),
            right_cell("Workers"),
            right_cell("Queued"),
            right_cell("Transitions"),
            right_cell("Valid"),
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .title(format!(
                "Stage Progress | total gadget completions: {}",
                format_total_rate(state.total_gadget_completion_rate())
            ))
            .borders(Borders::ALL),
    );
    frame.render_widget(table, area);
}

fn right_cell<'a>(content: impl Into<Line<'a>>) -> Cell<'a> {
    Cell::new(content.into().alignment(Alignment::Right))
}

pub fn stage_progress_bar_width(panel_width: u16) -> usize {
    const BORDER_WIDTH: u16 = 2;
    const COLUMN_SPACING: u16 = 8;
    const FIXED_COLUMNS: u16 = 5 + 8 + 8 + 6 + 7 + 6 + 11 + 5;
    const BRACKETS: u16 = 2;
    const MIN_BAR_WIDTH: u16 = 12;

    panel_width
        .saturating_sub(BORDER_WIDTH)
        .saturating_sub(COLUMN_SPACING)
        .saturating_sub(FIXED_COLUMNS)
        .saturating_sub(BRACKETS)
        .max(MIN_BAR_WIDTH) as usize
}

pub fn stage_progress_total(snapshot: &StageProgressSnapshot) -> usize {
    snapshot.progress_total().max(1)
}

fn format_attempt_rate(rate: Option<f64>) -> String {
    let Some(rate) = rate else {
        return "--".to_owned();
    };
    if rate < 10.0 {
        format!("{rate:.1}")
    } else if rate < 1000.0 {
        format!("{rate:.0}")
    } else {
        format!("{:.1}k", rate / 1000.0)
    }
}

fn format_total_rate(rate: Option<f64>) -> String {
    let Some(rate) = rate else {
        return "--/s".to_owned();
    };
    if rate < 10.0 {
        format!("{rate:.1}/s")
    } else if rate < 1000.0 {
        format!("{rate:.0}/s")
    } else {
        format!("{:.1}k/s", rate / 1000.0)
    }
}

pub fn render_progress_bar(
    attempts: usize,
    successes: usize,
    total: usize,
    width: usize,
) -> Line<'static> {
    let total = total.max(attempts).max(successes).max(1) as f64;
    let attempt_cells = (attempts as f64 / total).clamp(0.0, 1.0) * width as f64;
    let success_cells = (successes as f64 / total).clamp(0.0, 1.0) * width as f64;
    let attempt_style = Style::default().fg(Color::Yellow);
    let success_style = Style::default().fg(Color::Green);
    let success_on_attempt_style = Style::default().fg(Color::Green).bg(Color::Yellow);

    let mut spans = vec![Span::raw("[")];
    for cell in 0..width {
        let success_level = (success_cells - cell as f64).clamp(0.0, 1.0);
        let attempt_level = (attempt_cells - cell as f64).clamp(0.0, 1.0);

        if success_level >= 1.0 {
            spans.push(Span::styled("█", success_style));
            continue;
        }

        let success_block = fractional_block(success_level);
        if !success_block.is_empty() {
            let style = if attempt_level > success_level {
                success_on_attempt_style
            } else {
                success_style
            };
            spans.push(Span::styled(success_block, style));
            continue;
        }

        if attempt_level >= 1.0 {
            spans.push(Span::styled("█", attempt_style));
            continue;
        }

        let attempt_block = fractional_block(attempt_level);
        if !attempt_block.is_empty() {
            spans.push(Span::styled(attempt_block, attempt_style));
            continue;
        }

        spans.push(Span::raw(" "));
    }
    spans.push(Span::raw("]"));
    Line::from(spans)
}

fn fractional_block(level: f64) -> &'static str {
    let blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
    blocks[(level * 8.0).round() as usize]
}

fn render_selected_stage(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let snapshot = &state.stage_snapshots()[state.selected_stage()];
    let quit_status = if state.cancel_requested() {
        "waiting for engine shutdown"
    } else {
        "q requests shutdown"
    };
    let lines = vec![
        Line::from(format!("stage {}", snapshot.stage())),
        Line::from(format!("attempts: {}", snapshot.attempts())),
        Line::from(format!("outputs: {}", snapshot.distinct_outputs())),
        Line::from(format!("transitions: {}", snapshot.transition_count())),
        Line::from(format!("gadgets: {}", snapshot.total_gadgets())),
        Line::from(""),
        Line::from("Up/Down selects stage"),
        Line::from("Tab switches focus"),
        Line::from(quit_status),
    ];
    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .title(format!("Selected Stage {}", state.selected_stage()))
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn render_log(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let height = area.height.saturating_sub(2) as usize;
    let start = state.log_lines().len().saturating_sub(height);
    let lines = state
        .log_lines()
        .iter()
        .skip(start)
        .map(|line| Line::from(line.as_str()))
        .collect::<Vec<_>>();
    let paragraph =
        Paragraph::new(lines).block(Block::default().title("Runtime Log").borders(Borders::ALL));
    frame.render_widget(paragraph, area);
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let focus = match state.focused_pane() {
        FocusedPane::Stages => "stages",
        FocusedPane::Log => "log",
    };
    let controls = if state.cancel_requested() {
        "up/down stage | tab focus"
    } else {
        "up/down stage | tab focus | q quit"
    };
    let footer = Paragraph::new(format!("focus: {focus} | {controls}"))
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(footer, area);
}
