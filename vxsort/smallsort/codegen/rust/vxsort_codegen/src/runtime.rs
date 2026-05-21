use std::{
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
};

use crate::transition_table::StageStats;

#[derive(Clone, Debug, PartialEq)]
pub struct StageProgressSnapshot {
    stage: usize,
    attempts: usize,
    distinct_outputs: usize,
    transition_count: usize,
    total_gadgets: usize,
    progress_total: usize,
}

impl StageProgressSnapshot {
    pub fn new(stage: usize, stats: StageStats) -> Self {
        Self::with_progress_total(
            stage,
            stats.clone(),
            stats.attempts.max(stats.distinct_outputs),
        )
    }

    pub fn with_progress_total(stage: usize, stats: StageStats, progress_total: usize) -> Self {
        Self {
            stage,
            attempts: stats.attempts,
            distinct_outputs: stats.distinct_outputs,
            transition_count: stats.transition_count,
            total_gadgets: stats.total_gadgets,
            progress_total: progress_total
                .max(stats.attempts)
                .max(stats.distinct_outputs),
        }
    }

    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn attempts(&self) -> usize {
        self.attempts
    }

    pub fn distinct_outputs(&self) -> usize {
        self.distinct_outputs
    }

    pub fn transition_count(&self) -> usize {
        self.transition_count
    }

    pub fn total_gadgets(&self) -> usize {
        self.total_gadgets
    }

    pub fn progress_total(&self) -> usize {
        self.progress_total
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeEvent {
    RunStarted {
        stage_count: usize,
    },
    WaveStarted {
        wave: usize,
        target_stage: usize,
    },
    StageUpdated(StageProgressSnapshot),
    WaveFinished {
        wave: usize,
        target_stage: usize,
        attempts: usize,
        valid_outputs: usize,
        new_outputs: usize,
        discovered_paths: usize,
        scored_paths: usize,
    },
    RunFinished {
        wave_count: usize,
        search_exhausted: bool,
        scored_paths: usize,
        best_score: Option<f64>,
    },
}

pub trait RuntimeSession {
    fn on_event(&mut self, event: RuntimeEvent);

    fn should_stop(&self) -> bool {
        false
    }
}

#[derive(Default)]
pub struct NullRuntimeSession;

impl RuntimeSession for NullRuntimeSession {
    fn on_event(&mut self, _event: RuntimeEvent) {}
}

pub struct ChannelRuntimeSession {
    sender: Sender<RuntimeEvent>,
    cancel_flag: Arc<AtomicBool>,
}

impl ChannelRuntimeSession {
    pub fn new(sender: Sender<RuntimeEvent>, cancel_flag: Arc<AtomicBool>) -> Self {
        Self {
            sender,
            cancel_flag,
        }
    }
}

impl RuntimeSession for ChannelRuntimeSession {
    fn on_event(&mut self, event: RuntimeEvent) {
        let _ = self.sender.send(event);
    }

    fn should_stop(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }
}

pub struct LineRuntimeSession<W: Write> {
    writer: W,
}

impl LineRuntimeSession<io::Stdout> {
    pub fn stdout() -> Self {
        Self::new(io::stdout())
    }
}

impl<W: Write> LineRuntimeSession<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_line(&mut self, line: String) {
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }
}

impl<W: Write> RuntimeSession for LineRuntimeSession<W> {
    fn on_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::RunStarted { stage_count } => {
                self.write_line(format!("runtime: run started ({stage_count} stages)"));
            }
            RuntimeEvent::WaveStarted { wave, target_stage } => {
                self.write_line(format!(
                    "runtime: wave {wave} started (target stage {target_stage})"
                ));
            }
            RuntimeEvent::StageUpdated(snapshot) => {
                self.write_line(format!(
                    "runtime: stage {} attempts {} outputs {} transitions {} gadgets {}",
                    snapshot.stage(),
                    snapshot.attempts(),
                    snapshot.distinct_outputs(),
                    snapshot.transition_count(),
                    snapshot.total_gadgets()
                ));
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
                self.write_line(format!(
                    "runtime: wave {wave} finished (target stage {target_stage}, attempts {attempts}, valid outputs {valid_outputs}, new outputs {new_outputs}, discovered paths {discovered_paths}, scored paths {scored_paths})"
                ));
            }
            RuntimeEvent::RunFinished {
                wave_count,
                search_exhausted,
                scored_paths,
                best_score,
            } => {
                let best_score = best_score
                    .map(|score| score.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                self.write_line(format!(
                    "runtime: run finished (waves {wave_count}, exhausted {search_exhausted}, scored paths {scored_paths}, best score {best_score})"
                ));
            }
        }
    }
}
