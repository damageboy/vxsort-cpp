use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
};

use crate::transition_table::StageStats;

const LINE_STAGE_UPDATE_INTERVAL: usize = 250;

#[derive(Clone, Debug, PartialEq)]
pub struct StageProgressSnapshot {
    stage: usize,
    attempts: usize,
    distinct_outputs: usize,
    transition_count: usize,
    total_gadgets: usize,
    progress_total: usize,
    active_workers: usize,
    worker_capacity: usize,
    queued_jobs: usize,
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
        Self::with_worker_state(stage, stats, progress_total, 0, 0, 0)
    }

    pub fn with_worker_state(
        stage: usize,
        stats: StageStats,
        progress_total: usize,
        active_workers: usize,
        worker_capacity: usize,
        queued_jobs: usize,
    ) -> Self {
        Self {
            stage,
            attempts: stats.attempts,
            distinct_outputs: stats.distinct_outputs,
            transition_count: stats.transition_count,
            total_gadgets: stats.total_gadgets,
            progress_total: progress_total
                .max(stats.attempts)
                .max(stats.distinct_outputs),
            active_workers,
            worker_capacity,
            queued_jobs,
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

    pub fn active_workers(&self) -> usize {
        self.active_workers
    }

    pub fn worker_capacity(&self) -> usize {
        self.worker_capacity
    }

    pub fn queued_jobs(&self) -> usize {
        self.queued_jobs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeEvent {
    RunStarted {
        stage_count: usize,
    },
    TargetStarted {
        target_cpu: String,
        rough_candidate_limit: usize,
        final_top_k: usize,
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
    RoughSearchFinished {
        target_cpu: String,
        wave_count: usize,
        search_exhausted: bool,
        rough_candidate_count: usize,
        best_rough_score: Option<f64>,
    },
    FullScoringStarted {
        target_cpu: String,
        candidate_count: usize,
        final_top_k: usize,
    },
    FullScoringCandidateStarted {
        target_cpu: String,
        candidate_index: usize,
        candidate_count: usize,
        rough_score: f64,
    },
    FullScoringProgress {
        target_cpu: String,
        scored_count: usize,
        candidate_count: usize,
        best_score: Option<f64>,
    },
    FullScoringFinished {
        target_cpu: String,
        scored_count: usize,
        kept_count: usize,
        best_score: Option<f64>,
    },
    ExportStarted {
        target_cpu: String,
        kind: String,
        path: String,
    },
    ExportFinished {
        target_cpu: String,
        kind: String,
        path: String,
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
    last_stage_attempts: BTreeMap<usize, usize>,
    last_stage_worker_states: BTreeMap<usize, (usize, usize, usize)>,
}

impl LineRuntimeSession<io::Stdout> {
    pub fn stdout() -> Self {
        Self::new(io::stdout())
    }
}

impl<W: Write> LineRuntimeSession<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            last_stage_attempts: BTreeMap::new(),
            last_stage_worker_states: BTreeMap::new(),
        }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_line(&mut self, line: String) {
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }

    fn should_write_stage_update(&mut self, snapshot: &StageProgressSnapshot) -> bool {
        let last_attempts = self.last_stage_attempts.get(&snapshot.stage()).copied();
        let worker_state = (
            snapshot.active_workers(),
            snapshot.worker_capacity(),
            snapshot.queued_jobs(),
        );
        let worker_state_changed = self
            .last_stage_worker_states
            .get(&snapshot.stage())
            .is_none_or(|last| last.0 != worker_state.0 || last.1 != worker_state.1);
        let should_write = match last_attempts {
            None => true,
            _ if worker_state_changed => true,
            Some(last_attempts) if snapshot.attempts() <= last_attempts => false,
            Some(last_attempts) => {
                snapshot.attempts() == snapshot.progress_total()
                    || snapshot.attempts().saturating_sub(last_attempts)
                        >= LINE_STAGE_UPDATE_INTERVAL
            }
        };
        if should_write {
            self.last_stage_attempts
                .insert(snapshot.stage(), snapshot.attempts());
            self.last_stage_worker_states
                .insert(snapshot.stage(), worker_state);
        }
        should_write
    }
}

impl<W: Write> RuntimeSession for LineRuntimeSession<W> {
    fn on_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::RunStarted { stage_count } => {
                self.last_stage_attempts.clear();
                self.last_stage_worker_states.clear();
                self.write_line(format!("runtime: run started ({stage_count} stages)"));
            }
            RuntimeEvent::TargetStarted {
                target_cpu,
                rough_candidate_limit,
                final_top_k,
            } => {
                self.write_line(format!(
                    "runtime: target {target_cpu} started (rough candidates {rough_candidate_limit}, final top-k {final_top_k})"
                ));
            }
            RuntimeEvent::WaveStarted { wave, target_stage } => {
                self.write_line(format!(
                    "runtime: wave {wave} started (target stage {target_stage})"
                ));
            }
            RuntimeEvent::StageUpdated(snapshot) => {
                if !self.should_write_stage_update(&snapshot) {
                    return;
                }
                self.write_line(format!(
                    "runtime: stage {} attempts {} outputs {} transitions {} gadgets {} workers {}/{} queued {}",
                    snapshot.stage(),
                    snapshot.attempts(),
                    snapshot.distinct_outputs(),
                    snapshot.transition_count(),
                    snapshot.total_gadgets(),
                    snapshot.active_workers(),
                    snapshot.worker_capacity(),
                    snapshot.queued_jobs()
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
            RuntimeEvent::RoughSearchFinished {
                target_cpu,
                wave_count,
                search_exhausted,
                rough_candidate_count,
                best_rough_score,
            } => {
                let best_rough_score = best_rough_score
                    .map(|score| score.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                self.write_line(format!(
                    "runtime: rough search finished for {target_cpu} (waves {wave_count}, exhausted {search_exhausted}, rough candidates {rough_candidate_count}, best rough score {best_rough_score})"
                ));
            }
            RuntimeEvent::FullScoringStarted {
                target_cpu,
                candidate_count,
                final_top_k,
            } => {
                self.write_line(format!(
                    "runtime: final scoring started for {target_cpu} ({candidate_count} rough candidates, final top-k {final_top_k})"
                ));
            }
            RuntimeEvent::FullScoringCandidateStarted {
                target_cpu,
                candidate_index,
                candidate_count,
                rough_score,
            } => {
                self.write_line(format!(
                    "runtime: final scoring {target_cpu} candidate {candidate_index}/{candidate_count} (rough score {rough_score})"
                ));
            }
            RuntimeEvent::FullScoringProgress {
                target_cpu,
                scored_count,
                candidate_count,
                best_score,
            } => {
                let best_score = best_score
                    .map(|score| score.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                self.write_line(format!(
                    "runtime: final scoring progress for {target_cpu}: {scored_count}/{candidate_count} scored, best score {best_score}"
                ));
            }
            RuntimeEvent::FullScoringFinished {
                target_cpu,
                scored_count,
                kept_count,
                best_score,
            } => {
                let best_score = best_score
                    .map(|score| score.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                self.write_line(format!(
                    "runtime: final scoring finished for {target_cpu}: scored {scored_count}, kept {kept_count}, best score {best_score}"
                ));
            }
            RuntimeEvent::ExportStarted {
                target_cpu,
                kind,
                path,
            } => {
                self.write_line(format!(
                    "runtime: writing {kind} output for {target_cpu}: {path}"
                ));
            }
            RuntimeEvent::ExportFinished {
                target_cpu,
                kind,
                path,
            } => {
                self.write_line(format!(
                    "runtime: wrote {kind} output for {target_cpu}: {path}"
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
