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

/// Point-in-time progress for one synthesis stage.
///
/// The wave engine emits this snapshot through [`RuntimeEvent::StageUpdated`]
/// whenever stage-local synthesis state changes. Counters are cumulative for
/// the stage within the current run: `attempts` is the number of candidate
/// synthesis jobs applied to the transition table, `distinct_outputs` is the
/// number of unique states produced by this stage, and `transition_count` /
/// `total_gadgets` describe the current transition-table contents for the
/// stage.
///
/// Worker fields describe scheduler state at the instant the snapshot was
/// captured. They are especially useful when diagnosing process-backed worker
/// under-utilization: `active_workers` is the number of workers currently
/// holding jobs, `worker_capacity` is the configured worker capacity for this
/// stage, and `queued_jobs` is the amount of synthesis work known to the
/// coordinator.
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
    /// Builds a snapshot from stage statistics without explicit scheduler
    /// state.
    ///
    /// The progress denominator is inferred from whichever of attempts or
    /// distinct outputs is larger. Prefer [`Self::with_progress_total`] or
    /// [`Self::with_worker_state`] when the caller knows the intended progress
    /// denominator.
    pub fn new(stage: usize, stats: StageStats) -> Self {
        Self::with_progress_total(
            stage,
            stats.clone(),
            stats.attempts.max(stats.distinct_outputs),
        )
    }

    /// Builds a snapshot with an explicit progress denominator.
    ///
    /// `progress_total` is clamped up to at least the current attempts and
    /// distinct-output count, so renderers can safely display `attempts /
    /// progress_total` without showing impossible over-complete progress.
    pub fn with_progress_total(stage: usize, stats: StageStats, progress_total: usize) -> Self {
        Self::with_worker_state(stage, stats, progress_total, 0, 0, 0)
    }

    /// Builds a snapshot with explicit scheduler state.
    ///
    /// This is the normal constructor used by the wave engine while synthesis
    /// workers are running. The worker counters are sampled values, not
    /// cumulative totals.
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

    /// Zero-based synthesis stage index.
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// Number of candidate synthesis jobs applied to this stage so far.
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// Number of unique output states discovered at this stage.
    pub fn distinct_outputs(&self) -> usize {
        self.distinct_outputs
    }

    /// Number of distinct `(input_state, output_state)` transitions at this
    /// stage.
    pub fn transition_count(&self) -> usize {
        self.transition_count
    }

    /// Total number of concrete gadgets stored across this stage's
    /// transitions.
    pub fn total_gadgets(&self) -> usize {
        self.total_gadgets
    }

    /// Denominator used by progress renderers for this stage.
    ///
    /// This is usually the number of known synthesis jobs for the current pass,
    /// but it is always at least the current attempts and output counts.
    pub fn progress_total(&self) -> usize {
        self.progress_total
    }

    /// Workers currently executing synthesis jobs for this stage.
    pub fn active_workers(&self) -> usize {
        self.active_workers
    }

    /// Configured worker capacity available to this stage.
    pub fn worker_capacity(&self) -> usize {
        self.worker_capacity
    }

    /// Known synthesis jobs waiting or in flight for this stage.
    pub fn queued_jobs(&self) -> usize {
        self.queued_jobs
    }
}

/// Progress for asynchronous rough scoring.
///
/// Rough scoring is the cheap, online pruning phase that runs while synthesis
/// discovers complete paths. A "path" in this snapshot is a transition-table
/// path submitted for rough scoring; one path may produce multiple scored
/// candidates when equal-cost or near-tie gadget assignments are expanded.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoringProgressSnapshot {
    completed_paths: usize,
    total_paths: usize,
    pending_paths: usize,
    scored_paths: usize,
}

impl ScoringProgressSnapshot {
    /// Builds a rough-scoring progress snapshot.
    ///
    /// `total_paths` is clamped to at least `completed_paths + pending_paths`
    /// because new complete paths can be discovered while scoring is already
    /// running.
    pub fn new(
        completed_paths: usize,
        total_paths: usize,
        pending_paths: usize,
        scored_paths: usize,
    ) -> Self {
        Self {
            completed_paths,
            total_paths: total_paths.max(completed_paths + pending_paths),
            pending_paths,
            scored_paths,
        }
    }

    /// Number of submitted path-scoring jobs whose results have been applied.
    pub fn completed_paths(&self) -> usize {
        self.completed_paths
    }

    /// Current denominator for submitted plus known rough-scoring path work.
    pub fn total_paths(&self) -> usize {
        self.total_paths
    }

    /// Path-scoring jobs waiting to be submitted or waiting for a result.
    pub fn pending_paths(&self) -> usize {
        self.pending_paths
    }

    /// Number of scored candidate assignments retained by rough scoring.
    pub fn scored_paths(&self) -> usize {
        self.scored_paths
    }
}

/// Progress for live full uiCA scoring after rough pruning.
///
/// This snapshot tracks the expensive scoring phase that evaluates rough
/// survivors with uiCA. It exists separately from [`ScoringProgressSnapshot`]
/// because full scoring has a different cost model, a different candidate set,
/// and exposes the best full-score/cycles estimate found so far.
#[derive(Clone, Debug, PartialEq)]
pub struct FullScoringProgressSnapshot {
    completed_paths: usize,
    total_paths: usize,
    pending_paths: usize,
    scored_paths: usize,
    best_score: Option<f64>,
    best_estimated_cycles: Option<f64>,
}

impl FullScoringProgressSnapshot {
    /// Builds a live full-scoring progress snapshot.
    ///
    /// `total_paths` is clamped to at least `completed_paths + pending_paths`
    /// for the same reason as rough scoring: the producer can discover and
    /// enqueue additional candidates while scoring workers are active.
    pub fn new(
        completed_paths: usize,
        total_paths: usize,
        pending_paths: usize,
        scored_paths: usize,
        best_score: Option<f64>,
        best_estimated_cycles: Option<f64>,
    ) -> Self {
        Self {
            completed_paths,
            total_paths: total_paths.max(completed_paths + pending_paths),
            pending_paths,
            scored_paths,
            best_score,
            best_estimated_cycles,
        }
    }

    /// Number of full-scoring jobs whose results have been applied.
    pub fn completed_paths(&self) -> usize {
        self.completed_paths
    }

    /// Current denominator for known full-scoring work.
    pub fn total_paths(&self) -> usize {
        self.total_paths
    }

    /// Full-scoring jobs waiting to be submitted or waiting for a result.
    pub fn pending_paths(&self) -> usize {
        self.pending_paths
    }

    /// Number of fully scored candidates available to the top-k reducer.
    pub fn scored_paths(&self) -> usize {
        self.scored_paths
    }

    /// Best full-score value observed so far, if any candidate has completed.
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Best uiCA cycles/throughput estimate observed so far, if available.
    pub fn best_estimated_cycles(&self) -> Option<f64> {
        self.best_estimated_cycles
    }
}

/// Runtime notification emitted by the solver pipeline.
///
/// Runtime events are the boundary between the core solver and presentation
/// layers such as line logging, the TUI, tests, and trace-oriented tooling.
/// Events are intentionally higher-level than internal trace JSON events:
/// they describe the user-visible phase of a solve, progress snapshots for
/// synthesis and scoring, and export milestones.
///
/// The rough-search phase is driven by waves of Z3 synthesis work. Complete
/// paths discovered during that phase can be rough-scored asynchronously via
/// [`RuntimeEvent::ScoringProgress`]. After rough search finishes, the final
/// uiCA phase emits the `FullScoring*` events and may also emit
/// [`RuntimeEvent::LiveFullScoringProgress`] while live full scoring runs
/// concurrently with late rough-scoring results.
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeEvent {
    /// A solver run has started and initialized `stage_count` synthesis stages.
    RunStarted { stage_count: usize },
    /// Work for one target CPU has started.
    ///
    /// Multi-target runs emit this once per CPU. `rough_candidate_limit` is
    /// the expanded top-k used for rough pruning before final scoring trims to
    /// `final_top_k`.
    TargetStarted {
        target_cpu: String,
        rough_candidate_limit: usize,
        final_top_k: usize,
    },
    /// A synthesis wave has started and will prioritize `target_stage`.
    WaveStarted { wave: usize, target_stage: usize },
    /// Updated synthesis progress for a single stage.
    StageUpdated(StageProgressSnapshot),
    /// Updated progress for asynchronous rough scoring.
    ScoringProgress(ScoringProgressSnapshot),
    /// Updated progress for live full uiCA scoring.
    LiveFullScoringProgress(FullScoringProgressSnapshot),
    /// A synthesis wave has finished applying its generated worker results.
    WaveFinished {
        wave: usize,
        target_stage: usize,
        attempts: usize,
        valid_outputs: usize,
        new_outputs: usize,
        discovered_paths: usize,
        scored_paths: usize,
    },
    /// Rough search for a target CPU has finished.
    ///
    /// This event summarizes synthesis waves and rough-scored candidates before
    /// final uiCA scoring begins.
    RoughSearchFinished {
        target_cpu: String,
        wave_count: usize,
        search_exhausted: bool,
        rough_candidate_count: usize,
        best_rough_score: Option<f64>,
    },
    /// Final uiCA scoring has started for the rough-pruned candidate set.
    FullScoringStarted {
        target_cpu: String,
        candidate_count: usize,
        final_top_k: usize,
    },
    /// A specific final-scoring candidate is about to be evaluated.
    FullScoringCandidateStarted {
        target_cpu: String,
        candidate_index: usize,
        candidate_count: usize,
        rough_score: f64,
    },
    /// Synchronous final-scoring progress update.
    ///
    /// Live/asynchronous full scoring uses
    /// [`RuntimeEvent::LiveFullScoringProgress`] instead.
    FullScoringProgress {
        target_cpu: String,
        scored_count: usize,
        candidate_count: usize,
        best_score: Option<f64>,
    },
    /// Final uiCA scoring has finished and the best candidates have been kept.
    FullScoringFinished {
        target_cpu: String,
        scored_count: usize,
        kept_count: usize,
        best_score: Option<f64>,
    },
    /// Writing an output artifact has started.
    ExportStarted {
        target_cpu: String,
        kind: String,
        path: String,
    },
    /// Writing an output artifact has completed.
    ExportFinished {
        target_cpu: String,
        kind: String,
        path: String,
    },
    /// The full run has completed.
    RunFinished {
        wave_count: usize,
        search_exhausted: bool,
        scored_paths: usize,
        best_score: Option<f64>,
    },
}

/// Consumer interface for runtime progress events.
///
/// The solver takes a mutable [`RuntimeSession`] rather than hard-coding a UI.
/// Implementations can ignore events, print line-oriented logs, forward events
/// to a TUI thread, or record them in tests. `should_stop` is polled by the
/// solver so interactive sessions can request cooperative cancellation.
pub trait RuntimeSession {
    /// Handles one runtime event emitted by the solver.
    fn on_event(&mut self, event: RuntimeEvent);

    /// Returns whether the solver should stop at the next cooperative
    /// cancellation point.
    fn should_stop(&self) -> bool {
        false
    }
}

/// Runtime session that discards all events and never cancels.
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
    /// Creates a session that forwards events over `sender`.
    ///
    /// `cancel_flag` is shared with the UI/controller thread and is read by
    /// [`RuntimeSession::should_stop`].
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

/// Runtime session that writes human-readable progress lines to a writer.
///
/// This is the non-interactive logging implementation. It coalesces frequent
/// stage updates so stdout does not dominate long synthesis runs.
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
            RuntimeEvent::ScoringProgress(snapshot) => {
                self.write_line(format!(
                    "runtime: rough scoring paths {}/{} pending {} scored candidates {}",
                    snapshot.completed_paths(),
                    snapshot.total_paths(),
                    snapshot.pending_paths(),
                    snapshot.scored_paths()
                ));
            }
            RuntimeEvent::LiveFullScoringProgress(snapshot) => {
                let best_score = snapshot
                    .best_score()
                    .map(|score| score.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                let best_cycles = snapshot
                    .best_estimated_cycles()
                    .map(|cycles| cycles.to_string())
                    .unwrap_or_else(|| "none".to_owned());
                self.write_line(format!(
                    "runtime: live full scoring paths {}/{} pending {} scored candidates {}, best score {best_score}, best cycles/tp {best_cycles}",
                    snapshot.completed_paths(),
                    snapshot.total_paths(),
                    snapshot.pending_paths(),
                    snapshot.scored_paths()
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
