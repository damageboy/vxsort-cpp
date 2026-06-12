use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    env, fs,
    io::{self, ErrorKind, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use gadget_synth::{
    Arch as SynthArch, CandidateGraphTiers, DType as SynthDType, GadgetGraph, GadgetSynthesizer,
    InstructionSpec, LaneLabel, PermutationGadget, SynthesisError, SynthesisOptions, TargetPair,
    VectorState,
};

use crate::bitonic_sorter::{BitonicSorter, BitonicStage};
use crate::runtime::{
    RuntimeEvent, RuntimeSession, ScoringProgressSnapshot, StageProgressSnapshot,
};
use crate::runtime_trace::RuntimeTrace;
use crate::scoring::{
    AssignedPath, AssignedPathKey, DummyScorer, PathCost, PathScoringSnapshot, Scorer,
};
use crate::transition_table::{
    CompletePath, PathId, PathRegistry, StateId, TransitionRef, TransitionTable,
};
use crate::{ArchArg, DTypeArg, DeepSearchModeArg, WorkerBackendArg};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

const SYNTHESIS_WORKER_COMMAND: &str = "__synthesis-worker";
const ROUGH_TIE_EXPANSION_FACTOR: usize = 10;
const SCORING_RESULT_POLL_INTERVAL_PER_SYNTHESIS_JOBS: usize = 1024;
const SCORING_RESULT_POLL_LIMIT_PER_SYNTHESIS_INTERVAL: usize = 1024;
const SCORING_RESULT_POLL_LIMIT_PER_WAVE_BOUNDARY: usize = 1024;
const SCORING_OUTSTANDING_JOBS_PER_WORKER: usize = 4;
const MIN_SCORING_OUTSTANDING_JOBS: usize = 64;
const MAX_SCORING_OUTSTANDING_JOBS: usize = 512;
const SCORING_TRACE_DETAIL_LIMIT: usize = 1024;
const SCORING_TRACE_DETAIL_INTERVAL: usize = 1024;
const PROCESS_WORKER_RECYCLE_JOB_LIMIT: usize = 50_000;
const PROCESS_WORKER_RSS_TRACE_INTERVAL: usize = 16_384;
const PROCESS_WORKER_RSS_TRACE_IDLE_INTERVAL: Duration = Duration::from_secs(10);

/// Configuration for one wave-engine solve.
///
/// The wave engine owns the iterative search over bitonic-sort stages. It uses
/// this configuration to choose the vector lane shape, the Z3 gadget search
/// depth, worker scheduling mode, and how aggressively each transition may
/// retain concrete gadgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaveConfig {
    /// Number of vector registers sorted together.
    pub num_vecs: usize,
    /// Target SIMD instruction set.
    pub arch: ArchArg,
    /// Element type being sorted.
    pub dtype: DTypeArg,
    /// Maximum candidate gadget depth passed to the synthesizer.
    pub gadget_depth: u8,
    /// Search policy for spending depth-2 synthesis work.
    pub deep_search_mode: DeepSearchModeArg,
    /// Whether to append the final reorder stage that restores natural lane
    /// order.
    pub natural_order: bool,
    /// Whether stage 1 should be seeded from a lazy retroactive permutation
    /// source instead of only forwarding from the initial state.
    pub retroactive_input: bool,
    /// Number of rough-scored paths to keep. `None` means no top-k pruning.
    pub top_k: Option<usize>,
    /// Requested synthesis/scoring worker count after CLI resolution.
    pub worker_count: usize,
    /// In-process vs subprocess synthesis worker strategy.
    pub worker_backend: WorkerBackendArg,
    /// Maximum concrete gadget solutions retained for a single transition.
    pub max_unique_outputs: usize,
}

/// Stateful wave-search engine for synthesizing and scoring sorter paths.
///
/// `WaveEngine` maintains the transition table, bitonic stage definitions,
/// precomputed candidate gadget graphs, process-backed synthesis workers, and
/// asynchronous rough-scoring pool. A run proceeds in waves: each wave tries to
/// expand one target stage, propagates any new states forward, registers newly
/// complete paths, and schedules those paths for scoring.
///
/// The engine is intentionally stateful. Most counters and queues represent
/// live scheduler state, so callers should use the public snapshot/result APIs
/// instead of trying to infer progress from individual fields.
pub struct WaveEngine {
    /// Normalized solve configuration. `worker_count` is clamped to at least 1.
    config: WaveConfig,
    /// SIMD lanes in one vector for the selected architecture and dtype.
    elements_per_vector: usize,
    /// Total scalar elements being sorted across all configured vectors.
    total_elements: usize,
    /// Ordered bitonic stages, plus the optional natural-order restore stage.
    stages: Vec<BitonicStage>,
    /// Zero-based comparator target pairs for each stage.
    stage_target_pairs: Vec<Box<[TargetPair]>>,
    /// Upper bound on distinct output states each stage can produce.
    stage_output_limits: Vec<usize>,
    /// Candidate gadget graphs in the low-depth tier.
    shallow_candidates: Vec<GadgetGraph>,
    /// Candidate gadget graphs with a shared prefix and depth above one.
    shared_prefix_deep_candidates: Vec<GadgetGraph>,
    /// Candidate gadget graphs in the full higher-depth tier.
    full_deep_candidates: Vec<GadgetGraph>,
    /// Compatibility view of all higher-depth candidates.
    deep_candidates: Vec<GadgetGraph>,
    /// Lazy source for `--retroactive-input` stage-1 jobs.
    retroactive_input_source: Option<RetroactiveInputSource>,
    /// Interned states, transitions, gadgets, and per-stage attempt statistics.
    transition_table: TransitionTable,
    /// UI progress denominators, indexed by stage.
    stage_progress_totals: Vec<usize>,
    /// Currently active synthesis workers, indexed by stage.
    active_worker_counts: Vec<usize>,
    /// Current worker capacity advertised to the UI, indexed by stage.
    worker_capacities: Vec<usize>,
    /// Queued synthesis jobs not yet running, indexed by stage.
    queued_job_counts: Vec<usize>,
    /// Initial comparison-pair state built from the first bitonic stage.
    initial_state: VectorState,
    /// Interned id for `initial_state` in `transition_table`.
    initial_state_id: StateId,
    /// Number of completed wave iterations.
    wave_count: usize,
    /// Stages known to have no remaining search work.
    exhausted_stages: HashSet<usize>,
    /// Stages that cannot run now but may receive upstream inputs later.
    stalled_stages: HashSet<usize>,
    /// Assigned-path selection keys already accepted into `scored_paths`.
    scored_path_keys: HashSet<AssignedPathKey>,
    /// Retained top scored paths across the run.
    scored_paths: Vec<ScoredPath>,
    /// Recently scored paths retained for incremental progress reporting.
    recent_scored_paths: Vec<ScoredPath>,
    /// Bounded async pool used for rough path scoring.
    scoring_pool: ScoringPool,
    /// Monotonic submission order assigned to the next scoring job.
    next_scoring_job_order: usize,
    /// Lowest scoring-result order not yet applied.
    next_scoring_result_order: usize,
    /// Completed scoring results waiting for earlier orders to arrive.
    buffered_scoring_results: BTreeMap<usize, ScoringJobResult>,
    /// Last gadget-count signature scored for each registered path.
    latest_scoring_signatures: HashMap<PathId, PathScoringSignature>,
    /// FIFO backlog of registered paths waiting for scoring-pool capacity.
    pending_scoring_path_ids: VecDeque<PathId>,
    /// Membership set for `pending_scoring_path_ids`.
    pending_scoring_path_set: HashSet<PathId>,
    /// Transitions whose registered paths need rescoring after gadget growth.
    pending_registered_rescore_transitions: BTreeSet<TransitionRef>,
    /// Stable complete-path store and reverse transition-to-path index.
    path_registry: PathRegistry,
    /// Idle subprocess synthesis workers kept warm between stage runs.
    process_workers: Vec<ProcessSynthesisWorker>,
    /// Number of adaptive full-depth attempts already made by `(stage, input)`.
    adaptive_deep_attempts: HashMap<(usize, StateId), usize>,
    /// Runtime JSON trace sink. Disabled by default.
    trace: RuntimeTrace,
}

/// A complete path with one selected gadget per transition and its score.
///
/// Rough scoring can expand a single [`CompletePath`] into multiple
/// [`ScoredPath`] values when several gadget choices tie or are close enough
/// to survive top-k expansion. `discovery_order` preserves deterministic
/// ordering across asynchronous scoring completion.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPath {
    path_id: PathId,
    path: CompletePath,
    assigned_path: AssignedPath,
    cost: PathCost,
    discovery_order: usize,
}

impl ScoredPath {
    pub fn new(
        path_id: PathId,
        path: CompletePath,
        assigned_path: AssignedPath,
        cost: PathCost,
        discovery_order: usize,
    ) -> Self {
        Self {
            path_id,
            path,
            assigned_path,
            cost,
            discovery_order,
        }
    }

    pub fn path_id(&self) -> PathId {
        self.path_id
    }

    pub fn path(&self) -> &CompletePath {
        &self.path
    }

    pub fn assigned_path(&self) -> &AssignedPath {
        &self.assigned_path
    }

    pub fn cost(&self) -> &PathCost {
        &self.cost
    }

    pub fn discovery_order(&self) -> usize {
        self.discovery_order
    }

    pub fn with_cost(&self, cost: PathCost) -> Self {
        Self {
            path_id: self.path_id,
            path: self.path.clone(),
            assigned_path: self.assigned_path.clone(),
            cost,
            discovery_order: self.discovery_order,
        }
    }
}

/// Result of inserting one synthesized transition output.
///
/// This is the narrow result returned after a `(gadget, output_state)` pair is
/// applied to the transition table. It separates "a transition edge was new"
/// from "this edge completed one or more full paths".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionRecordResult {
    transition_added: bool,
    budget_output_added: bool,
    discovered_paths: usize,
}

impl TransitionRecordResult {
    pub fn transition_added(&self) -> bool {
        self.transition_added
    }

    pub fn budget_output_added(&self) -> bool {
        self.budget_output_added
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
    }
}

/// Stable identity for one `(stage, input, candidate)` synthesis attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SynthesisJob {
    stage: usize,
    input: StateId,
    candidate_index: usize,
    retroactive_rank: Option<u128>,
}

impl SynthesisJob {
    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn input(&self) -> StateId {
        self.input
    }

    pub fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    fn retroactive_rank(&self) -> Option<u128> {
        self.retroactive_rank
    }
}

#[derive(Debug)]
struct PreparedStageRun {
    stage: usize,
    jobs: Vec<SynthesisJob>,
    outputs_at_start: usize,
    attempts_at_start: usize,
    attempt_budget: usize,
    output_budget: usize,
}

/// Lazy factorial-rank source for retroactive stage-0 inputs.
///
/// The first bitonic stage consists of independent compare pairs. Retroactive
/// input treats any ordering of those pairs as a valid zero-cost prefix. This
/// source keeps that factorial space virtual and materializes only the ranks
/// needed by the current stage-1 page.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RetroactiveInputSource {
    pairs: Box<[(LaneLabel, LaneLabel)]>,
    factorials: Box<[u128]>,
    total_ranks: u128,
    next_rank_by_candidate: Vec<u128>,
}

impl RetroactiveInputSource {
    fn new(first_stage: &BitonicStage, candidate_count: usize) -> Self {
        let pairs = first_stage
            .pairs()
            .iter()
            .map(|(top, bottom)| {
                (
                    lane_label_from_one_based(*top),
                    lane_label_from_one_based(*bottom),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let factorials = checked_factorials_u128(pairs.len());
        let total_ranks = factorials[pairs.len()];

        Self {
            pairs,
            factorials,
            total_ranks,
            next_rank_by_candidate: vec![0; candidate_count],
        }
    }

    fn total_ranks(&self) -> u128 {
        self.total_ranks
    }

    fn next_rank(&self, candidate_index: usize) -> u128 {
        self.next_rank_by_candidate
            .get(candidate_index)
            .copied()
            .unwrap_or(self.total_ranks)
    }

    fn set_next_rank(&mut self, candidate_index: usize, rank: u128) {
        if let Some(next_rank) = self.next_rank_by_candidate.get_mut(candidate_index) {
            *next_rank = rank.min(self.total_ranks);
        }
    }

    fn state_for_rank(&self, rank: u128) -> VectorState {
        assert!(
            rank < self.total_ranks,
            "retroactive input rank should be inside factorial space"
        );

        let mut rank = rank;
        let mut remaining = self.pairs.to_vec();
        let mut ordered = Vec::with_capacity(remaining.len());

        for slot_count in (1..=self.pairs.len()).rev() {
            let chunk = self.factorials[slot_count - 1];
            let index =
                usize::try_from(rank / chunk).expect("factorial rank digit should fit in usize");
            rank %= chunk;
            ordered.push(remaining.remove(index));
        }

        VectorState::new(
            ordered.iter().map(|(top, _)| *top).collect(),
            ordered.iter().map(|(_, bottom)| *bottom).collect(),
        )
    }
}

/// In-process representation of a synthesis job with all execution payloads.
///
/// This is produced from [`SynthesisJob`] before handing work to either a
/// thread-backed worker or subprocess-backed worker. `target_pairs` is the
/// stage comparator set in zero-based lane labels.
#[derive(Clone, Debug, PartialEq)]
struct WorkerSynthesisJob {
    index: usize,
    id: SynthesisJob,
    input_state: VectorState,
    graph: GadgetGraph,
    target_pairs: Box<[TargetPair]>,
    allow_any_lane_order: bool,
}

/// Static configuration sent once to each synthesis subprocess.
///
/// Worker processes rebuild their own synthesizer and candidate graph cache
/// from this compact config. Keeping this separate from [`WaveConfig`] avoids
/// sending coordinator-only fields such as top-k and worker counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ProcessWorkerConfig {
    arch: ArchArg,
    dtype: DTypeArg,
    gadget_depth: u8,
    natural_order: bool,
    max_unique_outputs: usize,
}

impl From<WaveConfig> for ProcessWorkerConfig {
    fn from(config: WaveConfig) -> Self {
        Self {
            arch: config.arch,
            dtype: config.dtype,
            gadget_depth: config.gadget_depth,
            natural_order: config.natural_order,
            max_unique_outputs: config.max_unique_outputs,
        }
    }
}

/// Serializable synthesis job sent from the coordinator to a worker process.
///
/// This mirrors [`WorkerSynthesisJob`] but stores only data that can cross the
/// bincode IPC boundary. The worker reconstructs the candidate graph from
/// `candidate_index` and its local candidate cache.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ProcessWorkerJob {
    index: usize,
    stage: usize,
    input: u32,
    input_state: VectorState,
    candidate_index: usize,
    retroactive_rank: Option<u128>,
    target_pairs: Box<[TargetPair]>,
    allow_any_lane_order: bool,
}

impl ProcessWorkerJob {
    fn from_worker_job(worker_job: WorkerSynthesisJob) -> Self {
        Self {
            index: worker_job.index,
            stage: worker_job.id.stage,
            input: worker_job.id.input.0,
            input_state: worker_job.input_state,
            candidate_index: worker_job.id.candidate_index,
            retroactive_rank: worker_job.id.retroactive_rank,
            target_pairs: worker_job.target_pairs,
            allow_any_lane_order: worker_job.allow_any_lane_order,
        }
    }

    fn to_synthesis_job(&self) -> SynthesisJob {
        SynthesisJob {
            stage: self.stage,
            input: StateId(self.input),
            candidate_index: self.candidate_index,
            retroactive_rank: self.retroactive_rank,
        }
    }
}

/// Serializable worker-process response for a completed synthesis job.
///
/// Results are raw synthesized gadgets paired with their output states. The
/// coordinator is still responsible for applying them to the transition table
/// in candidate order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ProcessWorkerOutput {
    id: SynthesisJob,
    results: Vec<(PermutationGadget, VectorState)>,
}

impl ProcessWorkerOutput {
    fn into_synthesis_output(self) -> SynthesisJobOutput {
        SynthesisJobOutput {
            id: self.id,
            results: self.results,
        }
    }
}

/// Framed request protocol from the coordinator to a synthesis subprocess.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum ProcessWorkerRequest {
    Init(ProcessWorkerConfig),
    Run(ProcessWorkerJob),
    Shutdown,
}

/// Framed response protocol from a synthesis subprocess to the coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum ProcessWorkerResponse {
    Ready(Result<(), String>),
    Finished {
        index: usize,
        result: Result<ProcessWorkerOutput, String>,
    },
}

/// Non-serialized synthesis output used inside the coordinator.
///
/// Both in-process and subprocess worker paths normalize into this type before
/// the coordinator applies results to the transition table.
#[derive(Clone, Debug, PartialEq)]
struct SynthesisJobOutput {
    id: SynthesisJob,
    results: Vec<(PermutationGadget, VectorState)>,
}

/// Completion event emitted by an in-process worker thread.
///
/// Subprocess-backed execution has its own polling loop, but both paths track
/// start/finish events to maintain worker utilization counters.
#[derive(Clone, Debug, PartialEq)]
enum WorkerPoolEvent {
    Started(usize),
    Finished(usize, Result<SynthesisJobOutput, SynthesisError>),
}

/// Shared scorer instance used by asynchronous rough-scoring workers.
type SharedScorer = Arc<dyn Scorer>;

/// Compact fingerprint of a path's current gadget-list lengths.
///
/// The engine uses this to detect when an already-discovered path should be
/// rescored because a transition gained additional equal/near-tie gadgets.
type PathScoringSignature = Vec<(TransitionRef, usize)>;

/// Hook for observing wave progress and optionally scheduling additional work.
///
/// The production path uses this to bridge rough-search progress into live
/// full scoring. Tests can install custom observers to assert ordering and
/// cancellation behavior without coupling to the TUI.
pub trait WaveProgressObserver {
    fn on_wave_progress(
        &mut self,
        _engine: &WaveEngine,
        _recent_scored_paths: &[ScoredPath],
        _session: &mut dyn RuntimeSession,
        _reason: &str,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Work item for asynchronous rough scoring.
///
/// The job owns a [`PathScoringSnapshot`] so scorer threads can evaluate a
/// stable view of transition gadgets without cloning the full transition
/// table. `assignment_limit` is the per-path k-best expansion limit.
#[derive(Clone, Debug)]
struct ScoringJob {
    order: usize,
    path_id: PathId,
    path: CompletePath,
    snapshot: PathScoringSnapshot,
    assignment_limit: usize,
}

/// Result returned by a rough-scoring worker.
///
/// Results are buffered and applied in `order`, so scoring completion order
/// does not change deterministic top-k/discovery behavior.
#[derive(Clone, Debug)]
struct ScoringJobResult {
    order: usize,
    path_id: PathId,
    path_length: usize,
    scored_paths: Vec<ScoredPath>,
}

/// Small trace payload derived from a scoring result.
///
/// Keeping these fields separate avoids retaining full [`ScoredPath`] values
/// only for trace emission.
#[derive(Clone, Copy, Debug)]
struct ScoringJobTraceFields {
    order: usize,
    path_id: PathId,
    path_length: usize,
    assigned_candidates: usize,
}

/// Fixed-size thread pool for asynchronous rough scoring.
///
/// The pool intentionally bounds outstanding jobs so rough-scoring submission
/// cannot consume unbounded memory while synthesis continues to discover new
/// complete paths.
struct ScoringPool {
    sender: mpsc::Sender<Option<ScoringJob>>,
    receiver: mpsc::Receiver<ScoringJobResult>,
    workers: Vec<thread::JoinHandle<()>>,
    pending_jobs: usize,
    max_pending_jobs: usize,
}

/// Result of executing one synthesis job and applying its outputs.
///
/// This is the smallest execution result: it is per stage input/candidate
/// graph pair, before aggregation into stage or wave summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobExecutionResult {
    stage: usize,
    candidate_index: usize,
    attempts: usize,
    valid_output_count: usize,
    transition_count: usize,
    new_transition_count: usize,
    budget_output_count: usize,
    discovered_paths: usize,
}

impl JobExecutionResult {
    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    pub fn attempts(&self) -> usize {
        self.attempts
    }

    pub fn valid_output_count(&self) -> usize {
        self.valid_output_count
    }

    pub fn transition_count(&self) -> usize {
        self.transition_count
    }

    pub fn new_transition_count(&self) -> usize {
        self.new_transition_count
    }

    pub fn budget_output_count(&self) -> usize {
        self.budget_output_count
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
    }
}

/// Aggregated result for one stage pass.
///
/// A stage pass can execute many [`JobExecutionResult`] values. The aggregate
/// records the amount of synthesis attempted, how much new transition-table
/// state was found, and how many complete paths were discovered as a result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageRunResult {
    stage: usize,
    attempts: usize,
    valid_outputs: usize,
    new_outputs: usize,
    budget_outputs: usize,
    new_transitions: usize,
    discovered_paths: usize,
}

impl StageRunResult {
    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn attempts(&self) -> usize {
        self.attempts
    }

    pub fn valid_outputs(&self) -> usize {
        self.valid_outputs
    }

    pub fn new_outputs(&self) -> usize {
        self.new_outputs
    }

    pub fn budget_outputs(&self) -> usize {
        self.budget_outputs
    }

    pub fn new_transitions(&self) -> usize {
        self.new_transitions
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AdaptiveDeepRunResult {
    min_stage: usize,
    attempts: usize,
    valid_outputs: usize,
    new_outputs: usize,
    budget_outputs: usize,
    new_transitions: usize,
    discovered_paths: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AdaptiveTarget {
    stage: usize,
    input: StateId,
    topk_hits: usize,
    expensive_suffix_count: usize,
    priority: i64,
    exploratory: bool,
}

#[derive(Default)]
struct AdaptiveTargetAccumulator {
    topk_hits: usize,
    expensive_suffix_count: usize,
}

impl AdaptiveDeepRunResult {
    fn empty(stage_count: usize) -> Self {
        Self {
            min_stage: stage_count,
            attempts: 0,
            valid_outputs: 0,
            new_outputs: 0,
            budget_outputs: 0,
            new_transitions: 0,
            discovered_paths: 0,
        }
    }

    fn add_stage_result(&mut self, result: &StageRunResult) {
        self.min_stage = self.min_stage.min(result.stage());
        self.attempts += result.attempts();
        self.valid_outputs += result.valid_outputs();
        self.new_outputs += result.new_outputs();
        self.budget_outputs += result.budget_outputs();
        self.new_transitions += result.new_transitions();
        self.discovered_paths += result.discovered_paths();
    }
}

/// Result for one complete wave.
///
/// A wave has one target-stage expansion plus zero or more propagation stage
/// passes caused by newly discovered outputs. `scored_paths` counts rough
/// scoring results applied during the wave.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaveRunResult {
    wave: usize,
    target_stage: usize,
    target: StageRunResult,
    propagation: Vec<StageRunResult>,
    discovered_paths: usize,
    scored_paths: usize,
}

impl WaveRunResult {
    pub fn wave(&self) -> usize {
        self.wave
    }

    pub fn target_stage(&self) -> usize {
        self.target_stage
    }

    pub fn target(&self) -> &StageRunResult {
        &self.target
    }

    pub fn propagation(&self) -> &[StageRunResult] {
        &self.propagation
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
    }

    pub fn scored_paths(&self) -> usize {
        self.scored_paths
    }
}

/// Summary returned after running waves until exhaustion or a configured stop.
///
/// `search_exhausted` is true only when no stage remains eligible for frontier
/// expansion; a `max_waves` limit can end the search with this set to false.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaveSearchResult {
    wave_count: usize,
    search_exhausted: bool,
    waves: Vec<WaveRunResult>,
}

impl WaveSearchResult {
    pub fn wave_count(&self) -> usize {
        self.wave_count
    }

    pub fn search_exhausted(&self) -> bool {
        self.search_exhausted
    }

    pub fn waves(&self) -> &[WaveRunResult] {
        &self.waves
    }
}

impl ScoringPool {
    fn new_shared(worker_count: usize, scorer: SharedScorer) -> Self {
        let worker_count = worker_count.max(1);
        let max_pending_jobs = worker_count
            .saturating_mul(SCORING_OUTSTANDING_JOBS_PER_WORKER)
            .clamp(MIN_SCORING_OUTSTANDING_JOBS, MAX_SCORING_OUTSTANDING_JOBS);
        let (sender, job_receiver) = mpsc::channel::<Option<ScoringJob>>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (result_sender, receiver) = mpsc::channel::<ScoringJobResult>();
        let workers = (0..worker_count)
            .map(|_| {
                let scorer = Arc::clone(&scorer);
                let job_receiver = Arc::clone(&job_receiver);
                let result_sender = result_sender.clone();
                thread::spawn(move || {
                    loop {
                        let message = {
                            let receiver = job_receiver
                                .lock()
                                .expect("scoring job receiver lock should not be poisoned");
                            receiver.recv()
                        };
                        let Ok(Some(job)) = message else {
                            break;
                        };
                        let result = score_job(scorer.as_ref(), job);
                        if result_sender.send(result).is_err() {
                            break;
                        }
                    }
                })
            })
            .collect();

        Self {
            sender,
            receiver,
            workers,
            pending_jobs: 0,
            max_pending_jobs,
        }
    }

    fn submit(&mut self, job: ScoringJob) {
        self.sender
            .send(Some(job))
            .expect("scoring worker channel should stay open while engine is active");
        self.pending_jobs += 1;
    }

    fn has_capacity(&self) -> bool {
        self.pending_jobs < self.max_pending_jobs
    }
}

fn flatten_candidate_tiers(tiers: &CandidateGraphTiers) -> Vec<GadgetGraph> {
    let mut candidates = tiers.shallow.clone();
    candidates.extend(tiers.shared_prefix_deep.clone());
    candidates.extend(tiers.full_deep.clone());
    candidates
}

fn normal_candidate_count_for_mode(
    mode: DeepSearchModeArg,
    shallow_count: usize,
    shared_prefix_count: usize,
    full_deep_count: usize,
) -> usize {
    match mode {
        DeepSearchModeArg::Baseline => shallow_count + shared_prefix_count + full_deep_count,
        DeepSearchModeArg::SharedPrefix | DeepSearchModeArg::Adaptive => {
            shallow_count + shared_prefix_count
        }
    }
}

fn deterministic_jitter(stage: usize, input: StateId) -> i64 {
    let mut value = (stage as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(u64::from(input.0));
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    (value % 10) as i64
}

fn expensive_gadget_count(gadget: &PermutationGadget) -> usize {
    gadget
        .top_instructions()
        .iter()
        .chain(gadget.bottom_instructions())
        .filter(|instruction| is_expensive_permute_instruction(instruction))
        .count()
}

fn is_expensive_permute_instruction(instruction: &InstructionSpec) -> bool {
    let name = instruction.intrinsic_name();
    name.contains("permutexvar")
        || name.contains("permutevar")
        || name.contains("permutex2var")
        || instruction.args().contains_key("op_idx")
}

impl Drop for ScoringPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            let _ = self.sender.send(None);
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

/// Coordinator-side handle to a long-lived synthesis subprocess.
///
/// This is the per-worker resource pooled by
/// [`run_stage_worker_pool`](WaveEngine::run_stage_worker_pool) when the process
/// backend is selected. It owns a child process and the pipes used to drive it.
///
/// # Protocol
///
/// The worker is spawned as the *same executable* re-invoked with the hidden
/// `__synthesis-worker` subcommand (see [`run_synthesis_worker_stdio`]). The
/// coordinator and worker exchange length-framed messages over the child's
/// stdin/stdout: an [`Init`](ProcessWorkerRequest::Init) handshake answered by
/// [`Ready`](ProcessWorkerResponse::Ready), then a sequence of
/// [`Run`](ProcessWorkerRequest::Run) requests each answered by a
/// [`Finished`](ProcessWorkerResponse::Finished) response, and finally a
/// [`Shutdown`](ProcessWorkerRequest::Shutdown). The child's stderr is
/// inherited so worker diagnostics surface on the parent's terminal.
///
/// # Why subprocesses
///
/// Running synthesis out-of-process isolates expensive Z3 state and any
/// process-global locks per worker, and lets the coordinator *recycle* a worker
/// after [`PROCESS_WORKER_RECYCLE_JOB_LIMIT`] jobs
/// ([`should_recycle`](Self::should_recycle)) to bound external allocator / Z3
/// memory growth.
struct ProcessSynthesisWorker {
    /// The spawned worker process.
    child: Child,
    /// Pipe carrying framed requests to the worker.
    stdin: ChildStdin,
    /// Pipe carrying framed responses from the worker.
    stdout: ChildStdout,
    /// Jobs run since spawn; drives the recycle decision.
    jobs_since_spawn: usize,
}

impl ProcessSynthesisWorker {
    /// Spawns a worker subprocess and completes the init handshake.
    ///
    /// Re-launches the current executable with the `__synthesis-worker`
    /// subcommand, pipes its stdin/stdout (inheriting stderr), and sends an
    /// [`Init`](ProcessWorkerRequest::Init) frame carrying the solve `config`.
    /// Returns the ready worker once it acknowledges with
    /// [`Ready(Ok)`](ProcessWorkerResponse::Ready); any spawn, framing, or
    /// initialization failure — including the worker reporting an error or
    /// exiting early — is returned as [`SynthesisError::WorkerProcess`].
    fn spawn(config: WaveConfig) -> Result<Self, SynthesisError> {
        let current_exe = env::current_exe().map_err(|error| {
            SynthesisError::WorkerProcess(format!("failed to locate current executable: {error}"))
        })?;
        let mut child = Command::new(current_exe)
            .arg(SYNTHESIS_WORKER_COMMAND)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                SynthesisError::WorkerProcess(format!("failed to spawn synthesis worker: {error}"))
            })?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            SynthesisError::WorkerProcess("worker stdin was not piped".to_owned())
        })?;
        let mut stdout = child.stdout.take().ok_or_else(|| {
            SynthesisError::WorkerProcess("worker stdout was not piped".to_owned())
        })?;

        write_frame(
            &mut stdin,
            &ProcessWorkerRequest::Init(ProcessWorkerConfig::from(config)),
        )
        .map_err(|error| {
            SynthesisError::WorkerProcess(format!("failed to initialize worker: {error}"))
        })?;
        match read_frame::<_, ProcessWorkerResponse>(&mut stdout).map_err(|error| {
            SynthesisError::WorkerProcess(format!("failed to read worker ready response: {error}"))
        })? {
            Some(ProcessWorkerResponse::Ready(Ok(()))) => Ok(Self {
                child,
                stdin,
                stdout,
                jobs_since_spawn: 0,
            }),
            Some(ProcessWorkerResponse::Ready(Err(error))) => {
                Err(SynthesisError::WorkerProcess(error))
            }
            Some(other) => Err(SynthesisError::WorkerProcess(format!(
                "unexpected worker initialization response: {other:?}"
            ))),
            None => Err(SynthesisError::WorkerProcess(
                "worker exited before initialization completed".to_owned(),
            )),
        }
    }

    /// Runs a single job on the worker and waits for its result.
    ///
    /// Sends the `worker_job` as a [`Run`](ProcessWorkerRequest::Run) frame,
    /// blocks for the matching [`Finished`](ProcessWorkerResponse::Finished)
    /// response, and returns the synthesized output. Increments
    /// `jobs_since_spawn` (feeding [`should_recycle`](Self::should_recycle)). A
    /// worker-side error, an unexpected response, or the worker exiting before
    /// replying is surfaced as [`SynthesisError::WorkerProcess`].
    fn run_job(
        &mut self,
        worker_job: WorkerSynthesisJob,
    ) -> Result<SynthesisJobOutput, SynthesisError> {
        self.jobs_since_spawn += 1;
        let request = ProcessWorkerRequest::Run(ProcessWorkerJob::from_worker_job(worker_job));
        write_frame(&mut self.stdin, &request).map_err(|error| {
            SynthesisError::WorkerProcess(format!("failed to send worker job: {error}"))
        })?;
        match read_frame::<_, ProcessWorkerResponse>(&mut self.stdout).map_err(|error| {
            SynthesisError::WorkerProcess(format!("failed to read worker job result: {error}"))
        })? {
            Some(ProcessWorkerResponse::Finished {
                result: Ok(output), ..
            }) => Ok(output.into_synthesis_output()),
            Some(ProcessWorkerResponse::Finished {
                result: Err(error), ..
            }) => Err(SynthesisError::WorkerProcess(error)),
            Some(other) => Err(SynthesisError::WorkerProcess(format!(
                "unexpected worker job response: {other:?}"
            ))),
            None => Err(SynthesisError::WorkerProcess(
                "worker exited before sending job result".to_owned(),
            )),
        }
    }

    /// Asks the worker to exit and waits for it to reap.
    ///
    /// Consumes the handle: sends a best-effort
    /// [`Shutdown`](ProcessWorkerRequest::Shutdown) frame, closes stdin (so the
    /// worker's read loop ends even if the frame was lost), and waits on the
    /// child. Errors are intentionally ignored — this is cleanup.
    fn shutdown(mut self) {
        let _ = write_frame(&mut self.stdin, &ProcessWorkerRequest::Shutdown);
        drop(self.stdin);
        let _ = self.child.wait();
    }

    /// Operating-system process id of the worker, used for RSS tracing.
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Number of jobs this worker has run since it was spawned.
    fn jobs_since_spawn(&self) -> usize {
        self.jobs_since_spawn
    }

    /// Whether the worker has reached [`PROCESS_WORKER_RECYCLE_JOB_LIMIT`] and
    /// should be torn down and replaced to bound memory growth.
    fn should_recycle(&self) -> bool {
        self.jobs_since_spawn >= PROCESS_WORKER_RECYCLE_JOB_LIMIT
    }

    /// Resident set size of the worker process in kilobytes, if available on
    /// this platform.
    fn rss_kb(&self) -> Option<u64> {
        process_rss_kb(self.pid())
    }
}

/// Runs the hidden synthesis-worker subprocess protocol on stdin/stdout.
///
/// This is invoked by the main executable through the private
/// `__synthesis-worker` subcommand. It is public so the CLI layer can dispatch
/// to it, but normal callers should create a [`WaveEngine`] and use the
/// process worker backend instead of calling this directly.
pub fn run_synthesis_worker_stdio() -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_synthesis_worker_stdio_impl(&mut stdin.lock(), &mut stdout.lock())
}

fn run_synthesis_worker_stdio_impl<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> Result<(), String> {
    let Some(ProcessWorkerRequest::Init(config)) =
        read_frame::<_, ProcessWorkerRequest>(reader).map_err(|error| error.to_string())?
    else {
        return Err("worker expected an Init request".to_owned());
    };

    let runtime = match ProcessWorkerRuntime::new(config) {
        Ok(runtime) => {
            write_frame(writer, &ProcessWorkerResponse::Ready(Ok(())))
                .map_err(|error| error.to_string())?;
            runtime
        }
        Err(error) => {
            write_frame(
                writer,
                &ProcessWorkerResponse::Ready(Err(error.to_string())),
            )
            .map_err(|error| error.to_string())?;
            return Ok(());
        }
    };

    loop {
        match read_frame::<_, ProcessWorkerRequest>(reader).map_err(|error| error.to_string())? {
            Some(ProcessWorkerRequest::Run(job)) => {
                let index = job.index;
                let result = runtime.run_job(job).map_err(|error| error.to_string());
                write_frame(writer, &ProcessWorkerResponse::Finished { index, result })
                    .map_err(|error| error.to_string())?;
            }
            Some(ProcessWorkerRequest::Shutdown) | None => break,
            Some(ProcessWorkerRequest::Init(_)) => {
                return Err("worker received duplicate Init request".to_owned());
            }
        }
    }

    Ok(())
}

/// Runtime state owned inside a synthesis subprocess.
///
/// Each subprocess builds its own [`GadgetSynthesizer`] and candidate graph
/// cache. This is the point of the process-backed backend: expensive Z3 state
/// and process-global locks are isolated per worker process rather than shared
/// inside the main coordinator process.
struct ProcessWorkerRuntime {
    config: ProcessWorkerConfig,
    candidates: Vec<GadgetGraph>,
    synth: GadgetSynthesizer,
}

impl ProcessWorkerRuntime {
    fn new(config: ProcessWorkerConfig) -> Result<Self, SynthesisError> {
        let synth =
            GadgetSynthesizer::new(to_synth_arch(config.arch), to_synth_dtype(config.dtype));
        let tiers = synth.precompute_candidate_tiers(config.gadget_depth)?;
        let candidates = flatten_candidate_tiers(&tiers);
        Ok(Self {
            config,
            candidates,
            synth,
        })
    }

    fn run_job(&self, job: ProcessWorkerJob) -> Result<ProcessWorkerOutput, SynthesisError> {
        let graph = self.candidates.get(job.candidate_index).ok_or_else(|| {
            SynthesisError::WorkerProcess(format!(
                "candidate index {} is out of range for {} candidates",
                job.candidate_index,
                self.candidates.len()
            ))
        })?;
        let results = self.synth.synthesize_graph(
            graph,
            &job.input_state,
            &job.target_pairs,
            SynthesisOptions {
                max_unique_outputs: self.config.max_unique_outputs,
                allow_any_lane_order: job.allow_any_lane_order,
            },
        )?;
        Ok(ProcessWorkerOutput {
            id: job.to_synthesis_job(),
            results,
        })
    }
}

fn write_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let bytes = bincode::serialize(value).map_err(io::Error::other)?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "frame exceeds u32 length"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

fn read_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<Option<T>> {
    let mut len_bytes = [0_u8; 4];
    match reader.read_exact(&mut len_bytes) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes)?;
    bincode::deserialize(&bytes)
        .map(Some)
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))
}

fn score_job(scorer: &dyn Scorer, job: ScoringJob) -> ScoringJobResult {
    let path_length = job.path.len();
    let scored_paths = scorer
        .assign_path_gadgets_k_best_snapshot(&job.snapshot, job.assignment_limit)
        .into_iter()
        .enumerate()
        .map(|(assignment_index, assigned_path)| {
            let cost = scorer.score_assigned_path_snapshot(&assigned_path, &job.snapshot);
            ScoredPath::new(
                job.path_id,
                job.path.clone(),
                assigned_path,
                cost,
                job.order * job.assignment_limit.max(1) + assignment_index,
            )
        })
        .collect();

    ScoringJobResult {
        order: job.order,
        path_id: job.path_id,
        path_length,
        scored_paths,
    }
}

impl WaveEngine {
    pub fn new(config: WaveConfig) -> Result<Self, SynthesisError> {
        Self::with_scorer_factory(config, || Box::new(DummyScorer) as Box<dyn Scorer>)
    }

    pub fn with_scorer<S>(config: WaveConfig, scorer: S) -> Result<Self, SynthesisError>
    where
        S: Scorer + 'static,
    {
        let scorer: SharedScorer = Arc::new(scorer);
        Self::with_scoring_pool_factory(config, move |worker_count| {
            ScoringPool::new_shared(worker_count, Arc::clone(&scorer))
        })
    }

    pub fn with_scorer_factory<F>(
        config: WaveConfig,
        scorer_factory: F,
    ) -> Result<Self, SynthesisError>
    where
        F: Fn() -> Box<dyn Scorer> + Send + Sync + 'static,
    {
        let scorer: SharedScorer = Arc::from(scorer_factory());
        Self::with_scoring_pool_factory(config, move |worker_count| {
            ScoringPool::new_shared(worker_count, Arc::clone(&scorer))
        })
    }

    fn with_scoring_pool_factory(
        config: WaveConfig,
        scoring_pool_factory: impl FnOnce(usize) -> ScoringPool,
    ) -> Result<Self, SynthesisError> {
        let config = WaveConfig {
            worker_count: config.worker_count.max(1),
            ..config
        };
        let elements_per_vector = register_bits(config.arch) / dtype_bits(config.dtype);
        let total_elements = config.num_vecs * elements_per_vector;

        let sorter = BitonicSorter::new(total_elements);
        let mut stages = sorter.stages().to_vec();
        if config.natural_order {
            let stage_index = stages.len();
            stages.push(BitonicStage::new(
                stage_index,
                (0..elements_per_vector)
                    .map(|idx| (idx + 1, elements_per_vector + idx + 1))
                    .collect(),
            ));
        }

        let initial_state = create_initial_state(
            stages
                .first()
                .expect("bitonic sorter should produce at least one stage"),
        );
        let stage_target_pairs = stages
            .iter()
            .map(target_pairs_for_stage)
            .collect::<Vec<_>>();
        let stage_output_limits = stage_output_limits(&stages, config.natural_order);

        let synth =
            GadgetSynthesizer::new(to_synth_arch(config.arch), to_synth_dtype(config.dtype));
        let tiers = synth.precompute_candidate_tiers(config.gadget_depth)?;
        let CandidateGraphTiers {
            shallow: shallow_candidates,
            shared_prefix_deep: shared_prefix_deep_candidates,
            full_deep: full_deep_candidates,
        } = tiers;
        let mut deep_candidates = shared_prefix_deep_candidates.clone();
        deep_candidates.extend(full_deep_candidates.clone());
        let candidate_count = normal_candidate_count_for_mode(
            config.deep_search_mode,
            shallow_candidates.len(),
            shared_prefix_deep_candidates.len(),
            full_deep_candidates.len(),
        );
        let retroactive_input_source = if config.retroactive_input {
            Some(RetroactiveInputSource::new(&stages[0], candidate_count))
        } else {
            None
        };
        let mut transition_table =
            TransitionTable::new_with_lanes(stages.len(), elements_per_vector);
        let initial_state_id = transition_table.intern_zero_based_state(&initial_state);
        let mut exhausted_stages = HashSet::new();
        if config.retroactive_input {
            exhausted_stages.insert(0);
        }

        let stage_count = stages.len();

        Ok(Self {
            config,
            elements_per_vector,
            total_elements,
            stages,
            stage_target_pairs,
            stage_output_limits,
            shallow_candidates,
            shared_prefix_deep_candidates,
            full_deep_candidates,
            deep_candidates,
            retroactive_input_source,
            transition_table,
            stage_progress_totals: vec![0; stage_count],
            active_worker_counts: vec![0; stage_count],
            worker_capacities: vec![0; stage_count],
            queued_job_counts: vec![0; stage_count],
            initial_state,
            initial_state_id,
            wave_count: 0,
            exhausted_stages,
            stalled_stages: HashSet::new(),
            scored_path_keys: HashSet::new(),
            scored_paths: Vec::new(),
            recent_scored_paths: Vec::new(),
            scoring_pool: scoring_pool_factory(config.worker_count),
            next_scoring_job_order: 0,
            next_scoring_result_order: 0,
            buffered_scoring_results: BTreeMap::new(),
            latest_scoring_signatures: HashMap::new(),
            pending_scoring_path_ids: VecDeque::new(),
            pending_scoring_path_set: HashSet::new(),
            pending_registered_rescore_transitions: BTreeSet::new(),
            path_registry: PathRegistry::default(),
            process_workers: Vec::new(),
            adaptive_deep_attempts: HashMap::new(),
            trace: RuntimeTrace::disabled(),
        })
    }

    pub fn config(&self) -> WaveConfig {
        self.config
    }

    pub fn resolved_worker_backend(&self) -> WorkerBackendArg {
        match self.config.worker_backend {
            WorkerBackendArg::Auto if self.config.worker_count > 1 => WorkerBackendArg::Process,
            WorkerBackendArg::Auto => WorkerBackendArg::InProcess,
            backend => backend,
        }
    }

    pub fn elements_per_vector(&self) -> usize {
        self.elements_per_vector
    }

    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    pub fn stages(&self) -> &[BitonicStage] {
        &self.stages
    }

    pub fn stage_output_limit(&self, stage: usize) -> usize {
        self.stage_output_limits[stage]
    }

    pub fn stage_output_saturated(&self, stage: usize) -> bool {
        self.transition_table.unique_output_count(stage) >= self.stage_output_limits[stage]
    }

    pub fn shallow_candidates(&self) -> &[GadgetGraph] {
        &self.shallow_candidates
    }

    pub fn deep_candidates(&self) -> &[GadgetGraph] {
        &self.deep_candidates
    }

    pub fn shared_prefix_deep_candidates(&self) -> &[GadgetGraph] {
        &self.shared_prefix_deep_candidates
    }

    pub fn full_deep_candidates(&self) -> &[GadgetGraph] {
        &self.full_deep_candidates
    }

    pub fn normal_candidate_count(&self) -> usize {
        normal_candidate_count_for_mode(
            self.config.deep_search_mode,
            self.shallow_candidates.len(),
            self.shared_prefix_deep_candidates.len(),
            self.full_deep_candidates.len(),
        )
    }

    pub fn full_deep_candidate_indexes(&self) -> std::ops::Range<usize> {
        let start = self.shallow_candidates.len() + self.shared_prefix_deep_candidates.len();
        start..start + self.full_deep_candidates.len()
    }

    pub fn transition_table(&self) -> &TransitionTable {
        &self.transition_table
    }

    pub fn transition_table_mut(&mut self) -> &mut TransitionTable {
        &mut self.transition_table
    }

    pub fn runtime_trace(&self) -> &RuntimeTrace {
        &self.trace
    }

    pub fn set_runtime_trace(&mut self, runtime_trace: RuntimeTrace) {
        self.trace = runtime_trace;
    }

    pub fn initial_state(&self) -> &VectorState {
        &self.initial_state
    }

    pub fn wave_count(&self) -> usize {
        self.wave_count
    }

    pub fn exhausted_stages(&self) -> &HashSet<usize> {
        &self.exhausted_stages
    }

    pub fn stalled_stages(&self) -> &HashSet<usize> {
        &self.stalled_stages
    }

    pub fn scored_paths(&self) -> &[ScoredPath] {
        &self.scored_paths
    }

    pub fn scored_path_count(&self) -> usize {
        self.scored_paths.len()
    }

    pub fn recent_scored_paths(&self) -> &[ScoredPath] {
        &self.recent_scored_paths
    }

    pub fn scored_path_key_count(&self) -> usize {
        self.scored_path_keys.len()
    }

    pub fn pending_scoring_job_count(&self) -> usize {
        self.scoring_pool.pending_jobs + self.pending_scoring_path_ids.len()
    }

    fn in_flight_scoring_job_count(&self) -> usize {
        self.scoring_pool.pending_jobs
    }

    pub fn best_score(&self) -> Option<f64> {
        self.scored_paths
            .iter()
            .map(|path| path.cost().score())
            .min_by(f64::total_cmp)
    }

    pub fn discover_paths_for_transition(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
    ) -> usize {
        self.discover_paths_through_transition_for_reason(transition, max_paths, "manual_discovery")
    }

    pub fn discover_paths_through_transition(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
    ) -> usize {
        self.discover_paths_through_transition_for_reason(transition, max_paths, "manual_discovery")
    }

    fn discover_paths_through_transition_for_reason(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
        reason: &str,
    ) -> usize {
        let paths = self
            .transition_table
            .trace_paths_through(transition, max_paths, None);
        self.enqueue_scoring_paths(paths, reason)
    }

    pub fn poll_scoring_jobs(&mut self) -> usize {
        self.poll_scoring_jobs_for_reason("manual_poll")
    }

    pub fn drain_scoring_jobs(&mut self) -> usize {
        self.drain_scoring_jobs_for_reason("manual_drain")
    }

    fn poll_scoring_jobs_for_reason(&mut self, reason: &str) -> usize {
        self.poll_scoring_jobs_limited(reason, None)
    }

    fn poll_some_scoring_jobs(&mut self, reason: &str, result_limit: usize) -> usize {
        if result_limit == 0 {
            return 0;
        }
        self.poll_scoring_jobs_limited(reason, Some(result_limit))
    }

    fn poll_scoring_jobs_limited(&mut self, reason: &str, result_limit: Option<usize>) -> usize {
        let mut received = 0;
        while result_limit.is_none_or(|limit| received < limit) {
            let Ok(result) = self.scoring_pool.receiver.try_recv() else {
                break;
            };
            self.buffer_scoring_result(result, reason);
            received += 1;
        }
        let scored = self.apply_ready_scoring_results(reason, result_limit);
        self.top_off_scoring_jobs(reason);
        scored
    }

    fn drain_scoring_jobs_for_reason(&mut self, reason: &str) -> usize {
        self.flush_pending_registered_path_rescores(reason);
        self.top_off_scoring_jobs(reason);
        let mut scored = self.poll_scoring_jobs_for_reason(reason);
        while self.scoring_pool.pending_jobs > 0 || !self.pending_scoring_path_ids.is_empty() {
            self.top_off_scoring_jobs(reason);
            if self.scoring_pool.pending_jobs == 0 {
                break;
            }
            let Ok(result) = self.scoring_pool.receiver.recv() else {
                break;
            };
            self.buffer_scoring_result(result, reason);
            scored += self.apply_ready_scoring_results(reason, None);
            self.top_off_scoring_jobs(reason);
        }
        scored
    }

    fn buffer_scoring_result(&mut self, result: ScoringJobResult, reason: &str) {
        let fields = ScoringJobTraceFields {
            order: result.order,
            path_id: result.path_id,
            path_length: result.path_length,
            assigned_candidates: result.scored_paths.len(),
        };
        self.buffered_scoring_results.insert(fields.order, result);
        self.trace_scoring_job_completed(reason, fields);
    }

    pub fn record_transition(
        &mut self,
        stage: usize,
        input: StateId,
        output_state: &VectorState,
        gadget: PermutationGadget,
    ) -> TransitionRecordResult {
        if self.should_materialize_retroactive_prefix(stage) {
            self.materialize_retroactive_stage0_identity(input);
        }

        let insert =
            self.transition_table
                .add_transition_by_id(stage, input, output_state.clone(), gadget);
        let budget_output_added =
            insert.output_was_new && self.output_counts_toward_stage_budget(stage, insert.output);
        let discovered_paths = if insert.gadget_was_new {
            if insert.transition_was_new {
                self.discover_paths_through_transition_for_reason(
                    insert.transition,
                    None,
                    "new_transition",
                )
            } else {
                self.defer_registered_paths_for_transition_rescore(
                    insert.transition,
                    "new_gadget_on_registered_transition",
                )
            }
        } else {
            0
        };

        TransitionRecordResult {
            transition_added: insert.gadget_was_new,
            budget_output_added,
            discovered_paths,
        }
    }

    fn output_counts_toward_stage_budget(&self, stage: usize, output: StateId) -> bool {
        let next_stage = stage + 1;
        if next_stage >= self.stages.len() {
            return true;
        }

        !self
            .transition_table
            .was_input_attempted_by_id(next_stage, output)
            && !self
                .transition_table
                .stage(next_stage)
                .unique_input_ids()
                .contains(&output)
    }

    /// Chooses which stage the next wave should target.
    ///
    /// [`run_wave`](Self::run_wave) calls this at the start of every wave, and
    /// [`run`](Self::run) uses it as a loop guard (a `None` result ends the
    /// search). The choice drives where synthesis effort is spent.
    ///
    /// # Selection rules
    ///
    /// 1. **No stages**: returns `None`.
    /// 2. **First-wave bootstrap**: on wave 0, if stage 0 is not already
    ///    exhausted, stage 0 is chosen unconditionally. (Stage 0 *can* already
    ///    be exhausted on the first wave when seeded via `--retroactive-input`.)
    /// 3. **Least-explored stage**: otherwise `pick_target_stage` selects the
    ///    non-exhausted, non-stalled stage with the fewest distinct outputs
    ///    (ties broken by attempts, then index).
    /// 4. **Bubble-up**: if that stage shows signs of being starved — two or
    ///    more consecutive zero-output budgets, or three or more unproductive
    ///    waves — selection backs up to the nearest earlier stage that is
    ///    neither exhausted nor stalled, widening the input frontier feeding the
    ///    stuck stage. If no such upstream stage exists, the original target is
    ///    kept.
    ///
    /// Returns the chosen stage index, or `None` only when there are no stages.
    pub fn select_target_stage(&self) -> Option<usize> {
        if self.stages.is_empty() {
            return None;
        }

        // This kicks in when on the first wave but stage 0 is already exhausted.
        // The reason this is even possible is because of --retroactive-input.
        if self.wave_count == 0 && self.stage_is_frontier_selectable(0) {
            return Some(0);
        }

        let target = self.pick_target_stage()?;
        let stage_data = self.transition_table.stage(target);
        let needs_bubble_up =
            stage_data.consecutive_zero_budgets() >= 2 || stage_data.unproductive_waves() >= 3;

        if needs_bubble_up {
            for upstream in (0..target).rev() {
                if self.stage_is_frontier_selectable(upstream) {
                    return Some(upstream);
                }
            }
        }

        Some(target)
    }

    fn pick_target_stage(&self) -> Option<usize> {
        (0..self.stages.len())
            .filter(|stage| self.stage_is_frontier_selectable(*stage))
            .min_by_key(|stage| {
                let stats = self.transition_table.stage_stats(*stage);
                (stats.distinct_outputs, stats.attempts, *stage)
            })
    }

    fn stage_is_frontier_selectable(&self, stage: usize) -> bool {
        !self.exhausted_stages.contains(&stage)
            && !self.stalled_stages.contains(&stage)
            && !self.stage_output_saturated(stage)
    }

    /// Enumerates the [`SynthesisJob`]s for one sorting-network `stage`.
    ///
    /// Each job is a `(input state, candidate gadget graph)` pair that has not
    /// yet been attempted for this stage. Executing a job asks Z3 to synthesize
    /// a permutation gadget that takes the input state through the stage's
    /// comparators (see [`execute_job`](Self::execute_job)). The returned jobs
    /// are the unit of work consumed by the wave loop.
    ///
    /// # Inputs
    ///
    /// The set of input states is chosen as follows:
    ///
    /// - `Some(input_states)`: use exactly these states, keeping only the ones
    ///   already known to the transition table (unknown states are silently
    ///   dropped). Used to drive synthesis from an explicit frontier.
    /// - `None` and `stage == 0`: use the engine's single initial state.
    /// - `None`, `stage == 1`, and `--retroactive-input`: page virtual
    ///   factorial-ranked permutations from stage 0's comparison pairs.
    /// - `None` and `stage > 0`: use the unique output states produced by the
    ///   previous stage (`stage - 1`).
    ///
    /// In all non-trivial cases the inputs are ordered by
    /// `ordered_inputs_for_stage`, which moves not-yet-attempted inputs to the
    /// front so that limited budgets explore new states first.
    ///
    /// # Candidates
    ///
    /// Candidate gadget graphs are indexed across the concatenation of the
    /// shallow candidate pool followed by the deep candidate pool. A
    /// `candidate_index` of `0..shallow_candidates.len()` refers to a shallow
    /// graph; higher indices refer to deep graphs (see `candidate_graph`).
    ///
    /// # `limit`
    ///
    /// - `None`: emit every not-yet-attempted `(input, candidate)` pair for the
    ///   stage.
    /// - `Some(limit)`: emit at most `limit` jobs. The candidate space is walked
    ///   in windows of `max(limit / inputs.len(), 1)` candidates so that the
    ///   budget is spread across candidates rather than exhausting all inputs
    ///   for a single candidate first. The method returns as soon as `limit`
    ///   jobs have been collected.
    ///
    /// Already-attempted `(stage, input, candidate)` triples are skipped via
    /// [`was_attempted_by_id`](crate::transition_table). An empty vector is
    /// returned when there are no usable inputs or no candidates.
    pub fn make_jobs(
        &mut self,
        stage: usize,
        input_states: Option<&[VectorState]>,
        limit: Option<usize>,
    ) -> Vec<SynthesisJob> {
        if input_states.is_none() && self.should_use_virtual_retroactive_inputs(stage) {
            return self.make_retroactive_stage1_jobs(stage, limit);
        }

        let owned_inputs;
        let inputs = if let Some(input_states) = input_states {
            let input_states = input_states
                .iter()
                .filter_map(|state| {
                    self.transition_table
                        .lookup_zero_based_tuple(&state.as_tuple())
                })
                .collect::<Vec<_>>();
            self.ordered_inputs_for_stage(stage, input_states)
        } else if stage == 0 {
            vec![self.initial_state_id]
        } else {
            owned_inputs = self.ordered_outputs_from_stage(stage - 1);
            self.ordered_inputs_for_stage(stage, owned_inputs)
        };

        let candidate_count = self.normal_candidate_count();
        let mut jobs = Vec::new();

        if inputs.is_empty() || candidate_count == 0 {
            return jobs;
        }

        let window_size = limit
            .map(|limit| (limit / inputs.len()).max(1))
            .unwrap_or(candidate_count)
            .min(candidate_count);

        for window_start in (0..candidate_count).step_by(window_size) {
            let window_end = (window_start + window_size).min(candidate_count);
            for candidate_index in window_start..window_end {
                for input in &inputs {
                    if self
                        .transition_table
                        .was_attempted_by_id(stage, *input, candidate_index)
                    {
                        continue;
                    }

                    jobs.push(SynthesisJob {
                        stage,
                        input: *input,
                        candidate_index,
                        retroactive_rank: None,
                    });

                    if limit.is_some_and(|limit| jobs.len() >= limit) {
                        return jobs;
                    }
                }
            }
        }
        jobs
    }

    fn should_use_virtual_retroactive_inputs(&self, stage: usize) -> bool {
        self.config.retroactive_input && stage == 1 && self.retroactive_input_source.is_some()
    }

    fn make_retroactive_stage1_jobs(
        &mut self,
        stage: usize,
        limit: Option<usize>,
    ) -> Vec<SynthesisJob> {
        let candidate_count = self.normal_candidate_count();
        let job_limit = limit.unwrap_or(candidate_count);
        let mut jobs = Vec::new();

        if job_limit == 0 || candidate_count == 0 {
            return jobs;
        }

        for candidate_index in 0..candidate_count {
            let Some(mut rank) = self
                .retroactive_input_source
                .as_ref()
                .map(|source| source.next_rank(candidate_index))
            else {
                return jobs;
            };
            let total_ranks = self
                .retroactive_input_source
                .as_ref()
                .expect("retroactive source should exist")
                .total_ranks();

            while rank < total_ranks {
                let state = self
                    .retroactive_input_source
                    .as_ref()
                    .expect("retroactive source should exist")
                    .state_for_rank(rank);
                let input = self.transition_table.intern_zero_based_state(&state);

                if !self
                    .transition_table
                    .was_attempted_by_id(stage, input, candidate_index)
                {
                    jobs.push(SynthesisJob {
                        stage,
                        input,
                        candidate_index,
                        retroactive_rank: Some(rank),
                    });

                    if jobs.len() >= job_limit {
                        return jobs;
                    }
                }

                rank += 1;
            }
        }

        jobs
    }

    fn should_materialize_retroactive_prefix(&self, stage: usize) -> bool {
        self.config.retroactive_input && stage == 1
    }

    fn materialize_retroactive_stage0_identity(&mut self, input: StateId) {
        let state = self.transition_table.state_as_vector_state(input);
        self.transition_table.add_transition_by_id(
            0,
            input,
            state,
            PermutationGadget::new(Vec::new(), Vec::new()),
        );
        self.transition_table.mark_forwarded_ids(0, &[input]);
    }

    fn advance_retroactive_cursor_for_job(&mut self, job: SynthesisJob) {
        if job.retroactive_rank().is_none() {
            return;
        }

        let candidate_index = job.candidate_index();
        loop {
            let Some((rank, total_ranks)) = self
                .retroactive_input_source
                .as_ref()
                .map(|source| (source.next_rank(candidate_index), source.total_ranks()))
            else {
                return;
            };

            if rank >= total_ranks {
                return;
            }

            let state = self
                .retroactive_input_source
                .as_ref()
                .expect("retroactive source should exist")
                .state_for_rank(rank);
            let Some(input) = self
                .transition_table
                .lookup_zero_based_tuple(&state.as_tuple())
            else {
                return;
            };

            if !self
                .transition_table
                .was_attempted_by_id(job.stage(), input, candidate_index)
            {
                return;
            }

            self.retroactive_input_source
                .as_mut()
                .expect("retroactive source should exist")
                .set_next_rank(candidate_index, rank + 1);
        }
    }

    /// Synthesizes and applies a single [`SynthesisJob`] in-process.
    ///
    /// This is the serial counterpart to the worker pool: it runs one job on the
    /// calling thread and immediately folds the result into engine state. The
    /// inline branch of [`run_stage`](Self::run_stage) uses this path when there
    /// is only one worker or one job; it is also convenient for tests and
    /// step-by-step driving.
    ///
    /// Internally it expands the compact job into a full work item — resolving
    /// `candidate_index` to a concrete gadget graph via `candidate_graph` and
    /// attaching the stage's target comparator pairs — hands it to the Z3-backed
    /// `synthesize_worker_job`, then applies the output (recording any new
    /// transition and registering newly discovered terminal paths for later
    /// scoring).
    ///
    /// Returns a [`JobExecutionResult`] describing the attempt and any valid
    /// outputs, new transitions, and discovered paths it produced. Errors from
    /// the synthesizer are propagated.
    pub fn execute_job(
        &mut self,
        job: &SynthesisJob,
    ) -> Result<JobExecutionResult, SynthesisError> {
        let output = synthesize_worker_job(
            self.config,
            WorkerSynthesisJob {
                index: 0,
                id: *job,
                input_state: self.input_state_for_job(*job),
                graph: self
                    .candidate_graph(job.candidate_index())
                    .expect("job candidate index should be valid")
                    .clone(),
                target_pairs: self.stage_target_pairs(job.stage()),
                allow_any_lane_order: self.allow_any_lane_order_for_stage(job.stage()),
            },
        )?;

        Ok(self.apply_synthesis_output(output))
    }

    fn take_or_create_workers(
        &mut self,
        worker_count: usize,
    ) -> Result<Vec<ProcessSynthesisWorker>, SynthesisError> {
        while self.process_workers.len() < worker_count {
            let worker = ProcessSynthesisWorker::spawn(self.config)?;
            self.trace_process_worker_spawned("spawn", &worker);
            self.process_workers.push(worker);
        }

        let workers = self
            .process_workers
            .drain(0..worker_count)
            .collect::<Vec<_>>();
        self.trace_process_worker_pool_snapshot("checkout", &workers);
        Ok(workers)
    }

    fn return_process_workers(&mut self, workers: Vec<ProcessSynthesisWorker>) {
        self.trace_process_worker_pool_snapshot("return", &workers);
        for worker in workers {
            if worker.should_recycle() {
                self.trace_process_worker_recycled(&worker);
                worker.shutdown();
            } else {
                self.process_workers.push(worker);
            }
        }
        self.trace_process_worker_pool_snapshot("idle", &self.process_workers);
    }

    fn apply_synthesis_output(&mut self, output: SynthesisJobOutput) -> JobExecutionResult {
        let id = output.id;

        self.transition_table.record_attempt(id.stage, 1);
        self.transition_table
            .record_attempted_pair_by_id(id.stage, id.input, id.candidate_index);
        self.advance_retroactive_cursor_for_job(id);

        let valid_output_count = output.results.len();
        let mut new_transition_count = 0;
        let mut budget_output_count = 0;
        let mut discovered_paths = 0;
        for (gadget, output_state) in output.results {
            let result = self.record_transition(id.stage, id.input, &output_state, gadget);
            if result.transition_added() {
                new_transition_count += 1;
                discovered_paths += result.discovered_paths();
            }
            if result.budget_output_added() {
                budget_output_count += 1;
            }
        }

        JobExecutionResult {
            stage: id.stage,
            candidate_index: id.candidate_index,
            attempts: 1,
            valid_output_count,
            transition_count: valid_output_count,
            new_transition_count,
            budget_output_count,
            discovered_paths,
        }
    }

    /// Runs one *wave*: a target-stage pass plus downstream propagation.
    ///
    /// A wave is the engine's main progress increment. It is what
    /// [`run`](Self::run) loops over, and it ties together stage selection,
    /// per-stage synthesis, exhaustion bookkeeping, and forward propagation of
    /// newly discovered states.
    ///
    /// # Phases
    ///
    /// 1. **Target selection** — [`select_target_stage`](Self::select_target_stage)
    ///    picks the least-explored stage, then [`run_stage`](Self::run_stage) is
    ///    invoked on it with `input_states = None` and the given budgets.
    /// 2. **Exhaustion / stall bookkeeping** — if the target produced no new
    ///    outputs, a zero-output budget is recorded and the stage is marked
    ///    `exhausted` (no work left and inputs are settled) or `stalled`
    ///    (waiting on upstream inputs). Productive targets reset these markers
    ///    for the stage and clear stalls on downstream stages.
    /// 3. **Propagation** — for every later stage, the outputs newly produced by
    ///    its predecessor are forwarded as explicit `input_states` into another
    ///    [`run_stage`](Self::run_stage) call, so fresh states flow downstream
    ///    within the same wave instead of waiting for a future wave.
    /// 4. **Productivity tracking** — the count of unique outputs at the final
    ///    stage is compared before/after to mark the wave productive or
    ///    unproductive (which feeds the bubble-up heuristic in stage selection).
    ///
    /// Both budgets are applied per [`run_stage`](Self::run_stage) call (target
    /// and each propagation step), not shared across the wave. The wave boundary
    /// also flushes pending path rescores and polls a bounded number of
    /// asynchronous scoring results.
    ///
    /// Returns a [`WaveRunResult`] holding the target result, the per-stage
    /// propagation results, and the paths discovered/scored during the wave.
    /// Honors `session.should_stop()` between propagation stages.
    pub fn run_wave(
        &mut self,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<WaveRunResult, SynthesisError> {
        let wave = self.wave_count;
        let target_stage = self
            .select_target_stage()
            .expect("wave engine should have at least one stage");
        let last_stage = self.stages.len() - 1;
        self.poll_some_scoring_jobs("wave_started", SCORING_RESULT_POLL_LIMIT_PER_WAVE_BOUNDARY);
        let last_outputs_before = self.transition_table.unique_output_count(last_stage);
        let target = self.run_stage(target_stage, None, attempt_budget, output_budget, session)?;
        let mut propagation = Vec::new();

        if target.new_outputs() == 0 {
            self.transition_table
                .record_zero_output_budget(target_stage);
            if target.attempts() == 0 && !self.stage_has_remaining_jobs(target_stage) {
                if self.stage_has_inputs(target_stage)
                    || (0..target_stage).all(|stage| self.exhausted_stages.contains(&stage))
                {
                    self.exhausted_stages.insert(target_stage);
                    self.stalled_stages.remove(&target_stage);
                } else {
                    self.stalled_stages.insert(target_stage);
                }
            }
        } else {
            self.transition_table
                .reset_zero_output_budgets(target_stage);
            self.stalled_stages.remove(&target_stage);
            for downstream in target_stage + 1..self.stages.len() {
                self.stalled_stages.remove(&downstream);
            }
        }

        propagation.extend(self.propagate_downstream_from(
            target_stage,
            attempt_budget,
            output_budget,
            session,
        )?);

        let adaptive = self.run_adaptive_deep_search(attempt_budget, output_budget, session)?;
        if adaptive.new_outputs > 0 {
            propagation.extend(self.propagate_downstream_from(
                adaptive.min_stage,
                attempt_budget,
                output_budget,
                session,
            )?);
        }

        let last_outputs_after = self.transition_table.unique_output_count(last_stage);
        if last_outputs_after > last_outputs_before {
            self.transition_table.reset_unproductive_waves(target_stage);
        } else {
            self.transition_table.record_unproductive_wave(target_stage);
        }

        self.wave_count += 1;
        self.flush_pending_registered_path_rescores("wave_finished");
        let scored_paths = self
            .poll_some_scoring_jobs("wave_finished", SCORING_RESULT_POLL_LIMIT_PER_WAVE_BOUNDARY);
        let discovered_paths = target.discovered_paths()
            + adaptive.discovered_paths
            + propagation
                .iter()
                .map(StageRunResult::discovered_paths)
                .sum::<usize>();
        Ok(WaveRunResult {
            wave,
            target_stage,
            target,
            propagation,
            discovered_paths,
            scored_paths,
        })
    }

    fn propagate_downstream_from(
        &mut self,
        start_stage: usize,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<Vec<StageRunResult>, SynthesisError> {
        let mut propagation = Vec::new();
        for stage in start_stage + 1..self.stages.len() {
            if session.should_stop() {
                break;
            }
            let mut unforwarded = self.transition_table.get_unforwarded_output_ids(stage - 1);
            if unforwarded.is_empty() {
                continue;
            }
            unforwarded.sort_by_key(|(state_id, _)| {
                self.transition_table.state_as_zero_based_tuple(*state_id)
            });

            let forwarded = unforwarded
                .iter()
                .map(|(state_id, _)| *state_id)
                .collect::<Vec<_>>();
            let input_states = unforwarded
                .into_iter()
                .map(|(_, state)| state)
                .collect::<Vec<_>>();

            self.transition_table
                .mark_forwarded_ids(stage - 1, &forwarded);
            let result = self.run_stage(
                stage,
                Some(&input_states),
                attempt_budget,
                output_budget,
                session,
            )?;
            if result.attempts() > 0 || result.new_outputs() > 0 {
                propagation.push(result);
            }
        }
        Ok(propagation)
    }

    fn run_adaptive_deep_search(
        &mut self,
        wave_attempt_budget: usize,
        wave_output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<AdaptiveDeepRunResult, SynthesisError> {
        if self.config.deep_search_mode != DeepSearchModeArg::Adaptive
            || self.full_deep_candidates.is_empty()
        {
            return Ok(AdaptiveDeepRunResult::empty(self.stages.len()));
        }

        let attempt_budget = (wave_attempt_budget / 5).max(1);
        let output_budget = (wave_output_budget / 5).max(1);
        let targets = self.adaptive_deep_targets();
        let jobs = self.adaptive_deep_jobs(&targets, attempt_budget);

        crate::trace!(self.trace, "adaptive_deep_started", {
                "wave": self.wave_count,
                "attempt_budget": attempt_budget,
                "output_budget": output_budget,
                "target_count": targets.len(),
                "job_count": jobs.len(),
                "full_deep_candidates": self.full_deep_candidates.len(),
        });

        if jobs.is_empty() {
            crate::trace!(self.trace, "adaptive_deep_finished", {
                    "wave": self.wave_count,
                    "attempts": 0,
                    "valid_outputs": 0,
                    "new_outputs": 0,
                    "budget_outputs": 0,
                    "new_transitions": 0,
                    "discovered_paths": 0,
            });
            return Ok(AdaptiveDeepRunResult::empty(self.stages.len()));
        }

        let mut result = AdaptiveDeepRunResult::empty(self.stages.len());
        let mut remaining_attempts = attempt_budget;
        let mut remaining_outputs = output_budget;
        let mut jobs_by_stage: BTreeMap<usize, Vec<SynthesisJob>> = BTreeMap::new();
        for job in jobs {
            jobs_by_stage.entry(job.stage()).or_default().push(job);
        }

        for (stage, mut stage_jobs) in jobs_by_stage {
            if session.should_stop() || remaining_attempts == 0 || remaining_outputs == 0 {
                break;
            }
            stage_jobs.truncate(remaining_attempts);
            let selected_jobs = stage_jobs.clone();
            let outputs_at_start = self.transition_table.unique_output_count(stage);
            let attempts_at_start = self.transition_table.stage_stats(stage).attempts;
            let stage_result = self.run_prepared_stage_jobs(
                PreparedStageRun {
                    stage,
                    jobs: stage_jobs,
                    outputs_at_start,
                    attempts_at_start,
                    attempt_budget: remaining_attempts,
                    output_budget: remaining_outputs,
                },
                session,
            )?;
            for job in selected_jobs {
                if self.transition_table.was_attempted_by_id(
                    job.stage(),
                    job.input(),
                    job.candidate_index(),
                ) {
                    *self
                        .adaptive_deep_attempts
                        .entry((job.stage(), job.input()))
                        .or_insert(0) += 1;
                }
            }
            remaining_attempts = remaining_attempts.saturating_sub(stage_result.attempts());
            remaining_outputs = remaining_outputs.saturating_sub(stage_result.budget_outputs());
            result.add_stage_result(&stage_result);
        }

        crate::trace!(self.trace, "adaptive_deep_finished", {
                "wave": self.wave_count,
                "attempts": result.attempts,
                "valid_outputs": result.valid_outputs,
                "new_outputs": result.new_outputs,
                "budget_outputs": result.budget_outputs,
                "new_transitions": result.new_transitions,
                "discovered_paths": result.discovered_paths,
        });

        Ok(result)
    }

    fn adaptive_deep_targets(&self) -> Vec<AdaptiveTarget> {
        let mut accumulators: BTreeMap<(usize, StateId), AdaptiveTargetAccumulator> =
            BTreeMap::new();
        for scored_path in &self.scored_paths {
            let path_steps = scored_path.path().iter().collect::<Vec<_>>();
            let selected_gadgets = scored_path.assigned_path().gadgets();
            let mut suffix_expensive = vec![0; path_steps.len()];
            let mut running = 0;
            for index in (0..path_steps.len()).rev() {
                let (stage, transition) = path_steps[index];
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                let gadget = self
                    .transition_table
                    .transition(transition_ref)
                    .gadget(selected_gadgets[index]);
                running += expensive_gadget_count(gadget);
                suffix_expensive[index] = running;
            }

            for (index, (stage, transition)) in path_steps.into_iter().enumerate() {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                let input = self.transition_table.transition(transition_ref).input();
                let accumulator = accumulators.entry((stage, input)).or_default();
                accumulator.topk_hits += 1;
                accumulator.expensive_suffix_count += suffix_expensive[index];
            }
        }

        let mut targets = accumulators
            .into_iter()
            .filter_map(|((stage, input), accumulator)| {
                self.adaptive_target_from_parts(stage, input, accumulator, false)
            })
            .collect::<Vec<_>>();

        let primary_keys = targets
            .iter()
            .map(|target| (target.stage, target.input))
            .collect::<HashSet<_>>();
        targets.extend(
            self.exploratory_adaptive_targets(&primary_keys)
                .into_iter()
                .filter_map(|(stage, input)| {
                    self.adaptive_target_from_parts(
                        stage,
                        input,
                        AdaptiveTargetAccumulator::default(),
                        true,
                    )
                }),
        );

        targets.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.stage.cmp(&right.stage))
                .then_with(|| left.input.cmp(&right.input))
        });
        targets
    }

    fn adaptive_target_from_parts(
        &self,
        stage: usize,
        input: StateId,
        accumulator: AdaptiveTargetAccumulator,
        exploratory: bool,
    ) -> Option<AdaptiveTarget> {
        if !self.full_deep_candidate_indexes().any(|candidate_index| {
            !self
                .transition_table
                .was_attempted_by_id(stage, input, candidate_index)
        }) {
            return None;
        }

        let stage_data = self.transition_table.stage(stage);
        let stage_stall_bonus = usize::from(
            self.stalled_stages.contains(&stage) || stage_data.unproductive_waves() > 0,
        );
        let known_outputs = self
            .transition_table
            .transition_count_for_input_id(stage, input);
        let low_output_diversity_bonus = usize::from(known_outputs <= 1);
        let attempts = self
            .adaptive_deep_attempts
            .get(&(stage, input))
            .copied()
            .unwrap_or_else(|| self.full_deep_attempt_count_for_input(stage, input));
        let priority = 1000 * accumulator.topk_hits as i64
            + 100 * accumulator.expensive_suffix_count as i64
            + 50 * stage_stall_bonus as i64
            + 25 * low_output_diversity_bonus as i64
            + deterministic_jitter(stage, input)
            - 25 * attempts as i64;
        Some(AdaptiveTarget {
            stage,
            input,
            topk_hits: accumulator.topk_hits,
            expensive_suffix_count: accumulator.expensive_suffix_count,
            priority,
            exploratory,
        })
    }

    fn exploratory_adaptive_targets(
        &self,
        primary_keys: &HashSet<(usize, StateId)>,
    ) -> Vec<(usize, StateId)> {
        let mut inputs = Vec::new();
        for stage in 0..self.stages.len() {
            for input in self.current_input_ids_for_stage(stage) {
                if !primary_keys.contains(&(stage, input)) {
                    inputs.push((stage, input));
                }
            }
        }
        inputs.sort_by(|left, right| {
            deterministic_jitter(right.0, right.1)
                .cmp(&deterministic_jitter(left.0, left.1))
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.1.cmp(&right.1))
        });
        let reserve = self.config.top_k.unwrap_or(1).max(1);
        inputs.truncate(reserve);
        inputs
    }

    fn current_input_ids_for_stage(&self, stage: usize) -> Vec<StateId> {
        let mut inputs = if stage == 0 {
            vec![self.initial_state_id]
        } else {
            self.transition_table
                .get_unique_output_ids(stage - 1)
                .into_iter()
                .map(|(state_id, _)| state_id)
                .collect::<Vec<_>>()
        };
        inputs.sort_by_key(|input| self.transition_table.state_as_zero_based_tuple(*input));
        inputs
    }

    fn full_deep_attempt_count_for_input(&self, stage: usize, input: StateId) -> usize {
        self.full_deep_candidate_indexes()
            .filter(|candidate_index| {
                self.transition_table
                    .was_attempted_by_id(stage, input, *candidate_index)
            })
            .count()
    }

    fn adaptive_deep_jobs(
        &self,
        targets: &[AdaptiveTarget],
        attempt_budget: usize,
    ) -> Vec<SynthesisJob> {
        if attempt_budget == 0 {
            return Vec::new();
        }
        let exploratory_budget =
            (attempt_budget / 10).max(usize::from(targets.iter().any(|target| target.exploratory)));
        let primary_budget = attempt_budget.saturating_sub(exploratory_budget);
        let mut jobs = Vec::new();
        self.extend_adaptive_jobs(
            targets.iter().filter(|target| !target.exploratory),
            primary_budget,
            &mut jobs,
        );
        self.extend_adaptive_jobs(
            targets.iter().filter(|target| target.exploratory),
            attempt_budget.saturating_sub(jobs.len()),
            &mut jobs,
        );
        jobs.truncate(attempt_budget);
        jobs
    }

    fn extend_adaptive_jobs<'a>(
        &self,
        targets: impl Iterator<Item = &'a AdaptiveTarget> + Clone,
        budget: usize,
        jobs: &mut Vec<SynthesisJob>,
    ) {
        if budget == 0 {
            return;
        }
        let start_len = jobs.len();
        for candidate_index in self.full_deep_candidate_indexes() {
            for target in targets.clone() {
                if jobs.len().saturating_sub(start_len) >= budget {
                    return;
                }
                if self.transition_table.was_attempted_by_id(
                    target.stage,
                    target.input,
                    candidate_index,
                ) {
                    continue;
                }
                crate::trace!(self.trace, "adaptive_deep_candidate_selected", {
                        "wave": self.wave_count,
                        "stage": target.stage,
                        "input": target.input.0,
                        "candidate_index": candidate_index,
                        "priority": target.priority,
                        "topk_hits": target.topk_hits,
                        "expensive_suffix_count": target.expensive_suffix_count,
                        "exploratory": target.exploratory,
                });
                jobs.push(SynthesisJob {
                    stage: target.stage,
                    input: target.input,
                    candidate_index,
                    retroactive_rank: None,
                });
            }
        }
    }

    /// Runs the full wave search loop — the top-level synthesis entry point.
    ///
    /// This is the outermost driver: it repeatedly invokes
    /// [`run_wave`](Self::run_wave) until the search is exhausted, the wave cap
    /// is hit, or the session is asked to stop. It is the method the production
    /// solve path uses, and the one that wires in every progress and scoring
    /// extension point.
    ///
    /// # Loop and termination
    ///
    /// Each iteration checks, in order: `search_exhausted`
    /// (all stages exhausted), `session.should_stop()`, the optional
    /// `max_waves` cap, and whether [`select_target_stage`](Self::select_target_stage)
    /// still yields a stage. If all pass, it runs one wave and records its
    /// [`WaveRunResult`].
    ///
    /// # Budgets
    ///
    /// `attempt_budget` and `output_budget` are forwarded unchanged to every
    /// [`run_wave`](Self::run_wave) call (and thus to each
    /// [`run_stage`](Self::run_stage) within a wave); they bound per-stage work,
    /// not the run as a whole. `max_waves` bounds the number of waves.
    ///
    /// # Observation and ordering
    ///
    /// Beyond emitting [`RuntimeEvent`]s and JSON traces, this method calls
    /// `observer.on_wave_progress` after each wave and after the final
    /// rough-scoring drain, handing over the recently applied scored paths. The
    /// production path uses this hook to feed live full-uiCA scoring without
    /// blocking the synthesis coordinator.
    ///
    /// Determinism is preserved despite asynchronous workers: synthesis results
    /// are applied in candidate order and rough-scoring results in submission
    /// order before the observer ever sees them.
    ///
    /// # Finalization
    ///
    /// After the loop, any outstanding scoring jobs are drained, a final
    /// observer callback fires, and a [`WaveSearchResult`] is returned with the
    /// total wave count and whether the search was exhausted.
    pub fn run(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        observer: &mut impl WaveProgressObserver,
    ) -> Result<WaveSearchResult, SynthesisError> {
        session.on_event(RuntimeEvent::RunStarted {
            stage_count: self.stages.len(),
        });
        crate::trace!(self.trace, "engine_run_started", {
                    "stage_count": self.stages.len(),
                    "max_waves": max_waves,
                    "attempt_budget": attempt_budget,
                    "output_budget": output_budget,
                    "worker_count": self.config.worker_count,
                    "worker_backend": self.resolved_worker_backend().cli_name(),
        });
        for snapshot in self.stage_progress_snapshots() {
            session.on_event(RuntimeEvent::StageUpdated(snapshot));
        }
        self.emit_scoring_update(session, "run_started");

        let mut waves = Vec::new();
        while !self.search_exhausted() {
            if session.should_stop() {
                break;
            }
            if max_waves.is_some_and(|limit| self.wave_count >= limit) {
                break;
            }
            let Some(target_stage) = self.select_target_stage() else {
                break;
            };

            session.on_event(RuntimeEvent::WaveStarted {
                wave: self.wave_count,
                target_stage,
            });
            crate::trace!(self.trace, "wave_started", {
                        "wave": self.wave_count,
                        "target_stage": target_stage,
            });
            let wave = self.run_wave(attempt_budget, output_budget, session)?;
            for snapshot in self.stage_progress_snapshots() {
                session.on_event(RuntimeEvent::StageUpdated(snapshot));
            }
            session.on_event(RuntimeEvent::WaveFinished {
                wave: wave.wave(),
                target_stage: wave.target_stage(),
                attempts: wave.target().attempts(),
                valid_outputs: wave.target().valid_outputs(),
                new_outputs: wave.target().new_outputs(),
                discovered_paths: wave.discovered_paths(),
                scored_paths: wave.scored_paths(),
            });
            crate::trace!(self.trace, "wave_finished", {
                        "wave": wave.wave(),
                        "target_stage": wave.target_stage(),
                        "attempts": wave.target().attempts(),
                        "valid_outputs": wave.target().valid_outputs(),
                        "new_outputs": wave.target().new_outputs(),
                        "discovered_paths": wave.discovered_paths(),
                        "scored_paths": wave.scored_paths(),
            });
            self.trace_storage_snapshot("wave_finished");
            self.emit_scoring_update(session, "wave_finished");
            observer
                .on_wave_progress(self, &self.recent_scored_paths, session, "wave_finished")
                .map_err(SynthesisError::WorkerProcess)?;
            self.recent_scored_paths.clear();
            waves.push(wave);
        }

        self.drain_scoring_jobs_for_reason("run_finished_drain");
        self.trace_storage_snapshot("run_finished_drain");
        self.emit_scoring_update(session, "run_finished_drain");
        observer
            .on_wave_progress(
                self,
                &self.recent_scored_paths,
                session,
                "run_finished_drain",
            )
            .map_err(SynthesisError::WorkerProcess)?;
        self.recent_scored_paths.clear();
        let result = WaveSearchResult {
            wave_count: self.wave_count,
            search_exhausted: self.search_exhausted(),
            waves,
        };
        session.on_event(RuntimeEvent::RunFinished {
            wave_count: result.wave_count(),
            search_exhausted: result.search_exhausted(),
            scored_paths: self.scored_path_count(),
            best_score: self.best_score(),
        });
        crate::trace!(self.trace, "engine_run_finished", {
                    "wave_count": result.wave_count(),
                    "search_exhausted": result.search_exhausted(),
                    "scored_paths": self.scored_path_count(),
                    "best_score": self.best_score(),
        });

        Ok(result)
    }

    fn trace_storage_snapshot(&self, reason: &str) {
        crate::trace!(self.trace, "engine_storage_snapshot", {
            "reason": reason,
            "wave": self.wave_count,
            "states": self.transition_table.states().len(),
            "path_registry_paths": self.path_registry.len(),
            "path_registry_indexed_transitions": self.path_registry.indexed_transition_count(),
            "path_registry_indexed_path_refs": self.path_registry.indexed_path_ref_count(),
            "scored_path_keys": self.scored_path_keys.len(),
            "retained_scored_paths": self.scored_paths.len(),
            "recent_scored_paths": self.recent_scored_paths.len(),
            "latest_scoring_signatures": self.latest_scoring_signatures.len(),
            "pending_scoring_path_ids": self.pending_scoring_path_ids.len(),
            "pending_scoring_path_set": self.pending_scoring_path_set.len(),
            "pending_registered_rescore_transitions": self.pending_registered_rescore_transitions.len(),
            "buffered_scoring_results": self.buffered_scoring_results.len(),
            "pending_scoring_jobs": self.pending_scoring_job_count(),
            "in_flight_scoring_jobs": self.in_flight_scoring_job_count(),
            "stages": self
            .transition_table
            .stages()
            .iter()
            .enumerate()
            .map(|(stage, data)| {
                json!({
                    "stage": stage,
                    "transitions": data.transition_count(),
                    "gadgets": data.total_gadget_count(),
                    "unique_inputs": data.unique_input_ids().len(),
                    "unique_outputs": data.unique_output_ids().len(),
                    "output_limit": self.stage_output_limit(stage),
                    "output_saturated": self.stage_output_saturated(stage),
                    "attempted_pairs": data.attempted_pair_count(),
                    "forwarded_outputs": data.forwarded_output_count(),
                    "transitions_by_input_entries": data.transitions_by_input_entry_count(),
                    "transitions_by_input_refs": data.transitions_by_input_ref_count(),
                    "transitions_by_output_entries": data.transitions_by_output_entry_count(),
                    "transitions_by_output_refs": data.transitions_by_output_ref_count(),
                    "attempts": data.attempts(),
                })
            })
            .collect::<Vec<_>>(),
        });
    }

    pub fn stage_progress_snapshots(&self) -> Vec<StageProgressSnapshot> {
        (0..self.stages.len())
            .map(|stage| self.stage_progress_snapshot(stage))
            .collect()
    }

    fn emit_stage_update(&self, stage: usize, session: &mut impl RuntimeSession) {
        session.on_event(RuntimeEvent::StageUpdated(
            self.stage_progress_snapshot(stage),
        ));
    }

    fn emit_scoring_update(&self, session: &mut impl RuntimeSession, reason: &str) {
        session.on_event(RuntimeEvent::ScoringProgress(
            self.scoring_progress_snapshot(),
        ));
        self.trace_scoring_queue_snapshot(reason);
    }

    fn scoring_progress_snapshot(&self) -> ScoringProgressSnapshot {
        ScoringProgressSnapshot::new(
            self.next_scoring_result_order,
            self.next_scoring_job_order + self.pending_scoring_path_ids.len(),
            self.pending_scoring_job_count(),
            self.scored_path_count(),
        )
    }

    fn trace_scoring_job_queued(
        &self,
        reason: &str,
        order: usize,
        path_id: PathId,
        path: &CompletePath,
    ) {
        crate::trace!(self.trace, if should_trace_scoring_detail(order), "scoring_job_queued", [
            let (queued, submitted, completed, pending, scored) = self.scoring_trace_counts();
        ], {
                "reason": reason,
                "order": order,
                "path_id": path_id.0,
                "path_length": path.len(),
                "path_transitions": path_transitions_json(path),
                "queued": queued,
                "submitted": submitted,
                "completed": completed,
                "pending": pending,
                "scored": scored,
        });
    }

    fn trace_registered_path_rescore_deferred(
        &self,
        reason: &str,
        transition: TransitionRef,
        registered_paths: usize,
        was_new_pending_transition: bool,
    ) {
        crate::trace!(self.trace, "registered_path_rescore_deferred", {
                "reason": reason,
                "stage": transition.stage,
                "transition": transition.transition.0,
                "registered_paths": registered_paths,
                "was_new_pending_transition": was_new_pending_transition,
                "pending_registered_rescore_transitions": self.pending_registered_rescore_transitions.len(),
        });
    }

    fn trace_registered_path_rescore_flushed(
        &self,
        reason: &str,
        transition_count: usize,
        candidate_paths: usize,
        enqueued: usize,
    ) {
        crate::trace!(self.trace, "registered_path_rescore_flushed", {
                "reason": reason,
                "transitions": transition_count,
                "candidate_paths": candidate_paths,
                "enqueued": enqueued,
                "queued": self.next_scoring_job_order,
                "pending": self.pending_scoring_job_count(),
        });
    }

    fn trace_process_worker_spawned(&self, reason: &str, worker: &ProcessSynthesisWorker) {
        crate::trace!(self.trace, "process_worker_spawned", {
                "reason": reason,
                "pid": worker.pid(),
                "rss_kb": worker.rss_kb(),
                "recycle_job_limit": PROCESS_WORKER_RECYCLE_JOB_LIMIT,
        });
    }

    fn trace_process_worker_recycled(&self, worker: &ProcessSynthesisWorker) {
        crate::trace!(self.trace, "process_worker_recycled", {
                "pid": worker.pid(),
                "jobs_since_spawn": worker.jobs_since_spawn(),
                "rss_kb": worker.rss_kb(),
                "recycle_job_limit": PROCESS_WORKER_RECYCLE_JOB_LIMIT,
        });
    }

    fn trace_process_worker_pool_snapshot(&self, reason: &str, workers: &[ProcessSynthesisWorker]) {
        crate::trace!(self.trace, "process_worker_pool_snapshot", [
            let rss_values = workers
                .iter()
                .filter_map(|worker| worker.rss_kb())
                .collect::<Vec<_>>();
            let jobs_total = workers
                .iter()
                .map(ProcessSynthesisWorker::jobs_since_spawn)
                .sum::<usize>();
            let jobs_max = workers
                .iter()
                .map(ProcessSynthesisWorker::jobs_since_spawn)
                .max()
                .unwrap_or(0);
            let rss_total_kb = (!rss_values.is_empty()).then(|| rss_values.iter().sum::<u64>());
            let rss_min_kb = rss_values.iter().min().copied();
            let rss_max_kb = rss_values.iter().max().copied();
        ], {
                "reason": reason,
                "workers": workers.len(),
                "rss_total_kb": rss_total_kb,
                "rss_min_kb": rss_min_kb,
                "rss_max_kb": rss_max_kb,
                "jobs_since_spawn_total": jobs_total,
                "jobs_since_spawn_max": jobs_max,
                "recycle_job_limit": PROCESS_WORKER_RECYCLE_JOB_LIMIT,
        });
    }

    fn trace_process_worker_rss_snapshot(
        &self,
        reason: &str,
        pids: &[u32],
        counters: WorkerCounters,
    ) {
        crate::trace!(self.trace, if !pids.is_empty(), "process_worker_rss_snapshot", [
            let rss_values = pids
                .iter()
                .filter_map(|pid| process_rss_kb(*pid))
                .collect::<Vec<_>>();
            let rss_total_kb = (!rss_values.is_empty()).then(|| rss_values.iter().sum::<u64>());
            let rss_min_kb = rss_values.iter().min().copied();
            let rss_max_kb = rss_values.iter().max().copied();
        ], {
                "reason": reason,
                "stage": counters.stage,
                "workers": pids.len(),
                "rss_total_kb": rss_total_kb,
                "rss_min_kb": rss_min_kb,
                "rss_max_kb": rss_max_kb,
                "job_count": counters.job_count,
                "worker_capacity": counters.worker_capacity,
                "active_workers": counters.active_workers,
                "in_flight": counters.in_flight,
                "next_to_send": counters.next_to_send,
                "next_to_apply": counters.next_to_apply,
                "buffered_results": counters.buffered_results,
                "job_interval": PROCESS_WORKER_RSS_TRACE_INTERVAL,
                "idle_interval_ms": PROCESS_WORKER_RSS_TRACE_IDLE_INTERVAL.as_millis(),
        });
    }

    fn trace_scoring_job_completed(&self, reason: &str, fields: ScoringJobTraceFields) {
        crate::trace!(self.trace, if should_trace_scoring_detail(fields.order), "scoring_job_completed", [
            let (queued, submitted, completed, pending, scored) = self.scoring_trace_counts();
        ], {
                "reason": reason,
                "order": fields.order,
                "path_id": fields.path_id.0,
                "path_length": fields.path_length,
                "assigned_candidates": fields.assigned_candidates,
                "received": self.next_scoring_result_order + self.buffered_scoring_results.len(),
                "queued": queued,
                "submitted": submitted,
                "completed": completed,
                "pending": pending,
                "scored": scored,
        });
    }

    fn trace_scoring_job_applied(
        &self,
        reason: &str,
        fields: ScoringJobTraceFields,
        applied_candidates: usize,
    ) {
        crate::trace!(self.trace, if should_trace_scoring_detail(fields.order), "scoring_job_applied", [
            let (queued, submitted, completed, pending, scored) = self.scoring_trace_counts();
        ], {
                "reason": reason,
                "order": fields.order,
                "path_id": fields.path_id.0,
                "path_length": fields.path_length,
                "assigned_candidates": fields.assigned_candidates,
                "applied_candidates": applied_candidates,
                "queued": queued,
                "submitted": submitted,
                "completed": completed,
                "pending": pending,
                "scored": scored,
        });
    }

    fn trace_scoring_queue_snapshot(&self, reason: &str) {
        crate::trace!(self.trace, "scoring_queue_snapshot", [
            let snapshot = self.scoring_progress_snapshot();
            let (queued, submitted, completed, pending, scored) = self.scoring_trace_counts();
        ], {
                "reason": reason,
                "queued": queued,
                "submitted": submitted,
                "completed": completed,
                "pending": pending,
                "scored": scored,
                "total_paths": snapshot.total_paths(),
                "completed_paths": snapshot.completed_paths(),
                "pending_paths": snapshot.pending_paths(),
                "scored_paths": snapshot.scored_paths(),
                "pending_limit": self.scoring_pool.max_pending_jobs,
                "pending_backlog": self.pending_scoring_path_ids.len(),
                "in_flight": self.in_flight_scoring_job_count(),
        });
    }

    fn scoring_trace_counts(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.next_scoring_job_order + self.pending_scoring_path_ids.len(),
            self.next_scoring_job_order,
            self.next_scoring_result_order,
            self.pending_scoring_job_count(),
            self.scored_path_count(),
        )
    }

    fn stage_progress_snapshot(&self, stage: usize) -> StageProgressSnapshot {
        StageProgressSnapshot::with_worker_state(
            stage,
            self.transition_table.stage_stats(stage),
            self.stage_progress_totals[stage],
            self.active_worker_counts[stage],
            self.worker_capacities[stage],
            self.queued_job_counts[stage],
        )
    }

    fn set_stage_worker_state(
        &mut self,
        stage: usize,
        active_workers: usize,
        worker_capacity: usize,
        queued_jobs: usize,
    ) {
        self.active_worker_counts[stage] = active_workers;
        self.worker_capacities[stage] = worker_capacity;
        self.queued_job_counts[stage] = queued_jobs;
    }

    fn search_exhausted(&self) -> bool {
        !(0..self.stages.len()).any(|stage| self.stage_is_frontier_selectable(stage))
    }

    fn retain_top_k_paths(&mut self) {
        let Some(top_k) = self.config.top_k else {
            return;
        };
        if self.scored_paths.len() <= top_k {
            return;
        }

        self.scored_paths.sort_by(Self::compare_scored_paths);
        self.scored_paths.truncate(top_k);
        self.sync_scored_path_keys();
    }

    fn retain_recent_scored_paths(&mut self) {
        let Some(top_k) = self.config.top_k else {
            return;
        };
        let tie_limit = rough_tie_expanded_limit(top_k);
        retain_rough_frontier(&mut self.recent_scored_paths, top_k, tie_limit);
    }

    fn sync_scored_path_keys(&mut self) {
        self.scored_path_keys = self
            .scored_paths
            .iter()
            .map(|path| path.assigned_path().selection_key())
            .collect();
    }

    fn compare_scored_paths(left: &ScoredPath, right: &ScoredPath) -> std::cmp::Ordering {
        left.cost()
            .score()
            .total_cmp(&right.cost().score())
            .then_with(|| left.discovery_order().cmp(&right.discovery_order()))
    }

    fn enqueue_scoring_paths(&mut self, paths: Vec<CompletePath>, reason: &str) -> usize {
        if paths.is_empty() {
            return 0;
        }

        let mut enqueued = 0;
        for path in paths {
            let registration = self.path_registry.register(path);
            if self.queue_scoring_path(registration.id) {
                enqueued += 1;
            }
        }
        self.top_off_scoring_jobs(reason);
        self.trace_scoring_queue_snapshot(reason);
        enqueued
    }

    fn defer_registered_paths_for_transition_rescore(
        &mut self,
        transition: TransitionRef,
        reason: &str,
    ) -> usize {
        let registered_paths = self.path_registry.paths_for_transition(transition).len();
        if registered_paths == 0 {
            return 0;
        }

        let was_new_pending_transition = self
            .pending_registered_rescore_transitions
            .insert(transition);
        self.trace_registered_path_rescore_deferred(
            reason,
            transition,
            registered_paths,
            was_new_pending_transition,
        );
        0
    }

    fn flush_pending_registered_path_rescores(&mut self, reason: &str) -> usize {
        if self.pending_registered_rescore_transitions.is_empty() {
            return 0;
        }

        let transitions = std::mem::take(&mut self.pending_registered_rescore_transitions);
        let transition_count = transitions.len();
        let mut path_ids = BTreeSet::new();
        for transition in transitions {
            path_ids.extend(
                self.path_registry
                    .paths_for_transition(transition)
                    .iter()
                    .copied(),
            );
        }

        let candidate_paths = path_ids.len();
        let enqueued = self
            .enqueue_scoring_path_ids(path_ids.into_iter().collect(), "registered_gadget_rescore");
        self.trace_registered_path_rescore_flushed(
            reason,
            transition_count,
            candidate_paths,
            enqueued,
        );
        enqueued
    }

    fn enqueue_scoring_path_ids(&mut self, path_ids: Vec<PathId>, reason: &str) -> usize {
        if path_ids.is_empty() {
            return 0;
        }

        let mut enqueued = 0;
        for path_id in path_ids {
            if self.queue_scoring_path(path_id) {
                enqueued += 1;
            }
        }
        self.top_off_scoring_jobs(reason);
        self.trace_scoring_queue_snapshot(reason);
        enqueued
    }

    fn queue_scoring_path(&mut self, path_id: PathId) -> bool {
        let path = self.path_registry.path(path_id).clone();
        let signature = self.path_scoring_signature(&path);
        if self
            .latest_scoring_signatures
            .get(&path_id)
            .is_some_and(|existing| existing == &signature)
        {
            return false;
        }
        self.latest_scoring_signatures.insert(path_id, signature);
        let newly_pending = self.pending_scoring_path_set.insert(path_id);
        if newly_pending {
            self.pending_scoring_path_ids.push_back(path_id);
        }
        newly_pending
    }

    fn top_off_scoring_jobs(&mut self, reason: &str) -> usize {
        let mut submitted = 0;
        while self.scoring_pool.has_capacity() {
            let Some(path_id) = self.pending_scoring_path_ids.pop_front() else {
                break;
            };
            self.pending_scoring_path_set.remove(&path_id);
            self.submit_scoring_path(path_id, reason);
            submitted += 1;
        }
        submitted
    }

    fn submit_scoring_path(&mut self, path_id: PathId, reason: &str) {
        let path = self.path_registry.path(path_id).clone();
        let snapshot = self.path_scoring_snapshot(&path);
        let order = self.next_scoring_job_order;
        self.next_scoring_job_order += 1;
        self.scoring_pool.submit(ScoringJob {
            order,
            path_id,
            path: path.clone(),
            snapshot,
            assignment_limit: self.config.top_k.unwrap_or(1),
        });
        self.trace_scoring_job_queued(reason, order, path_id, &path);
    }

    fn path_scoring_snapshot(&self, path: &CompletePath) -> PathScoringSnapshot {
        PathScoringSnapshot::from_table(path, &self.transition_table)
    }

    fn path_scoring_signature(&self, path: &CompletePath) -> PathScoringSignature {
        path.iter()
            .map(|(stage, transition)| {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                (
                    transition_ref,
                    self.transition_table
                        .transition_gadgets(transition_ref)
                        .len(),
                )
            })
            .collect()
    }

    fn apply_ready_scoring_results(&mut self, reason: &str, result_limit: Option<usize>) -> usize {
        let mut scored = 0;
        let mut applied_results = 0;
        while result_limit.is_none_or(|limit| applied_results < limit) {
            let Some(result) = self
                .buffered_scoring_results
                .remove(&self.next_scoring_result_order)
            else {
                break;
            };
            let fields = ScoringJobTraceFields {
                order: result.order,
                path_id: result.path_id,
                path_length: result.path_length,
                assigned_candidates: result.scored_paths.len(),
            };
            let mut applied_candidates = 0;
            self.next_scoring_result_order += 1;
            self.scoring_pool.pending_jobs = self.scoring_pool.pending_jobs.saturating_sub(1);
            applied_results += 1;
            self.recent_scored_paths
                .extend(result.scored_paths.iter().cloned());
            self.retain_recent_scored_paths();
            for scored_path in result.scored_paths {
                let key = scored_path.assigned_path().selection_key();
                if self.scored_path_keys.contains(&key) {
                    continue;
                }
                self.scored_path_keys.insert(key);
                self.scored_paths.push(scored_path);
                scored += 1;
                applied_candidates += 1;
            }
            self.retain_top_k_paths();
            self.trace_scoring_job_applied(reason, fields, applied_candidates);
        }
        scored
    }

    fn stage_has_inputs(&self, stage: usize) -> bool {
        stage == 0
            || self.should_use_virtual_retroactive_inputs(stage)
            || self.transition_table.unique_output_count(stage - 1) > 0
    }

    fn stage_has_remaining_jobs(&mut self, stage: usize) -> bool {
        !self.make_jobs(stage, None, Some(1)).is_empty()
    }

    /// Runs one synthesis pass over a single sorting-network `stage`.
    ///
    /// This is the smallest budgeted unit of search: it enumerates the stage's
    /// pending jobs with [`make_jobs`](Self::make_jobs), executes them (locally
    /// or across a worker pool), and folds each valid result back into the
    /// transition table. [`run_wave`](Self::run_wave) calls this once per stage;
    /// [`run`](Self::run) drives `run_wave` in a loop.
    ///
    /// # Inputs
    ///
    /// - `input_states = Some(states)`: restrict the pass to exactly these
    ///   states. [`run_wave`](Self::run_wave) uses this to forward only the
    ///   *newly produced* upstream outputs downstream, avoiding redundant work.
    /// - `input_states = None`: use the stage's normal inputs (initial state for
    ///   stage 0, previous-stage outputs otherwise), as described on
    ///   [`make_jobs`](Self::make_jobs).
    ///
    /// # Budgets
    ///
    /// - `attempt_budget` bounds how many candidate jobs are attempted; it is
    ///   passed straight through to [`make_jobs`](Self::make_jobs) as its job
    ///   limit and also re-checked in the execution loop.
    /// - `output_budget` is an early-exit threshold: the pass returns as soon as
    ///   `output_budget` *new* unique output states have been discovered for the
    ///   stage, even if attempts remain.
    ///
    /// # Execution
    ///
    /// With `worker_count > 1` and more than one job, work is dispatched to a
    /// worker pool (`run_stage_worker_pool`) that preserves deterministic,
    /// candidate-ordered application of results; otherwise jobs run inline. The
    /// loop also honors `session.should_stop()` for cooperative cancellation and
    /// emits progress/scoring updates as it goes.
    ///
    /// # Side effects
    ///
    /// Valid syntheses are recorded as transitions in the transition table
    /// (feeding future [`make_jobs`](Self::make_jobs) dedup). New terminal paths
    /// discovered while applying transitions are registered and queued for
    /// asynchronous rough scoring; those scoring results are applied later by
    /// the wave/run boundaries, not here.
    ///
    /// Returns a [`StageRunResult`] summarizing attempts, valid outputs, new
    /// unique outputs, new transitions, and discovered paths for this pass.
    pub fn run_stage(
        &mut self,
        stage: usize,
        input_states: Option<&[VectorState]>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<StageRunResult, SynthesisError> {
        let outputs_at_start = self.transition_table.unique_output_count(stage);
        let attempts_at_start = self.transition_table.stage_stats(stage).attempts;
        let jobs = self.make_jobs(stage, input_states, Some(attempt_budget));
        self.run_prepared_stage_jobs(
            PreparedStageRun {
                stage,
                jobs,
                outputs_at_start,
                attempts_at_start,
                attempt_budget,
                output_budget,
            },
            session,
        )
    }

    fn run_prepared_stage_jobs(
        &mut self,
        request: PreparedStageRun,
        session: &mut impl RuntimeSession,
    ) -> Result<StageRunResult, SynthesisError> {
        let PreparedStageRun {
            stage,
            jobs,
            outputs_at_start,
            attempts_at_start,
            attempt_budget,
            output_budget,
        } = request;
        self.stage_progress_totals[stage] =
            self.stage_progress_totals[stage].max(attempts_at_start + jobs.len());
        let worker_capacity = if jobs.is_empty() {
            0
        } else {
            self.config.worker_count.min(jobs.len()).max(1)
        };
        crate::trace!(self.trace, "stage_jobs_prepared", {
                    "stage": stage,
                    "jobs": jobs.len(),
                    "attempt_budget": attempt_budget,
                    "output_budget": output_budget,
                    "attempts_at_start": attempts_at_start,
                    "outputs_at_start": outputs_at_start,
                    "worker_capacity": worker_capacity,
        });
        self.set_stage_worker_state(stage, 0, worker_capacity, jobs.len());
        self.emit_stage_update(stage, session);
        if self.config.worker_count > 1 && jobs.len() > 1 {
            return self.run_stage_worker_pool(
                stage,
                jobs,
                outputs_at_start,
                output_budget,
                session,
            );
        }

        let mut attempts = 0;
        let mut valid_outputs = 0;
        let mut budget_outputs = 0;
        let mut new_transitions = 0;
        let mut discovered_paths = 0;

        let job_count = jobs.len();
        for (job_index, job) in jobs.into_iter().enumerate() {
            if session.should_stop() {
                break;
            }
            if attempts >= attempt_budget {
                break;
            }
            self.set_stage_worker_state(stage, 1, 1, job_count.saturating_sub(job_index + 1));
            self.emit_stage_update(stage, session);
            let result = self.execute_job(&job)?;
            attempts += result.attempts();
            valid_outputs += result.valid_output_count();
            budget_outputs += result.budget_output_count();
            new_transitions += result.new_transition_count();
            discovered_paths += result.discovered_paths();
            self.poll_scoring_jobs_for_reason("stage_job_finished");
            self.emit_scoring_update(session, "stage_job_finished");
            self.set_stage_worker_state(stage, 0, 1, job_count.saturating_sub(job_index + 1));
            self.emit_stage_update(stage, session);

            let new_outputs = self.transition_table.unique_output_count(stage) - outputs_at_start;
            if budget_outputs >= output_budget {
                self.set_stage_worker_state(stage, 0, 0, 0);
                self.emit_stage_update(stage, session);
                return Ok(StageRunResult {
                    stage,
                    attempts,
                    valid_outputs,
                    new_outputs,
                    budget_outputs,
                    new_transitions,
                    discovered_paths,
                });
            }
        }
        self.set_stage_worker_state(stage, 0, 0, 0);
        self.emit_stage_update(stage, session);

        Ok(StageRunResult {
            stage,
            attempts,
            valid_outputs,
            new_outputs: self.transition_table.unique_output_count(stage) - outputs_at_start,
            budget_outputs,
            new_transitions,
            discovered_paths,
        })
    }

    /// Executes a stage's prepared `jobs` across a pool of parallel workers.
    ///
    /// This is the parallel counterpart to the inline loop in
    /// [`run_stage`](Self::run_stage), chosen when `worker_count > 1` and there
    /// is more than one job. It owns the full producer/consumer machinery:
    /// spawning workers, streaming jobs to them, collecting results, and folding
    /// them back into engine state.
    ///
    /// # Backends
    ///
    /// Depending on the resolved worker backend, work runs either on scoped
    /// threads calling `synthesize_worker_job` directly, or on pooled
    /// subprocess [`ProcessSynthesisWorker`]s checked out for the duration of
    /// the stage. Both share the same job channel and event protocol
    /// (`Started` / `Finished`).
    ///
    /// # Deterministic application despite parallelism
    ///
    /// Jobs are dispatched in index order, but workers finish out of order.
    /// Finished results are held in a `BTreeMap` keyed by job index and applied
    /// strictly in submission order (`applied_jobs` cursor). This guarantees the
    /// transition table and discovered paths are built identically regardless of
    /// worker timing — the determinism guarantee surfaced on [`run`](Self::run).
    ///
    /// # Early termination
    ///
    /// The pool stops sending and/or applying when any of these occur:
    /// `output_budget` new unique outputs (relative to `outputs_at_start`) have
    /// been reached, `session.should_stop()` is observed, or a worker returns an
    /// error (the first error is captured and returned after the pool drains).
    ///
    /// Throughout, it maintains UI worker-state counters and emits detailed
    /// trace events (submission, start, finish, apply, periodic RSS snapshots
    /// for subprocess workers). Returns the aggregated [`StageRunResult`].
    fn run_stage_worker_pool(
        &mut self,
        stage: usize,
        jobs: Vec<SynthesisJob>,
        outputs_at_start: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<StageRunResult, SynthesisError> {
        let worker_count = self.config.worker_count.min(jobs.len()).max(1);
        let config = self.config;
        let backend = self.resolved_worker_backend();
        let mut process_workers = if backend == WorkerBackendArg::Process {
            Some(self.take_or_create_workers(worker_count)?)
        } else {
            None
        };
        let process_worker_pids = process_workers
            .as_ref()
            .map(|workers| {
                workers
                    .iter()
                    .map(ProcessSynthesisWorker::pid)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let (job_sender, job_receiver) = mpsc::channel::<Option<WorkerSynthesisJob>>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (event_sender, event_receiver) = mpsc::channel::<WorkerPoolEvent>();
        crate::trace!(self.trace, "worker_pool_started", {
                    "stage": stage,
                    "jobs": jobs.len(),
                    "worker_capacity": worker_count,
                    "worker_backend": backend.cli_name(),
                    "outputs_at_start": outputs_at_start,
                    "output_budget": output_budget,
        });

        thread::scope(|scope| {
            let mut process_handles = Vec::new();
            if let Some(workers) = process_workers.take() {
                for mut process_worker in workers {
                    let job_receiver = Arc::clone(&job_receiver);
                    let event_sender = event_sender.clone();
                    process_handles.push(scope.spawn(move || {
                        loop {
                            let message = {
                                let receiver = job_receiver
                                    .lock()
                                    .expect("worker job receiver lock should not be poisoned");
                                receiver.recv()
                            };
                            let Ok(Some(worker_job)) = message else {
                                break;
                            };
                            let index = worker_job.index;
                            if event_sender.send(WorkerPoolEvent::Started(index)).is_err() {
                                break;
                            }
                            let result = process_worker.run_job(worker_job);
                            if event_sender
                                .send(WorkerPoolEvent::Finished(index, result))
                                .is_err()
                            {
                                break;
                            }
                        }
                        process_worker
                    }));
                }
            } else {
                for _ in 0..worker_count {
                    let job_receiver = Arc::clone(&job_receiver);
                    let event_sender = event_sender.clone();
                    scope.spawn(move || {
                        loop {
                            let message = {
                                let receiver = job_receiver
                                    .lock()
                                    .expect("worker job receiver lock should not be poisoned");
                                receiver.recv()
                            };
                            let Ok(Some(worker_job)) = message else {
                                break;
                            };
                            let index = worker_job.index;
                            if event_sender.send(WorkerPoolEvent::Started(index)).is_err() {
                                break;
                            }
                            let result = synthesize_worker_job(config, worker_job);
                            if event_sender
                                .send(WorkerPoolEvent::Finished(index, result))
                                .is_err()
                            {
                                break;
                            }
                        }
                    });
                }
            }
            drop(event_sender);

            let mut next_to_send = 0;
            let mut in_flight = 0;
            let mut active_workers = 0;
            let mut stop_sending = false;
            let mut stop_applying = false;
            let mut applied_jobs = 0;
            let mut attempts = 0;
            let mut valid_outputs = 0;
            let mut budget_outputs = 0;
            let mut new_transitions = 0;
            let mut discovered_paths = 0;
            let mut buffered_results = BTreeMap::new();
            let mut next_rss_trace_job = PROCESS_WORKER_RSS_TRACE_INTERVAL;

            while !session.should_stop() && in_flight < worker_count && next_to_send < jobs.len() {
                let job_index = next_to_send;
                self.send_worker_job(&job_sender, job_index, &jobs[job_index]);
                next_to_send += 1;
                in_flight += 1;
                let input_state = self.input_state_for_job(jobs[job_index]);
                self.trace_job_submitted(
                    &jobs[job_index],
                    &input_state,
                    job_index,
                    WorkerCounters {
                        stage,
                        job_count: jobs.len(),
                        worker_capacity: worker_count,
                        active_workers,
                        in_flight,
                        next_to_send,
                        next_to_apply: applied_jobs,
                        buffered_results: buffered_results.len(),
                    },
                );
            }
            self.set_stage_worker_state(
                stage,
                active_workers,
                worker_count,
                pending_worker_queue_len(jobs.len(), next_to_send, in_flight, active_workers),
            );
            self.emit_stage_update(stage, session);
            self.trace_worker_counters(
                "initial_submission",
                WorkerCounters {
                    stage,
                    job_count: jobs.len(),
                    worker_capacity: worker_count,
                    active_workers,
                    in_flight,
                    next_to_send,
                    next_to_apply: applied_jobs,
                    buffered_results: buffered_results.len(),
                },
            );
            self.trace_process_worker_rss_snapshot(
                "initial_submission",
                &process_worker_pids,
                WorkerCounters {
                    stage,
                    job_count: jobs.len(),
                    worker_capacity: worker_count,
                    active_workers,
                    in_flight,
                    next_to_send,
                    next_to_apply: applied_jobs,
                    buffered_results: buffered_results.len(),
                },
            );
            if session.should_stop() {
                stop_sending = true;
                stop_applying = true;
                crate::trace!(self.trace, "worker_pool_stop_requested", {
                        "stage": stage,
                        "reason": "session_stop",
                        "next_to_send": next_to_send,
                        "in_flight": in_flight,
                        "active_workers": active_workers,
                });
            }

            let mut first_error = None;
            while in_flight > 0 {
                let event = match event_receiver
                    .recv_timeout(PROCESS_WORKER_RSS_TRACE_IDLE_INTERVAL)
                {
                    Ok(event) => event,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        self.trace_process_worker_rss_snapshot(
                            "idle_timeout",
                            &process_worker_pids,
                            WorkerCounters {
                                stage,
                                job_count: jobs.len(),
                                worker_capacity: worker_count,
                                active_workers,
                                in_flight,
                                next_to_send,
                                next_to_apply: applied_jobs,
                                buffered_results: buffered_results.len(),
                            },
                        );
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        panic!("worker event channel should stay open while jobs are in flight");
                    }
                };

                match event {
                    WorkerPoolEvent::Started(index) => {
                        active_workers += 1;
                        self.set_stage_worker_state(
                            stage,
                            active_workers,
                            worker_count,
                            pending_worker_queue_len(
                                jobs.len(),
                                next_to_send,
                                in_flight,
                                active_workers,
                            ),
                        );
                        self.emit_stage_update(stage, session);
                        crate::trace!(self.trace, "job_started", {
                                    "stage": stage,
                                    "job_index": index,
                                    "candidate_index": jobs[index].candidate_index(),
                                    "active_workers": active_workers,
                                    "in_flight": in_flight,
                        });
                        self.trace_worker_counters(
                            "job_started",
                            WorkerCounters {
                                stage,
                                job_count: jobs.len(),
                                worker_capacity: worker_count,
                                active_workers,
                                in_flight,
                                next_to_send,
                                next_to_apply: applied_jobs,
                                buffered_results: buffered_results.len(),
                            },
                        );
                        continue;
                    }
                    WorkerPoolEvent::Finished(index, result) => {
                        in_flight -= 1;
                        active_workers = active_workers.saturating_sub(1);
                        self.set_stage_worker_state(
                            stage,
                            active_workers,
                            worker_count,
                            pending_worker_queue_len(
                                jobs.len(),
                                next_to_send,
                                in_flight,
                                active_workers,
                            ),
                        );
                        match &result {
                            Ok(output) => crate::trace!(self.trace, "job_finished", {
                                        "stage": stage,
                                        "job_index": index,
                                        "candidate_index": output.id.candidate_index(),
                                        "status": "ok",
                                        "valid_outputs": output.results.len(),
                                        "active_workers": active_workers,
                                        "in_flight": in_flight,
                            }),
                            Err(error) => crate::trace!(self.trace, "job_finished", {
                                        "stage": stage,
                                        "job_index": index,
                                        "candidate_index": jobs[index].candidate_index(),
                                        "status": "error",
                                        "error": error.to_string(),
                                        "active_workers": active_workers,
                                        "in_flight": in_flight,
                            }),
                        }
                        self.trace_worker_counters(
                            "job_finished",
                            WorkerCounters {
                                stage,
                                job_count: jobs.len(),
                                worker_capacity: worker_count,
                                active_workers,
                                in_flight,
                                next_to_send,
                                next_to_apply: applied_jobs,
                                buffered_results: buffered_results.len(),
                            },
                        );
                        if stop_applying {
                            continue;
                        }
                        buffered_results.insert(index, result);
                        while !stop_applying {
                            let Some(result) = buffered_results.remove(&applied_jobs) else {
                                break;
                            };
                            match result {
                                Ok(output) => {
                                    let job_index = applied_jobs;
                                    let job_stage = output.id.stage();
                                    let candidate_index = output.id.candidate_index();
                                    let worker_valid_outputs = output.results.len();
                                    crate::trace!(self.trace, "job_apply_started", {
                                                "stage": job_stage,
                                                "job_index": job_index,
                                                "candidate_index": candidate_index,
                                                "valid_outputs": worker_valid_outputs,
                                    });
                                    let apply_result = self.apply_synthesis_output(output);
                                    attempts += apply_result.attempts();
                                    valid_outputs += apply_result.valid_output_count();
                                    budget_outputs += apply_result.budget_output_count();
                                    new_transitions += apply_result.new_transition_count();
                                    discovered_paths += apply_result.discovered_paths();
                                    applied_jobs += 1;
                                    if applied_jobs.is_multiple_of(
                                        SCORING_RESULT_POLL_INTERVAL_PER_SYNTHESIS_JOBS,
                                    ) {
                                        self.poll_some_scoring_jobs(
                                            "job_apply_finished",
                                            SCORING_RESULT_POLL_LIMIT_PER_SYNTHESIS_INTERVAL,
                                        );
                                        self.emit_scoring_update(session, "job_apply_finished");
                                    }
                                    if applied_jobs >= next_rss_trace_job {
                                        self.trace_process_worker_rss_snapshot(
                                            "job_apply_interval",
                                            &process_worker_pids,
                                            WorkerCounters {
                                                stage,
                                                job_count: jobs.len(),
                                                worker_capacity: worker_count,
                                                active_workers,
                                                in_flight,
                                                next_to_send,
                                                next_to_apply: applied_jobs,
                                                buffered_results: buffered_results.len(),
                                            },
                                        );
                                        while next_rss_trace_job <= applied_jobs {
                                            next_rss_trace_job += PROCESS_WORKER_RSS_TRACE_INTERVAL;
                                        }
                                    }
                                    self.emit_stage_update(stage, session);
                                    crate::trace!(self.trace, "job_apply_finished", [
                                        let stage_new_outputs =
                                            self.transition_table.unique_output_count(stage)
                                                - outputs_at_start;
                                    ], {
                                                "stage": job_stage,
                                                "job_index": job_index,
                                                "candidate_index": candidate_index,
                                                "attempts": apply_result.attempts(),
                                                "valid_outputs": apply_result.valid_output_count(),
                                                "budget_outputs": apply_result.budget_output_count(),
                                                "new_transitions": apply_result.new_transition_count(),
                                                "discovered_paths": apply_result.discovered_paths(),
                                                "total_attempts": attempts,
                                                "total_valid_outputs": valid_outputs,
                                                "total_budget_outputs": budget_outputs,
                                                "total_new_transitions": new_transitions,
                                                "stage_new_outputs": stage_new_outputs,
                                    });
                                    self.trace_worker_counters(
                                        "job_apply_finished",
                                        WorkerCounters {
                                            stage,
                                            job_count: jobs.len(),
                                            worker_capacity: worker_count,
                                            active_workers,
                                            in_flight,
                                            next_to_send,
                                            next_to_apply: applied_jobs,
                                            buffered_results: buffered_results.len(),
                                        },
                                    );
                                    if session.should_stop() {
                                        crate::trace!(self.trace, "worker_pool_stop_requested", {
                                                "stage": stage,
                                                "reason": "session_stop",
                                                "next_to_send": next_to_send,
                                                "in_flight": in_flight,
                                                "active_workers": active_workers,
                                        });
                                        break;
                                    }

                                    let new_outputs =
                                        self.transition_table.unique_output_count(stage)
                                            - outputs_at_start;
                                    if budget_outputs >= output_budget {
                                        stop_sending = true;
                                        stop_applying = true;
                                        crate::trace!(self.trace, "worker_pool_stop_requested", {
                                                "stage": stage,
                                                "reason": "output_budget",
                                                "new_outputs": new_outputs,
                                                "budget_outputs": budget_outputs,
                                                "output_budget": output_budget,
                                                "next_to_send": next_to_send,
                                                "in_flight": in_flight,
                                                "active_workers": active_workers,
                                        });
                                    }
                                }
                                Err(error) => {
                                    crate::trace!(self.trace, "worker_pool_stop_requested", {
                                            "stage": stage,
                                            "reason": "worker_error",
                                            "error": error.to_string(),
                                            "next_to_send": next_to_send,
                                            "in_flight": in_flight,
                                            "active_workers": active_workers,
                                    });
                                    first_error = Some(error);
                                    stop_sending = true;
                                    stop_applying = true;
                                }
                            }
                        }
                    }
                }

                if session.should_stop() {
                    stop_sending = true;
                    stop_applying = true;
                    crate::trace!(self.trace, "worker_pool_stop_requested", {
                            "stage": stage,
                            "reason": "session_stop",
                            "next_to_send": next_to_send,
                            "in_flight": in_flight,
                            "active_workers": active_workers,
                    });
                }

                while !stop_sending && in_flight < worker_count && next_to_send < jobs.len() {
                    if session.should_stop() {
                        stop_sending = true;
                        crate::trace!(self.trace, "worker_pool_stop_requested", {
                                "stage": stage,
                                "reason": "session_stop",
                                "next_to_send": next_to_send,
                                "in_flight": in_flight,
                                "active_workers": active_workers,
                        });
                        break;
                    }
                    let job_index = next_to_send;
                    self.send_worker_job(&job_sender, job_index, &jobs[job_index]);
                    next_to_send += 1;
                    in_flight += 1;
                    let input_state = self.input_state_for_job(jobs[job_index]);
                    self.trace_job_submitted(
                        &jobs[job_index],
                        &input_state,
                        job_index,
                        WorkerCounters {
                            stage,
                            job_count: jobs.len(),
                            worker_capacity: worker_count,
                            active_workers,
                            in_flight,
                            next_to_send,
                            next_to_apply: applied_jobs,
                            buffered_results: buffered_results.len(),
                        },
                    );
                }
                self.set_stage_worker_state(
                    stage,
                    active_workers,
                    worker_count,
                    pending_worker_queue_len(jobs.len(), next_to_send, in_flight, active_workers),
                );
                self.emit_stage_update(stage, session);
                self.trace_worker_counters(
                    "post_refill",
                    WorkerCounters {
                        stage,
                        job_count: jobs.len(),
                        worker_capacity: worker_count,
                        active_workers,
                        in_flight,
                        next_to_send,
                        next_to_apply: applied_jobs,
                        buffered_results: buffered_results.len(),
                    },
                );
            }

            for _ in 0..worker_count {
                let _ = job_sender.send(None);
            }
            let returned_process_workers = process_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("process worker thread should not panic")
                })
                .collect::<Vec<_>>();
            if !returned_process_workers.is_empty() {
                self.return_process_workers(returned_process_workers);
            }

            self.set_stage_worker_state(stage, 0, 0, 0);
            self.emit_stage_update(stage, session);
            self.trace_worker_counters(
                "finished",
                WorkerCounters {
                    stage,
                    job_count: jobs.len(),
                    worker_capacity: 0,
                    active_workers: 0,
                    in_flight: 0,
                    next_to_send,
                    next_to_apply: applied_jobs,
                    buffered_results: buffered_results.len(),
                },
            );

            if let Some(error) = first_error {
                return Err(error);
            }

            let new_outputs = self.transition_table.unique_output_count(stage) - outputs_at_start;
            crate::trace!(self.trace, "worker_pool_finished", {
                    "stage": stage,
                    "attempts": attempts,
                    "valid_outputs": valid_outputs,
                    "new_outputs": new_outputs,
                    "budget_outputs": budget_outputs,
                    "new_transitions": new_transitions,
                    "discovered_paths": discovered_paths,
                    "job_count": jobs.len(),
                    "submitted_jobs": next_to_send,
                    "applied_jobs": applied_jobs,
                    "buffered_results": 0,
            });
            Ok(StageRunResult {
                stage,
                attempts,
                valid_outputs,
                new_outputs,
                budget_outputs,
                new_transitions,
                discovered_paths,
            })
        })
    }

    fn send_worker_job(
        &self,
        sender: &mpsc::Sender<Option<WorkerSynthesisJob>>,
        index: usize,
        job: &SynthesisJob,
    ) {
        sender
            .send(Some(WorkerSynthesisJob {
                index,
                id: *job,
                input_state: self.input_state_for_job(*job),
                graph: self
                    .candidate_graph(job.candidate_index())
                    .expect("job candidate index should be valid")
                    .clone(),
                target_pairs: self.stage_target_pairs(job.stage()),
                allow_any_lane_order: self.allow_any_lane_order_for_stage(job.stage()),
            }))
            .expect("worker job channel should stay open while scheduling");
    }

    fn ordered_outputs_from_stage(&self, stage: usize) -> Vec<StateId> {
        let mut outputs = self
            .transition_table
            .stage(stage)
            .unique_output_ids()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        outputs.sort_by_key(|state_id| self.transition_table.state_as_zero_based_tuple(*state_id));
        outputs
    }

    fn ordered_inputs_for_stage(&self, stage: usize, input_states: Vec<StateId>) -> Vec<StateId> {
        if stage == 0 {
            return input_states;
        }

        let (mut covered, mut uncovered): (Vec<_>, Vec<_>) =
            input_states.into_iter().partition(|state_id| {
                self.transition_table
                    .was_input_attempted_by_id(stage, *state_id)
            });
        uncovered.append(&mut covered);
        uncovered
    }

    fn candidate_graph(&self, candidate_index: usize) -> Option<&GadgetGraph> {
        let shared_start = self.shallow_candidates.len();
        let full_start = shared_start + self.shared_prefix_deep_candidates.len();
        if candidate_index < shared_start {
            self.shallow_candidates.get(candidate_index)
        } else if candidate_index < full_start {
            self.shared_prefix_deep_candidates
                .get(candidate_index.checked_sub(shared_start)?)
        } else {
            self.full_deep_candidates
                .get(candidate_index.checked_sub(full_start)?)
        }
    }

    fn input_state_for_job(&self, job: SynthesisJob) -> VectorState {
        self.transition_table.state_as_vector_state(job.input())
    }

    fn allow_any_lane_order_for_stage(&self, stage: usize) -> bool {
        !(self.config.natural_order && stage + 1 == self.stages.len())
    }

    fn stage_target_pairs(&self, stage: usize) -> Box<[TargetPair]> {
        self.stage_target_pairs[stage].clone()
    }

    fn trace_worker_counters(&self, reason: &str, counters: WorkerCounters) {
        crate::trace!(self.trace, "worker_counters", {
                "reason": reason,
                "stage": counters.stage,
                "job_count": counters.job_count,
                "worker_capacity": counters.worker_capacity,
                "active_workers": counters.active_workers,
                "in_flight": counters.in_flight,
                "queued_jobs": counters.queued_jobs(),
                "next_to_send": counters.next_to_send,
                "next_to_apply": counters.next_to_apply,
                "buffered_results": counters.buffered_results,
        });
    }

    fn trace_job_submitted(
        &self,
        job: &SynthesisJob,
        input_state: &VectorState,
        job_index: usize,
        counters: WorkerCounters,
    ) {
        crate::trace!(self.trace, "job_submitted", {
                "stage": job.stage(),
                "job_index": job_index,
                "candidate_index": job.candidate_index(),
                "input_top": input_state.top(),
                "input_bottom": input_state.bottom(),
                "job_count": counters.job_count,
                "worker_capacity": counters.worker_capacity,
                "active_workers": counters.active_workers,
                "in_flight": counters.in_flight,
                "queued_jobs": counters.queued_jobs(),
                "next_to_send": counters.next_to_send,
                "next_to_apply": counters.next_to_apply,
                "buffered_results": counters.buffered_results,
        });
    }
}

impl Drop for WaveEngine {
    fn drop(&mut self) {
        for worker in self.process_workers.drain(..) {
            worker.shutdown();
        }
    }
}

fn pending_worker_queue_len(
    job_count: usize,
    next_to_send: usize,
    in_flight: usize,
    active_workers: usize,
) -> usize {
    let sent_not_started = in_flight.saturating_sub(active_workers);
    job_count.saturating_sub(next_to_send) + sent_not_started
}

/// Snapshot of synthesis-worker scheduler counters for tracing.
///
/// This is emitted into runtime trace events and converted into stage progress
/// snapshots. `next_to_send` and `next_to_apply` are ordered job cursors; the
/// distinction matters because worker results may finish out of order but must
/// still be applied in candidate order.
#[derive(Clone, Copy)]
struct WorkerCounters {
    stage: usize,
    job_count: usize,
    worker_capacity: usize,
    active_workers: usize,
    in_flight: usize,
    next_to_send: usize,
    next_to_apply: usize,
    buffered_results: usize,
}

impl WorkerCounters {
    fn queued_jobs(self) -> usize {
        pending_worker_queue_len(
            self.job_count,
            self.next_to_send,
            self.in_flight,
            self.active_workers,
        )
    }
}

fn path_transitions_json(path: &CompletePath) -> Vec<Value> {
    path.iter()
        .map(|(stage, transition)| {
            json!({
                "stage": stage,
                "transition": transition.0,
            })
        })
        .collect()
}

fn rough_tie_expanded_limit(top_k: usize) -> usize {
    top_k
        .saturating_mul(ROUGH_TIE_EXPANSION_FACTOR)
        .max(top_k)
        .max(1)
}

fn retain_rough_frontier(paths: &mut Vec<ScoredPath>, top_k: usize, tie_limit: usize) {
    if top_k == 0 {
        paths.clear();
        return;
    }
    if paths.len() <= top_k {
        return;
    }

    paths.sort_by(compare_rough_frontier_paths);
    let cutoff_score = paths[top_k - 1].cost().score();
    let tied_len = paths
        .iter()
        .position(|path| path.cost().score().total_cmp(&cutoff_score).is_gt())
        .unwrap_or(paths.len());
    paths.truncate(tied_len.min(tie_limit.max(top_k)));
}

fn compare_rough_frontier_paths(left: &ScoredPath, right: &ScoredPath) -> std::cmp::Ordering {
    left.cost()
        .score()
        .total_cmp(&right.cost().score())
        .then_with(|| {
            left.assigned_path()
                .selection_key()
                .cmp(&right.assigned_path().selection_key())
        })
        .then_with(|| left.discovery_order().cmp(&right.discovery_order()))
}

fn should_trace_scoring_detail(order: usize) -> bool {
    order < SCORING_TRACE_DETAIL_LIMIT || order.is_multiple_of(SCORING_TRACE_DETAIL_INTERVAL)
}

fn process_rss_kb(pid: u32) -> Option<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?;
        value
            .split_whitespace()
            .next()
            .and_then(|kb| kb.parse::<u64>().ok())
    })
}

fn synthesize_worker_job(
    config: WaveConfig,
    worker_job: WorkerSynthesisJob,
) -> Result<SynthesisJobOutput, SynthesisError> {
    let synth = GadgetSynthesizer::new(to_synth_arch(config.arch), to_synth_dtype(config.dtype));
    let results = synth.synthesize_graph(
        &worker_job.graph,
        &worker_job.input_state,
        &worker_job.target_pairs,
        SynthesisOptions {
            max_unique_outputs: config.max_unique_outputs,
            allow_any_lane_order: worker_job.allow_any_lane_order,
        },
    )?;

    Ok(SynthesisJobOutput {
        id: worker_job.id,
        results,
    })
}

fn stage_output_limits(stages: &[BitonicStage], natural_order: bool) -> Vec<usize> {
    stages
        .iter()
        .enumerate()
        .map(|(stage, bitonic_stage)| {
            if natural_order && stage + 1 == stages.len() {
                1
            } else {
                saturating_factorial(bitonic_stage.pairs().len())
            }
        })
        .collect()
}

fn saturating_factorial(n: usize) -> usize {
    (1..=n).fold(1usize, usize::saturating_mul)
}

fn checked_factorials_u128(n: usize) -> Box<[u128]> {
    let mut factorials = vec![1_u128; n + 1];
    for value in 1..=n {
        factorials[value] = factorials[value - 1]
            .checked_mul(value as u128)
            .expect("retroactive input factorial should fit in u128");
    }
    factorials.into_boxed_slice()
}

fn create_initial_state(first_stage: &BitonicStage) -> VectorState {
    VectorState::new(
        first_stage
            .pairs()
            .iter()
            .map(|(top, _)| lane_label_from_one_based(*top))
            .collect(),
        first_stage
            .pairs()
            .iter()
            .map(|(_, bottom)| lane_label_from_one_based(*bottom))
            .collect(),
    )
}

fn target_pairs_for_stage(stage: &BitonicStage) -> Box<[TargetPair]> {
    stage
        .pairs()
        .iter()
        .map(|(left, right)| {
            (
                lane_label_from_one_based(*left),
                lane_label_from_one_based(*right),
            )
        })
        .collect()
}

fn lane_label_from_one_based(label: usize) -> LaneLabel {
    let zero_based = label
        .checked_sub(1)
        .expect("bitonic stage lane labels should be one-based");
    LaneLabel::try_from(zero_based).expect("lane label should fit in u8")
}

fn register_bits(arch: ArchArg) -> usize {
    match arch {
        ArchArg::Avx2 => 256,
        ArchArg::Avx512 => 512,
    }
}

fn dtype_bits(dtype: DTypeArg) -> usize {
    dtype.element_bits()
}

fn to_synth_arch(arch: ArchArg) -> SynthArch {
    match arch {
        ArchArg::Avx2 => SynthArch::Avx2,
        ArchArg::Avx512 => SynthArch::Avx512,
    }
}

fn to_synth_dtype(dtype: DTypeArg) -> SynthDType {
    match dtype {
        DTypeArg::I32 => SynthDType::I32,
        DTypeArg::I64 => SynthDType::I64,
        _ => unreachable!(
            "unsupported synthesis dtype should be rejected before constructing WaveEngine"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_wave_config() -> WaveConfig {
        WaveConfig {
            num_vecs: 2,
            arch: ArchArg::Avx2,
            dtype: DTypeArg::I64,
            gadget_depth: 2,
            deep_search_mode: DeepSearchModeArg::Adaptive,
            natural_order: false,
            retroactive_input: false,
            top_k: Some(1),
            worker_count: 1,
            worker_backend: WorkerBackendArg::InProcess,
            max_unique_outputs: 3,
        }
    }

    fn test_chain_state(index: u8) -> VectorState {
        let base = index * 8;
        VectorState::new(
            vec![base, base + 1, base + 2, base + 3],
            vec![base + 4, base + 5, base + 6, base + 7],
        )
    }

    fn test_empty_gadget() -> PermutationGadget {
        PermutationGadget::new(Vec::new(), Vec::new())
    }

    fn test_expensive_gadget() -> PermutationGadget {
        PermutationGadget::new(
            vec![InstructionSpec::new(
                "_mm256_permutexvar_epi32",
                BTreeMap::new(),
            )],
            Vec::new(),
        )
    }

    #[test]
    fn worker_job_payload_separates_identity_state_and_execution_context() {
        let id = SynthesisJob {
            stage: 0,
            input: StateId(7),
            candidate_index: 3,
            retroactive_rank: None,
        };
        let input_state = VectorState::new(vec![0_u8], vec![1]);
        let job: SynthesisJob = id;
        let worker_job = WorkerSynthesisJob {
            index: 11,
            id,
            input_state,
            graph: GadgetGraph::new(None, None),
            target_pairs: Box::new([(0_u8, 1)]),
            allow_any_lane_order: true,
        };

        assert_eq!(job, id);
        assert_eq!(worker_job.id, id);
        assert_eq!(&*worker_job.target_pairs, &[(0, 1)]);
        assert!(worker_job.allow_any_lane_order);
    }

    #[test]
    fn adaptive_deep_jobs_prioritize_topk_state_with_expensive_suffix() {
        let mut engine = WaveEngine::new(test_wave_config()).expect("engine should initialize");
        let states = (0..=engine.stages().len())
            .map(|index| test_chain_state(index as u8))
            .collect::<Vec<_>>();

        for stage in 0..engine.stages().len() {
            let gadget = if stage == 0 {
                test_expensive_gadget()
            } else {
                test_empty_gadget()
            };
            engine.transition_table_mut().add_transition(
                stage,
                &states[stage],
                &states[stage + 1],
                gadget,
            );
        }

        let first_transition = engine
            .transition_table()
            .transition_ref_for_zero_based_tuples(0, &states[0].as_tuple(), &states[1].as_tuple())
            .expect("first transition should resolve");
        let first_input = engine
            .transition_table()
            .transition(first_transition)
            .input();
        let terminal_stage = engine.stages().len() - 1;
        let terminal = engine
            .transition_table()
            .transition_ref_for_zero_based_tuples(
                terminal_stage,
                &states[terminal_stage].as_tuple(),
                &states[terminal_stage + 1].as_tuple(),
            )
            .expect("terminal transition should resolve");

        assert_eq!(engine.discover_paths_for_transition(terminal, None), 1);
        assert_eq!(engine.drain_scoring_jobs(), 1);

        let targets = engine.adaptive_deep_targets();
        let first_primary = targets
            .iter()
            .find(|target| !target.exploratory)
            .expect("retained top-k path should create primary adaptive targets");

        assert_eq!(first_primary.stage, 0);
        assert_eq!(first_primary.input, first_input);
        assert_eq!(first_primary.topk_hits, 1);
        assert_eq!(first_primary.expensive_suffix_count, 1);

        let jobs = engine.adaptive_deep_jobs(&targets, 10);

        assert_eq!(jobs.len(), 10);
        assert_eq!(jobs[0].stage(), 0);
        assert_eq!(jobs[0].input(), first_input);
        assert_eq!(
            jobs[0].candidate_index(),
            engine.full_deep_candidate_indexes().start
        );
    }
}
