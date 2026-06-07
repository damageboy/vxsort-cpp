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
    Arch as SynthArch, DType as SynthDType, GadgetGraph, GadgetSynthesizer, PermutationGadget,
    SynthesisError, SynthesisOptions, VectorState,
};

use crate::bitonic_sorter::{BitonicSorter, BitonicStage};
use crate::runtime::{
    NullRuntimeSession, RuntimeEvent, RuntimeSession, ScoringProgressSnapshot,
    StageProgressSnapshot,
};
use crate::runtime_trace::RuntimeTrace;
use crate::scoring::{
    AssignedPath, AssignedPathKey, DummyScorer, PathCost, PathScoringSnapshot, Scorer,
};
use crate::transition_table::{
    CompletePath, PathId, PathRegistry, State, StateId, TransitionRef, TransitionTable,
};
use crate::{ArchArg, DTypeArg, WorkerBackendArg};
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
    /// Whether to append the final reorder stage that restores natural lane
    /// order.
    pub natural_order: bool,
    /// Whether stage 0 should be prepopulated from retroactive inputs instead
    /// of only forwarding from the initial state.
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
    config: WaveConfig,
    elements_per_vector: usize,
    total_elements: usize,
    stages: Vec<BitonicStage>,
    shallow_candidates: Vec<GadgetGraph>,
    deep_candidates: Vec<GadgetGraph>,
    transition_table: TransitionTable,
    stage_progress_totals: Vec<usize>,
    active_worker_counts: Vec<usize>,
    worker_capacities: Vec<usize>,
    queued_job_counts: Vec<usize>,
    initial_state: VectorState,
    initial_state_id: StateId,
    wave_count: usize,
    exhausted_stages: HashSet<usize>,
    stalled_stages: HashSet<usize>,
    scored_path_keys: HashSet<AssignedPathKey>,
    scored_paths: Vec<ScoredPath>,
    recent_scored_paths: Vec<ScoredPath>,
    scoring_pool: ScoringPool,
    next_scoring_job_order: usize,
    next_scoring_result_order: usize,
    buffered_scoring_results: BTreeMap<usize, ScoringJobResult>,
    latest_scoring_signatures: HashMap<PathId, PathScoringSignature>,
    pending_scoring_path_ids: VecDeque<PathId>,
    pending_scoring_path_set: HashSet<PathId>,
    pending_registered_rescore_transitions: BTreeSet<TransitionRef>,
    path_registry: PathRegistry,
    process_workers: Vec<ProcessSynthesisWorker>,
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
    discovered_paths: usize,
}

impl TransitionRecordResult {
    pub fn transition_added(&self) -> bool {
        self.transition_added
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
    }
}

/// One synthesis unit scheduled against a stage input and candidate graph.
///
/// A job identifies the stage being expanded, the input state at that stage,
/// and the candidate graph index to ask Z3 to synthesize. The graph itself is
/// attached later by the internal worker job wrapper so the public job identity
/// stays compact and serializable through process-worker wrappers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthesisJob {
    stage: usize,
    input: StateId,
    input_state: VectorState,
    candidate_index: usize,
}

impl SynthesisJob {
    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn input_state(&self) -> &VectorState {
        &self.input_state
    }

    pub fn input(&self) -> StateId {
        self.input
    }

    pub fn candidate_index(&self) -> usize {
        self.candidate_index
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
    job: SynthesisJob,
    graph: GadgetGraph,
    target_pairs: Vec<(u64, u64)>,
    stage_count: usize,
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
    target_pairs: Vec<(u64, u64)>,
    stage_count: usize,
}

impl ProcessWorkerJob {
    fn from_worker_job(worker_job: WorkerSynthesisJob) -> Self {
        Self {
            index: worker_job.index,
            stage: worker_job.job.stage,
            input: worker_job.job.input.0,
            input_state: worker_job.job.input_state,
            candidate_index: worker_job.job.candidate_index,
            target_pairs: worker_job.target_pairs,
            stage_count: worker_job.stage_count,
        }
    }

    fn to_synthesis_job(&self) -> SynthesisJob {
        SynthesisJob {
            stage: self.stage,
            input: StateId(self.input),
            input_state: self.input_state.clone(),
            candidate_index: self.candidate_index,
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
    job: ProcessWorkerJob,
    results: Vec<(PermutationGadget, VectorState)>,
}

impl ProcessWorkerOutput {
    fn into_synthesis_output(self) -> SynthesisJobOutput {
        SynthesisJobOutput {
            job: self.job.to_synthesis_job(),
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
    job: SynthesisJob,
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
        _trace: &mut RuntimeTrace,
        _reason: &str,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Default observer used when no extra wave-progress behavior is needed.
struct NullWaveProgressObserver;

impl WaveProgressObserver for NullWaveProgressObserver {}

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

    pub fn new_transitions(&self) -> usize {
        self.new_transitions
    }

    pub fn discovered_paths(&self) -> usize {
        self.discovered_paths
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
/// `search_exhausted` is true only when all stages are exhausted or stalled
/// according to the wave scheduler; a `max_waves` limit can end the search
/// with this set to false.
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

/// Long-lived synthesis subprocess managed by the coordinator.
///
/// The worker is spawned as the same executable with the hidden
/// `__synthesis-worker` subcommand. The coordinator talks to it over framed
/// stdin/stdout messages and recycles it after a large number of jobs to bound
/// any external allocator/Z3 growth.
struct ProcessSynthesisWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    jobs_since_spawn: usize,
}

impl ProcessSynthesisWorker {
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

    fn shutdown(mut self) {
        let _ = write_frame(&mut self.stdin, &ProcessWorkerRequest::Shutdown);
        drop(self.stdin);
        let _ = self.child.wait();
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn jobs_since_spawn(&self) -> usize {
        self.jobs_since_spawn
    }

    fn should_recycle(&self) -> bool {
        self.jobs_since_spawn >= PROCESS_WORKER_RECYCLE_JOB_LIMIT
    }

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
        let (shallow, deep) = synth.precompute_candidates_stratified(config.gadget_depth)?;
        let mut candidates = shallow;
        candidates.extend(deep);
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
        let allow_any_lane_order = !(self.config.natural_order && job.stage + 1 == job.stage_count);
        let results = self.synth.synthesize_graph(
            graph,
            &job.input_state,
            &job.target_pairs,
            SynthesisOptions {
                max_unique_outputs: self.config.max_unique_outputs,
                allow_any_lane_order,
            },
        )?;
        Ok(ProcessWorkerOutput { job, results })
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

        let synth =
            GadgetSynthesizer::new(to_synth_arch(config.arch), to_synth_dtype(config.dtype));
        let (shallow_candidates, deep_candidates) =
            synth.precompute_candidates_stratified(config.gadget_depth)?;
        let mut transition_table =
            TransitionTable::new_with_lanes(stages.len(), elements_per_vector);
        let initial_state_id = transition_table.intern_zero_based_state(&initial_state);
        let mut exhausted_stages = HashSet::new();
        if config.retroactive_input {
            prepopulate_retroactive_stage0(&mut transition_table, &stages[0]);
            let forwarded = transition_table
                .get_unforwarded_output_ids(0)
                .into_iter()
                .map(|(state_id, _)| state_id)
                .collect::<Vec<_>>();
            transition_table.mark_forwarded_ids(0, &forwarded);
            exhausted_stages.insert(0);
        }

        let stage_count = stages.len();

        Ok(Self {
            config,
            elements_per_vector,
            total_elements,
            stages,
            shallow_candidates,
            deep_candidates,
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

    pub fn shallow_candidates(&self) -> &[GadgetGraph] {
        &self.shallow_candidates
    }

    pub fn deep_candidates(&self) -> &[GadgetGraph] {
        &self.deep_candidates
    }

    pub fn transition_table(&self) -> &TransitionTable {
        &self.transition_table
    }

    pub fn transition_table_mut(&mut self) -> &mut TransitionTable {
        &mut self.transition_table
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
        let mut trace = RuntimeTrace::disabled();
        self.discover_paths_for_transition_with_trace(transition, max_paths, &mut trace)
    }

    pub fn discover_paths_for_transition_with_trace(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
        trace: &mut RuntimeTrace,
    ) -> usize {
        self.discover_paths_through_transition_with_trace(
            transition,
            max_paths,
            trace,
            "manual_discovery",
        )
    }

    pub fn discover_paths_through_transition(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
    ) -> usize {
        let mut trace = RuntimeTrace::disabled();
        self.discover_paths_through_transition_with_trace(
            transition,
            max_paths,
            &mut trace,
            "manual_discovery",
        )
    }

    fn discover_paths_through_transition_with_trace(
        &mut self,
        transition: TransitionRef,
        max_paths: Option<usize>,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> usize {
        let paths = self
            .transition_table
            .trace_paths_through(transition, max_paths, None);
        self.enqueue_scoring_paths_with_trace(paths, trace, reason)
    }

    pub fn poll_scoring_jobs(&mut self) -> usize {
        let mut trace = RuntimeTrace::disabled();
        self.poll_scoring_jobs_with_trace(&mut trace, "manual_poll")
    }

    pub fn drain_scoring_jobs(&mut self) -> usize {
        let mut trace = RuntimeTrace::disabled();
        self.drain_scoring_jobs_with_trace(&mut trace, "manual_drain")
    }

    fn poll_scoring_jobs_with_trace(&mut self, trace: &mut RuntimeTrace, reason: &str) -> usize {
        self.poll_scoring_jobs_limited_with_trace(trace, reason, None)
    }

    fn poll_some_scoring_jobs_with_trace(
        &mut self,
        trace: &mut RuntimeTrace,
        reason: &str,
        result_limit: usize,
    ) -> usize {
        if result_limit == 0 {
            return 0;
        }
        self.poll_scoring_jobs_limited_with_trace(trace, reason, Some(result_limit))
    }

    fn poll_scoring_jobs_limited_with_trace(
        &mut self,
        trace: &mut RuntimeTrace,
        reason: &str,
        result_limit: Option<usize>,
    ) -> usize {
        let mut received = 0;
        while result_limit.is_none_or(|limit| received < limit) {
            let Ok(result) = self.scoring_pool.receiver.try_recv() else {
                break;
            };
            self.buffer_scoring_result_with_trace(result, trace, reason);
            received += 1;
        }
        let scored = self.apply_ready_scoring_results_with_trace(trace, reason, result_limit);
        self.top_off_scoring_jobs_with_trace(trace, reason);
        scored
    }

    fn drain_scoring_jobs_with_trace(&mut self, trace: &mut RuntimeTrace, reason: &str) -> usize {
        self.flush_pending_registered_path_rescores_with_trace(trace, reason);
        self.top_off_scoring_jobs_with_trace(trace, reason);
        let mut scored = self.poll_scoring_jobs_with_trace(trace, reason);
        while self.scoring_pool.pending_jobs > 0 || !self.pending_scoring_path_ids.is_empty() {
            self.top_off_scoring_jobs_with_trace(trace, reason);
            if self.scoring_pool.pending_jobs == 0 {
                break;
            }
            let Ok(result) = self.scoring_pool.receiver.recv() else {
                break;
            };
            self.buffer_scoring_result_with_trace(result, trace, reason);
            scored += self.apply_ready_scoring_results_with_trace(trace, reason, None);
            self.top_off_scoring_jobs_with_trace(trace, reason);
        }
        scored
    }

    fn buffer_scoring_result_with_trace(
        &mut self,
        result: ScoringJobResult,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) {
        let fields = ScoringJobTraceFields {
            order: result.order,
            path_id: result.path_id,
            path_length: result.path_length,
            assigned_candidates: result.scored_paths.len(),
        };
        self.buffered_scoring_results.insert(fields.order, result);
        self.trace_scoring_job_completed(trace, reason, fields);
    }

    pub fn record_transition(
        &mut self,
        stage: usize,
        input: StateId,
        output_state: &VectorState,
        gadget: PermutationGadget,
    ) -> TransitionRecordResult {
        let mut trace = RuntimeTrace::disabled();
        self.record_transition_with_trace(stage, input, output_state, gadget, &mut trace)
    }

    fn record_transition_with_trace(
        &mut self,
        stage: usize,
        input: StateId,
        output_state: &VectorState,
        gadget: PermutationGadget,
        trace: &mut RuntimeTrace,
    ) -> TransitionRecordResult {
        let output = State::try_from_zero_based_u64(output_state.top(), output_state.bottom())
            .expect("synthesized output labels should fit in compact state storage");
        let insert = self
            .transition_table
            .add_transition_by_id(stage, input, output, gadget);
        let discovered_paths = if insert.gadget_was_new {
            if insert.transition_was_new {
                self.discover_paths_through_transition_with_trace(
                    insert.transition,
                    None,
                    trace,
                    "new_transition",
                )
            } else {
                self.defer_registered_paths_for_transition_rescore_with_trace(
                    insert.transition,
                    trace,
                    "new_gadget_on_registered_transition",
                )
            }
        } else {
            0
        };

        TransitionRecordResult {
            transition_added: insert.gadget_was_new,
            discovered_paths,
        }
    }

    pub fn select_target_stage(&self) -> Option<usize> {
        if self.stages.is_empty() {
            return None;
        }

        if self.wave_count == 0 && !self.exhausted_stages.contains(&0) {
            return Some(0);
        }

        let target = self.pick_target_stage()?;
        let stage_data = self.transition_table.stage(target);
        let needs_bubble_up =
            stage_data.consecutive_zero_budgets() >= 2 || stage_data.unproductive_waves() >= 3;

        if needs_bubble_up {
            for upstream in (0..target).rev() {
                if !self.exhausted_stages.contains(&upstream)
                    && !self.stalled_stages.contains(&upstream)
                {
                    return Some(upstream);
                }
            }
        }

        Some(target)
    }

    fn pick_target_stage(&self) -> Option<usize> {
        (0..self.stages.len())
            .filter(|stage| {
                !self.exhausted_stages.contains(stage) && !self.stalled_stages.contains(stage)
            })
            .min_by_key(|stage| {
                let stats = self.transition_table.stage_stats(*stage);
                (stats.distinct_outputs, stats.attempts, *stage)
            })
    }

    pub fn make_jobs(
        &self,
        stage: usize,
        input_states: Option<&[VectorState]>,
        limit: Option<usize>,
    ) -> Vec<SynthesisJob> {
        let owned_inputs;
        let inputs = if let Some(input_states) = input_states {
            let input_states = input_states
                .iter()
                .filter_map(|state| {
                    self.transition_table
                        .lookup_zero_based_tuple(&state.as_tuple())
                        .map(|id| (id, state.clone()))
                })
                .collect::<Vec<_>>();
            self.ordered_inputs_for_stage(stage, input_states)
        } else if stage == 0 {
            vec![(self.initial_state_id, self.initial_state.clone())]
        } else {
            owned_inputs = self.ordered_outputs_from_stage(stage - 1);
            self.ordered_inputs_for_stage(stage, owned_inputs)
        };

        let candidate_count = self.shallow_candidates.len() + self.deep_candidates.len();
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
                for (input, input_state) in &inputs {
                    if self
                        .transition_table
                        .was_attempted_by_id(stage, *input, candidate_index)
                    {
                        continue;
                    }

                    jobs.push(SynthesisJob {
                        stage,
                        input: *input,
                        input_state: input_state.clone(),
                        candidate_index,
                    });

                    if limit.is_some_and(|limit| jobs.len() >= limit) {
                        return jobs;
                    }
                }
            }
        }
        jobs
    }

    pub fn execute_job(
        &mut self,
        job: &SynthesisJob,
    ) -> Result<JobExecutionResult, SynthesisError> {
        let mut trace = RuntimeTrace::disabled();
        self.execute_job_with_trace(job, &mut trace)
    }

    fn execute_job_with_trace(
        &mut self,
        job: &SynthesisJob,
        trace: &mut RuntimeTrace,
    ) -> Result<JobExecutionResult, SynthesisError> {
        let output = synthesize_worker_job(
            self.config,
            WorkerSynthesisJob {
                index: 0,
                job: job.clone(),
                graph: self
                    .candidate_graph(job.candidate_index)
                    .expect("job candidate index should be valid")
                    .clone(),
                target_pairs: self.stage_target_pairs(job.stage),
                stage_count: self.stages.len(),
            },
        )?;

        Ok(self.apply_synthesis_output_with_trace(output, trace))
    }

    fn take_process_workers(
        &mut self,
        worker_count: usize,
        trace: &mut RuntimeTrace,
    ) -> Result<Vec<ProcessSynthesisWorker>, SynthesisError> {
        while self.process_workers.len() < worker_count {
            let worker = ProcessSynthesisWorker::spawn(self.config)?;
            self.trace_process_worker_spawned(trace, "spawn", &worker);
            self.process_workers.push(worker);
        }

        let workers = self
            .process_workers
            .drain(0..worker_count)
            .collect::<Vec<_>>();
        self.trace_process_worker_pool_snapshot(trace, "checkout", &workers);
        Ok(workers)
    }

    fn return_process_workers(
        &mut self,
        workers: Vec<ProcessSynthesisWorker>,
        trace: &mut RuntimeTrace,
    ) {
        self.trace_process_worker_pool_snapshot(trace, "return", &workers);
        for worker in workers {
            if worker.should_recycle() {
                self.trace_process_worker_recycled(trace, &worker);
                worker.shutdown();
            } else {
                self.process_workers.push(worker);
            }
        }
        self.trace_process_worker_pool_snapshot(trace, "idle", &self.process_workers);
    }

    fn apply_synthesis_output_with_trace(
        &mut self,
        output: SynthesisJobOutput,
        trace: &mut RuntimeTrace,
    ) -> JobExecutionResult {
        let job = output.job;

        self.transition_table.record_attempt(job.stage, 1);
        self.transition_table.record_attempted_pair_by_id(
            job.stage,
            job.input,
            job.candidate_index,
        );

        let valid_output_count = output.results.len();
        let mut new_transition_count = 0;
        let mut discovered_paths = 0;
        for (gadget, output_state) in output.results {
            let result = self.record_transition_with_trace(
                job.stage,
                job.input,
                &output_state,
                gadget,
                trace,
            );
            if result.transition_added() {
                new_transition_count += 1;
                discovered_paths += result.discovered_paths();
            }
        }

        JobExecutionResult {
            stage: job.stage,
            candidate_index: job.candidate_index,
            attempts: 1,
            valid_output_count,
            transition_count: valid_output_count,
            new_transition_count,
            discovered_paths,
        }
    }

    /// Runs synthesis for one stage with no runtime events or trace output.
    ///
    /// This is the smallest public execution entry point. It prepares jobs for
    /// `stage`, applies up to `attempt_budget` candidate jobs, and stops early
    /// if `output_budget` new unique outputs are reached. New terminal paths
    /// discovered while applying transitions are still registered and queued
    /// for rough scoring; scoring results may be applied later by wave/run
    /// methods.
    pub fn run_stage_sync(
        &mut self,
        stage: usize,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<StageRunResult, SynthesisError> {
        self.run_stage_sync_with_inputs(stage, None, attempt_budget, output_budget)
    }

    fn run_stage_sync_with_session_and_trace(
        &mut self,
        stage: usize,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
    ) -> Result<StageRunResult, SynthesisError> {
        self.run_stage_sync_with_inputs_and_session(
            stage,
            None,
            attempt_budget,
            output_budget,
            session,
            trace,
        )
    }

    /// Runs one complete wave with the provided runtime session and trace.
    ///
    /// A wave selects the current target stage, runs that stage, then
    /// propagates newly discovered states through later stages. It also polls
    /// a bounded amount of completed rough-scoring work at wave boundaries.
    /// Use [`Self::run_sync`] when you want the normal repeated-wave search
    /// loop.
    pub fn run_wave_sync(
        &mut self,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
    ) -> Result<WaveRunResult, SynthesisError> {
        let wave = self.wave_count;
        let target_stage = self
            .select_target_stage()
            .expect("wave engine should have at least one stage");
        let last_stage = self.stages.len() - 1;
        self.poll_some_scoring_jobs_with_trace(
            trace,
            "wave_started",
            SCORING_RESULT_POLL_LIMIT_PER_WAVE_BOUNDARY,
        );
        let last_outputs_before = self.transition_table.unique_output_count(last_stage);
        let target = self.run_stage_sync_with_session_and_trace(
            target_stage,
            attempt_budget,
            output_budget,
            session,
            trace,
        )?;
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

        for stage in target_stage + 1..self.stages.len() {
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
            let result = self.run_stage_sync_with_inputs_and_session(
                stage,
                Some(&input_states),
                attempt_budget,
                output_budget,
                session,
                trace,
            )?;
            if result.attempts() > 0 || result.new_outputs() > 0 {
                propagation.push(result);
            }
        }

        let last_outputs_after = self.transition_table.unique_output_count(last_stage);
        if last_outputs_after > last_outputs_before {
            self.transition_table.reset_unproductive_waves(target_stage);
        } else {
            self.transition_table.record_unproductive_wave(target_stage);
        }

        self.wave_count += 1;
        self.flush_pending_registered_path_rescores_with_trace(trace, "wave_finished");
        let scored_paths = self.poll_some_scoring_jobs_with_trace(
            trace,
            "wave_finished",
            SCORING_RESULT_POLL_LIMIT_PER_WAVE_BOUNDARY,
        );
        let discovered_paths = target.discovered_paths()
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

    /// Runs the normal wave search loop without runtime events or trace output.
    ///
    /// This is the quiet programmatic entry point. It repeatedly runs waves
    /// until the search is exhausted, `max_waves` is reached, or the engine can
    /// no longer select a target stage. At the end it drains pending rough
    /// scoring jobs before returning the [`WaveSearchResult`].
    pub fn run_sync(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<WaveSearchResult, SynthesisError> {
        let mut session = NullRuntimeSession;
        self.run_sync_with_session(max_waves, attempt_budget, output_budget, &mut session)
    }

    /// Runs the normal wave search loop while emitting runtime events.
    ///
    /// This is the entry point used by line logging, the TUI, and tests that
    /// need user-visible progress without low-level JSON trace events.
    /// `session.should_stop()` is checked between waves and propagation stages
    /// for cooperative cancellation.
    pub fn run_sync_with_session(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<WaveSearchResult, SynthesisError> {
        let mut trace = RuntimeTrace::disabled();
        self.run_sync_with_session_and_trace(
            max_waves,
            attempt_budget,
            output_budget,
            session,
            &mut trace,
        )
    }

    /// Runs the normal wave search loop with runtime events and JSON tracing.
    ///
    /// This variant is useful when the caller wants both presentation-level
    /// [`RuntimeEvent`] updates and detailed trace events for profiling or
    /// postmortem analysis. It uses a no-op [`WaveProgressObserver`]; use
    /// [`Self::run_sync_with_session_trace_and_observer`] when live full
    /// scoring or a test hook must observe wave completions.
    pub fn run_sync_with_session_and_trace(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
    ) -> Result<WaveSearchResult, SynthesisError> {
        let mut observer = NullWaveProgressObserver;
        self.run_sync_with_session_trace_and_observer(
            max_waves,
            attempt_budget,
            output_budget,
            session,
            trace,
            &mut observer,
        )
    }

    /// Runs the full wave search loop with all progress extension points.
    ///
    /// This is the most complete execution entry point. In addition to runtime
    /// events and JSON tracing, it calls `observer` after each wave and after
    /// the final rough-scoring drain with the recently applied scored paths.
    /// The production solve path uses this to feed live full uiCA scoring
    /// without blocking the synthesis coordinator.
    ///
    /// The method preserves deterministic ordering despite asynchronous
    /// workers: synthesis results are applied in candidate order and rough
    /// scoring results are applied in submission order before they are exposed
    /// to the observer.
    pub fn run_sync_with_session_trace_and_observer(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
        observer: &mut impl WaveProgressObserver,
    ) -> Result<WaveSearchResult, SynthesisError> {
        session.on_event(RuntimeEvent::RunStarted {
            stage_count: self.stages.len(),
        });
        crate::trace!(trace, "engine_run_started", {
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
        self.emit_scoring_update_with_trace(session, trace, "run_started");

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
            crate::trace!(trace, "wave_started", {
                        "wave": self.wave_count,
                        "target_stage": target_stage,
            });
            let wave = self.run_wave_sync(attempt_budget, output_budget, session, trace)?;
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
            crate::trace!(trace, "wave_finished", {
                        "wave": wave.wave(),
                        "target_stage": wave.target_stage(),
                        "attempts": wave.target().attempts(),
                        "valid_outputs": wave.target().valid_outputs(),
                        "new_outputs": wave.target().new_outputs(),
                        "discovered_paths": wave.discovered_paths(),
                        "scored_paths": wave.scored_paths(),
            });
            self.trace_storage_snapshot(trace, "wave_finished");
            self.emit_scoring_update_with_trace(session, trace, "wave_finished");
            observer
                .on_wave_progress(
                    self,
                    &self.recent_scored_paths,
                    session,
                    trace,
                    "wave_finished",
                )
                .map_err(SynthesisError::WorkerProcess)?;
            self.recent_scored_paths.clear();
            waves.push(wave);
        }

        self.drain_scoring_jobs_with_trace(trace, "run_finished_drain");
        self.trace_storage_snapshot(trace, "run_finished_drain");
        self.emit_scoring_update_with_trace(session, trace, "run_finished_drain");
        observer
            .on_wave_progress(
                self,
                &self.recent_scored_paths,
                session,
                trace,
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
        crate::trace!(trace, "engine_run_finished", {
                    "wave_count": result.wave_count(),
                    "search_exhausted": result.search_exhausted(),
                    "scored_paths": self.scored_path_count(),
                    "best_score": self.best_score(),
        });

        Ok(result)
    }

    fn trace_storage_snapshot(&self, trace: &mut RuntimeTrace, reason: &str) {
        crate::trace!(trace, "engine_storage_snapshot", {
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

    fn emit_scoring_update(&self, session: &mut impl RuntimeSession) {
        session.on_event(RuntimeEvent::ScoringProgress(
            self.scoring_progress_snapshot(),
        ));
    }

    fn emit_scoring_update_with_trace(
        &self,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) {
        self.emit_scoring_update(session);
        self.trace_scoring_queue_snapshot(trace, reason);
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
        trace: &mut RuntimeTrace,
        reason: &str,
        order: usize,
        path_id: PathId,
        path: &CompletePath,
    ) {
        crate::trace!(trace, if should_trace_scoring_detail(order), "scoring_job_queued", [
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
        trace: &mut RuntimeTrace,
        reason: &str,
        transition: TransitionRef,
        registered_paths: usize,
        was_new_pending_transition: bool,
    ) {
        crate::trace!(trace, "registered_path_rescore_deferred", {
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
        trace: &mut RuntimeTrace,
        reason: &str,
        transition_count: usize,
        candidate_paths: usize,
        enqueued: usize,
    ) {
        crate::trace!(trace, "registered_path_rescore_flushed", {
                "reason": reason,
                "transitions": transition_count,
                "candidate_paths": candidate_paths,
                "enqueued": enqueued,
                "queued": self.next_scoring_job_order,
                "pending": self.pending_scoring_job_count(),
        });
    }

    fn trace_process_worker_spawned(
        &self,
        trace: &mut RuntimeTrace,
        reason: &str,
        worker: &ProcessSynthesisWorker,
    ) {
        crate::trace!(trace, "process_worker_spawned", {
                "reason": reason,
                "pid": worker.pid(),
                "rss_kb": worker.rss_kb(),
                "recycle_job_limit": PROCESS_WORKER_RECYCLE_JOB_LIMIT,
        });
    }

    fn trace_process_worker_recycled(
        &self,
        trace: &mut RuntimeTrace,
        worker: &ProcessSynthesisWorker,
    ) {
        crate::trace!(trace, "process_worker_recycled", {
                "pid": worker.pid(),
                "jobs_since_spawn": worker.jobs_since_spawn(),
                "rss_kb": worker.rss_kb(),
                "recycle_job_limit": PROCESS_WORKER_RECYCLE_JOB_LIMIT,
        });
    }

    fn trace_process_worker_pool_snapshot(
        &self,
        trace: &mut RuntimeTrace,
        reason: &str,
        workers: &[ProcessSynthesisWorker],
    ) {
        crate::trace!(trace, "process_worker_pool_snapshot", [
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
        trace: &mut RuntimeTrace,
        reason: &str,
        pids: &[u32],
        counters: WorkerCounters,
    ) {
        crate::trace!(trace, if !pids.is_empty(), "process_worker_rss_snapshot", [
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

    fn trace_scoring_job_completed(
        &self,
        trace: &mut RuntimeTrace,
        reason: &str,
        fields: ScoringJobTraceFields,
    ) {
        crate::trace!(trace, if should_trace_scoring_detail(fields.order), "scoring_job_completed", [
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
        trace: &mut RuntimeTrace,
        reason: &str,
        fields: ScoringJobTraceFields,
        applied_candidates: usize,
    ) {
        crate::trace!(trace, if should_trace_scoring_detail(fields.order), "scoring_job_applied", [
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

    fn trace_scoring_queue_snapshot(&self, trace: &mut RuntimeTrace, reason: &str) {
        crate::trace!(trace, "scoring_queue_snapshot", [
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
        self.exhausted_stages.len() >= self.stages.len()
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

    fn enqueue_scoring_paths_with_trace(
        &mut self,
        paths: Vec<CompletePath>,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> usize {
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
        self.top_off_scoring_jobs_with_trace(trace, reason);
        self.trace_scoring_queue_snapshot(trace, reason);
        enqueued
    }

    fn defer_registered_paths_for_transition_rescore_with_trace(
        &mut self,
        transition: TransitionRef,
        trace: &mut RuntimeTrace,
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
            trace,
            reason,
            transition,
            registered_paths,
            was_new_pending_transition,
        );
        0
    }

    fn flush_pending_registered_path_rescores_with_trace(
        &mut self,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> usize {
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
        let enqueued = self.enqueue_scoring_path_ids_with_trace(
            path_ids.into_iter().collect(),
            trace,
            "registered_gadget_rescore",
        );
        self.trace_registered_path_rescore_flushed(
            trace,
            reason,
            transition_count,
            candidate_paths,
            enqueued,
        );
        enqueued
    }

    fn enqueue_scoring_path_ids_with_trace(
        &mut self,
        path_ids: Vec<PathId>,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> usize {
        if path_ids.is_empty() {
            return 0;
        }

        let mut enqueued = 0;
        for path_id in path_ids {
            if self.queue_scoring_path(path_id) {
                enqueued += 1;
            }
        }
        self.top_off_scoring_jobs_with_trace(trace, reason);
        self.trace_scoring_queue_snapshot(trace, reason);
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

    fn top_off_scoring_jobs_with_trace(&mut self, trace: &mut RuntimeTrace, reason: &str) -> usize {
        let mut submitted = 0;
        while self.scoring_pool.has_capacity() {
            let Some(path_id) = self.pending_scoring_path_ids.pop_front() else {
                break;
            };
            self.pending_scoring_path_set.remove(&path_id);
            self.submit_scoring_path_with_trace(path_id, trace, reason);
            submitted += 1;
        }
        submitted
    }

    fn submit_scoring_path_with_trace(
        &mut self,
        path_id: PathId,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) {
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
        self.trace_scoring_job_queued(trace, reason, order, path_id, &path);
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

    fn apply_ready_scoring_results_with_trace(
        &mut self,
        trace: &mut RuntimeTrace,
        reason: &str,
        result_limit: Option<usize>,
    ) -> usize {
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
            self.trace_scoring_job_applied(trace, reason, fields, applied_candidates);
        }
        scored
    }

    fn stage_has_inputs(&self, stage: usize) -> bool {
        stage == 0 || self.transition_table.unique_output_count(stage - 1) > 0
    }

    fn stage_has_remaining_jobs(&self, stage: usize) -> bool {
        !self.make_jobs(stage, None, Some(1)).is_empty()
    }

    fn run_stage_sync_with_inputs(
        &mut self,
        stage: usize,
        input_states: Option<&[VectorState]>,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<StageRunResult, SynthesisError> {
        let mut session = NullRuntimeSession;
        let mut trace = RuntimeTrace::disabled();
        self.run_stage_sync_with_inputs_and_session(
            stage,
            input_states,
            attempt_budget,
            output_budget,
            &mut session,
            &mut trace,
        )
    }

    fn run_stage_sync_with_inputs_and_session(
        &mut self,
        stage: usize,
        input_states: Option<&[VectorState]>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
    ) -> Result<StageRunResult, SynthesisError> {
        let outputs_at_start = self.transition_table.unique_output_count(stage);
        let attempts_at_start = self.transition_table.stage_stats(stage).attempts;
        let jobs = self.make_jobs(stage, input_states, Some(attempt_budget));
        self.stage_progress_totals[stage] =
            self.stage_progress_totals[stage].max(attempts_at_start + jobs.len());
        let worker_capacity = if jobs.is_empty() {
            0
        } else {
            self.config.worker_count.min(jobs.len()).max(1)
        };
        crate::trace!(trace, "stage_jobs_prepared", {
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
                trace,
            );
        }

        let mut attempts = 0;
        let mut valid_outputs = 0;
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
            let result = self.execute_job_with_trace(&job, trace)?;
            attempts += result.attempts();
            valid_outputs += result.valid_output_count();
            new_transitions += result.new_transition_count();
            discovered_paths += result.discovered_paths();
            self.poll_scoring_jobs_with_trace(trace, "stage_job_finished");
            self.emit_scoring_update_with_trace(session, trace, "stage_job_finished");
            self.set_stage_worker_state(stage, 0, 1, job_count.saturating_sub(job_index + 1));
            self.emit_stage_update(stage, session);

            let new_outputs = self.transition_table.unique_output_count(stage) - outputs_at_start;
            if new_outputs >= output_budget {
                self.set_stage_worker_state(stage, 0, 0, 0);
                self.emit_stage_update(stage, session);
                return Ok(StageRunResult {
                    stage,
                    attempts,
                    valid_outputs,
                    new_outputs,
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
            new_transitions,
            discovered_paths,
        })
    }

    fn run_stage_worker_pool(
        &mut self,
        stage: usize,
        jobs: Vec<SynthesisJob>,
        outputs_at_start: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
        trace: &mut RuntimeTrace,
    ) -> Result<StageRunResult, SynthesisError> {
        let worker_count = self.config.worker_count.min(jobs.len()).max(1);
        let config = self.config;
        let backend = self.resolved_worker_backend();
        let mut process_workers = if backend == WorkerBackendArg::Process {
            Some(self.take_process_workers(worker_count, trace)?)
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
        crate::trace!(trace, "worker_pool_started", {
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
            let mut new_transitions = 0;
            let mut discovered_paths = 0;
            let mut buffered_results = BTreeMap::new();
            let mut next_rss_trace_job = PROCESS_WORKER_RSS_TRACE_INTERVAL;

            while !session.should_stop() && in_flight < worker_count && next_to_send < jobs.len() {
                let job_index = next_to_send;
                self.send_worker_job(&job_sender, job_index, &jobs[job_index]);
                next_to_send += 1;
                in_flight += 1;
                trace_job_submitted(
                    trace,
                    &jobs[job_index],
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
            trace_worker_counters(
                trace,
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
                trace,
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
                crate::trace!(trace, "worker_pool_stop_requested", {
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
                            trace,
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
                        crate::trace!(trace, "job_started", {
                                    "stage": stage,
                                    "job_index": index,
                                    "candidate_index": jobs[index].candidate_index(),
                                    "active_workers": active_workers,
                                    "in_flight": in_flight,
                        });
                        trace_worker_counters(
                            trace,
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
                            Ok(output) => crate::trace!(trace, "job_finished", {
                                        "stage": stage,
                                        "job_index": index,
                                        "candidate_index": output.job.candidate_index(),
                                        "status": "ok",
                                        "valid_outputs": output.results.len(),
                                        "active_workers": active_workers,
                                        "in_flight": in_flight,
                            }),
                            Err(error) => crate::trace!(trace, "job_finished", {
                                        "stage": stage,
                                        "job_index": index,
                                        "candidate_index": jobs[index].candidate_index(),
                                        "status": "error",
                                        "error": error.to_string(),
                                        "active_workers": active_workers,
                                        "in_flight": in_flight,
                            }),
                        }
                        trace_worker_counters(
                            trace,
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
                                    let job_stage = output.job.stage();
                                    let candidate_index = output.job.candidate_index();
                                    let worker_valid_outputs = output.results.len();
                                    crate::trace!(trace, "job_apply_started", {
                                                "stage": job_stage,
                                                "job_index": job_index,
                                                "candidate_index": candidate_index,
                                                "valid_outputs": worker_valid_outputs,
                                    });
                                    let apply_result =
                                        self.apply_synthesis_output_with_trace(output, trace);
                                    attempts += apply_result.attempts();
                                    valid_outputs += apply_result.valid_output_count();
                                    new_transitions += apply_result.new_transition_count();
                                    discovered_paths += apply_result.discovered_paths();
                                    applied_jobs += 1;
                                    if applied_jobs.is_multiple_of(
                                        SCORING_RESULT_POLL_INTERVAL_PER_SYNTHESIS_JOBS,
                                    ) {
                                        self.poll_some_scoring_jobs_with_trace(
                                            trace,
                                            "job_apply_finished",
                                            SCORING_RESULT_POLL_LIMIT_PER_SYNTHESIS_INTERVAL,
                                        );
                                        self.emit_scoring_update_with_trace(
                                            session,
                                            trace,
                                            "job_apply_finished",
                                        );
                                    }
                                    if applied_jobs >= next_rss_trace_job {
                                        self.trace_process_worker_rss_snapshot(
                                            trace,
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
                                    crate::trace!(trace, "job_apply_finished", [
                                        let stage_new_outputs =
                                            self.transition_table.unique_output_count(stage)
                                                - outputs_at_start;
                                    ], {
                                                "stage": job_stage,
                                                "job_index": job_index,
                                                "candidate_index": candidate_index,
                                                "attempts": apply_result.attempts(),
                                                "valid_outputs": apply_result.valid_output_count(),
                                                "new_transitions": apply_result.new_transition_count(),
                                                "discovered_paths": apply_result.discovered_paths(),
                                                "total_attempts": attempts,
                                                "total_valid_outputs": valid_outputs,
                                                "total_new_transitions": new_transitions,
                                                "stage_new_outputs": stage_new_outputs,
                                    });
                                    trace_worker_counters(
                                        trace,
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
                                        crate::trace!(trace, "worker_pool_stop_requested", {
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
                                    if new_outputs >= output_budget {
                                        stop_sending = true;
                                        stop_applying = true;
                                        crate::trace!(trace, "worker_pool_stop_requested", {
                                                "stage": stage,
                                                "reason": "output_budget",
                                                "new_outputs": new_outputs,
                                                "output_budget": output_budget,
                                                "next_to_send": next_to_send,
                                                "in_flight": in_flight,
                                                "active_workers": active_workers,
                                        });
                                    }
                                }
                                Err(error) => {
                                    crate::trace!(trace, "worker_pool_stop_requested", {
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
                    crate::trace!(trace, "worker_pool_stop_requested", {
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
                        crate::trace!(trace, "worker_pool_stop_requested", {
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
                    trace_job_submitted(
                        trace,
                        &jobs[job_index],
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
                trace_worker_counters(
                    trace,
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
                self.return_process_workers(returned_process_workers, trace);
            }

            self.set_stage_worker_state(stage, 0, 0, 0);
            self.emit_stage_update(stage, session);
            trace_worker_counters(
                trace,
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
            crate::trace!(trace, "worker_pool_finished", {
                    "stage": stage,
                    "attempts": attempts,
                    "valid_outputs": valid_outputs,
                    "new_outputs": new_outputs,
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
                job: job.clone(),
                graph: self
                    .candidate_graph(job.candidate_index)
                    .expect("job candidate index should be valid")
                    .clone(),
                target_pairs: self.stage_target_pairs(job.stage),
                stage_count: self.stages.len(),
            }))
            .expect("worker job channel should stay open while scheduling");
    }

    fn ordered_outputs_from_stage(&self, stage: usize) -> Vec<(StateId, VectorState)> {
        let mut outputs = self
            .transition_table
            .get_unique_output_ids(stage)
            .into_iter()
            .collect::<Vec<_>>();
        outputs.sort_by_key(|(state_id, _)| {
            self.transition_table.state_as_zero_based_tuple(*state_id)
        });
        outputs
    }

    fn ordered_inputs_for_stage(
        &self,
        stage: usize,
        input_states: Vec<(StateId, VectorState)>,
    ) -> Vec<(StateId, VectorState)> {
        if stage == 0 {
            return input_states;
        }

        let (mut covered, mut uncovered): (Vec<_>, Vec<_>) =
            input_states.into_iter().partition(|(state_id, _)| {
                self.transition_table
                    .was_input_attempted_by_id(stage, *state_id)
            });
        uncovered.append(&mut covered);
        uncovered
    }

    fn candidate_graph(&self, candidate_index: usize) -> Option<&GadgetGraph> {
        if candidate_index < self.shallow_candidates.len() {
            self.shallow_candidates.get(candidate_index)
        } else {
            self.deep_candidates
                .get(candidate_index.checked_sub(self.shallow_candidates.len())?)
        }
    }

    fn stage_target_pairs(&self, stage: usize) -> Vec<(u64, u64)> {
        self.stages[stage]
            .pairs()
            .iter()
            .map(|(left, right)| ((*left - 1) as u64, (*right - 1) as u64))
            .collect()
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

fn trace_worker_counters(trace: &mut RuntimeTrace, reason: &str, counters: WorkerCounters) {
    crate::trace!(trace, "worker_counters", {
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
    trace: &mut RuntimeTrace,
    job: &SynthesisJob,
    job_index: usize,
    counters: WorkerCounters,
) {
    crate::trace!(trace, "job_submitted", {
            "stage": job.stage(),
            "job_index": job_index,
            "candidate_index": job.candidate_index(),
            "input_top": job.input_state().top(),
            "input_bottom": job.input_state().bottom(),
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
    let allow_any_lane_order =
        !(config.natural_order && worker_job.job.stage + 1 == worker_job.stage_count);
    let synth = GadgetSynthesizer::new(to_synth_arch(config.arch), to_synth_dtype(config.dtype));
    let results = synth.synthesize_graph(
        &worker_job.graph,
        &worker_job.job.input_state,
        &worker_job.target_pairs,
        SynthesisOptions {
            max_unique_outputs: config.max_unique_outputs,
            allow_any_lane_order,
        },
    )?;

    Ok(SynthesisJobOutput {
        job: worker_job.job,
        results,
    })
}

fn prepopulate_retroactive_stage0(table: &mut TransitionTable, first_stage: &BitonicStage) {
    let mut pairs = first_stage.pairs().to_vec();
    permute_pairs(&mut pairs, 0, &mut |permuted| {
        let state = VectorState::new(
            permuted.iter().map(|(top, _)| (*top - 1) as u64).collect(),
            permuted
                .iter()
                .map(|(_, bottom)| (*bottom - 1) as u64)
                .collect(),
        );
        table.add_transition(
            0,
            &state,
            &state,
            PermutationGadget::new(Vec::new(), Vec::new()),
        );
    });
}

fn permute_pairs<F>(pairs: &mut [(usize, usize)], start: usize, visit: &mut F)
where
    F: FnMut(&[(usize, usize)]),
{
    if start >= pairs.len() {
        visit(pairs);
        return;
    }

    for idx in start..pairs.len() {
        pairs.swap(start, idx);
        permute_pairs(pairs, start + 1, visit);
        pairs.swap(start, idx);
    }
}

fn create_initial_state(first_stage: &BitonicStage) -> VectorState {
    VectorState::new(
        first_stage
            .pairs()
            .iter()
            .map(|(top, _)| (*top - 1) as u64)
            .collect(),
        first_stage
            .pairs()
            .iter()
            .map(|(_, bottom)| (*bottom - 1) as u64)
            .collect(),
    )
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
